"""Phase 3 Stage 2 — Experiment B: ex-slope corruption vs recovery mode (Option B driver).

The scientific question (stage2_cpp_patch.md): how much time does EB-fallback buy on the ex
slope, with detect-&-drop as the control? For each corruption pattern x corrupted-element
fraction, the SAME Python-corrupted index file is queried under all three recovery modes and
the recall deltas are recorded.

Division of labour (hard rules 1-2):
  Python (here): defines clean (CRC manifest of the clean index, written BEFORE injection),
    injects corruption into the serialized bytes via qp.faults primitives, computes the
    AUTHORITATIVE recall/collapse from the returned ids via qp.metrics, stamps provenance
    up-front. Injection never happens in C++.
  C++ (patched exp_dumpids --recovery): loads the corrupted file, CRC-scans it against the
    manifest, applies Samuel's exp-3 policy (none/drop/fallback_eb) during search, dumps ids.

Fraction axis = fraction of ELEMENTS whose ex_code is corrupted — the same x-axis as Samuel's
F3 fault-injection sweep ({0.01,0.05,0.08,0.20,0.57}, fraction of vectors marked corrupt), so
gate 3 can compare against Samuel's published fallback_eb value at f=0.05 (median 0.941865,
results/datasets/sift/faultinject_summary.json). Patterns choose WHICH elements: uniform
random / clustered runs / cross-row stride pairs / burst accumulation (E3c's 4 mechanisms at
element granularity); qp.faults then flips flips_per_element bits inside each chosen
element's ex_code (<=32 bits, so CRC-32 detection is guaranteed, and only ex_code is touched
-> corrupted_regions=["ex_code"], EB's bin+factors dependency intact).

Modes:
  --gate  : acceptance gates 1-5 in order, stop at first failure (stage2_cpp_patch.md §4).
  --sweep : patterns x fractions x recovery modes -> jsonl records + summary.
  --lazy-gate : the --crc-mode load-vs-lazy alignment gates (see below). Separate from --gate
    on purpose: gates 1-5 and expb_gates.json are frozen Stage-2 acceptance output and this
    round must not change their shape.

--crc-mode selects WHEN the CRC runs, never what it decides. `load` is the eager whole-index
scan at load time (the original, still the default and the mode every frozen artifact was
produced under); `lazy` recomputes an element's CRC as the search consults it, which is what
the paper's Sec 3.1/3.3 describe and which does not assume the index is immutable while
queries run. Because the predicate and its decision points are identical, the two modes must
return bit-identical ids on an index that does not change mid-run — that is gate L1, not an
assumption. Note the load-scan totals (stats.load.*) are ZERO under lazy: no scan ran.
"""

import argparse
import json
import math
import os
import subprocess
import zlib

import numpy as np

from qp import config, faults, metrics, provenance
from qp.bits import flip_bits
from qp.rabitq import crc_manifest, layout
from qp.rabitq.registry import get_adapter, adapter_name

PATTERNS = ("uniform_accum", "clustered_accum", "cross_row_accum", "burst_accum")
RECOVERY_MODES = ("none", "drop", "fallback_eb")
FIELD = "ex_code"

# Samuel F3 anchors (results/datasets/sift/faultinject_summary.json, b=7 SIFT, 10 seeds).
SAMUEL_EB_AT_5PCT = 0.941865       # fallback_eb median @ f=0.05 — the gate-3 anchor
GATE3_TOL = 0.01
CLEAN_ANCHOR = 0.98376             # qp dumpids clean baseline @ ef=2000, k=10 (Stage 0)
GATE2_TOL = 5e-4

DEFAULT_FRACTIONS = (0.01, 0.05, 0.08, 0.20, 0.57)
SMOKE_FRACTIONS = (0.05, 0.20)
SMOKE_PATTERNS = ("uniform_accum",)

CFG_DEFAULTS = {
    "flips_per_element": 4,   # bits flipped inside each corrupted element's ex_code (<=32)
    "run_len": 64,            # clustered: contiguous element run length
    "stride": 512,            # cross_row: element-id distance between pair members
    "burst_batches": 4,       # burst_accum: ~m/batches new elements per tick until m reached
    "ef": 2000,               # recall plateau (Stage 0 convention)
}


# ---------------------------------------------------------------------------
# Element selection (which elements get corrupted) — deterministic per (pattern, fraction, seed)
# ---------------------------------------------------------------------------

def _pattern_seed(seed, pattern, fraction):
    """Stable per-(pattern, fraction) RNG seed; zlib.crc32 keeps it text-stable across runs."""
    return (int(seed) ^ zlib.crc32(pattern.encode()) ^ int(round(fraction * 1e6))) & 0x7FFFFFFF


def _lane_seed(base, lane):
    """Independent deterministic sub-stream seed. NOT xor: `base ^ lane` aliases (e.g. bases
    b and b+1 with lanes 0 and 1 produce the SAME seed set, so adjacent --seed values would
    select identical element unions — caught by the determinism test). SeedSequence spawning
    is collision-free and platform-stable."""
    return int(np.random.SeedSequence([int(base), int(lane)]).generate_state(1)[0])


def select_elements(pattern, fraction, n, seed, cfg):
    """Element ids to corrupt: exactly ceil(fraction*n) of them, pattern-shaped, deterministic.

    uniform_accum   — iid element choice (Samuel's set_fault_injection analog, real bytes).
    clustered_accum — non-overlapping contiguous runs of run_len ids (grid-aligned so overlap
                      is impossible by construction at any fraction, incl. 0.57).
    cross_row_accum — pairs (e, e+stride): the rowhammer victim geometry at element scale.
    burst_accum     — batches accumulate over internal ticks (per-tick seed = seed^tick, the
                      E3c convention) until the union reaches the target.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    m = min(n, math.ceil(fraction * n))
    rng = np.random.default_rng(_pattern_seed(seed, pattern, fraction))

    if pattern == "uniform_accum":
        return sorted(int(e) for e in rng.choice(n, size=m, replace=False))

    if pattern == "clustered_accum":
        run_len = min(int(cfg["run_len"]), max(1, n // 8))
        n_blocks = n // run_len
        need_blocks = math.ceil(m / run_len)
        if need_blocks > n_blocks:
            raise ValueError(f"fraction {fraction} needs {need_blocks} runs of {run_len}, "
                             f"only {n_blocks} grid blocks exist")
        blocks = rng.choice(n_blocks, size=need_blocks, replace=False)
        elems = []
        for b in sorted(int(x) for x in blocks):
            elems.extend(range(b * run_len, b * run_len + run_len))
        return sorted(elems[:m])                      # trim the tail block to hit m exactly

    if pattern == "cross_row_accum":
        stride = min(int(cfg["stride"]), max(1, n // 2))
        used = set()
        attempts = 0
        while len(used) < m and attempts < 50 * (m + 1):
            attempts += 1
            base = int(rng.integers(0, n - stride))
            if base in used or base + stride in used:
                continue
            used.add(base)
            used.add(base + stride)
        e = 0
        while len(used) < m and e < n - stride:      # deterministic fill if rejection stalls
            if e not in used and e + stride not in used:
                used.add(e)
                used.add(e + stride)
            e += 1
        return sorted(used)[:m] if len(used) > m else sorted(used)

    if pattern == "burst_accum":
        batch = max(1, math.ceil(m / int(cfg["burst_batches"])))
        used = []
        seen = set()
        tick = 0
        while len(used) < m:
            trng = np.random.default_rng(_lane_seed(_pattern_seed(seed, pattern, fraction),
                                                    tick))
            for e in trng.choice(n, size=min(batch, n), replace=False):
                e = int(e)
                if e not in seen:
                    seen.add(e)
                    used.append(e)
            tick += 1
        return sorted(used[:m])                       # trim the last tick's batch

    raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")


def inject_elements(buf, rmap, elements, pattern, seed, cfg):
    """Flip flips_per_element bits inside each selected element's ex_code (qp.faults/qp.bits).

    cross_row pairs share the SAME in-field bit offsets (victim-row semantics); other patterns
    place an independent seeded burst per element (faults.temporal_burst = exactly-k distinct
    bits, so every selected element really is corrupted and the CRC scan must flag exactly
    len(elements)). Returns all flipped (byte, bit) positions; faults.restore undoes them.
    """
    flips = int(cfg["flips_per_element"])
    level0_start, spe, field_off, field_len, n = crc_manifest.field_geometry(rmap, FIELD)
    if not (0 < flips <= field_len * 8):
        raise ValueError(f"flips_per_element must be in 1..{field_len * 8}")
    # <=32 flips: CRC-32 detection is GUARANTEED (burst bound). Beyond that, a per-element
    # miss has probability ~2^-32 (random damage); _check_crc_fail_count still asserts the
    # exact count, so an astronomically-rare collision fails loudly instead of skewing rows.
    positions = []
    if pattern == "cross_row_accum":
        # elements were built as (e, e+stride) pairs; sorted order does not preserve the
        # pairing, so re-pair by stride membership: mirror offsets onto e+stride when present.
        eset = set(elements)
        stride = min(int(cfg["stride"]), max(1, n // 2))
        done = set()
        for e in elements:
            if e in done:
                continue
            partner = e + stride if (e + stride) in eset else None
            rng = np.random.default_rng(_lane_seed(_pattern_seed(seed, pattern, 0.0), e))
            rel_bits = rng.choice(field_len * 8, size=flips, replace=False)
            group = [e] if partner is None else [e, partner]
            for member in group:
                base = level0_start + member * spe + field_off
                pos = [(base + int(b) // 8, int(b) % 8) for b in rel_bits]
                flip_bits(buf, pos)
                positions.extend(pos)
                done.add(member)
    else:
        base = _pattern_seed(seed, pattern, 0.0)
        for e in elements:
            region = (level0_start + e * spe + field_off, field_len)
            pos = faults.temporal_burst(buf, region, [flips], _lane_seed(base, e))
            positions.extend(pos)
    return positions


# ---------------------------------------------------------------------------
# Shared setup: clean file + manifest + clean baseline + up-front provenance
# ---------------------------------------------------------------------------

def _measured_recall(res, gt, k):
    """AUTHORITATIVE recall — qp.metrics on the C++-dumped ids (hard rule #2)."""
    return float(metrics.recall_at_k(res["ids"], gt, k))


def setup_context(args):
    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)
    out_dir = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                           "phase3", "expb")
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(CFG_DEFAULTS)
    cfg["ef"] = int(args.ef)
    cfg["flips_per_element"] = int(args.flips_per_element)
    k = int(args.k)

    buf = adapter.serialize_index()
    rmap = adapter.region_map()
    gt = adapter.load_groundtruth()

    clean_path = os.path.join(out_dir, "_expb_clean.index")
    adapter.deserialize_index(buf, clean_path)

    manifest_path = os.path.join(out_dir, "expb_clean_ex_code.crcmf")
    crc_manifest.write_manifest(buf, manifest_path, field=FIELD, rmap=rmap)
    pre = crc_manifest.verify_buffer(buf, crc_manifest.read_manifest(manifest_path))
    assert pre == [], f"pre-flight: clean buffer fails its own manifest at elements {pre[:5]}"

    # Clean baselines: legacy positional path (backward-compat check) + recovery=none path.
    ids_legacy, cpp_legacy = adapter.query_ids(clean_path, k=k, ef=cfg["ef"],
                                               timeout=args.timeout)
    clean_recall = float(metrics.recall_at_k(ids_legacy, gt, k))

    # Provenance stamped NOW, before any corruption run (Stage-1 lesson: never backfill).
    meta = provenance.collect_provenance(
        adapter, aname, {"recall@10": clean_recall}, cfg, args, rmap,
        index_sha256=provenance.sha256_file(clean_path),
        corrupted_regions=[FIELD],
        recovery=list(RECOVERY_MODES),
        crc_manifest_sha256=provenance.sha256_file(manifest_path))
    # Tagged sweeps (e.g. --out-tag fstar_check) get their own meta file — a fixed name
    # would overwrite the original sweeps' provenance stamp.
    _tag = f"_{args.out_tag}" if getattr(args, "out_tag", None) else ""
    meta_path = os.path.join(out_dir, f"expb_meta{_tag}.json")
    with open(meta_path, "w") as fh:
        json.dump({"meta": meta}, fh, indent=2)

    print(f"[expb] adapter={aname} clean@10={clean_recall:.5f} (cpp {cpp_legacy:.5f}) "
          f"n={rmap['header']['cur_element_count']} ef={cfg['ef']} k={k}")
    return {
        "adapter": adapter, "aname": aname, "out_dir": out_dir, "cfg": cfg, "k": k,
        "buf": buf, "rmap": rmap, "gt": gt, "clean_path": clean_path,
        "manifest_path": manifest_path, "clean_recall": clean_recall,
        "clean_ids": ids_legacy, "meta": meta, "meta_path": meta_path,
        "corrupt_path": os.path.join(out_dir, "_expb_corrupt.index"),
        "timeout": args.timeout, "seed": int(args.seed),
        "out_tag": getattr(args, "out_tag", None),
        # getattr, not args.crc_mode: tests build the driver Namespace by hand and must not
        # have to learn every new flag (same convention as out_tag above).
        "crc_mode": getattr(args, "crc_mode", "load"),
    }


def _run_one(ctx, index_path, mode, crc_mode=None, crc_timer=False):
    """One C++ (or stub) run; returns (row_fragment, res) with authoritative recall."""
    adapter, k, cfg = ctx["adapter"], ctx["k"], ctx["cfg"]
    crc_mode = ctx.get("crc_mode", "load") if crc_mode is None else crc_mode
    # recovery=none loads no manifest, so "when to CRC" is meaningless there (and the adapters
    # reject the combination); keep the default so a none run is unchanged.
    kw = {} if mode == "none" else {"crc_mode": crc_mode, "crc_timer": crc_timer}
    # Distinct scratch per crc_mode: a load and a lazy run of the same config must not
    # overwrite each other's dumped ids, which gate L1 compares.
    suffix = "" if crc_mode == "load" else f"_{crc_mode}"
    out_path = os.path.join(ctx["out_dir"], f"_expb_dumpids{suffix}.ivecs")
    try:
        res = adapter.query_with_recovery(
            index_path, mode, ctx["manifest_path"] if mode != "none" else None,
            k=k, ef=cfg["ef"], out_path=out_path, timeout=ctx["timeout"], **kw)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return ({"recall@10": None, "cpp_recall": None, "failure_mode": metrics.CRASH,
                 "silent_collapse": False, "stats": None,
                 "error": type(exc).__name__}, None)
    recall = _measured_recall(res, ctx["gt"], k)
    fmode = metrics.classify_failure(None, None, res["ids"], recall, ctx["clean_recall"])
    return ({"recall@10": recall, "cpp_recall": res["cpp_recall"],
             "failure_mode": fmode,
             "silent_collapse": bool(metrics.is_silent_collapse(
                 recall, ctx["clean_recall"], failure_mode=fmode)),
             "stats": res.get("stats")}, res)


def _corrupt_index_file(ctx, pattern, fraction):
    """Select + inject + serialize the corrupted index; returns (elements, positions, sha)."""
    buf, rmap, cfg = ctx["buf"], ctx["rmap"], ctx["cfg"]
    n = int(rmap["header"]["cur_element_count"])
    elements = select_elements(pattern, fraction, n, ctx["seed"], cfg)
    positions = inject_elements(buf, rmap, elements, pattern, ctx["seed"], cfg)
    ctx["adapter"].deserialize_index(buf, ctx["corrupt_path"])
    sha = provenance.sha256_file(ctx["corrupt_path"])
    return elements, positions, sha


def _restore(ctx, positions):
    faults.restore(ctx["buf"], positions)
    drift = int(np.sum(ctx["buf"] != ctx["adapter"].serialize_index()))
    assert drift == 0, f"[expb] restore drift = {drift} bytes (injection bookkeeping bug)"


def _check_crc_fail_count(row, n_elements, mode, crc_mode="load"):
    """CRC detections must line up with what Python injected — the check is mode-aware.

    load: the scan covers every element, so it must flag EXACTLY the injected count (<=32-bit
      flips per element are CRC-32-guaranteed detectable; a mismatch means manifest/injection
      disagree). Unchanged from Stage 2 — every frozen record was checked this way.
    lazy: nothing is checked until the search consults it, so only the consulted subset can be
      flagged: 0 < distinct <= injected. The lower bound matters — `<= injected` alone is
      satisfied by a lazy path that never CRC'd anything at all, which is precisely the bug
      this would need to catch (see audits/GATE_DESIGN_MEMO.md: a check with no reachable
      failing input is not a gate).
    """
    stats = row.get("stats")
    if mode == "none" or stats is None or row["failure_mode"] == metrics.CRASH:
        return
    if crc_mode == "load":
        got = stats["load"]["elements_crc_fail"]
        assert got == n_elements, (f"CRC scan flagged {got} elements but Python injected "
                                   f"{n_elements} — manifest<->injection mismatch")
        return
    got = (stats.get("crc") or {}).get("distinct_elements_failed")
    if got is None:
        return          # stub: no real search, so nothing knows which elements were consulted
    assert 0 < got <= n_elements, (
        f"on-access CRC flagged {got} distinct elements against {n_elements} injected; "
        f"expected 0 < flagged <= injected (0 means the lazy path never checked anything, "
        f"> injected means it flagged elements Python never touched)")


# ---------------------------------------------------------------------------
# --sweep
# ---------------------------------------------------------------------------

def run_sweep(ctx, patterns, fractions, modes):
    all_rows = []
    tag = f"_{ctx['out_tag']}" if ctx.get("out_tag") else ""
    for pattern in patterns:
        rec_path = os.path.join(ctx["out_dir"],
                                f"expb_{pattern}_{FIELD}{tag}.records.jsonl")
        with open(rec_path, "w") as rec_f:
            for fraction in fractions:
                elements, positions, sha = _corrupt_index_file(ctx, pattern, fraction)
                frac_actual = len(elements) / int(ctx["rmap"]["header"]["cur_element_count"])
                print(f"[expb] {pattern} f={fraction} -> {len(elements)} elements "
                      f"({len(positions)} bit flips)")
                for mode in modes:
                    frag, _ = _run_one(ctx, ctx["corrupt_path"], mode)
                    _check_crc_fail_count(frag, len(elements), mode, ctx.get("crc_mode", "load"))
                    row = {"pattern": pattern, "fraction": fraction,
                           "fraction_actual": frac_actual, "n_elements": len(elements),
                           "bit_flips": len(positions),
                           "flips_per_element": ctx["cfg"]["flips_per_element"],
                           "recovery": mode,
                           "delta_vs_clean": (None if frag["recall@10"] is None else
                                              round(frag["recall@10"] - ctx["clean_recall"], 6)),
                           "region": FIELD, "seed": ctx["seed"], "ef": ctx["cfg"]["ef"],
                           "index_sha256": sha, "adapter": ctx["aname"], **frag}
                    rec_f.write(json.dumps(row) + "\n")
                    rec_f.flush()
                    all_rows.append(row)
                    r10 = "crash" if row["recall@10"] is None else f"{row['recall@10']:.5f}"
                    print(f"        recovery={mode:<11} recall@10={r10}")
                _restore(ctx, positions)

    summary = {
        "experiment": "expb_recovery",
        "clean_recall@10": ctx["clean_recall"],
        "patterns": list(patterns), "fractions": list(fractions), "recovery": list(modes),
        "rows": len(all_rows),
        "eb_minus_drop": {
            f"{p}@{f}": _delta(all_rows, p, f, "fallback_eb", "drop")
            for p in patterns for f in fractions},
        "eb_minus_none": {
            f"{p}@{f}": _delta(all_rows, p, f, "fallback_eb", "none")
            for p in patterns for f in fractions},
        "adapter": ctx["aname"],
        "meta": ctx["meta"],
    }
    out_json = os.path.join(ctx["out_dir"], f"expb_summary{tag}.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[expb] sweep done ({len(all_rows)} rows) -> {out_json}")
    return summary


def _delta(rows, pattern, fraction, mode_a, mode_b):
    def find(mode):
        for r in rows:
            if (r["pattern"], r["fraction"], r["recovery"]) == (pattern, fraction, mode):
                return r["recall@10"]
        return None
    a, b = find(mode_a), find(mode_b)
    return None if (a is None or b is None) else round(a - b, 6)


# ---------------------------------------------------------------------------
# --gate  (stage2_cpp_patch.md §4: sequential, stop at first failure)
# ---------------------------------------------------------------------------

def _stub_expected(ctx, mode, frac_actual):
    """The stub's documented-fake recall model (structural/plumbing check only)."""
    from qp.rabitq import stub_adapter
    return max(0.05, stub_adapter.CLEAN_RECALL *
               (1.0 - stub_adapter._RECOVERY_SLOPE[mode] * frac_actual))


def run_gates(ctx):
    real = ctx["aname"] == "real"
    results = []
    label = "" if real else " [STUB (non-scientific: plumbing check only)]"

    def gate(num, name, ok, detail):
        results.append({"gate": num, "name": name, "pass": bool(ok), "detail": detail})
        print(f"[expb] gate {num} {name}{label}: {'PASS' if ok else 'FAIL'} — {detail}")
        return bool(ok)

    def finish(all_ok):
        out = os.path.join(ctx["out_dir"], "expb_gates.json")
        with open(out, "w") as fh:
            json.dump({"all_pass": all_ok, "adapter": ctx["aname"], "gates": results,
                       "meta": ctx["meta"]}, fh, indent=2)
        print(f"[expb] gates {'ALL GREEN' if all_ok else 'STOPPED AT FAILURE'} -> {out}")
        return 0 if all_ok else 1

    # Gate 1 — build: binaries + the patched usage line.
    if real:
        from qp.rabitq import adapter as real_adapter
        built = real_adapter.binaries_built()
        usage = ""
        if built:
            proc = subprocess.run([os.path.join(real_adapter.BIN, "exp_dumpids")],
                                  capture_output=True, text=True)
            usage = proc.stdout + proc.stderr
        ok = built and "--recovery" in usage
        detail = f"binaries_built={built}, usage_has_recovery={'--recovery' in usage}"
    else:
        ok, detail = True, "stub adapter always 'built'"
    if not gate(1, "build", ok, detail):
        return finish(False)

    # Gate 2 — no-op anchor: clean index, all 3 modes == baseline ids and anchor recall.
    ok2, details = True, []
    for mode in RECOVERY_MODES:
        frag, res = _run_one(ctx, ctx["clean_path"], mode)
        ids_same = res is not None and np.array_equal(res["ids"], ctx["clean_ids"])
        crc0 = (mode == "none" or
                (frag["stats"] and frag["stats"]["load"]["elements_crc_fail"] == 0))
        anchor = CLEAN_ANCHOR if real else ctx["clean_recall"]
        tol = GATE2_TOL if real else 1e-9
        r_ok = frag["recall@10"] is not None and abs(frag["recall@10"] - anchor) <= tol
        ok2 = ok2 and ids_same and crc0 and r_ok
        details.append(f"{mode}: recall={frag['recall@10']}, ids_same={ids_same}, "
                       f"crc_fail_zero={crc0}")
    if not gate(2, "clean no-op anchor", ok2, "; ".join(details)):
        return finish(False)

    # Gate 3 — Fig-2/F3 anchor: 5% uniform elements + fallback_eb vs Samuel's median.
    elements, positions, _ = _corrupt_index_file(ctx, "uniform_accum", 0.05)
    frac_actual = len(elements) / int(ctx["rmap"]["header"]["cur_element_count"])
    frag3, res3 = _run_one(ctx, ctx["corrupt_path"], "fallback_eb")
    _check_crc_fail_count(frag3, len(elements), "fallback_eb", ctx.get("crc_mode", "load"))
    if real:
        anchor, tol = SAMUEL_EB_AT_5PCT, GATE3_TOL
    else:
        anchor, tol = _stub_expected(ctx, "fallback_eb", frac_actual), 2e-3
    ok3 = frag3["recall@10"] is not None and abs(frag3["recall@10"] - anchor) <= tol
    detail3 = (f"recall={frag3['recall@10']} vs anchor={anchor:.6f} ±{tol} "
               f"({len(elements)} elements corrupted)")
    if not ok3 and real:
        detail3 += ("  REPORT-AND-STOP: EB semantics may have diverged from Samuel exp-3 — "
                    "check the pessimistic formula (est + (est - low), hnsw.hpp EB site), the "
                    "error-bound field (bin_factors f_error via the 1-bit estimate), and "
                    "tie-breaking (BoundedKNN ascending est_dist). If marginal, retry at "
                    "ef=1000 (Samuel's plateau grid) before declaring divergence.")
    gate3_pass = gate(3, "F3 5% fallback_eb anchor", ok3, detail3)
    if not gate3_pass:
        _restore(ctx, positions)
        return finish(False)
    _restore(ctx, positions)

    # Gate 4 — ordering at f=0.20: fallback_eb >= drop, none the worst.
    #
    # "none should be worst (silently using BAD data)" presumes the data is actually garbage,
    # so the GATED run corrupts half of each element's ex_code bits (severe). Measured on this
    # workstation: at the default light damage (4/768 bits) none LEGITIMATELY outranks
    # drop/fallback_eb (slightly-wrong ex distances beat discarding 20% of the database:
    # none=0.907 vs drop=0.797/eb=0.816), while drop/fallback_eb are severity-INVARIANT
    # (detection only depends on which elements failed CRC). The light-damage ordering is
    # therefore recorded as informational, not gated.
    field_len = crc_manifest.field_geometry(ctx["rmap"], FIELD)[3]
    severe = max(1, field_len * 8 // 2)
    saved_flips = ctx["cfg"]["flips_per_element"]

    def ordering_at(flips):
        ctx["cfg"]["flips_per_element"] = flips
        try:
            elements4, positions4, _ = _corrupt_index_file(ctx, "uniform_accum", 0.20)
            out = {}
            for mode in RECOVERY_MODES:
                frag, _ = _run_one(ctx, ctx["corrupt_path"], mode)
                _check_crc_fail_count(frag, len(elements4), mode, ctx.get("crc_mode", "load"))
                out[mode] = frag["recall@10"]
            _restore(ctx, positions4)
            return out
        finally:
            ctx["cfg"]["flips_per_element"] = saved_flips

    rec = ordering_at(severe)
    light = ordering_at(saved_flips) if saved_flips != severe else rec
    ok4 = (None not in rec.values() and rec["fallback_eb"] >= rec["drop"]
           and rec["none"] < min(rec["drop"], rec["fallback_eb"]))
    if not gate(4, "ordering eb>=drop, none worst (severe damage)", ok4,
                f"severe({severe} flips/elem): none={rec['none']}, drop={rec['drop']}, "
                f"fallback_eb={rec['fallback_eb']}; informational light({saved_flips} "
                f"flips/elem): none={light['none']}, drop={light['drop']}, "
                f"fallback_eb={light['fallback_eb']}"):
        return finish(False)

    # Gate 5 — determinism: repeat the gate-3 config; injection AND ids must be identical.
    elements5, positions5, sha5 = _corrupt_index_file(ctx, "uniform_accum", 0.05)
    frag5, res5 = _run_one(ctx, ctx["corrupt_path"], "fallback_eb")
    _restore(ctx, positions5)
    ok5 = (elements5 == elements and res5 is not None and res3 is not None
           and np.array_equal(res5["ids"], res3["ids"]))
    if not gate(5, "determinism (same seed -> same ids)", ok5,
                f"elements_same={elements5 == elements}, "
                f"ids_same={res5 is not None and res3 is not None and np.array_equal(res5['ids'], res3['ids'])}"):
        return finish(False)

    return finish(True)


# ---------------------------------------------------------------------------
# --lazy-gate  (--crc-mode load vs lazy alignment)
# ---------------------------------------------------------------------------

LAZY_GATE_FRACTION = 0.05      # gate-3's corruption config: small, and the F3 anchor point
LAZY_GATE_PATTERN = "uniform_accum"


def _wall_ns(frag):
    return (frag.get("stats") or {}).get("search_wall_ns")


def _crc_block(frag):
    return (frag.get("stats") or {}).get("crc") or {}


def run_lazy_gates(ctx):
    """Gate the on-access CRC mode against the eager scan it replaces.

    L1 (the core) — same corrupted index, same seed, same queries: `--crc-mode lazy` must
      return top-k ids IDENTICAL to `--crc-mode load`, and the same recall. This follows from
      the decision points being untouched, so a difference is an implementation bug, not a
      finding: report and stop.
    L2 — counts: lazy can only flag what the search consulted, so
      0 < distinct_lazy <= elements_crc_fail_load; on a clean index both are 0.
    L3 — overhead, produced not asserted. The headline is the load-vs-lazy search_wall_ns
      delta, which is meaningful precisely BECAUSE L1 proved both runs do the same search
      work. Two cross-checks accompany it: the opt-in per-check timer (inflated by its own
      clock reads) and crc_bytes x the load scan's unperturbed ns/byte.

    Run for BOTH detecting policies on purpose. Their consult points differ — drop consults
    every unvisited neighbour before the bounds guard, EB only candidates that reach rerank —
    so drop CRCs bytes it then never reads. Expect drop's overhead to be much the larger; that
    asymmetry is the honest scope of the "piggyback / no extra memory traffic" claim.
    """
    real = ctx["aname"] == "real"
    results, numbers = [], {}
    label = "" if real else " [STUB (non-scientific: plumbing check only)]"

    def gate(num, name, ok, detail):
        results.append({"gate": num, "name": name, "pass": bool(ok), "detail": detail})
        print(f"[expb] gate {num} {name}{label}: {'PASS' if ok else 'FAIL'} — {detail}")
        return bool(ok)

    def finish(all_ok):
        out = os.path.join(ctx["out_dir"], "expb_lazy_gates.json")
        with open(out, "w") as fh:
            json.dump({"all_pass": all_ok, "adapter": ctx["aname"], "gates": results,
                       "overhead": numbers, "config": {
                           "pattern": LAZY_GATE_PATTERN, "fraction": LAZY_GATE_FRACTION,
                           "ef": ctx["cfg"]["ef"], "k": ctx["k"], "seed": ctx["seed"],
                           "flips_per_element": ctx["cfg"]["flips_per_element"]},
                       "meta": ctx["meta"]}, fh, indent=2)
        print(f"[expb] lazy gates {'ALL GREEN' if all_ok else 'STOPPED AT FAILURE'} -> {out}")
        return 0 if all_ok else 1

    # L0 — the binary must know the flag at all (a cached pre-flag build silently ignores it).
    if real:
        from qp.rabitq import adapter as real_adapter
        proc = subprocess.run([os.path.join(real_adapter.BIN, "exp_dumpids")],
                              capture_output=True, text=True)
        usage = proc.stdout + proc.stderr
        ok0 = "--crc-mode" in usage
        detail0 = f"usage_has_crc_mode={ok0}"
    else:
        ok0, detail0 = True, "stub adapter always 'built'"
    if not gate(0, "build has --crc-mode", ok0, detail0):
        return finish(False)

    # L2a — clean index: neither mode may flag anything.
    clean_ok, clean_detail = True, []
    for mode in ("drop", "fallback_eb"):
        f_load, _ = _run_one(ctx, ctx["clean_path"], mode, crc_mode="load")
        f_lazy, _ = _run_one(ctx, ctx["clean_path"], mode, crc_mode="lazy")
        n_load = (f_load.get("stats") or {}).get("load", {}).get("elements_crc_fail")
        n_lazy = _crc_block(f_lazy).get("distinct_elements_failed")
        ok = (n_load == 0) and (n_lazy in (0, None))
        clean_ok = clean_ok and ok
        clean_detail.append(f"{mode}: load_fail={n_load}, lazy_distinct={n_lazy}")
    if not gate(1, "clean index flags nothing in either mode", clean_ok, "; ".join(clean_detail)):
        return finish(False)

    # L1/L2/L3 on a corrupted index, one pass per detecting policy.
    elements, positions, sha = _corrupt_index_file(ctx, LAZY_GATE_PATTERN, LAZY_GATE_FRACTION)
    try:
        equiv_ok, equiv_detail = True, []
        count_ok, count_detail = True, []
        for mode in ("fallback_eb", "drop"):
            f_load, r_load = _run_one(ctx, ctx["corrupt_path"], mode, crc_mode="load")
            f_lazy, r_lazy = _run_one(ctx, ctx["corrupt_path"], mode, crc_mode="lazy")
            # Only the load arm is asserted here (it pins the scan against the injection).
            # The lazy count is gate 3 below, which reports rather than raising — a bare
            # assert would abort with a traceback before any gate could be written.
            _check_crc_fail_count(f_load, len(elements), mode, "load")
            ids_same = (r_load is not None and r_lazy is not None
                        and np.array_equal(r_load["ids"], r_lazy["ids"]))
            rec_same = (f_load["recall@10"] is not None
                        and f_load["recall@10"] == f_lazy["recall@10"])
            equiv_ok = equiv_ok and ids_same and rec_same
            equiv_detail.append(f"{mode}: ids_same={ids_same}, recall={f_load['recall@10']} "
                                f"vs {f_lazy['recall@10']}")

            n_load = (f_load.get("stats") or {}).get("load", {}).get("elements_crc_fail")
            n_lazy = _crc_block(f_lazy).get("distinct_elements_failed")
            if n_lazy is None:                      # stub: nothing knows who was consulted
                count_detail.append(f"{mode}: lazy_distinct=None (stub)")
            else:
                ok = 0 < n_lazy <= n_load
                count_ok = count_ok and ok
                count_detail.append(f"{mode}: 0 < {n_lazy} <= {n_load} -> {ok}")

            # L3 numbers. The timer run is SEPARATE so the wall-clock delta above stays clean.
            f_timed, _ = _run_one(ctx, ctx["corrupt_path"], mode,
                                  crc_mode="lazy", crc_timer=True)
            w_load, w_lazy = _wall_ns(f_load), _wall_ns(f_lazy)
            crc_lazy, crc_timed = _crc_block(f_lazy), _crc_block(f_timed)
            load_blk = (f_load.get("stats") or {}).get("load", {})
            nq = (f_load.get("stats") or {}).get("nq")
            entry = {"n_queries": nq,
                     "search_wall_ns_load": w_load, "search_wall_ns_lazy": w_lazy,
                     "crc_checks": crc_lazy.get("checks"), "crc_bytes": crc_lazy.get("bytes"),
                     "crc_oob_skipped": crc_lazy.get("oob_skipped"),
                     "crc_ns_timed": crc_timed.get("ns"),
                     "crc_checks_timed": crc_timed.get("checks"),
                     "load_scan_ns": load_blk.get("crc_scan_ns"),
                     "load_scan_elements": load_blk.get("elements_checked")}
            if w_load and w_lazy and nq:
                entry["headline_delta_ns_per_query"] = (w_lazy - w_load) / nq
                entry["headline_pct_of_search"] = 100.0 * (w_lazy - w_load) / w_load
            if crc_timed.get("ns") and crc_timed.get("checks"):
                entry["timed_ns_per_check"] = crc_timed["ns"] / crc_timed["checks"]
            # Analytic: the load scan's ns/byte is measured over the whole index with no
            # per-call clock in the way, so it is the least perturbed CRC rate available.
            scan_bytes = None
            if load_blk.get("elements_checked"):
                field_len = crc_manifest.field_geometry(ctx["rmap"], FIELD)[3]
                scan_bytes = load_blk["elements_checked"] * field_len
            if scan_bytes and load_blk.get("crc_scan_ns") and crc_lazy.get("bytes") and nq:
                entry["analytic_ns_per_query"] = (
                    crc_lazy["bytes"] * (load_blk["crc_scan_ns"] / scan_bytes) / nq)
            numbers[mode] = entry
            pct, per_q = (entry.get("headline_pct_of_search"),
                          entry.get("headline_delta_ns_per_query"))
            print(f"[expb]   {mode} overhead: "
                  f"{'n/a' if pct is None else f'{pct:.2f}%'} of search wall, "
                  f"{'n/a' if per_q is None else f'{per_q:.0f} ns'}/query "
                  f"({entry.get('crc_checks')} checks, {entry.get('crc_bytes')} bytes)")

        if not gate(2, "load vs lazy return identical ids", equiv_ok, "; ".join(equiv_detail)):
            return finish(False)
        if not gate(3, "lazy distinct failures <= load scan failures", count_ok,
                    "; ".join(count_detail)):
            return finish(False)
    finally:
        _restore(ctx, positions)

    gate(4, "overhead measured (reported, not asserted)", True,
         "; ".join(f"{m}: {numbers[m].get('headline_pct_of_search', 'n/a')}% of search wall"
                   for m in numbers))
    return finish(True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv(kind, allowed):
    def parse(s):
        vals = tuple(x.strip() for x in s.split(",") if x.strip())
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown {kind}: {bad} (choose from {allowed})")
        return vals
    return parse


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment B: ex-slope corruption vs recovery "
                                             "(Option B gates + sweep)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gate", action="store_true", help="run acceptance gates 1-5 in order")
    mode.add_argument("--sweep", action="store_true", help="patterns x fractions x recovery")
    mode.add_argument("--lazy-gate", action="store_true", dest="lazy_gate",
                      help="--crc-mode load vs lazy: id equivalence, counts, and the "
                           "query-path detection overhead")
    ap.add_argument("--adapter", default="auto", choices=("stub", "real", "auto"))
    ap.add_argument("--smoke", action="store_true",
                    help=f"small sweep: patterns={SMOKE_PATTERNS} fractions={SMOKE_FRACTIONS}; "
                         "outputs to artifacts_smoke/")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--ef", type=int, default=CFG_DEFAULTS["ef"])
    ap.add_argument("--k", type=int, default=config.K)
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-subprocess seconds; a hang under corruption records as crash")
    ap.add_argument("--fractions", type=lambda s: tuple(float(x) for x in s.split(",")),
                    default=None)
    ap.add_argument("--patterns", type=_csv("pattern", PATTERNS), default=None)
    ap.add_argument("--recovery", type=_csv("recovery mode", RECOVERY_MODES), default=None)
    ap.add_argument("--flips-per-element", type=int,
                    default=CFG_DEFAULTS["flips_per_element"], dest="flips_per_element")
    ap.add_argument("--out-tag", default=None, dest="out_tag",
                    help="suffix for sweep output filenames (e.g. sev384) so sweeps at "
                         "different severities can coexist")
    ap.add_argument("--crc-mode", default="load", choices=("load", "lazy"), dest="crc_mode",
                    help="WHEN the CRC runs: load = eager scan at load (default, the mode "
                         "every frozen artifact was produced under); lazy = on-access")
    args = ap.parse_args(argv)

    # Gates 1-5 are Stage-2 acceptance and are defined against the load scan (gate 2 asserts
    # elements_crc_fail == 0, which lazy makes vacuously true). --lazy-gate covers lazy.
    if args.gate and args.crc_mode != "load":
        ap.error("--gate is defined against the load scan; use --lazy-gate for crc_mode=lazy")

    ctx = setup_context(args)
    if args.gate:
        return run_gates(ctx)
    if args.lazy_gate:
        return run_lazy_gates(ctx)
    patterns = args.patterns or (SMOKE_PATTERNS if args.smoke else PATTERNS)
    fractions = args.fractions or (SMOKE_FRACTIONS if args.smoke else DEFAULT_FRACTIONS)
    modes = args.recovery or RECOVERY_MODES
    run_sweep(ctx, patterns, fractions, modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
