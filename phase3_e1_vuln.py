#!/usr/bin/env python3
"""Phase 3 — Stage 1, E1: RaBitQ single-bit vulnerability characterization (F1 risk localization).

Ports the Phase 1/2 FAISS sensitivity sweep to RaBitQ. For each serialized structure
(rotation / bin_factors / ex_factors / bin_code / ex_code / pointers / centroids / header) and
each bit-class (derived from the library source, qp.rabitq.bitclass — NOT assumed float32) we flip
ONE bit, re-search the fixed query set, and record how far recall@10 moved + the failure mode
(silent / nan-inf / crash). The 64-byte global rotation is enumerated exhaustively (512 bits, like
SQ8's sq_scale); per-vector structures are sampled over N elements x bit positions with bootstrap
CI. Aggregation REUSES phase1_sensitivity.aggregate / write_vuln_map (the single vuln_map schema +
the unified silent-collapse predicate qp.metrics.is_silent_collapse) — nothing re-implemented.

From the vuln_map this derives the Top-Down criticality ordering (reduction order) and a three-tier
scrub/protection allocation, priced with phase3_cost over the per-vector-aggregated region map.

ADAPTER INJECTION (the whole point of this stage): the real RaBitQ search is an x86-only C++
subprocess; on the arm64 dev box a deterministic stub stands in. `--adapter {stub,real,auto}`
(qp.rabitq.get_adapter) switches with NO code change. Develop/test on stub here; the human runs
`--adapter real` (or auto) on the x86 workstation. See artifacts/phase3/E1_RUNBOOK.md.

Outputs (under --out, default artifacts/phase3/e1; smoke -> artifacts_smoke/phase3/e1):
  raw/rabitq.records.jsonl   one row per flip (streamed, flushed) for offline debugging
  raw/rabitq.done            completion marker (enables --resume)
  vuln_map.json / .csv       aggregated by (region x bit_position_tag) — phase1 schema
  criticality.json           Top-Down reduction order + three-tier scrub allocation + cost

Usage:
  python phase3_e1_vuln.py --adapter stub --smoke      # dev pipeline gate (no RaBitQ)
  python phase3_e1_vuln.py                             # workstation full run (auto -> real)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from qp import config, metrics, buckets
from qp.bits import flip_bit
from qp.rabitq import get_adapter, adapter_name, layout, bitclass
import phase3_recall as pr
from phase1_sensitivity import aggregate, write_vuln_map   # reuse the canonical schema/aggregator
import phase1_sensitivity as p1                            # reuse _crash_record (single crash schema)

INDEX_NAME = "RABITQ"
PARITY_TOL = 1e-3

# Structures characterized, in the layout order; presence is filtered against the real region map.
STRUCTURES = ["rotation", "centroids", "bin_factors", "ex_factors", "bin_code", "ex_code",
              "links", "cluster_id", "label", "header"]

FULL = dict(n_elem=30, s_code=32, s_ptr=24, s_global=256, crash_bytes=32, ef=2000)
SMOKE = dict(n_elem=3, s_code=8, s_ptr=8, s_global=24, crash_bytes=8, ef=64)


def log(msg):
    print(msg, flush=True)


# --- region lookup ------------------------------------------------------------

def _region(rmap, name):
    return next((r for r in rmap["regions"] if r["name"] == name), None)


def _bucket_for(struct):
    return "crash_structure" if bitclass.sampling_class(struct) == "crash_probe" else "recall_relevant"


# --- sample planning ----------------------------------------------------------

def plan_samples(rmap, cfg, rng):
    """Yield sample dicts: {structure, region, kind, region_bits, element, byte_pos, bit, off}.

    rotation: every one of its 512 bits. centroids: s_global fp32-tag-stratified. header: first
    crash_bytes bytes x 8 (crash probe). per-vector structs: n_elem random elements, then full
    enumeration for the tiny float factors and sampled bits for the larger code/pointer fields.
    """
    hdr = rmap["header"]
    pd = hdr["padded_dim"]
    n_elem_total = hdr["cur_element_count"]
    samples = []

    for struct in STRUCTURES:
        sclass = bitclass.sampling_class(struct)
        if struct in ("rotation", "centroids", "header"):
            reg = _region(rmap, struct)
            if reg is None or reg["byte_start"] is None:
                continue
            region_bits = reg["byte_len"] * 8
            base = dict(structure=struct, region=struct, kind=reg["kind"],
                        region_bits=region_bits, element=None)
            if sclass == "exhaustive":                       # rotation: every bit
                for off in range(reg["byte_len"]):
                    for bit in range(8):
                        samples.append({**base, "off": off, "bit": bit,
                                        "byte_pos": reg["byte_start"] + off})
            elif sclass == "global_sampled":                 # centroids: fp32-tag stratified
                per = max(1, cfg["s_global"] // 4)
                n_floats = max(1, reg["byte_len"] // 4)
                for tag in buckets.FP32_TAGS:
                    for wbyte, bit in buckets.FP32_TAG_BITS[tag]:
                        for _ in range(max(1, per // len(buckets.FP32_TAG_BITS[tag]))):
                            off = int(rng.integers(n_floats)) * 4 + wbyte
                            if off < reg["byte_len"]:
                                samples.append({**base, "off": off, "bit": bit,
                                                "byte_pos": reg["byte_start"] + off})
            elif sclass == "crash_probe":                    # header: probe leading bytes
                for off in range(min(cfg["crash_bytes"], reg["byte_len"])):
                    for bit in range(8):
                        samples.append({**base, "off": off, "bit": bit,
                                        "byte_pos": reg["byte_start"] + off})
            continue

        # per-vector structure
        reg0 = _region(rmap, f"elem0.{struct}")
        if reg0 is None:
            continue                                          # e.g. ex_* absent on a b=1 index
        region_bits = reg0["byte_len"] * 8
        elems = sorted(set(int(e) for e in
                           rng.integers(0, n_elem_total, size=min(cfg["n_elem"], n_elem_total))))
        for e in elems:
            bstart, blen = layout.element_field_range(rmap, struct, e)
            base = dict(structure=struct, region=struct, kind=reg0["kind"],
                        region_bits=region_bits, element=e)
            if struct in ("bin_factors", "ex_factors"):       # tiny floats -> full enumeration
                for off in range(blen):
                    for bit in range(8):
                        samples.append({**base, "off": off, "bit": bit, "byte_pos": bstart + off})
            else:                                             # pointers / bin_code / ex_code
                cap = cfg["s_ptr"] if struct in ("links", "cluster_id", "label") else cfg["s_code"]
                budget = min(cap, blen * 8)
                # Sample DISTINCT bits (no replacement) so duplicates aren't counted as independent
                # samples in n_samples / the bootstrap CI; both pointer lanes still appear.
                for bid in rng.choice(blen * 8, size=budget, replace=False):
                    off, bit = int(bid) // 8, int(bid) % 8
                    samples.append({**base, "off": off, "bit": bit, "byte_pos": bstart + off})
    return samples


# --- measurement --------------------------------------------------------------

def _crash_record(clean, exc):
    """Crash record built on phase1's single crash schema, plus the 3 RaBitQ-only columns."""
    rec = p1._crash_record(clean, exc)                    # faulted_*/dRecall*/failure_mode/...
    rec.update({"nan_inf_supported": None, "cpp_recall": None, "parity_abs": None})
    return rec


def apply_and_measure(adapter, ref_buf, tmp, positions, clean, cfg, timeout):
    """Flip a list of (byte,bit), search via the adapter, classify, restore. phase1-shaped record.

    Shared by the single-bit E1 sweep and the multi-bit E3a/E3b experiments. Crash (subprocess
    nonzero/segfault) and timeout (hang) -> CRASH. A Python-side adapter failure (parse/IO/
    RuntimeError) is a HARNESS error and is allowed to propagate (abort loud) so it is never
    miscounted as a corruption crash. XOR-restores every position in `finally`.
    """
    # Co-locate the ids/dist output with tmp so the real adapter's `<out>.dist.fvecs` is written
    # next to tmp and cleaned every iteration (no stale-distance read across flips); the stub
    # ignores out_path harmlessly.
    out_path = tmp + ".out.ivecs"
    for bp, b in positions:
        flip_bit(ref_buf, bp, b)
    try:
        adapter.deserialize_index(ref_buf, tmp)
        try:
            res = adapter.search_corrupted(tmp, k=config.K, ef=cfg["ef"],
                                           out_path=out_path, timeout=timeout)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            return _crash_record(clean, e)
        ids = res["ids"]
        dist = res.get("distances")
        recalls = pr.recall_block(ids, clean["gt"])          # recall@100 skipped (W=10)
        f10 = recalls.get("recall@10")
        if f10 is None:
            # recall_block drops recall@10 only when the returned id width < k — a measurement
            # error (exp_dumpids pads to k), not a benign result. Abort loud rather than silently
            # recording it as a non-collapse (which clean_baseline's hard subscript would not).
            raise RuntimeError(
                f"search returned id width < k ({np.asarray(ids).shape[1]} < {config.K}); "
                f"cannot score recall@10 — harness/measurement error, not a corruption outcome")
        fm = metrics.classify_failure(distances=dist, recall=f10,
                                      clean_recall=clean["recall@10"])
        nan_inf = int((~np.isfinite(np.asarray(dist))).sum()) if dist is not None else 0
        cpp = res.get("cpp_recall")
        parity = abs(f10 - cpp) if (cpp is not None and f10 is not None) else None
        rec = {
            "faulted_recall@1": recalls.get("recall@1"), "faulted_recall@10": f10,
            "faulted_recall@100": None, "faulted_tol": None,
            "dRecall@1": (clean["recall@1"] - recalls["recall@1"]) if "recall@1" in recalls else None,
            "dRecall@10": (clean["recall@10"] - f10) if f10 is not None else None,
            "dRecall@100": None, "dTol": None,
            "failure_mode": fm, "nan_inf_count": nan_inf,
            "nan_inf_supported": dist is not None,
            "cpp_recall": cpp, "parity_abs": parity, "exception_repr": "",
        }
        rec["is_silent_collapse"] = metrics.is_silent_collapse(
            f10, clean["recall@10"], failure_mode=fm)
        return rec
    finally:
        for bp, b in positions:
            flip_bit(ref_buf, bp, b)                          # XOR restore
        for stale in (out_path, out_path + ".dist.fvecs"):
            try:
                os.remove(stale)
            except OSError:
                pass


def measure_flip(adapter, ref_buf, tmp, s, clean, cfg, timeout):
    """Single-bit E1 measurement (thin wrapper over apply_and_measure)."""
    return apply_and_measure(adapter, ref_buf, tmp, [(s["byte_pos"], s["bit"])],
                             clean, cfg, timeout)


def clean_baseline(adapter, ref_buf, tmp, cfg, timeout):
    adapter.deserialize_index(ref_buf, tmp)
    res = adapter.search_corrupted(tmp, k=config.K, ef=cfg["ef"],
                                   out_path=tmp + ".out.ivecs", timeout=timeout)
    gt = adapter.load_groundtruth()
    recalls = pr.recall_block(res["ids"], gt)
    nan_inf_supported = res.get("distances") is not None
    return {"recall@1": recalls.get("recall@1"), "recall@10": recalls["recall@10"],
            "gt": gt, "nan_inf_supported": nan_inf_supported}


def assert_no_state_leak(adapter, ref_buf, tmp, clean, cfg, timeout, tol):
    """Re-search the (restored) ref_buf and assert clean recall@10 is unchanged.

    The phase1 D1 guarantee: a leaked/incomplete XOR-restore is caught loudly rather than
    silently biasing later flips. Shared by E1's post-sweep check and the E3a/E3b runs. `tol`
    is exact (1e-9) for the deterministic stub and loosened for the real C++ search (float
    reduction / threading jitter). Returns the observed drift.
    """
    after = clean_baseline(adapter, ref_buf, tmp, cfg, timeout)
    drift = abs(after["recall@10"] - clean["recall@10"])
    if drift > tol:
        raise RuntimeError(f"state leaked across flips (clean drift {drift:.2e} > tol {tol:.0e})")
    return drift


def _gate_clean_baseline(adapter, aname, clean, args):
    """Two report-and-stop guards before any corruption number is trusted (real adapter only).

    #2 nan-inf detectability: if the adapter returns no distances, a non-finite-distance
       corruption is silently folded into silent collapse (wrong scrub tier). The stub always
       supplies distances, so this only ever fires on a real run lacking the exp_dumpids
       extension; --allow-no-distances opts into the documented degradation.
    #4 operating-point anchor: clean recall@10 must sit at the index's reference plateau
       (adapter.EXPECTED_CLEAN_RECALL10) within --clean-tol, else index/ef/query/gt are
       mis-pointed and every downstream dRecall / collapse threshold is rebased to garbage.
    """
    allow_no_dist = getattr(args, "allow_no_distances", False)
    clean_tol = getattr(args, "clean_tol", 0.02)

    if not clean["nan_inf_supported"]:
        if aname == "real" and not allow_no_dist:
            raise RuntimeError(
                "REPORT-AND-STOP: real adapter returned no distances -> nan-inf is undetectable "
                "and would be miscounted as silent collapse. Extend exp_dumpids to emit "
                "<out>.dist.fvecs (E1_RUNBOOK §2), or pass --allow-no-distances to accept the "
                "documented degradation.")
        log("[e1] [note] adapter returned no distances -> nan-inf is NOT detectable; "
            "n_nan_inf reported as 'not-detected' (see E1_RUNBOOK: extend exp_dumpids).")

    expected = getattr(adapter, "EXPECTED_CLEAN_RECALL10", None)
    if expected is not None:
        cdiff = abs(clean["recall@10"] - expected)
        if cdiff > clean_tol:
            msg = (f"clean recall@10={clean['recall@10']:.4f} differs from operating-point anchor "
                   f"{expected} by {cdiff:.4f} > clean_tol {clean_tol}")
            if aname == "real":
                raise RuntimeError("REPORT-AND-STOP: " + msg +
                                   " — check index/ef/query/gt paths (or raise --clean-tol).")
            log(f"[e1] [warn] {msg}")


# --- criticality ordering + three-tier scrub allocation -----------------------

def derive_criticality(rows, rmap, collapse_min=5.0, harmful_min=5.0, crash_min=5.0, agg=None):
    """Roll vuln_map rows up per structure and assign a Top-Down reduction order + scrub tier.

    Per-structure metrics are the worst (max) over its bit-classes — a structure is as dangerous
    as its most dangerous bit-class. Tiers follow the brief's policy and are driven by MEASURED
    percentages, not assumptions:
      bounds_check   pointer/header structures whose dominant mode is crash (detectable; guard ids)
      frequent_scrub small global/critical structures that silently collapse (rotation/factors/bin)
      crc_eb_lazy    structures harmful-but-not-collapsing (ex code/factors) -> CRC + EB + reload
      none           immune (no measurable harm)

    SEVERITY (not just frequency): a heavy-tailed GLOBAL structure (e.g. the 64-B rotation) can have
    a low collapse FREQUENCY (few of its bits cross the retention bar) yet a rare bit that alone
    halves recall (high max ΔRecall@10). Frequency thresholds miss it, so any GLOBAL structure
    (layout.GLOBAL_FIELDS) with at least one single-point-collapse bit is upgraded to frequent_scrub
    regardless of frequency. Per-vector structures keep the frequency rule (one bad bit among ~1e6
    elements is negligible). max/p99 ΔRecall@10 are surfaced for transparency.
    """
    by_struct = {}
    for r in rows:
        d = by_struct.setdefault(r["region"], {"pct_collapse": 0.0, "pct_harmful": 0.0,
                                               "pct_crash": 0.0, "max_dRecall": 0.0,
                                               "p99_dRecall": 0.0, "has_collapse_bit": False,
                                               "kind": r["kind"]})
        d["pct_collapse"] = max(d["pct_collapse"], r.get("pct_collapse") or 0.0)
        d["has_collapse_bit"] = d["has_collapse_bit"] or (r.get("pct_collapse") or 0.0) > 0.0
        d["max_dRecall"] = max(d["max_dRecall"], r.get("max_dRecall@10") or 0.0)
        d["p99_dRecall"] = max(d["p99_dRecall"], r.get("p99_dRecall@10") or 0.0)
        # pct_crash from failure-mode counts (n_crash / n_samples) per row
        n = r.get("n_samples") or 0
        crash_pct = 100.0 * (r.get("n_crash") or 0) / n if n else 0.0
        d["pct_crash"] = max(d["pct_crash"], crash_pct)
        # HARMFUL excluding crash: pct_catastrophic = (ΔR>0.01).sum()+n_crash over n_total
        # (phase1_sensitivity.aggregate). Subtract the crash share so a near-crash structure is
        # tiered by its silent harm, not by crashes it already detects via bounds_check.
        harmful_noncrash = (r.get("pct_catastrophic") or 0.0) - crash_pct
        d["pct_harmful"] = max(d["pct_harmful"], max(0.0, harmful_noncrash))

    if agg is None:
        agg = layout.aggregate_region_map(rmap)
    agg_sizes = {r["name"]: r["byte_len"] for r in agg["regions"]}
    out = []
    for struct, d in by_struct.items():
        is_global = struct in layout.GLOBAL_FIELDS
        if d["pct_crash"] >= crash_min and (d["pct_collapse"] < collapse_min):
            tier = "bounds_check"
        elif d["pct_collapse"] >= collapse_min or (is_global and d["has_collapse_bit"]):
            tier = "frequent_scrub"
        elif d["pct_harmful"] >= harmful_min:
            tier = "crc_eb_lazy"
        else:
            tier = "none"
        out.append({"structure": struct, "kind": d["kind"],
                    "pct_collapse": round(d["pct_collapse"], 3),
                    "pct_harmful": round(d["pct_harmful"], 3),
                    "pct_crash": round(d["pct_crash"], 3),
                    "max_dRecall@10": round(d["max_dRecall"], 4),
                    "p99_dRecall@10": round(d["p99_dRecall"], 4),
                    "footprint_bytes": agg_sizes.get(struct),
                    "scrub_tier": tier})
    # Top-Down reduction order: collapse fraction first, then worst-case severity (max ΔRecall),
    # then harmful, then crash, then smaller footprint.
    out.sort(key=lambda x: (-x["pct_collapse"], -x["max_dRecall@10"], -x["pct_harmful"],
                            -x["pct_crash"], x["footprint_bytes"] or 0))
    for i, o in enumerate(out):
        o["criticality_rank"] = i + 1
    return out


def cost_of_allocation(ranking, rmap, agg=None):
    """Price the three-tier allocation with phase3_cost over the per-vector-aggregated map."""
    import phase3_cost as cost
    tier_spec = {"frequent_scrub": {"mult": 3, "checksum_bytes": 4},
                 "crc_eb_lazy": {"mult": 1, "checksum_bytes": 4},
                 "bounds_check": {"mult": 1, "checksum_bytes": 4},
                 "none": None}
    if agg is None:
        agg = layout.aggregate_region_map(rmap)
    present = {r["name"] for r in agg["regions"]}
    assignment = {}
    for o in ranking:
        spec = tier_spec[o["scrub_tier"]]
        if not spec:
            continue
        if o["structure"] not in present:
            # A structure flagged for protection but missing from the aggregated cost map (e.g. an
            # absent ex_* region at ex_bits==0, or a container field) would otherwise be priced as
            # free and silently unprotected — surface it instead of dropping it quietly.
            log(f"[e1] [warn] structure {o['structure']!r} (tier {o['scrub_tier']}) is absent from "
                f"the aggregated region map — its protection is NOT priced (footprint unknown).")
            continue
        if spec["mult"] > 1 or spec["checksum_bytes"] > 0:
            assignment[o["structure"]] = spec
    mc = cost.mem_cost(agg, assignment)
    total_bytes = sum(r["byte_len"] for r in agg["regions"])
    return {"protection_assignment": assignment, "mem_cost": mc,
            "index_total_bytes": total_bytes,
            "protection_overhead_pct": round(100.0 * mc["total_bytes"] / total_bytes, 4)
            if total_bytes else None}


# --- orchestration ------------------------------------------------------------

def run(args):
    cfg = dict(SMOKE if args.smoke else FULL)
    cfg["ef"] = args.ef if args.ef is not None else cfg["ef"]
    timeout = args.timeout

    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    raw = os.path.join(out, "raw", "rabitq.records.jsonl")
    done = os.path.join(out, "raw", "rabitq.done")
    tmp = os.path.join(out, "raw", "_corrupt.index")

    log(f"[e1] adapter={aname}  out={out}  smoke={args.smoke}  cfg={cfg}")

    rmap = adapter.region_map()                              # computed once, reused everywhere

    if args.resume and os.path.exists(done):
        log("[e1] --resume: reloading completed shard")
        with open(raw) as fh:
            records = [json.loads(line) for line in fh]
    else:
        ref_buf = adapter.serialize_index()
        clean = clean_baseline(adapter, ref_buf, tmp, cfg, timeout)
        log(f"[e1] clean recall@10={clean['recall@10']:.4f}  "
            f"nan_inf_detectable={clean['nan_inf_supported']}")
        _gate_clean_baseline(adapter, aname, clean, args)

        rng = np.random.default_rng(args.seed)
        samples = plan_samples(rmap, cfg, rng)
        by = {}
        for s in samples:
            by[s["structure"]] = by.get(s["structure"], 0) + 1
        log(f"[e1] {len(samples)} flips planned across {len(by)} structures: {by}")

        records = []
        n = 0
        t0 = time.time()
        with open(raw, "w") as fh:
            for s in samples:
                rec = measure_flip(adapter, ref_buf, tmp, s, clean, cfg, timeout)
                tag = bitclass.bit_class(f"elem0.{s['structure']}" if s["element"] is not None
                                         else s["structure"],
                                         s["off"], s["bit"], padded_dim=rmap["header"]["padded_dim"])
                rec.update({
                    "index": INDEX_NAME, "region": s["structure"], "kind": s["kind"],
                    "bucket": _bucket_for(s["structure"]), "region_bits": s["region_bits"],
                    "bit_position_tag": tag, "byte_pos": s["byte_pos"], "bit": s["bit"],
                    "within_byte": s["off"] % 4, "element": s["element"],
                    "clean_recall@1": clean.get("recall@1"), "clean_recall@10": clean["recall@10"],
                    "clean_recall@100": None, "clean_tol": None,
                    "seed": args.seed, "sub_seed": f"{args.seed}:{s['structure']}",
                })
                if rec.get("parity_abs") is not None and rec["parity_abs"] > PARITY_TOL:
                    log(f"[e1] [warn] qp<->cpp recall parity {rec['parity_abs']:.4g} > {PARITY_TOL} "
                        f"at {s['structure']}/{tag}")
                fh.write(json.dumps(rec) + "\n")
                records.append(rec)
                n += 1
                if n % 200 == 0:
                    fh.flush()
        # XOR-restore correctness: pristine re-search must reproduce the clean baseline.
        drift_tol = 1e-9 if aname == "stub" else 1e-6      # real C++ search may have float jitter
        drift = assert_no_state_leak(adapter, ref_buf, tmp, clean, cfg, timeout, drift_tol)
        open(done, "w").close()
        log(f"[e1] {n} flips in {time.time() - t0:.1f}s (restore drift {drift:.2e})")

    rows = aggregate(records)
    write_vuln_map(out, rows)
    agg = layout.aggregate_region_map(rmap)                 # priced once, fed to both consumers
    ranking = derive_criticality(rows, rmap, agg=agg)
    crit = {"adapter": aname, "criticality_order": ranking,
            "cost": cost_of_allocation(ranking, rmap, agg=agg),
            "notes": [bitclass.EX_CODE_REPORT]}
    with open(os.path.join(out, "criticality.json"), "w") as f:
        json.dump(crit, f, indent=2)
    log(f"[e1] wrote vuln_map ({len(rows)} cells) + criticality.json")
    log("[e1] Top-Down reduction order: " +
        ", ".join(f"{o['structure']}({o['scrub_tier']})" for o in ranking))
    return rows, crit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, choices=["stub", "real", "auto"],
                    help="stub (dev) / real (x86) / auto (default: real if binaries else stub)")
    ap.add_argument("--smoke", action="store_true", help="tiny sample budget + small ef")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-no-distances", action="store_true",
                    help="real adapter only: accept the documented nan-inf-undetectable degradation "
                         "instead of report-and-stop (default off)")
    ap.add_argument("--clean-tol", type=float, default=0.02,
                    help="max |clean recall@10 - operating-point anchor| before report-and-stop (real)")
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=config.PHASE2_FLIP_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = (os.path.join(config.ROOT, "artifacts_smoke", "phase3", "e1") if args.smoke
                    else os.path.join(config.ROOT, "artifacts", "phase3", "e1"))
    run(args)
    log("\nE1 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
