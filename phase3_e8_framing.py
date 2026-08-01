#!/usr/bin/env python3
"""Phase 3 — E8: the upper-link FRAMING desync (a silent-collapse cliff the sweeps never sampled).

E1 and E6 sample level0 and the globals (rotation / centroids). Neither ever touches the
`upper_links` block, and none of the three protection layers does either — they protect index
DATA once it is loaded, not the FRAMING that tells the loader where that data ends. This driver
measures what lives in that blind spot.

THE MECHANISM (RaBitQ-Library `HierarchicalNSW`, third_party/RaBitQ-Library/include/rabitqlib/
index/hnsw/hnsw.hpp at the pinned commit):

  save()  L655-662  for each element: write `uint32 link_list_size` then that many bytes, where
                    link_list_size = size_links_per_element_ * element_levels_[i], or 0.
                    rotator_->save(output) then appends the rotation at L664.
  load()  L737-750  for each element: `input.read(&link_list_size, 4)`, `malloc(link_list_size)`,
                    `input.read(linkLists_[i], link_list_size)` — with NO bound on the value and
                    NO stream-health check — then `rotator_->load(input)` at L763 reads the
                    rotation from the SAME sequential stream.

The block has no index and no padding, so record i's position is the running sum of (4 + len) over
every earlier record. A single bit flipped in any length word therefore changes how many bytes that
record consumes and desynchronizes every later record AND the rotation read. Because ifstream
signals failure by setting failbit rather than by throwing, every subsequent `read` silently
no-ops: the loader finishes, `main` returns 0, the search runs, and recall is ~0. Nothing anywhere
reports a problem.

WHY IT IS 100% BY CONSTRUCTION, NOT BY LUCK: any bit flip changes the length value, hence changes
(4 + len), hence desynchronizes the stream. There is no benign flip in a length word. The sweep
below confirms that empirically rather than assuming it.

EXPOSURE. The length words are 4 bytes x cur_element_count = 4 MB on SIFT1M b=7 — 46.7% of the
upper_links block and 1.43% of the whole index, versus 64 bytes for the rotation (the study's
canonical single-point catastrophe). That is ~62,500x the rotation's byte area at the same
silent-collapse severity. NOTE the framing bytes are NOT a contiguous 4 MB slab: they are 1M
4-byte prefixes interleaved with variable-length payload, which is why this driver locates them by
PARSING the record stream (qp.rabitq.layout.parse_upper_link_records) instead of assuming a range.

Strata (single-bit flips; `rotation` and `ex_code` are the known-cliff / known-benign controls):
  len_word_any   uniform over ALL 4 MB of length-word bytes  -> the area-weighted headline
  len_byte0      byte lane 0 of a length word (len +/- 1..128)      -> the MISALIGN sub-mode
  len_byte3      byte lane 3 of a length word (len +/- 2^24..2^31)  -> the OVERCOMMIT sub-mode
  upper_payload  uniform over the upper-link PAYLOAD bytes    -> control, expected benign
  rotation       the 64-byte tail                             -> control, known cliff
  ex_code        a random element's ex_code                   -> control, known benign

Every row also carries what the flip did to the length value (clean_len, corrupt_len, len_delta)
and `bound_detects` — whether the framing guard's bound (link_list_size is 0 or a multiple of
size_links_per_element_ with quotient <= maxlevel_) would have REJECTED that exact corrupted word.
That column measures the fix's detection coverage on the very flips that produced the collapses,
without needing a rebuilt binary.

Outputs (under --out; default artifacts/phase3/e8, --smoke -> artifacts_smoke/phase3/e8):
  raw/e8.records.jsonl   one row per eval
  raw/e8.done            completion marker
  e8_results.csv         the per-eval table
  e8_summary.json        provenance meta + area exposure + per-stratum rates + gates

Usage:
  python phase3_e8_framing.py --adapter stub --smoke    # dev pipeline gate (plumbing only)
  python phase3_e8_framing.py                           # workstation run (auto -> real)
"""
import argparse
import csv
import json
import os
import struct
import sys
import time
import zlib

import numpy as np

from qp import config, metrics, provenance
from qp.bits import flip_bit
from qp.rabitq import layout
from qp.rabitq.registry import get_adapter, adapter_name
from qp.rawio import RawWriter

STRATA = ("len_word_any", "len_byte0", "len_byte3", "upper_payload", "rotation", "ex_code")

# Strata whose flips land in a length word BY CONSTRUCTION. The sweep asserts this against the
# independently-parsed record stream, so a sampler bug shows up as a gate failure rather than as
# a quietly mis-attributed collapse rate.
FRAMING_STRATA = ("len_word_any", "len_byte0", "len_byte3")

OUTCOMES = ("crash", "nan_inf", "silent_collapse", "silent_degraded", "benign")

# ef=64 sits at recall@10 ~= 0.95 and costs ~0.7 s/eval, versus ~15 s at the ef=2000 plateau. The
# effect measured here is a 5-orders-of-magnitude collapse, so the operating point is irrelevant
# to whether it reproduces and the cheap one buys ~20x more seeds per unit of wall clock.
FULL = {"seeds": 30, "ef": 64}          # 30 = the study's per-cell replicate convention
SMOKE = {"seeds": 2, "ef": 64}

# The index's reference recall plateau (adapter.EXPECTED_CLEAN_RECALL10 ~= 0.983 is quoted there).
# Measured once per run purely to confirm the platform/index, never used as a sweep baseline.
PLATEAU_EF = 2000

# Recall within DELTA_TOL of clean counts as benign (the phase1/E1 "harmful" bar). Collapse uses
# qp.metrics.is_silent_collapse (retention < config.COLLAPSE_RETENTION_FRAC), the study-wide rule.
DELTA_TOL = 0.01

CSV_COLUMNS = ["stratum", "seed_index", "seed", "byte_pos", "bit", "hit_len_word", "element",
               "len_lane", "clean_len", "corrupt_len", "len_delta", "bound_detects",
               "stream_records_walked", "stream_truncated", "stream_desynced",
               "recall", "delta_recall", "retention", "outcome", "cpp_recall", "wall_s", "error"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pure, seed-deterministic pieces (unit-tested in artifacts/phase3/tests/test_e8_framing.py)
# ---------------------------------------------------------------------------

def cell_seed(root_seed, stratum, index):
    """Deterministic per-(root, stratum, replicate) seed (same construction E6 uses)."""
    return int(np.random.SeedSequence([int(root_seed), zlib.crc32(stratum.encode()),
                                       int(index)]).generate_state(1)[0])


def framing_geometry(clean_buf, rmap):
    """Locate every length word and every payload byte in the upper_links block.

    Returns a dict with the parsed record stream plus the derived area accounting. The parse
    self-checks (sum of 4+len must tile the region exactly), so a successful return IS the
    region-math correctness proof.
    """
    hdr = rmap["header"]
    up = next(r for r in rmap["regions"] if r["name"] == "upper_links")
    rot = next(r for r in rmap["regions"] if r["name"] == "rotation")
    if up["byte_len"] <= 0:
        raise RuntimeError(
            "REPORT-AND-STOP: this index has a zero-length upper_links block, so there is no "
            "record framing to corrupt and E8 has nothing to measure.")
    recs = layout.parse_upper_link_records(clean_buf, hdr["cur_element_count"],
                                           up["byte_start"], up["byte_len"])
    lens = np.asarray(recs["lens"], dtype=np.int64)
    offsets = np.asarray(recs["len_offsets"], dtype=np.int64)
    total = int(clean_buf.size)
    return {
        "upper_start": int(up["byte_start"]),
        "upper_len": int(up["byte_len"]),
        "len_offsets": offsets,
        "lens": lens,
        "payload_cumsum": np.cumsum(lens),
        "framing_bytes": int(recs["framing_bytes"]),
        "payload_bytes": int(recs["payload_bytes"]),
        "rotation_start": int(rot["byte_start"]),
        "rotation_bytes": int(rot["byte_len"]),
        "index_bytes": total,
        "valid_lens": layout.valid_link_list_sizes(hdr["size_links_per_element"],
                                                   hdr["maxlevel"]),
        "n_elements": int(hdr["cur_element_count"]),
    }


def classify_len_word(byte_pos, geom):
    """(element, lane) if `byte_pos` is inside a length word, else (None, None).

    Binary search over the parsed record starts — the framing bytes are scattered 4-byte prefixes,
    so membership cannot be decided by a range test.
    """
    offs = geom["len_offsets"]
    i = int(np.searchsorted(offs, byte_pos, side="right")) - 1
    if i < 0:
        return None, None
    lane = int(byte_pos) - int(offs[i])
    if 0 <= lane < layout.LEN_WORD_BYTES:
        return i, lane
    return None, None


def sample_position(stratum, rng, geom, rmap):
    """One (byte_pos, bit) single-bit flip target for `stratum`. Pure and rng-deterministic."""
    if stratum == "len_word_any":
        # Uniform over the framing BYTES (each element contributes exactly 4), so the draw is
        # area-weighted the way a DRAM per-bit error rate would deliver it.
        idx = int(rng.integers(geom["framing_bytes"]))
        e, lane = divmod(idx, layout.LEN_WORD_BYTES)
        pos = int(geom["len_offsets"][e]) + lane
    elif stratum in ("len_byte0", "len_byte3"):
        lane = 0 if stratum == "len_byte0" else 3
        e = int(rng.integers(geom["n_elements"]))
        pos = int(geom["len_offsets"][e]) + lane
    elif stratum == "upper_payload":
        if geom["payload_bytes"] <= 0:
            raise RuntimeError("REPORT-AND-STOP: no upper-link payload bytes to sample")
        j = int(rng.integers(geom["payload_bytes"]))
        e = int(np.searchsorted(geom["payload_cumsum"], j, side="right"))
        before = int(geom["payload_cumsum"][e - 1]) if e else 0
        pos = int(geom["len_offsets"][e]) + layout.LEN_WORD_BYTES + (j - before)
    elif stratum == "rotation":
        pos = geom["rotation_start"] + int(rng.integers(geom["rotation_bytes"]))
    elif stratum == "ex_code":
        e = int(rng.integers(rmap["header"]["cur_element_count"]))
        start, flen = layout.element_field_range(rmap, "ex_code", e)
        pos = int(start) + int(rng.integers(flen))
    else:
        raise ValueError(f"unknown stratum {stratum!r}; choose from {list(STRATA)}")
    return pos, int(rng.integers(8))


def simulate_load_walk(buf, geom):
    """Emulate load()'s sequential record walk over a (possibly corrupted) buffer.

    Follows exactly what hnsw.hpp L737-750 does — read a uint32, skip that many bytes, repeat —
    against the bytes as they now stand, and reports where the stream runs out. Everything after
    the upper_links start belongs to this stream (the records, then the rotation at L763), so the
    budget is `buf.size - upper_start`.

    Returns {"records_walked", "consumed", "truncated", "desynced"}:
      records_walked  how many records the loader gets through before a read would run past EOF;
                      on the C++ side that read sets failbit and every read after it no-ops.
      truncated       the stream ran out (the read that fails is what makes the loss SILENT).
      desynced        the walk did not land exactly on the true end of upper_links, so the
                      rotation is read from the wrong offset (or not at all). This is the
                      universal consequence of a length flip: (4 + len) changes, so the total
                      consumed changes even when the stream never runs out.

    This is a PYTHON EMULATION of the read sequence, reported alongside the measured outcome,
    never as a substitute for it.
    """
    mv = memoryview(buf)
    start = geom["upper_start"]
    avail = int(buf.size) - start
    off = 0
    for i in range(geom["n_elements"]):
        if off + layout.LEN_WORD_BYTES > avail:
            return {"records_walked": i, "consumed": off, "truncated": True, "desynced": True}
        (L,) = struct.unpack_from("<I", mv, start + off)
        off += layout.LEN_WORD_BYTES
        if L:
            if off + L > avail:
                return {"records_walked": i, "consumed": off, "truncated": True, "desynced": True}
            off += L
    return {"records_walked": geom["n_elements"], "consumed": off,
            "truncated": off + geom["rotation_bytes"] > avail,
            "desynced": off != geom["upper_len"]}


def read_len_word(buf, geom, element):
    """The uint32 length value of `element` as it currently stands in `buf`."""
    off = int(geom["len_offsets"][element])
    return int(struct.unpack_from("<I", memoryview(buf)[off:off + layout.LEN_WORD_BYTES])[0])


def bound_coverage(valid_lens):
    """Exhaustive single-bit detection coverage of the framing guard's bound.

    For every length value a legitimate index can contain, flip each of the 32 bits and ask
    whether the result is still a valid length. Anything that is NOT is rejected by the bound, so
    the flip becomes a loud load failure instead of a silent desync. Enumerated, not sampled.
    """
    valid = set(int(v) for v in valid_lens)
    total = len(valid) * 32
    missed = [(v, b) for v in sorted(valid) for b in range(32) if (v ^ (1 << b)) in valid]
    return {
        "valid_lengths": sorted(valid),
        "cases_enumerated": total,
        "undetected_cases": [{"clean_len": v, "bit": b, "corrupt_len": v ^ (1 << b)}
                             for v, b in missed],
        "detected_fraction": (total - len(missed)) / total if total else None,
    }


def classify_outcome(recall, clean_recall, distances, crashed):
    """crash / nan_inf / silent_collapse / silent_degraded / benign for one eval."""
    if crashed:
        return "crash"
    mode = metrics.classify_failure(distances=distances, recall=recall,
                                    clean_recall=clean_recall)
    if mode == metrics.NAN_INF:
        return "nan_inf"
    if metrics.is_silent_collapse(recall, clean_recall, failure_mode=mode):
        return "silent_collapse"
    return "benign" if recall >= clean_recall - DELTA_TOL else "silent_degraded"


# ---------------------------------------------------------------------------
# Sandbox / CLI helpers (same house conventions as E6)
# ---------------------------------------------------------------------------

def stub_sandbox_ok(aname, out_dir):
    """False iff the stub would write documented-fake results into the real artifacts tree."""
    if aname != "stub":
        return True
    real_root = os.path.join(os.path.abspath(config.ROOT), "artifacts")
    target = os.path.abspath(out_dir)
    return not (target == real_root or target.startswith(real_root + os.sep))


def assert_stub_sandbox(aname, out_dir):
    if not stub_sandbox_ok(aname, out_dir):
        raise RuntimeError(
            f"REPORT-AND-STOP: stub adapter would write documented-fake results into the real "
            f"artifacts tree ({os.path.abspath(out_dir)}). Use --smoke (-> artifacts_smoke/) "
            f"or an explicit --out.")


def csv_list(kind, allowed):
    """argparse type for a comma-separated subset of `allowed` (a typo must be a loud error)."""
    def parse(s):
        vals = tuple(x.strip() for x in str(s).split(",") if x.strip())
        if not vals:
            raise argparse.ArgumentTypeError(f"empty {kind} list")
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(
                f"unknown {kind}: {bad} (choose from {list(allowed)})")
        return tuple(v for v in allowed if v in vals)
    return parse


def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 6) if vals else None


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_context(args, cfg):
    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)
    out = os.path.abspath(args.out)
    assert_stub_sandbox(aname, out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    tag = f"_{args.out_tag}" if getattr(args, "out_tag", None) else ""

    clean_buf = adapter.serialize_index()
    rmap = adapter.region_map()
    gt = adapter.load_groundtruth()
    geom = framing_geometry(clean_buf, rmap)
    tmp = os.path.join(out, "raw", f"_e8{tag}.index")

    # Clean baseline through the SAME path the sweep uses, so delta_recall is like-for-like.
    adapter.deserialize_index(clean_buf, tmp)
    t0 = time.perf_counter()
    clean_res = adapter.search_corrupted(tmp, k=config.K, ef=int(cfg["ef"]),
                                         out_path=tmp + ".clean.ivecs", timeout=args.timeout)
    clean_wall = time.perf_counter() - t0
    clean_recall = float(metrics.recall_at_k(clean_res["ids"], gt, config.K))
    if clean_recall <= 0:
        raise RuntimeError(f"REPORT-AND-STOP: clean recall@10 is {clean_recall} — the baseline "
                           f"search is broken, so every delta below would be meaningless.")

    # provenance's platform triangulation compares clean recall against the index's PLATEAU anchor
    # (~0.983). The sweep runs at ef=64 (~0.950) on purpose, which is off that anchor, so feeding
    # it the sweep's number would report platform_confirmed_real=False for a run that is entirely
    # real. Measure the plateau once and confirm against THAT; the sweep's operating point is
    # recorded separately so the two are never conflated.
    plateau = None
    if not args.skip_plateau_check:
        t0 = time.perf_counter()
        plateau_res = adapter.search_corrupted(tmp, k=config.K, ef=PLATEAU_EF,
                                               out_path=tmp + ".plateau.ivecs",
                                               timeout=args.timeout)
        plateau = {"ef": PLATEAU_EF,
                   "recall@10": float(metrics.recall_at_k(plateau_res["ids"], gt, config.K)),
                   "wall_s": round(time.perf_counter() - t0, 3)}
        log(f"[e8] plateau check: ef={PLATEAU_EF} clean@10={plateau['recall@10']:.6f} "
            f"(anchor {getattr(adapter, 'EXPECTED_CLEAN_RECALL10', None)}) "
            f"[{plateau['wall_s']:.1f}s]")

    meta = provenance.collect_provenance(
        adapter, aname, {"recall@10": (plateau or {}).get("recall@10", clean_recall)},
        cfg, args, rmap,
        index_sha256=provenance.sha256_file(tmp), corrupted_regions=list(STRATA),
        recovery="off")
    meta["plateau_check"] = plateau
    meta["sweep_operating_point"] = {"ef": int(cfg["ef"]), "clean_recall@10": clean_recall,
                                     "note": "the sweep runs here; the plateau above is the "
                                             "platform/index confirmation anchor"}
    meta["loader_citation"] = {
        "file": "third_party/RaBitQ-Library/include/rabitqlib/index/hnsw/hnsw.hpp",
        "save_link_list_size": "L655-662",
        "load_link_list_loop": "L737-750 (no bound on link_list_size, no stream-health check)",
        "load_rotation_read": "L763 (same sequential stream, after the link-list loop)",
        "maxlevel_invariant": "L920/L926 (maxlevel_ raised to curlevel => levels <= maxlevel_)",
    }
    meta["units"] = dict(meta.get("units", {}),
                         silent_collapse_rate="fraction_0_1",
                         framing_frac_of_index="fraction_0_1",
                         bound_detect_rate="fraction_0_1")

    log(f"[e8] adapter={aname} out={out} clean@10={clean_recall:.6f} ({clean_wall:.1f}s/eval) "
        f"ef={cfg['ef']}")
    log(f"[e8] upper_links @{geom['upper_start']} len={geom['upper_len']} "
        f"= {geom['framing_bytes']} framing + {geom['payload_bytes']} payload "
        f"({geom['framing_bytes'] / geom['upper_len'] * 100:.2f}% framing); "
        f"framing is {geom['framing_bytes'] / geom['index_bytes'] * 100:.4f}% of the index "
        f"and {geom['framing_bytes'] / geom['rotation_bytes']:.0f}x the rotation's area")
    return {"adapter": adapter, "aname": aname, "out": out, "tag": tag, "tmp": tmp,
            "clean_buf": clean_buf, "rmap": rmap, "gt": gt, "geom": geom,
            "clean_recall": clean_recall, "clean_wall_s": round(clean_wall, 3), "meta": meta,
            "timeout": args.timeout}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def measure_one(ctx, cfg, stratum, i, work, writer):
    """One single-bit eval: flip -> deserialize -> search -> recall -> restore + verify."""
    geom, adapter = ctx["geom"], ctx["adapter"]
    seed = cell_seed(cfg["seed"], stratum, i)
    rng = np.random.default_rng(seed)
    pos, bit = sample_position(stratum, rng, geom, ctx["rmap"])
    element, lane = classify_len_word(pos, geom)

    row = {"stratum": stratum, "seed_index": i, "seed": seed, "byte_pos": int(pos), "bit": bit,
           "hit_len_word": element is not None, "element": element, "len_lane": lane,
           "clean_len": None, "corrupt_len": None, "len_delta": None, "bound_detects": None,
           "stream_records_walked": None, "stream_truncated": None, "stream_desynced": None,
           "recall": None, "delta_recall": None, "retention": None, "outcome": None,
           "cpp_recall": None, "wall_s": None, "error": None}

    flip_bit(work, pos, bit)
    try:
        if element is not None:
            clean_len = int(geom["lens"][element])
            corrupt_len = read_len_word(work, geom, element)
            walk = simulate_load_walk(work, geom)
            row.update(clean_len=clean_len, corrupt_len=corrupt_len,
                       len_delta=corrupt_len - clean_len,
                       bound_detects=corrupt_len not in geom["valid_lens"],
                       stream_records_walked=walk["records_walked"],
                       stream_truncated=walk["truncated"],
                       stream_desynced=walk["desynced"])
        adapter.deserialize_index(work, ctx["tmp"])
        t0 = time.perf_counter()
        crashed = False
        try:
            res = adapter.search_corrupted(ctx["tmp"], k=config.K, ef=int(cfg["ef"]),
                                           out_path=ctx["tmp"] + ".ids.ivecs",
                                           timeout=ctx["timeout"])
        except Exception as exc:                                        # noqa: BLE001
            crashed, res = True, None
            row["error"] = repr(exc)[:400]
        row["wall_s"] = round(time.perf_counter() - t0, 3)
        if res is not None:
            recall = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
            row["recall"] = recall
            row["cpp_recall"] = res.get("cpp_recall")
            row["delta_recall"] = round(ctx["clean_recall"] - recall, 6)
            row["retention"] = round(recall / ctx["clean_recall"], 6)
        row["outcome"] = classify_outcome(row["recall"], ctx["clean_recall"],
                                          (res or {}).get("distances"), crashed)
    finally:
        flip_bit(work, pos, bit)                       # XOR is self-inverse
        leak = int(np.count_nonzero(work != ctx["clean_buf"]))
        if leak:
            raise RuntimeError(f"REPORT-AND-STOP: state leak after {stratum}[{i}] — {leak} byte(s) "
                               f"still differ from clean; later evals would be contaminated.")
    writer.write(row)
    return row


def run(args):
    cfg = dict(SMOKE if args.smoke else FULL)
    if args.seeds is not None:
        cfg["seeds"] = int(args.seeds)
    if args.ef is not None:
        cfg["ef"] = int(args.ef)
    cfg["seed"] = int(args.seed)
    strata = args.strata or STRATA

    ctx = setup_context(args, cfg)
    work = ctx["clean_buf"].copy()
    raw = os.path.join(ctx["out"], "raw", f"e8{ctx['tag']}.records.jsonl")
    records = []
    t_start = time.perf_counter()
    with RawWriter(raw, done_path=os.path.join(ctx["out"], "raw", f"e8{ctx['tag']}.done")) as w:
        for stratum in strata:
            for i in range(int(cfg["seeds"])):
                row = measure_one(ctx, cfg, stratum, i, work, w)
                records.append(row)
                log(f"[e8] {stratum:<14} #{i:<2} byte={row['byte_pos']} bit={row['bit']} "
                    f"len {row['clean_len']}->{row['corrupt_len']} "
                    f"recall={row['recall']} {row['outcome']}")
    log(f"[e8] {len(records)} evals in {time.perf_counter() - t_start:.1f}s")
    return summarize(ctx, cfg, strata, records)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(ctx, cfg, strata, records):
    out, tag, geom = ctx["out"], ctx["tag"], ctx["geom"]

    with open(os.path.join(out, f"e8_results{tag}.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        wr.writeheader()
        for r in records:
            wr.writerow(r)

    cells = {}
    for s in strata:
        rows = [r for r in records if r["stratum"] == s]
        if not rows:
            continue
        n = len(rows)
        framed = [r for r in rows if r["hit_len_word"]]
        n_collapse = sum(1 for r in rows if r["outcome"] == "silent_collapse")
        n_crash = sum(1 for r in rows if r["outcome"] == "crash")
        cells[s] = {
            "n": n,
            # SILENT collapse: exit 0, no diagnostic, recall gone. A crash is also a total loss
            # but it is LOUD, so the two are counted separately and `damaging_rate` is their sum.
            "silent_collapse_rate": round(n_collapse / n, 6),
            "n_silent_collapse": n_collapse,
            "crash_rate": round(n_crash / n, 6),
            "damaging_rate": round((n_collapse + n_crash) / n, 6),
            "outcomes": {o: sum(1 for r in rows if r["outcome"] == o)
                         for o in OUTCOMES if any(r["outcome"] == o for r in rows)},
            "median_recall": _median([r["recall"] for r in rows]),
            "median_delta_recall": _median([r["delta_recall"] for r in rows]),
            "median_retention": _median([r["retention"] for r in rows]),
            "n_hit_len_word": len(framed),
            "bound_detect_rate": (round(sum(1 for r in framed if r["bound_detects"]) / len(framed),
                                        6) if framed else None),
            "desync_rate": (round(sum(1 for r in framed if r["stream_desynced"]) / len(framed), 6)
                            if framed else None),
            "truncation_rate": (round(sum(1 for r in framed if r["stream_truncated"])
                                      / len(framed), 6) if framed else None),
        }

    exposure = {
        "index_bytes": geom["index_bytes"],
        "upper_links_bytes": geom["upper_len"],
        "framing_bytes": geom["framing_bytes"],
        "payload_bytes": geom["payload_bytes"],
        "framing_frac_of_upper_links": round(geom["framing_bytes"] / geom["upper_len"], 6),
        "framing_frac_of_index": round(geom["framing_bytes"] / geom["index_bytes"], 8),
        "rotation_bytes": geom["rotation_bytes"],
        "framing_over_rotation_area": round(geom["framing_bytes"] / geom["rotation_bytes"], 1),
        "n_elements": geom["n_elements"],
        "note": ("framing bytes are 4-byte length prefixes INTERLEAVED with variable-length "
                 "payload, not a contiguous slab; located by parsing the record stream"),
    }

    # Gates that can actually fail. Each is derived from the records, never asserted as a constant.
    framing_rows = [r for r in records if r["stratum"] in FRAMING_STRATA]
    payload_rows = [r for r in records if r["stratum"] == "upper_payload"]
    gates = {
        "framing_sampler_always_hits_a_length_word":
            all(r["hit_len_word"] for r in framing_rows) if framing_rows else None,
        "payload_sampler_never_hits_a_length_word":
            all(not r["hit_len_word"] for r in payload_rows) if payload_rows else None,
        "every_length_flip_changed_the_value":
            all(r["len_delta"] not in (None, 0) for r in framing_rows) if framing_rows else None,
        # The claim the whole finding rests on: a corrupt length word desynchronizes the stream.
        "every_length_flip_desynced_the_stream":
            all(r["stream_desynced"] for r in framing_rows) if framing_rows else None,
        "no_length_flip_was_benign":
            all(r["outcome"] != "benign" for r in framing_rows) if framing_rows else None,
        "row_count_ok": len(records) == len(strata) * int(cfg["seeds"]),
        "state_leak_bytes": 0,
        "stub_sandbox_ok": stub_sandbox_ok(ctx["aname"], out),
    }

    # How far the desynced walk gets before overrunning separates the two failure manifestations
    # cleanly in the measurement; the downstream reason it does is NOT established here, so this
    # is reported as an observed separation, not as an explanation.
    walked = {o: sorted(r["stream_records_walked"] for r in framing_rows
                        if r["outcome"] == o and r["stream_records_walked"] is not None)
              for o in ("silent_collapse", "crash")}
    silent_vs_crash = {
        "silent_records_walked_max": walked["silent_collapse"][-1] if walked["silent_collapse"]
        else None,
        "crash_records_walked_min": walked["crash"][0] if walked["crash"] else None,
        "separates_cleanly": (bool(walked["silent_collapse"] and walked["crash"]
                                   and walked["silent_collapse"][-1] < walked["crash"][0])
                              if (walked["silent_collapse"] and walked["crash"]) else None),
        "note": ("records the emulated loader walks before the stream runs out; a crash is a "
                 "SIGSEGV during search, not a diagnosed load failure"),
    }

    summary = {
        "experiment": "e8_framing_desync",
        "silent_vs_crash": silent_vs_crash,
        "clean_recall@10": ctx["clean_recall"],
        "clean_wall_s": ctx["clean_wall_s"],
        "cfg": cfg,
        "strata": list(strata),
        "area_exposure": exposure,
        "bound_coverage_analytic": bound_coverage(geom["valid_lens"]),
        "cells": cells,
        "gates": gates,
        "adapter": ctx["aname"],
        "meta": ctx["meta"],
    }
    with open(os.path.join(out, f"e8_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    log("")
    log(f"{'stratum':<15} {'n':>3} {'silent':>8} {'crash':>7} {'damaging':>9} "
        f"{'med recall':>11} {'bound det':>10}")
    for s, c in cells.items():
        bd = "-" if c["bound_detect_rate"] is None else f"{c['bound_detect_rate'] * 100:.0f}%"
        log(f"{s:<15} {c['n']:>3} {c['silent_collapse_rate'] * 100:>7.0f}% "
            f"{c['crash_rate'] * 100:>6.0f}% {c['damaging_rate'] * 100:>8.0f}% "
            f"{str(c['median_recall']):>11} {bd:>10}")
    bad = [k for k, v in gates.items() if v is False]
    if bad:
        raise RuntimeError(f"REPORT-AND-STOP: E8 gates failed: {bad}")
    log(f"[e8] wrote e8_results{tag}.csv ({len(records)} rows) + e8_summary{tag}.json")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, choices=["stub", "real", "auto"],
                    help="stub (dev) / real (x86) / auto (default: real if binaries else stub)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE['seeds']} seeds per stratum; outputs to artifacts_smoke/")
    ap.add_argument("--seeds", type=int, default=None, help="replicates per stratum")
    ap.add_argument("--strata", type=csv_list("stratum", STRATA), default=None,
                    help=f"comma-separated subset of {list(STRATA)}")
    ap.add_argument("--seed", type=int, default=config.SEED, help="root seed")
    ap.add_argument("--ef", type=int, default=None,
                    help=f"efSearch (default {FULL['ef']}; the collapse is 5 orders of magnitude "
                         f"so the operating point does not change the finding)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-search seconds; a hang under corruption records as crash")
    ap.add_argument("--skip-plateau-check", dest="skip_plateau_check", action="store_true",
                    help=f"skip the one-off ef={PLATEAU_EF} clean search that confirms the index "
                         f"sits on its reference plateau (costs ~15 s; without it "
                         f"platform_confirmed_real cannot be established)")
    ap.add_argument("--out-tag", dest="out_tag", default=None,
                    help="filename suffix so parallel/variant runs never clobber each other")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                                "phase3", "e8")
    run(args)
    log("\nE8 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
