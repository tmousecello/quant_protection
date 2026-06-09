#!/usr/bin/env python3
"""Phase 2 — burst / clustered injection sub-study (subprocess-isolated).

Single-bit flips model an isolated DRAM soft error. Real hardware also fails in SPATIAL
CLUSTERS (a bad DIMM region, a corrupted storage block) — a contiguous run of bits. The
hypothesis: burst exposure scales with the ABSOLUTE size of the critical structure, so a
burst is far likelier to land on PQ's 128KB `pq_codebook` than on SQ8's 1KB `sq_scale`, and
an fp32 index (IVF_FLAT) has no such structure to land on at all. We sweep burst length B and
ask, per index, how often a burst causes collapse and which region it tends to hit.

For each of IVF_PQ_M8 / HNSW_SQ8 / IVF_FLAT we sweep B ∈ config.PHASE2_BURST_B (bits), and per
B run N trials with a uniformly-random start (or, with --aligned-worstcase, the start aligned
to the index's smallest critical structure). Each burst is `qp.flip.burst_flip` (self-inverse
XOR over B consecutive bits). Because a burst can straddle the header / graph structure and
segfault FAISS, EVERY trial runs in its own subprocess via qp.isolation (segfault / hang ->
`crash`); the pristine buffer is dumped once to a temp .npy the children mmap.

Collapse is RELATIVE for PQ (retention < config.PHASE2_PQ_RETENTION_FRAC of own clean) and
ABSOLUTE (ΔR@10 > buckets.CATASTROPHIC_ABS) for the aligned indexes — matching Tier 1/3.

Outputs (under <out>/burst/):
  <index>_burst.csv   per (index, B): P(collapse), ΔR@10 distribution, failure counts,
                      region-hit histogram (JSON string).

Usage:
  python phase2_burst.py                                   # full sweep (humans)
  python phase2_burst.py --smoke --indexes HNSW_SQ8
  python phase2_burst.py --aligned-worstcase --trials 50
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

import faiss

from qp import config, data, buckets, metrics
from qp import indexes as ix
from qp.flip import to_buffer, burst_positions
from qp import isolation
from phase1_sensitivity import clean_baseline
from phase1_region_accounting import load_augmented_regions


BURST_GROUP = ["IVF_PQ_M8", "HNSW_SQ8", "IVF_FLAT"]
# smallest critical structure per index — the --aligned-worstcase landing zone.
CRITICAL_KIND = {"IVF_PQ_M8": "pq_codebook", "HNSW_SQ8": "sq_scale", "IVF_FLAT": "centroid"}
SMOKE = {"trials": 4, "queries": 200, "bursts": [1, 64, 8192]}


def log(msg):
    print(msg, flush=True)


def region_index(regions):
    spans = [(r["byte_start"], r["byte_start"] + r["byte_len"], r["name"]) for r in regions]
    spans.sort()

    def hit(byte_pos):
        for s, e, nm in spans:
            if s <= byte_pos < e:
                return nm
        return "unaccounted"
    return hit


def is_collapse(rec, clean10, baseline_mode, frac):
    if rec["failure_mode"] == metrics.CRASH:
        return True
    f10 = rec["faulted_recall@10"]
    if f10 is None:
        return True
    if baseline_mode == "own":
        return f10 < frac * clean10
    return (clean10 - f10) > buckets.CATASTROPHIC_ABS


def sweep_index(name, spec, knob_val, paths, sub, args):
    t0 = time.time()
    baseline_mode = "own" if name in ("IVF_PQ_M8", "IVF_PQ_M16") else "aligned"
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob_val)
    ref_buf = to_buffer(index)
    total_bits = ref_buf.size * 8        # ref_buf is already the serialized bytes; no re-serialize
    del index
    clean = clean_baseline(ref_buf, spec, knob_val, sub["xq"], sub["gt_ids"], sub["gt_dist"])
    clean10 = clean["recall"][10]
    log(f"        {name}: clean recall@10={clean10:.4f} mode={baseline_mode} "
        f"total_bits={total_bits:,}")
    buf_path = isolation.dump_buffer(ref_buf, os.path.join(sub["dir"], f"{name}.buf.npy"))
    del ref_buf

    rmap = load_augmented_regions(name, paths["regions_dir"])
    hit_of = region_index(rmap["regions"])
    crit = next((r for r in rmap["regions"] if r["kind"] == CRITICAL_KIND[name]), None)

    raw_fh = open(paths["raw"](name), "w")
    rows = []
    n_harness = 0
    for B in args.bursts:
        if B > total_bits:
            continue
        rng = np.random.default_rng([args.seed, BURST_GROUP.index(name), B])
        trials = []
        for _ in range(args.trials):
            if args.aligned_worstcase and crit is not None:
                start_bit = crit["byte_start"] * 8
            else:
                start_bit = int(rng.integers(0, max(1, total_bits - B)))
            # keep the WHOLE burst in-bounds: an aligned start near the buffer tail (or any
            # start with start+B > total_bits) would make burst_positions index past the buffer
            # and be swallowed as a fake `crash`. B <= total_bits here (guarded above).
            start_bit = max(0, min(start_bit, total_bits - B))
            rec = isolation.measure_flip_isolated(
                buf_path, burst_positions(start_bit, B), spec["kind"], knob_val,
                sub["q_path"], sub["gt_ids_path"], sub["gt_dist_path"], clean,
                k=100, timeout=args.timeout)
            rec.update({"index": name, "B_bits": B, "start_bit": start_bit,
                        "start_region": hit_of(start_bit // 8),
                        "end_region": hit_of((start_bit + B - 1) // 8),
                        "collapse": is_collapse(rec, clean10, baseline_mode, args.pq_retention_frac)})
            raw_fh.write(json.dumps(rec) + "\n")
            trials.append(rec)
            n_harness += int(rec["failure_mode"] == isolation.HARNESS_ERROR)

        d10 = np.array([t["dRecall@10"] for t in trials if t["dRecall@10"] is not None], float)
        fm = [t["failure_mode"] for t in trials]
        n = len(trials)
        hits = {}
        for t in trials:
            hits[t["start_region"]] = hits.get(t["start_region"], 0) + 1
        rows.append({
            "index": name, "B_bits": B, "n_trials": n,
            "p_collapse": sum(t["collapse"] for t in trials) / n if n else None,
            "mean_dR@10": float(d10.mean()) if d10.size else None,
            "p99_dR@10": float(np.percentile(d10, 99)) if d10.size else None,
            "max_dR@10": float(d10.max()) if d10.size else None,
            "n_crash": fm.count(metrics.CRASH), "n_nan_inf": fm.count(metrics.NAN_INF),
            "n_silent_wrong": fm.count(metrics.SILENT_WRONG), "n_clean": fm.count(metrics.CLEAN),
            "region_hits": json.dumps(hits), "baseline_mode": baseline_mode,
            "aligned_worstcase": bool(args.aligned_worstcase), "seed": args.seed,
        })
        log(f"          B={B:>7}: P(collapse)={rows[-1]['p_collapse']:.2f} "
            f"crash={rows[-1]['n_crash']} hits={hits}")
    raw_fh.close()
    if n_harness:
        raise RuntimeError(f"{name}: {n_harness} harness-error record(s) (not corruption "
                           f"crashes) — aborting; fix the child worker.")

    with open(paths["csv"](name), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    open(paths["done"](name), "w").close()
    log(f"        {name}: {len(args.bursts)} burst lengths in {time.time()-t0:.1f}s "
        f"-> {os.path.basename(paths['csv'](name))}")
    return rows


def make_paths(art_root, out, burst_dir):
    return {
        "index": lambda n: os.path.join(art_root, "indexes", f"{n}.faissindex"),
        "regions_dir": os.path.join(art_root, "phase1", "regions_aug"),
        "csv": lambda n: os.path.join(burst_dir, f"{n}_burst.csv"),
        "raw": lambda n: os.path.join(burst_dir, f"{n}_burst.raw.jsonl"),
        "done": lambda n: os.path.join(burst_dir, f"{n}_burst.done"),
    }


def make_substrate(out, art_root, nq):
    sub_dir = os.path.join(out, "_substrate")
    os.makedirs(sub_dir, exist_ok=True)
    xq = data.load_query()[:nq].astype("float32")
    gt_ids = np.load(os.path.join(art_root, "gt", "gt_ids.npy"))[:nq]
    gt_dist = np.load(os.path.join(art_root, "gt", "gt_dist.npy"))[:nq]
    q_path = os.path.join(sub_dir, "xq.npy")
    gi = os.path.join(sub_dir, "gt_ids.npy"); gd = os.path.join(sub_dir, "gt_dist.npy")
    np.save(q_path, xq); np.save(gi, gt_ids); np.save(gd, gt_dist)
    return {"dir": sub_dir, "xq": xq, "gt_ids": gt_ids, "gt_dist": gt_dist,
            "q_path": q_path, "gt_ids_path": gi, "gt_dist_path": gd}


def main():
    ap = argparse.ArgumentParser(description="Phase 2 burst/clustered injection sub-study.")
    ap.add_argument("--indexes", default=",".join(BURST_GROUP))
    ap.add_argument("--bursts", default=None,
                    help="comma-separated burst lengths in BITS "
                         "(default: config.PHASE2_BURST_B, or the small smoke set under --smoke)")
    ap.add_argument("--trials", type=int, default=50, help="random placements per burst length")
    ap.add_argument("--aligned-worstcase", action="store_true",
                    help="start each burst at the index's smallest critical structure")
    ap.add_argument("--pq-retention-frac", type=float, default=config.PHASE2_PQ_RETENTION_FRAC)
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--timeout", type=float, default=config.PHASE2_FLIP_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # An explicit --bursts is honored even under --smoke; otherwise default to the smoke set
    # (smoke) or the full config grid. (--trials/--queries are still clamped for smoke below.)
    if args.bursts is not None:
        args.bursts = [int(x) for x in args.bursts.split(",")]
    elif args.smoke:
        args.bursts = list(SMOKE["bursts"])
    else:
        args.bursts = list(config.PHASE2_BURST_B)

    art_root = os.path.join(config.ROOT, "artifacts")
    if args.smoke:
        args.trials = min(args.trials, SMOKE["trials"])
        args.queries = min(args.queries, SMOKE["queries"])
        out = args.out or os.path.join(config.ROOT, "artifacts_smoke", "phase2")
    else:
        out = args.out or os.path.join(art_root, "phase2")
    out = os.path.abspath(out)
    burst_dir = os.path.join(out, "burst")
    os.makedirs(burst_dir, exist_ok=True)
    paths = make_paths(art_root, out, burst_dir)

    names = [n.strip() for n in args.indexes.split(",") if n.strip() and n.strip() in BURST_GROUP]
    if not names:
        log(f"[phase2-burst] no valid index in --indexes={args.indexes!r}; "
            f"must be a subset of {BURST_GROUP}")
        return 1
    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        knob_by = {r["index"]: r["knob_value"] for r in json.load(f)["indexes"]}
    sub = make_substrate(out, art_root, args.queries)

    log(f"[phase2-burst] out={out} smoke={args.smoke} queries={args.queries} indexes={names} "
        f"trials={args.trials} aligned_worstcase={args.aligned_worstcase}")

    for name in names:
        log(f"[phase2-burst] {name}: sweeping burst lengths ...")
        sweep_index(name, spec_by[name], knob_by[name], paths, sub, args)

    if args.smoke:
        # verify burst_flip is exactly self-inverse on a real buffer (restore correctness).
        index = faiss.read_index(paths["index"](names[0]))
        buf = to_buffer(index); del index
        orig = buf.copy()
        from qp.flip import burst_flip
        burst_flip(buf, 12345, 777); assert not np.array_equal(buf, orig)
        burst_flip(buf, 12345, 777); assert np.array_equal(buf, orig), "burst_flip not self-inverse"
        log("        burst_flip self-inverse on real buffer: OK")
        log("\nPHASE2-BURST SMOKE OK")
        return 0
    log("\nPHASE2-BURST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
