#!/usr/bin/env python3
"""Phase 2 — Tier 2: graph_edges injection characterization (subprocess-isolated).

The sibling project first-try found that damaging HNSW graph structure mostly costs *speed*,
not recall. This pins that down for the corruption study by asking, for a flip inside the
`graph_edges` adjacency table, which of three things happens:

  crash          the corrupted id is out of bounds -> FAISS C++ dereferences it -> SIGSEGV
                 (DETECTABLE — the process dies; a guard or a checksum would catch it).
  silent-benign  the id is still a valid node, just a different/worse neighbour -> recall
                 barely moves (|ΔR@10| ~ 0). The "graph is a sponge" case.
  silent-harmful the misroute actually costs recall (ΔR@10 above --harmful-thresh). The
                 dangerous case: no crash, no NaN, recall quietly drops.

Each neighbour id is a little-endian int32. We tag the hit bit `id_high` (bytes 2-3 — node
ids on SIFT1M are < 2^21 so these bytes are ~zero; setting them yields a huge OOB id ->
crash) vs `id_low` (bytes 0-1 — a valid but wrong neighbour -> silent), and also record the
exact (within_byte, bit).

Because an OOB id segfaults the whole process, EVERY flip is evaluated in its own subprocess
via qp.isolation: a hard C++ crash kills only the child (recorded as `crash`) and a hang past
the timeout is terminated and also recorded as `crash`. The pristine 260MB+ buffer is dumped
once to a temp .npy that children mmap, so it is never piped or mutated.

Outputs (under --out, default artifacts/phase2; smoke -> artifacts_smoke/phase2):
  graph_edges_characterization.csv / .json   per (index x region x tag) + an ALL-tag rollup,
                                              with the 3-way split and failure counts.

Usage:
  python phase2_graph_sensitivity.py                       # full run (humans)
  python phase2_graph_sensitivity.py --smoke --indexes HNSW_SQ8
  python phase2_graph_sensitivity.py --n-edges 4000 --queries 10000
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
from qp.flip import to_buffer
from qp import isolation
from phase1_sensitivity import clean_baseline
from phase1_region_accounting import load_augmented_regions


GRAPH_GROUP = ["HNSW", "HNSW_SQ8"]
SMOKE = {"n_edges": 8, "queries": 200}


def log(msg):
    print(msg, flush=True)


def edge_tag(off_in_region, bit):
    """int32 neighbour-id bit tag. bytes 2-3 -> id_high (OOB-prone), bytes 0-1 -> id_low."""
    within = off_in_region % 4
    return "id_high" if within >= 2 else "id_low"


def plan_edge_samples(region, n_edges, rng):
    """Sample byte positions in graph_edges, stratified across the 4 int32 byte lanes so
    both id_high (high bytes) and id_low (low bytes) get equal coverage."""
    bs, bl = region["byte_start"], region["byte_len"]
    n_elems = max(1, bl // 4)
    per = max(1, n_edges // 4)
    out = []
    for within in range(4):                      # 0,1 -> id_low ; 2,3 -> id_high
        for _ in range(per):
            off = int(rng.integers(n_elems)) * 4 + within
            if off >= bl:
                continue
            bit = int(rng.integers(8))
            out.append({"byte_pos": int(bs + off), "bit": bit,
                        "within_byte": within, "bit_position_tag": edge_tag(off, bit)})
    return out


def classify_3way(rec, harmful_thresh):
    """Collapse a flip record into crash / nan-inf / silent-harmful / silent-benign."""
    if rec["failure_mode"] == metrics.CRASH:
        return "crash"
    if rec["failure_mode"] == metrics.NAN_INF:
        return "nan-inf"
    d10 = rec["dRecall@10"]
    if d10 is not None and d10 > harmful_thresh:
        return "silent-harmful"
    return "silent-benign"


def aggregate(records, harmful_thresh):
    groups = {}
    for r in records:
        for key in [(r["index"], r["region"], r["bit_position_tag"]),
                    (r["index"], r["region"], "ALL")]:
            groups.setdefault(key, []).append(r)

    rows = []
    for (index, region, tag), recs in sorted(groups.items()):
        d10 = np.array([r["dRecall@10"] for r in recs if r["dRecall@10"] is not None], float)
        three = [classify_3way(r, harmful_thresh) for r in recs]
        fm = [r["failure_mode"] for r in recs]
        n_total = len(recs)
        c = {k: three.count(k) for k in
             ("crash", "nan-inf", "silent-harmful", "silent-benign")}
        rows.append({
            "index": index, "region": region, "bit_position_tag": tag,
            "region_bits": recs[0]["region_bits"], "n": n_total, "n_samples": n_total,
            "pct_crash": 100.0 * c["crash"] / n_total,
            "pct_silent_benign": 100.0 * c["silent-benign"] / n_total,
            "pct_silent_harmful": 100.0 * c["silent-harmful"] / n_total,
            "pct_nan_inf": 100.0 * c["nan-inf"] / n_total,
            "mean_dR@10": float(d10.mean()) if d10.size else None,
            "p99_dR@10": float(np.percentile(d10, 99)) if d10.size else None,
            "max_dR@10": float(d10.max()) if d10.size else None,
            "n_crash": fm.count(metrics.CRASH),
            "n_silent_wrong": fm.count(metrics.SILENT_WRONG),
            "n_nan_inf": fm.count(metrics.NAN_INF),
            "n_clean": fm.count(metrics.CLEAN),
            "harmful_thresh": harmful_thresh, "baseline_mode": "aligned",
            "seed": recs[0]["seed"],
        })
    return rows


def write_out(out, rows):
    with open(os.path.join(out, "graph_edges_characterization.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    if rows:
        with open(os.path.join(out, "graph_edges_characterization.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def sweep_index(name, spec, knob_val, paths, sub, args):
    t0 = time.time()
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob_val)
    ref_buf = to_buffer(index)
    del index

    # clean baseline measured in-process (graph indexes deserialize cleanly).
    clean = clean_baseline(ref_buf, spec, knob_val, sub["xq"], sub["gt_ids"], sub["gt_dist"])
    log(f"        {name}: clean recall@10={clean['recall'][10]:.4f}")
    buf_path = isolation.dump_buffer(ref_buf, os.path.join(sub["dir"], f"{name}.buf.npy"))
    del ref_buf

    rmap = load_augmented_regions(name, paths["regions_dir"])
    edges = [r for r in rmap["regions"] if r["kind"] == "graph_edges"]
    if not edges:
        log(f"        {name}: no graph_edges region (skipping)")
        return []
    region = edges[0]
    rng = np.random.default_rng([args.seed, GRAPH_GROUP.index(name), 0])
    samples = plan_edge_samples(region, args.n_edges, rng)
    log(f"        {name}/graph_edges -> {len(samples)} isolated edge flips")

    shard = paths["raw"](name)
    records = []
    with open(shard, "w") as fh:
        for i, s in enumerate(samples):
            rec = isolation.measure_flip_isolated(
                buf_path, [(s["byte_pos"], s["bit"])], spec["kind"], knob_val,
                sub["q_path"], sub["gt_ids_path"], sub["gt_dist_path"], clean,
                k=100, timeout=args.timeout)
            rec.update({
                "index": name, "region": region["name"], "kind": region["kind"],
                "region_bits": buckets.region_bits(region),
                "byte_pos": s["byte_pos"], "bit": s["bit"],
                "within_byte": s["within_byte"], "bit_position_tag": s["bit_position_tag"],
                "three_way": classify_3way(rec, args.harmful_thresh), "seed": args.seed,
            })
            fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            if (i + 1) % 50 == 0:
                fh.flush()
                log(f"          {name}: {i+1}/{len(samples)} flips")
    n_harness = sum(1 for r in records if r["failure_mode"] == isolation.HARNESS_ERROR)
    if n_harness:
        first = next(r["exception_repr"] for r in records
                     if r["failure_mode"] == isolation.HARNESS_ERROR)
        raise RuntimeError(f"{name}: {n_harness} harness-error record(s) (not corruption "
                           f"crashes) — aborting; fix the child worker. First: {first}")
    open(paths["done"](name), "w").close()
    log(f"        {name}: {len(records)} isolated flips in {time.time()-t0:.1f}s")
    return records


def forced_crash_check(name, spec, knob_val, paths, sub, args):
    """Smoke gate: flip a graph_meta/header byte through the SAME isolated path and confirm
    it is contained and recorded as `crash` (proves segfault isolation works)."""
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob_val)
    ref_buf = to_buffer(index)
    del index
    clean = clean_baseline(ref_buf, spec, knob_val, sub["xq"], sub["gt_ids"], sub["gt_dist"])
    buf_path = isolation.dump_buffer(ref_buf, os.path.join(sub["dir"], f"{name}.crashprobe.npy"))
    del ref_buf
    rmap = load_augmented_regions(name, paths["regions_dir"])
    meta = [r for r in rmap["regions"] if buckets.bucket_of(r) == "crash_structure"]
    if not meta:
        return None
    region = meta[0]
    for off in range(0, min(64, region["byte_len"])):
        for bit in range(8):
            rec = isolation.measure_flip_isolated(
                buf_path, [(int(region["byte_start"] + off), bit)], spec["kind"], knob_val,
                sub["q_path"], sub["gt_ids_path"], sub["gt_dist_path"], clean,
                k=100, timeout=args.timeout)
            if rec["failure_mode"] == metrics.CRASH:
                return True
    return False


def make_paths(art_root, out):
    return {
        "index": lambda n: os.path.join(art_root, "indexes", f"{n}.faissindex"),
        "regions_dir": os.path.join(art_root, "phase1", "regions_aug"),
        "raw": lambda n: os.path.join(out, "raw", f"{n}.edges.jsonl"),
        "done": lambda n: os.path.join(out, "raw", f"{n}.edges.done"),
    }


def make_substrate(out, art_root, nq):
    """Dump the query/gt subset to .npy paths the isolated workers load (once per run)."""
    sub_dir = os.path.join(out, "_substrate")
    os.makedirs(sub_dir, exist_ok=True)
    xq = data.load_query()[:nq].astype("float32")
    gt_ids = np.load(os.path.join(art_root, "gt", "gt_ids.npy"))[:nq]
    gt_dist = np.load(os.path.join(art_root, "gt", "gt_dist.npy"))[:nq]
    q_path = os.path.join(sub_dir, "xq.npy")
    gt_ids_path = os.path.join(sub_dir, "gt_ids.npy")
    gt_dist_path = os.path.join(sub_dir, "gt_dist.npy")
    np.save(q_path, xq); np.save(gt_ids_path, gt_ids); np.save(gt_dist_path, gt_dist)
    return {"dir": sub_dir, "xq": xq, "gt_ids": gt_ids, "gt_dist": gt_dist,
            "q_path": q_path, "gt_ids_path": gt_ids_path, "gt_dist_path": gt_dist_path}


def main():
    ap = argparse.ArgumentParser(description="Phase 2 Tier 2: graph_edges characterization.")
    ap.add_argument("--indexes", default=",".join(GRAPH_GROUP))
    ap.add_argument("--n-edges", type=int, default=4000, help="edge flips per index (stratified)")
    ap.add_argument("--harmful-thresh", type=float, default=buckets.CATASTROPHIC_ABS,
                    help="dR@10 above this is silent-harmful (default 0.01)")
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--timeout", type=float, default=config.PHASE2_FLIP_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = os.path.join(config.ROOT, "artifacts")
    if args.smoke:
        args.n_edges = min(args.n_edges, SMOKE["n_edges"])
        args.queries = min(args.queries, SMOKE["queries"])
        out = args.out or os.path.join(config.ROOT, "artifacts_smoke", "phase2")
    else:
        out = args.out or os.path.join(art_root, "phase2")
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    paths = make_paths(art_root, out)

    names = [n.strip() for n in args.indexes.split(",") if n.strip() and n.strip() in GRAPH_GROUP]
    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        knob_by = {r["index"]: r["knob_value"] for r in json.load(f)["indexes"]}
    sub = make_substrate(out, art_root, args.queries)

    log(f"[phase2-T2] out={out} smoke={args.smoke} queries={args.queries} indexes={names} "
        f"n_edges={args.n_edges} harmful_thresh={args.harmful_thresh}")

    all_records = []
    for name in names:
        if args.resume and os.path.exists(paths["done"](name)):
            log(f"[phase2-T2] {name}: already complete (--resume), reloading shard")
            with open(paths["raw"](name)) as fh:
                all_records.extend(json.loads(line) for line in fh)
            continue
        log(f"[phase2-T2] {name}: characterizing graph_edges ...")
        all_records.extend(sweep_index(name, spec_by[name], knob_by[name], paths, sub, args))

    rows = aggregate(all_records, args.harmful_thresh)
    write_out(out, rows)
    log(f"[phase2-T2] wrote graph_edges_characterization.{{json,csv}} ({len(rows)} rows)")
    for r in rows:
        if r["bit_position_tag"] == "ALL":
            log(f"        {r['index']}: crash={r['pct_crash']:.1f}% "
                f"silent-benign={r['pct_silent_benign']:.1f}% "
                f"silent-harmful={r['pct_silent_harmful']:.1f}%")

    if args.smoke:
        log("[phase2-T2] forced-crash isolation check ...")
        crashed = False
        for name in names:
            res = forced_crash_check(name, spec_by[name], knob_by[name], paths, sub, args)
            if res:
                log(f"        {name}: isolated graph_meta/header flip -> crash captured "
                    f"(parent survived): OK")
                crashed = True
                break
        if not crashed:
            log("        [warn] no forced crash captured (subset may lack a crash byte)")
        log("\nPHASE2-T2 SMOKE OK")
        return 0

    log("\nPHASE2-T2 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
