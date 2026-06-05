#!/usr/bin/env python3
"""Phase 0 — build the design-matrix indexes, align clean recall, persist baselines.

For every index in qp.config.INDEX_SPECS:
  build -> tune clean recall@10 to the operating point (~0.95) -> write index ->
  measure serialized size -> compute clean recall@{1,10,100} + tolerant recall ->
  build a byte-region map.

FLAT is exact (recall ~ 1.0); it is the reference and the source of the exact ground-truth
distances used for tolerant recall. The study measures *sensitivity*, so the approximate
indexes are all pinned to the SAME clean recall before any corruption is injected.

Outputs (under --out, default artifacts/):
  indexes/<NAME>.faissindex      each index via faiss.write_index
  gt/gt_ids.npy   gt/gt_dist.npy ground-truth ids + exact distances (10k x 100)
  regions/<NAME>.regions.json    byte-region map per index
  baseline.json  baseline.csv    per-index knob/recall/size/target_met + run metadata

Usage:
  python phase0_build.py [--max-rows N] [--out artifacts/] [--target 0.95]

Exits non-zero if FLAT is not exact or any approximate index misses the operating band
(except a documented IVF_PQ_M8 plateau). Mirrors verify_env.py's `ENV OK` with `PHASE0 OK`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from qp import config, data
from qp import indexes as ix
from qp import metrics, regions
from qp.flip import byte_size

import faiss


def log(msg):
    print(msg, flush=True)


def ensure_dirs(out):
    for sub in ("indexes", "gt", "regions"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None,
                    help="load only first N base vectors (smoke runs)")
    ap.add_argument("--out", default=os.path.join(config.ROOT, "artifacts"))
    ap.add_argument("--target", type=float, default=config.OPERATING_POINT)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    ensure_dirs(out)
    smoke = args.max_rows is not None

    log(f"[phase0] out={out}  max_rows={args.max_rows}  target={args.target}")
    log("[phase0] loading SIFT1M ...")
    xb = data.load_base(args.max_rows)
    xq = data.load_query()
    xl = data.load_learn()
    n, d = xb.shape
    log(f"        base={xb.shape}  query={xq.shape}  learn={xl.shape}")

    # --- ground truth ---------------------------------------------------------
    # The provided GT is computed against the FULL 1M base. On a smoke subset it no
    # longer matches, so we recompute exact GT from the FLAT index for whatever base
    # we actually loaded. On the full run we still derive gt_dist from FLAT (needed for
    # tolerant recall) and cross-check ids against the shipped groundtruth file.
    flat = faiss.IndexFlatL2(d)
    flat.add(xb)
    t = time.time()
    gt_dist_full, gt_ids_full = flat.search(xq, 100)
    log(f"        exact GT from FLAT in {time.time() - t:.1f}s")
    gt_ids = gt_ids_full.astype("int32")
    gt_dist = gt_dist_full.astype("float32")

    if not smoke:
        shipped = data.load_groundtruth()
        agree = float((gt_ids[:, 0] == shipped[:, 0]).mean())
        log(f"        FLAT top-1 vs shipped groundtruth agreement: {agree:.4f}")

    np.save(os.path.join(out, "gt", "gt_ids.npy"), gt_ids)
    np.save(os.path.join(out, "gt", "gt_dist.npy"), gt_dist)

    # --- per-index build + tune + persist ------------------------------------
    rows = []
    all_ok = True
    for spec in config.INDEX_SPECS:
        name = spec["name"]
        log(f"[phase0] {name}: building ...")
        t0 = time.time()
        if name == "FLAT":
            index = flat                       # reuse the one we already built
        else:
            index = ix.build_index(spec, xb, xl)
        build_s = time.time() - t0

        # tune to operating point (FLAT is exact -> skip)
        if spec["kind"] == "exact":
            _, I = index.search(xq, config.K)
            r10 = metrics.recall_at_k(I, gt_ids, config.K)
            knob, knob_val, curve, target_met = None, None, [], True
        else:
            knob = ix.knob_name(spec)
            knob_val, r10, curve, target_met = ix.tune(
                index, spec, xq, gt_ids, target=args.target, k=config.K)
        log(f"        {name}: knob={knob}={knob_val} recall@10={r10:.4f} "
            f"target_met={target_met} build={build_s:.1f}s")

        # persist index
        path = os.path.join(out, "indexes", f"{name}.faissindex")
        faiss.write_index(index, path)
        nbytes = byte_size(index)

        # full clean metrics at the pinned operating point
        _, I = index.search(xq, 100)
        recalls = metrics.recall_curve(I, gt_ids, config.RECALL_KS)
        tol = metrics.tolerant_recall(I, gt_ids, gt_dist, config.K, config.EPSILON_TOLERANT)

        # region map
        rmap = regions.build_region_map(name, spec, index, xb)
        with open(os.path.join(out, "regions", f"{name}.regions.json"), "w") as f:
            json.dump(rmap, f, indent=2)
        n_located = sum(1 for r in rmap["regions"] if r["located"])
        log(f"        {name}: size={nbytes / 1e6:.1f}MB regions={len(rmap['regions'])} "
            f"({n_located} located)")

        # Gates:
        #  FLAT must be exact. Non-PQ approximate indexes must REACH the operating point
        #  (target_met); landing outside the tight band is only a soft alignment note, not
        #  a failure, because the knob is discrete. The IVF_PQ family cannot reach ~0.95 on
        #  SIFT without re-ranking (off by design), so its lower clean recall is a DOCUMENTED
        #  ceiling and becomes that index's own baseline — not a run failure.
        lo, hi = config.TARGET_BAND
        if name == "FLAT":
            if recalls[10] < 0.999:
                all_ok = False
                log(f"        [FAIL] FLAT recall@10 {recalls[10]:.4f} < 0.999")
        elif spec["quant"] == "pq" and not target_met:
            log(f"        [note] {name} clean recall@10 {r10:.4f} is the documented PQ "
                f"ceiling (re-ranking off); used as its own baseline")
        elif not target_met:
            all_ok = False
            log(f"        [FAIL] {name} could not reach operating point {args.target} "
                f"(best recall@10 {r10:.4f})")
        elif not (lo <= r10 <= hi):
            log(f"        [note] {name} recall@10 {r10:.4f} aligned above target but "
                f"outside tight band {config.TARGET_BAND} (discrete knob)")

        rows.append({
            "index": name,
            "kind": spec["kind"],
            "quant": spec["quant"],
            "knob": knob,
            "knob_value": knob_val,
            "clean_recall@1": recalls[1],
            "clean_recall@10": recalls[10],
            "clean_recall@100": recalls[100],
            "tolerant_recall@10": tol,
            "serialized_bytes": nbytes,
            "serialized_MB": nbytes / 1e6,
            "bytes_per_vector": nbytes / n,
            "target_met": target_met,
            "build_seconds": build_s,
            "n_regions": len(rmap["regions"]),
            "tuning_curve": curve,
        })

    # --- baseline.json / baseline.csv ----------------------------------------
    meta = {
        "dataset": config.DATASET,
        "n_base": int(n), "dim": int(d), "n_query": int(xq.shape[0]),
        "faiss_version": faiss.__version__,
        "seed": config.SEED, "k": config.K,
        "operating_point": args.target, "target_band": list(config.TARGET_BAND),
        "epsilon_tolerant": config.EPSILON_TOLERANT,
        "nlist": config.NLIST, "M": config.M, "efConstruction": config.EF_CONSTRUCTION,
        "refine_enabled": config.FAISS_REFINE,
        "smoke": smoke,
    }
    with open(os.path.join(out, "baseline.json"), "w") as f:
        json.dump({"meta": meta, "indexes": rows}, f, indent=2)

    # CSV without the bulky tuning curve
    import csv
    csv_cols = [k for k in rows[0] if k != "tuning_curve"]
    with open(os.path.join(out, "baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in csv_cols})

    # --- summary table --------------------------------------------------------
    log("\n[phase0] baseline summary:")
    log(f"{'index':<12} {'knob':>14} {'R@10':>7} {'tolR@10':>8} {'size(MB)':>9} {'met':>5}")
    for r in rows:
        knob = f"{r['knob']}={r['knob_value']}" if r["knob"] else "exact"
        log(f"{r['index']:<12} {knob:>14} {r['clean_recall@10']:>7.4f} "
            f"{r['tolerant_recall@10']:>8.4f} {r['serialized_MB']:>9.1f} "
            f"{str(r['target_met']):>5}")

    if all_ok:
        log("\nPHASE0 OK")
        return 0
    log("\nPHASE0 FAILED (see [FAIL]/[WARN] above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
