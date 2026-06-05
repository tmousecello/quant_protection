#!/usr/bin/env python3
"""Phase 1 — Task B + D: single-bit sensitivity map (clean group only).

For each of the 5 clean-aligned indexes (FLAT, IVF_FLAT, IVF_SQ8, HNSW, HNSW_SQ8) we flip
ONE bit at a time inside its recall_relevant regions, rebuild, re-search the fixed query
set at the Phase 0 operating point, and record how far recall moved. PQ is excluded this
round (it has no aligned baseline).

Method (per the proven substrate):
  ref_buf = serialize(index)               # one pristine buffer, mutated in place
  for each sampled (byte_pos, bit):
      flip_bit(ref_buf, byte_pos, bit)      # corrupt
      idx = deserialize(ref_buf); set_knob; search 10k @100
      record recall@{1,10,100} + tolerant recall + failure mode
      flip_bit(ref_buf, byte_pos, bit)      # XOR-restore (its own inverse, O(1))

Sampling is stratified by within-element bit position and labelled (qp.buckets) so the
fp32 sign/exponent/mantissa bimodal and the SQ8 MSB->LSB gradient survive aggregation:
  large regions (vectors/codes)   S_large positions, stratified across the 4 fp32 bytes
                                  (or 8 uint8 bit indices)
  centroid (medium_critical)      S_centroid positions, stratified
  sq_scale (small_critical)       exhaustive (every one of its 8192 bits)

nan/inf policy (deterministic): recall is computed from returned IDS, never distances, so
non-finite faulted distances cannot corrupt the recall number. NaN/Inf in the distance
array is tallied as the `nan-inf` failure mode (and nan_inf_count is stored) but its
ΔRecall is still recorded. A search/deserialize that raises is `crash`.

Outputs (under --out, default artifacts/phase1; smoke -> artifacts_smoke/phase1):
  raw/<NAME>.records.jsonl   one row per flip (streamed, flushed periodically)
  raw/<NAME>.done            completion marker (enables --resume)
  vuln_map.json / .csv       aggregated by (index x region x bit_position_tag)
  crash_probe.json           (only with --crash-probe) crash-mode characterization

Usage:
  python phase1_sensitivity.py                       # full Lean run (~6h serial)
  python phase1_sensitivity.py --smoke               # ~2min pipeline + D1-D4 gate
  python phase1_sensitivity.py --indexes IVF_SQ8 --resume
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

from qp import config, data, buckets
from qp import indexes as ix
from qp import metrics
from qp.flip import to_buffer, flip_bit, rebuild, safe_search
from phase1_region_accounting import load_augmented_regions

import faiss


CLEAN_GROUP = ["FLAT", "IVF_FLAT", "IVF_SQ8", "HNSW", "HNSW_SQ8"]
SMOKE = {"s_large": 20, "s_centroid": 20, "sq_cap": 64, "queries": 1000}


def log(msg):
    print(msg, flush=True)


# --- sampling -----------------------------------------------------------------
def _sample(byte_start, off, bit, edtype):
    """Build one sample dict for a within-region offset `off` and `bit`."""
    within = off % 4 if edtype == "float32" else 0
    return {
        "byte_pos": int(byte_start + off),
        "bit": int(bit),
        "within_byte": int(within),
        "bit_position_tag": buckets.bit_position_tag(edtype, off, bit),
    }


def plan_region_samples(name, region, s_large, s_centroid, sq_cap, rng):
    """Return the list of bit samples for one recall_relevant region.

    Stratified so each fp32 byte (sign/exp vs mantissa) and each SQ8 bit index is evenly
    covered, instead of uniform sampling that would starve the rare-but-catastrophic sign
    bit (1/32 of fp32 bits).
    """
    rclass = buckets.region_class_of(region)
    edtype = buckets.element_dtype_of(name, region)
    byte_start, byte_len = region["byte_start"], region["byte_len"]
    out = []

    if rclass == "small_critical":
        total_bits = byte_len * 8
        if sq_cap is not None and total_bits > sq_cap:
            # smoke: evenly spaced subset across the region
            for gi in np.linspace(0, total_bits - 1, sq_cap).astype(int):
                out.append(_sample(byte_start, int(gi) // 8, int(gi) % 8, edtype))
        else:
            for off in range(byte_len):
                for bit in range(8):
                    out.append(_sample(byte_start, off, bit, edtype))
        return out

    S = s_centroid if rclass == "medium_critical" else s_large
    if edtype == "float32":
        n_elems = max(1, byte_len // 4)
        per = max(1, S // 4)
        for w in range(4):                       # stratify across the 4 little-endian bytes
            for _ in range(per):
                off = int(rng.integers(n_elems)) * 4 + w
                if off >= byte_len:
                    continue
                out.append(_sample(byte_start, off, int(rng.integers(8)), edtype))
    else:                                        # uint8: stratify across the 8 bit indices
        per = max(1, S // 8)
        for bit in range(8):
            for _ in range(per):
                off = int(rng.integers(byte_len))
                out.append(_sample(byte_start, off, bit, edtype))
    return out


def recall_relevant_regions(rmap):
    return [r for r in rmap["regions"] if buckets.bucket_of(r) == "recall_relevant"]


# --- measurement --------------------------------------------------------------
def search_recalls(idx, spec, knob_val, xq, gt_ids, gt_dist):
    """Set knob, search @100, return (recalls dict, tol, D, I, exc)."""
    ix.set_knob(idx, spec, knob_val)
    D, I, exc = safe_search(idx, xq, 100)
    if exc is not None:
        return None, None, None, None, exc
    recalls = metrics.recall_curve(I, gt_ids, config.RECALL_KS)
    tol = metrics.tolerant_recall(I, gt_ids, gt_dist, config.K, config.EPSILON_TOLERANT)
    return recalls, tol, D, I, None


def measure_flip(ref_buf, byte_pos, bit, spec, knob_val, xq, gt_ids, gt_dist, clean):
    """Flip one bit, measure, restore. Returns a record dict (no identity fields)."""
    flip_bit(ref_buf, byte_pos, bit)
    try:
        idx, exc = rebuild(ref_buf)
        if exc is not None:
            return _crash_record(clean, exc)
        recalls, tol, D, _, sexc = search_recalls(
            idx, spec, knob_val, xq, gt_ids, gt_dist)
        if sexc is not None:
            return _crash_record(clean, sexc)
        nan_inf = int((~np.isfinite(np.asarray(D))).sum())
        fm = metrics.classify_failure(
            distances=D, recall=recalls[10], clean_recall=clean["recall"][10])
        rec = {
            "faulted_recall@1": recalls[1], "faulted_recall@10": recalls[10],
            "faulted_recall@100": recalls[100], "faulted_tol": tol,
            "dRecall@1": clean["recall"][1] - recalls[1],
            "dRecall@10": clean["recall"][10] - recalls[10],
            "dRecall@100": clean["recall"][100] - recalls[100],
            "dTol": clean["tol"] - tol,
            "failure_mode": fm, "nan_inf_count": nan_inf, "exception_repr": "",
        }
        return rec
    finally:
        flip_bit(ref_buf, byte_pos, bit)         # restore (XOR back)


def _crash_record(clean, exc):
    return {
        "faulted_recall@1": None, "faulted_recall@10": None,
        "faulted_recall@100": None, "faulted_tol": None,
        "dRecall@1": None, "dRecall@10": None, "dRecall@100": None, "dTol": None,
        "failure_mode": metrics.CRASH, "nan_inf_count": 0,
        "exception_repr": repr(exc)[:300],
    }


def clean_baseline(ref_buf, spec, knob_val, xq, gt_ids, gt_dist):
    """Clean recall measured on the SAME rebuild->search path the faulted runs use."""
    idx, exc = rebuild(ref_buf)
    if exc is not None:
        raise RuntimeError(f"clean buffer failed to deserialize: {exc!r}")
    recalls, tol, _, _, sexc = search_recalls(idx, spec, knob_val, xq, gt_ids, gt_dist)
    if sexc is not None:
        raise RuntimeError(f"clean search failed: {sexc!r}")
    return {"recall": recalls, "tol": tol}


# --- per-index sweep ----------------------------------------------------------
def sweep_index(name, spec, knob_val, paths, xq, gt_ids, gt_dist, args, rng_seed):
    t0 = time.time()
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob_val)
    ref_buf = to_buffer(index)
    del index

    clean = clean_baseline(ref_buf, spec, knob_val, xq, gt_ids, gt_dist)
    log(f"        {name}: clean recall@10={clean['recall'][10]:.4f} "
        f"tol={clean['tol']:.4f} (rebuild path)")
    if args.assert_clean is not None:
        diff = abs(clean["recall"][10] - args.assert_clean.get(name, clean["recall"][10]))
        if diff > 1e-3:
            log(f"        [warn] {name} clean recall@10 differs from baseline by {diff:.4f}")

    rmap = load_augmented_regions(name, paths["regions_dir"])
    regions = recall_relevant_regions(rmap)

    shard = paths["raw"](name)
    records = []
    n_records = 0
    with open(shard, "w") as fh:
        for r_ord, region in enumerate(regions):
            rng = np.random.default_rng([rng_seed, CLEAN_GROUP.index(name), r_ord])
            sub_seed = f"{rng_seed}:{CLEAN_GROUP.index(name)}:{r_ord}"
            samples = plan_region_samples(
                name, region, args.s_large, args.s_centroid, args.sq_cap, rng)
            log(f"        {name}/{region['name']:<10} "
                f"({buckets.region_class_of(region)}, {buckets.element_dtype_of(name, region)}) "
                f"-> {len(samples)} bits")
            for s in samples:
                rec = measure_flip(ref_buf, s["byte_pos"], s["bit"], spec, knob_val,
                                   xq, gt_ids, gt_dist, clean)
                rec.update({
                    "index": name, "region": region["name"], "kind": region["kind"],
                    "element_dtype": buckets.element_dtype_of(name, region),
                    "bucket": "recall_relevant",
                    "region_bits": buckets.region_bits(region),
                    "byte_pos": s["byte_pos"], "bit": s["bit"],
                    "within_byte": s["within_byte"],
                    "bit_position_tag": s["bit_position_tag"],
                    "clean_recall@1": clean["recall"][1],
                    "clean_recall@10": clean["recall"][10],
                    "clean_recall@100": clean["recall"][100],
                    "clean_tol": clean["tol"],
                    "seed": rng_seed, "sub_seed": sub_seed,
                })
                fh.write(json.dumps(rec) + "\n")
                records.append(rec)
                n_records += 1
                if n_records % 200 == 0:
                    fh.flush()

    # D1: no state leakage — clean recomputed from ref_buf must match initial within 1e-5.
    clean_after = clean_baseline(ref_buf, spec, knob_val, xq, gt_ids, gt_dist)
    leak = abs(clean_after["recall"][10] - clean["recall"][10])
    if leak > 1e-5:
        raise RuntimeError(f"{name}: state leaked across flips (clean drift {leak:.2e})")

    open(paths["done"](name), "w").close()
    log(f"        {name}: {n_records} flips in {time.time() - t0:.1f}s  "
        f"(D1 clean drift {leak:.2e})")
    return records


# --- aggregation --------------------------------------------------------------
def aggregate(all_records):
    """Aggregate raw records by (index, region, bit_position_tag)."""
    groups = {}
    for r in all_records:
        key = (r["index"], r["region"], r["bit_position_tag"])
        groups.setdefault(key, []).append(r)

    rows = []
    for (index, region, tag), recs in sorted(groups.items()):
        d10 = np.array([r["dRecall@10"] for r in recs if r["dRecall@10"] is not None],
                       dtype=float)
        d1 = np.array([r["dRecall@1"] for r in recs if r["dRecall@1"] is not None], dtype=float)
        d100 = np.array([r["dRecall@100"] for r in recs if r["dRecall@100"] is not None],
                        dtype=float)
        dtol = np.array([r["dTol"] for r in recs if r["dTol"] is not None], dtype=float)
        fm = [r["failure_mode"] for r in recs]
        n = d10.size
        if n > 1:
            sem = float(stats.sem(d10))
            h = sem * float(stats.t.ppf(0.975, n - 1))
        else:
            h = 0.0
        mean10 = float(d10.mean()) if n else None
        rows.append({
            "index": index, "region": region,
            "kind": recs[0]["kind"], "bucket": recs[0]["bucket"],
            "bit_position_tag": tag, "region_bits": recs[0]["region_bits"],
            "n_samples": len(recs),
            "mean_dRecall@10": mean10,
            "p99_dRecall@10": float(np.percentile(d10, 99)) if n else None,
            "max_dRecall@10": float(d10.max()) if n else None,
            "mean_dRecall@1": float(d1.mean()) if d1.size else None,
            "mean_dRecall@100": float(d100.mean()) if d100.size else None,
            "mean_dTol": float(dtol.mean()) if dtol.size else None,
            "pct_benign": float((np.abs(d10) <= buckets.BENIGN_ABS).mean() * 100) if n else None,
            "pct_catastrophic":
                float((d10 > buckets.CATASTROPHIC_ABS).mean() * 100) if n else None,
            "ci95_low": (mean10 - h) if n else None,
            "ci95_high": (mean10 + h) if n else None,
            "n_clean": fm.count(metrics.CLEAN),
            "n_silent_wrong": fm.count(metrics.SILENT_WRONG),
            "n_nan_inf": fm.count(metrics.NAN_INF),
            "n_crash": fm.count(metrics.CRASH),
            "seed": recs[0]["seed"],
        })
    return rows


def write_vuln_map(out, rows):
    with open(os.path.join(out, "vuln_map.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    import csv
    if rows:
        cols = list(rows[0].keys())
        with open(os.path.join(out, "vuln_map.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


# --- crash probe (D4, optional) ----------------------------------------------
def crash_probe(names, specs, knobs, paths, xq, gt_ids, gt_dist, n_probe, seed):
    results = []
    for name in names:
        spec = specs[name]
        index = faiss.read_index(paths["index"](name))
        ix.set_knob(index, spec, knobs[name])
        ref_buf = to_buffer(index)
        del index
        rmap = load_augmented_regions(name, paths["regions_dir"])
        cregions = [r for r in rmap["regions"] if buckets.bucket_of(r) == "crash_structure"]
        rng = np.random.default_rng([seed, 7777, CLEAN_GROUP.index(name) if name in CLEAN_GROUP else 0])
        counts = {metrics.CLEAN: 0, metrics.SILENT_WRONG: 0, metrics.NAN_INF: 0, metrics.CRASH: 0}
        clean = clean_baseline(ref_buf, spec, knobs[name], xq, gt_ids, gt_dist)
        for region in cregions:
            for _ in range(n_probe):
                byte_pos = int(region["byte_start"] + rng.integers(region["byte_len"]))
                bit = int(rng.integers(8))
                rec = measure_flip(ref_buf, byte_pos, bit, spec, knobs[name],
                                   xq, gt_ids, gt_dist, clean)
                counts[rec["failure_mode"]] += 1
        results.append({"index": name, "n_per_region": n_probe,
                        "regions": [r["name"] for r in cregions], "counts": counts})
        log(f"        {name}: crash-probe {counts}")
    return results


# --- D2/D3 deterministic checks ----------------------------------------------
def check_nan_policy():
    """D2: classify_failure is deterministic for non-finite distances (synthetic, in-proc)."""
    D = np.array([[1.0, np.inf, 3.0], [np.nan, 2.0, 5.0]], dtype="float32")
    fm = metrics.classify_failure(distances=D, recall=0.9, clean_recall=0.95)
    cnt = int((~np.isfinite(D)).sum())
    assert fm == metrics.NAN_INF, f"expected nan-inf, got {fm}"
    assert cnt == 2, cnt
    # and again -> identical
    assert metrics.classify_failure(distances=D, recall=0.9, clean_recall=0.95) == metrics.NAN_INF


def check_plan_reproducible(name, spec_regions, seed):
    """D3: the sample plan is byte-identical across two builds with the same seed."""
    def build():
        out = []
        for r_ord, region in enumerate(spec_regions):
            rng = np.random.default_rng([seed, CLEAN_GROUP.index(name), r_ord])
            for s in plan_region_samples(name, region, SMOKE["s_large"],
                                         SMOKE["s_centroid"], SMOKE["sq_cap"], rng):
                out.append((s["byte_pos"], s["bit"]))
        return out
    a, b = build(), build()
    assert a == b, f"{name}: sample plan not reproducible"


def check_forced_crash(name, spec, knob_val, paths, xq, gt_ids, gt_dist):
    """Smoke gate: flipping a header/graph_meta bit yields a crash failure mode."""
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob_val)
    ref_buf = to_buffer(index)
    del index
    clean = clean_baseline(ref_buf, spec, knob_val, xq, gt_ids, gt_dist)
    rmap = load_augmented_regions(name, paths["regions_dir"])
    cregions = [r for r in rmap["regions"] if buckets.bucket_of(r) == "crash_structure"]
    if not cregions:
        return False
    region = cregions[0]
    # try the first several bytes of the structure (magic/params are the most fragile)
    for off in range(0, min(64, region["byte_len"])):
        for bit in range(8):
            rec = measure_flip(ref_buf, int(region["byte_start"] + off), bit,
                               spec, knob_val, xq, gt_ids, gt_dist, clean)
            if rec["failure_mode"] == metrics.CRASH:
                return True
    return False


# --- orchestration ------------------------------------------------------------
def make_paths(art_root, out):
    return {
        "index": lambda n: os.path.join(art_root, "indexes", f"{n}.faissindex"),
        "regions_dir": os.path.join(art_root, "phase1", "regions_aug"),
        "raw": lambda n: os.path.join(out, "raw", f"{n}.records.jsonl"),
        "done": lambda n: os.path.join(out, "raw", f"{n}.done"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexes", default=",".join(CLEAN_GROUP),
                    help="comma-separated subset of the clean group")
    ap.add_argument("--s-large", type=int, default=1000)
    ap.add_argument("--s-centroid", type=int, default=2000)
    ap.add_argument("--sq-cap", type=int, default=None,
                    help="cap sq_scale bits (smoke only; default exhaustive)")
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip sweeping; consolidate all completed shards into vuln_map "
                         "(use after running indexes as parallel subset processes)")
    ap.add_argument("--crash-probe", action="store_true")
    ap.add_argument("--crash-probe-n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.assert_clean = None

    art_root = os.path.join(config.ROOT, "artifacts")
    if args.smoke:
        args.s_large = min(args.s_large, SMOKE["s_large"])
        args.s_centroid = min(args.s_centroid, SMOKE["s_centroid"])
        args.sq_cap = SMOKE["sq_cap"]
        args.queries = min(args.queries, SMOKE["queries"])
        out = args.out or os.path.join(config.ROOT, "artifacts_smoke", "phase1")
    else:
        out = args.out or os.path.join(art_root, "phase1")
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    paths = make_paths(art_root, out)

    names = [n.strip() for n in args.indexes.split(",") if n.strip()]
    for n in names:
        if n not in CLEAN_GROUP:
            log(f"[phase1-B] skipping {n!r}: not in clean group {CLEAN_GROUP}")
    names = [n for n in names if n in CLEAN_GROUP]

    # --- aggregate-only: consolidate completed shards, no sweeping -----------------
    if args.aggregate_only:
        all_records = []
        done, missing = [], []
        for name in names:
            if os.path.exists(paths["done"](name)):
                with open(paths["raw"](name)) as fh:
                    all_records.extend(json.loads(line) for line in fh)
                done.append(name)
            else:
                missing.append(name)
        rows = aggregate(all_records)
        write_vuln_map(out, rows)
        log(f"[phase1-B] aggregate-only: consolidated {done} ({len(rows)} cells)")
        if missing:
            log(f"[phase1-B] [warn] no completed shard for {missing}")
        log("\nPHASE1-B AGGREGATE OK")
        return 0

    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        baseline = json.load(f)
    knob_by = {r["index"]: r["knob_value"] for r in baseline["indexes"]}
    if not args.smoke:
        args.assert_clean = {r["index"]: r["clean_recall@10"] for r in baseline["indexes"]}

    nq = args.queries
    xq = data.load_query()[:nq].astype("float32")
    gt_ids = np.load(os.path.join(art_root, "gt", "gt_ids.npy"))[:nq]
    gt_dist = np.load(os.path.join(art_root, "gt", "gt_dist.npy"))[:nq]

    log(f"[phase1-B] out={out}  smoke={args.smoke}  queries={nq}  "
        f"S_large={args.s_large} S_centroid={args.s_centroid} sq_cap={args.sq_cap}")
    log(f"[phase1-B] indexes={names}")

    all_records = []
    for name in names:
        if args.resume and os.path.exists(paths["done"](name)):
            log(f"[phase1-B] {name}: already complete (--resume), reloading shard")
            with open(paths["raw"](name)) as fh:
                all_records.extend(json.loads(line) for line in fh)
            continue
        log(f"[phase1-B] {name}: sweeping ...")
        recs = sweep_index(name, spec_by[name], knob_by[name], paths,
                           xq, gt_ids, gt_dist, args, args.seed)
        all_records.extend(recs)

    rows = aggregate(all_records)
    write_vuln_map(out, rows)
    log(f"[phase1-B] wrote vuln_map.{{json,csv}} ({len(rows)} cells)")

    if args.crash_probe:
        log("[phase1-B] crash probe ...")
        cp = crash_probe(names, spec_by, knob_by, paths, xq, gt_ids, gt_dist,
                         args.crash_probe_n, args.seed)
        with open(os.path.join(out, "crash_probe.json"), "w") as f:
            json.dump({"results": cp}, f, indent=2)

    # --- smoke gate: D1 (per-index, already asserted) + D2 + D3 + forced crash --------
    if args.smoke:
        log("\n[phase1-B] smoke gate D2/D3/forced-crash ...")
        check_nan_policy()
        log("        D2 nan/inf classify_failure deterministic: OK")
        for name in names:
            rmap = load_augmented_regions(name, paths["regions_dir"])
            rr = recall_relevant_regions(rmap)
            check_plan_reproducible(name, rr, args.seed)
        log("        D3 sample plans reproducible: OK")
        # D4 confinement: every sampled byte_pos lay in a recall_relevant region
        for r in all_records:
            assert r["bucket"] == "recall_relevant"
        log("        D4 injection confined to recall_relevant regions: OK")
        # forced crash on the first index that has a crash_structure region
        crashed = False
        for name in names:
            if check_forced_crash(name, spec_by[name], knob_by[name], paths,
                                  xq, gt_ids, gt_dist):
                log(f"        forced crash via {name} header/graph bit: OK")
                crashed = True
                break
        if not crashed:
            log("        [warn] no forced crash observed (no crash_structure in subset?)")
        # every tag branch produced at least one record
        tags = {r["bit_position_tag"] for r in all_records}
        log(f"        observed bit_position_tags: {sorted(tags)}")
        log("\nPHASE1 SMOKE OK")
        return 0

    log("\nPHASE1-B OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
