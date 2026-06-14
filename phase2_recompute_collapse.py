#!/usr/bin/env python3
"""Phase 2 — recompute Curve B's collapse fraction under the UNIFIED retention rule.

Curve B's `cat_frac` historically came from `pct_catastrophic` = pct(ΔR@10 > 0.01) — the
weaker HARMFUL bar (qp.buckets.HARMFUL_ABS). The headline collapse notion is now the unified
retention rule: faulted recall@10 < frac*own-clean (qp.config.COLLAPSE_RETENTION_FRAC), SILENT
only (a crash is detectable, not collapse). See qp.metrics.is_silent_collapse.

KEY LEVER — collapse ⊆ harmful. A retention collapse has ΔR > ~0.475 ≫ 0.01, so any region
whose `pct_catastrophic` is already 0 has `pct_collapse` 0 by implication (no re-measurement).
Per artifacts/phase2/rollup/region_terms.csv the ONLY regions with nonzero harmful fraction are
`sq_scale` on IVF_SQ8 and HNSW_SQ8, so the whole recompute reduces to re-sweeping `sq_scale`
(8192 bits each, fully enumerable, deserializes cleanly -> fast in-process loop).

This script:
  1. Enumerates every sq_scale bit of IVF_SQ8 / HNSW_SQ8, flips it (serialize/flip/deserialize),
     measures faulted recall@10, classifies silent collapse, and aggregates `pct_collapse` per
     within-element fp32 tag. Persists per-flip raw jsonl — the data the original full run did NOT
     keep, which is why this re-sweep is needed at all. (phase1/tier1 aggregators now emit
     pct_collapse natively, so any FUTURE full run makes this script unnecessary.)
  2. Adds a `pct_collapse` column to vuln_map.csv/json (Phase 1) and vuln_map_pq.csv/json (Tier 1):
     sq_scale rows from the re-sweep; every other row 0 — asserting its pct_catastrophic is 0
     (the collapse⊆harmful guard; a violation means another region needs a real re-sweep).
  phase2_rollup then sources cat_frac from pct_collapse (falling back to pct_catastrophic only if
  the column is absent, for old/smoke data).

Usage:
  python phase2_recompute_collapse.py                 # full exhaustive re-sweep + map rewrite
  python phase2_recompute_collapse.py --queries 4000  # fewer queries (collapse has huge margin)
  python phase2_recompute_collapse.py --smoke         # tiny sampled sweep, no real-map rewrite
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
from qp.flip import to_buffer, flip_bit, rebuild, safe_search
from phase1_region_accounting import load_augmented_regions
from phase1_sensitivity import clean_baseline


SQ8_INDEXES = ["IVF_SQ8", "HNSW_SQ8"]
SQ_REGION = "sq_scale"


def log(msg):
    print(msg, flush=True)


def enum_positions(region, sample_per_tag, seed):
    """Yield (byte_off, bit, tag) for the sq_scale region. Exhaustive when sample_per_tag<=0;
    otherwise up to `sample_per_tag` random bits per fp32 tag (for smoke)."""
    byte_len = int(region["byte_len"])
    allp = [(off, bit, buckets.bit_position_tag("float32", off, bit))
            for off in range(byte_len) for bit in range(8)]
    if sample_per_tag and sample_per_tag > 0:
        rng = np.random.default_rng([seed, 99])
        by_tag = {}
        for p in allp:
            by_tag.setdefault(p[2], []).append(p)
        out = []
        for tag, ps in by_tag.items():
            idx = rng.choice(len(ps), size=min(sample_per_tag, len(ps)), replace=False)
            out += [ps[i] for i in idx]
        return out
    return allp


def resweep_sq_scale(name, spec, knob, paths, xq, gt_ids, gt_dist, frac, raw_dir,
                     sample_per_tag, seed):
    """Enumerate sq_scale bits, measure faulted recall@10, classify silent collapse.
    Returns {(index, region, tag): row_dict}. Writes per-flip raw jsonl."""
    t0 = time.time()
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob)
    ref = to_buffer(index)
    del index
    clean = clean_baseline(ref, spec, knob, xq, gt_ids, gt_dist)
    clean10 = clean["recall"][10]
    rmap = load_augmented_regions(name, paths["regions_dir"])
    region = next(r for r in rmap["regions"] if r["kind"] == SQ_REGION)
    positions = enum_positions(region, sample_per_tag, seed)
    log(f"        {name}: clean recall@10={clean10:.4f}  "
        f"collapse if faulted@10 < {frac*clean10:.4f}  ({len(positions)} bits)")

    os.makedirs(raw_dir, exist_ok=True)
    raw_fh = open(os.path.join(raw_dir, f"{name}.jsonl"), "w")
    by_tag = {}
    for i, (off, bit, tag) in enumerate(positions):
        byte_pos = region["byte_start"] + off
        flip_bit(ref, byte_pos, bit)
        idx, exc = rebuild(ref)
        if exc is None:
            ix.set_knob(idx, spec, knob)
            D, I, sexc = safe_search(idx, xq, 100)
        else:
            D = I = sexc = None
        if exc is not None or sexc is not None:
            faulted10, fm = None, metrics.CRASH
        else:
            faulted10 = metrics.recall_at_k(I, gt_ids, 10)
            fm = metrics.classify_failure(distances=D, recall=faulted10, clean_recall=clean10)
        flip_bit(ref, byte_pos, bit)             # XOR-restore

        collapse = metrics.is_silent_collapse(faulted10, clean10, frac)
        dR = (clean10 - faulted10) if faulted10 is not None else None
        raw_fh.write(json.dumps({
            "index": name, "region": SQ_REGION, "byte_off": off, "bit": bit,
            "bit_position_tag": tag, "clean_recall@10": clean10,
            "faulted_recall@10": faulted10, "dRecall@10": dR,
            "failure_mode": fm, "silent_collapse": bool(collapse)}) + "\n")
        g = by_tag.setdefault(tag, {"n": 0, "n_collapse": 0, "n_crash": 0, "n_nan_inf": 0,
                                    "dR": []})
        g["n"] += 1
        g["n_collapse"] += int(collapse)
        g["n_crash"] += int(fm == metrics.CRASH)
        g["n_nan_inf"] += int(fm == metrics.NAN_INF)
        if dR is not None:
            g["dR"].append(dR)
        if (i + 1) % 256 == 0:
            raw_fh.flush()
            done = sum(g["n_collapse"] for g in by_tag.values())
            log(f"          {name}: {i+1}/{len(positions)} bits "
                f"({time.time()-t0:.0f}s, {done} collapse so far)")
    raw_fh.close()

    out = {}
    for tag, g in by_tag.items():
        dR = np.array(g["dR"], float)
        out[(name, SQ_REGION, tag)] = {
            "index": name, "region": SQ_REGION, "kind": SQ_REGION, "bit_position_tag": tag,
            "n_samples": g["n"],
            "pct_collapse": 100.0 * g["n_collapse"] / g["n"] if g["n"] else None,
            "mean_dRecall@10": float(dR.mean()) if dR.size else None,
            "max_dRecall@10": float(dR.max()) if dR.size else None,
            "n_collapse": g["n_collapse"], "n_crash": g["n_crash"], "n_nan_inf": g["n_nan_inf"],
        }
    log(f"        {name}: done in {time.time()-t0:.0f}s -> "
        + ", ".join(f"{t[2]}={out[t]['pct_collapse']:.1f}%" for t in sorted(out)))
    return out


def add_pct_collapse(map_json, map_csv, sweep_map, label, harmful_col="pct_catastrophic"):
    """Rewrite a vuln map adding `pct_collapse`: sq_scale rows from `sweep_map`; every other row
    0, asserting `harmful_col` is 0 there (collapse⊆harmful). Updates both .json and .csv."""
    with open(map_json) as f:
        rows = json.load(f)["rows"]
    n_sq = n_zero = 0
    for r in rows:
        key = (r["index"], r["region"], r["bit_position_tag"])
        if r["region"] == SQ_REGION:
            if key not in sweep_map:
                raise RuntimeError(f"{label}: sq_scale row {key} not in re-sweep "
                                   f"(was the sweep run for this index?)")
            r["pct_collapse"] = sweep_map[key]["pct_collapse"]
            n_sq += 1
        else:
            harmful = r.get(harmful_col)
            harmful = float(harmful) if harmful not in (None, "") else 0.0
            if harmful > 0.0:
                raise RuntimeError(
                    f"{label}: {key} has {harmful_col}={harmful} > 0 but is not sq_scale — "
                    f"collapse⊆harmful guard broken; this region needs a real collapse re-sweep.")
            r["pct_collapse"] = 0.0
            n_zero += 1
    with open(map_json, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    if rows:
        with open(map_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log(f"[recompute] {label}: pct_collapse written "
        f"({n_sq} sq_scale rows re-swept, {n_zero} rows 0 by collapse⊆harmful)")


def write_summary(rollup_dir, sweep_map):
    os.makedirs(rollup_dir, exist_ok=True)
    rows = [sweep_map[k] for k in sorted(sweep_map)]
    if not rows:
        return
    cols = ["index", "region", "kind", "bit_position_tag", "n_samples", "pct_collapse",
            "mean_dRecall@10", "max_dRecall@10", "n_collapse", "n_crash", "n_nan_inf"]
    with open(os.path.join(rollup_dir, "sq_scale_collapse.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log(f"[recompute] wrote rollup/sq_scale_collapse.csv ({len(rows)} tag rows)")


def main():
    ap = argparse.ArgumentParser(description="Recompute Curve B collapse fraction (retention rule).")
    ap.add_argument("--indexes", default=",".join(SQ8_INDEXES))
    ap.add_argument("--queries", type=int, default=10000,
                    help="query count for recall@10 (collapse has a huge margin; 2000-4000 is "
                         "plenty — the original ΔR map used 10000)")
    ap.add_argument("--collapse-frac", type=float, default=config.COLLAPSE_RETENTION_FRAC)
    ap.add_argument("--sample-per-tag", type=int, default=0,
                    help="0 = exhaustive (every sq_scale bit); >0 = sample N bits per fp32 tag")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = os.path.join(config.ROOT, "artifacts")
    if args.smoke:
        args.queries = min(args.queries, 500)
        args.sample_per_tag = args.sample_per_tag or 4
        out = args.out or os.path.join(config.ROOT, "artifacts_smoke", "phase2")
    else:
        out = args.out or os.path.join(art_root, "phase2")
    out = os.path.abspath(out)
    rollup_dir = os.path.join(out, "rollup")
    raw_dir = os.path.join(rollup_dir, "sq_scale_collapse_raw")

    paths = {
        "index": lambda n: os.path.join(art_root, "indexes", f"{n}.faissindex"),
        "regions_dir": os.path.join(art_root, "phase1", "regions_aug"),
    }
    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        knob_by = {r["index"]: r["knob_value"] for r in json.load(f)["indexes"]}
    nq = args.queries
    xq = data.load_query()[:nq].astype("float32")
    gt_ids = np.load(os.path.join(art_root, "gt", "gt_ids.npy"))[:nq]
    gt_dist = np.load(os.path.join(art_root, "gt", "gt_dist.npy"))[:nq]

    names = [n.strip() for n in args.indexes.split(",")
             if n.strip() and n.strip() in SQ8_INDEXES]
    log(f"[recompute] out={out} smoke={args.smoke} queries={nq} indexes={names} "
        f"frac={args.collapse_frac} sample_per_tag={args.sample_per_tag}")

    sweep_map = {}
    for name in names:
        log(f"[recompute] {name}: re-sweeping sq_scale ...")
        sweep_map.update(resweep_sq_scale(
            name, spec_by[name], knob_by[name], paths, xq, gt_ids, gt_dist,
            args.collapse_frac, raw_dir, args.sample_per_tag, args.seed))
    write_summary(rollup_dir, sweep_map)

    if args.smoke:
        assert sweep_map, "re-sweep produced nothing"
        log("\nPHASE2-RECOMPUTE SMOKE OK")
        return 0

    # Rewrite the real maps in place (additive column). phase1 map is aligned (harmful=abs>0.01);
    # pq map's pct_catastrophic is the relative retention count (still a superset of silent collapse).
    add_pct_collapse(os.path.join(art_root, "phase1", "vuln_map.json"),
                     os.path.join(art_root, "phase1", "vuln_map.csv"), sweep_map, "phase1")
    pq_json = os.path.join(out, "vuln_map_pq.json")
    if os.path.exists(pq_json):
        add_pct_collapse(pq_json, os.path.join(out, "vuln_map_pq.csv"), sweep_map, "tier1-pq")
    log("\nPHASE2-RECOMPUTE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
