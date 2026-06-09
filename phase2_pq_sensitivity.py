#!/usr/bin/env python3
"""Phase 2 — Tier 1: PQ single-bit sensitivity map (own-baseline group).

Phase 1 deliberately left PQ out because it has no 0.95-aligned baseline. This script fills
that gap for IVF_PQ_M8 / IVF_PQ_M16, measuring ΔRecall against **each PQ index's own clean
recall** (M8≈0.379, M16≈0.563 — the documented re-rank-off ceiling, see memory
pq-cannot-align-to-095). It answers two questions the main narrative needs:

  1. Do PQ *codes* break the "codes are immune" rule?  PQ codes are codebook INDICES, so a
     flip is a discrete jump to another centroid, not a tiny fp32 perturbation. We tag those
     bits `code_b0..code_b7` (index bits — no IEEE-754 sign/exponent/mantissa meaning).
  2. Turn `pq_codebook` from a prediction into a measurement. The codebook is the PQ analogue
     of SQ8's sq_scale: a small (~128KB) fp32 structure shared by every encoded vector, so a
     single exponent flip there could move all of them. Tagged with the Phase 1 fp32 scheme.
  `centroid` (the IVF coarse quantizer, also fp32) is swept too for completeness.

Method, sampling, restore, failure classification, and the raw/aggregate output all reuse
Phase 1 (`phase1_sensitivity.py`, `qp.buckets`, `qp.metrics`, `qp.flip`) so Tier 1 is the
same harness pointed at the PQ family. Catastrophic is RELATIVE here (retention < 50% of own
clean, config.PHASE2_PQ_RETENTION_FRAC) because PQ's absolute recall is low; the absolute
>0.01 rule is also recorded for cross-comparison.

Outputs (under --out, default artifacts/phase2; smoke -> artifacts_smoke/phase2):
  raw/<NAME>.records.jsonl   one row per flip (also consumed by Tier 4 detection)
  raw/<NAME>.done            completion marker (enables --resume)
  vuln_map_pq.json / .csv     aggregated by (index x region x bit_position_tag), baseline_mode=own

Usage:
  python phase2_pq_sensitivity.py                          # full run (humans)
  python phase2_pq_sensitivity.py --smoke                  # seconds pipeline check
  python phase2_pq_sensitivity.py --indexes IVF_PQ_M8 --full-enum-codebook --resume
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy import stats

import faiss

from qp import config, data, buckets, metrics
from qp import indexes as ix
from qp.flip import to_buffer
# Reuse the Phase 1 primitives verbatim — same substrate, same record schema.
from phase1_sensitivity import (
    _sample, clean_baseline, measure_flip, recall_relevant_regions,
)
from phase1_region_accounting import load_augmented_regions


PQ_GROUP = ["IVF_PQ_M8", "IVF_PQ_M16"]
SMOKE = {"s_large": 16, "s_centroid": 16, "s_codebook": 32, "queries": 500}


def log(msg):
    print(msg, flush=True)


# --- sampling -----------------------------------------------------------------
def plan_pq_samples(name, region, s_large, s_centroid, s_codebook, full_enum, rng):
    """Bit samples for one PQ recall_relevant region.

    codes (uint8 PQ indices) : stratify across the 8 bit indices (S_large draws).
    centroid / pq_codebook   : fp32 — stratify equally across the 4 semantic tags so the
                               1/32 sign bit and the catastrophic exponent are covered like
                               mantissa (the codebook is 1.05M bits, far too many to exhaust
                               by default; --full-enum-codebook enumerates every bit instead).
    """
    kind = region["kind"]
    edtype = buckets.element_dtype_of(name, region)          # codes->uint8, else float32
    bs, bl = region["byte_start"], region["byte_len"]
    out = []

    if kind == "pq_codebook" and full_enum:
        for off in range(bl):
            for bit in range(8):
                out.append(_sample(bs, off, bit, edtype))
        return out

    S = {"codes": s_large, "centroid": s_centroid, "pq_codebook": s_codebook}.get(kind, s_large)
    if edtype == "float32":
        n_elems = max(1, bl // 4)
        per = max(1, S // 4)
        for tag in buckets.FP32_TAGS:
            choices = buckets.FP32_TAG_BITS[tag]
            for _ in range(per):
                wbyte, bit = choices[int(rng.integers(len(choices)))]
                off = int(rng.integers(n_elems)) * 4 + wbyte
                if off < bl:
                    out.append(_sample(bs, off, bit, edtype))
    else:                                                    # uint8 PQ codes
        per = max(1, S // 8)
        for bit in range(8):
            for _ in range(per):
                off = int(rng.integers(bl))
                out.append(_sample(bs, off, bit, edtype))
    return out


def pq_tag(region_kind, sample):
    """Bit-position tag, relabelling uint8 PQ codes as code_b* (index bits, not IEEE-754)."""
    if region_kind == "codes":
        return f"code_b{sample['bit']}"
    return sample["bit_position_tag"]


# --- aggregation (Phase 1 schema + baseline_mode + PQ relative catastrophic) ----------
def aggregate_pq(records, retention_frac):
    groups = {}
    for r in records:
        groups.setdefault((r["index"], r["region"], r["bit_position_tag"]), []).append(r)

    rows = []
    for (index, region, tag), recs in sorted(groups.items()):
        d10 = np.array([r["dRecall@10"] for r in recs if r["dRecall@10"] is not None], float)
        d1 = np.array([r["dRecall@1"] for r in recs if r["dRecall@1"] is not None], float)
        d100 = np.array([r["dRecall@100"] for r in recs if r["dRecall@100"] is not None], float)
        dtol = np.array([r["dTol"] for r in recs if r["dTol"] is not None], float)
        fm = [r["failure_mode"] for r in recs]
        n = d10.size
        n_total = len(recs)
        n_crash = fm.count(metrics.CRASH)
        # Relative catastrophic for PQ: retention < frac of own clean. Crashes are total loss.
        n_cat_rel = sum(1 for r in recs if r.get("cat_rel")) + n_crash
        n_cat_abs = int((d10 > buckets.CATASTROPHIC_ABS).sum()) + n_crash
        n_ben = int((np.abs(d10) <= buckets.BENIGN_ABS).sum())
        if n > 1:
            h = float(stats.sem(d10)) * float(stats.t.ppf(0.975, n - 1))
        else:
            h = 0.0
        mean10 = float(d10.mean()) if n else None
        rows.append({
            "index": index, "region": region, "kind": recs[0]["kind"],
            "bucket": recs[0]["bucket"], "bit_position_tag": tag,
            "region_bits": recs[0]["region_bits"], "n": n_total, "n_samples": n_total,
            "mean_dR@10": mean10, "mean_dRecall@10": mean10,
            "p99_dR@10": float(np.percentile(d10, 99)) if n else None,
            "max_dR@10": float(d10.max()) if n else None,
            "mean_dRecall@1": float(d1.mean()) if d1.size else None,
            "mean_dRecall@100": float(d100.mean()) if d100.size else None,
            "mean_dTol": float(dtol.mean()) if dtol.size else None,
            "pct_benign": 100.0 * n_ben / n_total if n_total else None,
            "pct_catastrophic": 100.0 * n_cat_rel / n_total if n_total else None,
            "pct_catastrophic_abs": 100.0 * n_cat_abs / n_total if n_total else None,
            "ci95_low": (mean10 - h) if n else None,
            "ci95_high": (mean10 + h) if n else None,
            "n_silent_wrong": fm.count(metrics.SILENT_WRONG),
            "n_nan_inf": fm.count(metrics.NAN_INF),
            "n_crash": n_crash, "n_clean": fm.count(metrics.CLEAN),
            "baseline_mode": "own", "seed": recs[0]["seed"],
        })
    return rows


def write_map(out, rows):
    with open(os.path.join(out, "vuln_map_pq.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    if rows:
        with open(os.path.join(out, "vuln_map_pq.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# --- per-index sweep ----------------------------------------------------------
def sweep_index(name, spec, knob_val, paths, xq, gt_ids, gt_dist, args):
    t0 = time.time()
    index = faiss.read_index(paths["index"](name))
    ix.set_knob(index, spec, knob_val)
    ref_buf = to_buffer(index)
    del index

    clean = clean_baseline(ref_buf, spec, knob_val, xq, gt_ids, gt_dist)
    frac = args.pq_retention_frac
    log(f"        {name}: own clean recall@10={clean['recall'][10]:.4f} "
        f"(catastrophic if retention < {frac:.0%} -> faulted@10 < {frac*clean['recall'][10]:.4f})")

    rmap = load_augmented_regions(name, paths["regions_dir"])
    regions = recall_relevant_regions(rmap)
    cat_thresh = frac * clean["recall"][10]

    shard = paths["raw"](name)
    records, n_records = [], 0
    with open(shard, "w") as fh:
        for r_ord, region in enumerate(regions):
            rng = np.random.default_rng([args.seed, PQ_GROUP.index(name), r_ord])
            samples = plan_pq_samples(name, region, args.s_large, args.s_centroid,
                                      args.s_codebook, args.full_enum_codebook, rng)
            log(f"        {name}/{region['name']:<11} "
                f"({buckets.region_class_of(region)}, {buckets.element_dtype_of(name, region)}) "
                f"-> {len(samples)} bits")
            for s in samples:
                rec = measure_flip(ref_buf, s["byte_pos"], s["bit"], spec, knob_val,
                                   xq, gt_ids, gt_dist, clean)
                f10 = rec["faulted_recall@10"]
                rec.update({
                    "index": name, "region": region["name"], "kind": region["kind"],
                    "element_dtype": buckets.element_dtype_of(name, region),
                    "bucket": "recall_relevant", "region_bits": buckets.region_bits(region),
                    "byte_pos": s["byte_pos"], "bit": s["bit"],
                    "bit_position_tag": pq_tag(region["kind"], s),
                    "clean_recall@10": clean["recall"][10], "clean_tol": clean["tol"],
                    "cat_rel": bool(f10 is not None and f10 < cat_thresh),
                    "cat_abs": bool(rec["dRecall@10"] is not None
                                    and rec["dRecall@10"] > buckets.CATASTROPHIC_ABS),
                    "baseline_mode": "own", "seed": args.seed,
                })
                fh.write(json.dumps(rec) + "\n")
                records.append(rec)
                n_records += 1
                if n_records % 200 == 0:
                    fh.flush()

    # state-leak guard (D1): clean recomputed from the buffer must match within 1e-5.
    clean_after = clean_baseline(ref_buf, spec, knob_val, xq, gt_ids, gt_dist)
    leak = abs(clean_after["recall"][10] - clean["recall"][10])
    if leak > 1e-5:
        raise RuntimeError(f"{name}: state leaked across flips (clean drift {leak:.2e})")
    open(paths["done"](name), "w").close()
    log(f"        {name}: {n_records} flips in {time.time()-t0:.1f}s (clean drift {leak:.2e})")
    return records


def make_paths(art_root, out):
    return {
        "index": lambda n: os.path.join(art_root, "indexes", f"{n}.faissindex"),
        "regions_dir": os.path.join(art_root, "phase1", "regions_aug"),
        "raw": lambda n: os.path.join(out, "raw", f"{n}.records.jsonl"),
        "done": lambda n: os.path.join(out, "raw", f"{n}.done"),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 2 Tier 1: PQ single-bit sensitivity map.")
    ap.add_argument("--indexes", default=",".join(PQ_GROUP),
                    help="comma-separated subset of the PQ group")
    ap.add_argument("--s-large", type=int, default=1000, help="samples for pq_codes")
    ap.add_argument("--s-centroid", type=int, default=2000, help="samples for centroid")
    ap.add_argument("--s-codebook", type=int, default=4000,
                    help="tag-stratified samples for pq_codebook (ignored with --full-enum-codebook)")
    ap.add_argument("--full-enum-codebook", action="store_true",
                    help="exhaust every pq_codebook bit (~1.05M flips/index — slow)")
    ap.add_argument("--pq-retention-frac", type=float, default=config.PHASE2_PQ_RETENTION_FRAC)
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = os.path.join(config.ROOT, "artifacts")
    if args.smoke:
        args.s_large = min(args.s_large, SMOKE["s_large"])
        args.s_centroid = min(args.s_centroid, SMOKE["s_centroid"])
        args.s_codebook = min(args.s_codebook, SMOKE["s_codebook"])
        args.queries = min(args.queries, SMOKE["queries"])
        out = args.out or os.path.join(config.ROOT, "artifacts_smoke", "phase2")
    else:
        out = args.out or os.path.join(art_root, "phase2")
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    paths = make_paths(art_root, out)

    names = [n.strip() for n in args.indexes.split(",") if n.strip()]
    for n in names:
        if n not in PQ_GROUP:
            log(f"[phase2-T1] skipping {n!r}: not in PQ group {PQ_GROUP}")
    names = [n for n in names if n in PQ_GROUP]

    if args.aggregate_only:
        recs = []
        for name in names:
            if os.path.exists(paths["done"](name)):
                with open(paths["raw"](name)) as fh:
                    recs.extend(json.loads(line) for line in fh)
        write_map(out, aggregate_pq(recs, args.pq_retention_frac))
        log(f"[phase2-T1] aggregate-only: {len(recs)} records -> vuln_map_pq")
        log("\nPHASE2-T1 AGGREGATE OK")
        return 0

    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        knob_by = {r["index"]: r["knob_value"] for r in json.load(f)["indexes"]}

    nq = args.queries
    xq = data.load_query()[:nq].astype("float32")
    gt_ids = np.load(os.path.join(art_root, "gt", "gt_ids.npy"))[:nq]
    gt_dist = np.load(os.path.join(art_root, "gt", "gt_dist.npy"))[:nq]

    log(f"[phase2-T1] out={out} smoke={args.smoke} queries={nq} indexes={names} "
        f"S_large={args.s_large} S_centroid={args.s_centroid} S_codebook={args.s_codebook} "
        f"full_enum_codebook={args.full_enum_codebook}")

    all_records = []
    for name in names:
        if args.resume and os.path.exists(paths["done"](name)):
            log(f"[phase2-T1] {name}: already complete (--resume), reloading shard")
            with open(paths["raw"](name)) as fh:
                all_records.extend(json.loads(line) for line in fh)
            continue
        log(f"[phase2-T1] {name}: sweeping ...")
        all_records.extend(sweep_index(name, spec_by[name], knob_by[name], paths,
                                       xq, gt_ids, gt_dist, args))

    rows = aggregate_pq(all_records, args.pq_retention_frac)
    write_map(out, rows)
    log(f"[phase2-T1] wrote vuln_map_pq.{{json,csv}} ({len(rows)} cells)")

    if args.smoke:
        tags = sorted({r["bit_position_tag"] for r in all_records})
        assert any(t.startswith("code_b") for t in tags), "expected code_b* tags from pq_codes"
        assert all(r["baseline_mode"] == "own" for r in rows)
        log(f"        observed tags: {tags}")
        log(f"        sample row: {rows[0]['index']}/{rows[0]['region']}/{rows[0]['bit_position_tag']} "
            f"pct_cat(rel)={rows[0]['pct_catastrophic']} pct_cat(abs)={rows[0]['pct_catastrophic_abs']}")
        log("\nPHASE2-T1 SMOKE OK")
        return 0

    log("\nPHASE2-T1 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
