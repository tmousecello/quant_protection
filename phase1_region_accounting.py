#!/usr/bin/env python3
"""Phase 1 — Task A: region accounting (no bit flips).

Reframes the faults/MB axis by asking, per index, *what fraction of its bytes is even
recall-relevant*. Reads the Phase 0 byte-region maps, buckets every region
(recall_relevant / crash_structure / benign) via qp.buckets, sums bytes per bucket, and
flags the small-but-high-leverage regions (centroid / sq_scale / pq_codebook) that are
shared by many vectors.

One gap is repaired here: HNSW_SQ8's ~128MB of SQ8 storage codes are not mapped by
regions.py (only graph_meta / graph_edges / sq_scale are). Without them the headline
IVF_SQ8-vs-HNSW_SQ8 "same quantizer, different structure" comparison has nothing to
sample on the HNSW_SQ8 side, so we add a DOCUMENTED estimated `codes` tail (the bytes
after the last mapped region, same convention regions.py already uses for IVF codes). The
Phase 0 regions.json on disk is left untouched; augmented copies go under
<out>/phase1/regions_aug/ and the sweep consumes those.

Outputs (under --out, default artifacts/):
  region_accounting.json / .csv / .md
  phase1/regions_aug/<NAME>.regions.json     (original + added HNSW_SQ8 codes tail)

Usage:
  python phase1_region_accounting.py [--out artifacts/]
"""
import argparse
import csv
import json
import os
import sys

from qp import config, buckets


def log(msg):
    print(msg, flush=True)


def load_regions_json(name, regions_dir):
    with open(os.path.join(regions_dir, f"{name}.regions.json")) as f:
        return json.load(f)


def load_augmented_regions(name, regions_dir):
    """Region map for one index with a quantized-graph storage codes tail added if missing.

    Shared by the accounting step and the sensitivity sweep so both see identical regions.
    Returns the full {index, total_bytes, regions:[...]} dict.

    regions.py maps a quantized graph index (e.g. HNSW_SQ8) as graph_meta/graph_edges +
    sq_scale only; its ~code_size*ntotal storage codes are the uncovered tail. We add that
    tail as an estimated `codes` region (same convention regions.py uses for IVF codes) so
    the fp32-vs-SQ8 codes comparison has something to sample on the graph-SQ side. The check
    is structural (graph index, no codes region, non-empty tail) rather than a hardcoded
    name, so a future HNSW_PQ/SQ4 is handled too; fp32 HNSW already covers its tail with a
    `vectors` region (tail==0), making this a no-op there. Non-graph indexes are unchanged.
    """
    rmap = load_regions_json(name, regions_dir)
    regions = list(rmap["regions"])
    total = int(rmap["total_bytes"])
    is_graph = any(r["kind"] in ("graph_edges", "graph_meta") for r in regions)
    has_codes = any(r["kind"] == "codes" for r in regions)
    covered_end = max((r["byte_start"] + r["byte_len"] for r in regions), default=0)
    tail = total - covered_end
    if is_graph and not has_codes and tail > 0:
        regions.append({
            "name": "codes",
            "kind": "codes",
            "byte_start": int(covered_end),
            "byte_len": int(tail),
            "dtype": "uint8",
            "semantic": "quantized storage codes (estimated: tail after the last mapped "
                        "region; added in Phase 1 to enable the fp32-vs-SQ8 codes comparison)",
            "located": False,
        })
    return {"index": name, "total_bytes": total, "regions": regions}


def bucket_for(region):
    """bucket_of, but tolerant of the synthetic 'unaccounted' residual."""
    if region["kind"] == "unaccounted":
        return "benign"
    return buckets.bucket_of(region)


def account_one(name, regions_dir):
    rmap = load_augmented_regions(name, regions_dir)
    total = int(rmap["total_bytes"])
    regions = rmap["regions"]

    # residual bytes not covered by any region -> a synthetic benign 'unaccounted' line so
    # bucket percentages sum to 100%.
    covered = sum(r["byte_len"] for r in regions)
    residual = total - covered
    region_rows = []
    for r in regions:
        region_rows.append({
            "index": name,
            "region": r["name"],
            "kind": r["kind"],
            "bucket": bucket_for(r),
            "byte_len": int(r["byte_len"]),
            "bits": buckets.region_bits(r),
            "pct_of_total": 100.0 * r["byte_len"] / total,
            "element_dtype": buckets.element_dtype_of(name, r),
            "located": bool(r["located"]),
        })
    if residual > 0:
        region_rows.append({
            "index": name, "region": "unaccounted", "kind": "unaccounted",
            "bucket": "benign", "byte_len": int(residual),
            "bits": int(residual) * 8, "pct_of_total": 100.0 * residual / total,
            "element_dtype": "uint8", "located": False,
        })

    bucket_tot = {"recall_relevant": 0, "crash_structure": 0, "benign": 0}
    bucket_regions = {"recall_relevant": [], "crash_structure": [], "benign": []}
    for rr in region_rows:
        bucket_tot[rr["bucket"]] += rr["byte_len"]
        bucket_regions[rr["bucket"]].append(rr["region"])
    buckets_summary = {
        b: {"bytes": bucket_tot[b], "MB": bucket_tot[b] / 1e6,
            "pct": 100.0 * bucket_tot[b] / total, "regions": bucket_regions[b]}
        for b in bucket_tot
    }

    # small-but-high-leverage: <1% of total and a shared structure (centroid/scale/codebook).
    small_high_leverage = [
        {"name": rr["region"], "bytes": rr["byte_len"], "bits": rr["bits"],
         "pct": rr["pct_of_total"]}
        for rr in region_rows
        if rr["kind"] in ("centroid", "sq_scale", "pq_codebook") and rr["pct_of_total"] < 1.0
    ]

    summary = {
        "index": name,
        "total_bytes": total,
        "total_MB": total / 1e6,
        "buckets": buckets_summary,
        "small_high_leverage": small_high_leverage,
    }
    return summary, region_rows


def write_md(path, summaries, prediction_gap):
    lines = []
    lines.append("# Phase 1 — Region Accounting\n")
    lines.append("How much of each index's memory is even recall-relevant? A uniformly random "
                 "physical bit flip lands in proportion to bucket size, so this fraction sets a "
                 "floor on bit-flip sensitivity before quantization is even considered.\n")
    lines.append("Buckets: **recall_relevant** (flips deserialize and can silently degrade "
                 "recall), **crash_structure** (flips crash/are detected on load), **benign** "
                 "(padding/unaccounted).\n")
    lines.append("\n## Per-index bucket breakdown\n")
    lines.append("| index | total MB | recall_relevant | crash_structure | benign |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summaries:
        b = s["buckets"]
        def cell(name):
            x = b[name]
            return f"{x['MB']:.1f} MB ({x['pct']:.1f}%)"
        lines.append(f"| {s['index']} | {s['total_MB']:.1f} | "
                     f"{cell('recall_relevant')} | {cell('crash_structure')} | "
                     f"{cell('benign')} |")

    lines.append("\n## Small but high-leverage regions\n")
    lines.append("Tiny regions shared by/ pointing at many vectors — a single flip here can move "
                 "every dependent vector.\n")
    lines.append("| index | region | bytes | bits | % of total |")
    lines.append("|---|---|---:|---:|---:|")
    for s in summaries:
        for r in s["small_high_leverage"]:
            lines.append(f"| {s['index']} | {r['name']} | {r['bytes']:,} | {r['bits']:,} | "
                         f"{r['pct']:.4f}% |")

    lines.append("\n## Headline prediction (A4)\n")
    lines.append(
        f"**IVF_SQ8 vs HNSW_SQ8 — same scalar quantizer, opposite memory profile.** "
        f"IVF_SQ8 is **{prediction_gap['IVF_SQ8_recall_relevant_pct']:.1f}%** recall-relevant "
        f"(nearly its whole footprint is codes), so almost every random flip hits recall. "
        f"HNSW_SQ8 is only **{prediction_gap['HNSW_SQ8_recall_relevant_pct']:.1f}%** "
        f"recall-relevant — about "
        f"{100 - prediction_gap['HNSW_SQ8_recall_relevant_pct']:.0f}% of it is graph "
        f"structure (crash-detectable), so at the same faults/MB most flips land on "
        f"detectable structure rather than silently corrupting recall. Predicted "
        f"recall-relevant gap: **{prediction_gap['delta']:.1f} percentage points**. "
        f"The single-bit sweep (Task B) tests whether this footprint difference, not the "
        f"quantizer, dominates sensitivity.\n")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(config.ROOT, "artifacts"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    regions_dir = os.path.join(out, "regions")
    aug_dir = os.path.join(out, "phase1", "regions_aug")
    os.makedirs(aug_dir, exist_ok=True)

    log(f"[phase1-A] reading regions from {regions_dir}")
    summaries, all_region_rows = [], []
    pct_recall_relevant = {}
    for spec in config.INDEX_SPECS:
        name = spec["name"]
        # persist the augmented region map for the sweep to consume
        aug = load_augmented_regions(name, regions_dir)
        with open(os.path.join(aug_dir, f"{name}.regions.json"), "w") as f:
            json.dump(aug, f, indent=2)

        summary, region_rows = account_one(name, regions_dir)
        summaries.append(summary)
        all_region_rows.extend(region_rows)
        rr_pct = summary["buckets"]["recall_relevant"]["pct"]
        pct_recall_relevant[name] = rr_pct
        log(f"        {name:<12} total={summary['total_MB']:>7.1f}MB  "
            f"recall_relevant={rr_pct:>5.1f}%  "
            f"crash={summary['buckets']['crash_structure']['pct']:>5.1f}%")

    prediction_gap = {
        "IVF_SQ8_recall_relevant_pct": pct_recall_relevant["IVF_SQ8"],
        "HNSW_SQ8_recall_relevant_pct": pct_recall_relevant["HNSW_SQ8"],
        "delta": pct_recall_relevant["IVF_SQ8"] - pct_recall_relevant["HNSW_SQ8"],
    }

    doc = {
        "meta": {
            "dataset": config.DATASET, "seed": config.SEED, "k": config.K,
            "operating_point": config.OPERATING_POINT,
            "note": "HNSW_SQ8 'codes' is an added estimated tail (located=false).",
        },
        "indexes": summaries,
        "prediction_gap": prediction_gap,
    }
    with open(os.path.join(out, "region_accounting.json"), "w") as f:
        json.dump(doc, f, indent=2)

    csv_cols = ["index", "region", "kind", "bucket", "byte_len", "bits",
                "pct_of_total", "element_dtype", "located"]
    with open(os.path.join(out, "region_accounting.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for rr in all_region_rows:
            w.writerow({k: rr[k] for k in csv_cols})

    write_md(os.path.join(out, "region_accounting.md"), summaries, prediction_gap)

    log(f"\n[phase1-A] prediction gap: IVF_SQ8={prediction_gap['IVF_SQ8_recall_relevant_pct']:.1f}% "
        f"vs HNSW_SQ8={prediction_gap['HNSW_SQ8_recall_relevant_pct']:.1f}%  "
        f"delta={prediction_gap['delta']:.1f}pts")
    log(f"[phase1-A] wrote region_accounting.{{json,csv,md}} and {aug_dir}")
    log("\nPHASE1-A OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
