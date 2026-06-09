#!/usr/bin/env python3
"""Phase 2 — Tier 4: range-check detection guard.

Phase 0/1 found the catastrophic failures are silent-finite-wrong (0 crash, 0 NaN/Inf at the
operating point): a flipped exponent in `sq_scale` / `pq_codebook` / `centroid` turns a value
into something still finite but wildly out of its trained range. So the right guard is a
VALUE-RANGE check on the metadata, not a NaN/Inf check. This script implements that guard and
measures how much of the damage it catches.

How it works (cheap — no full deserialize, no search):
  1. expected_ranges: from each CLEAN index buffer, slice every metadata region
     (sq_scale / centroid / pq_codebook) as fp32 and record [min, max] (+ robust quantiles).
  2. guard(buf): for each metadata region, view its bytes as fp32 and flag if ANY value is
     non-finite or outside [min, max] (± an optional pad). Returns (detected, which region).
  3. evaluate: replay every metadata-region flip recorded in the Phase 1 / Tier 1 raw shards
     by re-applying it to the clean buffer and running the guard — tallying coverage of
     catastrophic vs moderate flips, the false-positive rate on clean + benign flips, and the
     guard's wall-clock cost relative to one query.

Inputs:
  clean buffers  artifacts/indexes/<NAME>.faissindex
  region maps    artifacts/phase1/regions_aug/<NAME>.regions.json
  flip replays   --raw-dirs (default artifacts/phase1/raw + <out>/raw): *.records.jsonl with
                 byte_pos/bit/region/dRecall@10/failure_mode from Phase 1 + Tier 1.

Outputs (under <out>/detection/):
  expected_ranges.json   per (index, region) trained [min,max] (+ q01/q99)
  coverage.csv           per (index, region, severity): n, n_detected, coverage_pct
  detection_summary.json overall coverage, false-positive rate, overhead ratio

Usage:
  python phase2_detection.py                  # full maps under artifacts/phase2
  python phase2_detection.py --smoke
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

import faiss

from qp import config, buckets, data
from qp import indexes as ix
from qp.flip import to_buffer, flip_bit
from phase1_region_accounting import load_augmented_regions


GUARD_KINDS = ("sq_scale", "centroid", "pq_codebook")


def log(msg):
    print(msg, flush=True)


def metadata_regions(name, regions_dir):
    rmap = load_augmented_regions(name, regions_dir)
    return [r for r in rmap["regions"] if r["kind"] in GUARD_KINDS]


def region_values(buf, region):
    """View a region's bytes as fp32 (all guarded kinds are float32).

    Trim to a whole number of fp32 elements: .view(np.float32) requires a multiple-of-4 byte
    length, and a mis-sized/estimated region (byte_len % 4 != 0) would otherwise raise and take
    down the whole detection pass.
    """
    bs, bl = region["byte_start"], region["byte_len"]
    bl -= bl % 4
    seg = np.asarray(buf[bs:bs + bl], dtype=np.uint8)
    return seg.view(np.float32)


def compute_ranges(buf, region):
    v = region_values(buf, region)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        # No finite reference values to bound against — record a degenerate range so the guard
        # falls back to its non-finite check only (rather than crashing on min/percentile).
        return {
            "min": 0.0, "max": 0.0, "q01": 0.0, "q99": 0.0, "degenerate": True,
            "n": int(v.size), "byte_start": int(region["byte_start"]),
            "byte_len": int(region["byte_len"]),
        }
    return {
        "min": float(finite.min()), "max": float(finite.max()),
        "q01": float(np.percentile(finite, 1)), "q99": float(np.percentile(finite, 99)),
        "n": int(v.size), "byte_start": int(region["byte_start"]),
        "byte_len": int(region["byte_len"]),
    }


def guard(buf, regions, ranges, pad):
    """Return (detected, hit_region). Flags non-finite or out-of-[min,max]±pad in any region."""
    for region in regions:
        rg = ranges[region["name"]]
        v = region_values(buf, region)
        if not np.isfinite(v).all():
            return True, region["name"]
        span = rg["max"] - rg["min"]
        lo, hi = rg["min"] - pad * span, rg["max"] + pad * span
        if v.min() < lo or v.max() > hi:
            return True, region["name"]
    return False, None


def severity(rec, harmful):
    """catastrophic / moderate / benign / crash from a flip record."""
    if rec.get("failure_mode") == "crash":
        return "crash"
    d10 = rec.get("dRecall@10")
    if d10 is None:
        return "crash"
    if rec.get("cat_abs") or rec.get("cat_rel") or d10 > harmful:
        return "catastrophic"
    if abs(d10) > buckets.BENIGN_ABS:
        return "moderate"
    return "benign"


def load_raw(raw_dirs):
    """All flip records from *.records.jsonl across the given dirs, grouped by index."""
    by_index = {}
    for d in raw_dirs:
        for path in glob.glob(os.path.join(d, "*.records.jsonl")):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # ingests shards from several producers/dirs — skip any record missing the
                    # fields this replay needs rather than aborting the whole pass on a KeyError.
                    if not all(k in r for k in ("index", "region", "byte_pos", "bit")):
                        continue
                    by_index.setdefault(r["index"], []).append(r)
    return by_index


def main():
    ap = argparse.ArgumentParser(description="Phase 2 Tier 4: range-check detection guard.")
    ap.add_argument("--raw-dirs", default=None,
                    help="comma-separated dirs of *.records.jsonl (default phase1/raw + <out>/raw)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="fractional pad on [min,max] before flagging (default 0 = strict)")
    ap.add_argument("--harmful-thresh", type=float, default=buckets.CATASTROPHIC_ABS)
    ap.add_argument("--queries", type=int, default=2000, help="queries for the overhead baseline")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = os.path.join(config.ROOT, "artifacts")
    out = os.path.abspath(args.out or (os.path.join(config.ROOT, "artifacts_smoke", "phase2")
                                       if args.smoke else os.path.join(art_root, "phase2")))
    det_dir = os.path.join(out, "detection")
    os.makedirs(det_dir, exist_ok=True)
    regions_dir = os.path.join(art_root, "phase1", "regions_aug")

    if args.raw_dirs:
        raw_dirs = [d.strip() for d in args.raw_dirs.split(",") if d.strip()]
    elif args.smoke:
        raw_dirs = [os.path.join(config.ROOT, "artifacts_smoke", "phase1", "raw"),
                    os.path.join(out, "raw")]
    else:
        raw_dirs = [os.path.join(art_root, "phase1", "raw"), os.path.join(out, "raw")]

    # --- 1. expected ranges + cached clean buffers / region lists -----------------------
    spec_by = {s["name"]: s for s in config.INDEX_SPECS}
    with open(os.path.join(art_root, "baseline.json")) as f:
        knob_by = {r["index"]: r["knob_value"] for r in json.load(f)["indexes"]}

    ranges_doc, clean_bufs, region_lists = {}, {}, {}
    for spec in config.INDEX_SPECS:
        name = spec["name"]
        regs = metadata_regions(name, regions_dir)
        if not regs:
            continue
        index = faiss.read_index(os.path.join(art_root, "indexes", f"{name}.faissindex"))
        buf = to_buffer(index)
        del index
        clean_bufs[name] = buf
        region_lists[name] = regs
        ranges_doc[name] = {r["name"]: compute_ranges(buf, r) for r in regs}
        log(f"[phase2-T4] {name}: ranges for {[r['name'] for r in regs]}")
    with open(os.path.join(det_dir, "expected_ranges.json"), "w") as f:
        json.dump(ranges_doc, f, indent=2)

    # --- 2. overhead baseline: guard scan time vs one query ----------------------------
    overhead = {}
    for name, buf in clean_bufs.items():
        regs, rg = region_lists[name], ranges_doc[name]
        t0 = time.perf_counter()
        for _ in range(20):
            guard(buf, regs, rg, args.pad)
        guard_s = (time.perf_counter() - t0) / 20.0
        index = faiss.read_index(os.path.join(art_root, "indexes", f"{name}.faissindex"))
        ix.set_knob(index, spec_by[name], knob_by[name])
        xq = data.load_query()[:args.queries].astype("float32")
        t0 = time.perf_counter()
        index.search(xq, 100)
        query_s = time.perf_counter() - t0
        del index
        overhead[name] = {"guard_seconds": guard_s, "query_seconds": query_s,
                          "guard_vs_query": guard_s / query_s if query_s else None}

    # --- 3. replay recorded flips through the guard ------------------------------------
    by_index = load_raw(raw_dirs)
    cov = {}            # (index, region, severity) -> [n, n_detected]
    fp = {"clean": [0, 0], "benign": [0, 0]}   # [n, n_flagged]
    guarded_region_names = {n: {r["name"] for r in region_lists[n]} for n in region_lists}

    for name, recs in by_index.items():
        if name not in clean_bufs:
            continue
        buf, regs, rg = clean_bufs[name], region_lists[name], ranges_doc[name]
        # clean buffer should never be flagged (ranges came from it).
        det_clean, _ = guard(buf, regs, rg, args.pad)
        fp["clean"][0] += 1; fp["clean"][1] += int(det_clean)
        for r in recs:
            if r["region"] not in guarded_region_names[name]:
                continue
            sev = severity(r, args.harmful_thresh)
            bp, bit = int(r["byte_pos"]), int(r["bit"])
            flip_bit(buf, bp, bit)
            try:
                detected, _ = guard(buf, regs, rg, args.pad)
            finally:
                flip_bit(buf, bp, bit)         # restore (XOR back) even if guard raised,
                                               # so the shared clean buffer can't drift
            key = (name, r["region"], sev)
            cov.setdefault(key, [0, 0])
            cov[key][0] += 1; cov[key][1] += int(detected)
            if sev == "benign":
                fp["benign"][0] += 1; fp["benign"][1] += int(detected)

    rows = []
    for (name, region, sev), (n, nd) in sorted(cov.items()):
        rows.append({"index": name, "region": region, "severity": sev, "n": n,
                     "n_detected": nd, "coverage_pct": 100.0 * nd / n if n else None})
    with open(os.path.join(det_dir, "coverage.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    def cov_for(sev):
        n = sum(r["n"] for r in rows if r["severity"] == sev)
        nd = sum(r["n_detected"] for r in rows if r["severity"] == sev)
        return n, nd, (100.0 * nd / n if n else None)

    cat = cov_for("catastrophic"); mod = cov_for("moderate")
    summary = {
        "catastrophic": {"n": cat[0], "detected": cat[1], "coverage_pct": cat[2]},
        "moderate": {"n": mod[0], "detected": mod[1], "coverage_pct": mod[2]},
        "false_positive": {
            "clean_flagged": fp["clean"][1], "clean_n": fp["clean"][0],
            "benign_flagged": fp["benign"][1], "benign_n": fp["benign"][0],
            "benign_fp_pct": (100.0 * fp["benign"][1] / fp["benign"][0]) if fp["benign"][0] else None,
        },
        "overhead": overhead, "pad": args.pad, "raw_dirs": raw_dirs,
    }
    with open(os.path.join(det_dir, "detection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"[phase2-T4] wrote detection/{{expected_ranges.json, coverage.csv, detection_summary.json}}")
    log(f"        catastrophic coverage: {cat[1]}/{cat[0]} "
        f"({cat[2]:.1f}%)" if cat[0] else "        catastrophic coverage: n/a (no records)")
    log(f"        moderate coverage:     {mod[1]}/{mod[0]} "
        f"({mod[2]:.1f}%)" if mod[0] else "        moderate coverage: n/a")
    log(f"        benign false-positive: {fp['benign'][1]}/{fp['benign'][0]}")

    if args.smoke:
        assert ranges_doc, "no expected ranges computed"
        if not rows:
            log("        [warn] no metadata flip records found in raw dirs "
                "(run Tier 1 / Phase 1 smoke first to populate); guard+ranges still verified")
        log("\nPHASE2-T4 SMOKE OK")
        return 0
    log("\nPHASE2-T4 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
