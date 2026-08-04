"""Per-query cost of the cliff layer, decomposed so the number cannot be misquoted.

WHY THIS EXISTS. An earlier revision of FINDING-centroid-gap.md quoted two different figures for
"the per-query cost" and computed its percentages from a third, because three distinct quantities
were measured in isolation and then treated as interchangeable:

  * the vote for ONE region (the centroids), which is not the whole scrub;
  * the bare `(a&b)|(b&c)|(a&c)` expression timed on its own, which no code path executes —
    `_majority_vote` also runs 4 x (XOR + `.any()`) around it;
  * `_scrub_cliff` end to end, which is the thing that actually runs, and was never reported.

So this script measures all three, fits the size-dependent slope, and checks that the parts add
up to the whole. A decomposition that does not reconcile is a decomposition that is wrong.

Two numbers come out, and which one to quote depends on the question:

  measured   what `_scrub_cliff` costs as implemented, dominated by per-region numpy dispatch;
  floor      the size-dependent work plus the CRC, i.e. what survives a C++/SIMD implementation.

Usage:
  python bench_cliff_cost.py                 # against the real index
  python bench_cliff_cost.py --json out.json
"""

import argparse
import json
import statistics
import time
import zlib

import numpy as np

from qp.rabitq import layout
from qp.rabitq.registry import get_adapter
from phase3_e5_recovery import RecoveryGuard, _majority_vote

GUARD_CFG = {"R": 3, "cliff_scrub": True, "anchor_every": 0, "chunk_size": 4096}
REGIONS = ("rotation", "centroids", "header")
FIT_SIZES = (64, 256, 1024, 4096, 8192, 16384, 32768, 65536)

# Search cost per query, from fitting load+search wall across ef on this index (10k queries).
# Recorded here so the ratios below are reproducible without re-running the C++ binary.
SEARCH_US_PER_QUERY = {64: 44.99, 2000: 1405.91}


def timed(fn, reps):
    """Median-of-3 batches, to keep one scheduling hiccup out of the number."""
    fn()
    batches = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        batches.append((time.perf_counter() - t0) / reps * 1e6)
    return statistics.median(batches)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args(argv)

    adapter = get_adapter("real")
    buf = adapter.serialize_index()
    rmap = layout.serialized_region_map(layout.parse_header(bytes(buf[:layout.HEADER_BYTES])),
                                        file_size=len(buf))
    sizes = {r["name"]: int(r["byte_len"]) for r in rmap["regions"]}
    protected_bytes = sum(sizes[r] for r in REGIONS)

    out = {"protected_bytes": protected_bytes,
           "region_bytes": {r: sizes[r] for r in REGIONS}}

    # --- the thing that actually runs -----------------------------------------------
    print("_scrub_cliff, clean buffer (this is the cost; everything else explains it)")
    scrub = {}
    for label, regs in [(r, (r,)) for r in REGIONS] + [("all three", REGIONS)]:
        g = RecoveryGuard(adapter, {**GUARD_CFG, "cliff_regions": regs}, rmap)
        g.init_from_clean(buf)
        work = buf.copy()
        scrub[label] = timed(lambda: g._scrub_cliff(work), 5000)
        print(f"  {label:<12} {scrub[label]:7.2f} us")
    out["scrub_us"] = scrub
    total = scrub["all three"]

    # --- where it goes ---------------------------------------------------------------
    xs, ys = [], []
    for bl in FIT_SIZES:
        a = np.random.default_rng(bl).integers(0, 256, bl, dtype=np.uint8)
        copies = [a.copy() for _ in range(3)]
        b = a.copy()
        xs.append(bl)
        ys.append(timed(lambda: _majority_vote(copies, b), 20000 if bl <= 8192 else 5000))
    slope, intercept = (float(v) for v in np.polyfit(xs, ys, 1))

    crc_us = 0.0
    for r in REGIONS:
        blob = np.random.default_rng(0).integers(0, 256, sizes[r], dtype=np.uint8).tobytes()
        crc_us += timed(lambda: zlib.crc32(blob), 50000)

    dispatch = len(REGIONS) * intercept
    bytework = protected_bytes * slope
    predicted = dispatch + bytework + crc_us
    out.update({"fit_intercept_us": intercept, "fit_slope_us_per_byte": slope,
                "dispatch_us": dispatch, "bytework_us": bytework, "crc_us": crc_us,
                "predicted_us": predicted, "measured_us": total,
                "reconcile_error_pct": 100 * (predicted - total) / total})

    print()
    print(f"decomposition (fit: cost = {intercept:.2f} us + {slope*1024:.3f} us/KB)")
    for name, us in (("numpy dispatch, size-independent", dispatch),
                     (f"byte work, {protected_bytes:,} B", bytework),
                     ("zlib.crc32", crc_us)):
        print(f"  {name:<34} {us:6.2f} us   {100*us/total:4.0f}%")
    print(f"  {'predicted':<34} {predicted:6.2f} us")
    print(f"  {'measured':<34} {total:6.2f} us   "
          f"(reconciles to {abs(out['reconcile_error_pct']):.1f}%)")
    if abs(out["reconcile_error_pct"]) > 10:
        print("  WARNING: the parts do not add up to the whole; the decomposition is wrong.")

    # --- against one search ----------------------------------------------------------
    floor = bytework + crc_us
    out["floor_us"] = floor
    print()
    print("share of one search (search-only, load excluded)")
    print(f"  {'':<34} {'measured':>10} {'floor':>10}")
    ratios = {}
    for ef, q in sorted(SEARCH_US_PER_QUERY.items()):
        ratios[ef] = {"measured_pct": 100 * total / q, "floor_pct": 100 * floor / q}
        print(f"  ef={ef:<5} ({q:8.2f} us/query){'':<11} "
              f"{100*total/q:9.2f}% {100*floor/q:9.2f}%")
    out["share_of_search"] = ratios
    print()
    print("  measured = as implemented here, dominated by per-region numpy dispatch")
    print("  floor    = byte work + CRC, i.e. what a C++/SIMD implementation would still pay")
    print()
    print(f"  scaling: the size term is {slope*1024:.3f} us/KB, and the centroid block is "
          f"num_cluster x padded_dim x 4.")
    for nclu in (16, 256, 1024):
        blk = nclu * rmap["header"]["padded_dim"] * layout.FLOAT
        print(f"    num_cluster={nclu:>5} -> {blk:>8,} B centroids, "
              f"byte-work term {blk*slope:8.2f} us")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
