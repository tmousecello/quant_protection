#!/usr/bin/env python3
"""Phase 3 — E9: on-die-ECC MISCORRECTION x index REGION damage row (CIDR Fig 1(b)).

E6 measured three fault SHAPES against seven index REGIONS: a single cell, an 8 KB device row,
and a 512-row column stripe. The paper's motivational figure names one more event class, and it
is the one the DDR5 generation actually delivers to software: a MISCORRECTION.

DDR5 carries on-die ECC — single-error-correcting over 128 data bits + 8 parity per beat (A-003,
COMET/UCLA NanoCAD 2023; X-003, Criss et al., MEMSYS 2020 Fig. 5). Single-bit faults are
therefore mostly invisible to software (~80% of DDR5 faults masked, Chung MICRO 2025). What is
NOT invisible is a two-bit raw fault inside one ECC word: it exceeds SEC capacity, the syndrome
aliases onto an innocent bit, and the corrector flips a THIRD bit that was never broken. Software
sees 3 wrong bits confined to one ~16 B word (X-003 bounds an access's error span at <=16 bits).

So this event class is neither a single cell nor a row: it is a LOCALIZED MULTI-BIT BURST, and
whether that is benign or fatal depends entirely on which of RaBitQ's regions it lands in. This
driver measures exactly that — the same 7 strata x 30 seeds x 2 arms E6 used, for the one new
shape (420 evals).

WHY A SEPARATE DRIVER INSTEAD OF A 4TH E6 SHAPE: E6's published artifact (results/
jonathan-vuln-shapes/e6_results_merged.csv, 21 shards, its summary's row-count gate and shard
metadata) is built around a 3x7 grid. Growing SHAPES to 4 would silently retro-invalidate that
gate's meaning and make the merged CSV a partial grid. E9 instead REGISTERS its shape into E6's
dispatch (`phase3_e6_shapes.register_shape`) and reuses, verbatim and by import: the stratum
sampler, the anchor/seed derivation, `measure_cell` (both arms, the RecoveryGuard composition,
the per-eval fault-footprint gate), `bounds_check_full`, `classify_outcome`, `summarize`,
`write_csv` and `setup_context`'s preflight. No measurement logic is duplicated here.

DEVIATION FROM E6's OPERATING POINT: E9 sweeps at **ef=64** (clean recall@10 = 0.95035 on
SIFT1M) where E6 used ef=2000 (0.98376). At ~0.7 s/eval instead of ~15 s, the 420-eval grid runs
in minutes rather than hours. `delta_recall` is always measured against THIS run's own clean
baseline, re-measured through both arms' code paths at the same ef (setup_context gates that the
two agree), so every E9 number is internally consistent — but E6 and E9 deltas are NOT
interchangeable, because a 0.95 operating point has more slack above it than a 0.98 one. The
summary records both the baseline and the ef so a reader cannot miss it.

Arms are E6's: off = flip -> deserialize -> `search_corrupted`; on = fresh RecoveryGuard +
full-index `bounds_check_full` + `scrub_only` + `query_with_recovery`. Recall is always
recomputed in Python from returned ids via qp.metrics — never taken from the C++ side.

Outputs (under --out; default artifacts/phase3/e9, --smoke -> artifacts_smoke/phase3/e9):
  raw/e9.records.jsonl   one row per eval (anchor, coverage, field_hits, counters, error)
  raw/e9.done            completion marker (enables --resume)
  e9_results.csv         E6's 13-column schema, so E6+E9 rows concatenate into one matrix
  e9_summary.json        provenance meta + per-(stratum,arm) medians/p95 + smear + gates

Usage:
  python phase3_e9_miscorrection.py --adapter stub --smoke      # dev pipeline gate
  python phase3_e9_miscorrection.py --adapter real              # the 420-eval sweep
  python phase3_e9_miscorrection.py --adapter real --strata links --out-tag links   # one shard
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from qp import config, faults
from qp.rawio import RawWriter

import phase3_e6_shapes as e6
from phase3_e6_shapes import (ARMS, OUTCOMES, STRATA, RETENTION_TOL, csv_list, log,
                              measure_cell, sample_anchor, stub_sandbox_ok, summarize,
                              write_csv, _final_research_gate)

SHAPE = "miscorrection_word"

# ef=64 is the FAST operating point (clean recall@10 = 0.95035 on SIFT1M, ~0.7 s/eval) — see the
# module docstring's deviation note. word_bytes=16 is the 128-data-bit DDR5 beat (A-003/X-003);
# k_raw=2 is the smallest raw fault that exceeds SEC capacity, so observed bits = 3.
FULL = dict(seeds=30, ef=64, word_bytes=16, k_raw=2)
SMOKE = dict(FULL, seeds=2)

MISCORRECTION_CITATION = {
    "mechanism": ("DDR5 on-die ECC is SEC over 128 data + 8 parity bits per beat. A 2-bit RAW "
                  "fault inside one ECC word exceeds SEC capacity and MISCORRECTS: the decoder "
                  "flips a third, innocent bit. Software observes 3 wrong bits confined to one "
                  "~16 B word — a localized multi-bit burst, not a row or a stripe."),
    "sources": {
        "A-003": ("COMET: On-die and In-controller Collaborative Memory ECC Technique, UCLA "
                  "NanoCAD 2023 — 128 data + 8 parity per beat; a double-bit error can "
                  "miscorrect, producing a third erroneous bit within the beat boundary."),
        "X-003": ("Criss et al., Improving Memory Reliability by Bounding DRAM Faults, MEMSYS "
                  "2020, Fig. 5 — 128 data bits and 8 check-bits are read per access; errors "
                  "span up to 16 bits."),
    },
    "caveat": ("MODELING APPROXIMATION. The exact JEDEC JESD79-5 on-die-ECC parity-check matrix "
               "is confidential (A-GAP-03), so which bit a given double-error syndrome aliases "
               "onto cannot be reproduced; qp.faults.miscorrection_word draws the miscorrected "
               "bit uniformly within the word instead. Anchored in A-003/X-003: the 16 B word, "
               "the 2-raw -> 3-observed count, and the single-word confinement. Not anchored: "
               "the intra-word position of the miscorrected bit."),
}


def _kwargs_from_cfg(cfg):
    return {"word_bytes": int(cfg["word_bytes"]), "k_raw": int(cfg["k_raw"])}


# Registered, not appended to e6.SHAPES: E6's grid (and its published artifact's row-count gate)
# stays a 3x7 grid. See the module docstring.
e6.register_shape(SHAPE, faults.miscorrection_word, _kwargs_from_cfg)


def assert_row_count(records, seeds, strata=STRATA):
    """One shape x the (possibly sharded) strata slice x seeds x arms — no cell silently dropped."""
    expected = len(strata) * int(seeds) * len(ARMS)
    assert len(records) == expected, (
        f"row count {len(records)} != expected {expected} "
        f"(1 shape x {len(strata)} strata x {seeds} seeds x {len(ARMS)} arms)")
    return expected


def run(args):
    cfg = dict(SMOKE if args.smoke else FULL)
    for knob, cast in (("seeds", int), ("ef", int), ("word_bytes", int), ("k_raw", int)):
        if getattr(args, knob, None) is not None:
            cfg[knob] = cast(getattr(args, knob))

    strata = tuple(getattr(args, "strata", None) or STRATA)
    sharded = strata != STRATA

    ctx = e6.setup_context(args, cfg, prefix="e9")
    out, tag = ctx["out"], ctx["tag"]
    raw_path = os.path.join(out, "raw", f"e9{tag}.records.jsonl")
    done_path = os.path.join(out, "raw", f"e9{tag}.done")

    # E6 stamps its own grid-split rationale here; E9's grid is one shape, so say what E9 did.
    ctx["meta"]["spec_deviations"] = {
        "separate_driver": ("E9 registers `miscorrection_word` into E6's shape dispatch instead "
                            "of joining E6's SHAPES tuple, so E6's published 3x7 grid, its "
                            "row-count gate and its merged CSV keep their meaning. All "
                            "measurement logic (sampler, measure_cell, bounds_check_full, "
                            "classifier, summarizer) is imported from E6, not copied."),
        "operating_point": (f"E9 sweeps at ef={cfg['ef']} (clean recall@10 "
                            f"{ctx['clean_recall']:.5f}); E6 used ef=2000 (0.98376). "
                            "delta_recall is measured against this run's own baseline through "
                            "both arms' code paths, so E9 is internally consistent, but E6 and "
                            "E9 deltas are not interchangeable across operating points."),
    }
    ctx["meta"]["miscorrection_modeling"] = dict(
        MISCORRECTION_CITATION, word_bytes=int(cfg["word_bytes"]), k_raw=int(cfg["k_raw"]),
        observed_bits=int(cfg["k_raw"]) + 1)

    if args.resume and os.path.exists(done_path):
        log("[e9] --resume: reloading completed shard")
        with open(raw_path) as fh:
            records = [json.loads(line) for line in fh]
        prior = next((r["clean_recall"] for r in records if r.get("clean_recall") is not None),
                     None)
        if prior is not None and abs(prior - ctx["clean_recall"]) > RETENTION_TOL:
            raise RuntimeError(
                f"REPORT-AND-STOP: --resume shard was measured at clean recall@10={prior:.6f} "
                f"but this run's clean baseline is {ctx['clean_recall']:.6f} — different index, "
                f"ef, or adapter. Re-run without --resume.")
        leak = research_drift = None
    else:
        work = ctx["clean_buf"].copy()
        scratch = np.empty(work.size, dtype=bool)
        records = []
        t0 = time.time()
        if sharded:
            log(f"[e9] shard: strata={list(strata)} "
                f"({len(strata) * int(cfg['seeds']) * len(ARMS)} evals of "
                f"{len(STRATA) * int(cfg['seeds']) * len(ARMS)})")
        with RawWriter(raw_path, done_path=done_path) as w:
            for stratum in strata:
                tc = time.time()
                for i in range(int(cfg["seeds"])):
                    # Same (root, shape, stratum, i) derivation E6 uses, with E9's shape name in
                    # the mix — so a shard reproduces the full grid's cells exactly, and E9's
                    # anchors are independent of E6's rather than a re-run of them.
                    seed = e6.cell_seed(args.seed, SHAPE, stratum, i)
                    anchor = sample_anchor(ctx["rmap"], stratum, seed)
                    for arm in ARMS:            # paired: same anchor, same injector seed
                        records.append(w.write(measure_cell(
                            ctx, work, SHAPE, stratum, i, seed, anchor, arm, scratch=scratch)))
                cell = records[-2 * int(cfg["seeds"]):]
                hist = {o: sum(1 for r in cell if r["outcome"] == o) for o in OUTCOMES}
                log(f"[e9] {SHAPE:<20} {stratum:<12} {time.time() - tc:6.1f}s  "
                    f"bits~{int(np.median([r['bits_flipped'] for r in cell]))}  "
                    f"{ {k: v for k, v in hist.items() if v} }")
        leak = int(np.count_nonzero(work != ctx["clean_buf"]))
        if leak:
            raise RuntimeError(f"state leaked across the sweep: {leak} bytes differ from clean")
        research_drift = _final_research_gate(ctx, work)
        log(f"[e9] {len(records)} evals in {time.time() - t0:.1f}s "
            f"(leak {leak} bytes, re-search drift {research_drift:.2e})")

    expected = assert_row_count(records, cfg["seeds"], strata=strata)
    cells, smear = summarize(records, shapes=(SHAPE,), strata=strata)
    write_csv(os.path.join(out, f"e9_results{tag}.csv"), records)
    summary = {
        "experiment": "e9_miscorrection",
        "shapes": [SHAPE], "strata": list(strata), "arms": list(ARMS),
        "shard": {"is_shard": sharded, "full_grid_strata": list(STRATA),
                  "full_grid_rows": len(STRATA) * int(cfg["seeds"]) * len(ARMS),
                  "out_tag": getattr(args, "out_tag", None)},
        "seeds": int(cfg["seeds"]), "cfg": cfg,
        "clean_recall@10": ctx["clean_recall"],
        "cells": cells, "smear": smear,
        "sanity": {"row_count_ok": len(records) == expected, "expected_rows": expected,
                   "rows": len(records), "state_leak_bytes": leak,
                   "per_eval_footprint_gate_evals": 0 if leak is None else len(records),
                   "final_research_drift": research_drift,
                   "stub_sandbox_ok": stub_sandbox_ok(ctx["aname"], out)},
        "adapter": ctx["aname"], "meta": ctx["meta"],
    }
    with open(os.path.join(out, f"e9_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[e9] wrote e9_results{tag}.csv ({len(records)} rows) + e9_summary{tag}.json")
    return summary


def main(argv=None, return_summary=False):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, choices=["stub", "real", "auto"],
                    help="stub (dev) / real (x86) / auto (default: real if binaries else stub)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE['seeds']} seeds; outputs to artifacts_smoke/")
    ap.add_argument("--resume", action="store_true",
                    help="reload a completed raw shard instead of re-running the sweep")
    ap.add_argument("--seeds", type=int, default=None, help="replicates per stratum")
    ap.add_argument("--strata", type=csv_list("stratum", STRATA), default=None,
                    help=f"comma-separated subset of {list(STRATA)} — shards the sweep across "
                         f"processes. Cell seeds are (root, shape, stratum, i)-derived, so a "
                         f"shard's anchors and flips match the full sweep's. Pair with --out-tag.")
    ap.add_argument("--cliff-regions", dest="cliff_regions",
                    type=csv_list("cliff region", e6.CLIFF_REGIONS_ALLOWED), default=None,
                    help=f"comma-separated subset of {list(e6.CLIFF_REGIONS_ALLOWED)} that arm "
                         f"ON replicates and majority-votes (default "
                         f"{list(e6.CLIFF_REGIONS_DEFAULT)}). Same flag and same meaning as E6.")
    ap.add_argument("--seed", type=int, default=config.SEED, help="root seed")
    ap.add_argument("--ef", type=int, default=None,
                    help=f"search ef (default {FULL['ef']} — the fast operating point; E6 used "
                         f"2000, and deltas are not comparable across the two)")
    ap.add_argument("--word-bytes", dest="word_bytes", type=int, default=None,
                    help="ECC word size in bytes (default 16 = the DDR5 128-data-bit beat)")
    ap.add_argument("--k-raw", dest="k_raw", type=int, default=None,
                    help="raw fault bits inside the word (default 2); observed bits = k_raw + 1")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-search seconds; a hang under corruption records as crash")
    ap.add_argument("--weights", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-tag", dest="out_tag", default=None,
                    help="filename suffix so parallel/variant runs never clobber each other")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                                "phase3", "e9")
    summary = run(args)
    log("\nE9 OK")
    return summary if return_summary else 0


if __name__ == "__main__":
    sys.exit(main())
