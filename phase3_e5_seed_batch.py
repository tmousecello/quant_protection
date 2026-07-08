"""Phase 3 Stage 3 — task #2: multi-seed fuse first-failure distribution.

Run 4's tick-2 first vote failure is a single-seed (1234) sample; the §5 statistics note
estimates the expectation at O(10) ticks. This runner turns that into a distribution:
one legacy-fuse run (--inject-replicas, NO --cliff-scrub) per root seed, recall off
(counters only, seconds per run), first-failure tick = first tick whose
counters.cliff_irrecoverable >= 1.

Modes (composable; the formal invocations live in run_stage3_x86.sh Phase B):
  --launch       one no-recall e5 subprocess per seed, bounded parallel pool
  --analyze      aggregate first-failure ticks -> first_fail_summary.json
  --determinism  re-run one seed under a second tag, byte-compare the records jsonl
  --timelines    full-recall 100-tick timelines for the min/max first-failure seeds
  --smoke        3 seeds x 30 ticks, output under artifacts_smoke/

Replica seed lanes use the SeedSequence derivation (phase3_e5_recovery.replica_lane_seed),
so the distribution is free of the old XOR lane's aliasing risk across adjacent roots.
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from qp import config

E5_RUNNER = os.path.join(config.ROOT, "phase3_e5_recovery.py")
STEM_RE = re.compile(r"e5_uniform_accum_rotation_replicas_seed(\d+)\.records\.jsonl$")

SMOKE_SEEDS = 3
SMOKE_TICKS = 30
FULL_SEEDS = 30
FULL_TICKS = 150   # P(no fail within 100) ~ 2%/seed at ~3.8%/tick; 150 makes censoring negligible


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on synthetic rows)
# ---------------------------------------------------------------------------

def first_fail_tick(rows):
    """First tick whose cumulative cliff_irrecoverable count is >= 1; None if censored."""
    for row in rows:
        if int(row["counters"]["cliff_irrecoverable"]) >= 1:
            return int(row["tick"])
    return None


def summarize_first_fails(first_fails, ticks):
    """Distribution stats over per-seed first-failure ticks ({seed: tick|None})."""
    observed = sorted(t for t in first_fails.values() if t is not None)
    censored = [s for s, t in first_fails.items() if t is None]
    stats = {
        "n_seeds": len(first_fails),
        "n_observed": len(observed),
        "n_censored": len(censored),
        "censored_seeds": sorted(censored),
        "ticks_per_run": int(ticks),
    }
    if observed:
        stats.update({
            "min": observed[0],
            "max": observed[-1],
            "median": _quantile(observed, 0.50),
            "q1": _quantile(observed, 0.25),
            "q3": _quantile(observed, 0.75),
            "iqr": _quantile(observed, 0.75) - _quantile(observed, 0.25),
            "mean": sum(observed) / len(observed),
        })
    return stats


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile on a sorted list (numpy 'linear' method)."""
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def analytic_estimate(p=0.005, bits=512, R=3):
    """§5-note analytic model of the per-tick vote-failure probability.

    After every successful vote the copies are resynced to the majority, so each tick
    every replica carries a fresh Binomial(bits, p) batch of dirty bits (E[b] = bits*p).
    A vote fails when >= 2 replicas are dirty at the SAME bit position: birthday-style
    pairwise collision, P(fail/tick) ~ C(R,2) * E[b]^2 / bits. First-failure tick is then
    Geometric(P): median = ln(0.5)/ln(1-P), mean = 1/P.
    """
    eb = bits * p
    p_fail = min(1.0, math.comb(R, 2) * (eb ** 2) / bits)
    return {
        "p_per_bit": p, "bits": bits, "R": R,
        "expected_dirty_bits_per_replica": eb,
        "p_fail_per_tick": p_fail,
        "geometric_median": math.log(0.5) / math.log(1.0 - p_fail),
        "geometric_mean": 1.0 / p_fail,
    }


def records_identical(path_a, path_b):
    """Byte-compare two records files (rows carry no timestamps -> must be identical)."""
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        return fa.read() == fb.read()


def load_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

def _e5_cmd(seed, ticks, out_dir, out_tag, adapter, no_recall=True):
    cmd = [sys.executable, E5_RUNNER,
           "--adapter", adapter, "--region", "rotation", "--pattern", "uniform_accum",
           "--inject-replicas",
           "--ticks", str(int(ticks)), "--seed", str(int(seed)),
           "--out-tag", out_tag, "--out-dir", out_dir]
    if no_recall:
        cmd.append("--no-recall")
    return cmd


def _run_one(seed, ticks, out_dir, out_tag, adapter, no_recall=True):
    log_path = os.path.join(out_dir, f"{out_tag}.log")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(_e5_cmd(seed, ticks, out_dir, out_tag, adapter, no_recall),
                              stdout=log_f, stderr=subprocess.STDOUT, cwd=config.ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} run failed (rc={proc.returncode}); see {log_path}")
    return seed


def launch(seeds, ticks, out_dir, adapter, jobs):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[seed-batch] launching {len(seeds)} runs x {ticks} ticks "
          f"(adapter={adapter}, jobs={jobs}) -> {out_dir}")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(_run_one, s, ticks, out_dir, f"seed{s}", adapter)
                for s in seeds]
        for f in futs:
            f.result()   # propagate the first failure loudly
    print(f"[seed-batch] all {len(seeds)} runs done")


def batch_record_files(out_dir):
    """Per-seed records files from --launch only (excludes r2/full tags via strict regex)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "e5_*_seed*.records.jsonl"))):
        m = STEM_RE.search(os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def analyze(out_dir, ticks):
    files = batch_record_files(out_dir)
    if not files:
        raise RuntimeError(f"REPORT-AND-STOP: no per-seed records under {out_dir}; "
                           f"run --launch first")
    per_seed = {}
    for seed, path in files.items():
        per_seed[seed] = first_fail_tick(load_records(path))
    stats = summarize_first_fails(per_seed, ticks)
    analytic = analytic_estimate()

    # Provenance: lift the meta stamp from one per-seed summary (all runs same host/config).
    meta = None
    summaries = sorted(glob.glob(os.path.join(out_dir, "e5_*_seed*[0-9].json")))
    if summaries:
        with open(summaries[0]) as fh:
            meta = json.load(fh).get("meta")

    out = {
        "first_fail_tick_per_seed": {str(k): v for k, v in sorted(per_seed.items())},
        "stats": stats,
        "analytic": analytic,
        "seed_lane": "SeedSequence([root, tick, r+1]) (phase3_e5_recovery.replica_lane_seed)",
        "note": ("first failure = first tick with counters.cliff_irrecoverable >= 1 under "
                 "legacy fuse semantics (--inject-replicas, cliff_scrub off); None = censored "
                 "(no failure within ticks_per_run)"),
        "meta": meta,
    }
    out_path = os.path.join(out_dir, "first_fail_summary.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[seed-batch] analyze -> {out_path}")
    print(f"  observed={stats['n_observed']}/{stats['n_seeds']} "
          f"censored={stats['n_censored']}")
    if stats.get("median") is not None:
        print(f"  first-fail tick: median={stats['median']:.1f} "
              f"IQR=[{stats['q1']:.1f},{stats['q3']:.1f}] "
              f"min={stats['min']} max={stats['max']}  "
              f"(analytic geometric median~{analytic['geometric_median']:.1f})")
    return out


def determinism_gate(out_dir, ticks, adapter, seed=None):
    files = batch_record_files(out_dir)
    if not files:
        raise RuntimeError(f"REPORT-AND-STOP: no per-seed records under {out_dir}")
    seed = seed if seed is not None else sorted(files)[0]
    print(f"[seed-batch] determinism gate: re-running seed {seed}")
    _run_one(seed, ticks, out_dir, f"seed{seed}r2", adapter)
    a = files[seed]
    b = os.path.join(out_dir, os.path.basename(a).replace(f"seed{seed}.", f"seed{seed}r2."))
    same = records_identical(a, b)
    if not same:
        raise RuntimeError(f"REPORT-AND-STOP: determinism gate FAILED — {a} != {b}")
    print(f"[seed-batch] determinism gate OK (seed {seed}: records byte-identical)")
    return True


def run_timelines(out_dir, adapter, jobs=2, ticks=100):
    summary_path = os.path.join(out_dir, "first_fail_summary.json")
    if not os.path.isfile(summary_path):
        raise RuntimeError("REPORT-AND-STOP: first_fail_summary.json missing; run --analyze")
    with open(summary_path) as fh:
        per_seed = json.load(fh)["first_fail_tick_per_seed"]
    observed = {int(s): t for s, t in per_seed.items() if t is not None}
    if not observed:
        raise RuntimeError("REPORT-AND-STOP: no observed first failures to pick timelines from")
    earliest = min(observed, key=lambda s: (observed[s], s))
    latest = max(observed, key=lambda s: (observed[s], -s))
    picks = [earliest] if earliest == latest else [earliest, latest]
    print(f"[seed-batch] full-recall timelines for seeds {picks} "
          f"(first-fail ticks {[observed[s] for s in picks]})")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(_run_one, s, ticks, out_dir, f"seed{s}full", adapter,
                            False)  # no_recall=False: these ARE the recall timelines
                for s in picks]
        for f in futs:
            f.result()
    print(f"[seed-batch] timelines done -> {out_dir}/e5_*_seed{{{','.join(map(str, picks))}}}full.*")
    return picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E5 multi-seed fuse first-failure batch")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--timelines", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_SEEDS} seeds x {SMOKE_TICKS} ticks -> artifacts_smoke/")
    ap.add_argument("--adapter", default="real", choices=("stub", "real", "auto"))
    ap.add_argument("--seeds-start", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    n_seeds = args.n_seeds or (SMOKE_SEEDS if args.smoke else FULL_SEEDS)
    ticks = args.ticks or (SMOKE_TICKS if args.smoke else FULL_TICKS)
    out_dir = args.out_dir or os.path.join(
        config.ROOT, "artifacts_smoke" if args.smoke else "artifacts", "phase3", "e5_seeds")
    seeds = list(range(args.seeds_start, args.seeds_start + n_seeds))

    if not (args.launch or args.analyze or args.determinism or args.timelines):
        ap.error("pick at least one of --launch/--analyze/--determinism/--timelines")

    if args.launch:
        launch(seeds, ticks, out_dir, args.adapter, args.jobs)
    if args.analyze:
        analyze(out_dir, ticks)
    if args.determinism:
        determinism_gate(out_dir, ticks, args.adapter)
    if args.timelines:
        run_timelines(out_dir, args.adapter,
                      ticks=(10 if args.smoke else 100))


if __name__ == "__main__":
    main()
