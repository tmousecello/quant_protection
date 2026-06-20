#!/usr/bin/env python3
"""Phase 3 — Stage 1, E3b: spatial (W,k) clustering vs uniform at a fixed bit budget (RaBitQ).

Phase 2 found exposure ∝ 1/dispersion: a fixed number of flips concentrated in a small structure
is more dangerous than the same flips spread out. E3b tests that on RaBitQ. The FIXED-bit-budget
dispersion comparison is clustered vs uniform:
  clustered   qp.faults.spatial_cluster(W, k, n_win): n_win tight W-bit windows, k flips each
  uniform     the same budget of bits spread uniformly across the structure
A third distribution is reported alongside but is NOT budget-matched:
  cross_row   qp.faults.cross_row: the rowhammer adjacent-row victim pair — an inherent 2-bit
              pattern, so it is reported at its true budget_bits=2 and excluded from the
              clustered-vs-uniform verdict (comparing it at 2 bits to a 16-bit budget would read
              as "less dangerous" purely from flipping fewer bits).
and reports mean collapse fraction + mean ΔRecall@10 for each — so "clustered on a small structure
should be worse" is measured, not assumed. Injectors are imported from qp.faults (single source).

Adapter-injected + config-driven like E1/E3a; reuses E1's apply_and_measure + clean_baseline.
Output: <out>/e3b.json.

Usage:
  python phase3_e3b_spatial.py --adapter stub --smoke
  python phase3_e3b_spatial.py             # workstation full run (auto -> real)
"""
import argparse
import json
import os
import sys

import numpy as np

from qp import config, faults
from qp.rabitq import get_adapter, adapter_name
import phase3_e1_vuln as e1
from phase3_e3a_additivity import structure_region, sample_k_positions

STRUCTS = ["rotation", "bin_factors"]
FULL = dict(n_trials=40, budget=16, W=8, k_per_win=4, ef=2000)
SMOKE = dict(n_trials=8, budget=8, W=8, k_per_win=2, ef=64)


def _positions_via(byte_start, byte_len, seed, cfg, mode, budget):
    """Flip positions from a qp.faults injector, computed on a REGION-sized scratch.

    The injectors flip-in-place and return positions; we run them on a small throwaway buffer
    sized to the region (region=(0, byte_len)) — NOT a copy of the whole index, which can be GB on
    the real adapter — then shift the returned positions by byte_start. The injectors' random
    draws are region-relative (byte_start only enters at the final (byte,bit) mapping), so the
    shifted positions are identical to injecting directly into the full buffer.
    """
    scratch = np.zeros(byte_len, dtype=np.uint8)
    region = (0, byte_len)
    if mode == "clustered":
        n_win = max(1, budget // cfg["k_per_win"])
        rel = faults.spatial_cluster(scratch, region, W=cfg["W"], k=cfg["k_per_win"],
                                     n_win=n_win, seed=seed)
    elif mode == "cross_row":
        stride = max(1, byte_len // 2)            # region-spanning so off varies (not pinned to 0)
        rel = faults.cross_row(scratch, region, stride=stride, seed=seed)
    else:
        raise ValueError(mode)
    return [(byte_start + bp, bit) for bp, bit in rel]


def run(args):
    cfg = dict(SMOKE if args.smoke else FULL)
    cfg["ef"] = args.ef if args.ef is not None else cfg["ef"]
    timeout = args.timeout
    adapter = get_adapter(args.adapter)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_corrupt.index")

    ref_buf = adapter.serialize_index()
    rmap = adapter.region_map()
    clean = e1.clean_baseline(adapter, ref_buf, tmp, cfg, timeout)
    e1.log(f"[e3b] adapter={adapter_name(adapter)} clean@10={clean['recall@10']:.4f} cfg={cfg}")

    results = []
    for struct in STRUCTS:
        try:
            bstart, blen = structure_region(rmap, struct)
        except KeyError:
            e1.log(f"[e3b] {struct}: absent, skip")
            continue
        budget = min(cfg["budget"], blen * 8)
        modes = {}
        # Deterministic per-structure seed (stable integer index, NOT per-process-salted hash(str)).
        rng = np.random.default_rng([args.seed, STRUCTS.index(struct)])
        for mode in ("clustered", "uniform", "cross_row"):
            colls, drops = [], []
            nflips = 0
            for t in range(cfg["n_trials"]):
                seed = int(rng.integers(1 << 30))
                if mode == "uniform":
                    pos = sample_k_positions(bstart, blen, budget,
                                             np.random.default_rng(seed))
                else:
                    pos = _positions_via(bstart, blen, seed, cfg, mode, budget)
                nflips = len(pos)                            # actual flips (2 for cross_row)
                rec = e1.apply_and_measure(adapter, ref_buf, tmp, pos, clean, cfg, timeout)
                colls.append(int(bool(rec.get("is_silent_collapse"))))
                if rec.get("dRecall@10") is not None:
                    drops.append(rec["dRecall@10"])
            modes[mode] = {"n_trials": cfg["n_trials"], "budget_bits": nflips,
                           "collapse_frac": round(float(np.mean(colls)), 4),
                           "mean_dRecall@10": round(float(np.mean(drops)), 4) if drops else None,
                           "n_crash": cfg["n_trials"] - len(drops)}
        # exposure ∝ 1/dispersion: clustered collapse_frac should be >= uniform on a small struct.
        # cross_row is an inherent 2-bit pattern (budget_bits=2), reported but EXCLUDED from the
        # budget-matched verdict so its lower count isn't misread as a dispersion effect.
        modes["cross_row"]["note"] = "2-bit rowhammer victim pair; not budget-matched"
        modes["clustered_more_dangerous"] = (
            modes["clustered"]["collapse_frac"] >= modes["uniform"]["collapse_frac"])
        results.append({"structure": struct, "modes": modes})
        e1.log(f"[e3b] {struct}: clustered={modes['clustered']['collapse_frac']} "
               f"uniform={modes['uniform']['collapse_frac']} "
               f"cross_row={modes['cross_row']['collapse_frac']}")

    # phase1 D1: the shared ref_buf must be byte-pristine after all inject/restore cycles.
    drift_tol = 1e-9 if adapter_name(adapter) == "stub" else 1e-6
    e1.assert_no_state_leak(adapter, ref_buf, tmp, clean, cfg, timeout, drift_tol)

    with open(os.path.join(out, "e3b.json"), "w") as f:
        json.dump({"adapter": adapter_name(adapter), "results": results}, f, indent=2)
    e1.log(f"[e3b] wrote e3b.json ({len(results)} structures)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, choices=["stub", "real", "auto"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=config.PHASE2_FLIP_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = (os.path.join(config.ROOT, "artifacts_smoke", "phase3", "e3b") if args.smoke
                    else os.path.join(config.ROOT, "artifacts", "phase3", "e3b"))
    run(args)
    e1.log("\nE3b OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
