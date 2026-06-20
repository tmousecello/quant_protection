#!/usr/bin/env python3
"""Phase 3 — Stage 1, E3a: multi-bit additivity check for RaBitQ small-critical structures.

The single-bit map (E1) measures p1 = fraction of single-bit flips in a structure that cause a
silent collapse. The rollup model predicts the multi-bit collapse probability of k independent
flips as P_pred(k) = 1 - (1 - p1)^k (linear superposition of independent collapse events). E3a
tests that prediction: for each small-critical structure (rotation, bin_factors) it estimates p1,
predicts P(k) over a k grid, then ACTUALLY injects k random bits many times and measures the
empirical collapse fraction. It reports where additivity holds vs fails (failure = saturation or
compound effects: |measured - predicted| > tol).

Adapter-injected + config-driven exactly like E1 (--adapter {stub,real,auto}); develop/test on the
stub, run real on the x86 workstation. Reuses E1's apply_and_measure + clean_baseline and the
qp.metrics collapse predicate (single source of truth). Output: <out>/e3a.json.

Usage:
  python phase3_e3a_additivity.py --adapter stub --smoke
  python phase3_e3a_additivity.py            # workstation full run (auto -> real)
"""
import argparse
import json
import os
import sys

import numpy as np

from qp import config
from qp.rabitq import get_adapter, adapter_name, layout
import phase3_e1_vuln as e1

STRUCTS = ["rotation", "bin_factors"]
FULL = dict(p1_cap=512, n_trials=40, k_grid=[1, 2, 4, 8, 16, 32], ef=2000, tol=0.15)
SMOKE = dict(p1_cap=64, n_trials=8, k_grid=[1, 2, 4, 8], ef=64, tol=0.2)


def structure_region(rmap, struct):
    """(byte_start, byte_len) for a structure: rotation = global tail; factors = element 0 field.

    Raises KeyError when the structure is absent so the callers' `except KeyError: skip` contract
    is uniform across the global and per-vector branches (per-vector path already raises KeyError).
    """
    if struct in ("rotation", "centroids", "header"):
        r = e1._region(rmap, struct)
        if r is None or r["byte_start"] is None:
            raise KeyError(f"global region {struct!r} absent or unlocated in region map")
        return r["byte_start"], r["byte_len"]
    return layout.element_field_range(rmap, struct, 0)


def sample_k_positions(byte_start, byte_len, k, rng):
    """k distinct (byte,bit) positions drawn uniformly within a region (no replacement)."""
    n_bits = byte_len * 8
    offs = rng.choice(n_bits, size=min(k, n_bits), replace=False)
    return [((byte_start * 8 + int(o)) // 8, (byte_start * 8 + int(o)) % 8) for o in offs]


def estimate_p1(adapter, ref_buf, tmp, byte_start, byte_len, clean, cfg, timeout, cap):
    """Single-bit collapse fraction over a (capped, evenly spread) set of the structure's bits."""
    n_bits = byte_len * 8
    take = min(cap, n_bits)
    bit_idxs = np.linspace(0, n_bits - 1, take).astype(int)
    n_coll = 0
    for bo in bit_idxs:
        abs_bit = byte_start * 8 + int(bo)
        rec = e1.apply_and_measure(adapter, ref_buf, tmp, [(abs_bit // 8, abs_bit % 8)],
                                   clean, cfg, timeout)
        n_coll += int(bool(rec.get("is_silent_collapse")))
    return n_coll / take, take


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
    e1.log(f"[e3a] adapter={adapter_name(adapter)} clean@10={clean['recall@10']:.4f} cfg={cfg}")

    results = []
    for struct in STRUCTS:
        try:
            bstart, blen = structure_region(rmap, struct)
        except KeyError:
            e1.log(f"[e3a] {struct}: absent, skip")
            continue
        # Deterministic per-structure seed (stable integer index, NOT the per-process-salted
        # builtin hash(str)) so a given --seed reproduces the sampling across runs and machines.
        rng = np.random.default_rng([args.seed, STRUCTS.index(struct)])
        p1, n_p1 = estimate_p1(adapter, ref_buf, tmp, bstart, blen, clean, cfg, timeout,
                               cfg["p1_cap"])
        curve = []
        for k in cfg["k_grid"]:
            if k > blen * 8:
                continue
            n_coll = 0
            for _ in range(cfg["n_trials"]):
                pos = sample_k_positions(bstart, blen, k, rng)
                rec = e1.apply_and_measure(adapter, ref_buf, tmp, pos, clean, cfg, timeout)
                n_coll += int(bool(rec.get("is_silent_collapse")))
            measured = n_coll / cfg["n_trials"]
            predicted = 1.0 - (1.0 - p1) ** k
            curve.append({"k": k, "predicted": round(predicted, 4), "measured": round(measured, 4),
                          "abs_diff": round(abs(measured - predicted), 4),
                          "additive": abs(measured - predicted) <= cfg["tol"]})
        results.append({"structure": struct, "p1_single_bit_collapse": round(p1, 4),
                        "n_p1_samples": n_p1, "n_trials": cfg["n_trials"],
                        "tol": cfg["tol"], "curve": curve,
                        "additivity_holds": all(c["additive"] for c in curve) if curve else None})
        e1.log(f"[e3a] {struct}: p1={p1:.3f} -> " +
               " ".join(f"k{c['k']}:{c['measured']}/{c['predicted']}" for c in curve))

    # phase1 D1: the shared ref_buf must be byte-pristine after all multi-bit inject/restore cycles.
    drift_tol = 1e-9 if adapter_name(adapter) == "stub" else 1e-6
    e1.assert_no_state_leak(adapter, ref_buf, tmp, clean, cfg, timeout, drift_tol)

    with open(os.path.join(out, "e3a.json"), "w") as f:
        json.dump({"adapter": adapter_name(adapter), "results": results}, f, indent=2)
    e1.log(f"[e3a] wrote e3a.json ({len(results)} structures)")
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
        args.out = (os.path.join(config.ROOT, "artifacts_smoke", "phase3", "e3a") if args.smoke
                    else os.path.join(config.ROOT, "artifacts", "phase3", "e3a"))
    run(args)
    e1.log("\nE3a OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
