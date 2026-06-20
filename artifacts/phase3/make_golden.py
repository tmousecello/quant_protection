"""Generate the Stage 0 golden-file fixture (code-generated; never hand-edit the JSON).

`compute()` returns a dict of known-good numbers for the three RaBitQ-agnostic components on a
fixed synthetic fixture. Running this module writes artifacts/phase3/fixtures/golden_smoke.json;
test_golden.py re-runs compute() and asserts it reproduces the committed file. Any intended
change must come from re-running this generator, so the golden stays a pure function of code.
"""
import json
import os

import numpy as np

from qp import faults, metrics
from qp.rabitq import layout
import phase3_cost as cost
import phase3_recall as pr

FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
GOLDEN = os.path.join(FIX_DIR, "golden_smoke.json")


def _fingerprint(positions):
    """Stable, human-inspectable summary of a flip-position list (order-independent checksum)."""
    offs = sorted(bp * 8 + bit for bp, bit in positions)
    return {"count": len(offs), "checksum": int(sum(offs)), "first5": offs[:5]}


def compute():
    out = {}

    # --- fault models (fixed buffer/region/seed) ---
    region = (0, 4096)
    out["uniform_p"] = _fingerprint(
        faults.uniform_p(np.zeros(4096, np.uint8), region, p=1e-3, seed=7))
    out["spatial_cluster"] = _fingerprint(
        faults.spatial_cluster(np.zeros(4096, np.uint8), region, W=8, k=3, n_win=20, seed=5))
    out["cross_row"] = _fingerprint(
        faults.cross_row(np.zeros(256, np.uint8), (0, 256), stride=64, seed=9))
    out["temporal_burst"] = _fingerprint(
        faults.temporal_burst(np.zeros(4096, np.uint8), region, [0, 0, 0, 12, 0], seed=2))
    cc = faults.CumulativeCorruption(np.zeros(4096, np.uint8), region, seed=4)
    out["cumulative_curve"] = cc.run([0, 0, 0, 50, 0, 0])

    # --- recall + collapse predicate ---
    gt = np.tile(np.arange(20), (30, 1))
    gt_dist = np.tile(np.linspace(1.0, 2.0, 20), (30, 1))
    pred = gt[:, :10].copy()
    pred[:, 7:] = -1                       # knock out 3 of the top-10 per query
    out["recall_block"] = {k: round(v, 6)
                           for k, v in pr.recall_block(pred, gt, gt_dist=gt_dist, ks=(1, 10)).items()}
    out["collapse"] = {
        "silent_true": metrics.is_silent_collapse(0.4, 0.95),
        "silent_false": metrics.is_silent_collapse(0.89, 0.95),
        "nan_inf_excluded": metrics.is_silent_collapse(0.0, 0.95, failure_mode=metrics.NAN_INF),
    }

    # --- cost ---
    rmap = {"regions": [{"name": "rotation", "byte_len": 64},
                        {"name": "bin_factors", "byte_len": 12}]}
    mc = cost.mem_cost(rmap, {"rotation": {"mult": 3, "checksum_bytes": 4},
                              "bin_factors": {"mult": 2}})
    out["mem_cost"] = {k: mc[k] for k in ("replication_bytes", "checksum_bytes", "total_bytes")}

    # --- RaBitQ layout (source-derived; SIFT b=7 params, no binaries needed) ---
    # SIFT: dim=128 -> padded_dim=128; total_bits=7 -> ex_bits=6; M=16 -> maxM0=32.
    pd, ex_bits, maxM0 = 128, 6, 32
    out["layout"] = {
        "header_bytes": layout.HEADER_BYTES,
        "bin_data_bytes": layout.bin_data_bytes(pd),     # 16 + 12 = 28
        "ex_data_bytes": layout.ex_data_bytes(pd, ex_bits),  # 96 + 8 = 104
        "rotation_bytes": layout.rotation_bytes(pd),     # 64
        "size_links_level0": layout.size_links_level0(maxM0),  # 132
        "element_region_lens": {r["name"]: r["byte_len"]
                                for r in layout.element_regions(pd, ex_bits, maxM0)},
    }
    return out


def main():
    os.makedirs(FIX_DIR, exist_ok=True)
    data = compute()
    with open(GOLDEN, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
