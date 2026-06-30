"""Single-source provenance + units stamp for Phase 3 (Stage 1) result artifacts.

Every Stage-1 result JSON (E1 vuln_map / criticality, E3a, E3b) embeds the dict returned by
`collect_provenance` under a top-level ``"meta"`` key, mirroring phase0_build.py's
``{"meta": {...}, ...}`` shape, so a reader can tell WITHOUT guessing:

  * which platform actually produced the numbers — CONFIRMED, not inferred (see
    ``platform_confirmed_real``), and
  * the scale of every metric (the ``units`` legend) — so ``0.781`` (a percent) can never again
    be re-read as ``78`` (a fraction). vuln_map / criticality ``pct_*`` are percent 0-100; E3a/E3b
    ``collapse_frac`` / ``p1_single_bit_collapse`` and the criticality ``frac_*`` twins are
    fraction 0-1.

This module only READS state (platform, the resolved adapter, the index header, dep versions);
it never re-runs a measurement.
"""
import os
import platform
import socket
import subprocess

from qp import config
from qp.rabitq import adapter as real_adapter

# Metric scale legend — the fix for the percent-vs-fraction misread. Embedded in every meta block.
UNITS = {
    "pct_collapse": "percent_0_100",
    "pct_collapse_worst_bucket": "percent_0_100",
    "pct_collapse_overall": "percent_0_100",
    "pct_harmful": "percent_0_100",
    "pct_crash": "percent_0_100",
    "pct_catastrophic": "percent_0_100",
    "pct_benign": "percent_0_100",
    "frac_collapse_worst_bucket": "fraction_0_1",
    "frac_collapse_overall": "fraction_0_1",
    "p1_single_bit_collapse": "fraction_0_1",
    "collapse_frac": "fraction_0_1",
    "predicted": "fraction_0_1",
    "measured": "fraction_0_1",
    "dRecall@10": "abs_recall_delta_0_1",
    "mean_dRecall@10": "abs_recall_delta_0_1",
    "max_dRecall@10": "abs_recall_delta_0_1",
    "p99_dRecall@10": "abs_recall_delta_0_1",
}

CLEAN_TOL = 0.01  # |clean@10 - EXPECTED_CLEAN_RECALL10| within this -> tolerance_ok (on-plateau)


def _rabitq_lib_commit():
    """The exact RaBitQ-Library commit, READ at runtime via git (never assumed). None if unreadable."""
    repo = os.path.join(real_adapter.RABITQ_REPO, "third_party", "RaBitQ-Library")
    try:
        rev = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode == 0 and rev.stdout.strip():
            return rev.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _dep_versions():
    deps = {}
    try:
        import numpy
        deps["numpy"] = numpy.__version__
    except Exception:
        deps["numpy"] = None
    try:
        import faiss
        deps["faiss"] = faiss.__version__
    except Exception:                                   # stub path is faiss-free; absent is fine
        deps["faiss"] = None
    return deps


def collect_provenance(adapter, adapter_nm, clean, cfg, args, rmap=None):
    """Build the ``meta`` stamp for a Stage-1 result file.

    ``platform_confirmed_real`` is a TRIANGULATION, not an inference: True only when the resolved
    adapter is the real one AND the host arch is x86_64 AND the clean recall sits on the index's
    reference plateau. The stub's ``binaries_built()`` returns True on any host, so it is
    deliberately EXCLUDED from the confirmation (it is recorded for transparency only).
    """
    machine = platform.machine()
    expected = getattr(adapter, "EXPECTED_CLEAN_RECALL10", None)
    clean10 = clean.get("recall@10") if clean else None
    diff = abs(clean10 - expected) if (clean10 is not None and expected is not None) else None
    tol_ok = bool(diff is not None and diff <= CLEAN_TOL)

    confirmed_real = bool(adapter_nm == "real" and machine == "x86_64" and tol_ok)
    basis = [
        f"adapter_resolved={adapter_nm}",
        f"platform.machine()={machine}",
        f"clean@10={clean10} vs expected={expected} -> tolerance_ok={tol_ok}",
    ]

    geom = None
    try:
        header = (rmap or adapter.region_map())["header"]
        geom = {k: header[k] for k in ("dim", "padded_dim", "num_cluster", "ex_bits",
                                       "cur_element_count", "M", "maxM", "maxM0",
                                       "ef_construction") if k in header}
    except Exception:
        geom = None

    return {
        "adapter_type": adapter_nm,
        "platform": {
            "machine": machine,
            "system": platform.system(),
            "hostname": socket.gethostname(),
            "uname": list(os.uname()) if hasattr(os, "uname") else None,
        },
        "platform_confirmed_real": confirmed_real,
        "confirmation_basis": basis,
        "clean_baseline": {
            "recall@10": clean10,
            "expected": expected,
            "diff": round(diff, 6) if diff is not None else None,
            "tolerance_ok": tol_ok,
            "tol": CLEAN_TOL,
        },
        "index_geometry": geom,
        "rabitq": {
            "library_commit": _rabitq_lib_commit() if adapter_nm == "real" else None,
            "binaries_built_flag": getattr(adapter, "binaries_built", lambda: None)(),
            "note": ("binaries_built is NOT a platform signal (stub returns True); "
                     "platform is confirmed via platform_confirmed_real"),
        },
        "study_config": {
            "seed": getattr(args, "seed", config.SEED),
            "k": config.K,
            "dim": config.DIM,
            "ef": cfg.get("ef") if isinstance(cfg, dict) else None,
            "smoke": getattr(args, "smoke", None),
        },
        "deps": _dep_versions(),
        "units": dict(UNITS),
    }
