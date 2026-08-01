"""Verifications for the DRAM fault-shape injectors added to qp.faults (spec E0):

``single_cell`` / ``device_row`` / ``device_column`` / ``random_shape``. Same contract as
test_faults.py — deterministic in seed, restore round-trips, loud validation errors — plus a
weighted-shape-mix statistical validation of the ``random_shape`` DISPATCHER against the
vendor A CIDR distribution (shape_weights.json) and a uniformity check for single_cell.

The dispatcher test exercises qp.faults.random_shape itself (not just numpy's rng.choice
against the same weights it's being checked against) — it would fail if shape_weights.json
were corrupted, or if random_shape's own weighting logic were wrong.

Deliberately scipy-free (unlike test_faults.py, which already depends on scipy): a hand-rolled
chi-square helper (mirroring the one in test_faults.py) keeps this file runnable via
``uv run --with pytest --with numpy python -m pytest ...`` with no extra dependency.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from qp import faults

SHAPE_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[5]
    / "results" / "jonathan-vuln-shapes" / "shape_weights.json"
)
_HAVE_SHAPE_WEIGHTS = SHAPE_WEIGHTS_PATH.exists()


def _buf(n=1 << 16):  # 64 KiB
    return np.zeros(n, dtype=np.uint8)


def _chisquare_pvalue(observed, expected):
    """Scipy-free chi-square goodness-of-fit p-value (Wilson-Hilferty approx to chi2 CDF).

    Mirrors the pattern test_faults.py uses via scipy.stats.chisquare, but avoids the scipy
    dependency for this file. observed/expected are equal-length sequences of counts.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    stat = float(np.sum((observed - expected) ** 2 / expected))
    df = len(observed) - 1
    # Wilson-Hilferty: (chi2/df)^(1/3) is approximately normal with known mean/var.
    if df <= 0:
        return 1.0
    h = 2.0 / (9.0 * df)
    z = ((stat / df) ** (1.0 / 3.0) - (1 - h)) / math.sqrt(h)
    # upper-tail p-value = P(Z > z) via the standard normal survival function
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# --- single_cell --------------------------------------------------------------

def test_single_cell_one_bit_in_region_and_restores():
    buf = _buf(); region = (1000, 5000)
    pos, rec = faults.single_cell(buf, region, seed=7)
    assert len(pos) == 1 and rec["bits_flipped"] == 1
    byte, _ = pos[0]; assert 1000 <= byte < 6000
    faults.restore(buf, pos); assert not buf.any()


# --- device_row ----------------------------------------------------------------

def test_device_row_density_and_alignment():
    buf = _buf(1 << 20); region = (300000, 1000)
    pos, rec = faults.device_row(buf, region, seed=3)
    lo, hi = rec["coverage"]
    assert lo % 8192 == 0 and hi - lo == 8192
    assert lo <= 300000 + 999 and hi > 300000          # anchor byte inside region's block
    n = rec["bits_flipped"]                            # Binomial(65536, 0.5): ±4σ ≈ ±512
    assert abs(n - 32768) < 512 * 4
    faults.restore(buf, pos); assert not buf.any()


def test_row_clamps_at_buffer_end():
    buf = _buf(8192 + 100)                             # anchor in the partial tail block
    pos, rec = faults.device_row(buf, (8192, 100), seed=1)
    assert rec["coverage"][1] <= len(buf)


# --- device_column ---------------------------------------------------------------

def test_device_column_stripe_geometry():
    buf = _buf(1 << 22); region = (0, 1 << 22)
    pos, rec = faults.device_column(buf, region, seed=5, n_rows=64)
    bytes_hit = sorted(b for b, _ in pos)
    assert len(pos) == 64
    assert all((b - rec["col_offset"]) % 8192 == 0 for b in bytes_hit)
    assert len({bit for _, bit in pos}) == 1           # fixed bit lane
    faults.restore(buf, pos); assert not buf.any()


# --- shared contract ---------------------------------------------------------

def test_determinism():
    for fn, kw in [(faults.single_cell, {}), (faults.device_row, {}),
                   (faults.device_column, {"n_rows": 16})]:
        a, _ = fn(_buf(1 << 20), (0, 1 << 20), seed=42, **kw)
        b, _ = fn(_buf(1 << 20), (0, 1 << 20), seed=42, **kw)
        assert a == b


def test_records_json_serializable():
    for fn, kw in [(faults.single_cell, {}), (faults.device_row, {}),
                   (faults.device_column, {"n_rows": 8})]:
        _, rec = fn(_buf(1 << 20), (0, 1 << 20), seed=1, **kw)
        json.dumps(rec)  # raises TypeError on any leftover numpy scalar


def test_records_have_bytes_touched():
    """bytes_touched is an exact distinct-byte count; coverage is a (possibly loose) span."""
    for fn, kw in [(faults.single_cell, {}), (faults.device_row, {}),
                   (faults.device_column, {"n_rows": 8})]:
        pos, rec = fn(_buf(1 << 20), (0, 1 << 20), seed=1, **kw)
        assert rec["bytes_touched"] == len({b for b, _ in pos})
        lo, hi = rec["coverage"]
        assert rec["bytes_touched"] <= hi - lo         # exact count never exceeds the span


def test_shape_validation_errors():
    buf = _buf()
    with pytest.raises(ValueError):
        faults.single_cell(buf, (0, 0), seed=1)                 # empty region
    with pytest.raises(ValueError):
        faults.device_row(buf, (-5, 40), seed=1)                # negative byte_start
    with pytest.raises(ValueError):
        faults.device_row(buf, (0, 40), seed=1, row_bytes=0)    # bad row_bytes
    with pytest.raises(ValueError):
        faults.device_column(buf, (0, 40), seed=1, n_rows=0)    # bad n_rows
    with pytest.raises(ValueError):
        faults.device_column(buf, (0, 40), seed=1, row_bytes=-1)


# --- random_shape dispatcher --------------------------------------------------

_TOY_WEIGHTS = {"single_cell": 78.12, "device_row": 9.67, "device_column": 12.21}


def test_random_shape_returns_matching_injector_output():
    buf = _buf(1 << 20); region = (0, 1 << 20)
    shape_name, pos, rec = faults.random_shape(buf, region, seed=2, weights=_TOY_WEIGHTS)
    assert shape_name == rec["shape"]
    assert shape_name in _TOY_WEIGHTS


def test_random_shape_deterministic():
    a = faults.random_shape(_buf(1 << 20), (0, 1 << 20), seed=42, weights=_TOY_WEIGHTS)
    b = faults.random_shape(_buf(1 << 20), (0, 1 << 20), seed=42, weights=_TOY_WEIGHTS)
    assert a == b


def test_random_shape_validation_errors():
    buf = _buf()
    with pytest.raises(ValueError):
        faults.random_shape(buf, (0, 100), seed=1, weights={"single_cell": 1.0})  # missing keys
    with pytest.raises(ValueError):
        faults.random_shape(buf, (0, 100), seed=1,
                             weights={"single_cell": 1.0, "device_row": -1.0, "device_column": 1.0})


# --- Step 5: statistical validation (E0 deliverable) --------------------------

@pytest.mark.skipif(not _HAVE_SHAPE_WEIGHTS,
                     reason="monorepo weights file not present in standalone checkout")
def test_shape_weights_json_pinned():
    """Pin the data file's exact vendor A values so a corrupted/rewritten file is caught here,
    not silently accepted by the downstream chi-square test (which only checks the observed
    mix is consistent with WHATEVER the file currently says)."""
    with open(SHAPE_WEIGHTS_PATH) as f:
        weights = json.load(f)["vendorA"]
    assert weights == {"single_cell": 78.12, "device_row": 9.67, "device_column": 12.21}


@pytest.mark.skipif(not _HAVE_SHAPE_WEIGHTS,
                     reason="monorepo weights file not present in standalone checkout")
def test_shape_mix_matches_vendor_a_weights():
    """10,000 faults.random_shape draws should reproduce the vendor A CIDR distribution.

    Exercises the actual dispatcher (not a bare re-implementation of weighted sampling) so a
    bug in random_shape's own weighting logic, or a corrupted shape_weights.json, would show up
    here — see test_shape_weights_json_pinned for the latter, pinned separately.
    """
    with open(SHAPE_WEIGHTS_PATH) as f:
        weights = json.load(f)["vendorA"]
    shapes = ["single_cell", "device_row", "device_column"]
    buf = _buf(1 << 16)
    region = (0, 1 << 16)
    n = 10000
    counts = {s: 0 for s in shapes}
    for seed in range(n):
        shape_name, _pos, _rec = faults.random_shape(buf, region, seed=seed, weights=weights)
        counts[shape_name] += 1

    observed = np.array([counts[s] for s in shapes], dtype=float)
    probs = np.array([weights[s] for s in shapes], dtype=float)
    probs = probs / probs.sum()
    expected = probs * n

    p = _chisquare_pvalue(observed, expected)
    assert p > 0.01, f"dispatcher shape mix diverges from vendor A weights (chi2 p={p}); counts={counts}"


def test_single_cell_positions_uniform_ks():
    """10,000 single_cell draws in one region should land ~uniformly (KS-style bucket test)."""
    region = (0, 8192)
    n = 10000
    n_bits = 8192 * 8
    offs = []
    for seed in range(n):
        buf = _buf(8192)
        pos, _ = faults.single_cell(buf, region, seed=seed)
        b, bit = pos[0]
        offs.append(b * 8 + bit)
    offs = np.array(offs)
    counts, _ = np.histogram(offs, bins=32, range=(0, n_bits))
    expected = np.full(32, n / 32.0)
    p = _chisquare_pvalue(counts, expected)
    assert p > 0.01, f"single_cell positions not uniform (chi2 p={p})"
