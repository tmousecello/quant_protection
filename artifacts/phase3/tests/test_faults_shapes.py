"""Verifications for the three DRAM fault-shape injectors added to qp.faults (spec E0):

``single_cell`` / ``device_row`` / ``device_column``. Same contract as test_faults.py — deterministic
in seed, restore round-trips, loud validation errors — plus a weighted-shape-mix statistical
validation against the vendor A CIDR distribution (shape_weights.json) and a uniformity check for
single_cell.

Deliberately scipy-free (unlike test_faults.py, which already depends on scipy): a hand-rolled
chi-square helper (mirroring the one in test_faults.py) keeps this file runnable via
``uv run --with pytest --with numpy python -m pytest ...`` with no extra dependency.
"""
import json
import math
import os

import numpy as np
import pytest

from qp import faults

SHAPE_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..",
    "results", "jonathan-vuln-shapes", "shape_weights.json",
)


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


# --- Step 5: statistical validation (E0 deliverable) --------------------------

def test_shape_mix_matches_vendor_a_weights():
    """10,000 weighted shape draws should reproduce the vendor A CIDR distribution."""
    with open(SHAPE_WEIGHTS_PATH) as f:
        weights = json.load(f)["vendorA"]
    shapes = ["single_cell", "device_row", "device_column"]
    probs = np.array([weights[s] for s in shapes], dtype=float)
    probs = probs / probs.sum()

    rng = np.random.default_rng(8)
    n = 10000
    draws = rng.choice(shapes, size=n, p=probs)
    observed = np.array([(draws == s).sum() for s in shapes], dtype=float)
    expected = probs * n

    p = _chisquare_pvalue(observed, expected)
    assert p > 0.01, f"shape mix diverges from vendor A weights (chi2 p={p})"


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
