"""Verifications for qp.faults — the three fault models (Stage 0 component 1).

All run on a synthetic uint8 buffer (RaBitQ-agnostic), exactly as the brief specifies:
bit-count vs binomial CI, chi-square uniformity, per-window clustering, determinism, restore
roundtrip, and the temporal cumulative quiet->step curve.
"""
import numpy as np
import pytest
from scipy import stats

from qp import faults


def _buf(n_bytes=4096, fill=0):
    return np.full(n_bytes, fill, dtype=np.uint8)


# --- model 1: uniform_p ------------------------------------------------------

def test_uniform_p_count_in_binomial_ci():
    region = (0, 4096)            # 4096 bytes -> 32768 bits
    n_bits = 4096 * 8
    p = 1e-3
    buf = _buf()
    pos = faults.uniform_p(buf, region, p, seed=7)
    # 99.99% CI for Binomial(n_bits, p)
    lo, hi = stats.binom.ppf([5e-5, 1 - 5e-5], n_bits, p)
    assert lo <= len(pos) <= hi, f"flip count {len(pos)} outside binomial CI [{lo},{hi}]"


def test_uniform_p_positions_uniform_chisquare():
    region = (0, 8192)
    buf = _buf(8192)
    pos = faults.uniform_p(buf, region, p=5e-3, seed=11)
    # bucket flipped bit-offsets into 16 bins; expect ~uniform
    n_bits = 8192 * 8
    offs = np.array([b * 8 + bit for (b, bit) in pos])
    counts, _ = np.histogram(offs, bins=16, range=(0, n_bits))
    chi = stats.chisquare(counts)
    assert chi.pvalue > 0.01, f"positions not uniform (chi2 p={chi.pvalue})"


def test_uniform_p_deterministic_and_restore_roundtrip():
    region = (10, 2000)
    a, b = _buf(), _buf()
    pa = faults.uniform_p(a, region, p=2e-3, seed=42)
    pb = faults.uniform_p(b, region, p=2e-3, seed=42)
    assert pa == pb, "same seed must give identical positions"
    assert np.array_equal(a, b)
    before = _buf()
    work = before.copy()
    pos = faults.uniform_p(work, region, p=2e-3, seed=42)
    assert not np.array_equal(work, before)          # actually changed something
    faults.restore(work, pos)
    assert np.array_equal(work, before), "inject->restore must be byte-identical"


def test_uniform_p_confined_to_region():
    region = (100, 50)           # bytes [100,150)
    buf = _buf()
    pos = faults.uniform_p(buf, region, p=0.5, seed=3)
    assert all(100 <= bp < 150 for bp, _ in pos)


# --- model 2: spatial_cluster / cross_row ------------------------------------

def test_spatial_cluster_k_per_window_and_clustered():
    region = (0, 4096)
    W, k, n_win = 8, 3, 20
    buf = _buf()
    pos = faults.spatial_cluster(buf, region, W=W, k=k, n_win=n_win, seed=5)
    assert len(pos) == k * n_win
    offs = sorted(b * 8 + bit for (b, bit) in pos)
    # group consecutive flips by their W-window slot; each occupied slot must hold exactly k
    slots = {}
    for o in offs:
        slots.setdefault(o // W, []).append(o)
    assert len(slots) == n_win, "expected n_win distinct windows"
    assert all(len(v) == k for v in slots.values()), "each window must hold exactly k flips"
    # clustered, not uniform: k flips fall inside a W-bit span far tighter than chance.
    for v in slots.values():
        assert max(v) - min(v) < W, "flips in a window must lie within W bits"


def test_cross_row_pairs_same_offset():
    region = (0, 256)
    stride = 64
    buf = _buf(256)
    pos = faults.cross_row(buf, region, stride=stride, seed=9)
    assert len(pos) == 2
    (b0, bit0), (b1, bit1) = pos
    assert bit0 == bit1, "cross-row flips must hit the same in-row bit"
    assert b1 - b0 == stride, "second flip must be one row (stride) later"


def test_spatial_cluster_deterministic_and_restore():
    region = (0, 4096)
    before = _buf()
    work = before.copy()
    pos = faults.spatial_cluster(work, region, W=8, k=2, n_win=10, seed=1)
    pos2 = faults.spatial_cluster(before.copy(), region, W=8, k=2, n_win=10, seed=1)
    assert pos == pos2
    faults.restore(work, pos)
    assert np.array_equal(work, before)


# --- model 3: temporal_burst / CumulativeCorruption --------------------------

def test_temporal_burst_total_and_restore():
    region = (0, 4096)
    timeline = [0, 0, 0, 12, 0]
    before = _buf()
    work = before.copy()
    pos = faults.temporal_burst(work, region, timeline, seed=2)
    assert len(pos) == sum(timeline)
    faults.restore(work, pos)
    assert np.array_equal(work, before)


def test_cumulative_curve_quiet_then_step():
    region = (0, 4096)
    timeline = [0, 0, 0, 50, 0, 0]       # quiet, then a burst, then quiet
    cc = faults.CumulativeCorruption(_buf(), region, seed=4)
    curve = cc.run(timeline)
    # quiet ticks keep the cumulative count flat; the trigger steps it up; stays flat after.
    assert curve == [0, 0, 0, 50, 50, 50], f"unexpected cumulative curve {curve}"
    assert cc.corrupted_fraction == 50 / (4096 * 8)


def test_cumulative_monotonic_and_deterministic():
    region = (0, 4096)
    timeline = [5, 0, 7, 0, 3]
    a = faults.CumulativeCorruption(_buf(), region, seed=8).run(timeline)
    b = faults.CumulativeCorruption(_buf(), region, seed=8).run(timeline)
    assert a == b, "same seed -> identical cumulative timeline"
    assert all(a[i] >= a[i - 1] for i in range(1, len(a))), "cumulative must be non-decreasing"
    assert a[-1] == sum(timeline)


def test_cumulative_reset_restores_buffer():
    region = (0, 1024)
    before = _buf(1024)
    cc = faults.CumulativeCorruption(before, region, seed=6)
    cc.run([10, 0, 5])
    assert not np.array_equal(before, _buf(1024))
    cc.reset()
    assert np.array_equal(before, _buf(1024)), "reset (scrub) must restore the buffer"


# --- guards ------------------------------------------------------------------

def test_region_and_param_validation():
    buf = _buf()
    with pytest.raises(ValueError):
        faults.uniform_p(buf, (0, 0), p=0.1, seed=1)        # empty region
    with pytest.raises(ValueError):
        faults.spatial_cluster(buf, (0, 16), W=8, k=9, seed=1, n_win=1)  # k>W
    with pytest.raises(ValueError):
        faults.temporal_burst(buf, (0, 1), [9999], seed=1)  # burst exceeds region
