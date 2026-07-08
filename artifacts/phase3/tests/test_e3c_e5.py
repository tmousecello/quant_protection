"""Stage 2 stub tests: E3c (TemporalCorruptor) + E5 (RecoveryGuard) + §1 contract.

All tests run on arm64 (stub adapter, no x86 binaries). Tests cover:
  - E3c: monotone accumulation, spatial-distribution difference, double-flip XOR semantic,
         determinism and reset.
  - E5 cliff: majority-vote repair, irrecoverable detection (≥majority copies wrong).
  - E5 slope: CRC detection, EB-fallback trigger, counter tracking, lazy reload,
              parity repair and parity-exceeded fallback.
  - E5 bounds-check: OOB pointer skipped, no crash.
  - Integration: §1 mini-timeline with E3c + E5 jointly.
"""

import os
import sys
import tempfile
import zlib

import numpy as np
import pytest

# Make sure the project root is on the path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qp import config, metrics
from qp.rabitq import stub_adapter as adapter
from qp.rabitq import eb_policy, layout
from phase3_e3c_temporal import TemporalCorruptor, PATTERNS, SMOKE_CFG, _resolve_region
from phase3_e5_recovery import RecoveryGuard, SMOKE_CFG as E5_SMOKE_CFG


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _clean():
    return adapter.serialize_index()


def _rmap():
    return adapter.region_map()


def _rot_region(rmap=None):
    rmap = rmap or _rmap()
    return _resolve_region(rmap, "rotation")


def _guard(rmap=None, cfg=None):
    rmap = rmap or _rmap()
    cfg = cfg or {**E5_SMOKE_CFG, "chunk_size": 16, "reload_threshold": 0.5}
    g = RecoveryGuard(adapter, cfg, rmap)
    g.init_from_clean(_clean())
    return g, _clean()


# ---------------------------------------------------------------------------
# eb_policy unit test
# ---------------------------------------------------------------------------

def test_eb_policy_formula():
    """eb_rank_dist matches Samuel's formula: est + (est - low) = est + f_error * g_error."""
    est, f_error, g_error = 1.5, 0.1, 0.2
    low = est - f_error * g_error
    expected = est + (est - low)
    result = eb_policy.eb_rank_dist(est, f_error, g_error)
    assert abs(result - expected) < 1e-9
    assert result > est  # pessimistic: always > plain estimate


# ---------------------------------------------------------------------------
# E3c tests
# ---------------------------------------------------------------------------

class TestE3cUniformAccum:
    def setup_method(self):
        self.rmap = _rmap()
        self.region = _rot_region(self.rmap)
        self.cfg = {**SMOKE_CFG, "p": 0.01}
        self.corruptor = TemporalCorruptor(self.region, self.cfg)

    def test_monotone_accumulation(self):
        """Bit count is non-decreasing across ticks (modulo rare double-flip cancellations)."""
        buf = _clean()
        counts = []
        for tick in range(10):
            self.corruptor.inject_step(buf, "uniform_accum", seed=tick)
            cc = self.corruptor.cumulative_corruption()
            counts.append(cc["bits_flipped"])
        # Strictly non-decreasing is NOT guaranteed (double-flips cancel), but
        # should rarely decrease: assert the final count is > 0 and the trend goes up
        assert counts[-1] > 0, "expected some bits corrupted after 10 ticks at p=0.01"
        assert counts[-1] >= counts[0], "final count should not be less than first-tick count"

    def test_accumulation_rate(self):
        """Per-tick expected flip count ≈ n_bits * p (within 5σ)."""
        _, byte_len = self.region
        n_bits = byte_len * 8   # 512 for rotation
        p = float(self.cfg["p"])
        buf = _clean()
        self.corruptor.inject_step(buf, "uniform_accum", seed=99)
        cc = self.corruptor.cumulative_corruption()
        # After 1 tick: expected ≈ n_bits * p = 5.12
        expected = n_bits * p
        std = (n_bits * p * (1 - p)) ** 0.5
        assert abs(cc["bits_flipped"] - expected) <= 5 * max(std, 1), (
            f"first-tick count {cc['bits_flipped']} far from expected {expected:.1f}")

    def test_double_flip_xor_semantic(self):
        """Same position flipped twice → net XOR = 0 (not counted as corrupted)."""
        region = self.region
        buf = _clean()
        corruptor = TemporalCorruptor(region, {"p": 1.0})   # p=1 → flip everything
        corruptor.inject_step(buf, "uniform_accum", seed=42)
        first_dirty = len(corruptor._dirty_bits)
        assert first_dirty > 0, "p=1.0 should flip all bits"

        # Flip everything again with same seed → all bits cancel
        corruptor.inject_step(buf, "uniform_accum", seed=42)
        second_dirty = len(corruptor._dirty_bits)
        # A bit-exact double-flip: all n bits toggled twice = 0 corrupted
        assert second_dirty < first_dirty, (
            "second tick with same seed should cancel some/all flips")

    def test_xor_double_flip_restores_buf(self):
        """A bit flipped twice is physically restored in buf."""
        region = self.region
        clean = _clean()
        buf = clean.copy()
        corruptor = TemporalCorruptor(region, {"p": 0.5})
        corruptor.inject_step(buf, "uniform_accum", seed=7)
        corruptor.inject_step(buf, "uniform_accum", seed=7)   # same seed → same positions
        # Bits flipped twice are restored; the remaining dirty bits may not be zero
        # (because p=0.5 means ~half the bits flipped; the same half flip again → restore)
        # All doubly-flipped bits should be back to clean value
        # We verify: buf[i] for clean positions equals clean[i]
        bs, bl = region
        for rel in range(bl * 8):
            byte_abs = bs + rel // 8
            bit = rel % 8
            # If not in dirty_bits, this bit should match clean
            if rel not in corruptor._dirty_bits:
                assert (buf[byte_abs] >> bit) & 1 == (clean[byte_abs] >> bit) & 1, (
                    f"non-dirty bit {rel} doesn't match clean")

    def test_deterministic_same_seed(self):
        """Same seed → same trajectory."""
        region = self.region
        cfg = {**SMOKE_CFG, "p": 0.02}
        buf1 = _clean()
        c1 = TemporalCorruptor(region, cfg)
        for tick in range(5):
            c1.inject_step(buf1, "uniform_accum", seed=tick * 3)

        buf2 = _clean()
        c2 = TemporalCorruptor(region, cfg)
        for tick in range(5):
            c2.inject_step(buf2, "uniform_accum", seed=tick * 3)

        assert np.array_equal(buf1, buf2), "same seeds must produce identical buffers"
        assert c1._dirty_bits == c2._dirty_bits, "dirty-bits sets must match"

    def test_reset_restores_buf(self):
        """reset() restores buf to clean and clears state."""
        region = self.region
        clean = _clean()
        buf = clean.copy()
        corruptor = TemporalCorruptor(region, {**SMOKE_CFG, "p": 0.01})
        for tick in range(5):
            corruptor.inject_step(buf, "uniform_accum", seed=tick)
        assert not np.array_equal(buf, clean), "buf should be dirty after 5 ticks"

        corruptor.reset(buf)
        assert np.array_equal(buf, clean), "reset() must restore buf byte-for-byte"
        assert len(corruptor._dirty_bits) == 0
        assert corruptor._tick == 0

    def test_reset_then_replay_same_trajectory(self):
        """After reset, replaying same seeds gives same dirty_bits."""
        region = self.region
        clean = _clean()
        buf = clean.copy()
        corruptor = TemporalCorruptor(region, {**SMOKE_CFG, "p": 0.02})
        for tick in range(5):
            corruptor.inject_step(buf, "uniform_accum", seed=tick)
        first_dirty = frozenset(corruptor._dirty_bits)

        corruptor.reset(buf)
        for tick in range(5):
            corruptor.inject_step(buf, "uniform_accum", seed=tick)
        second_dirty = frozenset(corruptor._dirty_bits)

        assert first_dirty == second_dirty, "replay after reset must give same dirty set"


class TestE3cClusteredVsUniform:
    def test_spatial_distribution_differs(self):
        """clustered_accum concentrates in fewer windows than uniform_accum for same flip budget."""
        rmap = _rmap()
        region = _rot_region(rmap)
        _, byte_len = region
        n_bits = byte_len * 8  # 512

        W = 32   # window width in bits
        k_per_win = 4
        n_win = 2   # total = 8 flips per tick

        clustered_cfg = {**SMOKE_CFG, "W": W, "k": k_per_win, "n_win": n_win}
        uniform_cfg = {**SMOKE_CFG, "p": k_per_win * n_win / n_bits}

        buf_c = _clean()
        buf_u = _clean()
        c_corruptor = TemporalCorruptor(region, clustered_cfg)
        u_corruptor = TemporalCorruptor(region, uniform_cfg)

        c_corruptor.inject_step(buf_c, "clustered_accum", seed=0)
        u_corruptor.inject_step(buf_u, "uniform_accum", seed=0)

        def window_density(dirty_bits, w):
            """Max number of dirty bits in any W-bit window."""
            max_density = 0
            for start in range(0, n_bits, w):
                count = sum(1 for b in dirty_bits if start <= b < start + w)
                max_density = max(max_density, count)
            return max_density

        c_density = window_density(c_corruptor._dirty_bits, W)
        u_density = window_density(u_corruptor._dirty_bits, W)

        # Clustered injects k_per_win=4 in each of n_win windows → each window gets exactly k_per_win
        # Uniform distributes randomly → per-window density << k_per_win expected
        assert c_density >= k_per_win, (
            f"clustered max window density {c_density} should be >= k={k_per_win}")
        # For uniform at low p, the chance of 4 hits in any 32-bit window is very low
        # (Binomial(32, 8/512): P(≥4) ≈ 0.001) — this is a statistical check, not exact
        # We just verify the clustered has strictly higher density at 1 tick:
        assert c_density >= u_density, (
            f"clustered density {c_density} should be >= uniform density {u_density}")


class TestE3cRecoveryDesync:
    """Regression (#1): cumulative_corruption() must track the real buffer, not stale bookkeeping,
    even when ANOTHER writer (E5 cliff/slope repair) cleans the shared buf in place."""

    def test_cumulative_corruption_tracks_external_repair(self):
        rmap = _rmap()
        region = _rot_region(rmap)
        bs, bl = region
        clean = _clean()
        buf = clean.copy()
        corr = TemporalCorruptor(region, {"p": 0.5})

        def true_count():
            return int(np.unpackbits(buf[bs:bs + bl] ^ clean[bs:bs + bl]).sum())

        corr.inject_step(buf, "uniform_accum", seed=42)
        assert corr.cumulative_corruption()["bits_flipped"] == true_count() > 0

        # Simulate an E5 cliff repair: clean the rotation region in place, behind E3c's back.
        buf[bs:bs + bl] = clean[bs:bs + bl]
        assert true_count() == 0
        assert corr.cumulative_corruption()["bits_flipped"] == 0, (
            "metric must reflect the externally-repaired buffer, not internal _dirty_bits")

        # Re-inject the SAME seed (the old code would report 0 while re-corrupting the buffer).
        corr.inject_step(buf, "uniform_accum", seed=42)
        assert corr.cumulative_corruption()["bits_flipped"] == true_count() > 0, (
            "after re-injection the metric must match popcount(buf XOR clean), not desync to 0")


# ---------------------------------------------------------------------------
# E5 cliff tests
# ---------------------------------------------------------------------------

class TestE5Cliff:
    def setup_method(self):
        self.rmap = _rmap()
        self.cfg = {**E5_SMOKE_CFG, "R": 3, "chunk_size": 16}
        self.guard = RecoveryGuard(adapter, self.cfg, self.rmap)
        self.clean_buf = _clean()
        self.guard.init_from_clean(self.clean_buf)

    def test_majority_vote_repair(self):
        """One copy with 3 corrupted bits → majority-vote repairs them; cliff_repaired == 3."""
        bs, bl = self.guard._rot_region
        buf = self.clean_buf.copy()

        # Flip 3 bits in copy[0] only
        self.guard._rot_copies[0][0] ^= 0b00000111   # bits 0,1,2 of byte 0

        self.guard._scrub_cliff(buf)

        assert self.guard._cliff_repaired >= 3, (
            f"expected ≥3 bits repaired, got {self.guard._cliff_repaired}")
        assert self.guard._cliff_irrecoverable == 0
        # All copies should now match clean
        for r in range(3):
            assert np.array_equal(self.guard._rot_copies[r],
                                  np.frombuffer(bytes(self.clean_buf[bs:bs + bl]),
                                                dtype=np.uint8)), \
                f"copy[{r}] not repaired to clean"

    def test_irrecoverable_boundary(self):
        """≥ majority copies wrong at same bit → flagged as irrecoverable, NOT silently served."""
        bs, bl = self.guard._rot_region
        buf = self.clean_buf.copy()

        # Flip the same bit in copies[0] and copies[1] → 2 of 3 wrong (majority = wrong)
        self.guard._rot_copies[0][0] ^= 0b00000001
        self.guard._rot_copies[1][0] ^= 0b00000001
        # Clean CRC was stored in init_from_clean; majority will differ from it

        self.guard._scrub_cliff(buf)

        assert self.guard._cliff_irrecoverable >= 1, (
            "Should detect that majority-vote result fails clean CRC "
            "(2 of 3 copies corrupted the same way)")

    def test_irrecoverable_does_not_poison_buf(self):
        """Regression (#7): on irrecoverable, the wrong majority is NOT written into buf."""
        bs, bl = self.guard._rot_region
        buf = self.clean_buf.copy()
        # 2 of 3 copies wrong at the same bit -> majority is wrong -> irrecoverable.
        self.guard._rot_copies[0][0] ^= 0b00000001
        self.guard._rot_copies[1][0] ^= 0b00000001

        self.guard._scrub_cliff(buf)

        assert self.guard._cliff_irrecoverable >= 1
        # buf (which was clean) must be left untouched, NOT overwritten with the corrupt majority.
        assert np.array_equal(buf[bs:bs + bl], self.clean_buf[bs:bs + bl]), (
            "irrecoverable scrub must not serve the wrong majority into buf")
        assert self.guard._cliff_repaired == 0, "no repair should be counted when irrecoverable"

    def test_cliff_no_change_on_pristine_buf(self):
        """Clean buf + clean copies → no repairs, no irrecoverable, cliff_repaired == 0."""
        buf = self.clean_buf.copy()
        self.guard._scrub_cliff(buf)
        assert self.guard._cliff_repaired == 0
        assert self.guard._cliff_irrecoverable == 0

    def test_cliff_scrub_called_per_search(self):
        """search_with_recovery calls _scrub_cliff; checked count increments."""
        buf = self.clean_buf.copy()
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp = f.name
        try:
            self.guard.search_with_recovery(buf, tmp)
            assert self.guard._cliff_checked > 0, "cliff_checked must increment after search"
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class TestE5CliffWithE3c:
    def test_cliff_repair_prevents_accumulation(self):
        """E5 cliff repair keeps rotation bytes equal to clean across many E3c ticks."""
        rmap = _rmap()
        rot_region = _rot_region(rmap)
        bs, bl = rot_region
        guard_cfg = {**E5_SMOKE_CFG, "R": 3, "chunk_size": 16}
        guard = RecoveryGuard(adapter, guard_cfg, rmap)
        clean = _clean()
        guard.init_from_clean(clean)

        corruptor = TemporalCorruptor(rot_region, {**SMOKE_CFG, "p": 0.005})
        buf = clean.copy()

        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp = f.name
        try:
            for tick in range(10):
                corruptor.inject_step(buf, "uniform_accum", seed=tick)
                guard.search_with_recovery(buf, tmp)
                # After scrub, buf rotation bytes should equal clean
                assert np.array_equal(buf[bs:bs + bl], clean[bs:bs + bl]), (
                    f"tick={tick}: rotation not repaired to clean after scrub")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class TestE5CliffSelfScrub:
    """Stage-3 task #1: vote-failure-triggered full reload + low-frequency anchor backstop."""

    def setup_method(self):
        self.rmap = _rmap()
        self.cfg = {**E5_SMOKE_CFG, "R": 3, "chunk_size": 16,
                    "cliff_scrub": True, "anchor_every": 5}
        self.guard = RecoveryGuard(adapter, self.cfg, self.rmap)
        self.clean_buf = _clean()
        self.guard.init_from_clean(self.clean_buf)
        self.bs, self.bl = self.guard._rot_region

    def _clean_rot(self):
        return np.asarray(self.clean_buf[self.bs:self.bs + self.bl], dtype=np.uint8)

    def test_vote_failure_triggers_full_recovery(self):
        """Run-4 scenario (2 of 3 copies same bit) → reload → all clean → next vote normal."""
        buf = self.clean_buf.copy()
        self.guard._rot_copies[0][0] ^= 0b00000001
        self.guard._rot_copies[1][0] ^= 0b00000001

        self.guard._scrub_cliff(buf)

        assert self.guard._cliff_reload_triggered == 1
        assert self.guard._cliff_irrecoverable == 0, \
            "recovered event must not count as irrecoverable"
        assert np.array_equal(buf[self.bs:self.bs + self.bl], self._clean_rot())
        for r in range(3):
            assert np.array_equal(self.guard._rot_copies[r], self._clean_rot()), \
                f"copy[{r}] not reloaded to clean"

        # Voting must be functional again (not a blown fuse): a single-copy flip repairs.
        self.guard._rot_copies[0][3] ^= 0b00010000
        self.guard._scrub_cliff(buf)
        assert self.guard._cliff_irrecoverable == 0
        assert self.guard._cliff_reload_triggered == 1, "normal repair must not re-trigger reload"
        assert np.array_equal(self.guard._rot_copies[0], self._clean_rot())

    def test_full_reload_clears_unrelated_damage(self):
        """Full-region semantics: accumulated damage at OTHER positions is also cleared."""
        buf = self.clean_buf.copy()
        # The vote-failure double hit ...
        self.guard._rot_copies[0][0] ^= 0b00000001
        self.guard._rot_copies[1][0] ^= 0b00000001
        # ... plus unaligned accumulated damage elsewhere on every copy (run-4 tick 2 had
        # [1,3,5] bits spread across the replicas) and on buf itself.
        self.guard._rot_copies[0][5] ^= 0b00100000
        self.guard._rot_copies[1][9] ^= 0b00000110
        self.guard._rot_copies[2][2] ^= 0b10000001
        buf[self.bs + 7] ^= 0b00001000

        self.guard._scrub_cliff(buf)

        assert self.guard._cliff_reload_triggered == 1
        assert np.array_equal(buf[self.bs:self.bs + self.bl], self._clean_rot()), \
            "buf rotation must be fully clean, not just the failed bit"
        for r in range(3):
            assert np.array_equal(self.guard._rot_copies[r], self._clean_rot()), \
                f"copy[{r}] still carries damage at other positions after full reload"

    def test_anchor_catches_silent_wrong_vote(self):
        """Silent wrong majority (poisoned CRC anchor models a CRC collision / DRAM-hit
        anchor) → the periodic anchor check catches it from persistent storage and reloads."""
        buf = self.clean_buf.copy()
        # 2 of 3 copies same-position same-value: majority IS the wrong value.
        self.guard._rot_copies[0][0] ^= 0b00000001
        self.guard._rot_copies[1][0] ^= 0b00000001
        wrong = np.array(self._clean_rot(), dtype=np.uint8)
        wrong[0] ^= 0b00000001
        # Poison the in-memory CRC anchor so the vote verification passes silently.
        self.guard._rot_clean_crc = zlib.crc32(bytes(wrong))

        self.guard._scrub_cliff(buf)
        assert self.guard._cliff_reload_triggered == 0, "vote must succeed silently here"
        assert np.array_equal(buf[self.bs:self.bs + self.bl], wrong), \
            "precondition: wrong majority silently written into buf"

        # Not due yet -> nothing happens.
        self.guard.cliff_anchor_if_due(buf, tick=4)
        assert self.guard._cliff_anchor_checked == 0

        # Due -> caught from persistent storage, independent of the poisoned CRC.
        self.guard.cliff_anchor_if_due(buf, tick=5)
        assert self.guard._cliff_anchor_checked == 1
        assert self.guard._cliff_anchor_mismatch == 1
        assert self.guard._cliff_reload_triggered == 1
        assert np.array_equal(buf[self.bs:self.bs + self.bl], self._clean_rot())
        assert self.guard._rot_clean_crc == zlib.crc32(bytes(self._clean_rot())), \
            "reload must self-heal the poisoned in-memory CRC anchor"

    def test_anchor_switch_off_leaves_silent_error(self):
        """anchor_every=0 → the silent wrong vote is NOT caught (the switch controls it)."""
        cfg = {**self.cfg, "anchor_every": 0}
        guard = RecoveryGuard(adapter, cfg, self.rmap)
        guard.init_from_clean(self.clean_buf)
        buf = self.clean_buf.copy()
        guard._rot_copies[0][0] ^= 0b00000001
        guard._rot_copies[1][0] ^= 0b00000001
        wrong = np.array(self._clean_rot(), dtype=np.uint8)
        wrong[0] ^= 0b00000001
        guard._rot_clean_crc = zlib.crc32(bytes(wrong))

        guard._scrub_cliff(buf)
        for tick in range(1, 21):
            guard.cliff_anchor_if_due(buf, tick)
        assert guard._cliff_anchor_checked == 0
        assert guard._cliff_anchor_mismatch == 0
        assert np.array_equal(buf[self.bs:self.bs + self.bl], wrong), \
            "with the backstop off the wrong bytes must persist (validates the switch)"

    def test_cliff_scrub_off_keeps_legacy_fuse(self):
        """cliff_scrub=False → run-4 fuse semantics unchanged (irrecoverable, no reload)."""
        guard, clean = _guard(self.rmap, {**E5_SMOKE_CFG, "R": 3, "chunk_size": 16})
        buf = clean.copy()
        guard._rot_copies[0][0] ^= 0b00000001
        guard._rot_copies[1][0] ^= 0b00000001
        guard._scrub_cliff(buf)
        assert guard._cliff_irrecoverable == 1
        assert guard._cliff_reload_triggered == 0
        # anchor is inert without cliff_scrub even if anchor_every is set
        guard.cliff_anchor_if_due(buf, tick=10)
        assert guard._cliff_anchor_checked == 0

    def test_read_serialized_range_semantics(self):
        """Adapter clean-source reads match the serialized bytes and are independent copies."""
        full = adapter.serialize_index()
        a = adapter.read_serialized_range(self.bs, self.bl)
        assert np.array_equal(a, full[self.bs:self.bs + self.bl])
        a[0] ^= 0xFF
        b = adapter.read_serialized_range(self.bs, self.bl)
        assert np.array_equal(b, full[self.bs:self.bs + self.bl]), \
            "mutating a returned array must not affect subsequent reads (fresh copy per call)"


# ---------------------------------------------------------------------------
# E5 slope tests
# ---------------------------------------------------------------------------

class TestE5Slope:
    def setup_method(self):
        self.rmap = _rmap()
        self.cfg = {**E5_SMOKE_CFG, "chunk_size": 16, "reload_threshold": 0.5,
                    "parity_on": False}
        self.guard = RecoveryGuard(adapter, self.cfg, self.rmap)
        self.clean = _clean()
        self.guard.init_from_clean(self.clean)

    def _corrupt_ex_chunk(self, buf, chunk_idx=0):
        """XOR one byte in ex_code chunk `chunk_idx` of element 0."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        exs, exl = self.guard._ex_region
        sz = self.cfg["chunk_size"]
        byte_in_chunk = exs + chunk_idx * sz
        if byte_in_chunk < exs + exl:
            buf[byte_in_chunk] ^= 0xFF   # corrupt the whole byte

    def test_crc_detection(self):
        """Corrupting one ex chunk → slope_failed == 1, known_corrupted == 1."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        buf = self.clean.copy()
        self._corrupt_ex_chunk(buf, 0)
        self.guard._check_ex_crc(buf)
        ctr = self.guard.counters()
        assert ctr["slope_failed"] >= 1, f"expected slope_failed ≥ 1, got {ctr}"
        assert ctr["known_corrupted"] >= 1

    def test_eb_fallback_triggered(self):
        """CRC fail → adapter.search_with_eb_fallback called (_eb_path=True in result)."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        buf = self.clean.copy()
        self._corrupt_ex_chunk(buf, 0)
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp = f.name
        try:
            res = self.guard.search_with_recovery(buf, tmp)
            assert res.get("_eb_path") is True, (
                "EB-fallback path must be taken when ex chunk CRC fails")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_no_eb_fallback_on_clean(self):
        """Clean ex data → search_corrupted called (not EB path)."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        buf = self.clean.copy()
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp = f.name
        try:
            res = self.guard.search_with_recovery(buf, tmp)
            assert res.get("_eb_path") is not True, (
                "EB-fallback must NOT be taken on clean ex data")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_counter_tracks_injection_ratio(self):
        """Corrupt k chunks → known_corrupted == k, eb_fraction == k / total_chunks."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        buf = self.clean.copy()
        total = len(self.guard._clean_crcs)
        if total < 2:
            pytest.skip("not enough chunks for ratio test")
        k = min(2, total)
        for i in range(k):
            self._corrupt_ex_chunk(buf, i)
        self.guard._check_ex_crc(buf)
        ctr = self.guard.counters()
        assert ctr["known_corrupted"] == k, f"expected {k} corrupted chunks, got {ctr}"
        expected_frac = k / total
        assert abs(ctr["eb_fraction"] - expected_frac) < 1e-9

    def test_lazy_reload_resets_counter(self):
        """Corrupt enough chunks → scrub_if_due reloads and resets slope counters."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        buf = self.clean.copy()
        total = len(self.guard._clean_crcs)
        # Corrupt all chunks to exceed reload_threshold=0.5
        for i in range(total):
            exs, exl = self.guard._ex_region
            sz = self.cfg["chunk_size"]
            byte_in_chunk = exs + i * sz
            if byte_in_chunk < exs + exl:
                buf[byte_in_chunk] ^= 0xFF
        self.guard._check_ex_crc(buf)
        assert self.guard.counters()["known_corrupted"] > 0

        self.guard.scrub_if_due(buf, tick=1)

        ctr = self.guard.counters()
        assert ctr["known_corrupted"] == 0, "reload must clear failed_chunks"
        assert ctr["slope_reloaded"] > 0, "reload counter must increment"
        # Verify buf ex bytes match clean after reload
        exs, exl = self.guard._ex_region
        assert np.array_equal(buf[exs:exs + exl], self.clean[exs:exs + exl]), (
            "reload must restore ex bytes to clean")

    def test_ex_region_is_element0_only(self):
        """Regression (#5): the slope region is elem0.ex_code, NOT a ~99%-of-level0 span."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        expected = layout.element_field_range(self.rmap, "ex_code", 0)
        assert self.guard._ex_region == expected, (
            f"ex_region {self.guard._ex_region} must equal elem0 ex_code {expected}")
        level0 = next(r for r in self.rmap["regions"] if r["name"] == "level0")
        assert self.guard._ex_region[1] < level0["byte_len"] // 2, (
            "ex_region must be a small per-vector slice, not most of level0")

    def test_slope_failed_counts_distinct_events(self):
        """Regression (#8): a persistently-corrupted chunk counts once across repeated checks."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        buf = self.clean.copy()
        self._corrupt_ex_chunk(buf, 0)
        for _ in range(3):                       # same corruption checked three times
            self.guard._check_ex_crc(buf)
        ctr = self.guard.counters()
        assert ctr["slope_failed"] == 1, (
            f"persistent corruption must count as one failure event, got {ctr['slope_failed']}")
        assert ctr["known_corrupted"] == 1


class TestE5SlopeParity:
    def setup_method(self):
        self.rmap = _rmap()
        # Use tiny chunks (≤64 B) for parity repair path
        self.cfg = {**E5_SMOKE_CFG, "chunk_size": 16, "reload_threshold": 0.5,
                    "parity_on": True}
        self.guard = RecoveryGuard(adapter, self.cfg, self.rmap)
        self.clean = _clean()
        self.guard.init_from_clean(self.clean)

    def test_parity_on_single_bit_repair(self):
        """1-byte flip in a small (<=64 B) chunk: parity repair succeeds and restores the byte."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        if not self.guard._parity_bytes:
            pytest.skip("parity not initialized (no ex region)")
        exs, exl = self.guard._ex_region
        if exl < 1:
            pytest.skip("ex region too small")
        buf = self.clean.copy()
        # Flip exactly 1 bit in chunk 0. Clean stub chunk is all-zero, so actual_p=0x08 != 0,
        # diff_p=0x08, and the brute-force single-byte correction deterministically restores it.
        buf[exs] ^= 0x08
        self.guard._check_ex_crc(buf)
        ctr = self.guard.counters()
        assert ctr["slope_failed"] == 0, (
            "deterministic single-bit flip in a small chunk must be parity-repaired, not escalated")
        assert ctr["known_corrupted"] == 0, "parity-repaired chunk must not stay in known_corrupted"
        assert buf[exs] == self.clean[exs], "parity repair must restore the byte to clean"

    def test_parity_on_multi_bit_fallback(self):
        """Multi-byte corruption: parity cannot repair → slope_failed >= 1."""
        if self.guard._ex_region is None:
            pytest.skip("stub has no ex region")
        if not self.guard._parity_bytes:
            pytest.skip("parity not initialized")
        exs, exl = self.guard._ex_region
        if exl < 4:
            pytest.skip("ex region too small for multi-byte test")
        buf = self.clean.copy()
        # Flip 4 separate bytes in chunk 0 → parity can't repair all of them
        for i in range(4):
            if exs + i < exs + exl:
                buf[exs + i] ^= (1 << i)
        self.guard._check_ex_crc(buf)
        ctr = self.guard.counters()
        assert ctr["slope_failed"] >= 1, (
            "multi-byte corruption beyond parity capacity should trigger EB-fallback")


# ---------------------------------------------------------------------------
# E5 bounds-check test
# ---------------------------------------------------------------------------

class TestE5BoundsCheck:
    def test_oob_pointer_detected_and_no_crash(self):
        """OOB link id (even with count==0) → detected, field restored from clean, no crash."""
        rmap = _rmap()
        cfg = {**E5_SMOKE_CFG, "chunk_size": 16, "bounds_sample": 4}
        guard = RecoveryGuard(adapter, cfg, rmap)
        clean = _clean()
        guard.init_from_clean(clean)

        buf = clean.copy()
        try:
            bs_links, bl_links = layout.element_field_range(rmap, "links", 0)
        except KeyError:
            pytest.skip("no links region in stub rmap")
        if bl_links < 8:
            pytest.skip("links region too small")

        # Write an OOB link id (high bytes of the first slot) while leaving count==0 in the
        # pristine stub. The fixed-window scan must still find it, and restore-from-clean must
        # prevent the stub's high-byte-pointer crash.
        buf[bs_links + 6] = 0xFF
        buf[bs_links + 7] = 0xFF

        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp = f.name
        try:
            res = guard.search_with_recovery(buf, tmp)   # must NOT raise now
            assert guard._oob_elements >= 1, "OOB link id must be detected despite count==0"
            assert res.get("ids") is not None, "search must return after OOB restore (no crash)"
            # The corrupted links field must have been restored to clean.
            assert np.array_equal(buf[bs_links:bs_links + bl_links],
                                  clean[bs_links:bs_links + bl_links]), \
                "OOB field must be restored from the clean snapshot"
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Integration: §1 mini-timeline contract
# ---------------------------------------------------------------------------

class TestMiniTimeline:
    def test_e3c_e5_joint_loop(self):
        """§1 contract: E3c + E5 compose in a 5-tick loop without errors; schema correct."""
        rmap = _rmap()
        rot_region = _rot_region(rmap)
        bs, bl = rot_region
        gt = adapter.load_groundtruth()

        guard_cfg = {**E5_SMOKE_CFG, "R": 3, "chunk_size": 16, "reload_threshold": 0.5}
        guard = RecoveryGuard(adapter, guard_cfg, rmap)
        clean = _clean()
        guard.init_from_clean(clean)

        corruptor = TemporalCorruptor(rot_region, {**SMOKE_CFG, "p": 0.002})
        buf = clean.copy()

        REQUIRED_COUNTER_KEYS = {
            "cliff_checked", "cliff_repaired", "cliff_irrecoverable",
            "cliff_reload_triggered", "cliff_anchor_checked", "cliff_anchor_mismatch",
            "slope_checked", "slope_failed", "slope_reloaded",
            "known_corrupted", "eb_fraction", "oob_elements",
        }

        records = []
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            tmp = f.name
        try:
            for tick in range(5):
                # E3c: inject one step of cumulative corruption
                corruptor.inject_step(buf, "uniform_accum", seed=tick * 13)

                # E5: search with recovery
                res = guard.search_with_recovery(buf, tmp)
                ids = res.get("ids")
                if ids is not None:
                    recall = metrics.recall_at_k(ids, gt, config.K)
                else:
                    recall = None

                # Measure
                cc = corruptor.cumulative_corruption()
                ctr = guard.counters()

                # Record
                records.append({
                    "tick": tick,
                    "cumulative_corruption": cc,
                    "recall@10": recall,
                    "counters": ctr,
                })

                # Scrub if due
                guard.scrub_if_due(buf, tick)

        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        assert len(records) == 5, "must complete all 5 ticks"

        for row in records:
            assert "tick" in row
            assert "cumulative_corruption" in row
            assert "recall@10" in row
            assert "counters" in row
            missing = REQUIRED_COUNTER_KEYS - set(row["counters"].keys())
            assert not missing, f"counter schema missing keys: {missing}"

        final_cc = records[-1]["cumulative_corruption"]
        assert final_cc["tick"] == 5, f"expected tick=5 at end, got {final_cc}"
        assert final_cc["bits_flipped"] >= 0
        assert 0.0 <= final_cc["fraction"] <= 1.0

    def test_interface_contract_types(self):
        """E3c.cumulative_corruption() and E5.counters() return correct Python types."""
        rmap = _rmap()
        rot_region = _rot_region(rmap)
        corruptor = TemporalCorruptor(rot_region, SMOKE_CFG)
        buf = _clean()
        corruptor.inject_step(buf, "uniform_accum", seed=0)
        cc = corruptor.cumulative_corruption()
        assert isinstance(cc["bits_flipped"], int)
        assert isinstance(cc["fraction"], float)
        assert isinstance(cc["tick"], int)

        guard = RecoveryGuard(adapter, E5_SMOKE_CFG, rmap)
        guard.init_from_clean(_clean())
        ctr = guard.counters()
        for key in ("cliff_checked", "cliff_repaired", "cliff_irrecoverable",
                    "slope_checked", "slope_failed", "slope_reloaded",
                    "known_corrupted", "oob_elements"):
            assert isinstance(ctr[key], int), f"counter[{key!r}] must be int"
        assert isinstance(ctr["eb_fraction"], float), "eb_fraction must be float"

    def test_config_stub_vs_real_switch(self):
        """get_adapter('stub') returns the stub module; no code change between adapter types."""
        from qp.rabitq.registry import get_adapter, adapter_name
        stub = get_adapter("stub")
        assert adapter_name(stub) == "stub"
        # Create a guard with the stub adapter — interface identical to real
        rmap = stub.region_map()
        guard = RecoveryGuard(stub, E5_SMOKE_CFG, rmap)
        guard.init_from_clean(stub.serialize_index())
        # counters schema must be well-formed regardless of adapter
        ctr = guard.counters()
        assert "cliff_checked" in ctr
