"""Stage 3 tests: seed-batch aggregation (#2), F synthesis (#3), G detection (#4).

Pure-function tests on synthetic inputs — no real adapter, no subprocesses.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase3_e5_seed_batch import (first_fail_tick, summarize_first_fails,
                                  analytic_estimate, records_identical,
                                  batch_record_files, STEM_RE)


def _rows(irrecoverable_by_tick):
    """Synthetic e5 records: cumulative cliff_irrecoverable per tick."""
    return [{"tick": t, "counters": {"cliff_irrecoverable": c}}
            for t, c in enumerate(irrecoverable_by_tick)]


class TestFirstFail:
    def test_first_fail_found(self):
        # run-4 shape: fails at tick 2, cumulative counter rises thereafter
        assert first_fail_tick(_rows([0, 0, 1, 2, 3])) == 2

    def test_first_fail_at_tick_zero(self):
        assert first_fail_tick(_rows([1, 2])) == 0

    def test_censored_returns_none(self):
        assert first_fail_tick(_rows([0, 0, 0, 0])) is None

    def test_empty_rows(self):
        assert first_fail_tick([]) is None


class TestSummarize:
    def test_stats_with_censoring(self):
        ff = {1000: 2, 1001: 10, 1002: 6, 1003: None}
        s = summarize_first_fails(ff, ticks=150)
        assert s["n_seeds"] == 4
        assert s["n_observed"] == 3
        assert s["n_censored"] == 1
        assert s["censored_seeds"] == [1003]
        assert s["min"] == 2 and s["max"] == 10
        assert s["median"] == 6
        assert s["q1"] == 4 and s["q3"] == 8 and s["iqr"] == 4

    def test_all_censored(self):
        s = summarize_first_fails({1: None, 2: None}, ticks=30)
        assert s["n_observed"] == 0 and s["n_censored"] == 2
        assert "median" not in s


class TestAnalytic:
    def test_matches_section5_note(self):
        """§5 統計注: E[b]=2.56 dirty bits/replica, P(fail/tick)~3.8%, first fail O(10)."""
        a = analytic_estimate(p=0.005, bits=512, R=3)
        assert a["expected_dirty_bits_per_replica"] == pytest.approx(2.56)
        assert a["p_fail_per_tick"] == pytest.approx(3 * 2.56**2 / 512, rel=1e-9)
        assert 0.03 < a["p_fail_per_tick"] < 0.05
        assert 10 < a["geometric_median"] < 25       # O(10) ticks
        assert a["geometric_mean"] == pytest.approx(1 / a["p_fail_per_tick"])


class TestDeterminismCompare:
    def test_identical_and_differing(self, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        c = tmp_path / "c.jsonl"
        a.write_text('{"tick": 0}\n{"tick": 1}\n')
        b.write_text('{"tick": 0}\n{"tick": 1}\n')
        c.write_text('{"tick": 0}\n{"tick": 2}\n')
        assert records_identical(str(a), str(b))
        assert not records_identical(str(a), str(c))


class TestBatchFileDiscovery:
    def test_strict_stem_excludes_r2_and_full_tags(self, tmp_path):
        names = [
            "e5_uniform_accum_rotation_replicas_seed1000.records.jsonl",   # batch
            "e5_uniform_accum_rotation_replicas_seed1001.records.jsonl",   # batch
            "e5_uniform_accum_rotation_replicas_seed1000r2.records.jsonl",  # determinism rerun
            "e5_uniform_accum_rotation_replicas_seed1001full.records.jsonl",  # timeline
            "e5_uniform_accum_rotation_replicas.records.jsonl",            # run-4 style, no tag
        ]
        for n in names:
            (tmp_path / n).write_text("")
        found = batch_record_files(str(tmp_path))
        assert sorted(found) == [1000, 1001]
        assert STEM_RE.search(names[2]) is None
        assert STEM_RE.search(names[3]) is None
