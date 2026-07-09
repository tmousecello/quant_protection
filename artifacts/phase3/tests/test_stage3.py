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
                                  batch_record_files, STEM_RE, timeline_ticks)
from phase3_f_synth import (align_timelines, tolerance_curves, interp_recall,
                            solve_fstar, interval_ratio, fstar_table,
                            average_measured_at_fraction, run_check_gates,
                            _light_expb_records, RMIN_GRID)

_EXPB_DIR = os.path.join(_ROOT, "artifacts", "phase3", "expb")
_HAVE_EXPB = os.path.isfile(os.path.join(_EXPB_DIR, "expb_summary.json"))


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


class TestTimelineTicks:
    def test_extends_past_late_first_fail(self):
        # a seed failing at tick 118 (measured over 150) needs > 100 ticks to show collapse
        assert timeline_ticks(118, base=100) == 168
        assert timeline_ticks(118, base=100, post_fail=30) == 148

    def test_keeps_base_for_early_fail(self):
        assert timeline_ticks(20, base=100) == 100

    def test_censored_pick_keeps_base(self):
        assert timeline_ticks(None, base=100) == 100


class TestAnalytic:
    def test_matches_section5_note(self):
        """§5 統計注: E[b]=2.56 dirty bits/replica, P(fail/tick)~3.8%, first fail O(10)."""
        a = analytic_estimate(p=0.005, bits=512, R=3)
        assert a["expected_dirty_bits_per_replica"] == pytest.approx(2.56)
        assert a["p_fail_per_tick"] == pytest.approx(3 * 2.56**2 / 512, rel=1e-9)
        assert 0.03 < a["p_fail_per_tick"] < 0.05
        assert 10 < a["geometric_median"] < 25       # O(10) ticks
        # pin geometric_mean to an INDEPENDENT number, not 1/p_fail (which is tautological)
        assert a["geometric_mean"] == pytest.approx(26.042, abs=1e-2)

    def test_uses_pairwise_combination_not_R(self):
        """The §5 model is C(R,2) birthday-pairwise, not R. At R=3 comb(3,2)==3 hides the
        difference, so pin it at R=4 where comb(4,2)==6 != 4."""
        a = analytic_estimate(p=0.005, bits=512, R=4)
        eb = 512 * 0.005
        assert a["p_fail_per_tick"] == pytest.approx(6 * eb**2 / 512, rel=1e-9)
        assert a["p_fail_per_tick"] != pytest.approx(4 * eb**2 / 512, rel=1e-9)


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


# ---------------------------------------------------------------------------
# F synthesis (#3)
# ---------------------------------------------------------------------------

def _expb_row(pattern, fraction, recovery, recall):
    return {"pattern": pattern, "fraction": fraction, "recovery": recovery,
            "recall@10": recall}


class TestInterpolation:
    CURVE = [(0.0, 1.0), (0.1, 0.9), (0.3, 0.5)]

    def test_interp_midpoint(self):
        assert interp_recall(self.CURVE, 0.05) == pytest.approx(0.95)
        assert interp_recall(self.CURVE, 0.2) == pytest.approx(0.7)

    def test_interp_out_of_range_raises(self):
        with pytest.raises(ValueError):
            interp_recall(self.CURVE, 0.4)

    def test_solve_fstar_linear_inverse(self):
        assert solve_fstar(self.CURVE, 0.95) == pytest.approx(0.05)
        assert solve_fstar(self.CURVE, 0.7) == pytest.approx(0.2)

    def test_solve_fstar_below_range_returns_none(self):
        assert solve_fstar(self.CURVE, 0.4) is None

    def test_solve_fstar_nonmonotonic_returns_largest_f(self):
        # recall dips below 0.9 then recovers above it before finally dropping: the largest
        # f with recall>=0.9 is on the LAST crossing (~0.20), not the first (~0.067).
        curve = [(0.0, 1.0), (0.1, 0.85), (0.15, 0.92), (0.3, 0.5)]
        f = solve_fstar(curve, 0.9)
        assert f == pytest.approx(0.15 + (0.92 - 0.9) * (0.3 - 0.15) / (0.92 - 0.5))
        assert f > 0.15   # past the recovery, not stuck at the early dip

    def test_interval_ratio(self):
        # pin an INDEPENDENT numeric (the §7.2 headline ~1.09), not the code's own expression
        assert interval_ratio(0.101, 0.093) == pytest.approx(1.09076, abs=1e-4)
        assert interval_ratio(0.1, 0.1) == pytest.approx(1.0)


class TestToleranceCurves:
    def test_pattern_average_and_clean_anchor(self):
        rows = [
            _expb_row("uniform_accum", 0.05, "drop", 0.94),
            _expb_row("burst_accum", 0.05, "drop", 0.90),
            _expb_row("uniform_accum", 0.20, "drop", 0.60),
            _expb_row("burst_accum", 0.20, "drop", 0.80),
        ]
        curves = tolerance_curves(rows, clean_recall=0.98)
        assert curves["drop"] == [(0.0, 0.98), (0.05, pytest.approx(0.92)),
                                  (0.20, pytest.approx(0.70))]

    def test_crash_row_reports_and_stops(self):
        rows = [_expb_row("uniform_accum", 0.05, "drop", 0.94),
                _expb_row("burst_accum", 0.05, "drop", None)]   # crashed cell
        with pytest.raises(RuntimeError, match="crash/timeout"):
            tolerance_curves(rows, clean_recall=0.98)
        with pytest.raises(RuntimeError, match="burst_accum"):
            average_measured_at_fraction(rows, 0.05)

    def test_measured_average_at_fraction(self):
        rows = [
            _expb_row("uniform_accum", 0.101, "fallback_eb", 0.90),
            _expb_row("burst_accum", 0.101, "fallback_eb", 0.92),
            _expb_row("uniform_accum", 0.101, "drop", 0.88),
            _expb_row("uniform_accum", 0.05, "fallback_eb", 0.95),   # other f: excluded
        ]
        avg = average_measured_at_fraction(rows, 0.101)
        assert avg["fallback_eb"] == pytest.approx(0.91)
        assert avg["drop"] == pytest.approx(0.88)


class TestAlignTimelines:
    def test_shorter_series_padded_with_none(self):
        series = {"a": [(0, 1.0), (1, 0.9)], "b": [(0, 0.5)]}
        rows = align_timelines(series)
        assert rows == [{"tick": 0, "a": 1.0, "b": 0.5},
                        {"tick": 1, "a": 0.9, "b": None}]


class TestCheckGates:
    PRED = {"headline_fraction": 0.101,
            "pred_eb_at_headline": 0.8997, "pred_drop_at_headline": 0.8919}
    CLIFF_OK = {"deterministic_6bit": {"collapse": True, "delta": 0.002},
                "k8": {"collapse": True}}

    def test_all_pass(self):
        ok, gates = run_check_gates({"fallback_eb": 0.902, "drop": 0.885},
                                    self.PRED, self.CLIFF_OK)
        assert ok and all(g["ok"] for g in gates)

    def test_gate_a_fails_beyond_tolerance(self):
        ok, gates = run_check_gates({"fallback_eb": 0.92, "drop": 0.885},
                                    self.PRED, self.CLIFF_OK)
        assert not ok
        assert not next(g for g in gates if g["gate"] == "a_headline_eb")["ok"]

    def test_gate_b_requires_direction(self):
        # drop within tolerance of its prediction but NOT below eb -> direction fails
        ok, gates = run_check_gates({"fallback_eb": 0.890, "drop": 0.891},
                                    self.PRED, self.CLIFF_OK)
        assert not next(g for g in gates if g["gate"] == "b_drop_control")["ok"]

    def test_gate_c_needs_collapse_and_delta(self):
        bad = {"deterministic_6bit": {"collapse": True, "delta": 0.05},
               "k8": {"collapse": True}}
        ok, gates = run_check_gates({"fallback_eb": 0.902, "drop": 0.885},
                                    self.PRED, bad)
        assert not next(g for g in gates if g["gate"] == "c_cliff")["ok"]

    def test_gate_c_fails_when_delta_missing(self):
        # delta=None (run-2 reference absent) must FAIL gate c, not pass vacuously
        no_ref = {"deterministic_6bit": {"collapse": True, "delta": None},
                  "k8": {"collapse": True}}
        ok, gates = run_check_gates({"fallback_eb": 0.902, "drop": 0.885},
                                    self.PRED, no_ref)
        assert not next(g for g in gates if g["gate"] == "c_cliff")["ok"]


@pytest.mark.skipif(not _HAVE_EXPB, reason="expb artifacts only on the workstation")
class TestSection72Reproduction:
    """The pinned interpolation must reproduce E3C_E5_ANALYSIS.md §7.2 to table precision."""

    EXPECTED = {  # r_min: (fstar_drop, fstar_eb, ratio) as printed in the doc
        0.97: (0.016, 0.017, 1.06), 0.95: (0.039, 0.041, 1.07),
        0.90: (0.093, 0.101, 1.09), 0.85: (0.145, 0.159, 1.11),
        0.80: (0.197, 0.219, 1.13), 0.70: (0.298, 0.342, 1.18),
        0.60: (0.400, 0.465, 1.23),
    }

    def test_table_reproduced(self):
        with open(os.path.join(_EXPB_DIR, "expb_summary.json")) as fh:
            clean = json.load(fh)["clean_recall@10"]
        rows = _light_expb_records(_EXPB_DIR)
        curves = tolerance_curves(rows, clean)
        for r in fstar_table(curves, RMIN_GRID):
            exp_drop, exp_eb, exp_ratio = self.EXPECTED[r["r_min"]]
            assert r["fstar_drop"] == pytest.approx(exp_drop, abs=5e-4), r
            assert r["fstar_eb"] == pytest.approx(exp_eb, abs=5e-4), r
            assert r["interval_ratio"] == pytest.approx(exp_ratio, abs=5e-3), r


# ---------------------------------------------------------------------------
# G detection (#4)
# ---------------------------------------------------------------------------

from phase3_g_detection import extract_points, deviation_stats


def _g_row(pattern, fraction, actual, recovery, checked, crc_fail, consults, hits,
           n_elements=None):
    row = {"pattern": pattern, "fraction": fraction, "fraction_actual": actual,
           "recovery": recovery,
           "stats": {"load": {"elements_checked": checked, "elements_crc_fail": crc_fail},
                     "totals": {"consults": consults, "corrupt_hits": hits}}}
    if n_elements is not None:
        row["n_elements"] = n_elements
    return row


class TestGDetection:
    def test_extract_excludes_none_and_computes_ratios(self):
        rows = [
            _g_row("uniform_accum", 0.05, 0.05, "drop", 1000, 50, 2000, 100),
            _g_row("uniform_accum", 0.05, 0.05, "fallback_eb", 1000, 50, 400, 20),
            # recovery=none: no load scan ran -> excluded
            {"pattern": "uniform_accum", "fraction": 0.05, "fraction_actual": 0.05,
             "recovery": "none", "stats": {"load": {"elements_checked": 0,
                                                    "elements_crc_fail": 0}}},
        ]
        points, excluded = extract_points(rows, "light")
        assert excluded == 1
        assert len(points) == 2
        assert points[0]["observed_load"] == pytest.approx(0.05)
        assert points[0]["observed_access"] == pytest.approx(0.05)
        assert points[1]["observed_access"] == pytest.approx(0.05)
        assert points[0]["severity"] == "light"

    def test_denominator_invariant_ok(self):
        # checked * fraction_actual == n_elements  (elements_checked == cur_element_count)
        pts, _ = extract_points(
            [_g_row("uniform_accum", 0.05, 0.05, "drop", 1000, 50, 2000, 100,
                    n_elements=50)], "light")
        assert len(pts) == 1 and pts[0]["observed_load"] == pytest.approx(0.05)

    def test_denominator_invariant_violation_reports_and_stops(self):
        # checked=900 but n_elements=50 at f_actual=0.05 implies cur_element_count=1000,
        # i.e. elements_checked (900) != cur_element_count -> the diagonal claim is unsafe.
        with pytest.raises(RuntimeError, match="cur_element_count"):
            extract_points(
                [_g_row("uniform_accum", 0.05, 0.05, "drop", 900, 45, 2000, 90,
                        n_elements=50)], "light")

    def test_detecting_mode_zero_checked_excluded_not_divzero(self):
        # a drop/fallback_eb row with elements_checked == 0 must be excluded, not raise
        pts, excl = extract_points(
            [_g_row("uniform_accum", 0.05, 0.05, "drop", 0, 0, 2000, 0)], "light")
        assert pts == [] and excl == 1

    def test_zero_consults_gives_none_access_ratio(self):
        points, _ = extract_points(
            [_g_row("burst_accum", 0.2, 0.2, "drop", 100, 20, 0, 0)], "sev384")
        assert points[0]["observed_access"] is None
        assert points[0]["observed_load"] == pytest.approx(0.2)

    def test_deviation_stats_exact_diagonal(self):
        points, _ = extract_points(
            [_g_row("uniform_accum", 0.05, 0.0625, "drop", 64, 4, 10, 1)], "light")
        d = deviation_stats(points)
        assert d["n_points"] == 1
        assert d["max_abs_dev_vs_truth"] == 0.0
        assert d["identical_to_truth"] is True
        # nominal fraction differs from actual (rounding) -> nonzero vs nominal
        assert d["max_abs_dev_vs_nominal"] == pytest.approx(0.0125)

    def test_empty_points(self):
        assert deviation_stats([]) == {"n_points": 0}
