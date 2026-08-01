"""E6 (shape x region damage matrix) unit tests — the pure, seed-deterministic pieces.

Everything here runs offline against the STUB geometry (arm64-safe, faiss-free). Coverage:
  - `stratum_windows` agrees with qp.rabitq.layout arithmetic for every stratum (global and
    per-element), including the spec's "`ids` = two 4-byte windows per element" case;
  - `sample_anchor` is pure/deterministic in seed and byte-weighted across a stratum's windows
    (8192:64 for rotation_centroids, 12:8 for factors, 4:4 for ids);
  - `field_at_byte` / `count_field_hits` resolve absolute bytes back to the field they belong
    to — the smear accounting that makes "a row/column physically crosses interleaved fields"
    a measurement rather than a claim;
  - `classify_outcome`'s 6-case truth table;
  - `inject_shape` honours the sampled anchor and round-trips through `faults.restore`;
  - the driver's sanity gate + a full --smoke stub run (row count, CSV schema, no state leak).
"""

import csv
import json
import os
import sys
import types

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qp import faults
from qp.rabitq import layout
from qp.rabitq import stub_adapter

import phase3_e6_shapes as e6


@pytest.fixture(scope="module")
def rmap():
    return stub_adapter.region_map()


@pytest.fixture(scope="module")
def clean_buf():
    return stub_adapter.serialize_index()


# ---------------------------------------------------------------------------
# stratum_windows — must equal layout arithmetic, not a re-derivation
# ---------------------------------------------------------------------------

def test_stub_geometry_is_what_these_tests_assume(rmap):
    """Guard rail: if the stub geometry drifts, the concrete numbers below must be revisited."""
    hdr = rmap["header"]
    assert (hdr["padded_dim"], hdr["ex_bits"], hdr["cur_element_count"]) == (128, 6, 64)
    assert hdr["size_data_per_element"] == 272
    assert layout.element_field_range(rmap, "ex_code", 0)[1] == 96
    assert layout.element_field_range(rmap, "bin_code", 0)[1] == 16
    assert layout.element_field_range(rmap, "links", 0)[1] == 132


def test_rotation_centroids_is_the_union_of_the_two_global_windows(rmap):
    wins = e6.stratum_windows(rmap, "rotation_centroids")
    centroids = next(r for r in rmap["regions"] if r["name"] == "centroids")
    rotation = next(r for r in rmap["regions"] if r["name"] == "rotation")
    assert wins == [(centroids["byte_start"], centroids["byte_len"]),
                    (rotation["byte_start"], rotation["byte_len"])]
    assert [L for _, L in wins] == [8192, 64]          # the 8192:64 byte weighting of the spec


def test_ids_stratum_is_two_4_byte_windows_per_element(rmap):
    for e in (0, 1, 37, 63):
        wins = e6.stratum_windows(rmap, "ids", element=e)
        assert wins == [layout.element_field_range(rmap, "cluster_id", e),
                        layout.element_field_range(rmap, "label", e)]
        assert [L for _, L in wins] == [4, 4]


def test_factors_stratum_is_bin_then_ex_byte_weighted_12_to_8(rmap):
    wins = e6.stratum_windows(rmap, "factors", element=5)
    assert wins == [layout.element_field_range(rmap, "bin_factors", 5),
                    layout.element_field_range(rmap, "ex_factors", 5)]
    assert [L for _, L in wins] == [12, 8]


@pytest.mark.parametrize("stratum,field,length", [("ex_code", "ex_code", 96),
                                                  ("bin_code", "bin_code", 16),
                                                  ("links", "links", 132)])
def test_single_field_strata_match_element_field_range(rmap, stratum, field, length):
    for e in (0, 11, 63):
        assert e6.stratum_windows(rmap, stratum, element=e) == \
            [layout.element_field_range(rmap, field, e)]
        assert e6.stratum_windows(rmap, stratum, element=e)[0][1] == length


def test_element_stride_is_honoured(rmap):
    spe = rmap["header"]["size_data_per_element"]
    (s0, _), = e6.stratum_windows(rmap, "ex_code", element=0)
    (s7, _), = e6.stratum_windows(rmap, "ex_code", element=7)
    assert s7 - s0 == 7 * spe


def test_per_element_stratum_without_element_is_a_loud_error(rmap):
    with pytest.raises(ValueError, match="element"):
        e6.stratum_windows(rmap, "ex_code")


def test_unknown_stratum_is_a_loud_error(rmap):
    with pytest.raises(ValueError, match="stratum"):
        e6.stratum_windows(rmap, "not_a_stratum", element=0)


def test_every_declared_stratum_is_resolvable(rmap):
    for stratum in e6.STRATA:
        element = None if stratum in e6.GLOBAL_STRATA else 3
        wins = e6.stratum_windows(rmap, stratum, element=element)
        assert wins and all(L > 0 for _, L in wins)


# ---------------------------------------------------------------------------
# sample_anchor — pure, deterministic, byte-weighted
# ---------------------------------------------------------------------------

def test_sample_anchor_is_deterministic_in_seed(rmap):
    for stratum in e6.STRATA:
        a = e6.sample_anchor(rmap, stratum, 4242)
        b = e6.sample_anchor(rmap, stratum, 4242)
        assert a == b
    # ...and actually depends on the seed (not a constant)
    anchors = {e6.sample_anchor(rmap, "ex_code", s)["anchor_byte"] for s in range(64)}
    assert len(anchors) > 1


def test_sample_anchor_lands_inside_its_stratum(rmap):
    for stratum in e6.STRATA:
        for seed in range(200):
            a = e6.sample_anchor(rmap, stratum, seed)
            wins = e6.stratum_windows(rmap, stratum, element=a["element"])
            assert a["window"] in wins
            assert wins[a["window_index"]] == a["window"]
            start, length = a["window"]
            assert start <= a["anchor_byte"] < start + length


def test_sample_anchor_element_is_none_exactly_for_global_strata(rmap):
    for stratum in e6.STRATA:
        a = e6.sample_anchor(rmap, stratum, 7)
        if stratum in e6.GLOBAL_STRATA:
            assert a["element"] is None
        else:
            assert 0 <= a["element"] < rmap["header"]["cur_element_count"]


def test_sample_anchor_is_byte_weighted_across_windows(rmap):
    """rotation_centroids 8192:64, factors 12:8, ids 4:4 — deterministic seeds, so not flaky."""
    n = 2000
    rot_hits = sum(e6.sample_anchor(rmap, "rotation_centroids", s)["window_index"]
                   for s in range(n))
    assert 3 <= rot_hits <= 40, f"rotation share {rot_hits}/{n}, expected ~{n * 64 / 8256:.0f}"

    ex_hits = sum(e6.sample_anchor(rmap, "factors", s)["window_index"] for s in range(n))
    assert 0.35 < ex_hits / n < 0.45, f"ex_factors share {ex_hits / n:.3f}, expected ~0.40"

    label_hits = sum(e6.sample_anchor(rmap, "ids", s)["window_index"] for s in range(n))
    assert 0.45 < label_hits / n < 0.55, f"label share {label_hits / n:.3f}, expected ~0.50"


def test_sample_anchor_spreads_over_elements(rmap):
    elements = {e6.sample_anchor(rmap, "bin_code", s)["element"] for s in range(400)}
    assert len(elements) > 32          # 64 elements available; a constant would give 1


def test_cell_seed_is_deterministic_and_well_separated():
    assert e6.cell_seed(1234, "device_row", "ids", 3) == e6.cell_seed(1234, "device_row", "ids", 3)
    seeds = {e6.cell_seed(1234, sh, st, i)
             for sh in e6.SHAPES for st in e6.STRATA for i in range(30)}
    assert len(seeds) == len(e6.SHAPES) * len(e6.STRATA) * 30      # no collisions on the grid
    # adjacent roots must not alias (the `base ^ lane` bug expb caught)
    assert e6.cell_seed(1234, "single_cell", "ids", 1) != e6.cell_seed(1235, "single_cell", "ids", 0)


# ---------------------------------------------------------------------------
# smear accounting: absolute byte -> field
# ---------------------------------------------------------------------------

def test_field_at_byte_resolves_each_structure(rmap):
    resolve = e6.make_field_resolver(rmap)
    assert resolve(0) == "header"
    assert resolve(layout.HEADER_BYTES) == "centroids"
    rotation = next(r for r in rmap["regions"] if r["name"] == "rotation")
    assert resolve(rotation["byte_start"]) == "rotation"
    for field in ("links", "cluster_id", "label", "bin_code", "bin_factors",
                  "ex_code", "ex_factors"):
        start, length = layout.element_field_range(rmap, field, 9)
        assert resolve(start) == field
        assert resolve(start + length - 1) == field


def test_count_field_hits_counts_distinct_bytes(rmap):
    s_ex, _ = layout.element_field_range(rmap, "ex_code", 2)
    s_bin, _ = layout.element_field_range(rmap, "bin_code", 2)
    positions = [(s_ex, 0), (s_ex, 3), (s_ex + 1, 0), (s_bin, 7)]
    assert e6.count_field_hits(rmap, positions) == {"ex_code": 2, "bin_code": 1}


# ---------------------------------------------------------------------------
# classify_outcome — 6-case truth table
# ---------------------------------------------------------------------------

CLEAN = 0.98


@pytest.mark.parametrize("name,kw,expected", [
    ("crash",
     dict(arm="off", recall=None, elements_crc_fail=None, cliff_repaired=None,
          oob_restored=None, crashed=True), "crash"),
    ("repaired: cliff put it back, nothing left for CRC to flag",
     dict(arm="on", recall=0.98, elements_crc_fail=0, cliff_repaired=17, oob_restored=0,
          crashed=False), "repaired"),
    ("repaired via bounds-check restore",
     dict(arm="on", recall=0.977, elements_crc_fail=0, cliff_repaired=0, oob_restored=2,
          crashed=False), "repaired"),
    ("tolerated: EB carried it (recall held WITH crc failures)",
     dict(arm="on", recall=0.978, elements_crc_fail=5, cliff_repaired=0, oob_restored=0,
          crashed=False), "tolerated"),
    ("tolerated: damage simply too small to matter (arm off)",
     dict(arm="off", recall=0.9755, elements_crc_fail=None, cliff_repaired=None,
          oob_restored=None, crashed=False), "tolerated"),
    ("silent_wrong: recall gone, nothing detected",
     dict(arm="off", recall=0.42, elements_crc_fail=None, cliff_repaired=None,
          oob_restored=None, crashed=False), "silent_wrong"),
])
def test_classify_outcome_truth_table(name, kw, expected):
    assert e6.classify_outcome(clean_recall=CLEAN, **kw) == expected, name


def test_repaired_requires_a_recovery_counter(rmap):
    """arm on, recall fine, no CRC failures, but NO layer acted -> not 'repaired'."""
    assert e6.classify_outcome(arm="on", recall=0.98, clean_recall=CLEAN, elements_crc_fail=0,
                               cliff_repaired=0, oob_restored=0, crashed=False) == "tolerated"


def test_repaired_requires_clean_crc():
    """A repair counter fired but ex-data is still flagged -> EB carried it, not a repair."""
    assert e6.classify_outcome(arm="on", recall=0.98, clean_recall=CLEAN, elements_crc_fail=3,
                               cliff_repaired=9, oob_restored=0, crashed=False) == "tolerated"


def test_arm_off_can_never_be_repaired():
    assert e6.classify_outcome(arm="off", recall=0.98, clean_recall=CLEAN, elements_crc_fail=0,
                               cliff_repaired=99, oob_restored=99, crashed=False) == "tolerated"


def test_silent_wrong_needs_both_bands_missed():
    # just outside the 0.005 retention band but inside the 0.01 delta band -> tolerated
    assert e6.classify_outcome(arm="off", recall=CLEAN - 0.008, clean_recall=CLEAN,
                               elements_crc_fail=None, cliff_repaired=None, oob_restored=None,
                               crashed=False) == "tolerated"
    # outside both -> silent_wrong
    assert e6.classify_outcome(arm="off", recall=CLEAN - 0.02, clean_recall=CLEAN,
                               elements_crc_fail=None, cliff_repaired=None, oob_restored=None,
                               crashed=False) == "silent_wrong"


# ---------------------------------------------------------------------------
# inject_shape — the anchor the sampler chose is the anchor the injector uses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", e6.SHAPES)
def test_inject_shape_honours_the_anchor_and_restores(rmap, clean_buf, shape):
    buf = clean_buf.copy()
    anchor = e6.sample_anchor(rmap, "ex_code", 11)["anchor_byte"]
    positions, record = e6.inject_shape(buf, shape, anchor, seed=77)
    assert record["shape"] == shape
    assert record["anchor_byte"] == anchor
    assert positions, "every shape must flip at least one bit"
    assert record["bits_flipped"] == len(positions)
    touched = {b for b, _ in positions}
    if shape == "single_cell":
        assert touched == {anchor}
    else:
        # row: the row-aligned block containing the anchor; column: the stripe through it
        lo, hi = record["coverage"]
        assert lo <= anchor < hi
        assert all(lo <= b < hi for b in touched)
    assert not np.array_equal(buf, clean_buf)
    faults.restore(buf, positions)
    assert np.array_equal(buf, clean_buf), "restore must round-trip byte-for-byte"


def test_device_column_stripe_passes_through_the_anchor_byte(rmap, clean_buf):
    buf = clean_buf.copy()
    anchor = e6.sample_anchor(rmap, "bin_code", 5)["anchor_byte"]
    positions, record = e6.inject_shape(buf, "device_column", anchor, seed=3)
    assert anchor in {b for b, _ in positions}
    assert record["col_offset"] == anchor % record["row_bytes"]
    faults.restore(buf, positions)


def test_device_shapes_smear_outside_the_target_field(rmap, clean_buf):
    """The stated finding-surface: an 8 KB row crosses many interleaved per-element fields."""
    buf = clean_buf.copy()
    a = e6.sample_anchor(rmap, "ex_code", 21)
    positions, _ = e6.inject_shape(buf, "device_row", a["anchor_byte"], seed=21)
    hits = e6.count_field_hits(rmap, positions)
    assert len(hits) > 1, f"expected an 8 KB row to cross several fields, got {hits}"
    faults.restore(buf, positions)
    assert np.array_equal(buf, clean_buf)


def test_inject_shape_rejects_an_unknown_shape(clean_buf):
    with pytest.raises(ValueError, match="shape"):
        e6.inject_shape(clean_buf.copy(), "device_diagonal", 200, seed=1)


# ---------------------------------------------------------------------------
# driver: sanity gate + full --smoke stub run
# ---------------------------------------------------------------------------

def test_sanity_gate_refuses_to_let_the_stub_write_under_artifacts():
    with pytest.raises(RuntimeError, match="artifacts"):
        e6.assert_stub_sandbox("stub", os.path.join(_ROOT, "artifacts", "phase3", "e6"))
    e6.assert_stub_sandbox("stub", os.path.join(_ROOT, "artifacts_smoke", "phase3", "e6"))
    e6.assert_stub_sandbox("real", os.path.join(_ROOT, "artifacts", "phase3", "e6"))


def test_row_count_gate_is_enforced():
    with pytest.raises(AssertionError, match="row count"):
        e6.assert_row_count([{"x": 1}], seeds=30)


def _smoke_args(out):
    return types.SimpleNamespace(adapter="stub", smoke=True, resume=False, seeds=None,
                                 seed=1234, ef=None, timeout=60.0, out=out, out_tag=None,
                                 weights=None, p3=False, p3_seeds=None)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("e6_smoke"))
    summary = e6.run(_smoke_args(out))
    return out, summary


def test_smoke_run_writes_the_full_grid(smoke_run):
    out, summary = smoke_run
    with open(os.path.join(out, "e6_results.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(e6.SHAPES) * len(e6.STRATA) * 2 * 2      # 2 smoke seeds x 2 arms
    assert list(rows[0].keys()) == e6.CSV_COLUMNS
    assert {r["shape"] for r in rows} == set(e6.SHAPES)
    assert {r["region"] for r in rows} == set(e6.STRATA)
    assert {r["arm"] for r in rows} == {"off", "on"}
    assert all(r["outcome"] in e6.OUTCOMES for r in rows)


def test_smoke_summary_has_provenance_and_per_cell_stats(smoke_run):
    out, summary = smoke_run
    with open(os.path.join(out, "e6_summary.json")) as fh:
        blob = json.load(fh)
    assert blob["meta"]["adapter_type"] == "stub"
    assert blob["meta"]["platform_confirmed_real"] is False
    assert len(blob["cells"]) == len(e6.SHAPES) * len(e6.STRATA) * 2
    cell = blob["cells"][f"single_cell|ex_code|off"]
    assert cell["n"] == 2
    assert set(cell["outcomes"]) <= set(e6.OUTCOMES)
    assert blob["sanity"]["row_count_ok"] is True
    assert blob["sanity"]["state_leak_bytes"] == 0


def test_smoke_run_leaves_no_state_leak(smoke_run):
    """The buffer the driver corrupted must end byte-identical to the clean index."""
    out, summary = smoke_run
    assert summary["sanity"]["state_leak_bytes"] == 0


def test_smoke_raw_records_carry_the_smear_accounting(smoke_run):
    out, _ = smoke_run
    with open(os.path.join(out, "raw", "e6.records.jsonl")) as fh:
        recs = [json.loads(line) for line in fh]
    assert len(recs) == len(e6.SHAPES) * len(e6.STRATA) * 2 * 2
    rows = [r for r in recs if r["shape"] == "device_row"]
    assert rows and all("field_hits" in r for r in rows)
    assert any(len(r["field_hits"]) > 1 for r in rows), "device_row must be seen to smear"
    assert all(r["coverage_span_bytes"] >= r["coverage_bytes"] for r in recs)
    # coverage_bytes must be the EXACT touched-byte count from `positions`, never the injectors'
    # bounding span — which for device_column overstates the footprint by orders of magnitude.
    cols = [r for r in recs if r["shape"] == "device_column"]
    assert cols and all(r["coverage_span_bytes"] > 10 * r["coverage_bytes"] for r in cols)
    assert all(r["coverage_bytes"] == sum(r["field_hits"].values()) for r in recs)


def test_p3_comparison_runs_on_stub(tmp_path):
    args = _smoke_args(str(tmp_path))
    args.p3 = True
    args.p3_seeds = 2
    e6.run_p3(args)
    with open(os.path.join(str(tmp_path), "e6_p3.json")) as fh:
        blob = json.load(fh)
    assert set(blob["modes"]) == {"burst_one_element", "striped_64_elements"}
    for mode in blob["modes"].values():
        assert mode["bits_flipped"] == 64
        assert mode["n_seeds"] == 2
        assert "median_delta_recall_off" in mode
        assert "median_elements_crc_fail" in mode
