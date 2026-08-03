"""E6 (shape x region damage matrix) unit tests — the pure, seed-deterministic pieces.

Everything here runs offline against the STUB geometry (arm64-safe, faiss-free). Coverage:
  - `stratum_windows` agrees with qp.rabitq.layout arithmetic for every stratum (global and
    per-element), including the spec's "`ids` = two 4-byte windows per element" case;
  - `sample_anchor` is pure/deterministic in seed and byte-weighted across a stratum's windows
    (12:8 for factors, 4:4 for ids), and the split global strata each fill their own cell;
  - `field_at_byte` / `count_field_hits` resolve absolute bytes back to the field they belong
    to — the smear accounting that makes "a row/column physically crosses interleaved fields"
    a measurement rather than a claim;
  - `classify_outcome`'s 5-class truth table (including `detected_wrong`);
  - `bounds_check_full` — flag/restore/exempt per pointer field, reach of the last element, and
    chunk-size independence;
  - `inject_shape` honours the sampled anchor, aligns a row to the buffer, round-trips through
    `faults.restore`;
  - the state-leak gate is verified to FAIL on a stray write (it used to be 0 by construction);
  - the driver's sanity gates + a full --smoke stub run (row count, CSV schema, paired arms).
"""

import argparse
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
from qp.bits import flip_bits
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


def test_global_strata_are_split_not_merged(rmap):
    """Spec deviation I3(a): rotation and centroids are separate strata, each its own window."""
    rotation = next(r for r in rmap["regions"] if r["name"] == "rotation")
    centroids = next(r for r in rmap["regions"] if r["name"] == "centroids")
    assert e6.stratum_windows(rmap, "rotation") == [(rotation["byte_start"], 64)]
    assert e6.stratum_windows(rmap, "centroids") == [(centroids["byte_start"], 8192)]
    assert set(e6.GLOBAL_STRATA) == {"rotation", "centroids"}
    assert "rotation_centroids" not in e6.STRATA
    assert len(e6.STRATA) == 7


def test_splitting_makes_the_rotation_cell_sampleable(rmap):
    """The reason for the deviation: merged and byte-weighted, rotation is <1% of the draws."""
    anchors = [e6.sample_anchor(rmap, "rotation", s)["anchor_byte"] for s in range(50)]
    rot_start, rot_len = e6.stratum_windows(rmap, "rotation")[0]
    assert all(rot_start <= a < rot_start + rot_len for a in anchors)
    assert 64 * 64 / (8192 + 64) < 1.0, "merged weighting would give <1 rotation draw in 64 seeds"


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
    """factors 12:8, ids 4:4 — deterministic seeds, so pass/fail is fixed, not flaky."""
    n = 2000
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
    ("detected_wrong: the P3 stripe — 64 elements flagged, recall still gone",
     dict(arm="on", recall=0.54, elements_crc_fail=64, cliff_repaired=0, oob_restored=0,
          crashed=False), "detected_wrong"),
    ("detected_wrong via a bounds restore that did not save the recall",
     dict(arm="on", recall=0.31, elements_crc_fail=0, cliff_repaired=0, oob_restored=7,
          crashed=False), "detected_wrong"),
    ("silent_wrong: recall gone, nothing detected",
     dict(arm="off", recall=0.42, elements_crc_fail=None, cliff_repaired=None,
          oob_restored=None, crashed=False), "silent_wrong"),
    ("silent_wrong on arm on too: recall gone and every detector stayed quiet",
     dict(arm="on", recall=0.42, elements_crc_fail=0, cliff_repaired=0, oob_restored=0,
          crashed=False), "silent_wrong"),
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


def test_arm_off_can_never_be_repaired_or_detected_wrong():
    assert e6.classify_outcome(arm="off", recall=0.98, clean_recall=CLEAN, elements_crc_fail=0,
                               cliff_repaired=99, oob_restored=99, crashed=False) == "tolerated"
    # arm off has no detector at all, so a collapse there is silent by definition
    assert e6.classify_outcome(arm="off", recall=0.1, clean_recall=CLEAN, elements_crc_fail=99,
                               cliff_repaired=99, oob_restored=99,
                               crashed=False) == "silent_wrong"


def test_all_five_outcome_classes_are_reachable_and_declared():
    assert set(e6.OUTCOMES) == {"crash", "repaired", "tolerated", "detected_wrong", "silent_wrong"}


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


def test_device_row_block_is_aligned_to_the_buffer_not_the_region(rmap, clean_buf):
    """The row is a physical DRAM block: its start is row_bytes-aligned to the BUFFER origin."""
    buf = clean_buf.copy()
    for seed in range(8):
        anchor = e6.sample_anchor(rmap, "ex_code", seed)["anchor_byte"]
        positions, record = e6.inject_shape(buf, "device_row", anchor, seed=seed,
                                            **e6.shape_kwargs("device_row", e6.FULL))
        rb = record["row_bytes"]
        assert record["coverage"][0] == (anchor // rb) * rb
        assert record["coverage"][0] % rb == 0
        assert record["coverage"][1] == min(record["coverage"][0] + rb, buf.size)
        faults.restore(buf, positions)
    assert np.array_equal(buf, clean_buf)


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
# bounds_check_full — E6's full-index replacement for E5's 16-element sample
# ---------------------------------------------------------------------------

def _write_u32(buf, off, val):
    buf[off:off + 4] = np.frombuffer(int(val).to_bytes(4, "little"), dtype=np.uint8)


def test_bounds_check_is_a_no_op_on_a_clean_index(rmap, clean_buf):
    work = clean_buf.copy()
    restored, detail = e6.bounds_check_full(work, clean_buf, rmap)
    assert (restored, detail) == (0, {})
    assert np.array_equal(work, clean_buf)


@pytest.mark.parametrize("field,offset_in_field,bad_value", [
    ("links", 0, 10 ** 6),          # neighbour COUNT > maxM0 (E5's sampled check misses this)
    ("links", 4, 10 ** 6),          # first neighbour id >= cur_element_count
    ("cluster_id", 0, 9999),        # >= num_cluster
    ("label", 0, 10 ** 6),          # >= cur_element_count
])
def test_bounds_check_flags_and_restores_each_pointer_field(rmap, clean_buf, field,
                                                            offset_in_field, bad_value):
    work = clean_buf.copy()
    start, length = layout.element_field_range(rmap, field, 40)
    _write_u32(work, start + offset_in_field, bad_value)
    restored, detail = e6.bounds_check_full(work, clean_buf, rmap)
    assert restored == 1 and detail == {field: 1}
    assert np.array_equal(work, clean_buf), "the flagged field must be restored from clean"


def test_bounds_check_exempts_the_empty_slot_sentinel(rmap, clean_buf):
    """0xFFFFFFFF is an explicitly-empty neighbour slot, exactly as E5 treats it."""
    work = clean_buf.copy()
    start, _ = layout.element_field_range(rmap, "links", 7)
    _write_u32(work, start + 4, e6.PTR_SENTINEL)
    restored, _ = e6.bounds_check_full(work, clean_buf, rmap)
    assert restored == 0
    assert not np.array_equal(work, clean_buf), "an exempt value must be left in place"


def test_bounds_check_reaches_the_last_element(rmap, clean_buf):
    """The whole point of replacing the 16-element sample: element n-1 must be seen."""
    n = rmap["header"]["cur_element_count"]
    work = clean_buf.copy()
    start, _ = layout.element_field_range(rmap, "label", n - 1)
    _write_u32(work, start, 10 ** 6)
    restored, detail = e6.bounds_check_full(work, clean_buf, rmap)
    assert restored == 1 and detail == {"label": 1}


def test_bounds_check_survives_a_chunk_smaller_than_the_index(rmap, clean_buf, monkeypatch):
    """Chunking is a memory bound, not a semantic one: results must not depend on chunk size."""
    work = clean_buf.copy()
    for e in (0, 5, 33, 63):
        start, _ = layout.element_field_range(rmap, "cluster_id", e)
        _write_u32(work, start, 9999)
    monkeypatch.setattr(e6, "BOUNDS_CHUNK_ELEMENTS", 7)
    restored, detail = e6.bounds_check_full(work, clean_buf, rmap)
    assert restored == 4 and detail == {"cluster_id": 4}
    assert np.array_equal(work, clean_buf)


def test_bounds_check_counts_multiple_fields_of_one_element_separately(rmap, clean_buf):
    work = clean_buf.copy()
    for field in ("cluster_id", "label"):
        start, _ = layout.element_field_range(rmap, field, 12)
        _write_u32(work, start, 10 ** 6 if field == "label" else 9999)
    restored, detail = e6.bounds_check_full(work, clean_buf, rmap)
    assert restored == 2 and detail == {"cluster_id": 1, "label": 1}


# ---------------------------------------------------------------------------
# driver: sanity gate + full --smoke stub run
# ---------------------------------------------------------------------------

def test_sanity_gate_refuses_to_let_the_stub_write_under_artifacts():
    real_out = os.path.join(_ROOT, "artifacts", "phase3", "e6")
    smoke_out = os.path.join(_ROOT, "artifacts_smoke", "phase3", "e6")
    with pytest.raises(RuntimeError, match="artifacts"):
        e6.assert_stub_sandbox("stub", real_out)
    e6.assert_stub_sandbox("stub", smoke_out)
    e6.assert_stub_sandbox("real", real_out)
    # the summary's sanity flag is DERIVED from this predicate, never hardcoded
    assert e6.stub_sandbox_ok("stub", real_out) is False
    assert e6.stub_sandbox_ok("stub", smoke_out) is True
    assert e6.stub_sandbox_ok("real", real_out) is True


def test_row_count_gate_is_enforced():
    with pytest.raises(AssertionError, match="row count"):
        e6.assert_row_count([{"x": 1}], seeds=30)


def test_row_count_gate_adapts_to_a_shard():
    rows = [{"x": 1}] * (1 * 2 * 3 * 2)                # 1 shape x 2 strata x 3 seeds x 2 arms
    assert e6.assert_row_count(rows, seeds=3, shapes=("device_row",),
                               strata=("ids", "links")) == 12
    with pytest.raises(AssertionError, match="row count"):   # would pass against the full grid?
        e6.assert_row_count(rows, seeds=3, shapes=("device_row",), strata=("ids",))


# ---------------------------------------------------------------------------
# --shapes / --strata shard filters
# ---------------------------------------------------------------------------

def test_csv_list_validates_against_the_known_sets():
    parse = e6.csv_list("stratum", e6.STRATA)
    assert parse("ids,links") == ("links", "ids")       # canonicalised to STRATA order
    assert parse(" rotation , ids ") == ("rotation", "ids")
    for bad in ("idz", "ids,rotaton", "", "ids,ids"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse(bad)


def test_csv_list_rejects_a_stratum_that_is_only_a_shape_and_vice_versa():
    with pytest.raises(argparse.ArgumentTypeError, match="unknown stratum"):
        e6.csv_list("stratum", e6.STRATA)("device_row")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown shape"):
        e6.csv_list("shape", e6.SHAPES)("ex_code")


def test_cli_parses_the_shard_filters_and_rejects_typos(capsys):
    # exercise the parser through main()'s argv path: a typo must exit non-zero, not run a
    # 6-of-7-strata grid that quietly leaves a hole in the merged matrix.
    with pytest.raises(SystemExit):
        e6.main(["--adapter", "stub", "--smoke", "--strata", "ids,rotaton"])
    assert "unknown stratum" in capsys.readouterr().err


SHARD_SHAPES, SHARD_STRATA = ("device_column",), ("links", "ids")


@pytest.fixture(scope="module")
def shard_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("e6_shard"))
    args = _smoke_args(out)
    args.shapes, args.strata, args.out_tag = SHARD_SHAPES, SHARD_STRATA, "shard0"
    return out, e6.run(args)


def test_shard_runs_only_its_slice(shard_run):
    out, summary = shard_run
    with open(os.path.join(out, "e6_results_shard0.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(SHARD_SHAPES) * len(SHARD_STRATA) * 2 * 2
    assert {r["shape"] for r in rows} == set(SHARD_SHAPES)
    assert {r["region"] for r in rows} == set(SHARD_STRATA)
    assert summary["sanity"]["row_count_ok"] is True
    assert summary["shard"]["is_shard"] is True
    assert summary["shard"]["full_grid_rows"] == len(e6.SHAPES) * len(e6.STRATA) * 2 * 2
    # no empty placeholder cells for slices another shard is running
    assert len(summary["cells"]) == len(SHARD_SHAPES) * len(SHARD_STRATA) * 2
    assert all(c["n"] == 2 for c in summary["cells"].values())


def test_shard_reproduces_the_full_grid_exactly_for_shared_cells(shard_run, smoke_run):
    """A sharded run must be a partition of the full grid, not a different experiment.

    Cell seeds come from SeedSequence([root, crc32(shape), crc32(stratum), i]) — never from a
    loop counter — so the anchor, the injected positions (compared here via field_hits, which is
    derived from them) and the measured recall must be bit-identical either way.
    """
    shard_out, _ = shard_run
    full_out, _ = smoke_run

    def load(path):
        with open(path) as fh:
            return {(r["shape"], r["region"], r["seed_index"], r["arm"]): r
                    for r in (json.loads(line) for line in fh)}

    shard = load(os.path.join(shard_out, "raw", "e6_shard0.records.jsonl"))
    full = load(os.path.join(full_out, "raw", "e6.records.jsonl"))
    shared = set(shard) & set(full)
    assert len(shared) == len(SHARD_SHAPES) * len(SHARD_STRATA) * 2 * 2, "shard must be a subset"
    for key in sorted(shared):
        a, b = shard[key], full[key]
        for field in ("seed", "element", "anchor_byte", "anchor_window_index", "bits_flipped",
                      "coverage_bytes", "coverage_lo", "coverage_hi", "field_hits",
                      "bytes_in_anchor_field", "recall", "delta_recall", "outcome",
                      "elements_crc_fail", "cliff_repaired", "oob_restored"):
            assert a[field] == b[field], f"{key} differs on {field}: {a[field]} != {b[field]}"


# ---------------------------------------------------------------------------
# the state-leak gate must be able to FAIL (review I1)
# ---------------------------------------------------------------------------

def test_verify_and_restore_catches_a_write_outside_the_fault_footprint(clean_buf):
    work = clean_buf.copy()
    positions = [(100, 0), (101, 3)]
    flip_bits(work, positions)
    work[5000] ^= np.uint8(0x40)                       # a stray write nobody accounted for
    with pytest.raises(RuntimeError, match="outside"):
        e6._verify_and_restore(work, clean_buf, positions, "off")


def test_verify_and_restore_catches_it_on_the_arm_that_copies_clean_back(clean_buf):
    """arm on restores wholesale, so a post-restore check would see nothing — this one is pre."""
    work = clean_buf.copy()
    positions = [(300, 1)]
    flip_bits(work, positions)
    work[9000] ^= np.uint8(0x01)
    with pytest.raises(RuntimeError, match="outside"):
        e6._verify_and_restore(work, clean_buf, positions, "on")


def test_verify_and_restore_catches_positions_that_did_not_actually_flip(clean_buf):
    work = clean_buf.copy()
    with pytest.raises(RuntimeError, match="bookkeeping"):
        e6._verify_and_restore(work, clean_buf, [(700, 2)], "off")


def test_verify_and_restore_accepts_a_guard_style_partial_repair(clean_buf):
    """arm on legitimately ends with FEWER differing bytes than it touched (a layer repaired)."""
    work = clean_buf.copy()
    positions = [(400, 0), (401, 0), (402, 0)]
    flip_bits(work, positions)
    work[401] = clean_buf[401]                         # as if a recovery layer put it back
    assert e6._verify_and_restore(work, clean_buf, positions, "on") == 2
    assert np.array_equal(work, clean_buf)


def _smoke_args(out):
    return types.SimpleNamespace(adapter="stub", smoke=True, resume=False, seeds=None,
                                 seed=1234, ef=None, timeout=60.0, out=out, out_tag=None,
                                 weights=None, p3=False, p3_seeds=None, row_bytes=None,
                                 n_rows=None, p_in_row=None, shapes=None, strata=None)


def test_smoke_run_state_leak_gate_actually_fires(tmp_path, monkeypatch):
    """The gate this replaces reported 'leak 0' with a stray byte injected (review I1)."""
    real_inject = e6.inject_shape

    def scribbling_inject(buf, shape, anchor_byte, seed, **kwargs):
        positions, record = real_inject(buf, shape, anchor_byte, seed, **kwargs)
        buf[7] ^= np.uint8(0x80)                       # outside `positions`, so unaccounted for
        return positions, record

    monkeypatch.setattr(e6, "inject_shape", scribbling_inject)
    with pytest.raises(RuntimeError, match="outside"):
        e6.run(_smoke_args(str(tmp_path)))


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
    cell = blob["cells"]["single_cell|ex_code|off"]
    assert cell["n"] == 2
    assert set(cell["outcomes"]) <= set(e6.OUTCOMES)
    assert blob["sanity"]["row_count_ok"] is True
    assert blob["sanity"]["stub_sandbox_ok"] is True
    assert blob["sanity"]["state_leak_bytes"] == 0
    assert blob["sanity"]["final_research_drift"] == 0.0
    assert blob["sanity"]["per_eval_footprint_gate_evals"] == blob["sanity"]["rows"]
    # both clean baselines must be recorded, not just computed and dropped (review I2)
    both = blob["meta"]["clean_baseline_both_paths"]
    assert both["abs_diff"] <= e6.RETENTION_TOL
    assert both["query_with_recovery_recall@10"] is not None
    assert "strata_split" in blob["meta"]["spec_deviations"]
    assert "bounds_check" in blob["meta"]["spec_deviations"]


def test_smoke_run_leaves_no_state_leak(smoke_run):
    """Byte-identical AND still answers queries identically (the two independent gates)."""
    out, summary = smoke_run
    assert summary["sanity"]["state_leak_bytes"] == 0
    assert summary["sanity"]["final_research_drift"] == 0.0


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
    # the two smear denominators are distinct and ordered: element-field <= field <= touched
    assert all(r["bytes_in_anchor_element_field"] <= r["bytes_in_anchor_field"] <=
               r["coverage_bytes"] for r in recs)
    assert any(r["bytes_in_anchor_element_field"] < r["bytes_in_anchor_field"] for r in rows), \
        "a row leaves the anchor element but stays partly in the same KIND of field"


def test_smoke_arms_are_paired_on_the_same_physical_fault(smoke_run):
    """off/on must be the SAME injection measured twice, or their delta is not a comparison."""
    out, _ = smoke_run
    with open(os.path.join(out, "raw", "e6.records.jsonl")) as fh:
        recs = [json.loads(line) for line in fh]
    by_cell = {}
    for r in recs:
        by_cell.setdefault((r["shape"], r["region"], r["seed_index"]), {})[r["arm"]] = r
    assert len(by_cell) == len(e6.SHAPES) * len(e6.STRATA) * 2
    for key, pair in by_cell.items():
        assert set(pair) == {"off", "on"}, key
        off, on = pair["off"], pair["on"]
        assert off["seed"] == on["seed"], key           # same injector lane seed
        assert off["anchor_byte"] == on["anchor_byte"], key
        assert off["element"] == on["element"], key
        assert off["bits_flipped"] == on["bits_flipped"], key
        assert off["coverage_bytes"] == on["coverage_bytes"], key


def test_smoke_arm_on_supersedes_the_guards_sampled_bounds_check(smoke_run):
    """bounds_check_full runs first, so the guard's own sampled check must find nothing left."""
    out, _ = smoke_run
    with open(os.path.join(out, "raw", "e6.records.jsonl")) as fh:
        recs = [json.loads(line) for line in fh]
    on_rows = [r for r in recs if r["arm"] == "on"]
    assert on_rows and all(r["guard_oob_elements"] == 0 for r in on_rows)
    assert all(r["oob_restored"] is not None for r in on_rows)
    assert all(r["oob_restored"] is None for r in recs if r["arm"] == "off")


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


# ---------------------------------------------------------------------------
# --cliff-regions: which global regions arm ON protects
# ---------------------------------------------------------------------------

def test_cliff_regions_default_is_rotation_only():
    """Every shard produced before this flag existed ran under this default."""
    assert e6.CLIFF_REGIONS_DEFAULT == ("rotation",)
    args = _smoke_args("/tmp/unused")
    assert getattr(args, "cliff_regions", None) is None
    # setup_context resolves the default even when the namespace has no such attribute at all,
    # which is how the pre-existing tests keep working unchanged.
    assert tuple(getattr(args, "cliff_regions", None) or e6.CLIFF_REGIONS_DEFAULT) == ("rotation",)


def test_cliff_regions_cli_rejects_a_typo(capsys):
    """A silent fallback here would report centroid numbers produced without centroid protection."""
    with pytest.raises(SystemExit):
        e6.main(["--adapter", "stub", "--smoke", "--cliff-regions", "rotation,centriods"])
    assert "unknown cliff region" in capsys.readouterr().err


def test_cliff_regions_cli_rejects_a_stratum_that_is_not_a_cliff_region(capsys):
    """`links` is a valid stratum but not a global region; the two vocabularies must not blur."""
    with pytest.raises(SystemExit):
        e6.main(["--adapter", "stub", "--smoke", "--cliff-regions", "links"])
    assert "unknown cliff region" in capsys.readouterr().err


def test_every_allowed_cliff_region_is_in_the_region_map(rmap):
    """A name the flag accepts must be resolvable, or the sweep dies mid-run rather than at parse."""
    names = {r["name"] for r in rmap["regions"]}
    for region in e6.CLIFF_REGIONS_ALLOWED:
        assert region in names, f"{region} is offered by --cliff-regions but not in the region map"


@pytest.fixture(scope="module")
def cliff_run(tmp_path_factory):
    """A smoke sweep with the centroids and header protected, restricted to the cells they change."""
    out = str(tmp_path_factory.mktemp("e6_cliff"))
    args = _smoke_args(out)
    args.strata, args.out_tag = ("centroids",), "cliff"
    args.cliff_regions = ("rotation", "centroids", "header")
    return out, e6.run(args)


def test_cliff_regions_reach_the_summary_provenance(cliff_run):
    """A reader must be able to tell which configuration produced a given shard."""
    _, summary = cliff_run
    assert tuple(summary["cfg"]["cliff_regions"]) == ("rotation", "centroids", "header")


def test_centroid_protection_converts_the_crash_and_silent_cells(cliff_run):
    """Arm ON must no longer crash or go silently wrong anywhere in the centroid column.

    Arm OFF is the control and is expected to keep crashing: the flag configures protection,
    not the fault.
    """
    out, _ = cliff_run
    with open(os.path.join(out, "e6_results_cliff.csv")) as fh:
        rows = list(csv.DictReader(fh))
    on = [r for r in rows if r["arm"] == "on"]
    off = [r for r in rows if r["arm"] == "off"]
    assert on and off
    assert not [r for r in on if r["outcome"] in ("crash", "silent_wrong")], \
        f"protected centroid cells still failing: {[r['outcome'] for r in on]}"
    assert [r for r in off if r["outcome"] == "crash"], \
        "the unprotected arm should still crash — otherwise the fault is not being injected"


def test_raw_records_attribute_the_repair_to_a_region(cliff_run):
    """With three protected regions the flat counter cannot say which one acted."""
    out, _ = cliff_run
    with open(os.path.join(out, "raw", "e6_cliff.records.jsonl")) as fh:
        recs = [json.loads(line) for line in fh]
    on = [r for r in recs if r["arm"] == "on"]
    assert on, "no protected records"
    assert all(set(r["cliff_by_region"]) == {"rotation", "centroids", "header"} for r in on)
    assert all(r["cliff_repaired"] == sum(r["cliff_by_region"].values()) for r in on), \
        "the flat cliff_repaired must be the per-region sum"
    off = [r for r in recs if r["arm"] == "off"]
    assert all(r["cliff_by_region"] is None for r in off), \
        "arm off measures nothing, so it must report None rather than a measured zero"
