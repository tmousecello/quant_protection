"""E9 (on-die-ECC miscorrection x region) driver tests — the pieces E9 does NOT inherit from E6.

E9 is deliberately a thin driver over `phase3_e6_shapes`: the stratum sampler, the anchor
seeding, `measure_cell`, `bounds_check_full`, the outcome classifier and the state-leak gates are
all E6's, already covered by test_e6_shapes.py. What is tested here is exactly what is new:

  - registering `miscorrection_word` into E6's shape dispatch does NOT enlarge E6's own grid
    (the published E6 artifact's row-count gate and shard metadata must keep their meaning);
  - the registered shape dispatches through `inject_shape`, honours the stratified anchor, and
    lands its 3 bits inside ONE ECC word even when that word straddles the anchor's field;
  - the shape's damage is a LOCALIZED burst — orders of magnitude smaller than a device row's,
    which is the whole reason this event class needs its own row in the damage matrix;
  - the driver's own gates (row count over 1 shape x 7 strata, stub sandbox, ef default) and a
    full --smoke stub run producing the E6-schema CSV with paired arms.

Runs offline against the STUB geometry (arm64-safe, faiss-free), like test_e6_shapes.py.
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qp.rabitq import stub_adapter

import phase3_e6_shapes as e6
import phase3_e9_miscorrection as e9


@pytest.fixture(scope="module")
def rmap():
    return stub_adapter.region_map()


@pytest.fixture(scope="module")
def clean_buf():
    return stub_adapter.serialize_index()


# ---------------------------------------------------------------------------
# registration: E9 borrows E6's machinery without mutating E6's grid
# ---------------------------------------------------------------------------

def test_e6_grid_is_untouched_by_the_registration():
    """Importing E9 must not add a 4th shape to E6's 3x7 grid — the published E6 artifact's
    row-count gate, shard metadata and merged CSV all assume exactly those three."""
    assert e6.SHAPES == ("single_cell", "device_row", "device_column")
    assert e9.SHAPE not in e6.SHAPES
    assert e9.SHAPE in e6.EXTRA_SHAPES


def test_shape_kwargs_come_from_the_e9_config():
    cfg = dict(e9.FULL)
    assert e6.shape_kwargs(e9.SHAPE, cfg) == {"word_bytes": 16, "k_raw": 2}
    assert e6.shape_kwargs(e9.SHAPE, dict(cfg, word_bytes=8, k_raw=3)) == {"word_bytes": 8,
                                                                          "k_raw": 3}


def test_unknown_shape_error_lists_the_registered_one(clean_buf):
    with pytest.raises(ValueError, match="miscorrection_word"):
        e6.inject_shape(clean_buf.copy(), "not_a_shape", 0, 1)


# ---------------------------------------------------------------------------
# injection through the E6 dispatch, on the real index geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stratum", e6.STRATA)
def test_injection_honours_the_anchor_and_stays_in_one_word(rmap, clean_buf, stratum):
    work = clean_buf.copy()
    seed = e6.cell_seed(1234, e9.SHAPE, stratum, 0)
    anchor = e6.sample_anchor(rmap, stratum, seed)
    positions, record = e6.inject_shape(work, e9.SHAPE, anchor["anchor_byte"],
                                        e6.lane_seed(seed, e6.LANE_INJECT),
                                        **e6.shape_kwargs(e9.SHAPE, e9.FULL))
    assert record["anchor_byte"] == anchor["anchor_byte"]
    lo, hi = record["coverage"]
    assert lo % 16 == 0 and lo <= anchor["anchor_byte"] < hi
    assert record["bits_flipped"] == 3 and all(lo <= b < hi for b, _ in positions)
    assert 1 <= record["bytes_touched"] <= 3
    e6.faults.restore(work, positions)
    assert np.array_equal(work, clean_buf)


def test_miscorrection_is_far_more_localized_than_a_device_row(rmap, clean_buf):
    """The claim that earns this shape its own row: same anchor, ~4 orders of magnitude less
    damage than the row shape (3 bits in <=16 B vs ~32k bits over 8 KB)."""
    seed = e6.cell_seed(7, "ex_code", "ex_code", 0)
    anchor = e6.sample_anchor(rmap, "ex_code", seed)["anchor_byte"]
    mis, mrec = e6.inject_shape(clean_buf.copy(), e9.SHAPE, anchor, 5,
                                **e6.shape_kwargs(e9.SHAPE, e9.FULL))
    row, rrec = e6.inject_shape(clean_buf.copy(), "device_row", anchor, 5,
                                **e6.shape_kwargs("device_row", e6.FULL))
    assert mrec["bits_flipped"] == 3 < rrec["bits_flipped"]
    assert mrec["coverage"][1] - mrec["coverage"][0] == 16
    assert rrec["coverage"][1] - rrec["coverage"][0] > 100 * 16


# ---------------------------------------------------------------------------
# driver gates
# ---------------------------------------------------------------------------

def test_row_count_gate_is_one_shape_by_seven_strata():
    records = [{}] * (1 * len(e6.STRATA) * 3 * 2)
    assert e9.assert_row_count(records, 3) == len(records)
    with pytest.raises(AssertionError):
        e9.assert_row_count(records[:-1], 3)


def test_row_count_gate_adapts_to_a_stratum_shard():
    records = [{}] * (2 * 3 * 2)
    assert e9.assert_row_count(records, 3, strata=("links", "ids")) == len(records)


def test_cli_rejects_a_stratum_typo(capsys):
    with pytest.raises(SystemExit):
        e9.main(["--strata", "lnks"])
    assert "unknown stratum" in capsys.readouterr().err


def test_default_ef_is_the_fast_operating_point():
    """E9 sweeps at ef=64 (E6 used ef=2000). Recorded loudly because delta_recall is measured
    against THIS run's own clean baseline, so E6 and E9 deltas are not interchangeable."""
    assert e9.FULL["ef"] == 64 and e9.FULL["seeds"] == 30


def test_stub_cannot_write_into_the_real_artifacts_tree():
    assert not e6.stub_sandbox_ok("stub", os.path.join(e9.config.ROOT, "artifacts", "phase3", "e9"))
    assert e6.stub_sandbox_ok("stub", os.path.join(e9.config.ROOT, "artifacts_smoke", "phase3"))


# ---------------------------------------------------------------------------
# full --smoke stub run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("e9_smoke"))
    summary = e9.main(["--adapter", "stub", "--smoke", "--out", out], return_summary=True)
    return out, summary


def test_smoke_run_writes_the_full_stratum_sweep(smoke_run):
    out, summary = smoke_run
    with open(os.path.join(out, "e9_results.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(e6.STRATA) * e9.SMOKE["seeds"] * len(e6.ARMS)
    assert list(rows[0]) == e6.CSV_COLUMNS            # same schema as e6_results.csv
    assert {r["shape"] for r in rows} == {e9.SHAPE}
    assert {r["region"] for r in rows} == set(e6.STRATA)
    assert summary["sanity"]["row_count_ok"] and summary["sanity"]["state_leak_bytes"] == 0


def test_smoke_summary_carries_provenance_and_the_citation(smoke_run):
    _, summary = smoke_run
    meta = summary["meta"]
    assert meta["clean_baseline_both_paths"]["abs_diff"] <= e6.RETENTION_TOL
    model = meta["miscorrection_modeling"]
    assert model["word_bytes"] == 16 and model["k_raw"] == 2 and model["observed_bits"] == 3
    assert "A-003" in json.dumps(model) and "X-003" in json.dumps(model)
    assert "A-GAP-03" in json.dumps(model)            # the confidential-matrix caveat
    for stratum in e6.STRATA:
        for arm in e6.ARMS:
            cell = summary["cells"][f"{e9.SHAPE}|{stratum}|{arm}"]
            assert cell["n"] == e9.SMOKE["seeds"]
            assert cell["median_bits_flipped"] == 3


def test_smoke_arms_are_paired_on_the_same_physical_fault(smoke_run):
    out, _ = smoke_run
    with open(os.path.join(out, "raw", "e9.records.jsonl")) as fh:
        recs = [json.loads(line) for line in fh]
    by_cell = {}
    for r in recs:
        by_cell.setdefault((r["region"], r["seed_index"]), {})[r["arm"]] = r
    assert by_cell
    for (_stratum, _i), pair in by_cell.items():
        assert set(pair) == set(e6.ARMS)
        assert pair["off"]["anchor_byte"] == pair["on"]["anchor_byte"]
        assert pair["off"]["coverage_bytes"] == pair["on"]["coverage_bytes"]
