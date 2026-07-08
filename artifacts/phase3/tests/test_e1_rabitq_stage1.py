"""Stage 1 foundation tests: bit-class derivation, layout helpers, stub adapter, registry.

No RaBitQ binaries needed — everything here runs against the deterministic stub adapter and the
source-derived layout, which is the whole point of the stub: prove the injection/measurement
plumbing is correct offline so the runner can be carried to the x86 workstation unchanged.
"""
import os

import numpy as np
import pytest

from qp import metrics, bits
from qp.rabitq import get_adapter, adapter_name, layout, bitclass
from qp.rabitq import adapter as real_adapter
from qp.rabitq import stub_adapter


# --- bit-class derivation (from source, not assumed float32) ------------------

def test_rotation_four_stages():
    pd = 128
    # 64-byte flip_ = 4 sign-flip stages of padded_dim/8 = 16 bytes each (rotator.hpp rotate()).
    assert bitclass.bit_class("rotation", 0, 0, padded_dim=pd) == "rot_stage0"
    assert bitclass.bit_class("rotation", 15, 7, padded_dim=pd) == "rot_stage0"
    assert bitclass.bit_class("rotation", 16, 0, padded_dim=pd) == "rot_stage1"
    assert bitclass.bit_class("rotation", 32, 0, padded_dim=pd) == "rot_stage2"
    assert bitclass.bit_class("rotation", 63, 0, padded_dim=pd) == "rot_stage3"


def test_factor_fields_and_f_error_marked():
    pd = 128
    # bin_factors = [f_add(0..3), f_rescale(4..7), f_error(8..11)]; sign bit is byte3 bit7.
    assert bitclass.bit_class("elem0.bin_factors", 3, 7, padded_dim=pd) == "f_add:sign"
    assert bitclass.bit_class("elem0.bin_factors", 7, 7, padded_dim=pd) == "f_rescale:sign"
    assert bitclass.bit_class("elem0.bin_factors", 11, 7, padded_dim=pd) == "f_error:sign"
    assert bitclass.bit_class("elem0.bin_factors", 8, 0, padded_dim=pd) == "f_error:mantissa-low"
    # ex_factors = [f_add_ex(0..3), f_rescale_ex(4..7)]; NO f_error (ExDataMap has only 2 floats).
    assert bitclass.bit_class("elem0.ex_factors", 7, 7, padded_dim=pd) == "f_rescale_ex:sign"
    assert bitclass.bit_class("elem0.ex_factors", 4, 0, padded_dim=pd) == "f_rescale_ex:mantissa-low"


def test_pointer_lanes_and_flat_codes():
    pd = 128
    assert bitclass.bit_class("elem0.links", 0, 0, padded_dim=pd) == "ptr_low"
    assert bitclass.bit_class("elem0.links", 1, 0, padded_dim=pd) == "ptr_low"
    assert bitclass.bit_class("elem0.links", 2, 0, padded_dim=pd) == "ptr_high"
    assert bitclass.bit_class("elem0.cluster_id", 3, 0, padded_dim=pd) == "ptr_high"
    assert bitclass.bit_class("elem0.bin_code", 5, 2, padded_dim=pd) == "bin_sign"
    assert bitclass.bit_class("elem0.ex_code", 9, 1, padded_dim=pd) == "ex_code"   # flat (report)


def test_unknown_structure_raises():
    with pytest.raises(KeyError):
        bitclass.bit_class("elem0.mystery", 0, 0, padded_dim=128)


def test_ex_code_report_present():
    assert "REPORT-AND-STOP" in bitclass.EX_CODE_REPORT


# --- layout helpers -----------------------------------------------------------

def _rmap():
    return get_adapter("stub").region_map()


def test_element_field_range_matches_elem0_and_strides():
    rmap = _rmap()
    spe = rmap["header"]["size_data_per_element"]
    for field in ("links", "cluster_id", "bin_code", "bin_factors", "ex_code", "ex_factors"):
        elem0 = next(r for r in rmap["regions"] if r["name"] == f"elem0.{field}")
        bs0, blen0 = layout.element_field_range(rmap, field, 0)
        assert (bs0, blen0) == (elem0["byte_start"], elem0["byte_len"])
        bs1, _ = layout.element_field_range(rmap, field, 1)
        assert bs1 - bs0 == spe                      # element e is exactly e*spe later


def test_element_field_range_bounds():
    rmap = _rmap()
    n = rmap["header"]["cur_element_count"]
    with pytest.raises(IndexError):
        layout.element_field_range(rmap, "bin_factors", n)      # out of range


def test_aggregate_region_map_scales_per_vector_only():
    rmap = _rmap()
    n = rmap["header"]["cur_element_count"]
    agg = {r["name"]: r for r in layout.aggregate_region_map(rmap)["regions"]}
    # per-vector structures priced for ALL elements; globals at face value.
    assert agg["bin_factors"]["byte_len"] == 12 * n and agg["bin_factors"]["per_vector"]
    assert agg["rotation"]["byte_len"] == layout.rotation_bytes(128) and not agg["rotation"]["per_vector"]
    assert "level0" not in agg and "upper_links" not in agg   # containers dropped


# --- registry: config-driven switch + arm64 guard -----------------------------

def test_registry_explicit_and_env(monkeypatch):
    assert adapter_name(get_adapter("stub")) == "stub"
    monkeypatch.setenv("QP_RABITQ_ADAPTER", "stub")
    assert adapter_name(get_adapter()) == "stub"               # env honored
    assert adapter_name(get_adapter("stub")) == "stub"         # explicit beats env


def test_registry_auto_is_stub_without_binaries():
    # dev host has no x86 binaries -> auto resolves to stub (real on the workstation).
    if not real_adapter.binaries_built():
        assert adapter_name(get_adapter("auto")) == "stub"


def test_registry_real_reports_and_stops_without_binaries():
    if not real_adapter.binaries_built():
        with pytest.raises(RuntimeError, match="REPORT-AND-STOP"):
            get_adapter("real")                                # never silently downgrades


def test_registry_rejects_bad_name():
    with pytest.raises(ValueError):
        get_adapter("bogus")


# --- stub adapter: confinement, restore, determinism, structure resolution ----

def test_injection_confined_and_restore_byte_identical(tmp_path):
    a = get_adapter("stub")
    rmap = a.region_map()
    pristine = a.serialize_index()
    out = str(tmp_path / "c.index")
    for name in ("rotation", "centroids", "elem0.bin_factors", "elem0.links"):
        reg = next(r for r in rmap["regions"] if r["name"] == name)
        buf = pristine.copy()
        bits.flip_bit(buf, reg["byte_start"], 3)
        a.deserialize_index(buf, out)
        back = np.fromfile(out, dtype=np.uint8)
        diff = np.nonzero(back != pristine)[0]
        assert diff.size == 1
        assert reg["byte_start"] <= int(diff[0]) < reg["byte_start"] + reg["byte_len"], name
        bits.flip_bit(buf, reg["byte_start"], 3)               # XOR restore
        assert np.array_equal(buf, pristine), f"restore not byte-identical for {name}"


def test_stub_deterministic(tmp_path):
    a = get_adapter("stub")
    rmap = a.region_map()
    rot = next(r for r in rmap["regions"] if r["name"] == "rotation")
    out = str(tmp_path / "c.index")

    def run_once():
        buf = a.serialize_index()
        bits.flip_bit(buf, rot["byte_start"], 0)
        a.deserialize_index(buf, out)
        r = a.search_corrupted(out)
        return r["cpp_recall"], r["ids"].tolist()
    assert run_once() == run_once()                            # same seed -> identical


def test_stub_outcomes_drive_every_branch(tmp_path):
    """rotation -> silent collapse; ex_factors -> nan-inf (excluded); pointer-high/header -> crash;
    bin_code low bit -> clean. Collapse/nan-inf go through the imported qp.metrics predicate."""
    import subprocess
    a = get_adapter("stub")
    rmap = a.region_map()
    gt = a.load_groundtruth()
    out = str(tmp_path / "c.index")
    clean = a.clean_baseline_recall()

    def search_after(byte_pos, bit):
        buf = a.serialize_index()
        bits.flip_bit(buf, byte_pos, bit)
        a.deserialize_index(buf, out)
        return a.search_corrupted(out)

    rot = next(r for r in rmap["regions"] if r["name"] == "rotation")
    res = search_after(rot["byte_start"], 0)
    r10 = metrics.recall_at_k(res["ids"], gt, 10)
    assert metrics.is_silent_collapse(r10, clean) is True      # collapse via qp.metrics

    bs, _ = layout.element_field_range(rmap, "ex_factors", 0)
    res = search_after(bs, 0)
    r10 = metrics.recall_at_k(res["ids"], gt, 10)
    fm = metrics.classify_failure(distances=res["distances"], recall=r10, clean_recall=clean)
    assert fm == metrics.NAN_INF
    assert metrics.is_silent_collapse(r10, clean, failure_mode=fm) is False   # nan-inf excluded

    bs, _ = layout.element_field_range(rmap, "links", 0)
    with pytest.raises(subprocess.CalledProcessError):         # pointer high lane -> crash
        search_after(bs + 3, 0)

    bs, _ = layout.element_field_range(rmap, "bin_code", 0)
    res = search_after(bs, 0)
    assert metrics.recall_at_k(res["ids"], gt, 10) == pytest.approx(clean)     # clean


def test_stub_resolves_high_element(tmp_path):
    """A flip in element e>0 lands in level0 (not elem0.*) but resolves to the right structure."""
    a = get_adapter("stub")
    rmap = a.region_map()
    n = rmap["header"]["cur_element_count"]
    bs, _ = layout.element_field_range(rmap, "bin_factors", n - 1)
    assert stub_adapter._resolve_structure(bs).endswith("bin_factors")
