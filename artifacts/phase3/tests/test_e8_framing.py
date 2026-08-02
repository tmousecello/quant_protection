"""E8 — upper-link framing region math (qp.rabitq.layout.parse_upper_link_records).

The upper_links block is the only VARIABLE-length part of a serialized RaBitQ index: it is a
sequence of `[uint32 len][len bytes]` records, one per element, with no index and no padding.
Every E8 number (which bytes are length framing, which are payload, what a flip does to the
stream) is derived from that parse, so the parse itself needs an oracle.

THE ORACLE: sum over records of (4 + len_i) must equal the upper_links region size EXACTLY. The
region size is computed independently — from the header (level0 end) and the file size (rotation
is the last thing written, hnsw.hpp L664) — so an exact match can only happen if every record
boundary was found correctly. One byte of drift anywhere and the total misses.

The offline tests build a synthetic record stream and need no binaries. The real-index test is
skipped unless the x86 build + index are present.
"""
import csv
import json
import os
import struct

import numpy as np
import pytest

import phase3_e8_framing as e8
from qp import config
from qp.bits import flip_bit
from qp.rabitq import adapter, layout

SLPE = 68          # size_links_per_element on the SIFT b=7 index (maxM*4 + 4 = 16*4 + 4)
MAXLEVEL = 5


def _synth_stream(lens):
    """Serialize `lens` the way HierarchicalNSW::save() does (hnsw.hpp L655-662)."""
    out = bytearray()
    for i, L in enumerate(lens):
        out += struct.pack("<I", L)
        out += bytes((i + j) & 0xFF for j in range(L))       # recognizable payload
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


# --- parser -------------------------------------------------------------------

def test_parse_synthetic_stream_round_trips():
    lens = [0, SLPE, 0, 2 * SLPE, 0, 0, 5 * SLPE]
    buf = _synth_stream(lens)
    recs = layout.parse_upper_link_records(buf, len(lens), 0, buf.size)
    assert recs["lens"] == lens
    assert recs["consumed"] == buf.size
    # length words sit at the start of each record; payload follows
    assert recs["len_offsets"][0] == 0
    assert recs["len_offsets"][1] == 4 + lens[0]
    assert recs["framing_bytes"] == 4 * len(lens)
    assert recs["payload_bytes"] == sum(lens)
    assert recs["framing_bytes"] + recs["payload_bytes"] == buf.size


def test_parse_is_offset_relative_not_zero_based():
    """The region does not start at 0 in a real index; offsets must be absolute."""
    lens = [0, SLPE, 2 * SLPE]
    body = _synth_stream(lens)
    base = 1234
    buf = np.concatenate([np.zeros(base, dtype=np.uint8), body])
    recs = layout.parse_upper_link_records(buf, len(lens), base, body.size)
    assert recs["len_offsets"][0] == base
    assert recs["consumed"] == body.size
    assert all(base <= o < base + body.size for o in recs["len_offsets"])


def test_parse_rejects_under_consumption():
    """Records stop short of the region end -> 'does not tile' (the drift the oracle exists for)."""
    lens = [0, SLPE, 2 * SLPE]
    body = _synth_stream(lens)
    buf = np.concatenate([body, np.zeros(4, dtype=np.uint8)])       # 4 unexplained trailing bytes
    with pytest.raises(ValueError, match="do not tile"):
        layout.parse_upper_link_records(buf, len(lens), 0, buf.size)


def test_parse_rejects_over_consumption():
    """Records run past the region end -> loud overrun, never a plausible-looking partial parse."""
    lens = [0, SLPE, 2 * SLPE]
    buf = _synth_stream(lens)
    with pytest.raises(ValueError, match="overruns"):
        layout.parse_upper_link_records(buf, len(lens), 0, buf.size - 1)


def test_parse_rejects_a_corrupt_length_that_overruns():
    lens = [0, SLPE]
    buf = _synth_stream(lens)
    struct.pack_into("<I", buf.data, 0, 1 << 30)             # element 0's len -> 1 GB
    with pytest.raises(ValueError, match="overruns|do not tile"):
        layout.parse_upper_link_records(buf, len(lens), 0, buf.size)


# --- the save() invariant the framing guard enforces --------------------------

def test_valid_link_list_sizes_matches_save():
    """save() writes 0 or size_links_per_element_ * element_levels_[i], levels <= maxlevel_."""
    valid = layout.valid_link_list_sizes(SLPE, MAXLEVEL)
    assert valid == {0, 68, 136, 204, 272, 340}


def test_no_single_bit_flip_maps_one_valid_length_onto_another():
    """Why the bound alone is a COMPLETE single-bit detector on this geometry.

    If some valid length were one bit away from another valid length, the bound would silently
    accept that flip (and the stream would still desync). None is — so a bound check catches
    100% of single-bit length corruption here. This is a property of the geometry, not a
    guarantee for all (slpe, maxlevel), which is why the framing guard also checks stream
    health as a backstop.
    """
    valid = layout.valid_link_list_sizes(SLPE, MAXLEVEL)
    collisions = [(v, b) for v in valid for b in range(32) if (v ^ (1 << b)) in valid]
    assert collisions == []


def test_bound_rejects_the_two_measured_collapse_modes():
    valid = layout.valid_link_list_sizes(SLPE, MAXLEVEL)
    assert (0 ^ 1) not in valid                    # misalign mode: len 0 -> 1
    assert (0 ^ (1 << 30)) not in valid            # overcommit mode: len 0 -> 1 GB


# --- against the real index (skipped without the x86 build) -------------------

def _have_real_index():
    return os.path.isfile(adapter.INDEX_PATH)


@pytest.mark.skipif(not _have_real_index(), reason="real SIFT b=7 index not present")
def test_real_index_length_stream_tiles_the_region_exactly():
    """THE oracle, on the bytes E8 actually flips. If this fails, every E8 number is void."""
    hdr = adapter.read_header()
    rmap = adapter.region_map()
    up = next(r for r in rmap["regions"] if r["name"] == "upper_links")
    buf = adapter.serialize_index()
    recs = layout.parse_upper_link_records(buf, hdr["cur_element_count"],
                                           up["byte_start"], up["byte_len"])
    assert recs["consumed"] == up["byte_len"]
    assert recs["framing_bytes"] == 4 * hdr["cur_element_count"]
    # Every length the index actually contains must satisfy save()'s invariant, or the framing
    # guard's bound would reject a legitimate index (a false positive would break the fix).
    valid = layout.valid_link_list_sizes(hdr["size_links_per_element"], hdr["maxlevel"])
    assert set(recs["lens"]) <= valid


# --- the E8 driver's pure pieces (stub geometry; no binaries, no search) -------

def _stub_geom():
    from qp.rabitq import stub_adapter as stub
    return e8.framing_geometry(stub.serialize_index(), stub.region_map()), stub


def test_framing_geometry_accounting_is_self_consistent():
    geom, _ = _stub_geom()
    assert geom["framing_bytes"] + geom["payload_bytes"] == geom["upper_len"]
    assert geom["framing_bytes"] == 4 * geom["n_elements"]
    assert len(geom["len_offsets"]) == geom["n_elements"]


def test_framing_strata_only_ever_sample_length_bytes():
    """The sampler and the membership test are independent; they must agree on every draw."""
    geom, stub = _stub_geom()
    rmap = stub.region_map()
    for stratum in e8.FRAMING_STRATA:
        for i in range(60):
            rng = np.random.default_rng(e8.cell_seed(1234, stratum, i))
            pos, bit = e8.sample_position(stratum, rng, geom, rmap)
            element, lane = e8.classify_len_word(pos, geom)
            assert element is not None, f"{stratum}[{i}] byte {pos} is not a length word"
            assert 0 <= bit < 8
            if stratum == "len_byte0":
                assert lane == 0
            if stratum == "len_byte3":
                assert lane == 3


def test_payload_stratum_never_samples_a_length_byte():
    geom, stub = _stub_geom()
    rmap = stub.region_map()
    for i in range(120):
        rng = np.random.default_rng(e8.cell_seed(1234, "upper_payload", i))
        pos, _ = e8.sample_position("upper_payload", rng, geom, rmap)
        assert e8.classify_len_word(pos, geom) == (None, None), f"payload draw {pos} hit framing"
        assert geom["upper_start"] <= pos < geom["upper_start"] + geom["upper_len"]


def test_control_strata_land_in_their_regions():
    geom, stub = _stub_geom()
    rmap = stub.region_map()
    for i in range(30):
        rng = np.random.default_rng(e8.cell_seed(1234, "rotation", i))
        pos, _ = e8.sample_position("rotation", rng, geom, rmap)
        assert geom["rotation_start"] <= pos < geom["rotation_start"] + geom["rotation_bytes"]
    ex0, exlen = layout.element_field_range(rmap, "ex_code", 0)
    spe = rmap["header"]["size_data_per_element"]
    for i in range(30):
        rng = np.random.default_rng(e8.cell_seed(1234, "ex_code", i))
        pos, _ = e8.sample_position("ex_code", rng, geom, rmap)
        assert 0 <= (pos - ex0) % spe < exlen


def test_every_single_bit_flip_of_a_length_word_changes_the_consumed_size():
    """Why the collapse rate is 100% by construction: no flip leaves (4 + len) unchanged."""
    geom, _ = _stub_geom()
    for e in range(min(16, geom["n_elements"])):
        clean = int(geom["lens"][e])
        for b in range(32):
            assert (clean ^ (1 << b)) != clean


def test_sampling_is_deterministic_in_the_seed():
    geom, stub = _stub_geom()
    rmap = stub.region_map()
    for stratum in e8.STRATA:
        a = e8.sample_position(stratum, np.random.default_rng(e8.cell_seed(7, stratum, 3)),
                               geom, rmap)
        b = e8.sample_position(stratum, np.random.default_rng(e8.cell_seed(7, stratum, 3)),
                               geom, rmap)
        assert a == b


def test_load_walk_is_clean_on_the_pristine_buffer():
    """The emulation must agree with reality on an UNcorrupted index: all records, no truncation."""
    geom, stub = _stub_geom()
    walk = e8.simulate_load_walk(stub.serialize_index(), geom)
    assert walk["records_walked"] == geom["n_elements"]
    assert walk["consumed"] == geom["upper_len"]
    assert walk["truncated"] is False
    assert walk["desynced"] is False


def test_load_walk_detects_the_desync_from_a_single_length_flip():
    geom, stub = _stub_geom()
    rmap = stub.region_map()
    for stratum in e8.FRAMING_STRATA:
        for i in range(10):
            buf = stub.serialize_index()
            rng = np.random.default_rng(e8.cell_seed(1234, stratum, i))
            pos, bit = e8.sample_position(stratum, rng, geom, rmap)
            flip_bit(buf, pos, bit)
            assert e8.simulate_load_walk(buf, geom)["desynced"] is True, \
                f"{stratum}[{i}] flip at {pos}:{bit} did not desync the stream"


def test_load_walk_is_unmoved_by_a_payload_flip():
    """Payload bytes are not framing: corrupting one must not change the record boundaries."""
    geom, stub = _stub_geom()
    rmap = stub.region_map()
    for i in range(10):
        buf = stub.serialize_index()
        rng = np.random.default_rng(e8.cell_seed(1234, "upper_payload", i))
        pos, bit = e8.sample_position("upper_payload", rng, geom, rmap)
        flip_bit(buf, pos, bit)
        walk = e8.simulate_load_walk(buf, geom)
        assert walk["records_walked"] == geom["n_elements"]
        assert walk["desynced"] is False


def test_bound_coverage_is_exhaustive_not_sampled():
    cov = e8.bound_coverage(layout.valid_link_list_sizes(SLPE, MAXLEVEL))
    assert cov["cases_enumerated"] == 6 * 32
    assert cov["undetected_cases"] == []
    assert cov["detected_fraction"] == 1.0


def test_outcome_classification_boundaries():
    clean = 0.95
    assert e8.classify_outcome(None, clean, None, True) == "crash"
    assert e8.classify_outcome(2e-05, clean, None, False) == "silent_collapse"
    assert e8.classify_outcome(clean, clean, None, False) == "benign"
    # below the harmful bar but well above the 50%-retention collapse bar
    assert e8.classify_outcome(clean - 0.05, clean, None, False) == "silent_degraded"


def test_stub_sandbox_gate_blocks_the_real_artifacts_tree():
    real = os.path.join(config.ROOT, "artifacts", "phase3", "e8")
    assert e8.stub_sandbox_ok("stub", real) is False
    assert e8.stub_sandbox_ok("real", real) is True
    assert e8.stub_sandbox_ok("stub", os.path.join(config.ROOT, "artifacts_smoke", "x")) is True


def test_smoke_run_end_to_end_against_the_stub(tmp_path):
    """Full driver plumbing (sweep -> csv -> summary -> gates) with no binaries."""
    out = str(tmp_path)
    summary = e8.main(["--adapter", "stub", "--smoke", "--seeds", "2", "--out", out])
    with open(os.path.join(out, "e8_summary.json")) as fh:
        doc = json.load(fh)
    assert doc["adapter"] == "stub"
    assert doc["gates"]["framing_sampler_always_hits_a_length_word"] is True
    assert doc["gates"]["payload_sampler_never_hits_a_length_word"] is True
    assert doc["gates"]["every_length_flip_changed_the_value"] is True
    assert doc["gates"]["row_count_ok"] is True
    assert os.path.isfile(os.path.join(out, "raw", "e8.done"))
    with open(os.path.join(out, "e8_results.csv")) as fh:
        assert len(list(csv.DictReader(fh))) == len(e8.STRATA) * 2
    # the area accounting must be derived, not hardcoded
    ex = doc["area_exposure"]
    assert ex["framing_bytes"] + ex["payload_bytes"] == ex["upper_links_bytes"]
