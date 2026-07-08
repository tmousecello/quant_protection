"""Verifications for the source-derived RaBitQ byte layout (qp.rabitq.layout).

No binaries needed: the layout is pure arithmetic read from RaBitQ-Library's save() and
data_layout.hpp at the pinned commit. These tests pin the offsets so a future library bump
that shifts the layout is caught loudly (it would invalidate every Phase 3 flip target).
"""
import struct

import pytest

from qp.rabitq import layout


def test_header_is_156_bytes():
    assert layout.HEADER_BYTES == 156


def test_sift_b7_sizes():
    pd, ex_bits, maxM0 = 128, 6, 32       # SIFT dim128->pad128, b=7->ex_bits6, M=16->maxM0=32
    assert layout.bin_data_bytes(pd) == 28          # 16 (1-bit code) + 12 (factors)
    assert layout.ex_data_bytes(pd, ex_bits) == 104  # 96 (ex code) + 8 (factors)
    assert layout.rotation_bytes(pd) == 64           # 4*128/8 (FhtKacRotator flip_)
    assert layout.size_links_level0(maxM0) == 132    # 32*4 + 4
    assert layout.ex_data_bytes(pd, 0) == 0          # b=1 -> no ex block


def test_element_regions_contiguous_and_total():
    pd, ex_bits, maxM0 = 128, 6, 32
    regs = layout.element_regions(pd, ex_bits, maxM0)
    # regions must tile the element block with no gaps/overlaps
    cursor = 0
    for r in regs:
        assert r["offset_in_element"] == cursor, f"gap/overlap before {r['name']}"
        cursor += r["byte_len"]
    # total == size_data_per_element = offsetExData + size_ex_data
    expected = (layout.size_links_level0(maxM0) + layout.PID + layout.PID
                + layout.bin_data_bytes(pd) + layout.ex_data_bytes(pd, ex_bits))
    assert cursor == expected
    # the per-vector error bound lives in bin_factors (critical_global)
    bf = next(r for r in regs if r["name"] == "bin_factors")
    assert bf["protect"] == "critical_global"


def test_ex_factors_protect_matches_bin_factors():
    # ex_factors is a tiny per-vector decode scale (f_add_ex/f_rescale_ex), the same
    # criticality class as bin_factors — not bulk.
    regs = layout.element_regions(128, 6, 32)
    bf = next(r for r in regs if r["name"] == "bin_factors")
    ef = next(r for r in regs if r["name"] == "ex_factors")
    assert ef["protect"] == bf["protect"] == "critical_global"


def test_element_regions_no_ex_block_when_ex_bits_zero():
    # b=1 index: ex_bits=0 -> NO ex_code/ex_factors (mirror ex_data_bytes==0). The element
    # block must total exactly size_links_level0 + 2*PID + bin_data_bytes with no 8B overshoot.
    pd, maxM0 = 128, 32
    regs = layout.element_regions(pd, 0, maxM0)
    names = [r["name"] for r in regs]
    assert "ex_code" not in names and "ex_factors" not in names
    cursor = 0
    for r in regs:
        assert r["offset_in_element"] == cursor, f"gap/overlap before {r['name']}"
        cursor += r["byte_len"]
    expected = (layout.size_links_level0(maxM0) + 2 * layout.PID
                + layout.bin_data_bytes(pd) + layout.ex_data_bytes(pd, 0))
    assert cursor == expected == 168


def test_element_regions_honors_offset_overrides():
    # The authoritative header offsetBinData/offsetExData (real index w/ padding) win over the
    # source-derived formula when supplied.
    regs = layout.element_regions(128, 6, 32, off_bin=200, off_ex=300)
    bc = next(r for r in regs if r["name"] == "bin_code")
    ec = next(r for r in regs if r["name"] == "ex_code")
    assert bc["offset_in_element"] == 200
    assert ec["offset_in_element"] == 300


def test_serialized_region_map_uses_header_offsets():
    # When parse_header supplies offsetBinData/offsetExData, serialized_region_map locates the
    # per-element sub-regions from them, not the formula.
    header = {
        "padded_dim": 128, "ex_bits": 6, "num_cluster": 16, "cur_element_count": 10,
        "maxM0": 32, "offsetBinData": 200, "offsetExData": 300,
        "size_data_per_element": 400,
    }
    centroids = 16 * 128 * layout.FLOAT
    file_size = layout.HEADER_BYTES + centroids + 10 * 400 + layout.rotation_bytes(128)
    rmap = layout.serialized_region_map(header, file_size=file_size)
    level0_start = layout.HEADER_BYTES + centroids
    bc = next(r for r in rmap["regions"] if r["name"] == "elem0.bin_code")
    assert bc["byte_start"] == level0_start + 200


def test_parse_header_roundtrip():
    # build a synthetic 156-byte header (per-field struct code) and read it back
    vals = {name: (i + 1) for i, (name, _) in enumerate(layout.HEADER_FIELDS)}
    blob = b""
    for name, code in layout.HEADER_FIELDS:
        blob += struct.pack(code, vals[name] if code != "<d" else float(vals[name]))
    assert len(blob) == 156
    parsed = layout.parse_header(blob)
    assert parsed["padded_dim"] == vals["padded_dim"]
    assert parsed["num_cluster"] == vals["num_cluster"]
    assert parsed["ex_bits"] == vals["ex_bits"]


def test_parse_header_reads_size_t_as_int_not_double():
    # Regression for the _STRUCT_CODE byte-size collision: a size_t field written by the C++
    # save() as a raw uint64 must parse back as the SAME integer, not be reinterpreted as a
    # double. Under the old {8:"<Q",4:"<I",4:"<i",8:"<d"} -> {8:"<d",4:"<i"} collapse,
    # padded_dim=128 came back as 6.3e-322. We pack a real-format header and assert the
    # integer survives (and the one genuine double field still round-trips).
    blob = b""
    for name, code in layout.HEADER_FIELDS:
        blob += struct.pack(code, 128 if name == "padded_dim"
                            else (1.5 if code == "<d" else 1))
    parsed = layout.parse_header(blob)
    assert parsed["padded_dim"] == 128 and isinstance(parsed["padded_dim"], int)
    assert parsed["mult"] == 1.5                      # the lone <d field still parses as float
    assert isinstance(parsed["maxlevel"], int)        # <i field


def test_serialized_region_map_locates_rotation_at_tail():
    # minimal SIFT-like header; small element count so we can fake a file size
    header = {
        "padded_dim": 128, "ex_bits": 6, "num_cluster": 16, "cur_element_count": 10,
        "maxM0": 32,
        "size_data_per_element": (layout.size_links_level0(32) + 2 * layout.PID
                                  + layout.bin_data_bytes(128) + layout.ex_data_bytes(128, 6)),
    }
    centroids = 16 * 128 * layout.FLOAT
    level0 = 10 * header["size_data_per_element"]
    rot = layout.rotation_bytes(128)
    file_size = layout.HEADER_BYTES + centroids + level0 + 0 + rot   # 0 upper-links
    rmap = layout.serialized_region_map(header, file_size=file_size)
    rotation = next(r for r in rmap["regions"] if r["name"] == "rotation")
    assert rotation["byte_start"] == file_size - rot
    assert rotation["byte_len"] == rot
    assert rotation["protect"] == "critical_global"
    # elem0.bin_factors absolute offset lands inside level0
    bf = next(r for r in rmap["regions"] if r["name"] == "elem0.bin_factors")
    assert layout.HEADER_BYTES + centroids <= bf["byte_start"] < file_size - rot


def test_region_map_requires_file_for_rotation_offset():
    header = {"padded_dim": 128, "ex_bits": 6, "num_cluster": 16, "cur_element_count": 1,
              "maxM0": 32, "size_data_per_element": 272}
    rmap = layout.serialized_region_map(header, file_size=None)
    rotation = next(r for r in rmap["regions"] if r["name"] == "rotation")
    assert rotation["byte_start"] is None       # cannot locate tail without file size
    assert rotation["byte_len"] == 64
