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


def test_parse_header_roundtrip():
    # build a synthetic 156-byte header and read it back
    vals = {name: (i + 1) for i, (name, _) in enumerate(layout.HEADER_FIELDS)}
    blob = b""
    for name, sz in layout.HEADER_FIELDS:
        code = layout._STRUCT_CODE[sz]
        blob += struct.pack(code, vals[name] if code != "<d" else float(vals[name]))
    assert len(blob) == 156
    parsed = layout.parse_header(blob)
    assert parsed["padded_dim"] == vals["padded_dim"]
    assert parsed["num_cluster"] == vals["num_cluster"]
    assert parsed["ex_bits"] == vals["ex_bits"]


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
