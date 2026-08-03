"""Verifications for the source-derived RaBitQ byte layout (qp.rabitq.layout).

No binaries needed: the layout is pure arithmetic read from RaBitQ-Library's save() and
data_layout.hpp at the pinned commit. These tests pin the offsets so a future library bump
that shifts the layout is caught loudly (it would invalidate every Phase 3 flip target).
"""
import math
import numpy as np
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


# ---------------------------------------------------------------------------
# Header validity predicate
# ---------------------------------------------------------------------------

def _valid_header(n=1000000):
    """A header that satisfies every constraint, built from the same formulas save() used."""
    maxM0 = 32
    h = {
        "max_elements": n, "cur_element_count": n, "dim": 128, "padded_dim": 128,
        "num_cluster": 16, "ex_bits": 6,
        "size_bin_data": layout.bin_data_bytes(128),
        "size_ex_data": layout.ex_data_bytes(128, 6),
        "size_links_level0": layout.size_links_level0(maxM0),
        "label_offset": layout.size_links_level0(maxM0) + layout.PID,
        "offsetBinData": layout.size_links_level0(maxM0) + 2 * layout.PID,
        "size_links_per_element": 16 * layout.PID + layout.PID,
        "maxlevel": 5, "enterpoint_node": n // 2,
        "M": 16, "maxM": 16, "maxM0": maxM0,
        "mult": 1.0 / math.log(16), "ef_construction": 200,
    }
    h["offsetExData"] = h["offsetBinData"] + h["size_bin_data"]
    h["size_data_per_element"] = (h["size_links_level0"] + 2 * layout.PID
                                  + h["size_bin_data"] + h["size_ex_data"])
    return h


def _file_size_for(h, upper_links=None):
    """A file size consistent with `h`; defaults to the minimum save() could produce."""
    if upper_links is None:
        upper_links = layout.LEN_WORD_BYTES * h["cur_element_count"]
    return (layout.HEADER_BYTES + h["num_cluster"] * h["padded_dim"] * layout.FLOAT
            + h["cur_element_count"] * h["size_data_per_element"]
            + upper_links + layout.rotation_bytes(h["padded_dim"]))


def test_validate_header_accepts_a_consistent_header():
    h = _valid_header()
    assert layout.validate_header(h, _file_size_for(h)) == []
    assert layout.header_is_valid(h, _file_size_for(h))


def test_validate_header_accepts_without_a_file_size():
    """file_size gates only the accounting constraint; the rest must stand on their own."""
    h = _valid_header()
    assert layout.validate_header(h, None) == []


@pytest.mark.parametrize("field,value,constraint", [
    ("padded_dim", 192, "padded_dim_is_dim_rounded"),
    ("cur_element_count", 2000000, "count_within_capacity"),
    ("enterpoint_node", 1000000, "enterpoint_in_range"),
    ("maxlevel", -1, "maxlevel_sane"),
    ("maxM0", 33, "maxM_matches_M"),
    ("size_links_level0", 128, "size_links_level0_formula"),
    ("size_links_per_element", 64, "size_links_per_element_formula"),
    ("ex_bits", 99, "ex_bits_in_range"),
    ("size_bin_data", 24, "size_bin_data_formula"),
    ("size_ex_data", 100, "size_ex_data_formula"),
    ("label_offset", 100, "label_offset_formula"),
    ("offsetBinData", 100, "offset_bin_formula"),
    ("offsetExData", 100, "offset_ex_formula"),
    ("mult", 0.5, "mult_matches_M"),
    ("num_cluster", 0, "num_cluster_plausible"),
    ("ef_construction", 0, "ef_construction_plausible"),
])
def test_each_constraint_actually_fires(field, value, constraint):
    """Every constraint must be reachable — one that can never fail is decoration, not a check."""
    h = _valid_header()
    fs = _file_size_for(h)
    h[field] = value
    assert constraint in layout.validate_header(h, fs), \
        f"flipping {field} to {value} did not trip {constraint}"


def test_file_size_accounting_catches_an_overshoot():
    """The strongest single check: the blocks cannot claim more bytes than the file holds."""
    h = _valid_header()
    fs = _file_size_for(h)
    h["cur_element_count"] = h["cur_element_count"] * 2
    h["max_elements"] = h["cur_element_count"]
    assert "file_size_accounts" in layout.validate_header(h, fs)


def test_file_size_accounting_requires_room_for_the_length_words():
    """save() emits one uint32 length per element even when every element has zero upper links,
    so a residual smaller than that is impossible, not merely unusual."""
    h = _valid_header(n=1000)
    short = _file_size_for(h, upper_links=layout.LEN_WORD_BYTES * 1000 - 4)
    assert "file_size_accounts" in layout.validate_header(h, short)
    exact = _file_size_for(h, upper_links=layout.LEN_WORD_BYTES * 1000)
    assert layout.validate_header(h, exact) == []


def test_arithmetic_blowups_count_as_violations():
    """A header that makes the layout arithmetic raise is exactly what this rejects."""
    h = _valid_header()
    h["M"] = 0                                  # 1/log(0) -> ValueError inside mult_matches_M
    bad = layout.validate_header(h, _file_size_for(h))
    assert "mult_matches_M" in bad and "maxM_matches_M" in bad


def test_valid_link_lengths_matches_the_framing_guard_set():
    """On the SIFT b=7 geometry the framing guard reported exactly {0,68,...,340}."""
    h = _valid_header()
    assert layout.valid_link_lengths(h) == {0, 68, 136, 204, 272, 340}


def test_body_check_walks_the_record_chain():
    """A synthetic index whose records are self-consistent passes; a broken length does not."""
    h = _valid_header(n=4)
    n = h["cur_element_count"]
    upper = b""
    for i in range(n):
        length = (i % 3) * h["size_links_per_element"]
        upper += struct.pack("<I", length) + b"\x00" * length
    fs = _file_size_for(h, upper_links=len(upper))
    buf = bytearray(fs)
    buf[layout.first_upper_link_offset(h):layout.first_upper_link_offset(h) + len(upper)] = upper
    buf = np.frombuffer(bytes(buf), dtype=np.uint8)

    assert layout.validate_header_with_body(h, buf, fs) == []
    broken = bytearray(buf.tobytes())
    off = layout.first_upper_link_offset(h)
    struct.pack_into("<I", broken, off, 67)         # not a multiple of size_links_per_element
    assert "upper_link_framing" in layout.validate_header_with_body(
        h, np.frombuffer(bytes(broken), dtype=np.uint8), fs)


def test_full_walk_requires_landing_exactly_on_the_rotation():
    """records=None adds the framing guard's check 4, which is what closes a shifted chain."""
    h = _valid_header(n=4)
    upper = b"".join(struct.pack("<I", 0) for _ in range(4))
    fs = _file_size_for(h, upper_links=len(upper) + 4)      # one length word too many
    buf = bytearray(fs)
    off = layout.first_upper_link_offset(h)
    buf[off:off + len(upper)] = upper
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    assert layout.validate_header_with_body(h, arr, fs, records=4) == []
    assert "upper_link_not_exactly_at_rotation" in layout.validate_header_with_body(
        h, arr, fs, records=None)


def test_full_walk_pins_maxlevel_to_the_longest_record():
    """save() gave the entry point maxlevel levels, so the longest record pins maxlevel."""
    h = _valid_header(n=3)
    slpe = h["size_links_per_element"]
    h["enterpoint_node"] = 1
    upper = (struct.pack("<I", 0)
             + struct.pack("<I", h["maxlevel"] * slpe) + b"\x00" * (h["maxlevel"] * slpe)
             + struct.pack("<I", slpe) + b"\x00" * slpe)
    fs = _file_size_for(h, upper_links=len(upper))
    buf = bytearray(fs)
    buf[layout.first_upper_link_offset(h):layout.first_upper_link_offset(h) + len(upper)] = upper
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    assert layout.validate_header_with_body(h, arr, fs, records=None) == []

    raised = dict(h, maxlevel=h["maxlevel"] + 1)
    assert "maxlevel_exceeds_longest_record" in layout.validate_header_with_body(
        raised, arr, fs, records=None)


def test_full_walk_rejects_an_entry_point_that_is_not_top_level():
    """The flip no field bound can catch: a valid element id that is not at the top level.

    Measured on the real index: exactly one element carries a maxlevel*size_links_per_element
    record, and pointing enterpoint_node at any other one segfaults the stock binary.
    """
    h = _valid_header(n=3)
    slpe = h["size_links_per_element"]
    h["enterpoint_node"] = 1
    upper = (struct.pack("<I", 0)
             + struct.pack("<I", h["maxlevel"] * slpe) + b"\x00" * (h["maxlevel"] * slpe)
             + struct.pack("<I", slpe) + b"\x00" * slpe)
    fs = _file_size_for(h, upper_links=len(upper))
    buf = bytearray(fs)
    buf[layout.first_upper_link_offset(h):layout.first_upper_link_offset(h) + len(upper)] = upper
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)

    for bad_ep in (0, 2):                       # both are valid ids, neither is top level
        moved = dict(h, enterpoint_node=bad_ep)
        assert "enterpoint_is_not_a_top_level_element" in \
            layout.validate_header_with_body(moved, arr, fs, records=None), \
            f"entry point {bad_ep} accepted despite not being a top-level element"
        # the bounded walk cannot see this: it is a whole-file property
        assert layout.validate_header_with_body(moved, arr, fs, records=1) == []
