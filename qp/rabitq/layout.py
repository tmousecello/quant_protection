"""Serialized RaBitQ (HNSW+RaBitQ) index byte-region map — derived from library source.

Every offset/size here is read straight from RaBitQ-Library's `HierarchicalNSW::save()` and
`data_layout.hpp` at the pinned commit (7e39df2; see build_rabitq.sh), NOT guessed. This is
the report-and-stop resolution for "where do rotation / bin / ex / pointers live in the
bytes": they are all accounted for below.

On-disk file layout written by save() (all little-endian native; size_t=8, PID=uint32=4,
int=4, double=8, float=4):

  [HEADER 156 B]  (21 scalar fields, see HEADER_FIELDS)
  [CENTROIDS]     num_cluster * padded_dim * 4               (global IVF centroids)
  [LEVEL0]        cur_element_count * size_data_per_element  (per-vector blocks, verbatim
                  copy of data_level0_memory_ — so in-memory fi_* offsets == on-disk offsets
                  within each element block)
  [UPPER_LINKS]   per element: uint32 len + len bytes        (variable; graph edges, levels>0)
  [ROTATION]      flip_ : 4*padded_dim/8 = padded_dim/2 B    (FhtKacRotator sign-flips; the
                  global, shared decode structure — the RaBitQ analog of SQ8's sq_scale)

Per-element block (offsets relative to the element base inside LEVEL0), from save() offset
math (hnsw.hpp ~L422-432) and data_layout.hpp data_bytes():
  links       [0, size_links_level0)            size_links_level0 = maxM0*4 + 4  (count+edges)
  cluster_id  [size_links_level0, +4)           PID into centroids
  label       [size_links_level0+4, +4)         external id (PID)
  bin_code    [offsetBinData, +padded_dim/8)    1-bit sign code
  bin_factors [offsetBinData+padded_dim/8, +12) f_add/f_rescale/f_error (3*float32; f_error
                                                is the per-vector error bound EB recovery uses)
  ex_code     [offsetExData, +padded_dim*ex_bits/8)  high-precision refinement code
  ex_factors  [offsetExData+ex_code, +8)        f_add_ex/f_rescale_ex (2*float32)

provenance on every region: located="source-derived" (from the pinned library source).
verified_against_index is False here because that requires a built index (x86-64 build);
parse_header() upgrades the size fields to "header-parsed" when given a real index file.
"""
import struct

import numpy as np

# Byte widths used in level0 offset arithmetic (header field widths now live as struct codes
# in HEADER_FIELDS). PID = uint32 element/cluster id; FLOAT = sizeof(float) centroid/factor.
PID = 4
FLOAT = 4

BIN_FACTORS_BYTES = 3 * FLOAT   # f_add, f_rescale, f_error
EX_FACTORS_BYTES = 2 * FLOAT    # f_add_ex, f_rescale_ex

# Ordered header fields exactly as HierarchicalNSW::save() writes them (hnsw.hpp L468-492).
# (name, struct_code) — the little-endian format code is carried PER FIELD so the parser is
# unambiguous. (An earlier byte-size-keyed lookup table silently collapsed: sizeof(size_t)==
# sizeof(double)==8 and sizeof(PID)==sizeof(int)==4, so {8:"<Q",4:"<I",4:"<i",8:"<d"} became
# {8:"<d",4:"<i"} and every size_t parsed as a double.) 17 size_t (<Q) + label_offset (<I) +
# maxlevel (<i) + enterpoint (<I) + mult (<d) = 136 + 4 + 4 + 4 + 8 = 156 bytes.
HEADER_FIELDS = [
    ("max_elements", "<Q"),
    ("cur_element_count", "<Q"),
    ("dim", "<Q"),
    ("padded_dim", "<Q"),
    ("num_cluster", "<Q"),
    ("ex_bits", "<Q"),
    ("size_bin_data", "<Q"),
    ("size_ex_data", "<Q"),
    ("size_links_level0", "<Q"),
    ("offsetBinData", "<Q"),
    ("offsetExData", "<Q"),
    ("label_offset", "<I"),
    ("size_data_per_element", "<Q"),
    ("size_links_per_element", "<Q"),
    ("maxlevel", "<i"),
    ("enterpoint_node", "<I"),
    ("M", "<Q"),
    ("maxM", "<Q"),
    ("maxM0", "<Q"),
    ("mult", "<d"),
    ("ef_construction", "<Q"),
]
HEADER_BYTES = sum(struct.calcsize(code) for _, code in HEADER_FIELDS)   # == 156


def round_up_to_multiple(x, m):
    """padded_dim_ = round_up_to_multiple(dim, 64) for the FhtKacRotator (hnsw.hpp L388)."""
    return ((int(x) + m - 1) // m) * m


# --- size formulas (data_layout.hpp; T=float => sizeof(T)=4) ------------------

def bin_data_bytes(padded_dim):
    """BinDataMap<float>::data_bytes = padded_dim/8 + sizeof(float)*3 (data_layout.hpp L165)."""
    return padded_dim // 8 + BIN_FACTORS_BYTES


def ex_data_bytes(padded_dim, ex_bits):
    """ExDataMap<float>::data_bytes = ex_bits>0 ? padded_dim*ex_bits/8 + sizeof(float)*2 : 0."""
    return (padded_dim * ex_bits // 8 + EX_FACTORS_BYTES) if ex_bits > 0 else 0


def rotation_bytes(padded_dim):
    """FhtKacRotator dump = flip_.size() = 4*padded_dim/8 = padded_dim/2 bytes (rotator.hpp)."""
    return 4 * padded_dim // 8


def size_links_level0(maxM0):
    """maxM0*sizeof(PID) + sizeof(PID): the neighbour count + edge ids (hnsw.hpp L424)."""
    return maxM0 * PID + PID


# --- header parsing ----------------------------------------------------------

def parse_header(source):
    """Parse the 156-byte header from a serialized index (path, bytes, or file-like).

    Returns {field: int_value}. The size fields here are the AUTHORITATIVE on-disk sizes
    (header-parsed), so callers should prefer them over the formula fallbacks when a real
    index exists.
    """
    if hasattr(source, "read"):
        raw = source.read(HEADER_BYTES)
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source[:HEADER_BYTES])
    else:
        with open(source, "rb") as fh:
            raw = fh.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"index too short: need {HEADER_BYTES} header bytes, got {len(raw)}")
    out, off = {}, 0
    for name, code in HEADER_FIELDS:
        out[name] = struct.unpack_from(code, raw, off)[0]
        off += struct.calcsize(code)
    return out


# --- region construction -----------------------------------------------------

def _region(name, kind, byte_start, byte_len, *, protect, semantic, verified=False):
    return {
        "name": name,
        "kind": kind,
        "byte_start": int(byte_start),
        "byte_len": int(byte_len),
        "protect": protect,                  # "critical_global" | "large_bulk" | "pointer"
        "semantic": semantic,
        "located": "source-derived",
        "verified_against_index": bool(verified),
    }


def element_regions(padded_dim, ex_bits, maxM0, off_bin=None, off_ex=None):
    """Per-element sub-regions (offset within one LEVEL0 element block, length, kind).

    Returns a list of dicts with offset_in_element / byte_len / kind. Pure formula from the
    save() offset math — exact, independent of any built index.

    ``off_bin`` / ``off_ex`` override the formula offsets with the AUTHORITATIVE header
    offsetBinData / offsetExData when a real index is available (serialized_region_map passes
    them); left ``None`` they fall back to the source-derived formula — the only path the
    offline tests/golden exercise. A b=1 index has ``ex_bits==0`` and therefore NO ex block
    (mirrors ex_data_bytes); emitting one would overshoot size_data_per_element by 8 bytes and
    push every later offset into the next element.
    """
    links_len = size_links_level0(maxM0)
    if off_bin is None:
        off_bin = links_len + PID + PID            # cluster_id (PID) + label (PID)
    bin_code_len = padded_dim // 8
    if off_ex is None:
        off_ex = off_bin + bin_data_bytes(padded_dim)
    regs = [
        ("links",       "graph_edges", 0,                          links_len,        "pointer"),
        ("cluster_id",  "codes_meta",  links_len,                  PID,              "pointer"),
        ("label",       "codes_meta",  links_len + PID,            PID,              "pointer"),
        ("bin_code",    "codes",       off_bin,                    bin_code_len,     "large_bulk"),
        ("bin_factors", "sq_scale",    off_bin + bin_code_len,     BIN_FACTORS_BYTES,"critical_global"),
    ]
    if ex_bits > 0:                                # no ex block on a b=1 index (ex_data_bytes==0)
        ex_code_len = padded_dim * ex_bits // 8
        # ex_factors is a tiny per-vector decode scale (f_add_ex/f_rescale_ex), same criticality
        # class as bin_factors — not bulk.
        regs.append(("ex_code",    "codes",    off_ex,               ex_code_len,      "large_bulk"))
        regs.append(("ex_factors", "sq_scale", off_ex + ex_code_len, EX_FACTORS_BYTES, "critical_global"))
    return [
        {"name": n, "kind": k, "offset_in_element": int(o), "byte_len": int(L), "protect": p}
        for (n, k, o, L, p) in regs
    ]


def serialized_region_map(header, file_size=None):
    """File-level region map for a serialized index. `header` = parse_header(...) dict.

    Produces the top-level blocks (header, centroids, level0, upper_links, rotation) plus the
    per-element sub-regions for element 0 as absolute file byte ranges. The ROTATION block is
    located from the tail of the file when `file_size` is given (it is the last thing written),
    else flagged variable-offset.
    """
    pd = header["padded_dim"]
    ex_bits = header["ex_bits"]
    nclu = header["num_cluster"]
    n = header["cur_element_count"]
    spe = header["size_data_per_element"]

    centroids_len = nclu * pd * FLOAT
    level0_len = n * spe
    centroids_start = HEADER_BYTES
    level0_start = centroids_start + centroids_len
    upper_start = level0_start + level0_len
    rot_len = rotation_bytes(pd)

    regions = [
        _region("header", "header", 0, HEADER_BYTES,
                protect="critical_global", semantic="index geometry / offsets (crash if wrong)"),
        _region("centroids", "centroid", centroids_start, centroids_len,
                protect="critical_global", semantic="global IVF centroids (shared decode)"),
        _region("level0", "codes", level0_start, level0_len,
                protect="large_bulk", semantic="per-vector links+cluster+bin+ex blocks"),
    ]
    # Per-element sub-regions for element 0 (absolute file offsets), so a fault model can
    # target a specific structure (e.g. bin_factors == the per-vector error bound).
    for er in element_regions(pd, ex_bits, header["maxM0"],
                              off_bin=header.get("offsetBinData") or None,
                              off_ex=header.get("offsetExData") or None):
        regions.append(_region(
            f"elem0.{er['name']}", er["kind"],
            level0_start + er["offset_in_element"], er["byte_len"],
            protect=er["protect"], semantic=f"element 0 {er['name']}"))

    if file_size is not None:
        rot_start = file_size - rot_len
        regions.append(_region("rotation", "sq_scale", rot_start, rot_len,
                               protect="critical_global",
                               semantic="FhtKacRotator sign-flips (global shared rotation; "
                                        "single-point-catastrophe candidate)"))
        regions.append(_region("upper_links", "graph_edges", upper_start, rot_start - upper_start,
                               protect="pointer", semantic="upper-level graph edges (variable)"))
    else:
        # Without the file size the rotation tail offset is unknown (upper_links are variable).
        regions.append({
            "name": "rotation", "kind": "sq_scale", "byte_start": None, "byte_len": rot_len,
            "protect": "critical_global", "located": "source-derived-tail",
            "semantic": "FhtKacRotator sign-flips at EOF; pass file_size to locate",
            "verified_against_index": False,
        })

    return {
        "index": "RABITQ_HNSW",
        "header": header,
        "total_bytes": file_size,
        "regions": regions,
    }


# --- upper-link framing (E8) --------------------------------------------------

LEN_WORD_BYTES = 4      # sizeof(unsigned int) — the per-record length prefix save() writes


def valid_link_list_sizes(size_links_per_element, maxlevel):
    """The complete set of link_list_size values save() can emit (hnsw.hpp L656-657).

        link_list_size = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0

    and element_levels_[i] <= maxlevel_ by construction (maxlevel_ is raised to curlevel for
    every element whose level exceeds it, hnsw.hpp L920/L926). So the value is 0 or a multiple
    of size_links_per_element_ with quotient in [1, maxlevel]. This is the exact invariant the
    framing guard enforces on load — exact, so it cannot reject a legitimate index.
    """
    slpe, ml = int(size_links_per_element), int(maxlevel)
    if slpe <= 0:
        raise ValueError(f"size_links_per_element must be positive, got {slpe}")
    return {0} | {slpe * level for level in range(1, ml + 1)}


def parse_upper_link_records(buf, n_elements, region_start, region_len):
    """Walk the `[uint32 len][len bytes]` x n_elements stream and locate every length word.

    The upper_links block has no index and no padding, so the ONLY way to know where record i
    begins is to add up (4 + len) for every record before it — which is exactly why a single
    corrupt length word desynchronizes everything after it (see phase3_e8_framing.py).

    Returns {"lens", "len_offsets", "consumed", "framing_bytes", "payload_bytes"} where
    `len_offsets[i]` is the ABSOLUTE buffer offset of record i's length word.

    Raises ValueError unless the records tile `region_len` EXACTLY. That is the correctness
    oracle for all the E8 region math: an exact sum is unreachable if any boundary was missed,
    so a caller that gets a result back knows the parse is right.
    """
    buf = np.frombuffer(memoryview(buf), dtype=np.uint8) if not isinstance(buf, np.ndarray) \
        else buf
    start, length, n = int(region_start), int(region_len), int(n_elements)
    if start < 0 or length < 0 or start + length > buf.size:
        raise ValueError(f"upper_links region [{start},{start + length}) outside a "
                         f"{buf.size}-byte buffer")
    raw = buf[start:start + length].tobytes()
    lens, offsets, off = [], [], 0
    for i in range(n):
        if off + LEN_WORD_BYTES > length:
            raise ValueError(
                f"record {i}: length word at +{off} overruns the {length}-byte upper_links "
                f"region (stream desynchronized — the region math or the buffer is wrong)")
        (L,) = struct.unpack_from("<I", raw, off)
        offsets.append(start + off)
        lens.append(int(L))
        off += LEN_WORD_BYTES + int(L)
        if off > length:
            raise ValueError(
                f"record {i}: len={L} overruns the {length}-byte upper_links region at +{off}")
    if off != length:
        raise ValueError(
            f"upper-link records do not tile the region: consumed {off} B of {length} B over "
            f"{n} records (drift {off - length:+d}). The parse is wrong; STOP.")
    return {
        "lens": lens,
        "len_offsets": offsets,
        "consumed": off,
        "framing_bytes": LEN_WORD_BYTES * n,
        "payload_bytes": sum(lens),
    }


# --- per-element addressing + cost aggregation (Stage 1) ----------------------

# Per-vector kinds (one block PER element) vs global kinds (one copy for the whole index). Used
# both to address element e>0 and to scale per-vector structures by cur_element_count for cost.
PER_VECTOR_FIELDS = ("links", "cluster_id", "label",
                     "bin_code", "bin_factors", "ex_code", "ex_factors")
GLOBAL_FIELDS = ("header", "centroids", "rotation")


def base_name(name):
    """Strip the 'elem0.' element-0 prefix the region map uses for per-vector regions.

    Single source of truth for the prefix minted by serialized_region_map; bitclass and the
    stub adapter route through this so the delimiter lives in exactly one place.
    """
    return name.split(".", 1)[1] if name.startswith("elem0.") else name


def element_field_range(rmap, field, e):
    """Absolute (byte_start, byte_len) of per-vector `field` for element index `e`.

    The region map (serialized_region_map) exposes per-element sub-regions for ELEMENT 0 only
    (`elem0.<field>`). Element e's block is exactly e*size_data_per_element bytes later, so we add
    that to the already-correct elem0 byte_start rather than recomputing level0_start by hand
    (which would risk double-adding the 156-B header or the centroids block). Raises if `field`
    is absent (e.g. ex_* on a b=1 index) or `e` strays outside level0.
    """
    name = f"elem0.{field}"
    r = next((x for x in rmap["regions"] if x["name"] == name), None)
    if r is None:
        raise KeyError(f"no per-element region {name!r} in map (fields: "
                       f"{[x['name'] for x in rmap['regions'] if x['name'].startswith('elem0.')]})")
    hdr = rmap["header"]
    n = hdr["cur_element_count"]
    spe = hdr["size_data_per_element"]
    if not (0 <= int(e) < n):
        raise IndexError(f"element {e} out of range [0,{n})")
    byte_start = r["byte_start"] + int(e) * spe
    level0 = next(x for x in rmap["regions"] if x["name"] == "level0")
    level0_end = level0["byte_start"] + level0["byte_len"]
    if byte_start + r["byte_len"] > level0_end:
        raise IndexError(f"element {e} {field} [{byte_start},{byte_start + r['byte_len']}) "
                         f"overruns level0 end {level0_end}")
    return byte_start, r["byte_len"]


def aggregate_region_map(rmap):
    """Region map for COST accounting: per-vector structures priced for ALL elements.

    serialized_region_map's `elem0.<field>` is a single element's slice, so feeding it to
    phase3_cost.mem_cost prices one of ~10^6 vectors (Stage 0 known-limitation). Here each
    per-vector field's byte_len is multiplied by cur_element_count; global structures
    (header/centroids/rotation) are kept at face value. The output keys match what mem_cost reads
    ({'regions':[{name,byte_len}...]}), so phase3_cost stays untouched.
    """
    n = rmap["header"]["cur_element_count"]
    out = []
    for r in rmap["regions"]:
        bn = r["name"]
        base = base_name(bn)
        if base in PER_VECTOR_FIELDS and bn.startswith("elem0."):
            out.append({"name": base, "kind": r["kind"],
                        "byte_len": int(r["byte_len"]) * int(n),
                        "n_elements": int(n), "per_vector": True})
        elif bn in GLOBAL_FIELDS:
            out.append({"name": bn, "kind": r["kind"], "byte_len": int(r["byte_len"]),
                        "n_elements": 1, "per_vector": False})
        # level0 / upper_links are containers already covered by their per-element fields; skip.
    return {"index": rmap["index"], "header": rmap["header"], "regions": out}
