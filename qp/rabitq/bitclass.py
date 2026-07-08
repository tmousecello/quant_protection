"""Per-structure bit-class tagging for the RaBitQ vulnerability sweep (Phase 3 Stage 1).

Phase 1/2 (FAISS) tagged within-element bit positions with `qp.buckets.bit_position_tag`, which
knows only IEEE-754 float32 and raw uint8. RaBitQ's serialized structures are *not* uniformly
float32 (the brief: "bit-class 分類從 layout 推導,不要假設 float32"), so this module derives the
bit-class of each structure from the library source it was read off (pinned commit 7e39df2):

  rotation     `FhtKacRotator.flip_` is a flat sign-flip bit-vector applied in FOUR sequential
               stages (rotator.hpp rotate(): flip_.data() + s*(padded_dim/8), s=0..3). Each stage
               is padded_dim/8 bytes (16 B = 128 bits for SIFT). So a flip's class is its stage
               index `rot_stage{0..3}` — derived, unambiguous. No exponent/mantissa (sign bits).
  bin_factors  3*float32 {f_add, f_rescale, f_error} after the bin code (data_layout.hpp
               BinDataMap). Field = off//4; within-field = float32 sign/exp/mantissa. f_error is
               the per-vector error bound EB recovery uses -> tagged distinctly.
  ex_factors   2*float32 {f_add_ex, f_rescale_ex} (data_layout.hpp ExDataMap; NO f_error here).
  bin_code     1 sign bit per dim -> a single flat class `bin_sign` (every bit is one dim's sign).
  ex_code      ex_bits per dim, but packed by SIMD bit-plane packers (pack_excode.hpp
               packing_{1..}bit_excode) that interleave bit-planes across non-contiguous bytes in
               16/64-dim blocks. A clean contiguous high-vs-low byte split DOES NOT EXIST, so we
               tag it as a single flat class `ex_code` and REPORT-AND-STOP for finer resolution
               (see EX_CODE_REPORT). Do not hard-code a guessed high/low split.
  pointers     links / cluster_id / label are little-endian uint32 indices: byte lane off%4 in
               {2,3} = high (big jump / OOB) -> `ptr_high`; {0,1} = low (small perturbation) ->
               `ptr_low`.
  centroids    global fp32 IVF centroids -> float32 tagging, prefixed `centroid:`.
  header       index geometry; raw bytes -> `header_byte` (crash-probe structure, not recall).

The float32 within-field tagging REUSES qp.buckets.bit_position_tag (the single source of truth);
this module only routes each RaBitQ structure to the right scheme.
"""
from qp import buckets
from qp.rabitq.layout import base_name

FLOAT = 4

# Report-and-stop note surfaced by the runner when it characterizes ex_code: the within-bit
# (high vs low extension bit) ordering is SIMD-packer-specific and not a contiguous byte split.
EX_CODE_REPORT = (
    "ex_code high-vs-low extension-bit resolution is REPORT-AND-STOP: pack_excode.hpp packs "
    "ex_bits as bit-planes interleaved across non-contiguous bytes within 16/64-dim SIMD blocks "
    "(distinct packer per ex_bits), so there is no contiguous high/low byte region. Treated as a "
    "single flat class 'ex_code'. Finer resolution needs decoding packing_{ex_bits}bit_excode."
)

# float fields per factor structure, in serialized order (data_layout.hpp).
FACTOR_FIELDS = {
    "bin_factors": ["f_add", "f_rescale", "f_error"],   # f_error = per-vector EB
    "ex_factors": ["f_add_ex", "f_rescale_ex"],
}

# How each structure is sampled in the single-bit sweep (mirrors buckets.REGION_CLASS_OF_KIND):
#   exhaustive       small + global -> enumerate every bit (rotation, like sq_scale).
#   per_vector       per-element structure -> sample N elements x bit positions, bootstrap CI.
#   global_sampled   large global structure -> sample (centroids).
#   crash_probe      not recall-relevant; probe for crashes only (header).
SAMPLING_CLASS = {
    "rotation": "exhaustive",
    "bin_factors": "per_vector",
    "ex_factors": "per_vector",
    "bin_code": "per_vector",
    "ex_code": "per_vector",
    "links": "per_vector",
    "cluster_id": "per_vector",
    "label": "per_vector",
    "centroids": "global_sampled",
    "header": "crash_probe",
}

def rotation_stage(off_in_region, padded_dim):
    """Which of the 4 FhtKac sign-flip stages a rotation byte belongs to (rotator.hpp)."""
    stage_bytes = padded_dim // 8                     # 16 B = 128 bits for SIFT
    return int(off_in_region) // stage_bytes


def bit_class(region_name, off_in_region, bit, *, padded_dim):
    """Bit-class tag for a flip at byte offset `off_in_region` (from the region start), bit 0..7.

    Routes by structure to the scheme derived from the library source (see module docstring).
    Raises on an unknown structure so nothing is silently mislabeled.
    """
    name = base_name(region_name)
    off = int(off_in_region)

    if name == "rotation":
        return f"rot_stage{rotation_stage(off, padded_dim)}"

    if name in FACTOR_FIELDS:
        fields = FACTOR_FIELDS[name]
        field = fields[(off // FLOAT) % len(fields)]
        fp = buckets.bit_position_tag("float32", off, bit)   # sign / exponent / mantissa-*
        return f"{field}:{fp}"

    if name == "bin_code":
        return "bin_sign"

    if name == "ex_code":
        return "ex_code"                              # flat; see EX_CODE_REPORT

    if name in ("links", "cluster_id", "label"):
        lane = off % FLOAT                            # little-endian uint32 byte lane
        return "ptr_high" if lane >= 2 else "ptr_low"

    if name == "centroids":
        return f"centroid:{buckets.bit_position_tag('float32', off, bit)}"

    if name == "header":
        return "header_byte"

    raise KeyError(f"no RaBitQ bit-class for structure {name!r} (region {region_name!r})")


def sampling_class(region_name):
    """Sampling strategy for a structure, or None if it is not characterized."""
    return SAMPLING_CLASS.get(base_name(region_name))
