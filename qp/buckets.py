"""Region-bucket map + within-element bit-position tagging — shared by Phase 1 and 2.

A uniformly random physical bit flip does not pick semantics, so the first question for
any index is "what fraction of its bytes is even recall-relevant?". This module is the
single source of truth that answers it, plus the helper that labels *which bit of an
element* a flip hit (so the fp32 sign/exponent/mantissa bimodal and the SQ8 MSB->LSB
gradient are not hidden by uniform averaging). It is pure (no I/O) and is imported by
`phase1_region_accounting.py`, `phase1_sensitivity.py`, and the future Phase 2 sweep.

Buckets (grounded in the Phase 0 dry-run: code/centroid/scale flips deserialize cleanly,
header/graph-structure flips crash):
  recall_relevant  flips deserialize and can silently degrade recall -> the study target.
  crash_structure  flips crash on load or are structurally detected.
  benign           padding / irrelevant (reserved; regions.py emits none today).
"""

# kind -> bucket. Unknown kinds raise (nothing is silently misbucketed).
BUCKET_OF_KIND = {
    "vectors": "recall_relevant",       # fp32 raw vectors (FLAT, HNSW storage)
    "codes": "recall_relevant",         # IVF inverted-list codes (fp32 for IVF_FLAT, uint8 for SQ8)
    "centroid": "recall_relevant",      # IVF coarse-quantizer centroids
    "sq_scale": "recall_relevant",      # SQ vmin/vdiff per dim
    "pq_codebook": "recall_relevant",   # PQ sub-quantizer codebook (accounting only this round)
    "header": "crash_structure",        # fourcc magic + index params
    "graph_meta": "crash_structure",    # HNSW levels/offsets/entry_point
    "graph_edges": "crash_structure",   # HNSW neighbor adjacency (int32 ids -> OOB crash)
}

# kind -> sampling class for the single-bit sweep.
#   large           sample S bit positions.
#   medium_critical high-leverage but too big to exhaust (centroid = 512KB) -> larger sample.
#   small_critical  small enough to enumerate every bit (sq_scale = 8192 bits).
REGION_CLASS_OF_KIND = {
    "vectors": "large",
    "codes": "large",
    "centroid": "medium_critical",
    "sq_scale": "small_critical",
    "pq_codebook": "small_critical",
}

# Documented decision thresholds (prompt.txt B6: "先取絕對 >0.01").
CATASTROPHIC_ABS = 0.01   # a flip is catastrophic if dRecall@10 > this
BENIGN_ABS = 1e-4         # a flip is benign if |dRecall@10| <= this

# fp32 within-element bit-position tags, ordered MSB-effect -> LSB-effect for plotting.
FP32_TAGS = ["sign", "exponent", "mantissa-high", "mantissa-low"]
# SQ8 within-byte tags, ordered MSB -> LSB.
SQ8_TAGS = [f"sq8_b{b}" for b in (7, 6, 5, 4, 3, 2, 1, 0)]


def bucket_of(region):
    """Return the bucket for one region dict. Raises on an unmapped kind."""
    kind = region["kind"]
    if kind not in BUCKET_OF_KIND:
        raise KeyError(f"unmapped region kind {kind!r} — add it to BUCKET_OF_KIND")
    return BUCKET_OF_KIND[kind]


def region_class_of(region):
    """Sampling class for a region, or None if not in a sampled (recall_relevant) class."""
    return REGION_CLASS_OF_KIND.get(region["kind"])


def region_bits(region):
    """Total flippable bits in a region."""
    return int(region["byte_len"]) * 8


def element_dtype_of(index_name, region):
    """Logical element dtype for bit tagging — corrects the regions.json label mismatch.

    IVF_FLAT `codes` is stored as raw fp32 vectors yet labeled uint8 in regions.json, so it
    must be tagged as float32 (4-byte stride) for the fp32-vs-int8 contrast to be honest.
    NOTE (documented approximation): IVF code blocks interleave int64 list ids between the
    fp32 vectors, so the 4-byte stride is only approximately aligned across the whole region.
    All other `codes` are genuine SQ8 (uint8). centroid / sq_scale / pq_codebook / vectors
    are fp32.
    """
    kind = region["kind"]
    if kind in ("centroid", "sq_scale", "pq_codebook", "vectors"):
        return "float32"
    if kind == "codes":
        return "float32" if index_name == "IVF_FLAT" else "uint8"
    # crash_structure kinds are not sampled for recall; fall back to raw bytes.
    return "uint8"


def bit_position_tag(element_dtype, off_in_region, bit):
    """Label which bit of an element a flip hit.

    off_in_region = byte_pos - region["byte_start"]; bit in 0..7 (0 = LSB of that byte).
    Little-endian IEEE-754 float32: byte 0 = mantissa-low ... byte 3 = sign+exponent.
      byte 3: bit 7 -> sign, bits 6..0 -> exponent (top 7 exponent bits)
      byte 2: bit 7 -> exponent (LSB of exponent), bits 6..0 -> mantissa-high
      byte 1: -> mantissa-high
      byte 0: -> mantissa-low
    uint8 (SQ8): tag by bit index, bit 7 = MSB.
    """
    if element_dtype == "float32":
        within = off_in_region % 4
        if within == 3:
            return "sign" if bit == 7 else "exponent"
        if within == 2:
            return "exponent" if bit == 7 else "mantissa-high"
        if within == 1:
            return "mantissa-high"
        return "mantissa-low"
    # uint8 / raw byte
    return f"sq8_b{bit}"


def tag_order(element_dtype):
    """Canonical tag ordering for an element dtype (for stable plotting/sorting)."""
    return FP32_TAGS if element_dtype == "float32" else SQ8_TAGS
