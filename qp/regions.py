"""Byte-region map of a serialized index — the targeting substrate for Phase 1/2.

`faiss.serialize_index` returns a contiguous uint8 buffer. We locate each high-value
structure by searching the buffer for the structure's known content, which we can read
out of the live index object (centroids via reconstruct_n, SQ scales / PQ codebook via
faiss.vector_to_array, HNSW edges via vector_to_array(neighbors), flat storage = the base
vectors). Content search is robust to layout/version because it matches the bytes
themselves; we assert each match is unique.

Some structures are not contiguously locatable by content — IVF inverted-list codes are
reordered by list assignment, and small graph metadata (entry point / levels / offsets)
is not uniquely findable. For those we emit a DOCUMENTED byte-range estimate (the tail
after the located front blocks for IVF codes; the gap before the edge block for graph
metadata). prompt.txt explicitly permits this region map to be "a documented approach,
not fully solved".

Each region: {name, kind, byte_start, byte_len, dtype, semantic, located}.
  located=True  -> exact, content-matched.
  located=False -> documented estimate.
kind in {header, vectors, codes, centroid, sq_scale, pq_codebook, graph_edges, graph_meta}.
"""
import faiss
import numpy as np

from .flip import to_buffer


def _find_unique(bts, content_bytes):
    """Return start offset of content_bytes in bts, asserting a unique match. -1 if absent."""
    if len(content_bytes) == 0:
        return -1
    start = bts.find(content_bytes)
    if start < 0:
        return -1
    # Uniqueness check on a prefix keeps the scan cheap for huge blocks.
    probe = content_bytes[:64] if len(content_bytes) > 64 else content_bytes
    if bts.count(probe) != 1:
        # Ambiguous prefix: fall back to confirming the full block is unique.
        if bts.count(content_bytes) != 1:
            raise ValueError("non-unique block match; cannot place region safely")
    return start


def _region(name, kind, start, length, dtype, semantic, located=True):
    return {
        "name": name, "kind": kind,
        "byte_start": int(start), "byte_len": int(length),
        "dtype": dtype, "semantic": semantic, "located": bool(located),
    }


def _locate(bts, content, name, kind, dtype, semantic, out):
    """Append an exact content-matched region if found; return its end offset or None."""
    cb = np.ascontiguousarray(content).tobytes()
    start = _find_unique(bts, cb)
    if start < 0:
        return None
    out.append(_region(name, kind, start, len(cb), dtype, semantic, located=True))
    return start + len(cb)


def build_region_map(name, spec, index, xb):
    """Build the region list for one index. xb = base vectors added to the index."""
    buf = to_buffer(index)
    bts = buf.tobytes()
    total = len(buf)
    regions = []

    if name == "FLAT":
        _locate(bts, xb.astype("float32"), "vectors", "vectors", "float32",
                "fp32 base vectors (exact storage)", regions)

    elif spec["kind"] == "ivf":
        cent = index.quantizer.reconstruct_n(0, index.nlist).astype("float32")
        front_end = _locate(bts, cent, "centroid", "centroid", "float32",
                            "IVF coarse-quantizer centroids", regions) or 0
        if spec["quant"] == "sq8":
            sq = faiss.vector_to_array(index.sq.trained).astype("float32")
            e = _locate(bts, sq, "sq_scale", "sq_scale", "float32",
                        "scalar-quantizer vmin/vdiff per dim", regions)
            front_end = max(front_end, e or 0)
        elif spec["quant"] == "pq":
            pqc = faiss.vector_to_array(index.pq.centroids).astype("float32")
            e = _locate(bts, pqc, "pq_codebook", "pq_codebook", "float32",
                        "PQ sub-quantizer codebook", regions)
            front_end = max(front_end, e or 0)
        # IVF codes are reordered per list -> documented tail estimate.
        if front_end < total:
            regions.append(_region(
                "codes", "codes", front_end, total - front_end, "uint8",
                "inverted-list codes (estimated: tail after located metadata)",
                located=False))

    elif spec["kind"] == "graph":
        nb = faiss.vector_to_array(index.hnsw.neighbors).astype("int32")
        edge_start = _find_unique(bts, nb.tobytes())
        # graph metadata (levels/offsets/entry_point/params) lives before the edge block.
        if edge_start > 0:
            regions.append(_region(
                "graph_meta", "graph_meta", 0, edge_start, "mixed",
                f"HNSW params/levels/offsets, entry_point={int(index.hnsw.entry_point)}, "
                f"max_level={int(index.hnsw.max_level)} (estimated: header..edges)",
                located=False))
            regions.append(_region(
                "graph_edges", "graph_edges", edge_start, nb.nbytes, "int32",
                "HNSW neighbor adjacency table", located=True))
        # storage vectors
        if spec["quant"] is None:
            _locate(bts, xb.astype("float32"), "vectors", "vectors", "float32",
                    "fp32 storage vectors (HNSW flat storage)", regions)
        elif spec["quant"] == "sq8":
            st = faiss.downcast_index(index.storage)
            sq = faiss.vector_to_array(st.sq.trained).astype("float32")
            _locate(bts, sq, "sq_scale", "sq_scale", "float32",
                    "scalar-quantizer vmin/vdiff per dim (HNSW SQ storage)", regions)

    # header = bytes before the earliest located block (magic/params).
    located_starts = [r["byte_start"] for r in regions if r["located"]]
    if located_starts:
        first = min(located_starts)
        if first > 0 and not any(r["byte_start"] == 0 for r in regions):
            regions.insert(0, _region("header", "header", 0, first, "bytes",
                                      "fourcc magic + index parameters", located=False))

    _validate(regions, total)
    return {"index": name, "total_bytes": total, "regions": regions}


def _validate(regions, total):
    """Sanity: every region in-range; located regions ascending and non-overlapping."""
    for r in regions:
        s, n = r["byte_start"], r["byte_len"]
        if s < 0 or n < 0 or s + n > total:
            raise ValueError(f"region {r['name']} out of range: [{s},{s + n}) > {total}")
    located = sorted((r for r in regions if r["located"]), key=lambda r: r["byte_start"])
    for a, b in zip(located, located[1:]):
        if a["byte_start"] + a["byte_len"] > b["byte_start"]:
            raise ValueError(f"located regions overlap: {a['name']} / {b['name']}")


def sample_bit_in_region(region, rng):
    """Pick a uniform random (byte_pos, bit) inside a region — used by Phase 1/2."""
    byte_pos = int(region["byte_start"] + rng.integers(region["byte_len"]))
    bit = int(rng.integers(8))
    return byte_pos, bit
