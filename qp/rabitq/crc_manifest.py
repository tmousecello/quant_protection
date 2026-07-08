"""CRC manifest — Python-computed per-element CRCs of a CLEAN index, consumed by C++.

Option B contract (artifacts/phase3/plan/stage2_cpp_patch.md §2b): the definition of "clean"
stays in qp. Before injection, this module CRCs each element's ex field of the clean index and
writes a manifest; after injection, the corrupted index file + this manifest go to the patched
`exp_dumpids --recovery {drop|fallback_eb}`, whose `load_crc_manifest()` re-CRCs the loaded
bytes and marks mismatching elements corrupted. C++ only COMPARES — it never defines clean and
never injects (hard rule #1).

Granularity: one entry PER ELEMENT over that element's `ex_code` field (96 B on the SIFT b=7
index). This is the per-candidate refinement of the E5 slope layer's chunking
(phase3_e5_recovery.py: zlib.crc32, chunk_size 4096 over element-0's ex_code): a 4096-B chunk
over a 96-B field is the whole field, so manifest entry 0's CRC equals E5's `_clean_crcs[0]`
byte-for-byte — same algorithm, same bytes (asserted in tests/test_expb.py). `field` is
configurable for the future "bin also corrupted" experiments flagged by spec rule 3.

Binary format (little-endian, 64-byte header + 16 B/entry):

  off size field
  0    8   magic            b"QPCRCMF\\x01"
  8    4   u32 version      1
  12   4   u32 algo         1 = CRC-32/ISO-HDLC (poly 0xEDB88320 reflected, init 0xFFFFFFFF,
                            final xor 0xFFFFFFFF) — exactly zlib.crc32
  16   8   u64 n_entries    cur_element_count
  24   8   u64 index_file_size          } geometry echo: the C++ loader validates every one
  32   8   u64 level0_start             } against its own loaded header and fails loudly on
  40   8   u64 size_data_per_element    } mismatch — pairing the wrong manifest with the wrong
  48   8   u64 field_off_in_element     } index is the #1 foot-gun.
  56   4   u32 field_len
  60   4   u32 reserved     0
  64.. n_entries x { u64 abs_file_offset, u32 len, u32 crc32 }

The per-entry (offset, len) is redundant with the header geometry on purpose (spec §2b asks for
"simple (offset, len, crc)"): the C++ loader re-derives each offset and cross-checks, turning
the "entry i == element i" ordering assumption into a verified invariant.
"""
import os
import struct
import zlib

import numpy as np

from qp.rabitq import layout

MAGIC = b"QPCRCMF\x01"
VERSION = 1
ALGO_CRC32 = 1                      # zlib.crc32 == CRC-32/ISO-HDLC
HEADER_FMT = "<8sIIQQQQQII"
HEADER_BYTES = struct.calcsize(HEADER_FMT)          # == 64
ENTRY_DTYPE = np.dtype([("offset", "<u8"), ("len", "<u4"), ("crc", "<u4")])   # 16 B/entry
DEFAULT_FIELD = "ex_code"


def _field_geometry(rmap, field):
    """(level0_start, spe, field_off_in_element, field_len, n) with layout.py as the single
    source of truth: element 0 AND element n-1 go through element_field_range (which validates
    against level0's bounds); the loop between them is plain `+ e*spe` arithmetic."""
    hdr = rmap["header"]
    n = int(hdr["cur_element_count"])
    spe = int(hdr["size_data_per_element"])
    level0 = next(r for r in rmap["regions"] if r["name"] == "level0")
    s0, flen = layout.element_field_range(rmap, field, 0)
    if n > 1:
        s_last, _ = layout.element_field_range(rmap, field, n - 1)
        if s_last != s0 + (n - 1) * spe:
            raise AssertionError(f"element stride mismatch for {field!r}: "
                                 f"{s_last} != {s0} + {(n - 1) * spe}")
    return int(level0["byte_start"]), spe, int(s0 - level0["byte_start"]), int(flen), n


# Public alias: the Experiment-B driver reuses the stride-validated geometry (offsets from
# layout.element_field_range) instead of re-deriving element addressing by hand.
def field_geometry(rmap, field=DEFAULT_FIELD):
    """(level0_start, size_data_per_element, field_off_in_element, field_len, n_elements)."""
    return _field_geometry(rmap, field)


def build_manifest(buf, rmap, field=DEFAULT_FIELD):
    """Serialize a manifest (bytes) from a CLEAN index buffer. Pure — no filesystem."""
    buf = np.asarray(buf, dtype=np.uint8)
    level0_start, spe, field_off, field_len, n = _field_geometry(rmap, field)
    entries = np.empty(n, dtype=ENTRY_DTYPE)
    mv = memoryview(buf)                      # zero-copy slices for zlib.crc32
    for e in range(n):
        s = level0_start + e * spe + field_off
        entries[e] = (s, field_len, zlib.crc32(mv[s:s + field_len]))
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, ALGO_CRC32, n,
                         int(buf.size), level0_start, spe, field_off, field_len, 0)
    return header + entries.tobytes()


def write_manifest(index_source, out_path, field=DEFAULT_FIELD, rmap=None):
    """Build + write a manifest for a clean index (path or uint8 buffer). Returns a summary."""
    if isinstance(index_source, (str, os.PathLike)):
        buf = np.fromfile(index_source, dtype=np.uint8)
    else:
        buf = np.asarray(index_source, dtype=np.uint8)
    if rmap is None:
        rmap = layout.serialized_region_map(layout.parse_header(bytes(buf[:layout.HEADER_BYTES])),
                                            file_size=int(buf.size))
    blob = build_manifest(buf, rmap, field=field)
    with open(out_path, "wb") as fh:
        fh.write(blob)
    n = rmap["header"]["cur_element_count"]
    return {"path": str(out_path), "field": field, "n_entries": int(n),
            "manifest_bytes": len(blob), "algo": "crc32(zlib)",
            "index_file_size": int(buf.size)}


def read_manifest(path):
    """Parse a manifest file -> {"header": {...}, "entries": structured ndarray}."""
    with open(path, "rb") as fh:
        raw_hdr = fh.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"manifest too short: {len(raw_hdr)} < {HEADER_BYTES} header bytes")
        (magic, version, algo, n, file_size, level0_start, spe,
         field_off, field_len, _reserved) = struct.unpack(HEADER_FMT, raw_hdr)
        if magic != MAGIC:
            raise ValueError(f"bad manifest magic {magic!r} (want {MAGIC!r})")
        if version != VERSION or algo != ALGO_CRC32:
            raise ValueError(f"unsupported manifest version/algo {version}/{algo}")
        entries = np.fromfile(fh, dtype=ENTRY_DTYPE, count=n)
    if entries.size != n:
        raise ValueError(f"manifest truncated: {entries.size}/{n} entries")
    return {"header": {"version": version, "algo": algo, "n_entries": int(n),
                       "index_file_size": int(file_size), "level0_start": int(level0_start),
                       "size_data_per_element": int(spe),
                       "field_off_in_element": int(field_off), "field_len": int(field_len)},
            "entries": entries}


def verify_buffer(buf, manifest):
    """Element indices whose current bytes mismatch the manifest CRC (empty list == clean).

    The Python mirror of the C++ loader's eager scan — used by the stub adapter (so the manifest
    code path is genuinely exercised offline) and by the driver's pre-flight sanity check.
    """
    buf = np.asarray(buf, dtype=np.uint8)
    hdr = manifest["header"]
    if hdr["index_file_size"] and hdr["index_file_size"] != buf.size:
        raise ValueError(f"manifest is for a {hdr['index_file_size']}-byte index, "
                         f"got {buf.size} bytes")
    mv = memoryview(buf)
    failing = []
    for e, (off, ln, crc) in enumerate(manifest["entries"]):
        if zlib.crc32(mv[int(off):int(off) + int(ln)]) != crc:
            failing.append(e)
    return failing


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Write a per-element CRC manifest of a CLEAN RaBitQ index (Option B, §2b). "
                    "Run BEFORE injection; the corrupted index + this manifest feed "
                    "exp_dumpids --recovery {drop|fallback_eb}.")
    ap.add_argument("index", help="clean index file (e.g. hnsw_M16_efC200_b7.index)")
    ap.add_argument("out", help="manifest output path")
    ap.add_argument("--field", default=DEFAULT_FIELD,
                    help=f"per-element field to CRC (default {DEFAULT_FIELD})")
    args = ap.parse_args(argv)
    summary = write_manifest(args.index, args.out, field=args.field)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
