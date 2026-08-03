"""Phase 3 — E10: exhaustive single-bit detection coverage of the index header guard.

WHY. `HierarchicalNSW::load()` (hnsw.hpp L769-819) reads all 21 header fields with no
validation and then allocates on the values it just read:

    malloc(num_cluster_ * padded_dim_ * sizeof(float))
    malloc(max_elements_ * size_data_per_element_)

The centroids are their OWN 8 KB block at file offset 156 — not part of the header, contrary to
what the E6 centroid row was previously read as saying. But an 8 KB device row anchored in the
centroids is aligned to the buffer start (qp/faults.py), so it covers [0, 8192) and destroys all
156 header bytes on its way. That, not the centroid bytes, is what crashes 29 of 30 protected
seeds: garbage geometry followed by malloc + read.

WHAT THIS MEASURES. `layout.validate_header` is the predicate the C++ loader patch states, and
this driver enumerates EVERY single-bit flip of EVERY field of the real clean header — 156 bytes
= 1,248 cases — asking whether it is rejected. Enumerated, not sampled, exactly as
phase3_e8_framing.bound_coverage did for the upper-link length words.

Three tiers are measured, because they are what three different callers can afford:
  header-only        what C++ load() can check before it has read anything but the header
  + bounded walk     one pass over the first BODY_CHECK_RECORDS upper-link records
  + full walk        the whole record stream to EOF — the shipped framing guard's check 4

The claim this backs is NOT "100% of header flips are detected", and the number that matters is
not the raw fraction. Escapes are classified by what they would actually DO:

  desync         moves a computed offset, so load()'s sequential read misaligns and every later
                 record plus the rotation come from the wrong place
  level_overrun  sends the descent to a level that was never allocated — either by raising
                 maxlevel, or by moving enterpoint_node to an element that is not at the top
                 level. The second looks benign (the value is a valid element id) and is not:
                 it segfaults the stock binary.
  alloc          changes only how much room max_elements reserves
  benign         lands on a value a legitimate index could genuinely hold

The gate fails on a `desync` or `level_overrun` that survives every tier. It deliberately does
NOT fail on `alloc`/`benign`: rejecting those would mean rejecting legitimate indexes, which is
the bar framing-guard.patch set for itself and the bar this has to meet too.

Artifacts:
  artifacts/phase3/e10/header_coverage.json   per-field table + escape classes + gate verdict

Usage:
  python phase3_e10_header.py                 # against the real index
  python phase3_e10_header.py --index PATH
"""

import argparse
import json
import os
import struct

import numpy as np

from qp.rabitq import layout

OUT_DIR = os.path.join("artifacts", "phase3", "e10")

# Which fields load() consumes to compute an offset, an allocation size, or a search bound.
# Sources: hnsw.hpp load() L769-819 (the mallocs and the sequential reads) and the search path
# (`maxM0_`/`maxM_` cap the neighbour scan, `maxlevel_`/`enterpoint_node_` drive the descent,
# `num_cluster_` bounds the per-query q_to_centroids loop).
#
# The three excluded fields are build-time parameters that no query reads:
#   M                only feeds mult_ and the insert-time neighbour selection
#   mult             only assigns a level to a NEWLY INSERTED element
#   ef_construction  only bounds the build-time candidate list
LOAD_BEARING = {
    "max_elements", "cur_element_count", "dim", "padded_dim", "num_cluster", "ex_bits",
    "size_bin_data", "size_ex_data", "size_links_level0", "offsetBinData", "offsetExData",
    "label_offset", "size_data_per_element", "size_links_per_element", "maxlevel",
    "enterpoint_node", "maxM", "maxM0",
}


def field_spans():
    """[(name, struct_code, byte_offset, byte_len)] for the 21 header fields, in file order."""
    spans, off = [], 0
    for name, code in layout.HEADER_FIELDS:
        n = struct.calcsize(code)
        spans.append((name, code, off, n))
        off += n
    assert off == layout.HEADER_BYTES, f"header spans sum to {off}, not {layout.HEADER_BYTES}"
    return spans


def classify_escape(clean_hdr, hdr):
    """What a header flip that the header-only predicate ACCEPTS would actually do.

    "Undetected" and "dangerous" are not the same thing, and conflating them would either
    overstate the guard or condemn it for failing to reject headers a legitimate index could
    hold. Three classes:

      desync     the flip moves a computed offset (the centroid block's size, or where level0
                 ends), so load()'s sequential read lands in the wrong place and every later
                 record plus the rotation are read from garbage. This is the dangerous class —
                 the silent-collapse mode, reached through the header.
      alloc      the flip changes only an allocation size (max_elements). load() reads the same
                 bytes from the same offsets; it just reserves a different amount of room.
      benign     the flip lands on a value a legitimate index could genuinely hold AND that
                 load() acts on identically — a lower maxlevel, for instance. Rejecting these
                 would mean rejecting legitimate indexes, the bar the framing guard set.
    """
    if layout.first_upper_link_offset(hdr) != layout.first_upper_link_offset(clean_hdr):
        return "desync"
    if hdr["maxlevel"] > clean_hdr["maxlevel"]:
        # The descent starts at maxlevel_ and calls get_linklist(level) on the way down. A level
        # no element actually has means reading past that element's link list. Lowering it only
        # starts the descent further down, which a legitimate index could also do.
        return "level_overrun"
    if hdr["enterpoint_node"] != clean_hdr["enterpoint_node"]:
        # Same overrun by a different route, and the one that most looks benign: the flipped
        # value is a perfectly valid element id, so no bound on the field can reject it. But the
        # descent asks THAT element for level maxlevel, and only one element in the index is at
        # the top level. Measured, not assumed: such a flip segfaults the stock binary (-11).
        return "level_overrun"
    if hdr["max_elements"] != clean_hdr["max_elements"]:
        return "alloc"
    return "benign"


def header_bit_coverage(clean_bytes, file_size, body=None):
    """Flip every bit of every field; report which flips the guard rejects, and what escapes.

    Two predicates are measured, because they are what two different callers can afford:
      * header-only  (`validate_header`)          — what the C++ load() can check before it has
                                                    read anything but the header
      * header+body  (`validate_header_with_body`)— what the harness can check, since it holds
                                                    the whole buffer

    A flip that makes the header UNPARSEABLE counts as detected: load() would have died on it
    too. In practice every field here is a fixed-width scalar, so this only arises via `mult`.
    """
    clean = bytearray(clean_bytes)
    if len(clean) != layout.HEADER_BYTES:
        raise ValueError(f"need exactly {layout.HEADER_BYTES} header bytes, got {len(clean)}")
    clean_hdr = layout.parse_header(bytes(clean))

    per_field, total = [], 0
    det_hdr = det_body = 0
    escapes = {"desync": 0, "level_overrun": 0, "alloc": 0, "benign": 0}
    escaped_desync = []
    det_full_extra = [0]        # cases the full walk closes that the bounded one does not
    for name, code, off, nbytes in field_spans():
        detected, detected_body, undetected = 0, 0, []
        for bit in range(nbytes * 8):
            trial = bytearray(clean)
            trial[off + bit // 8] ^= (1 << (bit % 8))
            try:
                hdr = layout.parse_header(bytes(trial))
                violated = layout.validate_header(hdr, file_size)
            except Exception:                                   # noqa: BLE001 — see docstring
                detected += 1
                detected_body += 1
                continue
            if violated:
                detected += 1
                detected_body += 1
                continue
            kind = classify_escape(clean_hdr, hdr)
            escapes[kind] += 1
            caught_by_body = caught_by_full = False
            if body is not None:
                caught_by_body = bool(layout.validate_header_with_body(hdr, body, file_size))
                if caught_by_body:
                    detected_body += 1
                    caught_by_full = True
                elif kind in ("desync", "level_overrun"):
                    # These two classes change where or how far load() reads, and both are
                    # settled by the full pass: it pins the end of the record stream and the
                    # longest record in it. The other classes cannot be closed by reading more.
                    caught_by_full = bool(layout.validate_header_with_body(
                        hdr, body, file_size, records=None))
            if caught_by_full and not caught_by_body:
                det_full_extra[0] += 1
            if kind in ("desync", "level_overrun") and not caught_by_full:
                escaped_desync.append({"field": name, "bit": bit, "class": kind})
            undetected.append({"bit": bit, "value": struct.unpack_from(code, bytes(trial), off)[0],
                               "class": kind, "caught_by_body_check": caught_by_body,
                               "caught_by_full_walk": caught_by_full})
        total += nbytes * 8
        det_hdr += detected
        det_body += detected_body
        per_field.append({
            "field": name, "struct": code, "byte_offset": off, "bits": nbytes * 8,
            "detected_header_only": detected, "detected_with_body": detected_body,
            "undetected_with_body": nbytes * 8 - detected_body,
            "detected_fraction_header_only": detected / (nbytes * 8),
            "detected_fraction_with_body": detected_body / (nbytes * 8),
            "load_bearing": name in LOAD_BEARING,
            "escape_classes": {k: sum(1 for u in undetected if u["class"] == k)
                               for k in ("desync", "level_overrun", "alloc", "benign")},
            # Cap the examples: a fully-uncovered 64-bit field would otherwise dump 64 rows of
            # noise into an artifact whose point is the per-field verdict.
            "undetected_examples": undetected[:8],
        })
    return {
        "cases_enumerated": total,
        "detected_header_only": det_hdr,
        "detected_with_body": det_body,
        "detected_with_full_walk": det_body + det_full_extra[0],
        "detected_fraction_header_only": det_hdr / total if total else None,
        "detected_fraction_with_body": det_body / total if total else None,
        "detected_fraction_with_full_walk": ((det_body + det_full_extra[0]) / total
                                             if total else None),
        "escape_classes": escapes,
        "escaped_desync_cases": escaped_desync,
        "per_field": per_field,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=None,
                    help="index file to read the clean header from (default: the adapter's)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    path = args.index
    if path is None:
        from qp.rabitq import adapter as rq_adapter
        path = rq_adapter.INDEX_PATH
    if not os.path.isfile(path):
        raise SystemExit(f"index not found: {path} (build it via build_rabitq.sh)")

    file_size = os.path.getsize(path)
    with open(path, "rb") as fh:
        clean_bytes = fh.read(layout.HEADER_BYTES)
    # The body check reads one uint32 at a header-derived offset, so it needs the real bytes.
    # np.memmap keeps this off the heap: the index is 268 MiB and only 4 bytes per case are read.
    body = np.memmap(path, dtype=np.uint8, mode="r")

    hdr = layout.parse_header(clean_bytes)
    clean_violations = layout.validate_header(hdr, file_size)
    if clean_violations:
        raise SystemExit(
            f"REPORT-AND-STOP: the predicate rejects the REAL CLEAN header on "
            f"{clean_violations} — a guard that cannot accept a legitimate index is worse than "
            f"no guard. Fix the constraint before reading any coverage number below.")

    cov = header_bit_coverage(clean_bytes, file_size, body=body)
    summary = {
        "experiment": "e10_header_coverage",
        "index": os.path.abspath(path),
        "file_size": file_size,
        "header_bytes": layout.HEADER_BYTES,
        "clean_header": {k: v for k, v in hdr.items()},
        "constraints": [{"name": n, "kind": k} for n, k, _ in layout.HEADER_CONSTRAINTS],
        "coverage": cov,
        "gate": {
            "clean_header_accepted": True,
            "clean_header_accepted_with_body": not layout.validate_header_with_body(
                hdr, body, file_size),
            "no_escaped_desync": not cov["escaped_desync_cases"],
        },
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "header_coverage.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[e10] {cov['cases_enumerated']} single-bit header flips enumerated "
          f"(21 fields x 156 B x 8)")
    print(f"[e10]   header-only predicate : {cov['detected_header_only']:>5} detected "
          f"({cov['detected_fraction_header_only']:.4f})   <- what C++ load() can check")
    print(f"[e10]   + {layout.BODY_CHECK_RECORDS}-record walk    : {cov['detected_with_body']:>5} "
          f"detected ({cov['detected_fraction_with_body']:.4f})")
    print(f"[e10]   + full walk to EOF    : {cov['detected_with_full_walk']:>5} detected "
          f"({cov['detected_fraction_with_full_walk']:.4f})   <- + the shipped framing guard")
    print(f"[e10]   escapes by class      : {cov['escape_classes']}")
    print(f"[e10] {'field':<24} {'bits':>5} {'hdr':>5} {'+body':>6}  load-bearing  escapes")
    for row in cov["per_field"]:
        esc = {k: v for k, v in row["escape_classes"].items() if v}
        print(f"[e10] {row['field']:<24} {row['bits']:>5} {row['detected_header_only']:>5} "
              f"{row['detected_with_body']:>6}  {'yes' if row['load_bearing'] else 'no ':<12}  "
              f"{esc if esc else ''}")
    print(f"[e10] wrote {out_path}")

    esc = cov["escape_classes"]
    print(f"[e10] FINDING. The header-only predicate — the only tier C++ load() can afford "
          f"before it has read the body — leaves {esc['desync'] + esc['level_overrun']} flips "
          f"that change where or how far load() reads: {esc['desync']} move a computed offset, "
          f"{esc['level_overrun']} send the descent to a level that was never allocated (by "
          f"raising maxlevel, or by moving enterpoint_node to an element that is not at the "
          f"top level — a perfectly valid element id, which is why no bound on the field can "
          f"reject it, and which segfaults the stock binary). Walking the record stream to EOF "
          f"closes every one of them, which is why the header guard and the shipped framing "
          f"guard belong together rather than as alternatives. The remaining "
          f"{esc['alloc'] + esc['benign']} escapes are values a legitimate index could hold and "
          f"that load() acts on identically — a lower maxlevel, a larger max_elements — and "
          f"rejecting those would mean rejecting legitimate indexes, which is the bar "
          f"framing-guard.patch set for itself.")

    if cov["escaped_desync_cases"]:
        raise SystemExit(
            f"REPORT-AND-STOP: {len(cov['escaped_desync_cases'])} flip(s) move a computed "
            f"offset and survive even the full walk: {cov['escaped_desync_cases'][:8]}. That is "
            f"a silent read desync no tier covers — a finding that scopes the claim rather than "
            f"a number. See {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
