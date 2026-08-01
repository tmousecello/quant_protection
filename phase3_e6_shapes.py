#!/usr/bin/env python3
"""Phase 3 — E6: DRAM fault SHAPE x index REGION damage matrix (CIDR spec E1).

E1 asked "which bits matter"; E5/Experiment-B asked "does the recovery stack hold". E6 asks the
question the DRAM literature forces on us: real memory does not fail one bit at a time. A failed
row decoder takes out a whole ~8 KB row; a bad I/O gate / TSV takes out one bit lane down a
column of rows. So the damage a fault does to a RaBitQ index is a function of TWO variables — the
physical SHAPE of the fault and the index REGION it lands in — and this driver measures the full
cross product, with the two-layer recovery stack OFF and ON.

  shapes  {single_cell, device_row, device_column}                       (qp.faults, spec §B)
  strata  {rotation, centroids, ex_code, bin_code, links, factors, ids}  (the anchor's region)
  seeds   30 (--smoke: 2)
  arms    {off, on}                                                      -> 3*7*30*2 = 1260 evals

DEVIATION FROM THE MERGED-REGION SPEC: the spec's single `rotation_centroids` stratum is split
into `rotation` and `centroids`. Byte-weighting a merged window is 8192:64 on the stub geometry
(and worse on the real index), so the headline cell — a device_row landing IN the 64-byte global
rotation, the one structure whose corruption is a single-point catastrophe — would be sampled
under 1% of the time and is effectively unsampleable at 30 seeds. Splitting costs one extra
stratum and makes that cell a first-class measurement.

THE STRATUM PICKS THE ANCHOR, NOT THE DAMAGE. For `single_cell` the two coincide. For the device
shapes they emphatically do not, and that gap IS the finding: RaBitQ interleaves links / ids /
bin_code / bin_factors / ex_code / ex_factors every 272 B, so an 8 KB row anchored in one
element's ex_code physically shreds ~30 elements' worth of *every other* field too, and a column
stripe precesses across different fields row to row (8192 is not a multiple of 272). Every row
therefore records what was ACTUALLY hit: `bits_flipped`, `coverage_bytes` (exact distinct bytes
touched, derived from `positions` — NOT the injectors' bounding `coverage` span, which grossly
overstates device_column), and a per-field `field_hits` breakdown in the raw JSONL.

Arms:
  off — flip -> deserialize -> `adapter.search_corrupted` -> recall@10 via qp.metrics.
  on  — a FRESH `phase3_e5_recovery.RecoveryGuard` per injection (R=3 rotation replicas,
        cliff_scrub, chunk_size 4096), `init_from_clean(clean_buf)`, then after the flip
        `bounds_check_full` (below) followed by `guard.scrub_only(buf)` (cliff majority-vote
        repair + ex CRC) and `adapter.query_with_recovery(tmp, "fallback_eb", manifest)` with a
        CRC manifest written once from the clean index. Recall is recomputed from the returned
        ids — never taken from the C++ side.

DEVIATION FROM E5's BOUNDS CHECK: E5's `_bounds_check` scans a fixed 16-element sample, which is
right for E5 (a tick timeline where damage accumulates and any sample eventually sees it) and
wrong for E6. A device_row anchored at element 300,000 would have exactly zero chance of a
bounds-check hit on a 1M-element index, so `oob_restored` would be structurally ~0 and the
pointer-protection mechanism would look useless when it was simply never invoked. E6 measures
MECHANISM COVERAGE, not a timeline, so it runs its own vectorized FULL-index bounds check
(`bounds_check_full`) before handing the buffer to the guard; the guard's own sampled check then
finds nothing left, and its counter is recorded as `guard_oob_elements` as a cross-check.

Outputs (under --out; default artifacts/phase3/e6, --smoke -> artifacts_smoke/phase3/e6):
  raw/e6.records.jsonl   one row per eval (anchor, coverage, field_hits, counters, error)
  raw/e6.done            completion marker (enables --resume)
  e6_results.csv         the 13-column damage matrix (CSV_COLUMNS)
  e6_summary.json        provenance meta + per-(shape,stratum,arm) medians/p95 + smear + gates
  e6_p3.json             --p3 only: 64-bit burst in ONE element vs 64 single bits striped
                         across 64 elements (same bit budget, opposite CRC footprint)

Usage:
  python phase3_e6_shapes.py --adapter stub --smoke     # dev pipeline gate (plumbing only)
  python phase3_e6_shapes.py                            # workstation full run (auto -> real)
  python phase3_e6_shapes.py --p3                       # the burst-vs-stripe comparison
"""
import argparse
import csv
import json
import os
import sys
import time
import zlib

import numpy as np

from qp import config, faults, metrics, provenance
from qp.bits import burst_positions, flip_bits
from qp.rabitq import crc_manifest, layout
from qp.rabitq.registry import get_adapter, adapter_name
from qp.rawio import RawWriter
from phase3_e5_recovery import RecoveryGuard

SHAPES = ("single_cell", "device_row", "device_column")
STRATA = ("rotation", "centroids", "ex_code", "bin_code", "links", "factors", "ids")
GLOBAL_STRATA = ("rotation", "centroids")
ARMS = ("off", "on")
OUTCOMES = ("crash", "repaired", "tolerated", "detected_wrong", "silent_wrong")

# Per-element strata -> the layout fields they union over, IN byte-weighting order. A stratum with
# two fields is sampled byte-weighted across both (factors 12:8 bin:ex, ids 4:4 cluster_id:label),
# so a stratum's anchor distribution is uniform over its BYTES, not over its sub-fields.
ELEMENT_STRATUM_FIELDS = {
    "ex_code": ("ex_code",),
    "bin_code": ("bin_code",),
    "links": ("links",),
    "factors": ("bin_factors", "ex_factors"),
    "ids": ("cluster_id", "label"),
}
# Every stratum -> the field NAMES make_field_resolver emits for it, so "how much of this fault
# landed in the kind of structure it was aimed at" can be asked at field level (across all
# elements) as well as at the anchor element's own field.
STRATUM_FIELDS = dict(ELEMENT_STRATUM_FIELDS, rotation=("rotation",), centroids=("centroids",))

# Pointer sentinel E5's bounds check exempts (an explicitly-empty slot, not a corrupt id); kept
# identical here so guard_oob_elements is a meaningful cross-check of bounds_check_full.
PTR_SENTINEL = 0xFFFFFFFF
# Elements decoded per vectorized bounds-check pass. Bounds memory, not speed: a 1M-element index
# at maxM0=32 would otherwise materialize ~130M uint32 at once.
BOUNDS_CHUNK_ELEMENTS = 65536

# --- outcome thresholds (named, so a reader never has to guess what 0.005 means) ---------------
# RETENTION_TOL: recall within this of the clean baseline counts as "the query still works".
#   0.005 is half the E1/E3 catastrophic bar and ~5x the real search's float jitter.
RETENTION_TOL = 0.005
# DELTA_TOL: the phase1/E1 "harmful" bar (qp.metrics catastrophic threshold). Damage that moves
#   recall by less than this is tolerated whether or not any layer noticed it.
DELTA_TOL = 0.01

# RecoveryGuard configuration for arm ON (spec-fixed). anchor_every=0 disables the low-frequency
# anchor backstop: E6 is single-shot (one injection per eval, no tick timeline), so a periodic
# check has nothing to be periodic over.
GUARD_CFG = {"R": 3, "cliff_scrub": True, "anchor_every": 0, "chunk_size": 4096}

RECOVERY_MODE = "fallback_eb"
MANIFEST_FIELD = "ex_code"

# RNG lanes: the anchor sampler and the injector must not share a stream (both would otherwise
# consume default_rng(seed) from its first draw).
LANE_INJECT = 1
LANE_P3 = 2

FULL = dict(seeds=30, ef=2000, row_bytes=8192, n_rows=512, p_in_row=0.5)
SMOKE = dict(FULL, seeds=2, ef=64)

CSV_COLUMNS = ["shape", "region", "seed", "arm", "recall", "delta_recall", "outcome",
               "bits_flipped", "coverage_bytes", "elements_crc_fail", "fallbacks",
               "cliff_repaired", "oob_restored"]

DEFAULT_WEIGHTS = os.path.normpath(os.path.join(
    config.ROOT, "..", "..", "results", "jonathan-vuln-shapes", "shape_weights.json"))

P3_BITS = 64
P3_STRIPE_ELEMENTS = 64
P3_MODES = ("burst_one_element", "striped_64_elements")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pure, seed-deterministic pieces (unit-tested in artifacts/phase3/tests/test_e6_shapes.py)
# ---------------------------------------------------------------------------

def stratum_windows(rmap, stratum, element=None):
    """Absolute [(byte_start, byte_len), ...] windows a stratum's anchor may be drawn from.

    Global strata (`rotation`, `centroids`) ignore `element` and return that structure's single
    absolute window. They are kept SEPARATE rather than merged as the spec's `rotation_centroids`
    (see the module docstring): merged and byte-weighted, the 64-byte rotation would be drawn
    under 1% of the time and its cell would never fill.
    Per-element strata REQUIRE `element` and go through layout.element_field_range, so the
    element stride (e*size_data_per_element) is never re-derived here.
    """
    if stratum in GLOBAL_STRATA:
        r = next((x for x in rmap["regions"] if x["name"] == stratum), None)
        if r is None or r["byte_start"] is None:
            raise ValueError(f"stratum {stratum!r} is not located in the region map "
                             f"(was the map built without file_size?)")
        return [(int(r["byte_start"]), int(r["byte_len"]))]
    if stratum not in ELEMENT_STRATUM_FIELDS:
        raise ValueError(f"unknown stratum {stratum!r}; choose from {STRATA}")
    if element is None:
        raise ValueError(f"stratum {stratum!r} is per-element: pass element=<int>")
    return [layout.element_field_range(rmap, f, int(element))
            for f in ELEMENT_STRATUM_FIELDS[stratum]]


def sample_anchor(rmap, stratum, seed):
    """Pick the fault's anchor byte for `stratum`, byte-weighted and deterministic in `seed`.

    Pure: same (rmap geometry, stratum, seed) -> same result, no buffer touched. Per-element
    strata draw the element first (uniform over cur_element_count), then a byte uniformly over
    the union of that element's stratum windows — so a 12-B bin_factors window is picked 60% of
    the time against an 8-B ex_factors window, which is what "byte-weighted" means.

    Returns {"stratum", "element" (None for global strata), "window_index", "window", "anchor_byte"}.
    """
    rng = np.random.default_rng(seed)
    element = None
    if stratum not in GLOBAL_STRATA:
        n = int(rmap["header"]["cur_element_count"])
        if n <= 0:
            raise ValueError("index has no elements; per-element strata are unsamplable")
        element = int(rng.integers(n))
    windows = stratum_windows(rmap, stratum, element=element)
    total = sum(L for _, L in windows)
    offset = int(rng.integers(total))
    for idx, (start, length) in enumerate(windows):
        if offset < length:
            return {"stratum": stratum, "element": element, "window_index": idx,
                    "window": (int(start), int(length)), "anchor_byte": int(start + offset)}
        offset -= length
    raise AssertionError("unreachable: byte-weighted draw fell off the end of the windows")


def cell_seed(root_seed, shape, stratum, index):
    """Deterministic per-(root, shape, stratum, replicate) seed, collision-free by construction.

    SeedSequence entropy-list derivation rather than XOR mixing: `root ^ lane` aliases (adjacent
    roots with adjacent lanes produce identical seeds), which expb caught the hard way. The shape
    and stratum enter as crc32 of their names so the mapping is text-stable across runs.
    """
    return int(np.random.SeedSequence([int(root_seed), zlib.crc32(shape.encode()),
                                       zlib.crc32(stratum.encode()),
                                       int(index)]).generate_state(1)[0])


def lane_seed(seed, lane):
    """Independent sub-stream seed off `seed` (same construction as cell_seed, one level down)."""
    return int(np.random.SeedSequence([int(seed), int(lane)]).generate_state(1)[0])


def make_field_resolver(rmap):
    """Build `off -> field name` for absolute buffer bytes (the smear accounting's core).

    Inside level0 the offset is reduced modulo size_data_per_element and matched against
    layout.element_regions, so a byte in ANY element resolves to its field (links / cluster_id /
    label / bin_code / bin_factors / ex_code / ex_factors). Outside level0 the top-level blocks
    (header / centroids / upper_links / rotation) are matched directly. Returns a closure so the
    per-byte lookup does not rebuild the geometry for each of a row's ~8k bytes.
    """
    hdr = rmap["header"]
    level0 = next(r for r in rmap["regions"] if r["name"] == "level0")
    l0s = int(level0["byte_start"])
    l0e = l0s + int(level0["byte_len"])
    spe = int(hdr["size_data_per_element"])
    ers = [(er["name"], int(er["offset_in_element"]),
            int(er["offset_in_element"]) + int(er["byte_len"]))
           for er in layout.element_regions(hdr["padded_dim"], hdr["ex_bits"], hdr["maxM0"],
                                            off_bin=hdr.get("offsetBinData") or None,
                                            off_ex=hdr.get("offsetExData") or None)]
    blocks = [(r["name"], int(r["byte_start"]), int(r["byte_start"]) + int(r["byte_len"]))
              for r in rmap["regions"]
              if r["byte_start"] is not None and r["byte_len"] > 0
              and r["name"] != "level0" and not r["name"].startswith("elem0.")]

    def resolve(off):
        off = int(off)
        if l0s <= off < l0e:
            within = (off - l0s) % spe
            for name, lo, hi in ers:
                if lo <= within < hi:
                    return name
            return "level0_pad"          # trailing slack inside an element block, if any
        for name, lo, hi in blocks:
            if lo <= off < hi:
                return name
        return "unmapped"
    return resolve


def count_field_hits(rmap, positions, resolver=None):
    """{field: n_distinct_bytes_touched} for a list of flipped (byte, bit) positions.

    Counts BYTES, not bits, so a device_row's 32k flips over 8192 bytes read as "8192 bytes, of
    which N in ex_code" rather than a bit tally nobody can compare against a field size.
    """
    resolve = resolver or make_field_resolver(rmap)
    hits = {}
    for b in {int(p[0]) for p in positions}:
        name = resolve(b)
        hits[name] = hits.get(name, 0) + 1
    return dict(sorted(hits.items(), key=lambda kv: (-kv[1], kv[0])))


def shape_kwargs(shape, cfg):
    """The modeling knobs each shape takes (row_bytes/n_rows/p_in_row are NOT datasheet facts)."""
    if shape == "device_row":
        return {"row_bytes": int(cfg["row_bytes"]), "p_in_row": float(cfg["p_in_row"])}
    if shape == "device_column":
        return {"row_bytes": int(cfg["row_bytes"]), "n_rows": int(cfg["n_rows"])}
    return {}


def inject_shape(buf, shape, anchor_byte, seed, **kwargs):
    """Apply `shape` anchored at exactly `anchor_byte`. Returns (positions, record).

    The injectors in qp.faults draw their own anchor uniformly from the region they are handed,
    so the stratified anchor is imposed by handing them a ONE-BYTE region: the drawn anchor can
    then only be `anchor_byte`, while everything downstream of the anchor (which bit within the
    byte, the row block, the column stripe's (col_offset, col_bit) and its precession across
    rows) is still the injector's own seeded draw. The damage therefore extends far outside that
    one byte for the device shapes — by design; see the module docstring.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    region = (int(anchor_byte), 1)
    injector = {"single_cell": faults.single_cell, "device_row": faults.device_row,
                "device_column": faults.device_column}[shape]
    if shape == "single_cell":
        return injector(buf, region, seed)
    return injector(buf, region, seed, **kwargs)


def classify_outcome(*, arm, recall, clean_recall, elements_crc_fail, cliff_repaired,
                     oob_restored, crashed):
    """The 5-class outcome for one eval. Thresholds are RETENTION_TOL / DELTA_TOL above.

      crash          the search subprocess died or timed out (or returned nothing measurable).
      repaired       arm ON only: recall held AND a recovery layer actually acted (cliff
                     majority-vote repair or a pointer bounds-check restore) AND the CRC scan
                     found nothing left to flag — the damage was put back, not worked around.
      tolerated      recall held but the ex-data CRC still flags elements (EB-fallback carried
                     it rather than repairing), OR the damage moved recall by < DELTA_TOL.
      detected_wrong arm ON only: recall did NOT hold, but the stack SAW the damage (CRC flagged
                     elements, or the cliff/bounds layers fired). The operator gets a signal —
                     an operationally different (and far better) failure than the next class.
                     The P3 stripe is the archetype: 64 elements flagged, recall still gone.
      silent_wrong   recall dropped past both bands with NO detection signal at all — the
                     dangerous case, and the one the paper is about.

    Arm OFF passes None for the recovery counters (nothing was measured, as opposed to measured
    zero) and can therefore never be classified `repaired` or `detected_wrong`.
    """
    if crashed or recall is None:
        return "crash"
    crc_fail = 0 if elements_crc_fail is None else int(elements_crc_fail)
    repaired_ct = 0 if cliff_repaired is None else int(cliff_repaired)
    oob_ct = 0 if oob_restored is None else int(oob_restored)
    held = recall >= clean_recall - RETENTION_TOL
    detected = crc_fail > 0 or repaired_ct > 0 or oob_ct > 0
    if arm == "on" and held and crc_fail == 0 and (repaired_ct > 0 or oob_ct > 0):
        return "repaired"
    if (held and crc_fail > 0) or (clean_recall - recall) < DELTA_TOL:
        return "tolerated"
    if arm == "on" and detected:
        return "detected_wrong"
    return "silent_wrong"


# ---------------------------------------------------------------------------
# Full-index pointer bounds check (E6's replacement for E5's 16-element sample)
# ---------------------------------------------------------------------------

def _u32_columns(mat, offset, count):
    """Decode `count` little-endian uint32 starting at `offset` in every row of a (m, spe) view."""
    raw = np.ascontiguousarray(mat[:, offset:offset + 4 * count]).reshape(mat.shape[0], count, 4)
    return (raw[:, :, 0].astype(np.uint32)
            | (raw[:, :, 1].astype(np.uint32) << 8)
            | (raw[:, :, 2].astype(np.uint32) << 16)
            | (raw[:, :, 3].astype(np.uint32) << 24))


def bounds_check_full(work, clean_buf, rmap):
    """Vectorized FULL-index pointer bounds check + restore. Returns (n_fields_restored, detail).

    Why E6 has its own instead of using E5's `_bounds_check`: see the module docstring — E5's
    fixed 16-element sample cannot see a fault at element 300,000, so on a 1M-element index it
    would report `oob_restored ~ 0` for reasons that have nothing to do with the mechanism.

    Rules (a superset of E5's, which does not check the neighbour COUNT — the field whose
    corruption to 4 billion is the classic crash vector):
      links       count > maxM0, or any of the maxM0 physically-present neighbour slots holds an
                  id >= cur_element_count. Slots are scanned by their FIXED window, never by the
                  on-disk count, which may itself be the corrupted field.
      cluster_id  >= num_cluster        label  >= cur_element_count
    PTR_SENTINEL (0xFFFFFFFF) is exempt for ids exactly as in E5 (an explicitly-empty slot), but
    NOT for the count. A flagged field is restored wholesale from `clean_buf` — the faithful
    "skip the bad edge" that keeps the structure instead of losing it — and counted, one per
    (element, field), matching E5's `oob_elements` accounting.

    Elements are decoded in chunks (BOUNDS_CHUNK_ELEMENTS) so peak memory stays flat regardless
    of index size; on the SIFT1M geometry the whole pass is a couple of hundred milliseconds.
    """
    hdr = rmap["header"]
    n = int(hdr["cur_element_count"])
    spe = int(hdr["size_data_per_element"])
    maxM0 = int(hdr["maxM0"])
    num_cluster = int(hdr["num_cluster"])
    if n == 0:
        return 0, {}
    level0 = next(r for r in rmap["regions"] if r["name"] == "level0")
    l0s = int(level0["byte_start"])
    geom = {}
    for field in ("links", "cluster_id", "label"):
        try:
            start, length = layout.element_field_range(rmap, field, 0)
        except (KeyError, IndexError):
            continue
        geom[field] = (int(start) - l0s, int(length))

    restored, detail = 0, {}
    for lo in range(0, n, BOUNDS_CHUNK_ELEMENTS):
        hi = min(n, lo + BOUNDS_CHUNK_ELEMENTS)
        base = l0s + lo * spe
        mat = work[base: base + (hi - lo) * spe].reshape(hi - lo, spe)
        for field, (off, length) in geom.items():
            if field == "links":
                if length < 8:
                    continue
                n_slots = min(maxM0, (length - 4) // 4)
                words = _u32_columns(mat, off, 1 + n_slots)
                bad = (words[:, 0] > maxM0)
                ids = words[:, 1:]
                bad |= ((ids >= n) & (ids != PTR_SENTINEL)).any(axis=1)
            else:
                if length < 4:
                    continue
                val = _u32_columns(mat, off, 1)[:, 0]
                limit = num_cluster if field == "cluster_id" else n
                bad = (val >= limit) & (val != PTR_SENTINEL)
            flagged = np.flatnonzero(bad)
            if not flagged.size:
                continue
            for e in flagged.tolist():
                s = l0s + (lo + e) * spe + off
                work[s: s + length] = clean_buf[s: s + length]
            restored += int(flagged.size)
            detail[field] = detail.get(field, 0) + int(flagged.size)
    return restored, detail


# ---------------------------------------------------------------------------
# Sanity gates
# ---------------------------------------------------------------------------

def stub_sandbox_ok(aname, out_dir):
    """False iff the stub would write into the real artifacts tree. The gate's single predicate.

    `sanity.stub_sandbox_ok` in the summary is derived from THIS, not hardcoded — a hardcoded
    True is a gate that reports itself green without ever having run.
    """
    if aname != "stub":
        return True
    real_root = os.path.join(os.path.abspath(config.ROOT), "artifacts")
    target = os.path.abspath(out_dir)
    return not (target == real_root or target.startswith(real_root + os.sep))


def assert_stub_sandbox(aname, out_dir):
    """House convention: stub output NEVER lands under artifacts/ (only artifacts_smoke/).

    A stub number that ends up in the real artifacts tree is indistinguishable from a measured
    one three months later, so this is a hard stop rather than a warning.
    """
    if not stub_sandbox_ok(aname, out_dir):
        raise RuntimeError(
            f"REPORT-AND-STOP: stub adapter would write documented-fake results into the real "
            f"artifacts tree ({os.path.abspath(out_dir)}). Use --smoke (-> artifacts_smoke/) "
            f"or an explicit --out.")


def csv_list(kind, allowed):
    """argparse type for a comma-separated subset of `allowed` (the expb --patterns pattern).

    Sharding is by grid slice, so a typo must be a loud error: silently running 6 of 7 strata
    would leave a hole in the damage matrix that only shows up when the shards are merged.
    """
    def parse(s):
        vals = tuple(x.strip() for x in str(s).split(",") if x.strip())
        if not vals:
            raise argparse.ArgumentTypeError(f"empty {kind} list")
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(
                f"unknown {kind}: {bad} (choose from {list(allowed)})")
        seen = [v for i, v in enumerate(vals) if v in vals[:i]]
        if seen:
            raise argparse.ArgumentTypeError(f"duplicate {kind}: {sorted(set(seen))}")
        # Canonical order, so a shard's rows sort the same way as the full grid's.
        return tuple(v for v in allowed if v in vals)
    return parse


def assert_row_count(records, seeds, shapes=SHAPES, strata=STRATA):
    """The (possibly sharded) grid must be complete: no silently dropped cell.

    `shapes`/`strata` are the SLICE this process was asked to run (--shapes / --strata), so a
    shard is gated against its own expected size rather than the full grid's.
    """
    expected = len(shapes) * len(strata) * int(seeds) * len(ARMS)
    assert len(records) == expected, (
        f"row count {len(records)} != expected {expected} "
        f"({len(shapes)} shapes x {len(strata)} strata x {seeds} seeds x {len(ARMS)} arms)")
    return expected


# ---------------------------------------------------------------------------
# Shared setup (clean index + manifest + baseline + provenance stamped up front)
# ---------------------------------------------------------------------------

def _load_weights(path):
    """Provenance only: E6 enumerates the shapes explicitly rather than sampling the vendor mix."""
    p = os.path.abspath(path or DEFAULT_WEIGHTS)
    if not os.path.exists(p):
        log(f"[e6] [warn] shape weights not found at {p} — recorded as absent (E6 iterates "
            f"shapes explicitly, so the mix is provenance, not an input).")
        return {"path": p, "present": False}
    with open(p) as fh:
        blob = json.load(fh)
    return {"path": p, "present": True, "sha256": provenance.sha256_file(p), "weights": blob}


def setup_context(args, cfg):
    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)
    out = os.path.abspath(args.out)
    assert_stub_sandbox(aname, out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    tag = f"_{args.out_tag}" if getattr(args, "out_tag", None) else ""

    clean_buf = adapter.serialize_index()
    rmap = adapter.region_map()
    gt = adapter.load_groundtruth()
    tmp = os.path.join(out, "raw", f"_e6{tag}.index")

    manifest_path = os.path.join(out, f"e6_clean_{MANIFEST_FIELD}{tag}.crcmf")
    crc_manifest.write_manifest(clean_buf, manifest_path, field=MANIFEST_FIELD, rmap=rmap)
    pre = crc_manifest.verify_buffer(clean_buf, crc_manifest.read_manifest(manifest_path))
    if pre:
        raise RuntimeError(f"pre-flight: clean buffer fails its own manifest at elements {pre[:5]}")

    # Clean baseline through the SAME path arm off uses, so delta_recall is measured against a
    # like-for-like number rather than a published anchor.
    adapter.deserialize_index(clean_buf, tmp)
    clean_res = adapter.search_corrupted(tmp, k=config.K, ef=int(cfg["ef"]),
                                         out_path=tmp + ".clean.ivecs", timeout=args.timeout)
    clean_recall = float(metrics.recall_at_k(clean_res["ids"], gt, config.K))
    # ...and the arm-on path must be a no-op on the clean index (expb gate 2 in miniature).
    clean_rec = adapter.query_with_recovery(tmp, RECOVERY_MODE, manifest_path, k=config.K,
                                            ef=int(cfg["ef"]), out_path=tmp + ".cleanrec.ivecs",
                                            timeout=args.timeout)
    crc0 = int((clean_rec.get("stats") or {}).get("load", {}).get("elements_crc_fail", 0))
    if crc0 != 0:
        raise RuntimeError(f"pre-flight: recovery path flags {crc0} elements on the CLEAN index")
    # ...and its RECALL must match the arm-off baseline, else the two arms are measured against
    # different operating points and every delta_recall on the on-arm is rebased to garbage.
    # (Computing this search and throwing away its ids was review finding I2.)
    clean_rec_recall = float(metrics.recall_at_k(clean_rec["ids"], gt, config.K))
    if abs(clean_rec_recall - clean_recall) > RETENTION_TOL:
        raise RuntimeError(
            f"REPORT-AND-STOP: recovery path returns recall@10={clean_rec_recall:.6f} on the "
            f"CLEAN index vs {clean_recall:.6f} for the plain search — a gap of "
            f"{abs(clean_rec_recall - clean_recall):.6f} > {RETENTION_TOL}. The arms would be "
            f"measured against different baselines.")

    # The full-index bounds check must be a no-op on a clean index; if the real index's labels or
    # cluster ids fall outside the ranges E5's rule assumes, arm on would "restore" clean fields
    # on every eval and report fictitious oob_restored counts.
    clean_oob, clean_oob_detail = bounds_check_full(clean_buf.copy(), clean_buf, rmap)
    if clean_oob:
        raise RuntimeError(
            f"REPORT-AND-STOP: bounds_check_full flags {clean_oob} field(s) on the CLEAN index "
            f"({clean_oob_detail}) — the pointer validity rule does not match this index's "
            f"conventions, so arm-on oob_restored would be fiction.")
    _sweep_sidecars(tmp)

    meta = provenance.collect_provenance(
        adapter, aname, {"recall@10": clean_recall}, cfg, args, rmap,
        index_sha256=provenance.sha256_file(tmp),
        corrupted_regions=list(STRATA), recovery=["off", RECOVERY_MODE],
        crc_manifest_sha256=provenance.sha256_file(manifest_path))
    meta["shape_weights"] = _load_weights(getattr(args, "weights", None))
    meta["fault_shape_modeling"] = {
        "row_bytes": int(cfg["row_bytes"]), "n_rows": int(cfg["n_rows"]),
        "p_in_row": float(cfg["p_in_row"]),
        "caveat": ("row_bytes/n_rows are OUR modeling units (qp.faults group-4 caveat), not "
                   "datasheet DRAM geometry — treat as sensitivity knobs, not hardware facts."),
    }
    meta["outcome_thresholds"] = {"retention_tol": RETENTION_TOL, "delta_tol": DELTA_TOL}
    meta["clean_baseline_both_paths"] = {
        "search_corrupted_recall@10": clean_recall,
        "query_with_recovery_recall@10": clean_rec_recall,
        "abs_diff": round(abs(clean_rec_recall - clean_recall), 9),
        "gate": f"abs_diff <= RETENTION_TOL ({RETENTION_TOL})",
    }
    meta["spec_deviations"] = {
        "strata_split": ("spec E1 merges rotation+centroids into one stratum; E6 splits them. "
                         "Byte-weighted over the merged window the 64-B rotation is drawn <1% of "
                         "the time, so the row-in-rotation headline cell would never fill. "
                         f"Grid is {len(SHAPES)}x{len(STRATA)}x{cfg['seeds']}x{len(ARMS)}."),
        "bounds_check": ("arm ON uses E6's vectorized FULL-index pointer bounds check instead of "
                         "E5's fixed 16-element sample: E6 measures mechanism coverage, not E5's "
                         "accumulation timeline, and a sampled check cannot see a fault at "
                         "element 300,000. The guard's own counter is kept as guard_oob_elements."),
    }

    log(f"[e6] adapter={aname} out={out} clean@10={clean_recall:.5f} "
        f"(recovery path {clean_rec_recall:.5f}) n={rmap['header']['cur_element_count']} "
        f"ef={cfg['ef']} seeds={cfg['seeds']}")
    return {"adapter": adapter, "aname": aname, "out": out, "tag": tag, "rmap": rmap,
            "clean_buf": clean_buf, "gt": gt, "tmp": tmp, "manifest_path": manifest_path,
            "clean_recall": clean_recall, "meta": meta, "cfg": cfg,
            "timeout": args.timeout, "resolver": make_field_resolver(rmap)}


def _sweep_sidecars(tmp):
    for suffix in (".ids.ivecs", ".rec.ivecs", ".clean.ivecs", ".cleanrec.ivecs"):
        for stale in (tmp + suffix, tmp + suffix + ".dist.fvecs"):
            try:
                os.remove(stale)
            except OSError:
                pass


def _verify_and_restore(work, clean_buf, positions, arm, scratch=None):
    """Verify the buffer differs from clean ONLY where this injection reached, THEN restore.

    The verification has to happen BEFORE the restore, and the previous version of this function
    got that exactly backwards: arm ON restores by copying `clean_buf` wholesale, so any check
    made afterwards compares clean against clean and is 0 by construction. Since the arms loop
    ends on `on`, that also made the end-of-sweep whole-buffer check vacuous — review I1 proved
    it empirically by injecting a stray byte and still seeing "leak 0".

    What is actually checked here, against the diff BEFORE any restore:
      arm off — the differing byte set must equal the touched byte set EXACTLY. Every touched
                byte must differ (XOR-ing >=1 distinct bit always changes a byte), and nothing
                else may differ.
      arm on  — the differing set must be a SUBSET of the touched set. It is a strict subset
                whenever a recovery layer put something back; a byte outside the injection that
                differs from clean means something wrote where it had no business writing.

    `scratch` is an optional preallocated bool array (buffer-sized) so the whole-buffer compare
    doesn't allocate on every one of the 1,260 evals.
    """
    touched = {int(b) for b, _ in positions}
    if scratch is not None:
        np.not_equal(work, clean_buf, out=scratch)
        diff = set(int(b) for b in np.flatnonzero(scratch))
    else:
        diff = set(int(b) for b in np.flatnonzero(work != clean_buf))

    stray = sorted(diff - touched)
    if stray:
        raise RuntimeError(
            f"state leak (arm={arm}): {len(stray)} byte(s) differ from clean OUTSIDE the "
            f"{len(touched)} bytes this injection touched, e.g. {stray[:8]} — something wrote "
            f"outside the fault footprint, and every later eval would inherit it.")
    if arm == "off":
        unchanged = sorted(touched - diff)
        if unchanged:
            raise RuntimeError(
                f"injection bookkeeping bug: {len(unchanged)} byte(s) recorded as flipped are "
                f"identical to clean, e.g. {unchanged[:8]} — positions and buffer disagree.")

    if arm == "off":
        faults.restore(work, positions)
    else:
        # The guard writes into `work` itself (rotation repair, pointer restores), so re-XORing
        # the injected positions would leave its edits behind AND re-corrupt what it fixed.
        np.copyto(work, clean_buf)
    return len(diff)


# ---------------------------------------------------------------------------
# One eval
# ---------------------------------------------------------------------------

def measure_cell(ctx, work, shape, stratum, seed_index, seed, anchor, arm, scratch=None):
    """Inject one shape at one anchor, measure one arm, restore. Returns the raw record."""
    adapter, cfg = ctx["adapter"], ctx["cfg"]
    kwargs = shape_kwargs(shape, cfg)

    guard = None
    if arm == "on":
        guard = RecoveryGuard(adapter, {**GUARD_CFG, "ef": int(cfg["ef"]), "seed": int(seed)},
                              ctx["rmap"])
        guard.init_from_clean(ctx["clean_buf"])          # snapshot BEFORE any corruption

    positions, record = inject_shape(work, shape, anchor["anchor_byte"],
                                     lane_seed(seed, LANE_INJECT), **kwargs)
    recall = elements_crc_fail = fallbacks = cliff_repaired = oob_restored = None
    guard_oob = oob_detail = None
    crashed, error = False, ""
    try:
        if guard is not None:
            # E6's own full-index pointer check runs FIRST, so the guard's sampled one finds
            # nothing left and its counter becomes a cross-check rather than a second opinion.
            oob_restored, oob_detail = bounds_check_full(work, ctx["clean_buf"], ctx["rmap"])
            guard.scrub_only(work)                       # cliff vote/repair + ex CRC (public API)
        adapter.deserialize_index(work, ctx["tmp"])
        try:
            if arm == "off":
                res = adapter.search_corrupted(ctx["tmp"], k=config.K, ef=int(cfg["ef"]),
                                               out_path=ctx["tmp"] + ".ids.ivecs",
                                               timeout=ctx["timeout"])
                stats = None
            else:
                res = adapter.query_with_recovery(ctx["tmp"], RECOVERY_MODE, ctx["manifest_path"],
                                                  k=config.K, ef=int(cfg["ef"]),
                                                  out_path=ctx["tmp"] + ".rec.ivecs",
                                                  timeout=ctx["timeout"])
                stats = res.get("stats") or {}
        # Anything the search raises is an outcome of the corruption, not a harness bug: a
        # corrupted header can make the adapter's own parsing blow up (struct.error, ValueError,
        # MemoryError) long before a subprocess is ever spawned, and silently propagating that
        # would abort a 6-hour sweep at eval 300. KeyboardInterrupt/SystemExit are BaseException
        # and deliberately still propagate.
        except Exception as exc:                                    # noqa: BLE001
            crashed, error = True, repr(exc)
        else:
            recall = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
            if stats is not None:
                elements_crc_fail = int(stats.get("load", {}).get("elements_crc_fail", 0))
                fallbacks = int(stats.get("totals", {}).get("fallbacks", 0))
        if guard is not None:
            ctr = guard.counters()                       # counters are valid even after a crash
            cliff_repaired = int(ctr["cliff_repaired"])
            guard_oob = int(ctr["oob_elements"])
            if guard_oob:
                log(f"[e6] [warn] guard's sampled bounds check flagged {guard_oob} field(s) "
                    f"AFTER bounds_check_full ran — the two rules disagree ({shape}/{stratum}).")
    finally:
        _verify_and_restore(work, ctx["clean_buf"], positions, arm, scratch=scratch)
        _sweep_sidecars(ctx["tmp"])

    outcome = classify_outcome(arm=arm, recall=recall, clean_recall=ctx["clean_recall"],
                               elements_crc_fail=elements_crc_fail, cliff_repaired=cliff_repaired,
                               oob_restored=oob_restored, crashed=crashed)
    lo, hi = record["coverage"]
    windows = stratum_windows(ctx["rmap"], stratum, element=anchor["element"])
    touched = {int(b) for b, _ in positions}
    in_element_field = sum(1 for b in touched if any(s <= b < s + L for s, L in windows))
    field_hits = count_field_hits(ctx["rmap"], positions, resolver=ctx["resolver"])
    in_field = sum(field_hits.get(f, 0) for f in STRATUM_FIELDS[stratum])
    return {
        "shape": shape, "region": stratum, "seed": int(seed), "seed_index": int(seed_index),
        "arm": arm, "recall": recall,
        "delta_recall": None if recall is None else round(ctx["clean_recall"] - recall, 6),
        "outcome": outcome, "bits_flipped": int(record["bits_flipped"]),
        "coverage_bytes": int(record["bytes_touched"]),
        "elements_crc_fail": elements_crc_fail, "fallbacks": fallbacks,
        "cliff_repaired": cliff_repaired, "oob_restored": oob_restored,
        # --- context beyond the CSV: where the damage actually landed -------------------
        "element": anchor["element"], "anchor_byte": int(anchor["anchor_byte"]),
        "anchor_window": list(anchor["window"]), "anchor_window_index": anchor["window_index"],
        "coverage_lo": int(lo), "coverage_hi": int(hi), "coverage_span_bytes": int(hi - lo),
        # Two different questions, both worth answering. ELEMENT_FIELD: how much landed in the
        # anchor element's own field (near 0 for the device shapes — they leave the element).
        # FIELD: how much landed in that KIND of field anywhere in the index (much larger, since
        # a row crosses ~30 elements' worth of the same interleaved fields).
        "bytes_in_anchor_element_field": int(in_element_field),
        "bytes_in_anchor_field": int(in_field),
        "field_hits": field_hits,
        "guard_oob_elements": guard_oob, "oob_by_field": oob_detail,
        "clean_recall": ctx["clean_recall"], "adapter": ctx["aname"],
        "shape_kwargs": kwargs, "error": error,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 6) if vals else None


def _p95(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.percentile(vals, 95)), 6) if vals else None


def summarize(records, shapes=SHAPES, strata=STRATA):
    """Per-cell medians/p95 + the smear block, over the SLICE this process ran.

    Iterating the full grid on a shard would mint empty n=0 cells that look like measurements
    that came back blank rather than work another shard is doing.
    """
    cells, smear = {}, {}
    for shape in shapes:
        for stratum in strata:
            grp = [r for r in records if r["shape"] == shape and r["region"] == stratum]
            if grp:
                touched = [r["coverage_bytes"] for r in grp]
                in_elem = [r["bytes_in_anchor_element_field"] for r in grp]
                in_field = [r["bytes_in_anchor_field"] for r in grp]
                fields = {}
                for r in grp:
                    for f, c in r["field_hits"].items():
                        fields[f] = fields.get(f, 0) + c
                smear[f"{shape}|{stratum}"] = {
                    "median_bytes_touched": _median(touched),
                    # Two nested questions. ELEMENT_FIELD: did the damage stay in the exact field
                    # the anchor named? For the device shapes this collapses to ~0 — they walk
                    # straight out of the element. FIELD: did it at least stay in that KIND of
                    # field, anywhere in the index? Much larger, and the honest number to quote
                    # when asking whether a per-field protection scheme would have covered it.
                    "median_bytes_in_anchor_element_field": _median(in_elem),
                    "median_fraction_in_anchor_element_field": _median(
                        [i / t if t else None for i, t in zip(in_elem, touched)]),
                    "median_bytes_in_anchor_field": _median(in_field),
                    "median_fraction_in_anchor_field": _median(
                        [i / t if t else None for i, t in zip(in_field, touched)]),
                    "fields_hit": dict(sorted(fields.items(), key=lambda kv: (-kv[1], kv[0]))),
                }
            for arm in ARMS:
                rows = [r for r in grp if r["arm"] == arm]
                cells[f"{shape}|{stratum}|{arm}"] = {
                    "n": len(rows),
                    "n_crash": sum(1 for r in rows if r["outcome"] == "crash"),
                    "median_recall": _median([r["recall"] for r in rows]),
                    "median_delta_recall": _median([r["delta_recall"] for r in rows]),
                    "p95_delta_recall": _p95([r["delta_recall"] for r in rows]),
                    "median_bits_flipped": _median([r["bits_flipped"] for r in rows]),
                    "median_coverage_bytes": _median([r["coverage_bytes"] for r in rows]),
                    "median_elements_crc_fail": _median([r["elements_crc_fail"] for r in rows]),
                    "median_fallbacks": _median([r["fallbacks"] for r in rows]),
                    "median_cliff_repaired": _median([r["cliff_repaired"] for r in rows]),
                    "median_oob_restored": _median([r["oob_restored"] for r in rows]),
                    "outcomes": {o: sum(1 for r in rows if r["outcome"] == o)
                                 for o in OUTCOMES if any(r["outcome"] == o for r in rows)},
                }
    return cells, smear


def write_csv(path, records):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({c: r.get(c) for c in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# --sweep (default mode)
# ---------------------------------------------------------------------------

def _final_research_gate(ctx, work):
    """Re-search the RESTORED buffer once and require the clean recall back, unchanged.

    The house `assert_no_state_leak` pattern (phase3_e1_vuln.py). A byte compare says the bytes
    match; this says the index still ANSWERS the same, which also covers anything that leaked
    through a path the byte compare cannot see (a stale sidecar being read, a tmp file not
    rewritten). Tolerance is exact for the deterministic stub and loosened for the real C++
    search's float/threading jitter, as in E1.
    """
    ctx["adapter"].deserialize_index(work, ctx["tmp"])
    res = ctx["adapter"].search_corrupted(ctx["tmp"], k=config.K, ef=int(ctx["cfg"]["ef"]),
                                          out_path=ctx["tmp"] + ".clean.ivecs",
                                          timeout=ctx["timeout"])
    after = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
    _sweep_sidecars(ctx["tmp"])
    drift = abs(after - ctx["clean_recall"])
    tol = 1e-9 if ctx["aname"] == "stub" else 1e-6
    if drift > tol:
        raise RuntimeError(
            f"state leaked across the sweep: re-searching the restored buffer gives "
            f"recall@10={after:.9f} vs clean {ctx['clean_recall']:.9f} (drift {drift:.2e} > {tol})")
    return drift


def run(args):
    cfg = dict(SMOKE if args.smoke else FULL)
    if getattr(args, "seeds", None) is not None:
        cfg["seeds"] = int(args.seeds)
    if getattr(args, "ef", None) is not None:
        cfg["ef"] = int(args.ef)
    for knob in ("row_bytes", "n_rows"):
        if getattr(args, knob, None) is not None:
            cfg[knob] = int(getattr(args, knob))
    if getattr(args, "p_in_row", None) is not None:
        cfg["p_in_row"] = float(args.p_in_row)

    # Grid slice for this process. The C++ search is single-threaded, so a full run is sharded
    # across cores by shape/stratum; cell seeds are derived from (root, shape, stratum, i) and
    # never from a loop counter, so a shard reproduces exactly the cells the full grid would.
    shapes = tuple(getattr(args, "shapes", None) or SHAPES)
    strata = tuple(getattr(args, "strata", None) or STRATA)
    sharded = (shapes != SHAPES or strata != STRATA)

    ctx = setup_context(args, cfg)
    out, tag = ctx["out"], ctx["tag"]
    raw_path = os.path.join(out, "raw", f"e6{tag}.records.jsonl")
    done_path = os.path.join(out, "raw", f"e6{tag}.done")

    if args.resume and os.path.exists(done_path):
        log("[e6] --resume: reloading completed shard")
        with open(raw_path) as fh:
            records = [json.loads(line) for line in fh]
        # A shard measured against a different clean baseline (different index / ef / adapter)
        # would silently mix two operating points into one damage matrix.
        prior = next((r["clean_recall"] for r in records if r.get("clean_recall") is not None),
                     None)
        if prior is not None and abs(prior - ctx["clean_recall"]) > RETENTION_TOL:
            raise RuntimeError(
                f"REPORT-AND-STOP: --resume shard was measured at clean recall@10={prior:.6f} "
                f"but this run's clean baseline is {ctx['clean_recall']:.6f} — different index, "
                f"ef, or adapter. Re-run without --resume.")
        leak = None          # not re-verified on resume; the shard's own run already gated it
        research_drift = None
    else:
        work = ctx["clean_buf"].copy()
        scratch = np.empty(work.size, dtype=bool)        # reused by the per-eval leak gate
        records = []
        t0 = time.time()
        if sharded:
            log(f"[e6] shard: shapes={list(shapes)} strata={list(strata)} "
                f"({len(shapes) * len(strata) * int(cfg['seeds']) * len(ARMS)} evals of "
                f"{len(SHAPES) * len(STRATA) * int(cfg['seeds']) * len(ARMS)})")
        with RawWriter(raw_path, done_path=done_path) as w:
            for shape in shapes:
                for stratum in strata:
                    tc = time.time()
                    for i in range(int(cfg["seeds"])):
                        seed = cell_seed(args.seed, shape, stratum, i)
                        anchor = sample_anchor(ctx["rmap"], stratum, seed)
                        # Both arms share the anchor AND the injector seed, so off/on is a
                        # PAIRED comparison of the same physical fault, not two samples.
                        for arm in ARMS:
                            records.append(w.write(measure_cell(
                                ctx, work, shape, stratum, i, seed, anchor, arm,
                                scratch=scratch)))
                    cell = records[-2 * int(cfg["seeds"]):]
                    hist = {o: sum(1 for r in cell if r["outcome"] == o) for o in OUTCOMES}
                    log(f"[e6] {shape:<14} {stratum:<18} "
                        f"{time.time() - tc:6.1f}s  "
                        f"bits~{int(np.median([r['bits_flipped'] for r in cell]))}  "
                        f"{ {k: v for k, v in hist.items() if v} }")
        # Two independent end-of-sweep gates. The byte compare is nearly free but WEAK on its own
        # (the arms loop ends on `on`, whose restore copies clean bytes, so it is 0 by
        # construction — review I1); the per-eval before-restore gate in _verify_and_restore is
        # what actually catches a stray write. The re-search is the house assert_no_state_leak:
        # it proves the restored buffer still answers queries exactly like the clean index.
        leak = int(np.count_nonzero(work != ctx["clean_buf"]))
        if leak:
            raise RuntimeError(f"state leaked across the sweep: {leak} bytes differ from clean")
        research_drift = _final_research_gate(ctx, work)
        log(f"[e6] {len(records)} evals in {time.time() - t0:.1f}s "
            f"(leak {leak} bytes, re-search drift {research_drift:.2e})")

    expected = assert_row_count(records, cfg["seeds"], shapes=shapes, strata=strata)
    cells, smear = summarize(records, shapes=shapes, strata=strata)
    write_csv(os.path.join(out, f"e6_results{tag}.csv"), records)
    summary = {
        "experiment": "e6_shapes",
        "shapes": list(shapes), "strata": list(strata), "arms": list(ARMS),
        "shard": {"is_shard": sharded, "full_grid_shapes": list(SHAPES),
                  "full_grid_strata": list(STRATA),
                  "full_grid_rows": len(SHAPES) * len(STRATA) * int(cfg["seeds"]) * len(ARMS),
                  "out_tag": getattr(args, "out_tag", None)},
        "seeds": int(cfg["seeds"]), "cfg": cfg,
        "clean_recall@10": ctx["clean_recall"],
        "cells": cells, "smear": smear,
        "sanity": {"row_count_ok": len(records) == expected, "expected_rows": expected,
                   "rows": len(records), "state_leak_bytes": leak,
                   # The gate that can actually fail: every eval compared work against clean
                   # BEFORE restoring and required the diff to sit inside the fault footprint.
                   "per_eval_footprint_gate_evals": 0 if leak is None else len(records),
                   "final_research_drift": research_drift,
                   "stub_sandbox_ok": stub_sandbox_ok(ctx["aname"], out)},
        "adapter": ctx["aname"], "meta": ctx["meta"],
    }
    with open(os.path.join(out, f"e6_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[e6] wrote e6_results{tag}.csv ({len(records)} rows) + e6_summary{tag}.json")
    return summary


# ---------------------------------------------------------------------------
# --p3: same 64-bit budget, opposite CRC footprint
# ---------------------------------------------------------------------------

def _p3_positions(rmap, mode, seed):
    """The 64 flip positions for one P3 mode. Pure and deterministic in `seed`.

    burst_one_element    — 64 CONSECUTIVE bits (8 bytes) inside ONE element's 96 B ex_code, via
                           qp.bits.burst_positions: one element fails CRC.
    striped_64_elements  — the SAME in-chunk bit offset in 64 consecutive elements' ex_code, one
                           bit each: 64 elements fail CRC for the identical bit budget. This is
                           the device_column geometry expressed at element granularity.
    """
    rng = np.random.default_rng(seed)
    n = int(rmap["header"]["cur_element_count"])
    _, field_len = layout.element_field_range(rmap, "ex_code", 0)
    field_bits = field_len * 8
    if mode == "burst_one_element":
        if field_bits < P3_BITS:
            raise ValueError(f"ex_code is {field_bits} bits, cannot hold a {P3_BITS}-bit burst")
        e = int(rng.integers(n))
        start, _ = layout.element_field_range(rmap, "ex_code", e)
        rel = int(rng.integers(0, field_bits - P3_BITS + 1))
        return burst_positions(start * 8 + rel, P3_BITS), [e]
    if mode == "striped_64_elements":
        if n < P3_STRIPE_ELEMENTS:
            raise ValueError(f"index has {n} elements, need {P3_STRIPE_ELEMENTS} for the stripe")
        base = int(rng.integers(0, n - P3_STRIPE_ELEMENTS + 1))
        bit = int(rng.integers(0, field_bits))
        elements = list(range(base, base + P3_STRIPE_ELEMENTS))
        positions = []
        for e in elements:
            start, _ = layout.element_field_range(rmap, "ex_code", e)
            positions.append((start + bit // 8, bit % 8))
        return positions, elements
    raise ValueError(f"unknown P3 mode {mode!r}; choose from {P3_MODES}")


def run_p3(args):
    """P3 comparison: identical bit budget, one element vs 64 elements of CRC damage."""
    cfg = dict(SMOKE if args.smoke else FULL)
    if getattr(args, "ef", None) is not None:
        cfg["ef"] = int(args.ef)
    n_seeds = int(getattr(args, "p3_seeds", None) or (2 if args.smoke else 10))

    ctx = setup_context(args, cfg)
    work = ctx["clean_buf"].copy()
    scratch = np.empty(work.size, dtype=bool)
    rows, modes = [], {}
    for mode in P3_MODES:
        per_mode = []
        for i in range(n_seeds):
            seed = lane_seed(cell_seed(args.seed, "p3", mode, i), LANE_P3)
            positions, elements = _p3_positions(ctx["rmap"], mode, seed)
            row = {"mode": mode, "seed": int(seed), "seed_index": i,
                   "bits_flipped": len(positions), "n_elements_targeted": len(elements)}
            for arm in ARMS:
                flip_bits(work, positions)
                try:
                    if arm == "off":
                        ctx["adapter"].deserialize_index(work, ctx["tmp"])
                        res = ctx["adapter"].search_corrupted(
                            ctx["tmp"], k=config.K, ef=int(cfg["ef"]),
                            out_path=ctx["tmp"] + ".ids.ivecs", timeout=ctx["timeout"])
                        stats = None
                    else:
                        guard = RecoveryGuard(ctx["adapter"],
                                              {**GUARD_CFG, "ef": int(cfg["ef"]), "seed": seed},
                                              ctx["rmap"])
                        guard.init_from_clean(ctx["clean_buf"])
                        oob, _ = bounds_check_full(work, ctx["clean_buf"], ctx["rmap"])
                        guard.scrub_only(work)
                        row["oob_restored"] = oob
                        row["cliff_repaired"] = int(guard.counters()["cliff_repaired"])
                        ctx["adapter"].deserialize_index(work, ctx["tmp"])
                        res = ctx["adapter"].query_with_recovery(
                            ctx["tmp"], RECOVERY_MODE, ctx["manifest_path"], k=config.K,
                            ef=int(cfg["ef"]), out_path=ctx["tmp"] + ".rec.ivecs",
                            timeout=ctx["timeout"])
                        stats = res.get("stats") or {}
                    recall = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
                    row[f"recall_{arm}"] = recall
                    row[f"delta_recall_{arm}"] = round(ctx["clean_recall"] - recall, 6)
                    if stats is not None:
                        row["elements_crc_fail"] = int(
                            stats.get("load", {}).get("elements_crc_fail", 0))
                        row["fallbacks"] = int(stats.get("totals", {}).get("fallbacks", 0))
                except Exception as exc:                            # noqa: BLE001
                    row[f"recall_{arm}"] = None
                    row[f"delta_recall_{arm}"] = None
                    row[f"error_{arm}"] = repr(exc)
                finally:
                    _verify_and_restore(work, ctx["clean_buf"], positions, arm, scratch=scratch)
                    _sweep_sidecars(ctx["tmp"])
            row["outcome_on"] = classify_outcome(
                arm="on", recall=row.get("recall_on"), clean_recall=ctx["clean_recall"],
                elements_crc_fail=row.get("elements_crc_fail"),
                cliff_repaired=row.get("cliff_repaired"), oob_restored=row.get("oob_restored"),
                crashed=row.get("recall_on") is None)
            per_mode.append(row)
            rows.append(row)
        modes[mode] = {
            "n_seeds": n_seeds,
            "bits_flipped": int(np.median([r["bits_flipped"] for r in per_mode])),
            "median_elements_targeted": _median([r["n_elements_targeted"] for r in per_mode]),
            "median_delta_recall_off": _median([r.get("delta_recall_off") for r in per_mode]),
            "median_delta_recall_on": _median([r.get("delta_recall_on") for r in per_mode]),
            "median_elements_crc_fail": _median([r.get("elements_crc_fail") for r in per_mode]),
            "median_fallbacks": _median([r.get("fallbacks") for r in per_mode]),
            "outcomes_on": {o: sum(1 for r in per_mode if r.get("outcome_on") == o)
                            for o in OUTCOMES
                            if any(r.get("outcome_on") == o for r in per_mode)},
            "n_crash": sum(1 for r in per_mode if r.get("recall_off") is None
                           or r.get("recall_on") is None),
        }
        m = modes[mode]
        log(f"[e6-p3] {mode:<22} dRecall off={m['median_delta_recall_off']} "
            f"on={m['median_delta_recall_on']} crc_fail={m['median_elements_crc_fail']}")

    leak = int(np.count_nonzero(work != ctx["clean_buf"]))
    if leak:
        raise RuntimeError(f"P3 state leak: {leak} bytes differ from clean")
    blob = {"experiment": "e6_p3", "bits_per_injection": P3_BITS,
            "stripe_elements": P3_STRIPE_ELEMENTS, "n_seeds": n_seeds,
            "clean_recall@10": ctx["clean_recall"], "modes": modes, "rows": rows,
            "sanity": {"state_leak_bytes": leak}, "adapter": ctx["aname"], "meta": ctx["meta"]}
    path = os.path.join(ctx["out"], f"e6_p3{ctx['tag']}.json")
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)
    log(f"[e6-p3] wrote {path}")
    return blob


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, choices=["stub", "real", "auto"],
                    help="stub (dev) / real (x86) / auto (default: real if binaries else stub)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE['seeds']} seeds + small ef; outputs to artifacts_smoke/")
    ap.add_argument("--resume", action="store_true",
                    help="reload a completed raw shard instead of re-running the sweep")
    ap.add_argument("--p3", action="store_true",
                    help="run the P3 burst-vs-stripe comparison instead of the damage matrix")
    ap.add_argument("--seeds", type=int, default=None, help="replicates per (shape, stratum) cell")
    ap.add_argument("--shapes", type=csv_list("shape", SHAPES), default=None,
                    help=f"comma-separated subset of {list(SHAPES)} — shards the grid across "
                         f"processes (the C++ search is single-threaded). Pair with --out-tag.")
    ap.add_argument("--strata", type=csv_list("stratum", STRATA), default=None,
                    help=f"comma-separated subset of {list(STRATA)}; same sharding purpose. "
                         f"Cell seeds are (root, shape, stratum, i)-derived, so a shard's "
                         f"anchors and flips are identical to the full grid's.")
    ap.add_argument("--p3-seeds", dest="p3_seeds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED, help="root seed")
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--row-bytes", dest="row_bytes", type=int, default=None,
                    help="DRAM row modeling unit in bytes (default 8192 — a knob, not a fact)")
    ap.add_argument("--n-rows", dest="n_rows", type=int, default=None,
                    help="rows a device_column stripe runs through (default 512)")
    ap.add_argument("--p-in-row", dest="p_in_row", type=float, default=None,
                    help="per-bit flip probability inside a failed row (default 0.5 = a fully "
                         "failed row); lower values model a partially-failed row")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-search seconds; a hang under corruption records as crash")
    ap.add_argument("--weights", default=None,
                    help=f"shape_weights.json for provenance (default {DEFAULT_WEIGHTS})")
    ap.add_argument("--out-tag", dest="out_tag", default=None,
                    help="filename suffix so parallel/variant runs never clobber each other")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                                "phase3", "e6")
    if args.p3:
        run_p3(args)
        log("\nE6 P3 OK")
    else:
        run(args)
        log("\nE6 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
