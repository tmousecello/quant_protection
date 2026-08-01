#!/usr/bin/env python3
"""Phase 3 — E6: DRAM fault SHAPE x index REGION damage matrix (CIDR spec E1).

E1 asked "which bits matter"; E5/Experiment-B asked "does the recovery stack hold". E6 asks the
question the DRAM literature forces on us: real memory does not fail one bit at a time. A failed
row decoder takes out a whole ~8 KB row; a bad I/O gate / TSV takes out one bit lane down a
column of rows. So the damage a fault does to a RaBitQ index is a function of TWO variables — the
physical SHAPE of the fault and the index REGION it lands in — and this driver measures the full
cross product, with the two-layer recovery stack OFF and ON.

  shapes  {single_cell, device_row, device_column}                       (qp.faults, spec §B)
  strata  {rotation_centroids, ex_code, bin_code, links, factors, ids}   (the anchor's region)
  seeds   30 (--smoke: 2)
  arms    {off, on}                                                      -> 3*6*30*2 = 1080 evals

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
        `guard._pre_search_scrub(buf)` (cliff majority-vote repair + ex CRC + pointer
        bounds-check/restore) and `adapter.query_with_recovery(tmp, "fallback_eb", manifest)`
        with a CRC manifest written once from the clean index. Recall is recomputed from the
        returned ids — never taken from the C++ side.

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
import subprocess
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
STRATA = ("rotation_centroids", "ex_code", "bin_code", "links", "factors", "ids")
GLOBAL_STRATA = ("rotation_centroids",)
ARMS = ("off", "on")
OUTCOMES = ("crash", "repaired", "tolerated", "silent_wrong")

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

    Global strata ignore `element`: `rotation_centroids` is the union of the two global decode
    structures, centroids first (8192 B on the stub) then rotation (64 B) — the 8192:64 byte
    weighting the spec calls for falls straight out of sampling uniformly over the union's bytes.
    Per-element strata REQUIRE `element` and go through layout.element_field_range, so the
    element stride (e*size_data_per_element) is never re-derived here.
    """
    if stratum == "rotation_centroids":
        out = []
        for name in ("centroids", "rotation"):
            r = next((x for x in rmap["regions"] if x["name"] == name), None)
            if r is None or r["byte_start"] is None:
                raise ValueError(f"stratum {stratum!r}: region {name!r} is not located in the "
                                 f"region map (was the map built without file_size?)")
            out.append((int(r["byte_start"]), int(r["byte_len"])))
        return out
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
    """The 4-class outcome for one eval. Thresholds are RETENTION_TOL / DELTA_TOL above.

      crash        the search subprocess died or timed out (or returned nothing measurable).
      repaired     arm ON only: recall held AND a recovery layer actually acted (cliff
                   majority-vote repair or a pointer bounds-check restore) AND the CRC scan
                   found nothing left to flag — i.e. the damage was put back, not worked around.
      tolerated    recall held but the ex-data CRC still flags elements (EB-fallback carried it
                   rather than repairing), OR the damage simply moved recall by < DELTA_TOL.
      silent_wrong recall dropped past both bands with nothing detected — the dangerous case.

    Arm OFF passes None for the recovery counters (nothing was measured, as opposed to measured
    zero) and can therefore never be classified `repaired`.
    """
    if crashed or recall is None:
        return "crash"
    crc_fail = 0 if elements_crc_fail is None else int(elements_crc_fail)
    repaired_ct = 0 if cliff_repaired is None else int(cliff_repaired)
    oob_ct = 0 if oob_restored is None else int(oob_restored)
    held = recall >= clean_recall - RETENTION_TOL
    if arm == "on" and held and crc_fail == 0 and (repaired_ct > 0 or oob_ct > 0):
        return "repaired"
    if (held and crc_fail > 0) or (clean_recall - recall) < DELTA_TOL:
        return "tolerated"
    return "silent_wrong"


# ---------------------------------------------------------------------------
# Sanity gates
# ---------------------------------------------------------------------------

def assert_stub_sandbox(aname, out_dir):
    """House convention: stub output NEVER lands under artifacts/ (only artifacts_smoke/).

    A stub number that ends up in the real artifacts tree is indistinguishable from a measured
    one three months later, so this is a hard stop rather than a warning.
    """
    if aname != "stub":
        return
    real_root = os.path.join(os.path.abspath(config.ROOT), "artifacts")
    target = os.path.abspath(out_dir)
    if target == real_root or target.startswith(real_root + os.sep):
        raise RuntimeError(
            f"REPORT-AND-STOP: stub adapter would write documented-fake results into the real "
            f"artifacts tree ({target}). Use --smoke (-> artifacts_smoke/) or an explicit --out.")


def assert_row_count(records, seeds):
    """The grid must be complete: shapes x strata x seeds x arms, no silently dropped cell."""
    expected = len(SHAPES) * len(STRATA) * int(seeds) * len(ARMS)
    assert len(records) == expected, (
        f"row count {len(records)} != expected {expected} "
        f"({len(SHAPES)} shapes x {len(STRATA)} strata x {seeds} seeds x {len(ARMS)} arms)")
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

    log(f"[e6] adapter={aname} out={out} clean@10={clean_recall:.5f} "
        f"n={rmap['header']['cur_element_count']} ef={cfg['ef']} seeds={cfg['seeds']}")
    return {"adapter": adapter, "aname": aname, "out": out, "tag": tag, "rmap": rmap,
            "clean_buf": clean_buf, "gt": gt, "tmp": tmp, "manifest_path": manifest_path,
            "clean_recall": clean_recall, "meta": meta, "cfg": cfg,
            "timeout": args.timeout, "resolver": make_field_resolver(rmap)}


def _sweep_sidecars(tmp):
    for stale in (tmp + ".ids.ivecs", tmp + ".ids.ivecs.dist.fvecs",
                  tmp + ".rec.ivecs", tmp + ".rec.ivecs.dist.fvecs"):
        try:
            os.remove(stale)
        except OSError:
            pass


def _restore(work, clean_buf, positions, arm):
    """Undo one injection and verify locally that the buffer is back to clean.

    Arm OFF re-XORs the exact positions (the phase1 D1 identity guarantee). Arm ON CANNOT: the
    RecoveryGuard writes into `work` itself (rotation majority-vote repair, pointer restores), so
    re-XORing the injected positions would leave the guard's edits behind AND re-corrupt what it
    fixed; the clean bytes are copied back wholesale instead. Either way the touched bytes are
    then compared against clean — a leak would silently bias every later eval.
    """
    if arm == "off":
        faults.restore(work, positions)
    else:
        np.copyto(work, clean_buf)
    if positions:
        idx = np.fromiter({int(b) for b, _ in positions}, dtype=np.int64)
        drift = int(np.count_nonzero(work[idx] != clean_buf[idx]))
        if drift:
            raise RuntimeError(f"state leaked across evals: {drift} touched bytes differ from "
                               f"clean after restore (arm={arm})")


# ---------------------------------------------------------------------------
# One eval
# ---------------------------------------------------------------------------

def measure_cell(ctx, work, shape, stratum, seed_index, seed, anchor, arm):
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
    crashed, error = False, ""
    try:
        if guard is not None:
            guard._pre_search_scrub(work)                # cliff vote/repair + ex CRC + bounds
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
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            crashed, error = True, repr(exc)
        else:
            recall = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
            if stats is not None:
                elements_crc_fail = int(stats.get("load", {}).get("elements_crc_fail", 0))
                fallbacks = int(stats.get("totals", {}).get("fallbacks", 0))
        if guard is not None:
            ctr = guard.counters()                       # counters are valid even after a crash
            cliff_repaired = int(ctr["cliff_repaired"])
            oob_restored = int(ctr["oob_elements"])
    finally:
        _restore(work, ctx["clean_buf"], positions, arm)
        _sweep_sidecars(ctx["tmp"])

    outcome = classify_outcome(arm=arm, recall=recall, clean_recall=ctx["clean_recall"],
                               elements_crc_fail=elements_crc_fail, cliff_repaired=cliff_repaired,
                               oob_restored=oob_restored, crashed=crashed)
    lo, hi = record["coverage"]
    windows = stratum_windows(ctx["rmap"], stratum, element=anchor["element"])
    touched = {int(b) for b, _ in positions}
    in_stratum = sum(1 for b in touched if any(s <= b < s + L for s, L in windows))
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
        "bytes_in_anchor_stratum": int(in_stratum),
        "field_hits": count_field_hits(ctx["rmap"], positions, resolver=ctx["resolver"]),
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


def summarize(records):
    cells, smear = {}, {}
    for shape in SHAPES:
        for stratum in STRATA:
            grp = [r for r in records if r["shape"] == shape and r["region"] == stratum]
            if grp:
                touched = [r["coverage_bytes"] for r in grp]
                inside = [r["bytes_in_anchor_stratum"] for r in grp]
                fields = {}
                for r in grp:
                    for f, c in r["field_hits"].items():
                        fields[f] = fields.get(f, 0) + c
                smear[f"{shape}|{stratum}"] = {
                    "median_bytes_touched": _median(touched),
                    "median_bytes_in_anchor_stratum": _median(inside),
                    # The headline of the smear finding: for the device shapes this is ~0, i.e.
                    # almost none of the damage lands in the field the stratum aimed at.
                    "median_fraction_in_anchor_stratum": _median(
                        [i / t if t else None for i, t in zip(inside, touched)]),
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

def run(args):
    cfg = dict(SMOKE if args.smoke else FULL)
    if getattr(args, "seeds", None) is not None:
        cfg["seeds"] = int(args.seeds)
    if getattr(args, "ef", None) is not None:
        cfg["ef"] = int(args.ef)
    for knob in ("row_bytes", "n_rows"):
        if getattr(args, knob, None) is not None:
            cfg[knob] = int(getattr(args, knob))

    ctx = setup_context(args, cfg)
    out, tag = ctx["out"], ctx["tag"]
    raw_path = os.path.join(out, "raw", f"e6{tag}.records.jsonl")
    done_path = os.path.join(out, "raw", f"e6{tag}.done")

    if args.resume and os.path.exists(done_path):
        log("[e6] --resume: reloading completed shard")
        with open(raw_path) as fh:
            records = [json.loads(line) for line in fh]
        leak = None          # not re-verified on resume; the shard's own run already gated it
    else:
        work = ctx["clean_buf"].copy()
        records = []
        t0 = time.time()
        with RawWriter(raw_path, done_path=done_path) as w:
            for shape in SHAPES:
                for stratum in STRATA:
                    tc = time.time()
                    for i in range(int(cfg["seeds"])):
                        seed = cell_seed(args.seed, shape, stratum, i)
                        anchor = sample_anchor(ctx["rmap"], stratum, seed)
                        for arm in ARMS:
                            records.append(w.write(measure_cell(
                                ctx, work, shape, stratum, i, seed, anchor, arm)))
                    cell = records[-2 * int(cfg["seeds"]):]
                    hist = {o: sum(1 for r in cell if r["outcome"] == o) for o in OUTCOMES}
                    log(f"[e6] {shape:<14} {stratum:<18} "
                        f"{time.time() - tc:6.1f}s  "
                        f"bits~{int(np.median([r['bits_flipped'] for r in cell]))}  "
                        f"{ {k: v for k, v in hist.items() if v} }")
        # Whole-buffer leak check once at the end: the per-eval check covers the touched bytes,
        # this catches anything that wrote OUTSIDE them (e.g. a guard reload gone wrong).
        leak = int(np.count_nonzero(work != ctx["clean_buf"]))
        if leak:
            raise RuntimeError(f"state leaked across the sweep: {leak} bytes differ from clean")
        log(f"[e6] {len(records)} evals in {time.time() - t0:.1f}s (state leak {leak} bytes)")

    expected = assert_row_count(records, cfg["seeds"])
    cells, smear = summarize(records)
    write_csv(os.path.join(out, f"e6_results{tag}.csv"), records)
    summary = {
        "experiment": "e6_shapes",
        "shapes": list(SHAPES), "strata": list(STRATA), "arms": list(ARMS),
        "seeds": int(cfg["seeds"]), "cfg": cfg,
        "clean_recall@10": ctx["clean_recall"],
        "cells": cells, "smear": smear,
        "sanity": {"row_count_ok": len(records) == expected, "expected_rows": expected,
                   "rows": len(records), "state_leak_bytes": leak,
                   "stub_sandbox_ok": True},
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
                        guard._pre_search_scrub(work)
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
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    row[f"recall_{arm}"] = None
                    row[f"delta_recall_{arm}"] = None
                    row[f"error_{arm}"] = repr(exc)
                finally:
                    # arm on lets the guard write into `work`; copy clean back rather than
                    # re-XOR (same reasoning as _restore).
                    if arm == "off":
                        flip_bits(work, positions)
                    else:
                        np.copyto(work, ctx["clean_buf"])
                    _sweep_sidecars(ctx["tmp"])
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
    ap.add_argument("--p3-seeds", dest="p3_seeds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED, help="root seed")
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--row-bytes", dest="row_bytes", type=int, default=None,
                    help="DRAM row modeling unit in bytes (default 8192 — a knob, not a fact)")
    ap.add_argument("--n-rows", dest="n_rows", type=int, default=None,
                    help="rows a device_column stripe runs through (default 512)")
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
