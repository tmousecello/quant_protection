"""Deterministic STUB RaBitQ adapter — dev-machine substitute for the x86-only real adapter.

The real RaBitQ search runs in an AVX2/AVX512 C++ binary that will not build on this arm64 dev
machine, so the E1/E3a/E3b runners are developed and unit-tested against this stub. It mirrors the
real adapter's interface EXACTLY (same function names/returns, so the runner switches stub<->real
via qp.rabitq.registry with no code change) and returns DETERMINISTIC fake results — never real
recall numbers. Its only jobs:
  * serve a synthetic index FILE whose 156-byte header parses and whose region map is byte-exact
    (so injection confinement and the per-element math can be verified offline), and
  * map "which structure's bytes were corrupted" -> a deterministic outcome that exercises every
    branch of the runner's aggregation (silent-collapse / harmful / nan-inf / crash / clean).

It is faiss-free and import-safe on arm64. NOTHING here is a scientific measurement.
"""
import os
import struct
import subprocess
import tempfile

import numpy as np

from qp import config
from qp.rabitq import layout

EXPECTED_CLEAN_RECALL10 = 0.983       # mirrors the real adapter's reference anchor
CLEAN_RECALL = 0.98                   # the stub's deterministic "clean" recall@10
NQ = 64                               # synthetic query count
INDEX_PATH = os.path.join(tempfile.gettempdir(), "qp_rabitq_stub.index")

# Synthetic SIFT-b7-shaped geometry (small element count so a full file is a few tens of KB).
# The upper-link stream is DERIVED from _element_levels, not a byte constant: E8 studies the
# `[uint32 len][len bytes]` record framing, which cannot exist in a zero-length region.
_GEOM = dict(dim=128, padded_dim=128, ex_bits=6, num_cluster=16, cur_element_count=64,
             M=16, maxM=16, maxM0=32)


def _element_levels(n):
    """Deterministic HNSW level per element, shaped like a real index (most elements at level 0).

    Real indexes are dominated by len==0 records (937,001 of 1,000,000 on SIFT b=7), so the stub
    keeps that skew: every 8th element reaches level 1 and every 32nd reaches level 2. maxlevel is
    derived from this list, so the framing guard's bound (0 or slpe*level, level <= maxlevel) is
    exercised against the same invariant the real index satisfies.
    """
    return [2 if i % 32 == 0 else (1 if i % 8 == 0 else 0) for i in range(int(n))]


def _upper_link_stream(levels, size_links_per_element):
    """The upper_links block exactly as save() writes it (hnsw.hpp L655-662)."""
    out = bytearray()
    for lvl in levels:
        link_list_size = size_links_per_element * lvl if lvl > 0 else 0
        out += struct.pack("<I", link_list_size)
        out += bytes(link_list_size)                  # zero payload; only the framing matters here
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


def _build_header(g):
    """A consistent 156-byte header dict (sizes derived from layout formulas, never hand-picked)."""
    spe_links = layout.size_links_level0(g["maxM0"])
    off_bin = spe_links + layout.PID + layout.PID
    bin_bytes = layout.bin_data_bytes(g["padded_dim"])
    off_ex = off_bin + bin_bytes
    ex_bytes = layout.ex_data_bytes(g["padded_dim"], g["ex_bits"])
    spe = spe_links + layout.PID + layout.PID + bin_bytes + ex_bytes
    return {
        "max_elements": g["cur_element_count"], "cur_element_count": g["cur_element_count"],
        "dim": g["dim"], "padded_dim": g["padded_dim"], "num_cluster": g["num_cluster"],
        "ex_bits": g["ex_bits"], "size_bin_data": bin_bytes, "size_ex_data": ex_bytes,
        "size_links_level0": spe_links, "offsetBinData": off_bin, "offsetExData": off_ex,
        "label_offset": 0, "size_data_per_element": spe, "size_links_per_element": spe_links,
        "maxlevel": max(_element_levels(g["cur_element_count"])),
        "enterpoint_node": 0, "M": g["M"], "maxM": g["maxM"], "maxM0": g["maxM0"],
        "mult": 0.5, "ef_construction": 200,
    }


def _pack_header(hdr):
    return b"".join(struct.pack(code, hdr[name]) for name, code in layout.HEADER_FIELDS)


def _build_pristine():
    """The pristine synthetic index buffer: real header + zero body, self-consistent file size."""
    g = _GEOM
    hdr = _build_header(g)
    centroids = g["num_cluster"] * g["padded_dim"] * layout.FLOAT
    level0 = g["cur_element_count"] * hdr["size_data_per_element"]
    rot = layout.rotation_bytes(g["padded_dim"])
    upper = _upper_link_stream(_element_levels(g["cur_element_count"]),
                               hdr["size_links_per_element"])
    total = (layout.HEADER_BYTES + centroids + level0 + upper.size + rot)
    buf = np.zeros(total, dtype=np.uint8)
    buf[:layout.HEADER_BYTES] = np.frombuffer(_pack_header(hdr), dtype=np.uint8)
    upper_start = layout.HEADER_BYTES + centroids + level0
    buf[upper_start:upper_start + upper.size] = upper
    return buf


_PRISTINE = _build_pristine()


def _materialize_index_path():
    """Keep the on-disk INDEX_PATH in sync with _PRISTINE.

    Callers that read `adapter.INDEX_PATH` as a FILE (phase3_e7_episode.py's panel-B reload
    timing, provenance's index sha256) got whatever an earlier run happened to leave in /tmp.
    Nothing ever wrote it deliberately, so a geometry change left a stale file of the OLD size
    behind and search_corrupted then read it as "size changed => corrupted" and raised a
    simulated SIGSEGV — a failure with no connection to what the caller was testing.

    Written via a unique temp file + atomic rename so concurrent test processes never observe a
    half-written index.
    """
    try:
        if os.path.isfile(INDEX_PATH) and os.path.getsize(INDEX_PATH) == _PRISTINE.size:
            if np.array_equal(np.fromfile(INDEX_PATH, dtype=np.uint8), _PRISTINE):
                return
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(INDEX_PATH), suffix=".stubidx")
        os.close(fd)
        _PRISTINE.tofile(tmp)
        os.replace(tmp, INDEX_PATH)
    except OSError:
        pass          # a read-only/full tmpdir must not break importing the stub


_materialize_index_path()


# --- interface mirror: file substrate ----------------------------------------

def binaries_built():
    """The stub is always available (no x86 binaries needed)."""
    return True


def serialize_index(path=None):
    """A writable copy of the pristine synthetic buffer (ignores `path`)."""
    return _PRISTINE.copy()


def deserialize_index(buf, out_path):
    np.asarray(buf, dtype=np.uint8).tofile(out_path)
    return out_path


def byte_size(path=None):
    return int(_PRISTINE.size)


def read_serialized_range(byte_start, byte_len, path=None):
    """Read byte_len bytes at byte_start from the clean source (ignores `path`).

    _PRISTINE stands in for persistent storage here: on the real adapter this is a
    fresh file read every call (never cached), because the clean source's value is
    that it lives outside DRAM. Returns a fresh copy so callers can't alias it.
    """
    bs, bl = int(byte_start), int(byte_len)
    if bs < 0 or bs + bl > int(_PRISTINE.size):
        raise IOError(f"range ({bs}, {bl}) outside pristine buffer of {_PRISTINE.size} B")
    return _PRISTINE[bs:bs + bl].copy()


def read_header(path=None):
    return layout.parse_header(bytes(_PRISTINE[:layout.HEADER_BYTES]))


def region_map(path=None):
    return layout.serialized_region_map(read_header(), file_size=int(_PRISTINE.size))


def load_query():
    rng = np.random.default_rng(config.SEED)
    return rng.standard_normal((NQ, _GEOM["dim"]), dtype=np.float32)


def load_groundtruth():
    """Distinct ids per query so recall is unambiguous: gt[i] = [i*100 .. i*100+99]."""
    base = (np.arange(NQ) * 100)[:, None]
    return (base + np.arange(100)[None, :]).astype(np.int32)


# --- deterministic outcome model ---------------------------------------------
# structure base-name -> (recall_as_fraction_of_clean, mode). mode drives the branch exercised:
#   "ok"       finite distances, recall = frac*clean (collapse if frac<0.5, else harmful/clean)
#   "naninf"   finite recall number but a non-finite distance -> classify_failure -> NAN_INF
#   "crash"    raise CalledProcessError(-11) -> the sweep records a `crash` (segfault analog)
_OUTCOME = {
    "rotation": (0.20, "ok"),     # global decode basis mismatch -> silent collapse (the headline)
    "centroids": (0.60, "ok"),    # harmful, non-collapse (retention 0.6 > 0.5)
    "bin_factors": (0.85, "ok"),  # per-vector scale/EB -> harmful, non-collapse
    "ex_factors": (0.0, "naninf"),  # refinement scale -> non-finite distance estimate
    "bin_code": (1.00, "ok"),     # one dim's sign bit -> benign/clean
    "ex_code": (0.95, "ok"),      # extension bit -> mild
    "cluster_id": (0.90, "ok"),
    "label": (0.90, "ok"),
    "links": (0.90, "ok"),
    "header": (0.0, "crash"),
    # E8: the upper_links block splits into two structures with OPPOSITE outcomes. A flip in a
    # record's uint32 length prefix changes how many bytes that record consumes, desynchronizing
    # every later record AND the rotation read that follows them (hnsw.hpp L737-750, L763) ->
    # silent collapse. A flip in the payload perturbs one upper-level edge -> benign. Both
    # fractions mirror the measured real-index behaviour (phase3_e8_framing.py).
    "upper_link_len": (0.0, "ok"),
    "upper_links": (1.00, "ok"),
}


def _upper_len_offsets():
    """Absolute offsets of the upper-link length words in _PRISTINE (cached; pure geometry)."""
    global _UPPER_LEN_OFFSETS
    if _UPPER_LEN_OFFSETS is None:
        rmap = region_map()
        up = next(r for r in rmap["regions"] if r["name"] == "upper_links")
        recs = layout.parse_upper_link_records(
            _PRISTINE, rmap["header"]["cur_element_count"], up["byte_start"], up["byte_len"])
        _UPPER_LEN_OFFSETS = frozenset(
            o + j for o in recs["len_offsets"] for j in range(layout.LEN_WORD_BYTES))
    return _UPPER_LEN_OFFSETS


_UPPER_LEN_OFFSETS = None


def _resolve_structure(off):
    """Base structure name for an absolute byte offset, via the region map + per-element math."""
    rmap = region_map()
    hdr = rmap["header"]
    level0 = next(r for r in rmap["regions"] if r["name"] == "level0")
    l0s, l0e = level0["byte_start"], level0["byte_start"] + level0["byte_len"]
    if l0s <= off < l0e:
        within = (off - l0s) % hdr["size_data_per_element"]
        for er in layout.element_regions(hdr["padded_dim"], hdr["ex_bits"], hdr["maxM0"],
                                         off_bin=hdr["offsetBinData"], off_ex=hdr["offsetExData"]):
            if er["offset_in_element"] <= within < er["offset_in_element"] + er["byte_len"]:
                return er["name"]
    for r in rmap["regions"]:
        if r["name"] in ("level0",) or r["name"].startswith("elem0."):
            continue
        if r["byte_start"] is not None and r["byte_start"] <= off < r["byte_start"] + r["byte_len"]:
            # Split upper_links into its framing and payload halves — they behave oppositely.
            if r["name"] == "upper_links" and off in _upper_len_offsets():
                return "upper_link_len"
            return r["name"]
    return "header"   # offsets inside the 156-B header / unmapped -> crash structure


def _is_pointer_high(off, structure):
    """uint32 pointer high byte lane (off%4 in {2,3}) -> the big-jump / OOB-crash class."""
    if structure not in ("links", "cluster_id", "label"):
        return False
    return (off % layout.FLOAT) >= 2


def _ids_for_recall(target, gt, k):
    """Build (nq, k) ids whose qp.metrics.recall_at_k equals `target` exactly (deterministic)."""
    nq = gt.shape[0]
    total_hits = int(round(target * nq * k))
    base, rem = divmod(total_hits, nq)
    ids = np.full((nq, k), -1, dtype=np.int32)        # -1 = non-matching slot
    for i in range(nq):
        h = min(k, base + (1 if i < rem else 0))
        if h:
            ids[i, :h] = gt[i, :h]
        for j in range(h, k):
            ids[i, j] = 9_000_000 + i * k + j         # distinct non-gt id
    return ids


def query_ids(index_path=None, k=None, ef=2000, out_path=None, query_f=None, gt_f=None,
              timeout=None):
    res = search_corrupted(index_path, k=k, ef=ef, out_path=out_path,
                           query_f=query_f, gt_f=gt_f, timeout=timeout)
    return res["ids"], res["cpp_recall"]


def search_corrupted(index_path, k=None, ef=2000, out_path=None, query_f=None, gt_f=None,
                     timeout=None):
    """Deterministic stand-in for the real subprocess search. Returns {ids, cpp_recall, distances}.

    Reads the (possibly corrupted) file at `index_path`, diffs it against the pristine synthetic
    buffer, maps the changed byte(s) to a structure, and returns the structure's pre-assigned
    deterministic outcome. A pristine file -> clean recall. A pointer high-lane or header flip
    raises CalledProcessError to faithfully drive the runner's crash branch.
    """
    k = config.K if k is None else int(k)
    gt = load_groundtruth()
    buf = np.fromfile(index_path, dtype=np.uint8)
    if buf.size == _PRISTINE.size:
        changed = list(map(int, np.nonzero(buf != _PRISTINE)[0]))
    else:                                             # size change is itself a corruption
        changed = [0]

    if not changed:                                   # pristine -> clean baseline
        ids = _ids_for_recall(CLEAN_RECALL, gt, k)
        return {"ids": ids, "cpp_recall": float(np.round(_recall(ids, gt, k), 12)),
                "distances": _distances(ids, finite=True)}

    # Worst (lowest-frac / crash) structure among the changed bytes drives the outcome.
    structures = [layout.base_name(_resolve_structure(o)) for o in changed]
    if any(_is_pointer_high(o, s) for o, s in zip(changed, structures)) \
            or any(_OUTCOME.get(s, ("", "crash"))[1] == "crash" for s in structures):
        raise subprocess.CalledProcessError(-11, ["exp_dumpids", str(index_path)],
                                            output="", stderr="stub: simulated SIGSEGV")

    fracs = [(_OUTCOME.get(s, (1.0, "ok"))) for s in structures]
    frac = min(f for f, _ in fracs)
    mode = "naninf" if any(m == "naninf" for _, m in fracs) else "ok"
    ids = _ids_for_recall(frac * CLEAN_RECALL, gt, k)
    return {"ids": ids, "cpp_recall": float(np.round(_recall(ids, gt, k), 12)),
            "distances": _distances(ids, finite=(mode != "naninf"))}


def clean_baseline_recall(index_path=None):
    gt = load_groundtruth()
    ids = _ids_for_recall(CLEAN_RECALL, gt, config.K)
    return float(_recall(ids, gt, config.K))


# --- small helpers (kept faiss-free; recall uses the SAME formula qp.metrics does) ------------

def _recall(ids, gt, k):
    from qp import metrics
    return metrics.recall_at_k(ids, gt, k)


def search_with_eb_fallback(index_path, fraction, seed=0, *, crc_manifest=None, k=None,
                            ef=2000, out_path=None, query_f=None, gt_f=None, timeout=None,
                            crc_mode="load", crc_timer=False):
    """Stub stand-in for the EB-fallback search path (E5 slope layer / Option B).

    Same signature as the real adapter. With `crc_manifest` it routes through the Option-B
    query_with_recovery (real manifest verification, fake recall); without one it keeps the
    legacy stub behavior (delegate to search_corrupted) so pre-Option-B E5 stub tests are
    unchanged. Tags _eb_path=True so unit tests can assert the EB branch was taken.
    """
    if crc_manifest:
        res = query_with_recovery(index_path, "fallback_eb", crc_manifest, k=k, ef=ef,
                                  out_path=out_path, query_f=query_f, gt_f=gt_f,
                                  timeout=timeout, crc_mode=crc_mode, crc_timer=crc_timer)
        res["distances"] = None
    else:
        res = search_corrupted(index_path, k=k, ef=ef, out_path=out_path,
                               query_f=query_f, gt_f=gt_f, timeout=timeout)
    res["_eb_path"] = True
    res["eb_fraction"] = float(fraction)
    return res


# --- Option B: recovery-mode query (real CRC verification, FAKE recall model) -----------------

RECOVERY_MODES = ("none", "drop", "fallback_eb")
CRC_MODES = ("load", "lazy")

# recall-retention slopes per recovery mode as a function of the corrupted-element fraction f.
# CHOSEN, not measured: they exist only so the driver's gate/sweep plumbing has a deterministic
# outcome with the structurally-correct ordering (fallback_eb >= drop > none for f>0; all equal
# clean at f=0). NOTHING here is a scientific measurement.
_RECOVERY_SLOPE = {"none": 1.5, "drop": 0.6, "fallback_eb": 0.45}


def _corrupt_ex_elements(buf):
    """Element indices whose ex_code bytes differ from the pristine buffer (stub ground truth)."""
    rmap = region_map()
    hdr = rmap["header"]
    n, spe = hdr["cur_element_count"], hdr["size_data_per_element"]
    s0, flen = layout.element_field_range(rmap, "ex_code", 0)
    changed = np.nonzero(buf != _PRISTINE)[0]
    out = set()
    for off in changed:
        d, r = divmod(int(off) - s0, spe)
        if 0 <= d < n and 0 <= r < flen:
            out.add(int(d))
    return sorted(out)


def query_with_recovery(index_path, recovery="none", crc_manifest=None, *, k=None, ef=2000,
                        out_path=None, stats_json=None, query_f=None, gt_f=None, timeout=None,
                        crc_mode="load", crc_timer=False):
    """Deterministic stand-in for exp_dumpids --recovery (same contract as the real adapter).

    The CRC side is REAL: with a manifest, the failing-element set comes from
    qp.rabitq.crc_manifest.verify_buffer on the actual file bytes (so manifest format,
    geometry echo, and verification code paths are genuinely exercised offline); recovery=none
    diffs against the pristine buffer instead (no CRC check, mirroring the C++ none mode).
    Only the fraction -> recall mapping is a documented fake (_RECOVERY_SLOPE).

    `crc_mode` is accepted and echoed, but the stub deliberately does NOT model it. There is no
    real search here, so nothing knows which elements a query would have consulted — and the
    on-access counters are exactly the quantity that depends on that. Under crc_mode="lazy" the
    load-scan totals are zeroed (matching the binary: no scan ran) and every on-access counter
    is None, i.e. "not knowable offline", never a plausible-looking invented number. What the
    stub does check is that the flag reaches here and nothing breaks; the counters themselves
    are gated on the real adapter (phase3_expb_recovery.py --lazy-gate).
    """
    from qp.rabitq import crc_manifest as cm

    k = config.K if k is None else int(k)
    if recovery not in RECOVERY_MODES:
        raise ValueError(f"recovery must be one of {RECOVERY_MODES}, got {recovery!r}")
    if recovery != "none" and not crc_manifest:
        raise ValueError(f"--recovery {recovery} requires a crc_manifest path")
    if crc_mode not in CRC_MODES:
        raise ValueError(f"crc_mode must be one of {CRC_MODES}, got {crc_mode!r}")
    if recovery == "none" and (crc_mode != "load" or crc_timer):
        raise ValueError("recovery='none' never checks a CRC; crc_mode/crc_timer do not apply")
    gt = load_groundtruth()
    buf = np.fromfile(index_path, dtype=np.uint8)

    if recovery == "none":
        failing = _corrupt_ex_elements(buf)
        checked = 0                       # none mode never CRC-checks (spec rule 4)
    else:
        manifest = cm.read_manifest(crc_manifest)
        failing = cm.verify_buffer(buf, manifest)
        checked = int(manifest["header"]["n_entries"])

    n = int(region_map()["header"]["cur_element_count"])
    frac = len(failing) / n if n else 0.0
    recall = max(0.05, CLEAN_RECALL * (1.0 - _RECOVERY_SLOPE[recovery] * frac))
    ids = _ids_for_recall(recall, gt, k)
    per_hit = len(failing)                # deterministic fake counter model
    lazy = (crc_mode == "lazy")
    stats = {
        "recovery": recovery, "crc_manifest": crc_manifest or "", "ef": int(ef), "topk": k,
        "nq": NQ,
        "crc_mode": crc_mode,
        # Zeroed under lazy for the same reason the binary zeroes them: no load scan ran.
        "load": {"elements_checked": 0 if lazy else checked,
                 "elements_crc_fail": 0 if lazy else
                                      (len(failing) if recovery != "none" else 0)},
        # None = not knowable without a real search (see the docstring). Under load these are
        # 0 for the same reason as the binary: the query path computes no CRC.
        "crc": {"checks": None if lazy else 0, "bytes": None if lazy else 0,
                "ns": None if lazy else 0, "oob_skipped": None if lazy else 0,
                "timer_enabled": bool(crc_timer),
                "distinct_elements_failed": None if lazy else 0},
        "totals": {"consults": n * NQ if recovery != "none" else 0,
                   "corrupt_hits": per_hit * NQ if recovery != "none" else 0,
                   "fallbacks": per_hit * NQ if recovery == "fallback_eb" else 0,
                   "drops": per_hit * NQ if recovery == "drop" else 0},
        "per_query": None,
        "_stub": True,
    }
    return {"ids": ids, "cpp_recall": float(np.round(_recall(ids, gt, k), 12)),
            "stats": stats}


def _distances(ids, finite):
    """Synthetic top-k distances aligned with ids; one non-finite entry when finite=False."""
    nq, k = ids.shape
    d = np.tile(np.linspace(1.0, 2.0, k, dtype=np.float32), (nq, 1))
    if not finite:
        d[0, 0] = np.inf
    return d
