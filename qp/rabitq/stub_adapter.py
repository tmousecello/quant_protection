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
_GEOM = dict(dim=128, padded_dim=128, ex_bits=6, num_cluster=16, cur_element_count=64,
             M=16, maxM=16, maxM0=32, num_upper_link_bytes=0)


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
        "maxlevel": 0, "enterpoint_node": 0, "M": g["M"], "maxM": g["maxM"], "maxM0": g["maxM0"],
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
    total = (layout.HEADER_BYTES + centroids + level0 + g["num_upper_link_bytes"] + rot)
    buf = np.zeros(total, dtype=np.uint8)
    buf[:layout.HEADER_BYTES] = np.frombuffer(_pack_header(hdr), dtype=np.uint8)
    return buf


_PRISTINE = _build_pristine()


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
}


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


def _distances(ids, finite):
    """Synthetic top-k distances aligned with ids; one non-finite entry when finite=False."""
    nq, k = ids.shape
    d = np.tile(np.linspace(1.0, 2.0, k, dtype=np.float32), (nq, 1))
    if not finite:
        d[0, 0] = np.inf
    return d
