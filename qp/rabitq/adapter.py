"""RaBitQ adapter — file-based serialize/deserialize, loaders, and subprocess query wrappers.

Mirrors the qp.flip substrate (serialize -> flip bytes -> deserialize) but for a RaBitQ index,
which is an on-disk file produced by the C++ binaries rather than a faiss buffer:
  serialize_index(path)   -> uint8 numpy buffer (read the index file)
  deserialize_index(buf)  -> write the (possibly corrupted) buffer to a new file
A fault model from qp.faults flips bytes in that buffer at a region from qp.rabitq.layout,
the corrupted file is re-queried by the C++ binary, and the returned ids feed qp.metrics.

Live calls (build/query) shell out to Samuel's binaries and therefore require the x86-64
build (build_rabitq.sh). They gate on binaries_built(); on a machine without the binaries the
adapter still serves the byte layout (layout.py) and the file serialize/deserialize, so the
fault-injection plumbing is testable offline.
"""
import os
import subprocess

import numpy as np

from qp import config
from qp.data import read_fvecs, read_ivecs
from qp.rabitq import layout

# --- locations (overridable by env; defaults match build_rabitq.sh) ----------
RABITQ_REPO = os.environ.get(
    "RABITQ_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "..", "StorageSystemProject_RaBitQ_Free-recovery"))
RABITQ_REPO = os.path.normpath(RABITQ_REPO)
BIN = os.path.join(RABITQ_REPO, "third_party", "RaBitQ-Library", "bin")
PREP = os.path.join(RABITQ_REPO, "data", "sift", "prepared")
INDEX_PATH = os.path.join(RABITQ_REPO, "results", "datasets", "sift", "idx",
                          "hnsw_M16_efC200_b7.index")
METRIC = "l2"

BINARIES = ("hnsw_rabitq_indexing", "hnsw_rabitq_querying", "exp_faultinject",
            "exp_fieldflip", "exp_dumpids")

# Documented anchor for golden comparison (results/datasets/sift/ladder.csv plateau, b=7).
# This is a REFERENCE constant, never returned as if it were a fresh measurement.
EXPECTED_CLEAN_RECALL10 = 0.983


def binaries_built():
    """True iff Samuel's C++ binaries exist (i.e. build_rabitq.sh ran on an x86-64 host)."""
    return all(os.path.isfile(os.path.join(BIN, b)) for b in BINARIES)


def _require_binaries():
    if not binaries_built():
        raise RuntimeError(
            "REPORT-AND-STOP: RaBitQ binaries not built. The library is x86-64 only "
            f"(AVX2/AVX512); run build_rabitq.sh on an x86-64 Linux host. Looked in {BIN}.")


# --- file-based serialize / deserialize substrate ----------------------------

def serialize_index(path=None):
    """Read an index file into a writable uint8 buffer (the flip substrate's 'serialize')."""
    path = path or INDEX_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"index not found: {path} (build it via build_rabitq.sh)")
    return np.fromfile(path, dtype=np.uint8).copy()


def deserialize_index(buf, out_path):
    """Write a (possibly corrupted) buffer back to a file the C++ binary can load."""
    np.asarray(buf, dtype=np.uint8).tofile(out_path)
    return out_path


def byte_size(path=None):
    """Serialized length in bytes — the faults/MB denominator."""
    return os.path.getsize(path or INDEX_PATH)


# --- byte-region map ---------------------------------------------------------

def read_header(path=None):
    """Parse the 156-byte index header (authoritative on-disk sizes)."""
    return layout.parse_header(path or INDEX_PATH)


def region_map(path=None):
    """Full serialized region map (header/centroids/level0/elem0.*/rotation) for an index file.

    Rotation is located from the file tail (it is written last). Requires the index file to
    exist; the layout math itself is source-derived and needs no binaries.
    """
    path = path or INDEX_PATH
    header = layout.parse_header(path)
    return layout.serialized_region_map(header, file_size=os.path.getsize(path))


# --- dataset / query / ground-truth loaders ----------------------------------

def load_query():
    """Fixed query set (prepared/query.fvecs) — reuses qp.data.read_fvecs.

    read_fvecs returns ``(array, dim)``; return just the ``(N, dim)`` array so this matches
    load_groundtruth and qp.data.load_query (callers expect a bare ndarray, not a 2-tuple).
    """
    return read_fvecs(os.path.join(PREP, "query.fvecs"))[0]


def load_groundtruth():
    """Exact ground-truth neighbour ids (prepared/groundtruth.ivecs) — qp.data.read_ivecs."""
    return read_ivecs(os.path.join(PREP, "groundtruth.ivecs"))


# --- subprocess query wrappers (gated on the x86-64 build) -------------------

def _query_raw(index_path, query_f=None, gt_f=None):
    """Run hnsw_rabitq_querying; return its stdout (tab lines: ef\\tqps\\trecall10)."""
    _require_binaries()
    query_f = query_f or os.path.join(PREP, "query.fvecs")
    gt_f = gt_f or os.path.join(PREP, "groundtruth.ivecs")
    cmd = [os.path.join(BIN, "hnsw_rabitq_querying"), index_path, query_f, gt_f, METRIC]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def query_recall(index_path=None, query_f=None, gt_f=None):
    """Parse the querying binary's recall@10-by-ef output into {ef: recall10}.

    NOTE this reads the C++-computed recall. The qp-vs-C++ PARITY check (recompute recall
    from ids with qp.metrics) needs per-query ids, which the current binary does NOT emit —
    see query_ids().
    """
    out = _query_raw(index_path or INDEX_PATH, query_f, gt_f)
    table = {}
    for line in out.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
            ef, _qps, rec = parts[0], parts[1], parts[2]
            table[int(ef)] = float(rec)
    return table


def clean_baseline_recall(index_path=None):
    """Plateau (max-ef) clean recall@10 of the b=7 index. Must be ~EXPECTED_CLEAN_RECALL10.

    Gated on the build; raises report-and-stop otherwise. The caller should assert the value
    is within tolerance of 0.983 before trusting any downstream corruption numbers.
    """
    table = query_recall(index_path or INDEX_PATH)
    if not table:
        raise RuntimeError("querying produced no recall rows; check binary/index/query paths")
    return max(table.values())


def query_ids(index_path=None, k=None, ef=2000, out_path=None, query_f=None, gt_f=None,
              timeout=None):
    """Per-query top-k neighbour ids from the real search path, plus the C++-side recall.

    Runs the exp_dumpids instrument (built by build_rabitq.sh) at a single `ef`, which writes
    the ids as ivecs and prints `RECALL\\t<r>`. Returns (ids, cpp_recall):
      ids        : (nq, k) int32 array (qp.data.read_ivecs); -1 marks a padded/missing slot.
      cpp_recall : the binary's recall@k on those same ids — the parity counterpart to
                   qp.metrics.recall_at_k(ids, gt, k).
    `ef` defaults high (2000) to sit on the recall plateau (~0.983). Gated on the x86-64 build.
    `timeout` (seconds) bounds the subprocess: a corrupted index that hangs the search raises
    subprocess.TimeoutExpired (the sweep maps that to a `crash`). None = no limit.
    """
    _require_binaries()
    k = config.K if k is None else int(k)
    index_path = index_path or INDEX_PATH
    query_f = query_f or os.path.join(PREP, "query.fvecs")
    gt_f = gt_f or os.path.join(PREP, "groundtruth.ivecs")
    out_path = out_path or os.path.join(PREP, f"_dumpids_ef{int(ef)}_k{k}.ivecs")
    cmd = [os.path.join(BIN, "exp_dumpids"), index_path, query_f, gt_f, METRIC,
           str(int(ef)), out_path, str(k)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    cpp_recall = None
    for line in res.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0] == "RECALL":
            cpp_recall = float(parts[1])
    if cpp_recall is None:
        raise RuntimeError(f"exp_dumpids did not print a RECALL line; stdout:\n{res.stdout}")
    ids = read_ivecs(out_path)
    return ids, cpp_recall


def search_with_eb_fallback(index_path, fraction, seed=0, *, k=None, ef=2000,
                            out_path=None, query_f=None, gt_f=None, timeout=None):
    """REPORT-AND-STOP: pre-corrupted EB-fallback ranking is not available on current binaries.

    The E5 slope layer wants to re-rank an ALREADY-corrupted index with Samuel's EB policy
    (protected bin + error bound, hnsw.hpp FAULT_FALLBACK_EB). That cannot be done with the
    binaries build_rabitq.sh produces, so this path stops loudly instead of faking it:

      * The real exp_faultinject CLI is `<index> <query> <gt> [l2|ip] [seed]`. It loads ONE
        CLEAN index, sweeps ALL policies × fractions internally (set_fault_injection just marks
        a fraction of vectors corrupt in an in-memory bitmap — it does NOT flip bytes), and
        prints a CSV. It takes no policy/fraction/out_path/k/ef arg and emits no RECALL line, so
        it cannot rank a caller-supplied corrupted buffer. (The previous Option-A wrapper here
        was coded against a hypothetical patched binary and would have double-corrupted: it fed
        the already-corrupted index to exp_faultinject, which then injected MORE faults.)
      * EB ranking needs query-time quantities (est_dist, g_error) computed inside the C++
        search; they do not exist in Python, so qp.rabitq.eb_policy.eb_rank_dist is only a
        documented formula reference, not a runnable Python recovery path.

    True pre-corrupted EB requires Option B: patch exp_dumpids with a `--recovery fallback_eb`
    mode (workstation/C++ work). Until then, raise. See artifacts/phase3/E5_RUNBOOK.md.
    """
    raise RuntimeError(
        "REPORT-AND-STOP: pre-corrupted EB-fallback ranking needs the Option-B C++ patch "
        "(exp_dumpids --recovery fallback_eb); the current exp_faultinject binary cannot rank a "
        "supplied corrupted index (its CLI is <index> <query> <gt> [l2|ip] [seed], sweeps "
        "internally, prints CSV). See E5_RUNBOOK.md. "
        f"(index_path={index_path!r}, fraction={fraction!r}, seed={seed!r})")


def search_corrupted(index_path, k=None, ef=2000, out_path=None, query_f=None, gt_f=None,
                     timeout=None):
    """Runner contract: search a (possibly corrupted) index file, return ids/recall/distances.

    Returns a dict {"ids", "cpp_recall", "distances"}. `distances` is the per-query top-k distance
    array (nq, k) when exp_dumpids emitted a companion `<out_path>.dist.fvecs` (the workstation
    build adds this; see E1_RUNBOOK.md), else None — the runner's failure classifier then runs the
    SAME qp.metrics.classify_failure path FAISS uses, degrading gracefully (no nan-inf detection)
    when distances are absent. Subprocess crash/timeout propagate as CalledProcessError /
    TimeoutExpired for the sweep to record as `crash`.
    """
    k = config.K if k is None else int(k)
    out_path = out_path or os.path.join(PREP, f"_dumpids_ef{int(ef)}_k{k}.ivecs")
    ids, cpp_recall = query_ids(index_path, k=k, ef=ef, out_path=out_path,
                                query_f=query_f, gt_f=gt_f, timeout=timeout)
    distances = None
    dist_path = out_path + ".dist.fvecs"
    if os.path.isfile(dist_path):
        distances = read_fvecs(dist_path)[0]
    return {"ids": ids, "cpp_recall": cpp_recall, "distances": distances}
