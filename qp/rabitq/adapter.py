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

BINARIES = ("hnsw_rabitq_indexing", "hnsw_rabitq_querying", "exp_faultinject", "exp_fieldflip")

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
    """Fixed query set (prepared/query.fvecs) — reuses qp.data.read_fvecs."""
    return read_fvecs(os.path.join(PREP, "query.fvecs"))


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


def query_ids(index_path=None, k=10):
    """Per-query neighbour ids from the real search path — needed for qp-vs-C++ parity.

    REPORT-AND-STOP: hnsw_rabitq_querying / exp_faultinject print recall scalars only; neither
    emits per-query ids. Wiring the parity check requires a small instrumentation addition (an
    ids-dump flag writing an ivecs of the top-k ids), then parse it here. Flagged, not faked.
    """
    raise NotImplementedError(
        "query_ids requires an ids-dump from the C++ querying binary (a small instrumentation "
        "addition). Until then the qp-vs-C++ recall parity on a real index is blocked. See "
        "artifacts/phase3/STAGE0_STATUS.md.")
