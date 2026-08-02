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

# Option B recovery modes (stage2_cpp_patch.md rule 4). none = no CRC check, possibly-bad ex
# used as-is; drop = CRC fail -> skip candidate; fallback_eb = CRC fail -> bin + error-bound
# pessimistic rank (Samuel exp-3 semantics, computed in C++).
RECOVERY_MODES = ("none", "drop", "fallback_eb")

# WHEN the CRC is computed (orthogonal to WHAT it decides). load = eager whole-index scan at
# load, the original behaviour and still the default; lazy = on-access, recomputed as the
# search consults an element (the Sec 3.1/3.3 piggyback design, which does not assume the
# index is immutable during querying). Same predicate, same decision points, same ids.
CRC_MODES = ("load", "lazy")
# Which CRC kernel computes the check. All are bit-identical to zlib.crc32 (proven in
# artifacts/phase3/tests/test_crc_kernel.py), so this selects SPEED ONLY -- it can never
# change a verdict, an id or a recall. "table" is the original byte-at-a-time loop and
# stays the default so every frozen configuration re-runs unchanged.
CRC_IMPLS = ("table", "slice8", "clmul")
# Driver default. Measured on meow1 (median of 3 interleaved rounds, clean index, ef=2000):
# clmul cuts query-path detection cost by 70% for fallback_eb (+8.51% -> +2.55% of search
# wall) and 76% for drop (+209.83% -> +49.92%), and the load scan by 11.5x (112.9 -> 9.8 ms).
# See docs/crc_kernel.md. The BINARY still defaults to "table" so a bare command line stays
# backward compatible; the drivers pass this explicitly, and stats["crc_impl"] echoes what
# was actually used.
DEFAULT_CRC_IMPL = "clmul"

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


def read_serialized_range(byte_start, byte_len, path=None):
    """Read byte_len bytes at byte_start from the on-disk index (the clean source).

    Deliberately a fresh file read on every call — NEVER cached in memory. The whole
    point of the clean source is that it lives in persistent storage and is immune to
    the DRAM fault process; caching it in memory would turn it into a fourth corruptible
    replica and break that semantics.
    """
    path = path or INDEX_PATH
    with open(path, "rb") as fh:
        fh.seek(int(byte_start))
        data = fh.read(int(byte_len))
    if len(data) != int(byte_len):
        raise IOError(f"short read from {path}: wanted {byte_len} B at {byte_start}, "
                      f"got {len(data)} B")
    return np.frombuffer(data, dtype=np.uint8).copy()


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


def query_with_recovery(index_path, recovery="none", crc_manifest=None, *, k=None, ef=2000,
                        out_path=None, stats_json=None, query_f=None, gt_f=None, timeout=None,
                        crc_mode="load", crc_timer=False, crc_impl=None):
    """Run the Option-B exp_dumpids with query-time CRC recovery (stage2_cpp_patch.md §2a).

    `index_path` is a (possibly already-corrupted BY PYTHON) index file; `crc_manifest` is the
    qp.rabitq.crc_manifest file written from the CLEAN index BEFORE injection. The C++ side
    only compares CRCs and applies Samuel's exp-3 policy — it never injects (hard rule #1) and
    its RECALL stdout line is parity-only; recompute the authoritative recall from the returned
    ids with qp.metrics (hard rule #2).

    `crc_mode` picks WHEN the CRC runs, not what it decides:
      "load" (default) — eager whole-index scan at load; stats["load"] carries its totals.
      "lazy"           — on-access, recomputed as the search consults each element (the
                         Sec 3.1/3.3 piggyback design). stats["load"] is then all zeros (no
                         scan ran) and stats["crc"] carries the on-access totals instead.
    Both modes use the same predicate at the same decision points, so on an index that does not
    change mid-run they return bit-identical ids — asserted by phase3_expb_recovery --lazy-gate.
    `crc_timer` times each on-access CRC into stats["crc"]["ns"]; it is off by default because
    the clock reads both inflate that number and perturb stats["search_wall_ns"].

    Returns {"ids": (nq,k) int32 (-1 pads), "cpp_recall": float, "stats": dict|None} where
    `stats` is the binary's stats json (crc_mode + load-scan totals + on-access CRC totals +
    per-query counters). Subprocess crash/timeout propagate as CalledProcessError /
    TimeoutExpired for the runner to record as `crash`.
    """
    import json

    _require_binaries()
    if recovery not in RECOVERY_MODES:
        raise ValueError(f"recovery must be one of {RECOVERY_MODES}, got {recovery!r}")
    if recovery != "none" and not crc_manifest:
        raise ValueError(f"--recovery {recovery} requires a crc_manifest path")
    if crc_mode not in CRC_MODES:
        raise ValueError(f"crc_mode must be one of {CRC_MODES}, got {crc_mode!r}")
    # None means "caller did not specify", which is what lets recovery="none" reject an
    # EXPLICIT kernel choice as meaningless while still having a default for real runs.
    if crc_impl is not None and crc_impl not in CRC_IMPLS:
        raise ValueError(f"crc_impl must be one of {CRC_IMPLS}, got {crc_impl!r}")
    if recovery == "none" and (crc_mode != "load" or crc_timer or crc_impl is not None):
        raise ValueError("recovery='none' never checks a CRC; "
                         "crc_mode/crc_timer/crc_impl do not apply")
    crc_impl = DEFAULT_CRC_IMPL if crc_impl is None else crc_impl
    k = config.K if k is None else int(k)
    query_f = query_f or os.path.join(PREP, "query.fvecs")
    gt_f = gt_f or os.path.join(PREP, "groundtruth.ivecs")
    # The default scratch name carries crc_mode only for lazy, so every existing caller's
    # paths stay byte-identical while a load/lazy pair cannot overwrite each other's dumps.
    tag = "" if crc_mode == "load" else f"_{crc_mode}"
    out_path = out_path or os.path.join(PREP, f"_dumpids_{recovery}{tag}_ef{int(ef)}_k{k}.ivecs")
    stats_json = stats_json or out_path + ".stats.json"
    cmd = [os.path.join(BIN, "exp_dumpids"), index_path, query_f, gt_f, METRIC,
           str(int(ef)), out_path, str(k),
           "--recovery", recovery, "--stats-json", stats_json]
    if recovery != "none":
        cmd += ["--crc-manifest", crc_manifest]
        # Appended only for lazy: the load-mode command line stays literally what it was
        # before this flag existed, so re-running a frozen configuration is unchanged.
        if crc_mode != "load":
            cmd += ["--crc-mode", crc_mode]
        if crc_timer:
            cmd.append("--crc-timer")
        # Always explicit, unlike --crc-mode: the binary and the drivers deliberately have
        # DIFFERENT defaults (binary "table" for backward compatibility, drivers clmul for
        # speed), so omitting the flag would silently pick the slow kernel.
        cmd += ["--crc-impl", crc_impl]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    cpp_recall = None
    for line in res.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0] == "RECALL":
            cpp_recall = float(parts[1])
    if cpp_recall is None:
        raise RuntimeError(f"exp_dumpids did not print a RECALL line; stdout:\n{res.stdout}")
    ids = read_ivecs(out_path)
    stats = None
    if os.path.isfile(stats_json):
        with open(stats_json) as fh:
            stats = json.load(fh)
    return {"ids": ids, "cpp_recall": cpp_recall, "stats": stats}


def search_with_eb_fallback(index_path, fraction, seed=0, *, crc_manifest=None, k=None,
                            ef=2000, out_path=None, query_f=None, gt_f=None, timeout=None,
                            crc_mode="load", crc_timer=False, crc_impl=None):
    """EB-fallback search of an ALREADY-corrupted index (E5 slope layer contract).

    Requires `crc_manifest` — the qp-written CRC manifest of the CLEAN index (Option B). With
    it, delegates to query_with_recovery(..., "fallback_eb"): the patched exp_dumpids CRC-scans
    the loaded bytes and ranks CRC-failed candidates by Samuel's pessimistic error bound
    (est + (est - low), hnsw.hpp exp-3 code path, unchanged). `fraction`/`seed` are the
    CALLER's injection metadata, recorded for the result row only — nothing here injects or
    simulates (the earlier Option-A wrapper double-corrupted and was removed; the exp_faultinject
    binary still cannot rank a caller-supplied corrupted buffer).

    Without `crc_manifest` this remains a report-and-stop: EB needs query-time est_dist/g_error
    (C++-only) plus the clean-CRC definition (Python-only) — silently proceeding without the
    manifest would mean the C++ side inventing "clean", violating hard rule #1.
    """
    if not crc_manifest:
        raise RuntimeError(
            "REPORT-AND-STOP: EB-fallback needs the CLEAN index's CRC manifest "
            "(qp.rabitq.crc_manifest.write_manifest BEFORE injection); pass crc_manifest=. "
            "Without it the C++ side would have to invent the definition of 'clean' "
            f"(index_path={index_path!r}, fraction={fraction!r}, seed={seed!r})")
    res = query_with_recovery(index_path, "fallback_eb", crc_manifest, k=k, ef=ef,
                              out_path=out_path, query_f=query_f, gt_f=gt_f, timeout=timeout,
                              crc_mode=crc_mode, crc_timer=crc_timer,
                              crc_impl=crc_impl)
    res["distances"] = None
    res["_eb_path"] = True
    res["eb_fraction"] = float(fraction)
    return res


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
