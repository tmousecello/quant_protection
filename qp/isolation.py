"""Subprocess-isolated single-flip evaluation — crash containment for Phase 2.

Phase 1 sampled only recall_relevant regions that the Phase 0 dry-run proved deserialize
cleanly, so it could flip-measure-restore in one process. Phase 2 deliberately injects into
`graph_edges` (int32 node ids -> out-of-bounds neighbour -> FAISS C++ can segfault) and into
random `burst` ranges (which may straddle the header / graph structure). A segfault there
takes down the *whole* sweep, mislabelling every later flip.

The fix: evaluate each such flip in its own short-lived process. A clean Python deserialize
failure is already caught by `flip.rebuild` and returned as a `crash` record from inside the
child; a hard C++ segfault kills only the child, which the parent observes as a non-zero
(signal) exit code and records as `crash`. A child still running past the timeout is treated
as a hang -> `crash`.

Efficiency: the parent serializes the pristine index ONCE to a temp .npy (`dump_buffer`);
every child memory-maps that file, copies it, applies its flip, and never touches the
parent's buffer — so there is nothing to restore and the 260MB+ HNSW buffer is not piped per
flip. Query / ground-truth arrays are likewise passed as .npy paths the child loads.

The returned record has the SAME fields Phase 1's `measure_flip` produces, so the existing
aggregation in `phase1_sensitivity.aggregate` consumes Phase 2 records unchanged.
"""
import multiprocessing as mp
import queue as _queue

import numpy as np

from . import config
from . import metrics


# --- record shapes (identical to phase1_sensitivity for schema compatibility) --------
def _record(clean, recalls, tol, failure_mode, nan_inf):
    return {
        "faulted_recall@1": recalls[1], "faulted_recall@10": recalls[10],
        "faulted_recall@100": recalls[100], "faulted_tol": tol,
        "dRecall@1": clean["recall"][1] - recalls[1],
        "dRecall@10": clean["recall"][10] - recalls[10],
        "dRecall@100": clean["recall"][100] - recalls[100],
        "dTol": clean["tol"] - tol,
        "failure_mode": failure_mode, "nan_inf_count": int(nan_inf),
        "exception_repr": "",
    }


def crash_record(clean, exc):
    """A crash-mode record (deserialize/search raised, segfaulted, or hung)."""
    return {
        "faulted_recall@1": None, "faulted_recall@10": None,
        "faulted_recall@100": None, "faulted_tol": None,
        "dRecall@1": None, "dRecall@10": None, "dRecall@100": None, "dTol": None,
        "failure_mode": metrics.CRASH, "nan_inf_count": 0,
        "exception_repr": repr(exc)[:300],
    }


# A genuine corruption crash is what rebuild()/safe_search() return as an exception; ANY other
# exception inside the child is an UNEXPECTED harness failure (bad import under spawn, np.load,
# set_knob, a metrics bug). It must NOT be counted as the study's `crash` failure mode, so it
# gets its own marker. Drivers detect it and abort loudly rather than silently skew the stats.
HARNESS_ERROR = "harness-error"


def harness_error_record(clean, exc):
    rec = crash_record(clean, exc)
    rec["failure_mode"] = HARNESS_ERROR
    rec["exception_repr"] = "HARNESS_ERROR: " + rec["exception_repr"]
    return rec


# --- substrate prep -----------------------------------------------------------
def dump_buffer(buf, path):
    """Persist a pristine serialized-index buffer for children to mmap. Returns `path`."""
    np.save(path, np.asarray(buf, dtype=np.uint8))
    return path


# --- child worker (module-level so the spawn context can import + pickle it) ----------
def _worker(q, buf_path, positions, knob_kind, knob_val, q_path, gt_ids_path,
            gt_dist_path, clean, k):
    # Imports happen inside the child so a spawn start method re-imports faiss cleanly.
    from .flip import flip_bits, rebuild, safe_search
    from . import indexes as ix
    try:
        # mmap the pristine dump and take a private writable copy — only the pages we touch
        # materialize, so peak RSS is ~1× the buffer (mmap_mode=None would load the whole file
        # AND then copy, ~2×, risking OOM-induced spurious crashes on the 260MB+ HNSW buffers).
        buf = np.load(buf_path, mmap_mode="r").copy()
        flip_bits(buf, [(int(b), int(bit)) for b, bit in positions])
        idx, exc = rebuild(buf)
        if exc is not None:
            q.put(crash_record(clean, exc)); return
        ix.set_knob(idx, {"kind": knob_kind}, knob_val)
        xq = np.load(q_path).astype("float32")
        D, I, sexc = safe_search(idx, xq, k)
        if sexc is not None:
            q.put(crash_record(clean, sexc)); return
        gt_ids = np.load(gt_ids_path)
        gt_dist = np.load(gt_dist_path)
        recalls = metrics.recall_curve(I, gt_ids, config.RECALL_KS)
        tol = metrics.tolerant_recall(I, gt_ids, gt_dist, config.K, config.EPSILON_TOLERANT)
        nan_inf = int((~np.isfinite(np.asarray(D))).sum())
        fm = metrics.classify_failure(
            distances=D, recall=recalls[10], clean_recall=clean["recall"][10])
        q.put(_record(clean, recalls, tol, fm, nan_inf))
    except Exception as exc:  # noqa: BLE001 - reaching here means an UNEXPECTED harness failure
        # (rebuild/search corruption crashes are returned as exc and handled above), so tag it
        # as a harness error, not a corruption crash.
        q.put(harness_error_record(clean, exc))


# --- parent-side driver -------------------------------------------------------
def measure_flip_isolated(buf_path, positions, knob_kind, knob_val, q_path,
                          gt_ids_path, gt_dist_path, clean, k=100,
                          timeout=config.PHASE2_FLIP_TIMEOUT_S):
    """Apply `positions` (a list of (byte_pos, bit)) to the pristine buffer in a child and
    return a Phase-1-shaped record. A segfault / hang is contained and recorded as `crash`.

    positions  : one [(byte,bit)] for a single edge flip, or burst_positions(...) for a burst.
    knob_kind  : "ivf" | "graph" | "exact" (drives qp.indexes.set_knob in the child).
    clean      : {"recall": {1:.,10:.,100:.}, "tol": .} measured on the clean rebuild path.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker,
        args=(q, buf_path, list(positions), knob_kind, knob_val, q_path,
              gt_ids_path, gt_dist_path, clean, k),
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return crash_record(clean, RuntimeError(f"timeout > {timeout}s (hang)"))
    if p.exitcode == 0:
        try:
            return q.get(timeout=30)
        except _queue.Empty:
            return crash_record(clean, RuntimeError("child exited 0 but produced no record"))
    # non-zero / negative exit code => killed by a signal (segfault) or hard error.
    return crash_record(clean, RuntimeError(f"subprocess exitcode={p.exitcode} (segfault/hard crash)"))
