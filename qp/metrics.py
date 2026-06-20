"""Recall metrics and failure-mode classification.

`recall_at_k` is the standard metric. `tolerant_recall` measures recall against an
epsilon-EXPANDED ground-truth set: a predicted neighbor still counts if it is a true
neighbor whose exact distance is within (1+eps) of the kth true distance. This avoids
overcounting benign reshuffling of near-equidistant neighbors as recall loss
(prompt.txt: "tolerant recall 對只是重排不相關鄰居的微小擾動較不敏感"). It needs the
exact `gt_dist` produced once from the FLAT index in Phase 0.

`classify_failure` separates the three failure modes the study tracks. Phase 0 only ever
sees the clean path; Phase 1/2 import this for corrupted runs.
"""
import numpy as np

from qp import config


# --- failure-mode classification ---------------------------------------------
# (defined up here because is_silent_collapse needs CRASH / NAN_INF to enforce silent-only.)
CLEAN = "clean"
CRASH = "crash"
NAN_INF = "nan-inf"
SILENT_WRONG = "silent-wrong"


def is_silent_collapse(faulted10, clean10, frac=None, failure_mode=None):
    """Unified collapse predicate (all index families) — the single source of truth.

    Collapse = the index lost at least `frac` of its usable recall: faulted recall@10 <
    frac * own clean recall@10 (retention rule; frac defaults to config.COLLAPSE_RETENTION_FRAC).
    This replaces the old aligned/own split (absolute >0.01 vs retention) that made fp32 read
    as collapsing under a large burst while Curve B said it never collapses.

    SILENT-only: a DETECTABLE failure is never a silent collapse, so this returns False for
      - crash   -> faulted10 is None (and/or failure_mode == CRASH); and
      - nan-inf -> failure_mode == NAN_INF.
    NOTE: a nan-inf record still carries a NUMERIC faulted_recall@10, because recall is computed
    from the returned ids even when the distances are non-finite (see classify_failure /
    isolation._record / phase1 measure_flip). So the `faulted10 is None` check alone does NOT
    exclude nan-inf — the caller MUST pass `failure_mode` from the flip record for the
    silent-only guarantee. If `failure_mode` is omitted, only crashes are excluded (back-compat).
    Callers count crash / nan-inf separately (n_crash / n_nan_inf).
    """
    if frac is None:
        frac = config.COLLAPSE_RETENTION_FRAC
    if failure_mode in (CRASH, NAN_INF):
        return False
    return faulted10 is not None and faulted10 < frac * clean10


def recall_at_k(pred_ids, gt_ids, k):
    """Mean over queries of |set(pred_topk) ∩ set(true_topk)| / k.

    pred_ids, gt_ids: (N, >=k) int arrays. FAISS pads missing results with -1; those
    never match a (non-negative) ground-truth id, so they are handled correctly.

    Counting is done *per true id* (any-over-pred), not per pred slot, so a corrupted
    index that returns the SAME correct id in several slots cannot inflate recall — each
    true neighbour is credited at most once. Without this, result collapse (a real
    corruption mode, e.g. flipped IVF list ids) would silently raise recall and mask
    damage. Truth ids are distinct, so |intersection| is exactly the hit count.
    """
    pred = np.asarray(pred_ids)[:, :k]
    truth = np.asarray(gt_ids)[:, :k]
    # (N, k_pred, k_true) match tensor -> for each true id, did ANY pred slot hit it ->
    # count distinct true ids found per query.
    hits = (pred[:, :, None] == truth[:, None, :]).any(axis=1).sum(axis=1)
    return float(hits.mean() / k)


def recall_curve(pred_ids, gt_ids, ks):
    """Convenience: {k: recall_at_k} for several k."""
    return {int(k): recall_at_k(pred_ids, gt_ids, k) for k in ks}


def tolerant_recall(pred_ids, gt_ids, gt_dist, k, eps):
    """Recall against an eps-expanded GT set.

    For each query the acceptable set is every true neighbor whose exact distance is
    <= (1+eps) * (kth true distance). A predicted top-k id counts if it is in that set.
    gt_dist: exact L2 distances aligned COLUMN-FOR-COLUMN with gt_ids (SAME shape), from FLAT
    in Phase 0. The acceptable mask indexes gt_ids with a mask shaped like gt_dist, so the two
    must have the same column count — a k-wide gt_dist against a 100-wide gt_ids is a caller
    error (mis-sliced input), not a narrower ground truth.
    """
    pred = np.asarray(pred_ids)[:, :k]
    truth = np.asarray(gt_ids)
    gd = np.asarray(gt_dist)
    if gd.shape[1] != truth.shape[1]:
        raise ValueError(
            f"gt_dist and gt_ids must be column-aligned (same #neighbours); got gt_dist "
            f"{gd.shape} vs gt_ids {truth.shape} — slice both to the same width, not just one")
    thresh = gd[:, k - 1] * (1.0 + eps)                 # (N,)
    acceptable = gd <= thresh[:, None]                  # (N, Ngt) bool mask over GT
    n = pred.shape[0]
    hits = 0
    for q in range(n):
        ok_ids = truth[q][acceptable[q]]
        # np.unique(pred[q]) dedups so repeated correct ids count once; this both keeps
        # per-query tolerant recall <= 1.0 and prevents result-collapse from inflating it.
        hits += np.isin(np.unique(pred[q]), ok_ids).sum()
    return float(hits / (n * k))


def query_kth_gt_distance(gt_dist, k):
    """The kth true distance per query — handy for tolerant-recall diagnostics."""
    return np.asarray(gt_dist)[:, k - 1].copy()


def classify_failure(exception=None, distances=None, indices=None,
                     recall=None, clean_recall=None, drop_tol=1e-9):
    """Label a (possibly corrupted) search outcome.

    crash        deserialize/search raised (pass the exception).
    nan-inf      returned distances contain NaN or +/-Inf.
    silent-wrong distances are finite but recall fell below clean_recall - drop_tol.
    clean        finite results, recall not meaningfully below baseline.

    recall/clean_recall are optional; without them a finite result is reported as clean
    (Phase 0 path). Phase 1/2 pass both to detect silent-wrong.
    """
    if exception is not None:
        return CRASH
    if distances is not None:
        d = np.asarray(distances)
        if d.size and not np.isfinite(d).all():
            return NAN_INF
    if recall is not None and clean_recall is not None:
        if recall < clean_recall - drop_tol:
            return SILENT_WRONG
    return CLEAN
