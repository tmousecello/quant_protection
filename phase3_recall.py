"""Phase 3 recall measurement — a thin wrapper over qp.metrics (the single source of truth).

This module deliberately re-implements NO metric logic. recall@k, tolerant recall, the
collapse predicate and failure classification all live in qp.metrics; here we only bundle
the canonical calls at the locked operating point (k, eps from qp.config) and feed them the
neighbour IDs produced by the *real* RaBitQ search path (bin traversal + ex/refine rerank
with the recovery policy applied), parsed by the qp.rabitq adapter.

Keeping this a wrapper is the whole point of Stage 0: the RaBitQ work and the FAISS work
(Phase 1/2) score recall and collapse with the same code, so cross-repo parity holds.
"""
import numpy as np

from qp import config, metrics


def recall_block(pred_ids, gt_ids, gt_dist=None, ks=None, eps=None):
    """Canonical recall summary for one (corrupted or clean) result set.

    pred_ids : (N, W) neighbour ids from the search path (RaBitQ querying binary returns W=10).
    gt_ids   : (N, >=max(ks)) exact ground-truth ids (fixed across the study).
    gt_dist  : exact distances aligned COLUMN-FOR-COLUMN with gt_ids (same shape); required
               only for tolerant recall.
    Returns {'recall@k': ...} for each requested k that fits the prediction width W, plus
    'tolerant_recall@K' when gt_dist is given and config.K <= W. A requested k>W is SKIPPED,
    not reported: the RaBitQ path returns only W=10 ids, so recall@100 over it could never
    exceed 0.1 and would misread as catastrophic loss rather than a measurement-width artifact.
    """
    ks = tuple(config.RECALL_KS) if ks is None else tuple(ks)
    eps = config.EPSILON_TOLERANT if eps is None else eps
    ncols = np.asarray(pred_ids).shape[1]
    out = {f"recall@{k}": metrics.recall_at_k(pred_ids, gt_ids, k) for k in ks if k <= ncols}
    if gt_dist is not None and config.K <= ncols:
        out[f"tolerant_recall@{config.K}"] = metrics.tolerant_recall(
            pred_ids, gt_ids, gt_dist, config.K, eps)
    return out


def collapse_flags(faulted10, clean10, failure_mode=None, frac=None):
    """Delegate to metrics.is_silent_collapse — silent-only, retention<frac of own clean@10.

    failure_mode is the qp.metrics label (CLEAN/CRASH/NAN_INF/SILENT_WRONG) from the flip
    record; passing it is what guarantees crash and nan-inf are excluded from silent collapse.
    """
    return {
        "is_silent_collapse": metrics.is_silent_collapse(
            faulted10, clean10, frac=frac, failure_mode=failure_mode),
        "retention@10": (None if clean10 in (None, 0) or faulted10 is None
                         else faulted10 / clean10),
        "failure_mode": failure_mode,
    }


def classify(exception=None, distances=None, indices=None, recall=None, clean_recall=None):
    """Pass-through to metrics.classify_failure (crash / nan-inf / silent-wrong / clean)."""
    return metrics.classify_failure(
        exception=exception, distances=distances, indices=indices,
        recall=recall, clean_recall=clean_recall)
