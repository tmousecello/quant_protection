"""Phase 3 recall measurement — a thin wrapper over qp.metrics (the single source of truth).

This module deliberately re-implements NO metric logic. recall@k, tolerant recall, the
collapse predicate and failure classification all live in qp.metrics; here we only bundle
the canonical calls at the locked operating point (k, eps from qp.config) and feed them the
neighbour IDs produced by the *real* RaBitQ search path (bin traversal + ex/refine rerank
with the recovery policy applied), parsed by the qp.rabitq adapter.

Keeping this a wrapper is the whole point of Stage 0: the RaBitQ work and the FAISS work
(Phase 1/2) score recall and collapse with the same code, so cross-repo parity holds.
"""
from qp import config, metrics


def recall_block(pred_ids, gt_ids, gt_dist=None, ks=None, eps=None):
    """Canonical recall summary for one (corrupted or clean) result set.

    pred_ids : (N, >=max(ks)) neighbour ids from the search path (RaBitQ querying binary).
    gt_ids   : (N, >=max(ks)) exact ground-truth ids (fixed across the study).
    gt_dist  : (N, >=k) exact distances aligned with gt_ids; required only for tolerant recall.
    Returns a dict: {'recall@1':..., 'recall@10':..., 'recall@100':..., 'tolerant_recall@10':...}.
    """
    ks = tuple(config.RECALL_KS) if ks is None else tuple(ks)
    eps = config.EPSILON_TOLERANT if eps is None else eps
    out = {f"recall@{k}": metrics.recall_at_k(pred_ids, gt_ids, k) for k in ks}
    if gt_dist is not None:
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
