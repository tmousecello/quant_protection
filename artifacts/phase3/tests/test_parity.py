"""Parity check (Stage 0 component 4, total acceptance).

Two layers:
  1. metric-level parity (runs now, no binaries): qp.metrics.recall_at_k must equal an
     independent reference recall on the same (pred, gt) — proves the imported metric is the
     same arithmetic everyone else would compute.
  2. RaBitQ parity (BLOCKED): same clean index + same flip + same query -> qp-imported
     recall == Samuel's C++ recall. Requires the x86-64 build AND a per-query ids dump from
     the querying binary (adapter.query_ids is a flagged report-and-stop). Skipped until both
     exist, so the check is wired and visible rather than silently absent.
"""
import numpy as np
import pytest

from qp import metrics
from qp.rabitq import adapter


def _reference_recall_at_k(pred, gt, k):
    """Independent, obviously-correct recall@k (per-true-id hit), for cross-checking metrics."""
    pred, gt = np.asarray(pred)[:, :k], np.asarray(gt)[:, :k]
    hits = [len(set(p.tolist()) & set(t.tolist())) for p, t in zip(pred, gt)]
    return float(np.mean(hits) / k)


def test_metric_parity_against_reference():
    rng = np.random.default_rng(0)
    # distinct ids per query (as real ground truth is): each row a unique sample, no dupes.
    gt = np.stack([rng.choice(1000, size=10, replace=False) for _ in range(40)])
    pred = gt.copy()
    pred[::2, 6:] = rng.integers(1000, 2000, size=(20, 4))   # corrupt half the queries' tails
    for k in (1, 10):
        assert metrics.recall_at_k(pred, gt, k) == pytest.approx(_reference_recall_at_k(pred, gt, k))


@pytest.mark.skipif(not adapter.binaries_built(),
                    reason="RaBitQ C++ binaries not built (x86-64 only); see build_rabitq.sh")
def test_rabitq_parity_clean_index():
    """qp-imported recall == Samuel's C++ recall on the clean b=7 index — same ids, same query/gt.

    The keystone Stage 0 acceptance. exp_dumpids runs the real search path (bin traversal +
    ex/refine rerank, recovery policy applied) and emits both the top-k ids and its own
    recall@k on those ids. We recompute recall with the imported qp.metrics on the SAME ids:
    the two must agree (proves the single-source metric is identical to Samuel's), and the
    clean b=7 value must sit on the ~0.983 plateau (anchors the whole study).
    """
    from qp import config

    ids, cpp_recall = adapter.query_ids(adapter.INDEX_PATH, k=config.K, ef=2000)
    gt = adapter.load_groundtruth()
    qp_recall = metrics.recall_at_k(ids, gt, config.K)
    assert qp_recall == pytest.approx(cpp_recall, abs=1e-3), \
        f"qp recall {qp_recall} != C++ recall {cpp_recall} on identical ids"
    assert abs(qp_recall - adapter.EXPECTED_CLEAN_RECALL10) < 0.01, \
        f"clean b=7 recall {qp_recall} not near {adapter.EXPECTED_CLEAN_RECALL10}"
