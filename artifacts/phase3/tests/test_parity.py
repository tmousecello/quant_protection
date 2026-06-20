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
    """qp-imported recall == Samuel's C++ recall on the clean b=7 index, same query/gt.

    Also blocked on adapter.query_ids (needs a C++ ids dump). When both are available:
      ids = adapter.query_ids(adapter.INDEX_PATH, k=config.K)
      qp_recall = metrics.recall_at_k(ids, adapter.load_groundtruth(), config.K)
      cpp_recall = adapter.clean_baseline_recall()
      assert qp_recall == pytest.approx(cpp_recall, abs=1e-3)
    and the clean baseline must sit near adapter.EXPECTED_CLEAN_RECALL10 (~0.983).
    """
    cpp_recall = adapter.clean_baseline_recall()
    assert abs(cpp_recall - adapter.EXPECTED_CLEAN_RECALL10) < 0.01, \
        f"clean b=7 recall {cpp_recall} not near {adapter.EXPECTED_CLEAN_RECALL10}"
    pytest.skip("qp-vs-C++ id-level parity blocked on adapter.query_ids (C++ ids dump). "
                "Baseline anchor verified above.")
