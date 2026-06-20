"""Verifications for the recall measurement wrapper (Stage 0 component 2).

Predicate truth-table + gt/direction sanity. Everything delegates to qp.metrics (the single
source of truth); these tests pin the semantics the brief specifies.
"""
import numpy as np

from qp import metrics
import phase3_recall as pr


# --- is_silent_collapse truth table (the brief's exact cases) ----------------

def test_collapse_truth_table():
    assert metrics.is_silent_collapse(0.4, 0.95) is True            # retention 0.42 < 0.5
    assert metrics.is_silent_collapse(0.89, 0.95) is False          # retention 0.94 >= 0.5
    # crash: faulted10 is None -> never a silent collapse
    assert metrics.is_silent_collapse(None, 0.95) is False
    assert metrics.is_silent_collapse(None, 0.95, failure_mode=metrics.CRASH) is False
    # nan-inf carries a numeric recall but must be excluded when failure_mode is passed
    assert metrics.is_silent_collapse(0.0, 0.95, failure_mode=metrics.NAN_INF) is False


def test_collapse_flags_wrapper_matches_metrics():
    out = pr.collapse_flags(0.4, 0.95, failure_mode=metrics.SILENT_WRONG)
    assert out["is_silent_collapse"] is True
    assert abs(out["retention@10"] - 0.4 / 0.95) < 1e-12
    # nan-inf excluded via the wrapper too
    assert pr.collapse_flags(0.0, 0.95, failure_mode=metrics.NAN_INF)["is_silent_collapse"] is False


# --- gt sanity: exact NN search recalls itself ------------------------------

def test_gt_sanity_self_recall_is_one():
    # a trivial exact setup: pred == gt -> recall@1 == 1.0
    gt = np.array([[3, 7, 1], [5, 2, 9], [0, 4, 8]])
    pred = gt.copy()
    assert pr.recall_block(pred, gt, ks=(1,))["recall@1"] == 1.0


# --- direction sanity: corrupted <= clean, recall rises with more candidates --

def test_direction_clean_ge_corrupted():
    gt = np.tile(np.arange(10), (50, 1))                 # 50 queries, true ids 0..9
    clean = gt.copy()
    # corrupt: replace half the predicted ids per query with wrong ones
    corrupted = gt.copy()
    corrupted[:, 5:] = -1
    r_clean = pr.recall_block(clean, gt, ks=(10,))["recall@10"]
    r_corr = pr.recall_block(corrupted, gt, ks=(10,))["recall@10"]
    assert r_clean >= r_corr
    assert r_clean == 1.0 and r_corr < 1.0


def test_tolerant_recall_at_least_traditional():
    rng = np.random.default_rng(0)
    gt = np.tile(np.arange(20), (30, 1))
    gt_dist = np.tile(np.linspace(1.0, 2.0, 20), (30, 1))   # increasing true distances
    pred = gt[:, :10].copy()
    block = pr.recall_block(pred, gt, gt_dist=gt_dist, ks=(10,))
    assert block["tolerant_recall@10"] >= block["recall@10"] - 1e-9


def test_classify_passthrough():
    assert pr.classify(exception=RuntimeError("x")) == metrics.CRASH
    assert pr.classify(distances=np.array([[np.inf, 1.0]])) == metrics.NAN_INF
    assert pr.classify(distances=np.array([[1.0, 2.0]]), recall=0.5, clean_recall=0.95) == \
        metrics.SILENT_WRONG
