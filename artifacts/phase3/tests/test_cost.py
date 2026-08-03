"""Verifications for the cost harness (Stage 0 component 3).

Synthetic region map -> analytic protection bytes; reconciliation vs a measured serialized
delta; monotonicity in protection strength. Numbers are filled by Stage 1; here we prove the
formulas and the reconciliation gate.
"""
import pytest

import phase3_cost as cost


# a synthetic region map in the adapter's {'regions':[{name,byte_len}...]} shape
REGION_MAP = {"regions": [
    {"name": "rotation", "byte_len": 64},
    {"name": "centroids", "byte_len": 8192},
    {"name": "bin_factors", "byte_len": 12},
    {"name": "ex_code", "byte_len": 96},
]}


def test_mem_cost_replication_and_checksum():
    assign = {
        "rotation":    {"mult": 3, "checksum_bytes": 4},     # triplicate + 4B CRC
        "bin_factors": {"mult": 2, "checksum_bytes": 0},     # mirror
    }
    mc = cost.mem_cost(REGION_MAP, assign)
    # (3-1)*64 + (2-1)*12 = 128 + 12 = 140 replication; +4 checksum
    assert mc["replication_bytes"] == 140
    assert mc["checksum_bytes"] == 4
    assert mc["total_bytes"] == 144
    assert mc["by_struct"]["rotation"] == 2 * 64 + 4


def test_mem_cost_unprotected_is_free():
    assert cost.mem_cost(REGION_MAP, {})["total_bytes"] == 0
    assert cost.mem_cost(REGION_MAP, {"ex_code": {"mult": 1}})["total_bytes"] == 0


def test_mem_cost_rejects_unknown_struct():
    with pytest.raises(KeyError):
        cost.mem_cost(REGION_MAP, {"nope": {"mult": 2}})


def test_scrub_cost_and_pct():
    assert cost.scrub_cost(64, 10) == 640.0
    assert cost.scrub_overhead_pct(64, 10, throughput_bytes_per_s=6400) == pytest.approx(10.0)


def test_reconcile_pass_and_fail():
    # analytic protection bytes must equal the measured serialized-size delta
    res = cost.reconcile(144, 144)
    assert res["ok"] and res["abs_diff"] == 0
    cost.reconcile(144, 145, rtol=0.02)                      # within tolerance: no raise
    with pytest.raises(AssertionError):
        cost.reconcile(144, 200, rtol=0.01)                  # too far apart


def test_total_cost():
    mc = cost.mem_cost(REGION_MAP, {"rotation": {"mult": 3, "checksum_bytes": 4}})
    assert cost.total_cost(mc, scrub_overhead=640.0) == mc["total_bytes"] + 640.0
    # the query-path term is opt-in: omitting it must leave the two-term result unchanged
    assert cost.total_cost(mc, 640.0, detection_overhead=0.0) == cost.total_cost(mc, 640.0)
    assert cost.total_cost(mc, 640.0, detection_overhead=12.5) == mc["total_bytes"] + 652.5


def test_detection_cost():
    # 96 B CRC'd per query at 1000 qps = 96 kB/s on the service path
    assert cost.detection_cost(96.0, 1000.0) == 96000.0
    assert cost.detection_cost(0, 1000.0) == 0.0
    with pytest.raises(ValueError):
        cost.detection_cost(-1.0, 1000.0)


def test_detection_overhead_pct():
    assert cost.detection_overhead_pct(300_000, 15_000_000) == pytest.approx(2.0)
    assert cost.detection_overhead_pct(0, 15_000_000) == 0.0
    with pytest.raises(ValueError):
        cost.detection_overhead_pct(300_000, 0)


def test_monotonicity():
    assert cost.monotonicity_check([(1, 0), (2, 64), (3, 128)]) is True
    assert cost.monotonicity_check([0, 64, 64, 140]) is True
    with pytest.raises(AssertionError):
        cost.monotonicity_check([0, 140, 64])
