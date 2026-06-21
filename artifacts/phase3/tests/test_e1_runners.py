"""End-to-end runner tests for E1 / E3a / E3b against the stub adapter.

These are the Stage-1 delivery gate: the pipelines run, emit the correct schema, route recall and
collapse through the imported qp.metrics, switch adapter via --adapter with no code change, and are
deterministic. The scientific numbers come later from the x86 workstation (--adapter real).
"""
import argparse
import json
import os
import types

import pytest

import phase3_e1_vuln as e1
import phase3_e3a_additivity as e3a
import phase3_e3b_spatial as e3b


def _args(out, adapter="stub", **kw):
    base = dict(adapter=adapter, smoke=True, resume=False, ef=None,
                timeout=30.0, seed=1234, out=str(out))
    base.update(kw)
    return argparse.Namespace(**base)


# --- E1 -----------------------------------------------------------------------

def test_e1_end_to_end_schema_and_outputs(tmp_path):
    rows, crit = e1.run(_args(tmp_path))
    # vuln_map.json/csv + criticality.json written
    assert os.path.isfile(tmp_path / "vuln_map.json")
    assert os.path.isfile(tmp_path / "vuln_map.csv")
    assert os.path.isfile(tmp_path / "criticality.json")
    assert os.path.isfile(tmp_path / "raw" / "rabitq.records.jsonl")
    assert os.path.isfile(tmp_path / "raw" / "rabitq.done")
    # vuln_map schema: bootstrap CI + the unified pct_collapse must be present.
    r = rows[0]
    for key in ("index", "region", "bit_position_tag", "pct_collapse", "pct_catastrophic",
                "ci95_low", "ci95_high", "n_crash", "n_nan_inf", "n_samples"):
        assert key in r, key


def test_e1_rotation_collapses_pointers_crash_bincode_clean(tmp_path):
    rows, crit = e1.run(_args(tmp_path))
    by = {}
    for r in rows:
        by.setdefault(r["region"], []).append(r)
    # rotation is the single-point catastrophe: every rotation bit-class collapses.
    assert all(c["pct_collapse"] == 100.0 for c in by["rotation"])
    # pointers crash (detectable), never silent-collapse.
    assert any(c["n_crash"] > 0 for c in by["links"])
    assert all(c["pct_collapse"] == 0.0 for c in by["links"])
    # ex_factors registers nan-inf, excluded from silent collapse.
    assert any(c["n_nan_inf"] > 0 for c in by["ex_factors"])
    assert all(c["pct_collapse"] == 0.0 for c in by["ex_factors"])
    # bin_code low-impact -> not collapse.
    assert all(c["pct_collapse"] == 0.0 for c in by["bin_code"])


def test_e1_criticality_tiers_and_cost(tmp_path):
    _, crit = e1.run(_args(tmp_path))
    tier = {o["structure"]: o["scrub_tier"] for o in crit["criticality_order"]}
    assert tier["rotation"] == "frequent_scrub"        # collapse -> frequent scrub
    assert tier["links"] == "bounds_check"             # crash -> bounds check
    assert tier["bin_code"] == "none"                  # immune
    # Top-Down order puts rotation first (highest collapse).
    assert crit["criticality_order"][0]["structure"] == "rotation"
    # cost is priced over the per-vector-aggregated map and is a fraction of the index.
    cost = crit["cost"]
    assert cost["mem_cost"]["total_bytes"] > 0
    assert 0 < cost["protection_overhead_pct"] < 100
    # the report-and-stop note for ex_code travels with the artifact.
    assert any("REPORT-AND-STOP" in n for n in crit["notes"])


def _crit_row(region, *, pct_collapse=0.0, pct_catastrophic=0.0, n_crash=0, n_samples=100,
              max_dr=0.0, p99_dr=0.0, kind="data"):
    """Minimal vuln_map row carrying only the fields derive_criticality reads."""
    return {"region": region, "kind": kind, "pct_collapse": pct_collapse,
            "pct_catastrophic": pct_catastrophic, "n_crash": n_crash, "n_samples": n_samples,
            "max_dRecall@10": max_dr, "p99_dRecall@10": p99_dr}


def _agg(*names):
    return {"index": "RABITQ", "header": {}, "regions": [{"name": n, "byte_len": 64} for n in names]}


def test_criticality_global_singlepoint_upgrade():
    """A heavy-tailed GLOBAL structure (rare collapse bit, low frequency) upgrades to frequent_scrub;
    a per-vector structure with the SAME low frequency does not. Mirrors the real rotation data
    (mean pct_collapse ~0.2, max ΔRecall@10 ~0.5)."""
    rows = [
        # global, rare-but-devastating: pct_collapse 0.2 (<5) but >0 and max ΔRecall ~0.5
        _crit_row("rotation", pct_collapse=0.2, pct_catastrophic=12.0, max_dr=0.496, p99_dr=0.34),
        # per-vector, identical low frequency -> NOT upgraded (one bad bit among ~1e6 is negligible)
        _crit_row("bin_factors", pct_collapse=0.2, pct_catastrophic=12.0, max_dr=0.49, p99_dr=0.3),
    ]
    ranking = e1.derive_criticality(rows, rmap=None, agg=_agg("rotation", "bin_factors"))
    tier = {o["structure"]: o["scrub_tier"] for o in ranking}
    assert tier["rotation"] == "frequent_scrub"    # global + has a single-point-collapse bit
    assert tier["bin_factors"] == "crc_eb_lazy"     # harmful but not upgraded (per-vector)
    # severity tail is surfaced for transparency
    rot = next(o for o in ranking if o["structure"] == "rotation")
    assert rot["max_dRecall@10"] == 0.496 and rot["p99_dRecall@10"] == 0.34


def test_criticality_crash_does_not_count_as_harmful():
    """Crash share is detectable (bounds_check), not silent harm: a structure with sub-threshold
    silent harm + sub-threshold crash must NOT be escalated to crc_eb_lazy by summing the two."""
    # n_samples=100: 3 crashes (pct_crash 3 <5) + 3 silent-harm flips -> pct_catastrophic=6.
    rows = [_crit_row("bin_factors", pct_collapse=0.0, pct_catastrophic=6.0, n_crash=3,
                      n_samples=100, max_dr=0.02)]
    ranking = e1.derive_criticality(rows, rmap=None, agg=_agg("bin_factors"))
    # harmful_noncrash = 6 - 3 = 3 (<5) -> none (old code summed crash in and gave crc_eb_lazy).
    assert ranking[0]["scrub_tier"] == "none"


def test_e1_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    rows_a, _ = e1.run(_args(a))
    rows_b, _ = e1.run(_args(b))
    assert json.dumps(rows_a, sort_keys=True) == json.dumps(rows_b, sort_keys=True)


def test_e1_resume_reuses_shard(tmp_path):
    e1.run(_args(tmp_path))
    # second run with --resume must not re-sweep (done marker exists) and still emit a map.
    rows, _ = e1.run(_args(tmp_path, resume=True))
    assert rows and os.path.isfile(tmp_path / "vuln_map.json")


# --- E3a ----------------------------------------------------------------------

def test_e3a_prediction_formula_and_schema(tmp_path):
    results = e3a.run(_args(tmp_path))
    assert os.path.isfile(tmp_path / "e3a.json")
    rot = next(r for r in results if r["structure"] == "rotation")
    assert rot["p1_single_bit_collapse"] == 1.0
    for c in rot["curve"]:
        # predicted must equal the rollup model 1-(1-p1)^k exactly.
        expected = 1.0 - (1.0 - rot["p1_single_bit_collapse"]) ** c["k"]
        assert c["predicted"] == pytest.approx(round(expected, 4))
        assert c["additive"] == (c["abs_diff"] <= rot["tol"])
    assert rot["additivity_holds"] is True             # p1=1 -> pred=meas=1 for all k


# --- E3b ----------------------------------------------------------------------

def test_e3b_spatial_modes_and_schema(tmp_path):
    results = e3b.run(_args(tmp_path))
    assert os.path.isfile(tmp_path / "e3b.json")
    rot = next(r for r in results if r["structure"] == "rotation")
    for mode in ("clustered", "uniform", "cross_row"):
        m = rot["modes"][mode]
        assert m["n_trials"] == 8 and 0.0 <= m["collapse_frac"] <= 1.0
    # exposure ∝ 1/dispersion: clustered must be at least as dangerous as uniform on a small struct.
    assert rot["modes"]["clustered_more_dangerous"] is True
    # cross_row is reported at its TRUE 2-bit budget (not the fixed budget), with a provenance note.
    assert rot["modes"]["cross_row"]["budget_bits"] == 2
    assert rot["modes"]["clustered"]["budget_bits"] == rot["modes"]["uniform"]["budget_bits"]
    assert "not budget-matched" in rot["modes"]["cross_row"]["note"]


# --- reproducibility (#1: deterministic seed, no per-process hash()) -----------

def test_e3a_reproducible_across_runs(tmp_path):
    a, b = e3a.run(_args(tmp_path / "a")), e3a.run(_args(tmp_path / "b"))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_e3b_reproducible_across_runs(tmp_path):
    a, b = e3b.run(_args(tmp_path / "a")), e3b.run(_args(tmp_path / "b"))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- clean-baseline gates (#2 nan-inf detectability, #4 operating-point anchor) ----------------

def _fake_real_adapter(expected=0.983):
    """Minimal stand-in whose adapter_name resolves to 'real' (not the _stub module)."""
    return types.SimpleNamespace(EXPECTED_CLEAN_RECALL10=expected)


def _gate_args(allow_no_distances=False, clean_tol=0.02):
    return argparse.Namespace(allow_no_distances=allow_no_distances, clean_tol=clean_tol)


def test_gate_nan_inf_report_and_stop_on_real():
    adapter = _fake_real_adapter()
    clean = {"recall@10": 0.983, "nan_inf_supported": False}
    # real + no distances + not allowed -> report-and-stop
    with pytest.raises(RuntimeError, match="REPORT-AND-STOP"):
        e1._gate_clean_baseline(adapter, "real", clean, _gate_args())
    # explicit opt-in degrades instead of stopping
    e1._gate_clean_baseline(adapter, "real", clean, _gate_args(allow_no_distances=True))
    # stub never stops (distances always present in practice; here just verifies no raise)
    e1._gate_clean_baseline(adapter, "stub", clean, _gate_args())


def test_gate_clean_anchor_report_and_stop_on_real():
    adapter = _fake_real_adapter(expected=0.983)
    off = {"recall@10": 0.40, "nan_inf_supported": True}     # far below the plateau
    with pytest.raises(RuntimeError, match="REPORT-AND-STOP"):
        e1._gate_clean_baseline(adapter, "real", off, _gate_args())
    # within tolerance -> no raise
    ok = {"recall@10": 0.978, "nan_inf_supported": True}
    e1._gate_clean_baseline(adapter, "real", ok, _gate_args())
    # stub off-anchor only warns, never stops
    e1._gate_clean_baseline(adapter, "stub", off, _gate_args())
