"""Stage 2 Option B tests: CRC manifest + recovery adapters + Experiment-B driver.

All tests run offline against the stub adapter (arm64-safe) except the explicitly
skipif-gated real-binary twin at the bottom. Coverage:
  - manifest: round-trip, layout agreement, single-element detection, E5-CRC consistency
    (the spec's "chunk/CRC 與 E5 Python 側一致" rule, executable).
  - element selection/injection: determinism, confinement to ex_code, exact counts,
    restore identity, cross-row pair mirroring.
  - stub recovery contract: clean no-op, ordering fallback_eb >= drop > none,
    missing-manifest raise, stats shape.
  - driver: --gate green on stub, --sweep schema + provenance-before-records.
  - provenance: corruption block opt-in, backward-compat unchanged shape.
"""

import argparse
import json
import math
import os
import sys
import zlib

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qp import config, metrics, provenance
from qp.rabitq import crc_manifest as cm
from qp.rabitq import layout
from qp.rabitq import stub_adapter as adapter
import phase3_expb_recovery as expb


@pytest.fixture()
def clean_setup(tmp_path):
    buf = adapter.serialize_index()
    rmap = adapter.region_map()
    mf = str(tmp_path / "clean.crcmf")
    cm.write_manifest(buf, mf, rmap=rmap)
    return buf, rmap, mf, tmp_path


# ---------------------------------------------------------------------------
# CRC manifest
# ---------------------------------------------------------------------------

def test_manifest_roundtrip_matches_layout(clean_setup):
    buf, rmap, mf, _ = clean_setup
    m = cm.read_manifest(mf)
    n = rmap["header"]["cur_element_count"]
    assert m["header"]["n_entries"] == n
    assert m["header"]["index_file_size"] == buf.size
    assert m["header"]["algo"] == cm.ALGO_CRC32
    for e in (0, n // 2, n - 1):
        s, ln = layout.element_field_range(rmap, "ex_code", e)
        ent = m["entries"][e]
        assert (int(ent["offset"]), int(ent["len"])) == (s, ln)
        assert int(ent["crc"]) == zlib.crc32(bytes(buf[s:s + ln]))
    assert cm.verify_buffer(buf, m) == []


def test_manifest_detects_single_element_and_restores(clean_setup):
    buf, rmap, mf, _ = clean_setup
    m = cm.read_manifest(mf)
    b2 = buf.copy()
    s, _ = layout.element_field_range(rmap, "ex_code", 7)
    b2[s + 3] ^= 0x10
    assert cm.verify_buffer(b2, m) == [7]
    b2[s + 3] ^= 0x10
    assert cm.verify_buffer(b2, m) == []


def test_manifest_matches_e5_slope_crc(clean_setup):
    """Spec rule: C++/manifest chunk+CRC must agree with the E5 Python slope layer.

    E5 CRCs element-0's ex_code with zlib.crc32 in chunks of 4096 B; the field is 96 B, so
    E5's chunk 0 covers exactly the bytes manifest entry 0 covers -> identical CRC values.
    """
    from phase3_e5_recovery import RecoveryGuard, SMOKE_CFG as E5_CFG
    buf, rmap, mf, _ = clean_setup
    guard = RecoveryGuard(adapter, dict(E5_CFG), rmap)
    guard.init_from_clean(buf)
    m = cm.read_manifest(mf)
    assert guard._clean_crcs, "E5 slope layer found no ex chunks"
    assert int(m["entries"][0]["crc"]) == guard._clean_crcs[0]


def test_manifest_rejects_wrong_index_size(clean_setup):
    buf, _, mf, _ = clean_setup
    m = cm.read_manifest(mf)
    with pytest.raises(ValueError, match="byte index"):
        cm.verify_buffer(buf[:-1], m)


# ---------------------------------------------------------------------------
# Element selection + injection
# ---------------------------------------------------------------------------

def _cfg():
    return dict(expb.CFG_DEFAULTS)


@pytest.mark.parametrize("pattern", expb.PATTERNS)
def test_selection_deterministic_and_exact(pattern):
    n = 64
    for fraction in (0.05, 0.20, 0.57):
        a = expb.select_elements(pattern, fraction, n, config.SEED, _cfg())
        b = expb.select_elements(pattern, fraction, n, config.SEED, _cfg())
        assert a == b, "same seed must select the same elements"
        assert len(a) == min(n, math.ceil(fraction * n))
        assert len(set(a)) == len(a) and all(0 <= e < n for e in a)
        c = expb.select_elements(pattern, fraction, n, config.SEED + 1, _cfg())
        assert a != c or fraction > 0.9, "different seed should (generally) differ"


@pytest.mark.parametrize("pattern", expb.PATTERNS)
def test_injection_confined_and_restorable(clean_setup, pattern):
    buf, rmap, mf, _ = clean_setup
    m = cm.read_manifest(mf)
    n = rmap["header"]["cur_element_count"]
    elements = expb.select_elements(pattern, 0.20, n, config.SEED, _cfg())
    b2 = buf.copy()
    positions = expb.inject_elements(b2, rmap, elements, pattern, config.SEED, _cfg())

    # every flipped byte lies inside a SELECTED element's ex_code window
    level0_start, spe, field_off, field_len, _ = cm.field_geometry(rmap, "ex_code")
    sel = set(elements)
    for byte_pos, bit in positions:
        d, r = divmod(byte_pos - level0_start, spe)
        assert d in sel and field_off <= r < field_off + field_len, \
            f"flip at byte {byte_pos} escaped the selected ex_code windows"

    # the CRC scan must flag exactly the selected elements
    assert cm.verify_buffer(b2, m) == sorted(sel)

    # restore is identity
    from qp import faults
    faults.restore(b2, positions)
    assert np.array_equal(b2, buf)


def test_cross_row_pairs_share_offsets(clean_setup):
    buf, rmap, _, _ = clean_setup
    n = rmap["header"]["cur_element_count"]
    cfg = _cfg()
    stride = min(int(cfg["stride"]), max(1, n // 2))
    elements = expb.select_elements("cross_row_accum", 0.20, n, config.SEED, cfg)
    b2 = buf.copy()
    positions = expb.inject_elements(b2, rmap, elements, "cross_row_accum", config.SEED, cfg)
    level0_start, spe, field_off, _, _ = cm.field_geometry(rmap, "ex_code")
    by_elem = {}
    for byte_pos, bit in positions:
        d, r = divmod(byte_pos - level0_start, spe)
        by_elem.setdefault(d, set()).add((r, bit))
    eset = set(elements)
    mirrored = [e for e in elements if e + stride in eset]
    assert mirrored, "cross_row selection produced no intact pairs"
    for e in mirrored:
        assert by_elem[e] == by_elem[e + stride], \
            f"pair ({e},{e + stride}) must share in-field flip offsets"


# ---------------------------------------------------------------------------
# Stub recovery contract
# ---------------------------------------------------------------------------

def test_stub_recovery_contract(clean_setup):
    buf, rmap, mf, tmp_path = clean_setup
    clean_f = str(tmp_path / "clean.index")
    buf.tofile(clean_f)

    base_ids, _ = adapter.query_ids(clean_f)
    for mode in expb.RECOVERY_MODES:
        res = adapter.query_with_recovery(clean_f, mode, mf if mode != "none" else None)
        assert np.array_equal(res["ids"], base_ids), f"clean {mode} must equal baseline"
        assert res["stats"]["load"]["elements_crc_fail"] == 0

    n = rmap["header"]["cur_element_count"]
    b2 = buf.copy()
    elements = expb.select_elements("uniform_accum", 0.20, n, config.SEED, _cfg())
    expb.inject_elements(b2, rmap, elements, "uniform_accum", config.SEED, _cfg())
    cf = str(tmp_path / "corrupt.index")
    b2.tofile(cf)
    gt = adapter.load_groundtruth()
    rec = {}
    for mode in expb.RECOVERY_MODES:
        res = adapter.query_with_recovery(cf, mode, mf if mode != "none" else None)
        rec[mode] = metrics.recall_at_k(res["ids"], gt, config.K)
    assert rec["fallback_eb"] >= rec["drop"] > rec["none"]

    with pytest.raises(ValueError):
        adapter.query_with_recovery(cf, "drop", None)
    with pytest.raises(ValueError):
        adapter.query_with_recovery(cf, "bogus", mf)

    res = adapter.search_with_eb_fallback(cf, len(elements) / n, crc_manifest=mf)
    assert res["_eb_path"] is True
    assert res["stats"]["totals"]["fallbacks"] > 0


# ---------------------------------------------------------------------------
# --crc-mode pass-through (stub: plumbing only — see stub_adapter.query_with_recovery)
# ---------------------------------------------------------------------------

def test_stub_crc_mode_passthrough(clean_setup):
    """The flag must reach the adapter, be echoed, and not invent on-access numbers."""
    buf, rmap, mf, tmp_path = clean_setup
    clean_f = str(tmp_path / "clean.index")
    buf.tofile(clean_f)

    load = adapter.query_with_recovery(clean_f, "fallback_eb", mf, crc_mode="load")
    lazy = adapter.query_with_recovery(clean_f, "fallback_eb", mf, crc_mode="lazy")
    assert load["stats"]["crc_mode"] == "load"
    assert lazy["stats"]["crc_mode"] == "lazy"
    # Same predicate, same decision points -> same answer. (The stub cannot prove this for the
    # real search; that is phase3_expb_recovery --lazy-gate's job.)
    assert np.array_equal(load["ids"], lazy["ids"])
    # No load scan ran under lazy, so its totals are zero rather than carried over...
    assert lazy["stats"]["load"] == {"elements_checked": 0, "elements_crc_fail": 0}
    # ...and the on-access counters are None, not a plausible-looking invented number.
    assert lazy["stats"]["crc"]["checks"] is None
    assert lazy["stats"]["crc"]["distinct_elements_failed"] is None
    assert load["stats"]["crc"]["checks"] == 0

    lazy_t = adapter.query_with_recovery(clean_f, "fallback_eb", mf, crc_mode="lazy",
                                         crc_timer=True)
    assert lazy_t["stats"]["crc"]["timer_enabled"] is True

    eb = adapter.search_with_eb_fallback(clean_f, 0.0, crc_manifest=mf, crc_mode="lazy")
    assert eb["stats"]["crc_mode"] == "lazy"

    with pytest.raises(ValueError):
        adapter.query_with_recovery(clean_f, "fallback_eb", mf, crc_mode="bogus")
    # recovery=none never CRCs, so "when" is meaningless there — reject rather than echo.
    with pytest.raises(ValueError):
        adapter.query_with_recovery(clean_f, "none", None, crc_mode="lazy")


def test_crc_fail_count_check_is_mode_aware():
    """load pins the exact count; lazy bounds it — and 0 must still FAIL (not a tautology)."""
    def row(load_fail=None, distinct=None):
        return {"failure_mode": metrics.CLEAN,
                "stats": {"load": {"elements_crc_fail": load_fail},
                          "crc": {"distinct_elements_failed": distinct}}}

    expb._check_crc_fail_count(row(load_fail=50), 50, "fallback_eb", "load")
    with pytest.raises(AssertionError):
        expb._check_crc_fail_count(row(load_fail=49), 50, "fallback_eb", "load")

    expb._check_crc_fail_count(row(distinct=7), 50, "fallback_eb", "lazy")
    expb._check_crc_fail_count(row(distinct=50), 50, "fallback_eb", "lazy")
    with pytest.raises(AssertionError):      # nothing was ever checked
        expb._check_crc_fail_count(row(distinct=0), 50, "fallback_eb", "lazy")
    with pytest.raises(AssertionError):      # flagged elements Python never touched
        expb._check_crc_fail_count(row(distinct=51), 50, "fallback_eb", "lazy")
    # stub rows carry None (no real search knows who was consulted) -> skipped, not failed
    expb._check_crc_fail_count(row(distinct=None), 50, "fallback_eb", "lazy")


def test_real_adapter_requires_manifest_for_eb():
    """The real adapter must still report-and-stop without a manifest (never fake EB)."""
    from qp.rabitq import adapter as real_adapter
    with pytest.raises(RuntimeError, match="REPORT-AND-STOP"):
        real_adapter.search_with_eb_fallback("/nonexistent.index", 0.05)


# ---------------------------------------------------------------------------
# Driver: gates + sweep on stub
# ---------------------------------------------------------------------------

def _driver_args(**over):
    base = dict(gate=False, sweep=False, adapter="stub", smoke=True, seed=config.SEED,
                ef=2000, k=config.K, timeout=60.0, fractions=None, patterns=None,
                recovery=None, flips_per_element=4)
    base.update(over)
    return argparse.Namespace(**base)


def test_driver_gate_mode_stub_green(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", str(tmp_path))
    ctx = expb.setup_context(_driver_args(gate=True))
    assert expb.run_gates(ctx) == 0
    gates = json.load(open(os.path.join(ctx["out_dir"], "expb_gates.json")))
    assert gates["all_pass"] is True
    assert [g["pass"] for g in gates["gates"]] == [True] * 5


def test_driver_sweep_schema_and_provenance_first(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", str(tmp_path))
    args = _driver_args(sweep=True)
    ctx = expb.setup_context(args)
    meta_mtime = os.path.getmtime(ctx["meta_path"])
    summary = expb.run_sweep(ctx, ("uniform_accum",), (0.05, 0.20), expb.RECOVERY_MODES)

    rec = os.path.join(ctx["out_dir"], "expb_uniform_accum_ex_code.records.jsonl")
    rows = [json.loads(l) for l in open(rec)]
    assert len(rows) == 6
    required = {"pattern", "fraction", "fraction_actual", "n_elements", "bit_flips",
                "recovery", "recall@10", "cpp_recall", "delta_vs_clean", "failure_mode",
                "silent_collapse", "stats", "region", "seed", "ef", "index_sha256",
                "adapter"}
    assert required <= set(rows[0])
    assert {r["recovery"] for r in rows} == set(expb.RECOVERY_MODES)
    # the same corrupted file (same sha) is queried by all three modes at each fraction
    for f in (0.05, 0.20):
        shas = {r["index_sha256"] for r in rows if r["fraction"] == f}
        assert len(shas) == 1

    # provenance stamped BEFORE the first record was written
    assert meta_mtime <= os.path.getmtime(rec)
    meta = json.load(open(ctx["meta_path"]))["meta"]
    assert meta["corruption"]["corrupted_regions"] == ["ex_code"]
    assert meta["corruption"]["crc_manifest_sha256"]
    assert summary["meta"]["corruption"]["corrupted_regions"] == ["ex_code"]
    assert summary["eb_minus_drop"]["uniform_accum@0.2"] is not None


# ---------------------------------------------------------------------------
# Provenance corruption block
# ---------------------------------------------------------------------------

def test_provenance_corruption_block_opt_in():
    clean = {"recall@10": 0.98}
    args = argparse.Namespace(seed=config.SEED, smoke=True)
    old = provenance.collect_provenance(adapter, "stub", clean, {"ef": 2000}, args)
    assert "corruption" not in old
    new = provenance.collect_provenance(
        adapter, "stub", clean, {"ef": 2000}, args,
        index_sha256="ab" * 32, corrupted_regions=["ex_code"], recovery="fallback_eb")
    assert new["corruption"]["index_sha256"] == "ab" * 32
    assert new["corruption"]["recovery"] == "fallback_eb"
    # everything else identical to the old shape
    new.pop("corruption")
    assert new == old


def test_sha256_helpers(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert provenance.sha256_file(str(p)) == provenance.sha256_bytes(b"hello")
    assert provenance.sha256_bytes(np.frombuffer(b"hello", dtype=np.uint8)) == \
        provenance.sha256_file(str(p))


# ---------------------------------------------------------------------------
# Real-binary twin (x86 workstation only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("qp.rabitq.adapter", fromlist=["adapter"]).binaries_built(),
    reason="x86 binaries not built (run build_rabitq.sh on the workstation)")
def test_real_recovery_clean_noop(tmp_path):
    """Gate-2 twin: the REAL patched binary on the clean index is a byte-exact no-op."""
    from qp.rabitq import adapter as real_adapter
    mf = str(tmp_path / "real.crcmf")
    cm.write_manifest(real_adapter.INDEX_PATH, mf)
    base_ids, base_recall = real_adapter.query_ids(real_adapter.INDEX_PATH)
    res = real_adapter.query_with_recovery(
        real_adapter.INDEX_PATH, "fallback_eb", mf,
        out_path=str(tmp_path / "ids.ivecs"))
    assert np.array_equal(res["ids"], base_ids)
    assert res["stats"]["load"]["elements_crc_fail"] == 0
    assert abs(res["cpp_recall"] - base_recall) < 1e-9
