"""E7 (episode cost harness) unit tests — the pure, seed-deterministic pieces.

Everything here runs offline against the STUB geometry (arm64-safe, faiss-free). Coverage:
  - `build_schedule` is pure and deterministic in the root seed, places exactly one
    `single_cell` per step and exactly one `device_row` at `row_step`, and draws the
    single_cell anchor BYTE-UNIFORMLY over the whole index footprint (the documented
    interpretation of the brief's "vendor-A weights over strata" — see the module docstring);
  - `apply_injections` reproduces the qp.faults injectors exactly and round-trips through
    `faults.restore`, and the schedule accumulates (nothing is restored between steps);
  - `ReloadPolicy` is EDGE-triggered: it fires exactly once per crossing of the threshold and
    re-arms only after the fraction falls back below it (cite: RecoveryGuard `reload_threshold`,
    phase3_e5_recovery.py:50);
  - `repair_bytes` arithmetic (n_failed x manifest field_len, never a hardcoded 96);
  - `batch_repair` puts the ex_code windows back from the PRISTINE source and touches nothing
    else, and the manifest detector agrees it is clean afterwards;
  - `validate_cost_json` accepts the brief's schema, rejects structural damage, and treats the
    REQUIRES_MEASUREMENT sentinel as an explicitly-unmeasured (never invented) value;
  - a full `--adapter stub --smoke --panel a` run: CSV schema, paired arms, cumulative damage,
    and a genuinely-fired reload event.
"""

import csv
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qp import faults
from qp.rabitq import crc_manifest, layout
from qp.rabitq import stub_adapter

import phase3_e7_episode as e7


@pytest.fixture(scope="module")
def rmap():
    return stub_adapter.region_map()


@pytest.fixture(scope="module")
def clean_buf():
    return stub_adapter.serialize_index()


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, clean_buf, rmap):
    p = str(tmp_path_factory.mktemp("e7mf") / "clean.crcmf")
    crc_manifest.write_manifest(clean_buf, p, field="ex_code", rmap=rmap)
    return crc_manifest.read_manifest(p)


def test_stub_geometry_is_what_these_tests_assume(rmap):
    """Guard rail: if the stub geometry drifts, the concrete numbers below must be revisited."""
    hdr = rmap["header"]
    assert (hdr["padded_dim"], hdr["ex_bits"], hdr["cur_element_count"]) == (128, 6, 64)
    assert hdr["size_data_per_element"] == 272
    assert layout.element_field_range(rmap, "ex_code", 0)[1] == 96


# ---------------------------------------------------------------------------
# build_schedule — pure, deterministic, correct shape
# ---------------------------------------------------------------------------

def test_schedule_is_deterministic_in_seed(rmap):
    a = e7.build_schedule(rmap, steps=12, seed=7)
    b = e7.build_schedule(rmap, steps=12, seed=7)
    assert a == b, "same seed must reproduce the schedule byte-for-byte"


def test_schedule_changes_with_seed(rmap):
    a = e7.build_schedule(rmap, steps=12, seed=7)
    b = e7.build_schedule(rmap, steps=12, seed=8)
    assert [s["injections"][0]["anchor_byte"] for s in a] != \
           [s["injections"][0]["anchor_byte"] for s in b]


def test_schedule_is_pure(rmap, clean_buf):
    """Building a schedule must not touch any buffer (it is a plan, not an injection)."""
    before = clean_buf.copy()
    e7.build_schedule(rmap, steps=12, seed=7)
    assert np.array_equal(clean_buf, before)


def test_schedule_shape_one_single_cell_per_step_one_row_at_row_step(rmap):
    sched = e7.build_schedule(rmap, steps=10, seed=3, row_step=5)
    assert [s["step"] for s in sched] == list(range(1, 11))
    for s in sched:
        shapes = [i["shape"] for i in s["injections"]]
        assert shapes.count("single_cell") == 1
        if s["step"] == 5:
            assert shapes == ["single_cell", "device_row"]
        else:
            assert shapes == ["single_cell"]


def test_schedule_row_is_anchored_in_ex_code(rmap):
    sched = e7.build_schedule(rmap, steps=6, seed=3, row_step=5)
    row = [i for i in sched[4]["injections"] if i["shape"] == "device_row"][0]
    lo, length = layout.element_field_range(rmap, "ex_code", row["element"])
    assert lo <= row["anchor_byte"] < lo + length
    assert row["stratum"] == "ex_code"


def test_schedule_single_cell_anchor_is_byte_uniform_over_the_whole_index(rmap):
    """The documented ruling: the per-step cell is a random cell SOMEWHERE in the index.

    Byte-uniform means the share of anchors landing in level0 must match level0's byte share,
    not the 1/7 a stratum-uniform draw would give (level0 is 67% of the stub footprint).
    """
    total = int(rmap["total_bytes"])
    level0 = next(r for r in rmap["regions"] if r["name"] == "level0")
    lo, hi = level0["byte_start"], level0["byte_start"] + level0["byte_len"]
    anchors = [s["injections"][0]["anchor_byte"]
               for s in e7.build_schedule(rmap, steps=3000, seed=11)]
    assert all(0 <= a < total for a in anchors)
    share = sum(1 for a in anchors if lo <= a < hi) / len(anchors)
    expected = (hi - lo) / total
    assert abs(share - expected) < 0.05, f"level0 share {share:.3f} != byte share {expected:.3f}"


# ---------------------------------------------------------------------------
# apply_injections — the same damage qp.faults would do, and it accumulates
# ---------------------------------------------------------------------------

def test_apply_injections_matches_the_raw_injector(rmap, clean_buf):
    sched = e7.build_schedule(rmap, steps=1, seed=5)
    entry = sched[0]
    mine = clean_buf.copy()
    positions, _records = e7.apply_injections(mine, entry)

    theirs = clean_buf.copy()
    inj = entry["injections"][0]
    ref_pos, _ = faults.single_cell(theirs, (inj["anchor_byte"], 1), inj["seed"])
    assert positions == ref_pos
    assert np.array_equal(mine, theirs)


def test_apply_injections_round_trips_through_restore(rmap, clean_buf):
    sched = e7.build_schedule(rmap, steps=5, seed=5, row_step=5)
    work = clean_buf.copy()
    all_pos = []
    for entry in sched:
        pos, _ = e7.apply_injections(work, entry)
        all_pos.extend(pos)
    assert not np.array_equal(work, clean_buf), "5 steps incl. a device_row must change bytes"
    faults.restore(work, all_pos)
    assert np.array_equal(work, clean_buf)


def test_damage_accumulates_across_steps(rmap, clean_buf):
    """Cumulative, never restored: the differing-byte set may only grow."""
    sched = e7.build_schedule(rmap, steps=6, seed=9, row_step=5)
    work = clean_buf.copy()
    prev = 0
    for entry in sched:
        e7.apply_injections(work, entry)
        now = int(np.count_nonzero(work != clean_buf))
        assert now >= prev
        prev = now
    assert prev > 100, "the step-5 device_row must dominate the footprint"


# ---------------------------------------------------------------------------
# ReloadPolicy — edge-triggered, exactly once per crossing
# ---------------------------------------------------------------------------

def test_reload_policy_fires_once_per_crossing():
    p = e7.ReloadPolicy(threshold=0.10)
    fired = [p.observe(f) for f in (0.0, 0.05, 0.11, 0.12, 0.30, 0.02, 0.15)]
    assert fired == [False, False, True, False, False, False, True]
    assert p.n_fired == 2


def test_reload_policy_threshold_is_inclusive():
    """The brief says `failed_frac >= 0.10` (E5's own guard uses `>`); we follow the brief."""
    assert e7.ReloadPolicy(threshold=0.10).observe(0.10) is True
    assert e7.ReloadPolicy(threshold=0.10).observe(0.0999999) is False


def test_reload_policy_rearms_only_after_dropping_below():
    p = e7.ReloadPolicy(threshold=0.10)
    assert p.observe(0.5) is True
    assert p.observe(0.5) is False          # still above: already handled, do not re-fire
    assert p.armed is False
    assert p.observe(0.0) is False          # the drop re-arms but does not itself fire
    assert p.armed is True
    assert p.observe(0.5) is True


def test_reload_policy_default_threshold_is_the_cited_one():
    assert e7.ReloadPolicy().threshold == 0.10 == e7.RELOAD_THRESHOLD


# ---------------------------------------------------------------------------
# repair-bytes arithmetic
# ---------------------------------------------------------------------------

def test_repair_bytes_arithmetic(manifest):
    field_len = int(manifest["header"]["field_len"])
    assert field_len == 96
    assert e7.repair_bytes(0, field_len) == 0
    assert e7.repair_bytes(1, field_len) == 96
    assert e7.repair_bytes(1234, field_len) == 1234 * 96


def test_repair_bytes_reads_the_field_len_from_the_manifest_not_a_constant(manifest):
    """A hardcoded 96 would silently lie on any other ex_bits / padded_dim geometry."""
    assert e7.repair_bytes(10, 48) == 480
    assert e7.repair_bytes(10, int(manifest["header"]["field_len"])) == 960


def test_repair_bytes_rejects_nonsense():
    with pytest.raises(ValueError):
        e7.repair_bytes(-1, 96)
    with pytest.raises(ValueError):
        e7.repair_bytes(1, 0)


# ---------------------------------------------------------------------------
# batch_repair — pristine ex windows back in, nothing else touched
# ---------------------------------------------------------------------------

def test_batch_repair_restores_only_the_failed_ex_windows(rmap, clean_buf, manifest):
    work = clean_buf.copy()
    victims = [3, 17, 40]
    for e in victims:
        s, _ = layout.element_field_range(rmap, "ex_code", e)
        work[s] ^= 0xFF
    # ...plus one flip OUTSIDE the manifest's field, which the repair must NOT touch.
    bin_s, _ = layout.element_field_range(rmap, "bin_code", 1)
    work[bin_s] ^= 0x01

    assert e7.failed_elements(work, manifest) == victims
    n_bytes, wall_s = e7.batch_repair(stub_adapter, work, rmap, victims, field="ex_code")
    assert n_bytes == len(victims) * 96
    assert wall_s >= 0.0
    assert e7.failed_elements(work, manifest) == []
    diff = np.flatnonzero(work != clean_buf).tolist()
    assert diff == [bin_s], "repair must leave non-ex damage exactly where it was"


def test_batch_repair_of_nothing_is_a_no_op(rmap, clean_buf):
    work = clean_buf.copy()
    n_bytes, _ = e7.batch_repair(stub_adapter, work, rmap, [], field="ex_code")
    assert n_bytes == 0
    assert np.array_equal(work, clean_buf)


def test_batch_repair_reads_the_pristine_source_not_the_working_buffer(rmap, clean_buf,
                                                                      manifest, monkeypatch):
    """House rule: the clean source is re-read from persistent storage on every repair."""
    calls = []
    real = stub_adapter.read_serialized_range

    def spy(byte_start, byte_len, path=None):
        calls.append((int(byte_start), int(byte_len)))
        return real(byte_start, byte_len, path)

    monkeypatch.setattr(stub_adapter, "read_serialized_range", spy)
    work = clean_buf.copy()
    s, _ = layout.element_field_range(rmap, "ex_code", 5)
    work[s] ^= 0xFF
    e7.batch_repair(stub_adapter, work, rmap, [5], field="ex_code")
    assert calls == [(s, 96)]


# ---------------------------------------------------------------------------
# Pointer bounds-check layer (ours arm) — pread-backed, so its IO is counted
# ---------------------------------------------------------------------------

def test_pristine_source_preads_and_counts(clean_buf):
    src = e7.PristineSource(stub_adapter)
    got = src[100:164]
    assert np.array_equal(got, clean_buf[100:164])
    assert (src.bytes_read, src.n_reads) == (64, 1)


def test_pristine_source_refuses_non_contiguous_access():
    src = e7.PristineSource(stub_adapter)
    with pytest.raises(TypeError):
        src[0:100:2]
    with pytest.raises(TypeError):
        src[5]


def test_pointer_repair_restores_an_oob_neighbour_from_the_pristine_index(rmap, clean_buf):
    work = clean_buf.copy()
    start, length = layout.element_field_range(rmap, "links", 3)
    work[start + 4:start + 8] = np.frombuffer(np.uint32(50_000).tobytes(), dtype=np.uint8)
    res = e7.pointer_repair(stub_adapter, work, rmap)
    assert res["restored"] == 1 and res["detail"] == {"links": 1}
    assert res["bytes"] == length and res["reads"] == 1
    assert np.array_equal(work, clean_buf)


def test_pointer_repair_is_a_no_op_on_a_clean_index(rmap, clean_buf):
    """If it were not, the ours arm would report fictitious repair bytes on every step."""
    work = clean_buf.copy()
    res = e7.pointer_repair(stub_adapter, work, rmap)
    assert (res["restored"], res["bytes"], res["reads"]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Cold-cache verdict — the honesty gate, pure so it needs no GB of IO to test
# ---------------------------------------------------------------------------

def test_cold_verdict_true_when_the_cold_read_is_properly_slower():
    big = e7.MIN_COLD_TEST_BYTES * 4
    verified, reason = e7.cold_measurement_verdict(0.12, 0.022, big)
    assert verified is True and "really evicted" in reason


def test_cold_verdict_false_when_the_cold_read_was_served_from_cache():
    big = e7.MIN_COLD_TEST_BYTES * 4
    verified, reason = e7.cold_measurement_verdict(0.024, 0.022, big)
    assert verified is False and "page cache" in reason


def test_cold_verdict_boundary_is_the_documented_ratio():
    big = e7.MIN_COLD_TEST_BYTES * 4
    assert e7.cold_measurement_verdict(1.5, 1.0, big)[0] is True
    assert e7.cold_measurement_verdict(1.49, 1.0, big)[0] is False


def test_cold_verdict_is_undecidable_on_a_file_too_small_to_time():
    """A 25 KB stub index cannot distinguish NVMe from cache: None, not a coin-flip boolean."""
    verified, reason = e7.cold_measurement_verdict(0.0014, 0.0009, 25_820)
    assert verified is None and "cannot decide" in reason


def test_cold_verdict_is_undecidable_without_samples():
    assert e7.cold_measurement_verdict(None, 0.02, 1 << 30)[0] is None
    assert e7.cold_measurement_verdict(0.12, None, 1 << 30)[0] is None


# ---------------------------------------------------------------------------
# e7_cost.json schema
# ---------------------------------------------------------------------------

def _good_cost():
    return {
        "reload": {"seconds": 1.5, "bytes": 280573456, "gbps": 0.19, "method": "fresh_copy",
                   "downtime_upper_bound_full_batch_s": 18.0,
                   "definition": "seconds is the cold sequential read, the I/O component",
                   "method_detail": "fresh copy + posix_fadvise(POSIX_FADV_DONTNEED)",
                   "cache_eviction_verified": True},
        "crash_restart": {"control_path_s": 0.01, "control_path_method": "hwpoison",
                          "downtime_s": 1.51, "io_bytes": 280573456,
                          "downtime_upper_bound_full_batch_s": 18.01,
                          "method": "control_path + reload_full"},
        "eager": {"downtime_s": 1.5, "io_bytes": 280573456, "method": "reload_full",
                  "downtime_upper_bound_full_batch_s": 18.0},
        "ours": {"downtime_s": 0, "repair_io_bytes": 2880, "repair_wall_s": 0.004,
                 "method": "measured batch pread+patch (panel A)"},
        "meta": {"provenance": "..."},
    }


def test_cost_json_schema_accepts_a_complete_document():
    assert e7.validate_cost_json(_good_cost()) == []
    e7.assert_cost_json(_good_cost())          # must not raise


def test_cost_json_schema_allows_the_sentinel_for_unmeasured_values():
    doc = _good_cost()
    doc["crash_restart"]["control_path_s"] = e7.SENTINEL
    doc["crash_restart"]["control_path_method"] = e7.SENTINEL
    doc["crash_restart"]["downtime_s"] = e7.SENTINEL
    assert e7.validate_cost_json(doc) == []


def test_cost_json_schema_rejects_missing_blocks_and_keys():
    doc = _good_cost()
    del doc["eager"]
    assert any("eager" in p for p in e7.validate_cost_json(doc))

    doc = _good_cost()
    del doc["reload"]["gbps"]
    assert any("reload.gbps" in p for p in e7.validate_cost_json(doc))


def test_cost_json_schema_rejects_a_non_numeric_measurement():
    doc = _good_cost()
    doc["reload"]["seconds"] = "fast"
    problems = e7.validate_cost_json(doc)
    assert any("reload.seconds" in p for p in problems)
    with pytest.raises(ValueError):
        e7.assert_cost_json(doc)


def test_cost_json_schema_rejects_an_unknown_reload_method():
    doc = _good_cost()
    doc["reload"]["method"] = "vibes"
    assert any("reload.method" in p for p in e7.validate_cost_json(doc))


def test_cost_json_reload_block_is_self_describing_on_its_own():
    """A consumer reading only doc["reload"] must learn what `seconds` is, how the file was
    cooled, whether that was verified, and the other end of the bracket — without meta."""
    doc = _good_cost()
    for key in ("definition", "method_detail", "cache_eviction_verified",
                "downtime_upper_bound_full_batch_s"):
        broken = _good_cost()
        del broken["reload"][key]
        assert any(f"reload.{key}" in p for p in e7.validate_cost_json(broken)), key
    assert e7.validate_cost_json(doc) == []


def test_cost_json_rejects_a_cold_number_whose_eviction_check_did_not_pass():
    """The I2 gate: an unverified-cold read is page-cache bandwidth and must be sentinelled."""
    for verdict in (False, None):
        doc = _good_cost()
        doc["reload"]["cache_eviction_verified"] = verdict
        problems = e7.validate_cost_json(doc)
        assert any("reload.seconds" in p and "cache_eviction_verified" in p for p in problems)
        assert any("eager.downtime_s" in p for p in problems)

        # ...and the same document is valid once the gated fields are sentinelled.
        for dotted in ("reload.seconds", "reload.gbps",
                       "reload.downtime_upper_bound_full_batch_s", "eager.downtime_s",
                       "eager.downtime_upper_bound_full_batch_s"):
            b, k = dotted.split(".")
            doc[b][k] = e7.SENTINEL
        assert e7.validate_cost_json(doc) == []


def test_cost_json_cache_eviction_verified_must_be_tri_state():
    doc = _good_cost()
    doc["reload"]["cache_eviction_verified"] = "probably"
    assert any("cache_eviction_verified" in p for p in e7.validate_cost_json(doc))


def test_cost_json_io_bytes_must_be_the_real_index_size_not_a_guess():
    """The brief pins io_bytes = the serialized index size; a mismatch across blocks is a bug."""
    doc = _good_cost()
    doc["eager"]["io_bytes"] = 123
    assert any("io_bytes" in p for p in e7.validate_cost_json(doc))


# ---------------------------------------------------------------------------
# CSV contract
# ---------------------------------------------------------------------------

def test_csv_columns_are_exactly_the_briefs_schema():
    assert e7.CSV_COLUMNS == ["step", "queries_served", "arm", "recall", "elements_crc_fail",
                              "reload_event", "reload_bytes"]


# ---------------------------------------------------------------------------
# Stub sandbox gate (house rule: documented-fake numbers never land in artifacts/)
# ---------------------------------------------------------------------------

def test_stub_sandbox_gate_blocks_the_real_artifacts_tree(tmp_path):
    from qp import config
    real = os.path.join(config.ROOT, "artifacts", "phase3", "e7")
    with pytest.raises(RuntimeError):
        e7.assert_stub_sandbox("stub", real)
    e7.assert_stub_sandbox("stub", str(tmp_path))      # must not raise
    e7.assert_stub_sandbox("real", real)               # real results belong there


# ---------------------------------------------------------------------------
# End-to-end stub smoke (panel A)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("e7_panelA"))
    e7.main(["--adapter", "stub", "--smoke", "--panel", "a", "--out", out, "--steps", "6"])
    return out


def test_smoke_writes_the_csv_with_the_exact_schema(smoke_run):
    with open(os.path.join(smoke_run, "e7_panelA.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == e7.CSV_COLUMNS
    assert {r["arm"] for r in rows} == set(e7.ARMS)


def test_smoke_arms_are_paired_on_the_same_schedule(smoke_run):
    with open(os.path.join(smoke_run, "e7_panelA.csv")) as fh:
        rows = list(csv.DictReader(fh))
    per_arm = {a: [r for r in rows if r["arm"] == a] for a in e7.ARMS}
    assert len({len(v) for v in per_arm.values()}) == 1
    for a in e7.ARMS:
        steps = [int(r["step"]) for r in per_arm[a]]
        assert steps == sorted(steps)
        served = [int(r["queries_served"]) for r in per_arm[a]]
        assert served == sorted(served) and served[0] > 0


def test_smoke_ours_arm_actually_fires_a_reload(smoke_run):
    """The stub's 64-element geometry puts a device_row far over the 10% threshold, so the
    threshold branch and the batch pread+patch loop are genuinely executed offline."""
    with open(os.path.join(smoke_run, "e7_panelA.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == "ours"]
    events = [r for r in rows if int(r["reload_event"]) == 1]
    assert events, "no reload event fired in the stub smoke"
    for r in events:
        assert int(r["reload_bytes"]) == int(r["elements_crc_fail"]) * 96 or \
               int(r["reload_bytes"]) > 0


def test_smoke_summary_records_provenance_and_the_schedule(smoke_run):
    with open(os.path.join(smoke_run, "e7_panelA_summary.json")) as fh:
        s = json.load(fh)
    assert s["adapter"] == "stub"
    assert s["meta"]["adapter_type"] == "stub"
    assert s["meta"]["platform_confirmed_real"] is False
    assert s["schedule"]["row_step"] == e7.FULL["row_step"]
    assert s["schedule"]["single_cell_draw"].startswith("byte-uniform")
    assert s["reload_policy"]["threshold"] == e7.RELOAD_THRESHOLD


def test_smoke_raw_jsonl_and_done_marker_exist(smoke_run):
    raw = os.path.join(smoke_run, "raw", "e7_panelA.records.jsonl")
    assert os.path.exists(raw)
    assert os.path.exists(os.path.join(smoke_run, "raw", "e7_panelA.done"))
    recs = [json.loads(l) for l in open(raw)]
    assert recs and all("elements_crc_fail_py" in r for r in recs)
    assert all(r["elements_crc_fail_source"] in ("cpp_stats", "python_manifest") for r in recs)


def test_smoke_ours_arm_runs_and_accounts_for_the_pointer_layer(smoke_run):
    """The pointer layer is what keeps the ours arm alive through the device_row (the ex-only
    arm segfaults on the real index); its bytes must be measured, not free."""
    raw = os.path.join(smoke_run, "raw", "e7_panelA.records.jsonl")
    recs = [json.loads(l) for l in open(raw)]
    ours = [r for r in recs if r["arm"] == "ours"]
    ignore = [r for r in recs if r["arm"] == "ignore"]
    assert all(r["oob_restored"] is not None for r in ours)
    assert all(r["oob_restored"] is None for r in ignore), "the ignore arm repairs nothing"
    assert sum(r["oob_repair_bytes"] for r in ours) > 0, "the device_row must trip the check"
    for r in ours:
        if r["oob_restored"]:
            assert r["oob_repair_bytes"] > 0 and r["oob_repair_reads"] > 0


def test_smoke_crashed_arm_does_not_report_a_flattering_recall_min(smoke_run):
    """I3: the ignore arm is DOWN from the device_row on. A min over the steps it survived would
    read as its best case, so the headline aggregates must be null and say why."""
    with open(os.path.join(smoke_run, "e7_panelA_summary.json")) as fh:
        s = json.load(fh)
    ig = s["arm_summary"]["ignore"]
    assert ig["n_crash"] > 0 and ig["first_crash_step"] is not None
    assert ig["recall_min"] is None and ig["max_delta_recall"] is None
    assert ig["n_rows_with_recall"] == ig["rows"] - ig["n_crash"]
    assert "CRASHED" in ig["crash_blind_note"]
    # ...and the surviving-step figures are still available, under a name that says so.
    assert ig["recall_min_over_served_steps"] is not None

    ours = s["arm_summary"]["ours"]
    assert ours["n_crash"] == 0
    assert ours["recall_min"] == ours["recall_min_over_served_steps"] is not None
    assert ours["crash_blind_note"] is None


def test_smoke_panel_b_sentinels_an_index_too_small_to_verify_cold(tmp_path):
    """The stub index is 25 KB, so the cold/warm test is undecidable — every derived timing
    must come out as the sentinel rather than as a number nobody can defend."""
    out = str(tmp_path)
    e7.main(["--adapter", "stub", "--smoke", "--panel", "b", "--out", out, "--reps", "2"])
    with open(os.path.join(out, "e7_cost.json")) as fh:
        doc = json.load(fh)
    assert e7.validate_cost_json(doc) == []
    assert doc["reload"]["cache_eviction_verified"] is None
    assert doc["reload"]["seconds"] == doc["reload"]["gbps"] == e7.SENTINEL
    assert doc["eager"]["downtime_s"] == e7.SENTINEL
    # The raw medians survive for diagnosis even though the derived numbers are sentinels.
    assert doc["meta"]["reload_full"]["cold_read_s"]["median"] > 0


def test_smoke_summary_breaks_the_repair_cost_into_its_two_layers(smoke_run):
    with open(os.path.join(smoke_run, "e7_panelA_summary.json")) as fh:
        s = json.load(fh)
    rep = s["ours_repair"]
    bd = rep["breakdown"]
    assert bd["bounds_check_enabled"] is True
    assert rep["episode_total_repair_io_bytes"] == bd["ex_window_bytes"] + bd["pointer_field_bytes"]
    assert rep["repair_io_bytes"] <= rep["episode_total_repair_io_bytes"]
