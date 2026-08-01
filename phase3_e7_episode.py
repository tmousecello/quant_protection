#!/usr/bin/env python3
"""Phase 3 — E7: episode cost harness (CIDR spec E2, amended).

E1/E6 asked what a fault does to recall. E5/Experiment-B asked whether the recovery stack holds.
E7 asks the operator's question: over an EPISODE — a long run of queries during which damage
accumulates — what does each response policy actually cost? Two things are measured, and the
paper's cost table (E3) imports the second:

  Panel A (--panel a)  the RECALL TRAJECTORY of two arms against cumulative queries served.
  Panel B (--panel b)  the WALL-CLOCK / IO cost of the three alternative responses, measured.

PANEL A. S steps (default 40); each step is one full eval of the fixed query set. Injection
schedule (deterministic in --seed, replayed identically for both arms so the comparison is
PAIRED — same physical faults, different policy):
  * one `single_cell` per step, anchor drawn BYTE-UNIFORMLY over the whole index footprint;
  * one `device_row` anchored in an element's `ex_code` at step `--row-step` (default 5) — the
    spec's canonical fault;
  * damage is CUMULATIVE and never restored (the accumulation the episode is about).

  INTERPRETATION NOTE (brief wording "one single_cell per step drawn with vendor-A weights over
  strata"): shape_weights.json weights SHAPES (single_cell / device_row / device_column), not
  index strata, so there is no vendor-A distribution over strata to draw from. The physically
  honest model of "a random cell somewhere in the index" is a byte-uniform anchor over the whole
  serialized footprint — a DRAM cell does not know which index structure it holds — so that is
  what this harness does, and it is stamped into the summary meta as
  `schedule.single_cell_draw`. Byte-uniform automatically reproduces the structures' byte shares
  (level0 97%, upper_links 3%, centroids/rotation/header ~0%), which is the correct prior.

Arms:
  ignore — `adapter.search_corrupted` every step. Nothing detects, nothing repairs; the recall
           depression after the step-5 row is permanent (and a corrupted pointer may crash the
           search outright, which is recorded as `crash`, not smoothed away).
  ours   — the paper's recovery stack: the pointer bounds-check layer (E6's full-index
           `bounds_check_full`, restoring from fresh preads of the pristine index) runs first,
           then `adapter.query_with_recovery(..., "fallback_eb", manifest)` every step.
           `failed_frac`
           = elements_crc_fail / cur_element_count from the binary's stats JSON; when it reaches
           `--reload-threshold` (default 0.10 — cite: RecoveryGuard `reload_threshold`,
           phase3_e5_recovery.py:50) a BATCH REPAIR runs: each failed element's ex_code window is
           pread fresh from the pristine on-disk index (adapter.read_serialized_range — a new
           open/seek/read every call, never cached, per the house rule) and patched into the
           working buffer. The step's row records `reload_event=1` and `reload_bytes`, and recall
           recovers on the next step.

  WHICH ELEMENTS FAILED: the patched `exp_dumpids` stats JSON reports only COUNTS
  (`load.elements_checked` / `load.elements_crc_fail`; rabitq_instrumentation/exp_dumpids.cpp
  L193-194 and recovery-changes.patch — no per-element id list exists). The failed set is
  therefore derived in Python by re-verifying the working buffer against the same CRC manifest
  the C++ side was given (qp.rabitq.crc_manifest.read_manifest + verify_buffer). That is
  deployment-faithful rather than a workaround: the manifest IS the detector in Option B, and
  both sides run the identical CRC-32 over the identical 96-B windows. The two counts are
  cross-checked every step (`elements_crc_fail` vs `elements_crc_fail_py`) and a disagreement is
  logged loudly.

  THRESHOLD ARITHMETIC / final repair step. On the real 1M-element index the cited 0.10 threshold
  is 100,000 failed elements, while the canonical schedule damages ~30 (an 8 KB row spans
  8192/272 = 30.1 elements) plus a trickle from the single cells — so the threshold is NEVER
  crossed, which is itself a finding (EB-fallback carries that damage; no repair is due). To
  still obtain a MEASURED repair cost for Panel B's `ours` row, an extra final step repairs
  whatever is still failing and re-evaluates (`--no-final-repair` to switch off). Its row is
  tagged `phase="final_repair"` in the raw JSONL. The threshold branch itself is exercised for
  real in the stub smoke (64 elements, so a row is far over 10%) and unit-tested directly.

  DEVIATION — the ours arm carries the pointer bounds-check layer. The brief's one-line arm
  definition names only EB-fallback + the ex batch repair, both of which see only the CRC
  manifest's `ex_code` field. The canonical 8 KB row spans ~30 WHOLE elements, so it also
  destroys their links/cluster_id/label, and the first real validation run segfaulted the C++
  search at the row step in BOTH arms (SIGSEGV, CalledProcessError -11) — an ex-only ours arm
  does not survive its own headline fault and therefore has no trajectory to plot. The pointer
  layer is RecoveryGuard's third layer (phase3_e5_recovery.py) and E6's arm ON already runs the
  same full-index version, so this makes `ours` the stack as specified elsewhere rather than a
  new mechanism. Its restores pread from the pristine index (PristineSource) so every repaired
  byte is counted, and `--no-bounds-check` reproduces the literal brief arm and its crash.

PANEL B (--panel b). Three response policies, every number measured:
  reload_full   the cold-cache cost of reading the whole index back, reported as a BRACKET:
                (i) `cat index > /dev/null` — the pure sequential read, i.e. the I/O component
                of a reload, published as `seconds`/`gbps`; and (ii) one COMPLETE clean eval on
                the cold file (index load + all 10K queries), published as
                `downtime_upper_bound_full_batch_s`. (ii) is NOT "time to the first served
                batch" — it includes the entire search, so it overstates a reload; the true
                downtime lies between the two and the in-memory reconstruct inside hnsw.load()
                is not separately instrumented. Cold is obtained by `sudo -n` drop_caches when
                available (method="drop_caches"), else by a FRESH COPY of the index plus an
                explicit `posix_fadvise(POSIX_FADV_DONTNEED)` on it (method="fresh_copy") — a
                plain copy is NOT cold, since copying populates the page cache with the
                destination, so the fadvise eviction is what actually makes the fallback honest.
                Every repetition also times a WARM read, and `cache_eviction_verified` gates the
                whole thing: cold must be >=1.5x warm or the run report-and-stops (page-cache
                bandwidth is not an NVMe measurement). `--reps` repetitions, median + IQR.
  crash_restart HWPOISON control-path time (Task 5's helper, consumed via --control-path-json)
                + reload_full. With no helper output the control path is written as the
                REQUIRES_MEASUREMENT sentinel and downtime_s stays a sentinel too — never
                invented.
  eager         full reload on the first CRC failure -> downtime_s = reload_full.
  ours          downtime_s = 0 (service continues degraded; Panel A shows the quality dip
                instead); repair_io_bytes / repair_wall_s are the MEASURED batch pread+patch
                numbers imported from Panel A's summary.

MICROBENCH (--microbench). Per-vector CRC vs per-vector distance cost, both from the real binary.
The CRC numerator (`load.crc_scan_ns`) and the distance numerator (`search_wall_ns`) come from the
instrumentation patch; a binary without it yields REQUIRES_MEASUREMENT sentinels naming the exact
stats keys instead of a derived guess. Alongside them: elements_checked, totals.consults, the
subprocess wall, and a clearly labelled Python-side zlib reference that is explicitly NOT the
reported C++ ratio. The ratio is one-sided — the consult denominator carries graph traversal, so
it OVERSTATES pure distance and the ratio therefore UNDERSTATES: read it as "CRC costs at least
this fraction of a distance computation / is at most 1/ratio times cheaper", never the reverse.

Outputs (under --out; default artifacts/phase3/e7, --smoke -> artifacts_smoke/phase3/e7):
  raw/e7_panelA.records.jsonl  one row per eval (schedule, counters, timings, error)
  raw/e7_panelA.done           completion marker
  e7_panelA.csv                step,queries_served,arm,recall,elements_crc_fail,reload_event,reload_bytes
  e7_panelA_summary.json       provenance meta + schedule + reload events + trajectory summary
  e7_cost.json                 Panel B — THE FILE E3 IMPORTS
  e7_microbench.json           CRC-vs-distance ratio (or the sentinels naming what is missing)

Usage:
  python phase3_e7_episode.py --adapter stub --smoke --panel a     # dev pipeline gate
  python phase3_e7_episode.py --panel a                            # workstation, ~20 min
  python phase3_e7_episode.py --panel a --arms ours                # shard one arm per core
  python phase3_e7_episode.py --panel b --control-path-json <p>    # measured cost table
  python phase3_e7_episode.py --microbench
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from qp import config, metrics, provenance
from qp.rabitq import crc_manifest, layout
from qp.rabitq.registry import get_adapter, adapter_name
from qp.rawio import RawWriter

import phase3_e6_shapes as e6

# Reused verbatim from E6 rather than re-implemented (single source of truth for each).
csv_list = e6.csv_list
stub_sandbox_ok = e6.stub_sandbox_ok
assert_stub_sandbox = e6.assert_stub_sandbox
cell_seed = e6.cell_seed
lane_seed = e6.lane_seed
sample_anchor = e6.sample_anchor
inject_shape = e6.inject_shape
LANE_INJECT = e6.LANE_INJECT

ARMS = ("ignore", "ours")
RECOVERY_MODE = "fallback_eb"
MANIFEST_FIELD = "ex_code"
ROW_STRATUM = "ex_code"
ROW_SHAPE = "device_row"
CELL_SHAPE = "single_cell"
WHOLE_INDEX = "whole_index"          # the single_cell's "stratum" label (see module docstring)

# Lazy-reload trigger fraction. CITE: RecoveryGuard `reload_threshold`, phase3_e5_recovery.py:50
# (SMOKE_CFG["reload_threshold"] = 0.10). The brief specifies `>=`; E5's own guard uses `>`, and
# the difference only matters exactly at the boundary.
RELOAD_THRESHOLD = 0.10

# Written wherever a value is genuinely not measured yet. Never replaced by a plausible guess.
SENTINEL = "REQUIRES_MEASUREMENT"

FULL = dict(steps=40, ef=2000, row_step=5, row_bytes=8192, p_in_row=0.5,
            reload_threshold=RELOAD_THRESHOLD, reps=5, final_repair=True, bounds_check=True)
SMOKE = dict(FULL, steps=7, ef=64, reps=2)

CSV_COLUMNS = ["step", "queries_served", "arm", "recall", "elements_crc_fail",
               "reload_event", "reload_bytes"]

SINGLE_CELL_DRAW = (
    "byte-uniform over the whole serialized index footprint (a DRAM cell does not know which "
    "index structure it holds). The brief's 'vendor-A weights over strata' names shape_weights."
    "json, which weights SHAPES, not strata; see the module docstring's interpretation note.")

# Panel B method enums (the brief's schema).
RELOAD_METHODS = ("drop_caches", "fresh_copy")
CONTROL_PATH_METHODS = ("hwpoison", "mprotect_fallback")

# Cold-cache verification. A cold read must be at least COLD_WARM_MIN_RATIO x the warm read of
# the same file, or the "cold" number is page-cache bandwidth wearing an NVMe label. Below
# MIN_COLD_TEST_BYTES the comparison is process/syscall noise rather than I/O and cannot decide
# either way, so the verdict is None (undecidable) instead of a coin-flip True/False — that is
# what keeps the tiny-index stub path deterministic without special-casing the stub.
COLD_WARM_MIN_RATIO = 1.5
MIN_COLD_TEST_BYTES = 64 << 20          # 64 MiB


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pure, seed-deterministic pieces (unit-tested in artifacts/phase3/tests/test_e7_episode.py)
# ---------------------------------------------------------------------------

def build_schedule(rmap, steps, seed, *, row_step=5, row_bytes=8192, p_in_row=0.5):
    """The episode's whole injection plan, up front and PURE (no buffer is touched).

    Returns a list of `steps` entries `{"step": s, "injections": [...]}`, each injection a dict
    `{shape, stratum, element, anchor_byte, seed, kwargs}` ready for `apply_injections`. Building
    the plan separately from applying it is what makes the schedule unit-testable and lets both
    arms replay the IDENTICAL physical faults (the comparison is paired, not two samples).

    Seeds are SeedSequence-derived from (root, shape, stratum, step) via E6's `cell_seed`, so a
    step's draw does not depend on how many steps precede it and a re-run with the same --seed
    reproduces the episode exactly. The injector consumes a separate `lane_seed` sub-stream so
    the anchor draw and the in-shape draws never share an RNG.
    """
    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"steps must be >0, got {steps}")
    total = rmap.get("total_bytes")
    if not total:
        raise ValueError("region map has no total_bytes (built without file_size?) — the "
                         "byte-uniform single_cell draw needs the index footprint")
    sched = []
    for step in range(1, steps + 1):
        injections = []
        cs = cell_seed(seed, CELL_SHAPE, WHOLE_INDEX, step)
        anchor = int(np.random.default_rng(cs).integers(0, int(total)))
        injections.append({"shape": CELL_SHAPE, "stratum": WHOLE_INDEX, "element": None,
                           "anchor_byte": anchor, "seed": lane_seed(cs, LANE_INJECT),
                           "kwargs": {}})
        if step == int(row_step):
            rs = cell_seed(seed, ROW_SHAPE, ROW_STRATUM, step)
            a = sample_anchor(rmap, ROW_STRATUM, rs)
            injections.append({"shape": ROW_SHAPE, "stratum": ROW_STRATUM,
                               "element": a["element"], "anchor_byte": int(a["anchor_byte"]),
                               "seed": lane_seed(rs, LANE_INJECT),
                               "kwargs": {"row_bytes": int(row_bytes),
                                          "p_in_row": float(p_in_row)}})
        sched.append({"step": step, "injections": injections})
    return sched


def apply_injections(buf, entry):
    """Apply one schedule step's injections to `buf` IN PLACE. Returns (positions, records).

    Damage is cumulative by construction: nothing here restores, and the caller keeps the same
    working buffer across steps. Delegates to E6's `inject_shape`, which pins the anchor by
    handing the qp.faults injector a one-byte region while leaving every other draw (which bit,
    which row block, the column stripe) to the injector's own seeded stream.
    """
    positions, records = [], []
    for inj in entry["injections"]:
        pos, rec = inject_shape(buf, inj["shape"], inj["anchor_byte"], inj["seed"],
                                **inj.get("kwargs", {}))
        positions.extend(pos)
        records.append(dict(rec, stratum=inj["stratum"], element=inj["element"]))
    return positions, records


class ReloadPolicy:
    """Edge-triggered lazy-reload trigger: fires exactly ONCE per crossing of the threshold.

    `observe(failed_frac)` returns True on the step where the fraction first reaches the
    threshold and False while it stays there — a level-triggered version would re-fire a batch
    repair every step for damage the repair cannot clear (anything outside the manifest's ex_code
    field), turning one bounded repair into an unbounded loop and inflating `reload_bytes`
    without bound. Falling back below the threshold re-arms it (a later crossing fires again).

    Threshold default: RELOAD_THRESHOLD (cite: RecoveryGuard `reload_threshold`,
    phase3_e5_recovery.py:50). Comparison is `>=` per the brief.
    """

    def __init__(self, threshold=RELOAD_THRESHOLD):
        self.threshold = float(threshold)
        self.armed = True
        self.n_fired = 0

    def observe(self, failed_frac):
        if float(failed_frac) >= self.threshold:
            if not self.armed:
                return False
            self.armed = False
            self.n_fired += 1
            return True
        self.armed = True
        return False


def repair_bytes(n_failed, field_len):
    """Bytes a batch repair moves: one manifest field window per failed element.

    `field_len` comes from the CRC manifest header (96 B for ex_code on the SIFT b=7 index), never
    a literal — the same harness on a different padded_dim/ex_bits geometry would otherwise
    report a fabricated number.
    """
    n_failed, field_len = int(n_failed), int(field_len)
    if n_failed < 0:
        raise ValueError(f"n_failed must be >=0, got {n_failed}")
    if field_len <= 0:
        raise ValueError(f"field_len must be >0, got {field_len}")
    return n_failed * field_len


def failed_elements(work, manifest):
    """Element indices the CRC manifest flags in `work` — the detector, run in Python.

    The C++ side reports only counts (exp_dumpids.cpp L193-194), so the per-element ids come from
    re-running the SAME check the C++ loader runs, against the SAME manifest. See the module
    docstring.
    """
    return crc_manifest.verify_buffer(work, manifest)


def batch_repair(adapter, work, rmap, failed, field=MANIFEST_FIELD):
    """Patch each failed element's `field` window back from the PRISTINE on-disk index.

    One fresh `adapter.read_serialized_range` (open/seek/read) per element — deliberately never
    cached: the clean source's whole value is that it lives in persistent storage outside the
    DRAM fault process, and an in-memory copy would just be another corruptible replica.
    Returns (bytes_moved, wall_seconds) — both measured, both fed to Panel B's `ours` row.
    """
    t0 = time.perf_counter()
    moved = 0
    for e in failed:
        start, length = layout.element_field_range(rmap, field, int(e))
        work[start:start + length] = adapter.read_serialized_range(start, length)
        moved += int(length)
    return moved, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# e7_cost.json schema (the file E3 imports — validated before it is written)
# ---------------------------------------------------------------------------

_COST_NUMERIC = {
    "reload": ("seconds", "gbps", "downtime_upper_bound_full_batch_s"),
    "crash_restart": ("control_path_s", "downtime_s", "downtime_upper_bound_full_batch_s"),
    "eager": ("downtime_s", "downtime_upper_bound_full_batch_s"),
    "ours": ("downtime_s", "repair_io_bytes", "repair_wall_s"),
}
_COST_INT = {"reload": ("bytes",), "crash_restart": ("io_bytes",), "eager": ("io_bytes",)}
# `definition` and `method_detail` are REQUIRED, not decoration: doc["reload"]["seconds"] is
# meaningless to a downstream consumer without the sentence saying it is the I/O component and
# the one saying how the file was cooled.
_COST_STR = {"reload": ("method", "definition", "method_detail"),
             "crash_restart": ("control_path_method", "method"),
             "eager": ("method",), "ours": ("method",)}
# Tri-state: True verified cold / False known-warm / None undecidable (file too small).
_COST_TRISTATE = {"reload": ("cache_eviction_verified",)}
_COST_ENUM = {"reload.method": RELOAD_METHODS,
              "crash_restart.control_path_method": CONTROL_PATH_METHODS}


def validate_cost_json(doc):
    """Problems with an e7_cost.json document (empty list == valid). Never raises on content.

    Enforces the brief's schema: the four cost blocks + meta, numeric measurements that are
    actually numbers, method strings drawn from the documented enums, and one io_bytes value
    shared by every block that claims to move the whole index. SENTINEL is accepted anywhere a
    value is expected — "we have not measured this yet" is a legal, honest state; a plausible
    number in its place would not be.
    """
    problems = []
    if not isinstance(doc, dict):
        return ["cost json must be an object"]
    for block in ("reload", "crash_restart", "eager", "ours", "meta"):
        if block not in doc:
            problems.append(f"missing block {block!r}")
        elif not isinstance(doc[block], dict):
            problems.append(f"block {block!r} must be an object")
    if problems:
        return problems

    def check(block, keys, kinds, label):
        for key in keys:
            dotted = f"{block}.{key}"
            if key not in doc[block]:
                problems.append(f"missing {dotted}")
                continue
            val = doc[block][key]
            if val == SENTINEL:
                continue
            if not isinstance(val, kinds) or isinstance(val, bool):
                problems.append(f"{dotted} must be {label} (or {SENTINEL}), got {val!r}")

    for block, keys in _COST_NUMERIC.items():
        check(block, keys, (int, float), "numeric")
    for block, keys in _COST_INT.items():
        check(block, keys, (int,), "an integer byte count")
    for block, keys in _COST_STR.items():
        check(block, keys, (str,), "a string")
    for block, keys in _COST_TRISTATE.items():
        for key in keys:
            if key not in doc[block]:
                problems.append(f"missing {block}.{key}")
            elif doc[block][key] not in (True, False, None, SENTINEL):
                problems.append(f"{block}.{key} must be true/false/null (or {SENTINEL}), "
                                f"got {doc[block][key]!r}")

    # A cold-cache number that failed (or could not run) its own eviction check must not be
    # sitting in the file as a plain float — that is the exact failure this gate exists for.
    if doc["reload"].get("cache_eviction_verified") is not True:
        for dotted in ("reload.seconds", "reload.gbps",
                       "reload.downtime_upper_bound_full_batch_s", "eager.downtime_s",
                       "eager.downtime_upper_bound_full_batch_s"):
            block, key = dotted.split(".")
            if isinstance(doc[block].get(key), (int, float)) and not isinstance(
                    doc[block].get(key), bool):
                problems.append(
                    f"{dotted} is a number but reload.cache_eviction_verified is "
                    f"{doc['reload'].get('cache_eviction_verified')!r} — an unverified-cold read "
                    f"is page-cache bandwidth and must be written as {SENTINEL}")

    for dotted, allowed in _COST_ENUM.items():
        block, key = dotted.split(".")
        val = doc[block].get(key)
        if val not in (None, SENTINEL) and val not in allowed:
            problems.append(f"{dotted} must be one of {list(allowed)}, got {val!r}")

    sizes = {f"{b}.{k}": doc[b][k]
             for b, ks in (("reload", ("bytes",)), ("crash_restart", ("io_bytes",)),
                           ("eager", ("io_bytes",)))
             for k in ks if k in doc[b] and doc[b][k] != SENTINEL}
    if len(set(sizes.values())) > 1:
        problems.append(f"io_bytes disagree across blocks: {sizes} — every full-reload policy "
                        f"moves the same serialized index")
    return problems


def assert_cost_json(doc):
    """Hard gate before writing e7_cost.json (E3 imports it; a malformed file poisons the paper)."""
    problems = validate_cost_json(doc)
    if problems:
        raise ValueError("REPORT-AND-STOP: e7_cost.json fails its own schema: " +
                         "; ".join(problems))
    return doc


# ---------------------------------------------------------------------------
# Shared setup (clean index + manifest + baseline + provenance stamped up front)
# ---------------------------------------------------------------------------

def _sweep_sidecars(tmp):
    for suffix in (".ids.ivecs", ".rec.ivecs", ".clean.ivecs", ".cleanrec.ivecs"):
        for stale in (tmp + suffix, tmp + suffix + ".dist.fvecs", tmp + suffix + ".stats.json"):
            try:
                os.remove(stale)
            except OSError:
                pass


def setup_context(args, cfg, *, baseline=True):
    """Clean buffer, region map, CRC manifest, both-path clean baselines, provenance meta.

    The clean-baseline gates are E6's, kept identical on purpose: the two arms must be measured
    against the SAME operating point, and the recovery path must be a no-op on a clean index —
    otherwise every delta in Panel A is rebased onto garbage.
    """
    adapter = get_adapter(args.adapter)
    aname = adapter_name(adapter)
    out = os.path.abspath(args.out)
    assert_stub_sandbox(aname, out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    tag = f"_{args.out_tag}" if getattr(args, "out_tag", None) else ""

    clean_buf = adapter.serialize_index()
    rmap = adapter.region_map()
    gt = adapter.load_groundtruth()
    tmp = os.path.join(out, "raw", f"_e7{tag}.index")

    manifest_path = os.path.join(out, f"e7_clean_{MANIFEST_FIELD}{tag}.crcmf")
    crc_manifest.write_manifest(clean_buf, manifest_path, field=MANIFEST_FIELD, rmap=rmap)
    manifest = crc_manifest.read_manifest(manifest_path)
    pre = failed_elements(clean_buf, manifest)
    if pre:
        raise RuntimeError(f"pre-flight: clean buffer fails its own manifest at elements {pre[:5]}")

    clean_recall = clean_rec_recall = None
    if baseline:
        adapter.deserialize_index(clean_buf, tmp)
        t0 = time.perf_counter()
        clean_res = adapter.search_corrupted(tmp, k=config.K, ef=int(cfg["ef"]),
                                             out_path=tmp + ".clean.ivecs", timeout=args.timeout)
        eval_wall_s = time.perf_counter() - t0
        clean_recall = float(metrics.recall_at_k(clean_res["ids"], gt, config.K))
        clean_rec = adapter.query_with_recovery(tmp, RECOVERY_MODE, manifest_path, k=config.K,
                                                ef=int(cfg["ef"]),
                                                out_path=tmp + ".cleanrec.ivecs",
                                                timeout=args.timeout)
        crc0 = int((clean_rec.get("stats") or {}).get("load", {}).get("elements_crc_fail", 0))
        if crc0 != 0:
            raise RuntimeError(f"pre-flight: recovery path flags {crc0} elements on the CLEAN index")
        clean_rec_recall = float(metrics.recall_at_k(clean_rec["ids"], gt, config.K))
        if abs(clean_rec_recall - clean_recall) > e6.RETENTION_TOL:
            raise RuntimeError(
                f"REPORT-AND-STOP: recovery path returns recall@10={clean_rec_recall:.6f} on the "
                f"CLEAN index vs {clean_recall:.6f} for the plain search — the arms would be "
                f"measured against different baselines.")
        # The pointer rule must be a no-op on a CLEAN index; otherwise the ours arm would
        # "restore" clean fields every step and report fictitious oob counts and repair bytes.
        clean_oob, clean_oob_detail = e6.bounds_check_full(clean_buf.copy(), clean_buf, rmap)
        if clean_oob:
            raise RuntimeError(
                f"REPORT-AND-STOP: bounds_check_full flags {clean_oob} field(s) on the CLEAN "
                f"index ({clean_oob_detail}) — the pointer validity rule does not match this "
                f"index's conventions, so the ours arm's oob accounting would be fiction.")
        _sweep_sidecars(tmp)
    else:
        eval_wall_s = None

    meta = provenance.collect_provenance(
        adapter, aname, {"recall@10": clean_recall}, cfg, args, rmap,
        index_sha256=provenance.sha256_file(adapter.INDEX_PATH)
        if os.path.isfile(adapter.INDEX_PATH) else None,
        corrupted_regions=[WHOLE_INDEX, ROW_STRATUM], recovery=["ignore", RECOVERY_MODE],
        crc_manifest_sha256=provenance.sha256_file(manifest_path))
    meta["host_load"] = {"loadavg": list(os.getloadavg()), "cpu_count": os.cpu_count(),
                         "note": "recorded because a busy box inflates every wall-clock number"}

    log(f"[e7] adapter={aname} out={out} clean@10={clean_recall} "
        f"n={rmap['header']['cur_element_count']} ef={cfg['ef']} steps={cfg['steps']}")
    return {"adapter": adapter, "aname": aname, "out": out, "tag": tag, "rmap": rmap,
            "clean_buf": clean_buf, "gt": gt, "tmp": tmp, "manifest_path": manifest_path,
            "manifest": manifest, "clean_recall": clean_recall,
            "clean_rec_recall": clean_rec_recall, "clean_eval_wall_s": eval_wall_s,
            "meta": meta, "cfg": cfg, "timeout": args.timeout}


# ---------------------------------------------------------------------------
# Panel A — recall trajectory vs cumulative queries served
# ---------------------------------------------------------------------------

class PristineSource:
    """Slice-indexable stand-in for the clean buffer that PREADS from the on-disk index.

    `e6.bounds_check_full` restores a flagged pointer field with
    `work[s:s+L] = clean_buf[s:s+L]`. Handing it this object instead of an in-memory clean copy
    does two things at once: it keeps the house rule (the clean source lives in persistent
    storage — an in-memory copy is just another replica exposed to the same fault process), and
    it turns the pointer layer's repair from a free memcpy into MEASURED I/O, so `ours`' cost row
    accounts for every byte it moves.
    """

    def __init__(self, adapter):
        self._adapter = adapter
        self.bytes_read = 0
        self.n_reads = 0

    def __getitem__(self, sl):
        if not isinstance(sl, slice) or sl.start is None or sl.stop is None \
                or sl.step not in (None, 1):
            raise TypeError(f"PristineSource only serves contiguous [start:stop] slices, "
                            f"got {sl!r}")
        start, stop = int(sl.start), int(sl.stop)
        self.bytes_read += stop - start
        self.n_reads += 1
        return self._adapter.read_serialized_range(start, stop - start)


def pointer_repair(adapter, work, rmap):
    """Full-index pointer bounds check + pread-backed restore. Returns a measured dict.

    Detection is E6's vectorized `bounds_check_full` verbatim (links count > maxM0, any
    physically-present neighbour slot or cluster_id/label out of range, PTR_SENTINEL exempt);
    only the restore SOURCE differs — see PristineSource.

    WHY `ours` HAS THIS LAYER AT ALL (deviation from the brief's one-line arm definition): the
    brief describes the ours arm as EB-fallback plus the batch ex repair, which covers only the
    CRC manifest's `ex_code` field. The canonical 8 KB device_row spans ~30 whole elements, so it
    also shreds their `links`/`cluster_id`/`label`, and the first real validation run segfaulted
    the C++ search (SIGSEGV, CalledProcessError -11) at the row step in BOTH arms — an ex-only
    arm cannot even stay up, so it cannot have a recall trajectory to plot. The pointer
    bounds-check layer is part of the paper's stack anyway (RecoveryGuard's third layer,
    phase3_e5_recovery.py; E6's arm ON runs the same full-index version), so `ours` here is the
    stack as specified elsewhere, not a new mechanism invented for this panel. `--no-bounds-check`
    reproduces the literal brief arm (and its crash).
    """
    src = PristineSource(adapter)
    t0 = time.perf_counter()
    restored, detail = e6.bounds_check_full(work, src, rmap)
    return {"restored": int(restored), "detail": detail, "bytes": int(src.bytes_read),
            "reads": int(src.n_reads), "wall_s": round(time.perf_counter() - t0, 6)}


def _eval_step(ctx, work, arm):
    """One full eval of the working buffer for `arm`. Returns a dict of measured fields."""
    adapter = ctx["adapter"]
    adapter.deserialize_index(work, ctx["tmp"])
    out = {"recall": None, "cpp_recall": None, "crc_fail_cpp": None, "fallbacks": None,
           "crashed": False, "error": "", "eval_wall_s": None, "nq": None}
    t0 = time.perf_counter()
    try:
        if arm == "ignore":
            res = adapter.search_corrupted(ctx["tmp"], k=config.K, ef=int(ctx["cfg"]["ef"]),
                                           out_path=ctx["tmp"] + ".ids.ivecs",
                                           timeout=ctx["timeout"])
            stats = None
        else:
            res = adapter.query_with_recovery(ctx["tmp"], RECOVERY_MODE, ctx["manifest_path"],
                                              k=config.K, ef=int(ctx["cfg"]["ef"]),
                                              out_path=ctx["tmp"] + ".rec.ivecs",
                                              timeout=ctx["timeout"])
            stats = res.get("stats") or {}
    # A corrupted index legitimately kills the search (segfault / timeout / a header the adapter
    # cannot even parse). That is an OUTCOME of the episode, not a harness bug: record it and let
    # the trajectory continue. KeyboardInterrupt/SystemExit are BaseException and still propagate.
    except Exception as exc:                                        # noqa: BLE001
        out["crashed"], out["error"] = True, repr(exc)
    else:
        out["recall"] = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
        out["cpp_recall"] = res.get("cpp_recall")
        out["nq"] = int(np.asarray(res["ids"]).shape[0])
        if stats is not None:
            out["crc_fail_cpp"] = int(stats.get("load", {}).get("elements_crc_fail", 0))
            out["fallbacks"] = int(stats.get("totals", {}).get("fallbacks", 0))
    out["eval_wall_s"] = round(time.perf_counter() - t0, 6)
    _sweep_sidecars(ctx["tmp"])
    return out


def _step_record(ctx, arm, step, phase, work, ev, *, injections, reload_event, reload_bytes,
                 repair_wall_s, repair_elements, crc_fail_py, crc_py_wall_s, failed_frac,
                 ptr=None):
    """Assemble one raw JSONL row (a superset of the CSV's seven columns)."""
    n = int(ctx["rmap"]["header"]["cur_element_count"])
    nq = int(ctx["gt"].shape[0])
    crc_fail = ev["crc_fail_cpp"] if arm == "ours" and ev["crc_fail_cpp"] is not None \
        else crc_fail_py
    source = "cpp_stats" if (arm == "ours" and ev["crc_fail_cpp"] is not None) \
        else "python_manifest"
    recall = ev["recall"]
    return {
        # --- the CSV's seven columns -------------------------------------------------
        "step": int(step), "queries_served": int(step) * nq, "arm": arm,
        "recall": recall, "elements_crc_fail": crc_fail,
        "reload_event": int(reload_event), "reload_bytes": int(reload_bytes),
        # --- context beyond the CSV --------------------------------------------------
        "phase": phase,
        "delta_recall": None if recall is None else round(ctx["clean_recall"] - recall, 6),
        "clean_recall": ctx["clean_recall"],
        # Both detectors' counts, always, so their agreement is a measurement and not an
        # assumption. `elements_crc_fail` above is whichever one the arm actually acts on.
        "elements_crc_fail_cpp": ev["crc_fail_cpp"],
        "elements_crc_fail_py": int(crc_fail_py),
        "elements_crc_fail_source": source,
        "crc_verify_wall_s": crc_py_wall_s,
        "failed_frac": failed_frac,
        "repair_wall_s": repair_wall_s, "repair_elements": repair_elements,
        # The pointer bounds-check layer, measured separately from the ex-window repair: it runs
        # BEFORE the eval (it is what keeps the search from segfaulting), the ex repair after.
        "oob_restored": None if ptr is None else ptr["restored"],
        "oob_by_field": None if ptr is None else ptr["detail"],
        "oob_repair_bytes": None if ptr is None else ptr["bytes"],
        "oob_repair_reads": None if ptr is None else ptr["reads"],
        "oob_repair_wall_s": None if ptr is None else ptr["wall_s"],
        "fallbacks": ev["fallbacks"], "cpp_recall": ev["cpp_recall"],
        "crashed": ev["crashed"], "error": ev["error"], "eval_wall_s": ev["eval_wall_s"],
        "nq_returned": ev["nq"],
        "cumulative_bytes_changed": int(np.count_nonzero(work != ctx["clean_buf"])),
        "injections": injections,
        "n_elements": n, "adapter": ctx["aname"], "ef": int(ctx["cfg"]["ef"]),
    }


def run_arm(ctx, sched, arm, writer):
    """Replay the schedule for one arm. Returns (records, arm_summary)."""
    cfg = ctx["cfg"]
    n = int(ctx["rmap"]["header"]["cur_element_count"])
    field_len = int(ctx["manifest"]["header"]["field_len"])
    policy = ReloadPolicy(float(cfg["reload_threshold"]))
    work = ctx["clean_buf"].copy()
    records, events = [], []
    total_bits, prev_changed = 0, 0
    t_arm = time.perf_counter()

    for entry in sched:
        _pos, inj_records = apply_injections(work, entry)
        total_bits += sum(int(r["bits_flipped"]) for r in inj_records)

        # Pointer layer FIRST (before the search sees the buffer) — see pointer_repair's
        # docstring for why the ours arm has it and what happens without it.
        ptr = None
        if arm == "ours" and cfg["bounds_check"]:
            ptr = pointer_repair(ctx["adapter"], work, ctx["rmap"])
            if ptr["restored"]:
                log(f"[e7] step {entry['step']}: pointer bounds check restored "
                    f"{ptr['restored']} field(s) {ptr['detail']} ({ptr['bytes']} B) in "
                    f"{ptr['wall_s'] * 1e3:.1f} ms")

        ev = _eval_step(ctx, work, arm)

        # The manifest detector runs for BOTH arms every step: for `ours` it supplies the
        # per-element ids the C++ stats cannot (and cross-checks its count); for `ignore` it is
        # the only source for the elements_crc_fail column — recorded as "what a detector would
        # have flagged", explicitly NOT acted upon (source=python_manifest).
        t_crc = time.perf_counter()
        failed = failed_elements(work, ctx["manifest"])
        crc_py_wall_s = round(time.perf_counter() - t_crc, 6)
        crc_fail_py = len(failed)
        if arm == "ours" and ev["crc_fail_cpp"] is not None and ev["crc_fail_cpp"] != crc_fail_py:
            log(f"[e7] [warn] step {entry['step']}: C++ elements_crc_fail={ev['crc_fail_cpp']} "
                f"disagrees with the Python manifest verify ({crc_fail_py}) — the repair uses "
                f"the Python set (the C++ side emits no ids).")

        reload_event, reload_bytes, repair_wall_s, repair_elements = 0, 0, None, None
        failed_frac = crc_fail_py / n if n else 0.0
        if arm == "ours":
            observed = (ev["crc_fail_cpp"] if ev["crc_fail_cpp"] is not None else crc_fail_py) / n
            failed_frac = observed
            if policy.observe(observed):
                moved, repair_wall_s = batch_repair(ctx["adapter"], work, ctx["rmap"], failed)
                reload_event, reload_bytes = 1, moved
                repair_elements = len(failed)
                assert moved == repair_bytes(len(failed), field_len)
                events.append({"step": int(entry["step"]), "phase": "threshold",
                               "failed_frac": observed, "elements": len(failed),
                               "reload_bytes": moved, "repair_wall_s": round(repair_wall_s, 6)})
                log(f"[e7] step {entry['step']}: failed_frac={observed:.6f} >= "
                    f"{policy.threshold} -> batch repair of {len(failed)} elements "
                    f"({moved} B) in {repair_wall_s * 1e3:.1f} ms")

        rec = _step_record(ctx, arm, entry["step"], "episode", work, ev,
                           injections=inj_records, reload_event=reload_event,
                           reload_bytes=reload_bytes, repair_wall_s=repair_wall_s,
                           repair_elements=repair_elements, crc_fail_py=crc_fail_py,
                           crc_py_wall_s=crc_py_wall_s, failed_frac=failed_frac, ptr=ptr)
        # The ignore arm has no recovery, so its footprint can only grow. A shrink means
        # something wrote into the working buffer that had no business doing so, and every later
        # step of the episode would inherit it.
        if arm == "ignore" and rec["cumulative_bytes_changed"] < prev_changed:
            raise RuntimeError(
                f"state leak (arm=ignore, step {entry['step']}): bytes differing from clean fell "
                f"from {prev_changed} to {rec['cumulative_bytes_changed']} — nothing in this arm "
                f"repairs anything.")
        prev_changed = rec["cumulative_bytes_changed"]
        records.append(writer.write(rec))
        log(f"[e7] {arm:<6} step {entry['step']:>3}/{len(sched)}  "
            f"recall={'crash' if ev['recall'] is None else format(ev['recall'], '.5f')}  "
            f"crc_fail={rec['elements_crc_fail']}  reload={reload_event}  "
            f"{ev['eval_wall_s']:.1f}s")

    # --- final repair step: the MEASURED repair cost Panel B's `ours` row imports ---------
    if cfg["final_repair"]:
        step = len(sched) + 1
        reload_event, reload_bytes, repair_wall_s, repair_elements = 0, 0, None, None
        ptr = None
        if arm == "ours" and cfg["bounds_check"]:
            ptr = pointer_repair(ctx["adapter"], work, ctx["rmap"])
        if arm == "ours":
            failed = failed_elements(work, ctx["manifest"])
            if failed:
                moved, repair_wall_s = batch_repair(ctx["adapter"], work, ctx["rmap"], failed)
                reload_event, reload_bytes = 1, moved
                repair_elements = len(failed)
                events.append({"step": step, "phase": "final_repair",
                               "failed_frac": len(failed) / n if n else 0.0,
                               "elements": len(failed), "reload_bytes": moved,
                               "repair_wall_s": round(repair_wall_s, 6)})
                log(f"[e7] final repair: {len(failed)} elements ({moved} B) in "
                    f"{repair_wall_s * 1e3:.1f} ms")
        ev = _eval_step(ctx, work, arm)
        t_crc = time.perf_counter()
        crc_fail_py = len(failed_elements(work, ctx["manifest"]))
        crc_py_wall_s = round(time.perf_counter() - t_crc, 6)
        rec = _step_record(ctx, arm, step, "final_repair", work, ev, injections=[],
                           reload_event=reload_event, reload_bytes=reload_bytes,
                           repair_wall_s=repair_wall_s, repair_elements=repair_elements,
                           crc_fail_py=crc_fail_py, crc_py_wall_s=crc_py_wall_s,
                           failed_frac=crc_fail_py / n if n else 0.0, ptr=ptr)
        records.append(writer.write(rec))
        log(f"[e7] {arm:<6} step {step:>3} (final_repair) recall="
            f"{'crash' if ev['recall'] is None else format(ev['recall'], '.5f')}")

    recalls = [r["recall"] for r in records if r["recall"] is not None]
    deltas = [r["delta_recall"] for r in records if r["delta_recall"] is not None]
    n_crash = sum(1 for r in records if r["crashed"])
    # CRASH-BLIND AGGREGATES ARE A LIE. An arm that was DOWN for 4 of 6 steps has no meaningful
    # "worst recall" — averaging over only the steps it survived reports recall_min 0.98376 and
    # max_delta_recall 0.0, i.e. the exact inverse of what happened. So the headline keys go
    # None the moment anything crashed, and the surviving-step statistics stay available under
    # names that say out loud which steps they cover.
    summary = {
        "arm": arm, "rows": len(records),
        "wall_s": round(time.perf_counter() - t_arm, 3),
        "n_crash": n_crash,
        "first_crash_step": next((r["step"] for r in records if r["crashed"]), None),
        "n_rows_with_recall": len(recalls),
        "recall_first": records[0]["recall"] if records else None,
        "recall_min": (min(recalls) if recalls else None) if n_crash == 0 else None,
        "recall_last": records[-1]["recall"] if records else None,
        "max_delta_recall": (max(deltas, default=None)) if n_crash == 0 else None,
        "crash_blind_note": (
            None if n_crash == 0 else
            f"recall_min / max_delta_recall are null because {n_crash} of {len(records)} steps "
            f"CRASHED (first at step "
            f"{next((r['step'] for r in records if r['crashed']), None)}): the arm was down, not "
            f"degraded, and a min over the surviving steps would read as its best case. The "
            f"surviving-step figures are in *_over_served_steps below."),
        "recall_min_over_served_steps": min(recalls) if recalls else None,
        "max_delta_recall_over_served_steps": max(deltas, default=None),
        "final_elements_crc_fail": records[-1]["elements_crc_fail"] if records else None,
        "reload_events": events,
        "n_reload_events": len(events),
        "total_reload_bytes": sum(e["reload_bytes"] for e in events),
        "policy_n_fired": policy.n_fired,
        # Pointer layer totals over the whole episode (0/None on the ignore arm, which has no
        # recovery at all). These are real bytes the ours arm moved and belong in its cost.
        "bounds_check": bool(arm == "ours" and cfg["bounds_check"]),
        "total_oob_restored": sum(r["oob_restored"] or 0 for r in records),
        "total_oob_repair_bytes": sum(r["oob_repair_bytes"] or 0 for r in records),
        "total_oob_repair_wall_s": round(sum(r["oob_repair_wall_s"] or 0 for r in records), 6),
        "total_ex_repair_bytes": sum(r["reload_bytes"] for r in records),
        "cumulative_bits_flipped": int(total_bits),
        "final_bytes_changed": records[-1]["cumulative_bytes_changed"] if records else None,
    }
    return records, summary


def restamp_clean_baseline(ctx, args, recall):
    """Re-stamp provenance with a clean recall measured later in the run.

    Panels B and the microbench skip the up-front baseline evals (they would add two 15 s runs
    for nothing), but they DO measure a clean recall along the way. Without folding it back in,
    `platform_confirmed_real` would read False on a genuinely real run purely because the
    triangulation had no recall to compare against — a provenance stamp that lies in the safe
    direction is still a stamp that lies.
    """
    if recall is None:
        return ctx["meta"]
    fresh = provenance.collect_provenance(
        ctx["adapter"], ctx["aname"], {"recall@10": float(recall)}, ctx["cfg"], args,
        ctx["rmap"], crc_manifest_sha256=provenance.sha256_file(ctx["manifest_path"]))
    ctx["meta"].update({k: fresh[k] for k in
                        ("clean_baseline", "platform_confirmed_real", "confirmation_basis")})
    ctx["meta"]["clean_baseline"]["measured_by"] = (
        "measured during this run (not an up-front baseline eval)")
    return ctx["meta"]


def _threshold_note(rmap, cfg):
    """Whether THIS index/schedule pair can even reach the threshold — derived, not asserted.

    The answer flips between the 1M-element real index (it cannot: an 8 KB row covers ~30 of the
    100,000 elements the threshold asks for) and the 64-element stub (it trivially can, which is
    why the stub smoke genuinely exercises the batch-repair branch).
    """
    hdr = rmap["header"]
    n = int(hdr["cur_element_count"])
    spe = int(hdr["size_data_per_element"])
    need = float(cfg["reload_threshold"]) * n
    row_elems = int(cfg["row_bytes"]) // spe
    reachable = row_elems >= need
    return (f"threshold = {need:.0f} failed elements on a {n}-element index; the canonical "
            f"schedule damages ~{row_elems} elements (one {cfg['row_bytes']}-B row over a "
            f"{spe}-B element stride) plus a trickle from the per-step single cells, so on THIS "
            f"geometry the threshold "
            f"{'IS reachable' if reachable else 'is NOT reachable'} by this schedule"
            + ("." if reachable else
               " — the appended final_repair step is what supplies a measured repair cost."))


def run_panel_a(args, cfg):
    ctx = setup_context(args, cfg)
    out, tag = ctx["out"], ctx["tag"]
    arms = tuple(getattr(args, "arms", None) or ARMS)
    sched = build_schedule(ctx["rmap"], cfg["steps"], args.seed, row_step=cfg["row_step"],
                           row_bytes=cfg["row_bytes"], p_in_row=cfg["p_in_row"])

    raw_path = os.path.join(out, "raw", f"e7_panelA{tag}.records.jsonl")
    done_path = os.path.join(out, "raw", f"e7_panelA{tag}.done")
    records, arm_summaries = [], {}
    t0 = time.time()
    with RawWriter(raw_path, done_path=done_path) as w:
        for arm in arms:
            arm_records, arm_summaries[arm] = run_arm(ctx, sched, arm, w)
            records.extend(arm_records)

    expected = len(arms) * (len(sched) + (1 if cfg["final_repair"] else 0))
    if len(records) != expected:
        raise AssertionError(f"row count {len(records)} != expected {expected}")

    with open(os.path.join(out, f"e7_panelA{tag}.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        wr.writeheader()
        for r in records:
            wr.writerow({c: r.get(c) for c in CSV_COLUMNS})

    # The repair measurement Panel B imports. Prefer a threshold-triggered event (the deployed
    # path); fall back to the final-repair event, labelled as such so nothing is passed off as
    # something it is not.
    ours_events = arm_summaries.get("ours", {}).get("reload_events", [])
    threshold_ev = [e for e in ours_events if e["phase"] == "threshold"]
    chosen = (threshold_ev or ours_events or [None])[-1]
    # An ignore-only shard measured no repair at all: sentinels, not zeros — "that arm was not
    # run" and "the arm ran and moved nothing" are different claims and E3 must not conflate them.
    ours_arm = arm_summaries.get("ours") or {}
    ran_ours = "ours" in arm_summaries
    ours_repair = {
        # The brief's number: the ex-window batch repair of ONE reload event.
        "repair_io_bytes": chosen["reload_bytes"] if chosen else SENTINEL,
        "repair_wall_s": chosen["repair_wall_s"] if chosen else SENTINEL,
        "elements": chosen["elements"] if chosen else SENTINEL,
        "source_step": chosen["step"] if chosen else None,
        "source_phase": chosen["phase"] if chosen else None,
        "method": ("measured batch pread+patch loop (adapter.read_serialized_range per element, "
                   "fresh open/seek/read) over the elements the CRC manifest flagged"
                   if ran_ours else
                   f"NOT MEASURED: the ours arm was not run in this shard (arms={list(arms)})"),
        # ...and the honest episode total, which also includes every byte the pointer
        # bounds-check layer pread back. Quoting only the ex number would undercount `ours`.
        "episode_total_repair_io_bytes": ((int(ours_arm.get("total_ex_repair_bytes", 0))
                                           + int(ours_arm.get("total_oob_repair_bytes", 0)))
                                          if ran_ours else SENTINEL),
        "arm_was_run": ran_ours,
        "breakdown": {
            "ex_window_bytes": ours_arm.get("total_ex_repair_bytes"),
            "ex_reload_events": ours_arm.get("n_reload_events"),
            "pointer_field_bytes": ours_arm.get("total_oob_repair_bytes"),
            "pointer_fields_restored": ours_arm.get("total_oob_restored"),
            "pointer_repair_wall_s": ours_arm.get("total_oob_repair_wall_s"),
            "bounds_check_enabled": ours_arm.get("bounds_check"),
            "pointer_repair_wall_note": (
                "wall time of the whole bounds_check_full call = a full-index SCAN of all "
                "cur_element_count elements plus the restores. The scan dominates; the restore "
                "itself moves only pointer_field_bytes."),
            "detection_cost_note": (
                "the per-step Python CRC verify (crc_verify_wall_s in the raw rows) is HARNESS "
                "bookkeeping — it exists to recover the per-element failed ids the C++ stats "
                "JSON does not emit. In deployment that scan is the C++ loader's, whose cost is "
                "what --microbench measures. Do not quote the Python seconds as a system cost."),
        },
    }

    summary = {
        "experiment": "e7_episode_panelA",
        "arms": list(arms), "steps": int(cfg["steps"]), "cfg": cfg,
        "clean_recall@10": ctx["clean_recall"],
        "clean_recall_recovery_path@10": ctx["clean_rec_recall"],
        "queries_per_step": int(ctx["gt"].shape[0]),
        "queries_served_definition": (
            "cumulative OFFERED queries = step x queries_per_step, the episode's x-axis. On a "
            "step whose search crashed (recall empty, `crashed: true` in the raw row) nothing "
            "was actually answered — that is what the crash rows encode, and it is the ignore "
            "arm's real cost, not a gap in the data."),
        "schedule": {
            "row_step": int(cfg["row_step"]), "row_shape": ROW_SHAPE, "row_stratum": ROW_STRATUM,
            "row_bytes": int(cfg["row_bytes"]), "p_in_row": float(cfg["p_in_row"]),
            "single_cell_per_step": 1,
            "single_cell_draw": SINGLE_CELL_DRAW,
            "cumulative": True, "restored_between_steps": False,
            "seed": int(args.seed),
            "entries": [{"step": s["step"],
                         "injections": [{k: v for k, v in i.items() if k != "seed"}
                                        for i in s["injections"]]} for s in sched],
        },
        "reload_policy": {
            "threshold": float(cfg["reload_threshold"]),
            "comparison": ">=",
            "cite": "RecoveryGuard reload_threshold, phase3_e5_recovery.py:50",
            "edge_triggered": True,
            "denominator": "cur_element_count",
            "failed_element_ids_route": (
                "python: qp.rabitq.crc_manifest.read_manifest + verify_buffer on the working "
                "buffer. exp_dumpids' stats JSON reports only counts (load.elements_checked / "
                "load.elements_crc_fail), no per-element id list — see exp_dumpids.cpp L193-194 "
                "and recovery-changes.patch. The manifest IS the Option-B detector, so this is "
                "the deployed check, not a substitute for it."),
            "arithmetic_note": _threshold_note(ctx["rmap"], cfg),
        },
        "ours_repair": ours_repair,
        "arm_summary": arm_summaries,
        "sanity": {"rows": len(records), "expected_rows": expected,
                   "row_count_ok": len(records) == expected,
                   "stub_sandbox_ok": stub_sandbox_ok(ctx["aname"], out),
                   "clean_baseline_both_paths": {
                       "search_corrupted_recall@10": ctx["clean_recall"],
                       "query_with_recovery_recall@10": ctx["clean_rec_recall"]},
                   "wall_s": round(time.time() - t0, 1)},
        "adapter": ctx["aname"], "meta": ctx["meta"],
    }
    with open(os.path.join(out, f"e7_panelA_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[e7] wrote e7_panelA{tag}.csv ({len(records)} rows) + e7_panelA_summary{tag}.json "
        f"in {time.time() - t0:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# Panel B — measured wall-clock / IO cost of the three responses
# ---------------------------------------------------------------------------

def try_drop_caches():
    """Attempt `sudo -n echo 3 > /proc/sys/vm/drop_caches`; True iff it actually ran.

    Both the availability probe and the real per-repetition drop — there is no way to ask "could
    I?" without doing it, and dropping the cache is harmless. `-n` guarantees it never prompts
    (it fails immediately on a box without passwordless sudo, which is what selects the
    fresh-copy fallback).
    """
    try:
        res = subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                             capture_output=True, text=True, timeout=30)
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def cold_measurement_verdict(cold_median, warm_median, n_bytes):
    """Was the timed read really cold? -> (True | False | None, human-readable reason).

    Pure, so the honesty gate can be unit-tested without doing 5 GB of I/O.

      True   cold >= COLD_WARM_MIN_RATIO x warm on a file large enough for the comparison to be
             about I/O — the eviction worked and the bandwidth number means what it says.
      False  a big file whose "cold" read was no slower than its warm read. The cooling did not
             happen; the number is page-cache bandwidth. Panel B stops rather than shipping it.
      None   the file is smaller than MIN_COLD_TEST_BYTES (or a median is missing), so the test
             is undecidable — process startup dominates and the ratio is noise. Not an error,
             but not evidence either, so the derived seconds/gbps are sentinelled.
    """
    if not cold_median or not warm_median:
        return None, "no timing samples to compare"
    if int(n_bytes) < MIN_COLD_TEST_BYTES:
        return None, (f"index is {int(n_bytes)} B < {MIN_COLD_TEST_BYTES} B: the cold-vs-warm "
                      f"read comparison measures process overhead, not I/O, and cannot decide "
                      f"whether the page cache was evicted")
    ratio = cold_median / warm_median
    if ratio >= COLD_WARM_MIN_RATIO:
        return True, (f"cold/warm read ratio {ratio:.2f} >= {COLD_WARM_MIN_RATIO}: the page "
                      f"cache was really evicted")
    return False, (f"cold/warm read ratio {ratio:.2f} < {COLD_WARM_MIN_RATIO}: the 'cold' read "
                   f"was served from the page cache, so this is not an NVMe measurement")


def cool_file(path):
    """Evict `path` from the page cache without privileges: fsync then FADV_DONTNEED.

    This is what makes the sudo-less fallback honest. A plain `cp` does NOT produce a cold file —
    the copy leaves the destination's pages resident — so timing a read of a fresh copy without
    this step would measure page-cache bandwidth and call it NVMe bandwidth. FADV_DONTNEED drops
    a file's clean pages for any user; fsync first so nothing is still dirty (dirty pages cannot
    be dropped). The cold/warm pair reported alongside is the evidence that it worked.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _fresh_copy(src, dst):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 22)
        fo.flush()
        os.fsync(fo.fileno())
    cool_file(dst)
    return dst


def time_full_read(path):
    """Wall seconds for `cat path > /dev/null` — the pure sequential read of the whole file."""
    t0 = time.perf_counter()
    with open(os.devnull, "wb") as devnull:
        subprocess.run(["cat", path], stdout=devnull, check=True)
    return time.perf_counter() - t0


def _stats(values):
    a = np.asarray(values, dtype=float)
    q1, q3 = (float(np.percentile(a, 25)), float(np.percentile(a, 75))) if a.size else (None, None)
    return {"median": round(float(np.median(a)), 6) if a.size else None,
            "iqr": round(q3 - q1, 6) if a.size else None, "q1": round(q1, 6) if a.size else None,
            "q3": round(q3, 6) if a.size else None,
            "values": [round(float(v), 6) for v in a]}


def measure_reload_full(ctx, cfg, reps):
    """Cold-cache reload cost: pure read + one clean eval on the cold file. Median + IQR.

    Each repetition: cool -> time `cat` -> cool AGAIN (the cat just warmed it) -> time one clean
    eval -> time the same eval warm (the cache-residency control). Under `drop_caches` the
    pristine index itself is used; otherwise a fresh copy is made and FADV_DONTNEED'd, and the
    copy is deleted afterwards.
    """
    adapter = ctx["adapter"]
    src = adapter.INDEX_PATH
    if not os.path.isfile(src):
        adapter.deserialize_index(ctx["clean_buf"], src)      # stub: materialize its index file
    n_bytes = int(os.path.getsize(src))
    use_drop = (not cfg.get("no_sudo")) and try_drop_caches()
    method = "drop_caches" if use_drop else "fresh_copy"
    cold_read, cold_eval, warm_read, warm_eval, recalls = [], [], [], [], []
    copy_path = os.path.join(ctx["out"], "raw", f"_e7_cold{ctx['tag']}.index")

    for rep in range(int(reps)):
        if use_drop:
            path = src
            try_drop_caches()                                  # re-drop for this repetition
        else:
            path = _fresh_copy(src, copy_path)
        try:
            cold_read.append(time_full_read(path))
            warm_read.append(time_full_read(path))             # control: now certainly cached
            if use_drop:
                try_drop_caches()
            else:
                cool_file(path)
            t0 = time.perf_counter()
            res = adapter.search_corrupted(path, k=config.K, ef=int(cfg["ef"]),
                                           out_path=ctx["tmp"] + ".clean.ivecs",
                                           timeout=ctx["timeout"])
            cold_eval.append(time.perf_counter() - t0)
            recalls.append(float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K)))
            t0 = time.perf_counter()
            adapter.search_corrupted(path, k=config.K, ef=int(cfg["ef"]),
                                     out_path=ctx["tmp"] + ".clean.ivecs",
                                     timeout=ctx["timeout"])
            warm_eval.append(time.perf_counter() - t0)
            _sweep_sidecars(ctx["tmp"])
        finally:
            if not use_drop:
                try:
                    os.remove(copy_path)
                except OSError:
                    pass
        log(f"[e7] reload rep {rep + 1}/{reps}: cold_read={cold_read[-1]:.3f}s "
            f"warm_read={warm_read[-1]:.3f}s cold_eval={cold_eval[-1]:.3f}s "
            f"warm_eval={warm_eval[-1]:.3f}s")

    cr, wr, ce, we = (_stats(cold_read), _stats(warm_read), _stats(cold_eval), _stats(warm_eval))
    verified, verdict_reason = cold_measurement_verdict(cr["median"], wr["median"], n_bytes)
    # Every number that only means something if the file was ACTUALLY cold is gated on the
    # verdict. A warm read reported as `seconds` would put page-cache bandwidth in the paper.
    seconds = cr["median"] if verified else SENTINEL
    gbps = round(n_bytes / cr["median"] / 1e9, 6) if verified and cr["median"] else SENTINEL
    upper = ce["median"] if verified else SENTINEL
    return {
        "seconds": seconds,
        "bytes": n_bytes,
        "gbps": gbps,
        "downtime_upper_bound_full_batch_s": upper,
        "method": method,
        "method_detail": ("sudo -n `echo 3 > /proc/sys/vm/drop_caches` before every timed read"
                          if use_drop else
                          "fresh copy of the index + posix_fadvise(POSIX_FADV_DONTNEED) before "
                          "every timed read (a plain copy is NOT cold: copying populates the "
                          "page cache with the destination)"),
        "reps": int(reps),
        "definition": ("`seconds` is the median COLD sequential read of the whole serialized "
                       "index (`cat index > /dev/null`) — the I/O component of a reload, NOT the "
                       "time to a serving index. `downtime_upper_bound_full_batch_s` is the "
                       "other end of the bracket: one COMPLETE clean eval on the cold file "
                       "(index load + all 10K queries), so it overstates a reload by the whole "
                       "search. The true reload downtime lies between the two; the in-memory "
                       "reconstruct inside hnsw.load() is not separately instrumented."),
        "cold_read_s": cr, "warm_read_s": wr, "cold_eval_s": ce, "warm_eval_s": we,
        "cold_read_gbps": round(n_bytes / cr["median"] / 1e9, 6) if cr["median"] else None,
        "warm_read_gbps": round(n_bytes / wr["median"] / 1e9, 6) if wr["median"] else None,
        "cache_eviction_verified": verified,
        "cache_eviction_verdict": verdict_reason,
        "cache_eviction_evidence": (
            f"cold_read_s.median > {COLD_WARM_MIN_RATIO}x warm_read_s.median means the cooling "
            f"really evicted the file. False => the numbers are page-cache bandwidth and "
            f"`seconds`/`gbps`/`downtime_upper_bound_full_batch_s` are written as {SENTINEL} "
            f"(the raw medians stay here for diagnosis). null => the file is under "
            f"{MIN_COLD_TEST_BYTES} B, where the timing test cannot decide either way."),
        "clean_recall_on_cold_file": _stats(recalls)["median"],
    }


def read_control_path(path):
    """Task 5's HWPOISON helper output -> (seconds, method). Sentinels when it is absent.

    Accepted keys, in order: control_path_s | seconds | downtime_s, and control_path_method |
    method. Anything missing stays REQUIRES_MEASUREMENT — this harness never estimates the
    control path, because a plausible number here would silently become the paper's crash-restart
    downtime.
    """
    if not path:
        return SENTINEL, SENTINEL, "no --control-path-json given (Task 5 helper not run)"
    if not os.path.isfile(path):
        return SENTINEL, SENTINEL, f"--control-path-json {path} does not exist"
    with open(path) as fh:
        blob = json.load(fh)
    seconds = next((blob[k] for k in ("control_path_s", "seconds", "downtime_s") if k in blob),
                   SENTINEL)
    method = next((blob[k] for k in ("control_path_method", "method") if k in blob), SENTINEL)
    return seconds, method, f"read from {path}"


def run_panel_b(args, cfg):
    ctx = setup_context(args, cfg, baseline=False)
    out, tag = ctx["out"], ctx["tag"]
    reload_full = measure_reload_full(ctx, dict(cfg, no_sudo=args.no_sudo), args.reps or cfg["reps"])
    n_bytes = reload_full["bytes"]
    restamp_clean_baseline(ctx, args, reload_full["clean_recall_on_cold_file"])

    # An unverified-cold measurement is page-cache bandwidth. Stopping is the default because a
    # sentinel-laden cost table that nobody notices is nearly as bad as a wrong number.
    if reload_full["cache_eviction_verified"] is False and not args.allow_unverified_cold:
        raise RuntimeError(
            f"REPORT-AND-STOP: the cold-cache measurement failed its own check — "
            f"{reload_full['cache_eviction_verdict']}. Re-run on a box where the eviction works "
            f"(passwordless sudo enables drop_caches), or pass --allow-unverified-cold to write "
            f"the run with seconds/gbps as {SENTINEL} and the raw medians kept for diagnosis.")

    cold_ok = reload_full["cache_eviction_verified"] is True
    seconds = reload_full["seconds"]
    upper = reload_full["downtime_upper_bound_full_batch_s"]
    cp_s, cp_method, cp_note = read_control_path(args.control_path_json)
    downtime_crash = (round(cp_s + seconds, 6)
                      if isinstance(cp_s, (int, float)) and isinstance(seconds, (int, float))
                      else SENTINEL)

    panel_a_path = args.panel_a_json or os.path.join(out, f"e7_panelA_summary{tag}.json")
    ours_repair, ours_note = None, f"panel A summary not found at {panel_a_path}"
    if os.path.isfile(panel_a_path):
        with open(panel_a_path) as fh:
            ours_repair = json.load(fh).get("ours_repair")
        ours_note = f"imported from {panel_a_path}"

    doc = {
        # SELF-DESCRIBING ON PURPOSE: a consumer that reads only doc["reload"] must be able to
        # tell what `seconds` is (the I/O component, not a time-to-serving), how the file was
        # cooled, whether that cooling was verified, and what the other end of the bracket is —
        # without having to know that meta.reload_full exists.
        "reload": {k: reload_full[k] for k in
                   ("seconds", "bytes", "gbps", "method", "downtime_upper_bound_full_batch_s",
                    "definition", "method_detail", "cache_eviction_verified")},
        "crash_restart": {
            "control_path_s": cp_s,
            "control_path_method": cp_method,
            "downtime_s": downtime_crash,
            "io_bytes": n_bytes,
            "method": "HWPOISON control path (Task 5 helper) + reload_full.seconds",
            "note": cp_note,
            # Named for what it actually is: control path + one COMPLETE 10K-query eval on the
            # cold file, so it overstates a restart by the whole search batch.
            "downtime_upper_bound_full_batch_s": (
                round(cp_s + upper, 6) if isinstance(cp_s, (int, float))
                and isinstance(upper, (int, float)) else SENTINEL),
        },
        "eager": {
            "downtime_s": seconds,
            "io_bytes": n_bytes,
            "method": "full reload on the first CRC failure = reload_full.seconds (measured)",
            "downtime_upper_bound_full_batch_s": upper,
        },
        "ours": {
            "downtime_s": 0,
            "repair_io_bytes": (ours_repair or {}).get("repair_io_bytes", SENTINEL),
            "repair_wall_s": (ours_repair or {}).get("repair_wall_s", SENTINEL),
            "method": ((ours_repair or {}).get("method")
                       or "measured batch pread+patch loop (Panel A)"),
            "note": ("downtime_s is 0 by construction — the service keeps answering, degraded; "
                     "Panel A's recall trajectory IS the cost. " + ours_note),
            "source_step": (ours_repair or {}).get("source_step"),
            "source_phase": (ours_repair or {}).get("source_phase"),
            "elements_repaired": (ours_repair or {}).get("elements", SENTINEL),
            # repair_io_bytes above is the brief's number (one ex-window batch repair). This is
            # every byte the arm moved across the whole episode, ex windows plus the pointer
            # bounds-check layer's restores — the honest total to compare against a full reload.
            "episode_total_repair_io_bytes": (ours_repair or {}).get(
                "episode_total_repair_io_bytes", SENTINEL),
            "breakdown": (ours_repair or {}).get("breakdown", SENTINEL),
        },
        "meta": dict(ctx["meta"], reload_full=reload_full,
                     panel_b_note=("every value here is measured on this host; sentinels mark "
                                   "what is not measured yet and are never filled with an "
                                   "estimate"),
                     sentinel=SENTINEL,
                     index_path=ctx["adapter"].INDEX_PATH,
                     adapter=ctx["aname"]),
    }
    assert_cost_json(doc)
    path = os.path.join(out, f"e7_cost{tag}.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    log(f"[e7] wrote {path} (reload {seconds}s, {reload_full['gbps']} GB/s, "
        f"method={reload_full['method']}, cold_verified={reload_full['cache_eviction_verified']})")
    if not cold_ok:
        log(f"[e7] [warn] cold-cache check did not pass: "
            f"{reload_full['cache_eviction_verdict']} — seconds/gbps written as {SENTINEL}.")
    return doc


# ---------------------------------------------------------------------------
# --microbench — per-vector CRC vs per-vector distance
# ---------------------------------------------------------------------------

def run_microbench(args, cfg):
    """CRC ns/vector vs distance ns/consult, both from the real binary on real data.

    Both numerators come from the instrumentation patch (`load.crc_scan_ns` and
    `search_wall_ns` in the stats JSON). If the built binary does not emit them they come out as
    sentinels that NAME the missing key rather than as a derived guess. Alongside: elements_checked,
    totals.consults, the subprocess wall, and a Python zlib reference explicitly fenced off from
    the reported number.

    The ratio is ONE-SIDED and the direction is easy to state backwards — see `distance.caveat`
    and `ratio_interpretation`, which spell out that it is a lower bound on the ratio and hence
    an upper bound on how cheap CRC is.
    """
    ctx = setup_context(args, cfg, baseline=False)
    out, tag = ctx["out"], ctx["tag"]
    adapter = ctx["adapter"]
    adapter.deserialize_index(ctx["clean_buf"], ctx["tmp"])

    t0 = time.perf_counter()
    res = adapter.query_with_recovery(ctx["tmp"], RECOVERY_MODE, ctx["manifest_path"],
                                      k=config.K, ef=int(cfg["ef"]),
                                      out_path=ctx["tmp"] + ".rec.ivecs", timeout=ctx["timeout"])
    wall_s = time.perf_counter() - t0
    stats = res.get("stats") or {}
    load = stats.get("load", {})
    totals = stats.get("totals", {})
    recall = float(metrics.recall_at_k(res["ids"], ctx["gt"], config.K))
    _sweep_sidecars(ctx["tmp"])
    restamp_clean_baseline(ctx, args, recall)

    crc_ns = load.get("crc_scan_ns")
    checked = load.get("elements_checked")
    search_ns = stats.get("search_wall_ns")
    consults = totals.get("consults")

    ns_per_vector = (crc_ns / checked) if (crc_ns and checked) else SENTINEL
    ns_per_consult = (search_ns / consults) if (search_ns and consults) else SENTINEL

    # Measured here and clearly fenced off: a Python zlib.crc32 pass over the same 96-B windows.
    # It is NOT the reported CRC cost (different language, different compiler flags, per-slice
    # interpreter overhead) — it exists only as an order-of-magnitude sanity anchor.
    t0 = time.perf_counter()
    failed_elements(ctx["clean_buf"], ctx["manifest"])
    py_wall = time.perf_counter() - t0
    n = int(ctx["manifest"]["header"]["n_entries"])

    ratio = (round(ns_per_vector / ns_per_consult, 6)
             if isinstance(ns_per_vector, float) and isinstance(ns_per_consult, float)
             else SENTINEL)
    doc = {
        "experiment": "e7_microbench",
        "crc": {
            "crc_scan_ns": crc_ns if crc_ns is not None else SENTINEL,
            "elements_checked": checked if checked is not None else SENTINEL,
            "ns_per_vector": ns_per_vector,
            "missing_key": None if crc_ns is not None else "stats.load.crc_scan_ns",
            "note": ("measured by the patched load_crc_manifest (stats.load.crc_scan_ns): the "
                     "eager CRC-32 scan over every element's 96 B ex window, C++ table "
                     "implementation under the binary's own compiler flags."
                     if crc_ns is not None else
                     "stats.load.crc_scan_ns is absent from this binary's stats JSON, so the "
                     "C++ CRC time is NOT measured; rebuild with the instrumentation patch."),
        },
        "distance": {
            "search_wall_ns": search_ns if search_ns is not None else SENTINEL,
            "consults": consults if consults is not None else SENTINEL,
            "ns_per_consult": ns_per_consult,
            "missing_key": None if search_ns is not None else "stats.search_wall_ns",
            # DIRECTION MATTERS AND IS EASY TO STATE BACKWARDS. ns_per_consult is search wall /
            # consults, so it carries the graph traversal on top of the distance arithmetic and
            # OVERSTATES a pure distance computation. Dividing by an overstated denominator
            # understates the ratio => the reported ratio is a LOWER BOUND ON THE RATIO, i.e.
            # CRC costs AT LEAST this fraction of a distance computation. It is therefore an
            # UPPER bound on how cheap CRC is: "at most 1/ratio x cheaper", never "at least".
            "caveat": ("ns_per_consult = search_wall_ns / consults includes graph traversal, so "
                       "it overstates pure distance arithmetic. Dividing by it therefore "
                       "UNDERSTATES the ratio: ratio_crc_per_distance is a LOWER BOUND on the "
                       "true CRC-to-pure-distance ratio. Read it as 'CRC costs at least this "
                       "fraction of a distance computation' — i.e. at most 1/ratio times "
                       "cheaper, not at least."),
        },
        "ratio_crc_per_distance": ratio,
        "ratio_interpretation": (
            SENTINEL if ratio == SENTINEL else
            f"CRC costs at least {ratio * 100:.2f}% of one distance consult "
            f"({ns_per_vector:.1f} ns/vector vs {ns_per_consult:.1f} ns/consult), i.e. AT MOST "
            f"{1 / ratio:.0f}x cheaper. The bound is one-sided because the consult denominator "
            f"includes traversal — see distance.caveat."),
        "measured_today": {
            "subprocess_wall_s": round(wall_s, 6),
            "subprocess_wall_note": ("whole exp_dumpids run: index load + CRC scan + the full "
                                     "query batch + IO. An upper bound on each part, not a "
                                     "substitute for either numerator."),
            "recall@10": recall,
            "ef": int(cfg["ef"]),
        },
        "python_zlib_reference": {
            "wall_s": round(py_wall, 6), "elements": n,
            "ns_per_vector": round(py_wall / n * 1e9, 3) if n else None,
            "warning": ("NOT the reported CRC cost and NOT an input to the ratio: this is "
                        "Python zlib.crc32 with per-slice interpreter overhead, not the C++ "
                        "table implementation under the binary's compiler flags."),
        },
        "adapter": ctx["aname"], "meta": ctx["meta"],
    }
    path = os.path.join(out, f"e7_microbench{tag}.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    log(f"[e7] wrote {path} (ratio={doc['ratio_crc_per_distance']})")
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="a", choices=["a", "b"],
                    help="a = recall trajectory (default); b = measured cost table")
    ap.add_argument("--microbench", action="store_true",
                    help="run the CRC-vs-distance microbench instead of a panel")
    ap.add_argument("--adapter", default=None, choices=["stub", "real", "auto"],
                    help="stub (dev) / real (x86) / auto (default: real if binaries else stub)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE['steps']} steps + small ef; outputs to artifacts_smoke/")
    ap.add_argument("--steps", type=int, default=None, help="episode length S (default 40)")
    ap.add_argument("--arms", type=csv_list("arm", ARMS), default=None,
                    help=f"comma-separated subset of {list(ARMS)} — shards the two arms across "
                         f"processes (the C++ search is single-threaded). Pair with --out-tag.")
    ap.add_argument("--seed", type=int, default=config.SEED, help="root seed for the schedule")
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--row-step", dest="row_step", type=int, default=None,
                    help="step at which the canonical device_row lands (default 5)")
    ap.add_argument("--row-bytes", dest="row_bytes", type=int, default=None,
                    help="DRAM row modeling unit in bytes (default 8192 — a knob, not a fact)")
    ap.add_argument("--p-in-row", dest="p_in_row", type=float, default=None,
                    help="per-bit flip probability inside the failed row (default 0.5)")
    ap.add_argument("--reload-threshold", dest="reload_threshold", type=float, default=None,
                    help=f"failed_frac at which the batch repair fires (default "
                         f"{RELOAD_THRESHOLD}; cite phase3_e5_recovery.py:50)")
    ap.add_argument("--no-final-repair", dest="final_repair", action="store_false", default=None,
                    help="skip the appended measured-repair step of the ours arm")
    ap.add_argument("--no-bounds-check", dest="bounds_check", action="store_false", default=None,
                    help="drop the pointer bounds-check layer from the ours arm, reproducing the "
                         "brief's literal ex-only arm (which segfaults at the device_row step)")
    ap.add_argument("--reps", type=int, default=None, help="panel B repetitions (default 5)")
    ap.add_argument("--no-sudo", action="store_true",
                    help="panel B: never attempt drop_caches; force the fresh-copy+fadvise path")
    ap.add_argument("--allow-unverified-cold", action="store_true",
                    help="panel B: do not stop when the cold-cache check fails; write "
                         f"seconds/gbps as {SENTINEL} and keep the raw medians for diagnosis")
    ap.add_argument("--control-path-json", dest="control_path_json", default=None,
                    help="Task 5 HWPOISON helper output (keys control_path_s / "
                         "control_path_method); absent -> REQUIRES_MEASUREMENT sentinel")
    ap.add_argument("--panel-a-json", dest="panel_a_json", default=None,
                    help="panel A summary to import the measured repair cost from "
                         "(default: e7_panelA_summary.json in --out)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-search seconds; a hang under corruption records as crash")
    ap.add_argument("--out-tag", dest="out_tag", default=None,
                    help="filename suffix so parallel/variant runs never clobber each other")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(config.ROOT, "artifacts_smoke" if args.smoke else "artifacts",
                                "phase3", "e7")

    cfg = dict(SMOKE if args.smoke else FULL)
    for knob in ("steps", "ef", "row_step", "row_bytes", "reps"):
        if getattr(args, knob, None) is not None:
            cfg[knob] = int(getattr(args, knob))
    for knob in ("p_in_row", "reload_threshold"):
        if getattr(args, knob, None) is not None:
            cfg[knob] = float(getattr(args, knob))
    for flag in ("final_repair", "bounds_check"):
        if getattr(args, flag) is not None:
            cfg[flag] = bool(getattr(args, flag))
    if cfg["row_step"] > cfg["steps"]:
        log(f"[e7] [warn] --row-step {cfg['row_step']} > --steps {cfg['steps']}: the canonical "
            f"device_row never lands in this episode.")

    if args.microbench:
        run_microbench(args, cfg)
        log("\nE7 MICROBENCH OK")
    elif args.panel == "a":
        run_panel_a(args, cfg)
        log("\nE7 PANEL A OK")
    else:
        run_panel_b(args, cfg)
        log("\nE7 PANEL B OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
