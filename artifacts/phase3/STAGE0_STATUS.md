# Phase 3 — Stage 0 status

Measurement infrastructure for RaBitQ corruption study. Core rule honored: all metric /
collapse / fault-injection logic is **imported from `qp/`** (single source of truth); nothing
is re-implemented in Samuel's repo.

## Components (all built; verifications passing)

| # | Component | Where | Verified by |
|---|-----------|-------|-------------|
| 1 | Fault models (`uniform_p`, `spatial_cluster`/`cross_row`, `temporal_burst` + `CumulativeCorruption`) | `qp/faults.py` (faiss-free, on `qp/bits.py` primitives) | `tests/test_faults.py` (binomial CI, chi-square uniformity, per-window clustering, determinism, restore roundtrip, quiet→step curve) |
| 2 | Recall measurement (recall@{1,10,100}, tolerant recall, `is_silent_collapse`) | `phase3_recall.py` → delegates to `qp.metrics` | `tests/test_recall_predicate.py` (collapse truth table, gt/direction sanity, classify passthrough) |
| 3 | Cost harness (mem / scrub / total + reconciliation + monotonicity) | `phase3_cost.py` | `tests/test_cost.py` |
| 0 | RaBitQ adapter + byte-region map | `qp/rabitq/layout.py`, `qp/rabitq/adapter.py` | `tests/test_layout.py` (source-derived offsets) |
| 4 | Parity + golden | `artifacts/phase3/make_golden.py`, `tests/test_golden.py`, `tests/test_parity.py` | golden regression + metric-level parity (RaBitQ parity skipped, gated) |

Run all: `python -m pytest artifacts/phase3/tests/ -q` → **33 passed, 1 skipped**.

## Byte layout — RESOLVED from source (not guessed)

Read directly from RaBitQ-Library `HierarchicalNSW::save()`, `data_layout.hpp`, and
`rotator.hpp` at the pinned commit `7e39df2`. Serialized file:
`[header 156B][centroids][level0 per-vector blocks][upper links (variable)][rotation @ EOF]`.
Per element: `links(maxM0·4+4) | cluster_id(4) | label(4) | bin_code(pdim/8) | bin_factors(12)
| ex_code(pdim·ex_bits/8) | ex_factors(8)`. SIFT b=7: bin_data 28B, ex_data 104B,
size_links_level0 132B.

**Rotation = `FhtKacRotator.flip_`, `4·padded_dim/8 = padded_dim/2` bytes (64B for SIFT).**
It is a *structured* FHT+Kac rotation (a sign-flip bit-vector), not a dense matrix — a tiny,
globally-shared decode structure written at end-of-file. This is the RaBitQ analog of SQ8's
`sq_scale` and the predicted single-point-catastrophe target. `layout.serialized_region_map`
locates it at the file tail.

## Blocked items (report-and-stop; nothing faked)

- **D — RaBitQ C++ build is x86-64 only.** The library's core (`utils/space.hpp`,
  `quantization/rabitq_impl.hpp`, `index/ivf|hnsw`) unconditionally includes `<immintrin.h>`
  and uses AVX2/AVX512 intrinsics with **no NEON path**, so it does not compile on this Apple
  Silicon (arm64) host. `build_rabitq.sh` is complete and correct: it clones + pins upstream
  to `7e39df2` (patch applies cleanly), installs cmake/libomp, fixes AppleClang OpenMP, then
  cleanly **reports-and-stops at the arch gate**. Run it **unchanged on an x86-64 Linux host**
  to produce the binaries and the b=7 index. The live clean baseline (≈0.983) and any live
  corruption run are blocked until then. (Samuel's own results were produced on x86 Linux.)

- **Parity id-dump — RESOLVED (implemented; runs on the x86-64 build).** Added
  `rabitq_instrumentation/exp_dumpids.cpp` (mirrors `hnsw_rabitq_querying.cpp`; for one `ef`
  writes top-k ids as ivecs and prints `RECALL\t<r>`). `build_rabitq.sh` builds it,
  `adapter.query_ids` runs it and returns `(ids, cpp_recall)`, and
  `tests/test_parity.py::test_rabitq_parity_clean_index` now asserts
  `qp.metrics.recall_at_k(ids, gt, k) == cpp_recall ≈ 0.983` on identical ids (no skip on x86;
  still skips on arm64 where no binaries exist). The metric-level parity passes everywhere.

## Workstation handoff

`run_stage0_x86.sh` + `X86_WORKSTATION.md`: one command on an x86-64 Linux host clones the
sibling repo, fetches SIFT, builds the env + RaBitQ binaries (incl `exp_dumpids`) + b=7 index,
and runs the full suite (parity included) + the live ≈0.983 baseline gate.

## Known limitation — deferred to Stage 1

- **Per-vector cost aggregation.** `layout.serialized_region_map` exposes per-element
  sub-regions for **element 0 only** (`elem0.bin_factors` is one 12-byte slice). Feeding that
  map to `phase3_cost.mem_cost` and protecting a per-vector structure prices only ONE element,
  so the protection-memory estimate is short by a factor of `cur_element_count` (~10⁶ for
  SIFT1M). Stage 1 must add an aggregate per-structure region (size × `cur_element_count`) or
  pass an element count into `mem_cost` before any per-vector protection budget is trusted.
  The cost formulas themselves are correct for the region map they are given.

## Stage 1 update (E1 + E3a/E3b built; stub-green, awaiting x86 run)

Built on top of Stage 0 (see `plan/stage1_plan.md` and `E1_RUNBOOK.md`):
- **Adapter injection** — `qp/rabitq/registry.py::get_adapter` + `qp/rabitq/stub_adapter.py`
  (deterministic synthetic-index adapter). `--adapter {stub,real,auto}` switches with no code
  change; `real` without binaries report-and-stops (never downgrades to stub).
- **Bit-class** — `qp/rabitq/bitclass.py`, derived from source: rotation = 4 FhtKac sign-flip
  stages (`rot_stage0..3`); bin/ex factors = float fields × fp32 tag (f_error marked); bin_code =
  flat sign; pointers = int32 high/low lane; **ex_code = flat (report-and-stop**: SIMD
  bit-plane packing has no contiguous high/low split).
- **Runners** — `phase3_e1_vuln.py` (single-bit map; rotation exhaustive, per-vector sampled with
  bootstrap CI; reuses `phase1_sensitivity.aggregate` + the unified `is_silent_collapse`; emits
  `vuln_map` + criticality order + three-tier scrub allocation priced via `phase3_cost`),
  `phase3_e3a_additivity.py`, `phase3_e3b_spatial.py`.
- **Resolves the two Stage-0 deferrals below**: per-vector addressing via
  `layout.element_field_range` and per-vector cost via `layout.aggregate_region_map`
  (×`cur_element_count`); nan-inf via the distances-optional `adapter.search_corrupted`
  (runbook §2 extends `exp_dumpids` to dump distances on the workstation).
- Tests: `tests/test_e1_rabitq_stage1.py` + `tests/test_e1_runners.py` (full suite now **64
  passed, 1 skipped**). Scientific numbers pending the x86 run (E1_RUNBOOK).

## To finish Stage 0 acceptance on the right machine
1. `bash build_rabitq.sh` on an x86-64 Linux host (cmake + libomp present) → binaries + b=7 index.
2. Confirm `adapter.clean_baseline_recall()` ≈ 0.983 (anchors the whole study).
3. Add the C++ ids-dump, implement `adapter.query_ids`, unskip `test_rabitq_parity_clean_index`.
4. Regenerate the golden if any number legitimately changes: `python artifacts/phase3/make_golden.py`.
