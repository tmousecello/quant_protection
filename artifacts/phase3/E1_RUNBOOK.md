# Phase 3 — Stage 1 runbook (x86 workstation)

The RaBitQ search is AVX2/AVX512 C++ and runs **only on an x86-64 Linux host**. The E1/E3a/E3b
runners were developed and unit-tested on the arm64 dev machine against a deterministic **stub**
adapter; the real scientific numbers (the RaBitQ `vuln_map`, criticality order, E3a/E3b results)
are produced here. The code needs **no edits** on the workstation — only `--adapter real` (or the
auto-default, which picks `real` once the binaries exist).

## 0. Sync + environment
1. `rsync` / deploy-key the `quant_protection/` tree to the workstation (sibling RaBitQ repo too).
2. `bash setup.sh && source .venv/bin/activate` (Python 3.13, faiss-cpu, scipy, numpy).
3. `bash build_rabitq.sh` — clones + pins RaBitQ-Library to `7e39df2`, builds the binaries and the
   b=7 SIFT index. This produces `hnsw_rabitq_indexing/querying`, `exp_faultinject`, `exp_fieldflip`,
   `exp_dumpids`. Confirm: `python -c "from qp.rabitq import adapter; print(adapter.binaries_built())"`
   → `True`, and `adapter.clean_baseline_recall()` ≈ **0.983** (anchors the whole study).

## 1. First shot — is the region map aligned, and how dangerous is the 64-byte rotation?
Before the full sweep, flip rotation bits and look at recall. This is both (a) a physical
confirmation that the region map is aligned to the real file and (b) E1's first datapoint.

```bash
python phase3_e1_vuln.py --adapter real --smoke --clean-tol 0.05   # smoke ef=64 -> clean ~0.95
```
The gate (in `run_stage1_x86.sh` step 8) is an **alignment** check: it passes iff rotation flips
demonstrably move recall — `max ΔRecall@10` over the rotation rows ≥ `ROT_ALIGN_DRECALL_MIN`
(default 0.05). It does **not** gate on a collapse fraction. **Report-and-stop (exit 7) only when
rotation is inert** (signal ≈ 0) → the region map is likely offset against this build; inspect
`artifacts_smoke/phase3/e1/vuln_map.json` (rotation rows).

Observed on the x86 workstation: rotation is **aligned but heavy-tailed** — the map is correct
(header/links/cluster_id flips crash, centroids move recall, all 512 rotation flips register as
`silent_wrong`), yet only ~0.2 % of rotation bits cross the 50 %-retention collapse bar while the
worst single bit drops recall@10 by ~0.5. So rotation is **not** a `sq_scale`-style uniform
single-point catastrophe; its danger lives in the tail (`max`/`p99 ΔRecall@10`), which the runner
reports and `derive_criticality` now uses to upgrade rotation (a GLOBAL structure with ≥1
single-point-collapse bit) to `frequent_scrub`. Note: first-shot magnitudes are at smoke ef=64; the
full §3 run at ef=2000 gives the operating-point numbers.

## 2. (Recommended) enable nan-inf detection — extend exp_dumpids
The failure taxonomy separates silent / **nan-inf** / crash. `exp_dumpids` currently emits ids +
`RECALL` only, so the runner detects crash + silent but reports `n_nan_inf` as **not-detected**
(`nan_inf_supported=false`). To get the complete taxonomy via the *identical* `qp.metrics`
classifier FAISS uses, add a per-query distances dump (the search already has the distances):

In `rabitq_instrumentation/exp_dumpids.cpp`, alongside the ids `ivecs`, write a companion
`<out>.dist.fvecs` of the top-k distances (`res[i][j].first`), same fixed-k row width as the ids,
padded with a large sentinel. Rebuild via `build_rabitq.sh`. The adapter
(`adapter.search_corrupted`) already reads `<out_path>.dist.fvecs` when present and the runner then
runs `metrics.classify_failure(distances=…)` → nan-inf is detected. No Python change needed.

This step is optional: skip it and E1 still produces the silent-collapse map (the F1 headline);
nan-inf simply stays "not-detected" and is honestly flagged, never reported as zero.

## 3. Full runs (verbose log + per-flip jsonl preserved for offline debugging)
```bash
python phase3_e1_vuln.py        --adapter real 2>&1 | tee artifacts/phase3/e1/run.log
python phase3_e3a_additivity.py --adapter real 2>&1 | tee artifacts/phase3/e3a/run.log
python phase3_e3b_spatial.py    --adapter real 2>&1 | tee artifacts/phase3/e3b/run.log
```
- `--adapter real` is explicit insurance; plain (auto) also resolves to real once binaries exist.
- A corrupted index that hangs the search is bounded by `--timeout` (default 120 s) and recorded as
  a `crash`; a segfault (nonzero exit) is also a `crash`. A Python-side adapter error aborts loudly
  (it is a harness bug, never counted as a corruption crash).
- Outputs per experiment: `vuln_map.{json,csv}` + `criticality.json` (E1), `e3a.json`, `e3b.json`,
  and a per-flip audit trail under `raw/`: E1 `raw/rabitq.records.jsonl` (+ `raw/rabitq.done`),
  E3a `raw/e3a.records.jsonl`, E3b `raw/e3b.records.jsonl`. `--resume` skips a completed E1 shard.
- **Every result JSON now carries a top-level `meta` block** (`qp.provenance.collect_provenance`):
  - `units` — the scale legend. `pct_*` (vuln_map/criticality) are **percent 0-100**; `collapse_frac`,
    `p1_single_bit_collapse`, and the criticality `frac_*` twins are **fraction 0-1**. (This is the
    fix for reading `0.781`% as `78`.)
  - `platform_confirmed_real` — **CONFIRMED, not inferred**: True only when adapter=real **and**
    `platform.machine()==x86_64` **and** clean@10 is on the plateau (within 0.01 of 0.983). On the
    arm64/stub dev box it is False. `confirmation_basis` records the evidence; `clean_baseline`,
    `study_config.seed`, `index_geometry`, `rabitq.library_commit`, and dep versions are stamped too.
- `criticality.json` collapse fields: `pct_collapse_worst_bucket` (worst per-bit-class bucket — what
  drives the rank) **and** `pct_collapse_overall` (n-weighted across all enumerated bits), each with a
  `frac_*` twin. The 64-B rotation reads worst_bucket ≈ 0.78% vs overall ≈ 0.2% — both, so neither is
  mistaken for the other.

## 4. Bring results back
Copy the WHOLE `artifacts/phase3/{e1,e3a,e3b}/` tree — maps + criticality + **`raw/` (all three
records.jsonl)** + `run.log` — back to the dev machine for analysis, commit, and (if any number
legitimately changed) `python artifacts/phase3/make_golden.py`. Only a run whose `meta.
platform_confirmed_real` is True is a stamped scientific result; the arm64/stub outputs are
plumbing checks. If a run misbehaved, the per-flip jsonl + run.log reproduce/debug it offline against
the stub. Then proceed to Stage 2 (scrub mechanism + temporal).

## Known report-and-stop
- **ex_code bit-class** is flat (`ex_code`). A clean high-vs-low extension-bit split does not exist:
  `pack_excode.hpp` packs `ex_bits` as bit-planes interleaved across non-contiguous bytes in 16/64-dim
  SIMD blocks (distinct packer per `ex_bits`). Finer resolution needs decoding
  `packing_{ex_bits}bit_excode`; carried as a note in `criticality.json`.
