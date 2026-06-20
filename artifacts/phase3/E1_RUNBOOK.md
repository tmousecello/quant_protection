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

## 1. First shot — is the 64-byte rotation catastrophic?
Before the full sweep, flip a single bit of the **rotation** region and look at recall. This is
both (a) a physical confirmation that the region map is aligned to the real file and (b) E1's first
datapoint (the headline hypothesis: rotation = RaBitQ's `sq_scale` single-point catastrophe).

```bash
python phase3_e1_vuln.py --adapter real --smoke    # tiny budget; rotation is enumerated first
```
Inspect `artifacts_smoke/phase3/e1/vuln_map.json`: the `rotation` rows should show
`pct_collapse` near 100 (silent collapse: finite-but-wrong distances, not nan-inf/crash). **If
recall does NOT collapse, STOP and report** — either the region map is offset against this build, or
the structured rotation is more graceful than predicted. Both are findings; do not proceed blindly.

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
  and `raw/rabitq.records.jsonl` (one row per flip) + `raw/rabitq.done`. `--resume` skips a
  completed shard.

## 4. Bring results back
Copy `artifacts/phase3/e1|e3a|e3b/` (maps + criticality + raw jsonl + run.log) back to the dev
machine for analysis, commit, and (if any number legitimately changed) `python
artifacts/phase3/make_golden.py`. If a run misbehaved, the per-flip jsonl + run.log are enough to
reproduce/debug offline against the stub. Then proceed to Stage 2 (scrub mechanism + temporal).

## Known report-and-stop
- **ex_code bit-class** is flat (`ex_code`). A clean high-vs-low extension-bit split does not exist:
  `pack_excode.hpp` packs `ex_bits` as bit-planes interleaved across non-contiguous bytes in 16/64-dim
  SIMD blocks (distinct packer per `ex_bits`). Finer resolution needs decoding
  `packing_{ex_bits}bit_excode`; carried as a note in `criticality.json`.
