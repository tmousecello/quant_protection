# Phase 2 — How to run the full experiments

This directory holds Phase 2: does a *quantized* index carry a catastrophic single-bit /
burst structure that full-precision indexes lack? Five scripts (+ a `burst_flip` extension)
were **implemented and smoke-tested only** — no full measurement has been run. This README is
the human runbook: per script, the full command, parameters, outputs/schema, expected cost,
and the recommended order. `prompt.md` is the authoritative design.

Everything reuses the Phase 0/1 substrate (`qp/flip.py`, `qp/buckets.py`, `qp/metrics.py`,
`qp/indexes.py`, `phase1_sensitivity.py`, `artifacts/phase1/regions_aug/*`) and writes only
under `artifacts/phase2/`. Phase 0/1 artifacts and the locked `qp/config.py` control
variables are untouched; Phase 2 added one **additive** config block (`PHASE2_*`) and
`qp/flip.burst_flip` / `qp/isolation.py`.

```bash
cd quant_protection && source .venv/bin/activate     # FAISS 1.14.2, Python 3.13
```

## Recommended order

```
Tier 1 (PQ sensitivity)  ─┐
                          ├─►  Tier 3 rollup  ─►  Tier 4 detection
Tier 2 (graph edges)     ─┘
Burst sub-study  ── independent (any time)
```
Tier 1 and Tier 2 are independent and can run in parallel (different indexes). Tier 3 reads
both their maps plus `artifacts/phase1/vuln_map.csv`. Tier 4 reads Tier 1's raw shards. The
burst sub-study is standalone.

> ### ⚠ Clean-start prerequisite — run before ANY Tier 3 / Tier 4 measurement
>
> The byte-region geometry changed: the IVF code block is now split into a crash-prone
> `codes_meta` header + the recall-relevant `codes` body, with `codes` anchored at the first
> real inverted-list code (so the fp32 grid is phase-aligned). The committed
> `artifacts/phase1/vuln_map.csv` is **stale AND incomplete** — built on the OLD geometry and
> **missing IVF_SQ8** (whose `sq_scale` is the headline catastrophic structure) and the PQ
> family. **Do not use it for Tier 3/Tier 4.** Run, in order:
>
> ```bash
> python phase0_build.py --regions-only    # rebuild artifacts/regions/* (now carries codes_meta)
> python phase1_region_accounting.py       # propagate into artifacts/phase1/regions_aug/* (fast)
> python phase1_sensitivity.py             # FULL single-bit sweep on the NEW geometry,
>                                          # INCLUDING IVF_SQ8 (~6 h; run on the workstation)
> ```
>
> Tier 3 (rollup) and Tier 4 (detection) are only meaningful once `phase1/vuln_map.csv` +
> `phase1/raw/*.records.jsonl` are freshly produced on the new geometry. Tier 1 / Tier 2 / the
> burst study generate their own data and need only the regenerated `regions_aug` (first two
> steps).

## Shared conventions

- Output schema extends Phase 1 `vuln_map.csv` with **`baseline_mode`** (`own` for PQ, else
  `aligned`). Catastrophic is **absolute** ΔR@10 > 0.01 for aligned indexes and **relative**
  (retention < `PHASE2_PQ_RETENTION_FRAC`=0.5 of own clean) for PQ — both are recorded.
- Seeds fixed (`config.SEED`=1234); every run reproducible — including the Tier 3
  `--validate-multiflip` placements (a stable per-target stream id, not `hash(name)`).
- **Failure modes** are counted separately as `silent-wrong / nan-inf / crash`. Only true
  corruption crashes (a flipped buffer that fails to deserialize/search, or a C++ segfault)
  count as `crash`.
- **Crash isolation:** Tier 2 and the burst study run **each flip in its own subprocess**
  (`qp/isolation.py`) so a FAISS C++ segfault is recorded as `crash` instead of killing the
  sweep. An *unexpected harness* failure inside a child (bad import under spawn, etc.) is
  tagged `harness-error` — **not** counted as a corruption `crash` — and aborts the sweep
  loudly so it can't silently skew the crash statistics. Tier 1 deserializes cleanly (Phase 0
  dry-run) and runs in-process.
- `--smoke` on every script runs a seconds-scale subset into `artifacts_smoke/phase2/`
  (NOT `artifacts/phase2/`) and is what was used to validate the pipeline.

---

## Tier 1 — `phase2_pq_sensitivity.py` (PQ single-bit map)

PQ single-bit sensitivity for `IVF_PQ_M8` / `IVF_PQ_M16` vs each index's **own** clean recall
(M8≈0.379, M16≈0.563). Tests whether PQ *codes* break code-immunity and turns `pq_codebook`
from prediction into measurement.

```bash
python phase2_pq_sensitivity.py --resume                 # both PQ indexes, full sampling
# or split across two processes, then consolidate:
python phase2_pq_sensitivity.py --indexes IVF_PQ_M8  --resume
python phase2_pq_sensitivity.py --indexes IVF_PQ_M16 --resume
python phase2_pq_sensitivity.py --aggregate-only
```

- **Key flags:** `--s-large` (pq_codes, default 1000), `--s-centroid` (2000),
  `--s-codebook` (tag-stratified codebook samples, default 4000),
  `--full-enum-codebook` (exhaust all ~1.05M codebook bits — slow), `--queries` (10000),
  `--resume`, `--aggregate-only`.
- **Outputs:**
  - `artifacts/phase2/vuln_map_pq.csv` / `.json` — aggregated by (index × region ×
    bit_position_tag). Columns: `index, region, kind, bucket, bit_position_tag, region_bits,
    n, mean_dR@10, p99_dR@10, max_dR@10, mean_dRecall@{1,100}, mean_dTol, pct_benign,
    pct_catastrophic` (relative), `pct_catastrophic_abs, ci95_low/high, n_silent_wrong,
    n_nan_inf, n_crash, n_clean, baseline_mode=own, seed`. Tags: `code_b0..code_b7` for
    `pq_codes` (index bits, not IEEE-754), `sign/exponent/mantissa-high/mantissa-low` for
    `pq_codebook`/`centroid`.
  - `artifacts/phase2/raw/<NAME>.records.jsonl` — per-flip rows (consumed by Tier 4) + `.done`.
- **Expected cost:** ≈ (S_codes + S_centroid + S_codebook) × 2 indexes searches over 10k
  queries at @100 ≈ a few hours serial at the default S; `--full-enum-codebook` is ~1.05M
  flips/index (overnight+). Each search ≈ the index's clean QPS.
- **Smoke (passed):** `python phase2_pq_sensitivity.py --smoke --indexes IVF_PQ_M8` (~4 s).

## Tier 2 — `phase2_graph_sensitivity.py` (graph_edges characterization)

Classifies `graph_edges` flips for `HNSW` / `HNSW_SQ8` into **crash** (OOB id → segfault,
detectable) / **silent-benign** (valid wrong neighbour, ΔR≈0) / **silent-harmful** (ΔR above
threshold). Subprocess-isolated.

```bash
python phase2_graph_sensitivity.py --n-edges 4000 --queries 10000 --resume
```

- **Key flags:** `--n-edges` (stratified across the 4 int32 byte lanes, default 4000),
  `--harmful-thresh` (default 0.01), `--queries`, `--timeout` (per-flip subprocess budget,
  default 120 s), `--resume`.
- **Outputs:** `artifacts/phase2/graph_edges_characterization.csv` / `.json`, per
  (index × graph_edges × tag) plus an `ALL` rollup. Columns: `index, region,
  bit_position_tag` (`id_high`/`id_low`/`ALL`), `region_bits, n, pct_crash,
  pct_silent_benign, pct_silent_harmful, pct_nan_inf, mean_dR@10, p99_dR@10, max_dR@10,
  n_crash, n_silent_wrong, n_nan_inf, n_clean, harmful_thresh, baseline_mode=aligned, seed`.
  Plus per-flip `artifacts/phase2/raw/<NAME>.edges.jsonl` + `.done`.
- **Expected cost:** one subprocess **per** edge flip (each re-deserializes the index from a
  temp `.npy` and searches), so ≈ `n_edges × 2` × (deserialize + search). At `--n-edges 4000`,
  budget a few hours per index; cut `--queries` to speed the search component.
- **Smoke (passed):** `python phase2_graph_sensitivity.py --smoke --indexes HNSW_SQ8` (~6 s;
  confirms an isolated `graph_meta` flip is captured as `crash` without killing the parent).

## Tier 3 — `phase2_rollup.py` (faults/MB rollup — analysis only, no flipping)

Synthesizes the two headline curves over an absolute per-bit rate `r`. **Reads existing maps,
injects nothing.**

```bash
python phase2_rollup.py                                   # uses maps under artifacts/phase2
python phase2_rollup.py --validate-multiflip --multiflip-trials 10   # optional non-additivity check
```

- **Key flags:** `--phase1-map` (default `artifacts/phase1/vuln_map.csv`), `--pq-map`,
  `--graph-map`, `--rates` (comma list; default `config.PHASE2_RATE_GRID` =
  `1e-9…1e-5`), `--validate-multiflip` (+`--multiflip-trials/-ks/-queries`).
- **Normalization:** absolute **per-bit** error rate. Convert from a DRAM field-study FIT/Mbit
  figure (e.g. Schroeder et al.) to a per-bit-hour probability for the x-axis. Collapse risk
  is driven by the **absolute bit count** of a critical structure (`cat_bits = catastrophic_
  fraction × region_bits`), not a per-flip conditional probability.
- **Outputs (under `artifacts/phase2/rollup/`):**
  - `curveB_collapse_prob.csv` — **PRIMARY.** `index, baseline_mode, r, cat_bits_total,
    P_collapse`. **Excludes `graph_edges`** (its own series, below), so fp32 indexes
    (FLAT/IVF_FLAT/HNSW) → `cat_bits=0` → `P_collapse≈0`; SQ8 driven by `sq_scale`, PQ by
    `pq_codebook`. This is the "quantization introduces a catastrophic single-point structure
    fp32 lacks" plot.
  - `curveB_graph_edges.csv` — graph_edges as its **own** series (same columns). HNSW and
    HNSW_SQ8 share an IDENTICAL graph_edges table — a *controlled* variable, not an encoding
    effect — whose ~2e9-bit count would otherwise dominate Curve B and make fp32 HNSW appear to
    collapse. It is kept out of the primary curve and shown here instead. (Edge crashes are
    detectable, so this uses `silent-harmful` only.)
  - `curveA_expected_dR.csv` (+ `curveA_graph_edges.csv`) — APPENDIX. `index, baseline_mode, r,
    expected_dRecall@10`. Linear superposition, valid only while `region_bits·r ≪ 1`; past that
    regime the total is **clamped to the physical [-1, 1] recall-drop range** (non-finite terms
    dropped) so the curve never reports an impossible recall loss.
  - `region_terms.csv` — per-(index,region) `region_bits, E_dR, cat_frac, series, source`
    attribution (`series` = `main` | `graph_edges`).
  - `validate_multiflip.csv` (only with the flag) — `index, region, k, trial, combined_dR@10,
    sum_singles_dR@10, ratio`.
- **Expected cost:** seconds (pure analysis). `--validate-multiflip` adds a few hundred
  in-process searches (minutes).
- **Smoke (passed):** `python phase2_rollup.py --smoke` (reads the Phase 1 map + Tier 1/2
  smoke maps; confirms fp32 Curve B ≈ 0, HNSW_SQ8 sq_scale > 0).

## Tier 4 — `phase2_detection.py` (range-check guard)

Builds a value-range guard from clean metadata and measures how much catastrophic damage it
catches. No search needed to evaluate the guard — it scans raw metadata bytes.

```bash
python phase2_detection.py                                # replays phase1/raw + phase2/raw flips
```

- **Key flags:** `--raw-dirs` (default `artifacts/phase1/raw` + `artifacts/phase2/raw`),
  `--pad` (fractional slack on [min,max] before flagging, default 0 = strict),
  `--harmful-thresh` (0.01), `--queries` (overhead baseline, default 2000).
- **Outputs (under `artifacts/phase2/detection/`):**
  - `expected_ranges.json` — per (index, region) `min/max/q01/q99/n` for `sq_scale /
    centroid / pq_codebook`.
  - `coverage.csv` — `index, region, severity` (catastrophic/moderate/benign/crash), `n,
    n_detected, coverage_pct`.
  - `detection_summary.json` — overall catastrophic/moderate coverage, benign false-positive
    rate, and guard-scan vs one-query overhead.
- **Expected cost:** minutes (replays are byte-range checks, not searches). Coverage is only
  meaningful once Tier 1 + Phase 1 raw shards exist.
- **Smoke (passed):** `python phase2_detection.py --smoke`.

## Burst sub-study — `phase2_burst.py` (clustered injection)

Spatially-clustered faults. Sweeps burst length `B` (bits) for `IVF_PQ_M8` / `HNSW_SQ8` /
`IVF_FLAT`; subprocess-isolated.

```bash
python phase2_burst.py --trials 50 --queries 10000               # random placement
python phase2_burst.py --aligned-worstcase --trials 50           # land on the critical structure
```

- **Key flags:** `--bursts` (default `config.PHASE2_BURST_B` =
  `1,8,64,512,1024,8192,65536,262144`), `--trials` (placements per B, default 50),
  `--aligned-worstcase` (start each burst at the index's smallest critical structure:
  pq_codebook / sq_scale / centroid), `--queries`, `--timeout`, `--pq-retention-frac`.
- **Outputs (under `artifacts/phase2/burst/`):** `<index>_burst.csv` per (index × B):
  `B_bits, n_trials, p_collapse, mean_dR@10, p99_dR@10, max_dR@10, n_crash, n_nan_inf,
  n_silent_wrong, n_clean, region_hits` (JSON histogram), `baseline_mode, aligned_worstcase,
  seed`. Plus per-trial `<index>_burst.raw.jsonl` + `.done`.
- **Expected cost:** one subprocess per trial × `len(bursts)` × 3 indexes. At `--trials 50`
  and 8 burst lengths, budget a few hours; large `B` trials are no slower than small ones
  (the search dominates, not the XOR).
- **Smoke (passed):** `python phase2_burst.py --smoke --indexes HNSW_SQ8 --aligned-worstcase`
  (a 64-bit burst into sq_scale collapses recall every trial; confirms `burst_flip` is exactly
  self-inverse and subprocess isolation catches crashes).

---

## What "collapse" / "catastrophic" mean (recap)

| index family | baseline_mode | catastrophic rule |
|---|---|---|
| FLAT / IVF_FLAT / HNSW / IVF_SQ8 / HNSW_SQ8 | `aligned` | ΔR@10 > 0.01 (absolute) |
| IVF_PQ_M8 / IVF_PQ_M16 | `own` | faulted recall@10 < 0.5 × own clean recall@10 |

Both the relative and absolute counts are stored in every Tier 1 row, so Tier 3 can choose.

## Files added by Phase 2

- Scripts: `phase2_pq_sensitivity.py`, `phase2_graph_sensitivity.py`, `phase2_rollup.py`,
  `phase2_detection.py`, `phase2_burst.py`.
- Library (additive): `qp/isolation.py` (subprocess crash isolation), `qp/flip.burst_flip` /
  `qp/flip.burst_positions`, and a `PHASE2_*` block in `qp/config.py`.
- `artifacts/baseline.*` and the locked `qp/config.py` values are untouched. The byte-region
  maps (`artifacts/regions/*`, `artifacts/phase1/regions_aug/*`) **were regenerated** for the
  `codes_meta` split — re-derived from the saved indexes, not hand-edited.

## Correctness fixes applied (post-review)

- **Region geometry:** `qp/regions.py` splits the crash-prone IVF invlist header into
  `codes_meta` and anchors `codes` at the first real list code (longer, unique needle). Region
  maps regenerated. Phase 1 must be re-run on this geometry (see the clean-start prerequisite).
- **Tier 3 rollup:** `graph_edges` moved to its own Curve A/B series; primary Curve B excludes
  it (restores fp32 ≈ 0). Curve A clamped to the physical range. `--validate-multiflip` seed
  made reproducible.
- **Isolation:** children `mmap` the pristine dump (no 2× RSS / OOM-induced false crashes);
  harness errors are separated from corruption crashes.
- **Tier 4 detection:** guard hardened (fp32 view trimmed to whole elements, empty-range &
  malformed-record guards, restore wrapped in `try/finally`).
- **Burst:** burst start clamped in-bounds (no fake `crash` from out-of-range XOR); `--bursts`
  honored under `--smoke`; empty `--indexes` handled; redundant re-serialize removed.
```
