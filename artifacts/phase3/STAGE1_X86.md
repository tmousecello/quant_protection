# Phase 3 Stage 1 — x86-64 Workstation Handoff

One command builds the whole environment from scratch and runs **every** Stage 1 acceptance
item — both the engineering pipeline (stub-green tests, which on x86 also execute the parity
keystone against the real binaries) and the **scientific** runs that can only be produced here:
the real RaBitQ **E1 vulnerability map**, **E3a additivity**, and **E3b spatial** results.

Use this on an **x86-64 Linux** workstation. The RaBitQ C++ core is AVX2/AVX512-only and does
**not** build on Apple Silicon — on arm64 the byte layout is readable from source, but the live
binaries / baseline / parity / corruption sweeps are all blocked. This handoff is what unblocks
them. (Engineering for Stage 1 was developed and unit-tested on arm64 against a deterministic
**stub** adapter; here the same code runs unchanged with `--adapter real`.)

## TL;DR

```bash
git clone https://github.com/tmousecello/quant_protection.git   # or: git pull
cd quant_protection
git checkout phase3-stage1
bash run_stage1_x86.sh
```

That's it. Expect **all tests passed**, **clean b=7 recall@10 ≈ 0.983**, the **rotation
first-shot collapsing** (the F1 headline), and the three scientific outputs written under
`artifacts/phase3/{e1,e3a,e3b}/`.

Want to validate the pipeline fast first (build + tests + first-shot, **no** long sweep)?

```bash
bash run_stage1_x86.sh --smoke      # minutes; stops before the full E1/E3a/E3b runs
bash run_stage1_x86.sh              # then the real thing (add --resume to continue a partial run)
```

## Prerequisites

- **x86-64 Linux** (the script aborts on other arches at preflight).
- Toolchain: `git`, `cmake`, `make`, a C++ compiler (`g++`/`c++`), `python3` (3.13). gcc supplies
  OpenMP natively — no libomp needed (that workaround is macOS-only).
- Network access (clones two repos + downloads SIFT1M ~168 MB).
- ~3 GB free disk (SIFT1M ~550 MB unpacked, the 1M b=7 index, the venv).

Install toolchain on Debian/Ubuntu if needed:
```bash
sudo apt-get update && sudo apt-get install -y git cmake build-essential python3 python3-venv
```

## What `run_stage1_x86.sh` does

| Step | Action |
|---|---|
| 0 preflight | assert x86-64 + required tools; print mode (full vs `--smoke`) |
| 1 locate | `QP_ROOT` = this repo; `RQ_ROOT` = sibling `../StorageSystemProject_RaBitQ_Free-recovery` |
| 2 sibling repo | **auto-clones** the RaBitQ repo if missing (`RQ_REMOTE` overridable) |
| 3 SIFT | `fetch_sift.sh` → `sift/` (idempotent) |
| 4 env | `setup.sh` (`.venv` + `requirements.txt` + `verify_env.py`) then `pip install -e .` |
| 5 build | `build_rabitq.sh` → clone/pin RaBitQ-Library@`7e39df2`, patch, build all binaries incl `exp_dumpids`, prepare SIFT, build the b=7 index |
| 6 adapter | `adapter.binaries_built()` → True, `adapter.clean_baseline_recall()` ≈ 0.983 (anchors the study) |
| 7 pytest | full suite (**parity keystone now runs**: `qp.metrics` recall on the real top-k ids must equal the C++'s own recall, ≈0.983) |
| 8 first shot | single-bit **rotation** flip on the real index via `phase3_e1_vuln.py --adapter real --smoke`; gate on mean `pct_collapse` ≥ `ROT_COLLAPSE_MIN` (default 50) |
| 9 full runs | `phase3_e1_vuln.py` / `e3a` / `e3b` with `--adapter real`, tee'd to per-experiment `run.log` *(skipped under `--smoke`)* |
| 10 summary | PASS/FAIL per item + output-file presence + artifact paths |

Everything is idempotent: re-running skips cached clone/build/index/dataset steps. The runners'
`--resume` (pass `bash run_stage1_x86.sh --resume`) continues a partially-completed shard.

## Acceptance items (what "all green" means)

1. **Live clean baseline** — `clean_baseline_recall()` ≈ 0.983. If this is off, **stop**: every
   corruption number is measured relative to it.
2. **Engineering pipeline** — the full pytest suite, including everything that was stub-only on the
   laptop. On x86 the **parity** test runs for real: `exp_dumpids` executes the real search path,
   emits top-k ids + its own recall, and the imported `qp.metrics.recall_at_k` must reproduce that
   recall on the *same* ids (≈0.983). A mismatch is a real defect, not a tolerance issue.
3. **First shot (region-map alignment + E1 headline)** — flipping one bit of the 64-byte rotation
   (RaBitQ's `sq_scale`-analog single-point structure) must silently collapse recall. This both
   confirms the region map is byte-aligned to *this* build and is E1's first datapoint.
   **Report-and-stop**: if rotation does **not** collapse (mean `pct_collapse` < `ROT_COLLAPSE_MIN`),
   the script exits 7 with a message — that is a finding (map offset, or rotation more graceful than
   predicted), not necessarily a bug. Do not proceed blindly; inspect
   `artifacts_smoke/phase3/e1/vuln_map.json`.
4. **E1 full vuln map** — `artifacts/phase3/e1/vuln_map.{json,csv}` + `criticality.json` (Top-Down
   reduction order + three-tier scrub allocation + cost). Rotation exhaustive (512 bits),
   per-vector structures sampled with bootstrap CI.
5. **E3a additivity** — `artifacts/phase3/e3a/e3a.json`: measured vs predicted `P(k)=1-(1-p1)^k`.
6. **E3b spatial** — `artifacts/phase3/e3b/e3b.json`: clustered-vs-uniform collapse at a fixed bit
   budget (+ `cross_row` reported at its true 2-bit budget, excluded from the verdict).

Items 1–3 also gate `--smoke`; 4–6 are the full-run deliverables.

## Optional: complete the failure taxonomy (nan-inf)

`exp_dumpids` currently emits ids + `RECALL`, so the runner detects **crash** + **silent** but
reports `n_nan_inf` as *not-detected* (`nan_inf_supported=false`, honestly flagged — never zero).
To get the full silent / nan-inf / crash split via the identical `qp.metrics` classifier, add a
companion `<out>.dist.fvecs` top-k distance dump in `rabitq_instrumentation/exp_dumpids.cpp` and
rebuild via `build_rabitq.sh`; the adapter already reads it when present. See **E1_RUNBOOK.md §2** —
this is optional and the silent-collapse map (the F1 headline) is produced either way.

## Overrides (env vars)

| Var | Default | Purpose |
|---|---|---|
| `RQ_ROOT` | `../StorageSystemProject_RaBitQ_Free-recovery` | sibling repo location |
| `RQ_REMOTE` | ZMYsamuel GitHub URL | sibling clone source (set if private/forked) |
| `RABITQ_LIB_COMMIT` | `7e39df2…` | pinned upstream RaBitQ-Library commit (build_rabitq.sh) |
| `SIFT_DIR` | `./sift` | SIFT1M input dir (build_rabitq.sh) |
| `ROT_COLLAPSE_MIN` | `50` | first-shot rotation collapse gate (mean `pct_collapse`) |

## Bring results back

Copy `artifacts/phase3/{e1,e3a,e3b}/` (maps + criticality + `e3a.json`/`e3b.json` + per-flip
`raw/rabitq.records.jsonl` + `run.log`) back to the dev machine for analysis and commit. If any
number legitimately changed, regenerate the golden:
`python artifacts/phase3/make_golden.py`. If a run misbehaved, the per-flip jsonl + run.log
reproduce/debug it offline against the stub. Then proceed to **Stage 2** (scrub mechanism +
temporal).

## Troubleshooting

- **Aborts at preflight on arm64** — expected; use an x86-64 host.
- **Sibling clone fails / private** — set `RQ_REMOTE=…` (or `RQ_ROOT=…` at an existing checkout).
- **`cmake/make failed`** — confirm `build-essential` (g++); the library uses `-march=native
  -fopenmp`, both supported by Linux gcc.
- **Baseline not ≈0.983** — stop; check `data/sift/prepared/{base,query,groundtruth}` exist and the
  b=7 index built.
- **First shot exits 7** — read the printed report-and-stop block; it is the designed gate, not a
  crash. Inspect the rotation rows before re-running.
- **Parity mismatch** — the imported metric and the C++ recall disagree on identical ids; a real
  defect to fix.

Related: **E1_RUNBOOK.md** (per-experiment runbook + the nan-inf extension), **X86_WORKSTATION.md**
(Stage 0 handoff, `run_stage0_x86.sh`), **STAGE0_STATUS.md** (component-by-component status).
