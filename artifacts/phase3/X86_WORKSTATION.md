# Phase 3 Stage 0 — x86-64 Workstation Handoff

One command builds the whole environment from scratch and runs **every** Stage 0 acceptance
item. Use this on an **x86-64 Linux** workstation; the RaBitQ C++ core is AVX2/AVX512-only and
does **not** build on Apple Silicon (on arm64 the byte layout is still readable from source,
but the live binaries / baseline / parity are blocked — that is exactly what this handoff
unblocks).

## TL;DR

```bash
git clone https://github.com/tmousecello/quant_protection.git
cd quant_protection
git checkout phase3-stage0
bash run_stage0_x86.sh
```

That's it. Expect **all tests passed (0 skipped)** and **clean b=7 recall@10 ≈ 0.983**.

## Prerequisites

- **x86-64 Linux** (the script aborts on other arches).
- Toolchain: `git`, `cmake`, `make`, a C++ compiler (`g++`/`c++`), `python3` (3.13). gcc supplies
  OpenMP natively — no libomp needed (that workaround is macOS-only).
- Network access (clones two repos + downloads SIFT1M ~168 MB).
- ~3 GB free disk (SIFT1M ~550 MB unpacked, the 1M b=7 index, the venv).

Install toolchain on Debian/Ubuntu if needed:
```bash
sudo apt-get update && sudo apt-get install -y git cmake build-essential python3 python3-venv
```

## What `run_stage0_x86.sh` does

| Stage | Action |
|---|---|
| 0 preflight | assert x86-64 + required tools |
| 1 locate | `QP_ROOT` = this repo; `RQ_ROOT` = sibling `../StorageSystemProject_RaBitQ_Free-recovery` |
| 2 sibling repo | **auto-clones** the RaBitQ repo if missing (`RQ_REMOTE` overridable) |
| 3 SIFT | `fetch_sift.sh` → `sift/` (idempotent) |
| 4 env | `setup.sh` (`.venv` + `requirements.txt` + `verify_env.py`) then `pip install -e .` |
| 5 build | `build_rabitq.sh` → clone/pin RaBitQ-Library@`7e39df2`, patch, **build all binaries incl `exp_dumpids`**, prepare SIFT, build the b=7 index, print recall-by-ef |
| 6 acceptance | full pytest suite (**parity now runs**) + live clean-baseline gate (`≈0.983`) |
| 7 summary | PASS/FAIL per item + artifact paths |

Everything is idempotent: re-running skips cached clone/build/index/dataset steps.

## Acceptance items (what "all green" means)

1. **Fault models** (`qp/faults.py`) — `tests/test_faults.py`: binomial CI, χ² uniformity,
   per-window clustering, determinism, restore roundtrip, quiet→step temporal curve.
2. **Recall** (`phase3_recall.py` → `qp.metrics`) — `tests/test_recall_predicate.py`:
   `is_silent_collapse` truth table, gt/direction sanity.
3. **Cost** (`phase3_cost.py`) — `tests/test_cost.py`: reconciliation + monotonicity.
4. **Byte layout** (`qp/rabitq/layout.py`) — `tests/test_layout.py`: source-derived offsets.
5. **Golden** — `tests/test_golden.py` reproduces `fixtures/golden_smoke.json`.
6. **Parity (keystone)** — `tests/test_parity.py::test_rabitq_parity_clean_index`: `exp_dumpids`
   runs the real search path and emits both the top-k ids and its own recall; the imported
   `qp.metrics.recall_at_k` recomputes recall on the **same ids** and must equal it, and the
   value must be ≈0.983. On the laptop this test skips (no binaries); on the workstation it runs.
7. **Live baseline** — `adapter.clean_baseline_recall()` ≈ 0.983 (anchors the whole study).

## The parity instrument (`exp_dumpids`)

`rabitq_instrumentation/exp_dumpids.cpp` mirrors the upstream `hnsw_rabitq_querying.cpp` (same
`hnsw.search(query, nq, topk, ef, 1)` call) but, for one `ef`, writes the per-query top-k ids
as ivecs and prints `RECALL\t<r>`. `build_rabitq.sh` copies it into the library `sample/` and
registers a CMake target. `qp/rabitq/adapter.query_ids()` runs it, parses the ivecs with
`qp.data.read_ivecs`, and returns `(ids, cpp_recall)` for the parity assertion.

## Overrides (env vars)

| Var | Default | Purpose |
|---|---|---|
| `RQ_ROOT` | `../StorageSystemProject_RaBitQ_Free-recovery` | sibling repo location |
| `RQ_REMOTE` | ZMYsamuel GitHub URL | sibling clone source |
| `RABITQ_LIB_COMMIT` | `7e39df2…` | pinned upstream RaBitQ-Library commit (build_rabitq.sh) |
| `SIFT_DIR` | `./sift` | SIFT1M input dir (build_rabitq.sh) |

## Troubleshooting

- **Aborts at preflight on arm64** — expected; use an x86-64 host (see top of this doc).
- **Sibling clone fails / private** — set `RQ_REMOTE=…` (or `RQ_ROOT=…` pointing at an existing
  local checkout) and re-run.
- **`cmake/make failed`** — confirm `build-essential` (g++) is installed; the library uses
  `-march=native -fopenmp`, both supported by Linux gcc.
- **Baseline not ≈0.983** — stop and investigate before trusting any corruption numbers; check
  the prepare step produced `data/sift/prepared/{base,query,groundtruth}` and the index built.
- **Parity mismatch (qp vs C++)** — means the imported metric and Samuel's recall disagree on
  identical ids; that is a real defect to fix, not a tolerance issue.

See `artifacts/phase3/STAGE0_STATUS.md` for component-by-component status and the (now resolved)
blocked items.
