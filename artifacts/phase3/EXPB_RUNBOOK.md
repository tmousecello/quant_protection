# Experiment B runbook — ex-slope corruption vs recovery mode (Option B)

x86-64 workstation. Prereqs: `setup.sh` done, `verify_env.py` exits 0, sibling repo
`../StorageSystemProject_RaBitQ_Free-recovery` present, SIFT in `sift/`.

The scientific question: **how much time does EB-fallback buy on the ex slope**, with
detect-&-drop as the control (F3's slope half). Division of labour is hard-ruled
(`plan/stage2_cpp_patch.md`): Python injects and computes recall; C++ only loads, CRC-checks
against the qp-written manifest, and applies Samuel's exp-3 policy.

## 1. Build (gate 1)

```bash
bash build_rabitq.sh
```

Expect: `applying recovery-changes.patch (Option B)` (or `recovery patch: already applied`),
9 binaries in `third_party/RaBitQ-Library/bin/`, and the step-7 plateau at
`ef=2000 recall@10=0.98376`. A cached `exp_dumpids` that predates `--recovery` is deleted and
the whole sample set recompiles (the patched `hnsw.hpp` is shared).

If `recovery-changes.patch` fails to apply: it must land ON TOP of `library-changes.patch`
at pin `7e39df2`; re-clone and rerun the script (idempotent).

## 2. Acceptance gates 2–5

```bash
source .venv/bin/activate
python phase3_expb_recovery.py --adapter real --gate
```

Runs in order, stops at the first failure, exit 0 only when all green
(`artifacts/phase3/expb/expb_gates.json`):

| gate | check | expected |
|---|---|---|
| 1 | binaries + usage has `--recovery` | pass |
| 2 | clean index, all 3 modes | ids identical to the no-flag baseline; recall == 0.98376 ± 5e-4; `elements_crc_fail == 0` |
| 3 | 5% uniform elements + `fallback_eb` | recall within ±0.01 of Samuel's F3 median **0.941865** (measured here: 0.94328) |
| 4 | f=0.20, three modes, SEVERE damage (half of ex_code bits per element) | `fallback_eb ≥ drop`, `none` worst |
| 5 | rerun gate-3 config, same seed | identical element selection AND identical ids |

**Gate 3 failure = report-and-stop** (EB semantics diverged): check the pessimistic formula
(`est + (est − low)` at the hnsw.hpp EB site), the error-bound field (`bin_factors.f_error`
via the 1-bit estimate), and tie-breaking (`BoundedKNN` ascending `est_dist`). If marginal,
retry `--ef 1000` (Samuel's plateau grid) before declaring divergence.

**Gate 4 severity note** (measured on this workstation, f=0.20 uniform): `drop`/`fallback_eb`
are severity-INVARIANT (≈0.797 / ≈0.815 whether 4 or 384 bits are flipped per element —
detection only depends on which elements fail CRC), but `none` crosses from BEST at light
damage (0.907 @ 4 flips) to catastrophic at garbage (0.126 @ 384 flips), crossover below
32 flips (0.625). The gated ordering uses severe damage; the light-damage ordering is printed
as informational. This severity dependence is itself an Experiment-B result — report it, don't
average over it.

## 3. Sweep (Experiment B main body)

```bash
# light per-element damage (default 4 flips / 96-B ex_code) — the DRAM-ramp-realistic end
python phase3_expb_recovery.py --adapter real --sweep

# severe per-element damage (garbage chunks) — the spec's "silently using bad data" end
python phase3_expb_recovery.py --adapter real --sweep --flips-per-element 384 --out-tag sev384
```

4 patterns × fractions {0.01, 0.05, 0.08, 0.20, 0.57} × 3 recovery modes = 60 C++ runs per
sweep; ~15 s each at ef=2000 plus injection time → budget **≤ 1 h per sweep**. Long runs:
`tmux` per WORKSTATION.md.

**Never run two expb processes concurrently** — `_expb_clean.index`, `_expb_corrupt.index`,
the CRC manifest and the dumpids scratch files under `expb/` are shared per-directory
scratch; parallel sweeps would cross-write them. Tagged sweeps (`--out-tag`) shard their
records and meta (`expb_meta_<tag>.json`) but still share the scratch — run sequentially.

Outputs in `artifacts/phase3/expb/` (git-ignored; bring numbers back per `git_tasks.sh`):
- `expb_meta[_<tag>].json` — provenance, stamped BEFORE any corruption run
  (`corruption.corrupted_regions == ["ex_code"]`, manifest + clean index sha256).
- `expb_<pattern>_ex_code[_<tag>].records.jsonl` — one row per (fraction × recovery):
  `recall@10` (qp.metrics, authoritative), `cpp_recall` (parity only), `delta_vs_clean`,
  `stats` (load-time CRC scan + per-query consults/corrupt_hits/fallbacks/drops),
  `failure_mode`, `silent_collapse`, `index_sha256`, `flips_per_element`.
- `expb_summary[_<tag>].json` — `eb_minus_drop` / `eb_minus_none` per (pattern, fraction).

Sanity invariants the driver asserts per run: the CRC scan flags EXACTLY the injected element
count, and inject→restore is byte-identity against the clean buffer.

## 4. Cheap follow-ups (Experiment A close-out, can run alongside the sweep)

```bash
# per-tick NO-RECOVERY recall timeline (turns A's "cliff ≤ 2 ticks" derivation into data)
python phase3_e3c_temporal.py --adapter real --region rotation --measure-recall

# replica-failure boundary: corrupt the R=3 rotation copies too (same p, independent streams)
python phase3_e5_recovery.py --adapter real --region rotation --inject-replicas
```

Both flag-gated; without the flags the runners produce the original record schema unchanged.

## 5. `--crc-mode lazy` alignment run (gate 6, one command)

```bash
source .venv/bin/activate
python phase3_expb_recovery.py --adapter real --lazy-gate
```

Requires an **AVX512BW** CPU like every other real run here (RaBitQ aborts at
`utils/space.hpp:937` otherwise — check with `grep -o avx512bw /proc/cpuinfo | head -1`).
Rebuild first if the tree predates the flag: `bash build_rabitq.sh` and confirm
`exp_dumpids` usage lists `--crc-mode` (the build script forces a rebuild when it does not).

Runs `uniform_accum` @ f=0.05 under both detecting policies and stops at the first failure
(`artifacts/phase3/expb/expb_lazy_gates.json`):

| gate | check | expected |
|---|---|---|
| 0 | binary usage has `--crc-mode` | pass |
| 1 | clean index, both policies, both modes | nothing flagged in either mode |
| 2 | **load vs lazy return identical top-k ids** and identical recall | pass |
| 3 | `0 < lazy distinct failures <= load scan failures` | pass |
| 4 | overhead measured (reported, never asserted) | numbers written |

**Gate 2 failure = report-and-stop.** The decision points are untouched between the modes, so
identical ids is a consequence, not a hypothesis — a difference is an implementation bug in
`ex_bad()` / `ex_corrupted_now()` (`hnsw.hpp`), not a finding. Do not "investigate the
discrepancy" in the paper; fix the code.

Three overhead numbers land in `expb_lazy_gates.json` per policy and must agree in magnitude:

- `headline_delta_ns_per_query` / `headline_pct_of_search` — `search_wall_ns(lazy) −
  search_wall_ns(load)`. **The one to quote.** It is unperturbed by any per-check clock, and
  it is sound precisely because gate 2 proved both runs do the same search work.
- `timed_ns_per_check` — from the separate `--crc-timer` run. Inflated by two
  `steady_clock` reads (~40–50 ns) around a ~96-byte CRC (~100–200 ns); a cross-check, not the
  headline.
- `analytic_ns_per_query` — `crc_bytes × (load scan ns/byte)`, the least perturbed CRC rate
  available since the load scan is timed once over 1M elements.

Expect `drop` ≫ `fallback_eb`: drop consults every unvisited neighbour before the bounds guard
and never reads the ex block it just CRC'd, while EB CRCs exactly the bytes `get_full_est` is
about to load. That asymmetry is the honest scope of the piggyback claim — see
`docs/claim_impl_map.md`.

Feed the numbers forward with `python phase3_f_synth.py --synthesize --adapter real`, which
reads `expb_lazy_gates.json` into `frontier.csv`'s `detect_*` columns (blank, never zero, if
the run is absent or failed).

**Regression**: gates 1–5 (§2) must still be green in load mode, and Phase D's G sanity
(`identical_to_truth`) unchanged. G rejects lazy records by design — never point it at a
`--crc-mode lazy` sweep.

## 6. Bring results back

`expb/*.jsonl`, `expb_summary*.json`, `expb_gates.json`, `expb_meta.json` + run logs, via
`git_tasks.sh save` conventions (raw jsonl stays on the workstation, numbers/curves travel).
