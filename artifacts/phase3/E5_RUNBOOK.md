# E5 Recovery + E3c Temporal — Workstation Runbook (Stage 2 Round 1)

## Prerequisites

Same as `E1_RUNBOOK.md`: x86-64 Linux, binaries built (`build_rabitq.sh`), real adapter configured.

```bash
source .venv/bin/activate
python verify_env.py          # should exit 0
python -c "from qp.rabitq.adapter import binaries_built; assert binaries_built()"
```

---

## Smoke A — E3c temporal accumulation (rotation, uniform pattern)

```bash
python phase3_e3c_temporal.py --adapter real --region rotation \
    --pattern uniform_accum --ticks 20 --p 0.001 --smoke
```

**Expected output** (stdout):
```
[e3c] adapter=real  region=rotation  pattern=uniform_accum  ticks=20
  tick=  0  bits_flipped=   X  fraction=0.XXXX
  ...
[e3c] reset OK (drift=0)
```

**Verify:**
- `bits_flipped` increases monotonically (rare double-flip cancellations are fine)
- `reset OK (drift=0)` confirms XOR-toggle correctness
- Check `artifacts_smoke/phase3/e3c/e3c_uniform_accum_rotation.json` exists

**If drift ≠ 0**: bug in `_toggle` XOR accounting — REPORT-AND-STOP.

---

## Smoke B — E5 cliff layer (rotation repair, short timeline)

```bash
python phase3_e5_recovery.py --adapter real --region rotation \
    --pattern uniform_accum --ticks 20 --p 0.001 --smoke
```

**Expected output:**
```
[e5] adapter=real  cliff+slope recovery  ticks=20
  tick=  0  cliff_repaired=X  slope_failed=0  recall=0.9XX
  ...
[e5] done → artifacts_smoke/phase3/e5/e5_rotation.json
```

**Verify:**
- `cliff_repaired` increments per tick (copies accumulate, buf is repaired each tick)
- `cliff_irrecoverable == 0` (p=0.001 is too low for ≥2 copies to flip same bit in 20 ticks)
- `recall ≈ 0.983` (cliff repair keeps the rotation clean → baseline recall maintained)

**If recall drops below 0.95**: cliff repair not working, check majority-vote logic and `init_from_clean` copy storage.

**If `cliff_irrecoverable > 0` at p=0.001**: unexpected — REPORT-AND-STOP.

---

## Smoke C — E5 slope layer (ex_code CRC + EB-fallback)

```bash
python phase3_e5_recovery.py --adapter real --region ex_code \
    --pattern uniform_accum --ticks 20 --p 0.005 --smoke
```

**Expected output:**
```
  tick=  X  cliff_repaired=0  slope_failed=Y  eb_fraction=0.XXX  recall=0.9XX
```

**Verify:**
- `slope_failed > 0` by mid-run (ex_code chunks accumulate errors)
- `slope_reloaded` increments when `len(failed_chunks)/total_chunks > 0.10` (lazy reload fires)
- `_eb_path=True` in search result when slope corruption detected — **stub only**.

**EB-fallback on x86 — REPORT-AND-STOP (see §Design Gap below)**:
On the real adapter, the slope EB path now **raises** (`adapter.search_with_eb_fallback` cannot run
on the current binaries — see the gap section). So on x86 the ex/slope smoke will stop the moment a
chunk CRC fails. This is intended: it is not faked. Rotation/cliff smoke and Experiment A are
unaffected. Build Option B (`exp_dumpids --recovery fallback_eb`) before running Experiment B.

---

## Full Run — Experiment A (time-to-cliff)

Sweeps rotation corruption accumulation with E3c + E5 cliff guard. Measures: at what tick (and what
physical bit-error count) does recall drop from 0.983 to <0.80 WITHOUT the cliff guard, and how
many extra ticks does the cliff guard buy?

```bash
python phase3_e5_recovery.py --adapter real --region rotation \
    --pattern uniform_accum --ticks 100 --p 0.005
# → artifacts/phase3/e5/e5_rotation.json + e5_rotation.records.jsonl

# Also run without recovery (plain E3c) for baseline:
python phase3_e3c_temporal.py --adapter real --region rotation \
    --pattern uniform_accum --ticks 100 --p 0.005
# → artifacts/phase3/e3c/e3c_uniform_accum_rotation.json
```

Compare tick-of-collapse (recall < 0.80) with and without E5.

---

## Full Run — Experiment B (ex slope + EB recall curve)

```bash
for pattern in uniform_accum clustered_accum cross_row_accum burst_accum; do
    python phase3_e5_recovery.py --adapter real --region ex_code \
        --pattern $pattern --ticks 100 --p 0.005
done
```

Outputs: `artifacts/phase3/e5/e5_ex_code_<pattern>.records.jsonl`

Key measurements per tick:
- `recall` (E5 path, EB-fallback when slope fails)
- `slope_failed / slope_checked` (fraction of chunks corrupted)
- `eb_fraction` (fraction fed to EB policy)

Plot: x=tick, y=recall; one curve per pattern. Annotate `slope_reloaded` events.

---

## Design Gap — EB-fallback and pre-corrupted buffer

**The problem**: the real `exp_faultinject` binary cannot rank a caller-supplied corrupted
index. Its actual CLI is `<index> <query> <gt> [l2|ip] [seed]`: it loads ONE clean index, sweeps
ALL policies × fractions internally (`set_fault_injection` only marks a fraction of vectors
corrupt in an in-memory bitmap — it does NOT flip bytes), and prints a CSV. It takes no
policy/fraction/out_path/k/ef argument and emits no `RECALL` line. And EB ranking needs query-time
quantities (`est_dist`, `g_error`) computed inside the C++ search, which do not exist in Python —
so `qp.rabitq.eb_policy.eb_rank_dist` is only a documented formula reference, not a runnable Python
recovery path.

**Current behaviour — REPORT-AND-STOP (honest)**: `adapter.search_with_eb_fallback(...)` now
**raises** rather than faking it. (The earlier "Option A" wrapper was coded against a hypothetical
patched binary and would have double-corrupted: it fed the already-corrupted index to
`exp_faultinject`, which then injected MORE faults.) On x86, rotation/cliff experiments are
unaffected (they never call the EB path); ex/slope experiments stop here until Option B is built.
The stub adapter keeps a working `search_with_eb_fallback` (tagging `_eb_path=True`) so dev-machine
branch-coverage tests still exercise the slope EB branch.

**Option B (the path forward — requires C++ build)**:
Patch `exp_dumpids` to add a `--recovery fallback_eb` mode so E5 can serialize the pre-corrupted
buffer to a temp file and run the EB policy directly on it. This is a workstation build patch and
is the human decision recorded in `artifacts/phase3/plan/stage2_plan1.md` when Experiment B runs.

---

## Counter Schema (from `e5.counters()`)

| Key | Meaning |
|-----|---------|
| `cliff_checked` | rotation bits checked (64B × 8 per search call) |
| `cliff_repaired` | bits corrected across copies + buf by majority-vote |
| `cliff_irrecoverable` | searches where majority-vote result failed clean CRC |
| `slope_checked` | ex_code chunks CRC-checked |
| `slope_failed` | chunks that failed CRC this run |
| `slope_reloaded` | chunks batch-reloaded (lazy scrub) |
| `known_corrupted` | currently tracked failed chunk count |
| `eb_fraction` | `known_corrupted / total_chunks` (fraction fed to EB policy) |
| `oob_elements` | OOB pointer values detected by bounds-check pass |

---

## When Done

Report in `artifacts/phase3/plan/stage2_plan1.md`:
- Exp A: tick-to-cliff with/without E5 (write numbers, not "to be determined")
- Exp B: EB-fraction vs recall curve per pattern
- Which EB gap option was used and what was the recall delta

Then: proceed to Stage 2 Round 2 (cross-index comparison: RaBitQ vs IVF_SQ8 slope sensitivity).
