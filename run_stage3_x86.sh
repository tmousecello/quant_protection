#!/usr/bin/env bash
# Phase 3 Stage 3 — formal experiment runner (the ONLY place formal runs are launched).
#
# Implementation was validated with pytest + --smoke only (artifacts_smoke/); this script
# produces the formal artifacts under artifacts/phase3/ after the implementation report.
#
#   A  self-scrub 200-tick timeline (main figure 4th line)            ~52 min
#      e5 --inject-replicas --cliff-scrub --ticks 200 + sanity gate
#   B  30-seed fuse first-failure distribution                        ~15 min + 2x27 min
#      seed batch launch -> analyze -> determinism gate -> min/max timelines
#   C  F validation + synthesis                                       ~20 min
#      expb fstar_check sweep (4 patterns x f=0.101 x {eb,drop}) ->
#      f_synth --run-cliff-check -> --synthesize -> --check (|Δ|<=0.01 gate, exit!=0 stops)
#   D  G detection analysis (offline, seconds)
#
# Phases run SEQUENTIALLY (the C++ search saturates the CPU; expb must never run twice
# concurrently — shared scratch). Any gate failure aborts the script: report-and-stop,
# never hand-edit artifacts.
#
# Usage:
#   bash run_stage3_x86.sh            # all phases A -> B -> C -> D
#   bash run_stage3_x86.sh A          # a single phase (A|B|C|D)
#   bash run_stage3_x86.sh --dry-run  # print the exact commands without executing
set -euo pipefail

QP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$QP_ROOT/.venv/bin/python"
LOG_DIR="$QP_ROOT/artifacts/phase3/logs"
TS="$(date +%Y%m%d_%H%M%S)"

DRY=0
PHASES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    A|B|C|D) PHASES+=("$arg") ;;
    *) echo "usage: bash run_stage3_x86.sh [--dry-run] [A|B|C|D ...]" >&2; exit 2 ;;
  esac
done
[ ${#PHASES[@]} -eq 0 ] && PHASES=(A B C D)

run() {
  # run <logname> <cmd...> — echo, then execute teeing to the phase log
  local logname="$1"; shift
  echo "+ $*"
  if [ "$DRY" -eq 1 ]; then return 0; fi
  mkdir -p "$LOG_DIR"
  "$@" 2>&1 | tee -a "$LOG_DIR/stage3_${logname}_${TS}.log"
  return "${PIPESTATUS[0]}"
}

sanity() {
  # sanity <logname> <python -c body>
  local logname="$1"; shift
  echo "+ [sanity] python -c '...'"
  if [ "$DRY" -eq 1 ]; then return 0; fi
  "$PY" -c "$1" 2>&1 | tee -a "$LOG_DIR/stage3_${logname}_${TS}.log"
  return "${PIPESTATUS[0]}"
}

phase_A() {
  echo "=== Phase A: self-scrub 200-tick timeline (~52 min) ==="
  run A "$PY" "$QP_ROOT/phase3_e5_recovery.py" --adapter real --region rotation \
      --pattern uniform_accum --inject-replicas --cliff-scrub --ticks 200
  sanity A "
import json
rows = [json.loads(l) for l in open('$QP_ROOT/artifacts/phase3/e5/e5_uniform_accum_rotation_replicas_scrub.records.jsonl')]
assert len(rows) == 200, f'expected 200 ticks, got {len(rows)}'
recalls = [r['recall@10'] for r in rows]
final = rows[-1]['counters']
assert all(abs(x - 0.98376) < 1e-9 for x in recalls), f'recall not flat: min={min(recalls)}'
assert final['cliff_reload_triggered'] > 0, 'no vote failure occurred in 200 ticks (implausible; check lanes)'
assert final['cliff_irrecoverable'] == 0, f'irrecoverable={final[\"cliff_irrecoverable\"]}'
print(f'[sanity A] OK: recall flat 0.98376 x200, reloads={final[\"cliff_reload_triggered\"]}, '
      f'anchor_checked={final[\"cliff_anchor_checked\"]}, anchor_mismatch={final[\"cliff_anchor_mismatch\"]}')
"
}

phase_B() {
  echo "=== Phase B: 30-seed fuse first-failure distribution (~15 min + 2x27 min) ==="
  run B "$PY" "$QP_ROOT/phase3_e5_seed_batch.py" --launch --analyze --determinism
  run B "$PY" "$QP_ROOT/phase3_e5_seed_batch.py" --timelines
  sanity B "
import json
s = json.load(open('$QP_ROOT/artifacts/phase3/e5_seeds/first_fail_summary.json'))
st = s['stats']
assert st['n_seeds'] == 30, st
assert st['n_censored'] <= 2, f'unexpectedly many censored seeds: {st[\"n_censored\"]}'
print(f'[sanity B] OK: {st[\"n_observed\"]}/30 observed, median={st[\"median\"]}, '
      f'IQR=[{st[\"q1\"]},{st[\"q3\"]}], min={st[\"min\"]}, max={st[\"max\"]}, '
      f'analytic median~{s[\"analytic\"][\"geometric_median\"]:.1f}')
"
}

phase_C() {
  echo "=== Phase C: F validation sweep + synthesis + gates (~20 min) ==="
  # (a)+(b): 4 patterns x f=0.101 x {fallback_eb, drop}, LIGHT severity (default flips=4)
  run C "$PY" "$QP_ROOT/phase3_expb_recovery.py" --sweep --adapter real \
      --patterns uniform_accum,clustered_accum,cross_row_accum,burst_accum \
      --fractions 0.101 --recovery fallback_eb,drop --out-tag fstar_check
  # (c): deterministic run-2 tick-0 reproduction + k=8 secondary
  run C "$PY" "$QP_ROOT/phase3_f_synth.py" --run-cliff-check --adapter real
  # synthesis (needs Phase A's scrub timeline) + the |Δ|<=0.01 gates
  run C "$PY" "$QP_ROOT/phase3_f_synth.py" --synthesize --adapter real
  run C "$PY" "$QP_ROOT/phase3_f_synth.py" --check
}

phase_D() {
  echo "=== Phase D: G detection analysis (offline) ==="
  run D "$PY" "$QP_ROOT/phase3_g_detection.py"
  sanity D "
import json
s = json.load(open('$QP_ROOT/artifacts/phase3/g_detection/g_detection_summary.json'))
d = s['deviation']
assert d['n_points'] > 0
assert d['identical_to_truth'], f'load-scan ratio deviates from truth: {d}'
print(f'[sanity D] OK: {d[\"n_points\"]} points, load-scan == truth exactly, '
      f'severities={s[\"severities_found\"]}')
"
}

echo "[stage3] phases: ${PHASES[*]}  dry_run=$DRY  log_dir=$LOG_DIR"
if [ "$DRY" -eq 0 ] && [ ! -x "$PY" ]; then
  echo "REPORT-AND-STOP: $PY missing — run setup.sh first" >&2; exit 1
fi
for ph in "${PHASES[@]}"; do
  "phase_$ph"
done
echo "[stage3] requested phases complete. Artifacts under artifacts/phase3/{e5,e5_seeds,expb,f_synth,g_detection}."
