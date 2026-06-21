#!/usr/bin/env bash
# Phase 3 Stage 1 — one-shot x86-64 Linux workstation build + FULL acceptance.
#
# After `git clone` / `git pull` of quant_protection (branch phase3-stage1), this single script
# stands up the whole environment from scratch and runs EVERY acceptance item — engineering
# (the stub-green pipeline, which on x86 also runs the parity keystone) AND scientific (the real
# RaBitQ E1 vuln map, E3a additivity, E3b spatial — the numbers that can only be produced here):
#
#   0 preflight (x86-64 + toolchain)        1 locate repos
#   2 auto-clone the sibling RaBitQ repo     3 fetch SIFT1M
#   4 venv + deps + editable qp              5 build RaBitQ C++ (incl exp_dumpids) + b=7 index
#   6 adapter ready + live clean baseline ~0.983
#   7 engineering acceptance: full pytest suite (parity NOW runs on real binaries)
#   8 first shot: single-bit rotation flip on the REAL index — region-map alignment + E1 datapoint
#   9 full scientific runs: E1 vuln map + E3a + E3b (--adapter real)   [skipped under --smoke]
#  10 summary
#
# Idempotent: every underlying step skips work whose output already exists; the E1 --resume
# continues a partial sweep (E3a/E3b are small single-shot). The runners need NO edits on the
# workstation — --adapter real is passed explicitly, with --allow-no-distances (stock exp_dumpids
# has no distance dump) and, for the smoke first shot, --clean-tol 0.05 (ef=64 lowers clean recall).
#
# Usage:
#   bash run_stage1_x86.sh            # full: build + pytest + first-shot + E1/E3a/E3b real runs
#   bash run_stage1_x86.sh --smoke    # fast: build + pytest + first-shot only (stop before §9)
#   bash run_stage1_x86.sh --resume   # full, but resume a partially-completed E1 sweep
set -euo pipefail

QP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RQ_ROOT="${RQ_ROOT:-$(cd "$QP_ROOT/.." && pwd)/StorageSystemProject_RaBitQ_Free-recovery}"
RQ_REMOTE="${RQ_REMOTE:-https://github.com/ZMYsamuel/StorageSystemProject_RaBitQ_Free-recovery.git}"
PY="$QP_ROOT/.venv/bin/python"
# First-shot is an ALIGNMENT check, not a severity verdict. rotation is heavy-tailed (most bits
# perturb recall slightly; a few are catastrophic), so we gate on "rotation flips demonstrably move
# recall" (max ΔRecall@10 over rotation rows) and merely REPORT the severity (collapse fraction,
# max/p99 ΔRecall). An inert rotation (signal ~0) means the region map is offset against this build.
ROT_ALIGN_DRECALL_MIN="${ROT_ALIGN_DRECALL_MIN:-0.05}"

SMOKE=0
RESUME=""
for a in "$@"; do
  case "$a" in
    --smoke)  SMOKE=1 ;;
    --resume) RESUME="--resume" ;;
    *) echo "unknown arg: $a (accepted: --smoke, --resume)" >&2; exit 64 ;;
  esac
done

banner() { echo; echo "########## [stage1-x86] $* ##########"; }
die()    { echo "!!! [stage1-x86] $*" >&2; exit 1; }

# 0. PREFLIGHT ------------------------------------------------------------
banner "0/10 preflight"
[ "$(uname -m)" = "x86_64" ] || die "this script is for x86-64 (got $(uname -m)). RaBitQ-Library is AVX2/AVX512-only; on arm64 the byte layout is readable from source but the binaries cannot build and no real numbers can be produced."
[ "$(uname -s)" = "Linux" ] || echo "warn: not Linux ($(uname -s)); build_rabitq.sh has an x86-64 macOS path but Linux is the intended host."
for t in git cmake make python3; do command -v "$t" >/dev/null || die "missing required tool: $t"; done
command -v g++ >/dev/null || command -v c++ >/dev/null || die "missing a C++ compiler (g++/c++)"
echo "ok: $(uname -m) $(uname -s); $(cmake --version | head -1); $(python3 --version)"
[ "$SMOKE" -eq 1 ] && echo "mode: --smoke (build + pytest + first-shot only)" || echo "mode: full (incl §9 real E1/E3a/E3b)"

# 1. LOCATE ---------------------------------------------------------------
banner "1/10 locate repos"
echo "QP_ROOT=$QP_ROOT"
echo "RQ_ROOT=$RQ_ROOT"

# 2. SIBLING RABITQ REPO --------------------------------------------------
banner "2/10 sibling RaBitQ repo"
if [ ! -d "$RQ_ROOT/.git" ]; then
  echo "cloning $RQ_REMOTE -> $RQ_ROOT"
  git clone "$RQ_REMOTE" "$RQ_ROOT" || die "clone of sibling RaBitQ repo failed (network? private? set RQ_REMOTE= or RQ_ROOT= to an existing checkout)"
else
  echo "present: $RQ_ROOT"
fi
[ -f "$RQ_ROOT/rabitq_instrumentation/library-changes.patch" ] || die "sibling repo missing instrumentation patch — wrong repo at $RQ_ROOT?"

# 3. SIFT1M ---------------------------------------------------------------
banner "3/10 fetch SIFT1M"
bash "$QP_ROOT/fetch_sift.sh"

# 4. ENV: venv + deps + editable qp --------------------------------------
banner "4/10 environment (setup.sh + editable qp)"
bash "$QP_ROOT/setup.sh"
[ -x "$PY" ] || die "setup.sh did not produce $PY"
"$PY" -m pip install -e "$QP_ROOT" -q
"$PY" -c "import qp.faults, qp.metrics, qp.rabitq.adapter; print('editable qp OK')"

# 5. BUILD RABITQ + b=7 INDEX --------------------------------------------
banner "5/10 build RaBitQ (incl exp_dumpids) + b=7 index"
bash "$QP_ROOT/build_rabitq.sh"

# 6. ADAPTER READY + LIVE CLEAN BASELINE ---------------------------------
banner "6/10 adapter ready + live clean baseline (~0.983)"
set +e
"$PY" - <<'PYEOF'
import sys
from qp.rabitq import adapter
if not adapter.binaries_built():
    print("binaries_built() = False — build step 5 did not produce the RaBitQ binaries"); sys.exit(1)
r = adapter.clean_baseline_recall()
print(f"binaries_built() = True; clean b=7 recall@10 = {r:.5f}  (expected ~{adapter.EXPECTED_CLEAN_RECALL10})")
sys.exit(0 if abs(r - adapter.EXPECTED_CLEAN_RECALL10) < 0.01 else 1)
PYEOF
baseline_rc=$?
set -e
[ $baseline_rc -eq 0 ] || die "adapter not ready or baseline off (~0.983 anchors the whole study) — stop and investigate before trusting any corruption number."

# 7. ENGINEERING ACCEPTANCE: pytest (parity now runs) --------------------
banner "7/10 engineering acceptance: full pytest suite (parity keystone now runs on real binaries)"
set +e
"$PY" -m pytest "$QP_ROOT/artifacts/phase3/tests/" -q
pytest_rc=$?
set -e

# 8. FIRST SHOT: real rotation flips (alignment check) -------------------
# Confirms the region map is aligned to THIS build (rotation bytes are consumed by the search) and
# is E1's first datapoint. Two flags make it run unmodified on the stock workstation:
#   --clean-tol 0.05      smoke ef=64 -> clean@10 ~0.95 vs the 0.983 plateau anchor (real ΔRecall
#                         magnitudes come from §9 at ef=2000, not this fast pass).
#   --allow-no-distances  stock exp_dumpids emits ids+RECALL only (no per-query distances), so
#                         nan-inf is undetectable and honestly flagged 'not-detected' rather than
#                         report-and-stopping. (Optional exp_dumpids extension: E1_RUNBOOK §2.)
banner "8/10 first shot: rotation flips on the real index (alignment check)"
set +e
"$PY" "$QP_ROOT/phase3_e1_vuln.py" --adapter real --smoke --clean-tol 0.05 --allow-no-distances
firstshot_rc=$?
set -e
rot_report=""
if [ $firstshot_rc -eq 0 ]; then
  set +e
  rot_report="$("$PY" - "$QP_ROOT" "$ROT_ALIGN_DRECALL_MIN" <<'PYEOF'
import json, os, sys
qp_root, thr = sys.argv[1], float(sys.argv[2])
vm = os.path.join(qp_root, "artifacts_smoke", "phase3", "e1", "vuln_map.json")
rows = json.load(open(vm)).get("rows", [])
rot = [r for r in rows if r.get("region") == "rotation"]
if not rot:
    print("NO_ROTATION_ROWS"); sys.exit(3)
def col(name):  # max over rotation bit-class rows, treating None as 0
    return max((r.get(name) or 0.0) for r in rot)
align = col("max_dRecall@10")                    # alignment signal: rotation perturbs recall
pct_collapse_mean = sum((r.get("pct_collapse") or 0.0) for r in rot) / len(rot)
n_silent = sum((r.get("n_silent_wrong") or 0) for r in rot)
# One-line severity REPORT (not a gate): heavy-tailed -> max/p99 carry the danger, not the mean.
print(f"align(maxΔR@10)={align:.4f} | collapse_pct(mean)={pct_collapse_mean:.2f} "
      f"maxΔR@10={col('max_dRecall@10'):.4f} p99ΔR@10={col('p99_dRecall@10'):.4f} "
      f"n_silent_wrong={n_silent}")
sys.exit(0 if align >= thr else 7)             # stop ONLY if rotation is inert (likely map offset)
PYEOF
)"
  rot_rc=$?
  set -e
else
  rot_rc=$firstshot_rc
fi
if [ "${rot_rc:-1}" -eq 0 ]; then
  echo "region map ALIGNED — rotation flips move recall (signal >= ${ROT_ALIGN_DRECALL_MIN})."
  echo "  severity: ${rot_report}"
  echo "  (rotation is heavy-tailed: rare catastrophic bits, not a uniform collapse — see §9 at ef=2000.)"
elif [ "${rot_rc:-1}" -eq 7 ]; then
  cat >&2 <<EOF
!!! [stage1-x86] REPORT-AND-STOP: rotation is INERT (${rot_report}; align signal < ${ROT_ALIGN_DRECALL_MIN}).
    Flipping rotation bytes barely moves recall, so the region map is likely OFFSET against this
    build (or rotation is genuinely not consumed). Per E1_RUNBOOK §1 this is a FINDING — do not
    proceed to §9 blindly. Inspect:
      $QP_ROOT/artifacts_smoke/phase3/e1/vuln_map.json   (rotation rows)
EOF
  exit 7
elif [ "${rot_rc:-1}" -eq 3 ]; then
  die "first-shot produced NO rotation rows in vuln_map.json — runner output schema changed or rotation region absent; inspect $QP_ROOT/artifacts_smoke/phase3/e1/vuln_map.json."
else
  die "first-shot rotation smoke failed to run (rc=$firstshot_rc). See output above."
fi

# 9. FULL SCIENTIFIC RUNS (skipped under --smoke) ------------------------
e1_rc=0; e3a_rc=0; e3b_rc=0
if [ "$SMOKE" -eq 1 ]; then
  banner "9/10 full scientific runs — SKIPPED (--smoke)"
  echo "re-run without --smoke on this same host to produce the real E1/E3a/E3b numbers."
else
  banner "9/10 full scientific runs: E1 vuln map + E3a + E3b (--adapter real)"
  mkdir -p "$QP_ROOT/artifacts/phase3/e1" "$QP_ROOT/artifacts/phase3/e3a" "$QP_ROOT/artifacts/phase3/e3b"
  # --allow-no-distances: stock exp_dumpids has no distance dump (nan-inf undetectable, honestly
  # flagged). §9 runs at ef=2000 so clean ~0.983 sits on the anchor — no --clean-tol needed here.
  # --resume is E1-only (E3a/E3b are small single-shot experiments with no resume support).
  set +e
  echo "--- E1: full vuln map (rotation exhaustive + per-vector sampled w/ CI) ---"
  "$PY" "$QP_ROOT/phase3_e1_vuln.py" --adapter real --allow-no-distances $RESUME 2>&1 | tee "$QP_ROOT/artifacts/phase3/e1/run.log"
  e1_rc=${PIPESTATUS[0]}
  echo "--- E3a: multi-bit additivity ---"
  "$PY" "$QP_ROOT/phase3_e3a_additivity.py" --adapter real --allow-no-distances 2>&1 | tee "$QP_ROOT/artifacts/phase3/e3a/run.log"
  e3a_rc=${PIPESTATUS[0]}
  echo "--- E3b: spatial (W,k) clustering vs uniform ---"
  "$PY" "$QP_ROOT/phase3_e3b_spatial.py" --adapter real --allow-no-distances 2>&1 | tee "$QP_ROOT/artifacts/phase3/e3b/run.log"
  e3b_rc=${PIPESTATUS[0]}
  set -e
fi

# 10. SUMMARY -------------------------------------------------------------
banner "10/10 summary"
ok() { [ "${1:-1}" -eq 0 ] && echo "PASS" || echo "FAIL"; }
have() { [ -f "$1" ] && echo "ok ($1)" || echo "MISSING ($1)"; }
echo "  clean baseline ~0.983 ......... $(ok $baseline_rc)"
echo "  pytest suite (incl parity) .... $(ok $pytest_rc)"
echo "  first-shot alignment gate ..... $(ok ${rot_rc:-1})  [${rot_report:-n/a}]"
acc_fail=0
[ $baseline_rc -eq 0 ] || acc_fail=1
[ $pytest_rc   -eq 0 ] || acc_fail=1
[ "${rot_rc:-1}" -eq 0 ] || acc_fail=1
if [ "$SMOKE" -eq 1 ]; then
  echo "  E1/E3a/E3b real runs .......... SKIPPED (--smoke)"
  [ "$acc_fail" -ne 0 ] && die "Stage 1 smoke acceptance FAILED (see above)."
  echo "=== Stage 1 SMOKE acceptance PASSED — pipeline verified on real binaries. Re-run without --smoke for the scientific numbers. ==="
else
  echo "  E1 full run ................... $(ok $e1_rc)   outputs: $(have "$QP_ROOT/artifacts/phase3/e1/vuln_map.json")"
  echo "                                          $(have "$QP_ROOT/artifacts/phase3/e1/criticality.json")"
  echo "  E3a full run .................. $(ok $e3a_rc)   outputs: $(have "$QP_ROOT/artifacts/phase3/e3a/e3a.json")"
  echo "  E3b full run .................. $(ok $e3b_rc)   outputs: $(have "$QP_ROOT/artifacts/phase3/e3b/e3b.json")"
  [ $e1_rc  -eq 0 ] || acc_fail=1
  [ $e3a_rc -eq 0 ] || acc_fail=1
  [ $e3b_rc -eq 0 ] || acc_fail=1
  echo "  per-flip jsonl + run.log preserved under artifacts/phase3/{e1,e3a,e3b}/ for offline debugging."
  [ "$acc_fail" -ne 0 ] && die "Stage 1 acceptance FAILED (see above). Per-flip jsonl + run.log are enough to reproduce/debug offline against the stub."
  echo "=== Stage 1 acceptance PASSED — all items green. Copy artifacts/phase3/{e1,e3a,e3b}/ back to the dev machine for analysis + golden update. ==="
fi
