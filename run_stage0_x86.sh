#!/usr/bin/env bash
# Phase 3 Stage 0 — one-shot x86-64 Linux workstation build + full acceptance.
#
# After `git clone`/`git pull` of quant_protection (branch phase3-stage0), this single script
# stands up the whole environment from scratch and runs EVERY Stage 0 acceptance item:
#   0 preflight (x86-64 + toolchain)   1 locate repos
#   2 auto-clone the sibling RaBitQ repo if missing
#   3 fetch SIFT1M                      4 venv + deps + editable qp
#   5 build RaBitQ C++ (incl exp_dumpids) + b=7 index
#   6 acceptance: pytest suite (parity now RUNS) + live clean baseline ~0.983
#   7 summary
# Idempotent: every underlying step skips work whose output already exists.
set -euo pipefail

QP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RQ_ROOT="${RQ_ROOT:-$(cd "$QP_ROOT/.." && pwd)/StorageSystemProject_RaBitQ_Free-recovery}"
RQ_REMOTE="${RQ_REMOTE:-https://github.com/ZMYsamuel/StorageSystemProject_RaBitQ_Free-recovery.git}"
PY="$QP_ROOT/.venv/bin/python"

banner() { echo; echo "########## [stage0-x86] $* ##########"; }
die()    { echo "!!! [stage0-x86] $*" >&2; exit 1; }

# 0. PREFLIGHT ------------------------------------------------------------
banner "0/7 preflight"
[ "$(uname -m)" = "x86_64" ] || die "this script is for x86-64 (got $(uname -m)). RaBitQ-Library is AVX2/AVX512-only; on arm64 the byte layout is readable but the binaries cannot build."
[ "$(uname -s)" = "Linux" ] || echo "warn: not Linux ($(uname -s)); build_rabitq.sh has an x86-64 macOS path but Linux is the intended host."
for t in git cmake make python3; do command -v "$t" >/dev/null || die "missing required tool: $t"; done
command -v g++ >/dev/null || command -v c++ >/dev/null || die "missing a C++ compiler (g++/c++)"
echo "ok: $(uname -m) $(uname -s); $(cmake --version | head -1); $(python3 --version)"

# 1. LOCATE ---------------------------------------------------------------
banner "1/7 locate repos"
echo "QP_ROOT=$QP_ROOT"
echo "RQ_ROOT=$RQ_ROOT"

# 2. SIBLING RABITQ REPO --------------------------------------------------
banner "2/7 sibling RaBitQ repo"
if [ ! -d "$RQ_ROOT/.git" ]; then
  echo "cloning $RQ_REMOTE -> $RQ_ROOT"
  git clone "$RQ_REMOTE" "$RQ_ROOT" || die "clone of sibling RaBitQ repo failed (network? RQ_REMOTE override?)"
else
  echo "present: $RQ_ROOT"
fi
[ -f "$RQ_ROOT/rabitq_instrumentation/library-changes.patch" ] || die "sibling repo missing instrumentation patch — wrong repo at $RQ_ROOT?"

# 3. SIFT1M ---------------------------------------------------------------
banner "3/7 fetch SIFT1M"
bash "$QP_ROOT/fetch_sift.sh"

# 4. ENV: venv + deps + editable qp --------------------------------------
banner "4/7 environment (setup.sh + editable qp)"
bash "$QP_ROOT/setup.sh"
[ -x "$PY" ] || die "setup.sh did not produce $PY"
"$PY" -m pip install -e "$QP_ROOT" -q
"$PY" -c "import qp.faults, qp.metrics, qp.rabitq.adapter; print('editable qp OK')"

# 5. BUILD RABITQ + b=7 INDEX --------------------------------------------
banner "5/7 build RaBitQ (incl exp_dumpids) + b=7 index"
bash "$QP_ROOT/build_rabitq.sh"

# 6. ACCEPTANCE -----------------------------------------------------------
banner "6/7 acceptance"
acc_fail=0
set +e
echo "--- pytest (full Stage 0 suite; parity now runs) ---"
"$PY" -m pytest "$QP_ROOT/artifacts/phase3/tests/" -q
pytest_rc=$?
[ $pytest_rc -eq 0 ] || acc_fail=1

echo "--- live clean baseline (b=7 recall@10 ~ 0.983) ---"
"$PY" - <<'PYEOF'
import sys
from qp.rabitq import adapter
r = adapter.clean_baseline_recall()
print(f"clean b=7 recall@10 = {r:.5f}  (expected ~{adapter.EXPECTED_CLEAN_RECALL10})")
sys.exit(0 if abs(r - adapter.EXPECTED_CLEAN_RECALL10) < 0.01 else 1)
PYEOF
baseline_rc=$?
[ $baseline_rc -eq 0 ] || acc_fail=1
set -e

# 7. SUMMARY --------------------------------------------------------------
banner "7/7 summary"
ok() { [ "$1" -eq 0 ] && echo "PASS" || echo "FAIL"; }
echo "  pytest suite ............. $(ok $pytest_rc)"
echo "  clean baseline ~0.983 .... $(ok $baseline_rc)"
echo "  binaries: $RQ_ROOT/third_party/RaBitQ-Library/bin"
echo "  b=7 index: $RQ_ROOT/results/datasets/sift/idx/hnsw_M16_efC200_b7.index"
if [ "$acc_fail" -ne 0 ]; then
  die "Stage 0 acceptance FAILED (see above)."
fi
echo "=== Stage 0 acceptance PASSED — all items green. ==="
