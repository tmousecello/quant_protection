#!/usr/bin/env bash
# Build Samuel's RaBitQ C++ toolchain and a clean b=7 SIFT index for Phase 3 Stage 0.
#
# Idempotent: every step skips if its output already exists. Steps:
#   1. clone RaBitQ-Library into <rabitq_repo>/third_party/RaBitQ-Library
#   2. git apply rabitq_instrumentation/library-changes.patch
#   3. copy exp_faultinject.cpp / exp_fieldflip.cpp into sample/
#   4. cmake -DCMAKE_BUILD_TYPE=Release && make -j   (binaries -> third_party/.../bin)
#   5. prepare SIFT (base/query/gt/centroids/clusterids) via prepare_rabitq.py
#   6. build the b=7 index (M=16, efConstruction=200)
#   7. clean baseline sanity: query the b=7 index, print recall@10 by ef (expect ~0.983)
#
# All measurement infra lives in quant_protection (qp/); this only stands up the RaBitQ
# substrate the Stage 0 adapter shells out to. Heavy step (network clone + native build +
# index build); the rest of Stage 0 is smoke-only.
set -uo pipefail

QP_ROOT="${QP_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
RQ_ROOT="${RQ_ROOT:-$(cd "$QP_ROOT/.." && pwd)/StorageSystemProject_RaBitQ_Free-recovery}"
PY="${BUILD_PY:-$QP_ROOT/.venv/bin/python}"
RABITQ_LIB_URL="${RABITQ_LIB_URL:-https://github.com/VectorDB-NTU/RaBitQ-Library.git}"
# Pin to the exact upstream commit Samuel's library-changes.patch was authored against
# (verified by matching the patch pre-image blobs: hnsw.hpp=80cd573, sample/CMakeLists.txt=bc88b1d).
# Upstream HEAD has since moved samples to sample/cpp/, which breaks the patch — do not use HEAD.
RABITQ_LIB_COMMIT="${RABITQ_LIB_COMMIT:-7e39df2fcff6161af5c4a77cc3dd1709684fb52b}"

LIB="$RQ_ROOT/third_party/RaBitQ-Library"
BIN="$LIB/bin"
INSTR="$RQ_ROOT/rabitq_instrumentation"
PREP="$RQ_ROOT/data/sift/prepared"
IDX_DIR="$RQ_ROOT/results/datasets/sift/idx"
IDXF="$IDX_DIR/hnsw_M16_efC200_b7.index"
SIFT_DIR="${SIFT_DIR:-$QP_ROOT/sift}"
METRIC="l2"

say()  { echo "=== [build_rabitq] $* ==="; }
fail() { echo "!!! [build_rabitq] FAIL: $*" >&2; exit 1; }

[ -d "$RQ_ROOT" ]   || fail "RaBitQ repo not found at $RQ_ROOT"
[ -x "$PY" ]        || fail "python not found at $PY (set BUILD_PY=...)"
[ -d "$SIFT_DIR" ]  || fail "SIFT dir not found at $SIFT_DIR (set SIFT_DIR=...)"
command -v cmake >/dev/null || fail "cmake not installed"
command -v git   >/dev/null || fail "git not installed"

# 1. CLONE + PIN ----------------------------------------------------------
if [ ! -d "$LIB/.git" ]; then
  say "[1/7] cloning RaBitQ-Library -> $LIB"
  mkdir -p "$RQ_ROOT/third_party"
  git clone "$RABITQ_LIB_URL" "$LIB" || fail "clone failed (network?)"
else
  say "[1/7] clone: cached"
fi
say "[1/7] pinning upstream to $RABITQ_LIB_COMMIT"
( cd "$LIB" && git checkout -q "$RABITQ_LIB_COMMIT" ) \
  || fail "checkout of pinned commit failed (fetch may be shallow; run: git -C $LIB fetch --unshallow)"
[ -f "$LIB/sample/CMakeLists.txt" ] || fail "sample/CMakeLists.txt missing at pinned commit (layout mismatch)"

# 2. PATCH ----------------------------------------------------------------
# Detect application by a sentinel symbol the patch adds; only apply if absent.
if ! grep -q "set_fault_injection" "$LIB/include/rabitqlib/index/hnsw/hnsw.hpp" 2>/dev/null; then
  say "[2/7] applying library-changes.patch"
  ( cd "$LIB" && git apply "$INSTR/library-changes.patch" ) || fail "git apply failed"
else
  say "[2/7] patch: already applied"
fi

# 3. COPY EXPERIMENT SOURCES ---------------------------------------------
say "[3/7] copying experiment sources into sample/"
cp "$INSTR/exp_faultinject.cpp" "$INSTR/exp_fieldflip.cpp" "$LIB/sample/" || fail "copy exp_*.cpp failed"

# 3b. macOS/AppleClang OpenMP flag fix (build-portability only; no logic change) ----
# Upstream CMAKE_CXX_FLAGS hard-codes "-fopenmp -lrt" via SET(...), which clobbers any -D
# override and which Apple's stock clang rejects (no bundled OpenMP; -lrt is Linux-only).
# Rewrite ONLY the flags line to the libomp recipe: -Xclang -fopenmp + libomp include/lib.
# Re-applied every run because the step-1 checkout restores the pristine file first.
if [ "$(uname)" = "Darwin" ]; then
  LIBOMP="$(brew --prefix libomp 2>/dev/null)"
  [ -d "$LIBOMP" ] || fail "libomp not found (brew install libomp)"
  say "[3/7] patching CMAKE_CXX_FLAGS for AppleClang+libomp ($LIBOMP)"
  NEWFLAGS="-Wall -Ofast -Wextra -march=native -fpic -Xclang -fopenmp -I$LIBOMP/include -L$LIBOMP/lib -lomp -ftree-vectorize -fexceptions"
  /usr/bin/sed -i '' -E "s|^SET\\(CMAKE_CXX_FLAGS.*|SET(CMAKE_CXX_FLAGS  \"$NEWFLAGS\")|" "$LIB/CMakeLists.txt"
  grep -q 'Xclang' "$LIB/CMakeLists.txt" || fail "CMAKE_CXX_FLAGS flag-patch did not apply"
fi

# 3c. ARCHITECTURE GATE ---------------------------------------------------
# RaBitQ-Library's core (utils/space.hpp, quantization/rabitq_impl.hpp, index/ivf|hnsw)
# unconditionally includes <emmintrin.h>/<immintrin.h> and uses native AVX2/AVX512
# instructions with NO aarch64/NEON path. It does not compile on Apple Silicon (arm64).
# Steps 1-3 (clone/pin/patch — useful for reading the byte layout from source) already ran;
# the build + live baseline require an x86-64 host (where Samuel's results were produced).
# Override with FORCE_BUILD=1 only if you have ported the SIMD.
ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ] && [ "${FORCE_BUILD:-0}" != "1" ]; then
  cat >&2 <<EOF
!!! [build_rabitq] REPORT-AND-STOP: cannot build RaBitQ-Library on '$ARCH'.
    The upstream library is x86-64 only (AVX2/AVX512 intrinsics, no NEON path).
    Steps 1-3 succeeded: source cloned + pinned to $RABITQ_LIB_COMMIT and the
    instrumentation patch applies cleanly, so the byte layout is readable from source at:
      $LIB/include/rabitqlib/index/hnsw/hnsw.hpp
    To produce the live b=7 baseline (~0.983) and parity numbers, run this same script
    UNCHANGED on an x86-64 Linux host (cmake + libomp present). It is idempotent.
EOF
  exit 2
fi

# 4. BUILD ----------------------------------------------------------------
if [ ! -x "$BIN/hnsw_rabitq_querying" ] || [ ! -x "$BIN/exp_faultinject" ]; then
  say "[4/7] cmake + make (native build; this takes a few minutes)"
  ( cd "$LIB" && rm -rf build && mkdir build && cd build \
      && cmake .. -DCMAKE_BUILD_TYPE=Release \
      && make -j ) || fail "cmake/make failed (see output; -march=native may be unsupported on this CPU)"
else
  say "[4/7] build: cached"
fi
for b in hnsw_rabitq_indexing hnsw_rabitq_querying exp_faultinject exp_fieldflip; do
  [ -x "$BIN/$b" ] || fail "expected binary $BIN/$b not produced"
done

# 5. PREPARE SIFT ---------------------------------------------------------
if [ ! -f "$PREP/base.fvecs" ] || [ ! -f "$PREP/centroids_16.fvecs" ]; then
  say "[5/7] preparing SIFT (KMeans K=16) -> $PREP"
  mkdir -p "$PREP"
  "$PY" "$RQ_ROOT/scripts/prepare_rabitq.py" \
      --data "$SIFT_DIR" --out-dir "$PREP" --metric "$METRIC" --M 16 || fail "prepare_rabitq failed"
else
  say "[5/7] prepare: cached"
fi

# 6. BUILD b=7 INDEX ------------------------------------------------------
if [ ! -f "$IDXF" ]; then
  say "[6/7] building b=7 index (M=16, efC=200) -> $IDXF"
  mkdir -p "$IDX_DIR"
  "$BIN/hnsw_rabitq_indexing" "$PREP/base.fvecs" "$PREP/centroids_16.fvecs" \
      "$PREP/clusterids_16.ivecs" 16 200 7 "$IDXF" "$METRIC" || fail "indexing failed"
else
  say "[6/7] index: cached"
fi

# 7. CLEAN BASELINE SANITY ------------------------------------------------
say "[7/7] clean baseline: recall@10 by ef (expect plateau ~0.983)"
"$BIN/hnsw_rabitq_querying" "$IDXF" "$PREP/query.fvecs" "$PREP/groundtruth.ivecs" "$METRIC" \
    | awk -F'\t' '/^[0-9]/{print "    ef="$1"  qps="$2"  recall@10="$3}'

say "DONE. binaries: $BIN   index: $IDXF"
