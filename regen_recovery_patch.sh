#!/usr/bin/env bash
# Regenerate rabitq_instrumentation/recovery-changes.patch from the edited working header.
#
# WHY THIS EXISTS
# ---------------
# recovery-changes.patch is applied by build_rabitq.sh on top of Samuel's library-changes.patch.
# Until now it was maintained by hand: ~330 lines, 5 hunks, @@ anchors, and no `index` line (so
# `git apply --3way` is unavailable and there is no fuzz). Hand-editing it means hand-editing
# hunk line counts, and a wrong count is a hard "corrupt patch" while a wrong context is a
# silent misapply. Neither is a good way to spend an afternoon.
#
# The pre-image the patch is written against — pristine-at-pin PLUS library-changes.patch, and
# nothing else — does not exist anywhere on disk. This script materialises it, diffs the edited
# header against it, and then VERIFIES THE ROUND TRIP before overwriting anything.
#
# WORKFLOW
#   1. edit  <RQ_ROOT>/third_party/RaBitQ-Library/include/rabitqlib/index/hnsw/hnsw.hpp in place
#   2. build and test there (cd build && make -j)
#   3. bash regen_recovery_patch.sh
#   4. if the patch added a new symbol, bump RECOVERY_SENTINEL in build_rabitq.sh to it, and if
#      it added a new exp_dumpids flag, bump the step-4 grep too. Forgetting either means the
#      next build silently keeps the OLD header. See build_rabitq.sh's comments.
#
# The script leaves the working header exactly as it found it, and never touches the git index
# (build_rabitq.sh's `git checkout -- hnsw.hpp` restores from the index, so a staged hnsw.hpp
# would break its stale-revert path).
set -uo pipefail

QP_ROOT="${QP_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
RQ_ROOT="${RQ_ROOT:-$(cd "$QP_ROOT/.." && pwd)/StorageSystemProject_RaBitQ_Free-recovery}"
LIB="$RQ_ROOT/third_party/RaBitQ-Library"
REL="include/rabitqlib/index/hnsw/hnsw.hpp"
LIBPATCH="$RQ_ROOT/rabitq_instrumentation/library-changes.patch"
OUT="$QP_ROOT/rabitq_instrumentation/recovery-changes.patch"

say()  { echo "=== [regen] $* ==="; }
fail() { echo "!!! [regen] FAIL: $*" >&2; exit 1; }

[ -f "$LIB/$REL" ]  || fail "no working header at $LIB/$REL (run build_rabitq.sh first)"
[ -f "$LIBPATCH" ]  || fail "no library-changes.patch at $LIBPATCH"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp "$LIB/$REL" "$TMP/edited.hpp" || fail "could not snapshot the edited header"

restore() { cp "$TMP/edited.hpp" "$LIB/$REL"; }

say "materialising the pre-image (pristine@pin + library-changes.patch)"
( cd "$LIB" && git checkout -- "$REL" ) || { restore; fail "could not revert to pristine"; }
( cd "$LIB" && git apply --include="$REL" "$LIBPATCH" ) \
  || { restore; fail "library-changes.patch did not apply to the pristine header"; }
cp "$LIB/$REL" "$TMP/baseline.hpp"
restore

say "diffing edited against the pre-image"
{
  echo "diff --git a/$REL b/$REL"
  diff -u --label "a/$REL" --label "b/$REL" "$TMP/baseline.hpp" "$TMP/edited.hpp"
} > "$TMP/new.patch"
if [ ! -s "$TMP/new.patch" ] || ! grep -q '^@@' "$TMP/new.patch"; then
  fail "generated patch has no hunks — is the working header actually modified?"
fi

# The only check that matters: pristine + library + new patch must reproduce the edited header
# byte for byte. Nothing in CI does this, so it happens here or not at all.
say "verifying round trip"
( cd "$LIB" && git checkout -- "$REL" ) || { restore; fail "revert failed during verify"; }
( cd "$LIB" && git apply --include="$REL" "$LIBPATCH" ) \
  || { restore; fail "library-changes.patch re-apply failed during verify"; }
( cd "$LIB" && git apply "$TMP/new.patch" ) \
  || { restore; fail "GENERATED PATCH DOES NOT APPLY — not written"; }
if ! diff -q "$LIB/$REL" "$TMP/edited.hpp" >/dev/null; then
  restore
  fail "ROUND TRIP MISMATCH — generated patch does not reproduce the edited header; not written"
fi

cp "$TMP/new.patch" "$OUT" || fail "could not write $OUT"
say "round trip OK -> $OUT ($(wc -l < "$OUT") lines, $(grep -c '^@@' "$OUT") hunks)"

# Nudge about the trap that cost this project weeks of uncompiled code.
SENT="$(grep -o 'RECOVERY_SENTINEL="[^"]*"' "$QP_ROOT/build_rabitq.sh" | head -1 | cut -d'"' -f2)"
if [ -n "$SENT" ] && ! grep -q "$SENT" "$LIB/$REL"; then
  echo "!!! [regen] WARNING: build_rabitq.sh's RECOVERY_SENTINEL ('$SENT') is NOT in the new"
  echo "!!! [regen] header. Step 2b would report 'already applied' and silently skip this patch."
fi
say "reminder: if this patch added a new symbol, bump RECOVERY_SENTINEL in build_rabitq.sh"
