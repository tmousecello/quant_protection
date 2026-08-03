#!/usr/bin/env bash
# The centroid-gap sweeps: E6 centroids under two cliff configurations, E6 rotation as the
# vectorization regression, and the full E9 miscorrection grid.
#
# The ablation is two values of --cliff-regions separated by --out-tag, NOT two new arms:
# ARMS = ("off","on") is threaded through classify_outcome and the row-count gates, and adding
# to it would have rippled into all of them. Each shard re-runs its own paired `off` arm, which
# also re-checks that the unprotected baseline reproduces.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=artifacts/phase3/logs/centroid_sweeps
mkdir -p "$LOG"

CENT="rotation,centroids"
BOTH="rotation,centroids,header"

pids=()
for shape in single_cell device_row device_column; do
  # centroid protection alone: separates the two problems the centroid row conflates
  $PY phase3_e6_shapes.py --adapter real --seeds 30 --shapes "$shape" --strata centroids \
      --cliff-regions "$CENT" --out-tag "${shape}__centroids_v2cent" \
      > "$LOG/${shape}__centroids_v2cent.log" 2>&1 &
  pids+=($!)
  # + the header: the device_row cell needs both, since the row destroys header and centroids
  $PY phase3_e6_shapes.py --adapter real --seeds 30 --shapes "$shape" --strata centroids \
      --cliff-regions "$BOTH" --out-tag "${shape}__centroids_v2cent_hdr" \
      > "$LOG/${shape}__centroids_v2cent_hdr.log" 2>&1 &
  pids+=($!)
  # regression: the rotation cells must be unchanged by the vectorized vote
  $PY phase3_e6_shapes.py --adapter real --seeds 30 --shapes "$shape" --strata rotation \
      --cliff-regions "$BOTH" --out-tag "${shape}__rotation_v2cent" \
      > "$LOG/${shape}__rotation_v2cent.log" 2>&1 &
  pids+=($!)
done

# E9 is ef=64 and unsharded (~455 s for all 420 evals).
$PY phase3_e9_miscorrection.py --adapter real --cliff-regions "$BOTH" \
    --out-tag v2cent_hdr > "$LOG/e9_v2cent_hdr.log" 2>&1 &
pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "=== centroid sweeps done: ${#pids[@]} shards, $fail non-zero exits ==="
exit $fail
