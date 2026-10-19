#!/usr/bin/env bash
# Ink-predict one micro-segment zarr at three z windows and two checkpoints, then make a contact strip.
# Usage: INKENV=... MODELS=... scripts/search/ink_patch.sh data/microseg/<name>.zarr
set -euo pipefail
zarr=$1; stem=${zarr%.zarr}
root=$(cd "$(dirname "$0")/../.." && pwd)
for ck in hybrid_3d2d-seed42/step-040000.pth hybrid_3d2d-seed43/step-075000.pth; do
  tag=$(echo "$ck" | sed 's#hybrid_3d2d-##;s#/step-0*#_s#;s#\.pth##')
  for s in 3 5 7; do
    out=${stem}_ink_${tag}_z$s.tif
    [ -f "$out" ] && continue
    "$INKENV/bin/python" -m koine_machines.inference.infer "$zarr" "$MODELS/ink_9um/$ck" "$out" \
      --layer-start $s --layer-end $((s + 21)) --no-compile --batch-size 16 2>&1 | grep -E "Wrote|Error" || true
  done
done
cd "$root" && uv run python -m vesuvius_sweep.strip "$stem"
