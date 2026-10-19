#!/usr/bin/env bash
# Run the scrollprize ink_9um model over every published segment of a scroll,
# at several z windows, caching surface volumes locally.
# Usage: INKENV=/path/to/venv MODELS=/path/to/models scripts/search/ink_sweep.sh PHerc0800 [ckpt ...]
set -euo pipefail
scroll=$1; shift
ckpts=("${@:-hybrid_3d2d-seed42/step-040000.pth}")
root=$(cd "$(dirname "$0")/../.." && pwd)
cache=$root/data/$scroll; out=$root/ink/$scroll
mkdir -p "$cache" "$out"
B=https://vesuvius-challenge-open-data.s3.amazonaws.com
for zp in $(cd "$root" && uv run python -c "from vesuvius_sweep import surface_volumes as s; print('\n'.join(s('$scroll')))"); do
  seg=$(echo "$zp" | cut -d/ -f3)
  local_zarr=$cache/$seg.zarr
  if [ ! -d "$local_zarr" ]; then
    # Pull level 0 + metadata with plain HTTP (bucket is public).
    "$INKENV/bin/python" - "$B/${zp%/}" "$local_zarr" <<'PY'
import sys, fsspec, zarr
src = zarr.open_group(fsspec.get_mapper(sys.argv[1]), mode="r")
dst = zarr.open_group(sys.argv[2], mode="w")
dst.attrs.update(dict(src.attrs))
a = src["0"]; dst.create_dataset("0", data=a[:], chunks=a.chunks)
PY
  fi
  depth=$("$INKENV/bin/python" -c "import zarr,sys;print(zarr.open_group(sys.argv[1],'r')['0'].shape[0])" "$local_zarr")
  mid=$((depth / 2))
  for ck in "${ckpts[@]}"; do
    tag=$(echo "$ck" | sed 's#hybrid_3d2d-##;s#/step-0*#_s#;s#\.pth##')
    for off in -2 0 2; do
      s=$((mid - 10 + off)); e=$((s + 21))
      dest=$out/${seg}__${tag}__z${s}-${e}.tif
      [ -f "$dest" ] && continue
      echo "== $seg $tag z$s-$e"
      "$INKENV/bin/python" -m koine_machines.inference.infer "$local_zarr" "$MODELS/ink_9um/$ck" "$dest" \
        --layer-start $s --layer-end $e --no-compile --batch-size 16 2>&1 | grep -E "Wrote|Error|error" || true
    done
  done
done
