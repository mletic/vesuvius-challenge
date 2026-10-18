#!/usr/bin/env bash
# Hunt for ink: grow 2.4 µm micro-segments at a list of level-2 seeds and run the 2 µm ink model.
# Usage: INKENV=... MODELS=... VILLA=... scripts/hunt.sh SCROLL "z y x" ["z y x" ...]
set -uo pipefail
scroll=$1; shift
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
for seed in "$@"; do
  read -r z y x <<< "$seed"
  name=data/grow/${scroll}_z${z}_y${y}_x${x}_S512_hires
  if [ ! -d "$name.zarr" ]; then
    uv run python -m vesuvius_sweep.grow "$scroll" "$z" "$y" "$x" --size 512 --tile 256 --layers 65 --hires 2>&1 | grep -E "coverage|wrote|no sheet|Error" || continue
  fi
  [ -d "$name.zarr" ] || continue
  [ -f "${name}_r152.tif" ] || "$INKENV/bin/python" scripts/infer_r152.py "$name.zarr" "$MODELS/ink_2um/r152_3ddec_v2_l5_epoch13.ckpt" "${name}_r152.tif" --villa "$VILLA" 2>&1 | tail -1
  uv run python -c "
import numpy as np, tifffile, zarr
from PIL import Image
a=zarr.open_group('$name.zarr','r')['0']; ct=a[28:37].mean(0); m=ct>0
p=tifffile.imread('${name}_r152.tif').astype(np.float32)/255
lo,hi=np.percentile(ct[m],[1,99]); ct=np.clip((ct-lo)/(hi-lo),0,1)*m
Image.fromarray((np.concatenate([ct,np.clip(p/0.9,0,1)*m],1)*255).astype(np.uint8)).resize((1600,800)).save('${name}_view.png')
print('$name', 'cover', round(float(m.mean()),2), 'ink frac>0.5', round(float((p[m]>0.5).mean()),3), 'p99', round(float(np.percentile(p[m],99)),2))"
done
