"""PREREG B2: published 2.4 µm canonical predictions of the 28 labelled segments, reduced to 9.6 µm.

Writes data/bench/b2/<scroll>_<segment>.npy (uint8, 4x4 block mean) and deletes the full TIFF.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import tifffile

from vesuvius_sweep import BUCKET

OUT = Path("data/bench/b2"); OUT.mkdir(parents=True, exist_ok=True)
for o in json.load(open("data/label_inventory.json")):
    name = o["segment"].split("/")[0] + "_" + o["segment"].split("/")[2]
    dst = OUT / f"{name}.npy"
    if dst.exists():
        continue
    pred = [p for p in o["preds"] if "2.4" in p or "2.39" in p or "2.40" in p] or o["preds"]
    tif = OUT / "_tmp.tif"
    for _ in range(3):
        r = subprocess.run(["curl", "-sf", "-o", str(tif), f"{BUCKET}/{pred[0]}"])
        if r.returncode == 0:
            break
    if r.returncode != 0:
        print(name, "download FAILED", flush=True); continue
    a = tifffile.imread(tif)
    H, W = a.shape[0] // 4 * 4, a.shape[1] // 4 * 4
    b = a[:H, :W].reshape(H // 4, 4, W // 4, 4).mean((1, 3)).astype(np.uint8)
    np.save(dst, b); tif.unlink()
    print(name, a.shape, "->", b.shape, Path(pred[0]).name[:60], flush=True)
