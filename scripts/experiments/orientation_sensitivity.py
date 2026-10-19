"""How much do the published ink models depend on in-plane orientation?

Runs a model on all 8 in-plane rotations/flips of a labelled surface volume
(optionally also with the depth order reversed), maps each prediction back to
the original frame, and scores it against the published ink labels with AUC
inside the supervision mask.

usage (needs INKENV / MODELS / VILLA env vars):
  uv run python scripts/experiments/orientation_sensitivity.py 9um   # ink_9um on data/orient/w00_9um.zarr
  uv run python scripts/experiments/orientation_sensitivity.py 2um   # ink_canonical_2um on the central 2048² of w00_2um.zarr
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

from vesuvius_sweep.microseg import write_zarr

OUT = Path("data/orient")
# (name, k quarter-turns, flip columns afterwards); together the 8 elements of the dihedral group.
TRANSFORMS = [(f"rot{90 * k}{'+flip' if f else ''}", k, f) for k in range(4) for f in (False, True)]


def fwd(a, k, f):  # a: (..., H, W)
    a = np.rot90(a, k, axes=(-2, -1))
    return a[..., ::-1] if f else a


def inv(a, k, f):
    if f:
        a = a[..., ::-1]
    return np.rot90(a, -k, axes=(-2, -1))


def auc(score, label):
    """Rank AUC (Mann-Whitney), ties averaged."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    mode = sys.argv[1]
    inkenv, models, villa = os.environ["INKENV"], os.environ["MODELS"], os.environ.get("VILLA", "")
    lab = np.load(OUT / "w00_inklabels.npy") > 0
    sup = np.load(OUT / "w00_supervision.npy") > 0
    if mode == "9um":
        vol = zarr.open_group(str(OUT / "w00_9um.zarr"), "r")["0"][:]
        lab, sup = lab[2::4, 2::4], sup[2::4, 2::4]
        depths = [("as-published", False), ("depth-reversed", True)]
    else:
        vol = zarr.open_group(str(OUT / "w00_2um.zarr"), "r")["0"][:, 1024:3072, 1024:3072]
        lab, sup = lab[1024:3072, 1024:3072], sup[1024:3072, 1024:3072]
        depths = [("as-published", False)]
    results = []
    for dname, rev in depths:
        for name, k, f in TRANSFORMS:
            tag = f"{mode}_{dname}_{name}"
            pred_path = OUT / f"{tag}.tif"
            if not pred_path.exists():
                v = fwd(vol, k, f)
                if rev:
                    v = v[::-1]
                tmp = OUT / f"_{tag}.zarr"
                write_zarr(tmp, np.ascontiguousarray(v), 9.612 if mode == "9um" else 2.403, {})
                if mode == "9um":
                    cmd = [f"{inkenv}/bin/python", "-m", "koine_machines.inference.infer", str(tmp),
                           f"{models}/ink_9um/hybrid_3d2d-seed42/step-040000.pth", str(pred_path), "--no-compile", "--batch-size", "16"]
                else:
                    cmd = [f"{inkenv}/bin/python", "scripts/infer_r152.py", str(tmp),
                           f"{models}/ink_2um/r152_3ddec_v2_l5_epoch13.ckpt", str(pred_path), "--villa", villa]
                subprocess.run(cmd, capture_output=True)
                subprocess.run(["rm", "-rf", str(tmp)])
            p = inv(tifffile.imread(pred_path).astype(np.float32), k, f)
            m = sup & (fwd(np.ones_like(sup), 0, False))
            a = auc(p[m], lab[m])
            results.append(dict(model=mode, depth=dname, transform=name, auc=round(a, 4)))
            print(f"{mode:4s} {dname:15s} {name:10s} AUC {a:.3f}", flush=True)
    (OUT / f"results_{mode}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
