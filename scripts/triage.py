"""Stage-1 ink triage: many cheap 9.6 µm micro-segments per scroll, ranked by line periodicity.

For each scroll: pick seeds on surface-prediction sheets spread over the scroll
(Poisson-disk in x/y at several heights), grow a --size field at level 2,
run the ink_9um model (two checkpoints, default window) and record the
line-periodicity "peak" score (vesuvius_sweep.textscore) to data/triage_<scroll>.csv.
Known text scores ~+1.6..+1.8 at 9 µm; presumed-blank segments -0.9..+0.35.

usage: uv run python scripts/triage.py SCROLL [--n-z 4] [--spacing 1100] [--size 1536]
       (needs INKENV and MODELS env vars, as in scripts/ink_sweep.sh)
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from vesuvius_sweep.microseg import SOURCES, open_arr, read_retry
from vesuvius_sweep.textscore import line_score

CKPTS = ["hybrid_3d2d-seed42/step-040000.pth", "hybrid_3d2d-seed43/step-075000.pth"]


def pick_seeds(scroll: str, n_z: int, spacing: int, margin: int, rng: np.random.Generator):
    """Seeds (z, y, x) in level-2 voxels, on sheet voxels, spread out in x/y per height."""
    _, sf_p, _, _ = SOURCES[scroll]
    coarse = open_arr(sf_p, "2")  # 4x down from level 2
    full = open_arr(sf_p, "0")
    seeds = []
    for zf in np.linspace(0.15, 0.85, n_z):
        zc = int(coarse.shape[0] * zf)
        s = read_retry(coarse, zc).astype(np.float32)
        dens = ndi.uniform_filter((s > 100).astype(np.float32), 31)
        inside = ndi.binary_erosion(dens > 0.08, iterations=margin // 4)
        cand = np.argwhere(inside)
        rng.shuffle(cand)
        chosen = []
        for y, x in cand:
            if all(np.hypot(y - a, x - b) * 4 >= spacing for a, b in chosen):
                chosen.append((y, x))
        for y, x in chosen:
            z0, y0, x0 = zc * 4, y * 4, x * 4
            blk = read_retry(full, (slice(z0 - 8, z0 + 8), slice(y0 - 16, y0 + 16), slice(x0 - 16, x0 + 16))).astype(np.float32)
            if blk.max() < 150:
                continue
            k = np.unravel_index(np.argmax(ndi.gaussian_filter(blk, 1)), blk.shape)
            seeds.append((z0 - 8 + k[0], y0 - 16 + k[1], x0 - 16 + k[2]))
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scroll", choices=sorted(SOURCES))
    ap.add_argument("--n-z", type=int, default=4)
    ap.add_argument("--spacing", type=int, default=1100)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=999)
    a = ap.parse_args()
    inkenv, models = os.environ["INKENV"], os.environ["MODELS"]
    out_csv = Path(f"data/triage_{a.scroll}.csv")
    done = set()
    if out_csv.exists():
        done = {row["name"] for row in csv.DictReader(out_csv.open())}
    seeds = pick_seeds(a.scroll, a.n_z, a.spacing, margin=a.size // 2, rng=np.random.default_rng(a.seed))[:a.max]
    print(f"{a.scroll}: {len(seeds)} seeds", flush=True)
    new = not out_csv.exists()
    with out_csv.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["name", "z", "y", "x", "coverage", "peak", "peak_v", "aniso", "ink_frac"])
        for z, y, x in seeds:
            name = f"{a.scroll}_z{z}_y{y}_x{x}_S{a.size}"
            if name in done:
                continue
            stem = Path("data/grow") / name
            for _ in range(3):  # S3 occasionally drops a read
                if (stem.parent / f"{name}.zarr").exists():
                    break
                subprocess.run([sys.executable, "-m", "vesuvius_sweep.grow", a.scroll, str(z), str(y), str(x),
                                "--size", str(a.size)], capture_output=True)
            if not (stem.parent / f"{name}.zarr").exists():
                print(f"{name}: grow failed", flush=True)
                continue
            preds = []
            for ck in CKPTS:
                tif = f"{stem}_ink9_{ck.split('/')[0][-6:]}.tif"
                if not Path(tif).exists():
                    subprocess.run([f"{inkenv}/bin/python", "-m", "koine_machines.inference.infer", f"{stem}.zarr",
                                    f"{models}/ink_9um/{ck}", tif, "--no-compile", "--batch-size", "16"],
                                   capture_output=True)
                if Path(tif).exists():
                    preds.append(tifffile.imread(tif).astype(np.float32) / 255)
            if not preds:
                print(f"{name}: inference failed", flush=True)
                continue
            p = np.mean(preds, 0)
            valid = np.asarray(Image.open(f"{stem}_mid.png").convert("L")) > 0
            d = line_score(p, valid, SOURCES[a.scroll][3])
            ink = float((np.clip((p[valid] - 0.25) / 0.5, 0, 1) > 0.5).mean()) if valid.any() else float("nan")
            w.writerow([name, z, y, x, round(d["coverage"], 3), round(d.get("peak", np.nan), 3),
                        round(d.get("peak_v", np.nan), 3), round(d["score"], 3), round(ink, 3)])
            fh.flush()
            print(f"{name}: cov {d['coverage']:.2f} peak {d.get('peak', float('nan')):+.2f}", flush=True)


if __name__ == "__main__":
    main()
