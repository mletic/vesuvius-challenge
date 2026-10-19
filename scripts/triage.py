"""Stage-1 ink triage: many cheap 9.6 µm micro-segments per scroll, ranked by line periodicity.

For each scroll: pick seeds on surface-prediction sheets spread over the scroll
(Poisson-disk in x/y at several heights), grow a --size field at level 2,
run the ink_9um model (two checkpoints, default window) and record the
line-periodicity "peak" score (vesuvius_sweep.textscore) to data/triage_<scroll>.csv.
Known text scores ~+1.6..+1.8 at 9 µm; presumed-blank segments -0.9..+0.35.

usage: uv run python scripts/triage.py SCROLL [--n-z 4] [--spacing 1100] [--size 1536] [--z-range 0.25 0.75]
       uv run python scripts/triage.py SCROLL --rescore   # recompute scores and guards in place
A field counts as "clean" when coverage >= 0.6, holes <= 12% of its outline and mask_peak <= 1.0.
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
from vesuvius_sweep.textscore import line_score, mask_peak

CKPTS = ["hybrid_3d2d-seed42/step-040000.pth", "hybrid_3d2d-seed43/step-075000.pth"]


def pick_seeds(scroll: str, n_z: int, spacing: int, margin: int, rng: np.random.Generator,
               z_range=(0.25, 0.75), avoid=()):
    """Seeds (z, y, x) in level-2 voxels, on sheet voxels, spread out in x/y per height."""
    _, sf_p, _, _ = SOURCES[scroll]
    coarse = open_arr(sf_p, "2")  # 4x down from level 2
    full = open_arr(sf_p, "0")
    seeds = []
    for zf in np.linspace(*z_range, n_z):
        zc = int(coarse.shape[0] * zf)
        s = read_retry(coarse, zc).astype(np.float32)
        dens = ndi.uniform_filter((s > 100).astype(np.float32), 31)
        inside = ndi.binary_erosion(dens > 0.08, iterations=margin // 4)
        cand = np.argwhere(inside)
        rng.shuffle(cand)
        # Do not re-sample near fields already scored at (about) this height.
        chosen = [(ay / 4, ax / 4) for az, ay, ax in avoid if abs(az - zc * 4) < spacing / 2]
        n_avoid = len(chosen)
        for y, x in cand:
            if all(np.hypot(y - a, x - b) * 4 >= spacing for a, b in chosen):
                chosen.append((y, x))
        for y, x in chosen[n_avoid:]:
            y, x = int(y), int(x)
            z0, y0, x0 = zc * 4, y * 4, x * 4
            blk = read_retry(full, (slice(z0 - 8, z0 + 8), slice(y0 - 16, y0 + 16), slice(x0 - 16, x0 + 16))).astype(np.float32)
            if blk.max() < 150:
                continue
            k = np.unravel_index(np.argmax(ndi.gaussian_filter(blk, 1)), blk.shape)
            seeds.append((z0 - 8 + k[0], y0 - 16 + k[1], x0 - 16 + k[2]))
    return seeds


COLUMNS = ["name", "z", "y", "x", "coverage", "peak", "peak_v", "aniso", "ink_frac", "mask_peak", "hole_frac", "clean"]
CLEAN = dict(min_coverage=0.6, max_hole_frac=0.12, max_mask_peak=1.0)


def score_field(scroll: str, stem: Path) -> dict | None:
    """Text score of a grown field plus guards against geometry-driven scores.

    mask_peak: the same peakness score computed on the field's own hole/outline
    map; a high value means tears or the outline alone make horizontal periodicity.
    hole_frac: holes inside the field's filled outline (torn, frayed surfaces).
    """
    preds = [tifffile.imread(t).astype(np.float32) / 255 for t in sorted(stem.parent.glob(stem.name + "_ink9_*.tif"))]
    if not preds:
        return None
    p = np.mean(preds, 0)
    valid = np.asarray(Image.open(f"{stem}_mid.png").convert("L")) > 0
    um = SOURCES[scroll][3]
    d = line_score(p, valid, um)
    mp = mask_peak(valid, um)
    hull = ndi.binary_fill_holes(ndi.binary_closing(valid, iterations=15))
    hole_frac = 1 - valid.sum() / max(hull.sum(), 1)
    ink = float((np.clip((p[valid] - 0.25) / 0.5, 0, 1) > 0.5).mean()) if valid.any() else float("nan")
    clean = (d["coverage"] >= CLEAN["min_coverage"] and hole_frac <= CLEAN["max_hole_frac"]
             and np.isfinite(mp) and mp <= CLEAN["max_mask_peak"])
    return dict(coverage=round(d["coverage"], 3), peak=round(d.get("peak", np.nan), 3),
                peak_v=round(d.get("peak_v", np.nan), 3), aniso=round(d["score"], 3), ink_frac=round(ink, 3),
                mask_peak=round(mp, 3), hole_frac=round(float(hole_frac), 3), clean=int(clean))


def rescore(scroll: str, out_csv: Path) -> None:
    rows = list(csv.DictReader(out_csv.open()))
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        for r in rows:
            d = score_field(scroll, Path("data/grow") / r["name"])
            if d:
                w.writerow({**{k: r[k] for k in ("name", "z", "y", "x")}, **d})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scroll", choices=sorted(SOURCES))
    ap.add_argument("--n-z", type=int, default=4)
    ap.add_argument("--spacing", type=int, default=1100)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=999)
    ap.add_argument("--z-range", type=float, nargs=2, default=(0.25, 0.75),
                    help="fraction of the scan height to seed in (ends of the band are torn)")
    ap.add_argument("--margin", type=int, default=400, help="min distance (L2 vox) of seeds from the scroll edge")
    ap.add_argument("--rescore", action="store_true", help="recompute all scores in the CSV and exit")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.scroll, Path(f"data/triage_{a.scroll}.csv"))
        return
    inkenv, models = os.environ["INKENV"], os.environ["MODELS"]
    out_csv = Path(f"data/triage_{a.scroll}.csv")
    done, avoid = set(), []
    if out_csv.exists():
        for row in csv.DictReader(out_csv.open()):
            done.add(row["name"]); avoid.append((int(row["z"]), int(row["y"]), int(row["x"])))
    seeds = pick_seeds(a.scroll, a.n_z, a.spacing, margin=a.margin, rng=np.random.default_rng(a.seed),
                       z_range=tuple(a.z_range), avoid=avoid)
    np.random.default_rng(a.seed + 1000).shuffle(seeds)  # a --max cap then samples all heights evenly
    seeds = seeds[:a.max]
    print(f"{a.scroll}: {len(seeds)} seeds", flush=True)
    new = not out_csv.exists()
    with out_csv.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
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
            for ck in CKPTS:
                tif = f"{stem}_ink9_{ck.split('/')[0][-6:]}.tif"
                if not Path(tif).exists():
                    subprocess.run([f"{inkenv}/bin/python", "-m", "koine_machines.inference.infer", f"{stem}.zarr",
                                    f"{models}/ink_9um/{ck}", tif, "--no-compile", "--batch-size", "16"],
                                   capture_output=True)
            d = score_field(a.scroll, stem)
            if d is None:
                print(f"{name}: inference failed", flush=True)
                continue
            w.writerow([name, z, y, x] + [d[k] for k in COLUMNS[4:]])
            fh.flush()
            print(f"{name}: cov {d['coverage']:.2f} peak {d['peak']:+.2f} mask {d['mask_peak']:+.2f} "
                  f"holes {d['hole_frac']:.2f} clean {d['clean']}", flush=True)


if __name__ == "__main__":
    main()
