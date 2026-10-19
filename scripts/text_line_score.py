"""Score an ink-probability map for lines of writing, and draw where they are.

Lines of writing in a published-layout render run horizontally and repeat
vertically, 5.65-7.27 mm apart (measured from the organisers' labels). For each
tile (default 20 mm, step 5 mm) this reports the row-profile periodicity score
(v3, `vesuvius_sweep.textscore.line_profile_score`): positive when the tile has
more line structure along rows than along columns. A geometry guard (`mask_peak`,
the same kind of score on the tile's own hole/outline map) flags tiles whose
shape alone could produce the pattern.

What the score can and cannot do (see PREREG.md and results/):
  - On crisp maps such as the published 2.4 µm predictions it separates text from
    the same map rotated 90° (AUC 0.92).
  - On 9 µm ink_9um outputs it is at chance, even on known text.
There is no absolute "text" threshold; compare tiles within a map, or a map with
its rotated copy (--rotate).

usage: uv run python scripts/text_line_score.py INK.tif --um 2.4 [--mask MASK.png] [--out heat.png] [--csv tiles.csv]
       INK may be uint8 (0-255) or float (0-1). --score v1 uses the superseded peakness score.
"""

import argparse
import csv

import numpy as np
import tifffile
from PIL import Image

from vesuvius_sweep.textscore import LINE_BAND_MM, line_profile_score, line_score, mask_peak


def score_tile(p, valid, um, which):
    if which == "v1":
        return line_score(p, valid, um, period_mm=LINE_BAND_MM).get("peak", np.nan)
    return line_profile_score(p, valid, um, band_mm=LINE_BAND_MM)["s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ink")
    ap.add_argument("--um", type=float, required=True, help="pixel size of the ink map in µm")
    ap.add_argument("--mask", help="image whose non-zero pixels are valid surface (default: ink > 1%%)")
    ap.add_argument("--tile-mm", type=float, default=20.0)
    ap.add_argument("--step-mm", type=float, default=5.0)
    ap.add_argument("--score", choices=("v3", "v1"), default="v3")
    ap.add_argument("--rotate", action="store_true", help="score the map rotated 90° (a control: lines run vertically)")
    ap.add_argument("--out", help="heat overlay PNG")
    ap.add_argument("--csv", help="per-tile scores")
    a = ap.parse_args()

    p = tifffile.imread(a.ink).astype(np.float32)
    if p.max() > 1.5:
        p /= 255
    valid = (np.asarray(Image.open(a.mask).convert("L")) > 0) if a.mask else p > 0.01
    if a.rotate:
        p, valid = np.ascontiguousarray(np.rot90(p)), np.ascontiguousarray(np.rot90(valid))
    t, s = int(a.tile_mm * 1000 / a.um), int(a.step_mm * 1000 / a.um)
    if p.shape[0] < t or p.shape[1] < t:
        t = min(p.shape)  # small map: one tile
        print(f"map smaller than a {a.tile_mm:g} mm tile; scoring it whole ({t * a.um / 1000:.1f} mm)")

    rows = []
    heat = np.full(p.shape, np.nan, np.float32)
    for r0 in range(0, p.shape[0] - t + 1, max(s, 1)):
        for c0 in range(0, p.shape[1] - t + 1, max(s, 1)):
            sl = (slice(r0, r0 + t), slice(c0, c0 + t))
            if valid[sl].mean() < 0.6:
                continue
            score = score_tile(p[sl], valid[sl], a.um, a.score)
            mp = mask_peak(valid[sl], a.um)
            rows.append(dict(row=r0, col=c0, coverage=round(float(valid[sl].mean()), 3), score=round(score, 3),
                             mask_peak=round(mp, 3), geometry_flag=int(mp > 1.0)))
            if np.isfinite(score) and mp <= 1.0:
                cr, cc = r0 + t // 2, c0 + t // 2
                heat[cr - s // 2:cr + s // 2, cc - s // 2:cc + s // 2] = score

    ok = [r for r in rows if np.isfinite(r["score"]) and not r["geometry_flag"]]
    if ok:
        sc = np.array([r["score"] for r in ok])
        best = max(ok, key=lambda r: r["score"])
        print(f"{len(ok)} tiles scored ({a.score}{', rotated' if a.rotate else ''}); median {np.median(sc):+.2f}, "
              f"best {best['score']:+.2f} at row {best['row']}, col {best['col']}; "
              f"{sum(r['geometry_flag'] for r in rows)} tiles flagged by the geometry guard")
    else:
        print("no tile had enough valid surface to score")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if a.out:
        W = 1400
        H = max(1, int(W * p.shape[0] / p.shape[1]))
        base = np.asarray(Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize((W, H)), np.float32) / 255
        hs = np.nan_to_num(np.clip(heat / 1.2, 0, 1), nan=0)  # v3 saturates near +1.1
        hs = np.asarray(Image.fromarray((hs * 255).astype(np.uint8)).resize((W, H), Image.NEAREST), np.float32) / 255
        rgb = np.stack([np.maximum(base * 0.8, hs), base * 0.8, base * 0.8 * (1 - hs)], -1)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(a.out)


if __name__ == "__main__":
    main()
