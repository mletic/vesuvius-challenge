"""Score an ink-probability map for lines of writing, and draw where they are.

Text lines in a published-layout render run horizontally and repeat every
~4-7 mm. For each tile (default 20 mm, step 5 mm) this reports the "peakness"
of the vertical spectrum in that period band relative to its flanks, minus the
same for the horizontal direction (vesuvius_sweep.textscore.line_score), and a
geometry guard: the same score computed on the tile's own hole/outline map.

Calibration (README): known text +1.5..+1.9 per 20 mm tile, blank margins and
presumed-blank segments -0.9..+0.35. Treat a tile as text-like when
peak >= 1.0 and mask_peak <= 1.0 and coverage >= 0.6.

usage: uv run python scripts/text_line_score.py INK.tif --um 2.4 [--mask MASK.png] [--tile-mm 20] [--out heat.png]
       INK may be uint8 (0-255) or float (0-1); 2.4 µm maps are block-averaged to 40 µm first.
"""

import argparse
import csv

import numpy as np
import tifffile
from PIL import Image

from vesuvius_sweep.textscore import line_score, mask_peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ink"); ap.add_argument("--um", type=float, required=True, help="pixel size of the ink map")
    ap.add_argument("--mask", help="image whose non-zero pixels are valid surface (default: ink > 1%)")
    ap.add_argument("--tile-mm", type=float, default=20.0); ap.add_argument("--step-mm", type=float, default=5.0)
    ap.add_argument("--out", help="heat overlay PNG"); ap.add_argument("--csv", help="per-tile scores")
    a = ap.parse_args()

    p = tifffile.imread(a.ink).astype(np.float32)
    if p.max() > 1.5:
        p /= 255
    valid = (np.asarray(Image.open(a.mask).convert("L")) > 0) if a.mask else p > 0.01
    t, s = int(a.tile_mm * 1000 / a.um), int(a.step_mm * 1000 / a.um)
    if p.shape[0] < t or p.shape[1] < t:
        t = min(p.shape)  # small map: one tile
        print(f"map smaller than a {a.tile_mm} mm tile; scoring it whole ({t * a.um / 1000:.1f} mm)")
    rows = []
    heat = np.full(p.shape, np.nan, np.float32)
    for r0 in range(0, p.shape[0] - t + 1, max(s, 1)):
        for c0 in range(0, p.shape[1] - t + 1, max(s, 1)):
            sl = (slice(r0, r0 + t), slice(c0, c0 + t))
            d = line_score(p[sl], valid[sl], a.um)
            mp = mask_peak(valid[sl], a.um)
            text_like = d["coverage"] >= 0.6 and d.get("peak", -9) >= 1.0 and mp <= 1.0
            rows.append(dict(row=r0, col=c0, coverage=round(d["coverage"], 3), peak=round(d.get("peak", np.nan), 3),
                             mask_peak=round(mp, 3), text_like=int(text_like)))
            if np.isfinite(d.get("peak", np.nan)):
                cr, cc = r0 + t // 2, c0 + t // 2
                heat[cr - s // 2:cr + s // 2, cc - s // 2:cc + s // 2] = d["peak"] if mp <= 1.0 else -9
    ok = [r for r in rows if np.isfinite(r["peak"])]
    n_text = sum(r["text_like"] for r in rows)
    if ok:
        best = max(ok, key=lambda r: r["peak"] if r["mask_peak"] <= 1.0 else -9)
        print(f"{len(ok)} tiles scored, {n_text} text-like; best peak {best['peak']:+.2f} "
              f"(mask {best['mask_peak']:+.2f}, coverage {best['coverage']:.2f}) at row {best['row']}, col {best['col']}")
    else:
        print("no tile had enough valid surface to score")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.out:
        W = 1400; H = max(1, int(W * p.shape[0] / p.shape[1]))
        base = np.asarray(Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize((W, H)), np.float32) / 255
        hs = np.nan_to_num(np.clip((heat - 0.5) / 1.5, 0, 1), nan=0)
        hs = np.asarray(Image.fromarray((hs * 255).astype(np.uint8)).resize((W, H), Image.NEAREST), np.float32) / 255
        Image.fromarray((np.stack([np.maximum(base * 0.8, hs), base * 0.8, base * 0.8 * (1 - hs)], -1) * 255).astype(np.uint8)).save(a.out)


if __name__ == "__main__":
    main()
