"""Contact strip for a micro-segment: CT render | forward ink | reverse ink | overlay of the stronger."""

import glob
import sys

import numpy as np
import tifffile
from PIL import Image


def ensemble(paths: list[str]) -> np.ndarray:
    preds = [tifffile.imread(p).astype(np.float32) / 255 for p in paths]
    return np.clip((np.mean(preds, 0) - 0.25) / 0.5, 0, 1)


def main(stem: str) -> None:
    allp = sorted(glob.glob(f"{stem}_ink_*.tif"))
    fwd = ensemble([p for p in allp if not p.endswith("_reverse.tif")])
    rev = ensemble([p for p in allp if p.endswith("_reverse.tif")]) if any(p.endswith("_reverse.tif") for p in allp) else np.zeros_like(fwd)
    ct = np.asarray(Image.open(f"{stem}_mid.png").convert("L"), np.float32) / 255
    lo, hi = np.percentile(ct[ct > 0], [1, 99]) if (ct > 0).any() else (0, 1)
    ct = np.clip((ct - lo) / max(hi - lo, 1e-6), 0, 1)
    # Pick the direction whose map has more contrast (letters give a high p99 - median).
    score = lambda m: float(np.percentile(m, 99) - np.median(m))
    ink = fwd if score(fwd) >= score(rev) else rev
    over = np.stack([np.maximum(ct, ink), ct * (1 - ink), ct * (1 - ink)], -1)
    g = lambda m: np.stack([m] * 3, -1)
    row = np.concatenate([g(ct), g(fwd), g(rev), over], 1)
    Image.fromarray((row * 255).astype(np.uint8)).save(f"{stem}_strip.png")
    print(f"{stem}_strip.png  fwd={score(fwd):.3f} rev={score(rev):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
