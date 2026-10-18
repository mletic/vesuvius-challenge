"""Ensemble the ink predictions of each published segment and save CT | ink review images."""

import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def main(scroll: str, max_side: int = 1600) -> None:
    groups = defaultdict(list)
    for p in sorted(glob.glob(f"ink/{scroll}/*.tif")):
        groups[Path(p).name.split("__")[0]].append(p)
    for seg, paths in groups.items():
        ink = np.mean([tifffile.imread(p).astype(np.float32) / 255 for p in paths], 0)
        ink = np.clip((ink - 0.25) / 0.5, 0, 1)
        ct = np.asarray(Image.open(f"renders/{scroll}/{seg}_L0.png").convert("L"), np.float32) / 255
        if ct.shape != ink.shape:
            ct = np.asarray(Image.fromarray((ct * 255).astype(np.uint8)).resize(ink.shape[::-1]), np.float32) / 255
        im = Image.fromarray((np.concatenate([ct, ink], 1) * 255).astype(np.uint8))
        im.thumbnail((max_side * 2, max_side))
        out = f"ink/{scroll}/{seg}_review.png"
        im.save(out)
        print(out, len(paths), f"p99-med={np.percentile(ink, 99) - np.median(ink):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
