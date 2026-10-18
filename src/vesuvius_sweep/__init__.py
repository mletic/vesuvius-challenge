"""Sweep published segments of unread scrolls for visible ink.

Streams segment surface volumes straight from the public S3 bucket (no sign-up,
CC-BY-NC 4.0) and writes flattened renders that can be eyeballed for letters.
"""

import argparse
import re
import sys
from pathlib import Path

import fsspec
import numpy as np
import requests
import zarr
from PIL import Image, ImageDraw
from skimage import exposure

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

# Scrolls the Vesuvius Challenge lists as not yet read (October 2026).
UNREAD_SCROLLS = [
    "PHerc0125", "PHerc0175A", "PHerc0175B", "PHerc0191", "PHerc0211",
    "PHerc0257", "PHerc0268", "PHerc0306B", "PHerc0358", "PHerc0483A",
    "PHerc0483B", "PHerc0490A", "PHerc0490B", "PHerc0800", "PHerc0813",
    "PHerc0826", "PHerc0846A", "PHerc0846B", "PHerc1203", "PHerc1218",
    "PHerc1545",
]


def list_prefixes(prefix: str) -> list[str]:
    """Immediate sub-prefixes of `prefix` in the bucket (handles pagination)."""
    out, token = [], None
    while True:
        params = {"list-type": "2", "delimiter": "/", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        xml = requests.get(BUCKET + "/", params=params, timeout=60).text
        out += [p for p in re.findall(r"<Prefix>(.*?)</Prefix>", xml) if p != prefix]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1)


def surface_volumes(scroll: str) -> list[str]:
    """Surface-volume zarr prefixes for every published segment of a scroll."""
    vols = []
    for seg in list_prefixes(f"{scroll}/segments/"):
        if seg.endswith("/raw/"):
            continue
        vols += [v for v in list_prefixes(seg + "surface-volumes/") if v.rstrip("/").endswith(".zarr")]
    return vols


def open_level(zarr_prefix: str, level: int) -> tuple[zarr.Array, float]:
    """Open one pyramid level over HTTP; returns the array and its pixel size in µm."""
    root = zarr.open_group(fsspec.get_mapper(f"{BUCKET}/{zarr_prefix.rstrip('/')}"), mode="r")
    ds = root.attrs["multiscales"][0]["datasets"]
    level = min(level, len(ds) - 1)
    um = next(t["scale"][-1] for t in ds[level]["coordinateTransformations"] if t["type"] == "scale")
    return root[ds[level]["path"]], float(um)


def render(vol: np.ndarray, um_per_px: float, band: int) -> Image.Image:
    """Mean of the `band` layers around the middle, contrast-equalised, with a 1 cm bar."""
    mid = vol.shape[0] // 2
    lo, hi = max(0, mid - band // 2), min(vol.shape[0], mid + band // 2 + 1)
    img = vol[lo:hi].astype(np.float32).mean(axis=0)
    mask = img > 0
    if mask.any():
        img = (img - img[mask].min()) / max(np.ptp(img[mask]), 1e-6)
    img = exposure.equalize_adapthist(np.clip(img, 0, 1), clip_limit=0.02)
    im = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    bar = round(10_000 / um_per_px)  # 1 cm in pixels
    d = ImageDraw.Draw(im)
    y = im.height - 20
    d.rectangle([10, y, 10 + bar, y + 6], fill=(255, 64, 64))
    d.text((10, y - 14), "1 cm", fill=(255, 64, 64))
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scrolls", nargs="*", default=UNREAD_SCROLLS, help="scroll ids (default: all unread scrolls)")
    ap.add_argument("--level", type=int, default=1, help="pyramid level (0 = full res; 1 = 2x down)")
    ap.add_argument("--band", type=int, default=7, help="number of middle layers to average")
    ap.add_argument("--out", type=Path, default=Path("renders"))
    ap.add_argument("--list", action="store_true", help="only list segments, do not render")
    args = ap.parse_args()

    for scroll in args.scrolls:
        vols = surface_volumes(scroll)
        print(f"{scroll}: {len(vols)} surface volume(s)", flush=True)
        for v in vols:
            seg = v.split("/")[2]
            print(f"  {seg}", flush=True)
            if args.list:
                continue
            dest = args.out / scroll / f"{seg}_L{args.level}.png"
            if dest.exists():
                continue
            try:
                arr, um = open_level(v, args.level)
                im = render(arr[:], um, args.band)
            except Exception as e:  # keep sweeping past broken stores
                print(f"    skipped: {e}", file=sys.stderr)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest)
            print(f"    -> {dest} ({arr.shape}, {um} µm/px)", flush=True)
