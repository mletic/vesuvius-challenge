"""Audit which way the scroll axis runs in every published segment mesh.

For each segment's tifxyz mesh (excluding mesh/intermediate/), reads z.tif and
takes the median gradient of z along rows and columns. The direction of +z in
the image tells how the segment's renders are oriented:

  rows-down  : z increases down the rows (e.g. PHerc0139 w043)
  rows-up    : z increases up the rows (render is upside down relative to rows-down)
  cols-right / cols-left : scroll axis runs along the columns (render is rotated 90°)

Writes data/orientation_audit.csv.

usage: uv run python scripts/audit_orientation.py [SCROLL ...]
"""

import csv
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile

from vesuvius_sweep import BUCKET, list_prefixes

S = requests.Session()


def get(url, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, timeout=60)
            r.raise_for_status()
            return r
        except Exception:
            if i == tries - 1:
                raise


def meshes(seg_prefix):
    xml = get(f"{BUCKET}/?list-type=2&prefix={seg_prefix}mesh/").text
    keys = re.findall(r"<Key>(.*?)</Key>", xml)
    return sorted({k.rsplit("/", 1)[0] for k in keys if k.endswith("/z.tif") and "/intermediate/" not in k}), \
        any("ink-detection/" in k for k in re.findall(r"<Prefix>(.*?)</Prefix>", get(f"{BUCKET}/?list-type=2&delimiter=/&prefix={seg_prefix}").text))


def classify(mesh):
    z = tifffile.imread(io.BytesIO(get(f"{BUCKET}/{mesh}/z.tif").content)).astype(np.float64)
    z[z <= 0] = np.nan
    if np.isnan(z).all():
        return np.nan, "empty"
    gr = np.nanmedian(np.gradient(z, axis=0))
    gc = np.nanmedian(np.gradient(z, axis=1))
    ang = float(np.degrees(np.arctan2(gc, gr)))  # 0 = +z down the rows, 90 = +z to the right
    cls = "rows-down" if -45 <= ang < 45 else "cols-right" if 45 <= ang < 135 else "cols-left" if -135 <= ang < -45 else "rows-up"
    return round(ang, 1), cls


def main() -> None:
    scrolls = sys.argv[1:] or [p.rstrip("/") for p in list_prefixes("") if not p.startswith("_")]
    rows = []
    for scroll in scrolls:
        segs = list_prefixes(f"{scroll}/segments/")
        segs = [s for s in segs if not s.endswith("/raw/")]
        if not segs:
            continue
        with ThreadPoolExecutor(16) as ex:
            info = list(ex.map(meshes, segs))
        jobs = [(scroll, s.split("/")[2], m, ink) for s, (ms, ink) in zip(segs, info) for m in ms]
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda j: classify(j[2]), jobs))
        for (sc, seg, m, ink), (ang, cls) in zip(jobs, res):
            rows.append(dict(scroll=sc, segment=seg, mesh=m.rsplit("/", 1)[1], angle=ang, orientation=cls, has_ink_detection=int(ink)))
        counts = {c: sum(1 for r in rows if r["scroll"] == scroll and r["orientation"] == c) for c in ("rows-down", "rows-up", "cols-right", "cols-left", "empty")}
        print(scroll, len(segs), "segments", counts, flush=True)
    with open("data/orientation_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
