"""Benchmark inputs: which segments carry organiser ink labels, their ~9.4 µm volumes, and the labels.

Writes data/label_inventory.json, data/bench/vol_inventory.json and
data/bench/labels/<scroll>_<segment>.npz (level 2 = 9.6 µm; zarr v3 read via a throwaway env).

usage: uv run python scripts/bench_inventory.py
"""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from vesuvius_sweep import BUCKET, list_prefixes

SCROLLS = ["PHerc0009B", "PHerc0139", "PHerc0172", "PHerc0343P", "PHerc0500P2", "PHerc0814", "PHerc0841",
           "PHerc1447", "PHerc1667", "PHercParis4"]
S = requests.Session()


def seg_info(seg):
    xml = S.get(f"{BUCKET}/?list-type=2&prefix={seg}ink-labels/", timeout=60).text
    labs = sorted({k.split("/inklabels.zarr")[0] for k in re.findall(r"<Key>(.*?)</Key>", xml) if "/inklabels.zarr/" in k})
    xml = S.get(f"{BUCKET}/?list-type=2&prefix={seg}ink-detection/", timeout=60).text
    preds = [k for k in re.findall(r"<Key>(.*?)</Key>", xml) if k.endswith(".tif") and "/downsampled/" not in k]
    return dict(segment=seg, labels=labs, preds=preds)


def vol_info(o):
    vols = [p.split("/")[-2] for p in list_prefixes(o["segment"] + "surface-volumes/")]
    v9 = [v for v in vols if v.startswith(("9.3", "8.64", "7.9", "9.6"))]
    shape = None
    if v9:
        shape = json.loads(S.get(f"{BUCKET}/{o['segment']}surface-volumes/{v9[0]}/0/.zarray", timeout=30).text)["shape"]
    return [o["segment"].split("/")[0], o["segment"].split("/")[2][:40], v9[:1], shape]


LABELS = r'''
import json, sys, zarr, numpy as np
B = "%s/"
for o in json.load(open("data/label_inventory.json")):
    name = o["segment"].split("/")[0] + "_" + o["segment"].split("/")[2]
    out = f"data/bench/labels/{name}.npz"
    try:
        open(out); continue
    except OSError:
        pass
    arrs = {}
    for z in ("inklabels", "supervision"):
        for _ in range(4):
            try:
                arrs[z] = np.asarray(zarr.open_array(f"{B}{o['labels'][0]}/{z}.zarr", path="2", mode="r")[:]) > 0
                break
            except Exception:
                pass
    if len(arrs) == 2:
        np.savez_compressed(out, ink=arrs["inklabels"], sup=arrs["supervision"]); print(name, arrs["supervision"].shape)
''' % BUCKET


def main():
    segs = [s for sc in SCROLLS for s in list_prefixes(f"{sc}/segments/") if not s.endswith("/raw/")]
    with ThreadPoolExecutor(16) as ex:
        inv = [o for o in ex.map(seg_info, segs) if o["labels"]]
    Path("data").mkdir(exist_ok=True)
    Path("data/label_inventory.json").write_text(json.dumps(inv, indent=1))
    with ThreadPoolExecutor(16) as ex:
        vols = list(ex.map(vol_info, inv))
    Path("data/bench/labels").mkdir(parents=True, exist_ok=True)
    Path("data/bench/vol_inventory.json").write_text(json.dumps(vols))
    print(f"{len(inv)} labelled segments, {sum(1 for v in vols if v[2])} with a ~9.4 µm volume")
    subprocess.run(["uv", "run", "--no-project", "--with", "zarr>=3", "--with", "fsspec", "--with", "aiohttp",
                    "--with", "requests", "--with", "numpy", "python", "-c", LABELS], check=True)


if __name__ == "__main__":
    main()
