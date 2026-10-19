"""Copy a (rows, cols) window of a published surface volume to a local zarr."""

import argparse
import json

import fsspec
import numpy as np
import zarr

from vesuvius_sweep import BUCKET
from vesuvius_sweep.microseg import read_retry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zarr_prefix"); ap.add_argument("out")
    ap.add_argument("--rows", type=int, nargs=2, required=True)
    ap.add_argument("--cols", type=int, nargs=2, required=True)
    a = ap.parse_args()
    url = f"{BUCKET}/{a.zarr_prefix.rstrip('/')}"
    attrs = json.loads(read_retry(fsspec.get_mapper(url), ".zattrs"))
    arr = zarr.open_array(fsspec.get_mapper(url), path="0", mode="r")
    (r0, r1), (c0, c1) = a.rows, a.cols
    dst = zarr.open_group(a.out, mode="w")
    dst.attrs.update(attrs)
    out = dst.create_dataset("0", shape=(arr.shape[0], r1 - r0, c1 - c0), chunks=(arr.shape[0], 128, 128), dtype=arr.dtype)
    band = 512  # read in bands so a dropped connection only costs one band
    for r in range(r0, r1, band):
        out[:, r - r0:min(r + band, r1) - r0] = read_retry(arr, (slice(None), slice(r, min(r + band, r1)), slice(c0, c1)))
    print(a.out, out.shape)


if __name__ == "__main__":
    main()
