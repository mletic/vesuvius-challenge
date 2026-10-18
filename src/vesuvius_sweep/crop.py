"""Copy a (rows, cols) window of a published surface volume to a local zarr."""

import argparse
import time

import fsspec
import zarr

from vesuvius_sweep import BUCKET


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zarr_prefix"); ap.add_argument("out")
    ap.add_argument("--rows", type=int, nargs=2, required=True)
    ap.add_argument("--cols", type=int, nargs=2, required=True)
    a = ap.parse_args()
    src = zarr.open_group(fsspec.get_mapper(f"{BUCKET}/{a.zarr_prefix.rstrip('/')}"), mode="r")
    arr = src["0"]
    (r0, r1), (c0, c1) = a.rows, a.cols
    for attempt in range(5):  # S3 occasionally drops long HTTP reads
        try:
            data = arr[:, r0:r1, c0:c1]
            break
        except Exception as e:
            print("retry:", e); time.sleep(3)
    dst = zarr.open_group(a.out, mode="w")
    dst.attrs.update(dict(src.attrs))
    dst.create_dataset("0", data=data, chunks=(data.shape[0], 128, 128))
    print(a.out, data.shape)


if __name__ == "__main__":
    main()
