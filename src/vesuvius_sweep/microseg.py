"""Micro-segmentation: turn a small block of a scroll into a surface volume.

No hand-traced mesh is needed. Within a block, papyrus sheets are close to
parallel, so we:

1. estimate the dominant sheet normal from the published surface prediction
   (structure tensor of its gradient),
2. resample the block into a frame (n, v, u) whose first axis is that normal,
3. track a single sheet as a height field n = h(v, u) by region-growing over
   surface-prediction peaks from a seed column,
4. sample the CT along the local normal at h ± k to get an N-layer surface
   volume in the same layout as published segments (layers, rows, cols).

The result is written as a one-level OME-Zarr that ink models accept.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

import fsspec
import numpy as np
import zarr
from PIL import Image
from scipy import ndimage as ndi

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

# Scroll -> (CT volume zarr, surface-prediction zarr, CT level matching surface level 0, µm/px)
SOURCES = {
    # Unread scrolls:
    "PHerc1203": (
        "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr",
        "PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr",
        "2", 9.612,
    ),
    "PHerc0846A": (
        "PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr",
        "PHerc0846A/representations/predictions/surfaces/20260319102732-surface-20260413222639-surface-m7-L2-th0.2.zarr",
        "2", 9.612,
    ),
    "PHerc0841": (
        "PHerc0841/volumes/20260319124803-2.403um-0.2m-77keV-masked.zarr",
        "PHerc0841/representations/predictions/surfaces/20260319124803-surface-20260413222639-surface-m7-L2-th0.2.zarr",
        "2", 9.612,
    ),
    "PHerc1203-9um": (
        "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr",
        "PHerc1203/representations/predictions/surfaces/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "0", 9.362,
    ),
    # Positive controls (scrolls with known, model-detectable ink):
    "PHercParis4": (
        "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr",
        "PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr",
        "2", 9.6,
    ),
    "PHerc0139-9um": (
        "PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr",
        "PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "0", 9.362,
    ),
}

# Unread scrolls with only a 9.36/8.64 µm scan follow the same pattern.
for _s, _v in {
    "PHerc0125": "20250821151825-9.362um", "PHerc0191": "20250821151635-9.362um", "PHerc0211": "20250821151803-9.362um",
    "PHerc0257": "20250821151750-9.362um", "PHerc0358": "20250821151737-9.362um", "PHerc0813": "20250821151723-9.362um",
    "PHerc0826": "20250821151701-9.362um", "PHerc1545": "20250821151648-9.362um", "PHerc0800": "20250521135224-8.640um",
}.items():
    _ts, _um = _v.split("-")
    _kev = "1.2m-113keV" if _um.startswith("9.362") else "1.2m-116keV"
    SOURCES[_s] = (f"{_s}/volumes/{_v}-{_kev}-masked.zarr",
                   f"{_s}/representations/predictions/surfaces/{_ts}-surface-20260413222639-surface-m7-L0-th0.2.zarr",
                   "0", float(_um.rstrip("um")))


def open_arr(prefix: str, level: str, tries: int = 4) -> zarr.Array:
    for i in range(tries):  # S3 metadata reads occasionally fail transiently
        try:
            return zarr.open_array(fsspec.get_mapper(f"{BUCKET}/{prefix}"), path=level, mode="r")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def read_retry(arr: zarr.Array, sl, tries: int = 5) -> np.ndarray:
    """arr[sl] with retries: long HTTP reads from the bucket sometimes drop."""
    for i in range(tries):
        try:
            return arr[sl]
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def dominant_normal(surf: np.ndarray) -> np.ndarray:
    """Unit normal (z, y, x) of the sheets in a block, from the gradient structure tensor."""
    s = ndi.gaussian_filter(surf.astype(np.float32), 2.0)
    g = np.stack(np.gradient(s))  # (3, Z, Y, X)
    g = g.reshape(3, -1)
    T = g @ g.T
    w, V = np.linalg.eigh(T)
    n = V[:, -1]
    return n / np.linalg.norm(n)


def frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (n, v, u) with u as close to the scroll axis (z) as possible."""
    z = np.array([1.0, 0, 0])
    u = z - n * (z @ n)
    if np.linalg.norm(u) < 1e-3:
        u = np.array([0, 0, 1.0]) - n * n[2]
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, v, u


def resample(vol: np.ndarray, origin: np.ndarray, axes, shape, order=1) -> np.ndarray:
    """Sample `vol` on a grid origin + i*a0 + j*a1 + k*a2 (block-local voxel coords)."""
    idx = [np.arange(s, dtype=np.float32) - s / 2 for s in shape]
    I, J, K = np.meshgrid(*idx, indexing="ij")
    coords = (origin[:, None, None, None]
              + axes[0][:, None, None, None] * I
              + axes[1][:, None, None, None] * J
              + axes[2][:, None, None, None] * K)
    return ndi.map_coordinates(vol, coords, order=order, mode="constant", cval=0)


def track_sheet(surf_r: np.ndarray, seed_vu: tuple[int, int], seed_n: int | None, tol: float = 2.5) -> np.ndarray:
    """Height field h(v, u) of one sheet in a normal-aligned surface block (n, v, u).

    Peaks of the (smoothed) surface prediction along n are sheet candidates.
    Growth goes best-first from the seed; each column takes the peak nearest to
    the median height of its already-assigned neighbours, within `tol` voxels.
    """
    N, V, U = surf_r.shape
    s = ndi.gaussian_filter1d(surf_r.astype(np.float32), 1.0, axis=0)
    is_peak = (s > 0.35 * 255) & (s >= np.roll(s, 1, 0)) & (s >= np.roll(s, -1, 0))
    is_peak[0] = is_peak[-1] = False
    h = np.full((V, U), np.nan, np.float32)
    v0, u0 = seed_vu
    cand = np.flatnonzero(is_peak[:, v0, u0])
    if cand.size == 0:
        raise RuntimeError("no sheet at the seed column; pick another seed")
    target = N / 2 if seed_n is None else seed_n
    h[v0, u0] = cand[np.argmin(np.abs(cand - target))]
    heap = [(0.0, v0, u0)]
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        _, v, u = heapq.heappop(heap)
        for dv, du in nb:
            vv, uu = v + dv, u + du
            if not (0 <= vv < V and 0 <= uu < U) or not np.isnan(h[vv, uu]):
                continue
            win = h[max(0, vv - 2):vv + 3, max(0, uu - 2):uu + 3]
            pred = np.nanmedian(win)
            c = np.flatnonzero(is_peak[:, vv, uu])
            if c.size == 0:
                continue
            d = np.abs(c - pred)
            if d.min() > tol:
                continue
            h[vv, uu] = c[np.argmin(d)]
            heapq.heappush(heap, (float(d.min()), vv, uu))
    return h


def fill_and_smooth(h: np.ndarray, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Fill holes by normalised convolution and smooth; returns (h, valid-mask)."""
    valid = ~np.isnan(h)
    w = ndi.gaussian_filter(valid.astype(np.float32), sigma)
    hs = ndi.gaussian_filter(np.where(valid, h, 0), sigma) / np.maximum(w, 1e-6)
    # Keep only the connected region the tracker actually reached, slightly dilated.
    reach = ndi.binary_dilation(ndi.binary_closing(valid, iterations=3), iterations=2)
    return hs, reach


def surface_volume(ct: np.ndarray, origin, axes, h: np.ndarray, layers: int, offset: float) -> np.ndarray:
    """Sample CT along the local normal at the sheet, layers centred on h + offset.

    `h` holds sheet heights as indices along n in the resampled frame, whose
    centre index (depth / 2) sits at `origin`; callers pass h already centred.
    """
    n, v, u = axes
    V, U = h.shape
    hv, hu = np.gradient(h)
    # Local normal in block coords: n - dh/dv * v - dh/du * u, normalised.
    ln = n[:, None, None] - hv[None] * v[:, None, None] - hu[None] * u[:, None, None]
    ln /= np.linalg.norm(ln, axis=0, keepdims=True)
    jj, kk = np.meshgrid(np.arange(V) - V / 2, np.arange(U) - U / 2, indexing="ij")
    sheet = (origin[:, None, None] + n[:, None, None] * h[None]
             + v[:, None, None] * jj[None] + u[:, None, None] * kk[None])
    out = np.empty((layers, V, U), np.uint8)
    for i, k in enumerate(np.arange(layers) - layers // 2):
        out[i] = np.clip(ndi.map_coordinates(ct, sheet + ln * (k + offset), order=1, mode="constant", cval=0), 0, 255)
    return out


def orient_body_positive(sv: np.ndarray) -> tuple[np.ndarray, bool]:
    """Make the papyrus body lie on the +n side of the tracked (recto) surface.

    The structure-tensor normal has an arbitrary sign. Flipping n -> -n together
    with v -> -v is a proper rotation of the frame, so the image is not mirrored;
    in array terms it reverses the layers and the v axis. Validated orientation
    (PHerc0139 w043) has the body on +n.
    """
    L = sv.shape[0]
    q = max(2, L // 6)
    valid = sv[L // 2] > 0
    if not valid.any():
        return sv, False
    plus = sv[L // 2 + 1:L // 2 + 1 + 2 * q][:, valid].mean()
    minus = sv[L // 2 - 2 * q:L // 2][:, valid].mean()
    if minus > plus:
        return np.ascontiguousarray(sv[::-1, ::-1, :]), True
    return sv, False


def to_published_layout(sv: np.ndarray) -> np.ndarray:
    """(k along +n, v, u) -> published segment layout.

    Published surface volumes have the scroll axis down the rows and their layer
    index running opposite to our +n (checked against PHerc0139 w043: their
    layer i == our layer L-1-i, and their rows == our u). The ink models are
    neither rotation- nor depth-flip-invariant: in the wrong layout the 9 µm
    model's output correlates 0.07 with the published prediction, in this one 0.55.
    """
    return np.ascontiguousarray(sv[::-1].transpose(0, 2, 1))


def write_zarr(path: Path, vol: np.ndarray, um: float, meta: dict) -> None:
    g = zarr.open_group(str(path), mode="w")
    g.create_dataset("0", data=vol, chunks=(vol.shape[0], 128, 128))
    g.attrs["multiscales"] = [{
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [um] * 3}]}],
    }]
    g.attrs["microseg"] = meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scroll", choices=sorted(SOURCES))
    ap.add_argument("z", type=int); ap.add_argument("y", type=int); ap.add_argument("x", type=int)
    ap.add_argument("--size", type=int, default=512, help="patch width/height in voxels")
    ap.add_argument("--depth", type=int, default=96, help="search depth along the normal")
    ap.add_argument("--layers", type=int, default=31)
    ap.add_argument("--offset", type=float, default=0.0, help="shift layer centre along the normal")
    ap.add_argument("--seed-n", type=int, default=None, help="which sheet: height in the n-frame (default: nearest to centre)")
    ap.add_argument("--out", type=Path, default=Path("data/microseg"))
    args = ap.parse_args()

    ct_p, sf_p, ct_level, um = SOURCES[args.scroll]
    ct_a, sf_a = open_arr(ct_p, ct_level), open_arr(sf_p, "0")
    # Load an axis-aligned cube big enough for any rotation of the patch.
    r = int(np.ceil(np.sqrt(2 * (args.size / 2) ** 2 + (args.depth / 2 + args.layers) ** 2))) + 4
    c = np.array([args.z, args.y, args.x])
    lo = np.maximum(c - r, 0); hi = np.minimum(c + r, ct_a.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    print(f"loading block {tuple(hi - lo)} at {tuple(lo)} ...", flush=True)
    ct = ct_a[sl]; sf = sf_a[sl]
    origin = (c - lo).astype(np.float32)

    m = 48  # normal from a central sub-block
    core = sf[tuple(slice(int(o) - m, int(o) + m) for o in origin)]
    axes = frame(dominant_normal(core))
    print("normal (z,y,x):", np.round(axes[0], 3), flush=True)

    shape = (args.depth, args.size, args.size)
    sf_r = resample(sf, origin, axes, shape)
    h = track_sheet(sf_r, (args.size // 2, args.size // 2), args.seed_n)
    cover = float(np.mean(~np.isnan(h)))
    print(f"tracked {cover:.0%} of the patch", flush=True)
    hs, reach = fill_and_smooth(h)
    sv = surface_volume(ct, origin, axes, hs - args.depth / 2, args.layers, args.offset)
    sv[:, ~reach] = 0
    sv, flipped = orient_body_positive(sv)
    if flipped:
        h = h[::-1]
        axes = (-axes[0], -axes[1], axes[2])
    sv = to_published_layout(sv)
    h = h.T

    name = f"{args.scroll}_z{args.z}_y{args.y}_x{args.x}_s{args.size}" + (f"_n{args.seed_n}" if args.seed_n is not None else "")
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(scroll=args.scroll, center_zyx=c.tolist(), level=ct_level, um=um, normal=axes[0].tolist(),
                v=axes[1].tolist(), u=axes[2].tolist(), coverage=cover, seed_n=args.seed_n)
    write_zarr(args.out / f"{name}.zarr", sv, um, meta)
    mid = sv[args.layers // 2 - 3: args.layers // 2 + 4].mean(0)
    Image.fromarray(np.clip(mid, 0, 255).astype(np.uint8)).save(args.out / f"{name}_mid.png")
    Image.fromarray((np.nan_to_num(h, nan=0) * 255 / args.depth).astype(np.uint8)).save(args.out / f"{name}_h.png")
    (args.out / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out / name}.zarr  {sv.shape}")


if __name__ == "__main__":
    main()
