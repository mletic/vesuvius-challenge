"""Grow a large micro-segment tile by tile.

One global frame (n, v, u) is fixed from the seed. The (v, u) plane is cut into
overlapping tiles processed outward from the seed; each tile loads only the
axis-aligned box around its own oriented slab, recentred along n on the height
its neighbours predict (so the depth window follows a curving sheet), and the
sheet tracker is seeded with the heights already known in the overlap.

Output is a surface volume in published layout (see microseg.to_published_layout).
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from vesuvius_sweep.microseg import (SOURCES, dominant_normal, fill_and_smooth, frame, open_arr,
                                     orient_body_positive, to_published_layout, write_zarr)


def peaks_along_n(sf, origin, axes, n0, depth, vs, us):
    """Surface-prediction peak mask on the slab (n0 + [-depth/2, depth/2), vs, us) of the global frame."""
    n, v, u = axes
    kk = np.arange(depth, dtype=np.float32) - depth / 2
    K, J, I = np.meshgrid(kk, vs, us, indexing="ij")
    c = origin[:, None, None, None] + n[:, None, None, None] * (n0[None] + K) \
        + v[:, None, None, None] * J + u[:, None, None, None] * I
    s = ndi.map_coordinates(sf, c, order=1, mode="constant", cval=0)
    s = ndi.gaussian_filter1d(s.astype(np.float32), 1.0, axis=0)
    pk = (s > 0.35 * 255) & (s >= np.roll(s, 1, 0)) & (s >= np.roll(s, -1, 0))
    pk[0] = pk[-1] = False
    return pk


def track_from(pk, h0, n0, tol=2.5):
    """Best-first growth inside a tile; h0 holds known global heights (NaN = unknown)."""
    D, V, U = pk.shape
    h = h0.copy()
    heap = [(0.0, int(a), int(b)) for a, b in zip(*np.nonzero(~np.isnan(h)))]
    heapq.heapify(heap)
    while heap:
        _, a, b = heapq.heappop(heap)
        for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            aa, bb = a + da, b + db
            if not (0 <= aa < V and 0 <= bb < U) or not np.isnan(h[aa, bb]):
                continue
            pred = np.nanmedian(h[max(0, aa - 2):aa + 3, max(0, bb - 2):bb + 3])
            cand = np.flatnonzero(pk[:, aa, bb]) - D / 2 + n0[aa, bb]
            if cand.size == 0:
                continue
            d = np.abs(cand - pred)
            if d.min() > tol:
                continue
            h[aa, bb] = cand[np.argmin(d)]
            heapq.heappush(heap, (float(d.min()), aa, bb))
    return h


def load_box(arr, pts, pad=4):
    lo = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(pts.max(0)).astype(int) + pad, arr.shape)
    return arr[tuple(slice(a, b) for a, b in zip(lo, hi))], lo


def sample_hires(ct0, c, axes, hs, reach, layers, scale=4, tile=128):
    """Sample a surface volume from the full-resolution CT (level 0) along the
    surface tracked on level 2: in-plane and along-normal spacing are both one
    level-0 voxel, so the output is `scale` times larger in-plane."""
    n, v, u = axes
    S = hs.shape[0]
    out = np.zeros((layers, S * scale, S * scale), np.uint8)
    gv, gu = np.gradient(hs)
    fine = (np.arange(scale) + 0.5) / scale - 0.5  # sub-positions of fine pixels inside a coarse one
    for a in range(0, S, tile):
        for b in range(0, S, tile):
            if not reach[a:a + tile, b:b + tile].any():
                continue
            ta, tb = min(tile, S - a), min(tile, S - b)
            # Fine grid coordinates (in level-2 units) and bilinear-upsampled heights.
            fa = (np.arange(a, a + ta)[:, None] + fine[None]).ravel()
            fb = (np.arange(b, b + tb)[:, None] + fine[None]).ravel()
            FA, FB = np.meshgrid(fa, fb, indexing="ij")
            h = ndi.map_coordinates(hs, [FA, FB], order=1, mode="nearest")
            hv = ndi.map_coordinates(gv, [FA, FB], order=1, mode="nearest")
            hu = ndi.map_coordinates(gu, [FA, FB], order=1, mode="nearest")
            ln = n[:, None, None] - hv[None] * v[:, None, None] - hu[None] * u[:, None, None]
            ln /= np.linalg.norm(ln, axis=0, keepdims=True)
            sheet2 = c[:, None, None] + n[:, None, None] * h[None] \
                + v[:, None, None] * (FA - S / 2)[None] + u[:, None, None] * (FB - S / 2)[None]
            sheet0 = sheet2 * scale + (scale - 1) / 2  # level-2 voxel centre -> level-0 coords
            pts = np.concatenate([(sheet0 + ln * k).reshape(3, -1).T[::211] for k in (-layers, layers)])
            ct, lo = load_box(ct0, pts, pad=6)
            loc = sheet0 - lo[:, None, None]
            for i, k in enumerate(np.arange(layers) - layers // 2):
                out[i, a * scale:(a + ta) * scale, b * scale:(b + tb) * scale] = np.clip(
                    ndi.map_coordinates(ct, loc + ln * k, order=1, mode="constant", cval=0), 0, 255)
    m = np.kron(reach, np.ones((scale, scale), bool))
    out[:, ~m] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scroll", choices=sorted(SOURCES))
    ap.add_argument("z", type=int); ap.add_argument("y", type=int); ap.add_argument("x", type=int)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--depth", type=int, default=64, help="search depth along n per tile")
    ap.add_argument("--layers", type=int, default=31)
    ap.add_argument("--out", type=Path, default=Path("data/grow"))
    ap.add_argument("--hires", action="store_true", help="sample from CT level 0 (sources tracked on level 2 only)")
    args = ap.parse_args()

    ct_p, sf_p, lv, um = SOURCES[args.scroll]
    ct_a, sf_a = open_arr(ct_p, lv), open_arr(sf_p, "0")
    c = np.array([args.z, args.y, args.x], float)
    core = sf_a[tuple(slice(int(q) - 48, int(q) + 48) for q in c)]
    axes = frame(dominant_normal(core))
    n, v, u = axes
    print("normal (z,y,x):", np.round(n, 3), flush=True)

    S, T = args.size, args.tile
    step = T - 64
    H = np.full((S, S), np.nan, np.float32)  # global heights along n, relative to c
    starts = list(range(0, S - T + 1, step))
    if starts[-1] != S - T:
        starts.append(S - T)
    tiles = sorted(((a, b) for a in starts for b in starts),
                   key=lambda t: np.hypot(t[0] + T / 2 - S / 2, t[1] + T / 2 - S / 2))
    seeded = False
    tracked_at = {}  # tile -> known-cell count when it was last tracked
    for pas in range(4):  # repeat: a tile may be reached from a new side later
        before = int(np.isnan(H).sum())
        for ti, (a, b) in enumerate(tiles):
            h0 = H[a:a + T, b:b + T].copy()
            known = ~np.isnan(h0)
            if seeded and (not known.any() or tracked_at.get(ti) == int(known.sum())):
                continue
            vs = np.arange(a, a + T, dtype=np.float32) - S / 2
            us = np.arange(b, b + T, dtype=np.float32) - S / 2
            if seeded:
                # Expected sheet height over the tile: smooth extrapolation of known heights.
                w = ndi.gaussian_filter(known.astype(np.float32), T / 4)
                n0 = ndi.gaussian_filter(np.where(known, h0, 0), T / 4) / np.maximum(w, 1e-6)
                n0 = np.where(w > 1e-4, n0, np.nanmean(h0)).astype(np.float32)
            else:
                n0 = np.zeros((T, T), np.float32)
            corners = np.array([c + n * (n0.mean() + s * (args.depth / 2 + np.ptp(n0) / 2 + 2))
                                + v * vv + u * uu for s in (-1, 1) for vv in (vs[0], vs[-1]) for uu in (us[0], us[-1])])
            sf, lo = load_box(sf_a, corners)
            pk = peaks_along_n(sf, (c - lo).astype(np.float32), axes, n0, args.depth, vs, us)
            if not seeded:
                cand = np.flatnonzero(pk[:, T // 2, T // 2]) - args.depth / 2
                if cand.size == 0:
                    raise SystemExit("no sheet at seed")
                h0[T // 2, T // 2] = cand[np.argmin(np.abs(cand))]
                seeded = True
            h = track_from(pk, h0, n0)
            newly = np.isnan(H[a:a + T, b:b + T]) & ~np.isnan(h)
            H[a:a + T, b:b + T][newly] = h[newly]
            tracked_at[ti] = int((~np.isnan(H[a:a + T, b:b + T])).sum())
            print(f"pass {pas} tile {ti + 1}/{len(tiles)} ({a},{b}): +{newly.mean():.0%}, total {np.mean(~np.isnan(H)):.0%}", flush=True)
        if int(np.isnan(H).sum()) == before:
            break


    hs, reach = fill_and_smooth(H, 3.0)
    print(f"coverage {np.mean(~np.isnan(H)):.0%}; sampling CT ...", flush=True)
    L = args.layers
    if args.hires:
        assert lv == "2", "--hires needs a source tracked on level 2"
        sv, flipped = orient_body_positive(sample_hires(open_arr(ct_p, "0"), c, axes, hs, reach, L))
        if flipped:
            H, n, v = H[::-1], -n, -v
        sv = to_published_layout(sv)
        name = f"{args.scroll}_z{args.z}_y{args.y}_x{args.x}_S{S}_hires"
        args.out.mkdir(parents=True, exist_ok=True)
        meta = dict(scroll=args.scroll, center_zyx=c.tolist(), level="0", um=um / 4, normal=n.tolist(), v=v.tolist(),
                    u=u.tolist(), coverage=float(np.mean(~np.isnan(H))), size=S * 4)
        write_zarr(args.out / f"{name}.zarr", sv, um / 4, meta)
        np.save(args.out / f"{name}_h.npy", H.T)
        Image.fromarray(np.clip(sv[L // 2 - 3:L // 2 + 4].mean(0), 0, 255).astype(np.uint8)).save(args.out / f"{name}_mid.png")
        (args.out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print("wrote", args.out / f"{name}.zarr", sv.shape)
        return
    sv = np.zeros((L, S, S), np.uint8)
    gv, gu = np.gradient(hs)
    for a in starts:
        for b in starts:
            if not reach[a:a + T, b:b + T].any():
                continue
            hv, hu = gv[a:a + T, b:b + T], gu[a:a + T, b:b + T]
            ht = hs[a:a + T, b:b + T]
            ln = n[:, None, None] - hv[None] * v[:, None, None] - hu[None] * u[:, None, None]
            ln /= np.linalg.norm(ln, axis=0, keepdims=True)
            jj, kk = np.meshgrid(np.arange(a, a + T) - S / 2, np.arange(b, b + T) - S / 2, indexing="ij")
            sheet = c[:, None, None] + n[:, None, None] * ht[None] + v[:, None, None] * jj[None] + u[:, None, None] * kk[None]
            pts = np.concatenate([(sheet + ln * k).reshape(3, -1).T[::97] for k in (-L, L)])
            ct, lo = load_box(ct_a, pts)
            loc = sheet - lo[:, None, None]
            for i, k in enumerate(np.arange(L) - L // 2):
                sv[i, a:a + T, b:b + T] = np.clip(ndi.map_coordinates(ct, loc + ln * k, order=1, mode="constant", cval=0), 0, 255)
    sv[:, ~reach] = 0
    sv, flipped = orient_body_positive(sv)
    if flipped:
        H, n, v = H[::-1], -n, -v
    sv = to_published_layout(sv)

    name = f"{args.scroll}_z{args.z}_y{args.y}_x{args.x}_S{S}"
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(scroll=args.scroll, center_zyx=c.tolist(), level=lv, um=um, normal=n.tolist(), v=v.tolist(), u=u.tolist(),
                coverage=float(np.mean(~np.isnan(H))), size=S)
    write_zarr(args.out / f"{name}.zarr", sv, um, meta)
    np.save(args.out / f"{name}_h.npy", H.T)
    Image.fromarray(np.clip(sv[L // 2 - 3:L // 2 + 4].mean(0), 0, 255).astype(np.uint8)).save(args.out / f"{name}_mid.png")
    (args.out / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print("wrote", args.out / f"{name}.zarr", sv.shape)


if __name__ == "__main__":
    main()
