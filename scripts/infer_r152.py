"""Run the scrollprize 2 µm ResNet-152 3D-decoder ink model on a surface-volume zarr.

Mirrors villa/ink-detection/optimized_inference (tile 256, clip 200, /200,
62 layers centred on the surface, sigmoid, Hann-blended overlap) but runs on
Apple MPS. Needs the villa checkout on --villa and the ink_canonical_2um ckpt.

usage: python scripts/infer_r152.py <in.zarr> <ckpt> <out.tif> --villa <path> [--offset K] [--reverse]
"""

import argparse
import sys

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import zarr


def hann2d(n):
    w = np.hanning(n + 2)[1:-1].astype(np.float32)
    return np.outer(w, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--villa", required=True)
    ap.add_argument("--frames", type=int, default=62)
    ap.add_argument("--offset", type=int, default=0, help="shift the layer window centre")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    sys.path.insert(0, f"{a.villa}/ink-detection/optimized_inference")
    from model_resnet3d_3d_decoder import load_model  # noqa: E402

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(a.ckpt, dev, num_frames=a.frames)
    vol = zarr.open_group(a.zarr, mode="r")["0"][:]
    D, H, W = vol.shape
    c0 = D // 2 + a.offset - a.frames // 2
    if c0 < 0 or c0 + a.frames > D:
        raise SystemExit(f"window {c0}..{c0 + a.frames} outside depth {D}")
    v = vol[c0:c0 + a.frames]
    if a.reverse:
        v = v[::-1]
    v = np.clip(v, 0, 200).astype(np.float32) / 200.0
    T, s = a.tile, a.stride
    ys = list(range(0, max(1, H - T + 1), s)); xs = list(range(0, max(1, W - T + 1), s))
    if ys[-1] != max(0, H - T): ys.append(max(0, H - T))
    if xs[-1] != max(0, W - T): xs.append(max(0, W - T))
    tiles = [(y, x) for y in ys for x in xs if vol[D // 2, y:y + T, x:x + T].any()]
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    wk = hann2d(T)
    print(f"{len(tiles)} tiles on {dev}", flush=True)
    with torch.no_grad():
        for i in range(0, len(tiles), a.batch):
            chunk = tiles[i:i + a.batch]
            x = np.stack([v[:, y:y + T, xx:xx + T] for y, xx in chunk])[:, None]  # B,1,D,T,T
            if x.shape[-1] != T or x.shape[-2] != T:
                pad = [(0, 0)] * 3 + [(0, T - x.shape[-2]), (0, T - x.shape[-1])]
                x = np.pad(x, pad)
            with torch.autocast(device_type=dev.type, enabled=dev.type != "cpu"):
                p = model.forward(torch.from_numpy(x).to(dev))
            p = F.interpolate(torch.sigmoid(p.float()), size=(T, T), mode="bilinear", align_corners=False)
            p = p[:, 0].cpu().numpy()
            for (y, xx), pi in zip(chunk, p):
                hh, ww = min(T, H - y), min(T, W - xx)
                acc[y:y + hh, xx:xx + ww] += pi[:hh, :ww] * wk[:hh, :ww]
                wsum[y:y + hh, xx:xx + ww] += wk[:hh, :ww]
            if (i // a.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(tiles)}", flush=True)
    pred = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 0)
    tifffile.imwrite(a.out, (np.clip(pred, 0, 1) * 255).astype(np.uint8))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
