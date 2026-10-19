"""Can a label-free text score rank ink checkpoints on a scroll the model never saw?

Pilot on PHerc0841 w00, the 9.366 µm surface volume (held out for ink_9um,
see villa issue #1867). For each of the 14 ink_9um checkpoints and both depth
directions: run inference, then compute

  auc        : against the organisers' labels inside the supervision mask (ground truth)
  text       : label-free, median line-periodicity peak over clean 20 mm tiles
               (coverage >= 0.6, mask_peak <= 1) -- fixed before looking at results
  separation : label-free baseline from #1867, mean(p | p > .5) - mean(p | p <= .5)

and report Spearman correlations across checkpoints in the forward direction, plus
how often each label-free score picks the better direction.

Deviation from the plan fixed before the run: ranking was to use the reverse
direction (correct for the 4.681 µm label-bucket renders in #1867). On this
9.366 µm array the smoke test gave AUC 0.748 forward vs 0.501 reverse
(seed42 step-075000), so the ranking uses forward. Direction-picking is reported
for both.

usage: INKENV=... MODELS=... uv run python scripts/benchmark/ckpt_rank.py [--infer-only | --score-only]
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image

from vesuvius_sweep.textscore import line_profile_score, line_score, mask_peak

D = Path("data/ckrank")
UM = 9.366
CKPTS = [f"hybrid_3d2d-seed{s}/step-{st}.pth" for s in (42, 43)
         for st in ("010000", "020000", "030000", "040000", "050000", "060000", "075000")]


def tag(ck, direction):
    return ck.replace("hybrid_3d2d-", "").replace("/step-", "_").replace(".pth", "") + "_" + direction


def infer(inkenv, models):
    for direction in ("forward", "reverse"):
        for ck in CKPTS:
            out = D / f"pred_{tag(ck, direction)}.tif"
            if out.exists():
                continue
            cmd = [f"{inkenv}/bin/python", "-m", "koine_machines.inference.infer", str(D / "w00_9366.zarr"),
                   f"{models}/ink_9um/{ck}", str(out), "--no-compile", "--batch-size", "16", "--direction", direction]
            r = subprocess.run(cmd, capture_output=True, text=True)
            # infer writes <out> for a single direction; some versions suffix _reverse
            alt = out.with_name(out.stem + "_reverse.tif")
            if not out.exists() and alt.exists():
                alt.rename(out)
            print(f"{tag(ck, direction)}: {'ok' if out.exists() else 'FAILED ' + r.stderr[-300:]}", flush=True)


def labels(shape):
    """Organiser labels (2.403 µm grid) box-resampled onto the 9.366 µm canvas."""
    out = []
    for name in ("inklabels", "supervision"):
        a = np.load(D / f"w00_{name}_2um.npy").astype(np.float32) / 255
        out.append(np.asarray(Image.fromarray(a).resize((shape[1], shape[0]), Image.BOX)) >= 0.5)
    return out


def auc(score, label):
    h1, _ = np.histogram(score[label], bins=256, range=(0, 1))
    h0, _ = np.histogram(score[~label], bins=256, range=(0, 1))
    c0 = np.cumsum(h0) - h0  # negatives strictly below each bin
    return float(((h1 * (c0 + 0.5 * h0)).sum()) / (h1.sum() * h0.sum()))


def text_score(p, valid, tile_mm=20.0, step_mm=5.0):
    t, s = int(tile_mm * 1000 / UM), int(step_mm * 1000 / UM)
    peaks = []
    for r0 in range(0, p.shape[0] - t + 1, s):
        for c0 in range(0, p.shape[1] - t + 1, s):
            sl = (slice(r0, r0 + t), slice(c0, c0 + t))
            d = line_score(p[sl], valid[sl], UM)
            if d["coverage"] >= 0.6 and np.isfinite(d.get("peak", np.nan)) and mask_peak(valid[sl], UM) <= 1.0:
                peaks.append(d["peak"])
    return (float(np.median(peaks)) if peaks else np.nan), len(peaks)


def text_score_v3(p, valid, tile_mm=20.0, step_mm=5.0):
    """Exploratory (added after PREREG deviation 4): median v3 profile score over the same clean tiles."""
    band = tuple(json.load(open("data/bench/line_period.json"))[k] for k in ("p10_mm", "p90_mm"))
    t, s = int(tile_mm * 1000 / UM), int(step_mm * 1000 / UM)
    vals = []
    for r0 in range(0, p.shape[0] - t + 1, s):
        for c0 in range(0, p.shape[1] - t + 1, s):
            sl = (slice(r0, r0 + t), slice(c0, c0 + t))
            if valid[sl].mean() >= 0.6 and mask_peak(valid[sl], UM) <= 1.0:
                v = line_profile_score(p[sl], valid[sl], UM, band_mm=band)["s"]
                if np.isfinite(v):
                    vals.append(v)
    return float(np.median(vals)) if vals else np.nan


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def score():
    vol0 = zarr.open_group(str(D / "w00_9366.zarr"), "r")["0"]
    valid = vol0[vol0.shape[0] // 2] > 0
    lab, sup = labels(valid.shape)
    m = sup & valid
    rows = []
    for ck in CKPTS:
        for direction in ("reverse", "forward"):
            f = D / f"pred_{tag(ck, direction)}.tif"
            if not f.exists():
                continue
            p = tifffile.imread(f).astype(np.float32) / 255
            pv = p[valid]
            sep = float(pv[pv > 0.5].mean() - pv[pv <= 0.5].mean()) if (pv > 0.5).any() else 0.0
            ts, ntiles = text_score(p, valid)
            rows.append(dict(ckpt=ck, direction=direction, auc=round(auc(p[m], lab[m]), 4),
                             text=round(ts, 4), tiles=ntiles, separation=round(sep, 4),
                             text_v3=round(text_score_v3(p, valid), 4)))
            print(rows[-1], flush=True)
    (D / "scores.json").write_text(json.dumps(rows, indent=1))

    rev = [r for r in rows if r["direction"] == "forward"]
    a = [r["auc"] for r in rev]
    print(f"\nforward direction, {len(rev)} checkpoints: AUC {min(a):.3f}..{max(a):.3f}")
    for k in ("text", "separation", "text_v3"):
        print(f"  Spearman(AUC, {k}) = {spearman(a, [r[k] for r in rev]):+.3f}{'  (exploratory)' if k == 'text_v3' else ''}")
    best = max(rev, key=lambda r: r["auc"])
    for k in ("text", "separation", "text_v3"):
        pick = max(rev, key=lambda r: r[k])
        print(f"  checkpoint picked by {k}: {pick['ckpt']} AUC {pick['auc']:.3f} (best {best['auc']:.3f}, median {np.median(a):.3f})")
    pairs = {}
    for r in rows:
        pairs.setdefault(r["ckpt"], {})[r["direction"]] = r
    full = [v for v in pairs.values() if len(v) == 2]
    for k in ("text", "separation", "text_v3"):
        right = sum((v["reverse"][k] > v["forward"][k]) == (v["reverse"]["auc"] > v["forward"]["auc"]) for v in full)
        print(f"  direction picked correctly by {k}: {right}/{len(full)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer-only", action="store_true"); ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if not a.score_only:
        infer(os.environ["INKENV"], os.environ["MODELS"])
    if not a.infer_only:
        score()


if __name__ == "__main__":
    main()
