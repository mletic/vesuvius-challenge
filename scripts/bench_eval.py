"""Evaluate the pre-registered outcomes P1-P6 (see PREREG.md). Writes data/bench/results.json.

usage: uv run python scripts/bench_eval.py [--skip-b2] [--skip-b3]
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw, ImageFont

from vesuvius_sweep.microseg import SOURCES
from vesuvius_sweep.textscore import line_profile_score, line_score, line_z, mask_peak

BAND = tuple(json.load(open("data/bench/line_period.json"))[k] for k in ("p10_mm", "p90_mm"))
SPACING = json.load(open("data/bench/line_period.json"))["median_mm"]
TILE_MM, STEP_MM = 20.0, 10.0
RES = {}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    # average ties
    order = np.argsort(allv, kind="mergesort"); s = allv[order]; r = np.empty_like(ranks)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pix_auc(score, label):
    h1, _ = np.histogram(score[label], bins=256, range=(0, 1))
    h0, _ = np.histogram(score[~label], bins=256, range=(0, 1))
    c0 = np.cumsum(h0) - h0
    return float((h1 * (c0 + 0.5 * h0)).sum() / max(h1.sum() * h0.sum(), 1))


def tiles(p, valid, um, fn):
    t, s = int(TILE_MM * 1000 / um), int(STEP_MM * 1000 / um)
    out = []
    for r0 in range(0, max(1, p.shape[0] - t + 1), s):
        for c0 in range(0, max(1, p.shape[1] - t + 1), s):
            sl = (slice(r0, r0 + t), slice(c0, c0 + t))
            v = valid[sl]
            if v.mean() < 0.6 or mask_peak(v, um) > 1.0:
                continue
            out.append(((r0, c0), fn(p[sl], v)))
    return out


def tile_metrics(pt, v, um):
    z = line_z(pt, v, um, band_mm=BAND)
    s = line_score(pt, v, um, period_mm=BAND)
    pv = pt[v]
    return dict(s3=line_profile_score(pt, v, um, band_mm=BAND)["s"], z=z["z"], peak=s.get("peak", np.nan), aniso=s.get("score", np.nan), mean=float(pv.mean()),
                frac=float((pv > 0.5).mean()),
                sep=float(pv[pv > 0.5].mean() - pv[pv <= 0.5].mean()) if (pv > 0.5).any() and (pv <= 0.5).any() else 0.0)


def map_sep(p, valid):
    pv = p[valid]
    return float(pv[pv > 0.5].mean() - pv[pv <= 0.5].mean()) if (pv > 0.5).any() else 0.0


def load_labels(name, shape):
    z = np.load(f"data/bench/labels/{name}.npz")
    rs = lambda a: np.asarray(Image.fromarray(a.astype(np.float32)).resize((shape[1], shape[0]), Image.BOX)) >= 0.5
    return rs(z["ink"]), rs(z["sup"])


# ---------------------------------------------------------------- B1
def b1():
    segs = []
    for d in sorted(Path("data/bench/b1").iterdir()):
        f, r = d / "pred_forward.tif", d / "pred_reverse.tif"
        if not (f.exists() and r.exists()):
            continue
        um = float(json.loads((d / "vol.zarr" / ".zattrs").read_text())["multiscales"][0]["datasets"][0]
                   ["coordinateTransformations"][0]["scale"][-1])
        vol = zarr.open_group(str(d / "vol.zarr"), "r")["0"]
        valid = vol[vol.shape[0] // 2] > 0
        P = {k: tifffile.imread(x).astype(np.float32) / 255 for k, x in (("forward", f), ("reverse", r))}
        ink, sup = load_labels(d.name, valid.shape)
        best = None
        for dy in range(-16, 17, 4):
            for dx in range(-16, 17, 4):
                L, S = np.roll(ink, (dy, dx), (0, 1)), np.roll(sup, (dy, dx), (0, 1)) & valid
                a = {k: pix_auc(P[k][S], L[S]) for k in P}
                if best is None or a["forward"] + a["reverse"] > sum(best[0].values()):
                    best = (a, dy, dx, L, S)
        a, dy, dx, L, S = best
        text_dir = max(a, key=a.get)
        ambiguous = abs(a["forward"] - a["reverse"]) < 0.05 or max(a.values()) < 0.60
        contrast = float(P[text_dir][S & L].mean() - P[text_dir][S & ~L].mean())
        tl = {k: tiles(P[k], valid, um, lambda pt, v: tile_metrics(pt, v, um)) for k in P}
        mapscore = {k: {m: float(np.nanmedian([t[1][m] for t in tl[k]])) if tl[k] else np.nan
                        for m in ("s3", "z", "peak", "aniso", "mean", "frac")} | {"sep_map": map_sep(P[k], valid)} for k in P}
        segs.append(dict(name=d.name, scroll=d.name.split("_")[0], um=um, auc=a, shift=(dy, dx), text_dir=text_dir,
                         ambiguous=bool(ambiguous), contrast=contrast, mapscore=mapscore,
                         tiles={k: [t[1] for t in tl[k]] for k in tl}))
        print(f"B1 {d.name[:48]:48s} AUC f {a['forward']:.3f} r {a['reverse']:.3f} -> {text_dir}{' (ambiguous)' if ambiguous else ''}"
              f"  tiles {len(tl['forward'])}  s3 f {mapscore['forward']['s3']:+.2f} r {mapscore['reverse']['s3']:+.2f}", flush=True)
    RES["b1_segments"] = [{k: v for k, v in s.items() if k != "tiles"} for s in segs]
    use = [s for s in segs if not s["ambiguous"]]
    other = lambda s: "reverse" if s["text_dir"] == "forward" else "forward"
    # P1 direction choice
    p1 = {}
    for m in ("s3", "z", "peak", "aniso", "mean", "frac", "sep_map"):
        right = [s["mapscore"][s["text_dir"]][m] > s["mapscore"][other(s)][m] for s in use
                 if np.isfinite(s["mapscore"][s["text_dir"]][m]) and np.isfinite(s["mapscore"][other(s)][m])]
        p1[m] = dict(correct=int(sum(right)), n=len(right))
    RES["P1_direction"] = p1
    # P2 pooled tile AUC; P4 LOSO threshold
    p2, p4 = {}, {}
    for m in ("s3", "z", "peak", "aniso", "mean", "frac", "sep"):
        pos = [t[m] for s in use for t in s["tiles"][s["text_dir"]]]
        neg = [t[m] for s in use for t in s["tiles"][other(s)]]
        p2[m] = dict(auc=auc(pos, neg), n_pos=len(pos), n_neg=len(neg),
                     per_scroll={sc: auc([t[m] for s in use if s["scroll"] == sc for t in s["tiles"][s["text_dir"]]],
                                         [t[m] for s in use if s["scroll"] == sc for t in s["tiles"][other(s)]])
                                 for sc in sorted({s["scroll"] for s in use})})
        fp = tp = nneg = npos = 0
        for sc in sorted({s["scroll"] for s in use}):
            train_neg = [t[m] for s in use if s["scroll"] != sc for t in s["tiles"][other(s)] if np.isfinite(t[m])]
            if not train_neg:
                continue
            thr = np.percentile(train_neg, 95)
            hn = [t[m] for s in use if s["scroll"] == sc for t in s["tiles"][other(s)] if np.isfinite(t[m])]
            hp = [t[m] for s in use if s["scroll"] == sc for t in s["tiles"][s["text_dir"]] if np.isfinite(t[m])]
            fp += sum(x > thr for x in hn); nneg += len(hn); tp += sum(x > thr for x in hp); npos += len(hp)
        p4[m] = dict(fpr=fp / max(nneg, 1), tpr=tp / max(npos, 1), n_neg=nneg, n_pos=npos)
    RES["P2_tile_auc"], RES["P4_loso"] = p2, p4
    RES["real_contrast"] = dict(median=float(np.median([s["contrast"] for s in use])) if use else np.nan,
                                per_segment={s["name"]: s["contrast"] for s in use})
    return segs, use


# ---------------------------------------------------------------- B2
def b2():
    pos, neg, pos_lab, neg_lab = [], [], [], []
    for f in sorted(glob.glob("data/bench/b2/*.npy")):
        p = np.load(f).astype(np.float32) / 255
        valid = p > 0.01
        name = Path(f).stem
        lab = np.load(f"data/bench/labels/{name}.npz")
        ink = lab["ink"][:p.shape[0], :p.shape[1]]; sup = lab["sup"][:p.shape[0], :p.shape[1]]
        t, s = int(TILE_MM * 1000 / 9.6), int(STEP_MM * 1000 / 9.6)
        for r0 in range(0, max(1, p.shape[0] - t + 1), s):
            for c0 in range(0, max(1, p.shape[1] - t + 1), s):
                sl = (slice(r0, r0 + t), slice(c0, c0 + t))
                v = valid[sl]
                if v.shape != (t, t) or v.mean() < 0.6 or mask_peak(v, 9.6) > 1.0:
                    continue
                a = line_profile_score(p[sl], v, 9.6, band_mm=BAND)["s"]
                b = line_profile_score(np.rot90(p[sl]), np.rot90(v), 9.6, band_mm=BAND)["s"]
                pos.append(a); neg.append(b)
                if ink[sl].shape == (t, t) and (ink[sl] & sup[sl]).mean() >= 0.02:
                    pos_lab.append(a); neg_lab.append(b)
        print(f"B2 {name[:50]:50s} tiles so far {len(pos)}", flush=True)
    RES["P3_rotation_auc"] = dict(all_tiles=auc(pos, neg), n=len(pos), label_confirmed=auc(pos_lab, neg_lab), n_label=len(pos_lab))


# ---------------------------------------------------------------- B3
def glyph_mask(shape, um, rng):
    font_px = int(round(2.0 * 1000 / um / 0.70))  # Times cap height ~0.70 em -> 2.0 mm letters
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", font_px)
    im = Image.new("L", (shape[1], shape[0]), 0); d = ImageDraw.Draw(im)
    letters = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    step = int(SPACING * 1000 / um)
    y = int(rng.integers(0, step))
    while y < shape[0]:
        line = "".join(letters[i] if rng.random() > 0.15 else " " for i in rng.integers(0, len(letters), 60))
        d.text((int(rng.integers(-font_px, 0)), y), line, fill=255, font=font)
        y += step
    return np.asarray(im, np.float32) / 255


def b3(segs, use):
    amps = [0, 0.02, 0.05, 0.1, 0.2, 0.4]
    rng = np.random.default_rng(7)
    all_neg = [t["s3"] for s in use for t in s["tiles"]["reverse" if s["text_dir"] == "forward" else "forward"] if np.isfinite(t["s3"])]
    thr = float(np.percentile(all_neg, 95)) if all_neg else np.nan
    det = {a: [0, 0] for a in amps}
    for s in use:
        d = Path("data/bench/b1") / s["name"]
        wrong = "reverse" if s["text_dir"] == "forward" else "forward"
        p = tifffile.imread(d / f"pred_{wrong}.tif").astype(np.float32) / 255
        vol = zarr.open_group(str(d / "vol.zarr"), "r")["0"]; valid = vol[vol.shape[0] // 2] > 0
        g = glyph_mask(p.shape, s["um"], rng)
        for a in amps:
            q = np.clip(p + a * g, 0, 1)
            for _, z in tiles(q, valid, s["um"], lambda pt, v: line_profile_score(pt, v, s["um"], band_mm=BAND)["s"]):
                if np.isfinite(z):
                    det[a][0] += z > thr; det[a][1] += 1
        print(f"B3 {s['name'][:48]} done", flush=True)
    RES["P5_injection"] = dict(threshold=thr, detection={str(a): det[a][0] / max(det[a][1], 1) for a in amps},
                               n_tiles={str(a): det[a][1] for a in amps})


# ---------------------------------------------------------------- B4
def b4(use):
    null = np.array([t["s3"] for s in use for t in s["tiles"]["reverse" if s["text_dir"] == "forward" else "forward"]
                     if np.isfinite(t["s3"])])
    fields = []
    for scroll in ("PHerc1203", "PHerc0846A"):
        p = Path(f"data/triage_{scroll}.csv")
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            if r.get("clean") != "1":
                continue
            stem = Path("data/grow") / r["name"]
            preds = [tifffile.imread(x).astype(np.float32) / 255 for x in sorted(stem.parent.glob(stem.name + "_ink9_*.tif"))]
            if not preds:
                continue
            pm = np.mean(preds, 0); valid = np.asarray(Image.open(f"{stem}_mid.png").convert("L")) > 0
            z = line_profile_score(pm, valid, SOURCES[scroll][3], band_mm=BAND)["s"]
            pv = (1 + (null >= z).sum()) / (1 + len(null)) if np.isfinite(z) and len(null) else np.nan
            fields.append(dict(name=r["name"], z=z, p=float(pv)))
    ps = np.array([f["p"] for f in fields])
    order = np.argsort(ps); m = len(ps); passed = set()
    for k, i in enumerate(order, 1):
        if ps[i] <= 0.10 * k / m:
            passed = set(order[:k])
    for i, f in enumerate(fields):
        f["bh_pass"] = i in passed
    RES["P6_search"] = dict(n_clean_fields=len(fields), n_pass_bh=len(passed), fields=fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-b2", action="store_true"); ap.add_argument("--skip-b3", action="store_true")
    a = ap.parse_args()
    RES["band_mm"], RES["spacing_mm"] = BAND, SPACING
    segs, use = b1()
    if not a.skip_b2:
        b2()
    if not a.skip_b3 and use:
        b3(segs, use)
    if use:
        b4(use)
    Path("data/bench/results.json").write_text(json.dumps(RES, indent=1, default=float))
    print(json.dumps({k: v for k, v in RES.items() if k.startswith("P")}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
