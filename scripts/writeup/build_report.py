"""Build the write-up page (report/index.html) from the results in data/.

Figures are embedded as JPEG data URIs; the triage chart is drawn as inline SVG
from data/triage_*.csv; the checkpoint-ranking section is filled from
data/ckrank/scores.json when it exists.

usage: uv run python scripts/writeup/build_report.py
"""

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("report/index.html")
DOCS = Path("docs/index.html")


def img(path, width=1100, quality=78):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(path, alt, caption, width=1100):
    return (f'<figure><img src="{img(path, width)}" alt="{html.escape(alt)}" loading="lazy">'
            f'<figcaption>{caption}</figcaption></figure>')


def triage_chart():
    rows = []
    for scroll in ("PHerc1203", "PHerc0846A"):
        p = Path(f"data/triage_{scroll}.csv")
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            try:
                pk = float(r["peak"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(pk):
                rows.append((scroll, pk, r.get("clean") == "1"))
    lo, hi = -2.5, 2.5
    W, H, left, right, top = 720, 230, 110, 20, 28
    x = lambda v: left + (min(max(v, lo), hi) - lo) / (hi - lo) * (W - left - right)
    lanes = {"PHerc1203": 95, "PHerc0846A": 165}
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Line-periodicity scores of triage fields" class="chart">']
    # reference bands
    parts.append(f'<rect x="{x(-0.9):.1f}" y="{top}" width="{x(0.35) - x(-0.9):.1f}" height="{H - top - 40}" class="band-blank"/>')
    parts.append(f'<rect x="{x(1.5):.1f}" y="{top}" width="{x(1.9) - x(1.5):.1f}" height="{H - top - 40}" class="band-text"/>')
    parts.append(f'<text x="{(x(-0.9) + x(0.35)) / 2:.1f}" y="{top - 8}" class="lab" text-anchor="middle">presumed blank</text>')
    parts.append(f'<text x="{(x(1.5) + x(1.9)) / 2:.1f}" y="{top - 8}" class="lab" text-anchor="middle">known text</text>')
    for v in range(-2, 3):
        parts.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{H - 40}" class="grid"/>')
        parts.append(f'<text x="{x(v):.1f}" y="{H - 22}" class="tick" text-anchor="middle">{v:+d}</text>')
    parts.append(f'<text x="{(left + W - right) / 2:.1f}" y="{H - 4}" class="lab" text-anchor="middle">peakness score (20 mm field, 9.6 µm)</text>')
    rng = np.random.default_rng(3)
    for scroll, y in lanes.items():
        parts.append(f'<text x="{left - 12}" y="{y + 4}" class="lane" text-anchor="end">{scroll}</text>')
        for s, pk, clean in rows:
            if s != scroll:
                continue
            jy = y + rng.uniform(-14, 14)
            cls = "dot-clean" if clean else "dot-flag"
            parts.append(f'<circle cx="{x(pk):.1f}" cy="{jy:.1f}" r="5" class="{cls}"><title>{scroll} peak {pk:+.2f}'
                         f'{"" if clean else " (fails a geometry guard)"}</title></circle>')
    parts.append("</svg>")
    n = len(rows)
    n_clean = sum(c for _, _, c in rows)
    n_hi_clean = sum(c and pk >= 1.0 for _, pk, c in rows)
    n_textrange = sum(pk >= 1.5 for _, pk, _ in rows)
    n_textrange_clean = sum(c and pk >= 1.5 for _, pk, c in rows)
    return "".join(parts), n, n_clean, n_hi_clean, n_textrange, n_textrange_clean


def ckrank_section():
    p = Path("data/ckrank/scores.json")
    if not p.exists():
        return ('<p class="note">This experiment is running now: 14 checkpoints × 2 depth directions on the '
                '9.366 µm volume of PHerc0841 w00. This section will hold the rank correlations.</p>')
    rows = json.loads(p.read_text())
    fw = [r for r in rows if r["direction"] == "forward"]

    def sp(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    a = [r["auc"] for r in fw]
    s_text, s_sep = sp(a, [r["text"] for r in fw]), sp(a, [r["separation"] for r in fw])
    pairs = {}
    for r in rows:
        pairs.setdefault(r["ckpt"], {})[r["direction"]] = r
    full = [v for v in pairs.values() if len(v) == 2]
    right = {k: sum((v["forward"][k] > v["reverse"][k]) == (v["forward"]["auc"] > v["reverse"]["auc"]) for v in full)
             for k in ("text", "separation")}
    body = "".join(f'<tr><td>{html.escape(r["ckpt"].replace("hybrid_3d2d-", "").replace(".pth", ""))}</td>'
                   f'<td>{r["auc"]:.3f}</td><td>{r["text"]:+.2f}</td><td>{r["separation"]:.3f}</td></tr>'
                   for r in sorted(fw, key=lambda r: -r["auc"]))
    return (f'<p>Across {len(fw)} checkpoints (forward direction, AUC {min(a):.3f} to {max(a):.3f}), the rank correlation '
            f'with label AUC is <strong>{s_text:+.2f}</strong> for the text score and <strong>{s_sep:+.2f}</strong> for the '
            f'separation rule from villa #1867. Picking the depth direction: text score right in {right["text"]} of {len(full)} '
            f'checkpoints, separation rule in {right["separation"]} of {len(full)}.</p>'
            f'<div class="table-wrap"><table><thead><tr><th>checkpoint</th><th>AUC (labels)</th><th>text score</th>'
            f'<th>separation</th></tr></thead><tbody>{body}</tbody></table></div>')


def orient_table():
    p = Path("data/orient/results_9um.json")
    if not p.exists():
        return ""
    r = json.loads(p.read_text())
    by = {(x["depth"], x["transform"]): x["auc"] for x in r}
    names = [x["transform"] for x in r if x["depth"] == "as-published"]
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    rows = "".join(f'<tr><td>{d}</td>' + "".join(f'<td class="n">{by[(d, n)]:.3f}</td>' for n in names) + "</tr>"
                   for d in ("as-published", "depth-reversed"))
    return (f'<div class="table-wrap"><table><thead><tr><th>depth order</th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


SCORE_NAMES = {"s3": "v3 row-profile (primary)", "z": "v2 red-noise", "peak": "v1 peakness", "aniso": "anisotropy",
               "mean": "mean ink", "frac": "fraction > 0.5", "sep": "separation (#1867)", "sep_map": "separation (#1867)"}


def bench_section():
    p = Path("data/bench/results.json")
    if not p.exists():
        return '<p class="note">The benchmark is running; this section fills in from data/bench/results.json.</p>'
    R = json.loads(p.read_text())
    segs = R.get("b1_segments", [])
    use = [x for x in segs if not x["ambiguous"]]
    out = [f'<p>B1 used {len(segs)} labelled segments; {len(use)} had a clear text direction '
           f'(AUC gap ≥ 0.05 and best AUC ≥ 0.60). Line band {R["band_mm"][0]:.2f}–{R["band_mm"][1]:.2f} mm, '
           f'measured from the labels.</p>']
    seg_rows = "".join(f'<tr><td>{html.escape(x["name"][:46])}</td><td class="n">{x["auc"]["forward"]:.3f}</td>'
                       f'<td class="n">{x["auc"]["reverse"]:.3f}</td><td>{x["text_dir"]}'
                       f'{" (ambiguous)" if x["ambiguous"] else ""}</td></tr>' for x in segs)
    out.append(f'<div class="table-wrap"><table><thead><tr><th>segment</th><th>AUC forward</th><th>AUC reverse</th>'
               f'<th>text direction</th></tr></thead><tbody>{seg_rows}</tbody></table></div>')
    p1, p2, p4 = R.get("P1_direction", {}), R.get("P2_tile_auc", {}), R.get("P4_loso", {})
    keys = [k for k in ("s3", "z", "peak", "aniso", "mean", "frac") if k in p2] + ["sep"]
    rows = ""
    for k in keys:
        d1 = p1.get(k if k != "sep" else "sep_map", {})
        rows += (f'<tr><td>{SCORE_NAMES[k]}</td><td class="n">{d1.get("correct", "–")}/{d1.get("n", "–")}</td>'
                 f'<td class="n">{p2.get(k, {}).get("auc", float("nan")):.3f}</td>'
                 f'<td class="n">{p4.get(k, {}).get("fpr", float("nan")):.3f}</td>'
                 f'<td class="n">{p4.get(k, {}).get("tpr", float("nan")):.3f}</td></tr>')
    out.append('<div class="table-wrap"><table><thead><tr><th>score</th><th>P1 direction correct</th>'
               '<th>P2 tile AUC</th><th>P4 held-out FPR</th><th>held-out TPR</th></tr></thead>'
               f'<tbody>{rows}</tbody></table></div>')
    if "P3_rotation_auc" in R:
        q = R["P3_rotation_auc"]
        out.append(f'<p><strong>P3.</strong> Published 2.4 µm prediction tiles versus their 90°-rotated copies: AUC '
                   f'{q["all_tiles"]:.3f} over {q["n"]} tiles; {q["label_confirmed"]:.3f} over the {q["n_label"]} tiles '
                   f'where labels confirm ink.</p>')
    if "P5_injection" in R:
        q = R["P5_injection"]
        det = " · ".join(f"{float(a):g}: {v:.0%}" for a, v in q["detection"].items())
        out.append(f'<p><strong>P5.</strong> Synthetic Greek text added to wrong-direction maps, detection rate at the '
                   f'calibrated threshold ({q["threshold"]:.2f}) by added amplitude: {det}. Real text contrast in the '
                   f'text-direction maps (labelled ink minus supervised non-ink): median {R["real_contrast"]["median"]:.3f}.</p>')
    if "P6_search" in R:
        q = R["P6_search"]
        out.append(f'<p><strong>P6.</strong> Of {q["n_clean_fields"]} clean search fields on PHerc1203 and PHerc0846A, '
                   f'{q["n_pass_bh"]} pass the calibrated test after Benjamini–Hochberg at q = 0.10.</p>')
    return "".join(out)


def bench_explore():
    p = Path("data/bench/exploratory_label_confirmed.json")
    if not p.exists():
        return ""
    e = json.loads(p.read_text())
    return (f'<p><strong>Exploratory, not pre-registered.</strong> Labels cover only 1–13% of each segment, so most '
            f'"text-direction" tiles hold no text. Restricted to the {e["n_label"]} tiles where labels confirm ink, v3 '
            f'still reaches only AUC {e["auc_label"]:.3f} against all {e["n_neg"]} wrong-direction tiles, and flags '
            f'{e["tpr_label_at_5pct"]:.0%} of them at a 5% false-positive rate. Injection agrees: synthetic lines on top of '
            f'real 9 µm output are detected only when they are stronger than real ink. The most line-like wrong-direction '
            f'tiles are plain blob noise, not text from the next winding.</p>'
            '<p><strong>Reading the result.</strong> The score recognises lines in crisp maps (P3) but not in 9 µm '
            'outputs, where ink comes out as millimetre blobs. Any label-free text test on 9 µm maps needs more than '
            'three line periods per tile, or a sharper ink model. For choosing the depth direction, use the separation '
            'rule: it was right on every segment here.</p>')


def main():
    chart, n, n_clean, n_hi_clean, n_tr, n_tr_clean = triage_chart()
    figs = {
        "v9": figure("data/control/cmp_ink.png", "Ink predictions at 9 µm: published segment versus mesh-free surface",
                     "PHerc0139 w043, same 7.2 mm window. Left: ink_9um on the published, hand-traced segment. Middle: this "
                     "pipeline's sampler run along the published mesh (r = 0.57 with the left map). Right: the same with the "
                     "depth order reversed (r = 0.13). The fully mesh-free surface from the tracker gives r = 0.55."),
        "v2": figure("data/control3/_cmp2.png", "CT renders and 2 µm ink predictions, published versus mesh-free",
                     "PHerc0841 w00, 4.9 mm window. Top: CT at the recto surface, published segment (left) and mesh-free "
                     "surface (right). Bottom: ink_canonical_2um on each. After the −16° rotation between the two frames, "
                     "r = 0.72 (other rotations 0.08 to 0.23)."),
        "text": figure("data/published_ink/paris4_20231005123336_linescore.png", "Line-periodicity score over a published ink map",
                       "Line-periodicity score on the published 2.4 µm prediction of PHerc. Paris 4 segment 20231005123336, "
                       "20 mm tiles every 5 mm. Red marks high scores: the text columns light up and the blank margin "
                       "between them stays dark.", 1300),
        "fp1": figure("data/grow/PHerc1203_z3223_y3724_x4235_S1536_triage_view.png", "Torn PHerc1203 field with a high score",
                      "A rejected hit (PHerc1203, peak +1.82). CT at the surface, 9 µm ink, and the row profile. The surface "
                      "is torn; the model fires on frayed fragment edges. Coverage 0.58 fails the 0.6 guard."),
        "fp2": figure("data/grow/PHerc0846A_z3215_y2915_x4265_S1536_triage_view.png", "PHerc0846A field whose outline drives the score",
                      "A rejected hit (PHerc0846A, peak +1.56). The field's own hole and outline map scores +2.34, so the "
                      "periodicity comes from geometry. Its 1 cm read at 2.4 µm has 6% sparse ink blobs, several beside holes."),
    }
    page = TEMPLATE
    for k, v in figs.items():
        page = page.replace("{{" + k + "}}", v)
    page = (page.replace("{{chart}}", chart).replace("{{n}}", str(n)).replace("{{n_clean}}", str(n_clean))
            .replace("{{n_hi_clean}}", str(n_hi_clean)).replace("{{ckrank}}", ckrank_section())
            .replace("{{orient_table}}", orient_table())
            .replace("{{n_tr}}", str(n_tr)).replace("{{n_tr_clean}}", str(n_tr_clean))
            .replace("{{bench}}", bench_section()).replace("{{bench_explore}}", bench_explore()))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
    # Standalone copy for GitHub Pages: the artifact host adds the document skeleton, Pages does not.
    DOCS.parent.mkdir(exist_ok=True)
    DOCS.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                    '<meta name="description" content="Mesh-free surface volumes from the published surface predictions, '
                    'a pre-registered benchmark of label-free ink diagnostics, and a search of two unread scrolls.">\n'
                    '<style>*,*::before,*::after{box-sizing:border-box}html{-webkit-text-size-adjust:100%}'
                    'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
                    + page.replace("<main", "</head>\n<body>\n<main", 1) + "\n</body>\n</html>\n")
    print(DOCS, f"{DOCS.stat().st_size / 1e6:.1f} MB")


TEMPLATE = Path(__file__).with_name("report_template.html").read_text()

if __name__ == "__main__":
    main()
