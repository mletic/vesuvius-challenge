"""PREREG step 1: line spacing measured from the organisers' ink labels only.

For each labelled segment (labels at 9.6 µm, level 2 of the 2.4 µm grid), in 5 mm wide
column strips, take the row profile of label ink over runs of >= 10 mm with supervision
(PREREG deviation 1),
and the first autocorrelation peak between 2.5 and 10 mm. Writes data/bench/line_period.json.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

PX_UM = 9.6
spacings, per_seg = [], {}
for f in sorted(glob.glob("data/bench/labels/*.npz")):
    z = np.load(f); ink, sup = z["ink"], z["sup"]
    w = int(5_000 / PX_UM); minrun = int(10_000 / PX_UM)  # PREREG deviation 1 (was 10 mm strips, 15 mm runs)
    lo, hi = int(2500 / PX_UM), int(10_000 / PX_UM)
    seg = []
    for c0 in range(0, ink.shape[1] - w + 1, w // 2):
        s = sup[:, c0:c0 + w].mean(1) > 0.5
        prof = np.where(s, (ink[:, c0:c0 + w] & sup[:, c0:c0 + w]).sum(1) / np.maximum(sup[:, c0:c0 + w].sum(1), 1), np.nan)
        # contiguous supervised runs
        idx = np.flatnonzero(s)
        if idx.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        for run in np.split(idx, breaks + 1):
            if run.size < minrun:
                continue
            x = prof[run] - np.nanmean(prof[run])
            ac = np.correlate(x, x, "full")[x.size - 1:]
            ac = ac / ac[0] if ac[0] > 0 else ac
            pk, props = find_peaks(ac[lo:hi], prominence=0.05)
            if pk.size:
                best = pk[np.argmax(props["prominences"])] + lo
                seg.append(best * PX_UM / 1000)
    per_seg[Path(f).stem] = seg
    spacings += seg
sp = np.array(spacings)
res = dict(n=int(sp.size), median_mm=float(np.median(sp)), p10_mm=float(np.percentile(sp, 10)),
           p90_mm=float(np.percentile(sp, 90)), per_segment_median={k: (float(np.median(v)) if v else None) for k, v in per_seg.items()})
Path("data/bench/line_period.json").write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "per_segment_median"})
for k, v in res["per_segment_median"].items():
    print(f"  {k[:55]:55s} {v if v is None else round(v, 2)}  (n={len(per_seg[k])})")
