"""Text-structure scores for ink-probability maps in published layout.

In published layout the scroll axis runs down the rows, so lines of writing run
horizontally and repeat vertically every ~4-7 mm. The line score compares
spectral power in that period band along the vertical direction (within
±`cone` degrees) against the same band along the horizontal direction; blobs
and clouds are roughly isotropic and score near 0, text lines score > 0.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def to_grid(p: np.ndarray, valid: np.ndarray, px_um: float, grid_um: float = 40.0):
    """Block-average an ink map (and its mask) to ~grid_um pixels."""
    f = max(1, int(round(grid_um / px_um)))
    H, W = (p.shape[0] // f) * f, (p.shape[1] // f) * f
    pv = (p[:H, :W] * valid[:H, :W]).reshape(H // f, f, W // f, f).sum((1, 3))
    vv = valid[:H, :W].reshape(H // f, f, W // f, f).sum((1, 3))
    return np.where(vv > 0, pv / np.maximum(vv, 1), 0), vv > f * f / 2, px_um * f


def line_score(p: np.ndarray, valid: np.ndarray, px_um: float,
               period_mm=(3.5, 7.5), cone: float = 20.0) -> dict:
    """Directional band-power ratio. Returns dict(score, vert, horiz, coverage)."""
    g, m, um = to_grid(p.astype(np.float32), valid, px_um)
    if m.mean() < 0.3 or m.sum() < 400:
        return dict(score=np.nan, vert=np.nan, horiz=np.nan, coverage=float(m.mean()))
    x = np.where(m, g - g[m].mean(), 0)
    # Soft taper at mask and image borders to avoid edge ringing.
    w = ndi.gaussian_filter(m.astype(np.float32), 3) * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * w))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(x.shape[0], d=um / 1000))  # cycles / mm
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=um / 1000))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    r = np.hypot(FY, FX)
    band = (r >= 1 / period_mm[1]) & (r <= 1 / period_mm[0])
    ang = np.degrees(np.arctan2(np.abs(FX), np.abs(FY)))  # 0 = vertical frequency (horizontal lines)
    vert = P[band & (ang <= cone)].mean()
    horiz = P[band & (ang >= 90 - cone)].mean()
    # Peakness: line spacing is regular, so text puts a bump in the period band
    # relative to its flanks; cracks and fibres give a smooth 1/f fall-off.
    flank = ((r >= 1 / 15.0) & (r < 1 / period_mm[1])) | ((r > 1 / period_mm[0]) & (r <= 1 / 2.0))
    fv, fh = P[flank & (ang <= cone)].mean(), P[flank & (ang >= 90 - cone)].mean()
    if min(vert, horiz, fv, fh) <= 0:  # featureless input (e.g. an all-zero map)
        return dict(score=np.nan, peak=np.nan, peak_v=np.nan, vert=float(vert), horiz=float(horiz), coverage=float(m.mean()))
    peak_v, peak_h = vert / fv, horiz / fh
    return dict(score=float(np.log(vert / horiz)), peak=float(np.log(peak_v / peak_h)),
                peak_v=float(np.log(peak_v)), vert=float(vert), horiz=float(horiz), coverage=float(m.mean()))


def mask_peak(valid: np.ndarray, px_um: float) -> float:
    """Peakness of the surface's own hole/outline map: high values mean tears or the
    field outline alone produce horizontal periodicity. 0 when there are hardly any holes."""
    if (~valid).mean() < 0.01:
        return 0.0
    return line_score((~valid).astype(np.float32), np.ones_like(valid), px_um).get("peak", np.nan)


def tile_scores(p: np.ndarray, valid: np.ndarray, px_um: float, tile_mm: float = 20.0, step_mm: float = 10.0):
    """Line scores on a sliding grid of square tiles; returns list of (row0, col0, score dict)."""
    t, s = int(tile_mm * 1000 / px_um), int(step_mm * 1000 / px_um)
    out = []
    for r0 in range(0, max(1, p.shape[0] - t + 1), s):
        for c0 in range(0, max(1, p.shape[1] - t + 1), s):
            d = line_score(p[r0:r0 + t, c0:c0 + t], valid[r0:r0 + t, c0:c0 + t], px_um)
            out.append((r0, c0, d))
    return out
