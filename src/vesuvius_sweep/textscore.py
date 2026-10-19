"""Label-free scores for lines of writing in ink-probability maps.

In published layout the scroll axis runs down the rows, so lines of writing run
horizontally and repeat vertically, 5.65-7.27 mm apart (`LINE_BAND_MM`, P10-P90
of 73 spacings measured from the organisers' labels; median 6.55 mm).

Which function to use
---------------------
`line_profile_score` (v3, primary). Averages each row across the tile, then
    compares the fraction of row-profile power in the line band with the same
    for columns. Returns dict(s, s_rows, s_cols, coverage); `s` > 0 means more
    horizontal line structure. Validated on crisp maps (published 2.4 µm
    predictions: AUC 0.92 against rotated copies); at chance on 9 µm ink_9um
    outputs (see PREREG.md and results/).
`mask_peak`. Geometry guard: the v1 score of a tile's own hole/outline map.
    Values > 1 mean tears or the outline alone could produce line structure.
`line_score` (v1, superseded). Directional spectral "peakness"; its key `peak`
    was the early score and `score` is plain anisotropy. Too noisy on 20 mm tiles.
`line_z` (v2, secondary). Per-direction excess over a fitted power-law
    background. Underpowered on 20 mm tiles (unit tests).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


LINE_BAND_MM = (5.65, 7.27)


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


def line_z(p: np.ndarray, valid: np.ndarray, px_um: float, band_mm=(5.65, 7.27),
           fit_mm=(2.0, 15.0), cone: float = 20.0, pad: int = 4) -> dict:
    """Score v2 (PREREG deviations 2-3): red-noise excess of line-period power, per direction.

    The tapered tile is zero-padded `pad`x so the spectrum is sampled finely enough for a
    20 mm tile; in each cone (vertical frequencies = horizontal lines, and horizontal
    frequencies) the mean power in rings one padded frequency step wide is log-transformed
    over fit_mm periods; a power law fitted to the rings outside band_mm predicts the band;
    z_dir is the mean band residual over the fit's residual SD. z = z_v - z_h.
    """
    g, m, um = to_grid(p.astype(np.float32), valid, px_um)
    if m.mean() < 0.3 or m.sum() < 400:
        return dict(z=np.nan, z_v=np.nan, z_h=np.nan, coverage=float(m.mean()))
    x = np.where(m, g - g[m].mean(), 0)
    w = ndi.gaussian_filter(m.astype(np.float32), 3) * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    shp = (x.shape[0] * pad, x.shape[1] * pad)
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * w, s=shp))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(shp[0], d=um / 1000))
    fx = np.fft.fftshift(np.fft.fftfreq(shp[1], d=um / 1000))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    r = np.hypot(FY, FX)
    ang = np.degrees(np.arctan2(np.abs(FX), np.abs(FY)))
    step = max(fy[1] - fy[0], fx[1] - fx[0])
    edges = np.arange(1 / fit_mm[1], 1 / fit_mm[0] + step, step)
    centers = (edges[:-1] + edges[1:]) / 2
    in_band = (centers >= 1 / band_mm[1]) & (centers <= 1 / band_mm[0])
    # Lines are not sinusoids: keep their 2nd and 3rd harmonics out of the background fit.
    harmonic = np.zeros_like(in_band)
    for h in (2, 3):
        harmonic |= (centers >= h / band_mm[1]) & (centers <= h / band_mm[0])
    ring = np.digitize(r, edges) - 1
    out = {}
    for name, sel in (("v", ang <= cone), ("h", ang >= 90 - cone)):
        k = ring[sel]; pw = P[sel]
        ok_k = (k >= 0) & (k < len(centers))
        sums = np.bincount(k[ok_k], weights=pw[ok_k], minlength=len(centers))
        cnt = np.bincount(k[ok_k], minlength=len(centers))
        lp = np.where((cnt > 0) & (sums > 0), np.log(np.maximum(sums, 1e-30) / np.maximum(cnt, 1)), np.nan)
        ok = np.isfinite(lp)
        fit = ok & ~in_band & ~harmonic
        if fit.sum() < 6 or (ok & in_band).sum() < 1:
            return dict(z=np.nan, z_v=np.nan, z_h=np.nan, coverage=float(m.mean()))
        A = np.vstack([np.ones(fit.sum()), np.log(centers[fit])]).T
        coef, *_ = np.linalg.lstsq(A, lp[fit], rcond=None)
        pred = coef[0] + coef[1] * np.log(centers)
        sd = np.std(lp[fit] - pred[fit], ddof=2)
        out[name] = float(np.mean(lp[ok & in_band] - pred[ok & in_band]) / max(sd, 1e-6))
    return dict(z=out["v"] - out["h"], z_v=out["v"], z_h=out["h"], coverage=float(m.mean()))


def _profile_power(g, m, um, band_mm, fit_mm, pad):
    """Fraction of detrended row-mean-profile power that lies in the line band (harmonics excluded)."""
    cnt = m.sum(1)
    prof = np.where(cnt > 0.3 * m.shape[1], (g * m).sum(1) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(prof)
    if ok.mean() < 0.6:
        return np.nan
    idx = np.arange(prof.size)
    prof = np.interp(idx, idx[ok], prof[ok])
    prof = prof - ndi.gaussian_filter1d(prof, 10_000 / um)  # remove trends longer than ~1 cm
    prof = prof * np.hanning(prof.size)
    P = np.abs(np.fft.rfft(prof, n=prof.size * pad)) ** 2
    f = np.fft.rfftfreq(prof.size * pad, d=um / 1000)
    band = (f >= 1 / band_mm[1]) & (f <= 1 / band_mm[0])
    harm = np.zeros_like(band)
    for h in (2, 3):
        harm |= (f >= h / band_mm[1]) & (f <= h / band_mm[0])
    bg = (f >= 1 / fit_mm[1]) & (f <= 1 / fit_mm[0]) & ~band & ~harm
    tot = P[band].sum() + P[bg].sum()
    if not band.any() or bg.sum() < 5 or tot <= 0:
        return np.nan
    return float(P[band].sum() / tot)  # fraction of profile power in the line band


def line_profile_score(p: np.ndarray, valid: np.ndarray, px_um: float, band_mm=(5.65, 7.27),
                       fit_mm=(2.0, 15.0), tilts=(-6, -3, 0, 3, 6), pad: int = 8) -> dict:
    """Score v3 (PREREG deviation 4): periodicity of row-mean profiles.

    Lines of writing run across the whole tile, so averaging each row over the tile's
    width suppresses noise before looking for the line period. For each small tilt the
    tile is rotated, the row-mean profile is detrended and the fraction of its power in the
    line band is taken (2nd/3rd harmonic bands excluded from the total). The best tilt is
    kept; the same is done with column profiles (lines running the other way) as the
    control. s = log(frac_rows / frac_cols).
    """
    g, m, um = to_grid(p.astype(np.float32), valid, px_um)
    if m.mean() < 0.3 or m.sum() < 400:
        return dict(s=np.nan, s_rows=np.nan, s_cols=np.nan, coverage=float(m.mean()))
    res = {}
    for name, (G, M) in (("rows", (g, m)), ("cols", (g.T, m.T))):
        best = -np.inf
        for t in tilts:
            Gr = ndi.rotate(G, t, reshape=False, order=1, mode="constant", cval=0) if t else G
            Mr = (ndi.rotate(M.astype(np.float32), t, reshape=False, order=0, mode="constant", cval=0) > 0.5) if t else M
            v = _profile_power(Gr, Mr, um, band_mm, fit_mm, pad)
            if np.isfinite(v):
                best = max(best, v)
        res[name] = best if np.isfinite(best) else np.nan
    s_ = float(np.log(res["rows"] / res["cols"])) if res["rows"] > 0 and res["cols"] > 0 else np.nan
    return dict(s=s_, s_rows=res["rows"], s_cols=res["cols"], coverage=float(m.mean()))
