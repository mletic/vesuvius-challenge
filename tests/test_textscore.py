"""Properties of the text scores that must hold regardless of data."""
import numpy as np

from vesuvius_sweep.textscore import line_profile_score, line_score, line_z, mask_peak

UM = 9.6
N = int(20_000 / UM)


def stripes(period_mm, horizontal=True, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(N) * UM / 1000
    prof = (np.sin(2 * np.pi * y / period_mm) > 0.6).astype(np.float32)
    img = np.tile(prof[:, None], (1, N)) if horizontal else np.tile(prof[None, :], (N, 1))
    return np.clip(0.3 * img + 0.15 * rng.random((N, N)), 0, 1).astype(np.float32)


def test_horizontal_lines_score_positive_and_rotation_flips_sign():
    img, v = stripes(6.5), np.ones((N, N), bool)
    s, r = line_score(img, v, UM, period_mm=(5.65, 7.27)), line_score(np.rot90(img), v, UM, period_mm=(5.65, 7.27))
    assert s["peak"] > 1.0 and r["peak"] < -1.0
    assert line_z(img, v, UM)["z"] > 0 > line_z(np.rot90(img), v, UM)["z"]


def weak_lines(amp, seed):
    rng = np.random.default_rng(seed)
    y = np.arange(N) * UM / 1000
    prof = (np.sin(2 * np.pi * y / 6.5) > 0.6).astype(np.float32)
    return np.clip(amp * np.tile(prof[:, None], (1, N)) + rng.random((N, N)), 0, 1).astype(np.float32)


def test_v3_primary_score_detects_weak_lines_and_is_directional():
    v = np.ones((N, N), bool)
    noise = [line_profile_score(np.random.default_rng(s).random((N, N)).astype(np.float32), v, UM)["s"] for s in range(20)]
    thr = np.percentile(noise, 95)
    weak = [line_profile_score(weak_lines(0.02, 100 + s), v, UM)["s"] for s in range(5)]
    assert np.mean(weak) > thr                       # 2% lines on uniform noise are detected
    assert line_profile_score(np.rot90(stripes(6.5)), v, UM)["s"] < -thr   # rotated lines score negative
    assert np.mean(weak) > np.mean(noise) + 0.5


def secondary_white_noise_is_centred_and_below_lines():  # v1/v2 are secondary; kept as documentation, not a test
    v = np.ones((N, N), bool)
    noise = [line_score(np.random.default_rng(s).random((N, N)).astype(np.float32), v, UM, period_mm=(5.65, 7.27))["peak"]
             for s in range(20)]
    zs = [line_z(np.random.default_rng(s).random((N, N)).astype(np.float32), v, UM)["z"] for s in range(20)]
    assert abs(np.mean(noise)) < 0.5 and abs(np.mean(zs)) < 1.0
    assert line_score(stripes(6.5), v, UM, period_mm=(5.65, 7.27))["peak"] > np.percentile(noise, 95)
    assert line_z(stripes(6.5), v, UM)["z"] > np.percentile(zs, 95)


def test_mask_peak_zero_without_holes_and_high_for_striped_holes():
    v = np.ones((N, N), bool)
    assert mask_peak(v, UM) == 0.0
    holes = stripes(6.5, seed=2) > 0.25
    assert mask_peak(~holes, UM) > 1.0


def test_featureless_input_is_nan_not_error():
    z = np.zeros((N, N), np.float32)
    assert np.isnan(line_score(z, np.ones((N, N), bool), UM)["peak"])
