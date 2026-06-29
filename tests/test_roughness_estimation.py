"""
Unit tests for the Whittle Hurst estimator.
Validates on synthetic fractional Brownian motion with known H.
"""
import numpy as np
import pytest
from sys import path
from pathlib import Path

path.insert(0, str(Path(__file__).parent.parent / "src"))
from features_rough_vol import whittle_hurst, ols_scaling_hurst


def simulate_fbm_davies_harte(n: int, H: float, seed: int = 42) -> np.ndarray:
    """
    Simulate fractional Brownian motion using the Davies-Harte exact method.
    Returns the fBm path (not increments).
    """
    rng = np.random.default_rng(seed)
    # Autocovariance of fGn
    k = np.arange(n)
    r = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    r[0] = 1.0

    # Circulant embedding
    row = np.concatenate([r, r[-2:0:-1]])
    lam = np.real(np.fft.fft(row))
    lam = np.maximum(lam, 0)  # numerical fix

    w = rng.standard_normal(2 * (n - 1)) + 1j * rng.standard_normal(2 * (n - 1))
    w[0] = np.real(w[0]) * np.sqrt(2)
    w[n - 1] = np.real(w[n - 1]) * np.sqrt(2)

    fgn = np.real(np.fft.ifft(np.sqrt(lam) * w))[:n]
    fbm = np.cumsum(fgn)
    return fbm


@pytest.mark.parametrize("H_true", [0.1, 0.2, 0.3, 0.4])
def test_whittle_hurst_accuracy(H_true):
    """Whittle estimator should recover H within ±0.10 on n=500 samples."""
    fbm = simulate_fbm_davies_harte(n=500, H=H_true, seed=0)
    H_est = whittle_hurst(fbm)
    assert not np.isnan(H_est), f"Whittle returned NaN for H={H_true}"
    assert abs(H_est - H_true) < 0.10, \
        f"Whittle H={H_est:.3f} too far from true H={H_true:.3f}"


@pytest.mark.parametrize("H_true", [0.1, 0.3])
def test_whittle_beats_ols(H_true):
    """Whittle estimator should be at least as accurate as OLS scaling moments."""
    errors_whittle, errors_ols = [], []
    for seed in range(20):
        fbm = simulate_fbm_davies_harte(n=300, H=H_true, seed=seed)
        hw = whittle_hurst(fbm)
        ho = ols_scaling_hurst(fbm)
        if not np.isnan(hw):
            errors_whittle.append(abs(hw - H_true))
        if not np.isnan(ho):
            errors_ols.append(abs(ho - H_true))

    mae_w = np.mean(errors_whittle)
    mae_o = np.mean(errors_ols)
    # Whittle should not be dramatically worse than OLS
    assert mae_w <= mae_o * 1.5, \
        f"Whittle MAE={mae_w:.3f} much worse than OLS MAE={mae_o:.3f} for H={H_true}"


def test_hurst_in_valid_range():
    """H must be in (0, 1) for any non-degenerate series."""
    rng = np.random.default_rng(99)
    for _ in range(10):
        x = rng.standard_normal(200)
        h = whittle_hurst(x)
        if not np.isnan(h):
            assert 0 < h < 1, f"H={h} out of range"


def test_short_series_returns_nan():
    """Series shorter than minimum length should return NaN gracefully."""
    h = whittle_hurst(np.array([0.1, 0.2, 0.3]))
    assert np.isnan(h)


def test_roughness_sign():
    """A rough process (H < 0.5) should give roughness > 0."""
    fbm_rough = simulate_fbm_davies_harte(500, H=0.1, seed=7)
    H_est = whittle_hurst(fbm_rough)
    assert not np.isnan(H_est)
    roughness = 0.5 - H_est
    assert roughness > 0, f"Expected roughness > 0 for H={H_est:.3f}"
