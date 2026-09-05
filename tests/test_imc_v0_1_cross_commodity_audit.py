from __future__ import annotations

import numpy as np
import pandas as pd

from imc_v0_1_cross_commodity_audit import (
    empirical_ks,
    empirical_wasserstein,
    imc_transform_window,
    rolling_volume_baseline,
    valid_window_starts,
)


def synthetic_window(length: int = 64):
    index = np.arange(length, dtype=np.float64)
    close = 100.0 * np.exp(0.001 * index + 0.0001 * np.sin(index))
    open_ = close * 0.999
    high = close * 1.003
    low = close * 0.997
    oi = 10_000.0 * np.exp(0.002 * index)
    volume = 200.0 + (index % 17) * 13.0
    prior = 150.0 + np.arange(20, dtype=np.float64)
    return open_, high, low, close, oi, volume, prior


def test_imc_is_scale_invariant_and_reconstructable():
    window = synthetic_window()
    original = imc_transform_window(*window)
    open_, high, low, close, oi, volume, prior = window
    scaled = imc_transform_window(
        open_ * 3.7,
        high * 3.7,
        low * 3.7,
        close * 3.7,
        oi * 11.2,
        volume * 53.0,
        prior * 53.0,
    )
    for key in ("price_open", "price_high", "price_low", "price_close", "oi", "volume_fixed", "volume_q20"):
        np.testing.assert_allclose(original[key], scaled[key], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(original["p0"] * np.exp(original["price_close"]), close, rtol=1e-12)
    np.testing.assert_allclose(original["oi0"] * np.exp(original["oi"]), oi, rtol=1e-12)
    np.testing.assert_allclose(original["m0_volume"] * original["volume_fixed"], volume, rtol=1e-12)


def test_oi_and_price_path_identities_are_exact_to_tolerance():
    window = synthetic_window(512)
    transformed = imc_transform_window(*window)
    close, oi = window[3], window[4]
    for lag in (1, 4, 16, 64, 256):
        np.testing.assert_allclose(
            transformed["price_close"][lag:] - transformed["price_close"][:-lag],
            np.log(close[lag:] / close[:-lag]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transformed["oi"][lag:] - transformed["oi"][:-lag],
            np.log(oi[lag:] / oi[:-lag]),
            rtol=0.0,
            atol=1e-12,
        )


def test_volume_q20_uses_strictly_prior_bars():
    volume = np.arange(1.0, 42.0)
    baseline = rolling_volume_baseline(volume)
    assert np.isnan(baseline[19])
    assert baseline[20] == np.median(volume[:20])
    changed = volume.copy()
    changed[21:] *= 10_000
    changed_baseline = rolling_volume_baseline(changed)
    assert changed_baseline[20] == baseline[20]
    assert changed_baseline[21] == baseline[21]


def test_valid_windows_require_positive_baseline_and_oi():
    rows = 100
    frame = pd.DataFrame(
        {
            "Open": np.full(rows, 100.0),
            "High": np.full(rows, 101.0),
            "Low": np.full(rows, 99.0),
            "Close": np.full(rows, 100.0),
            "Volume": np.full(rows, 10.0),
            "OpenInterest": np.full(rows, 1_000.0),
        }
    )
    starts = valid_window_starts(frame, 64)
    assert starts[0] == 20
    frame.loc[25, "OpenInterest"] = 0.0
    starts_after = valid_window_starts(frame, 64)
    assert 20 not in starts_after


def test_empirical_distance_helpers_match_simple_examples():
    values = np.asarray([0.0, 1.0, 2.0])
    np.testing.assert_allclose(empirical_wasserstein(values, values), 0.0, atol=1e-12)
    np.testing.assert_allclose(empirical_ks(values, values), 0.0, atol=1e-12)
    assert empirical_wasserstein(values, values + 2.0) > 1.99
    assert empirical_ks(values, values + 2.0) > 0.6
