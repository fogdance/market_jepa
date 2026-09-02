from __future__ import annotations

import numpy as np
import pandas as pd

from market_jepa.data.pipeline import prepare_market_data


def test_normalizer_fit_uses_train_dates_only(causal_config) -> None:
    causal_config["data"]["splits"]["train"] = ["2024-01-05", "2024-01-08"]
    causal_config["data"]["splits"]["validation"] = ["2024-01-09", "2024-01-09"]
    causal_config["data"]["splits"]["test"] = ["2024-01-10", "2024-01-10"]
    data = prepare_market_data(causal_config)
    mask = data.minute["trading_day"].between(pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"))
    expected = data.minute_market_raw[mask.to_numpy()].mean(axis=0)
    assert np.allclose(data.normalizers.minute_market.mean, expected)
    assert data.normalizers.minute_market.fit_count == int(mask.sum())
