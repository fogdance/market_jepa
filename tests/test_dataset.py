from __future__ import annotations

import numpy as np

from market_jepa.data.dataset import MarketDataset, build_split_indices, future_outcomes


def test_future_target_schema_is_market_only(causal_data, causal_config) -> None:
    dataset = MarketDataset(causal_data, causal_config, "train", indices=np.asarray([4]))
    sample = dataset[0]
    assert sample["minute_context"].shape[-1] == 5
    assert sample["targets"][1].shape == (1, 14)
    assert set(sample["targets"]) == {1}


def test_outcome_formulas(causal_data) -> None:
    outcome = future_outcomes(causal_data, anchor=4, horizon=2)
    close_t = 105.0
    expected_return = 107.0 / close_t - 1
    expected_mfe = 999.0 / close_t - 1
    expected_mae = 104.0 / close_t - 1
    expected_rv = np.sqrt(np.mean(np.diff(np.log([105.0, 106.0, 107.0])) ** 2))
    assert np.allclose(outcome, [expected_return, expected_mfe, expected_mae, expected_rv])


def test_split_only_purges_future_not_past_context(causal_data, causal_config) -> None:
    causal_config["data"]["splits"]["train"] = ["2024-01-08", "2024-01-08"]
    indices = build_split_indices(causal_data, causal_config, "train")
    assert indices[0] == 2
    assert causal_data.minute.iloc[indices[0] - 1]["trading_day"].strftime("%Y-%m-%d") == "2024-01-05"
    assert causal_data.minute.iloc[indices[-1] + 1]["trading_day"].strftime("%Y-%m-%d") == "2024-01-08"
