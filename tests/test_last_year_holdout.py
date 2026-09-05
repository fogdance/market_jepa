from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.data import build_split_indices
from train_market_jepa_last_year_holdout import (
    _classify,
    expected_config,
    load_benchmark_config,
    split_integrity,
)


def test_frozen_holdout_config_changes_only_identity_split_output_and_protocol() -> None:
    config = load_benchmark_config(
        "configs/market_jepa_last_year_holdout_2025.yaml"
    )
    assert config == expected_config()
    assert config["model"] == DEFAULT_CONFIG["model"]
    assert config["training"] == DEFAULT_CONFIG["training"]
    assert config["data"]["splits"] == {
        "train": ["2018-01-02", "2024-12-31"],
        "validation": ["2025-01-01", "2025-06-30"],
        "test": ["2025-07-01", "2025-12-02"],
    }
    assert config["benchmark"]["checkpoint_selection"] == "fixed_budget_final"
    assert config["benchmark"]["evaluate_validation_during_training"] is False
    assert config["benchmark"]["test_authorized"] is True


def test_split_integrity_purges_future_at_both_2025_boundaries() -> None:
    days = pd.to_datetime(
        [
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
            "2025-06-30",
            "2025-07-01",
            "2025-12-01",
            "2025-12-02",
        ]
    )
    frame = pd.DataFrame({"timestamp": days, "trading_day": days})
    data = SimpleNamespace(minute=frame)
    config = {
        "data": {
            "minute_context_length": 1,
            "horizons": [1],
            "splits": expected_config()["data"]["splits"],
        }
    }
    validation = build_split_indices(data, config, "validation")
    test = build_split_indices(data, config, "test")
    np.testing.assert_array_equal(validation, np.asarray([2]))
    np.testing.assert_array_equal(test, np.asarray([4, 5]))
    report = split_integrity(data, config)
    assert report["validation"]["target_max_trading_day"] == "2025-06-30"
    assert report["test"]["target_max_trading_day"] == "2025-12-02"


def _effect(effect: float, low: float, high: float) -> dict:
    return {"effect": effect, "ci95": [low, high]}


def test_category_uses_test_prediction_ci_and_frozen_structural_point_rules() -> None:
    prediction = {"scopes": {"AVG": _effect(0.1, 0.01, 0.2)}}
    structural = {
        "effects": {
            name: {"AVG": _effect(0.1, -0.1, 0.2)}
            for name in ("oi_use", "volume_use", "ov_use", "price_x_ov")
        }
    }
    assert _classify(prediction, structural)["code"] == "D"
    structural["effects"]["price_x_ov"]["AVG"]["effect"] = -0.01
    assert _classify(prediction, structural)["code"] == "C"
    structural["effects"]["ov_use"]["AVG"]["effect"] = -0.01
    assert _classify(prediction, structural)["code"] == "B"
    prediction["scopes"]["AVG"]["ci95"][0] = -0.01
    assert _classify(prediction, structural)["code"] == "A"
