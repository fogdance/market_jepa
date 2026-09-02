from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.data.pipeline import prepare_market_data


@pytest.fixture
def causal_csv(tmp_path: Path) -> Path:
    rows = [
        ("2024-01-05 09:01:00", 100, 102, 99, 101, 10, 1000),
        ("2024-01-05 15:00:00", 101, 103, 100, 102, 20, 1010),
        ("2024-01-05 21:01:00", 103, 104, 102, 103, 30, 1020),
        ("2024-01-05 23:00:00", 103, 105, 101, 104, 40, 1030),
        ("2024-01-08 09:01:00", 104, 106, 103, 105, 50, 1040),
        ("2024-01-08 15:00:00", 105, 999, 104, 106, 60, 1050),
        ("2024-01-08 21:01:00", 107, 108, 106, 107, 70, 1060),
        ("2024-01-09 09:01:00", 107, 109, 105, 108, 80, 1070),
        ("2024-01-09 15:00:00", 108, 110, 107, 109, 90, 1080),
    ]
    frame = pd.DataFrame(
        rows, columns=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"]
    )
    path = tmp_path / "minute.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def causal_config(causal_csv: Path) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["data"].update(
        csv_path=str(causal_csv),
        minute_context_length=2,
        horizons=[1],
        realized_vol_window=3,
    )
    config["data"]["splits"] = {
        "train": ["2024-01-01", "2024-01-31"],
        "validation": ["2024-02-01", "2024-02-29"],
        "test": ["2024-03-01", "2024-03-31"],
    }
    return config


@pytest.fixture
def causal_data(causal_config: dict):
    return prepare_market_data(causal_config)
