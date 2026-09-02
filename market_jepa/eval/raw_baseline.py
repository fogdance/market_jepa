from __future__ import annotations

from typing import Iterable

import numpy as np

from market_jepa.data.pipeline import MarketData
from market_jepa.data.schema import CONTEXT_FEATURES, MARKET_FEATURES


STATISTICS = ("last", "mean", "std", "min", "max", "slope")


def _summary(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if not len(values):
        raise ValueError("raw baseline sequences must be non-empty")
    if len(values) == 1:
        slope = np.zeros(values.shape[1], dtype=np.float64)
    else:
        position = np.linspace(-1.0, 1.0, len(values))
        centered = values - values.mean(axis=0, keepdims=True)
        slope = (position[:, None] * centered).sum(axis=0) / np.square(position).sum()
    return np.concatenate(
        [values[-1], values.mean(0), values.std(0), values.min(0), values.max(0), slope]
    )


def _window_summaries(sequence: np.ndarray, windows: Iterable[int | None]) -> np.ndarray:
    parts = []
    for window in windows:
        selected = sequence if window is None else sequence[-min(window, len(sequence)) :]
        parts.append(_summary(selected))
    return np.concatenate(parts)


def raw_feature_names(ablation: str) -> list[str]:
    names: list[str] = []
    groups = [("minute", [16, 64, 256, 512])]
    if ablation in {"minute_daily", "minute_daily_weekly"}:
        groups.append(("daily", [5, 20, 60, None]))
    if ablation == "minute_daily_weekly":
        groups.append(("weekly", [4, 13, 26, None]))
    for timeframe, windows in groups:
        for window in windows:
            label = "all" if window is None else str(window)
            for statistic in STATISTICS:
                names.extend(
                    f"{timeframe}_{label}_{statistic}_{feature}" for feature in MARKET_FEATURES
                )
    names.extend(f"current_{name}" for name in CONTEXT_FEATURES)
    if ablation in {"minute_daily", "minute_daily_weekly"}:
        names.extend(["current_daily_source_bar_count", "daily_token_count"])
    if ablation == "minute_daily_weekly":
        names.extend(["current_weekly_source_bar_count", "weekly_token_count"])
    return names


def raw_baseline_features(data: MarketData, anchor: int, context_length: int, ablation: str) -> np.ndarray:
    minute = data.minute_market_raw[anchor - context_length + 1 : anchor + 1]
    parts = [_window_summaries(minute, [16, 64, 256, 512])]
    daily_market, daily_count = data.daily_snapshot_raw(anchor)
    weekly_market, weekly_count = data.weekly_snapshot_raw(anchor)
    if ablation in {"minute_daily", "minute_daily_weekly"}:
        parts.append(_window_summaries(daily_market, [5, 20, 60, None]))
    if ablation == "minute_daily_weekly":
        parts.append(_window_summaries(weekly_market, [4, 13, 26, None]))
    parts.append(data.minute_context_raw[anchor])
    if ablation in {"minute_daily", "minute_daily_weekly"}:
        parts.append(np.asarray([daily_count[-1, 0], len(daily_market)]))
    if ablation == "minute_daily_weekly":
        parts.append(np.asarray([weekly_count[-1, 0], len(weekly_market)]))
    result = np.concatenate([np.atleast_1d(part) for part in parts]).astype(np.float32)
    if len(result) != len(raw_feature_names(ablation)):
        raise RuntimeError("raw baseline feature/name schema mismatch")
    return result
