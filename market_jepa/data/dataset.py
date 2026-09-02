from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .pipeline import MarketData


def build_split_indices(data: MarketData, config: dict[str, Any], split: str) -> np.ndarray:
    start, end = map(pd.Timestamp, config["data"]["splits"][split])
    context = int(config["data"]["minute_context_length"])
    max_horizon = max(config["data"]["horizons"])
    n = len(data.minute)
    candidates = np.arange(context - 1, n - max_horizon, dtype=np.int64)
    trading_days = data.minute["trading_day"].to_numpy(dtype="datetime64[ns]")
    start64, end64 = np.datetime64(start), np.datetime64(end)
    mask = (trading_days[candidates] >= start64) & (trading_days[candidates] <= end64)
    mask &= trading_days[candidates + max_horizon] <= end64
    return candidates[mask]


def future_outcomes(data: MarketData, anchor: int, horizon: int) -> np.ndarray:
    frame = data.minute
    close_t = float(frame.iloc[anchor]["close"])
    future = frame.iloc[anchor + 1 : anchor + horizon + 1]
    future_return = float(future.iloc[-1]["close"] / close_t - 1.0)
    mfe = float((future["high"] / close_t - 1.0).max())
    mae = float((future["low"] / close_t - 1.0).min())
    closes = frame.iloc[anchor : anchor + horizon + 1]["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(closes))
    rv = float(np.sqrt(np.mean(log_returns**2)))
    return np.asarray([future_return, mfe, mae, rv], dtype=np.float32)


@dataclass(frozen=True)
class MarketSample:
    anchor: int


class MarketDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data: MarketData,
        config: dict[str, Any],
        split: str,
        indices: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.config = config
        self.split = split
        self.indices = build_split_indices(data, config, split) if indices is None else np.asarray(indices)
        self.context_length = int(config["data"]["minute_context_length"])
        self.horizons = tuple(int(value) for value in config["data"]["horizons"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        anchor = int(self.indices[item])
        start = anchor - self.context_length + 1
        daily_market, daily_observation, daily_source_max = self.data.daily_snapshot(anchor)
        weekly_market, weekly_observation, weekly_source_max = self.data.weekly_snapshot(anchor)
        result: dict[str, Any] = {
            "minute_market": torch.from_numpy(self.data.minute_market[start : anchor + 1]),
            "minute_context": torch.from_numpy(self.data.minute_context[start : anchor + 1]),
            "daily": torch.from_numpy(np.concatenate([daily_market, daily_observation], axis=1)),
            "weekly": torch.from_numpy(np.concatenate([weekly_market, weekly_observation], axis=1)),
            "daily_source_max": torch.from_numpy(daily_source_max),
            "weekly_source_max": torch.from_numpy(weekly_source_max),
            "anchor_index": anchor,
            "timestamp_ns": int(self.data.minute.iloc[anchor]["timestamp"].value),
            "trading_day_ns": int(self.data.minute.iloc[anchor]["trading_day"].value),
            "targets": {},
            "persistence": {},
            "outcomes": {},
        }
        for horizon in self.horizons:
            result["targets"][horizon] = torch.from_numpy(
                self.data.minute_market[anchor + 1 : anchor + horizon + 1]
            )
            result["persistence"][horizon] = torch.from_numpy(
                self.data.minute_market[anchor - horizon + 1 : anchor + 1]
            )
            result["outcomes"][horizon] = torch.from_numpy(future_outcomes(self.data, anchor, horizon))
        return result


def _pad(items: Iterable[torch.Tensor], padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    values = list(items)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    return pad_sequence(values, batch_first=True, padding_value=padding_value), lengths


def collate_market_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    daily, daily_lengths = _pad(sample["daily"] for sample in samples)
    weekly, weekly_lengths = _pad(sample["weekly"] for sample in samples)
    daily_source, _ = _pad(
        (sample["daily_source_max"] for sample in samples), padding_value=-1
    )
    weekly_source, _ = _pad(
        (sample["weekly_source_max"] for sample in samples), padding_value=-1
    )
    horizons = tuple(samples[0]["targets"])
    return {
        "minute_market": torch.stack([sample["minute_market"] for sample in samples]),
        "minute_context": torch.stack([sample["minute_context"] for sample in samples]),
        "daily": daily,
        "daily_lengths": daily_lengths,
        "weekly": weekly,
        "weekly_lengths": weekly_lengths,
        "daily_source_max": daily_source,
        "weekly_source_max": weekly_source,
        "anchor_index": torch.tensor([sample["anchor_index"] for sample in samples]),
        "timestamp_ns": torch.tensor([sample["timestamp_ns"] for sample in samples]),
        "trading_day_ns": torch.tensor([sample["trading_day_ns"] for sample in samples]),
        "targets": {
            horizon: torch.stack([sample["targets"][horizon] for sample in samples])
            for horizon in horizons
        },
        "persistence": {
            horizon: torch.stack([sample["persistence"][horizon] for sample in samples])
            for horizon in horizons
        },
        "outcomes": {
            horizon: torch.stack([sample["outcomes"][horizon] for sample in samples])
            for horizon in horizons
        },
    }
