from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    Normalizer,
    completed_market_features,
    minute_context_observations,
    partial_market_features,
)
from .schema import CONTEXT_FEATURES, MARKET_FEATURES, PERIOD_CONTEXT_FEATURES
from .trading_day import causal_partial_bars, completed_bars, load_minute_csv


@dataclass(frozen=True)
class NormalizerBundle:
    minute_market: Normalizer
    minute_context: Normalizer
    daily_market: Normalizer
    daily_context: Normalizer
    weekly_market: Normalizer
    weekly_context: Normalizer

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizerBundle":
        return cls(**{name: Normalizer.from_dict(value[name]) for name in cls.__dataclass_fields__})


@dataclass
class MarketData:
    minute: pd.DataFrame
    daily_partial: pd.DataFrame
    weekly_partial: pd.DataFrame
    daily_completed: pd.DataFrame
    weekly_completed: pd.DataFrame
    minute_market_raw: np.ndarray
    minute_context_raw: np.ndarray
    daily_partial_market_raw: np.ndarray
    weekly_partial_market_raw: np.ndarray
    daily_completed_market_raw: np.ndarray
    weekly_completed_market_raw: np.ndarray
    minute_market: np.ndarray
    minute_context: np.ndarray
    daily_partial_market: np.ndarray
    weekly_partial_market: np.ndarray
    daily_completed_market: np.ndarray
    weekly_completed_market: np.ndarray
    daily_partial_context: np.ndarray
    weekly_partial_context: np.ndarray
    daily_completed_context: np.ndarray
    weekly_completed_context: np.ndarray
    daily_position: np.ndarray
    weekly_position: np.ndarray
    normalizers: NormalizerBundle

    def daily_snapshot(self, anchor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = int(self.daily_position[anchor])
        market = np.concatenate(
            [self.daily_completed_market[:position], self.daily_partial_market[anchor : anchor + 1]], axis=0
        )
        context = np.concatenate(
            [self.daily_completed_context[:position], self.daily_partial_context[anchor : anchor + 1]], axis=0
        )
        source_max = np.concatenate(
            [
                self.daily_completed["source_max_index"].to_numpy(dtype=np.int64)[:position],
                np.asarray([anchor], dtype=np.int64),
            ]
        )
        return market, context, source_max

    def weekly_snapshot(self, anchor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = int(self.weekly_position[anchor])
        market = np.concatenate(
            [self.weekly_completed_market[:position], self.weekly_partial_market[anchor : anchor + 1]], axis=0
        )
        context = np.concatenate(
            [self.weekly_completed_context[:position], self.weekly_partial_context[anchor : anchor + 1]], axis=0
        )
        source_max = np.concatenate(
            [
                self.weekly_completed["source_max_index"].to_numpy(dtype=np.int64)[:position],
                np.asarray([anchor], dtype=np.int64),
            ]
        )
        return market, context, source_max

    def daily_snapshot_raw(self, anchor: int) -> tuple[np.ndarray, np.ndarray]:
        position = int(self.daily_position[anchor])
        market = np.concatenate(
            [
                self.daily_completed_market_raw[:position],
                self.daily_partial_market_raw[anchor : anchor + 1],
            ],
            axis=0,
        )
        count = np.concatenate(
            [
                self.daily_completed[["source_bar_count"]].to_numpy(dtype=np.float64)[:position],
                self.daily_partial[["source_bar_count"]].to_numpy(dtype=np.float64)[anchor : anchor + 1],
            ],
            axis=0,
        )
        return market, count

    def weekly_snapshot_raw(self, anchor: int) -> tuple[np.ndarray, np.ndarray]:
        position = int(self.weekly_position[anchor])
        market = np.concatenate(
            [
                self.weekly_completed_market_raw[:position],
                self.weekly_partial_market_raw[anchor : anchor + 1],
            ],
            axis=0,
        )
        count = np.concatenate(
            [
                self.weekly_completed[["source_bar_count"]].to_numpy(dtype=np.float64)[:position],
                self.weekly_partial[["source_bar_count"]].to_numpy(dtype=np.float64)[anchor : anchor + 1],
            ],
            axis=0,
        )
        return market, count


def _fit_normalizers(
    config: dict[str, Any],
    minute: pd.DataFrame,
    minute_market: np.ndarray,
    minute_context: np.ndarray,
    daily_partial: pd.DataFrame,
    daily_partial_market: np.ndarray,
    daily_completed: pd.DataFrame,
    daily_completed_market: np.ndarray,
    weekly_partial: pd.DataFrame,
    weekly_partial_market: np.ndarray,
    weekly_completed: pd.DataFrame,
    weekly_completed_market: np.ndarray,
) -> NormalizerBundle:
    start, end = map(pd.Timestamp, config["data"]["splits"]["train"])
    minute_mask = minute["trading_day"].between(start, end).to_numpy()
    daily_complete_mask = daily_completed["trading_day"].between(start, end).to_numpy()
    weekly_complete_mask = weekly_completed["trading_day"].between(start, end).to_numpy()

    daily_market_fit = np.concatenate(
        [daily_completed_market[daily_complete_mask], daily_partial_market[minute_mask]], axis=0
    )
    weekly_market_fit = np.concatenate(
        [weekly_completed_market[weekly_complete_mask], weekly_partial_market[minute_mask]], axis=0
    )
    daily_context_fit = np.concatenate(
        [
            daily_completed.loc[daily_complete_mask, ["source_bar_count"]].to_numpy(dtype=np.float64),
            daily_partial.loc[minute_mask, ["source_bar_count"]].to_numpy(dtype=np.float64),
        ],
        axis=0,
    )
    weekly_context_fit = np.concatenate(
        [
            weekly_completed.loc[weekly_complete_mask, ["source_bar_count"]].to_numpy(dtype=np.float64),
            weekly_partial.loc[minute_mask, ["source_bar_count"]].to_numpy(dtype=np.float64),
        ],
        axis=0,
    )
    return NormalizerBundle(
        minute_market=Normalizer.fit(minute_market[minute_mask], MARKET_FEATURES),
        minute_context=Normalizer.fit(minute_context[minute_mask], CONTEXT_FEATURES),
        daily_market=Normalizer.fit(daily_market_fit, MARKET_FEATURES),
        daily_context=Normalizer.fit(daily_context_fit, PERIOD_CONTEXT_FEATURES),
        weekly_market=Normalizer.fit(weekly_market_fit, MARKET_FEATURES),
        weekly_context=Normalizer.fit(weekly_context_fit, PERIOD_CONTEXT_FEATURES),
    )


def prepare_market_data(
    config: dict[str, Any], normalizers: NormalizerBundle | None = None
) -> MarketData:
    minute = load_minute_csv(config["data"]["csv_path"])
    window = int(config["data"]["realized_vol_window"])
    daily_partial = causal_partial_bars(minute, "trading_day")
    weekly_partial = causal_partial_bars(minute, "iso_key")
    daily_completed = completed_bars(daily_partial, "trading_day")
    weekly_completed = completed_bars(weekly_partial, "iso_key")
    weekly_completed["trading_day"] = minute.loc[
        weekly_completed["source_max_index"].to_numpy(dtype=np.int64), "trading_day"
    ].to_numpy()

    minute_market_frame = completed_market_features(minute, window)
    minute_context_frame = minute_context_observations(minute)
    daily_completed_market_frame = completed_market_features(daily_completed, window)
    weekly_completed_market_frame = completed_market_features(weekly_completed, window)
    daily_partial_market_frame = partial_market_features(
        daily_partial, daily_completed, "trading_day", window
    )
    weekly_partial_market_frame = partial_market_features(
        weekly_partial, weekly_completed, "iso_key", window
    )

    raw_arrays = {
        "minute_market": minute_market_frame.to_numpy(dtype=np.float64),
        "minute_context": minute_context_frame.to_numpy(dtype=np.float64),
        "daily_partial_market": daily_partial_market_frame.to_numpy(dtype=np.float64),
        "weekly_partial_market": weekly_partial_market_frame.to_numpy(dtype=np.float64),
        "daily_completed_market": daily_completed_market_frame.to_numpy(dtype=np.float64),
        "weekly_completed_market": weekly_completed_market_frame.to_numpy(dtype=np.float64),
    }
    if normalizers is None:
        normalizers = _fit_normalizers(
            config,
            minute,
            raw_arrays["minute_market"],
            raw_arrays["minute_context"],
            daily_partial,
            raw_arrays["daily_partial_market"],
            daily_completed,
            raw_arrays["daily_completed_market"],
            weekly_partial,
            raw_arrays["weekly_partial_market"],
            weekly_completed,
            raw_arrays["weekly_completed_market"],
        )

    daily_position = np.searchsorted(
        daily_completed["trading_day"].to_numpy(dtype="datetime64[ns]"),
        minute["trading_day"].to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    weekly_position = np.searchsorted(
        weekly_completed["iso_key"].to_numpy(dtype=np.int64),
        minute["iso_key"].to_numpy(dtype=np.int64),
        side="left",
    )
    daily_partial_count = daily_partial[["source_bar_count"]].to_numpy(dtype=np.float64)
    weekly_partial_count = weekly_partial[["source_bar_count"]].to_numpy(dtype=np.float64)
    daily_completed_count = daily_completed[["source_bar_count"]].to_numpy(dtype=np.float64)
    weekly_completed_count = weekly_completed[["source_bar_count"]].to_numpy(dtype=np.float64)
    return MarketData(
        minute=minute,
        daily_partial=daily_partial,
        weekly_partial=weekly_partial,
        daily_completed=daily_completed,
        weekly_completed=weekly_completed,
        minute_market_raw=raw_arrays["minute_market"],
        minute_context_raw=raw_arrays["minute_context"],
        daily_partial_market_raw=raw_arrays["daily_partial_market"],
        weekly_partial_market_raw=raw_arrays["weekly_partial_market"],
        daily_completed_market_raw=raw_arrays["daily_completed_market"],
        weekly_completed_market_raw=raw_arrays["weekly_completed_market"],
        minute_market=normalizers.minute_market.transform(raw_arrays["minute_market"]),
        minute_context=normalizers.minute_context.transform(raw_arrays["minute_context"]),
        daily_partial_market=normalizers.daily_market.transform(raw_arrays["daily_partial_market"]),
        weekly_partial_market=normalizers.weekly_market.transform(raw_arrays["weekly_partial_market"]),
        daily_completed_market=normalizers.daily_market.transform(raw_arrays["daily_completed_market"]),
        weekly_completed_market=normalizers.weekly_market.transform(raw_arrays["weekly_completed_market"]),
        daily_partial_context=normalizers.daily_context.transform(daily_partial_count),
        weekly_partial_context=normalizers.weekly_context.transform(weekly_partial_count),
        daily_completed_context=normalizers.daily_context.transform(daily_completed_count),
        weekly_completed_context=normalizers.weekly_context.transform(weekly_completed_count),
        daily_position=daily_position,
        weekly_position=weekly_position,
        normalizers=normalizers,
    )
