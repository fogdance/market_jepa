from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .schema import CONTEXT_FEATURES, MARKET_FEATURES, RAW_COLUMNS


def _feature_frame(
    frame: pd.DataFrame,
    previous_close: np.ndarray,
    previous_volume_log: np.ndarray,
    previous_oi_log: np.ndarray,
    close_return: np.ndarray,
    realized_volatility: np.ndarray,
) -> pd.DataFrame:
    values = frame.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
    open_, high, low, close, volume, oi = values.T
    log_open = np.log(open_)
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)
    log_prev_close = np.log(previous_close)
    log_volume = np.log1p(volume)
    log_oi = np.log1p(oi)
    result = pd.DataFrame(
        {
            "log_open": log_open,
            "log_high": log_high,
            "log_low": log_low,
            "log_close": log_close,
            "close_log_return": close_return,
            "open_to_prev_close": log_open - log_prev_close,
            "high_to_prev_close": log_high - log_prev_close,
            "low_to_prev_close": log_low - log_prev_close,
            "normalized_range": (high - low) / previous_close,
            "log1p_volume": log_volume,
            "volume_log_change": log_volume - previous_volume_log,
            "log1p_open_interest": log_oi,
            "open_interest_log_change": log_oi - previous_oi_log,
            "realized_volatility": realized_volatility,
        },
        index=frame.index,
    )
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("market feature generation produced non-finite values")
    return result.loc[:, MARKET_FEATURES]


def completed_market_features(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("realized volatility window must be positive")
    close = frame["close"].to_numpy(dtype=np.float64)
    volume_log = np.log1p(frame["volume"].to_numpy(dtype=np.float64))
    oi_log = np.log1p(frame["open_interest"].to_numpy(dtype=np.float64))
    previous_close = np.roll(close, 1)
    previous_volume = np.roll(volume_log, 1)
    previous_oi = np.roll(oi_log, 1)
    previous_close[0] = close[0]
    previous_volume[0] = volume_log[0]
    previous_oi[0] = oi_log[0]
    returns = np.log(close / previous_close)
    rv = (
        pd.Series(returns * returns)
        .rolling(window=window, min_periods=1)
        .mean()
        .pow(0.5)
        .to_numpy()
    )
    return _feature_frame(frame, previous_close, previous_volume, previous_oi, returns, rv)


def partial_market_features(
    partial: pd.DataFrame,
    completed: pd.DataFrame,
    key_column: str,
    window: int,
) -> pd.DataFrame:
    """Feature each causal partial against strictly earlier completed periods."""

    keys = completed[key_column].to_numpy()
    row_keys = partial[key_column].to_numpy()
    positions = np.searchsorted(keys, row_keys)
    if np.any(positions >= len(keys)) or np.any(keys[positions] != row_keys):
        raise ValueError("partial period key is absent from completed periods")

    completed_close = completed["close"].to_numpy(dtype=np.float64)
    completed_volume_log = np.log1p(completed["volume"].to_numpy(dtype=np.float64))
    completed_oi_log = np.log1p(completed["open_interest"].to_numpy(dtype=np.float64))
    partial_close = partial["close"].to_numpy(dtype=np.float64)
    partial_volume_log = np.log1p(partial["volume"].to_numpy(dtype=np.float64))
    partial_oi_log = np.log1p(partial["open_interest"].to_numpy(dtype=np.float64))

    has_previous = positions > 0
    previous_close = partial_close.copy()
    previous_volume_log = partial_volume_log.copy()
    previous_oi_log = partial_oi_log.copy()
    previous_close[has_previous] = completed_close[positions[has_previous] - 1]
    previous_volume_log[has_previous] = completed_volume_log[positions[has_previous] - 1]
    previous_oi_log[has_previous] = completed_oi_log[positions[has_previous] - 1]
    current_return = np.log(partial_close / previous_close)

    completed_previous = np.roll(completed_close, 1)
    completed_previous[0] = completed_close[0]
    completed_returns_sq = np.log(completed_close / completed_previous) ** 2
    prefix = np.concatenate([[0.0], np.cumsum(completed_returns_sq)])
    starts = np.maximum(0, positions - (window - 1))
    past_sum = prefix[positions] - prefix[starts]
    past_count = positions - starts
    rv = np.sqrt((past_sum + current_return**2) / (past_count + 1))
    return _feature_frame(
        partial,
        previous_close,
        previous_volume_log,
        previous_oi_log,
        current_return,
        rv,
    )


def minute_context_observations(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = frame["timestamp"]
    minute_of_day = timestamp.dt.hour.to_numpy() * 60 + timestamp.dt.minute.to_numpy()
    trading_weekday = frame["trading_day"].dt.weekday.to_numpy()
    if np.any(trading_weekday > 4):
        raise ValueError("inferred trading days must be Monday through Friday")
    delta_minutes = timestamp.diff().dt.total_seconds().div(60).fillna(0.0).to_numpy()
    if np.any(delta_minutes[1:] <= 0):
        raise ValueError("timestamps must be strictly increasing")
    result = pd.DataFrame(
        {
            "time_of_day_sin": np.sin(2 * np.pi * minute_of_day / 1440.0),
            "time_of_day_cos": np.cos(2 * np.pi * minute_of_day / 1440.0),
            "day_of_week_sin": np.sin(2 * np.pi * trading_weekday / 5.0),
            "day_of_week_cos": np.cos(2 * np.pi * trading_weekday / 5.0),
            "log1p_delta_minutes": np.log1p(delta_minutes),
        },
        index=frame.index,
    )
    return result.loc[:, CONTEXT_FEATURES]


@dataclass(frozen=True)
class Normalizer:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    fit_count: int

    @classmethod
    def fit(cls, values: np.ndarray, names: Sequence[str]) -> "Normalizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != len(names):
            raise ValueError("invalid normalizer fit matrix")
        if not np.isfinite(array).all():
            raise ValueError("normalizer fit matrix is non-finite")
        mean = array.mean(axis=0)
        std = array.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0
        return cls(tuple(names), mean, std, len(array))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.names):
            raise ValueError("normalizer schema mismatch")
        return ((array - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fit_count": self.fit_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Normalizer":
        return cls(
            tuple(value["names"]),
            np.asarray(value["mean"], dtype=np.float64),
            np.asarray(value["std"], dtype=np.float64),
            int(value["fit_count"]),
        )
