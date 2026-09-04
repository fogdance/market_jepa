from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import RAW_COLUMNS


def infer_trading_day(timestamp: pd.Series, night_start_hour: int = 18) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="raise")
    natural = ts.dt.normalize().to_numpy(dtype="datetime64[ns]")
    night = ts.dt.hour.to_numpy() >= night_start_hour
    observed_day_dates = np.unique(natural[~night])
    if not len(observed_day_dates):
        raise ValueError("cannot infer trading day without daytime observations")
    result = natural.copy()
    positions = np.searchsorted(observed_day_dates, natural[night], side="right")
    valid = positions < len(observed_day_dates)
    inferred = np.full(positions.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    inferred[valid] = observed_day_dates[positions[valid]]
    result[night] = inferred
    return pd.Series(pd.to_datetime(result), index=timestamp.index, name="trading_day")


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "date": "timestamp",
        "timestamp": "timestamp",
        "datetime": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "openinterest": "open_interest",
        "open_interest": "open_interest",
    }
    rename: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "").replace("-", "_")
        if key in aliases:
            rename[column] = aliases[key]
    result = frame.rename(columns=rename)
    missing = {"timestamp", *RAW_COLUMNS}.difference(result.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return result


def _rows_through_trading_day(path: str | Path, cutoff: pd.Timestamp) -> int:
    """Count the source prefix without materializing post-cutoff rows."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        timestamp_columns = [
            name
            for name in reader.fieldnames
            if name.strip().lower() in {"date", "datetime", "timestamp"}
        ]
        if len(timestamp_columns) != 1:
            raise ValueError("CSV must contain exactly one timestamp column")
        timestamp_column = timestamp_columns[0]
        rows = 0
        for row in reader:
            timestamp = pd.Timestamp(row[timestamp_column])
            calendar_day = timestamp.normalize()
            # `infer_trading_day` assigns a night row to the next observed
            # daytime date. Consequently, night rows stamped on the cutoff
            # calendar day necessarily belong after the cutoff and are not
            # materialized. Earlier Friday/holiday-eve nights remain included.
            if calendar_day > cutoff or (
                calendar_day == cutoff and timestamp.hour >= 18
            ):
                break
            rows += 1
    return rows


def load_minute_csv(
    path: str | Path, max_trading_day: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    cutoff = pd.Timestamp(max_trading_day) if max_trading_day is not None else None
    nrows = _rows_through_trading_day(path, cutoff) if cutoff is not None else None
    frame = _canonicalize(pd.read_csv(path, nrows=nrows))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for column in RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamp")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps are not increasing")
    raw = frame.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("OHLCV/OI contains NaN or Inf")
    if np.any(raw[:, :4] <= 0):
        raise ValueError("OHLC must be positive")
    if np.any(raw[:, 4:] < 0):
        raise ValueError("Volume/OI must be non-negative")
    invalid = (frame["high"] < frame[["open", "low", "close"]].max(axis=1)) | (
        frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    )
    if invalid.any():
        raise ValueError(f"invalid OHLC rows: {int(invalid.sum())}")
    frame["trading_day"] = infer_trading_day(frame["timestamp"])
    frame = frame.loc[frame["trading_day"].notna()].copy().reset_index(drop=True)
    if cutoff is not None:
        frame = frame.loc[frame["trading_day"] <= cutoff].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError("max_trading_day removes every minute row")
    frame["source_index"] = np.arange(len(frame), dtype=np.int64)
    iso = frame["trading_day"].dt.isocalendar()
    frame["iso_key"] = iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)
    return frame


def causal_partial_bars(minute: pd.DataFrame, key: str) -> pd.DataFrame:
    group = minute.groupby(key, sort=False)
    result = pd.DataFrame(index=minute.index)
    result[key] = minute[key]
    result["open"] = group["open"].transform("first")
    result["high"] = group["high"].cummax()
    result["low"] = group["low"].cummin()
    result["close"] = minute["close"]
    result["volume"] = group["volume"].cumsum()
    result["open_interest"] = minute["open_interest"]
    result["source_bar_count"] = group.cumcount().to_numpy(dtype=np.int64) + 1
    result["source_min_index"] = group["source_index"].transform("first")
    result["source_max_index"] = minute["source_index"]
    return result


def completed_bars(partial: pd.DataFrame, key: str) -> pd.DataFrame:
    result = partial.groupby(key, sort=True, as_index=False).tail(1).copy()
    return result.sort_values(key).reset_index(drop=True)
