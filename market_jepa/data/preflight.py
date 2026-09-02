from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import MarketData
from .schema import MARKET_FEATURES


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(values: np.ndarray, names: list[str]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "features": {}}
    quantiles = np.quantile(array, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)
    features: dict[str, Any] = {}
    for index, name in enumerate(names):
        features[name] = {
            "mean": float(array[:, index].mean()),
            "std": float(array[:, index].std(ddof=0)),
            "q01": float(quantiles[0, index]),
            "q25": float(quantiles[1, index]),
            "q50": float(quantiles[2, index]),
            "q75": float(quantiles[3, index]),
            "q99": float(quantiles[4, index]),
        }
    return {"count": int(len(array)), "features": features}


def _period_distributions(
    data: MarketData,
    timeframe: str,
    train_mask: np.ndarray,
    completed_mask: np.ndarray,
) -> dict[str, Any]:
    partial_frame = getattr(data, f"{timeframe}_partial")
    completed_frame = getattr(data, f"{timeframe}_completed")
    partial_market = getattr(data, f"{timeframe}_partial_market_raw")
    completed_market = getattr(data, f"{timeframe}_completed_market_raw")
    key = "trading_day" if timeframe == "daily" else "iso_key"
    final_count = partial_frame.groupby(key)["source_bar_count"].transform("max").to_numpy()
    progress = partial_frame["source_bar_count"].to_numpy() / final_count
    partial_values = np.concatenate(
        [partial_market, partial_frame[["source_bar_count"]].to_numpy(dtype=np.float64)], axis=1
    )
    completed_values = np.concatenate(
        [
            completed_market,
            completed_frame[["source_bar_count"]].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    names = [*MARKET_FEATURES, "source_bar_count"]
    return {
        "completed": _statistics(completed_values[completed_mask], names),
        "partial_all": _statistics(partial_values[train_mask], names),
        "partial_early": _statistics(partial_values[train_mask & (progress <= 0.25)], names),
        "partial_late": _statistics(partial_values[train_mask & (progress > 0.75)], names),
    }


def _top_records(data: MarketData, category: str, score: np.ndarray, top_n: int) -> list[dict[str, Any]]:
    source_index = data.minute["source_index"].to_numpy(dtype=np.int64)
    valid = np.flatnonzero(np.isfinite(score))
    order = np.lexsort((source_index[valid], -score[valid]))[:top_n]
    records: list[dict[str, Any]] = []
    for rank, row in enumerate(valid[order], start=1):
        current = data.minute.iloc[row]
        previous = data.minute.iloc[row - 1] if row else current
        records.append(
            {
                "category": category,
                "rank": rank,
                "source_index": int(current["source_index"]),
                "previous_source_index": int(previous["source_index"]),
                "timestamp": current["timestamp"].isoformat(),
                "previous_timestamp": previous["timestamp"].isoformat(),
                "score": float(score[row]),
                "open": float(current["open"]),
                "close": float(current["close"]),
                "previous_close": float(previous["close"]),
                "volume": float(current["volume"]),
                "previous_volume": float(previous["volume"]),
                "open_interest": float(current["open_interest"]),
                "previous_open_interest": float(previous["open_interest"]),
                "delta_minutes": float(
                    (current["timestamp"] - previous["timestamp"]).total_seconds() / 60.0
                ),
            }
        )
    return records


def build_preflight_report(data: MarketData, config: dict[str, Any], top_n: int = 100) -> dict[str, Any]:
    frame = data.minute
    close = frame["close"].to_numpy(dtype=np.float64)
    open_ = frame["open"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)
    oi = frame["open_interest"].to_numpy(dtype=np.float64)
    previous_close = np.roll(close, 1)
    previous_close[0] = close[0]
    delta = frame["timestamp"].diff().dt.total_seconds().div(60).to_numpy()
    scores = {
        "absolute_1m_log_return": np.abs(np.log(close / previous_close)),
        "cross_observation_gap": np.where(delta > 1, np.abs(np.log(open_ / previous_close)), np.nan),
        "volume_log_jump": np.abs(np.diff(np.log1p(volume), prepend=np.log1p(volume[0]))),
        "open_interest_log_jump": np.abs(np.diff(np.log1p(oi), prepend=np.log1p(oi[0]))),
    }
    for values in scores.values():
        values[0] = np.nan

    start, end = map(pd.Timestamp, config["data"]["splits"]["train"])
    train_mask = frame["trading_day"].between(start, end).to_numpy()
    daily_completed_mask = data.daily_completed["trading_day"].between(start, end).to_numpy()
    weekly_completed_mask = data.weekly_completed["trading_day"].between(start, end).to_numpy()
    source_path = config["data"]["csv_path"]
    report = {
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "rows": int(len(frame)),
            "timestamp_start": frame.iloc[0]["timestamp"].isoformat(),
            "timestamp_end": frame.iloc[-1]["timestamp"].isoformat(),
            "trading_days": int(frame["trading_day"].nunique()),
            "trading_weeks": int(frame["iso_key"].nunique()),
        },
        "structural_checks": {
            "missing_values": int(frame.isna().sum().sum()),
            "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
            "strictly_increasing": bool(frame["timestamp"].is_monotonic_increasing),
            "non_finite_numeric": int(
                (~np.isfinite(frame[["open", "high", "low", "close", "volume", "open_interest"]])).sum().sum()
            ),
        },
        "normalization_distributions": {
            "daily": _period_distributions(data, "daily", train_mask, daily_completed_mask),
            "weekly": _period_distributions(data, "weekly", train_mask, weekly_completed_mask),
        },
        "anomalies": {
            name: _top_records(data, name, values, top_n) for name, values in scores.items()
        },
    }
    return report


def run_preflight(
    data: MarketData,
    config: dict[str, Any],
    output_dir: str | Path,
    top_n: int = 100,
) -> tuple[dict[str, Any], Path, Path]:
    report = build_preflight_report(data, config, top_n=top_n)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "preflight.json"
    csv_path = output / "preflight_anomalies.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    records = [record for group in report["anomalies"].values() for record in group]
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return report, json_path, csv_path
