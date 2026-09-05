#!/usr/bin/env python3
"""Independent Price x OI x Volume structural audit (2018-2024 only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


LOOKBACKS = (4, 16, 64)
HORIZONS = (16, 64, 256)
MAX_HORIZON = 256
VOLUME_BASELINE_WINDOWS = 16
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
CUTOFF = pd.Timestamp("2024-12-31")
PERIODS = {
    "2018-2021": (pd.Timestamp("2018-01-02"), pd.Timestamp("2021-12-31")),
    "2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "2023-2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
}
YEARS = {
    str(year): (pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
    for year in range(2018, 2025)
}
STATES = (
    "price_up_oi_up",
    "price_down_oi_up",
    "price_up_oi_down",
    "price_down_oi_down",
)
STATE_LABELS = {
    "price_up_oi_up": "Price Up + OI Up",
    "price_down_oi_up": "Price Down + OI Up",
    "price_up_oi_down": "Price Up + OI Down",
    "price_down_oi_down": "Price Down + OI Down",
}
RAW_COLUMNS = ("open", "high", "low", "close", "volume", "open_interest")


def _rows_through_cutoff(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        candidates = [
            name
            for name in reader.fieldnames
            if name.strip().lower() in {"date", "datetime", "timestamp"}
        ]
        if len(candidates) != 1:
            raise ValueError("CSV must contain exactly one timestamp column")
        timestamp_column = candidates[0]
        rows = 0
        for row in reader:
            timestamp = pd.Timestamp(row[timestamp_column])
            calendar_day = timestamp.normalize()
            if calendar_day > CUTOFF or (
                calendar_day == CUTOFF and timestamp.hour >= 18
            ):
                break
            rows += 1
    return rows


def _infer_trading_day(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="raise")
    natural = ts.dt.normalize().to_numpy(dtype="datetime64[ns]")
    night = ts.dt.hour.to_numpy() >= 18
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


def load_minutes(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source, nrows=_rows_through_cutoff(source))
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
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "").replace("-", "_")
        if key in aliases:
            rename[column] = aliases[key]
    frame = frame.rename(columns=rename)
    missing = {"timestamp", *RAW_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for column in RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be unique and increasing")
    raw = frame.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("OHLCV/OI contains non-finite values")
    if np.any(raw[:, :4] <= 0) or np.any(raw[:, 4:] < 0):
        raise ValueError("invalid non-positive price or negative Volume/OI")
    frame["trading_day"] = _infer_trading_day(frame["timestamp"])
    frame = frame.loc[frame["trading_day"].notna()].copy().reset_index(drop=True)
    frame = frame.loc[frame["trading_day"] <= CUTOFF].copy().reset_index(drop=True)
    frame["source_index"] = np.arange(len(frame), dtype=np.int64)
    iso = frame["trading_day"].dt.isocalendar()
    frame["iso_key"] = iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)
    if frame.empty or frame["trading_day"].max() > CUTOFF:
        raise RuntimeError("2025 embargo failed")
    return frame


def future_outcomes(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    log_return_sq = np.square(np.diff(np.log(close)))
    result: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        rows = len(close) - horizon
        anchors = np.arange(rows, dtype=np.int64)
        future_high = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)[:rows]
        future_low = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)[:rows]
        future_rv = np.lib.stride_tricks.sliding_window_view(log_return_sq, horizon)[:rows]
        result[horizon] = np.column_stack(
            [
                close[anchors + horizon] / close[anchors] - 1.0,
                future_high.max(axis=1) / close[anchors] - 1.0,
                future_low.min(axis=1) / close[anchors] - 1.0,
                np.sqrt(future_rv.mean(axis=1)),
            ]
        ).astype(np.float32)
    return result


def _rolling_sum(values: np.ndarray, length: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[length - 1 :] = cumulative[length:] - cumulative[:-length]
    return result


def structural_samples(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    close = frame["close"].to_numpy(dtype=np.float64)
    oi = frame["open_interest"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)
    first_anchor = (VOLUME_BASELINE_WINDOWS + 1) * lookback - 1
    anchors = np.arange(first_anchor, len(frame) - MAX_HORIZON, dtype=np.int64)
    current_volume = _rolling_sum(volume, lookback)[anchors]
    window_volume = _rolling_sum(volume, lookback)
    previous = np.stack(
        [window_volume[anchors - offset * lookback] for offset in range(1, 17)],
        axis=1,
    )
    baseline = np.median(previous, axis=1)
    price_change = np.log(close[anchors] / close[anchors - lookback])
    oi_valid = (oi[anchors] > 0) & (oi[anchors - lookback] > 0)
    oi_change = np.full(len(anchors), np.nan, dtype=np.float64)
    oi_change[oi_valid] = np.log(oi[anchors[oi_valid]] / oi[anchors[oi_valid] - lookback])
    valid = (
        oi_valid
        & np.isfinite(price_change)
        & np.isfinite(oi_change)
        & (price_change != 0)
        & (oi_change != 0)
        & np.isfinite(current_volume)
        & np.isfinite(baseline)
        & (current_volume > 0)
        & (baseline > 0)
    )
    anchors = anchors[valid]
    price_change = price_change[valid]
    oi_change = oi_change[valid]
    intensity = np.log(current_volume[valid] / baseline[valid])
    state = np.select(
        [
            (price_change > 0) & (oi_change > 0),
            (price_change < 0) & (oi_change > 0),
            (price_change > 0) & (oi_change < 0),
            (price_change < 0) & (oi_change < 0),
        ],
        STATES,
        default="invalid",
    )
    trading_day = frame["trading_day"].to_numpy(dtype="datetime64[ns]")
    return pd.DataFrame(
        {
            "anchor_index": anchors,
            "trading_day": trading_day[anchors],
            "target_end_day": trading_day[anchors + MAX_HORIZON],
            "iso_key": frame["iso_key"].to_numpy(dtype=np.int64)[anchors],
            "state": state,
            "volume_intensity": intensity,
        }
    )


def _period_mask(samples: pd.DataFrame, bounds: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    start, end = (np.datetime64(value) for value in bounds)
    day = samples["trading_day"].to_numpy(dtype="datetime64[ns]")
    target_end = samples["target_end_day"].to_numpy(dtype="datetime64[ns]")
    return (day >= start) & (day <= end) & (target_end <= end)


def fit_volume_thresholds(samples: dict[int, pd.DataFrame]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for lookback, values in samples.items():
        train = values.loc[_period_mask(values, PERIODS["2018-2021"]), "volume_intensity"].to_numpy()
        if not len(train) or not np.isfinite(train).all():
            raise RuntimeError(f"no finite Train volume intensity for L={lookback}")
        q1, q2, q3 = np.quantile(train, [0.25, 0.5, 0.75])
        result[lookback] = {
            "q1": float(q1),
            "q2": float(q2),
            "q3": float(q3),
            "train_samples": int(len(train)),
            "fit_period": "2018-2021",
        }
    return result


def assign_volume_bucket(intensity: np.ndarray, threshold: dict[str, Any]) -> np.ndarray:
    cuts = np.asarray([threshold["q1"], threshold["q2"], threshold["q3"]])
    return np.searchsorted(cuts, intensity, side="left").astype(np.int8) + 1


def _cell_stats(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {
            "N": 0,
            "p_return_up": np.nan,
            "p_return_down": np.nan,
            "return_mean": np.nan,
            "return_median": np.nan,
            "mfe_mean": np.nan,
            "mfe_median": np.nan,
            "mae_mean": np.nan,
            "mae_median": np.nan,
            "rv_mean": np.nan,
            "rv_median": np.nan,
        }
    return {
        "N": int(len(values)),
        "p_return_up": float(np.mean(values[:, 0] > 0)),
        "p_return_down": float(np.mean(values[:, 0] < 0)),
        "return_mean": float(np.mean(values[:, 0], dtype=np.float64)),
        "return_median": float(np.median(values[:, 0])),
        "mfe_mean": float(np.mean(values[:, 1], dtype=np.float64)),
        "mfe_median": float(np.median(values[:, 1])),
        "mae_mean": float(np.mean(values[:, 2], dtype=np.float64)),
        "mae_median": float(np.median(values[:, 2])),
        "rv_mean": float(np.mean(values[:, 3], dtype=np.float64)),
        "rv_median": float(np.median(values[:, 3])),
    }


def build_cells(
    samples: dict[int, pd.DataFrame],
    thresholds: dict[int, dict[str, Any]],
    outcomes: dict[int, np.ndarray],
    populations: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, bounds in populations.items():
        for lookback, source in samples.items():
            current = source.loc[_period_mask(source, bounds)].copy()
            current["volume_bucket"] = assign_volume_bucket(
                current["volume_intensity"].to_numpy(), thresholds[lookback]
            )
            anchors = current["anchor_index"].to_numpy(dtype=np.int64)
            for horizon in HORIZONS:
                future = outcomes[horizon][anchors]
                for state in STATES:
                    state_mask = current["state"].to_numpy() == state
                    for volume_name, bucket in [("ALL", None), ("V1", 1), ("V2", 2), ("V3", 3), ("V4", 4)]:
                        selected = state_mask
                        if bucket is not None:
                            selected = selected & (current["volume_bucket"].to_numpy() == bucket)
                        rows.append(
                            {
                                "period": period,
                                "lookback": lookback,
                                "horizon": horizon,
                                "state": state,
                                "state_label": STATE_LABELS[state],
                                "volume_bucket": volume_name,
                                **_cell_stats(future[selected]),
                            }
                        )
    return pd.DataFrame(rows)


def _block_probability_effect(
    success_a: np.ndarray,
    blocks_a: np.ndarray,
    week_universe: np.ndarray,
    success_b: np.ndarray | None = None,
    blocks_b: np.ndarray | None = None,
) -> dict[str, Any]:
    def aggregate(success: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.searchsorted(week_universe, blocks)
        sums = np.bincount(positions, weights=success.astype(np.float64), minlength=len(week_universe))
        counts = np.bincount(positions, minlength=len(week_universe)).astype(np.float64)
        return sums, counts

    sum_a, count_a = aggregate(success_a, blocks_a)
    if not count_a.sum():
        raise RuntimeError("bootstrap group A is empty")
    if success_b is not None:
        if blocks_b is None:
            raise ValueError("blocks_b is required with success_b")
        sum_b, count_b = aggregate(success_b, blocks_b)
        if not count_b.sum():
            raise RuntimeError("bootstrap group B is empty")
        point = sum_a.sum() / count_a.sum() - sum_b.sum() / count_b.sum()
    else:
        sum_b = count_b = None
        point = sum_a.sum() / count_a.sum() - 0.5
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[np.ndarray] = []
    for start in range(0, BOOTSTRAP_SAMPLES, 1000):
        size = min(1000, BOOTSTRAP_SAMPLES - start)
        draw = rng.integers(0, len(week_universe), size=(size, len(week_universe)))
        denominator_a = count_a[draw].sum(axis=1)
        estimate = sum_a[draw].sum(axis=1) / denominator_a
        valid = denominator_a > 0
        if sum_b is not None and count_b is not None:
            denominator_b = count_b[draw].sum(axis=1)
            valid &= denominator_b > 0
            estimate = estimate - sum_b[draw].sum(axis=1) / denominator_b
        else:
            estimate = estimate - 0.5
        estimates.append(estimate[valid])
    bootstrap = np.concatenate(estimates)
    if len(bootstrap) != BOOTSTRAP_SAMPLES:
        raise RuntimeError("a bootstrap resample contained no observations for a required group")
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "effect": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "blocks": int(len(week_universe)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def build_bootstrap_effects(
    samples: dict[int, pd.DataFrame],
    thresholds: dict[int, dict[str, Any]],
    outcomes: dict[int, np.ndarray],
) -> pd.DataFrame:
    definitions = (
        ("long_build_up_minus_0_5", "price_up_oi_up", "up", None),
        ("short_build_down_minus_0_5", "price_down_oi_up", "down", None),
        ("up_oi_up_minus_oi_down", "price_up_oi_up", "up", "price_up_oi_down"),
        ("down_oi_up_minus_oi_down", "price_down_oi_up", "down", "price_down_oi_down"),
    )
    rows: list[dict[str, Any]] = []
    for period, bounds in PERIODS.items():
        for lookback, source in samples.items():
            current = source.loc[_period_mask(source, bounds)].copy()
            current["volume_bucket"] = assign_volume_bucket(
                current["volume_intensity"].to_numpy(), thresholds[lookback]
            )
            current = current.loc[current["volume_bucket"] == 4]
            weeks = np.sort(current["iso_key"].unique())
            anchors = current["anchor_index"].to_numpy(dtype=np.int64)
            states = current["state"].to_numpy()
            blocks = current["iso_key"].to_numpy(dtype=np.int64)
            for horizon in HORIZONS:
                returns = outcomes[horizon][anchors, 0]
                for effect_name, state_a, direction, state_b in definitions:
                    mask_a = states == state_a
                    success = returns > 0 if direction == "up" else returns < 0
                    kwargs: dict[str, Any] = {}
                    n_b = 0
                    if state_b is not None:
                        mask_b = states == state_b
                        kwargs = {
                            "success_b": success[mask_b],
                            "blocks_b": blocks[mask_b],
                        }
                        n_b = int(mask_b.sum())
                    result = _block_probability_effect(
                        success[mask_a], blocks[mask_a], weeks, **kwargs
                    )
                    rows.append(
                        {
                            "period": period,
                            "lookback": lookback,
                            "horizon": horizon,
                            "volume_bucket": "V4",
                            "effect_name": effect_name,
                            "n_a": int(mask_a.sum()),
                            "n_b": n_b,
                            **result,
                            "ci_positive": bool(result["ci95_low"] > 0),
                            "ci_negative": bool(result["ci95_high"] < 0),
                        }
                    )
    return pd.DataFrame(rows)


def _question_answers(cells: pd.DataFrame, yearly: pd.DataFrame, bootstrap: pd.DataFrame) -> list[str]:
    oos = bootstrap[bootstrap["period"].isin(["2022", "2023-2024"])]

    def positive_count(effect: str) -> tuple[int, int]:
        selected = oos[oos["effect_name"] == effect]
        return int(selected["ci_positive"].sum()), len(selected)

    long_positive, long_total = positive_count("long_build_up_minus_0_5")
    short_positive, short_total = positive_count("short_build_down_minus_0_5")
    up_difference = oos[oos["effect_name"] == "up_oi_up_minus_oi_down"]
    down_difference = oos[oos["effect_name"] == "down_oi_up_minus_oi_down"]

    monotonic = 0
    strengthened = 0
    volume_cases = 0
    for period in ("2022", "2023-2024"):
        for lookback in LOOKBACKS:
            for horizon in HORIZONS:
                for state, column in (
                    ("price_up_oi_up", "p_return_up"),
                    ("price_down_oi_up", "p_return_down"),
                ):
                    rows = cells[
                        (cells["period"] == period)
                        & (cells["lookback"] == lookback)
                        & (cells["horizon"] == horizon)
                        & (cells["state"] == state)
                        & cells["volume_bucket"].isin(["V1", "V2", "V3", "V4"])
                    ].sort_values("volume_bucket")
                    values = rows[column].to_numpy(dtype=np.float64)
                    if len(values) == 4 and np.isfinite(values).all():
                        volume_cases += 1
                        monotonic += int(np.all(np.diff(values) >= 0))
                        strengthened += int(values[-1] > values[0])

    stable: list[tuple[int, float, int, int, str]] = []
    for lookback in LOOKBACKS:
        for horizon in HORIZONS:
            for state in STATES:
                rows = yearly[
                    (yearly["lookback"] == lookback)
                    & (yearly["horizon"] == horizon)
                    & (yearly["state"] == state)
                    & (yearly["volume_bucket"] == "V4")
                ].sort_values("period")
                column = "p_return_up" if "price_up" in state else "p_return_down"
                values = rows[column].to_numpy(dtype=np.float64)
                valid = np.isfinite(values)
                count = int(np.sum(values[valid] > 0.5))
                dispersion = float(np.std(values[valid])) if valid.any() else np.inf
                stable.append((count, dispersion, lookback, horizon, state))
    stable.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    top = "; ".join(
        f"{STATE_LABELS[state]} L={lookback}/H={horizon}: {count}/7 years"
        for count, _, lookback, horizon, state in stable[:5]
    )
    up_significant = int((up_difference["ci_positive"] | up_difference["ci_negative"]).sum())
    down_significant = int((down_difference["ci_positive"] | down_difference["ci_negative"]).sum())
    return [
        f"1. OOS V4 long-build P(up)-0.5 has positive 95% CI in {long_positive}/{long_total} L/H/period cells.",
        f"2. OOS V4 short-build P(down)-0.5 has positive 95% CI in {short_positive}/{short_total} L/H/period cells.",
        f"3. Directional probability is monotonic V1->V4 in {monotonic}/{volume_cases} OOS cases and V4 exceeds V1 in {strengthened}/{volume_cases} cases.",
        f"4. Price-up OI-up versus OI-down probability difference excludes zero in {up_significant}/{len(up_difference)} OOS cells.",
        f"5. Price-down OI-up versus OI-down probability difference excludes zero in {down_significant}/{len(down_difference)} OOS cells.",
        f"6. Highest cross-year directional consistency: {top}.",
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_markdown(
    path: Path,
    frame: pd.DataFrame,
    thresholds: dict[int, dict[str, Any]],
    bootstrap: pd.DataFrame,
    answers: list[str],
) -> None:
    lines = [
        "# Structural Market Pattern Audit — Price × OI × Volume",
        "",
        f"Data rows through 2024-12-31: {len(frame):,}",
        "",
        "2025 Test consumed: **false**",
        "",
        "## Train-only volume-intensity quartiles",
        "",
        "| L | Q1 | Q2 | Q3 | Train N |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lookback in LOOKBACKS:
        value = thresholds[lookback]
        lines.append(
            f"| {lookback} | {value['q1']:.6g} | {value['q2']:.6g} | {value['q3']:.6g} | {value['train_samples']:,} |"
        )
    lines.extend(
        [
            "",
            "## ISO-week bootstrap overview",
            "",
            "Positive means the named directional effect is stronger; no threshold was tuned after seeing OOS.",
            "",
            "| Period | Effect | CI lower > 0 | Total | Median effect |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for period in PERIODS:
        for effect_name in bootstrap["effect_name"].unique():
            selected = bootstrap[
                (bootstrap["period"] == period) & (bootstrap["effect_name"] == effect_name)
            ]
            lines.append(
                f"| {period} | {effect_name} | {int(selected['ci_positive'].sum())} | {len(selected)} | {selected['effect'].median():.6g} |"
            )
    lines.extend(["", "## Required questions", "", *answers, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(csv_path: str | Path, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame = load_minutes(csv_path)
    outcomes = future_outcomes(frame)
    samples = {lookback: structural_samples(frame, lookback) for lookback in LOOKBACKS}
    thresholds = fit_volume_thresholds(samples)
    cells = build_cells(samples, thresholds, outcomes, PERIODS)
    yearly = build_cells(samples, thresholds, outcomes, YEARS)
    bootstrap = build_bootstrap_effects(samples, thresholds, outcomes)
    answers = _question_answers(cells, yearly, bootstrap)

    cells.to_csv(output / "all_cells.csv", index=False)
    yearly.to_csv(output / "yearly_cells.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_effects.csv", index=False)
    (output / "volume_thresholds.json").write_text(
        json.dumps(_json_safe(thresholds), indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": "Structural Market Pattern Audit — Price x OI x Volume",
        "source": str(csv_path),
        "max_loaded_trading_day": str(frame["trading_day"].max().date()),
        "rows": len(frame),
        "lookbacks": list(LOOKBACKS),
        "horizons": list(HORIZONS),
        "volume_threshold_fit_period": "2018-2021",
        "bootstrap": {
            "block": "ISO trading week",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "estimator": "sample-weighted block sums / block counts",
        },
        "questions": answers,
        "bootstrap_effects": bootstrap.to_dict(orient="records"),
        "test_consumed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_markdown(output / "summary.md", frame, thresholds, bootstrap, answers)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="8Y_DCE_JM2601_1m.csv")
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/structural_market_pattern_audit",
    )
    args = parser.parse_args()
    result = run_audit(args.csv, args.output)
    print(result)
    print("test_consumed=false")


if __name__ == "__main__":
    main()
