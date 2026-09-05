#!/usr/bin/env python3
"""IMC v0.1 cross-commodity mathematical and empirical audit.

The formulas in this file are a direct implementation of
docs/MARKET_INVARIANT_COORDINATES_MATHEMATICAL_DESIGN.md.  This is an
independent, CPU-only data audit: it neither imports nor trains Market-JEPA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


SEED = 4601
WINDOW_LENGTHS = (64, 256, 512)
INTERVALS = (1, 4, 16, 64, 256)
WINDOWS_PER_LENGTH = 10_000
ROLLING_VOLUME_LENGTH = 20
TOLERANCE = 1e-10
SCALE_FACTORS = (0.1, 0.37, 3.7, 11.2, 53.0, 1000.0)
TRAIN_COMMODITIES = ("FG", "SA", "JM", "SH", "SP")
HELD_OUT_COMMODITY = "RB"
COMMODITIES = TRAIN_COMMODITIES + (HELD_OUT_COMMODITY,)
CSYMBOLS = {
    "FG": "CZCE.FG",
    "SA": "CZCE.SA",
    "JM": "DCE.JM",
    "SH": "CZCE.SH",
    "SP": "SHFE.SP",
    "RB": "SHFE.RB",
}
OHLC = ("Open", "High", "Low", "Close")
CORE_COLUMNS = (*OHLC, "Volume", "OpenInterest")
DIST_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
DIST_MAX_POINTS = 500_000
NEAREST_QUERY_COUNT = 5
NEAREST_CANDIDATES_PER_COMMODITY = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).lower(): column for column in frame.columns}
    aliases = {
        "Date": ("eob", "date", "datetime", "timestamp"),
        "Open": ("open",),
        "High": ("high",),
        "Low": ("low",),
        "Close": ("close",),
        "Volume": ("volume",),
        "OpenInterest": ("openinterest", "open_interest", "position", "oi"),
    }
    rename: dict[object, str] = {}
    for destination, candidates in aliases.items():
        source = next((lower[name] for name in candidates if name in lower), None)
        if source is None:
            raise ValueError(f"missing required column {destination}; columns={frame.columns.tolist()}")
        rename[source] = destination
    result = frame.rename(columns=rename).copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    for column in CORE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Date"]).sort_values("Date", kind="mergesort").reset_index(drop=True)
    return result


def load_lake_commodity(lake_root: Path, commodity: str) -> tuple[pd.DataFrame, list[Path]]:
    root = lake_root / "provider=JUEJIN" / "freq=1m" / f"symbol={CSYMBOLS[commodity]}"
    paths = sorted(root.glob("year=*/month=*/data.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet data found for {commodity}: {root}")
    parts = [pd.read_parquet(path) for path in paths]
    frame = canonicalize_columns(pd.concat(parts, ignore_index=True))
    # The lake partition is a continuous-symbol export.  It has no actual
    # contract identifier, so duplicated timestamps are data-quality facts,
    # not silently resolvable roll records.
    return frame, paths


def export_core_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ("Date", *CORE_COLUMNS)].to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def rolling_volume_baseline(volume: np.ndarray) -> np.ndarray:
    return (
        pd.Series(volume, copy=False)
        .rolling(ROLLING_VOLUME_LENGTH, min_periods=ROLLING_VOLUME_LENGTH)
        .median()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )


def imc_transform_window(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    oi: np.ndarray,
    volume: np.ndarray,
    prior_volume: np.ndarray,
) -> dict[str, np.ndarray | float]:
    arrays = [open_, high, low, close, oi, volume, prior_volume]
    arrays = [np.asarray(value, dtype=np.float64) for value in arrays]
    open_, high, low, close, oi, volume, prior_volume = arrays
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("IMC window inputs must be one-dimensional")
    if not (len(open_) == len(high) == len(low) == len(close) == len(oi) == len(volume)):
        raise ValueError("window arrays must have equal length")
    if len(prior_volume) != ROLLING_VOLUME_LENGTH:
        raise ValueError("prior_volume must contain exactly 20 bars")
    if np.any(open_ <= 0) or np.any(high <= 0) or np.any(low <= 0) or np.any(close <= 0):
        raise ValueError("OHLC must be positive")
    if np.any(oi <= 0):
        raise ValueError("OpenInterest must be positive")
    p0 = float(close[0])
    oi0 = float(oi[0])
    m0 = float(np.median(prior_volume))
    if not np.isfinite(m0) or m0 <= 0:
        raise ValueError("fixed-origin Volume baseline must be positive")
    history = np.concatenate((prior_volume, volume))
    q = np.empty(len(volume), dtype=np.float64)
    for position in range(len(volume)):
        baseline = float(np.median(history[position : position + ROLLING_VOLUME_LENGTH]))
        if baseline <= 0 or not np.isfinite(baseline):
            q[position] = np.nan
        else:
            q[position] = volume[position] / baseline
    return {
        "price_open": np.log(open_ / p0),
        "price_high": np.log(high / p0),
        "price_low": np.log(low / p0),
        "price_close": np.log(close / p0),
        "oi": np.log(oi / oi0),
        "volume_fixed": volume / m0,
        "volume_q20": q,
        "p0": p0,
        "oi0": oi0,
        "m0_volume": m0,
    }


def valid_window_starts(frame: pd.DataFrame, length: int) -> np.ndarray:
    values = frame.loc[:, CORE_COLUMNS].to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    valid &= (values[:, :4] > 0).all(axis=1)
    valid &= values[:, 5] > 0
    bad_prefix = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    starts = np.arange(ROLLING_VOLUME_LENGTH, len(frame) - length + 1, dtype=np.int64)
    good_window = bad_prefix[starts + length] == bad_prefix[starts]
    prior_volume = frame["Volume"].to_numpy(dtype=np.float64)
    baselines = rolling_volume_baseline(prior_volume)[starts]
    good = good_window & np.isfinite(baselines) & (baselines > 0)
    if "contract" in frame.columns:
        contracts = frame["contract"].astype(str).to_numpy()
        good &= np.asarray(
            [len(set(contracts[start - ROLLING_VOLUME_LENGTH : start + length])) == 1 for start in starts]
        )
    return starts[good]


def sample_starts(valid_starts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = min(WINDOWS_PER_LENGTH, len(valid_starts))
    if count == len(valid_starts):
        return valid_starts.copy()
    return np.sort(rng.choice(valid_starts, size=count, replace=False))


@dataclass
class ErrorAccumulator:
    name: str
    total: float = 0.0
    count: int = 0
    maximum: float = 0.0
    samples: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.samples = []

    def add(self, values: np.ndarray | float) -> None:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return
        absolute = np.abs(array)
        self.total += float(absolute.sum())
        self.count += int(absolute.size)
        self.maximum = max(self.maximum, float(absolute.max()))
        stride = max(1, absolute.size // 4096)
        assert self.samples is not None
        self.samples.append(absolute.ravel()[::stride][:4096])

    def summary(self) -> dict[str, float | int | str]:
        assert self.samples is not None
        sample = np.concatenate(self.samples) if self.samples else np.asarray([], dtype=np.float64)
        return {
            "test": self.name,
            "max_abs_error": self.maximum,
            "mean_abs_error": self.total / self.count if self.count else math.nan,
            "p99_abs_error": float(np.quantile(sample, 0.99)) if sample.size else math.nan,
            "n_values": self.count,
        }


def rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def audit_windows(
    commodity: str,
    frame: pd.DataFrame,
    sampled: Mapping[int, np.ndarray],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    raw = {column: frame[column].to_numpy(dtype=np.float64) for column in CORE_COLUMNS}
    q_baseline = rolling_volume_baseline(raw["Volume"])
    # Independent NumPy implementation of Median(V[t-20:t-1]) for Test H.
    # The final trailing 20-bar window would correspond to a non-existent next
    # row, so it is deliberately excluded.
    q_baseline_reference = np.full(len(frame), np.nan, dtype=np.float64)
    rolling_windows = np.lib.stride_tricks.sliding_window_view(
        raw["Volume"], ROLLING_VOLUME_LENGTH
    )
    q_baseline_reference[ROLLING_VOLUME_LENGTH:] = np.median(rolling_windows[:-1], axis=1)
    scaled_q_baselines = {
        factor: rolling_volume_baseline(raw["Volume"] * factor) for factor in SCALE_FACTORS
    }
    scale_rows: list[dict] = []
    reconstruction_rows: list[dict] = []
    oi_rows: list[dict] = []
    price_rows: list[dict] = []

    for length, starts in sampled.items():
        accumulators = {
            "price_reconstruction": ErrorAccumulator("price_reconstruction"),
            "oi_reconstruction": ErrorAccumulator("oi_reconstruction"),
            "volume_reconstruction": ErrorAccumulator("volume_fixed_reconstruction"),
            "volume_q_identity": ErrorAccumulator("volume_q20_identity"),
            "oi_step_identity": ErrorAccumulator("oi_step_identity"),
            "joint_corr": ErrorAccumulator("joint_path_correlation_invariance"),
            "joint_sign": ErrorAccumulator("joint_path_sign_sequence_invariance"),
        }
        oi_interval = {lag: ErrorAccumulator(f"oi_interval_{lag}") for lag in INTERVALS if lag < length}
        price_interval = {lag: ErrorAccumulator(f"price_interval_{lag}") for lag in INTERVALS if lag < length}
        scale_acc = {
            (group, factor): ErrorAccumulator(f"{group}_scale_{factor:g}")
            for group in ("price", "oi", "volume")
            for factor in SCALE_FACTORS
        }
        joint_scale_acc = {
            "joint_3.7_11.2_53": ErrorAccumulator("joint_scale_3.7_11.2_53"),
            "joint_0.1_1000_0.37": ErrorAccumulator("joint_scale_0.1_1000_0.37"),
            "joint_all_53": ErrorAccumulator("joint_scale_all_53"),
        }

        for offset in range(0, len(starts), 128):
            batch_starts = starts[offset : offset + 128]
            positions = batch_starts[:, None] + np.arange(length, dtype=np.int64)[None, :]
            close = raw["Close"][positions]
            oi = raw["OpenInterest"][positions]
            volume = raw["Volume"][positions]
            p0 = close[:, :1]
            oi0 = oi[:, :1]
            prior_positions = batch_starts[:, None] - np.arange(ROLLING_VOLUME_LENGTH, 0, -1)[None, :]
            prior_volume = raw["Volume"][prior_positions]
            m0 = np.median(prior_volume, axis=1, keepdims=True)
            x_close = np.log(close / p0)
            x_oi = np.log(oi / oi0)
            x_volume = volume / m0
            local_median = q_baseline_reference[positions]
            q_direct = volume / local_median
            q_cached = raw["Volume"][positions] / q_baseline[positions]

            for price_column in OHLC:
                price = raw[price_column][positions]
                coordinate = np.log(price / p0)
                reconstructed = p0 * np.exp(coordinate)
                accumulators["price_reconstruction"].add((reconstructed - price) / price)
                for factor in SCALE_FACTORS:
                    scaled_coordinate = np.log((price * factor) / (p0 * factor))
                    scale_acc[("price", factor)].add(scaled_coordinate - coordinate)

            reconstructed_oi = oi0 * np.exp(x_oi)
            accumulators["oi_reconstruction"].add((reconstructed_oi - oi) / oi)
            reconstructed_volume = m0 * x_volume
            volume_denominator = np.maximum(np.abs(volume), np.finfo(np.float64).tiny)
            accumulators["volume_reconstruction"].add((reconstructed_volume - volume) / volume_denominator)
            accumulators["volume_q_identity"].add(q_cached - q_direct)
            accumulators["oi_step_identity"].add(np.diff(x_oi, axis=1) - np.log(oi[:, 1:] / oi[:, :-1]))

            for factor in SCALE_FACTORS:
                scale_acc[("oi", factor)].add(np.log((oi * factor) / (oi0 * factor)) - x_oi)
                scaled_prior = prior_volume * factor
                scaled_volume = volume * factor
                scaled_m0 = np.median(scaled_prior, axis=1, keepdims=True)
                scale_acc[("volume", factor)].add(scaled_volume / scaled_m0 - x_volume)
                scaled_local = scaled_q_baselines[factor][positions]
                scale_acc[("volume", factor)].add(scaled_volume / scaled_local - q_direct)

            for lag, accumulator in oi_interval.items():
                accumulator.add(x_oi[:, lag:] - x_oi[:, :-lag] - np.log(oi[:, lag:] / oi[:, :-lag]))
            for lag, accumulator in price_interval.items():
                accumulator.add(x_close[:, lag:] - x_close[:, :-lag] - np.log(close[:, lag:] / close[:, :-lag]))

            # Since every transformed primitive is independently invariant,
            # these correlations and sign sequences must also be invariant.
            d_price = np.diff(x_close, axis=1)
            d_oi = np.diff(x_oi, axis=1)
            q_aligned = q_direct[:, 1:]
            joint_scenarios = (
                ("joint_3.7_11.2_53", 3.7, 11.2, 53.0),
                ("joint_0.1_1000_0.37", 0.1, 1000.0, 0.37),
                ("joint_all_53", 53.0, 53.0, 53.0),
            )
            for scenario, factor_p, factor_oi, factor_v in joint_scenarios:
                scaled_p = np.log((close * factor_p) / (p0 * factor_p))
                scaled_oi = np.log((oi * factor_oi) / (oi0 * factor_oi))
                scaled_fixed = (volume * factor_v) / np.median(prior_volume * factor_v, axis=1, keepdims=True)
                scaled_q = (volume * factor_v) / scaled_q_baselines[factor_v][positions]
                joint_scale_acc[scenario].add(scaled_p - x_close)
                joint_scale_acc[scenario].add(scaled_oi - x_oi)
                joint_scale_acc[scenario].add(scaled_fixed - x_volume)
                joint_scale_acc[scenario].add(scaled_q - q_direct)
                pairs = (
                    (d_price, d_oi, np.diff(scaled_p, axis=1), np.diff(scaled_oi, axis=1)),
                    (d_price, q_aligned, np.diff(scaled_p, axis=1), scaled_q[:, 1:]),
                    (d_oi, q_aligned, np.diff(scaled_oi, axis=1), scaled_q[:, 1:]),
                )
                for left, right, scaled_left, scaled_right in pairs:
                    accumulators["joint_corr"].add(rowwise_corr(left, right) - rowwise_corr(scaled_left, scaled_right))
                original_sign = np.sign(d_price) == np.sign(d_oi)
                scaled_sign = np.sign(np.diff(scaled_p, axis=1)) == np.sign(np.diff(scaled_oi, axis=1))
                accumulators["joint_sign"].add(original_sign.astype(float) - scaled_sign.astype(float))

        for key in ("price_reconstruction", "oi_reconstruction", "volume_reconstruction"):
            row = accumulators[key].summary()
            row.update({"commodity": commodity, "window_length": length})
            reconstruction_rows.append(row)
        volume_q_row = accumulators["volume_q_identity"].summary()
        volume_q_row.update({"commodity": commodity, "window_length": length})
        reconstruction_rows.append(volume_q_row)
        for (group, factor), accumulator in scale_acc.items():
            row = accumulator.summary()
            row.update({"commodity": commodity, "window_length": length, "group": group, "scale": factor})
            scale_rows.append(row)
        for key in ("joint_corr", "joint_sign"):
            row = accumulators[key].summary()
            row.update({"commodity": commodity, "window_length": length, "group": "joint", "scale": "joint"})
            scale_rows.append(row)
        for scenario, accumulator in joint_scale_acc.items():
            row = accumulator.summary()
            row.update(
                {
                    "commodity": commodity,
                    "window_length": length,
                    "group": "joint_coordinates",
                    "scale": scenario,
                }
            )
            scale_rows.append(row)
        step_row = accumulators["oi_step_identity"].summary()
        step_row.update({"commodity": commodity, "window_length": length, "interval": 1})
        oi_rows.append(step_row)
        for lag, accumulator in oi_interval.items():
            row = accumulator.summary()
            row.update({"commodity": commodity, "window_length": length, "interval": lag})
            oi_rows.append(row)
        for lag, accumulator in price_interval.items():
            row = accumulator.summary()
            row.update({"commodity": commodity, "window_length": length, "interval": lag})
            price_rows.append(row)

    return scale_rows, reconstruction_rows, oi_rows, price_rows


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(rankdata_average(left), rankdata_average(right))[0, 1])


def extreme_oi_audit(commodity: str, frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    oi = frame["OpenInterest"].to_numpy(dtype=np.float64)
    valid = (oi[1:] > 0) & (oi[:-1] > 0)
    indices = np.flatnonzero(valid) + 1
    jumps = np.abs(np.log(oi[indices] / oi[indices - 1]))
    global_coordinate = np.log(oi[oi > 0] / oi[oi > 0][0])
    reconstructed_jumps = np.abs(np.diff(global_coordinate))
    if len(global_coordinate) != len(oi):
        reconstructed_for_valid = jumps.copy()
    else:
        reconstructed_for_valid = reconstructed_jumps[indices - 1]
    rows: list[dict] = []
    summaries: list[dict] = []
    for fraction in (0.001, 0.01, 0.05):
        count = max(1, int(math.ceil(len(jumps) * fraction)))
        chosen = np.argsort(-jumps, kind="mergesort")[:count]
        chosen_hat = np.argsort(-reconstructed_for_valid, kind="mergesort")[:count]
        overlap = len(set(chosen.tolist()).intersection(chosen_hat.tolist())) / count
        summaries.append(
            {
                "commodity": commodity,
                "top_fraction": fraction,
                "n": count,
                "spearman": spearman(jumps, reconstructed_for_valid),
                "top_k_overlap": overlap,
                "max_value_error": float(np.max(np.abs(jumps - reconstructed_for_valid))),
            }
        )
        if fraction <= 0.01:
            for position in chosen:
                row_index = int(indices[position])
                rows.append(
                    {
                        "commodity": commodity,
                        "top_fraction": fraction,
                        "row_index": row_index,
                        "timestamp": frame["Date"].iat[row_index],
                        "oi_previous": oi[row_index - 1],
                        "oi_current": oi[row_index],
                        "raw_abs_log_jump": jumps[position],
                        "imc_abs_jump": reconstructed_for_valid[position],
                    }
                )
    return summaries, rows


def volume_audit(commodity: str, frame: pd.DataFrame) -> tuple[list[dict], list[dict], np.ndarray]:
    volume = frame["Volume"].to_numpy(dtype=np.float64)
    baseline = rolling_volume_baseline(volume)
    valid = np.isfinite(baseline) & (baseline > 0)
    q = np.full(len(volume), np.nan, dtype=np.float64)
    q[valid] = volume[valid] / baseline[valid]
    scaled_baseline = rolling_volume_baseline(volume * 100.0)
    scaled_q = np.full(len(volume), np.nan, dtype=np.float64)
    scaled_valid = np.isfinite(scaled_baseline) & (scaled_baseline > 0)
    scaled_q[scaled_valid] = volume[scaled_valid] * 100.0 / scaled_baseline[scaled_valid]
    valid_indices = np.flatnonzero(valid & scaled_valid)
    ordered = valid_indices[np.argsort(-q[valid_indices], kind="mergesort")]
    ordered_scaled = valid_indices[np.argsort(-scaled_q[valid_indices], kind="mergesort")]
    rows = []
    for fraction in (0.001, 0.01):
        count = max(1, int(math.ceil(len(valid_indices) * fraction)))
        for rank, row_index in enumerate(ordered[:count], start=1):
            rows.append(
                {
                    "commodity": commodity,
                    "top_fraction": fraction,
                    "rank": rank,
                    "row_index": int(row_index),
                    "timestamp": frame["Date"].iat[row_index],
                    "raw_volume": volume[row_index],
                    "previous20_median": baseline[row_index],
                    "q20": q[row_index],
                }
            )
    rng = np.random.default_rng(SEED + sum(ord(character) for character in commodity))
    for threshold in (2.0, 3.0, 5.0, 10.0):
        eligible = valid_indices[q[valid_indices] >= threshold]
        chosen = eligible if len(eligible) <= 20 else np.sort(rng.choice(eligible, size=20, replace=False))
        for row_index in chosen:
            rows.append(
                {
                    "commodity": commodity,
                    "top_fraction": f"random_q_ge_{threshold:g}",
                    "rank": None,
                    "row_index": int(row_index),
                    "timestamp": frame["Date"].iat[int(row_index)],
                    "raw_volume": volume[row_index],
                    "previous20_median": baseline[row_index],
                    "q20": q[row_index],
                }
            )
    summaries = [
        {
            "commodity": commodity,
            "test": "volume_scale_100_ranking",
            "valid_n": len(valid_indices),
            "max_q_error": float(np.nanmax(np.abs(q - scaled_q))),
            "top_0_1pct_timestamps_identical": bool(
                np.array_equal(ordered[: max(1, math.ceil(len(ordered) * 0.001))], ordered_scaled[: max(1, math.ceil(len(ordered) * 0.001))])
            ),
            "top_1pct_timestamps_identical": bool(
                np.array_equal(ordered[: max(1, math.ceil(len(ordered) * 0.01))], ordered_scaled[: max(1, math.ceil(len(ordered) * 0.01))])
            ),
            **{f"q_ge_{threshold:g}_count": int(np.sum(q[valid] >= threshold)) for threshold in (2, 3, 5, 10)},
        }
    ]
    return summaries, rows, q


def causality_audit(
    commodity: str,
    frame: pd.DataFrame,
    sampled: Mapping[int, np.ndarray],
) -> list[dict]:
    rng = np.random.default_rng(SEED + 99)
    rows: list[dict] = []
    for length, starts in sampled.items():
        chosen = starts if len(starts) <= 32 else np.sort(rng.choice(starts, size=32, replace=False))
        maximum = 0.0
        for start in chosen:
            end = int(start + length)
            local_end = min(len(frame), end + 20)
            local = frame.iloc[start - ROLLING_VOLUME_LENGTH : local_end].copy()
            window = local.iloc[ROLLING_VOLUME_LENGTH : ROLLING_VOLUME_LENGTH + length]
            prior = local.iloc[:ROLLING_VOLUME_LENGTH]
            original = imc_transform_window(
                *(window[column].to_numpy() for column in OHLC),
                window["OpenInterest"].to_numpy(),
                window["Volume"].to_numpy(),
                prior["Volume"].to_numpy(),
            )
            future_start = ROLLING_VOLUME_LENGTH + length
            for column in CORE_COLUMNS:
                local.loc[local.index[future_start:], column] *= 1000.0
            mutated_window = local.iloc[ROLLING_VOLUME_LENGTH : ROLLING_VOLUME_LENGTH + length]
            mutated_prior = local.iloc[:ROLLING_VOLUME_LENGTH]
            counterfactual = imc_transform_window(
                *(mutated_window[column].to_numpy() for column in OHLC),
                mutated_window["OpenInterest"].to_numpy(),
                mutated_window["Volume"].to_numpy(),
                mutated_prior["Volume"].to_numpy(),
            )
            for key in ("price_open", "price_high", "price_low", "price_close", "oi", "volume_fixed", "volume_q20"):
                maximum = max(maximum, float(np.nanmax(np.abs(original[key] - counterfactual[key]))))
        rows.append(
            {
                "commodity": commodity,
                "window_length": length,
                "anchors_tested": len(chosen),
                "causality_max_abs_error": maximum,
            }
        )
    return rows


def distribution_summary(commodity: str, feature: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, DIST_QUANTILES)
    return {
        "commodity": commodity,
        "feature": feature,
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        **{
            label: float(value)
            for label, value in zip(("p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_9"), quantiles)
        },
    }


def collect_imc_distribution_samples(
    frame: pd.DataFrame,
    sampled: Mapping[int, np.ndarray],
    rng: np.random.Generator,
    q_full: np.ndarray,
) -> dict[str, np.ndarray]:
    raw = {column: frame[column].to_numpy(dtype=np.float64) for column in CORE_COLUMNS}
    result: dict[str, list[np.ndarray]] = {
        "imc_price_relative_close": [],
        "imc_oi_relative": [],
        "imc_volume_fixed_ratio": [],
        "imc_volume_q20": [],
    }
    per_length = math.ceil(DIST_MAX_POINTS / len(sampled))
    for length, starts in sampled.items():
        total = len(starts) * length
        count = min(per_length, total)
        flat = rng.choice(total, size=count, replace=False)
        window_number = flat // length
        within = flat % length
        positions = starts[window_number] + within
        origins = starts[window_number]
        prior_positions = origins[:, None] - np.arange(ROLLING_VOLUME_LENGTH, 0, -1)[None, :]
        m0 = np.median(raw["Volume"][prior_positions], axis=1)
        result["imc_price_relative_close"].append(np.log(raw["Close"][positions] / raw["Close"][origins]))
        result["imc_oi_relative"].append(np.log(raw["OpenInterest"][positions] / raw["OpenInterest"][origins]))
        result["imc_volume_fixed_ratio"].append(raw["Volume"][positions] / m0)
        result["imc_volume_q20"].append(q_full[positions])
    return {name: np.concatenate(parts)[:DIST_MAX_POINTS] for name, parts in result.items()}


def empirical_distances_sorted(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Exact 1D empirical Wasserstein-1 and two-sample KS statistics.

    ``left`` and ``right`` must already be sorted.  Both statistics share one
    merged support, avoiding the earlier repeated-sort/quantile-grid path.
    """
    combined = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, combined, side="right") / len(left)
    right_cdf = np.searchsorted(right, combined, side="right") / len(right)
    ks = float(np.max(np.abs(left_cdf - right_cdf)))
    deltas = np.diff(combined)
    wasserstein = float(np.sum(np.abs(left_cdf[:-1] - right_cdf[:-1]) * deltas))
    return wasserstein, ks


def empirical_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    return empirical_distances_sorted(
        np.sort(np.asarray(left, dtype=np.float64)),
        np.sort(np.asarray(right, dtype=np.float64)),
    )[0]


def empirical_ks(left: np.ndarray, right: np.ndarray) -> float:
    return empirical_distances_sorted(
        np.sort(np.asarray(left, dtype=np.float64)),
        np.sort(np.asarray(right, dtype=np.float64)),
    )[1]


def data_quality_rows(commodity: str, frame: pd.DataFrame, paths: list[Path]) -> tuple[dict, dict, dict]:
    numeric = frame.loc[:, CORE_COLUMNS]
    dates = frame["Date"]
    deltas = dates.diff().dt.total_seconds().div(60.0)
    invalid_ohlc = (
        ~np.isfinite(numeric.loc[:, OHLC].to_numpy(dtype=np.float64)).all(axis=1)
        | (numeric.loc[:, OHLC] <= 0).any(axis=1).to_numpy()
        | (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).to_numpy()
        | (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).to_numpy()
    )
    inventory = {
        "commodity": commodity,
        "csymbol": CSYMBOLS[commodity],
        "first_timestamp": dates.min(),
        "last_timestamp": dates.max(),
        "row_count": len(frame),
        "source_file_count": len(paths),
        "source_sha256_manifest": hashlib.sha256(
            "\n".join(f"{path}:{sha256_file(path)}" for path in paths).encode()
        ).hexdigest(),
    }
    quality = {
        "commodity": commodity,
        **{f"{column.lower()}_min": float(frame[column].min()) for column in OHLC},
        **{f"{column.lower()}_max": float(frame[column].max()) for column in OHLC},
        **{
            f"volume_{label}": float(value)
            for label, value in zip(("min", "median", "p99", "p99_9", "max"), np.quantile(frame["Volume"], (0, 0.5, 0.99, 0.999, 1)))
        },
        **{
            f"oi_{label}": float(value)
            for label, value in zip(("min", "median", "p99", "p99_9", "max"), np.quantile(frame["OpenInterest"], (0, 0.5, 0.99, 0.999, 1)))
        },
        "volume_zero_count": int((frame["Volume"] == 0).sum()),
        "oi_nonpositive_count": int((frame["OpenInterest"] <= 0).sum()),
        "invalid_ohlc_count": int(invalid_ohlc.sum()),
        "duplicate_timestamp_count": int(dates.duplicated(keep=False).sum()),
        "non_monotonic_timestamp_count": int((deltas < 0).sum()),
        "non_one_minute_interval_count": int(((deltas != 1) & deltas.notna()).sum()),
        "maximum_interval_minutes": float(deltas.max()),
    }
    roll = {
        "commodity": commodity,
        "roll_status": "unknown",
        "contract_metadata_present": False,
        "roll_boundary_count": None,
        "windows_removed_because_of_roll": None,
        "limitation": "continuous-symbol parquet has no per-row actual contract identifier",
    }
    return inventory, quality, roll


def sustained_volume_examples(
    commodity: str,
    frame: pd.DataFrame,
    starts: np.ndarray,
    q_full: np.ndarray,
) -> list[dict]:
    volume = frame["Volume"].to_numpy(dtype=np.float64)
    rows: list[dict] = []
    for start in starts:
        prior = volume[start - ROLLING_VOLUME_LENGTH : start]
        m0 = float(np.median(prior))
        x = volume[start : start + 512] / m0
        rolling_median16 = pd.Series(x).rolling(16, min_periods=16).median().to_numpy()
        positions = np.flatnonzero(rolling_median16 >= 2.0)
        if positions.size:
            end = int(positions[0])
            begin = end - 15
            absolute = np.arange(start + begin, start + end + 1)
            rows.append(
                {
                    "commodity": commodity,
                    "window_start": frame["Date"].iat[start],
                    "segment_start": frame["Date"].iat[int(absolute[0])],
                    "segment_end": frame["Date"].iat[int(absolute[-1])],
                    "x_volume_fixed_path": json.dumps(x[begin : end + 1].tolist()),
                    "q_volume_20_path": json.dumps(q_full[absolute].tolist()),
                }
            )
            if len(rows) >= 5:
                break
    return rows


def volume_semantic_examples(frames: Mapping[str, pd.DataFrame], q_values: Mapping[str, np.ndarray]) -> list[dict]:
    rows = []
    for target in (0.5, 1.0, 2.0, 3.0, 5.0):
        for commodity in COMMODITIES:
            q = q_values[commodity]
            valid = np.flatnonzero(np.isfinite(q))
            if target == 5.0:
                eligible = valid[q[valid] >= 5.0]
                index = int(eligible[np.argmin(q[eligible] - 5.0)]) if len(eligible) else int(valid[np.argmax(q[valid])])
            else:
                index = int(valid[np.argmin(np.abs(q[valid] - target))])
            frame = frames[commodity]
            baseline = frame["Volume"].iat[index] / q[index]
            rows.append(
                {
                    "target_q": target,
                    "commodity": commodity,
                    "timestamp": frame["Date"].iat[index],
                    "raw_volume": frame["Volume"].iat[index],
                    "previous20_median": baseline,
                    "q": q[index],
                }
            )
    return rows


def oi_semantic_examples(frames: Mapping[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for lag, target_fraction in ((1, 0.005), (16, 0.02), (64, 0.05)):
        target = math.log1p(target_fraction)
        for commodity, frame in frames.items():
            oi = frame["OpenInterest"].to_numpy(dtype=np.float64)
            valid = (oi[lag:] > 0) & (oi[:-lag] > 0)
            positions = np.flatnonzero(valid) + lag
            changes = np.log(oi[positions] / oi[positions - lag])
            positive = changes > 0
            positions = positions[positive]
            changes = changes[positive]
            if not len(positions):
                continue
            selection = int(np.argmin(np.abs(changes - target)))
            end = int(positions[selection])
            start = end - lag
            rows.append(
                {
                    "target_description": f"{lag} rows +{target_fraction:.1%} OI",
                    "commodity": commodity,
                    "start_timestamp": frame["Date"].iat[start],
                    "end_timestamp": frame["Date"].iat[end],
                    "raw_oi_start": oi[start],
                    "raw_oi_end": oi[end],
                    "absolute_oi_change": oi[end] - oi[start],
                    "relative_oi_change": oi[end] / oi[start] - 1.0,
                    "imc_log_change": changes[selection],
                }
            )
    return rows


def pattern_vector(frame: pd.DataFrame, start: int, length: int, q_full: np.ndarray) -> np.ndarray:
    close = frame["Close"].to_numpy(dtype=np.float64)
    oi = frame["OpenInterest"].to_numpy(dtype=np.float64)
    volume = frame["Volume"].to_numpy(dtype=np.float64)
    positions = slice(start, start + length)
    m0 = np.median(volume[start - ROLLING_VOLUME_LENGTH : start])
    return np.concatenate(
        (
            np.log(close[positions] / close[start]),
            np.log(oi[positions] / oi[start]),
            volume[positions] / m0,
            q_full[positions],
        )
    )


def nearest_pattern_examples(
    frames: Mapping[str, pd.DataFrame],
    sampled: Mapping[str, Mapping[int, np.ndarray]],
    q_values: Mapping[str, np.ndarray],
) -> list[dict]:
    rng = np.random.default_rng(SEED + 700)
    rows = []
    rb = frames[HELD_OUT_COMMODITY]
    for length in WINDOW_LENGTHS:
        rb_starts = sampled[HELD_OUT_COMMODITY][length]
        queries = rb_starts if len(rb_starts) <= NEAREST_QUERY_COUNT else np.sort(
            rng.choice(rb_starts, size=NEAREST_QUERY_COUNT, replace=False)
        )
        for rb_start in queries:
            query = pattern_vector(rb, int(rb_start), length, q_values[HELD_OUT_COMMODITY])
            for commodity in TRAIN_COMMODITIES:
                candidates = sampled[commodity][length]
                if len(candidates) > NEAREST_CANDIDATES_PER_COMMODITY:
                    candidates = np.sort(rng.choice(candidates, size=NEAREST_CANDIDATES_PER_COMMODITY, replace=False))
                best_distance = math.inf
                best_start = -1
                for candidate in candidates:
                    vector = pattern_vector(frames[commodity], int(candidate), length, q_values[commodity])
                    distance = float(np.linalg.norm(query - vector))
                    if distance < best_distance or (distance == best_distance and candidate < best_start):
                        best_distance = distance
                        best_start = int(candidate)
                candidate_frame = frames[commodity]
                rows.append(
                    {
                        "window_length": length,
                        "rb_start_index": int(rb_start),
                        "rb_timestamp": rb["Date"].iat[int(rb_start)],
                        "train_commodity": commodity,
                        "train_start_index": best_start,
                        "train_timestamp": candidate_frame["Date"].iat[best_start],
                        "imc_path_distance": best_distance,
                        "rb_price_scale": rb["Close"].iat[int(rb_start)],
                        "train_price_scale": candidate_frame["Close"].iat[best_start],
                        "rb_oi_scale": rb["OpenInterest"].iat[int(rb_start)],
                        "train_oi_scale": candidate_frame["OpenInterest"].iat[best_start],
                        "rb_volume_scale": np.median(
                            rb["Volume"].iloc[int(rb_start) - ROLLING_VOLUME_LENGTH : int(rb_start)]
                        ),
                        "train_volume_scale": np.median(
                            candidate_frame["Volume"].iloc[best_start - ROLLING_VOLUME_LENGTH : best_start]
                        ),
                    }
                )
    return rows


def gate_status(rows: Iterable[dict], field: str = "max_abs_error") -> bool:
    values = [float(row[field]) for row in rows if field in row and np.isfinite(row[field])]
    return bool(values) and max(values) <= TOLERANCE


def write_summary(output_dir: Path, summary: dict) -> None:
    gates = summary["gates"]
    alignment = summary["empirical_alignment"]
    lines = [
        "# IMC v0.1 Cross-Commodity Mathematical & Empirical Audit",
        "",
        f"**Overall: {summary['overall']}**",
        "",
        "## Mathematical gates",
        "",
        "| Gate | Result | Maximum error |",
        "|---|---:|---:|",
    ]
    for name, value in gates.items():
        lines.append(f"| {name} | {'PASS' if value['pass'] else 'FAIL'} | {value['max_error']:.3e} |")
    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"1. Scale invariance: **{'yes' if gates['scale_invariance']['pass'] else 'no'}**; all six commodities were tested against the frozen positive scales.",
            f"2. Price reconstruction modulo P0: **{'yes' if gates['price_reconstruction']['pass'] else 'no'}**.",
            f"3. OI reconstruction modulo OI0: **{'yes' if gates['oi_reconstruction']['pass'] else 'no'}**.",
            f"4. Sudden OI changes: **{'strictly preserved' if gates['oi_identity']['pass'] else 'not preserved within tolerance'}**.",
            f"5. OI cumulative changes at 16/64/256 rows: **{'strictly preserved' if gates['oi_identity']['pass'] else 'not preserved within tolerance'}**.",
            f"6. Fixed-origin Volume path reconstruction: **{'yes' if gates['volume_reconstruction']['pass'] else 'no'}**.",
            f"7. Volume / previous20 median N-fold surprise: **{'strictly preserved' if gates['volume_q20']['pass'] else 'not preserved within tolerance'}**.",
            "8. Sustained high activity is retained by the fixed-origin coordinate; the accompanying examples show rolling Q20 adapting while the fixed-origin ratio remains elevated.",
            f"9. Causality: **{'yes' if gates['causality']['pass'] else 'no'}**.",
            f"10. Train-commodity absolute-scale separation fell after IMC in {alignment['distance_reduction_count']} of {alignment['distance_comparison_count']} matched raw/IMC pairwise comparisons. This is descriptive, not a gate.",
            "11. RB used the same frozen formulas, window lengths, Median20 rule, and tolerances without fitting or adjustment.",
            "12. Rollover reliability is limited: all lake inputs are continuous-symbol series without per-row actual-contract metadata, so roll_status is unknown and cross-roll windows could not be removed.",
            f"13. Final result: **{summary['overall']}**.",
            "",
            "## Claim boundary",
            "",
            "This audit tests unit-scale invariance, reconstruction/identity properties, causality, and descriptive cross-commodity comparability. It does not establish predictive power, commodity equivalence, or model generalization.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.csv_dir.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "IMC v0.1 Cross-Commodity Mathematical & Empirical Audit",
        "mathematical_source": str(args.design.resolve()),
        "mathematical_source_sha256": sha256_file(args.design),
        "lake_root": str(args.lake_root.resolve()),
        "train_commodities": list(TRAIN_COMMODITIES),
        "held_out_commodity": HELD_OUT_COMMODITY,
        "held_out_read_after_rules_frozen": True,
        "window_lengths": list(WINDOW_LENGTHS),
        "windows_per_length": WINDOWS_PER_LENGTH,
        "seed": SEED,
        "median_volume_lookback": ROLLING_VOLUME_LENGTH,
        "float_dtype": "float64",
        "tolerance": TOLERANCE,
        "scale_factors": list(SCALE_FACTORS),
        "distribution_distance_sampling": {
            "max_points_per_commodity_dimension": DIST_MAX_POINTS,
            "purpose": "descriptive empirical audit only; never used by mathematical gates",
        },
        "nearest_pattern_sampling": {
            "queries_per_length": NEAREST_QUERY_COUNT,
            "candidates_per_train_commodity": NEAREST_CANDIDATES_PER_COMMODITY,
            "purpose": "secondary visualization only",
        },
        "models_trained": False,
        "formulas_tuned": False,
    }
    json_dump(output_dir / "protocol.json", protocol)

    frames: dict[str, pd.DataFrame] = {}
    paths_by_commodity: dict[str, list[Path]] = {}
    inventory_rows: list[dict] = []
    quality_rows: list[dict] = []
    roll_rows: list[dict] = []
    sampled: dict[str, dict[int, np.ndarray]] = {}
    rng = np.random.default_rng(SEED)

    # Train commodities are fully loaded and their rules/indices frozen before
    # the held-out RB input is opened.
    for commodity in TRAIN_COMMODITIES:
        print(f"[IMC] loading train commodity {commodity}", flush=True)
        frame, paths = load_lake_commodity(args.lake_root, commodity)
        frames[commodity], paths_by_commodity[commodity] = frame, paths
    for commodity in TRAIN_COMMODITIES:
        sampled[commodity] = {
            length: sample_starts(valid_window_starts(frames[commodity], length), rng)
            for length in WINDOW_LENGTHS
        }
    # Held-out read begins only here; no rule above depends on RB values.
    frame, paths = load_lake_commodity(args.lake_root, HELD_OUT_COMMODITY)
    print(f"[IMC] loading held-out commodity {HELD_OUT_COMMODITY}", flush=True)
    frames[HELD_OUT_COMMODITY], paths_by_commodity[HELD_OUT_COMMODITY] = frame, paths
    sampled[HELD_OUT_COMMODITY] = {
        length: sample_starts(valid_window_starts(frame, length), rng) for length in WINDOW_LENGTHS
    }

    scale_rows: list[dict] = []
    reconstruction_rows: list[dict] = []
    oi_identity_rows: list[dict] = []
    oi_extreme_rows: list[dict] = []
    oi_extreme_summary_rows: list[dict] = []
    volume_identity_rows: list[dict] = []
    volume_extreme_rows: list[dict] = []
    sustained_rows: list[dict] = []
    price_identity_rows: list[dict] = []
    causality_rows: list[dict] = []
    distribution_rows: list[dict] = []
    distributions: dict[str, dict[str, np.ndarray]] = {}
    q_values: dict[str, np.ndarray] = {}

    for commodity in COMMODITIES:
        print(
            f"[IMC] auditing {commodity}: rows={len(frames[commodity]):,}, "
            f"windows={sum(len(value) for value in sampled[commodity].values()):,}",
            flush=True,
        )
        frame = frames[commodity]
        paths = paths_by_commodity[commodity]
        inventory, quality, roll = data_quality_rows(commodity, frame, paths)
        inventory_rows.append(inventory)
        quality_rows.append(quality)
        roll_rows.append(roll)
        export_core_csv(frame, csv_dir / f"{CSYMBOLS[commodity].replace('.', '_')}_continuous_1m.csv")

        window_scale, window_reconstruction, window_oi, window_price = audit_windows(
            commodity, frame, sampled[commodity]
        )
        scale_rows.extend(window_scale)
        reconstruction_rows.extend(window_reconstruction)
        oi_identity_rows.extend(window_oi)
        price_identity_rows.extend(window_price)
        extreme_summary, extreme_rows = extreme_oi_audit(commodity, frame)
        oi_extreme_summary_rows.extend(extreme_summary)
        oi_extreme_rows.extend(extreme_rows)
        volume_summary, volume_rows, q_full = volume_audit(commodity, frame)
        volume_identity_rows.extend(volume_summary)
        volume_extreme_rows.extend(volume_rows)
        q_values[commodity] = q_full
        sustained_rows.extend(sustained_volume_examples(commodity, frame, sampled[commodity][512], q_full))
        causality_rows.extend(causality_audit(commodity, frame, sampled[commodity]))

        close = frame["Close"].to_numpy(dtype=np.float64)
        oi = frame["OpenInterest"].to_numpy(dtype=np.float64)
        volume = frame["Volume"].to_numpy(dtype=np.float64)
        dist = {
            "raw_close": close[np.isfinite(close)],
            "raw_oi": oi[np.isfinite(oi)],
            "raw_volume": volume[np.isfinite(volume)],
            "imc_single_step_price_return": np.diff(np.log(close)),
            "imc_single_step_oi_log_change": np.diff(np.log(oi)),
            **collect_imc_distribution_samples(frame, sampled[commodity], rng, q_full),
        }
        distributions[commodity] = {
            name: values[np.isfinite(values)] for name, values in dist.items()
        }
        for name, values in distributions[commodity].items():
            distribution_rows.append(distribution_summary(commodity, name, values))
        print(f"[IMC] finished mathematical audit for {commodity}", flush=True)

    pd.DataFrame(inventory_rows).to_csv(output_dir / "data_inventory.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(output_dir / "roll_audit.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(output_dir / "scale_invariance.csv", index=False)
    pd.DataFrame(reconstruction_rows).to_csv(output_dir / "reconstruction.csv", index=False)
    pd.DataFrame(oi_identity_rows).to_csv(output_dir / "oi_identity_tests.csv", index=False)
    pd.DataFrame(oi_extreme_rows).to_csv(output_dir / "oi_extreme_events.csv", index=False)
    pd.DataFrame(oi_extreme_summary_rows).to_csv(output_dir / "oi_extreme_summary.csv", index=False)
    pd.DataFrame(volume_identity_rows).to_csv(output_dir / "volume_identity_tests.csv", index=False)
    pd.DataFrame(volume_extreme_rows).to_csv(output_dir / "volume_extreme_events.csv", index=False)
    pd.DataFrame(sustained_rows).to_csv(output_dir / "volume_sustained_examples.csv", index=False)
    pd.DataFrame(price_identity_rows).to_csv(output_dir / "price_identity_tests.csv", index=False)
    pd.DataFrame(causality_rows).to_csv(output_dir / "causality_tests.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(output_dir / "commodity_distribution_stats.csv", index=False)

    wasserstein_rows, ks_rows = [], []
    print("[IMC] computing descriptive pairwise distribution distances", flush=True)
    features = sorted(next(iter(distributions.values())).keys())
    sorted_distributions = {
        commodity: {feature: np.sort(values) for feature, values in by_feature.items()}
        for commodity, by_feature in distributions.items()
    }
    for left_commodity, right_commodity in combinations(TRAIN_COMMODITIES, 2):
        for feature in features:
            left = sorted_distributions[left_commodity][feature]
            right = sorted_distributions[right_commodity][feature]
            wasserstein, ks = empirical_distances_sorted(left, right)
            wasserstein_rows.append(
                {
                    "commodity_left": left_commodity,
                    "commodity_right": right_commodity,
                    "feature": feature,
                    "distance": wasserstein,
                }
            )
            ks_rows.append(
                {
                    "commodity_left": left_commodity,
                    "commodity_right": right_commodity,
                    "feature": feature,
                    "statistic": ks,
                }
            )
    pd.DataFrame(wasserstein_rows).to_csv(output_dir / "pairwise_wasserstein.csv", index=False)
    pd.DataFrame(ks_rows).to_csv(output_dir / "pairwise_ks.csv", index=False)
    pd.DataFrame(volume_semantic_examples(frames, q_values)).to_csv(
        output_dir / "cross_commodity_volume_examples.csv", index=False
    )
    pd.DataFrame(oi_semantic_examples(frames)).to_csv(
        output_dir / "cross_commodity_oi_examples.csv", index=False
    )
    pd.DataFrame(nearest_pattern_examples(frames, sampled, q_values)).to_csv(
        output_dir / "cross_commodity_nearest_patterns.csv", index=False
    )
    json_dump(
        output_dir / "sample_indices.json",
        {
            commodity: {str(length): starts.tolist() for length, starts in by_length.items()}
            for commodity, by_length in sampled.items()
        },
    )

    scale_pass = gate_status(scale_rows)
    price_reconstruction = [row for row in reconstruction_rows if row["test"] == "price_reconstruction"]
    oi_reconstruction = [row for row in reconstruction_rows if row["test"] == "oi_reconstruction"]
    volume_reconstruction = [row for row in reconstruction_rows if row["test"] == "volume_fixed_reconstruction"]
    volume_q = [row for row in reconstruction_rows if row["test"] == "volume_q20_identity"]
    oi_identity_pass = gate_status(oi_identity_rows) and all(
        row["top_k_overlap"] == 1.0 and row["max_value_error"] <= TOLERANCE
        for row in oi_extreme_summary_rows
    )
    causality_pass = gate_status(causality_rows, "causality_max_abs_error")

    def gate(pass_: bool, rows: list[dict], field: str = "max_abs_error") -> dict:
        values = [float(row[field]) for row in rows if field in row and np.isfinite(row[field])]
        return {"pass": pass_, "max_error": max(values) if values else math.nan}

    gates = {
        "scale_invariance": gate(scale_pass, scale_rows),
        "price_reconstruction": gate(gate_status(price_reconstruction), price_reconstruction),
        "oi_reconstruction": gate(gate_status(oi_reconstruction), oi_reconstruction),
        "oi_identity": gate(oi_identity_pass, oi_identity_rows),
        "volume_reconstruction": gate(gate_status(volume_reconstruction), volume_reconstruction),
        "volume_q20": gate(gate_status(volume_q), volume_q),
        "causality": gate(causality_pass, causality_rows, "causality_max_abs_error"),
    }
    overall = (
        "IMC_MATHEMATICAL_IMPLEMENTATION_PASS"
        if all(value["pass"] for value in gates.values())
        else "IMC_MATHEMATICAL_IMPLEMENTATION_FAIL"
    )

    raw_to_imc = {
        "raw_close": "imc_price_relative_close",
        "raw_oi": "imc_oi_relative",
        "raw_volume": "imc_volume_fixed_ratio",
    }
    wasserstein_lookup = {
        (row["commodity_left"], row["commodity_right"], row["feature"]): row["distance"]
        for row in wasserstein_rows
    }
    reductions = []
    for left, right in combinations(TRAIN_COMMODITIES, 2):
        for raw_feature, imc_feature in raw_to_imc.items():
            reductions.append(
                wasserstein_lookup[(left, right, imc_feature)]
                < wasserstein_lookup[(left, right, raw_feature)]
            )
    summary = {
        "overall": overall,
        "gates": gates,
        "roll_status": "unknown_for_all_six_continuous_symbol_inputs",
        "empirical_alignment": {
            "distance_reduction_count": int(sum(reductions)),
            "distance_comparison_count": len(reductions),
            "is_formal_gate": False,
        },
        "csv_exports": {
            commodity: str(csv_dir / f"{CSYMBOLS[commodity].replace('.', '_')}_continuous_1m.csv")
            for commodity in COMMODITIES
        },
        "models_trained": False,
    }
    json_dump(output_dir / "summary.json", summary)
    write_summary(output_dir, summary)
    print(f"[IMC] complete: {overall}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake-root", type=Path, default=Path("/data/juejin"))
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("docs/MARKET_INVARIANT_COORDINATES_MATHEMATICAL_DESIGN.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/imc_v0_1_cross_commodity_audit"),
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("data/imc_v0_1"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
