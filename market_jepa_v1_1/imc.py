from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import HELD_OUT_COMMODITY, IMC_FEATURES, TRAIN_COMMODITIES


def _positive_median(values: np.ndarray) -> tuple[float, bool]:
    values = np.asarray(values, dtype=np.float64)
    if values.size != 20 or not np.isfinite(values).all():
        return 0.0, False
    value = float(np.median(values))
    return (value, True) if value > 0 else (0.0, False)


@dataclass(frozen=True)
class IMCOrigin:
    price: float
    open_interest: float
    volume_baseline: float
    price_valid: bool
    oi_valid: bool
    volume_valid: bool


class V11IMCTransform:
    """Frozen nine-coordinate IMC transform with explicit validity.

    Invalid coordinates are returned as numeric zero. No epsilon or future
    observation is used to repair a missing/zero baseline.
    """

    feature_order = IMC_FEATURES

    @staticmethod
    def volume_baseline(prior_volume: np.ndarray) -> tuple[float, bool]:
        return _positive_median(prior_volume)

    @staticmethod
    def origin(close: np.ndarray, oi: np.ndarray, prior_volume: np.ndarray) -> IMCOrigin:
        close = np.asarray(close, dtype=np.float64)
        oi = np.asarray(oi, dtype=np.float64)
        if close.size == 0 or oi.size != close.size:
            raise ValueError("origin requires a nonempty aligned close/OI sequence")
        baseline, volume_valid = _positive_median(prior_volume)
        price = float(close[0])
        open_interest = float(oi[0])
        return IMCOrigin(
            price=price if np.isfinite(price) and price > 0 else 0.0,
            open_interest=open_interest if np.isfinite(open_interest) and open_interest > 0 else 0.0,
            volume_baseline=baseline,
            price_valid=bool(np.isfinite(price) and price > 0),
            oi_valid=bool(np.isfinite(open_interest) and open_interest > 0),
            volume_valid=volume_valid,
        )

    @classmethod
    def transform(
        cls,
        bars: np.ndarray,
        *,
        origin: IMCOrigin,
        prior_volume: np.ndarray,
        previous_close: float | None = None,
        previous_oi: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform `[N,6]` OHLC, Volume, OI bars using a supplied origin."""

        bars = np.asarray(bars, dtype=np.float64)
        if bars.ndim != 2 or bars.shape[1] != 6:
            raise ValueError("bars must be [N,6] ordered OHLC,Volume,OI")
        n = len(bars)
        result = np.zeros((n, len(IMC_FEATURES)), dtype=np.float32)
        valid = np.zeros_like(result, dtype=np.bool_)
        if n == 0:
            return result, valid
        open_, high, low, close, volume, oi = bars.T

        if origin.price_valid:
            for column, values in enumerate((open_, high, low, close)):
                ok = np.isfinite(values) & (values > 0)
                result[ok, column] = np.log(values[ok] / origin.price).astype(np.float32)
                valid[ok, column] = True

        prior_close = np.concatenate((
            np.asarray([np.nan if previous_close is None else previous_close]), close[:-1]
        ))
        ok = np.isfinite(close) & (close > 0) & np.isfinite(prior_close) & (prior_close > 0)
        result[ok, 4] = np.log(close[ok] / prior_close[ok]).astype(np.float32)
        valid[ok, 4] = True

        if origin.oi_valid:
            ok = np.isfinite(oi) & (oi > 0)
            result[ok, 5] = np.log(oi[ok] / origin.open_interest).astype(np.float32)
            valid[ok, 5] = True

        prior_oi_values = np.concatenate((
            np.asarray([np.nan if previous_oi is None else previous_oi]), oi[:-1]
        ))
        ok = np.isfinite(oi) & (oi > 0) & np.isfinite(prior_oi_values) & (prior_oi_values > 0)
        result[ok, 6] = np.log(oi[ok] / prior_oi_values[ok]).astype(np.float32)
        valid[ok, 6] = True

        if origin.volume_valid:
            ok = np.isfinite(volume) & (volume >= 0)
            result[ok, 7] = (volume[ok] / origin.volume_baseline).astype(np.float32)
            valid[ok, 7] = True

        prior_volume = np.asarray(prior_volume, dtype=np.float64)
        volume_path = np.concatenate((prior_volume, volume))
        offset = len(prior_volume)
        for index, value in enumerate(volume):
            start = offset + index - 20
            if start < 0 or not np.isfinite(value) or value < 0:
                continue
            baseline, baseline_valid = _positive_median(volume_path[start : offset + index])
            if baseline_valid:
                result[index, 8] = np.float32(value / baseline)
                valid[index, 8] = True

        if not np.isfinite(result).all():
            raise FloatingPointError("IMC transform produced NaN/Inf")
        return result, valid

    @classmethod
    def window(
        cls,
        bars: np.ndarray,
        *,
        prior_volume: np.ndarray,
        previous_close: float | None = None,
        previous_oi: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, IMCOrigin]:
        bars = np.asarray(bars, dtype=np.float64)
        origin = cls.origin(bars[:, 3], bars[:, 5], np.asarray(prior_volume, dtype=np.float64))
        values, validity = cls.transform(
            bars, origin=origin, prior_volume=prior_volume,
            previous_close=previous_close, previous_oi=previous_oi,
        )
        return values, validity, origin


@dataclass
class SharedIMCScaler:
    mean: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    fitted_commodities: tuple[str, ...]
    source_counts: dict[str, int]
    feature_order: tuple[str, ...] = IMC_FEATURES
    checksum: str = ""

    @classmethod
    def fit(
        cls,
        population: Iterable[tuple[str, str, np.ndarray, np.ndarray]],
        *,
        expected_commodities: Iterable[str] = TRAIN_COMMODITIES,
        held_out_commodity: str = HELD_OUT_COMMODITY,
    ) -> "SharedIMCScaler":
        expected = tuple(str(value) for value in expected_commodities)
        if (
            not expected or any(not value for value in expected)
            or len(set(expected)) != len(expected) or held_out_commodity in expected
        ):
            raise ValueError("invalid shared scaler Train/held-out commodity configuration")
        count = np.zeros(len(IMC_FEATURES), dtype=np.int64)
        total = np.zeros(len(IMC_FEATURES), dtype=np.float64)
        total_sq = np.zeros(len(IMC_FEATURES), dtype=np.float64)
        commodities: set[str] = set()
        source_counts: dict[str, int] = {}
        for commodity, source, values, validity in population:
            if commodity == held_out_commodity or commodity not in expected:
                raise ValueError(f"scaler fit population contains forbidden commodity {commodity}")
            values = np.asarray(values, dtype=np.float64)
            validity = np.asarray(validity, dtype=np.bool_)
            if values.shape != validity.shape or values.ndim < 2 or values.shape[-1] != len(IMC_FEATURES):
                raise ValueError("scaler values/validity must align on the IMC feature axis")
            flat, mask = values.reshape(-1, len(IMC_FEATURES)), validity.reshape(-1, len(IMC_FEATURES))
            safe = np.where(mask, flat, 0.0)
            if not np.isfinite(safe).all():
                raise ValueError("valid scaler population contains NaN/Inf")
            count += mask.sum(axis=0)
            total += safe.sum(axis=0)
            total_sq += (safe * safe).sum(axis=0)
            commodities.add(commodity)
            source_counts[source] = source_counts.get(source, 0) + int(mask.sum())
        if commodities != set(expected):
            missing = set(expected) - commodities
            raise ValueError(f"shared scaler fit must include every configured Train commodity; missing={sorted(missing)}")
        if np.any(count == 0):
            raise ValueError("shared scaler population is empty for a commodity or feature")
        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 0.0)
        std = np.sqrt(variance)
        std[std == 0] = 1.0
        scaler = cls(mean, std, count, tuple(sorted(commodities)), dict(sorted(source_counts.items())))
        scaler.checksum = scaler.compute_checksum()
        return scaler

    def transform(self, values: np.ndarray, validity: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        validity = np.asarray(validity, dtype=np.bool_)
        if values.shape != validity.shape or values.shape[-1] != len(self.feature_order):
            raise ValueError("shared scaler input shape mismatch")
        scaled = (values.astype(np.float64) - self.mean) / self.std
        scaled[~validity] = 0.0
        if not np.isfinite(scaled).all():
            raise FloatingPointError("shared scaler produced NaN/Inf")
        return scaled.astype(np.float32)

    def _payload(self) -> dict:
        return {
            "feature_order": list(self.feature_order),
            "mean": self.mean.tolist(), "std": self.std.tolist(), "counts": self.counts.tolist(),
            "fitted_commodities": list(self.fitted_commodities),
            "source_counts": self.source_counts,
        }

    def compute_checksum(self) -> str:
        raw = json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        return {**self._payload(), "checksum": self.checksum or self.compute_checksum()}

    @classmethod
    def from_dict(cls, value: dict) -> "SharedIMCScaler":
        scaler = cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            std=np.asarray(value["std"], dtype=np.float64),
            counts=np.asarray(value["counts"], dtype=np.int64),
            fitted_commodities=tuple(value["fitted_commodities"]),
            source_counts={str(k): int(v) for k, v in value["source_counts"].items()},
            feature_order=tuple(value["feature_order"]), checksum=str(value["checksum"]),
        )
        if scaler.feature_order != IMC_FEATURES or scaler.compute_checksum() != scaler.checksum:
            raise ValueError("shared scaler feature order/checksum mismatch")
        if not scaler.fitted_commodities or len(set(scaler.fitted_commodities)) != len(scaler.fitted_commodities):
            raise ValueError("shared scaler fitted commodity population is invalid")
        return scaler
