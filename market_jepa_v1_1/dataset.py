from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    DAILY_CONTEXT_FEATURES, IMC_FEATURES, MINUTE_CONTEXT_FEATURES,
    TRAIN_COMMODITIES, WEEKLY_CONTEXT_FEATURES,
)
from .imc import IMCOrigin, SharedIMCScaler, V11IMCTransform


BAR_COLUMNS = ("open", "high", "low", "close", "volume", "open_interest")


@dataclass(frozen=True)
class ContractEpisode:
    commodity: str
    contract_uid: str
    episode_id: int
    main_start: pd.Timestamp
    main_end: pd.Timestamp
    anchor_end: pd.Timestamp
    role: str

    @property
    def key(self) -> tuple[str, str, int]:
        return self.commodity, self.contract_uid, self.episode_id


@dataclass
class EpisodeArrays:
    episode: ContractEpisode
    minute_frame: pd.DataFrame
    minute_bars: np.ndarray
    anchors: np.ndarray
    daily_raw: pd.DataFrame
    daily_frame: pd.DataFrame
    daily_origin: IMCOrigin
    daily_prior_volume: np.ndarray
    daily_previous_close: float | None
    daily_previous_oi: float | None
    weekly_raw: pd.DataFrame
    current_weekly_frame: pd.DataFrame
    weekly_origin: IMCOrigin
    weekly_prior_volume: np.ndarray
    weekly_previous_close: float | None
    weekly_previous_oi: float | None


def _aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["period_end", *BAR_COLUMNS])
    values = daily.copy()
    iso = values["trading_date"].dt.isocalendar()
    values["iso_year"] = iso.year.to_numpy()
    values["iso_week"] = iso.week.to_numpy()
    grouped = values.groupby(["iso_year", "iso_week"], sort=True, observed=True)
    result = grouped.agg(
        period_end=("trading_date", "max"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        open_interest=("open_interest", "last"), trading_day_count=("trading_date", "nunique"),
    ).reset_index(drop=True)
    return result


def _previous_row(frame: pd.DataFrame, time_column: str, timestamp: pd.Timestamp) -> pd.Series | None:
    previous = frame.loc[frame[time_column] < timestamp]
    return None if previous.empty else previous.iloc[-1]


def _transform_lifecycle(
    lifecycle: pd.DataFrame,
    raw: pd.DataFrame,
    time_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    if lifecycle.empty:
        empty = np.empty((0, len(IMC_FEATURES)), dtype=np.float32)
        return empty, empty.astype(np.bool_)
    start = pd.Timestamp(lifecycle.iloc[0][time_column])
    previous = raw.loc[raw[time_column] < start].tail(20)
    bars = lifecycle.loc[:, BAR_COLUMNS].to_numpy(dtype=np.float64)
    prior_volume = previous["volume"].to_numpy(dtype=np.float64)
    immediate = previous.iloc[-1] if not previous.empty else None
    return V11IMCTransform.window(
        bars, prior_volume=prior_volume,
        previous_close=None if immediate is None else float(immediate["close"]),
        previous_oi=None if immediate is None else float(immediate["open_interest"]),
    )[:2]


def _partial_bar(minute: pd.DataFrame, time_column: str) -> pd.DataFrame:
    if minute.empty:
        return pd.DataFrame(columns=[time_column, *BAR_COLUMNS])
    return pd.DataFrame({
        time_column: [minute["trading_date"].max()],
        "open": [minute.iloc[0].open], "high": [minute.high.max()],
        "low": [minute.low.min()], "close": [minute.iloc[-1].close],
        "volume": [minute.volume.sum()], "open_interest": [minute.iloc[-1].open_interest],
    })


def _read_csv(path: Path, scale: str, contracts: set[str] | None = None) -> pd.DataFrame:
    time_column = {"1m": "datetime", "1d": "trading_date", "1w": "week_end_date"}[scale]
    pieces = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        if contracts is not None:
            chunk = chunk.loc[chunk["contract_uid"].isin(contracts)]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=["commodity", "contract_uid", time_column, *BAR_COLUMNS])
    frame = pd.concat(pieces, ignore_index=True)
    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    if scale == "1m":
        frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="raise")
    for column in BAR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["contract_uid", time_column], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["contract_uid", time_column]).any():
        raise ValueError(f"duplicate real-contract {scale} rows in {path}")
    return frame


class V11DataStore:
    def __init__(self, episodes: pd.DataFrame, frames: dict[str, dict[str, pd.DataFrame]], source_root: Path | None = None) -> None:
        required = {
            "commodity", "contract_uid", "episode_id", "main_start_date",
            "main_end_date", "anchor_end_date", "role",
        }
        if required - set(episodes):
            raise ValueError(f"episode table missing columns: {sorted(required - set(episodes))}")
        table = episodes.copy()
        for column in ("main_start_date", "main_end_date", "anchor_end_date"):
            table[column] = pd.to_datetime(table[column], errors="raise")
        if table.duplicated(["commodity", "contract_uid", "episode_id"]).any():
            raise ValueError("duplicate contract episode key")
        if ((table.main_start_date > table.main_end_date) | (table.main_end_date > table.anchor_end_date)).any():
            raise ValueError("invalid main/anchor episode interval")
        self.episode_table = table
        self.frames = frames
        self.source_root = source_root
        self.episodes = {
            (str(row.commodity), str(row.contract_uid), int(row.episode_id)): ContractEpisode(
                str(row.commodity), str(row.contract_uid), int(row.episode_id),
                pd.Timestamp(row.main_start_date), pd.Timestamp(row.main_end_date),
                pd.Timestamp(row.anchor_end_date), str(row.role),
            )
            for row in table.itertuples(index=False)
        }
        for commodity, scales in frames.items():
            for scale, frame in scales.items():
                if not frame.empty and set(frame["commodity"].astype(str)) != {commodity}:
                    raise ValueError(f"{commodity}/{scale} contains another commodity")
                unknown = set(frame["contract_uid"].astype(str)) - set(
                    table.loc[table.commodity == commodity, "contract_uid"].astype(str)
                )
                if unknown:
                    raise ValueError(f"{commodity}/{scale} contains unknown contracts: {sorted(unknown)[:3]}")

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        commodities: Iterable[str] = TRAIN_COMMODITIES,
        *,
        max_contracts_per_commodity: int | None = None,
    ) -> "V11DataStore":
        root = Path(root)
        episodes = pd.read_csv(root / "contract_episodes.csv")
        requested = tuple(commodities)
        episodes = episodes.loc[episodes["commodity"].isin(requested)].copy()
        frames: dict[str, dict[str, pd.DataFrame]] = {}
        for commodity in requested:
            candidates = episodes.loc[episodes.commodity == commodity].sort_values("main_start_date")
            selected = candidates
            if max_contracts_per_commodity is not None:
                selected = candidates.tail(max_contracts_per_commodity)
            minute_contracts = set(selected["contract_uid"].astype(str))
            frames[commodity] = {
                "minute": _read_csv(root / commodity / f"{commodity}_1m.csv", "1m", minute_contracts),
                # Daily/weekly stay complete because earlier contracts form commodity history.
                "daily": _read_csv(root / commodity / f"{commodity}_1d.csv", "1d"),
                "weekly": _read_csv(root / commodity / f"{commodity}_1w.csv", "1w"),
            }
        return cls(episodes, frames, root)


class V11ContractDataset(Dataset[dict[str, Any]]):
    """Real-contract V1.1 samples. Metadata never enters model tensors."""

    def __init__(self, store: V11DataStore, config: dict, *, role: str = "train", scaler: SharedIMCScaler | None = None) -> None:
        self.store, self.config, self.role, self.scaler = store, config, role, scaler
        self.horizons = tuple(int(x) for x in config["data"]["horizons"])
        self.minute_capacity = int(config["data"]["minute_capacity"])
        self.daily_capacity = int(config["data"]["daily_capacity"])
        self.current_weekly_capacity = int(config["data"]["current_weekly_capacity"])
        self.history_weekly_capacity = int(config["data"]["history_weekly_capacity"])
        self.history_years = int(config["data"]["history_years"])
        self.episode_arrays: list[EpisodeArrays] = []
        self.offsets = [0]
        self.daily_truncation_count = 0
        self.daily_truncated_tokens = 0
        self.excluded_episodes: list[dict[str, str]] = []
        self._history_cache: dict[tuple[str, str, int], tuple[np.ndarray, ...]] = {}
        self._build()

    def _build(self) -> None:
        stride, max_horizon = int(self.config["data"]["anchor_stride"]), max(self.horizons)
        rows = self.store.episode_table.loc[self.store.episode_table.role == self.role]
        for row in rows.sort_values(["commodity", "main_start_date"]).itertuples(index=False):
            episode = self.store.episodes[(str(row.commodity), str(row.contract_uid), int(row.episode_id))]
            scales = self.store.frames.get(episode.commodity)
            if scales is None:
                continue
            minute = scales["minute"].loc[scales["minute"].contract_uid == episode.contract_uid].copy()
            if minute.empty:
                self.excluded_episodes.append({"contract_uid": episode.contract_uid, "reason": "missing_minute_bars"})
                continue
            minute = minute.sort_values("datetime", kind="mergesort").reset_index(drop=True)
            legal = minute["trading_date"].between(episode.main_start, episode.anchor_end).to_numpy()
            positions = np.flatnonzero(legal)
            positions = positions[positions + max_horizon < len(minute)][::stride]
            if not len(positions):
                self.excluded_episodes.append({"contract_uid": episode.contract_uid, "reason": "no_all_horizon_anchor"})
                continue
            daily_raw = scales["daily"].loc[scales["daily"].contract_uid == episode.contract_uid].copy()
            daily = daily_raw.loc[daily_raw["trading_date"].between(episode.main_start, episode.anchor_end)].copy()
            daily = daily.sort_values("trading_date", kind="mergesort").reset_index(drop=True)
            weekly = _aggregate_weekly(daily)
            weekly_raw = scales["weekly"].loc[scales["weekly"].contract_uid == episode.contract_uid].copy()
            weekly_raw = weekly_raw.rename(columns={"week_end_date": "period_end"}).sort_values("period_end").reset_index(drop=True)
            first_minute = minute.loc[minute.trading_date >= episode.main_start].iloc[0]
            daily_previous = _previous_row(daily_raw, "trading_date", episode.main_start)
            daily_prior = daily_raw.loc[daily_raw.trading_date < episode.main_start].tail(20).volume.to_numpy(dtype=np.float64)
            daily_baseline = V11IMCTransform.origin(
                np.asarray([float(first_minute.open)]), np.asarray([float(first_minute.open_interest)]), daily_prior
            )
            weekly_previous = _previous_row(weekly_raw, "period_end", episode.main_start)
            weekly_prior = weekly_raw.loc[weekly_raw.period_end < episode.main_start].tail(20).volume.to_numpy(dtype=np.float64)
            weekly_baseline = V11IMCTransform.origin(
                np.asarray([float(first_minute.open)]), np.asarray([float(first_minute.open_interest)]), weekly_prior
            )
            arrays = EpisodeArrays(
                episode, minute, minute.loc[:, BAR_COLUMNS].to_numpy(dtype=np.float64), positions,
                daily_raw, daily, daily_baseline, daily_prior,
                None if daily_previous is None else float(daily_previous.close),
                None if daily_previous is None else float(daily_previous.open_interest),
                weekly_raw, weekly, weekly_baseline, weekly_prior,
                None if weekly_previous is None else float(weekly_previous.close),
                None if weekly_previous is None else float(weekly_previous.open_interest),
            )
            self.episode_arrays.append(arrays)
            self.offsets.append(self.offsets[-1] + len(positions))
        if not self.episode_arrays:
            raise ValueError(f"no valid {self.role} real-contract episodes")

    def __len__(self) -> int:
        return self.offsets[-1]

    @property
    def hierarchy(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for index, arrays in enumerate(self.episode_arrays):
            result.setdefault(arrays.episode.commodity, []).append(index)
        return result

    def global_index(self, episode_index: int, local_anchor_index: int) -> int:
        if not 0 <= local_anchor_index < len(self.episode_arrays[episode_index].anchors):
            raise IndexError("anchor index out of range")
        return self.offsets[episode_index] + local_anchor_index

    @staticmethod
    def _pad(values: np.ndarray, validity: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(values) > capacity:
            values, validity = values[-capacity:], validity[-capacity:]
        result = np.zeros((capacity, values.shape[-1]), dtype=np.float32)
        result_valid = np.zeros_like(result, dtype=np.bool_)
        mask = np.ones(capacity, dtype=np.bool_)
        if len(values):
            result[-len(values):] = values
            result_valid[-len(values):] = validity
            mask[-len(values):] = False
        return result, result_valid, mask

    def _scaled(self, values: np.ndarray, validity: np.ndarray) -> np.ndarray:
        return values if self.scaler is None else self.scaler.transform(values, validity)

    @staticmethod
    def _minute_context(frame: pd.DataFrame) -> np.ndarray:
        stamp = frame["datetime"]
        minute = stamp.dt.hour.to_numpy() * 60 + stamp.dt.minute.to_numpy()
        weekday = frame["trading_date"].dt.weekday.to_numpy()
        delta = stamp.diff().dt.total_seconds().div(60).fillna(0).to_numpy()
        return np.column_stack((
            np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440),
            np.sin(2 * np.pi * weekday / 5), np.cos(2 * np.pi * weekday / 5),
            np.log1p(np.maximum(delta, 0)),
        )).astype(np.float32)

    def _history(self, current: ContractEpisode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if current.key in self._history_cache:
            return self._history_cache[current.key]
        cutoff = current.main_start - pd.DateOffset(years=self.history_years)
        pieces: list[tuple[ContractEpisode, pd.DataFrame]] = []
        candidates = self.store.episode_table.loc[
            (self.store.episode_table.commodity == current.commodity)
            & (self.store.episode_table.role == current.role)
            & (pd.to_datetime(self.store.episode_table.main_start_date) < current.main_start)
        ]
        raw_daily = self.store.frames[current.commodity]["daily"]
        for row in candidates.sort_values("main_start_date").itertuples(index=False):
            episode = self.store.episodes[(current.commodity, str(row.contract_uid), int(row.episode_id))]
            daily = raw_daily.loc[(raw_daily.contract_uid == episode.contract_uid)
                                  & raw_daily.trading_date.between(max(cutoff, episode.main_start), min(current.main_start - pd.Timedelta(days=1), episode.main_end))]
            weekly = _aggregate_weekly(daily)
            if not weekly.empty:
                weekly["contract_uid"] = episode.contract_uid
                weekly["main_start"] = episode.main_start
                pieces.append((episode, weekly))
        market_parts, valid_parts, context_parts, boundary_parts = [], [], [], []
        raw_weekly_all = self.store.frames[current.commodity]["weekly"]
        for episode, weekly in pieces:
            raw = raw_weekly_all.loc[raw_weekly_all.contract_uid == episode.contract_uid].rename(columns={"week_end_date": "period_end"})
            market, validity = _transform_lifecycle(weekly, raw, "period_end")
            ages = ((weekly.period_end - episode.main_start).dt.days // 7).clip(lower=0).to_numpy()
            ago = ((current.main_start - weekly.period_end).dt.days // 7).clip(lower=0).to_numpy()
            context = np.column_stack((
                ago, ages, np.zeros(len(weekly)), np.zeros(len(weekly)), weekly.trading_day_count,
            )).astype(np.float32)
            boundary = np.zeros(len(weekly), dtype=np.float32)
            boundary[0] = 1.0
            market_parts.append(market); valid_parts.append(validity); context_parts.append(context); boundary_parts.append(boundary)
        if market_parts:
            market = np.concatenate(market_parts); validity = np.concatenate(valid_parts)
            context = np.concatenate(context_parts); boundary = np.concatenate(boundary_parts)
            contract_ids = np.concatenate([
                np.full(len(weekly), episode.contract_uid, dtype=object) for episode, weekly in pieces
            ])
            if len(market) > self.history_weekly_capacity:
                market, validity, context, boundary, contract_ids = (
                    item[-self.history_weekly_capacity:] for item in (market, validity, context, boundary, contract_ids)
                )
                boundary[:] = 0
                boundary[0] = 1
                boundary[1:] = (contract_ids[1:] != contract_ids[:-1]).astype(np.float32)
        else:
            market = np.empty((0, len(IMC_FEATURES)), dtype=np.float32)
            validity = np.empty_like(market, dtype=np.bool_)
            context = np.empty((0, len(WEEKLY_CONTEXT_FEATURES)), dtype=np.float32)
            boundary = np.empty(0, dtype=np.float32)
        market, validity, mask = self._pad(self._scaled(market, validity), validity, self.history_weekly_capacity)
        padded_context = np.zeros((self.history_weekly_capacity, len(WEEKLY_CONTEXT_FEATURES)), dtype=np.float32)
        padded_boundary = np.zeros(self.history_weekly_capacity, dtype=np.float32)
        n = int((~mask).sum())
        if n:
            padded_context[-n:] = context[-n:]
            padded_boundary[-n:] = boundary[-n:]
        result = market, padded_context, mask, validity, padded_boundary
        self._history_cache[current.key] = result
        return result

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        episode_index = bisect_right(self.offsets, item) - 1
        arrays = self.episode_arrays[episode_index]
        local = item - self.offsets[episode_index]
        anchor = int(arrays.anchors[local])
        start = max(0, anchor - self.minute_capacity + 1)
        window = arrays.minute_bars[start : anchor + 1]
        prior_start = max(0, start - 20)
        prior = arrays.minute_bars[prior_start:start]
        previous = arrays.minute_bars[start - 1] if start else None
        minute_market, minute_validity, origin = V11IMCTransform.window(
            window, prior_volume=prior[:, 4],
            previous_close=None if previous is None else float(previous[3]),
            previous_oi=None if previous is None else float(previous[5]),
        )
        minute_context = self._minute_context(arrays.minute_frame.iloc[start : anchor + 1])
        minute_market, minute_validity, minute_mask = self._pad(
            self._scaled(minute_market, minute_validity), minute_validity, self.minute_capacity
        )
        padded_minute_context = np.zeros((self.minute_capacity, len(MINUTE_CONTEXT_FEATURES)), dtype=np.float32)
        padded_minute_context[-len(window):] = minute_context

        anchor_date = pd.Timestamp(arrays.minute_frame.iloc[anchor].trading_date)
        daily_n = int(np.searchsorted(arrays.daily_frame.trading_date.to_numpy(dtype="datetime64[ns]"), np.datetime64(anchor_date), side="left"))
        current_minutes = arrays.minute_frame.iloc[: anchor + 1]
        current_minutes = current_minutes.loc[current_minutes.trading_date == anchor_date]
        daily_partial = _partial_bar(current_minutes, "trading_date")
        daily_sequence = pd.concat((arrays.daily_frame.iloc[:daily_n], daily_partial), ignore_index=True)
        daily_market_raw, daily_validity_raw = V11IMCTransform.transform(
            daily_sequence.loc[:, BAR_COLUMNS].to_numpy(dtype=np.float64),
            origin=arrays.daily_origin, prior_volume=arrays.daily_prior_volume,
            previous_close=arrays.daily_previous_close, previous_oi=arrays.daily_previous_oi,
        )
        if len(daily_sequence) > self.daily_capacity:
            self.daily_truncation_count += 1
            self.daily_truncated_tokens += len(daily_sequence) - self.daily_capacity
        daily_values, daily_valid, daily_mask = self._pad(
            self._scaled(daily_market_raw, daily_validity_raw), daily_validity_raw, self.daily_capacity,
        )
        daily_rows = daily_sequence.iloc[-self.daily_capacity:]
        daily_context = np.zeros((self.daily_capacity, len(DAILY_CONTEXT_FEATURES)), dtype=np.float32)
        if len(daily_rows):
            dates = daily_rows.trading_date
            partial_flags = np.zeros(len(daily_rows), dtype=np.float32)
            partial_flags[-1] = 1.0
            elapsed = np.zeros(len(daily_rows), dtype=np.float32)
            elapsed[-1] = float((current_minutes.iloc[-1].datetime - current_minutes.iloc[0].datetime).total_seconds() / 60)
            context = np.column_stack((
                (dates - arrays.episode.main_start).dt.days,
                (dates <= arrays.episode.main_end).astype(np.float32),
                np.maximum((dates - arrays.episode.main_end).dt.days, 0),
                partial_flags, elapsed,
            )).astype(np.float32)
            daily_context[-len(context):] = context

        anchor_iso = anchor_date.isocalendar()
        week_keys = np.asarray([
            int(date.isocalendar().year) * 100 + int(date.isocalendar().week)
            for date in arrays.current_weekly_frame.period_end
        ], dtype=np.int64)
        anchor_week_key = int(anchor_iso.year) * 100 + int(anchor_iso.week)
        week_n = int(np.searchsorted(week_keys, anchor_week_key, side="left"))
        minute_iso = arrays.minute_frame.trading_date.dt.isocalendar()
        weekly_minutes = arrays.minute_frame.iloc[: anchor + 1].loc[
            (minute_iso.year.iloc[: anchor + 1].to_numpy() == anchor_iso.year)
            & (minute_iso.week.iloc[: anchor + 1].to_numpy() == anchor_iso.week)
            & (arrays.minute_frame.trading_date.iloc[: anchor + 1].to_numpy(dtype="datetime64[ns]") >= np.datetime64(arrays.episode.main_start))
        ]
        weekly_partial = _partial_bar(weekly_minutes, "period_end")
        weekly_partial["trading_day_count"] = weekly_minutes.trading_date.nunique()
        weekly_sequence = pd.concat((arrays.current_weekly_frame.iloc[:week_n], weekly_partial), ignore_index=True)
        weekly_market_raw, weekly_validity_raw = V11IMCTransform.transform(
            weekly_sequence.loc[:, BAR_COLUMNS].to_numpy(dtype=np.float64),
            origin=arrays.weekly_origin, prior_volume=arrays.weekly_prior_volume,
            previous_close=arrays.weekly_previous_close, previous_oi=arrays.weekly_previous_oi,
        )
        weekly_values, weekly_valid, weekly_mask = self._pad(
            self._scaled(weekly_market_raw, weekly_validity_raw), weekly_validity_raw,
            self.current_weekly_capacity,
        )
        week_rows = weekly_sequence.iloc[-self.current_weekly_capacity:]
        weekly_context = np.zeros((self.current_weekly_capacity, len(WEEKLY_CONTEXT_FEATURES)), dtype=np.float32)
        if len(week_rows):
            partial_flags = np.zeros(len(week_rows), dtype=np.float32)
            partial_flags[-1] = 1.0
            context = np.column_stack((
                ((anchor_date - week_rows.period_end).dt.days // 7).clip(lower=0),
                ((week_rows.period_end - arrays.episode.main_start).dt.days // 7).clip(lower=0),
                np.ones(len(week_rows)),
                partial_flags, week_rows.trading_day_count,
            )).astype(np.float32)
            weekly_context[-len(context):] = context

        history_market, history_context, history_mask, history_valid, history_boundary = self._history(arrays.episode)
        targets, target_validity, target_masks = {}, {}, {}
        target_prior = arrays.minute_bars[max(0, anchor + 1 - 20) : anchor + 1, 4]
        for horizon in self.horizons:
            future = arrays.minute_bars[anchor + 1 : anchor + horizon + 1]
            previous_bar = arrays.minute_bars[anchor]
            values, validity = V11IMCTransform.transform(
                future, origin=origin, prior_volume=target_prior,
                previous_close=float(previous_bar[3]), previous_oi=float(previous_bar[5]),
            )
            targets[horizon] = torch.from_numpy(self._scaled(values, validity))
            target_validity[horizon] = torch.from_numpy(validity)
            target_masks[horizon] = torch.zeros(horizon, dtype=torch.bool)

        return {
            "minute_market": torch.from_numpy(minute_market),
            "minute_context": torch.from_numpy(padded_minute_context),
            "minute_mask": torch.from_numpy(minute_mask),
            "minute_imc_validity": torch.from_numpy(minute_validity),
            "daily_market": torch.from_numpy(daily_values),
            "daily_context": torch.from_numpy(daily_context),
            "daily_mask": torch.from_numpy(daily_mask),
            "daily_imc_validity": torch.from_numpy(daily_valid),
            "current_weekly_market": torch.from_numpy(weekly_values),
            "current_weekly_context": torch.from_numpy(weekly_context),
            "current_weekly_mask": torch.from_numpy(weekly_mask),
            "current_weekly_imc_validity": torch.from_numpy(weekly_valid),
            "history_weekly_market": torch.from_numpy(history_market),
            "history_weekly_context": torch.from_numpy(history_context),
            "history_weekly_mask": torch.from_numpy(history_mask),
            "history_weekly_imc_validity": torch.from_numpy(history_valid),
            "history_weekly_contract_boundary": torch.from_numpy(history_boundary),
            "target_minute_market": targets,
            "target_minute_imc_validity": target_validity,
            "target_minute_mask": target_masks,
            "metadata": {
                "commodity": arrays.episode.commodity, "contract_uid": arrays.episode.contract_uid,
                "episode_id": arrays.episode.episode_id,
                "anchor_datetime": str(arrays.minute_frame.iloc[anchor].datetime),
                "anchor_position": anchor,
            },
        }


def collate_v11_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [key for key, value in samples[0].items() if isinstance(value, torch.Tensor)]
    result = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in ("target_minute_market", "target_minute_imc_validity", "target_minute_mask"):
        result[key] = {
            horizon: torch.stack([sample[key][horizon] for sample in samples])
            for horizon in samples[0][key]
        }
    result["metadata"] = [sample["metadata"] for sample in samples]
    return result


def fit_v11_shared_scaler(
    dataset: V11ContractDataset, anchors_per_commodity: int = 32,
) -> SharedIMCScaler:
    """Fit one scaler from deterministic Train-only online-memory snapshots."""
    if dataset.role != "train" or dataset.scaler is not None:
        raise ValueError("shared scaler requires an unscaled Train dataset")
    if anchors_per_commodity <= 0:
        raise ValueError("anchors_per_commodity must be positive")
    population: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for commodity in TRAIN_COMMODITIES:
        episode_indices = dataset.hierarchy.get(commodity, [])
        candidates = [
            dataset.global_index(episode_index, local)
            for episode_index in episode_indices
            for local in range(len(dataset.episode_arrays[episode_index].anchors))
        ]
        if not candidates:
            raise ValueError(f"shared scaler has no valid anchors for {commodity}")
        selected = np.linspace(
            0, len(candidates) - 1, min(anchors_per_commodity, len(candidates)), dtype=np.int64,
        )
        for position in selected:
            sample = dataset[candidates[int(position)]]
            for source in ("minute", "daily", "current_weekly", "history_weekly"):
                population.append((
                    commodity, source,
                    sample[f"{source}_market"].numpy(),
                    sample[f"{source}_imc_validity"].numpy(),
                ))
    return SharedIMCScaler.fit(population)
