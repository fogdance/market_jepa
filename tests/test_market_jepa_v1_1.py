from __future__ import annotations

import inspect
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from benchmark_market_jepa_v1_1_batch import profile_plan
from market_jepa.model import jepa_loss
from market_jepa_v1_1 import DEFAULT_V11_CONFIG, MarketJEPAV11
from market_jepa_v1_1.config import (
    DAILY_CONTEXT_FEATURES, IMC_FEATURES, MINUTE_CONTEXT_FEATURES,
    TRAIN_COMMODITIES, WEEKLY_CONTEXT_FEATURES, validate_v11_config,
)
from market_jepa_v1_1.dataset import (
    BAR_COLUMNS, V11ContractDataset, V11DataStore, collate_v11_batch,
    fit_v11_shared_scaler,
)
from market_jepa_v1_1.imc import IMCOrigin, SharedIMCScaler, V11IMCTransform
from market_jepa_v1_1.sampler import HierarchicalCommodityContractSampler
from market_jepa_v1_1.checkpoint import (
    load_v11_checkpoint, model_from_checkpoint, validate_v11_checkpoint,
    v11_implementation_manifest,
)
from market_jepa_v1_1.formal_training import (
    EPOCH_FIELDS, _filter_train_commodities_by_history, validate_formal_training_config,
)
from market_jepa_v1_1.development import audit_history_week_eligibility
from market_jepa_v1_1.training import V11Trainer


def _bars(count: int, *, base: float = 100.0) -> np.ndarray:
    close = base + np.arange(count, dtype=np.float64)
    return np.column_stack((close - 0.5, close + 1, close - 1, close,
                            np.arange(1, count + 1) * 10.0,
                            1000 + np.arange(count) * 5.0))


def _debug_model_config() -> dict:
    value = deepcopy(DEFAULT_V11_CONFIG["model"])
    value.update(
        d_model=16, belief_dim=16, num_heads=4, ffn_dim=32,
        minute_layers=1, minute_capacity=8, daily_capacity=4,
        current_weekly_capacity=3, history_weekly_capacity=5,
        commodity_state_tokens=2, contract_state_tokens=2,
        belief_tokens=2, predictor_hidden=32, dropout=0.0,
    )
    return value


def _model_batch(batch_size: int = 2) -> dict:
    torch.manual_seed(11)
    lengths = {"minute": 8, "daily": 4, "current_weekly": 3, "history_weekly": 5}
    contexts = {
        "minute": len(MINUTE_CONTEXT_FEATURES), "daily": len(DAILY_CONTEXT_FEATURES),
        "current_weekly": len(WEEKLY_CONTEXT_FEATURES),
        "history_weekly": len(WEEKLY_CONTEXT_FEATURES),
    }
    batch = {}
    for source, length in lengths.items():
        batch[f"{source}_market"] = torch.randn(batch_size, length, len(IMC_FEATURES))
        batch[f"{source}_context"] = torch.randn(batch_size, length, contexts[source])
        batch[f"{source}_mask"] = torch.zeros(batch_size, length, dtype=torch.bool)
        batch[f"{source}_imc_validity"] = torch.ones(batch_size, length, len(IMC_FEATURES), dtype=torch.bool)
    boundary = torch.zeros(batch_size, lengths["history_weekly"])
    boundary[:, 0] = 1
    batch["history_weekly_contract_boundary"] = boundary
    batch["target_minute_market"] = {h: torch.randn(batch_size, h, len(IMC_FEATURES)) for h in (16, 64, 256)}
    batch["target_minute_imc_validity"] = {
        h: torch.ones(batch_size, h, len(IMC_FEATURES), dtype=torch.bool) for h in (16, 64, 256)
    }
    batch["target_minute_mask"] = {h: torch.zeros(batch_size, h, dtype=torch.bool) for h in (16, 64, 256)}
    return batch


@pytest.fixture
def v11_model():
    torch.manual_seed(12)
    return MarketJEPAV11(_debug_model_config(), debug=True).eval()


def _make_store(commodities=("FG",), contracts_per_commodity: int = 3) -> V11DataStore:
    episode_rows, frames = [], {}
    for commodity_index, commodity in enumerate(commodities):
        minute_parts, daily_parts, weekly_parts = [], [], []
        for contract_index in range(contracts_per_commodity):
            contract = f"{commodity}{contract_index + 1:02d}"
            start = pd.Timestamp("2022-01-03") + pd.DateOffset(months=contract_index * 2)
            days = pd.bdate_range(start, periods=10)
            base = 100 + commodity_index * 100 + contract_index * 20
            minute_rows = []
            for day_index, day in enumerate(days):
                for minute_index in range(100):
                    stamp = day + pd.Timedelta(hours=9, minutes=minute_index)
                    close = base + day_index + minute_index / 1000
                    minute_rows.append({
                        "commodity": commodity, "contract_uid": contract,
                        "datetime": stamp, "trading_date": day,
                        "open": close - 0.02, "high": close + 0.05,
                        "low": close - 0.05, "close": close,
                        "volume": float(10 + minute_index),
                        "open_interest": float(1000 + day_index * 100 + minute_index),
                    })
            minute = pd.DataFrame(minute_rows)
            minute_parts.append(minute)

            closed = minute.groupby("trading_date", sort=True).agg(
                open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"),
                open_interest=("open_interest", "last"),
            ).reset_index()
            warm_dates = pd.bdate_range(days[0] - pd.Timedelta(days=45), periods=20)
            warm = pd.DataFrame({
                "trading_date": warm_dates, "open": base - 2, "high": base - 1,
                "low": base - 3, "close": base - 2,
                "volume": np.arange(20) + 100, "open_interest": np.arange(20) + 800,
            })
            daily = pd.concat((warm, closed), ignore_index=True)
            daily.insert(0, "contract_uid", contract); daily.insert(0, "commodity", commodity)
            daily_parts.append(daily)

            warm_weeks = pd.date_range(days[0] - pd.Timedelta(weeks=20), periods=20, freq="W-FRI")
            weekly = pd.DataFrame({
                "commodity": commodity, "contract_uid": contract,
                "week_end_date": warm_weeks, "open": base - 4, "high": base - 3,
                "low": base - 5, "close": base - 4,
                "volume": np.arange(20) + 1000, "open_interest": np.arange(20) + 700,
            })
            weekly_parts.append(weekly)
            episode_rows.append({
                "commodity": commodity, "contract_uid": contract,
                "delivery_year": 2022 + contract_index, "delivery_month": 1,
                "series_key": f"{commodity}-01",
                "episode_id": contract_index, "main_start_date": days[0],
                "main_end_date": days[7], "anchor_end_date": days[9], "role": "train",
            })
        frames[commodity] = {
            "minute": pd.concat(minute_parts, ignore_index=True),
            "daily": pd.concat(daily_parts, ignore_index=True),
            "weekly": pd.concat(weekly_parts, ignore_index=True),
        }
    return V11DataStore(pd.DataFrame(episode_rows), frames)


def _make_fg_lineage_store() -> V11DataStore:
    store = _make_store(("FG",), contracts_per_commodity=6)
    identities = (
        ("FG202401", 2024, 1, "FG-01"), ("FG202405", 2024, 5, "FG-05"),
        ("FG202501", 2025, 1, "FG-01"), ("FG202505", 2025, 5, "FG-05"),
        ("FG202601", 2026, 1, "FG-01"), ("FG202605", 2026, 5, "FG-05"),
    )
    mapping = {f"FG{index + 1:02d}": uid for index, (uid, _, _, _) in enumerate(identities)}
    episodes = store.episode_table.copy()
    episodes["contract_uid"] = episodes.contract_uid.map(mapping)
    for index, (uid, year, month, series_key) in enumerate(identities):
        episodes.loc[episodes.episode_id == index, [
            "delivery_year", "delivery_month", "series_key", "role",
        ]] = [year, month, series_key, "train" if index >= 4 else "validation"]
    frames = {}
    for scale, original in store.frames["FG"].items():
        frame = original.copy()
        frame["contract_uid"] = frame.contract_uid.map(mapping)
        frames[scale] = frame
    return V11DataStore(episodes, {"FG": frames})


def _dataset_config() -> dict:
    value = deepcopy(DEFAULT_V11_CONFIG)
    value["data"].update(
        minute_capacity=16, daily_capacity=8, current_weekly_capacity=4,
        anchor_stride=17,
    )
    value["history_week"].update(capacity=8, require_full_history=False)
    return value


def _prepend_reliable_week(store: V11DataStore, starts: dict[str, str]) -> V11DataStore:
    for commodity, date in starts.items():
        frame = store.frames[commodity]["weekly"]
        contract = str(store.episode_table.loc[store.episode_table.commodity == commodity, "contract_uid"].iloc[0])
        row = pd.DataFrame([{
            "commodity": commodity, "contract_uid": contract, "week_end_date": pd.Timestamp(date),
            "open": 90.0, "high": 92.0, "low": 89.0, "close": 91.0,
            "volume": 1000.0, "open_interest": 800.0,
        }])
        store.frames[commodity]["weekly"] = pd.concat((row, frame), ignore_index=True)
    return store


@pytest.fixture(scope="module")
def contract_dataset():
    return V11ContractDataset(_make_store(), _dataset_config())


def _sample_for_anchor(dataset: V11ContractDataset, episode_index: int, anchor_position: int):
    local = int(np.flatnonzero(dataset.episode_arrays[episode_index].anchors == anchor_position)[0])
    return dataset[dataset.global_index(episode_index, local)]


def test_v11_minute_imc_fixed_origin():
    bars = _bars(4)
    values, validity, origin = V11IMCTransform.window(bars, prior_volume=np.arange(1, 21))
    assert origin.price == bars[0, 3]
    np.testing.assert_allclose(values[:, 3], np.log(bars[:, 3] / bars[0, 3]), rtol=1e-6)
    assert validity[:, 3].all()


def test_v11_minute_oi_step_identity():
    bars = _bars(4)
    values, validity, _ = V11IMCTransform.window(bars, prior_volume=np.arange(1, 21))
    np.testing.assert_allclose(values[1:, 6], np.log(bars[1:, 5] / bars[:-1, 5]), rtol=1e-6)
    assert not validity[0, 6] and validity[1:, 6].all()


def test_v11_nonpositive_oi_is_explicitly_invalid():
    bars = _bars(3); bars[1, 5] = 0
    values, validity, _ = V11IMCTransform.window(bars, prior_volume=np.arange(1, 21))
    assert values[1, 5] == values[1, 6] == 0
    assert not validity[1, 5] and not validity[1, 6]
    assert not validity[2, 6]


def test_v11_minute_volume_fixed_baseline():
    bars = _bars(2)
    values, validity, origin = V11IMCTransform.window(bars, prior_volume=np.arange(1, 21))
    assert origin.volume_baseline == 10.5
    np.testing.assert_allclose(values[:, 7], bars[:, 4] / 10.5)
    assert validity[:, 7].all()


def test_v11_minute_q20():
    bars = _bars(2)
    prior = np.arange(1, 21, dtype=np.float64)
    values, validity, _ = V11IMCTransform.window(bars, prior_volume=prior)
    assert values[0, 8] == pytest.approx(bars[0, 4] / np.median(prior))
    expected_second_baseline = np.median(np.concatenate((prior[1:], bars[:1, 4])))
    assert values[1, 8] == pytest.approx(bars[1, 4] / expected_second_baseline)
    assert validity[:, 8].all()


def test_v11_missing_baseline_never_reads_future():
    first = _bars(4); second = first.copy(); second[1:, 4] *= 1000
    a, av, _ = V11IMCTransform.window(first, prior_volume=np.arange(19))
    b, bv, _ = V11IMCTransform.window(second, prior_volume=np.arange(19))
    assert not av[0, 7] and not av[0, 8]
    assert not bv[0, 7] and not bv[0, 8]
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(av[0], bv[0])


def _scaler_population(include_rb: bool = False):
    values = np.arange(18, dtype=np.float32).reshape(2, 9) + 1
    valid = np.ones_like(values, dtype=np.bool_)
    population = [(commodity, "minute", values + index, valid) for index, commodity in enumerate(TRAIN_COMMODITIES)]
    if include_rb:
        population.append(("RB", "minute", values, valid))
    return population


def test_v11_shared_scaler_only():
    scaler = SharedIMCScaler.fit(_scaler_population())
    assert scaler.fitted_commodities == tuple(sorted(TRAIN_COMMODITIES))
    restored = SharedIMCScaler.from_dict(scaler.to_dict())
    assert restored.checksum == scaler.checksum
    with pytest.raises(ValueError, match="every configured Train commodity"):
        SharedIMCScaler.fit(_scaler_population()[:-1])


def test_v11_rb_does_not_fit_scaler():
    with pytest.raises(ValueError, match="forbidden commodity RB"):
        SharedIMCScaler.fit(_scaler_population(include_rb=True))


def test_v11_minute_never_crosses_contract(contract_dataset):
    sample = contract_dataset[0]
    arrays = contract_dataset.episode_arrays[0]
    position = sample["metadata"]["anchor_position"]
    source = arrays.minute_frame.iloc[max(0, position - 15): position + 1]
    assert source.contract_uid.nunique() == 1
    assert source.contract_uid.iloc[0] == sample["metadata"]["contract_uid"]


def test_v11_daily_never_crosses_contract(contract_dataset):
    sample = contract_dataset[0]
    arrays = contract_dataset.episode_arrays[0]
    assert arrays.daily_frame.contract_uid.nunique() == 1
    assert arrays.daily_frame.contract_uid.iloc[0] == sample["metadata"]["contract_uid"]
    assert sample["daily_context"][~sample["daily_mask"], 3].sum() == 1


def test_v11_current_weekly_never_crosses_contract(contract_dataset):
    sample = contract_dataset[0]
    arrays = contract_dataset.episode_arrays[0]
    assert arrays.weekly_raw.contract_uid.nunique() == 1
    assert arrays.weekly_raw.contract_uid.iloc[0] == sample["metadata"]["contract_uid"]
    assert sample["current_weekly_context"][~sample["current_weekly_mask"], 3].sum() == 1


def test_v11_history_weekly_boundary_flags():
    config = _dataset_config(); config["history_week"]["capacity"] = 64
    dataset = V11ContractDataset(_make_store(), config)
    arrays = dataset.episode_arrays[2]
    market, context, mask, validity, boundary = dataset._history(arrays.episode)
    assert market.shape == validity.shape == (64, len(IMC_FEATURES))
    assert boundary[~mask].sum() == 3
    assert np.all(boundary[mask] == 0)
    assert np.all(context[~mask, 2] == 0)


def test_lineage_contract_boundary_resets_imc():
    config = _dataset_config(); config["history_week"]["capacity"] = 64
    dataset = V11ContractDataset(_make_store(), config)
    current = dataset.episode_arrays[2]
    market, _, mask, validity, boundary = dataset._history(current.episode)
    boundary_positions = np.flatnonzero((~mask) & boundary.astype(bool))
    assert len(boundary_positions) == 3
    np.testing.assert_allclose(market[boundary_positions, 3], 0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(market[boundary_positions, 5], 0, rtol=0, atol=1e-7)
    assert validity[boundary_positions, 3].all() and validity[boundary_positions, 5].all()


def test_fg601_history_contains_only_fg01_lineage():
    dataset = V11ContractDataset(_make_fg_lineage_store(), _dataset_config())
    current = next(item for item in dataset.episode_arrays if item.episode.contract_uid == "FG202601")
    rows = dataset.history_weekly_rows(current.episode)
    assert set(rows.series_key) == {"FG-01"}
    assert set(rows.contract_uid) <= {"FG202401", "FG202501", "FG202601"}


def test_fg605_history_contains_only_fg05_lineage():
    dataset = V11ContractDataset(_make_fg_lineage_store(), _dataset_config())
    current = next(item for item in dataset.episode_arrays if item.episode.contract_uid == "FG202605")
    rows = dataset.history_weekly_rows(current.episode)
    assert set(rows.series_key) == {"FG-05"}
    assert set(rows.contract_uid) <= {"FG202405", "FG202505", "FG202605"}


def test_history_excludes_other_delivery_months():
    dataset = V11ContractDataset(_make_fg_lineage_store(), _dataset_config())
    for current in dataset.episode_arrays:
        rows = dataset.history_weekly_rows(current.episode)
        assert (rows.delivery_month == current.episode.delivery_month).all()
        assert (rows.series_key == current.episode.series_key).all()


def test_lineage_overlap_keeps_previous_delivery_year():
    store = _make_fg_lineage_store()
    weekly = store.frames["FG"]["weekly"]
    template = weekly.loc[weekly.contract_uid == "FG202401"].iloc[-1].copy()
    overlap_date = pd.Timestamp("2022-08-25")
    additions = []
    for contract_uid, close in (("FG202401", 111.0), ("FG202501", 222.0)):
        row = template.copy()
        row["contract_uid"], row["week_end_date"] = contract_uid, overlap_date
        row[["open", "high", "low", "close"]] = [close, close + 1, close - 1, close]
        additions.append(row)
    store.frames["FG"]["weekly"] = pd.concat((weekly, pd.DataFrame(additions)), ignore_index=True)
    dataset = V11ContractDataset(store, _dataset_config())
    current = next(item for item in dataset.episode_arrays if item.episode.contract_uid == "FG202601")
    rows = dataset.history_weekly_rows(current.episode)
    iso = rows.period_end.dt.isocalendar()
    target = overlap_date.isocalendar()
    selected = rows.loc[(iso.year == target.year) & (iso.week == target.week)]
    assert len(selected) == 1
    assert selected.iloc[0].contract_uid == "FG202401"
    assert int(selected.iloc[0].delivery_year) == 2024


def test_v11_history_cache_respects_shared_scaler():
    dataset = V11ContractDataset(_make_store(), _dataset_config())
    index = dataset.global_index(2, 0)
    raw = dataset[index]
    assert dataset._history_cache
    scaler = SharedIMCScaler.fit(_scaler_population())
    dataset.set_scaler(scaler)
    scaled = dataset[index]
    expected = scaler.transform(
        raw["history_weekly_market"].numpy(), raw["history_weekly_imc_validity"].numpy(),
    )
    np.testing.assert_allclose(scaled["history_weekly_market"].numpy(), expected, rtol=0, atol=0)
    assert not torch.equal(raw["history_weekly_market"], scaled["history_weekly_market"])
    with pytest.raises(AttributeError):
        dataset.scaler = scaler


def test_current_contract_pre_main_enters_history():
    dataset = V11ContractDataset(_make_fg_lineage_store(), _dataset_config())
    current = next(item for item in dataset.episode_arrays if item.episode.contract_uid == "FG202601")
    rows = dataset.history_weekly_rows(current.episode)
    current_rows = rows.loc[rows.contract_uid == "FG202601"]
    assert not current_rows.empty
    assert (current_rows.period_end < current.episode.main_start).all()


def test_current_weekly_starts_at_main_start():
    dataset = V11ContractDataset(_make_fg_lineage_store(), _dataset_config())
    for current in dataset.episode_arrays:
        assert not current.current_weekly_frame.empty
        assert (current.current_weekly_frame.period_end >= current.episode.main_start).all()


def test_v11_context_features_are_bounded_counts_not_wall_clock():
    dataset = V11ContractDataset(_make_store(), _dataset_config())
    sample = dataset[dataset.global_index(0, 3)]
    for source in ("daily", "current_weekly", "history_weekly"):
        context = sample[f"{source}_context"][~sample[f"{source}_mask"]]
        assert ((0 <= context) & (context <= 1)).all()
    anchor = sample["metadata"]["anchor_position"]
    current = dataset.episode_arrays[0].minute_frame.iloc[:anchor + 1]
    anchor_date = current.iloc[-1].trading_date
    expected = min(1.0, len(current.loc[current.trading_date == anchor_date]) / 512.0)
    assert sample["daily_context"][-1, 4] == pytest.approx(expected)


def test_v11_history_visibility_is_causal_not_role_partitioned():
    store = _make_store()
    store.episode_table.loc[store.episode_table.episode_id == 0, "role"] = "validation"
    config = _dataset_config(); config["history_week"]["capacity"] = 64
    dataset = V11ContractDataset(store, config, role="train")
    current = next(arrays for arrays in dataset.episode_arrays if arrays.episode.episode_id == 2)
    history = dataset._history(current.episode)
    assert history[4][~history[2]].sum() == 3


def test_v11_history_truncation_preserves_real_contract_boundary():
    config = _dataset_config(); config["history_week"]["capacity"] = 12
    dataset = V11ContractDataset(_make_store(), config)
    current = dataset.episode_arrays[2]
    rows = dataset.history_weekly_rows(current.episode).tail(12)
    _, _, mask, _, boundary = dataset._history(current.episode)
    expected = np.r_[True, rows.contract_uid.to_numpy()[1:] != rows.contract_uid.to_numpy()[:-1]]
    np.testing.assert_array_equal(boundary[~mask].astype(bool), expected)


def test_v11_target_never_crosses_contract(contract_dataset):
    for arrays in contract_dataset.episode_arrays:
        last = int(arrays.anchors[-1])
        assert last + max(contract_dataset.horizons) < len(arrays.minute_frame)
        assert arrays.minute_frame.iloc[last:last + max(contract_dataset.horizons) + 1].contract_uid.nunique() == 1


def test_v11_history_weekly_is_causal(contract_dataset):
    arrays = contract_dataset.episode_arrays[2]
    original = tuple(value.copy() for value in contract_dataset._history(arrays.episode))
    future = contract_dataset.store.frames["FG"]["daily"]
    future.loc[future.trading_date >= arrays.episode.main_start, "close"] *= 100
    contract_dataset._history_cache.clear()
    changed = contract_dataset._history(arrays.episode)
    for first, second in zip(original, changed):
        np.testing.assert_array_equal(first, second)


def test_v11_masks_and_padding(contract_dataset):
    sample = contract_dataset[0]
    for source, capacity in (("minute", 16), ("daily", 8), ("current_weekly", 4), ("history_weekly", 8)):
        mask = sample[f"{source}_mask"]
        assert mask.shape == (capacity,) and mask.dtype == torch.bool
        assert torch.equal(mask, torch.sort(mask, descending=True).values)
        assert (sample[f"{source}_market"][mask] == 0).all()
        assert not sample[f"{source}_imc_validity"][mask].any()


def test_v11_daily_origin_fixed_for_contract(contract_dataset):
    arrays = contract_dataset.episode_arrays[0]
    first = contract_dataset[contract_dataset.global_index(0, 0)]
    later = contract_dataset[contract_dataset.global_index(0, 5)]
    expected = float(arrays.minute_frame.iloc[0].open)
    assert arrays.daily_origin.price == expected
    for sample in (first, later):
        first_valid = int(torch.nonzero(~sample["daily_mask"], as_tuple=False)[0, 0])
        assert sample["daily_market"][first_valid, 0] == pytest.approx(0.0, abs=1e-6)


def test_v11_current_weekly_origin_fixed_for_contract(contract_dataset):
    arrays = contract_dataset.episode_arrays[0]
    assert arrays.weekly_origin.price == float(arrays.minute_frame.iloc[0].open)
    first = contract_dataset[contract_dataset.global_index(0, 0)]
    later = contract_dataset[contract_dataset.global_index(0, 20)]
    for sample in (first, later):
        index = int(torch.nonzero(~sample["current_weekly_mask"], as_tuple=False)[0, 0])
        assert sample["current_weekly_market"][index, 0] == pytest.approx(0.0, abs=1e-6)


def test_daily_partial_does_not_see_future_minutes(contract_dataset):
    arrays = contract_dataset.episode_arrays[0]
    item = contract_dataset.global_index(0, 3)
    anchor = int(arrays.anchors[3])
    before = contract_dataset[item]
    arrays.minute_frame.loc[anchor + 1:, [*BAR_COLUMNS]] *= 100
    arrays.minute_bars[anchor + 1:] *= 100
    after = contract_dataset[item]
    torch.testing.assert_close(before["daily_market"], after["daily_market"], rtol=0, atol=0)
    torch.testing.assert_close(before["daily_context"], after["daily_context"], rtol=0, atol=0)


def test_weekly_partial_does_not_see_future_minutes(contract_dataset):
    # Use a fresh fixture because the preceding test intentionally mutates its future.
    dataset = V11ContractDataset(_make_store(), _dataset_config())
    arrays = dataset.episode_arrays[0]
    item = dataset.global_index(0, 3); anchor = int(arrays.anchors[3])
    before = dataset[item]
    arrays.minute_frame.loc[anchor + 1:, [*BAR_COLUMNS]] *= 100
    arrays.minute_bars[anchor + 1:] *= 100
    after = dataset[item]
    torch.testing.assert_close(before["current_weekly_market"], after["current_weekly_market"], rtol=0, atol=0)
    torch.testing.assert_close(before["current_weekly_context"], after["current_weekly_context"], rtol=0, atol=0)


def test_daily_partial_close_parity():
    config = _dataset_config(); config["data"]["anchor_stride"] = 1
    dataset = V11ContractDataset(_make_store(), config)
    arrays = dataset.episode_arrays[0]
    sample = _sample_for_anchor(dataset, 0, 499)
    anchor_date = arrays.minute_frame.iloc[499].trading_date
    cached_sequence = arrays.daily_frame.loc[arrays.daily_frame.trading_date <= anchor_date]
    expected, _ = V11IMCTransform.transform(
        cached_sequence.loc[:, list(BAR_COLUMNS)].to_numpy(dtype=np.float64), origin=arrays.daily_origin,
        prior_volume=arrays.daily_prior_volume, previous_close=arrays.daily_previous_close,
        previous_oi=arrays.daily_previous_oi,
    )
    torch.testing.assert_close(sample["daily_market"][-1], torch.from_numpy(expected[-1]), rtol=1e-5, atol=1e-6)


def test_weekly_partial_close_parity():
    config = _dataset_config(); config["data"]["anchor_stride"] = 1
    dataset = V11ContractDataset(_make_store(), config)
    arrays = dataset.episode_arrays[0]
    sample = _sample_for_anchor(dataset, 0, 499)
    cached_sequence = arrays.current_weekly_frame.iloc[:1]
    expected, _ = V11IMCTransform.transform(
        cached_sequence.loc[:, list(BAR_COLUMNS)].to_numpy(dtype=np.float64), origin=arrays.weekly_origin,
        prior_volume=arrays.weekly_prior_volume, previous_close=arrays.weekly_previous_close,
        previous_oi=arrays.weekly_previous_oi,
    )
    torch.testing.assert_close(sample["current_weekly_market"][-1], torch.from_numpy(expected[-1]), rtol=1e-5, atol=1e-6)


def _intervention(model, batch, field, intermediate):
    changed = deepcopy(batch); changed[field][:, 0] += 10
    with torch.no_grad():
        first = model(**batch, return_intermediates=True)["intermediates"][intermediate]
        second = model(**changed, return_intermediates=True)["intermediates"][intermediate]
    assert not torch.allclose(first, second, rtol=1e-5, atol=1e-6)


def test_v11_history_affects_commodity_state(v11_model):
    _intervention(v11_model, _model_batch(), "history_weekly_market", "commodity_state")


def test_v11_commodity_affects_contract_state(v11_model):
    _intervention(v11_model, _model_batch(), "history_weekly_market", "contract_state")


def test_v11_daily_affects_contract_state(v11_model):
    _intervention(v11_model, _model_batch(), "daily_market", "contract_state")


def test_v11_current_weekly_affects_contract_state(v11_model):
    _intervention(v11_model, _model_batch(), "current_weekly_market", "contract_state")


def test_v11_contract_affects_minute_before_belief(v11_model):
    _intervention(v11_model, _model_batch(), "daily_market", "minute_conditioned_tokens")


def test_v11_commodity_affects_minute_before_belief(v11_model):
    _intervention(v11_model, _model_batch(), "history_weekly_market", "minute_conditioned_tokens")


def test_v11_minute_affects_belief(v11_model):
    _intervention(v11_model, _model_batch(), "minute_market", "final_belief")


def test_v11_higher_scale_ablation_changes_belief(v11_model):
    batch = _model_batch()
    with torch.no_grad():
        regular = v11_model(**batch)["z_market"]
    handle = v11_model.minute_conditioner.read.register_forward_hook(
        lambda module, inputs, output: (torch.zeros_like(output[0]), output[1]))
    try:
        with torch.no_grad():
            ablated = v11_model(**batch)["z_market"]
    finally:
        handle.remove()
    assert not torch.allclose(regular, ablated)


def test_v11_all_masked_source_safety(v11_model):
    batch = _model_batch()
    for source in ("daily", "current_weekly", "history_weekly"):
        batch[f"{source}_mask"][:] = True
        batch[f"{source}_market"][:] = float("nan")
    with torch.no_grad():
        result = v11_model(**batch, return_intermediates=True)
    assert torch.isfinite(result["z_market"]).all()
    assert torch.equal(result["intermediates"]["source_gate_values"]["contract"], torch.zeros(2, 2))


def test_v11_no_commodity_embedding(v11_model):
    forbidden = ("commodity_embedding", "symbol_embedding", "symbol_id", "instrument_embedding")
    texts = [*v11_model.state_dict(), str(inspect.signature(MarketJEPAV11.__init__)), str(inspect.signature(v11_model.forward))]
    assert not any(word in text.lower() for text in texts for word in forbidden)
    assert v11_model.architecture_config["commodity_embedding"] is False


def test_v11_target_market_only(v11_model):
    signature = str(inspect.signature(v11_model.target_minute.forward))
    assert "context" not in signature and "daily" not in signature and "weekly" not in signature
    assert not any("context" in name for name in v11_model.target_minute.state_dict())


def test_v11_future_context_does_not_affect_target(v11_model):
    batch = _model_batch(); changed = deepcopy(batch)
    changed["minute_context"] += 100
    with torch.no_grad():
        first, second = v11_model(**batch), v11_model(**changed)
    for horizon in v11_model.horizons:
        torch.testing.assert_close(first["targets"][horizon], second["targets"][horizon], rtol=0, atol=0)


def test_v11_future_daily_does_not_affect_target(v11_model):
    batch = _model_batch(); changed = deepcopy(batch)
    changed["daily_market"] -= 100; changed["current_weekly_market"] += 100
    with torch.no_grad():
        first, second = v11_model(**batch), v11_model(**changed)
    for horizon in v11_model.horizons:
        torch.testing.assert_close(first["targets"][horizon], second["targets"][horizon], rtol=0, atol=0)


def test_v11_same_origin_future_imc(contract_dataset):
    sample = contract_dataset[0]
    arrays = contract_dataset.episode_arrays[0]
    anchor = sample["metadata"]["anchor_position"]
    start = max(0, anchor - contract_dataset.minute_capacity + 1)
    p0 = arrays.minute_bars[start, 3]
    oi0 = arrays.minute_bars[start, 5]
    future = arrays.minute_bars[anchor + 1:anchor + 17]
    np.testing.assert_allclose(sample["target_minute_market"][16][:, 3], np.log(future[:, 3] / p0), rtol=1e-6)
    np.testing.assert_allclose(sample["target_minute_market"][16][:, 5], np.log(future[:, 5] / oi0), rtol=1e-6)
    assert sample["target_minute_market"][16][0, 3] != 0


def test_v11_target_encoder_frozen(v11_model):
    loss, _ = jepa_loss(v11_model(**_model_batch()))
    loss.backward()
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in v11_model.target_minute.parameters())


def test_v11_ema_update(v11_model):
    before = {name: value.clone() for name, value in v11_model.target_minute.named_parameters()}
    with torch.no_grad():
        for value in v11_model.minute_local.market_core.parameters():
            value.add_(1)
    v11_model.update_target(0.996)
    for name, value in v11_model.target_minute.named_parameters():
        torch.testing.assert_close(value, before[name] + 0.004)


def test_v11_all_online_trainable_params_have_gradient():
    model = MarketJEPAV11(_debug_model_config(), debug=True)
    loss, _ = jepa_loss(model(**_model_batch()))
    loss.backward()
    failures = [name for name, parameter in model.named_parameters()
                if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all())]
    assert failures == []


def test_v11_no_dead_branch():
    model = MarketJEPAV11(_debug_model_config(), debug=True)
    output = model(**_model_batch(), return_intermediates=True)
    output["intermediates"]["minute_conditioned_tokens"].retain_grad()
    loss, _ = jepa_loss(output); loss.backward()
    gradient = output["intermediates"]["minute_conditioned_tokens"].grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_v11_balanced_commodity_sampling():
    dataset = V11ContractDataset(_make_store(TRAIN_COMMODITIES, 2), _dataset_config())
    sampler = HierarchicalCommodityContractSampler(dataset, 50_000, seed=123)
    counts = dict.fromkeys(TRAIN_COMMODITIES, 0)
    for index in sampler:
        episode_index = np.searchsorted(dataset.offsets, index, side="right") - 1
        counts[dataset.episode_arrays[episode_index].episode.commodity] += 1
    expected = len(sampler) / len(TRAIN_COMMODITIES)
    assert all(abs(value - expected) / expected < 0.03 for value in counts.values())


def test_v11_balanced_contract_sampling():
    dataset = V11ContractDataset(_make_store(TRAIN_COMMODITIES, 2), _dataset_config())
    sampler = HierarchicalCommodityContractSampler(dataset, 50_000, seed=124)
    counts = {(commodity, episode): 0 for commodity in TRAIN_COMMODITIES for episode in (0, 1)}
    for index in sampler:
        episode_index = np.searchsorted(dataset.offsets, index, side="right") - 1
        episode = dataset.episode_arrays[episode_index].episode
        counts[(episode.commodity, episode.episode_id)] += 1
    expected = len(sampler) / 10
    assert all(abs(value - expected) / expected < 0.06 for value in counts.values())


def test_v11_sampler_determinism():
    dataset = V11ContractDataset(_make_store(TRAIN_COMMODITIES, 2), _dataset_config())
    first = HierarchicalCommodityContractSampler(dataset, 100, seed=7)
    second = HierarchicalCommodityContractSampler(dataset, 100, seed=7)
    assert list(first) == list(second)
    second.set_epoch(1)
    assert list(first) != list(second)


def test_v11_contract_filtered_if_history_shorter_than_config():
    store = _prepend_reliable_week(_make_store(), {"FG": "2019-03-01"})
    config = _dataset_config(); config["history_week"].update(years=3, require_full_history=True)
    dataset = V11ContractDataset(store, config)
    assert [arrays.episode.episode_id for arrays in dataset.episode_arrays] == [1, 2]
    summary = dataset.history_week_eligibility_summary["FG"]
    assert summary["total_contract_count"] == 3
    assert summary["filtered_insufficient_history_count"] == 1
    assert summary["eligible_contract_count"] == 2


def test_v11_history_years_is_configurable():
    assert DEFAULT_V11_CONFIG["history_week"]["years"] == 3
    assert DEFAULT_V11_CONFIG["history_week"]["commodity_years"] == {"SH": 2}
    assert DEFAULT_V11_CONFIG["history_week"]["series_mode"] == "same_delivery_month"
    formal_two_years = deepcopy(DEFAULT_V11_CONFIG)
    formal_two_years["history_week"]["years"] = 2
    validate_v11_config(formal_two_years)
    store = _prepend_reliable_week(_make_store(), {"FG": "2019-03-01"})
    two_years = _dataset_config(); two_years["history_week"].update(years=2, require_full_history=True)
    three_years = _dataset_config(); three_years["history_week"].update(years=3, require_full_history=True)
    assert V11ContractDataset(store, two_years).history_week_eligibility_summary["FG"]["eligible_contract_count"] == 3
    assert V11ContractDataset(store, three_years).history_week_eligibility_summary["FG"]["eligible_contract_count"] == 2


def test_v11_late_listed_commodity_uses_own_history_start():
    store = _prepend_reliable_week(
        _make_store(("FG", "SH")), {"FG": "2018-01-05", "SH": "2021-01-08"},
    )
    config = _dataset_config(); config["history_week"].update(
        years=3, commodity_years={"SH": 1}, require_full_history=True,
    )
    dataset = V11ContractDataset(store, config)
    assert dataset.history_week_eligibility_summary["FG"]["eligible_contract_count"] == 3
    assert dataset.history_week_eligibility_summary["FG"]["required_history_years"] == 3
    assert dataset.history_week_eligibility_summary["SH"]["eligible_contract_count"] == 2
    assert dataset.history_week_eligibility_summary["SH"]["required_history_years"] == 1


def test_history_eligibility_is_lineage_specific():
    store = _make_fg_lineage_store()
    current_05 = store.episodes[("FG", "FG202605", 5)]
    series_05_contracts = {
        identity.contract_uid for identity in store.contracts.values()
        if identity.series_key == "FG-05"
    }
    weekly = store.frames["FG"]["weekly"]
    store.frames["FG"]["weekly"] = weekly.loc[
        ~weekly.contract_uid.isin(series_05_contracts)
        | (weekly.week_end_date >= current_05.main_start - pd.Timedelta(days=180))
    ].copy()
    config = _dataset_config(); config["history_week"].update(
        years=1, commodity_years={}, require_full_history=True,
    )
    dataset = V11ContractDataset(store, config)
    decisions = {
        record["contract_uid"]: record for record in dataset.history_week_eligibility_records
    }
    assert decisions["FG202601"]["eligible"] is True
    assert decisions["FG202605"]["eligible"] is False
    assert decisions["FG202601"]["series_key"] == "FG-01"
    assert decisions["FG202605"]["series_key"] == "FG-05"
    assert [item.episode.contract_uid for item in dataset.episode_arrays] == ["FG202601"]


def test_v11_sampler_never_selects_ineligible_contract():
    store = _prepend_reliable_week(
        _make_store(TRAIN_COMMODITIES), {commodity: "2019-03-01" for commodity in TRAIN_COMMODITIES},
    )
    config = _dataset_config(); config["history_week"].update(
        years=3, commodity_years={}, require_full_history=True,
    )
    dataset = V11ContractDataset(store, config)
    sampler = HierarchicalCommodityContractSampler(dataset, 10_000, seed=29)
    for index in sampler:
        episode_index = np.searchsorted(dataset.offsets, index, side="right") - 1
        assert dataset.episode_arrays[episode_index].episode.episode_id in (1, 2)


def test_sampler_never_uses_short_lineage_history():
    original = _make_store(TRAIN_COMMODITIES)
    episodes = original.episode_table.copy()
    episodes.loc[
        (episodes.commodity == "FG") & (episodes.episode_id == 0),
        ["delivery_month", "series_key"],
    ] = [5, "FG-05"]
    store = V11DataStore(episodes, original.frames)
    for commodity in TRAIN_COMMODITIES:
        weekly = store.frames[commodity]["weekly"]
        contract_uid = f"{commodity}02" if commodity == "FG" else f"{commodity}01"
        row = weekly.loc[weekly.contract_uid == contract_uid].iloc[0].copy()
        row["week_end_date"] = pd.Timestamp("2020-01-03")
        store.frames[commodity]["weekly"] = pd.concat((pd.DataFrame([row]), weekly), ignore_index=True)
    config = _dataset_config(); config["history_week"].update(
        years=1, commodity_years={}, require_full_history=True,
    )
    dataset = V11ContractDataset(store, config)
    assert ("FG", "FG01", 0) not in set(dataset.eligible_contracts_by_commodity["FG"])
    sampler = HierarchicalCommodityContractSampler(dataset, 10_000, seed=31)
    for index in sampler:
        episode_index = np.searchsorted(dataset.offsets, index, side="right") - 1
        episode = dataset.episode_arrays[episode_index].episode
        assert not (episode.commodity == "FG" and episode.contract_uid == "FG01")


def test_v11_no_short_history_silent_fallback():
    store = _prepend_reliable_week(_make_store(("SH",)), {"SH": "2021-01-08"})
    config = _dataset_config(); config["history_week"].update(years=3, require_full_history=True)
    with pytest.raises(ValueError, match="insufficient history_week coverage"):
        V11ContractDataset(store, config)


def test_v11_collate_keeps_metadata_outside_model(contract_dataset):
    batch = collate_v11_batch([contract_dataset[0], contract_dataset[1]])
    assert isinstance(batch["metadata"], list)
    assert not any(key in inspect.signature(MarketJEPAV11.forward).parameters for key in ("symbol", "contract_uid", "commodity_id"))


class _TrainerDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.scaler = SharedIMCScaler.fit(_scaler_population())
        self.episode_arrays = [
            SimpleNamespace(
                episode=SimpleNamespace(commodity=commodity, key=(commodity, f"X{index}", 0)),
                anchors=np.arange(1),
            )
            for index, commodity in enumerate(TRAIN_COMMODITIES)
        ]
        self.eligible_contracts_by_commodity = {
            commodity: ((commodity, f"X{index}", 0),)
            for index, commodity in enumerate(TRAIN_COMMODITIES)
        }

    @property
    def hierarchy(self):
        return {commodity: [index] for index, commodity in enumerate(TRAIN_COMMODITIES)}

    def global_index(self, episode_index, local_anchor_index):
        assert local_anchor_index == 0
        return episode_index

    def __len__(self):
        return len(self.episode_arrays)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        sample = {}
        lengths = {"minute": 512, "daily": 256, "current_weekly": 64, "history_weekly": 156}
        contexts = {
            "minute": len(MINUTE_CONTEXT_FEATURES), "daily": len(DAILY_CONTEXT_FEATURES),
            "current_weekly": len(WEEKLY_CONTEXT_FEATURES),
            "history_weekly": len(WEEKLY_CONTEXT_FEATURES),
        }
        for source, length in lengths.items():
            sample[f"{source}_market"] = torch.randn(length, len(IMC_FEATURES), generator=generator)
            sample[f"{source}_context"] = torch.randn(length, contexts[source], generator=generator)
            sample[f"{source}_mask"] = torch.zeros(length, dtype=torch.bool)
            sample[f"{source}_imc_validity"] = torch.ones(length, len(IMC_FEATURES), dtype=torch.bool)
        sample["history_weekly_contract_boundary"] = torch.zeros(156)
        sample["history_weekly_contract_boundary"][0] = 1
        sample["target_minute_market"] = {
            h: torch.randn(h, len(IMC_FEATURES), generator=generator) for h in (16, 64, 256)
        }
        sample["target_minute_imc_validity"] = {
            h: torch.ones(h, len(IMC_FEATURES), dtype=torch.bool) for h in (16, 64, 256)
        }
        sample["target_minute_mask"] = {h: torch.zeros(h, dtype=torch.bool) for h in (16, 64, 256)}
        sample["metadata"] = {
            "commodity": TRAIN_COMMODITIES[index], "contract_uid": f"X{index}",
            "daily_was_truncated": True, "daily_truncated_tokens": 3,
        }
        return sample


def _trainer_config(path) -> dict:
    config = deepcopy(DEFAULT_V11_CONFIG)
    config["profile"] = "debug"
    config["model"].update(
        d_model=16, belief_dim=16, num_heads=4, ffn_dim=32, minute_layers=1,
        commodity_state_tokens=2, contract_state_tokens=2, belief_tokens=2,
        predictor_hidden=32, dropout=0.0,
    )
    config["training"].update(
        batch_size=2, gradient_accumulation=1, max_epochs=1, num_workers=0,
        amp=False, checkpoint_dir=str(path),
    )
    return config


@pytest.fixture(scope="module")
def trained_v11_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("v11-checkpoint")
    config = _trainer_config(path)
    dataset = _TrainerDataset()
    model = MarketJEPAV11(config["model"], debug=True)
    trainer = V11Trainer(
        model, config, dataset, torch.device("cpu"),
        validation_dataset=dataset, samples_per_epoch=2, data_manifest_sha256="synthetic-data",
    )
    trainer.fit()
    return path, config, dataset, model, trainer


def test_v11_fixed_budget_checkpoint_only(trained_v11_checkpoint):
    path, _, _, _, trainer = trained_v11_checkpoint
    assert trainer.checkpoint_selection == "fixed_budget_final"
    assert (path / "last.pt").is_file()
    assert not list(path.glob("best*"))
    state = load_v11_checkpoint(path / "last.pt")
    assert state["checkpoint_selection"] == "fixed_budget_final"
    assert state["history"][0]["validation"] is not None
    assert set(EPOCH_FIELDS) <= set(state["history"][0])
    assert state["history"][0]["optimizer_steps_this_epoch"] == 1
    assert state["daily_truncation_count"] == 2
    assert state["daily_truncated_tokens"] == 6
    assert state["model_size"] == "DEBUG"
    assert state["d_model"] == 16
    assert state["num_heads"] == 4
    assert state["ffn_dim"] == 32
    assert state["minute_layers"] == 1
    assert state["predictor_hidden_dim"] == 32
    assert state["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    tampered = deepcopy(state); tampered["checkpoint_selection"] = "validation_h64"
    with pytest.raises(ValueError, match="fixed_budget_final"):
        validate_v11_checkpoint(tampered)
    tampered = deepcopy(state); tampered["model_size"] = "XL"
    with pytest.raises(ValueError, match="model_size mismatch"):
        validate_v11_checkpoint(tampered)
    tampered = deepcopy(state); tampered["trainable_parameter_count"] += 1
    with pytest.raises(ValueError, match="parameter count mismatch"):
        model_from_checkpoint(tampered)
    tampered = deepcopy(state); del tampered["trainable_parameter_count"]
    with pytest.raises(ValueError, match="incomplete.*capacity"):
        validate_v11_checkpoint(tampered)


def test_v11_formal_training_contract_is_frozen():
    config = deepcopy(DEFAULT_V11_CONFIG)
    validate_formal_training_config(config)
    assert config["training"]["batch_size"] == 64
    assert config["training"]["gradient_accumulation"] == 2
    assert config["training"]["num_workers"] == 8
    changed = deepcopy(config); changed["history_week"]["commodity_years"] = {"SH": 3}
    with pytest.raises(ValueError, match="frozen commodity history"):
        validate_formal_training_config(changed)
    changed = deepcopy(config); changed["training"]["batch_size"] = 32
    with pytest.raises(ValueError, match="training configuration mismatch"):
        validate_formal_training_config(changed)
    changed = deepcopy(config); changed["training"]["max_epochs"] = 100
    validate_formal_training_config(changed)
    changed = deepcopy(config); changed["training"]["max_epochs"] = 0
    with pytest.raises(ValueError, match="max_epochs must be a positive integer"):
        validate_formal_training_config(changed)


def test_v11_formal_entrypoint_is_checkpointed_implementation():
    manifest = v11_implementation_manifest()
    assert "train_market_jepa_v1_1.py" in manifest["files"]
    assert "market_jepa_v1_1/formal_training.py" in manifest["files"]


def test_v11_multiworker_runtime_reports_truncation(tmp_path):
    config = _trainer_config(tmp_path)
    config["training"]["num_workers"] = 1
    dataset = _TrainerDataset()
    trainer = V11Trainer(
        MarketJEPAV11(config["model"], debug=True), config, dataset, torch.device("cpu"),
        samples_per_epoch=2, data_manifest_sha256="synthetic-data",
    )
    trainer.fit()
    assert trainer.daily_truncation_count == 2
    assert trainer.daily_truncated_tokens == 6
    state = load_v11_checkpoint(tmp_path / "last.pt")
    assert state["daily_truncation_count"] == 2
    assert state["daily_truncated_tokens"] == 6


def test_v11_batch_sweep_preserves_only_comparable_effective_batch():
    plans = profile_plan((64, 128, 192, 256))
    assert [(item["batch_size"], item["gradient_accumulation"]) for item in plans] == [
        (64, 2), (128, 1), (192, 1), (256, 1),
    ]
    assert [item["effective_batch"] for item in plans] == [128, 128, 192, 256]
    assert [item["formal_effective_batch_128_candidate"] for item in plans] == [True, True, False, False]


def test_v11_train_commodity_population_is_configurable():
    config = deepcopy(DEFAULT_V11_CONFIG)
    config["profile"] = "debug"
    config["data"]["train_commodities"] = ["FG"]
    config["history_week"].update(commodity_years={}, require_full_history=False)
    validate_v11_config(config)
    unscaled = V11ContractDataset(_make_store(("FG",)), config)
    scaler = fit_v11_shared_scaler(unscaled, anchors_per_commodity=2)
    dataset = V11ContractDataset(_make_store(("FG",)), config, scaler=scaler)
    sampler = HierarchicalCommodityContractSampler(dataset, num_samples=10, seed=42)
    assert sampler.commodities == ("FG",)
    assert scaler.fitted_commodities == ("FG",)


def test_v11_eligibility_audit_uses_only_configured_commodities(tmp_path):
    store = _make_store(("FG", "SA"))
    store.episode_table.to_csv(tmp_path / "contract_episodes.csv", index=False)
    commodity_path = tmp_path / "FG"
    commodity_path.mkdir()
    store.frames["FG"]["weekly"].to_csv(commodity_path / "FG_1w.csv", index=False)
    config = deepcopy(DEFAULT_V11_CONFIG)
    config["profile"] = "debug"
    config["data"]["train_commodities"] = ["FG"]
    config["history_week"].update(commodity_years={}, require_full_history=False)
    report = audit_history_week_eligibility(tmp_path, config, tmp_path / "audit")
    assert set(report["commodities"]) == {"FG"}
    assert {record["commodity"] for record in report["contracts"]} == {"FG"}


def test_v11_eligibility_audit_reports_configured_commodity_without_train_episode(tmp_path):
    store = _make_store(("FG",))
    store.episode_table["role"] = "unassigned"
    store.episode_table.to_csv(tmp_path / "contract_episodes.csv", index=False)
    commodity_path = tmp_path / "FG"
    commodity_path.mkdir()
    store.frames["FG"]["weekly"].iloc[0:0].to_csv(commodity_path / "FG_1w.csv", index=False)
    config = deepcopy(DEFAULT_V11_CONFIG)
    config["profile"] = "debug"
    config["data"]["train_commodities"] = ["FG"]
    config["history_week"].update(commodity_years={}, require_full_history=False)

    report = audit_history_week_eligibility(tmp_path, config, tmp_path / "audit")

    assert report["status"] == "BLOCKED"
    assert report["commodities"]["FG"]["eligible_contract_count"] == 0
    assert report["commodities"]["FG"]["reason"] == "no train contract episodes"

    overridden = audit_history_week_eligibility(
        tmp_path, config, tmp_path / "overridden-audit", episode_role="train",
    )
    assert overridden["status"] == "PASS"
    assert overridden["commodities"]["FG"]["eligible_contract_count"] == 3
    assert overridden["episode_role_override"] == "train"


def test_v11_data_store_can_override_selected_episode_roles_in_memory(tmp_path):
    store = _make_store(("FG",))
    episodes = store.episode_table.copy()
    episodes["role"] = "unassigned"
    episodes.to_csv(tmp_path / "contract_episodes.csv", index=False)
    commodity_path = tmp_path / "FG"
    commodity_path.mkdir()
    for scale, suffix in (("minute", "1m"), ("daily", "1d"), ("weekly", "1w")):
        store.frames["FG"][scale].to_csv(commodity_path / f"FG_{suffix}.csv", index=False)

    loaded = V11DataStore.from_directory(tmp_path, ("FG",), episode_role="train")

    assert set(loaded.episode_table["role"]) == {"train"}
    assert set(pd.read_csv(tmp_path / "contract_episodes.csv")["role"]) == {"unassigned"}


def test_v11_formal_history_filter_removes_only_zero_eligibility_commodities():
    config = deepcopy(DEFAULT_V11_CONFIG)
    config["profile"] = "debug"
    config["data"]["train_commodities"] = ["FG", "TC"]
    config["history_week"]["commodity_years"] = {"TC": 1}
    eligibility = {
        "commodities": {
            "FG": {
                "eligible_contract_count": 2, "reason": "", "total_contract_count": 3,
                "filtered_insufficient_history_count": 1,
            },
            "TC": {
                "eligible_contract_count": 0, "reason": "no train contract episodes",
                "total_contract_count": 0, "filtered_insufficient_history_count": 0,
            },
        },
    }

    effective, removed = _filter_train_commodities_by_history(config, eligibility)

    assert effective == ("FG",)
    assert [record["commodity"] for record in removed] == ["TC"]
    assert config["data"]["train_commodities"] == ["FG"]
    assert config["history_week"]["commodity_years"] == {}


def test_v11_checkpoint_roundtrip(trained_v11_checkpoint):
    path, _, _, model, _ = trained_v11_checkpoint
    state = load_v11_checkpoint(path / "last.pt")
    restored = model_from_checkpoint(state).eval()
    model.eval()
    batch = collate_v11_batch([_TrainerDataset()[0], _TrainerDataset()[1]])
    kwargs = {key: value for key, value in batch.items() if key != "metadata"}
    with torch.no_grad():
        expected, actual = model(**kwargs), restored(**kwargs)
    torch.testing.assert_close(expected["z_market"], actual["z_market"], rtol=0, atol=0)
    for branch in ("predictions", "targets"):
        for horizon in (16, 64, 256):
            torch.testing.assert_close(expected[branch][horizon], actual[branch][horizon], rtol=0, atol=0)


def test_v11_resume_roundtrip(trained_v11_checkpoint):
    path, config, dataset, _, trainer = trained_v11_checkpoint
    state = load_v11_checkpoint(path / "last.pt")
    restored_model = MarketJEPAV11(config["model"], debug=True)
    restored = V11Trainer(
        restored_model, config, dataset, torch.device("cpu"),
        validation_dataset=dataset, samples_per_epoch=2, data_manifest_sha256="synthetic-data",
    )
    restored.resume(state)
    assert restored.global_step == trainer.global_step
    assert restored.start_epoch == 1
    assert restored.history == trainer.history
    assert restored.sampler.state_dict() == trainer.sampler.state_dict()


def test_v11_formal_amp_dtype_is_bfloat16():
    config = deepcopy(DEFAULT_V11_CONFIG)
    assert config["training"]["amp"] is True
    assert config["training"]["amp_dtype"] == "bfloat16"
    validate_v11_config(config)
    validate_formal_training_config(config)
    changed = deepcopy(config)
    changed["training"]["amp_dtype"] = "float16"
    with pytest.raises(ValueError, match="training configuration mismatch"):
        validate_formal_training_config(changed)


def test_v11_jepa_loss_fp32_keeps_fp32_reductions_and_gradients():
    from market_jepa_v1_1.training import jepa_loss_fp32

    generator = torch.Generator().manual_seed(20260907)
    predictions = {
        h: torch.randn(4, 16, generator=generator, dtype=torch.float16).requires_grad_()
        for h in (16, 64, 256)
    }
    targets = {
        h: torch.randn(4, 16, generator=generator, dtype=torch.float16)
        for h in (16, 64, 256)
    }
    latent = torch.randn(4, 16, generator=generator, dtype=torch.float16, requires_grad=True)
    loss, metrics = jepa_loss_fp32(
        {"predictions": predictions, "targets": targets, "z_market": latent},
        lambda_var=0.0, lambda_cov=0.0, variance_floor=1.0,
    )
    assert loss.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in metrics.values())
    loss.backward()
    assert latent.grad is None  # formal lambda_var=lambda_cov=0: z_market enters loss through real predictors
    for value in predictions.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()


class _FakeV11Scaler:
    def __init__(self, before: float, after: float):
        self.scale = before
        self.after = after
        self.steps = 0

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        del optimizer
        self.steps += 1

    def update(self):
        self.scale = self.after


class _FakeV11Optimizer:
    def __init__(self):
        self.zero_grad_calls = 0

    def zero_grad(self, *, set_to_none: bool):
        assert set_to_none is True
        self.zero_grad_calls += 1


class _FakeV11Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _FakeV11ModelForStep:
    def __init__(self):
        self.ema_updates = 0

    def update_target(self, tau: float):
        assert tau == 0.996
        self.ema_updates += 1


def _fake_v11_step_trainer(*, after_scale: float):
    trainer = V11Trainer.__new__(V11Trainer)
    trainer.scaler = _FakeV11Scaler(1024.0, after_scale)
    trainer.optimizer = _FakeV11Optimizer()
    trainer.scheduler = _FakeV11Scheduler()
    trainer.model = _FakeV11ModelForStep()
    trainer.config = {"training": {"ema_tau": 0.996}}
    trainer.grad_scaler_enabled = True
    trainer.global_step = 0
    trainer.skipped_optimizer_steps = 0
    trainer.consecutive_amp_overflows = 0
    trainer.max_consecutive_amp_overflows = 8
    return trainer


def test_v11_float16_amp_overflow_skips_optimizer_dependent_state():
    trainer = _fake_v11_step_trainer(after_scale=512.0)
    assert trainer._finish_optimizer_step() is False
    assert trainer.skipped_optimizer_steps == 1
    assert trainer.consecutive_amp_overflows == 1
    assert trainer.global_step == 0
    assert trainer.scheduler.steps == 0
    assert trainer.model.ema_updates == 0
    assert trainer.optimizer.zero_grad_calls == 1


def test_v11_float16_amp_success_advances_optimizer_dependent_state():
    trainer = _fake_v11_step_trainer(after_scale=1024.0)
    assert trainer._finish_optimizer_step() is True
    assert trainer.skipped_optimizer_steps == 0
    assert trainer.consecutive_amp_overflows == 0
    assert trainer.global_step == 1
    assert trainer.scheduler.steps == 1
    assert trainer.model.ema_updates == 1


def test_v11_repeated_float16_amp_overflow_hard_fails():
    trainer = _fake_v11_step_trainer(after_scale=512.0)
    trainer.consecutive_amp_overflows = 7
    with pytest.raises(FloatingPointError, match="8 consecutive"):
        trainer._finish_optimizer_step()
    assert trainer.global_step == 0
    assert trainer.scheduler.steps == 0
    assert trainer.model.ema_updates == 0


def test_v11_finite_gradient_norm_reduction_overflow_uses_fp64_fallback(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])

    class Model:
        def optimizer_parameters(self):
            yield parameter

    trainer = V11Trainer.__new__(V11Trainer)
    trainer.model = Model()
    trainer.config = {"training": {"gradient_clip_norm": 1.0}}
    trainer.grad_scaler_enabled = False
    trainer.amp_dtype_name = "bfloat16"

    monkeypatch.setattr(
        torch.nn.utils, "get_total_norm",
        lambda *args, **kwargs: torch.tensor(float("inf")),
    )
    assert trainer._clip_gradients_or_skip() is True
    torch.testing.assert_close(parameter.grad.norm(), torch.tensor(1.0), rtol=1e-5, atol=1e-6)


def test_v11_nonfinite_gradient_is_hard_failure_without_grad_scaler():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float("nan"), 1.0])

    class Model:
        def optimizer_parameters(self):
            yield parameter

    trainer = V11Trainer.__new__(V11Trainer)
    trainer.model = Model()
    trainer.config = {"training": {"gradient_clip_norm": 1.0}}
    trainer.grad_scaler_enabled = False
    trainer.amp_dtype_name = "bfloat16"
    with pytest.raises(FloatingPointError, match="non-finite gradient elements"):
        trainer._clip_gradients_or_skip()


def test_v11_nonfinite_gradient_is_deferred_to_float16_grad_scaler():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float("inf"), 1.0])

    class Model:
        def optimizer_parameters(self):
            yield parameter

    trainer = V11Trainer.__new__(V11Trainer)
    trainer.model = Model()
    trainer.config = {"training": {"gradient_clip_norm": 1.0}}
    trainer.grad_scaler_enabled = True
    trainer.amp_dtype_name = "float16"
    assert trainer._clip_gradients_or_skip() is False


def test_v11_trainer_emits_successful_optimizer_step_metrics(tmp_path):
    class StepRecorder:
        active = True

        def __init__(self):
            self.values = []

        def log_step(self, **value):
            self.values.append(value)

    config = _trainer_config(tmp_path)
    dataset = _TrainerDataset()
    recorder = StepRecorder()
    trainer = V11Trainer(
        MarketJEPAV11(config["model"], debug=True), config, dataset, torch.device("cpu"),
        samples_per_epoch=2, data_manifest_sha256="synthetic-data",
        step_logger=recorder, wandb_run_id="tracking-id",
    )
    history = trainer.fit()
    assert len(recorder.values) == 1
    step = recorder.values[0]
    assert step["global_step"] == 1
    assert step["samples"] == 2
    assert step["loss"] == pytest.approx(history[0]["train_loss"])
    for horizon in (16, 64, 256):
        assert step[f"h{horizon}_loss"] == pytest.approx(
            history[0][f"prediction_loss_h{horizon}"]
        )
    state = load_v11_checkpoint(tmp_path / "last.pt")
    assert state["wandb_run_id"] == "tracking-id"
