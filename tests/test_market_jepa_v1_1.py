from __future__ import annotations

import inspect
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from market_jepa.model import jepa_loss
from market_jepa_v1_1 import DEFAULT_V11_CONFIG, MarketJEPAV11
from market_jepa_v1_1.config import (
    DAILY_CONTEXT_FEATURES, IMC_FEATURES, MINUTE_CONTEXT_FEATURES,
    TRAIN_COMMODITIES, WEEKLY_CONTEXT_FEATURES,
)
from market_jepa_v1_1.dataset import (
    BAR_COLUMNS, V11ContractDataset, V11DataStore, collate_v11_batch,
)
from market_jepa_v1_1.imc import IMCOrigin, SharedIMCScaler, V11IMCTransform
from market_jepa_v1_1.sampler import HierarchicalCommodityContractSampler
from market_jepa_v1_1.checkpoint import load_v11_checkpoint, model_from_checkpoint, validate_v11_checkpoint
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
                "episode_id": contract_index, "main_start_date": days[0],
                "main_end_date": days[7], "anchor_end_date": days[9], "role": "train",
            })
        frames[commodity] = {
            "minute": pd.concat(minute_parts, ignore_index=True),
            "daily": pd.concat(daily_parts, ignore_index=True),
            "weekly": pd.concat(weekly_parts, ignore_index=True),
        }
    return V11DataStore(pd.DataFrame(episode_rows), frames)


def _dataset_config() -> dict:
    value = deepcopy(DEFAULT_V11_CONFIG)
    value["data"].update(
        minute_capacity=16, daily_capacity=8, current_weekly_capacity=4,
        history_weekly_capacity=8, anchor_stride=17,
    )
    return value


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
    with pytest.raises(ValueError, match="all Train commodities"):
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


def test_v11_history_weekly_boundary_flags(contract_dataset):
    arrays = contract_dataset.episode_arrays[2]
    market, context, mask, validity, boundary = contract_dataset._history(arrays.episode)
    assert market.shape == validity.shape == (8, len(IMC_FEATURES))
    assert boundary[~mask].sum() == 2
    assert np.all(boundary[mask] == 0)
    assert np.all(context[~mask, 2] == 0)


def test_v11_history_weekly_resets_on_contract_boundary(contract_dataset):
    sample = contract_dataset[contract_dataset.global_index(2, 0)]
    valid = ~sample["history_weekly_mask"]
    boundary = sample["history_weekly_contract_boundary"].bool() & valid
    # Each segment begins at its own first weekly close/OI origin.
    assert boundary.sum() == 2
    torch.testing.assert_close(sample["history_weekly_market"][boundary, 3], torch.zeros(2))
    torch.testing.assert_close(sample["history_weekly_market"][boundary, 5], torch.zeros(2))


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


def test_v11_collate_keeps_metadata_outside_model(contract_dataset):
    batch = collate_v11_batch([contract_dataset[0], contract_dataset[1]])
    assert isinstance(batch["metadata"], list)
    assert not any(key in inspect.signature(MarketJEPAV11.forward).parameters for key in ("symbol", "contract_uid", "commodity_id"))


class _TrainerDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.scaler = SharedIMCScaler.fit(_scaler_population())
        self.daily_truncation_count = 0
        self.daily_truncated_tokens = 0
        self.episode_arrays = [
            SimpleNamespace(episode=SimpleNamespace(commodity=commodity), anchors=np.arange(1))
            for commodity in TRAIN_COMMODITIES
        ]

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
        sample["metadata"] = {"commodity": TRAIN_COMMODITIES[index], "contract_uid": f"X{index}"}
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
    tampered = deepcopy(state); tampered["checkpoint_selection"] = "validation_h64"
    with pytest.raises(ValueError, match="fixed_budget_final"):
        validate_v11_checkpoint(tampered)


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
