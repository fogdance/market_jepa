from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from export_frozen_context import ALLOWED_SPLITS, FrozenContextDataset, export_context_split
from market_jepa.data import CONTEXT_FEATURES, MARKET_FEATURES
from market_jepa.eval.metrics import block_bootstrap
from market_jepa.frozen_state import (
    PredictiveStateHead,
    RFFMap,
    Standardizer,
    _weekly_blocks,
    evaluate_gates,
    fit_full_train_head,
    inner_fit_dev_indices,
    nearest_euclidean_indices,
    select_epoch_budget,
)
from market_jepa.model import MarketJEPA


def _small_model(config) -> MarketJEPA:
    model_config = deepcopy(config["model"])
    model_config.update(
        minute_d_model=16,
        latent_dim=16,
        minute_layers=1,
        minute_heads=4,
        minute_ffn_dim=32,
        time_hidden=4,
        daily_hidden=8,
        weekly_hidden=8,
        recurrent_layers=1,
        fusion_hidden=32,
        predictor_hidden=32,
        dropout=0.0,
    )
    return MarketJEPA(
        len(MARKET_FEATURES),
        len(CONTEXT_FEATURES),
        config["data"]["horizons"],
        model_config,
    )


def test_frozen_context_export_cannot_access_test() -> None:
    assert ALLOWED_SPLITS == ("train", "validation")
    with pytest.raises(ValueError, match="only Train and Validation"):
        FrozenContextDataset(None, {}, "test")


def test_context_export_keeps_jepa_parameters_bitwise_unchanged(
    tmp_path, causal_data, causal_config
) -> None:
    model = _small_model(causal_config)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    dataset = FrozenContextDataset(causal_data, causal_config, "train")
    path = export_context_split(
        model,
        dataset,
        tmp_path / "context_train.npz",
        torch.device("cpu"),
        batch_size=2,
        metadata={"split": "train"},
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    with np.load(path, allow_pickle=False) as exported:
        assert exported["z_context"].shape[1] == 4
        assert exported["context_progress"].shape[1] == len(CONTEXT_FEATURES) + 4
        assert np.allclose(
            exported["context_progress"][0, : len(CONTEXT_FEATURES)],
            causal_data.minute_context_raw[dataset.indices[0]],
        )


def test_rff_and_train_pair_sampling_are_deterministic() -> None:
    train = np.random.default_rng(7).normal(size=(100, 12)).astype(np.float32)
    first, first_left, first_right = RFFMap.fit(train)
    second, second_left, second_right = RFFMap.fit(train)
    assert first.sigma0 == second.sigma0
    assert np.array_equal(first.bandwidths, second.bandwidths)
    assert np.array_equal(first.frequencies, second.frequencies)
    assert np.array_equal(first.phases, second.phases)
    assert np.array_equal(first_left, second_left)
    assert np.array_equal(first_right, second_right)
    assert np.all(first_left != first_right)


def test_scaler_and_kernel_fit_are_train_only() -> None:
    train = np.random.default_rng(8).normal(size=(100, 12)).astype(np.float32)
    validation_a = np.zeros((20, 12), dtype=np.float32)
    validation_b = np.full((20, 12), 1e6, dtype=np.float32)
    scaler_a = Standardizer.fit(train)
    rff_a, _, _ = RFFMap.fit(scaler_a.transform(train))
    scaler_b = Standardizer.fit(train)
    rff_b, _, _ = RFFMap.fit(scaler_b.transform(train))
    assert not np.array_equal(validation_a, validation_b)
    assert np.array_equal(scaler_a.mean, scaler_b.mean)
    assert np.array_equal(scaler_a.std, scaler_b.std)
    assert rff_a.sigma0 == rff_b.sigma0
    assert np.array_equal(rff_a.frequencies, rff_b.frequencies)


def test_inner_selection_excludes_formal_validation_years() -> None:
    days = np.asarray(
        ["2018-01-02", "2021-12-31", "2022-01-03", "2022-12-30", "2023-01-03"],
        dtype="datetime64[ns]",
    ).astype(np.int64)
    fit, dev = inner_fit_dev_indices(days)
    assert np.array_equal(fit, np.asarray([0, 1]))
    assert np.array_equal(dev, np.asarray([2, 3]))
    assert 4 not in fit and 4 not in dev


def test_state_neighbor_distance_is_euclidean_not_cosine() -> None:
    reference = np.asarray([[1.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    query = np.asarray([[9.0, 0.0]], dtype=np.float32)
    indices = nearest_euclidean_indices(
        reference,
        query,
        np.asarray([1, 2]),
        np.asarray([3]),
        k=1,
        device=torch.device("cpu"),
    )
    assert indices.tolist() == [[1]]


def test_trading_week_bootstrap_preserves_sample_weighted_estimand() -> None:
    days = np.asarray(
        ["2024-01-02"] * 100 + ["2024-01-09"], dtype="datetime64[ns]"
    ).astype(np.int64)
    effects = np.concatenate([np.ones(100), np.asarray([-10.0])])
    blocks = _weekly_blocks(days)
    result = block_bootstrap(effects, blocks, samples=10_000, seed=42)
    assert len(np.unique(blocks)) == 2
    assert result["effect"] == pytest.approx(effects.mean())
    assert result["effect"] == pytest.approx(0.8910891089108911)


def test_predictive_state_head_has_no_output_normalization() -> None:
    head = PredictiveStateHead(16)
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in head.modules())
    output = head(torch.zeros(2, 16))
    assert output.shape == (2, 3072)


def test_train_only_epoch_selection_and_gate_smoke(tmp_path, monkeypatch) -> None:
    import market_jepa.frozen_state as frozen_state

    monkeypatch.setattr(frozen_state, "PAIR_SAMPLES", 100)
    monkeypatch.setattr(frozen_state, "RFF_PER_BANDWIDTH", 8)
    monkeypatch.setattr(frozen_state, "HEAD_BATCH_SIZE", 4)
    monkeypatch.setattr(frozen_state, "HEAD_MAX_EPOCHS", 2)
    monkeypatch.setattr(frozen_state, "K_NEIGHBORS", 2)
    monkeypatch.setattr(frozen_state, "BOOTSTRAP_SAMPLES", 100)
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(20, 3)).astype(np.float32)
    train_y = rng.normal(size=(20, 12)).astype(np.float32)
    days = np.asarray(
        ["2018-01-02"] * 4
        + ["2019-01-02"] * 4
        + ["2020-01-02"] * 4
        + ["2021-01-03"] * 4
        + ["2022-01-03"] * 4,
        dtype="datetime64[ns]",
    ).astype(np.int64)
    rff, _, _ = RFFMap.fit(train_y)
    selected, history = select_epoch_budget(
        train_x, train_y, days, rff, torch.device("cpu")
    )
    assert selected in (0, 1)
    assert len(history) == 2
    head = fit_full_train_head(
        train_x, train_y, rff, selected, torch.device("cpu")
    )
    assert head(torch.from_numpy(train_x[:2])).shape == (2, 24)

    validation_y = rng.normal(size=(4, 12)).astype(np.float32)
    train_state = rng.normal(size=(20, 24)).astype(np.float32)
    validation_state = rng.normal(size=(4, 24)).astype(np.float32)
    train_context = rng.normal(size=(20, 24)).astype(np.float32)
    validation_context = rng.normal(size=(4, 24)).astype(np.float32)
    result = evaluate_gates(
        train_y,
        validation_y,
        train_state,
        validation_state,
        train_context,
        validation_context,
        np.arange(20),
        np.arange(100, 104),
        np.asarray(
            ["2023-01-03", "2023-01-04", "2023-01-10", "2023-01-11"],
            dtype="datetime64[ns]",
        ).astype(np.int64),
        rff,
        tmp_path,
        torch.device("cpu"),
    )
    assert result["gate2"]["distance"] == "euclidean"
    assert result["test_consumed"] is False
