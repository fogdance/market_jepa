from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from market_jepa.data import CONTEXT_FEATURES, MARKET_FEATURES
from market_jepa.model import MarketJEPA
from train_frozen_jepa_cdf_state import (
    CONTEXT_DIM,
    HORIZONS,
    STATE_DIM,
    ContextCDFHead,
    FrozenJEPACDFHead,
    binary_targets,
    fit_thresholds,
    load_frozen_v0,
    load_protocol,
    per_sample_bce,
    state_schema,
    train_head,
    unconditional_probabilities,
)


def _v0_model_config() -> dict:
    return {
        "ablation": "minute_daily_weekly",
        "minute_d_model": 8,
        "minute_layers": 1,
        "minute_heads": 2,
        "minute_ffn_dim": 16,
        "time_hidden": 4,
        "daily_hidden": 4,
        "weekly_hidden": 4,
        "recurrent_layers": 1,
        "fusion_hidden": 16,
        "latent_dim": 8,
        "predictor_hidden": 16,
        "dropout": 0.0,
    }


def _training_config(max_epochs: int = 2) -> dict:
    return {
        "model": {"latent_dim": 256, "head_hidden": 256, "context_hidden": 128},
        "training": {
            "seed": 42,
            "learning_rate": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "max_epochs": max_epochs,
            "gradient_clip_norm": 1.0,
        },
    }


def test_v0_encoder_is_fully_frozen_and_eval() -> None:
    model = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), list(HORIZONS), _v0_model_config()
    )
    checkpoint = {"config": {"model": _v0_model_config()}, "model": deepcopy(model.state_dict())}
    frozen = load_frozen_v0(checkpoint, torch.device("cpu"))
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert all(parameter.grad is None for parameter in frozen.parameters())


def test_cdf_head_and_target_shapes_are_exactly_108() -> None:
    y = np.zeros((5, 12), dtype=np.float32)
    thresholds = np.zeros((12, 9), dtype=np.float32)
    target = binary_targets(y, thresholds)
    logits = FrozenJEPACDFHead()(torch.zeros(5, 256))
    assert target.shape == (5, STATE_DIM)
    assert logits.shape == (5, STATE_DIM)
    assert torch.sigmoid(logits).min() >= 0
    assert torch.sigmoid(logits).max() <= 1


def test_head_has_no_dropout_normalization_or_residual_modules() -> None:
    head = FrozenJEPACDFHead()
    assert not any(isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.LayerNorm)) for module in head.modules())
    assert [type(module) for module in head.network] == [torch.nn.Linear, torch.nn.GELU, torch.nn.Linear]


def test_unconditional_baseline_is_exact_deciles_repeated_12_times() -> None:
    expected = np.tile(
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
        12,
    )
    np.testing.assert_array_equal(unconditional_probabilities(), expected)


def test_context_head_accepts_only_fixed_seven_dimensional_context() -> None:
    head = ContextCDFHead()
    assert CONTEXT_DIM == 7
    assert head.network[0].in_features == CONTEXT_DIM
    assert head(torch.zeros(3, CONTEXT_DIM)).shape == (3, STATE_DIM)
    with pytest.raises(RuntimeError):
        head(torch.zeros(3, len(MARKET_FEATURES)))


def test_threshold_metadata_freezes_inner_and_final_fit_ranges() -> None:
    y = np.arange(1200, dtype=np.float32).reshape(100, 12)
    inner, inner_metadata = fit_thresholds(y, "inner_fit", ["2018-01-02", "2021-12-31"])
    final, final_metadata = fit_thresholds(y, "final_train", ["2018-01-02", "2022-12-31"])
    assert inner.shape == final.shape == (12, 9)
    assert inner_metadata["fit_date_range"][1] == "2021-12-31"
    assert final_metadata["fit_date_range"][1] == "2022-12-31"
    assert inner_metadata["method"] == "numpy.quantile(method='linear')"


def test_schema_order_is_horizon_outcome_then_quantile() -> None:
    schema = state_schema()
    assert len(schema) == STATE_DIM
    assert schema[0]["name"] == "H16_Return_q10"
    assert schema[8]["name"] == "H16_Return_q90"
    assert schema[9]["name"] == "H16_MFE_q10"
    assert schema[-1]["name"] == "H256_RV_q90"


def test_loss_is_unweighted_mean_bce_with_logits() -> None:
    logits = torch.tensor([[0.0] * STATE_DIM, [1.0] * STATE_DIM], requires_grad=True)
    targets = torch.tensor([[0.0] * STATE_DIM, [1.0] * STATE_DIM])
    actual = per_sample_bce(logits, targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(1)
    torch.testing.assert_close(actual, expected)


def test_same_seed_repeats_selected_epoch_and_metrics() -> None:
    rng = np.random.Generator(np.random.PCG64(9001))
    x_fit = rng.normal(size=(32, 256)).astype(np.float32)
    y_fit = rng.normal(size=(32, 12)).astype(np.float32)
    x_dev = rng.normal(size=(16, 256)).astype(np.float32)
    y_dev = rng.normal(size=(16, 12)).astype(np.float32)
    thresholds, _ = fit_thresholds(y_fit, "inner_fit", ["2018-01-02", "2021-12-31"])
    config = _training_config(max_epochs=2)
    left, left_history, left_selection = train_head(
        "jepa", x_fit, y_fit, thresholds, config, torch.device("cpu"),
        x_selection=x_dev, y_selection=y_dev,
    )
    right, right_history, right_selection = train_head(
        "jepa", x_fit, y_fit, thresholds, config, torch.device("cpu"),
        x_selection=x_dev, y_selection=y_dev,
    )
    assert left_selection == right_selection
    assert left_history == right_history
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=0, atol=0)


def test_optimizer_run_updates_only_head_parameters() -> None:
    rng = np.random.Generator(np.random.PCG64(12))
    x = rng.normal(size=(16, 256)).astype(np.float32)
    y = rng.normal(size=(16, 12)).astype(np.float32)
    thresholds, _ = fit_thresholds(y, "final_train", ["2018-01-02", "2022-12-31"])
    head, _, _ = train_head(
        "jepa", x, y, thresholds, _training_config(max_epochs=1), torch.device("cpu"), fixed_epochs=1
    )
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_entrypoint_config_embargoes_2025_and_has_no_test_execution_option() -> None:
    config = load_protocol(Path("configs/market_predictive_state_v0_8_frozen_jepa_cdf.yaml"))
    assert config["data"]["max_trading_day"] == "2024-12-31"
    assert "run_test" not in config
    assert "test" not in config["artifacts"]["output_dir"]
