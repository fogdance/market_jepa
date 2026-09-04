from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.data import prepare_market_data
from market_jepa.data.dataset import future_outcomes
from market_jepa.data.trading_day import load_minute_csv
from market_jepa.frozen_state import PredictiveStateHead as FrozenStateHead
from market_jepa.predictive_state import (
    EndToEndPredictiveState,
    PredictiveStateDataset,
    PredictiveStateHead,
    TargetPreprocessing,
    all_future_outcomes,
    build_population_indices,
    context_only_batch,
    exact_euclidean_indices,
    fit_target_preprocessing,
    identical_models,
    load_protocol_config,
    state_loss,
)
from market_jepa.frozen_state import RFFMap, Standardizer
from market_jepa.train.checkpoint import load_checkpoint, save_checkpoint
from market_jepa.train.predictive_state_trainer import (
    PredictiveStateRun,
    initial_state,
    state_dict_sha256,
)
from market_jepa.train.trainer import configure_determinism


def _small_model_config() -> dict:
    config = deepcopy(DEFAULT_CONFIG["model"])
    config.update(
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
        dropout=0.0,
    )
    return config


def _batch() -> dict[str, torch.Tensor]:
    return {
        "minute_market": torch.randn(2, 8, 14),
        "minute_context": torch.randn(2, 8, 5),
        "daily": torch.randn(2, 4, 15),
        "daily_lengths": torch.tensor([4, 3]),
        "weekly": torch.randn(2, 2, 15),
        "weekly_lengths": torch.tensor([2, 1]),
    }


def test_v07_config_is_frozen_and_has_no_test_execution_option() -> None:
    config = load_protocol_config("configs/market_predictive_state_v0_7.yaml")
    assert config["protocol_version"] == "0.7.1"
    assert config["data"]["populations"]["final_test"] == [
        "2025-01-01",
        "2025-12-02",
    ]
    assert set(config["artifacts"]) == {"checkpoint_dir", "evaluation_dir"}


def test_v07_config_rejects_model_or_runtime_drift(tmp_path) -> None:
    import yaml

    source = load_protocol_config("configs/market_predictive_state_v0_7.yaml")
    for path, value in (
        (("model", "minute_d_model"), 128),
        (("training", "num_workers"), 2),
        (("data", "minute_context_length"), 256),
    ):
        changed = deepcopy(source)
        changed[path[0]][path[1]] = value
        config_path = tmp_path / f"{path[0]}_{path[1]}.yaml"
        config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="differs from frozen protocol"):
            load_protocol_config(config_path)


def test_custom_normalizer_population_and_data_cutoff(causal_config) -> None:
    data = prepare_market_data(
        causal_config,
        normalizer_fit_range=("2024-01-05", "2024-01-08"),
        max_trading_day="2024-01-08",
    )
    expected = data.minute["trading_day"].between("2024-01-05", "2024-01-08")
    assert data.normalizers.minute_market.fit_count == int(expected.sum())
    assert data.minute["trading_day"].max() <= np.datetime64("2024-01-08")


def test_csv_cutoff_is_applied_before_post_cutoff_rows_are_materialized(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "minutes.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume,OpenInterest\n"
        "2024-12-31 14:59:00,100,101,99,100,1,10\n"
        "2024-12-31 21:00:00,100,101,99,100,1,10\n"
        "2025-01-02 09:01:00,100,101,99,100,1,10\n",
        encoding="utf-8",
    )
    observed: dict[str, int | None] = {}
    original = pd.read_csv

    def recording_read_csv(*args, **kwargs):
        observed["nrows"] = kwargs.get("nrows")
        return original(*args, **kwargs)

    monkeypatch.setattr("market_jepa.data.trading_day.pd.read_csv", recording_read_csv)
    frame = load_minute_csv(path, max_trading_day="2024-12-31")
    assert observed["nrows"] == 1
    assert len(frame) == 1


def test_population_target_end_is_row_based_and_inside_boundary(causal_data) -> None:
    indices = build_population_indices(
        causal_data,
        ("2024-01-08", "2024-01-08"),
        context_length=2,
        max_horizon=1,
    )
    assert indices[0] == 2
    assert np.all(indices + 1 < len(causal_data.minute))
    days = causal_data.minute["trading_day"].to_numpy(dtype="datetime64[ns]")
    assert np.all(days[indices + 1] <= np.datetime64("2024-01-08"))


def test_lean_dataset_matches_legacy_model_inputs_without_metadata(causal_data) -> None:
    anchor = 3
    y = np.zeros((1, 12), dtype=np.float32)
    lean_dataset = PredictiveStateDataset(
        causal_data, np.asarray([anchor]), y, context_length=2
    )
    audited_dataset = PredictiveStateDataset(
        causal_data,
        np.asarray([anchor]),
        y,
        context_length=2,
        include_metadata=True,
    )
    assert lean_dataset.daily_partial is audited_dataset.daily_partial
    assert lean_dataset.weekly_partial is audited_dataset.weekly_partial
    lean = lean_dataset[0]
    audited = audited_dataset[0]
    daily_market, daily_context, _ = causal_data.daily_snapshot(anchor)
    weekly_market, weekly_context, _ = causal_data.weekly_snapshot(anchor)
    expected = {
        "minute_market": torch.from_numpy(causal_data.minute_market[2:4]),
        "minute_context": torch.from_numpy(causal_data.minute_context[2:4]),
        "daily": torch.from_numpy(np.concatenate([daily_market, daily_context], axis=1)),
        "weekly": torch.from_numpy(np.concatenate([weekly_market, weekly_context], axis=1)),
        "y": torch.from_numpy(y[0]),
    }
    assert set(lean) == set(expected)
    for name, value in expected.items():
        assert torch.equal(lean[name], value), name
        assert torch.equal(audited[name], value), name
    assert audited["anchor_index"] == anchor
    assert "timestamp_ns" in audited and "trading_day_ns" in audited


def test_vectorized_outcomes_match_frozen_definition(causal_data) -> None:
    calculated = all_future_outcomes(causal_data, horizons=(1, 2))
    for horizon in (1, 2):
        for anchor in range(len(causal_data.minute) - horizon):
            np.testing.assert_array_equal(
                calculated[horizon][anchor], future_outcomes(causal_data, anchor, horizon)
            )


def test_target_preprocessing_is_train_only_and_deterministic(monkeypatch) -> None:
    import market_jepa.frozen_state as frozen

    monkeypatch.setattr(frozen, "PAIR_SAMPLES", 100)
    monkeypatch.setattr(frozen, "RFF_PER_BANDWIDTH", 8)
    train = np.random.default_rng(8).normal(size=(100, 12)).astype(np.float32)
    dev_a = np.zeros((20, 12), dtype=np.float32)
    dev_b = np.full((20, 12), 1e6, dtype=np.float32)
    first = fit_target_preprocessing(train)
    second = fit_target_preprocessing(train)
    assert not np.array_equal(dev_a, dev_b)
    assert first.hashes == second.hashes
    assert first.audit == second.audit
    assert np.array_equal(first.mu_phi, second.mu_phi)
    pair_rng = np.random.Generator(np.random.PCG64(4242))
    left = pair_rng.integers(0, len(train), size=100)
    offset = pair_rng.integers(1, len(train), size=100)
    expected_pairs = np.stack([left, (left + offset) % len(train)], axis=1)
    from market_jepa.predictive_state import array_sha256

    assert first.hashes["pairs"] == array_sha256(expected_pairs)


def test_market_and_context_initial_states_are_bitwise_identical() -> None:
    market, context = identical_models(_small_model_config(), seed=42)
    assert sum(p.numel() for p in market.parameters()) == sum(
        p.numel() for p in context.parameters()
    )
    for name, value in market.state_dict().items():
        assert torch.equal(value, context.state_dict()[name]), name


def test_state_head_replays_the_frozen_seed42_initialization() -> None:
    configure_determinism(42)
    expected = FrozenStateHead(256)
    configure_determinism(42)
    config = _small_model_config()
    config["latent_dim"] = 256
    actual = EndToEndPredictiveState(config).head
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name


def test_independent_runs_replay_the_same_dropout_rng_path() -> None:
    config = _small_model_config()
    config["dropout"] = 0.2
    frozen_state, frozen_hash = initial_state(config, 42)
    outputs = []
    for _ in range(2):
        configure_determinism(42)
        model = EndToEndPredictiveState(config).train()
        model.load_state_dict(frozen_state)
        outputs.append(model(_batch())["state"])
    assert frozen_hash
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_context_control_zeros_only_normalized_market_channels() -> None:
    batch = _batch()
    minute_storage = batch["minute_market"].data_ptr()
    daily_storage = batch["daily"].data_ptr()
    minute_context = batch["minute_context"].clone()
    daily_context = batch["daily"][..., 14:].clone()
    weekly_context = batch["weekly"][..., 14:].clone()
    controlled = context_only_batch(batch)
    assert controlled is batch
    assert controlled["minute_market"].data_ptr() == minute_storage
    assert controlled["daily"].data_ptr() == daily_storage
    assert torch.count_nonzero(controlled["minute_market"]) == 0
    assert torch.count_nonzero(controlled["daily"][..., :14]) == 0
    assert torch.count_nonzero(controlled["weekly"][..., :14]) == 0
    assert torch.equal(controlled["minute_context"], minute_context)
    assert torch.equal(controlled["daily"][..., 14:], daily_context)
    assert torch.equal(controlled["weekly"][..., 14:], weekly_context)
    assert torch.equal(controlled["daily_lengths"], batch["daily_lengths"])


def test_state_loss_is_fp32_squared_euclidean_and_reaches_every_branch() -> None:
    model = EndToEndPredictiveState(_small_model_config())
    output = model(_batch())["state"]
    target = torch.randn_like(output)
    loss, per_sample = state_loss(output.half(), target.half())
    expected = torch.square(output.half().float() - target.half().float()).sum(dim=-1)
    assert loss.dtype == torch.float32
    assert per_sample.dtype == torch.float32
    torch.testing.assert_close(per_sample, expected)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.head.modules())
    assert isinstance(model.head, PredictiveStateHead)


def test_exact_knn_matches_brute_force_and_uses_anchor_tie_rule() -> None:
    reference = np.asarray([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    query = np.asarray([[0.0, 0.0], [2.2, 0.0]], dtype=np.float32)
    anchors = np.asarray([20, 10, 30, 40], dtype=np.int64)
    actual = exact_euclidean_indices(
        reference,
        query,
        anchors,
        np.asarray([1, 2, 3, 4]),
        np.asarray([10, 11]),
        k=2,
        device=torch.device("cpu"),
        query_chunk_size=1,
    )
    brute = []
    for row in query:
        distance = np.square(reference - row).sum(axis=1)
        brute.append(np.lexsort((anchors, distance))[:2])
    assert np.array_equal(actual, np.asarray(brute))
    assert actual[0].tolist() == [1, 0]


def test_week_bootstrap_keeps_the_sample_weighted_estimand() -> None:
    from market_jepa.predictive_state import _bootstrap_with_blocks, _weekly_blocks

    effects = np.concatenate([np.ones(100), np.asarray([-10.0])])
    trading_days = np.concatenate(
        [
            np.full(100, np.datetime64("2024-01-02"), dtype="datetime64[ns]"),
            np.asarray([np.datetime64("2024-01-09")], dtype="datetime64[ns]"),
        ]
    )
    result = _bootstrap_with_blocks(effects, _weekly_blocks(trading_days))
    assert result["effect"] == pytest.approx(effects.mean())
    assert result["blocks"] == 2


def test_final_run_rejects_checkpoint_selection_dataset() -> None:
    from market_jepa.train.predictive_state_trainer import PredictiveStateRun, schedule_factor

    # A selected 23-epoch Final budget must follow the first 23 epochs of the
    # same 100-epoch schedule, not a compressed 23-epoch cosine.
    step = 22 * 10
    assert schedule_factor(step, 10, horizon_epochs=100) != schedule_factor(
        step, 10, horizon_epochs=23
    )

    # Constructor validation happens before model/data access.
    with pytest.raises(ValueError, match="Final run cannot"):
        PredictiveStateRun(
            run_name="market",
            stage="final",
            context_only=False,
            model_config={},
            training_config={},
            train_dataset=None,
            dev_dataset=object(),
            target=None,
            normalizers=None,
            initial_model_state={},
            initial_model_hash="",
            output_dir=None,
            source_path=None,
            source_sha256="",
            protocol_path=None,
            protocol_sha256="",
            config={},
            epochs_to_run=1,
            device=torch.device("cpu"),
        )


def test_predictive_state_run_resume_matches_continuous_training(
    causal_data, causal_config, tmp_path
) -> None:
    rng = np.random.default_rng(7)
    rff = RFFMap(
        sigma0=1.0,
        bandwidths=np.asarray([0.5, 1.0, 2.0]),
        frequencies=rng.normal(size=(3, 12, 1024)).astype(np.float32),
        phases=rng.uniform(0, 2 * np.pi, size=(3, 1024)).astype(np.float32),
    )
    target = TargetPreprocessing(
        scaler=Standardizer(np.zeros(12), np.ones(12)),
        rff=rff,
        mu_phi=np.zeros(3072),
        audit={},
        hashes={},
    )
    indices = np.asarray([2, 3, 4, 5], dtype=np.int64)
    dataset = PredictiveStateDataset(
        causal_data,
        indices,
        rng.normal(size=(len(indices), 12)).astype(np.float32),
        context_length=2,
    )
    model_config = _small_model_config()
    training = {
        "seed": 42,
        "microbatch_size": 2,
        "gradient_accumulation": 1,
        "gradient_clip_norm": 1.0,
        "amp": False,
        "warmup_ratio": 0.05,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "encoder_learning_rate": 3e-4,
        "encoder_weight_decay": 0.05,
        "head_learning_rate": 1e-3,
        "head_weight_decay": 1e-4,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    frozen, frozen_hash = initial_state(model_config, 42)

    def make_run(directory):
        return PredictiveStateRun(
            run_name="market",
            stage="final",
            context_only=False,
            model_config=model_config,
            training_config=training,
            train_dataset=dataset,
            dev_dataset=None,
            target=target,
            normalizers=causal_data.normalizers,
            initial_model_state=frozen,
            initial_model_hash=frozen_hash,
            output_dir=directory,
            source_path=causal_config["data"]["csv_path"],
            source_sha256="synthetic-source",
            protocol_path=tmp_path / "protocol.md",
            protocol_sha256="synthetic-protocol",
            config={"test": True},
            epochs_to_run=2,
            device=torch.device("cpu"),
            run_metadata={"test": True},
        )

    continuous = make_run(tmp_path / "continuous")
    continuous.fit()
    continuous_hash = state_dict_sha256(continuous.model.state_dict())

    interrupted = make_run(tmp_path / "resumed")
    interrupted.sampler.set_epoch(0)
    train_loss = interrupted._epoch(interrupted.train_loader, training=True)
    interrupted.history.append(
        {
            "epoch": 0,
            "train_state_mse": train_loss,
            "inner_dev_state_mse": None,
            "elapsed_seconds": 0.0,
        }
    )
    save_checkpoint(interrupted._checkpoint(0), interrupted.last_path)
    resumed = make_run(tmp_path / "resumed")
    resumed.fit(resume=True)

    assert state_dict_sha256(resumed.model.state_dict()) == continuous_hash
    checkpoint = load_checkpoint(resumed.last_path)
    assert checkpoint["sampler_epoch"] == checkpoint["epoch"] == 1
