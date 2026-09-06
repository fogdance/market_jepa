from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from market_jepa.data import MarketDataset, collate_market_batch, prepare_market_data
from market_jepa.train.checkpoint import save_checkpoint
from market_jepa.train.trainer import configure_determinism
from market_jepa_v1 import DEFAULT_V1_CONFIG, MarketJEPAV1, validate_v1_config
from market_jepa_v1.checkpoint import load_v1_checkpoint, model_from_checkpoint, validate_v1_checkpoint
from market_jepa_v1.data import adapt_market_batch
from market_jepa_v1.training import V1Trainer


@pytest.fixture
def integration(tmp_path):
    rows = []
    for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
        for minute in range(300):
            index = len(rows)
            stamp = pd.Timestamp(f"{day} 09:00") + pd.Timedelta(minutes=minute)
            close = 100 + 0.003 * index + 0.03 * np.sin(index / 7)
            rows.append((stamp, close - 0.02, close + 0.1, close - 0.1, close, 10 + index % 31, 1000 + index))
    csv_path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"]).to_csv(csv_path, index=False)
    config = deepcopy(DEFAULT_V1_CONFIG)
    config["profile"] = "debug"
    config["model"].update(d_model=16, belief_dim=16, num_heads=4, ffn_dim=32, minute_layers=1,
                           daily_gru_hidden=8, weekly_gru_hidden=8, predictor_hidden=32, dropout=0.15)
    config["data"].update(csv_path=str(csv_path), minute_context_length=256,
                          splits={"train": ["2024-01-02", "2024-01-03"],
                                  "validation": ["2024-01-04", "2024-01-04"], "test": ["2024-01-05", "2024-01-05"]})
    config["training"].update(batch_size=2, gradient_accumulation=1, max_epochs=2, amp=False,
                              checkpoint_dir=str(tmp_path / "checkpoints"))
    data = prepare_market_data(config)
    dataset = MarketDataset(data, config, "train")
    train = MarketDataset(data, config, "train", dataset.indices[[0, 10, 20, 30]])
    return config, data, train


def trainer_for(config, train, validation=None):
    return V1Trainer(MarketJEPAV1(config["model"], debug=True), config, train, "synthetic_source", {}, torch.device("cpu"),
                     validation_dataset=validation)


def test_market_jepa_v1_checkpoint_roundtrip(integration):
    config, _, train = integration
    configure_determinism(44)
    trainer = trainer_for(config, train)
    history = trainer.fit(stop_before_epoch=1)
    assert history[0]["validation"] is None
    assert not (trainer.checkpoint_dir / "best.pt").exists()
    state = load_v1_checkpoint(trainer.checkpoint_dir / "last.pt")
    restored = model_from_checkpoint(state).eval()
    trainer.model.eval()
    batch = adapt_market_batch(collate_market_batch([train[0], train[1]]), config["model"])
    with torch.no_grad():
        expected, actual = trainer.model(batch), restored(batch)
    torch.testing.assert_close(expected["z_market"], actual["z_market"], rtol=0, atol=0)
    for h in (16, 64, 256):
        for key in ("targets", "predictions"):
            torch.testing.assert_close(expected[key][h], actual[key][h], rtol=0, atol=0)
    assert state["design_version"] == "1.0"
    assert state["num_state_tokens"] == 8 and state["cross_scale_rounds"] == 2
    assert state["global_step"] == 2 and state["optimizer"]["state"]
    for key in ("architecture_config", "feature_dimensions", "rng_state", "v1_implementation_manifest"):
        assert key in state
    wrong = deepcopy(state)
    wrong["design_version"] = "0.6.1"
    path = trainer.checkpoint_dir / "wrong.pt"
    save_checkpoint(wrong, path)
    with pytest.raises(ValueError, match="design_version"):
        load_v1_checkpoint(path)
    wrong = deepcopy(state)
    wrong["feature_dimensions"]["minute_market_dim"] = 7
    with pytest.raises(ValueError, match="feature dimensions"):
        validate_v1_checkpoint(wrong)
    wrong = deepcopy(state)
    wrong["config"]["design_version"] = "0.6.1"
    with pytest.raises(ValueError, match="training config"):
        validate_v1_checkpoint(wrong)
    wrong = deepcopy(state)
    del wrong["model"]["online.state_tokens"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model_from_checkpoint(wrong)
    wrong = deepcopy(state)
    del wrong["optimizer"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_v1_checkpoint(wrong)


def test_market_jepa_v1_resume_preserves_training_trajectory(integration):
    config, _, train = integration
    configure_determinism(42)
    continuous = trainer_for(config, train)
    expected = continuous.fit()
    configure_determinism(42)
    interrupted = trainer_for(config, train)
    interrupted.fit(stop_before_epoch=1)
    state = load_v1_checkpoint(interrupted.checkpoint_dir / "last.pt")
    configure_determinism(42)
    resumed = trainer_for(config, train)
    resumed.resume(state)
    actual = resumed.fit()
    assert expected == actual
    assert continuous.global_step == resumed.global_step == 4
    for key, value in continuous.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    for parameter, values in continuous.optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][parameter][key], rtol=0, atol=0)
    assert not resumed.model.target_minute.training


def test_market_jepa_v1_evaluation_compatibility_on_synthetic_only(integration):
    config, data, train = integration
    base = MarketDataset(data, config, "validation")
    validation = MarketDataset(data, config, "validation", base.indices[:2])
    trainer = trainer_for(config, train, validation)
    result = trainer._epoch(trainer.validation_loader, training=False)
    assert np.isfinite(result["prediction_loss_h64"])
    assert trainer.global_step == 0
    assert trainer.model.encode_persistence(adapt_market_batch(collate_market_batch([validation[0], validation[1]]), config["model"]))[64].shape == (2, 16)


def test_market_jepa_v1_data_integrity_and_adapter(integration):
    config, data, train = integration
    sample = train[0]
    anchor = sample["anchor_index"]
    batch = collate_market_batch([sample, train[1]])
    batch["commodity"] = ["JM", "JM"]
    adapted = adapt_market_batch(batch, config["model"])
    assert "commodity" not in adapted and "outcomes" not in adapted and "anchor_index" not in adapted
    assert (batch["daily_source_max"] <= batch["anchor_index"][:, None]).all()
    assert (batch["weekly_source_max"] <= batch["anchor_index"][:, None]).all()
    for h in (16, 64, 256):
        np.testing.assert_array_equal(sample["targets"][h], data.minute_market[anchor + 1:anchor + h + 1])
        assert data.minute.iloc[anchor + h]["trading_day"] <= pd.Timestamp(config["data"]["splits"]["train"][1])
    original = deepcopy(adapted)
    # Perturb future source rows while preserving the train-fitted normalizers.
    frame = pd.read_csv(config["data"]["csv_path"])
    frame.loc[anchor + 1:, ["Open", "High", "Low", "Close"]] *= 2
    frame.to_csv(config["data"]["csv_path"], index=False)
    changed = prepare_market_data(config, normalizers=data.normalizers)
    changed_train = MarketDataset(changed, config, "train", train.indices)
    changed_batch = adapt_market_batch(collate_market_batch([changed_train[0], changed_train[1]]), config["model"])
    for key in ("minute_market", "minute_context", "daily_market", "daily_context", "weekly_market", "weekly_context"):
        torch.testing.assert_close(original[key][0], changed_batch[key][0], rtol=0, atol=0)
    batch["daily_source_max"][0, 0] = anchor + 1
    with pytest.raises(ValueError, match="future source"):
        adapt_market_batch(batch, config["model"])


def test_market_jepa_v1_config_is_separate_from_frozen_v0():
    from market_jepa.config import validate_config
    with pytest.raises(ValueError, match="0.6.1"):
        validate_config(DEFAULT_V1_CONFIG)
    for name, value in (("cross_scale_rounds", 6), ("d_model", 512), ("num_state_tokens", 16)):
        config = deepcopy(DEFAULT_V1_CONFIG)
        config["model"][name] = value
        with pytest.raises(ValueError):
            validate_v1_config(config)
    config = deepcopy(DEFAULT_V1_CONFIG)
    config["training"]["lambda_var"] = 0.1
    with pytest.raises(ValueError, match="preserves V0"):
        validate_v1_config(config)


def test_market_jepa_v1_period_gru_sequence_is_causal():
    from market_jepa_v1.encoders import PeriodSequenceEncoder
    encoder = PeriodSequenceEncoder(7, 2, 8, 2, 16, 0.0).eval()
    market, context = torch.randn(2, 5, 7), torch.randn(2, 5, 2)
    with torch.no_grad():
        original = encoder(market, context)[0]
        market[:, 3:] += 100
        context[:, 3:] -= 100
        changed = encoder(market, context)[0]
    torch.testing.assert_close(original[:, :3], changed[:, :3], rtol=0, atol=0)
    assert not torch.equal(original[:, 3:], changed[:, 3:])
