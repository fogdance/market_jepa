from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from export_market_latents import export_split
from market_jepa.config import DEFAULT_CONFIG
from market_jepa.data import (
    CONTEXT_FEATURES,
    MARKET_FEATURES,
    MarketDataset,
    prepare_market_data,
    run_preflight,
)
from market_jepa.eval.pipeline import evaluate_exports
from market_jepa.model import MarketJEPA
from market_jepa.train import Trainer, configure_determinism, load_checkpoint
from market_jepa.train.checkpoint import sha256


def _write_synthetic_csv(path: Path) -> None:
    rows = []
    index = 0
    for date in ("2024-01-02", "2024-01-03", "2024-01-04"):
        for minute in range(20):
            timestamp = pd.Timestamp(f"{date} 09:01") + pd.Timedelta(minutes=minute)
            close = 100.0 + 0.05 * index + 0.02 * np.sin(index)
            rows.append(
                (timestamp, close - 0.02, close + 0.05, close - 0.05, close, 10 + index, 1_000 + index)
            )
            index += 1
    pd.DataFrame(
        rows,
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"],
    ).to_csv(path, index=False)


def _integration_config(csv_path: Path, checkpoint_dir: Path) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["experiment_id"] = "synthetic_integration"
    config["data"].update(
        csv_path=str(csv_path),
        minute_context_length=4,
        horizons=[1, 2, 4],
        realized_vol_window=3,
        splits={
            "train": ["2024-01-02", "2024-01-02"],
            "validation": ["2024-01-03", "2024-01-03"],
            "test": ["2024-01-04", "2024-01-04"],
        },
    )
    config["model"].update(
        minute_d_model=8,
        minute_layers=1,
        minute_heads=2,
        minute_ffn_dim=16,
        time_hidden=4,
        daily_hidden=4,
        weekly_hidden=4,
        recurrent_layers=2,
        fusion_hidden=16,
        latent_dim=8,
        predictor_hidden=16,
        dropout=0.2,
    )
    config["training"].update(
        batch_size=2,
        gradient_accumulation=1,
        max_epochs=2,
        amp=False,
        checkpoint_dir=str(checkpoint_dir),
    )
    config["evaluation"].update(ks=[2], bootstrap_samples=20)
    return config


def _model(config: dict) -> MarketJEPA:
    return MarketJEPA(
        len(MARKET_FEATURES),
        len(CONTEXT_FEATURES),
        config["data"]["horizons"],
        config["model"],
    )


def _trainer(
    config: dict,
    data,
    model: MarketJEPA,
    preflight: dict,
    runtime_options: dict | None = None,
) -> Trainer:
    return Trainer(
        model,
        config,
        MarketDataset(data, config, "train"),
        MarketDataset(data, config, "validation"),
        source_sha256="synthetic-source",
        preflight_metadata=preflight,
        device=torch.device("cpu"),
        runtime_options=runtime_options,
    )


def test_train_resume_export_eval_chain_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)
    continuous_config = _integration_config(csv_path, tmp_path / "continuous")
    resumed_config = _integration_config(csv_path, tmp_path / "resumed")
    data = prepare_market_data(continuous_config)
    report, json_path, csv_report_path = run_preflight(data, continuous_config, tmp_path / "preflight")
    preflight = {
        "json": str(json_path),
        "csv": str(csv_report_path),
        "source": report["source"],
    }

    configure_determinism(42)
    continuous = _trainer(continuous_config, data, _model(continuous_config), preflight)
    continuous_history = continuous.fit()

    configure_determinism(42)
    interrupted = _trainer(resumed_config, data, _model(resumed_config), preflight)
    first_history = interrupted.fit(stop_before_epoch=1)
    assert len(first_history) == 1
    interrupted_checkpoint = load_checkpoint(
        Path(resumed_config["training"]["checkpoint_dir"])
        / resumed_config["experiment_id"]
        / "last.pt"
    )

    # Match the real restart path: process-level seeding and model construction
    # happen before resume restores the checkpoint RNG state.
    configure_determinism(42)
    resumed = _trainer(resumed_config, data, _model(resumed_config), preflight)
    resumed.resume(interrupted_checkpoint)
    resumed_history = resumed.fit()
    assert resumed_history == continuous_history
    for name, continuous_value in continuous.model.state_dict().items():
        assert torch.equal(continuous_value, resumed.model.state_dict()[name]), name

    checkpoint_path = (
        Path(resumed_config["training"]["checkpoint_dir"])
        / resumed_config["experiment_id"]
        / "last.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path)
    export_model = _model(resumed_config)
    export_model.load_state_dict(checkpoint["model"])
    latent_dir = tmp_path / "latents"
    provenance = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_sha256": checkpoint["source_sha256"],
    }
    for split in ("train", "validation", "test"):
        export_split(
            export_model,
            MarketDataset(data, resumed_config, split),
            latent_dir / f"{split}.npz",
            torch.device("cpu"),
            batch_size=2,
            provenance=provenance,
        )
    evaluation = evaluate_exports(
        latent_dir,
        resumed_config,
        expected_checkpoint_sha256=provenance["checkpoint_sha256"],
        expected_source_sha256=provenance["source_sha256"],
    )
    assert evaluation["design_version"] == "0.6.1"
    assert evaluation["go_no_go"]["primary_horizon"] == 2
    assert set(evaluation["ranges"]) == {"train", "validation", "test"}


def test_multiworker_runtime_preserves_resume_trajectory(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)
    continuous_config = _integration_config(csv_path, tmp_path / "continuous_workers")
    resumed_config = _integration_config(csv_path, tmp_path / "resumed_workers")
    data = prepare_market_data(continuous_config)
    runtime = {
        "num_workers": 2,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

    configure_determinism(42)
    continuous = _trainer(
        continuous_config, data, _model(continuous_config), {}, runtime
    )
    continuous_history = continuous.fit()

    configure_determinism(42)
    interrupted = _trainer(
        resumed_config, data, _model(resumed_config), {}, runtime
    )
    interrupted.fit(stop_before_epoch=1)
    checkpoint = load_checkpoint(
        Path(resumed_config["training"]["checkpoint_dir"])
        / resumed_config["experiment_id"]
        / "last.pt"
    )
    configure_determinism(42)
    resumed = _trainer(resumed_config, data, _model(resumed_config), {}, runtime)
    resumed.resume(checkpoint)
    resumed_history = resumed.fit()

    assert resumed_history == continuous_history
    for name, continuous_value in continuous.model.state_dict().items():
        assert torch.equal(continuous_value, resumed.model.state_dict()[name]), name
