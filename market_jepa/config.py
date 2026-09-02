from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "market_jepa_v0_default",
    "design_version": "0.6.1",
    "data": {
        "csv_path": "8Y_DCE_JM2601_1m.csv",
        "symbol": "JM",
        "series_id": "8Y_DCE_JM2601",
        "minute_context_length": 512,
        "horizons": [16, 64, 256],
        "realized_vol_window": 32,
        "anchor_stride": 1,
        "splits": {
            "train": ["2018-01-02", "2022-12-31"],
            "validation": ["2023-01-01", "2024-12-31"],
            "test": ["2025-01-01", "2025-12-02"],
        },
    },
    "model": {
        "ablation": "minute_daily_weekly",
        "minute_d_model": 256,
        "minute_layers": 4,
        "minute_heads": 8,
        "minute_ffn_dim": 1024,
        "time_hidden": 32,
        "daily_hidden": 128,
        "weekly_hidden": 128,
        "recurrent_layers": 2,
        "fusion_hidden": 512,
        "latent_dim": 256,
        "predictor_hidden": 512,
        "dropout": 0.1,
    },
    "training": {
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 3.0e-4,
        "weight_decay": 0.05,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "batch_size": 64,
        "gradient_accumulation": 2,
        "max_epochs": 50,
        "warmup_ratio": 0.05,
        "scheduler": "cosine",
        "gradient_clip_norm": 1.0,
        "ema_tau": 0.996,
        "amp": True,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "variance_floor": 1.0,
        "collapse_threshold": 0.05,
        "num_workers": 0,
        "checkpoint_dir": "artifacts/checkpoints",
    },
    "evaluation": {
        "ks": [20, 50, 100],
        "ridge_alpha": 1.0,
        "bootstrap_samples": 10_000,
        "seed": 42,
        "output_dir": "artifacts/evaluation",
    },
}

ABLATIONS = {"minute", "minute_daily", "minute_daily_weekly"}
FROZEN_EXPERIMENTS = {
    "market_jepa_v0_default": "minute_daily_weekly",
    "market_jepa_v0_minute": "minute",
    "market_jepa_v0_minute_daily": "minute_daily",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(value, smoke=False)
    return value


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an isolated smoke profile; never mutate the frozen config."""

    value = deepcopy(config)
    value["experiment_id"] = f'{config["experiment_id"]}_smoke'
    value["profile"] = "smoke"
    value["model"].update(
        minute_d_model=32,
        minute_layers=1,
        minute_heads=4,
        minute_ffn_dim=64,
        time_hidden=8,
        daily_hidden=16,
        weekly_hidden=16,
        recurrent_layers=1,
        fusion_hidden=64,
        latent_dim=32,
        predictor_hidden=64,
        dropout=0.0,
    )
    value["training"].update(
        batch_size=4,
        gradient_accumulation=1,
        max_epochs=2,
        amp=False,
    )
    value["evaluation"].update(ks=[2, 4, 8], bootstrap_samples=200)
    value["data"]["minute_context_length"] = 32
    value["data"]["horizons"] = [4, 8, 16]
    value["smoke"] = {"max_samples_per_split": 16}
    validate_config(value, smoke=True)
    return value


def validate_config(config: dict[str, Any], smoke: bool | None = None) -> None:
    required = {"experiment_id", "design_version", "data", "model", "training", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"configuration missing keys: {sorted(missing)}")
    if config["design_version"] != "0.6.1":
        raise ValueError("implementation accepts only frozen design_version 0.6.1")
    is_smoke = config.get("profile") == "smoke" if smoke is None else smoke
    if not is_smoke:
        expected = deepcopy(DEFAULT_CONFIG)
        experiment_id = config.get("experiment_id")
        if experiment_id not in FROZEN_EXPERIMENTS:
            raise ValueError("formal V0 experiment_id is not a frozen ablation")
        expected["experiment_id"] = experiment_id
        expected["model"]["ablation"] = FROZEN_EXPERIMENTS[experiment_id]
        if config != expected:
            raise ValueError("formal V0 config differs from the frozen default protocol")
    data = config["data"]
    horizons = data["horizons"]
    if sorted(set(horizons)) != horizons or min(horizons) <= 0:
        raise ValueError("horizons must be unique, positive, and ascending")
    if data["minute_context_length"] < max(horizons):
        raise ValueError("minute context must cover the largest persistence horizon")
    if data["anchor_stride"] != 1:
        raise ValueError("V0 anchor_stride is frozen at 1")
    if config["model"]["ablation"] not in ABLATIONS:
        raise ValueError(f"unknown ablation: {config['model']['ablation']}")
    if config["model"]["minute_d_model"] != config["model"]["latent_dim"]:
        raise ValueError("minute_d_model and latent_dim must match the target space")
    if config["model"]["minute_d_model"] % config["model"]["minute_heads"]:
        raise ValueError("minute_d_model must be divisible by minute_heads")
    ranges = list(data["splits"].values())
    if not (ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]):
        raise ValueError("split ranges must be ordered and non-overlapping")
