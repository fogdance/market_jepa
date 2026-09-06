from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from market_jepa.config import DEFAULT_CONFIG


FEATURE_DIMENSIONS = (
    "minute_market_dim", "minute_context_dim", "daily_market_dim",
    "daily_context_dim", "weekly_market_dim", "weekly_context_dim",
)

DEFAULT_V1_CONFIG: dict[str, Any] = {
    "design_version": "1.0",
    "experiment_id": "market_jepa_v1_development",
    "data": deepcopy(DEFAULT_CONFIG["data"]),
    "model": {
        "minute_market_dim": 14, "minute_context_dim": 5,
        "daily_market_dim": 14, "daily_context_dim": 1,
        "weekly_market_dim": 14, "weekly_context_dim": 1,
        "d_model": 256, "num_heads": 8, "ffn_dim": 1024,
        "minute_layers": 4, "minute_max_length": 512,
        "daily_gru_layers": 2, "daily_gru_hidden": 128,
        "weekly_gru_layers": 2, "weekly_gru_hidden": 128,
        "num_state_tokens": 8, "cross_scale_rounds": 2,
        "belief_dim": 256, "predictor_hidden": 512,
        "dropout": 0.1, "use_commodity_embedding": False,
    },
    "training": deepcopy(DEFAULT_CONFIG["training"]),
    "evaluation": deepcopy(DEFAULT_CONFIG["evaluation"]),
    "development": {"optimizer_steps": 100, "batch_size": 2, "train_days": 30},
}


def validate_model_config(config: dict[str, Any], *, debug: bool = False) -> None:
    expected = DEFAULT_V1_CONFIG["model"]
    if set(config) != set(expected):
        raise ValueError(f"V1 model config keys differ: {sorted(set(config) ^ set(expected))}")
    for name, value in config.items():
        if name == "use_commodity_embedding":
            if value is not False:
                raise ValueError("V1 forbids commodity embeddings")
        elif name == "dropout":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError("dropout must be in [0,1)")
        else:
            minimum = 0 if name.endswith("context_dim") or name == "cross_scale_rounds" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid V1 model field: {name}")
    if config["d_model"] != config["belief_dim"]:
        raise ValueError("belief_dim must equal the minute target dimension")
    if config["d_model"] % config["num_heads"] or config["d_model"] % 2:
        raise ValueError("d_model must be even and divisible by num_heads")
    if not debug:
        flexible = {*FEATURE_DIMENSIONS, "minute_max_length"}
        for name in set(expected) - flexible:
            if config[name] != expected[name]:
                raise ValueError(f"formal V1 freezes {name}={expected[name]}")
    elif config["cross_scale_rounds"] not in {0, 2}:
        raise ValueError("V1 supports only two rounds, or zero for debug")


def validate_v1_config(config: dict[str, Any]) -> None:
    if config.get("design_version") != "1.0":
        raise ValueError('V1 requires design_version="1.0"')
    validate_model_config(config["model"], debug=config.get("profile") == "debug")
    data, training = config["data"], config["training"]
    if data["horizons"] != [16, 64, 256]:
        raise ValueError("V1 JEPA horizons are frozen at H16/H64/H256")
    if not max(data["horizons"]) <= data["minute_context_length"] <= config["model"]["minute_max_length"]:
        raise ValueError("minute context must cover persistence and fit positional capacity")
    for name in ("optimizer", "learning_rate", "weight_decay", "betas", "eps", "ema_tau",
                 "lambda_var", "lambda_cov", "variance_floor", "gradient_clip_norm", "scheduler", "warmup_ratio"):
        if training[name] != DEFAULT_CONFIG["training"][name]:
            raise ValueError(f"V1 architecture development preserves V0 training field {name}")
    ranges = [data["splits"][name] for name in ("train", "validation", "test")]
    if any(start > end for start, end in ranges) or not ranges[0][1] < ranges[1][0] <= ranges[1][1] < ranges[2][0]:
        raise ValueError("split ranges must be ordered and non-overlapping")
    development = config["development"]
    if not 100 <= development["optimizer_steps"] <= 500:
        raise ValueError("real-data development smoke is limited to 100–500 optimizer steps")
    if development["batch_size"] not in {2, 8} or development["train_days"] < 2:
        raise ValueError("invalid development batch size or training-day budget")


def load_v1_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("V1 configuration root must be a mapping")
    validate_v1_config(config)
    return config
