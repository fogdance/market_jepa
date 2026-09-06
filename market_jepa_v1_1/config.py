from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from market_jepa.config import DEFAULT_CONFIG


TRAIN_COMMODITIES = ("FG", "SA", "JM", "SH", "SP")
HELD_OUT_COMMODITY = "RB"
IMC_FEATURES = (
    "open_origin_log",
    "high_origin_log",
    "low_origin_log",
    "close_origin_log",
    "close_step_log",
    "oi_origin_log",
    "oi_step_log",
    "volume_fixed_ratio",
    "volume_q20",
)
MINUTE_CONTEXT_FEATURES = (
    "time_of_day_sin", "time_of_day_cos", "day_of_week_sin",
    "day_of_week_cos", "log1p_delta_minutes",
)
DAILY_CONTEXT_FEATURES = (
    "days_since_main", "is_main", "days_since_lost_main",
    "bar_is_partial", "elapsed_minutes_in_trading_day",
)
WEEKLY_CONTEXT_FEATURES = (
    "weeks_ago", "contract_age", "memory_source",
    "bar_is_partial", "elapsed_trading_days_in_week",
)

DEFAULT_V11_CONFIG: dict[str, Any] = {
    "design_version": "1.1",
    "experiment_id": "market_jepa_v1_1_development",
    "data": {
        "root": "/data/jepa/v1_1_raw",
        "train_commodities": list(TRAIN_COMMODITIES),
        "held_out_commodity": HELD_OUT_COMMODITY,
        "minute_capacity": 512,
        "daily_capacity": 256,
        "current_weekly_capacity": 64,
        "history_weekly_capacity": 156,
        "history_years": 3,
        "horizons": [16, 64, 256],
        "anchor_stride": 1,
    },
    "model": {
        "minute_market_dim": len(IMC_FEATURES),
        "minute_context_dim": len(MINUTE_CONTEXT_FEATURES),
        "daily_market_dim": len(IMC_FEATURES),
        "daily_context_dim": len(DAILY_CONTEXT_FEATURES),
        "current_weekly_market_dim": len(IMC_FEATURES),
        "current_weekly_context_dim": len(WEEKLY_CONTEXT_FEATURES),
        "history_weekly_market_dim": len(IMC_FEATURES),
        "history_weekly_context_dim": len(WEEKLY_CONTEXT_FEATURES),
        "d_model": 256,
        "num_heads": 8,
        "ffn_dim": 1024,
        "minute_layers": 4,
        "minute_capacity": 512,
        "daily_capacity": 256,
        "current_weekly_capacity": 64,
        "history_weekly_capacity": 156,
        "commodity_state_tokens": 4,
        "contract_state_tokens": 4,
        "belief_tokens": 8,
        "belief_dim": 256,
        "predictor_hidden": 512,
        "dropout": 0.1,
        "commodity_embedding": False,
    },
    "training": deepcopy(DEFAULT_CONFIG["training"]),
    "development": {
        "optimizer_steps": 100,
        "batch_size": 2,
        "scaler_anchors_per_commodity": 32,
        "anchors_per_contract": 16,
    },
}


def validate_model_config(config: dict[str, Any], *, debug: bool = False) -> None:
    expected = DEFAULT_V11_CONFIG["model"]
    if set(config) != set(expected):
        raise ValueError(f"V1.1 model config keys differ: {sorted(set(config) ^ set(expected))}")
    for name, value in config.items():
        if name == "commodity_embedding":
            if value is not False:
                raise ValueError("V1.1 forbids commodity embeddings")
        elif name == "dropout":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError("dropout must be in [0,1)")
        elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid V1.1 model field: {name}")
    if config["d_model"] != config["belief_dim"]:
        raise ValueError("belief_dim must equal d_model")
    if config["d_model"] % config["num_heads"] or config["d_model"] % 2:
        raise ValueError("d_model must be even and divisible by num_heads")
    if not debug:
        for name, value in expected.items():
            if config[name] != value:
                raise ValueError(f"formal V1.1 freezes {name}={value}")


def validate_v11_config(config: dict[str, Any]) -> None:
    if config.get("design_version") != "1.1":
        raise ValueError('V1.1 requires design_version="1.1"')
    validate_model_config(config["model"], debug=config.get("profile") == "debug")
    data = config["data"]
    if tuple(data["train_commodities"]) != TRAIN_COMMODITIES:
        raise ValueError("V1.1 train commodities must be FG/SA/JM/SH/SP in that order")
    if data["held_out_commodity"] != HELD_OUT_COMMODITY:
        raise ValueError("V1.1 held-out commodity must be RB")
    fixed = {
        "minute_capacity": 512, "daily_capacity": 256,
        "current_weekly_capacity": 64, "history_weekly_capacity": 156,
        "history_years": 3, "horizons": [16, 64, 256],
    }
    for name, value in fixed.items():
        if data[name] != value:
            raise ValueError(f"V1.1 freezes data.{name}={value}")
    if any(data[name] != config["model"][name] for name in (
        "minute_capacity", "daily_capacity", "current_weekly_capacity", "history_weekly_capacity"
    )):
        raise ValueError("data/model capacities differ")
    training = config["training"]
    for name in ("optimizer", "learning_rate", "weight_decay", "betas", "eps", "ema_tau",
                 "lambda_var", "lambda_cov", "variance_floor", "gradient_clip_norm",
                 "scheduler", "warmup_ratio"):
        if training[name] != DEFAULT_CONFIG["training"][name]:
            raise ValueError(f"V1.1 preserves V0 training field {name}")
    development = config["development"]
    if not 100 <= development["optimizer_steps"] <= 500:
        raise ValueError("development optimizer_steps must be 100-500")
    if development["batch_size"] not in {2, 8}:
        raise ValueError("development batch_size must be 2 or 8")


def load_v11_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("V1.1 configuration root must be a mapping")
    validate_v11_config(config)
    return config
