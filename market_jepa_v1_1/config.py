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
    "days_since_main_fraction", "is_main", "days_since_lost_main_fraction",
    "bar_is_partial", "observed_minute_count_fraction",
)
WEEKLY_CONTEXT_FEATURES = (
    "weeks_ago_fraction", "contract_age_fraction", "memory_source",
    "bar_is_partial", "elapsed_trading_days_fraction",
)

MODEL_SIZE_PROFILES: dict[str, dict[str, int]] = {
    "S": {
        "d_model": 256, "num_heads": 8, "ffn_dim": 1024,
        "minute_layers": 4, "belief_dim": 256, "predictor_hidden": 512,
    },
    "M": {
        "d_model": 384, "num_heads": 6, "ffn_dim": 1536,
        "minute_layers": 6, "belief_dim": 384, "predictor_hidden": 768,
    },
    "L": {
        "d_model": 512, "num_heads": 8, "ffn_dim": 2048,
        "minute_layers": 8, "belief_dim": 512, "predictor_hidden": 1024,
    },
    "XL": {
        "d_model": 768, "num_heads": 12, "ffn_dim": 3072,
        "minute_layers": 8, "belief_dim": 768, "predictor_hidden": 1536,
    },
}

PROFILE_TRAINING: dict[str, dict[str, int | bool]] = {
    "S": {"batch_size": 64, "gradient_accumulation": 2, "gradient_checkpointing": False},
    "M": {"batch_size": 64, "gradient_accumulation": 2, "gradient_checkpointing": False},
    "L": {"batch_size": 64, "gradient_accumulation": 2, "gradient_checkpointing": True},
    "XL": {"batch_size": 64, "gradient_accumulation": 2, "gradient_checkpointing": True},
}

DEFAULT_V11_CONFIG: dict[str, Any] = {
    "design_version": "1.1",
    "model_size": "S",
    "experiment_id": "market_jepa_v1_1_development",
    "data": {
        "root": "/data/jepa/v1_1_raw",
        "train_commodities": list(TRAIN_COMMODITIES),
        "held_out_commodity": HELD_OUT_COMMODITY,
        "minute_capacity": 512,
        "daily_capacity": 256,
        "current_weekly_capacity": 64,
        "horizons": [16, 64, 256],
        "anchor_stride": 1,
    },
    "history_week": {
        "years": 3,
        "commodity_years": {"SH": 2},
        "capacity": 156,
        "require_full_history": True,
        "series_mode": "same_delivery_month",
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
    "training": {
        **deepcopy(DEFAULT_CONFIG["training"]), "num_workers": 8,
        "gradient_checkpointing": False,
        "amp_dtype": "bfloat16",
    },
    "development": {
        "optimizer_steps": 100,
        "batch_size": 2,
        "scaler_anchors_per_commodity": 32,
        "anchors_per_contract": 16,
    },
}


def model_size_from_config(config: dict[str, Any]) -> str:
    matches = [
        size for size, values in MODEL_SIZE_PROFILES.items()
        if all(config.get(name) == value for name, value in values.items())
    ]
    if len(matches) != 1:
        raise ValueError("V1.1 model dimensions do not match a registered S/M/L/XL profile")
    return matches[0]


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
        size = model_size_from_config(config)
        scalable = set(MODEL_SIZE_PROFILES[size])
        for name, value in expected.items():
            if name not in scalable and config[name] != value:
                raise ValueError(f"formal V1.1 freezes non-capacity field {name}={value}")


def validate_v11_config(config: dict[str, Any]) -> None:
    if config.get("design_version") != "1.1":
        raise ValueError('V1.1 requires design_version="1.1"')
    validate_model_config(config["model"], debug=config.get("profile") == "debug")
    if config.get("profile") != "debug":
        inferred_size = model_size_from_config(config["model"])
        if config.get("model_size", "S") != inferred_size:
            raise ValueError(f"model_size must be {inferred_size} for configured dimensions")
    data = config["data"]
    train_commodities = data.get("train_commodities")
    if (
        not isinstance(train_commodities, list) or not train_commodities
        or any(not isinstance(value, str) or not value for value in train_commodities)
        or len(set(train_commodities)) != len(train_commodities)
    ):
        raise ValueError("V1.1 data.train_commodities must be a nonempty unique string list")
    held_out = data.get("held_out_commodity")
    if not isinstance(held_out, str) or not held_out or held_out in train_commodities:
        raise ValueError("V1.1 held-out commodity must be nonempty and absent from Train commodities")
    fixed = {
        "minute_capacity": 512, "daily_capacity": 256,
        "current_weekly_capacity": 64, "horizons": [16, 64, 256],
    }
    for name, value in fixed.items():
        if data[name] != value:
            raise ValueError(f"V1.1 freezes data.{name}={value}")
    if any(data[name] != config["model"][name] for name in (
        "minute_capacity", "daily_capacity", "current_weekly_capacity"
    )):
        raise ValueError("data/model capacities differ")
    history = config.get("history_week")
    if not isinstance(history, dict) or set(history) != {
        "years", "commodity_years", "capacity", "require_full_history", "series_mode",
    }:
        raise ValueError(
            "history_week requires years/commodity_years/capacity/require_full_history/series_mode"
        )
    if isinstance(history["years"], bool) or not isinstance(history["years"], int) or history["years"] <= 0:
        raise ValueError("history_week.years must be a positive integer")
    commodity_years = history["commodity_years"]
    if not isinstance(commodity_years, dict):
        raise ValueError("history_week.commodity_years must be a mapping")
    unknown = set(commodity_years) - set((*train_commodities, held_out))
    if unknown:
        raise ValueError(f"unknown history_week commodity override: {sorted(unknown)}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in commodity_years.values()
    ):
        raise ValueError("history_week.commodity_years values must be positive integers")
    if history["capacity"] != 156 or history["capacity"] != config["model"]["history_weekly_capacity"]:
        raise ValueError("V1.1 freezes history_week.capacity=156")
    if not isinstance(history["require_full_history"], bool):
        raise ValueError("history_week.require_full_history must be boolean")
    if config.get("profile") != "debug" and history["require_full_history"] is not True:
        raise ValueError("formal V1.1 requires the configured full historical coverage")
    if history["series_mode"] != "same_delivery_month":
        raise ValueError("V1.1 requires history_week.series_mode=same_delivery_month")
    training = config["training"]
    if not isinstance(training.get("gradient_checkpointing", False), bool):
        raise ValueError("training.gradient_checkpointing must be boolean")
    amp_dtype = training.get("amp_dtype", "float16")
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for name, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _deep_merge(result[name], value)
        else:
            result[name] = deepcopy(value)
    return result


def _load_config_tree(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"cyclic V1.1 config extends: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("V1.1 configuration root must be a mapping")
    parent = value.pop("extends", None)
    if parent is None:
        return value
    if not isinstance(parent, str) or not parent:
        raise ValueError("V1.1 config extends must be a nonempty relative path")
    parent_path = (resolved.parent / parent).resolve()
    return _deep_merge(_load_config_tree(parent_path, {*seen, resolved}), value)


def load_v11_config(path: str | Path) -> dict[str, Any]:
    config = _load_config_tree(Path(path), set())
    if not isinstance(config, dict):
        raise ValueError("V1.1 configuration root must be a mapping")
    validate_v11_config(config)
    return config
