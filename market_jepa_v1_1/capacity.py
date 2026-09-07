from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MODEL_SIZE_PROFILES, load_v11_config
from .model import MarketJEPAV11


PROFILE_CONFIG_PATHS = {
    size: Path(f"configs/v1_1/market_jepa_v1_1_{size.lower()}.yaml")
    for size in MODEL_SIZE_PROFILES
}


def _online_category(name: str) -> str:
    if name.startswith("predictors."):
        horizon = name.split(".", 2)[1]
        return f"predictor_h{horizon}"
    projection_prefixes = (
        "minute_local.market_core.market_projection.",
        "minute_local.market_core.validity_projection.",
        "minute_local.context_projection.",
        "commodity_memory.tokenizer.market_projection.",
        "commodity_memory.tokenizer.validity_projection.",
        "commodity_memory.tokenizer.context_projection.",
        "commodity_memory.tokenizer.boundary_projection.",
        "commodity_memory.tokenizer.source_embedding",
        "contract_lifecycle.daily.market_projection.",
        "contract_lifecycle.daily.validity_projection.",
        "contract_lifecycle.daily.context_projection.",
        "contract_lifecycle.daily.source_embedding",
        "contract_lifecycle.weekly.market_projection.",
        "contract_lifecycle.weekly.validity_projection.",
        "contract_lifecycle.weekly.context_projection.",
        "contract_lifecycle.weekly.source_embedding",
    )
    if name.startswith(projection_prefixes):
        return "input_projections"
    roots = {
        "minute_local.": "minute_local_encoder",
        "commodity_memory.": "commodity_memory_stage",
        "contract_lifecycle.": "contract_state_stage",
        "minute_conditioner.": "minute_higher_scale_conditioner",
        "belief_encoder.": "belief_stage",
    }
    for prefix, category in roots.items():
        if name.startswith(prefix):
            return category
    raise RuntimeError(f"unclassified V1.1 parameter: {name}")


def parameter_report(model: MarketJEPAV11) -> dict[str, Any]:
    categories = {
        "input_projections": 0,
        "minute_local_encoder": 0,
        "commodity_memory_stage": 0,
        "contract_state_stage": 0,
        "minute_higher_scale_conditioner": 0,
        "belief_stage": 0,
        "predictor_h16": 0,
        "predictor_h64": 0,
        "predictor_h256": 0,
    }
    ema = 0
    non_trainable_other = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("target_minute."):
            ema += count
        elif parameter.requires_grad:
            categories[_online_category(name)] += count
        else:
            non_trainable_other += count
    trainable = sum(categories.values())
    direct_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != direct_trainable:
        raise RuntimeError("V1.1 parameter decomposition does not cover all trainable parameters")
    total_non_trainable = ema + non_trainable_other
    resident = trainable + total_non_trainable
    return {
        "model_size": model.model_size,
        "architecture": {
            name: model.architecture_config[name]
            for name in (
                "d_model", "num_heads", "ffn_dim", "minute_layers",
                "commodity_state_tokens", "contract_state_tokens", "belief_tokens",
                "predictor_hidden", "minute_capacity", "daily_capacity",
                "current_weekly_capacity", "history_weekly_capacity",
            )
        },
        "trainable_parameters": trainable,
        "trainable_decomposition": categories,
        "ema_target_parameters": ema,
        "other_non_trainable_parameters": non_trainable_other,
        "total_non_trainable_parameters": total_non_trainable,
        "total_resident_parameters": resident,
    }


def load_capacity_configs(root: str | Path = ".") -> dict[str, dict[str, Any]]:
    root = Path(root)
    return {
        size: load_v11_config(root / path)
        for size, path in PROFILE_CONFIG_PATHS.items()
    }
