from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from market_jepa.implementation import ROOT, implementation_manifest, manifest_sha256
from market_jepa.train.checkpoint import load_checkpoint, sha256

from .config import FEATURE_DIMENSIONS, validate_v1_config
from .model import MarketJEPAV1


def v1_implementation_manifest() -> dict:
    result = deepcopy(implementation_manifest())
    paths = [*ROOT.glob("market_jepa_v1/**/*.py"), *ROOT.glob("configs/v1/*.yaml"),
             ROOT / "develop_market_jepa_v1.py"]
    for path in sorted(paths):
        result["files"][path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def validate_v1_checkpoint(state: dict) -> None:
    if state.get("design_version") != "1.0":
        raise ValueError('V1 checkpoint requires design_version="1.0"; cannot load V0 as V1')
    required = {"config", "v1_config", "architecture_config", "feature_dimensions", "num_state_tokens", "cross_scale_rounds",
                "model", "optimizer", "scheduler", "scaler", "global_step", "rng_state", "normalizers",
                "v1_implementation_manifest", "v1_implementation_sha256"}
    if required - set(state):
        raise ValueError(f"incomplete V1 checkpoint: {sorted(required - set(state))}")
    validate_v1_config(state["v1_config"])
    architecture = state["v1_config"]["model"]
    trainer_config = deepcopy(state["v1_config"])
    trainer_config["model"]["latent_dim"] = architecture["belief_dim"]
    if state["config"] != trainer_config:
        raise ValueError("V1 checkpoint training config mismatch")
    if state["architecture_config"] != architecture:
        raise ValueError("V1 checkpoint architecture config mismatch")
    if state["feature_dimensions"] != {key: architecture[key] for key in FEATURE_DIMENSIONS}:
        raise ValueError("V1 checkpoint feature dimensions mismatch")
    for name in ("num_state_tokens", "cross_scale_rounds"):
        if state[name] != architecture[name]:
            raise ValueError(f"V1 checkpoint {name} mismatch")
    if not any(key.startswith("target_minute.") for key in state["model"]):
        raise ValueError("V1 checkpoint is missing EMA state")
    if manifest_sha256(state["v1_implementation_manifest"]) != state["v1_implementation_sha256"]:
        raise ValueError("V1 checkpoint manifest digest mismatch")


def load_v1_checkpoint(path: str | Path, map_location="cpu") -> dict:
    state = load_checkpoint(path, map_location=map_location)
    validate_v1_checkpoint(state)
    return state


def model_from_checkpoint(state: dict) -> MarketJEPAV1:
    validate_v1_checkpoint(state)
    model = MarketJEPAV1(state["architecture_config"], debug=state["v1_config"].get("profile") == "debug")
    model.load_state_dict(state["model"], strict=True)
    return model
