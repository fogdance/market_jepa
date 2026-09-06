from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from market_jepa.implementation import ROOT, implementation_manifest, manifest_sha256
from market_jepa.train.checkpoint import load_checkpoint, sha256

from .config import validate_v11_config
from .imc import SharedIMCScaler
from .model import MarketJEPAV11


def v11_implementation_manifest() -> dict:
    result = deepcopy(implementation_manifest())
    paths = [
        *ROOT.glob("market_jepa_v1/**/*.py"), *ROOT.glob("market_jepa_v1_1/**/*.py"),
        *ROOT.glob("configs/v1/*.yaml"), *ROOT.glob("configs/v1_1/*.yaml"),
        ROOT / "develop_market_jepa_v1.py", ROOT / "develop_market_jepa_v1_1.py",
        ROOT / "train_market_jepa_v1_1.py",
    ]
    for path in sorted(path for path in paths if path.is_file()):
        result["files"][path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def validate_v11_checkpoint(state: dict) -> None:
    if state.get("design_version") != "1.1":
        raise ValueError('V1.1 checkpoint requires design_version="1.1"')
    required = {
        "v11_config", "architecture_config", "model", "optimizer", "scheduler", "scaler",
        "epoch", "global_step", "rng_state", "sampler", "shared_imc_scaler",
        "data_manifest_sha256", "checkpoint_selection", "v11_implementation_manifest",
        "v11_implementation_sha256", "daily_truncation_count", "daily_truncated_tokens",
    }
    if required - set(state):
        raise ValueError(f"incomplete V1.1 checkpoint: {sorted(required - set(state))}")
    validate_v11_config(state["v11_config"])
    if state["architecture_config"] != state["v11_config"]["model"]:
        raise ValueError("V1.1 checkpoint architecture mismatch")
    if state["checkpoint_selection"] != "fixed_budget_final":
        raise ValueError("V1.1 checkpoint selection must be fixed_budget_final")
    scaler = SharedIMCScaler.from_dict(state["shared_imc_scaler"])
    train_commodities = set(state["v11_config"]["data"]["train_commodities"])
    held_out = state["v11_config"]["data"]["held_out_commodity"]
    if set(scaler.fitted_commodities) != train_commodities or held_out in scaler.fitted_commodities:
        raise ValueError("V1.1 checkpoint shared scaler Train/held-out population mismatch")
    if not any(name.startswith("target_minute.") for name in state["model"]):
        raise ValueError("V1.1 checkpoint is missing EMA target")
    if manifest_sha256(state["v11_implementation_manifest"]) != state["v11_implementation_sha256"]:
        raise ValueError("V1.1 implementation manifest checksum mismatch")


def load_v11_checkpoint(path: str | Path, map_location="cpu") -> dict:
    state = load_checkpoint(path, map_location=map_location)
    validate_v11_checkpoint(state)
    return state


def model_from_checkpoint(state: dict) -> MarketJEPAV11:
    validate_v11_checkpoint(state)
    model = MarketJEPAV11(state["architecture_config"], debug=state["v11_config"].get("profile") == "debug")
    model.load_state_dict(state["model"], strict=True)
    return model
