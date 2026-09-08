from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from market_jepa.implementation import ROOT, implementation_manifest, manifest_sha256
from market_jepa.train.checkpoint import load_checkpoint, sha256

from .config import FIXED_BUDGET_PROTOCOL, model_size_from_config, validate_v11_config
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
    common_required = {
        "v11_config", "architecture_config", "model", "optimizer", "scheduler", "scaler",
        "global_step", "rng_state", "sampler", "shared_imc_scaler",
        "data_manifest_sha256", "checkpoint_selection", "v11_implementation_manifest",
        "v11_implementation_sha256", "daily_truncation_count", "daily_truncated_tokens",
    }
    fixed = state.get("training_protocol_version") == FIXED_BUDGET_PROTOCOL
    required = common_required | ({
        "budget_mode", "max_optimizer_steps", "effective_batch_size", "samples_seen",
        "target_samples_seen", "warmup_optimizer_steps", "checkpoint_every_optimizer_steps",
        "sampling_cycle", "sampling_cycle_sample_offset", "sampling_population_sha256",
        "interval_history",
    } if fixed else {"epoch", "history"})
    if required - set(state):
        raise ValueError(f"incomplete V1.1 checkpoint: {sorted(required - set(state))}")
    validate_v11_config(state["v11_config"])
    if state["architecture_config"] != state["v11_config"]["model"]:
        raise ValueError("V1.1 checkpoint architecture mismatch")
    if state["checkpoint_selection"] != "fixed_budget_final":
        raise ValueError("V1.1 checkpoint selection must be fixed_budget_final")
    if fixed:
        training = state["v11_config"]["training"]
        expected = {
            "budget_mode": "fixed_optimizer_steps",
            "max_optimizer_steps": training["max_optimizer_steps"],
            "effective_batch_size": training["batch_size"] * training["gradient_accumulation"],
            "target_samples_seen": training["max_optimizer_steps"] * training["batch_size"] * training["gradient_accumulation"],
            "warmup_optimizer_steps": training["warmup_optimizer_steps"],
            "checkpoint_every_optimizer_steps": training["checkpoint_every_optimizer_steps"],
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"fixed-budget checkpoint {name} mismatch")
        if state["samples_seen"] != state["global_step"] * state["effective_batch_size"]:
            raise ValueError("fixed-budget checkpoint samples_seen mismatch")
        if not 0 <= state["global_step"] <= state["max_optimizer_steps"]:
            raise ValueError("fixed-budget checkpoint global_step outside budget")
        if not isinstance(state["sampling_population_sha256"], str) or len(state["sampling_population_sha256"]) != 64:
            raise ValueError("invalid sampling population SHA256")

        interval = int(state["checkpoint_every_optimizer_steps"])
        global_step = int(state["global_step"])
        maximum = int(state["max_optimizer_steps"])
        effective_batch = int(state["effective_batch_size"])
        if global_step >= maximum:
            expected_cycle, expected_num_samples, expected_offset = global_step // interval, 0, 0
        else:
            expected_cycle = global_step // interval
            cycle_start = expected_cycle * interval
            cycle_steps = min(interval, maximum - cycle_start)
            expected_num_samples = cycle_steps * effective_batch
            expected_offset = (global_step - cycle_start) * effective_batch
        if state["sampling_cycle"] != expected_cycle or state["sampling_cycle_sample_offset"] != expected_offset:
            raise ValueError("fixed-budget checkpoint sampling position mismatch")
        sampler = state["sampler"]
        if (
            sampler.get("sampling_cycle") != expected_cycle
            or sampler.get("num_samples") != expected_num_samples
            or sampler.get("start_offset") != expected_offset
            or sampler.get("sampling_population_sha256") != state["sampling_population_sha256"]
            or sampler.get("seed") != training["seed"]
        ):
            raise ValueError("fixed-budget checkpoint sampler state mismatch")
    scaler = SharedIMCScaler.from_dict(state["shared_imc_scaler"])
    train_commodities = set(state["v11_config"]["data"]["train_commodities"])
    held_out = state["v11_config"]["data"]["held_out_commodity"]
    if set(scaler.fitted_commodities) != train_commodities or held_out in scaler.fitted_commodities:
        raise ValueError("V1.1 checkpoint shared scaler Train/held-out population mismatch")
    if not any(name.startswith("target_minute.") for name in state["model"]):
        raise ValueError("V1.1 checkpoint is missing EMA target")
    if manifest_sha256(state["v11_implementation_manifest"]) != state["v11_implementation_sha256"]:
        raise ValueError("V1.1 implementation manifest checksum mismatch")
    expected_size = (
        "DEBUG" if state["v11_config"].get("profile") == "debug"
        else model_size_from_config(state["architecture_config"])
    )
    capacity_metadata = {
        "model_size": expected_size,
        "d_model": state["architecture_config"]["d_model"],
        "num_heads": state["architecture_config"]["num_heads"],
        "ffn_dim": state["architecture_config"]["ffn_dim"],
        "minute_layers": state["architecture_config"]["minute_layers"],
        "predictor_hidden_dim": state["architecture_config"]["predictor_hidden"],
    }
    capacity_metadata_names = {*capacity_metadata, "trainable_parameter_count"}
    present = capacity_metadata_names & set(state)
    if present and present != capacity_metadata_names:
        raise ValueError("incomplete V1.1 capacity checkpoint metadata")
    for name, expected in capacity_metadata.items():
        if name in state and state[name] != expected:
            raise ValueError(f"V1.1 checkpoint {name} mismatch")
    if "trainable_parameter_count" in state and (
        isinstance(state["trainable_parameter_count"], bool)
        or not isinstance(state["trainable_parameter_count"], int)
        or state["trainable_parameter_count"] <= 0
    ):
        raise ValueError("invalid V1.1 checkpoint trainable parameter count")
    if "wandb_run_id" in state and state["wandb_run_id"] is not None and (
        not isinstance(state["wandb_run_id"], str) or not state["wandb_run_id"]
    ):
        raise ValueError("invalid V1.1 checkpoint wandb_run_id")


def load_v11_checkpoint(path: str | Path, map_location="cpu") -> dict:
    state = load_checkpoint(path, map_location=map_location)
    validate_v11_checkpoint(state)
    return state


def model_from_checkpoint(state: dict) -> MarketJEPAV11:
    validate_v11_checkpoint(state)
    if state.get("evaluation_variant", "Full") != "Full":
        raise ValueError("control checkpoint requires explicit evaluation control model loader")
    model = MarketJEPAV11(state["architecture_config"], debug=state["v11_config"].get("profile") == "debug")
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if state.get("trainable_parameter_count", actual) != actual:
        raise ValueError("V1.1 checkpoint trainable parameter count mismatch")
    model.load_state_dict(state["model"], strict=True)
    return model
