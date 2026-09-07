"""Explicit from-scratch S controls using a completed Full S run's protocol."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from market_jepa.train.trainer import configure_determinism
from market_jepa_v1_1.checkpoint import load_v11_checkpoint
from market_jepa_v1_1.evaluation.controls import ControlTrainer, control_model
from market_jepa_v1_1.evaluation.protocol import file_hash, write_json, verify_data_files, completed_run_gate, semantic_implementation_gate
from market_jepa_v1_1.evaluation.runner import dataset_for
from market_jepa_v1_1.formal_training import _save_epoch_outputs
from market_jepa_v1_1.imc import SharedIMCScaler


def train_control(reference, variant, output, resume=None):
    reference = Path(reference)
    state = load_v11_checkpoint(reference)
    completed_run_gate(reference, state)
    semantic_implementation_gate(state)
    config = deepcopy(state["v11_config"])
    if state.get("model_size", "S") != "S" or state.get("evaluation_variant", "Full") != "Full":
        raise ValueError("matched controls require a Full S reference")
    if state["epoch"] != config["training"]["max_epochs"] - 1:
        raise ValueError("reference is not fixed-budget endpoint")
    if file_hash(Path(config["data"]["root"]) / "build_manifest.json") != state["data_manifest_sha256"]:
        raise ValueError("reference data changed")
    if "RB" in config["data"]["train_commodities"]:
        raise ValueError("RB contamination")
    verify_data_files(config["data"]["root"], config["data"]["train_commodities"])
    output = Path(output)
    if (output / "checkpoints" / "last.pt").exists() and resume is None:
        raise FileExistsError("control checkpoint exists; use --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("formal control training requires CUDA")
    config["training"]["checkpoint_dir"] = str(output / "checkpoints")
    scaler = SharedIMCScaler.from_dict(state["shared_imc_scaler"])
    dataset = dataset_for(config, scaler, config["data"]["train_commodities"])
    if len(dataset) != state["sampler"]["num_samples"]:
        raise ValueError("reference/control anchor population size mismatch")
    configure_determinism(config["training"]["seed"])
    model = control_model(config["model"], variant)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if variant == "LateFusion" and abs(count / state["trainable_parameter_count"] - 1) > .05:
        raise ValueError("LateFusion parameter match exceeds 5%")
    trainer = ControlTrainer(model, config, dataset, torch.device("cuda"),
                             data_manifest_sha256=state["data_manifest_sha256"])
    if resume is not None:
        trainer.resume(load_v11_checkpoint(resume))
    write_json(output / "control_protocol.json", {
        "variant": variant, "reference_sha256": file_hash(reference), "config": config,
        "trainable_params": count, "scaler_sha256": scaler.checksum,
        "from_scratch": resume is None, "checkpoint_policy": "fixed_budget_final",
        "data_manifest_sha256": state["data_manifest_sha256"], "RB_read": False,
    })
    trainer.fit(epoch_callback=lambda _: _save_epoch_outputs(output, trainer.history))
    endpoint = output / "checkpoints" / "last.pt"
    restored = load_v11_checkpoint(endpoint)
    check = control_model(config["model"], variant)
    check.load_state_dict(restored["model"], strict=True)
    write_json(output / "final_summary.json", {
        "status": "V1_1_CONTROL_TRAINING_PASS", "variant": variant,
        "final_epoch": restored["epoch"], "checkpoint_sha256": file_hash(endpoint), "RB_read": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--variant", choices=["LateFusion", "MinuteOnly", "NoHistoricalWeekly"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    torch.set_num_threads(2)
    train_control(args.reference, args.variant, args.output, args.resume)
