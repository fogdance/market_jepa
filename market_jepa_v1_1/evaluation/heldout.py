"""Explicit frozen RB execution. Inventory alone does not evaluate RB outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from market_jepa_v1_1.checkpoint import load_v11_checkpoint, model_from_checkpoint
from market_jepa_v1_1.imc import SharedIMCScaler

from .controls import control_model
from .metrics import bootstrap, predictive_status
from .probe import predict_probe
from .protocol import digest, evaluation_code_hash, file_hash, final_checkpoint_gate, validate_manifest, write_json, verify_data_files, completed_run_gate, semantic_implementation_gate
from .runner import extract, write_csv
from .campaign import campaign_plan, exact_cohort
from .protocol import matched_control_gate


def freeze(checkpoints, evaluations, manifest_path, output, *, plan_path):
    expected, plan_hash = campaign_plan(plan_path, output)
    if Path(output).exists():
        raise FileExistsError("frozen campaign already exists")
    if len(checkpoints) != len(evaluations) or not checkpoints:
        raise ValueError("one Train evaluation per checkpoint required")
    manifest = json.loads(Path(manifest_path).read_text()); validate_manifest(manifest)
    if not manifest["rb_inventory"]:
        raise ValueError("RB inventory required")
    models = []
    common = None
    protocols = []
    for checkpoint, directory in zip(checkpoints, evaluations):
        state = load_v11_checkpoint(checkpoint); final_checkpoint_gate(state, manifest)
        semantic_implementation_gate(state)
        completed_run_gate(checkpoint, state)
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        protocol = summary["protocol"]
        if (protocol["model_size"] != state.get("model_size", "S") or
                protocol["variant"] != state.get("evaluation_variant", "Full")):
            raise ValueError("evaluation/checkpoint identity mismatch")
        if (protocol.get("sampler_num_samples") != state["sampler"]["num_samples"] or
                (protocol["variant"] != "Full" and protocol.get("control_provenance") != state.get("control_provenance"))):
            raise ValueError("evaluation/checkpoint control provenance mismatch")
        if json.loads((directory / "evaluation_protocol.json").read_text()) != protocol:
            raise ValueError("evaluation summary/protocol mismatch")
        protocols.append(protocol)
        shared = {key: protocol[key] for key in ("manifest_sha256", "epochs", "scaler_sha256", "data_manifest_sha256")}
        if common is not None and common != shared:
            raise ValueError("RB campaign training populations/budgets/scalers are unmatched")
        common = shared
        if protocol["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Train evaluation checkpoint mismatch")
        if protocol["evaluation_code_sha256"] != evaluation_code_hash():
            raise ValueError("evaluation code changed after Train evaluation")
        models.append({"model_size": protocol["model_size"], "variant": protocol["variant"],
                       "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_hash(checkpoint),
                       "train_evaluation": str(directory.resolve()), "probe_sha256": file_hash(directory / "a2_probe_hyperparams.json"),
                       "train_protocol_sha256": file_hash(directory / "evaluation_protocol.json")})
    exact_cohort([(p["model_size"], p["variant"]) for p in protocols], expected)
    for p in protocols:
        if p["variant"] != "Full":
            references = [r for r in protocols if (r["model_size"], r["variant"]) == ("S", "Full")]
            if len(references) != 1:
                raise ValueError("control cohort requires Full S reference")
            matched_control_gate(p, references[0])
    payload = {"models": models, "campaign_plan": str(Path(plan_path).resolve()), "campaign_plan_sha256": plan_hash,
               "manifest": str(Path(manifest_path).resolve()), "manifest_sha256": manifest["sha256"],
               "evaluation_code_sha256": evaluation_code_hash(), "human_review_pass": False,
               "tests_pass": False, "rb_test_consumed": False}
    payload["freeze_sha256"] = digest(payload)
    write_json(output, payload)
    return payload


def run_heldout(freeze_path, output, *, partition, approval=None, device="cpu", batch_size=8):
    frozen = json.loads(Path(freeze_path).read_text())
    if digest({k: v for k, v in frozen.items() if k != "freeze_sha256"}) != frozen["freeze_sha256"]:
        raise ValueError("freeze checksum mismatch")
    if frozen["evaluation_code_sha256"] != evaluation_code_hash():
        raise ValueError("evaluation code changed since freeze")
    if partition not in {"rb_dev", "rb_test"}:
        raise ValueError("invalid RB partition")
    output = Path(output)
    if partition == "rb_test":
        if approval is None:
            raise PermissionError("formal RB-Test requires human-reviewed approval file")
        approved = json.loads(Path(approval).read_text())
        if (approved.get("freeze_sha256") != frozen["freeze_sha256"] or
            approved.get("human_review_pass") is not True or approved.get("tests_pass") is not True):
            raise PermissionError("approval does not match frozen protocol/tests")
    if not frozen.get("models"):
        raise ValueError("empty frozen checkpoint cohort")
    expected, plan_hash = campaign_plan(frozen["campaign_plan"], freeze_path)
    if plan_hash != frozen["campaign_plan_sha256"]:
        raise ValueError("campaign plan changed after freeze")
    exact_cohort([(m["model_size"], m["variant"]) for m in frozen["models"]], expected)
    manifest = json.loads(Path(frozen["manifest"]).read_text()); validate_manifest(manifest)
    if manifest["sha256"] != frozen["manifest_sha256"]:
        raise ValueError("RB manifest changed")
    # Validate every model/probe before consuming the one-shot campaign.
    for spec in frozen["models"]:
        directory = Path(spec["train_evaluation"])
        if file_hash(spec["checkpoint"]) != spec["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint changed")
        if file_hash(directory / "a2_probe_hyperparams.json") != spec["probe_sha256"]:
            raise ValueError("frozen probe changed")
        if file_hash(directory / "evaluation_protocol.json") != spec["train_protocol_sha256"]:
            raise ValueError("Train evaluation protocol changed")
        state = load_v11_checkpoint(spec["checkpoint"])
        final_checkpoint_gate(state, manifest)
        semantic_implementation_gate(state)
        if file_hash(Path(state["v11_config"]["data"]["root"]) / "build_manifest.json") != manifest["data_manifest_sha256"]:
            raise ValueError("RB data build changed")
        verify_data_files(state["v11_config"]["data"]["root"], ["RB"])
    records = [r for r in manifest["anchors"] if r["split"] == partition]
    if not records:
        raise ValueError("empty heldout partition")
    if partition == "rb_test":
        # Ledger belongs to the frozen campaign, independent of --output choice.
        ledger = Path(freeze_path).resolve().parent / "rb_test_consumption.json"
        with ledger.open("x") as handle:
            json.dump({"rb_test_consumed": True, "state": "STARTED", "freeze_sha256": frozen["freeze_sha256"]}, handle)
    results = []
    for index, spec in enumerate(frozen["models"]):
        state = load_v11_checkpoint(spec["checkpoint"]); final_checkpoint_gate(state, manifest)
        config = state["v11_config"]
        if file_hash(Path(config["data"]["root"]) / "build_manifest.json") != manifest["data_manifest_sha256"]:
            raise ValueError("RB data build changed")
        variant = state.get("evaluation_variant", "Full")
        model = model_from_checkpoint(state) if variant == "Full" else control_model(state["architecture_config"], variant)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval().requires_grad_(False)
        scaler = SharedIMCScaler.from_dict(state["shared_imc_scaler"])
        exports, ordered = extract(model, config, scaler, records, torch.device(device), batch_size=batch_size,
                                   diagnostics=False, rb=True)
        probes = json.loads((Path(spec["train_evaluation"]) / "a2_probe_hyperparams.json").read_text())
        errors = {key: np.square((predict_probe(probes[key], exports[key]) - exports["outcomes"]) / probes[key]["y_std"])
                  for key in ("belief", "summary")}
        a1 = bootstrap(exports["jepa"], exports["persistence"], ordered)
        a2 = bootstrap(errors["belief"], errors["summary"], ordered)
        result = {"checkpoint_sha256": spec["checkpoint_sha256"], "partition": partition,
                  "RB_Gain": a1, "RB_Skill": a2, "claim": "Unseen Commodity Generalization",
                  "status": ("V1_1_RB_GENERALIZATION_PASS" if partition == "rb_test" and
                             predictive_status(a1) == predictive_status(a2) == "PASS" else
                             "SANITY_ONLY" if partition == "rb_dev" else "NOT_PASS")}
        directory = output / str(index); directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "rb_bootstrap.json", result)
        write_csv(directory / "rb_results.csv", [{"metric": "RB_Gain_ALL", **a1["all"]}, {"metric": "RB_Skill_ALL", **a2["all"]}])
        np.savez_compressed(directory / "paired_errors.npz", jepa=exports["jepa"], persistence=exports["persistence"],
                            belief_error=errors["belief"], summary_error=errors["summary"])
        write_json(directory / "paired_records.json", ordered)
        results.append(result)
    if partition == "rb_test":
        write_json(ledger, {"rb_test_consumed": True, "state": "COMPLETED", "freeze_sha256": frozen["freeze_sha256"]})
    write_json(output / "summary.json", {"partition": partition, "models": results})
    return results
