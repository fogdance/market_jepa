from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from market_jepa.data import prepare_market_data
from market_jepa.data.preflight import file_sha256 as source_file_sha256
from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.predictive_state import (
    EndToEndPredictiveState,
    FixedRFF,
    PredictiveStateDataset,
    all_future_outcomes,
    build_population_indices,
    collate_predictive_state,
    context_only_batch,
    evaluate_development,
    export_states,
    file_sha256,
    fit_target_preprocessing,
    future_y,
    load_protocol_config,
    state_loss,
)
from market_jepa.train import configure_determinism, load_checkpoint
from market_jepa.train.predictive_state_trainer import PredictiveStateRun, initial_state


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], check=True, text=True, capture_output=True
    ).stdout.strip()


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_regression_tests() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise SystemExit(f"Formal V0.7 pytest gate failed:\n{output}")
    matches = re.findall(r"(\d+) passed", output)
    if not matches:
        raise SystemExit(f"Formal V0.7 could not parse pytest result:\n{output}")
    return f"{matches[-1]} passed"


def _markdown(report: dict[str, Any]) -> str:
    inner = report["inner"]
    audit = report["final_target_audit"]
    gate1 = report["gate1"]
    gate2 = report["gate2"]
    return "\n".join(
        [
            "# V0.7 End-to-End Predictive-State",
            "",
            "## Inner",
            "",
            f"- Market selected epoch = {inner['market_selected_epoch']}",
            f"- Context selected epoch = {inner['context_selected_epoch']}",
            f"- Market budget hit ceiling = {str(inner['market_budget_hit_ceiling']).lower()}",
            f"- Context budget hit ceiling = {str(inner['context_budget_hit_ceiling']).lower()}",
            "",
            "## Final target audit",
            "",
            f"- sigma0 = {audit['sigma0']:.16g}",
            f"- RFF RMSE = {audit['rmse']:.16g}",
            f"- RFF MAE = {audit['mae']:.16g}",
            f"- RFF correlation = {audit['correlation']:.16g}",
            "",
            "## Gate 1",
            "",
            f"- Market error = {gate1['market_error']:.10g}",
            f"- Context error = {gate1['context_error']:.10g}",
            f"- Unconditional error = {gate1['unconditional_error']:.10g}",
            f"- Delta_info = {gate1['delta_info']['effect']:.10g}",
            f"- 95% CI = [{gate1['delta_info']['ci95'][0]:.10g}, {gate1['delta_info']['ci95'][1]:.10g}]",
            f"- Delta_info = {'PASS' if gate1['delta_info']['pass'] else 'FAIL'}",
            f"- Delta_unconditional = {gate1['delta_unconditional']['effect']:.10g}",
            f"- 95% CI = [{gate1['delta_unconditional']['ci95'][0]:.10g}, {gate1['delta_unconditional']['ci95'][1]:.10g}]",
            f"- Delta_unconditional = {'PASS' if gate1['delta_unconditional']['pass'] else 'FAIL'}",
            "",
            "## Gate 2",
            "",
            f"- Market-kNN error = {gate2['market_knn_error']:.10g}",
            f"- Context-kNN error = {gate2['context_knn_error']:.10g}",
            f"- Random error = {gate2['random_error']:.10g}",
            f"- Delta_geometry = {gate2['delta_geometry']['effect']:.10g}",
            f"- 95% CI = [{gate2['delta_geometry']['ci95'][0]:.10g}, {gate2['delta_geometry']['ci95'][1]:.10g}]",
            f"- Delta_geometry = {'PASS' if gate2['delta_geometry']['pass'] else 'FAIL'}",
            f"- Delta_random = {gate2['delta_random']['effect']:.10g}",
            f"- 95% CI = [{gate2['delta_random']['ci95'][0]:.10g}, {gate2['delta_random']['ci95'][1]:.10g}]",
            f"- Delta_random = {'PASS' if gate2['delta_random']['pass'] else 'FAIL'}",
            "",
            f"Overall: **{report['overall']}**",
            "",
            "Test consumed = **false**",
            "",
            f"pytest = {report.get('pytest', 'pending')}",
            "",
        ]
    )


def _make_dataset(
    data,
    indices: np.ndarray,
    outcomes: dict[int, np.ndarray],
    target,
    context_length: int,
) -> PredictiveStateDataset:
    raw_y = future_y(outcomes, indices)
    return PredictiveStateDataset(
        data, indices, target.scaler.transform(raw_y), context_length=context_length
    )


def _run_one(
    *,
    run_name: str,
    stage: str,
    config: dict[str, Any],
    train_dataset: PredictiveStateDataset,
    dev_dataset: PredictiveStateDataset | None,
    target,
    normalizers,
    initial_model_state,
    initial_model_hash: str,
    output_dir: Path,
    source_path: Path,
    source_hash: str,
    protocol_path: Path,
    protocol_hash: str,
    epochs: int,
    device: torch.device,
    implementation,
    run_metadata: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise RuntimeError(f"{output_dir} already exists; use --resume or choose a clean run")
    trainer = PredictiveStateRun(
        run_name=run_name,
        stage=stage,
        context_only=run_name == "context",
        model_config=config["model"],
        training_config=config["training"],
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        target=target,
        normalizers=normalizers,
        initial_model_state=initial_model_state,
        initial_model_hash=initial_model_hash,
        output_dir=output_dir,
        source_path=source_path,
        source_sha256=source_hash,
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        config=config,
        epochs_to_run=epochs,
        device=device,
        implementation=implementation,
        run_metadata=run_metadata,
    )
    summary = trainer.fit(resume=resume)
    del trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def formal(config: dict[str, Any], resume: bool) -> Path:
    dirty = _git(["status", "--porcelain", "--untracked-files=all"])
    if dirty:
        raise SystemExit("Formal V0.7 requires a clean git worktree")
    if not torch.cuda.is_available():
        raise SystemExit("Formal V0.7 requires CUDA")
    pytest_result = _run_regression_tests()
    protocol_path = Path(config["protocol_path"])
    protocol_hash = file_sha256(protocol_path)
    source_path = Path(config["data"]["csv_path"])
    source_hash = source_file_sha256(source_path)
    implementation = implementation_manifest()
    run_metadata = {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": "clean",
        "implementation_sha256": manifest_sha256(implementation),
        "pytest": pytest_result,
    }
    inner_metadata = {
        **run_metadata,
        "preprocessing_fit_population": "inner_fit",
        "train_population": "inner_fit",
        "checkpoint_selection_population": "inner_dev",
        "development_used": False,
    }
    device = torch.device("cuda")
    populations = config["data"]["populations"]
    context_length = int(config["data"]["minute_context_length"])
    checkpoint_root = Path(config["artifacts"]["checkpoint_dir"])
    evaluation_root = Path(config["artifacts"]["evaluation_dir"])
    if (evaluation_root / "development_summary.json").exists():
        raise SystemExit("V0.7 Development result already exists; refusing to overwrite")

    inner_data = prepare_market_data(
        config,
        normalizer_fit_range=tuple(populations["inner_fit"]),
        max_trading_day=populations["inner_dev"][1],
    )
    inner_outcomes = all_future_outcomes(inner_data)
    inner_fit_indices = build_population_indices(
        inner_data, populations["inner_fit"], context_length
    )
    inner_dev_indices = build_population_indices(
        inner_data, populations["inner_dev"], context_length
    )
    inner_target = fit_target_preprocessing(
        future_y(inner_outcomes, inner_fit_indices)
    )
    inner_fit = _make_dataset(
        inner_data, inner_fit_indices, inner_outcomes, inner_target, context_length
    )
    inner_dev = _make_dataset(
        inner_data, inner_dev_indices, inner_outcomes, inner_target, context_length
    )
    inner_initial, inner_initial_hash = initial_state(
        config["model"], int(config["training"]["seed"])
    )
    inner_market = _run_one(
        run_name="market",
        stage="inner",
        config=config,
        train_dataset=inner_fit,
        dev_dataset=inner_dev,
        target=inner_target,
        normalizers=inner_data.normalizers,
        initial_model_state=inner_initial,
        initial_model_hash=inner_initial_hash,
        output_dir=checkpoint_root / "inner_market",
        source_path=source_path,
        source_hash=source_hash,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        epochs=100,
        device=device,
        implementation=implementation,
        run_metadata=inner_metadata,
        resume=resume,
    )
    inner_context = _run_one(
        run_name="context",
        stage="inner",
        config=config,
        train_dataset=inner_fit,
        dev_dataset=inner_dev,
        target=inner_target,
        normalizers=inner_data.normalizers,
        initial_model_state=inner_initial,
        initial_model_hash=inner_initial_hash,
        output_dir=checkpoint_root / "inner_context",
        source_path=source_path,
        source_hash=source_hash,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        epochs=100,
        device=device,
        implementation=implementation,
        run_metadata=inner_metadata,
        resume=resume,
    )
    del inner_fit, inner_dev, inner_data, inner_outcomes, inner_target, inner_initial
    gc.collect()

    final_data = prepare_market_data(
        config,
        normalizer_fit_range=tuple(populations["final_train"]),
        max_trading_day=populations["final_train"][1],
    )
    final_outcomes = all_future_outcomes(final_data)
    final_indices = build_population_indices(
        final_data, populations["final_train"], context_length
    )
    final_target = fit_target_preprocessing(
        future_y(final_outcomes, final_indices), final_regression=True
    )
    final_train = _make_dataset(
        final_data, final_indices, final_outcomes, final_target, context_length
    )
    final_initial, final_initial_hash = initial_state(
        config["model"], int(config["training"]["seed"])
    )
    final_market = _run_one(
        run_name="market",
        stage="final",
        config=config,
        train_dataset=final_train,
        dev_dataset=None,
        target=final_target,
        normalizers=final_data.normalizers,
        initial_model_state=final_initial,
        initial_model_hash=final_initial_hash,
        output_dir=checkpoint_root / "final_market",
        source_path=source_path,
        source_hash=source_hash,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        epochs=int(inner_market["selected_budget"]),
        device=device,
        implementation=implementation,
        run_metadata={
            **run_metadata,
            "preprocessing_fit_population": "final_train",
            "train_population": "final_train",
            "checkpoint_selection_population": None,
            "development_used": False,
            "selected_from_inner_epoch": inner_market["selected_epoch"],
        },
        resume=resume,
    )
    final_context = _run_one(
        run_name="context",
        stage="final",
        config=config,
        train_dataset=final_train,
        dev_dataset=None,
        target=final_target,
        normalizers=final_data.normalizers,
        initial_model_state=final_initial,
        initial_model_hash=final_initial_hash,
        output_dir=checkpoint_root / "final_context",
        source_path=source_path,
        source_hash=source_hash,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        epochs=int(inner_context["selected_budget"]),
        device=device,
        implementation=implementation,
        run_metadata={
            **run_metadata,
            "preprocessing_fit_population": "final_train",
            "train_population": "final_train",
            "checkpoint_selection_population": None,
            "development_used": False,
            "selected_from_inner_epoch": inner_context["selected_epoch"],
        },
        resume=resume,
    )
    del final_train, final_data, final_outcomes, final_initial
    gc.collect()
    torch.cuda.empty_cache()

    # Development is first constructed only after both Final runs complete.
    development_data = prepare_market_data(
        config,
        normalizers=final_target_normalizers(
            checkpoint_root / "final_market" / "last.pt", expected_run_name="market"
        ),
        max_trading_day=populations["development"][1],
    )
    development_outcomes = all_future_outcomes(development_data)
    train_indices = build_population_indices(
        development_data, populations["final_train"], context_length
    )
    development_indices = build_population_indices(
        development_data, populations["development"], context_length
    )
    train_dataset = _make_dataset(
        development_data, train_indices, development_outcomes, final_target, context_length
    )
    development_dataset = _make_dataset(
        development_data,
        development_indices,
        development_outcomes,
        final_target,
        context_length,
    )
    cache = evaluation_root / "cache"
    market_model = _load_final_model(
        checkpoint_root / "final_market" / "last.pt", config, device, "market"
    )
    market_train_state = export_states(
        market_model, train_dataset, cache / "market_train.npy", context_only=False, device=device
    )
    market_dev_state = export_states(
        market_model,
        development_dataset,
        cache / "market_development.npy",
        context_only=False,
        device=device,
    )
    del market_model
    torch.cuda.empty_cache()
    context_model = _load_final_model(
        checkpoint_root / "final_context" / "last.pt", config, device, "context"
    )
    context_train_state = export_states(
        context_model,
        train_dataset,
        cache / "context_train.npy",
        context_only=True,
        device=device,
    )
    context_dev_state = export_states(
        context_model,
        development_dataset,
        cache / "context_development.npy",
        context_only=True,
        device=device,
    )
    del context_model
    torch.cuda.empty_cache()
    train_y = final_target.scaler.transform(future_y(development_outcomes, train_indices))
    development_y = final_target.scaler.transform(
        future_y(development_outcomes, development_indices)
    )
    gates = evaluate_development(
        train_state=market_train_state,
        development_state=market_dev_state,
        train_context_state=context_train_state,
        development_context_state=context_dev_state,
        train_y=train_y,
        development_y=development_y,
        train_anchor=train_indices,
        train_timestamp=development_data.minute.iloc[train_indices]["timestamp"].astype("int64").to_numpy(),
        development_timestamp=development_data.minute.iloc[development_indices]["timestamp"].astype("int64").to_numpy(),
        development_trading_day=development_data.minute.iloc[development_indices]["trading_day"].astype("int64").to_numpy(),
        target=final_target,
        cache_dir=cache,
        device=device,
    )
    report = {
        "protocol_version": "0.7.1",
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hash,
        "implementation_sha256": manifest_sha256(implementation),
        "inner": {
            "market_selected_epoch": inner_market["selected_epoch"],
            "context_selected_epoch": inner_context["selected_epoch"],
            "market_budget_hit_ceiling": inner_market["budget_hit_ceiling"],
            "context_budget_hit_ceiling": inner_context["budget_hit_ceiling"],
        },
        "final": {"market": final_market, "context": final_context},
        "final_target_audit": {
            "sigma0": final_target.rff.sigma0,
            **{key: final_target.audit[key] for key in ("rmse", "mae", "correlation")},
            "hashes": final_target.hashes,
        },
        **gates,
        "pytest": pytest_result,
        "test_consumed": False,
    }
    result_path = evaluation_root / "development_summary.json"
    _atomic_json(report, result_path)
    (evaluation_root / "development_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(result_path)
    print(f"overall={report['overall']}")
    print("test_consumed=false")
    return result_path


def final_target_normalizers(checkpoint_path: Path, expected_run_name: str):
    from market_jepa.data.pipeline import NormalizerBundle

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if (
        checkpoint["stage"] != "final"
        or checkpoint["run_name"] != expected_run_name
        or checkpoint["test_consumed"] is not False
        or checkpoint["scheduler_horizon_epochs"] != 100
        or checkpoint["epoch"] != checkpoint["epochs_to_run"] - 1
    ):
        raise RuntimeError("invalid Final checkpoint")
    return NormalizerBundle.from_dict(checkpoint["normalizers"])


def _load_final_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
    expected_run_name: str,
) -> EndToEndPredictiveState:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if (
        checkpoint["stage"] != "final"
        or checkpoint["run_name"] != expected_run_name
        or checkpoint["test_consumed"] is not False
        or checkpoint["scheduler_horizon_epochs"] != 100
        or checkpoint["epoch"] != checkpoint["epochs_to_run"] - 1
    ):
        raise RuntimeError("Development requires a Final V0.7 checkpoint")
    model = EndToEndPredictiveState(config["model"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def smoke(config: dict[str, Any], device: torch.device) -> None:
    populations = config["data"]["populations"]
    data = prepare_market_data(
        config,
        normalizer_fit_range=tuple(populations["inner_fit"]),
        max_trading_day=populations["inner_dev"][1],
    )
    outcomes = all_future_outcomes(data)
    indices = build_population_indices(data, populations["inner_fit"])[-64:]
    target = fit_target_preprocessing(future_y(outcomes, indices))
    dataset = _make_dataset(data, indices, outcomes, target, 512)
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_predictive_state)))
    model_config = deepcopy(config["model"])
    model_config.update(
        minute_d_model=32,
        latent_dim=32,
        minute_layers=1,
        minute_heads=4,
        minute_ffn_dim=64,
        time_hidden=8,
        daily_hidden=16,
        weekly_hidden=16,
        recurrent_layers=1,
        fusion_hidden=64,
        dropout=0.0,
    )
    configure_determinism(42)
    for context in (False, True):
        model = EndToEndPredictiveState(model_config).to(device).train()
        fixed = FixedRFF(target.rff).to(device)
        moved = {
            key: value if key in {"daily_lengths", "weekly_lengths", "anchor_index", "timestamp_ns", "trading_day_ns"} else value.to(device)
            for key, value in batch.items()
        }
        if context:
            moved = context_only_batch(moved)
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            state = model(moved)["state"]
        loss, _ = state_loss(state, fixed(moved["y"]))
        loss.backward()
        if not all(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("smoke loss did not reach every model parameter")
    print(json.dumps({"smoke": "PASS", "device": str(device), "test_consumed": False}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market_predictive_state_v0_7.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()
    if sum((args.smoke, args.cuda_smoke)) > 1:
        parser.error("choose at most one smoke mode")
    config = load_protocol_config(args.config)
    if args.smoke:
        smoke(config, torch.device("cpu"))
    elif args.cuda_smoke:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA smoke requested but CUDA is unavailable")
        smoke(config, torch.device("cuda"))
    else:
        formal(config, args.resume)


if __name__ == "__main__":
    main()
