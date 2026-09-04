from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from market_jepa.frozen_state import (
    HORIZONS,
    RFFMap,
    Standardizer,
    evaluate_gates,
    fit_full_train_head,
    predict_head,
    rff_audit,
    save_json,
    select_epoch_budget,
)
from market_jepa.train import load_checkpoint
from market_jepa.train.checkpoint import sha256


def _load(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = names.difference(loaded.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
        return {name: loaded[name] for name in names}


def _load_latent(path: Path) -> dict[str, np.ndarray]:
    return _load(
        path,
        {
            "anchor_index",
            "timestamp_ns",
            "trading_day_ns",
            "z_market",
            "outcomes_h16",
            "outcomes_h64",
            "outcomes_h256",
            "checkpoint_sha256",
            "source_sha256",
            "split",
        },
    )


def _load_context(path: Path) -> dict[str, np.ndarray]:
    return _load(
        path,
        {
            "anchor_index",
            "timestamp_ns",
            "trading_day_ns",
            "z_context",
            "context_progress",
            "checkpoint_sha256",
            "source_sha256",
            "split",
            "jepa_state_sha256",
        },
    )


def _validate_pair(
    latent: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    split: str,
    checkpoint_sha256: str,
    source_sha256: str,
) -> None:
    if set(latent["split"].astype(str)) != {split}:
        raise ValueError(f"latent file is not {split}")
    if str(context["split"][0]) != split:
        raise ValueError(f"context file is not {split}")
    for data in (latent, context):
        if str(data["checkpoint_sha256"][0]) != checkpoint_sha256:
            raise ValueError(f"{split} data uses another checkpoint")
        if str(data["source_sha256"][0]) != source_sha256:
            raise ValueError(f"{split} data uses another source")
    for name in ("anchor_index", "timestamp_ns", "trading_day_ns"):
        if not np.array_equal(latent[name], context[name]):
            raise ValueError(f"{split} latent/context {name} alignment differs")


def _future_y(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([data[f"outcomes_h{horizon}"] for horizon in HORIZONS], axis=1)


def _save_parameters(
    path: Path,
    y_scaler: Standardizer,
    b_scaler: Standardizer,
    c_scaler: Standardizer,
    rff: RFFMap,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_mean=y_scaler.mean,
        y_std=y_scaler.std,
        b_mean=b_scaler.mean,
        b_std=b_scaler.std,
        c_mean=c_scaler.mean,
        c_std=c_scaler.std,
        sigma0=np.asarray([rff.sigma0]),
        bandwidths=rff.bandwidths,
        rff_frequencies=rff.frequencies,
        rff_phases=rff.phases,
    )


def _report_markdown(report: dict[str, Any]) -> str:
    gate1 = report["gate1"]
    gate2 = report["gate2"]
    information = gate1["delta_information"]
    geometry = gate2["delta_geometry"]
    return "\n".join(
        [
            "# V0.7-Frozen Predictive-State Candidate",
            "",
            "## Kernel/RFF audit",
            "",
            f"- sigma0 = {report['kernel_rff_audit']['sigma0']:.10g}",
            f"- RMSE = {report['kernel_rff_audit']['rmse']:.10g}",
            f"- MAE = {report['kernel_rff_audit']['mae']:.10g}",
            f"- correlation = {report['kernel_rff_audit']['correlation']:.10g}",
            "",
            "## Inner training",
            "",
            f"- B head selected epoch = {report['inner_training']['b_selected_epoch']}",
            f"- Context head selected epoch = {report['inner_training']['context_selected_epoch']}",
            "",
            "## Gate 1 — Conditional Distribution Information",
            "",
            f"- B error = {gate1['b_error']:.10g}",
            f"- Context error = {gate1['context_error']:.10g}",
            f"- Unconditional error = {gate1['unconditional_error']:.10g}",
            f"- Delta information = {information['effect']:.10g}",
            f"- 95% CI = [{information['ci95'][0]:.10g}, {information['ci95'][1]:.10g}]",
            f"- Delta unconditional = {gate1['delta_unconditional']['effect']:.10g}",
            f"- Sanity 95% CI = [{gate1['delta_unconditional']['ci95'][0]:.10g}, {gate1['delta_unconditional']['ci95'][1]:.10g}]",
            f"- {'PASS' if gate1['pass'] else 'FAIL'}",
            "",
            "## Gate 2 — Predictive-State Geometry",
            "",
            f"- State-kNN error = {gate2['state_knn_error']:.10g}",
            f"- Context-kNN error = {gate2['context_knn_error']:.10g}",
            f"- Random error = {gate2['random_error']:.10g}",
            f"- Delta geometry = {geometry['effect']:.10g}",
            f"- 95% CI = [{geometry['ci95'][0]:.10g}, {geometry['ci95'][1]:.10g}]",
            f"- Delta random = {gate2['delta_random']['effect']:.10g}",
            f"- Sanity 95% CI = [{gate2['delta_random']['ci95'][0]:.10g}, {gate2['delta_random']['ci95'][1]:.10g}]",
            f"- {'PASS' if gate2['pass'] else 'FAIL'}",
            "",
            f"Overall: **{report['overall']}**",
            "",
            "Test consumed = **false**",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V0.7-Frozen predictive-state experiment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--context-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation/market_jepa_v0_7_frozen",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    result_path = output_dir / "frozen_predictive_state.json"
    report_path = output_dir / "frozen_predictive_state.md"
    if result_path.exists() or report_path.exists():
        raise SystemExit("V0.7-Frozen result already exists; refusing to overwrite")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = sha256(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint_path.name != "last.pt" or int(checkpoint.get("epoch", -1)) != 49:
        raise SystemExit("V0.7-Frozen requires epoch 49 last.pt")
    source_sha256 = str(checkpoint["source_sha256"])
    device = torch.device(args.device)

    # Formal Validation is deliberately not opened until both epoch budgets
    # have been selected exclusively from the Train period and both heads have
    # been refit on full Train.
    train = _load_latent(Path(args.latents_dir) / "train.npz")
    train_context = _load_context(Path(args.context_dir) / "context_train.npz")
    _validate_pair(train, train_context, "train", checkpoint_sha256, source_sha256)
    train_y_raw = _future_y(train)
    train_b_raw = train["z_market"]
    train_c_raw = np.concatenate(
        [train_context["z_context"], train_context["context_progress"]], axis=1
    )

    y_scaler = Standardizer.fit(train_y_raw)
    b_scaler = Standardizer.fit(train_b_raw)
    c_scaler = Standardizer.fit(train_c_raw)
    train_y = y_scaler.transform(train_y_raw)
    train_b = b_scaler.transform(train_b_raw)
    train_c = c_scaler.transform(train_c_raw)
    rff, pair_left, pair_right = RFFMap.fit(train_y)
    audit = rff_audit(train_y, rff, pair_left, pair_right)

    b_selected_epoch, b_history = select_epoch_budget(
        train_b, train_y, train["trading_day_ns"], rff, device
    )
    c_selected_epoch, c_history = select_epoch_budget(
        train_c, train_y, train["trading_day_ns"], rff, device
    )
    b_head = fit_full_train_head(train_b, train_y, rff, b_selected_epoch, device)
    c_head = fit_full_train_head(train_c, train_y, rff, c_selected_epoch, device)

    cache_dir = output_dir / "cache"
    train_b_state = predict_head(b_head, train_b, cache_dir / "state_b_train.npy", device)
    train_c_state = predict_head(c_head, train_c, cache_dir / "state_c_train.npy", device)

    validation = _load_latent(Path(args.latents_dir) / "validation.npz")
    validation_context = _load_context(
        Path(args.context_dir) / "context_validation.npz"
    )
    _validate_pair(
        validation,
        validation_context,
        "validation",
        checkpoint_sha256,
        source_sha256,
    )
    validation_y = y_scaler.transform(_future_y(validation))
    validation_b = b_scaler.transform(validation["z_market"])
    validation_c = c_scaler.transform(
        np.concatenate(
            [validation_context["z_context"], validation_context["context_progress"]],
            axis=1,
        )
    )
    validation_b_state = predict_head(
        b_head, validation_b, cache_dir / "state_b_validation.npy", device
    )
    validation_c_state = predict_head(
        c_head, validation_c, cache_dir / "state_c_validation.npy", device
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    b_head.to("cpu")
    c_head.to("cpu")
    torch.save(b_head.state_dict(), output_dir / "b_head.pt")
    torch.save(c_head.state_dict(), output_dir / "context_head.pt")
    del b_head, c_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gates = evaluate_gates(
        train_y,
        validation_y,
        train_b_state,
        validation_b_state,
        train_c_state,
        validation_c_state,
        train["timestamp_ns"],
        validation["timestamp_ns"],
        validation["trading_day_ns"],
        rff,
        cache_dir,
        device,
    )
    if sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("frozen Market-JEPA checkpoint changed during V0.7-Frozen")

    _save_parameters(output_dir / "frozen_parameters.npz", y_scaler, b_scaler, c_scaler, rff)
    save_json(b_history, output_dir / "inner_b_history.json")
    save_json(c_history, output_dir / "inner_context_history.json")
    report: dict[str, Any] = {
        "experiment": "V0.7-Frozen",
        "checkpoint_epoch": 49,
        "checkpoint_sha256": checkpoint_sha256,
        "kernel_rff_audit": audit,
        "inner_training": {
            "inner_fit": "2018-2021",
            "inner_dev": "2022",
            "b_selected_epoch": b_selected_epoch,
            "b_full_train_epochs": b_selected_epoch + 1,
            "context_selected_epoch": c_selected_epoch,
            "context_full_train_epochs": c_selected_epoch + 1,
        },
        **gates,
    }
    save_json(report, result_path)
    report_path.write_text(_report_markdown(report), encoding="utf-8")
    print(result_path)
    print(report_path)
    print(f"overall={report['overall']}")
    print("test_consumed=false")


if __name__ == "__main__":
    main()
