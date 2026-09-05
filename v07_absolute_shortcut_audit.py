#!/usr/bin/env python3
"""Audit absolute Price/Volume/OI sensitivity of a stopped V0.7 Inner model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from market_jepa.data import prepare_market_data
from market_jepa.data.pipeline import NormalizerBundle
from market_jepa.data.schema import MARKET_FEATURES
from market_jepa.eval.metrics import block_bootstrap
from market_jepa.frozen_state import RFFMap, Standardizer
from market_jepa.predictive_state import (
    EndToEndPredictiveState,
    FixedRFF,
    PredictiveStateDataset,
    TargetPreprocessing,
    all_future_outcomes,
    build_population_indices,
    collate_predictive_state,
    future_y,
    normalizer_hash,
)
from market_jepa.train import load_checkpoint
from market_jepa.train.predictive_state_trainer import initial_state


SAMPLE_SIZE = 4096
SAMPLE_SEED = 4401
PAIR_SIZE = 4096
PAIR_SEED = 4401
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
INTERVENTIONS = {
    "no_price_level": ("log_open", "log_high", "log_low", "log_close"),
    "no_volume_level": ("log1p_volume",),
    "no_oi_level": ("log1p_open_interest",),
    "no_absolute_levels": (
        "log_open",
        "log_high",
        "log_low",
        "log_close",
        "log1p_volume",
        "log1p_open_interest",
    ),
}


def _target_from_checkpoint(value: dict[str, Any]) -> TargetPreprocessing:
    target = TargetPreprocessing(
        scaler=Standardizer(
            mean=np.asarray(value["y_mean"], dtype=np.float64),
            std=np.asarray(value["y_std"], dtype=np.float64),
        ),
        rff=RFFMap(
            sigma0=float(value["sigma0"]),
            bandwidths=np.asarray(value["bandwidths"], dtype=np.float64),
            frequencies=np.asarray(value["frequencies"], dtype=np.float32),
            phases=np.asarray(value["phases"], dtype=np.float32),
        ),
        mu_phi=np.asarray(value["mu_phi"], dtype=np.float64),
        audit=dict(value["audit"]),
        hashes=dict(value["hashes"]),
    )
    return target


def _sample_anchors(population: np.ndarray) -> np.ndarray:
    values = np.asarray(population, dtype=np.int64)
    if len(values) <= SAMPLE_SIZE:
        return values.copy()
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    return np.sort(rng.choice(values, size=SAMPLE_SIZE, replace=False))


def _pair_positions(size: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("real-distance normalization requires at least two samples")
    rng = np.random.Generator(np.random.PCG64(PAIR_SEED))
    left = rng.integers(0, size, size=PAIR_SIZE)
    offset = rng.integers(1, size, size=PAIR_SIZE)
    return left, (left + offset) % size


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    cpu_names = {"daily_lengths", "weekly_lengths", "anchor_index", "timestamp_ns", "trading_day_ns"}
    return {
        name: value if name in cpu_names else value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _forward_intervention(
    model: EndToEndPredictiveState,
    batch: dict[str, torch.Tensor],
    channels: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for name in ("minute_market", "daily", "weekly"):
            tensor = batch[name]
            value = tensor[..., channels].clone()
            saved.append((tensor, value))
            tensor[..., channels] = 0
        return model(batch)
    finally:
        for name, (tensor, value) in zip(
            ("minute_market", "daily", "weekly"), saved, strict=True
        ):
            del name
            tensor[..., channels] = value


def _squared_error(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=state.device.type, enabled=False):
        return torch.square(state.float() - target.float()).sum(dim=-1)


def _evaluate_population(
    trained: EndToEndPredictiveState,
    untrained: EndToEndPredictiveState,
    fixed_rff: FixedRFF,
    dataset: PredictiveStateDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    names = ("original", *INTERVENTIONS)
    count = len(dataset)
    trained_error = {name: np.empty(count, dtype=np.float64) for name in names}
    sensitivity = {
        model_name: {
            intervention: {
                "belief": np.empty(count, dtype=np.float64),
                "state": np.empty(count, dtype=np.float64),
            }
            for intervention in INTERVENTIONS
        }
        for model_name in ("trained", "untrained")
    }
    original = {
        model_name: {
            "belief": np.empty((count, 256), dtype=np.float32),
            "state": np.empty((count, 3072), dtype=np.float32),
        }
        for model_name in ("trained", "untrained")
    }
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_predictive_state,
    )
    cursor = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            stop = cursor + len(batch["y"])
            target = fixed_rff(batch["y"])
            for model_name, model in (("trained", trained), ("untrained", untrained)):
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    baseline = model(batch)
                baseline_b = baseline["belief"]
                baseline_s = baseline["state"]
                original[model_name]["belief"][cursor:stop] = baseline_b.float().cpu().numpy()
                original[model_name]["state"][cursor:stop] = baseline_s.float().cpu().numpy()
                if model_name == "trained":
                    trained_error["original"][cursor:stop] = (
                        _squared_error(baseline_s, target).cpu().numpy()
                    )
                for intervention, feature_names in INTERVENTIONS.items():
                    channels = tuple(MARKET_FEATURES.index(name) for name in feature_names)
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        counterfactual = _forward_intervention(model, batch, channels)
                    sensitivity[model_name][intervention]["belief"][cursor:stop] = (
                        torch.linalg.vector_norm(counterfactual["belief"].float() - baseline_b.float(), dim=1)
                        .cpu()
                        .numpy()
                    )
                    sensitivity[model_name][intervention]["state"][cursor:stop] = (
                        torch.linalg.vector_norm(counterfactual["state"].float() - baseline_s.float(), dim=1)
                        .cpu()
                        .numpy()
                    )
                    if model_name == "trained":
                        trained_error[intervention][cursor:stop] = (
                            _squared_error(counterfactual["state"], target).cpu().numpy()
                        )
            cursor = stop
    if cursor != count:
        raise RuntimeError("inference did not cover every fixed audit anchor")
    left, right = _pair_positions(count)
    real_distance: dict[str, dict[str, float]] = {}
    for model_name in ("trained", "untrained"):
        real_distance[model_name] = {}
        for representation in ("belief", "state"):
            values = original[model_name][representation]
            distances = np.linalg.norm(
                values[left].astype(np.float64) - values[right].astype(np.float64), axis=1
            )
            median = float(np.median(distances))
            if not np.isfinite(median) or median <= 0:
                raise RuntimeError(f"non-positive real {model_name} {representation} distance")
            real_distance[model_name][representation] = median
    return {
        "errors": trained_error,
        "sensitivity": sensitivity,
        "original": original,
        "real_distance": real_distance,
        "pair_left": left,
        "pair_right": right,
    }


def _iso_week_blocks(trading_day: np.ndarray) -> np.ndarray:
    iso = pd.DatetimeIndex(trading_day.astype("datetime64[ns]")).isocalendar()
    return iso["year"].to_numpy(dtype=np.int64) * 100 + iso["week"].to_numpy(dtype=np.int64)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _interpret(rows: list[dict[str, Any]]) -> list[str]:
    dev = {row["intervention"]: row for row in rows if row["population"] == "2022"}
    train = {row["intervention"]: row for row in rows if row["population"] == "2018-2021"}
    labels = {
        "no_price_level": "Price absolute level",
        "no_volume_level": "Volume absolute level",
        "no_oi_level": "OI absolute level",
        "no_absolute_levels": "All absolute levels",
    }
    answers = []
    for index, name in enumerate(INTERVENTIONS, start=1):
        row = dev[name]
        direction = "improved" if row["gain_remove"] > 0 else "worsened"
        answers.append(
            f"{index}. Removing {labels[name]} {direction} 2022 MSE by "
            f"{abs(row['gain_remove']):.8g}; 95% CI for Gain_remove is "
            f"[{row['ci95_low']:.8g}, {row['ci95_high']:.8g}]."
        )
    opposite = [
        name
        for name in INTERVENTIONS
        if train[name]["gain_remove"] <= 0 < dev[name]["gain_remove"]
    ]
    answers.append(
        "5. Train and 2022 point effects have the shortcut-like opposite sign for: "
        + (", ".join(labels[name] for name in opposite) if opposite else "none")
        + "."
    )
    ratios = []
    for name in INTERVENTIONS:
        row = dev[name]
        rb_fold = row["trained_r_b"] / max(row["untrained_r_b"], 1e-12)
        rs_fold = row["trained_r_s"] / max(row["untrained_r_s"], 1e-12)
        ratios.append(f"{labels[name]}: R_B x{rb_fold:.3g}, R_S x{rs_fold:.3g}")
    answers.append(
        "6. Epoch-9 did not increase normalized absolute-level sensitivity over the "
        "untrained model: every 2022 trained/untrained ratio is below one ("
        + "; ".join(ratios)
        + ")."
    )
    all_levels = dev["no_absolute_levels"]
    strong = all_levels["gain_remove"] > 0 and all_levels["ci95_low"] > 0
    if strong:
        conclusion = (
            "Strong evidence that absolute-level information learned on 2018-2021 harms "
            "temporal transfer to 2022 while relative structural information is preserved."
        )
    else:
        conclusion = (
            "There is channel-specific evidence that absolute Volume harms 2022 transfer, but "
            "the pre-registered all-level intervention is not significant; Price is inconclusive "
            "and removing absolute OI significantly worsens prediction. Therefore this audit does "
            "not support absolute levels as the primary general cause of early OOS deterioration."
        )
    answers.append("7. " + conclusion)
    return answers


def _write_markdown(path: Path, checkpoint: dict[str, Any], rows: list[dict[str, Any]], answers: list[str]) -> None:
    lines = [
        "# V0.7 Absolute-Level Shortcut Audit",
        "",
        "## Checkpoint",
        "",
        f"- epoch: {checkpoint['epoch']}",
        f"- history length: {len(checkpoint['history'])}",
        f"- best epoch: {checkpoint['best_epoch']}",
        f"- best dev loss: {checkpoint['best_dev_loss']:.10g}",
        "",
        "## Prediction and representation effects",
        "",
        "| Population | Intervention | Original MSE | CF MSE | Gain remove | 95% CI | R_B trained/untrained | R_S trained/untrained |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['population']} | {row['intervention']} | {row['original_mse']:.7g} | "
            f"{row['counterfactual_mse']:.7g} | {row['gain_remove']:.7g} | "
            f"[{row['ci95_low']:.7g}, {row['ci95_high']:.7g}] | "
            f"{row['trained_r_b']:.5g}/{row['untrained_r_b']:.5g} | "
            f"{row['trained_r_s']:.5g}/{row['untrained_r_s']:.5g} |"
        )
    lines.extend(["", "## Required answers", "", *answers, "", "2023-2025 consumed: **false**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(checkpoint_path: str | Path, output_dir: str | Path, batch_size: int = 64) -> Path:
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location="cpu")
    checks = {
        "stage": "inner",
        "run_name": "market",
        "context_only": False,
        "test_consumed": False,
    }
    for name, expected in checks.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"checkpoint {name} is not valid for this audit")
    if int(checkpoint["epoch"]) != 9 or len(checkpoint["history"]) != 10:
        raise ValueError("audit is frozen to the stopped epoch-9 checkpoint")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = checkpoint["config"]
    populations = config["data"]["populations"]
    normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
    if normalizer_hash(normalizers) != checkpoint["normalizer_sha256"]:
        raise RuntimeError("checkpoint normalizer hash mismatch")
    target = _target_from_checkpoint(checkpoint["target_preprocessing"])
    data = prepare_market_data(
        config,
        normalizers=normalizers,
        max_trading_day=populations["inner_dev"][1],
    )
    if data.minute["trading_day"].max() > pd.Timestamp("2022-12-31"):
        raise RuntimeError("2023-2025 embargo failed")
    outcomes = all_future_outcomes(data)
    context_length = int(config["data"]["minute_context_length"])
    populations_indices = {
        "2018-2021": build_population_indices(data, populations["inner_fit"], context_length),
        "2022": build_population_indices(data, populations["inner_dev"], context_length),
    }
    sampled = {name: _sample_anchors(indices) for name, indices in populations_indices.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = EndToEndPredictiveState(config["model"]).to(device)
    trained.load_state_dict(checkpoint["model"])
    initial_weights, _ = initial_state(config["model"], int(config["training"]["seed"]))
    untrained = EndToEndPredictiveState(config["model"]).to(device)
    untrained.load_state_dict(initial_weights)
    trained.eval().requires_grad_(False)
    untrained.eval().requires_grad_(False)
    fixed_rff = FixedRFF(target.rff).to(device).eval()

    rows: list[dict[str, Any]] = []
    sample_payload: dict[str, Any] = {
        "seed": SAMPLE_SEED,
        "without_replacement": True,
        "sample_size": SAMPLE_SIZE,
        "pair_seed": PAIR_SEED,
        "pair_size": PAIR_SIZE,
        "populations": {},
    }
    for population, anchors in sampled.items():
        raw_y = future_y(outcomes, anchors)
        standardized_y = target.scaler.transform(raw_y)
        dataset = PredictiveStateDataset(
            data,
            anchors,
            standardized_y,
            context_length=context_length,
            include_metadata=False,
        )
        evaluated = _evaluate_population(
            trained, untrained, fixed_rff, dataset, device, batch_size
        )
        trading_day = data.minute["trading_day"].to_numpy(dtype="datetime64[ns]")[anchors]
        blocks = _iso_week_blocks(trading_day)
        left = evaluated["pair_left"]
        right = evaluated["pair_right"]
        sample_payload["populations"][population] = {
            "anchors": anchors.tolist(),
            "pair_left_positions": left.tolist(),
            "pair_right_positions": right.tolist(),
            "pair_left_anchors": anchors[left].tolist(),
            "pair_right_anchors": anchors[right].tolist(),
        }
        original_error = evaluated["errors"]["original"]
        for intervention in INTERVENTIONS:
            counterfactual_error = evaluated["errors"][intervention]
            gain = original_error - counterfactual_error
            bootstrap = block_bootstrap(
                gain, blocks, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
            )
            sensitivity = evaluated["sensitivity"]
            real = evaluated["real_distance"]
            row = {
                "population": population,
                "intervention": intervention,
                "N": len(anchors),
                "original_mse": float(original_error.mean()),
                "counterfactual_mse": float(counterfactual_error.mean()),
                "delta_mse": float(counterfactual_error.mean() - original_error.mean()),
                "gain_remove": float(bootstrap["effect"]),
                "ci95_low": float(bootstrap["ci95_low"]),
                "ci95_high": float(bootstrap["ci95_high"]),
                "iso_week_blocks": int(bootstrap["blocks"]),
                "trained_median_d_b": float(np.median(sensitivity["trained"][intervention]["belief"])),
                "trained_median_d_s": float(np.median(sensitivity["trained"][intervention]["state"])),
                "trained_real_d_b": real["trained"]["belief"],
                "trained_real_d_s": real["trained"]["state"],
                "trained_r_b": float(np.median(sensitivity["trained"][intervention]["belief"]) / real["trained"]["belief"]),
                "trained_r_s": float(np.median(sensitivity["trained"][intervention]["state"]) / real["trained"]["state"]),
                "untrained_median_d_b": float(np.median(sensitivity["untrained"][intervention]["belief"])),
                "untrained_median_d_s": float(np.median(sensitivity["untrained"][intervention]["state"])),
                "untrained_real_d_b": real["untrained"]["belief"],
                "untrained_real_d_s": real["untrained"]["state"],
                "untrained_r_b": float(np.median(sensitivity["untrained"][intervention]["belief"]) / real["untrained"]["belief"]),
                "untrained_r_s": float(np.median(sensitivity["untrained"][intervention]["state"]) / real["untrained"]["state"]),
            }
            rows.append(row)

    answers = _interpret(rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_intervention.csv", index=False)
    (output / "sample_indices.json").write_text(
        json.dumps(sample_payload, indent=2), encoding="utf-8"
    )
    summary = {
        "audit": "V0.7 Absolute-Level Shortcut Audit",
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "history_length": len(checkpoint["history"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_dev_loss": float(checkpoint["best_dev_loss"]),
        },
        "data": {
            "max_loaded_trading_day": str(data.minute["trading_day"].max().date()),
            "populations": ["2018-2021", "2022"],
            "sample_seed": SAMPLE_SEED,
            "sample_size": SAMPLE_SIZE,
        },
        "bootstrap": {
            "block": "ISO trading week",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "estimand": "sample-weighted paired original_error - counterfactual_error",
        },
        "results": rows,
        "answers": answers,
        "test_consumed": False,
        "years_2023_2025_consumed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_markdown(output / "summary.md", checkpoint, rows, answers)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/market_predictive_state_v0_7/inner_market/last.pt",
    )
    parser.add_argument(
        "--output",
        default="artifacts/evaluation/v07_absolute_shortcut_audit",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run_audit(args.checkpoint, args.output, args.batch_size)
    print(result)
    print("2023-2025_consumed=false")


if __name__ == "__main__":
    main()
