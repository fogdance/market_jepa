from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from market_jepa.data import (
    CONTEXT_FEATURES,
    MARKET_FEATURES,
    MarketDataset,
    NormalizerBundle,
    build_split_indices,
    collate_market_batch,
    prepare_market_data,
)
from market_jepa.eval.metrics import block_bootstrap
from market_jepa.model import MarketJEPA
from market_jepa.train.checkpoint import load_checkpoint, save_checkpoint, sha256


EXPECTED_SOURCE_SHA256 = "d6887df95e0a8f0b165e87b190d3af438ceb23445702c35f191e5d546c03a688"
HORIZONS = (16, 64, 256)
OUTCOMES = ("Return", "MFE", "MAE", "RV")
QUANTILES = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
STATE_DIM = 108
CONTEXT_DIM = len(CONTEXT_FEATURES) + 2
CACHE_NAMES = ("inner_fit", "inner_dev", "final_train", "development")


class FrozenJEPACDFHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ContextCDFHead(nn.Module):
    def __init__(self, context_dim: int = CONTEXT_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_protocol(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "0.8.0":
        raise ValueError("V0.8 protocol version drift")
    if config["checkpoint"]["required_epoch"] != 49:
        raise ValueError("V0.8 requires epoch49")
    if config["checkpoint"]["required_design_version"] != "0.6.1":
        raise ValueError("V0.8 requires V0 design 0.6.1")
    if config["checkpoint"]["required_source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("V0.8 source hash drift")
    if tuple(config["cdf"]["horizons"]) != HORIZONS:
        raise ValueError("V0.8 horizon drift")
    if tuple(config["cdf"]["outcomes"]) != OUTCOMES:
        raise ValueError("V0.8 outcome drift")
    if not np.allclose(np.asarray(config["cdf"]["quantiles"]), QUANTILES, rtol=0.0, atol=1e-15):
        raise ValueError("V0.8 quantile drift")
    if config["cdf"]["state_dim"] != STATE_DIM:
        raise ValueError("V0.8 state dimension drift")
    expected_training = {
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0001,
        "batch_size": 1024,
        "max_epochs": 50,
        "gradient_clip_norm": 1.0,
        "scheduler": "none",
        "amp": False,
    }
    if config["training"] != expected_training:
        raise ValueError("V0.8 training protocol drift")
    if config["data"]["max_trading_day"] != "2024-12-31":
        raise ValueError("V0.8 entrypoint must embargo 2025")
    return config


def state_schema() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    index = 0
    for horizon in HORIZONS:
        for outcome_index, outcome in enumerate(OUTCOMES):
            for quantile in QUANTILES:
                result.append(
                    {
                        "index": index,
                        "name": f"H{horizon}_{outcome}_q{int(round(quantile * 100)):02d}",
                        "horizon": horizon,
                        "outcome": outcome,
                        "outcome_index": outcome_index,
                        "quantile": float(quantile),
                    }
                )
                index += 1
    if len(result) != STATE_DIM:
        raise AssertionError("CDF schema must contain 108 components")
    return result


def fit_thresholds(y: np.ndarray, fit_name: str, date_range: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("Y must have shape [N,12]")
    thresholds = np.quantile(values, QUANTILES, axis=0, method="linear").T
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for outcome_index in range(12):
        horizon = HORIZONS[outcome_index // len(OUTCOMES)]
        outcome = OUTCOMES[outcome_index % len(OUTCOMES)]
        unique = int(len(np.unique(thresholds[outcome_index])))
        if unique < len(QUANTILES):
            warnings.append(f"H{horizon}_{outcome} has {unique}/9 unique thresholds")
        entries.append(
            {
                "horizon": horizon,
                "outcome": outcome,
                "thresholds": thresholds[outcome_index].tolist(),
                "unique_thresholds": unique,
            }
        )
    metadata = {
        "fit_population": fit_name,
        "fit_date_range": date_range,
        "fit_samples": len(values),
        "method": "numpy.quantile(method='linear')",
        "quantiles": QUANTILES.tolist(),
        "outcomes": entries,
        "warnings": warnings,
        "threshold_sha256": canonical_sha256(thresholds.tolist()),
    }
    return thresholds, metadata


def binary_targets(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or threshold_values.shape != (12, 9):
        raise ValueError("invalid Y or threshold shape")
    result = (values[:, :, None] <= threshold_values[None, :, :]).reshape(len(values), STATE_DIM)
    return result.astype(np.float32)


def unconditional_probabilities() -> np.ndarray:
    return np.tile(QUANTILES.astype(np.float32), 12)


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _save_metadata(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def _context_vectors(data: Any, anchors: np.ndarray) -> np.ndarray:
    result = np.concatenate(
        [
            data.minute_context[anchors],
            data.daily_partial_context[anchors],
            data.weekly_partial_context[anchors],
        ],
        axis=1,
    ).astype(np.float32)
    if result.shape != (len(anchors), CONTEXT_DIM):
        raise AssertionError("context baseline vector schema changed")
    return result


def _load_export(path: Path, expected_checkpoint_sha256: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as export:
        if str(export["checkpoint_sha256"][0]) != expected_checkpoint_sha256:
            raise RuntimeError(f"latent export checkpoint mismatch: {path}")
        if str(export["source_sha256"][0]) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(f"latent export source mismatch: {path}")
        return {
            "anchor": export["anchor_index"].astype(np.int64),
            "timestamp_ns": export["timestamp_ns"].astype(np.int64),
            "trading_day_ns": export["trading_day_ns"].astype(np.int64),
            "z": export["z_market"].astype(np.float32),
            "y": np.concatenate(
                [export[f"outcomes_h{horizon}"].astype(np.float32) for horizon in HORIZONS],
                axis=1,
            ),
        }


def build_latent_cache(
    config: dict[str, Any], checkpoint: dict[str, Any], checkpoint_hash: str
) -> tuple[Path, Any]:
    cache_dir = Path(config["data"]["cache_dir"])
    latent_dir = Path(config["data"]["latent_export_dir"])
    v0_config = checkpoint["config"]
    normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
    data = prepare_market_data(
        v0_config,
        normalizers=normalizers,
        max_trading_day=config["data"]["max_trading_day"],
    )
    maximum = pd.Timestamp(data.minute["trading_day"].max())
    if maximum > pd.Timestamp("2024-12-31"):
        raise AssertionError("2025 data was materialized")

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    required = [cache_dir / f"{name}_{kind}.npy" for name in CACHE_NAMES for kind in ("z", "y", "context")]
    required += [cache_dir / f"{name}_metadata.npz" for name in CACHE_NAMES]
    if manifest_path.exists() and all(path.exists() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("checkpoint_sha256") == checkpoint_hash
            and manifest.get("source_sha256") == EXPECTED_SOURCE_SHA256
            and manifest.get("protocol_version") == config["protocol_version"]
            and manifest.get("context_schema")
            == [*CONTEXT_FEATURES, "daily_source_bar_count", "weekly_source_bar_count"]
            and manifest.get("test_consumed") is False
        ):
            return cache_dir, data

    train = _load_export(latent_dir / "train.npz", checkpoint_hash)
    development = _load_export(latent_dir / "validation.npz", checkpoint_hash)
    expected_train = build_split_indices(data, v0_config, "train")
    expected_development = build_split_indices(data, v0_config, "validation")
    np.testing.assert_array_equal(train["anchor"], expected_train)
    np.testing.assert_array_equal(development["anchor"], expected_development)

    train_days = train["trading_day_ns"].astype("datetime64[ns]")
    inner_fit_mask = train_days <= np.datetime64("2021-12-31")
    inner_dev_mask = (train_days >= np.datetime64("2022-01-01")) & (
        train_days <= np.datetime64("2022-12-31")
    )
    if np.any(inner_fit_mask & inner_dev_mask) or not np.all(inner_fit_mask | inner_dev_mask):
        raise AssertionError("inner population partition is incomplete")

    populations = {
        "inner_fit": {key: value[inner_fit_mask] for key, value in train.items()},
        "inner_dev": {key: value[inner_dev_mask] for key, value in train.items()},
        "final_train": train,
        "development": development,
    }
    manifest_populations: dict[str, Any] = {}
    for name, population in populations.items():
        anchors = population["anchor"]
        context = _context_vectors(data, anchors)
        _save_npy(cache_dir / f"{name}_z.npy", population["z"])
        _save_npy(cache_dir / f"{name}_y.npy", population["y"])
        _save_npy(cache_dir / f"{name}_context.npy", context)
        iso_week = data.minute.iloc[anchors]["iso_key"].to_numpy(dtype=np.int64)
        _save_metadata(
            cache_dir / f"{name}_metadata.npz",
            anchor_index=anchors,
            timestamp_ns=population["timestamp_ns"],
            trading_day_ns=population["trading_day_ns"],
            iso_week=iso_week,
        )
        days = population["trading_day_ns"].astype("datetime64[ns]")
        manifest_populations[name] = {
            "samples": len(anchors),
            "first_trading_day": str(days.min().astype("datetime64[D]")),
            "last_trading_day": str(days.max().astype("datetime64[D]")),
        }
    manifest = {
        "protocol_version": config["protocol_version"],
        "checkpoint_sha256": checkpoint_hash,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_exports": {
            "train": str(latent_dir / "train.npz"),
            "development": str(latent_dir / "validation.npz"),
        },
        "context_schema": [*CONTEXT_FEATURES, "daily_source_bar_count", "weekly_source_bar_count"],
        "populations": manifest_populations,
        "max_loaded_trading_day": str(maximum.date()),
        "dtype": "float32",
        "test_consumed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return cache_dir, data


def _move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_batch(item, device) for key, item in value.items()}
    return value


def load_frozen_v0(checkpoint: dict[str, Any], device: torch.device) -> MarketJEPA:
    v0_config = checkpoint["config"]
    model = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), list(HORIZONS), v0_config["model"]
    )
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("V0 JEPA was not fully frozen")
    return model


def validate_cached_latents(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    cache_dir: Path,
    data: Any,
    device: torch.device,
) -> dict[str, Any]:
    count = int(config["evaluation"]["cache_consistency_samples"])
    if count != 128:
        raise ValueError("frozen V0.8 consistency audit requires exactly 128 samples")
    rng = np.random.Generator(np.random.PCG64(config["evaluation"]["cache_consistency_seed"]))
    per_population = count // 2
    selected_anchors: list[np.ndarray] = []
    expected_values: list[np.ndarray] = []
    for name in ("final_train", "development"):
        with np.load(cache_dir / f"{name}_metadata.npz", allow_pickle=False) as metadata:
            anchors = metadata["anchor_index"]
        z = np.load(cache_dir / f"{name}_z.npy", mmap_mode="r")
        # The original formal export used chronological batches of 64. Recompute
        # randomly selected complete export batches so packed-GRU numerical order
        # is identical; arbitrary regrouping changes CUDA GRU rounding (~1e-4)
        # even though the cached float32 values themselves are exact.
        batch_start = int(rng.integers(0, len(anchors) // per_population)) * per_population
        positions = np.arange(batch_start, batch_start + per_population, dtype=np.int64)
        selected_anchors.append(anchors[positions])
        expected_values.append(np.asarray(z[positions]))
    anchors = np.concatenate(selected_anchors)
    expected = np.concatenate(expected_values)

    v0_config = checkpoint["config"]
    model = load_frozen_v0(checkpoint, device)
    if any(parameter.requires_grad for parameter in model.online.parameters()):
        raise AssertionError("V0 encoder was not frozen")
    dataset = MarketDataset(data, v0_config, "cache_consistency", anchors)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_market_batch)
    direct: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            direct.append(model.online(_move_batch(batch, device)).cpu().numpy())
    direct_array = np.concatenate(direct)
    maximum = float(np.max(np.abs(direct_array - expected)))
    tolerance = float(config["evaluation"]["cache_max_abs_error"])
    if maximum > tolerance:
        raise RuntimeError(f"cached latent mismatch: max_abs_error={maximum} > {tolerance}")
    if any(parameter.grad is not None for parameter in model.online.parameters()):
        raise AssertionError("frozen encoder accumulated gradients")
    return {
        "samples": len(anchors),
        "max_abs_error": maximum,
        "tolerance": tolerance,
        "passed": True,
        "encoder_requires_grad": False,
        "encoder_gradients_none": True,
    }


def _load_cache(cache_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    z = np.load(cache_dir / f"{name}_z.npy", mmap_mode="r")
    y = np.load(cache_dir / f"{name}_y.npy", mmap_mode="r")
    context = np.load(cache_dir / f"{name}_context.npy", mmap_mode="r")
    with np.load(cache_dir / f"{name}_metadata.npz", allow_pickle=False) as values:
        metadata = {key: values[key] for key in values.files}
    return z, y, context, metadata


def _fresh_head(kind: str, config: dict[str, Any], device: torch.device) -> nn.Module:
    set_determinism(int(config["training"]["seed"]))
    if kind == "jepa":
        head: nn.Module = FrozenJEPACDFHead(
            config["model"]["latent_dim"], config["model"]["head_hidden"]
        )
    elif kind == "context":
        head = ContextCDFHead(CONTEXT_DIM, config["model"]["context_hidden"])
    else:
        raise ValueError(f"unknown head kind: {kind}")
    return head.to(device)


def per_sample_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape[-1] != STATE_DIM:
        raise ValueError("CDF logits/target shape mismatch")
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none").mean(1)


def _evaluate_bce(
    head: nn.Module, x: torch.Tensor, targets: torch.Tensor, batch_size: int
) -> tuple[float, np.ndarray]:
    head.eval()
    losses: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            logits = head(x[start : start + batch_size])
            losses.append(per_sample_bce(logits, targets[start : start + batch_size]).cpu().numpy())
    values = np.concatenate(losses)
    return float(values.mean(dtype=np.float64)), values


def train_head(
    kind: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    thresholds: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    *,
    x_selection: np.ndarray | None = None,
    y_selection: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    training = config["training"]
    head = _fresh_head(kind, config, device)
    optimizer = torch.optim.AdamW(
        list(head.parameters()),
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
        weight_decay=training["weight_decay"],
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in head.parameters()}:
        raise AssertionError("optimizer contains non-head parameters")
    epochs = int(training["max_epochs"] if fixed_epochs is None else fixed_epochs)
    if not 1 <= epochs <= training["max_epochs"]:
        raise ValueError("invalid fixed head training budget")
    # mmap caches are intentionally read-only. torch.tensor makes an owned copy,
    # avoiding the undefined-write warning emitted by torch.from_numpy(memmap).
    x_train = torch.tensor(np.asarray(x_fit), device=device)
    target_train = torch.from_numpy(binary_targets(y_fit, thresholds)).to(device)
    if x_selection is not None:
        if y_selection is None or fixed_epochs is not None:
            raise ValueError("selection data is only valid for inner training")
        x_dev = torch.tensor(np.asarray(x_selection), device=device)
        target_dev = torch.from_numpy(binary_targets(y_selection, thresholds)).to(device)
    else:
        x_dev = target_dev = None
    rng = np.random.Generator(np.random.PCG64(training["seed"]))
    batch_size = int(training["batch_size"])
    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_loss = float("inf")
    started = time.monotonic()
    for epoch in range(epochs):
        head.train()
        permutation = rng.permutation(len(x_train))
        total = 0.0
        count = 0
        for start in range(0, len(permutation), batch_size):
            indices = torch.from_numpy(permutation[start : start + batch_size]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x_train[indices])
            loss = per_sample_bce(logits, target_train[indices]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), training["gradient_clip_norm"])
            optimizer.step()
            size = len(indices)
            total += float(loss.detach()) * size
            count += size
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_bce": total / count,
        }
        if x_dev is not None and target_dev is not None:
            selection_loss, _ = _evaluate_bce(head, x_dev, target_dev, batch_size)
            record["selection_bce"] = selection_loss
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_epoch = epoch
        history.append(record)
        print(
            json.dumps(
                {
                    "phase": "head_training",
                    "kind": kind,
                    **record,
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    if x_dev is None:
        best_epoch = epochs - 1
        best_loss = float("nan")
    selection = {
        "kind": kind,
        "best_epoch_zero_based": best_epoch,
        "selected_budget_epochs": best_epoch + 1,
        "best_selection_bce": best_loss,
        "selection_population": "inner_dev_2022" if x_dev is not None else None,
    }
    return head, history, selection


def _predict_logits(head: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    tensor = torch.tensor(np.asarray(x), device=device)
    values: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            values.append(head(tensor[start : start + batch_size]).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _component_losses(logits: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits64 = np.asarray(logits, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits64, -50.0, 50.0)))
    bce = np.logaddexp(0.0, logits64) - target64 * logits64
    brier = np.square(probabilities - target64)
    return probabilities, bce, brier


def _scopes(schema: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result = {"ALL": np.arange(STATE_DIM, dtype=np.int64)}
    for horizon in HORIZONS:
        result[f"H{horizon}"] = np.asarray(
            [entry["index"] for entry in schema if entry["horizon"] == horizon], dtype=np.int64
        )
    for outcome in OUTCOMES:
        result[outcome] = np.asarray(
            [entry["index"] for entry in schema if entry["outcome"] == outcome], dtype=np.int64
        )
    return result


def _aggregate_metrics(
    component_bce: np.ndarray,
    component_brier: np.ndarray,
    scopes: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    summary: dict[str, dict[str, float]] = {}
    per_sample: dict[str, np.ndarray] = {}
    for name, indices in scopes.items():
        bce = component_bce[:, indices].mean(axis=1)
        brier = component_brier[:, indices].mean(axis=1)
        summary[name] = {
            "bce": float(bce.mean(dtype=np.float64)),
            "brier": float(brier.mean(dtype=np.float64)),
        }
        per_sample[f"bce_{name}"] = bce
        per_sample[f"brier_{name}"] = brier
    return summary, per_sample


def calibration_rows(
    model_name: str, probabilities: np.ndarray, targets: np.ndarray, schema: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    predicted = probabilities.mean(axis=0)
    observed = targets.mean(axis=0)
    error = np.abs(predicted - observed)
    rows = [
        {
            "model": model_name,
            **entry,
            "mean_probability": float(predicted[entry["index"]]),
            "mean_target": float(observed[entry["index"]]),
            "absolute_calibration_error": float(error[entry["index"]]),
        }
        for entry in schema
    ]
    summary = {
        "mean": float(error.mean()),
        "median": float(np.median(error)),
        "p90": float(np.quantile(error, 0.9)),
        "max": float(error.max()),
    }
    return rows, summary


def monotonicity_audit(
    model_name: str, probabilities: np.ndarray, schema: list[dict[str, Any]]
) -> dict[str, Any]:
    reshaped = probabilities.reshape(len(probabilities), 12, 9)
    violation = reshaped[:, :, :-1] > reshaped[:, :, 1:]
    magnitude = np.maximum(reshaped[:, :, :-1] - reshaped[:, :, 1:], 0.0)
    outcomes: list[dict[str, Any]] = []
    for outcome_index in range(12):
        entries = schema[outcome_index * 9 : (outcome_index + 1) * 9]
        outcomes.append(
            {
                "horizon": entries[0]["horizon"],
                "outcome": entries[0]["outcome"],
                "violation_rate": float(violation[:, outcome_index].mean()),
                "violation_magnitude": float(magnitude[:, outcome_index].mean()),
                "samples_with_any_violation_fraction": float(
                    violation[:, outcome_index].any(axis=1).mean()
                ),
            }
        )
    return {
        "model": model_name,
        "overall": {
            "violation_rate": float(violation.mean()),
            "violation_magnitude": float(magnitude.mean()),
            "samples_with_any_violation_fraction": float(violation.any(axis=(1, 2)).mean()),
        },
        "outcomes": outcomes,
    }


def _bootstrap(
    per_sample: dict[str, dict[str, np.ndarray]], blocks: np.ndarray, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison, baseline in (("jepa_vs_unconditional", "unconditional"), ("jepa_vs_context", "context")):
        for scope in ("ALL", "H16", "H64", "H256"):
            effects = per_sample[baseline][f"bce_{scope}"] - per_sample["jepa"][f"bce_{scope}"]
            result = block_bootstrap(
                effects,
                blocks,
                int(config["evaluation"]["bootstrap_samples"]),
                int(config["evaluation"]["bootstrap_seed"]),
            )
            rows.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "effect": result["effect"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                    "n": len(effects),
                    "number_of_weeks": result["blocks"],
                }
            )
    return pd.DataFrame(rows)


def _effect(bootstrap: pd.DataFrame, comparison: str, scope: str) -> dict[str, Any]:
    row = bootstrap.loc[
        (bootstrap["comparison"] == comparison) & (bootstrap["scope"] == scope)
    ].iloc[0]
    return {
        "effect": float(row["effect"]),
        "ci95": [float(row["ci95_low"]), float(row["ci95_high"])],
        "n": int(row["n"]),
        "weeks": int(row["number_of_weeks"]),
    }


def _save_head(
    path: Path,
    head: nn.Module,
    kind: str,
    budget: int,
    thresholds: np.ndarray,
    config: dict[str, Any],
    checkpoint_hash: str,
) -> None:
    save_checkpoint(
        {
            "experiment_id": config["experiment_id"],
            "protocol_version": config["protocol_version"],
            "kind": kind,
            "model": head.state_dict(),
            "budget_epochs": budget,
            "thresholds": thresholds,
            "schema": state_schema(),
            "v0_checkpoint_sha256": checkpoint_hash,
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "test_consumed": False,
        },
        path,
    )


def _report(summary: dict[str, Any], path: Path) -> None:
    effects = summary["effects"]
    metrics = summary["development_metrics"]
    calibration = summary["calibration"]
    monotonicity = summary["monotonicity"]
    outcome_gains = {
        outcome: metrics["context"][outcome]["bce"] - metrics["jepa"][outcome]["bce"]
        for outcome in OUTCOMES
    }
    easiest = max(OUTCOMES, key=outcome_gains.__getitem__)
    lines = [
        "# Market Predictive State V0.8 — Frozen V0 JEPA + 108D Conditional-CDF Head",
        "",
        f"- Frozen checkpoint epoch: {summary['checkpoint']['epoch']}",
        f"- V0 checkpoint SHA-256: `{summary['checkpoint']['sha256']}`",
        f"- JEPA selected epoch: {summary['inner_selection']['jepa']['best_epoch_zero_based']}",
        f"- Context selected epoch: {summary['inner_selection']['context']['best_epoch_zero_based']}",
        f"- Test consumed: `{str(summary['test_consumed']).lower()}`",
        "",
        "## Development BCE",
        "",
        "| Scope | JEPA | Context | Unconditional | JEPA−Context gain | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ("ALL", "H16", "H64", "H256"):
        effect = effects["jepa_vs_context"][scope]
        lines.append(
            f"| {scope} | {metrics['jepa'][scope]['bce']:.8g} | {metrics['context'][scope]['bce']:.8g} | "
            f"{metrics['unconditional'][scope]['bce']:.8g} | {effect['effect']:.8g} | "
            f"[{effect['ci95'][0]:.8g}, {effect['ci95'][1]:.8g}] |"
        )
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. JEPA vs unconditional: {'significantly better' if summary['gates']['A_jepa_vs_unconditional'] else 'not significantly better'}; effect={effects['jepa_vs_unconditional']['ALL']['effect']:.8g}, CI={effects['jepa_vs_unconditional']['ALL']['ci95']}.",
            f"2. JEPA vs Context-only: {'significantly better' if summary['gates']['B_jepa_vs_context'] else 'not significantly better'}; effect={effects['jepa_vs_context']['ALL']['effect']:.8g}, CI={effects['jepa_vs_context']['ALL']['ci95']}.",
            "3. H16/H64/H256 results are shown in the table; cross-horizon Gate C is " + ("PASS." if summary["gates"]["C_cross_horizon"] else "FAIL."),
            f"4. Largest JEPA-over-Context outcome-family BCE gain is {easiest}; gains are "
            + ", ".join(f"{name}={value:.6g}" for name, value in outcome_gains.items())
            + ".",
            f"5. JEPA calibration absolute error mean/median/p90/max = {calibration['jepa']['mean']:.6g}/{calibration['jepa']['median']:.6g}/{calibration['jepa']['p90']:.6g}/{calibration['jepa']['max']:.6g}.",
            f"6. JEPA monotonic violation rate={monotonicity['jepa']['overall']['violation_rate']:.6g}, magnitude={monotonicity['jepa']['overall']['violation_magnitude']:.6g}, samples-any={monotonicity['jepa']['overall']['samples_with_any_violation_fraction']:.6g}. No post-processing was applied.",
            f"7. **{summary['decision']}**",
            "8. **2025 test_consumed = false**",
            "",
            "Claim boundary: this experiment evaluates only a Marginal-CDF Y-Predictive State Candidate; it does not model the full joint future distribution or establish profitability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, device: torch.device) -> dict[str, Any]:
    config = load_protocol(config_path)
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["checkpoint"]["path"])
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint.get("epoch") != 49 or checkpoint.get("design_version") != "0.6.1":
        raise RuntimeError("V0.8 requires V0 epoch49/design 0.6.1")
    if checkpoint.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("V0.8 dataset hash mismatch")

    schema = state_schema()
    (output_dir / "cdf_state_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    cache_dir, data = build_latent_cache(config, checkpoint, checkpoint_hash)
    consistency = validate_cached_latents(config, checkpoint, cache_dir, data, device)

    fit_z, fit_y, fit_context, _ = _load_cache(cache_dir, "inner_fit")
    dev_z, dev_y, dev_context, _ = _load_cache(cache_dir, "inner_dev")
    final_z, final_y, final_context, _ = _load_cache(cache_dir, "final_train")
    del data

    inner_thresholds, inner_threshold_metadata = fit_thresholds(
        fit_y, "inner_fit", config["data"]["populations"]["inner_fit"]
    )
    (output_dir / "cdf_thresholds_inner.json").write_text(
        json.dumps(inner_threshold_metadata, indent=2) + "\n", encoding="utf-8"
    )
    inner_jepa, history_jepa, selection_jepa = train_head(
        "jepa",
        fit_z,
        fit_y,
        inner_thresholds,
        config,
        device,
        x_selection=dev_z,
        y_selection=dev_y,
    )
    del inner_jepa
    if device.type == "cuda":
        torch.cuda.empty_cache()
    inner_context, history_context, selection_context = train_head(
        "context",
        fit_context,
        fit_y,
        inner_thresholds,
        config,
        device,
        x_selection=dev_context,
        y_selection=dev_y,
    )
    del inner_context
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pd.DataFrame(history_jepa).to_csv(output_dir / "inner_history_jepa.csv", index=False)
    pd.DataFrame(history_context).to_csv(output_dir / "inner_history_context.csv", index=False)
    inner_best = {"jepa": selection_jepa, "context": selection_context}
    (output_dir / "inner_best.json").write_text(
        json.dumps(_jsonable(inner_best), indent=2) + "\n", encoding="utf-8"
    )

    final_thresholds, final_threshold_metadata = fit_thresholds(
        final_y, "final_train", config["data"]["populations"]["final_train"]
    )
    (output_dir / "cdf_thresholds_final.json").write_text(
        json.dumps(final_threshold_metadata, indent=2) + "\n", encoding="utf-8"
    )
    final_jepa, _, _ = train_head(
        "jepa",
        final_z,
        final_y,
        final_thresholds,
        config,
        device,
        fixed_epochs=selection_jepa["selected_budget_epochs"],
    )
    final_context_head, _, _ = train_head(
        "context",
        final_context,
        final_y,
        final_thresholds,
        config,
        device,
        fixed_epochs=selection_context["selected_budget_epochs"],
    )
    _save_head(
        output_dir / "final_head.pt",
        final_jepa,
        "jepa",
        selection_jepa["selected_budget_epochs"],
        final_thresholds,
        config,
        checkpoint_hash,
    )
    _save_head(
        output_dir / "final_context_head.pt",
        final_context_head,
        "context",
        selection_context["selected_budget_epochs"],
        final_thresholds,
        config,
        checkpoint_hash,
    )

    # Development is first loaded for metrics only after both final heads exist.
    development_z, development_y, development_context, development_metadata = _load_cache(
        cache_dir, "development"
    )
    batch_size = int(config["training"]["batch_size"])
    jepa_logits = _predict_logits(final_jepa, development_z, device, batch_size)
    context_logits = _predict_logits(final_context_head, development_context, device, batch_size)
    targets = binary_targets(development_y, final_thresholds).astype(np.float64)
    uncond_probability = unconditional_probabilities().astype(np.float64)
    probabilities_jepa, bce_jepa, brier_jepa = _component_losses(jepa_logits, targets)
    probabilities_context, bce_context, brier_context = _component_losses(context_logits, targets)
    probabilities_uncond = np.broadcast_to(uncond_probability, targets.shape)
    bce_uncond = -(
        targets * np.log(probabilities_uncond)
        + (1.0 - targets) * np.log(1.0 - probabilities_uncond)
    )
    brier_uncond = np.square(probabilities_uncond - targets)
    scopes = _scopes(schema)
    metric_summary: dict[str, Any] = {}
    per_sample: dict[str, dict[str, np.ndarray]] = {}
    for name, bce, brier in (
        ("jepa", bce_jepa, brier_jepa),
        ("context", bce_context, brier_context),
        ("unconditional", bce_uncond, brier_uncond),
    ):
        metric_summary[name], per_sample[name] = _aggregate_metrics(bce, brier, scopes)

    calibration: dict[str, Any] = {}
    calibration_table: list[dict[str, Any]] = []
    for name, probabilities in (("jepa", probabilities_jepa), ("context", probabilities_context)):
        rows, calibration[name] = calibration_rows(name, probabilities, targets, schema)
        calibration_table.extend(rows)
    pd.DataFrame(calibration_table).to_csv(output_dir / "calibration.csv", index=False)
    monotonicity = {
        "jepa": monotonicity_audit("jepa", probabilities_jepa, schema),
        "context": monotonicity_audit("context", probabilities_context, schema),
    }
    (output_dir / "monotonicity.json").write_text(
        json.dumps(monotonicity, indent=2) + "\n", encoding="utf-8"
    )

    blocks = development_metadata["iso_week"]
    bootstrap = _bootstrap(per_sample, blocks, config)
    bootstrap.to_csv(output_dir / "bootstrap_effects.csv", index=False)
    effects = {
        comparison: {
            scope: _effect(bootstrap, comparison, scope)
            for scope in ("ALL", "H16", "H64", "H256")
        }
        for comparison in ("jepa_vs_unconditional", "jepa_vs_context")
    }
    gate_a = effects["jepa_vs_unconditional"]["ALL"]["ci95"][0] > 0
    gate_b = effects["jepa_vs_context"]["ALL"]["ci95"][0] > 0
    horizon_context = [effects["jepa_vs_context"][f"H{horizon}"] for horizon in HORIZONS]
    gate_c = sum(value["effect"] > 0 for value in horizon_context) >= 2 and not any(
        value["ci95"][1] < 0 for value in horizon_context
    )
    decision = (
        "V0.8_FROZEN_JEPA_CDF_GO"
        if gate_a and gate_b and gate_c
        else "V0.8_FROZEN_JEPA_CDF_NO_GO"
    )

    development_frame: dict[str, Any] = {
        "anchor_index": development_metadata["anchor_index"],
        "timestamp": development_metadata["timestamp_ns"].astype("datetime64[ns]"),
        "trading_day": development_metadata["trading_day_ns"].astype("datetime64[ns]"),
        "iso_week": blocks,
    }
    for model_name in ("jepa", "context", "unconditional"):
        for scope in ("ALL", "H16", "H64", "H256"):
            development_frame[f"{model_name}_bce_{scope.lower()}"] = per_sample[model_name][
                f"bce_{scope}"
            ]
            development_frame[f"{model_name}_brier_{scope.lower()}"] = per_sample[model_name][
                f"brier_{scope}"
            ]
    pd.DataFrame(development_frame).to_csv(output_dir / "development_per_sample.csv", index=False)
    development_metrics = {
        "metrics": metric_summary,
        "effects": effects,
        "calibration": calibration,
        "monotonicity": monotonicity,
        "samples": len(targets),
        "weeks": int(len(np.unique(blocks))),
        "test_consumed": False,
    }
    (output_dir / "development_metrics.json").write_text(
        json.dumps(_jsonable(development_metrics), indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "config": config,
        "config_sha256": _file_sha256(config_path),
        "v0_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "epoch": checkpoint["epoch"],
            "design_version": checkpoint["design_version"],
            "source_sha256": checkpoint["source_sha256"],
        },
        "cache_consistency": consistency,
        "test_consumed": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(_jsonable(protocol), indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "experiment": "Market Predictive State V0.8 — Frozen V0 JEPA + 108D Conditional-CDF Head",
        "claim": "Marginal-CDF Y-Predictive State Candidate",
        "checkpoint": protocol["v0_checkpoint"],
        "cache_consistency": consistency,
        "thresholds": {
            "inner": inner_threshold_metadata,
            "final": final_threshold_metadata,
        },
        "inner_selection": inner_best,
        "development_metrics": metric_summary,
        "effects": effects,
        "calibration": calibration,
        "monotonicity": monotonicity,
        "gates": {
            "A_jepa_vs_unconditional": gate_a,
            "B_jepa_vs_context": gate_b,
            "C_cross_horizon": gate_c,
        },
        "decision": decision,
        "test_consumed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    _report(summary, output_dir / "summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="V0.8 frozen-JEPA Conditional-CDF experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/market_predictive_state_v0_8_frozen_jepa_cdf.yaml"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    summary = run(args.config, torch.device(args.device))
    print(
        json.dumps(
            {"decision": summary["decision"], "test_consumed": summary["test_consumed"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
