from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from market_jepa.eval.metrics import block_bootstrap


HORIZONS = (16, 64, 256)
OUTCOME_DIM = 4
Y_DIM = len(HORIZONS) * OUTCOME_DIM
RFF_SEED = 4243
PAIR_SEED = 4242
PAIR_SAMPLES = 100_000
RFF_PER_BANDWIDTH = 1024
BANDWIDTH_MULTIPLIERS = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)
HEAD_SEED = 42
HEAD_BATCH_SIZE = 1024
HEAD_MAX_EPOCHS = 100
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4
HEAD_GRADIENT_CLIP = 1.0
K_NEIGHBORS = 50
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=np.float64)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std[std < 1e-12] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.std).astype(
            np.float32
        )


@dataclass(frozen=True)
class RFFMap:
    sigma0: float
    bandwidths: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray

    @classmethod
    def fit(cls, train_y: np.ndarray) -> tuple["RFFMap", np.ndarray, np.ndarray]:
        if len(train_y) < 2:
            raise ValueError("RFF fitting requires at least two Train outcomes")
        pair_rng = np.random.Generator(np.random.PCG64(PAIR_SEED))
        left = pair_rng.integers(0, len(train_y), size=PAIR_SAMPLES)
        offsets = pair_rng.integers(1, len(train_y), size=PAIR_SAMPLES)
        right = (left + offsets) % len(train_y)
        distances = np.linalg.norm(
            train_y[left].astype(np.float64) - train_y[right].astype(np.float64),
            axis=1,
        )
        sigma0 = float(np.median(distances))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise RuntimeError("non-finite or non-positive Train-only sigma0")
        bandwidths = sigma0 * BANDWIDTH_MULTIPLIERS
        rng = np.random.Generator(np.random.PCG64(RFF_SEED))
        frequencies = np.stack(
            [
                rng.normal(
                    0.0,
                    1.0 / bandwidth,
                    size=(Y_DIM, RFF_PER_BANDWIDTH),
                )
                for bandwidth in bandwidths
            ]
        ).astype(np.float32)
        phases = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(len(bandwidths), RFF_PER_BANDWIDTH),
        ).astype(np.float32)
        return cls(sigma0, bandwidths, frequencies, phases), left, right

    @property
    def output_dim(self) -> int:
        return len(self.bandwidths) * RFF_PER_BANDWIDTH

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        blocks = [
            np.sqrt(2.0 / RFF_PER_BANDWIDTH)
            * np.cos(np.asarray(values, dtype=np.float32) @ frequency + phase)
            / np.sqrt(3.0)
            for frequency, phase in zip(self.frequencies, self.phases, strict=True)
        ]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def torch_parameters(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.frequencies).to(device),
            torch.from_numpy(self.phases).to(device),
        )


class PredictiveStateHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3 * RFF_PER_BANDWIDTH),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _rff_torch(
    values: torch.Tensor,
    frequencies: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    scale = math.sqrt(2.0 / RFF_PER_BANDWIDTH) / math.sqrt(3.0)
    return torch.cat(
        [torch.cos(values @ frequency + phase) * scale for frequency, phase in zip(frequencies, phases)],
        dim=1,
    )


def rff_audit(
    train_y: np.ndarray,
    rff: RFFMap,
    left: np.ndarray,
    right: np.ndarray,
    chunk_size: int = 4096,
) -> dict[str, float | int]:
    exact_parts: list[np.ndarray] = []
    approximate_parts: list[np.ndarray] = []
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        y_left = train_y[left[start:stop]]
        y_right = train_y[right[start:stop]]
        squared_distance = np.square(
            y_left.astype(np.float64) - y_right.astype(np.float64)
        ).sum(axis=1)
        exact_parts.append(
            np.mean(
                [
                    np.exp(-squared_distance / (2.0 * bandwidth**2))
                    for bandwidth in rff.bandwidths
                ],
                axis=0,
            )
        )
        approximate_parts.append(
            np.sum(
                rff.transform_numpy(y_left).astype(np.float64)
                * rff.transform_numpy(y_right).astype(np.float64),
                axis=1,
            )
        )
    exact = np.concatenate(exact_parts)
    approximate = np.concatenate(approximate_parts)
    error = approximate - exact
    correlation = float(np.corrcoef(exact, approximate)[0, 1])
    result = {
        "pairs": int(len(left)),
        "sigma0": rff.sigma0,
        "bandwidths": rff.bandwidths.tolist(),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": correlation,
    }
    if not all(np.isfinite(value) for value in (result["rmse"], result["mae"], correlation)):
        raise RuntimeError("non-finite RFF numerical audit")
    if correlation <= 0 or result["rmse"] >= 1.0:
        raise RuntimeError("RFF numerical audit indicates an implementation error")
    return result


def _loss_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(prediction - target).sum(dim=1)


def _evaluate_head_loss(
    head: PredictiveStateHead,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    frequencies: torch.Tensor,
    phases: torch.Tensor,
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), HEAD_BATCH_SIZE):
            selected = indices[start : start + HEAD_BATCH_SIZE]
            x_batch = torch.from_numpy(x[selected]).to(device)
            y_batch = torch.from_numpy(y[selected]).to(device)
            loss = _loss_per_sample(
                head(x_batch), _rff_torch(y_batch, frequencies, phases)
            )
            total += float(loss.sum().item())
            count += len(selected)
    return total / count


def inner_fit_dev_indices(trading_day_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    years = pd.DatetimeIndex(trading_day_ns.astype("datetime64[ns]")).year.to_numpy()
    fit_indices = np.flatnonzero((years >= 2018) & (years <= 2021))
    dev_indices = np.flatnonzero(years == 2022)
    if not len(fit_indices) or not len(dev_indices):
        raise ValueError("inner-fit 2018-2021 and inner-dev 2022 must both be non-empty")
    return fit_indices, dev_indices


def select_epoch_budget(
    x: np.ndarray,
    y: np.ndarray,
    trading_day_ns: np.ndarray,
    rff: RFFMap,
    device: torch.device,
) -> tuple[int, list[dict[str, float | int]]]:
    fit_indices, dev_indices = inner_fit_dev_indices(trading_day_ns)

    _seed_everything(HEAD_SEED)
    head = PredictiveStateHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    frequencies, phases = rff.torch_parameters(device)
    rng = np.random.default_rng(HEAD_SEED)
    history: list[dict[str, float | int]] = []
    best_epoch = -1
    best_loss = math.inf
    for epoch in range(HEAD_MAX_EPOCHS):
        order = rng.permutation(fit_indices)
        head.train()
        total = 0.0
        count = 0
        for start in range(0, len(order), HEAD_BATCH_SIZE):
            selected = order[start : start + HEAD_BATCH_SIZE]
            x_batch = torch.from_numpy(x[selected]).to(device)
            y_batch = torch.from_numpy(y[selected]).to(device)
            with torch.no_grad():
                target = _rff_torch(y_batch, frequencies, phases)
            optimizer.zero_grad(set_to_none=True)
            loss_by_sample = _loss_per_sample(head(x_batch), target)
            loss = loss_by_sample.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), HEAD_GRADIENT_CLIP)
            optimizer.step()
            total += float(loss_by_sample.detach().sum().item())
            count += len(selected)
        dev_loss = _evaluate_head_loss(
            head, x, y, dev_indices, frequencies, phases, device
        )
        record = {
            "epoch": epoch,
            "inner_fit_loss": total / count,
            "inner_dev_loss": dev_loss,
        }
        history.append(record)
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
    return best_epoch, history


def fit_full_train_head(
    x: np.ndarray,
    y: np.ndarray,
    rff: RFFMap,
    selected_epoch: int,
    device: torch.device,
) -> PredictiveStateHead:
    if not 0 <= selected_epoch < HEAD_MAX_EPOCHS:
        raise ValueError("selected epoch is outside the frozen budget")
    _seed_everything(HEAD_SEED)
    head = PredictiveStateHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    frequencies, phases = rff.torch_parameters(device)
    rng = np.random.default_rng(HEAD_SEED)
    indices = np.arange(len(x))
    for _ in range(selected_epoch + 1):
        order = rng.permutation(indices)
        head.train()
        for start in range(0, len(order), HEAD_BATCH_SIZE):
            selected = order[start : start + HEAD_BATCH_SIZE]
            x_batch = torch.from_numpy(x[selected]).to(device)
            y_batch = torch.from_numpy(y[selected]).to(device)
            with torch.no_grad():
                target = _rff_torch(y_batch, frequencies, phases)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_per_sample(head(x_batch), target).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), HEAD_GRADIENT_CLIP)
            optimizer.step()
    head.eval()
    return head


def predict_head(
    head: PredictiveStateHead,
    x: np.ndarray,
    output_path: Path,
    device: torch.device,
) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(x), 3 * RFF_PER_BANDWIDTH),
    )
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(x), HEAD_BATCH_SIZE):
            stop = min(start + HEAD_BATCH_SIZE, len(x))
            output[start:stop] = head(
                torch.from_numpy(x[start:stop]).to(device)
            ).cpu().numpy()
    output.flush()
    return output


def write_phi(
    y: np.ndarray, rff: RFFMap, output_path: Path, chunk_size: int = 4096
) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(y), rff.output_dim)
    )
    for start in range(0, len(y), chunk_size):
        stop = min(start + chunk_size, len(y))
        output[start:stop] = rff.transform_numpy(y[start:stop])
    output.flush()
    return output


def nearest_euclidean_indices(
    reference: np.ndarray,
    query: np.ndarray,
    reference_timestamp: np.ndarray,
    query_timestamp: np.ndarray,
    k: int,
    device: torch.device,
    query_chunk_size: int = 128,
) -> np.ndarray:
    if k > len(reference):
        raise ValueError("k exceeds reference sample count")
    if np.max(reference_timestamp) >= np.min(query_timestamp):
        raise ValueError("state kNN reference contains a query-time-or-future row")
    reference_tensor = torch.from_numpy(np.asarray(reference)).to(device)
    reference_norm = torch.square(reference_tensor).sum(dim=1)
    result = np.empty((len(query), k), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(query), query_chunk_size):
            stop = min(start + query_chunk_size, len(query))
            query_tensor = torch.from_numpy(np.asarray(query[start:stop])).to(device)
            distance = (
                torch.square(query_tensor).sum(dim=1, keepdim=True)
                + reference_norm.unsqueeze(0)
                - 2.0 * query_tensor @ reference_tensor.T
            )
            result[start:stop] = torch.topk(
                distance, k=k, dim=1, largest=False, sorted=True
            ).indices.cpu().numpy()
    return result


def _weekly_blocks(trading_day_ns: np.ndarray) -> np.ndarray:
    iso = pd.DatetimeIndex(trading_day_ns.astype("datetime64[ns]")).isocalendar()
    return (iso["year"].to_numpy(dtype=np.int64) * 100 + iso["week"].to_numpy(dtype=np.int64))


def _bootstrap_effect(effects: np.ndarray, blocks: np.ndarray) -> dict[str, Any]:
    result = block_bootstrap(effects, blocks, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    return {
        "effect": result["effect"],
        "ci95": [result["ci95_low"], result["ci95_high"]],
        "blocks": result["blocks"],
        "pass": result["ci95_low"] > 0,
    }


def _state_losses(
    prediction: np.ndarray,
    target: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    result = np.empty(len(target), dtype=np.float64)
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        difference = prediction[start:stop].astype(np.float64) - target[start:stop].astype(
            np.float64
        )
        result[start:stop] = np.square(difference).sum(axis=1)
    return result


def _constant_losses(
    target: np.ndarray, constant: np.ndarray, chunk_size: int = 1024
) -> np.ndarray:
    result = np.empty(len(target), dtype=np.float64)
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        difference = target[start:stop].astype(np.float64) - constant
        result[start:stop] = np.square(difference).sum(axis=1)
    return result


def _neighbor_losses(
    train_phi: np.ndarray,
    validation_phi: np.ndarray,
    neighbors: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    result = np.empty(len(neighbors), dtype=np.float64)
    for start in range(0, len(neighbors), chunk_size):
        stop = min(start + chunk_size, len(neighbors))
        estimate = train_phi[neighbors[start:stop]].astype(np.float64).mean(axis=1)
        difference = estimate - validation_phi[start:stop].astype(np.float64)
        result[start:stop] = np.square(difference).sum(axis=1)
    return result


def evaluate_gates(
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_state: np.ndarray,
    validation_state: np.ndarray,
    train_context_state: np.ndarray,
    validation_context_state: np.ndarray,
    train_timestamp: np.ndarray,
    validation_timestamp: np.ndarray,
    validation_trading_day: np.ndarray,
    rff: RFFMap,
    cache_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_phi = write_phi(train_y, rff, cache_dir / "phi_train.npy")
    validation_phi = write_phi(validation_y, rff, cache_dir / "phi_validation.npy")
    unconditional = np.asarray(train_phi).mean(axis=0, dtype=np.float64)

    loss_state = _state_losses(validation_state, validation_phi)
    loss_context = _state_losses(validation_context_state, validation_phi)
    loss_unconditional = _constant_losses(validation_phi, unconditional)
    blocks = _weekly_blocks(validation_trading_day)
    information = _bootstrap_effect(loss_context - loss_state, blocks)
    unconditional_effect = _bootstrap_effect(loss_unconditional - loss_state, blocks)
    gate1_pass = information["pass"] and unconditional_effect["pass"]

    state_neighbors = nearest_euclidean_indices(
        train_state,
        validation_state,
        train_timestamp,
        validation_timestamp,
        K_NEIGHBORS,
        device,
    )
    context_neighbors = nearest_euclidean_indices(
        train_context_state,
        validation_context_state,
        train_timestamp,
        validation_timestamp,
        K_NEIGHBORS,
        device,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    random_neighbors = np.stack(
        [
            rng.choice(len(train_state), size=K_NEIGHBORS, replace=False)
            for _ in range(len(validation_state))
        ]
    )
    error_state = _neighbor_losses(train_phi, validation_phi, state_neighbors)
    error_context = _neighbor_losses(train_phi, validation_phi, context_neighbors)
    error_random = _neighbor_losses(train_phi, validation_phi, random_neighbors)
    geometry = _bootstrap_effect(error_context - error_state, blocks)
    random_effect = _bootstrap_effect(error_random - error_state, blocks)
    gate2_pass = geometry["pass"] and random_effect["pass"]

    return {
        "gate1": {
            "b_error": float(loss_state.mean()),
            "context_error": float(loss_context.mean()),
            "unconditional_error": float(loss_unconditional.mean()),
            "delta_information": information,
            "delta_unconditional": unconditional_effect,
            "pass": gate1_pass,
        },
        "gate2": {
            "distance": "euclidean",
            "k": K_NEIGHBORS,
            "state_knn_error": float(error_state.mean()),
            "context_knn_error": float(error_context.mean()),
            "random_error": float(error_random.mean()),
            "delta_geometry": geometry,
            "delta_random": random_effect,
            "pass": gate2_pass,
        },
        "overall": "V0.7-FROZEN_GO" if gate1_pass and gate2_pass else "V0.7-FROZEN_NO_GO",
        "test_consumed": False,
    }


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
