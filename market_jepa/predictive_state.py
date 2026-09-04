from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from market_jepa.data.pipeline import MarketData, NormalizerBundle
from market_jepa.data.schema import CONTEXT_FEATURES, MARKET_FEATURES
from market_jepa.eval.metrics import block_bootstrap
from market_jepa.frozen_state import HEAD_SEED, RFFMap, Standardizer, rff_audit
from market_jepa.model.encoders import MarketEncoder


PROTOCOL_VERSION = "0.7.1"
PROTOCOL_PATH = Path("docs/market_predictive_state_v0_7_end_to_end_protocol.md")
HORIZONS = (16, 64, 256)
MAX_HORIZON = 256
STATE_DIM = 3072
STATE_HIDDEN = 512
MARKET_DIM = len(MARKET_FEATURES)
CONTEXT_DIM = len(CONTEXT_FEATURES)
FINAL_REFERENCES: dict[str, Any] = {
    "sigma0": 3.0125319812116054,
    "rmse": 0.011788527790449407,
    "mae": 0.009348590317651785,
    "correlation": 0.998792590439722,
    "y_mean_hash": "9158a8251a3b52c82fd2d9b88f979a95fdbcd0cb3b5321070af6c1e7858eea8e",
    "y_std_hash": "58145f4c51bdd64e38c2c073b8e9b1a71dde12e46e8e3789289ffdd316143b96",
    "pair_hash": "d6d2951c3a2bb53b268ca516278f9b725e772309b7ed1d5b171a5489a7ae9990",
    "w_hash": "0f696e854d938331011b317b129f6568ff7b7960f3f8ad9bf2d118cfb5a2c50b",
    "b_hash": "ef45a53d4f680e207e2b9ddcd0834f48b9c0161237be7bd3ed95e799c1790a6f",
}


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("V0.7 config root must be a mapping")
    required = {"experiment_id", "protocol_version", "protocol_path", "data", "model", "training", "target", "evaluation", "artifacts"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"V0.7 config missing keys: {sorted(missing)}")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("V0.7 protocol version differs from frozen implementation")
    if tuple(config["data"]["horizons"]) != HORIZONS:
        raise ValueError("V0.7 horizons are frozen at 16/64/256")
    if config["data"]["anchor_stride"] != 1:
        raise ValueError("V0.7 anchor stride is frozen at one")
    frozen_data = {
        "csv_path": "8Y_DCE_JM2601_1m.csv",
        "symbol": "JM",
        "series_id": "8Y_DCE_JM2601",
        "minute_context_length": 512,
        "horizons": [16, 64, 256],
        "realized_vol_window": 32,
        "anchor_stride": 1,
    }
    for key, expected in frozen_data.items():
        if config["data"].get(key) != expected:
            raise ValueError(f"V0.7 data.{key} differs from frozen protocol")
    frozen_model = {
        "ablation": "minute_daily_weekly",
        "minute_d_model": 256,
        "minute_layers": 4,
        "minute_heads": 8,
        "minute_ffn_dim": 1024,
        "time_hidden": 32,
        "daily_hidden": 128,
        "weekly_hidden": 128,
        "recurrent_layers": 2,
        "fusion_hidden": 512,
        "latent_dim": 256,
        "predictor_hidden": 512,
        "dropout": 0.1,
        "state_hidden": 512,
        "state_dim": 3072,
    }
    if config["model"] != frozen_model:
        raise ValueError("V0.7 model differs from frozen protocol")
    training = config["training"]
    frozen_training = {
        "seed": 42,
        "microbatch_size": 64,
        "gradient_accumulation": 2,
        "max_epochs": 100,
        "gradient_clip_norm": 1.0,
        "amp": True,
        "warmup_ratio": 0.05,
        "scheduler": "cosine",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "encoder_learning_rate": 3e-4,
        "encoder_weight_decay": 0.05,
        "head_learning_rate": 1e-3,
        "head_weight_decay": 1e-4,
    }
    for key, expected in frozen_training.items():
        if training.get(key) != expected:
            raise ValueError(f"V0.7 training.{key} differs from frozen protocol")
    frozen_runtime = {
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    for key, expected in frozen_runtime.items():
        if training.get(key) != expected:
            raise ValueError(f"V0.7 training.{key} differs from frozen protocol")
    target = config["target"]
    if target != {
        "pair_samples": 100_000,
        "pair_seed": 4242,
        "rff_seed": 4243,
        "rff_per_bandwidth": 1024,
        "bandwidth_multipliers": [0.5, 1.0, 2.0],
    }:
        raise ValueError("V0.7 target construction differs from frozen protocol")
    evaluation = config["evaluation"]
    if evaluation != {
        "k": 50,
        "random_seed": 4244,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 42,
        "bootstrap_block": "iso_trading_week",
    }:
        raise ValueError("V0.7 evaluation differs from frozen protocol")
    populations = config["data"]["populations"]
    if populations != {
        "inner_fit": ["2018-01-02", "2021-12-31"],
        "inner_dev": ["2022-01-01", "2022-12-31"],
        "final_train": ["2018-01-02", "2022-12-31"],
        "development": ["2023-01-01", "2024-12-31"],
        "final_test": ["2025-01-01", "2025-12-02"],
    }:
        raise ValueError("V0.7 populations differ from frozen protocol")
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_population_indices(
    data: MarketData,
    date_range: tuple[str, str] | list[str],
    context_length: int = 512,
    max_horizon: int = MAX_HORIZON,
) -> np.ndarray:
    start, end = map(pd.Timestamp, date_range)
    trading_days = data.minute["trading_day"].to_numpy(dtype="datetime64[ns]")
    candidates = np.arange(context_length - 1, len(data.minute) - max_horizon, dtype=np.int64)
    start64, end64 = np.datetime64(start), np.datetime64(end)
    mask = (trading_days[candidates] >= start64) & (trading_days[candidates] <= end64)
    target_end = candidates + max_horizon
    mask &= target_end < len(trading_days)
    mask &= trading_days[target_end] <= end64
    return candidates[mask]


def all_future_outcomes(data: MarketData, horizons: tuple[int, ...] = HORIZONS) -> dict[int, np.ndarray]:
    frame = data.minute
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    log_return_sq = np.square(np.diff(np.log(close)))
    result: dict[int, np.ndarray] = {}
    for horizon in horizons:
        rows = len(close) - horizon
        anchor = np.arange(rows, dtype=np.int64)
        future_high = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)[:rows]
        future_low = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)[:rows]
        # Preserve the frozen `future_outcomes` reduction order. A prefix-sum
        # implementation is algebraically equivalent but changes a few float32
        # RV values after cancellation, which breaks the target-space hashes.
        future_rv = np.lib.stride_tricks.sliding_window_view(
            log_return_sq, horizon
        )[:rows]
        values = np.column_stack(
            [
                close[anchor + horizon] / close[anchor] - 1.0,
                future_high.max(axis=1) / close[anchor] - 1.0,
                future_low.min(axis=1) / close[anchor] - 1.0,
                np.sqrt(future_rv.mean(axis=1)),
            ]
        )
        result[horizon] = values.astype(np.float32)
    return result


def future_y(outcomes: dict[int, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.concatenate([outcomes[horizon][indices] for horizon in HORIZONS], axis=1)


@dataclass(frozen=True)
class TargetPreprocessing:
    scaler: Standardizer
    rff: RFFMap
    mu_phi: np.ndarray
    audit: dict[str, Any]
    hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "y_mean": self.scaler.mean.tolist(),
            "y_std": self.scaler.std.tolist(),
            "sigma0": self.rff.sigma0,
            "bandwidths": self.rff.bandwidths.tolist(),
            "frequencies": self.rff.frequencies,
            "phases": self.rff.phases,
            "mu_phi": self.mu_phi,
            "audit": self.audit,
            "hashes": self.hashes,
        }


def fit_target_preprocessing(y_raw: np.ndarray, final_regression: bool = False) -> TargetPreprocessing:
    scaler = Standardizer.fit(y_raw)
    y = scaler.transform(y_raw)
    rff, left, right = RFFMap.fit(y)
    pairs = np.stack([left, right], axis=1)
    audit = rff_audit(y, rff, left, right)
    total = np.zeros(rff.output_dim, dtype=np.float64)
    for start in range(0, len(y), 4096):
        total += rff.transform_numpy(y[start : start + 4096]).sum(axis=0, dtype=np.float64)
    mu_phi = (total / len(y)).astype(np.float64)
    hashes = {
        "y_mean": array_sha256(scaler.mean),
        "y_std": array_sha256(scaler.std),
        "pairs": array_sha256(pairs),
        "frequencies": array_sha256(rff.frequencies),
        "phases": array_sha256(rff.phases),
    }
    if not np.isfinite(mu_phi).all():
        raise RuntimeError("target preprocessing produced non-finite mu_phi")
    if final_regression:
        expected_hashes = {
            "y_mean": FINAL_REFERENCES["y_mean_hash"],
            "y_std": FINAL_REFERENCES["y_std_hash"],
            "pairs": FINAL_REFERENCES["pair_hash"],
            "frequencies": FINAL_REFERENCES["w_hash"],
            "phases": FINAL_REFERENCES["b_hash"],
        }
        if hashes != expected_hashes:
            raise RuntimeError(f"Final target array regression failed: {hashes}")
        for name in ("sigma0", "rmse", "mae", "correlation"):
            actual = rff.sigma0 if name == "sigma0" else float(audit[name])
            if not np.isclose(actual, FINAL_REFERENCES[name], rtol=1e-9, atol=1e-12):
                raise RuntimeError(f"Final target scalar regression failed for {name}: {actual}")
    return TargetPreprocessing(scaler, rff, mu_phi, audit, hashes)


class PredictiveStateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data: MarketData,
        indices: np.ndarray,
        standardized_y: np.ndarray,
        context_length: int = 512,
        include_metadata: bool = False,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.standardized_y = np.asarray(standardized_y, dtype=np.float32)
        self.context_length = context_length
        self.include_metadata = include_metadata
        if len(self.indices) != len(self.standardized_y):
            raise ValueError("indices and Y must have equal length")
        # These feature-axis concatenations are invariant across anchors. Doing
        # them once avoids rebuilding the same completed and partial token
        # tables in every worker for every sample.
        (
            self.daily_completed,
            self.daily_partial,
            self.weekly_completed,
            self.weekly_partial,
        ) = data.combined_token_arrays(
        )
        if include_metadata:
            self.timestamp_ns = data.minute["timestamp"].astype("int64").to_numpy()
            self.trading_day_ns = (
                data.minute["trading_day"].astype("int64").to_numpy()
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        anchor = int(self.indices[item])
        start = anchor - self.context_length + 1
        daily_position = int(self.data.daily_position[anchor])
        weekly_position = int(self.data.weekly_position[anchor])
        sample = {
            "minute_market": torch.from_numpy(self.data.minute_market[start : anchor + 1]),
            "minute_context": torch.from_numpy(self.data.minute_context[start : anchor + 1]),
            "daily": torch.from_numpy(
                np.concatenate(
                    [
                        self.daily_completed[:daily_position],
                        self.daily_partial[anchor : anchor + 1],
                    ],
                    axis=0,
                )
            ),
            "weekly": torch.from_numpy(
                np.concatenate(
                    [
                        self.weekly_completed[:weekly_position],
                        self.weekly_partial[anchor : anchor + 1],
                    ],
                    axis=0,
                )
            ),
            "y": torch.from_numpy(self.standardized_y[item]),
        }
        if self.include_metadata:
            sample.update(
                anchor_index=anchor,
                timestamp_ns=int(self.timestamp_ns[anchor]),
                trading_day_ns=int(self.trading_day_ns[anchor]),
            )
        return sample


def _pad(items: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    values = list(items)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    return pad_sequence(values, batch_first=True), lengths


def collate_predictive_state(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    daily, daily_lengths = _pad(sample["daily"] for sample in samples)
    weekly, weekly_lengths = _pad(sample["weekly"] for sample in samples)
    batch = {
        "minute_market": torch.stack([sample["minute_market"] for sample in samples]),
        "minute_context": torch.stack([sample["minute_context"] for sample in samples]),
        "daily": daily,
        "daily_lengths": daily_lengths,
        "weekly": weekly,
        "weekly_lengths": weekly_lengths,
        "y": torch.stack([sample["y"] for sample in samples]),
    }
    if "anchor_index" in samples[0]:
        batch.update(
            anchor_index=torch.tensor([sample["anchor_index"] for sample in samples]),
            timestamp_ns=torch.tensor([sample["timestamp_ns"] for sample in samples]),
            trading_day_ns=torch.tensor(
                [sample["trading_day_ns"] for sample in samples]
            ),
        )
    return batch


class PredictiveStateHead(nn.Module):
    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, STATE_HIDDEN),
            nn.GELU(),
            nn.Linear(STATE_HIDDEN, STATE_DIM),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        return self.network(belief)


class FixedRFF(nn.Module):
    def __init__(self, rff: RFFMap) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.from_numpy(rff.frequencies.copy()))
        self.register_buffer("phases", torch.from_numpy(rff.phases.copy()))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(2.0 / 1024.0) / math.sqrt(3.0)
        with torch.amp.autocast(device_type=y.device.type, enabled=False):
            values = y.float()
            return torch.cat(
                [
                    torch.cos(values @ frequency.float() + phase.float()) * scale
                    for frequency, phase in zip(
                        self.frequencies, self.phases, strict=True
                    )
                ],
                dim=1,
            )


class EndToEndPredictiveState(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = MarketEncoder(MARKET_DIM, CONTEXT_DIM, model_config)
        # Frozen V0.7 initialized the head immediately after setting seed 42.
        # Reset here so encoder initialization does not shift that exact stream.
        torch.manual_seed(HEAD_SEED)
        self.head = PredictiveStateHead(model_config["latent_dim"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        belief = self.encoder(batch)
        return {"belief": belief, "state": self.head(belief)}

    def optimizer_groups(self, training: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "params": list(self.encoder.optimizer_parameters()),
                "lr": training["encoder_learning_rate"],
                "weight_decay": training["encoder_weight_decay"],
            },
            {
                "params": list(self.head.parameters()),
                "lr": training["head_learning_rate"],
                "weight_decay": training["head_weight_decay"],
            },
        ]


def identical_models(model_config: dict[str, Any], seed: int = 42) -> tuple[EndToEndPredictiveState, EndToEndPredictiveState]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    market = EndToEndPredictiveState(model_config)
    context = EndToEndPredictiveState(model_config)
    context.load_state_dict(deepcopy(market.state_dict()))
    return market, context


def context_only_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mask normalized market channels in a disposable collated batch in place."""

    batch["minute_market"].zero_()
    for name in ("daily", "weekly"):
        batch[name][..., :MARKET_DIM].zero_()
    return batch


def state_loss(state: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(device_type=state.device.type, enabled=False):
        per_sample = torch.square(state.float() - target.float()).sum(dim=-1)
        return per_sample.mean(), per_sample


def normalizer_hash(normalizers: NormalizerBundle) -> str:
    return canonical_sha256(normalizers.to_dict())


def _move_for_inference(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    cpu_names = {"daily_lengths", "weekly_lengths", "anchor_index", "timestamp_ns", "trading_day_ns"}
    return {
        name: value if name in cpu_names else value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def export_states(
    model: EndToEndPredictiveState,
    dataset: PredictiveStateDataset,
    output_path: Path,
    *,
    context_only: bool,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    amp: bool = True,
) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(dataset), STATE_DIM)
    )
    loader_options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_predictive_state,
        **loader_options,
    )
    model.to(device).eval()
    position = 0
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                batch = _move_for_inference(raw_batch, device)
                if context_only:
                    batch = context_only_batch(batch)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp and device.type == "cuda",
                ):
                    state = model(batch)["state"]
                stop = position + len(state)
                output[position:stop] = state.float().cpu().numpy()
                position = stop
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None
    if position != len(dataset):
        raise RuntimeError("state export ended before every sample was written")
    output.flush()
    return output


def exact_euclidean_indices(
    reference: np.ndarray,
    query: np.ndarray,
    reference_anchor: np.ndarray,
    reference_timestamp: np.ndarray,
    query_timestamp: np.ndarray,
    *,
    k: int = 50,
    device: torch.device,
    query_chunk_size: int = 128,
) -> np.ndarray:
    if k > len(reference):
        raise ValueError("k exceeds reference population")
    if np.max(reference_timestamp) >= np.min(query_timestamp):
        raise ValueError("KNN reference contains a query-time-or-future sample")
    reference_array = np.asarray(reference, dtype=np.float32)
    query_array = np.asarray(query, dtype=np.float32)
    anchors = np.asarray(reference_anchor, dtype=np.int64)
    reference_tensor = torch.from_numpy(reference_array).to(device)
    reference_norm = torch.square(reference_tensor).sum(dim=1)
    result = np.empty((len(query_array), k), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(query_array), query_chunk_size):
            stop = min(start + query_chunk_size, len(query_array))
            query_tensor = torch.from_numpy(query_array[start:stop]).to(device)
            distance = (
                torch.square(query_tensor).sum(dim=1, keepdim=True)
                + reference_norm.unsqueeze(0)
                - 2.0 * query_tensor @ reference_tensor.T
            )
            values, candidates = torch.topk(
                distance, k=k, dim=1, largest=False, sorted=False
            )
            cutoff = values.max(dim=1).values
            less_count = (distance < cutoff[:, None]).sum(dim=1)
            equal_count = (distance == cutoff[:, None]).sum(dim=1)
            boundary_tie = equal_count > (k - less_count)
            candidate_np = candidates.cpu().numpy()
            for row in range(stop - start):
                if bool(boundary_tie[row].item()):
                    lower = torch.nonzero(distance[row] < cutoff[row], as_tuple=False).flatten().cpu().numpy()
                    equal = torch.nonzero(distance[row] == cutoff[row], as_tuple=False).flatten().cpu().numpy()
                    needed = k - len(lower)
                    equal = equal[np.argsort(anchors[equal], kind="stable")[:needed]]
                    selected = np.concatenate([lower, equal])
                else:
                    selected = candidate_np[row]
                selected_distance = distance[row, torch.from_numpy(selected).to(device)].cpu().numpy()
                order = np.lexsort((anchors[selected], selected_distance))
                result[start + row] = selected[order]
    return result


def _weekly_blocks(trading_day_ns: np.ndarray) -> np.ndarray:
    iso = pd.DatetimeIndex(trading_day_ns.astype("datetime64[ns]")).isocalendar()
    return iso["year"].to_numpy(dtype=np.int64) * 100 + iso["week"].to_numpy(dtype=np.int64)


def _bootstrap_with_blocks(effects: np.ndarray, blocks: np.ndarray) -> dict[str, Any]:
    result = block_bootstrap(effects, blocks, samples=10_000, seed=42)
    return {
        "effect": result["effect"],
        "ci95": [result["ci95_low"], result["ci95_high"]],
        "blocks": result["blocks"],
        "pass": result["ci95_low"] > 0,
    }


def _row_squared_error(
    prediction: np.ndarray, target: np.ndarray, chunk_size: int = 1024
) -> np.ndarray:
    result = np.empty(len(target), dtype=np.float64)
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        difference = prediction[start:stop].astype(np.float64) - target[start:stop].astype(np.float64)
        result[start:stop] = np.square(difference).sum(axis=1)
    return result


def _neighbor_error(
    train_phi: np.ndarray,
    development_phi: np.ndarray,
    neighbors: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    result = np.empty(len(development_phi), dtype=np.float64)
    for start in range(0, len(development_phi), chunk_size):
        stop = min(start + chunk_size, len(development_phi))
        estimate = train_phi[neighbors[start:stop]].astype(np.float64).mean(axis=1)
        difference = estimate - development_phi[start:stop].astype(np.float64)
        result[start:stop] = np.square(difference).sum(axis=1)
    return result


def write_phi(y: np.ndarray, rff: RFFMap, path: Path) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(y), STATE_DIM)
    )
    for start in range(0, len(y), 4096):
        stop = min(start + 4096, len(y))
        output[start:stop] = rff.transform_numpy(y[start:stop])
    output.flush()
    return output


def evaluate_development(
    *,
    train_state: np.ndarray,
    development_state: np.ndarray,
    train_context_state: np.ndarray,
    development_context_state: np.ndarray,
    train_y: np.ndarray,
    development_y: np.ndarray,
    train_anchor: np.ndarray,
    train_timestamp: np.ndarray,
    development_timestamp: np.ndarray,
    development_trading_day: np.ndarray,
    target: TargetPreprocessing,
    cache_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_phi = write_phi(train_y, target.rff, cache_dir / "phi_train.npy")
    development_phi = write_phi(
        development_y, target.rff, cache_dir / "phi_development.npy"
    )
    blocks = _weekly_blocks(development_trading_day)

    market_loss = _row_squared_error(development_state, development_phi)
    context_loss = _row_squared_error(development_context_state, development_phi)
    unconditional = np.broadcast_to(target.mu_phi, development_phi.shape)
    unconditional_loss = _row_squared_error(unconditional, development_phi)
    delta_info = _bootstrap_with_blocks(context_loss - market_loss, blocks)
    delta_unconditional = _bootstrap_with_blocks(unconditional_loss - market_loss, blocks)
    gate1_pass = delta_info["pass"] and delta_unconditional["pass"]

    market_neighbors = exact_euclidean_indices(
        train_state,
        development_state,
        train_anchor,
        train_timestamp,
        development_timestamp,
        k=50,
        device=device,
    )
    context_neighbors = exact_euclidean_indices(
        train_context_state,
        development_context_state,
        train_anchor,
        train_timestamp,
        development_timestamp,
        k=50,
        device=device,
    )
    rng = np.random.Generator(np.random.PCG64(4244))
    random_neighbors = np.stack(
        [rng.choice(len(train_state), size=50, replace=False) for _ in range(len(development_state))]
    )
    market_knn_error = _neighbor_error(train_phi, development_phi, market_neighbors)
    context_knn_error = _neighbor_error(train_phi, development_phi, context_neighbors)
    random_error = _neighbor_error(train_phi, development_phi, random_neighbors)
    delta_geometry = _bootstrap_with_blocks(context_knn_error - market_knn_error, blocks)
    delta_random = _bootstrap_with_blocks(random_error - market_knn_error, blocks)
    gate2_pass = delta_geometry["pass"] and delta_random["pass"]
    return {
        "gate1": {
            "market_error": float(market_loss.mean()),
            "context_error": float(context_loss.mean()),
            "unconditional_error": float(unconditional_loss.mean()),
            "delta_info": delta_info,
            "delta_unconditional": delta_unconditional,
            "pass": gate1_pass,
        },
        "gate2": {
            "distance": "exact_squared_euclidean",
            "k": 50,
            "tie_rule": "smaller_train_anchor_index_first",
            "market_knn_error": float(market_knn_error.mean()),
            "context_knn_error": float(context_knn_error.mean()),
            "random_error": float(random_error.mean()),
            "delta_geometry": delta_geometry,
            "delta_random": delta_random,
            "pass": gate2_pass,
        },
        "overall": (
            "V0.7 DEVELOPMENT_GO"
            if gate1_pass and gate2_pass
            else "V0.7 DEVELOPMENT_NO_GO"
        ),
        "test_consumed": False,
    }
