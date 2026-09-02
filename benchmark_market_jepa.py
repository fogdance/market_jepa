from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from market_jepa.config import load_config
from market_jepa.data import (
    CONTEXT_FEATURES,
    MARKET_FEATURES,
    MarketDataset,
    collate_market_batch,
    prepare_market_data,
)
from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.model import MarketJEPA, jepa_loss
from market_jepa.train import configure_determinism
from market_jepa.train.diagnostics import LatentAccumulator, MetricAverage
from market_jepa.train.trainer import EpochSampler


def _loader(dataset: MarketDataset, config: dict[str, Any], workers: int, prefetch: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config["training"]["batch_size"],
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": collate_market_batch,
        "pin_memory": True,
        "generator": torch.Generator().manual_seed(config["training"]["seed"]),
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=prefetch)
    return DataLoader(**kwargs)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _batch_hash(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()

    def update(value: Any, prefix: str) -> None:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(prefix.encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                update(value[key], f"{prefix}/{key}")

    update(batch, "batch")
    return digest.hexdigest()


def _loader_benchmark(
    dataset: MarketDataset,
    config: dict[str, Any],
    workers: int,
    prefetch: int,
    batches: int,
) -> dict[str, Any]:
    loader = _loader(dataset, config, workers, prefetch)
    iterator = iter(loader)
    waits: list[float] = []
    hashes: list[str] = []
    sequence_digest = hashlib.sha256()
    for index in range(batches):
        started = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - started)
        batch_digest = _batch_hash(batch)
        sequence_digest.update(batch_digest.encode("ascii"))
        if index < 3:
            hashes.append(batch_digest)
    return {
        "workers": workers,
        "prefetch_factor": prefetch if workers else None,
        "wait_seconds": _summary(waits),
        "batches_per_second": float(batches / sum(waits)),
        "first_batch_hashes": hashes,
        "measured_sequence_sha256": sequence_digest.hexdigest(),
    }


def _optimizer(model: MarketJEPA, config: dict[str, Any]) -> torch.optim.AdamW:
    training = config["training"]
    return torch.optim.AdamW(
        list(model.optimizer_parameters()),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
    )


def _run_fragment(
    model: MarketJEPA,
    dataset: MarketDataset,
    config: dict[str, Any],
    workers: int,
    prefetch: int,
    training: bool,
    warmup: int,
) -> dict[str, Any]:
    """Measure Trainer-equivalent wall time without benchmark-only CUDA syncs."""
    device = torch.device("cuda")
    loader = _loader(dataset, config, workers, prefetch)
    iterator = iter(loader)
    optimizer = _optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    accumulation = config["training"]["gradient_accumulation"]
    timings: dict[str, list[float]] = {
        name: [] for name in ("loader", "diagnostics", "wall")
    }
    averages = MetricAverage()
    latent = LatentAccumulator(config["model"]["latent_dim"])
    targets = {
        horizon: LatentAccumulator(config["model"]["latent_dim"])
        for horizon in model.horizons
    }
    model.train(training)
    optimizer.zero_grad(set_to_none=True)
    for batch_index in range(len(loader)):
        wall_started = time.perf_counter()
        started = time.perf_counter()
        batch = next(iterator)
        loader_seconds = time.perf_counter() - started

        batch = _move(batch, device)
        grad_context = torch.enable_grad if training else torch.inference_mode
        with grad_context(), torch.amp.autocast(
            "cuda", dtype=torch.float16
        ):
            output = model(batch)
            loss, metrics = jepa_loss(
                output,
                lambda_var=config["training"]["lambda_var"],
                lambda_cov=config["training"]["lambda_cov"],
                variance_floor=config["training"]["variance_floor"],
            )
        if training:
            scaler.scale(loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.optimizer_parameters()),
                    config["training"]["gradient_clip_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                model.update_target(config["training"]["ema_tau"])

        started = time.perf_counter()
        averages.update(metrics, batch["minute_market"].shape[0])
        latent.update(output["z_market"])
        for horizon in model.horizons:
            targets[horizon].update(output["targets"][horizon])
        diagnostics_seconds = time.perf_counter() - started

        if batch_index >= warmup:
            timings["loader"].append(loader_seconds)
            timings["diagnostics"].append(diagnostics_seconds)
            timings["wall"].append(time.perf_counter() - wall_started)
    finalize_started = time.perf_counter()
    averages.result()
    threshold = config["training"]["collapse_threshold"]
    latent.metrics(threshold)
    for accumulator in targets.values():
        accumulator.metrics(threshold)
    return {
        "training": training,
        "measured_batches": len(timings["wall"]),
        "timing_seconds": {name: _summary(values) for name, values in timings.items()},
        "epoch_finalize_seconds": time.perf_counter() - finalize_started,
    }


def _phase_fragment(
    model: MarketJEPA,
    dataset: MarketDataset,
    config: dict[str, Any],
    workers: int,
    prefetch: int,
    batches: int,
) -> dict[str, Any]:
    """Short phase diagnostic. Explicit syncs make it invalid for duration projection."""
    device = torch.device("cuda")
    loader = _loader(dataset, config, workers, prefetch)
    optimizer = _optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    accumulation = config["training"]["gradient_accumulation"]
    phases = {
        name: []
        for name in (
            "loader",
            "h2d",
            "online_minute_market",
            "online_minute_context",
            "online_daily",
            "online_weekly",
            "online_fusion_predictors",
            "target_h16",
            "target_h64",
            "target_h256",
            "loss",
            "backward",
            "optimizer_ema",
        )
    }
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    for batch_index in range(min(batches, len(loader))):
        started = time.perf_counter()
        batch = next(iterator)
        phases["loader"].append(time.perf_counter() - started)

        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = _move(batch, device)
        torch.cuda.synchronize()
        phases["h2d"].append(time.perf_counter() - started)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            started = time.perf_counter()
            minute_market = model.online.minute_market(batch["minute_market"])
            torch.cuda.synchronize()
            phases["online_minute_market"].append(time.perf_counter() - started)

            started = time.perf_counter()
            minute_context = model.online.minute_context(batch["minute_context"])
            torch.cuda.synchronize()
            phases["online_minute_context"].append(time.perf_counter() - started)

            started = time.perf_counter()
            daily = model.online.daily(batch["daily"], batch["daily_lengths"])
            torch.cuda.synchronize()
            phases["online_daily"].append(time.perf_counter() - started)

            started = time.perf_counter()
            weekly = model.online.weekly(batch["weekly"], batch["weekly_lengths"])
            torch.cuda.synchronize()
            phases["online_weekly"].append(time.perf_counter() - started)

            started = time.perf_counter()
            z_market = model.online.fusion(
                torch.cat([minute_market, minute_context, daily, weekly], dim=-1)
            )
            predictions = {
                h: model.predictors[str(h)](z_market) for h in model.horizons
            }
            torch.cuda.synchronize()
            phases["online_fusion_predictors"].append(time.perf_counter() - started)

            with torch.no_grad():
                target_values = {}
                for horizon in model.horizons:
                    started = time.perf_counter()
                    target_values[horizon] = model.target_minute(
                        batch["targets"][horizon]
                    )
                    torch.cuda.synchronize()
                    phases[f"target_h{horizon}"].append(
                        time.perf_counter() - started
                    )

            started = time.perf_counter()
            output = {
                "z_market": z_market,
                "predictions": predictions,
                "targets": target_values,
            }
            loss, _ = jepa_loss(
                output,
                lambda_var=config["training"]["lambda_var"],
                lambda_cov=config["training"]["lambda_cov"],
                variance_floor=config["training"]["variance_floor"],
            )
            torch.cuda.synchronize()
            phases["loss"].append(time.perf_counter() - started)

        started = time.perf_counter()
        scaler.scale(loss / accumulation).backward()
        torch.cuda.synchronize()
        phases["backward"].append(time.perf_counter() - started)
        optimizer_seconds = 0.0
        if (batch_index + 1) % accumulation == 0:
            started = time.perf_counter()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.optimizer_parameters()),
                config["training"]["gradient_clip_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.update_target(config["training"]["ema_tau"])
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter() - started
        phases["optimizer_ema"].append(optimizer_seconds)
    return {
        "warning": "synchronization overhead included; do not use for duration projection",
        "batches": len(phases["loader"]),
        "phase_seconds": {name: _summary(values) for name, values in phases.items()},
    }


def _take_train_subset(data, config: dict[str, Any], batches: int) -> MarketDataset:
    dataset = MarketDataset(data, config, "train")
    effective = config["training"]["batch_size"] * config["training"]["gradient_accumulation"]
    sampler = EpochSampler(len(dataset), effective, config["training"]["seed"])
    positions = np.fromiter(iter(sampler), dtype=np.int64, count=batches * config["training"]["batch_size"])
    return MarketDataset(data, config, "train", indices=dataset.indices[positions])


def _take_validation_bands(data, config: dict[str, Any], batches_per_band: int) -> MarketDataset:
    dataset = MarketDataset(data, config, "validation")
    count = batches_per_band * config["training"]["batch_size"]
    starts = (0, (len(dataset) - count) // 2, len(dataset) - count)
    indices = np.concatenate([dataset.indices[start : start + count] for start in starts])
    return MarketDataset(data, config, "validation", indices=indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market_jepa_v0.yaml")
    parser.add_argument("--train-batches", type=int, default=500)
    parser.add_argument("--validation-batches-per-band", type=int, default=100)
    parser.add_argument("--loader-batches", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--phase-batches", type=int, default=30)
    parser.add_argument("--worker-candidates", default="0,2,4")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--output", default="artifacts/performance/baseline.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("formal performance benchmark requires CUDA")

    config = load_config(args.config)
    configure_determinism(config["training"]["seed"])
    prepare_started = time.perf_counter()
    data = prepare_market_data(config)
    prepare_seconds = time.perf_counter() - prepare_started
    train_batches = max(args.train_batches, args.loader_batches, args.warmup + 1)
    train_subset = _take_train_subset(data, config, train_batches)
    validation_subset = _take_validation_bands(data, config, args.validation_batches_per_band)

    candidates = [int(value) for value in args.worker_candidates.split(",")]
    loader_results = [
        _loader_benchmark(
            train_subset,
            config,
            workers,
            args.prefetch_factor,
            min(args.loader_batches, len(train_subset) // config["training"]["batch_size"]),
        )
        for workers in candidates
    ]
    expected_hashes = loader_results[0]["first_batch_hashes"]
    expected_sequence = loader_results[0]["measured_sequence_sha256"]
    if any(
        result["first_batch_hashes"] != expected_hashes
        or result["measured_sequence_sha256"] != expected_sequence
        for result in loader_results[1:]
    ):
        raise RuntimeError("worker candidates produced different batch tensors")
    fastest = min(result["wait_seconds"]["mean"] for result in loader_results)
    selected_workers = min(
        result["workers"]
        for result in loader_results
        if result["wait_seconds"]["mean"] <= fastest * 1.03
    )

    device = torch.device("cuda")
    model = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), config["data"]["horizons"], config["model"]
    ).to(device)
    torch.cuda.reset_peak_memory_stats()
    train_measured = MarketDataset(
        data,
        config,
        "train",
        indices=train_subset.indices[: args.train_batches * config["training"]["batch_size"]],
    )
    train_timing = _run_fragment(
        model,
        train_measured,
        config,
        selected_workers,
        args.prefetch_factor,
        training=True,
        warmup=args.warmup,
    )
    validation_timing = _run_fragment(
        model,
        validation_subset,
        config,
        selected_workers,
        args.prefetch_factor,
        training=False,
        warmup=0,
    )
    phase_count = min(
        args.phase_batches,
        len(train_measured) // config["training"]["batch_size"],
    )
    phase_dataset = MarketDataset(
        data,
        config,
        "train",
        indices=train_measured.indices[
            : phase_count * config["training"]["batch_size"]
        ],
    )
    phase_timing = _phase_fragment(
        model,
        phase_dataset,
        config,
        selected_workers,
        args.prefetch_factor,
        phase_count,
    )

    train_full = MarketDataset(data, config, "train")
    validation_full = MarketDataset(data, config, "validation")
    batch_size = config["training"]["batch_size"]
    effective = batch_size * config["training"]["gradient_accumulation"]
    train_microbatches = (len(train_full) - len(train_full) % effective) // batch_size
    validation_batches = math.ceil(len(validation_full) / batch_size)
    train_epoch_seconds = (
        train_timing["timing_seconds"]["wall"]["mean"] * train_microbatches
    )
    validation_epoch_seconds = (
        validation_timing["timing_seconds"]["wall"]["mean"] * validation_batches
    )
    manifest = implementation_manifest()
    report = {
        "config": args.config,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "data_prepare_seconds": prepare_seconds,
        "sample_counts": {"train": len(train_full), "validation": len(validation_full)},
        "batch_counts": {
            "train_microbatches": train_microbatches,
            "validation_batches": validation_batches,
        },
        "loader_candidates": loader_results,
        "selected_runtime": {
            "num_workers": selected_workers,
            "persistent_workers": selected_workers > 0,
            "prefetch_factor": args.prefetch_factor if selected_workers else None,
        },
        "train": train_timing,
        "validation": validation_timing,
        "phase_diagnostic": phase_timing,
        "implementation_manifest": manifest,
        "implementation_sha256": manifest_sha256(manifest),
        "projection": {
            "train_epoch_minutes": train_epoch_seconds / 60.0,
            "validation_epoch_minutes": validation_epoch_seconds / 60.0,
            "zero_validation_50_epoch_hours": train_epoch_seconds * 50 / 3600.0,
            "total_50_epoch_hours": (train_epoch_seconds + validation_epoch_seconds) * 50 / 3600.0,
        },
        "memory_gib": {
            "max_allocated": torch.cuda.max_memory_allocated() / 2**30,
            "max_reserved": torch.cuda.max_memory_reserved() / 2**30,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    runtime = output.with_name("selected_runtime.json")
    runtime.write_text(json.dumps(report["selected_runtime"], indent=2), encoding="utf-8")
    print(json.dumps(report["projection"], indent=2))
    print(json.dumps(report["memory_gib"], indent=2))
    print(f"report={output} runtime={runtime}")


if __name__ == "__main__":
    main()
