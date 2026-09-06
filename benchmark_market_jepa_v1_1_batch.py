"""Bounded Train-only V1.1 micro-batch throughput sweep; never starts an epoch run."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path

import torch

from market_jepa.model.jepa import jepa_loss
from market_jepa.train.trainer import configure_determinism
from market_jepa_v1_1 import MarketJEPAV11, V11ContractDataset, V11DataStore
from market_jepa_v1_1.config import load_v11_config
from market_jepa_v1_1.dataset import fit_v11_shared_scaler
from market_jepa_v1_1.formal_training import write_json
from market_jepa_v1_1.training import (
    V11Trainer, assert_all_trainable_gradients, model_inputs,
)


def profile_plan(batch_sizes: tuple[int, ...]) -> list[dict]:
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    return [
        {
            "batch_size": batch_size,
            "gradient_accumulation": 2 if batch_size == 64 else 1,
            "effective_batch": batch_size * (2 if batch_size == 64 else 1),
            "formal_effective_batch_128_candidate": batch_size in {64, 128},
        }
        for batch_size in batch_sizes
    ]


class NvidiaDmon:
    def __init__(self) -> None:
        self.samples: list[dict[str, int]] = []
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        def collect() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                values = line.split()
                if len(values) >= 3 and values[0].isdigit():
                    self.samples.append({"gpu_util_percent": int(values[1]), "memory_util_percent": int(values[2])})

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)
        gpu = [sample["gpu_util_percent"] for sample in self.samples]
        memory = [sample["memory_util_percent"] for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "gpu_util_mean_percent": statistics.fmean(gpu) if gpu else None,
            "gpu_util_peak_percent": max(gpu) if gpu else None,
            "memory_util_mean_percent": statistics.fmean(memory) if memory else None,
            "samples": self.samples,
        }


def benchmark_profile(
    base_config: dict, dataset: V11ContractDataset, plan: dict,
    *, warmup_updates: int, measured_updates: int, num_workers: int,
) -> dict:
    config = deepcopy(base_config)
    config["training"].update(
        batch_size=plan["batch_size"],
        gradient_accumulation=plan["gradient_accumulation"],
        max_epochs=1,
        num_workers=num_workers,
        checkpoint_dir="artifacts/performance/v1_1_batch_sweep_unused_checkpoints",
    )
    configure_determinism(int(config["training"]["seed"]))
    model = MarketJEPAV11(config["model"])
    total_updates = warmup_updates + measured_updates
    trainer = V11Trainer(
        model, config, dataset, torch.device("cuda"), validation_dataset=None,
        samples_per_epoch=plan["effective_batch"] * total_updates,
        data_manifest_sha256="bounded-throughput-sweep",
    )
    iterator = iter(trainer.train_loader)
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    measured_times: list[float] = []
    measured_data_wait: list[float] = []
    measured_losses: list[float] = []
    skipped_steps = 0
    connectivity = None
    monitor = NvidiaDmon()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor.start()
    started = time.perf_counter()
    try:
        for update in range(total_updates):
            update_started = time.perf_counter()
            data_wait = 0.0
            losses = []
            for _ in range(plan["gradient_accumulation"]):
                wait_started = time.perf_counter()
                raw = next(iterator)
                data_wait += time.perf_counter() - wait_started
                kwargs = model_inputs(raw, torch.device("cuda"))
                trainer._calibrate_amp(kwargs)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = trainer.model(**kwargs)
                    loss, _ = jepa_loss(
                        output,
                        lambda_var=config["training"]["lambda_var"],
                        lambda_cov=config["training"]["lambda_cov"],
                        variance_floor=config["training"]["variance_floor"],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("throughput sweep loss is nonfinite")
                trainer.scaler.scale(loss / plan["gradient_accumulation"]).backward()
                losses.append(float(loss.detach()))
            trainer.scaler.unscale_(trainer.optimizer)
            if connectivity is None:
                connectivity = assert_all_trainable_gradients(trainer.model)
            torch.nn.utils.clip_grad_norm_(
                list(trainer.model.optimizer_parameters()),
                config["training"]["gradient_clip_norm"], error_if_nonfinite=True,
            )
            scale_before = trainer.scaler.get_scale()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            if trainer.scaler.get_scale() < scale_before:
                skipped_steps += 1
            else:
                trainer.model.update_target(config["training"]["ema_tau"])
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - update_started
            if update >= warmup_updates:
                measured_times.append(elapsed)
                measured_data_wait.append(data_wait)
                measured_losses.append(statistics.fmean(losses))
    finally:
        utilization = monitor.stop()
    total_elapsed = time.perf_counter() - started
    measured_samples = measured_updates * plan["effective_batch"]
    measured_elapsed = sum(measured_times)
    return {
        **plan,
        "status": "PASS",
        "num_workers": num_workers,
        "warmup_optimizer_updates": warmup_updates,
        "measured_optimizer_updates": measured_updates,
        "measured_samples": measured_samples,
        "samples_per_second": measured_samples / measured_elapsed,
        "optimizer_step_mean_seconds": statistics.fmean(measured_times),
        "optimizer_step_p50_seconds": statistics.median(measured_times),
        "data_wait_mean_seconds": statistics.fmean(measured_data_wait),
        "data_wait_fraction": sum(measured_data_wait) / measured_elapsed,
        "mean_loss": statistics.fmean(measured_losses),
        "skipped_amp_steps": skipped_steps,
        "amp_calibration": trainer.amp_calibration,
        "gradient_connectivity": connectivity,
        "peak_vram_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved": int(torch.cuda.max_memory_reserved()),
        "wall_seconds_including_warmup": total_elapsed,
        "gpu_utilization": utilization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1_1/market_jepa_v1_1.yaml")
    parser.add_argument("--output", type=Path, default=Path("artifacts/performance/v1_1_batch_sweep.json"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 192, 256])
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=10)
    parser.add_argument("--scaler-anchors-per-commodity", type=int, default=32)
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader workers (default: training.num_workers from config)",
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (
        args.warmup_updates < 1 or args.measured_updates < 1
        or args.threads < 1 or (args.num_workers is not None and args.num_workers < 0)
    ):
        parser.error("warmup/measured updates and threads must be positive; workers must be nonnegative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    torch.set_num_threads(args.threads)
    config = load_v11_config(args.config)
    num_workers = int(config["training"]["num_workers"] if args.num_workers is None else args.num_workers)
    train_commodities = tuple(config["data"]["train_commodities"])
    setup_started = time.perf_counter()
    store = V11DataStore.from_directory(config["data"]["root"], train_commodities)
    unscaled = V11ContractDataset(store, config, role="train")
    scaler = fit_v11_shared_scaler(unscaled, args.scaler_anchors_per_commodity)
    dataset = V11ContractDataset(store, config, role="train", scaler=scaler)
    setup_seconds = time.perf_counter() - setup_started
    profiles = []
    for plan in profile_plan(tuple(args.batch_sizes)):
        try:
            result = benchmark_profile(
                config, dataset, plan,
                warmup_updates=args.warmup_updates,
                measured_updates=args.measured_updates,
                num_workers=num_workers,
            )
        except torch.cuda.OutOfMemoryError as error:
            result = {**plan, "status": "OOM", "error": str(error)}
        profiles.append(result)
        gc.collect()
        torch.cuda.empty_cache()
    candidates = [
        profile for profile in profiles
        if profile["formal_effective_batch_128_candidate"] and profile["status"] == "PASS"
    ]
    fastest = max(candidates, key=lambda item: item["samples_per_second"]) if candidates else None
    batch64 = next((profile for profile in candidates if profile["batch_size"] == 64), None)
    minimum_gain = 0.05
    if fastest is not None and batch64 is not None:
        gain = fastest["samples_per_second"] / batch64["samples_per_second"] - 1
        recommended = fastest if gain >= minimum_gain else batch64
    else:
        gain = None
        recommended = fastest
    report = {
        "status": "PASS" if candidates else "FAIL",
        "purpose": "bounded throughput calibration; no formal epoch/checkpoint",
        "model_scale": "V1.1-S",
        "train_commodities": list(train_commodities),
        "held_out_commodity": config["data"]["held_out_commodity"],
        "held_out_read": False,
        "dataset_anchors": len(dataset),
        "setup_seconds": setup_seconds,
        "scaler_anchors_per_commodity": args.scaler_anchors_per_commodity,
        "num_workers": num_workers,
        "profiles": profiles,
        "batch_selection_rule": "change from 64x2 only if measured throughput gain is at least 5%",
        "fastest_gain_vs_batch64": gain,
        "recommended_effective_batch_128_profile": None if recommended is None else {
            key: recommended[key]
            for key in ("batch_size", "gradient_accumulation", "samples_per_second", "peak_vram_allocated")
        },
        "formal_config_changed_by_benchmark": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
