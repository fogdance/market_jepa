from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from market_jepa.data.dataset import MarketDataset, collate_market_batch
from market_jepa.implementation import (
    changed_implementation_files,
    implementation_manifest,
    manifest_sha256,
)
from market_jepa.model.jepa import MarketJEPA, jepa_loss

from .checkpoint import save_checkpoint
from .diagnostics import LatentAccumulator, MetricAverage


def configure_determinism(seed: int) -> None:
    # Required by CUDA deterministic GEMM; set before any CUDA tensor operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def capture_rng_state(include_cuda: bool = False) -> dict[str, Any]:
    if include_cuda and not torch.cuda.is_available():
        raise RuntimeError("cannot capture CUDA RNG state because CUDA is unavailable")
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("checkpoint RNG state schema is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG device count differs from runtime")
        torch.cuda.set_rng_state_all(cuda_state)


class EpochSampler(Sampler[int]):
    def __init__(self, length: int, effective_batch: int, seed: int) -> None:
        self.length = length
        self.effective_batch = effective_batch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.length, generator=generator).tolist()
        usable = self.length - self.length % self.effective_batch
        return iter(order[:usable])

    def __len__(self) -> int:
        return self.length - self.length % self.effective_batch


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


class Trainer:
    def __init__(
        self,
        model: MarketJEPA,
        config: dict[str, Any],
        train_dataset: MarketDataset,
        validation_dataset: MarketDataset,
        source_sha256: str,
        preflight_metadata: dict[str, Any],
        device: torch.device,
        runtime_options: dict[str, Any] | None = None,
        *,
        checkpoint_selection: str = "validation_h64",
        evaluate_validation_during_training: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.source_sha256 = source_sha256
        self.preflight_metadata = preflight_metadata
        self.device = device
        if checkpoint_selection not in {"validation_h64", "fixed_budget_final"}:
            raise ValueError(f"unknown checkpoint selection rule: {checkpoint_selection}")
        if checkpoint_selection == "validation_h64" and not evaluate_validation_during_training:
            raise ValueError("validation checkpoint selection requires validation evaluation")
        self.checkpoint_selection = checkpoint_selection
        self.evaluate_validation_during_training = bool(evaluate_validation_during_training)
        self.implementation_manifest = implementation_manifest()
        self.implementation_sha256 = manifest_sha256(self.implementation_manifest)
        training = config["training"]
        supplied_runtime = runtime_options or {}
        unknown_runtime = set(supplied_runtime) - {
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
        }
        if unknown_runtime:
            raise ValueError(f"unknown runtime options: {sorted(unknown_runtime)}")
        workers = supplied_runtime.get("num_workers", training["num_workers"])
        persistent = supplied_runtime.get("persistent_workers", False)
        prefetch = supplied_runtime.get("prefetch_factor")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not isinstance(persistent, bool):
            raise ValueError("persistent_workers must be a boolean")
        if prefetch is not None and (
            isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch <= 0
        ):
            raise ValueError("prefetch_factor must be a positive integer or null")
        self.runtime_options = {
            "num_workers": workers,
            "persistent_workers": persistent,
            "prefetch_factor": prefetch,
        }
        if not self.runtime_options["num_workers"]:
            self.runtime_options["persistent_workers"] = False
            self.runtime_options["prefetch_factor"] = None
        elif self.runtime_options["prefetch_factor"] is None:
            self.runtime_options["prefetch_factor"] = 2
        self.accumulation = int(training["gradient_accumulation"])
        self.amp_enabled = bool(training["amp"] and device.type == "cuda")
        self.optimizer = torch.optim.AdamW(
            list(model.optimizer_parameters()),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            betas=tuple(training["betas"]),
            eps=training["eps"],
        )
        effective_batch = training["batch_size"] * self.accumulation
        self.sampler = EpochSampler(len(train_dataset), effective_batch, training["seed"])
        self.train_loader_generator = torch.Generator().manual_seed(
            int(training["seed"]) + 10_000
        )
        self.validation_loader_generator = torch.Generator().manual_seed(
            int(training["seed"]) + 20_000
        )
        loader_options: dict[str, Any] = {
            "num_workers": self.runtime_options["num_workers"],
            "pin_memory": device.type == "cuda",
        }
        if self.runtime_options["num_workers"]:
            loader_options.update(
                persistent_workers=self.runtime_options["persistent_workers"],
                prefetch_factor=int(self.runtime_options["prefetch_factor"]),
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=training["batch_size"],
            sampler=self.sampler,
            collate_fn=collate_market_batch,
            generator=self.train_loader_generator,
            **loader_options,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=training["batch_size"],
            shuffle=False,
            collate_fn=collate_market_batch,
            generator=self.validation_loader_generator,
            **loader_options,
        )
        self.steps_per_epoch = len(self.sampler) // effective_batch
        if self.steps_per_epoch == 0:
            raise ValueError("training dataset is smaller than one effective batch")
        total_steps = self.steps_per_epoch * training["max_epochs"]
        warmup_steps = math.ceil(training["warmup_ratio"] * total_steps)

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.start_epoch = 0
        self.global_step = 0
        self.best_error = math.inf
        self.best_epoch = -1
        self.skipped_optimizer_steps = 0
        self.resume_history: list[dict[str, Any]] = []
        self.checkpoint_dir = Path(training["checkpoint_dir"]) / config["experiment_id"]

    def resume(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint["source_sha256"] != self.source_sha256:
            raise ValueError("cannot resume: source hash changed")
        if checkpoint["config"] != self.config:
            raise ValueError("cannot resume: frozen configuration changed")
        if checkpoint["normalizers"] != self.train_dataset.data.normalizers.to_dict():
            raise ValueError("cannot resume: normalization statistics changed")
        if checkpoint.get("runtime_options") != self.runtime_options:
            raise ValueError("cannot resume exactly: runtime options changed")
        if checkpoint.get("checkpoint_selection", "validation_h64") != self.checkpoint_selection:
            raise ValueError("cannot resume exactly: checkpoint selection rule changed")
        if checkpoint.get("evaluate_validation_during_training", True) != self.evaluate_validation_during_training:
            raise ValueError("cannot resume exactly: validation timing changed")
        if "loader_rng_state" not in checkpoint:
            raise ValueError("cannot resume exactly: checkpoint has no DataLoader RNG state")
        if "rng_state" not in checkpoint:
            raise ValueError("cannot resume exactly: checkpoint has no RNG state")
        if "implementation_manifest" not in checkpoint or "implementation_sha256" not in checkpoint:
            raise ValueError("cannot resume exactly: checkpoint has no implementation manifest")
        if checkpoint["implementation_sha256"] != self.implementation_sha256:
            changed = changed_implementation_files(
                checkpoint["implementation_manifest"], self.implementation_manifest
            )
            raise ValueError(f"cannot resume: implementation changed: {changed}")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_error = float(checkpoint["best_error"])
        self.best_epoch = int(checkpoint["best_epoch"])
        self.skipped_optimizer_steps = int(checkpoint.get("skipped_optimizer_steps", 0))
        self.resume_history = list(checkpoint.get("history", []))
        if len(self.resume_history) != self.start_epoch:
            raise ValueError("cannot resume exactly: checkpoint history does not match epoch")
        restore_rng_state(checkpoint["rng_state"])
        self.train_loader_generator.set_state(checkpoint["loader_rng_state"]["train"])
        self.validation_loader_generator.set_state(
            checkpoint["loader_rng_state"]["validation"]
        )

    def _finish_optimizer_step(self) -> bool:
        """Advance optimizer-dependent state only when AMP applies the update."""
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        step_succeeded = not self.amp_enabled or scale_after >= scale_before
        self.optimizer.zero_grad(set_to_none=True)
        if step_succeeded:
            self.scheduler.step()
            self.model.update_target(self.config["training"]["ema_tau"])
            self.global_step += 1
        else:
            self.skipped_optimizer_steps += 1
        return step_succeeded

    def _epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        averages = MetricAverage()
        latent = LatentAccumulator(self.config["model"]["latent_dim"])
        targets = {
            horizon: LatentAccumulator(self.config["model"]["latent_dim"])
            for horizon in self.model.horizons
        }
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            batch = _move(batch, self.device)
            grad_context = torch.enable_grad if training else torch.inference_mode
            with grad_context(), torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                output = self.model(batch)
                loss, metrics = jepa_loss(
                    output,
                    lambda_var=self.config["training"]["lambda_var"],
                    lambda_cov=self.config["training"]["lambda_cov"],
                    variance_floor=self.config["training"]["variance_floor"],
                )
            if training:
                self.scaler.scale(loss / self.accumulation).backward()
                if (batch_index + 1) % self.accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.optimizer_parameters()),
                        self.config["training"]["gradient_clip_norm"],
                    )
                    self._finish_optimizer_step()
            batch_size = batch["minute_market"].shape[0]
            averages.update(metrics, batch_size)
            latent.update(output["z_market"])
            for horizon in self.model.horizons:
                targets[horizon].update(output["targets"][horizon])
        result = averages.result()
        threshold = self.config["training"]["collapse_threshold"]
        result.update({f"z_market_{k}": v for k, v in latent.metrics(threshold).items()})
        for horizon, accumulator in targets.items():
            result.update(
                {f"target_h{horizon}_{k}": v for k, v in accumulator.metrics(threshold).items()}
            )
        return result

    def _checkpoint_state(
        self, epoch: int, metrics: dict[str, Any], history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        current_manifest = implementation_manifest()
        current_digest = manifest_sha256(current_manifest)
        if current_digest != self.implementation_sha256:
            changed = changed_implementation_files(
                self.implementation_manifest, current_manifest
            )
            raise RuntimeError(
                f"implementation changed during training; refusing checkpoint: {changed}"
            )
        return {
            "design_version": self.config["design_version"],
            "config": self.config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_error": self.best_error,
            "best_epoch": self.best_epoch,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "checkpoint_selection": self.checkpoint_selection,
            "evaluate_validation_during_training": self.evaluate_validation_during_training,
            "normalizers": self.train_dataset.data.normalizers.to_dict(),
            "source_sha256": self.source_sha256,
            "preflight": self.preflight_metadata,
            "metrics": metrics,
            "history": history,
            "feature_schema": {
                "minute_market": list(self.train_dataset.data.normalizers.minute_market.names),
                "minute_context": list(self.train_dataset.data.normalizers.minute_context.names),
                "daily_market": list(self.train_dataset.data.normalizers.daily_market.names),
                "daily_context": list(self.train_dataset.data.normalizers.daily_context.names),
                "weekly_market": list(self.train_dataset.data.normalizers.weekly_market.names),
                "weekly_context": list(self.train_dataset.data.normalizers.weekly_context.names),
            },
            "rng_state": capture_rng_state(include_cuda=self.device.type == "cuda"),
            "runtime_options": self.runtime_options,
            "loader_rng_state": {
                "train": self.train_loader_generator.get_state(),
                "validation": self.validation_loader_generator.get_state(),
            },
            "implementation_manifest": self.implementation_manifest,
            "implementation_sha256": self.implementation_sha256,
        }

    def _shutdown_loader_workers(self) -> None:
        for loader in (self.train_loader, self.validation_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None

    def fit(self, stop_before_epoch: int | None = None) -> list[dict[str, Any]]:
        """Run the frozen epoch range; stop_before_epoch exists only for interruption tests."""
        history_path = self.checkpoint_dir / "history.json"
        history = list(self.resume_history)
        final_epoch = self.config["training"]["max_epochs"]
        if stop_before_epoch is not None:
            if stop_before_epoch <= self.start_epoch or stop_before_epoch > final_epoch:
                raise ValueError("invalid stop_before_epoch")
            final_epoch = stop_before_epoch
        selection_horizon = 64 if 64 in self.model.horizons else self.model.horizons[len(self.model.horizons) // 2]
        try:
            for epoch in range(self.start_epoch, final_epoch):
                self.sampler.set_epoch(epoch)
                skipped_before = self.skipped_optimizer_steps
                train_metrics = self._epoch(self.train_loader, training=True)
                validation_metrics = (
                    self._epoch(self.validation_loader, training=False)
                    if self.evaluate_validation_during_training
                    else None
                )
                record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "skipped_optimizer_steps": self.skipped_optimizer_steps - skipped_before,
                    "skipped_optimizer_steps_total": self.skipped_optimizer_steps,
                }
                is_best = False
                if self.checkpoint_selection == "validation_h64":
                    assert validation_metrics is not None
                    selection = validation_metrics[f"prediction_loss_h{selection_horizon}"]
                    is_best = selection < self.best_error
                    if is_best:
                        self.best_error = selection
                        self.best_epoch = epoch
                history.append(record)
                state = self._checkpoint_state(epoch, record, history)
                save_checkpoint(state, self.checkpoint_dir / "last.pt")
                if is_best:
                    save_checkpoint(state, self.checkpoint_dir / "best.pt")
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(record, ensure_ascii=False))
        finally:
            self._shutdown_loader_workers()
        return history
