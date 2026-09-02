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
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.source_sha256 = source_sha256
        self.preflight_metadata = preflight_metadata
        self.device = device
        training = config["training"]
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
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=training["batch_size"],
            sampler=self.sampler,
            num_workers=training["num_workers"],
            collate_fn=collate_market_batch,
            pin_memory=device.type == "cuda",
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=training["batch_size"],
            shuffle=False,
            num_workers=training["num_workers"],
            collate_fn=collate_market_batch,
            pin_memory=device.type == "cuda",
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
        self.checkpoint_dir = Path(training["checkpoint_dir"]) / config["experiment_id"]

    def resume(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint["source_sha256"] != self.source_sha256:
            raise ValueError("cannot resume: source hash changed")
        if checkpoint["config"] != self.config:
            raise ValueError("cannot resume: frozen configuration changed")
        if checkpoint["normalizers"] != self.train_dataset.data.normalizers.to_dict():
            raise ValueError("cannot resume: normalization statistics changed")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_error = float(checkpoint["best_error"])
        self.best_epoch = int(checkpoint["best_epoch"])

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
            with torch.set_grad_enabled(training), torch.amp.autocast(
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
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.model.update_target(self.config["training"]["ema_tau"])
                    self.global_step += 1
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

    def _checkpoint_state(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
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
            "normalizers": self.train_dataset.data.normalizers.to_dict(),
            "source_sha256": self.source_sha256,
            "preflight": self.preflight_metadata,
            "metrics": metrics,
            "feature_schema": {
                "minute_market": list(self.train_dataset.data.normalizers.minute_market.names),
                "minute_context": list(self.train_dataset.data.normalizers.minute_context.names),
                "daily_market": list(self.train_dataset.data.normalizers.daily_market.names),
                "daily_context": list(self.train_dataset.data.normalizers.daily_context.names),
                "weekly_market": list(self.train_dataset.data.normalizers.weekly_market.names),
                "weekly_context": list(self.train_dataset.data.normalizers.weekly_context.names),
            },
        }

    def fit(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        selection_horizon = 64 if 64 in self.model.horizons else self.model.horizons[len(self.model.horizons) // 2]
        for epoch in range(self.start_epoch, self.config["training"]["max_epochs"]):
            self.sampler.set_epoch(epoch)
            train_metrics = self._epoch(self.train_loader, training=True)
            validation_metrics = self._epoch(self.validation_loader, training=False)
            record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            selection = validation_metrics[f"prediction_loss_h{selection_horizon}"]
            is_best = selection < self.best_error
            if is_best:
                self.best_error = selection
                self.best_epoch = epoch
            state = self._checkpoint_state(epoch, record)
            save_checkpoint(state, self.checkpoint_dir / "last.pt")
            if is_best:
                save_checkpoint(state, self.checkpoint_dir / "best.pt")
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
        history_path = self.checkpoint_dir / "history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        return history
