from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from market_jepa.data.pipeline import NormalizerBundle
from market_jepa.data.schema import CONTEXT_FEATURES, MARKET_FEATURES
from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.predictive_state import (
    EndToEndPredictiveState,
    FixedRFF,
    PredictiveStateDataset,
    TargetPreprocessing,
    canonical_sha256,
    collate_predictive_state,
    context_only_batch,
    normalizer_hash,
    state_loss,
)
from market_jepa.train.checkpoint import load_checkpoint, save_checkpoint
from market_jepa.train.trainer import (
    EpochSampler,
    capture_rng_state,
    configure_determinism,
    restore_rng_state,
)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = __import__("hashlib").sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initial_state(model_config: dict[str, Any], seed: int) -> tuple[dict[str, torch.Tensor], str]:
    configure_determinism(seed)
    model = EndToEndPredictiveState(model_config)
    state = deepcopy(model.state_dict())
    return state, state_dict_sha256(state)


def schedule_factor(
    step: int,
    steps_per_epoch: int,
    *,
    horizon_epochs: int = 100,
    warmup_ratio: float = 0.05,
) -> float:
    total_steps = steps_per_epoch * horizon_epochs
    warmup_steps = math.ceil(warmup_ratio * total_steps)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    cpu_names = {"daily_lengths", "weekly_lengths", "anchor_index", "timestamp_ns", "trading_day_ns"}
    return {
        name: value if name in cpu_names else value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


class PredictiveStateRun:
    """One independent deterministic Market or Context training run."""

    def __init__(
        self,
        *,
        run_name: str,
        stage: str,
        context_only: bool,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        train_dataset: PredictiveStateDataset,
        dev_dataset: PredictiveStateDataset | None,
        target: TargetPreprocessing,
        normalizers: NormalizerBundle,
        initial_model_state: dict[str, torch.Tensor],
        initial_model_hash: str,
        output_dir: Path,
        source_path: Path,
        source_sha256: str,
        protocol_path: Path,
        protocol_sha256: str,
        config: dict[str, Any],
        epochs_to_run: int,
        device: torch.device,
        implementation: dict[str, Any] | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        if run_name not in {"market", "context"}:
            raise ValueError("run_name must be market or context")
        if stage not in {"inner", "final"}:
            raise ValueError("stage must be inner or final")
        if not 1 <= epochs_to_run <= 100:
            raise ValueError("epoch budget must be between 1 and 100")
        if stage == "inner" and (epochs_to_run != 100 or dev_dataset is None):
            raise ValueError("Inner run requires all 100 epochs and an Inner Dev dataset")
        if stage == "final" and dev_dataset is not None:
            raise ValueError("Final run cannot use a checkpoint-selection dataset")

        self.run_name = run_name
        self.stage = stage
        self.context_only = context_only
        self.training = training_config
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.target = target
        self.normalizers = normalizers
        self.initial_model_hash = initial_model_hash
        self.output_dir = output_dir
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self.protocol_path = protocol_path
        self.protocol_sha256 = protocol_sha256
        self.config = config
        self.epochs_to_run = epochs_to_run
        self.device = device
        self.implementation = implementation or implementation_manifest()
        self.implementation_sha256 = manifest_sha256(self.implementation)
        self.run_metadata = dict(run_metadata or {})

        configure_determinism(int(training_config["seed"]))
        self.model = EndToEndPredictiveState(model_config).to(device)
        self.model.load_state_dict(initial_model_state)
        if state_dict_sha256(self.model.state_dict()) != initial_model_hash:
            raise RuntimeError("run did not start from the frozen initial state")
        self.parameters = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.model.optimizer_groups(training_config),
            betas=tuple(training_config["betas"]),
            eps=float(training_config["eps"]),
        )
        self.accumulation = int(training_config["gradient_accumulation"])
        effective_batch = int(training_config["microbatch_size"]) * self.accumulation
        self.sampler = EpochSampler(len(train_dataset), effective_batch, int(training_config["seed"]))
        workers = int(training_config["num_workers"])
        loader_options: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
        }
        if workers:
            loader_options.update(
                persistent_workers=bool(training_config["persistent_workers"]),
                prefetch_factor=int(training_config["prefetch_factor"]),
            )
        self.train_generator = torch.Generator().manual_seed(int(training_config["seed"]) + 10_000)
        self.dev_generator = torch.Generator().manual_seed(int(training_config["seed"]) + 20_000)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_config["microbatch_size"]),
            sampler=self.sampler,
            collate_fn=collate_predictive_state,
            generator=self.train_generator,
            **loader_options,
        )
        self.dev_loader = (
            DataLoader(
                dev_dataset,
                batch_size=int(training_config["microbatch_size"]),
                shuffle=False,
                collate_fn=collate_predictive_state,
                generator=self.dev_generator,
                **loader_options,
            )
            if dev_dataset is not None
            else None
        )
        steps_per_epoch = len(self.sampler) // effective_batch
        if steps_per_epoch <= 0:
            raise ValueError("training population is smaller than one effective batch")
        def schedule(step: int) -> float:
            return schedule_factor(
                step,
                steps_per_epoch,
                horizon_epochs=100,
                warmup_ratio=float(training_config["warmup_ratio"]),
            )

        self.scheduler_horizon_epochs = 100
        self.scheduler = LambdaLR(self.optimizer, schedule)
        self.amp_enabled = bool(training_config["amp"] and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.fixed_rff = FixedRFF(target.rff).to(device).eval()
        self.start_epoch = 0
        self.global_step = 0
        self.best_epoch = -1
        self.best_dev_loss = math.inf
        self.history: list[dict[str, Any]] = []
        self.last_path = output_dir / "last.pt"

    def _epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        count = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(loader):
            batch = _move(raw_batch, self.device)
            if self.context_only:
                batch = context_only_batch(batch)
            grad_context = torch.enable_grad if training else torch.inference_mode
            with grad_context(), torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                output = self.model(batch)
            with torch.no_grad():
                target = self.fixed_rff(batch["y"])
            loss, per_sample = state_loss(output["state"], target)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite {self.stage}/{self.run_name} state loss"
                )
            if training:
                self.scaler.scale(loss / self.accumulation).backward()
                if (batch_index + 1) % self.accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters, float(self.training["gradient_clip_norm"]))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1
            total += float(per_sample.detach().sum().item())
            count += len(per_sample)
        return total / count

    def _checkpoint(self, epoch: int) -> dict[str, Any]:
        current_implementation = implementation_manifest()
        current_digest = manifest_sha256(current_implementation)
        if current_digest != self.implementation_sha256:
            raise RuntimeError("implementation changed during a formal V0.7 run")
        return {
            "protocol_version": "0.7.1",
            "stage": self.stage,
            "run_name": self.run_name,
            "context_only": self.context_only,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "sampler_epoch": epoch,
            "epochs_to_run": self.epochs_to_run,
            "scheduler_horizon_epochs": self.scheduler_horizon_epochs,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_dev_loss": self.best_dev_loss,
            "budget_hit_ceiling": self.best_epoch == 99 if self.stage == "inner" else None,
            "history": self.history,
            "normalizers": self.normalizers.to_dict(),
            "normalizer_sha256": normalizer_hash(self.normalizers),
            "target_preprocessing": self.target.to_dict(),
            "target_preprocessing_sha256": canonical_sha256(
                {"audit": self.target.audit, "hashes": self.target.hashes}
            ),
            "initial_state_sha256": self.initial_model_hash,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "protocol_path": str(self.protocol_path),
            "protocol_sha256": self.protocol_sha256,
            "config": self.config,
            "feature_schema": {
                "minute_market": list(MARKET_FEATURES),
                "minute_context": list(CONTEXT_FEATURES),
                "daily_market": list(MARKET_FEATURES),
                "daily_context": ["source_bar_count"],
                "weekly_market": list(MARKET_FEATURES),
                "weekly_context": ["source_bar_count"],
            },
            "implementation_manifest": self.implementation,
            "implementation_sha256": self.implementation_sha256,
            "run_metadata": self.run_metadata,
            "rng_state": capture_rng_state(include_cuda=self.device.type == "cuda"),
            "loader_rng_state": {
                "train": self.train_generator.get_state(),
                "dev": self.dev_generator.get_state(),
            },
            "test_consumed": False,
        }

    def resume(self) -> None:
        checkpoint = load_checkpoint(self.last_path, map_location="cpu")
        checks = {
            "protocol_version": "0.7.1",
            "protocol_sha256": self.protocol_sha256,
            "source_sha256": self.source_sha256,
            "implementation_sha256": self.implementation_sha256,
            "initial_state_sha256": self.initial_model_hash,
            "normalizer_sha256": normalizer_hash(self.normalizers),
            "stage": self.stage,
            "run_name": self.run_name,
            "context_only": self.context_only,
            "epochs_to_run": self.epochs_to_run,
            "scheduler_horizon_epochs": 100,
            "run_metadata": self.run_metadata,
            "test_consumed": False,
        }
        for name, expected in checks.items():
            if checkpoint.get(name) != expected:
                raise ValueError(f"cannot resume V0.7: {name} changed")
        if int(checkpoint.get("sampler_epoch", -1)) != int(checkpoint["epoch"]):
            raise ValueError("cannot resume V0.7: sampler epoch is missing or inconsistent")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_epoch = int(checkpoint["best_epoch"])
        self.best_dev_loss = float(checkpoint["best_dev_loss"])
        self.history = list(checkpoint["history"])
        if len(self.history) != self.start_epoch:
            raise ValueError("cannot resume V0.7: history length differs")
        restore_rng_state(checkpoint["rng_state"])
        self.train_generator.set_state(checkpoint["loader_rng_state"]["train"])
        self.dev_generator.set_state(checkpoint["loader_rng_state"]["dev"])

    def fit(self, resume: bool = False) -> dict[str, Any]:
        if resume and self.last_path.exists():
            self.resume()
        started_run = time.perf_counter()
        try:
            for epoch in range(self.start_epoch, self.epochs_to_run):
                started = time.perf_counter()
                self.sampler.set_epoch(epoch)
                train_loss = self._epoch(self.train_loader, training=True)
                dev_loss = self._epoch(self.dev_loader, training=False) if self.dev_loader is not None else None
                if dev_loss is not None and dev_loss < self.best_dev_loss:
                    self.best_dev_loss = dev_loss
                    self.best_epoch = epoch
                record = {
                    "epoch": epoch,
                    "train_state_mse": train_loss,
                    "inner_dev_state_mse": dev_loss,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                self.history.append(record)
                save_checkpoint(self._checkpoint(epoch), self.last_path)
                _atomic_json(self.history, self.output_dir / "history.json")
                print(json.dumps({"stage": self.stage, "run": self.run_name, **record}))
        finally:
            for loader in (self.train_loader, self.dev_loader):
                if loader is not None:
                    iterator = getattr(loader, "_iterator", None)
                    if iterator is not None:
                        iterator._shutdown_workers()
                        loader._iterator = None
        selected_epoch = self.best_epoch if self.stage == "inner" else self.epochs_to_run - 1
        if self.stage == "inner" and selected_epoch < 0:
            raise RuntimeError("Inner run completed without a finite checkpoint-selection loss")
        summary = {
            "stage": self.stage,
            "run_name": self.run_name,
            "selected_epoch": selected_epoch,
            "selected_budget": selected_epoch + 1,
            "budget_hit_ceiling": selected_epoch == 99 if self.stage == "inner" else False,
            "best_dev_loss": self.best_dev_loss if self.stage == "inner" else None,
            "epochs_completed": len(self.history),
            "elapsed_seconds_this_process": time.perf_counter() - started_run,
            "checkpoint": str(self.last_path),
            "test_consumed": False,
        }
        _atomic_json(summary, self.output_dir / "complete.json")
        return summary
