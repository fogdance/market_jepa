from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from market_jepa.train.checkpoint import save_checkpoint
from market_jepa.train.trainer import capture_rng_state, restore_rng_state

from .checkpoint import validate_v11_checkpoint, v11_implementation_manifest
from .config import FIXED_BUDGET_PROTOCOL, validate_v11_config
from .dataset import V11ContractDataset, collate_v11_batch
from .sampler import HierarchicalCommodityContractSampler, sampling_population_sha256
from market_jepa.implementation import manifest_sha256


MODEL_INPUT_KEYS = (
    "minute_market", "minute_context", "minute_mask", "minute_imc_validity",
    "daily_market", "daily_context", "daily_mask", "daily_imc_validity",
    "current_weekly_market", "current_weekly_context", "current_weekly_mask",
    "current_weekly_imc_validity", "history_weekly_market", "history_weekly_context",
    "history_weekly_mask", "history_weekly_imc_validity",
    "history_weekly_contract_boundary", "target_minute_market",
    "target_minute_imc_validity", "target_minute_mask",
)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def move(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def model_inputs(batch: dict, device: torch.device) -> dict:
    return {key: move(batch[key], device) for key in MODEL_INPUT_KEYS}


def jepa_loss_fp32(
    output: dict[str, Any],
    lambda_var: float = 0.0,
    lambda_cov: float = 0.0,
    variance_floor: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V1.1 JEPA loss with numerically sensitive reductions forced to FP32.

    The model may run under FP16/BF16 autocast, but cosine normalization and
    latent variance/covariance reductions should not. Casting keeps gradients
    connected to the original tensors while avoiding low-precision overflow.
    """
    predictions = {h: value.float() for h, value in output["predictions"].items()}
    targets = {h: value.float() for h, value in output["targets"].items()}
    horizon_losses = {
        horizon: (1.0 - F.cosine_similarity(predictions[horizon], target, dim=-1)).mean()
        for horizon, target in targets.items()
    }
    prediction = torch.stack(list(horizon_losses.values())).mean()

    latent = output["z_market"].float()
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    variance = torch.relu(float(variance_floor) - std).mean()
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance_matrix = centered.T @ centered / max(latent.shape[0] - 1, 1)
    diagonal = torch.diagonal(covariance_matrix)
    covariance = (covariance_matrix.square().sum() - diagonal.square().sum()) / latent.shape[1]

    # Mathematically identical for zero coefficients, but avoids IEEE 0 * inf -> nan
    # if a diagnostic term is extremely large while disabled in the formal objective.
    total = prediction
    if lambda_var:
        total = total + float(lambda_var) * variance
    if lambda_cov:
        total = total + float(lambda_cov) * covariance

    metrics = {f"prediction_loss_h{h}": value.detach() for h, value in horizon_losses.items()}
    metrics.update(
        prediction_loss=prediction.detach(),
        variance_loss=variance.detach(),
        covariance_loss=covariance.detach(),
        total_loss=total.detach(),
    )
    return total, metrics


def _amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported V1.1 amp_dtype: {name}")


def _gradient_elements_finite(parameters: list[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in parameters
    )


def _stable_total_norm_fp64(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float64)
    device = grads[0].device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for grad in grads:
        total = total + grad.detach().double().square().sum()
    return total.sqrt()


def fixed_budget_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


def assert_all_trainable_gradients(model) -> dict:
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    if missing or nonfinite:
        raise RuntimeError(f"V1.1 gradient connectivity failure: missing={missing}, nonfinite={nonfinite}")
    return {"intended_trainable": sum(1 for p in model.parameters() if p.requires_grad), "missing": [], "nonfinite": []}


class V11Trainer:
    checkpoint_selection = "fixed_budget_final"

    def __init__(
        self, model, config: dict, train_dataset: V11ContractDataset, device: torch.device,
        *, validation_dataset: V11ContractDataset | None = None,
        data_manifest_sha256: str = "", sampling_population_digest: str | None = None,
        step_logger: Any | None = None, wandb_run_id: str | None = None,
    ) -> None:
        validate_v11_config(config)
        if train_dataset.scaler is None:
            raise ValueError("V1.1 training requires a frozen shared IMC scaler")
        if model.architecture_config != config["model"]:
            raise ValueError("V1.1 model/config mismatch")
        self.model, self.config, self.device = model.to(device), deepcopy(config), device
        self.model.set_gradient_checkpointing(bool(config["training"].get("gradient_checkpointing", False)))
        self.train_dataset, self.validation_dataset = train_dataset, validation_dataset
        self.data_manifest_sha256 = str(data_manifest_sha256)
        training = config["training"]
        if training.get("protocol_version") != FIXED_BUDGET_PROTOCOL:
            raise ValueError(
                "V11Trainer only runs fixed_sample_budget_v1; legacy epoch checkpoints are evaluation-only"
            )
        self.accumulation = int(training["gradient_accumulation"])
        self.batch_size = int(training["batch_size"])
        self.effective_batch_size = self.batch_size * self.accumulation
        self.max_optimizer_steps = int(training["max_optimizer_steps"])
        self.warmup_optimizer_steps = int(training["warmup_optimizer_steps"])
        self.checkpoint_every_optimizer_steps = int(training["checkpoint_every_optimizer_steps"])
        self.progress_every_optimizer_steps = int(training["progress_every_optimizer_steps"])
        self.target_samples_seen = self.max_optimizer_steps * self.effective_batch_size
        actual_population = sampling_population_sha256(train_dataset, self.data_manifest_sha256)
        if sampling_population_digest is not None and sampling_population_digest != actual_population:
            raise ValueError("supplied sampling population SHA256 differs from dataset")
        self.sampling_population_sha256 = actual_population
        initial_samples = min(self.checkpoint_every_optimizer_steps, self.max_optimizer_steps) * self.effective_batch_size
        self.sampler = HierarchicalCommodityContractSampler(
            train_dataset, initial_samples, int(training["seed"]),
            commodities=config["data"]["train_commodities"],
            sampling_population_sha256=self.sampling_population_sha256,
        )
        self.validation_loader = None if validation_dataset is None else DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_v11_batch, num_workers=0,
        )
        self.optimizer = torch.optim.AdamW(
            list(model.optimizer_parameters()), lr=training["learning_rate"],
            weight_decay=training["weight_decay"], betas=tuple(training["betas"]), eps=training["eps"],
        )
        total_steps = self.max_optimizer_steps
        warmup = self.warmup_optimizer_steps
        def schedule(step: int) -> float:
            return fixed_budget_lr_multiplier(step, total_steps, warmup)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.amp_enabled = bool(training["amp"] and device.type == "cuda")
        self.amp_dtype_name = str(training.get("amp_dtype", "float16"))
        self.amp_dtype = _amp_dtype(self.amp_dtype_name)
        if self.amp_enabled and self.amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("V1.1 amp_dtype=bfloat16 requested but CUDA device does not support BF16")
        self.grad_scaler_enabled = self.amp_enabled and self.amp_dtype is torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.grad_scaler_enabled)
        self.amp_calibrated = not self.grad_scaler_enabled
        self.amp_calibration: list[dict[str, float | bool]] = []
        self.global_step = 0; self.samples_seen = 0
        self.history: list[dict] = []
        self.interrupted = False
        self.skipped_optimizer_steps = 0
        self.last_grad_norm: float | None = None
        self.step_logger = step_logger
        self.wandb_run_id = wandb_run_id
        self.consecutive_amp_overflows = 0
        self.max_consecutive_amp_overflows = 8
        self.gradient_connectivity: dict | None = None
        self.daily_truncation_count = 0; self.daily_truncated_tokens = 0
        self.checkpoint_dir = Path(training["checkpoint_dir"])
        self.manifest = v11_implementation_manifest(); self.manifest_digest = manifest_sha256(self.manifest)

    def _calibrate_amp(self, kwargs: dict) -> None:
        """Find a finite initial scale without consuming an optimizer/EMA step."""
        if self.amp_calibrated:
            return
        if not self.grad_scaler_enabled:
            raise RuntimeError("V1.1 AMP calibration is only valid for float16 GradScaler mode")
        rng = capture_rng_state(include_cuda=True)
        scale = 65536.0
        try:
            for _ in range(8):
                restore_rng_state(rng)
                self.optimizer.zero_grad(set_to_none=True)
                probe = torch.amp.GradScaler("cuda", init_scale=scale)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = self.model(**kwargs)
                loss, _ = jepa_loss_fp32(
                    output, lambda_var=self.config["training"]["lambda_var"],
                    lambda_cov=self.config["training"]["lambda_cov"],
                    variance_floor=self.config["training"]["variance_floor"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("V1.1 AMP calibration loss is nonfinite")
                probe.scale(loss / self.accumulation).backward()
                probe.unscale_(self.optimizer)
                finite = all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in self.model.parameters() if parameter.requires_grad
                )
                self.amp_calibration.append({"loss_scale": scale, "gradients_finite": bool(finite)})
                if finite:
                    self.scaler = torch.amp.GradScaler("cuda", init_scale=scale)
                    self.amp_calibrated = True
                    return
                scale *= 0.5
            raise FloatingPointError(f"V1.1 AMP calibration exhausted eight attempts: {self.amp_calibration}")
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng_state(rng)

    def _finish_optimizer_step(self) -> bool:
        """Advance optimizer-dependent state only when the update is actually applied."""
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        step_succeeded = not self.grad_scaler_enabled or scale_after >= scale_before
        self.optimizer.zero_grad(set_to_none=True)
        if step_succeeded:
            self.consecutive_amp_overflows = 0
            self.scheduler.step()
            self.model.update_target(self.config["training"]["ema_tau"])
            self.global_step += 1
        else:
            self.skipped_optimizer_steps += 1
            self.consecutive_amp_overflows += 1
            if self.consecutive_amp_overflows >= self.max_consecutive_amp_overflows:
                raise FloatingPointError(
                    "V1.1 float16 AMP overflowed for "
                    f"{self.consecutive_amp_overflows} consecutive optimizer attempts"
                )
        return step_succeeded

    def _clip_gradients_or_skip(self) -> bool:
        """Clip finite gradients; let GradScaler recover true FP16 overflow steps.

        The common path uses PyTorch's fast norm reduction. Only if that norm is
        non-finite do we scan gradient elements and, when all elements are finite,
        recompute the norm in FP64 to distinguish reduction overflow from real NaN/Inf.
        """
        parameters = list(self.model.optimizer_parameters())
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not grads:
            raise RuntimeError("V1.1 optimizer step has no gradients")
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type=2.0, error_if_nonfinite=False,
        )
        if torch.isfinite(total_norm):
            self.last_grad_norm = float(total_norm.detach())
            torch.nn.utils.clip_grads_with_norm_(
                parameters, self.config["training"]["gradient_clip_norm"], total_norm,
            )
            return True

        if _gradient_elements_finite(parameters):
            stable_norm = _stable_total_norm_fp64(parameters)
            if not torch.isfinite(stable_norm):
                raise FloatingPointError("V1.1 FP64 gradient norm is non-finite")
            self.last_grad_norm = float(stable_norm.detach())
            torch.nn.utils.clip_grads_with_norm_(
                parameters, self.config["training"]["gradient_clip_norm"], stable_norm,
            )
            return True

        if self.grad_scaler_enabled:
            # unscale_ has already recorded found_inf for GradScaler. Do not clip
            # non-finite gradients; scaler.step() will skip the optimizer update and
            # scaler.update() will lower the scale.
            self.last_grad_norm = None
            return False
        raise FloatingPointError(
            f"V1.1 non-finite gradient elements under amp_dtype={self.amp_dtype_name}"
        )

    def _cycle_position(self) -> tuple[int, int, int]:
        cycle = self.global_step // self.checkpoint_every_optimizer_steps
        if self.global_step >= self.max_optimizer_steps:
            return cycle, 0, 0
        cycle_start = cycle * self.checkpoint_every_optimizer_steps
        cycle_steps = min(self.checkpoint_every_optimizer_steps, self.max_optimizer_steps - cycle_start)
        return cycle, cycle_steps * self.effective_batch_size, (self.global_step - cycle_start) * self.effective_batch_size

    def _cycle_loader(self) -> DataLoader:
        cycle, samples, offset = self._cycle_position()
        if samples <= 0:
            raise RuntimeError("cannot create a loader after the fixed optimizer budget")
        self.sampler.configure_cycle(cycle, samples, offset)
        generator = torch.Generator().manual_seed(int(self.config["training"]["seed"]) + 1_000_003 * cycle)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=self.sampler,
            collate_fn=collate_v11_batch, num_workers=int(self.config["training"]["num_workers"]),
            pin_memory=self.device.type == "cuda", generator=generator,
        )

    def _run_segment(
        self, loader: DataLoader, expected_optimizer_steps: int,
        stop_requested: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, float], bool]:
        self.model.train(True)
        totals = {f"prediction_loss_h{h}": 0.0 for h in self.model.horizons}
        total_loss, count, interrupted = 0.0, 0, False
        segment_started = time.perf_counter()
        optimizer_step = 0
        step_started = time.perf_counter()
        step_samples = 0
        step_loss = 0.0
        step_horizons = {horizon: 0.0 for horizon in self.model.horizons}
        self.optimizer.zero_grad(set_to_none=True)
        for index, raw in enumerate(loader):
            self.daily_truncation_count += sum(
                bool(metadata.get("daily_was_truncated", False)) for metadata in raw["metadata"]
            )
            self.daily_truncated_tokens += sum(
                int(metadata.get("daily_truncated_tokens", 0)) for metadata in raw["metadata"]
            )
            kwargs = model_inputs(raw, self.device)
            self._calibrate_amp(kwargs)
            with torch.enable_grad(), torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                output = self.model(**kwargs)
            loss, metrics = jepa_loss_fp32(
                output, lambda_var=self.config["training"]["lambda_var"],
                lambda_cov=self.config["training"]["lambda_cov"],
                variance_floor=self.config["training"]["variance_floor"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("V1.1 JEPA loss is nonfinite")
            size = kwargs["minute_market"].shape[0]
            loss_value = float(loss.detach())
            horizon_values = {
                horizon: float(metrics[f"prediction_loss_h{horizon}"])
                for horizon in self.model.horizons
            }
            total_loss += loss_value * size
            count += size
            for horizon in self.model.horizons:
                totals[f"prediction_loss_h{horizon}"] += horizon_values[horizon] * size
            step_samples += size
            step_loss += loss_value * size
            for horizon in self.model.horizons:
                step_horizons[horizon] += horizon_values[horizon] * size
            self.scaler.scale(loss / self.accumulation).backward()
            if (index + 1) % self.accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                gradients_finite = self._clip_gradients_or_skip()
                if gradients_finite and self.gradient_connectivity is None:
                    self.gradient_connectivity = assert_all_trainable_gradients(self.model)
                step_succeeded = self._finish_optimizer_step()
                if not gradients_finite and step_succeeded:
                    raise FloatingPointError("GradScaler applied non-finite gradients")
                if not step_succeeded:
                    raise FloatingPointError("fixed sample budget requires every sampled optimizer update to succeed")
                if step_samples != self.effective_batch_size:
                    raise RuntimeError("fixed sample budget encountered a partial effective batch")
                self.samples_seen += step_samples
                optimizer_step += 1
                if self.step_logger is not None:
                    self.step_logger.log_step(
                            global_step=self.global_step,
                            samples_seen=self.samples_seen,
                            loss=step_loss / step_samples,
                            h16_loss=step_horizons[16] / step_samples,
                            h64_loss=step_horizons[64] / step_samples,
                            h256_loss=step_horizons[256] / step_samples,
                            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                            grad_norm=self.last_grad_norm,
                            skipped_optimizer_steps=self.skipped_optimizer_steps,
                            step_seconds=time.perf_counter() - step_started,
                            samples=step_samples,
                            allocated_vram_bytes=(
                                int(torch.cuda.memory_allocated()) if self.device.type == "cuda" else None
                            ),
                            reserved_vram_bytes=(
                                int(torch.cuda.memory_reserved()) if self.device.type == "cuda" else None
                            ),
                    )
                if self.global_step % self.progress_every_optimizer_steps == 0 or self.global_step == self.max_optimizer_steps:
                    elapsed = time.perf_counter() - segment_started
                    remaining = self.max_optimizer_steps - self.global_step
                    eta = elapsed / optimizer_step * remaining if optimizer_step else 0
                    print(
                            f"global_step={self.global_step}/{self.max_optimizer_steps} "
                            f"{100.0 * self.global_step / self.max_optimizer_steps:.1f}% "
                            f"samples_seen={self.samples_seen}/{self.target_samples_seen} "
                            f"loss={step_loss / step_samples:.8f} "
                            f"lr={float(self.optimizer.param_groups[0]['lr']):.8g} "
                            f"elapsed={_format_duration(elapsed)} ETA={_format_duration(eta)}",
                            flush=True,
                    )
                step_started = time.perf_counter()
                step_samples = 0
                step_loss = 0.0
                step_horizons = {horizon: 0.0 for horizon in self.model.horizons}
                if stop_requested is not None and stop_requested():
                    interrupted = True
                    break
                if optimizer_step >= expected_optimizer_steps:
                    break
        if step_samples:
            raise RuntimeError("training segment ended inside gradient accumulation")
        if not interrupted and optimizer_step != expected_optimizer_steps:
            raise RuntimeError(f"training segment produced {optimizer_step} updates, expected {expected_optimizer_steps}")
        if not count:
            raise RuntimeError("empty fixed-budget training segment")
        return ({"loss": total_loss / count, **{key: value / count for key, value in totals.items()},
                 "optimizer_steps": optimizer_step, "samples": count}, interrupted)

    def _state(self) -> dict:
        current = v11_implementation_manifest()
        if manifest_sha256(current) != self.manifest_digest:
            raise RuntimeError("V1.1 implementation changed during training")
        cycle, cycle_samples, cycle_offset = self._cycle_position()
        sampler_state = {
            "seed": self.sampler.seed, "sampling_cycle": cycle,
            "num_samples": cycle_samples, "start_offset": cycle_offset,
            "sampling_population_sha256": self.sampling_population_sha256,
        }
        from .temporal import training_contracts
        split = getattr(self.train_dataset, "stage_a_temporal_split", None)
        return {
            "stage_a_temporal_split_sha256": split["sha256"] if split else None,
            "stage_a_temporal_split": split,
            "stage_a_training_contracts": [list(x) for x in training_contracts(self.train_dataset)] if split else None,
            "stage_a_scaler_selected_anchors": getattr(self.train_dataset, "scaler_selected_anchors", None),
            "design_version": "1.1", "v11_config": deepcopy(self.config),
            "model_size": self.model.model_size,
            "d_model": int(self.model.architecture_config["d_model"]),
            "num_heads": int(self.model.architecture_config["num_heads"]),
            "ffn_dim": int(self.model.architecture_config["ffn_dim"]),
            "minute_layers": int(self.model.architecture_config["minute_layers"]),
            "predictor_hidden_dim": int(self.model.architecture_config["predictor_hidden"]),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
            ),
            "architecture_config": deepcopy(self.model.architecture_config),
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
            "training_protocol_version": self.config["training"]["protocol_version"],
            "budget_mode": self.config["training"]["budget_mode"],
            "global_step": self.global_step,
            "max_optimizer_steps": self.max_optimizer_steps,
            "effective_batch_size": self.effective_batch_size,
            "samples_seen": self.samples_seen,
            "target_samples_seen": self.target_samples_seen,
            "warmup_optimizer_steps": self.warmup_optimizer_steps,
            "checkpoint_every_optimizer_steps": self.checkpoint_every_optimizer_steps,
            "sampling_cycle": cycle,
            "sampling_cycle_sample_offset": cycle_offset,
            "sampling_population_sha256": self.sampling_population_sha256,
            "rng_state": capture_rng_state(include_cuda=self.device.type == "cuda"),
            "sampler": sampler_state,
            "shared_imc_scaler": self.train_dataset.scaler.to_dict(),
            "data_manifest_sha256": self.data_manifest_sha256,
            "checkpoint_selection": self.checkpoint_selection,
            "v11_implementation_manifest": self.manifest,
            "v11_implementation_sha256": self.manifest_digest,
            "daily_truncation_count": self.daily_truncation_count,
            "daily_truncated_tokens": self.daily_truncated_tokens,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "consecutive_amp_overflows": self.consecutive_amp_overflows,
            "amp_dtype": self.amp_dtype_name,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "amp_calibration": self.amp_calibration,
            "wandb_run_id": self.wandb_run_id,
            "interval_history": self.history, "gradient_connectivity": self.gradient_connectivity,
            "metric_logger_state": (
                self.step_logger.state_dict() if self.step_logger is not None and hasattr(self.step_logger, "state_dict") else None
            ),
        }

    def fit(
        self, stop_after_optimizer_step: int | None = None,
        *, interval_callback: Callable[[dict], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[dict]:
        target = self.max_optimizer_steps if stop_after_optimizer_step is None else int(stop_after_optimizer_step)
        if not self.global_step <= target <= self.max_optimizer_steps:
            raise ValueError("stop_after_optimizer_step is outside remaining fixed budget")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        while self.global_step < target:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            step_start = self.global_step
            samples_start = self.samples_seen
            skipped_before = self.skipped_optimizer_steps
            cycle_boundary = min(
                ((self.global_step // self.checkpoint_every_optimizer_steps) + 1) * self.checkpoint_every_optimizer_steps,
                self.max_optimizer_steps, target,
            )
            expected_steps = cycle_boundary - self.global_step
            train, interrupted = self._run_segment(self._cycle_loader(), expected_steps, stop_requested)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started
            record = {
                "sampling_cycle": step_start // self.checkpoint_every_optimizer_steps,
                "step_start": step_start + 1, "step_end": self.global_step,
                "samples_start": samples_start, "samples_end": self.samples_seen,
                "checkpoint_reason": "interrupted" if interrupted else (
                    "final" if self.global_step == self.max_optimizer_steps else "periodic"
                ),
                "train": train, "validation": None,
                "train_loss": train["loss"],
                **{
                    f"prediction_loss_h{horizon}": train[f"prediction_loss_h{horizon}"]
                    for horizon in self.model.horizons
                },
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "global_step": self.global_step,
                "optimizer_steps": self.global_step - step_start,
                "skipped_amp_steps": self.skipped_optimizer_steps - skipped_before,
                "elapsed_seconds": elapsed_seconds,
                "samples_per_sec": (
                    (self.samples_seen - samples_start) / elapsed_seconds if elapsed_seconds > 0 else 0.0
                ),
                "peak_vram_allocated": (
                    int(torch.cuda.max_memory_allocated()) if self.device.type == "cuda" else None
                ),
                "peak_vram_reserved": (
                    int(torch.cuda.max_memory_reserved()) if self.device.type == "cuda" else None
                ),
            }
            self.history.append(record)
            # Commit the recovery checkpoint before publishing interval artifacts.
            # If the process dies between these operations, resume state remains the
            # authoritative prefix and local interval files are safely rewritten later.
            save_checkpoint(self._state(), self.checkpoint_dir / "last.pt")
            history_path = self.checkpoint_dir / "interval_history.json"
            history_temporary = history_path.with_suffix(".json.tmp")
            history_temporary.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            history_temporary.replace(history_path)
            if interval_callback is not None:
                interval_callback(deepcopy(record))
            if interrupted:
                self.interrupted = True
                break
        return self.history

    def resume(self, state: dict) -> None:
        validate_v11_checkpoint(state)
        split = getattr(self.train_dataset, "stage_a_temporal_split", None)
        if state.get("stage_a_temporal_split_sha256") != (split["sha256"] if split else None):
            raise ValueError("cannot resume with changed Stage-A temporal split")
        if state.get("training_protocol_version") != FIXED_BUDGET_PROTOCOL:
            raise ValueError(
                "Cannot resume legacy epoch-budget checkpoint into training_protocol_version="
                "fixed_sample_budget_v1. Legacy checkpoints remain evaluation-only."
            )
        legacy_size = "DEBUG" if state["v11_config"].get("profile") == "debug" else self.model.model_size
        state_size = state.get("model_size", legacy_size)
        if state_size != self.model.model_size:
            raise ValueError(f"cannot resume {self.model.model_size} from {state_size} checkpoint")
        actual_trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        if state.get("trainable_parameter_count", actual_trainable) != actual_trainable:
            raise ValueError("cannot resume V1.1 with changed trainable parameter count")
        if state["v11_config"] != self.config or state["data_manifest_sha256"] != self.data_manifest_sha256:
            raise ValueError("cannot resume V1.1 with changed config/data")
        if state["v11_implementation_sha256"] != self.manifest_digest:
            raise ValueError("cannot resume V1.1 with changed implementation")
        if state["shared_imc_scaler"] != self.train_dataset.scaler.to_dict():
            raise ValueError("cannot resume V1.1 with changed shared scaler")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"]); self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"]); restore_rng_state(state["rng_state"])
        for name, expected in (
            ("max_optimizer_steps", self.max_optimizer_steps),
            ("effective_batch_size", self.effective_batch_size),
            ("target_samples_seen", self.target_samples_seen),
            ("warmup_optimizer_steps", self.warmup_optimizer_steps),
            ("checkpoint_every_optimizer_steps", self.checkpoint_every_optimizer_steps),
            ("sampling_population_sha256", self.sampling_population_sha256),
        ):
            if state.get(name) != expected:
                raise ValueError(f"cannot resume V1.1 with changed {name}")
        self.global_step = int(state["global_step"])
        self.samples_seen = int(state["samples_seen"])
        if self.samples_seen != self.global_step * self.effective_batch_size:
            raise ValueError("checkpoint samples_seen/global_step mismatch")
        if self.global_step < self.max_optimizer_steps:
            cycle, samples, offset = self._cycle_position()
            expected_sampler = dict(state["sampler"])
            if (expected_sampler.get("sampling_cycle"), expected_sampler.get("num_samples"), expected_sampler.get("start_offset")) != (cycle, samples, offset):
                raise ValueError("checkpoint sampler position differs from committed samples")
            self.sampler.load_state_dict(expected_sampler)
        self.history = list(state["interval_history"])
        self.skipped_optimizer_steps = int(state.get("skipped_optimizer_steps", 0))
        self.consecutive_amp_overflows = int(state.get("consecutive_amp_overflows", 0))
        self.daily_truncation_count = int(state["daily_truncation_count"])
        self.daily_truncated_tokens = int(state["daily_truncated_tokens"])
        self.gradient_connectivity = state.get("gradient_connectivity")
        self.amp_calibration = list(state.get("amp_calibration", [])); self.amp_calibrated = True
        self.wandb_run_id = state.get("wandb_run_id")
        if self.step_logger is not None and hasattr(self.step_logger, "load_state_dict"):
            self.step_logger.load_state_dict(state.get("metric_logger_state"))
