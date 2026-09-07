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
from .config import validate_v11_config
from .dataset import V11ContractDataset, collate_v11_batch
from .sampler import HierarchicalCommodityContractSampler
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
PROGRESS_INTERVAL_STEPS = 200


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
        samples_per_epoch: int | None = None, data_manifest_sha256: str = "",
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
        self.accumulation = int(training["gradient_accumulation"])
        batch_size = int(training["batch_size"])
        samples = int(samples_per_epoch or len(train_dataset))
        self.sampler = HierarchicalCommodityContractSampler(
            train_dataset, samples, int(training["seed"]),
            commodities=config["data"]["train_commodities"],
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=self.sampler,
            collate_fn=collate_v11_batch, num_workers=int(training["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        self.validation_loader = None if validation_dataset is None else DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_v11_batch, num_workers=0,
        )
        self.optimizer = torch.optim.AdamW(
            list(model.optimizer_parameters()), lr=training["learning_rate"],
            weight_decay=training["weight_decay"], betas=tuple(training["betas"]), eps=training["eps"],
        )
        steps_per_epoch = math.ceil(math.ceil(samples / batch_size) / self.accumulation)
        total_steps = max(1, steps_per_epoch * int(training["max_epochs"]))
        warmup = int(total_steps * float(training["warmup_ratio"]))
        def schedule(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
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
        self.global_step = 0; self.start_epoch = 0; self.history: list[dict] = []
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

    def _run_epoch(self, loader, training: bool, *, epoch: int | None = None) -> dict[str, float]:
        self.model.train(training)
        totals = {f"prediction_loss_h{h}": 0.0 for h in self.model.horizons}
        total_loss, count = 0.0, 0
        epoch_started = time.perf_counter()
        optimizer_step = 0
        optimizer_steps_total = math.ceil(len(loader) / self.accumulation) if training else 0
        step_started = time.perf_counter()
        step_samples = 0
        step_loss = 0.0
        step_horizons = {horizon: 0.0 for horizon in self.model.horizons}
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for index, raw in enumerate(loader):
            if training:
                self.daily_truncation_count += sum(
                    bool(metadata.get("daily_was_truncated", False))
                    for metadata in raw["metadata"]
                )
                self.daily_truncated_tokens += sum(
                    int(metadata.get("daily_truncated_tokens", 0))
                    for metadata in raw["metadata"]
                )
            kwargs = model_inputs(raw, self.device)
            if training:
                self._calibrate_amp(kwargs)
            with (torch.enable_grad() if training else torch.inference_mode()):
                with torch.amp.autocast(
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
            if training:
                step_samples += size
                step_loss += loss_value * size
                for horizon in self.model.horizons:
                    step_horizons[horizon] += horizon_values[horizon] * size
            if training:
                self.scaler.scale(loss / self.accumulation).backward()
                if (index + 1) % self.accumulation == 0 or index + 1 == len(loader):
                    optimizer_step += 1
                    self.scaler.unscale_(self.optimizer)
                    gradients_finite = self._clip_gradients_or_skip()
                    if gradients_finite and self.gradient_connectivity is None:
                        self.gradient_connectivity = assert_all_trainable_gradients(self.model)
                    step_succeeded = self._finish_optimizer_step()
                    if not gradients_finite and step_succeeded:
                        raise FloatingPointError(
                            "V1.1 GradScaler failed to skip an optimizer step with non-finite gradients"
                        )
                    if (
                        step_succeeded and self.step_logger is not None
                        and getattr(self.step_logger, "active", True)
                    ):
                        self.step_logger.log_step(
                            global_step=self.global_step,
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
                    if (
                        optimizer_step % PROGRESS_INTERVAL_STEPS == 0
                        or optimizer_step == optimizer_steps_total
                    ):
                        elapsed = time.perf_counter() - epoch_started
                        eta = elapsed / optimizer_step * (optimizer_steps_total - optimizer_step)
                        print(
                            f"epoch={epoch} step={optimizer_step}/{optimizer_steps_total} "
                            f"{100.0 * optimizer_step / optimizer_steps_total:.1f}% "
                            f"loss={step_loss / step_samples:.8f} "
                            f"lr={float(self.optimizer.param_groups[0]['lr']):.8g} "
                            f"elapsed={_format_duration(elapsed)} ETA={_format_duration(eta)}",
                            flush=True,
                        )
                    step_started = time.perf_counter()
                    step_samples = 0
                    step_loss = 0.0
                    step_horizons = {horizon: 0.0 for horizon in self.model.horizons}
            total_loss += loss_value * size; count += size
            for horizon in self.model.horizons:
                totals[f"prediction_loss_h{horizon}"] += horizon_values[horizon] * size
        return {"loss": total_loss / count, **{key: value / count for key, value in totals.items()}}

    def _state(self, epoch: int) -> dict:
        current = v11_implementation_manifest()
        if manifest_sha256(current) != self.manifest_digest:
            raise RuntimeError("V1.1 implementation changed during training")
        return {
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
            "epoch": epoch, "global_step": self.global_step,
            "rng_state": capture_rng_state(include_cuda=self.device.type == "cuda"),
            "sampler": self.sampler.state_dict(),
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
            "history": self.history, "gradient_connectivity": self.gradient_connectivity,
        }

    def fit(
        self, stop_before_epoch: int | None = None,
        *, epoch_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        final_epoch = int(self.config["training"]["max_epochs"])
        if stop_before_epoch is not None:
            final_epoch = int(stop_before_epoch)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.start_epoch, final_epoch):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            global_step_before = self.global_step
            skipped_before = self.skipped_optimizer_steps
            self.sampler.set_epoch(epoch)
            train = self._run_epoch(self.train_loader, True, epoch=epoch)
            validation = None if self.validation_loader is None else self._run_epoch(self.validation_loader, False)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            record = {
                "epoch": epoch, "train": train, "validation": validation,
                "train_loss": train["loss"],
                **{
                    f"prediction_loss_h{horizon}": train[f"prediction_loss_h{horizon}"]
                    for horizon in self.model.horizons
                },
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "global_step": self.global_step,
                "optimizer_steps_this_epoch": self.global_step - global_step_before,
                "skipped_amp_steps": self.skipped_optimizer_steps - skipped_before,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_vram_allocated": (
                    int(torch.cuda.max_memory_allocated()) if self.device.type == "cuda" else None
                ),
                "peak_vram_reserved": (
                    int(torch.cuda.max_memory_reserved()) if self.device.type == "cuda" else None
                ),
            }
            self.history.append(record)
            save_checkpoint(self._state(epoch), self.checkpoint_dir / "last.pt")
            history_path = self.checkpoint_dir / "history.json"
            history_temporary = history_path.with_suffix(".json.tmp")
            history_temporary.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            history_temporary.replace(history_path)
            if epoch_callback is not None:
                epoch_callback(deepcopy(record))
        return self.history

    def resume(self, state: dict) -> None:
        validate_v11_checkpoint(state)
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
        self.sampler.load_state_dict(state["sampler"])
        self.global_step = int(state["global_step"]); self.start_epoch = int(state["epoch"]) + 1
        self.history = list(state["history"]); self.skipped_optimizer_steps = int(state.get("skipped_optimizer_steps", 0))
        self.consecutive_amp_overflows = int(state.get("consecutive_amp_overflows", 0))
        self.daily_truncation_count = int(state["daily_truncation_count"])
        self.daily_truncated_tokens = int(state["daily_truncated_tokens"])
        self.gradient_connectivity = state.get("gradient_connectivity")
        self.amp_calibration = list(state.get("amp_calibration", [])); self.amp_calibrated = True
        self.wandb_run_id = state.get("wandb_run_id")
