from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from market_jepa.data.dataset import MarketDataset
from market_jepa.implementation import changed_implementation_files, manifest_sha256
from market_jepa.model.jepa import jepa_loss
from market_jepa.train.diagnostics import LatentAccumulator, MetricAverage
from market_jepa.train.trainer import Trainer, _move

from .checkpoint import v1_implementation_manifest, validate_v1_checkpoint
from .config import FEATURE_DIMENSIONS, validate_v1_config
from .data import adapt_market_batch


class V1Trainer(Trainer):
    """Retain V0 optimizer/scheduler/AMP/resume mechanics with explicit V1 IO."""

    def __init__(self, model, config, train_dataset, source_sha256, preflight_metadata, device,
                 *, validation_dataset=None, runtime_options=None) -> None:
        validate_v1_config(config)
        if model.architecture_config != config["model"]:
            raise ValueError("V1 trainer/model architecture mismatch")
        self.v1_config = deepcopy(config)
        trainer_config = deepcopy(config)
        # The unchanged V0 diagnostic code uses this name for vector dimension.
        trainer_config["model"]["latent_dim"] = config["model"]["belief_dim"]
        evaluate = validation_dataset is not None
        if validation_dataset is None:
            validation_dataset = MarketDataset(train_dataset.data, train_dataset.config, "train", np.empty(0, dtype=np.int64))
        super().__init__(model, trainer_config, train_dataset, validation_dataset, source_sha256,
                         preflight_metadata, device, runtime_options,
                         checkpoint_selection="fixed_budget_final",
                         evaluate_validation_during_training=evaluate)
        self.v1_manifest = v1_implementation_manifest()
        self.v1_digest = manifest_sha256(self.v1_manifest)
        self.max_preclip_gradient_norm = 0.0

    def _epoch(self, loader, training: bool) -> dict[str, float]:
        # This loop adapts batches and adds finite checks; update ordering remains
        # in the unchanged Trainer._finish_optimizer_step (including AMP skips).
        self.model.train(training)
        averages = MetricAverage()
        dim = self.v1_config["model"]["belief_dim"]
        latent = LatentAccumulator(dim)
        targets = {h: LatentAccumulator(dim) for h in self.model.horizons}
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(loader):
            batch = _move(adapt_market_batch(raw_batch, self.v1_config["model"]), self.device)
            with (torch.enable_grad() if training else torch.inference_mode()), torch.amp.autocast(
                self.device.type, dtype=torch.float16, enabled=self.amp_enabled
            ):
                output = self.model(batch)
                loss, metrics = jepa_loss(output,
                    lambda_var=self.config["training"]["lambda_var"],
                    lambda_cov=self.config["training"]["lambda_cov"],
                    variance_floor=self.config["training"]["variance_floor"])
            if not torch.isfinite(loss):
                raise FloatingPointError("V1 JEPA loss is nonfinite")
            if training:
                self.scaler.scale(loss / self.accumulation).backward()
                if (index + 1) % self.accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(list(self.model.optimizer_parameters()),
                        self.config["training"]["gradient_clip_norm"], error_if_nonfinite=not self.amp_enabled)
                    if torch.isfinite(norm):
                        self.max_preclip_gradient_norm = max(self.max_preclip_gradient_norm, float(norm))
                    self._finish_optimizer_step()
                    if self.global_step and self.global_step % 10 == 0:
                        print(f"V1 development optimizer_step={self.global_step} loss={float(loss.detach()):.6f}", flush=True)
            size = batch["minute_market"].shape[0]
            averages.update(metrics, size)
            latent.update(output["z_market"])
            for h in self.model.horizons:
                targets[h].update(output["targets"][h])
        result = averages.result()
        threshold = self.config["training"]["collapse_threshold"]
        result.update({f"z_market_{k}": v for k, v in latent.metrics(threshold).items()})
        for h, accumulator in targets.items():
            result.update({f"target_h{h}_{k}": v for k, v in accumulator.metrics(threshold).items()})
        return result

    def _checkpoint_state(self, epoch, metrics, history) -> dict:
        current = v1_implementation_manifest()
        if manifest_sha256(current) != self.v1_digest:
            raise RuntimeError(f"V1 implementation changed during training: {changed_implementation_files(self.v1_manifest, current)}")
        state = super()._checkpoint_state(epoch, metrics, history)
        architecture = deepcopy(self.v1_config["model"])
        state.update(v1_config=deepcopy(self.v1_config), architecture_config=architecture,
                     feature_dimensions={key: architecture[key] for key in FEATURE_DIMENSIONS},
                     num_state_tokens=architecture["num_state_tokens"], cross_scale_rounds=architecture["cross_scale_rounds"],
                     v1_implementation_manifest=self.v1_manifest, v1_implementation_sha256=self.v1_digest,
                     max_preclip_gradient_norm=self.max_preclip_gradient_norm)
        return state

    def resume(self, checkpoint: dict) -> None:
        validate_v1_checkpoint(checkpoint)
        if checkpoint["v1_config"] != self.v1_config:
            raise ValueError("cannot resume V1: configuration changed")
        if checkpoint["v1_implementation_sha256"] != self.v1_digest:
            raise ValueError("cannot resume V1: implementation changed")
        super().resume(checkpoint)
        self.max_preclip_gradient_norm = float(checkpoint.get("max_preclip_gradient_norm", 0.0))
