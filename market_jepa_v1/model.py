from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from market_jepa.model.jepa import Predictor

from .config import validate_model_config
from .encoders import CrossScaleStateEncoder


class MarketJEPAV1(nn.Module):
    design_version = "1.0"

    def __init__(self, config: dict, horizons: tuple[int, ...] = (16, 64, 256), *, debug: bool = False) -> None:
        super().__init__()
        validate_model_config(config, debug=debug)
        if tuple(horizons) != (16, 64, 256):
            raise ValueError("V1 predictors require H16/H64/H256")
        self.architecture_config = deepcopy(config)
        self.horizons = tuple(horizons)
        self.online = CrossScaleStateEncoder(config)
        self.target_minute = deepcopy(self.online.minute_market).requires_grad_(False).eval()
        self.predictors = nn.ModuleDict({str(h): Predictor(config["belief_dim"], config["predictor_hidden"], config["dropout"]) for h in self.horizons})
        self.assert_no_identity_parameters()

    def assert_no_identity_parameters(self) -> None:
        forbidden = ("commodity", "symbol_id", "instrument_embedding")
        if any(any(word in key.lower() for word in forbidden) for key in self.state_dict()):
            raise RuntimeError("V1 state contains a forbidden identity parameter")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_minute.eval()
        return self

    def forward(self, batch: dict, *, return_intermediates: bool = False) -> dict:
        allowed = {"minute_market", "minute_context", "daily_market", "daily_context", "weekly_market", "weekly_context",
                   "minute_padding_mask", "daily_padding_mask", "weekly_padding_mask", "targets", "target_padding_masks",
                   "persistence", "persistence_padding_masks"}
        if set(batch) - allowed:
            raise ValueError(f"unexpected V1 model inputs: {sorted(set(batch) - allowed)}")
        online = self.online(batch, return_intermediates=return_intermediates)
        belief, intermediates = online if return_intermediates else (online, None)
        predictions = {h: self.predictors[str(h)](belief) for h in self.horizons}
        if set(batch["targets"]) != set(self.horizons):
            raise ValueError("target horizon keys do not match the V1 predictors")
        for horizon, market in batch["targets"].items():
            if market.ndim != 3 or market.shape[:2] != (belief.shape[0], horizon):
                raise ValueError(f"H{horizon} target must have shape [B,{horizon},market_dim]")
        self.target_minute.eval()
        with torch.no_grad():
            targets = {h: self.target_minute(batch["targets"][h], batch.get("target_padding_masks", {}).get(h)) for h in self.horizons}
        output = {"z_market": belief, "predictions": predictions, "targets": targets}
        if intermediates is not None:
            output["intermediates"] = intermediates
        return output

    @torch.no_grad()
    def encode_persistence(self, batch: dict) -> dict[int, torch.Tensor]:
        self.target_minute.eval()
        if set(batch["persistence"]) != set(self.horizons):
            raise ValueError("persistence horizon keys do not match the V1 predictors")
        for horizon, market in batch["persistence"].items():
            if market.ndim != 3 or market.shape[:2] != (batch["minute_market"].shape[0], horizon):
                raise ValueError(f"H{horizon} persistence must have shape [B,{horizon},market_dim]")
        return {h: self.target_minute(batch["persistence"][h], batch.get("persistence_padding_masks", {}).get(h)) for h in self.horizons}

    @torch.no_grad()
    def update_target(self, tau: float = 0.996) -> None:
        if not 0 <= tau <= 1:
            raise ValueError("EMA tau must be in [0,1]")
        online = dict(self.online.minute_market.named_parameters())
        target = dict(self.target_minute.named_parameters())
        if online.keys() != target.keys():
            raise RuntimeError("online/target minute parameter schemas differ")
        for name, value in target.items():
            value.mul_(tau).add_(online[name], alpha=1 - tau)
        source_buffers = dict(self.online.minute_market.named_buffers())
        target_buffers = dict(self.target_minute.named_buffers())
        if source_buffers.keys() != target_buffers.keys():
            raise RuntimeError("online/target minute buffer schemas differ")
        for name, value in target_buffers.items():
            value.copy_(source_buffers[name])

    def optimizer_parameters(self):
        yield from self.online.parameters()
        yield from self.predictors.parameters()
