from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import MarketEncoder, MinuteMarketEncoder


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MarketJEPA(nn.Module):
    def __init__(self, market_dim: int, context_dim: int, horizons: list[int], config: dict[str, Any]) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        self.online = MarketEncoder(market_dim, context_dim, config)
        self.target_minute: MinuteMarketEncoder = deepcopy(self.online.minute_market)
        self.target_minute.requires_grad_(False)
        self.target_minute.eval()
        self.predictors = nn.ModuleDict(
            {
                str(horizon): Predictor(config["latent_dim"], config["predictor_hidden"], config["dropout"])
                for horizon in horizons
            }
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_minute.eval()
        return self

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        z_market = self.online(batch)
        predictions = {h: self.predictors[str(h)](z_market) for h in self.horizons}
        self.target_minute.eval()
        with torch.no_grad():
            targets = {h: self.target_minute(batch["targets"][h]) for h in self.horizons}
        return {"z_market": z_market, "predictions": predictions, "targets": targets}

    @torch.no_grad()
    def encode_persistence(self, batch: dict[str, Any]) -> dict[int, torch.Tensor]:
        self.target_minute.eval()
        return {h: self.target_minute(batch["persistence"][h]) for h in self.horizons}

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        online = dict(self.online.minute_market.named_parameters())
        target = dict(self.target_minute.named_parameters())
        if online.keys() != target.keys():
            raise RuntimeError("online/target MinuteMarketEncoder schemas differ")
        for name, target_parameter in target.items():
            target_parameter.mul_(tau).add_(online[name], alpha=1.0 - tau)
        for target_buffer, online_buffer in zip(
            self.target_minute.buffers(), self.online.minute_market.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)

    def optimizer_parameters(self):
        yield from self.online.optimizer_parameters()
        yield from self.predictors.parameters()


def jepa_loss(
    output: dict[str, Any],
    lambda_var: float = 0.0,
    lambda_cov: float = 0.0,
    variance_floor: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    horizon_losses = {
        horizon: (1.0 - F.cosine_similarity(output["predictions"][horizon], target, dim=-1)).mean()
        for horizon, target in output["targets"].items()
    }
    prediction = torch.stack(list(horizon_losses.values())).mean()
    latent = output["z_market"]
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    variance = torch.relu(variance_floor - std).mean()
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance_matrix = centered.T @ centered / max(latent.shape[0] - 1, 1)
    diagonal = torch.diagonal(covariance_matrix)
    covariance = (covariance_matrix.square().sum() - diagonal.square().sum()) / latent.shape[1]
    total = prediction + lambda_var * variance + lambda_cov * covariance
    metrics = {f"prediction_loss_h{h}": value.detach() for h, value in horizon_losses.items()}
    metrics.update(
        prediction_loss=prediction.detach(),
        variance_loss=variance.detach(),
        covariance_loss=covariance.detach(),
        total_loss=total.detach(),
    )
    return total, metrics
