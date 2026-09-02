from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length + 1, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length + 1, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence exceeds configured positional encoding length")
        return value + self.encoding[:, : value.shape[1]].to(dtype=value.dtype)


class MinuteMarketEncoder(nn.Module):
    """Market-only encoder shared semantically by online and EMA target."""

    def __init__(
        self,
        market_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.market_dim = market_dim
        self.input_projection = nn.Linear(market_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.position = SinusoidalPosition(d_model, max_length=max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        if market_features.ndim != 3 or market_features.shape[-1] != self.market_dim:
            raise ValueError(
                f"MinuteMarketEncoder expects [B,T,{self.market_dim}] market features only"
            )
        projected = self.input_projection(market_features)
        cls = self.cls.expand(projected.shape[0], -1, -1)
        encoded = self.position(torch.cat([cls, projected], dim=1))
        return self.output_norm(self.transformer(encoded)[:, 0])


class TemporalContextEncoder(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.gru = nn.GRU(context_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, context_observations: torch.Tensor) -> torch.Tensor:
        if context_observations.ndim != 3 or context_observations.shape[-1] != self.context_dim:
            raise ValueError(f"TemporalContextEncoder expects [B,T,{self.context_dim}]")
        _, hidden = self.gru(context_observations)
        return hidden[-1]


class PeriodEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(f"PeriodEncoder expects [B,T,{self.input_dim}]")
        if torch.any(lengths <= 0):
            raise ValueError("Daily/Weekly sequence must contain its current partial token")
        packed = pack_padded_sequence(
            values, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class MarketEncoder(nn.Module):
    def __init__(self, market_dim: int, context_dim: int, config: dict) -> None:
        super().__init__()
        self.ablation = config["ablation"]
        d_model = config["minute_d_model"]
        self.minute_market = MinuteMarketEncoder(
            market_dim=market_dim,
            d_model=d_model,
            layers=config["minute_layers"],
            heads=config["minute_heads"],
            ffn_dim=config["minute_ffn_dim"],
            dropout=config["dropout"],
            max_length=512,
        )
        self.minute_context = TemporalContextEncoder(context_dim, config["time_hidden"])
        period_input = market_dim + 1
        self.daily = PeriodEncoder(
            period_input, config["daily_hidden"], config["recurrent_layers"], config["dropout"]
        )
        self.weekly = PeriodEncoder(
            period_input, config["weekly_hidden"], config["recurrent_layers"], config["dropout"]
        )
        fusion_dim = d_model + config["time_hidden"] + config["daily_hidden"] + config["weekly_hidden"]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config["fusion_hidden"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_hidden"], config["latent_dim"]),
            nn.LayerNorm(config["latent_dim"]),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        minute_market = self.minute_market(batch["minute_market"])
        minute_context = self.minute_context(batch["minute_context"])
        batch_size = minute_market.shape[0]
        daily = torch.zeros(
            batch_size, self.daily.gru.hidden_size, device=minute_market.device, dtype=minute_market.dtype
        )
        weekly = torch.zeros(
            batch_size, self.weekly.gru.hidden_size, device=minute_market.device, dtype=minute_market.dtype
        )
        if self.ablation in {"minute_daily", "minute_daily_weekly"}:
            daily = self.daily(batch["daily"], batch["daily_lengths"])
        if self.ablation == "minute_daily_weekly":
            weekly = self.weekly(batch["weekly"], batch["weekly_lengths"])
        return self.fusion(torch.cat([minute_market, minute_context, daily, weekly], dim=-1))

    def optimizer_parameters(self):
        modules: list[nn.Module] = [self.minute_market, self.minute_context, self.fusion]
        if self.ablation in {"minute_daily", "minute_daily_weekly"}:
            modules.append(self.daily)
        if self.ablation == "minute_daily_weekly":
            modules.append(self.weekly)
        for module in modules:
            yield from module.parameters()
