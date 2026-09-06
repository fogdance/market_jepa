from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, length: int) -> None:
        super().__init__()
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        rates = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim))
        table = torch.zeros(1, length, dim)
        table[0, :, 0::2] = torch.sin(positions * rates)
        table[0, :, 1::2] = torch.cos(positions * rates)
        self.register_buffer("encoding", table, persistent=True)


def validate_sequence(values: torch.Tensor, mask: torch.Tensor, validity: torch.Tensor | None = None) -> None:
    if values.ndim != 3 or not values.is_floating_point() or values.shape[0] == 0:
        raise ValueError("sequence must be floating [B,T,F]")
    if mask.shape != values.shape[:2] or mask.dtype != torch.bool or mask.device != values.device:
        raise ValueError("mask must be bool [B,T] on the input device; True means PAD")
    if validity is not None and (validity.shape != values.shape or validity.dtype != torch.bool or validity.device != values.device):
        raise ValueError("IMC validity must be bool and match market shape")
    effective = values.masked_fill(mask.unsqueeze(-1), 0)
    if not torch.isfinite(effective).all():
        raise ValueError("valid tokens contain NaN/Inf")


def ffn(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, dim), nn.Dropout(dropout),
    )


class SafeCrossAttention(nn.Module):
    """MHA whose fully masked source has an exact zero contribution."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if memory.ndim != 3 or memory.shape[0] != query.shape[0]:
            raise ValueError("attention memory must be [B,T,D]")
        if memory.shape[1] == 0:
            return torch.zeros_like(query), torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        if mask is None:
            mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        if mask.shape != memory.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("attention padding mask mismatch")
        available = ~mask.all(dim=1)
        safe_mask = mask.clone()
        safe_memory = memory
        if (~available).any():
            safe_mask[~available, 0] = False
            safe_memory = memory.clone()
            safe_memory[~available] = 0
        keys = self.memory_norm(safe_memory)
        output = self.attention(self.query_norm(query), keys, keys, key_padding_mask=safe_mask, need_weights=False)[0]
        output = output * available[:, None, None].to(output.dtype)
        return output, available


def available_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Normalize source logits over available sources; all-empty rows become zero."""
    if logits.ndim != 1 or available.ndim != 2 or available.shape[1] != logits.shape[0]:
        raise ValueError("source gate shape mismatch")
    expanded = logits[None].expand(available.shape[0], -1)
    safe = expanded.masked_fill(~available, -torch.inf)
    any_available = available.any(dim=1)
    safe = torch.where(any_available[:, None], safe, torch.zeros_like(safe))
    weights = torch.softmax(safe, dim=1)
    return weights * available.to(weights.dtype)


class MemoryTokenizer(nn.Module):
    def __init__(self, market_dim: int, context_dim: int, dim: int, capacity: int, *, boundary: bool = False) -> None:
        super().__init__()
        self.market_dim, self.context_dim = market_dim, context_dim
        self.market_projection = nn.Linear(market_dim, dim)
        self.validity_projection = nn.Linear(market_dim, dim, bias=False)
        self.context_projection = nn.Linear(context_dim, dim, bias=False)
        self.boundary_projection = nn.Linear(1, dim, bias=False) if boundary else None
        self.source_embedding = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.source_embedding, std=0.02)
        self.position = SinusoidalPosition(dim, capacity)

    def forward(
        self, market: torch.Tensor, context: torch.Tensor, mask: torch.Tensor,
        validity: torch.Tensor, boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_sequence(market, mask, validity)
        if market.shape[-1] != self.market_dim or context.shape != (*market.shape[:2], self.context_dim):
            raise ValueError("memory market/context dimensions differ from config")
        if not torch.isfinite(context.masked_fill(mask.unsqueeze(-1), 0)).all():
            raise ValueError("valid memory context contains NaN/Inf")
        values = market.masked_fill(~validity, 0).masked_fill(mask.unsqueeze(-1), 0)
        context = context.masked_fill(mask.unsqueeze(-1), 0)
        tokens = self.market_projection(values) + self.validity_projection(validity.to(values.dtype))
        tokens = tokens + self.context_projection(context) + self.source_embedding
        if self.boundary_projection is not None:
            if boundary is None or boundary.shape != mask.shape:
                raise ValueError("history boundary must be [B,T]")
            if not torch.isfinite(boundary.masked_fill(mask, 0)).all():
                raise ValueError("valid history boundary contains NaN/Inf")
            tokens = tokens + self.boundary_projection(boundary.unsqueeze(-1).to(tokens.dtype))
        elif boundary is not None:
            raise ValueError("boundary is only valid for historical weekly memory")
        positions = ((~mask).long().cumsum(dim=1) - 1).clamp_min(0)
        tokens = tokens + self.position.encoding[0, positions].to(tokens.dtype)
        return tokens.masked_fill(mask.unsqueeze(-1), 0)


class MinuteMarketCore(nn.Module):
    """EMA-compatible market-only minute core."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dim, capacity = config["d_model"], config["minute_capacity"]
        self.market_dim = config["minute_market_dim"]
        self.market_projection = nn.Linear(self.market_dim, dim)
        self.validity_projection = nn.Linear(self.market_dim, dim, bias=False)
        self.cls = nn.Parameter(torch.empty(1, 1, dim))
        self.source_embedding = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.source_embedding, std=0.02)
        # The same market core encodes 512 online bars and H16/H64/H256 targets.
        self.position = SinusoidalPosition(dim, max(capacity, 256) + 1)
        layer = nn.TransformerEncoderLayer(
            dim, config["num_heads"], config["ffn_dim"], config["dropout"],
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config["minute_layers"], enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dim)

    def encode_tokens(
        self, market: torch.Tensor, validity: torch.Tensor, mask: torch.Tensor,
        *, additive_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_sequence(market, mask, validity)
        if market.shape[-1] != self.market_dim or mask.all(dim=1).any():
            raise ValueError("minute market dimensions mismatch or sample is fully padded")
        values = market.masked_fill(~validity, 0).masked_fill(mask.unsqueeze(-1), 0)
        projected = self.market_projection(values) + self.validity_projection(validity.to(values.dtype))
        if additive_context is not None:
            if additive_context.shape != projected.shape:
                raise ValueError("minute context projection shape mismatch")
            projected = projected + additive_context
        tokens = torch.cat((self.cls.expand(market.shape[0], -1, -1), projected), dim=1)
        full_mask = torch.cat((torch.zeros_like(mask[:, :1]), mask), dim=1)
        positions = ((~full_mask).long().cumsum(dim=1) - 1).clamp_min(0)
        tokens = tokens + self.source_embedding + self.position.encoding[0, positions].to(tokens.dtype)
        tokens = tokens.masked_fill(full_mask.unsqueeze(-1), 0)
        encoded = self.output_norm(self.transformer(tokens, src_key_padding_mask=full_mask))
        return encoded.masked_fill(full_mask.unsqueeze(-1), 0), full_mask

    def forward(self, market: torch.Tensor, validity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encode_tokens(market, validity, mask)[0][:, 0]


class MinuteLocalEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.context_dim = config["minute_context_dim"]
        self.market_core = MinuteMarketCore(config)
        self.context_projection = nn.Linear(self.context_dim, config["d_model"], bias=False)

    def forward(self, market: torch.Tensor, context: torch.Tensor, mask: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        if context.shape != (*market.shape[:2], self.context_dim):
            raise ValueError("minute context dimensions differ from config")
        if not torch.isfinite(context.masked_fill(mask.unsqueeze(-1), 0)).all():
            raise ValueError("valid minute context contains NaN/Inf")
        projected = self.context_projection(context.masked_fill(mask.unsqueeze(-1), 0))
        return self.market_core.encode_tokens(market, validity, mask, additive_context=projected)[0][:, 1:]


class CommodityMemoryEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config["d_model"]
        self.tokenizer = MemoryTokenizer(
            config["history_weekly_market_dim"], config["history_weekly_context_dim"],
            dim, config["history_weekly_capacity"], boundary=True,
        )
        self.state_tokens = nn.Parameter(torch.empty(1, config["commodity_state_tokens"], dim))
        nn.init.normal_(self.state_tokens, std=0.02)
        self.read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.ffn_norm = nn.LayerNorm(dim); self.ffn = ffn(dim, config["ffn_dim"], config["dropout"])

    def forward(self, market, context, mask, validity, boundary):
        memory = self.tokenizer(market, context, mask, validity, boundary)
        state = self.state_tokens.expand(market.shape[0], -1, -1)
        update, available = self.read(state, memory, mask)
        state = state + update
        return state + self.ffn(self.ffn_norm(state)), memory, available


class ContractLifecycleEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config["d_model"]
        self.daily = MemoryTokenizer(config["daily_market_dim"], config["daily_context_dim"], dim, config["daily_capacity"])
        self.weekly = MemoryTokenizer(config["current_weekly_market_dim"], config["current_weekly_context_dim"], dim, config["current_weekly_capacity"])
        self.state_tokens = nn.Parameter(torch.empty(1, config["contract_state_tokens"], dim))
        nn.init.normal_(self.state_tokens, std=0.02)
        self.commodity_read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.daily_read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.weekly_read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.source_logits = nn.Parameter(torch.zeros(2))
        self.ffn_norm = nn.LayerNorm(dim); self.ffn = ffn(dim, config["ffn_dim"], config["dropout"])

    def forward(self, commodity, daily_market, daily_context, daily_mask, daily_validity,
                weekly_market, weekly_context, weekly_mask, weekly_validity):
        daily = self.daily(daily_market, daily_context, daily_mask, daily_validity)
        weekly = self.weekly(weekly_market, weekly_context, weekly_mask, weekly_validity)
        state = self.state_tokens.expand(commodity.shape[0], -1, -1)
        state = state + self.commodity_read(state, commodity)[0]
        daily_update, daily_available = self.daily_read(state, daily, daily_mask)
        weekly_update, weekly_available = self.weekly_read(state, weekly, weekly_mask)
        gates = available_softmax(self.source_logits, torch.stack((daily_available, weekly_available), dim=1))
        state = state + gates[:, 0, None, None] * daily_update + gates[:, 1, None, None] * weekly_update
        return state + self.ffn(self.ffn_norm(state)), daily, weekly, gates


class MinuteHigherScaleConditioner(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config["d_model"]
        self.read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.ffn_norm = nn.LayerNorm(dim); self.ffn = ffn(dim, config["ffn_dim"], config["dropout"])

    def forward(self, minute: torch.Tensor, minute_mask: torch.Tensor, commodity: torch.Tensor, contract: torch.Tensor) -> torch.Tensor:
        higher = torch.cat((commodity, contract), dim=1)
        result = minute + self.read(minute, higher)[0]
        result = result + self.ffn(self.ffn_norm(result))
        return result.masked_fill(minute_mask.unsqueeze(-1), 0)


class BeliefEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config["d_model"]
        self.tokens = nn.Parameter(torch.empty(1, config["belief_tokens"], dim))
        nn.init.normal_(self.tokens, std=0.02)
        self.minute_read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.contract_read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.commodity_read = SafeCrossAttention(dim, config["num_heads"], config["dropout"])
        self.source_logits = nn.Parameter(torch.zeros(3))
        self.ffn_norm = nn.LayerNorm(dim); self.ffn = ffn(dim, config["ffn_dim"], config["dropout"])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config["belief_dim"]))

    def forward(self, minute, minute_mask, contract, commodity):
        tokens = self.tokens.expand(minute.shape[0], -1, -1)
        minute_update, minute_available = self.minute_read(tokens, minute, minute_mask)
        contract_update, contract_available = self.contract_read(tokens, contract)
        commodity_update, commodity_available = self.commodity_read(tokens, commodity)
        available = torch.stack((minute_available, contract_available, commodity_available), dim=1)
        gates = available_softmax(self.source_logits, available)
        tokens = tokens + gates[:, 0, None, None] * minute_update
        tokens = tokens + gates[:, 1, None, None] * contract_update
        tokens = tokens + gates[:, 2, None, None] * commodity_update
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return self.output(tokens[:, 0]), tokens, gates
