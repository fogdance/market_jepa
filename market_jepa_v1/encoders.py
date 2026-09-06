from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from market_jepa.model.encoders import SinusoidalPosition


def padding_mask(values: torch.Tensor, mask: torch.Tensor | None, *, allow_empty: bool) -> torch.Tensor:
    if values.ndim != 3 or not values.is_floating_point() or values.shape[0] == 0:
        raise ValueError("sequence must be a nonempty floating batch [B,T,F]")
    if mask is None:
        mask = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    if mask.shape != values.shape[:2] or mask.dtype != torch.bool or mask.device != values.device:
        raise ValueError("padding mask must be bool [B,T] on the input device; True means padding")
    if not allow_empty and (mask.shape[1] == 0 or torch.any(mask.all(dim=1))):
        raise ValueError("minute market history must contain at least one valid token per sample")
    if not torch.isfinite(values.masked_fill(mask.unsqueeze(-1), 0)).all():
        raise ValueError("valid input tokens contain NaN or Inf")
    return mask


class MinuteSequenceCore(nn.Module):
    """EMA-compatible market path. Context projection lives outside this module."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config["d_model"]
        self.market_dim = config["minute_market_dim"]
        self.input_projection = nn.Linear(self.market_dim, dim)
        self.cls = nn.Parameter(torch.empty(1, 1, dim))
        self.scale_embedding = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.scale_embedding, std=0.02)
        self.position = SinusoidalPosition(dim, config["minute_max_length"])
        layer = nn.TransformerEncoderLayer(
            dim, config["num_heads"], config["ffn_dim"], config["dropout"],
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config["minute_layers"], enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dim)

    def encode_tokens(
        self, market: torch.Tensor, mask: torch.Tensor | None = None,
        *, context_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = padding_mask(market, mask, allow_empty=False)
        if market.shape[-1] != self.market_dim:
            raise ValueError(f"minute market expects {self.market_dim} features")
        projected = self.input_projection(market.masked_fill(mask.unsqueeze(-1), 0))
        if context_tokens is not None:
            if context_tokens.shape != projected.shape:
                raise ValueError("projected context must match minute token shape")
            projected = projected + context_tokens
        tokens = torch.cat((self.cls.expand(market.shape[0], -1, -1), projected), dim=1)
        full_mask = torch.cat((torch.zeros_like(mask[:, :1]), mask), dim=1)
        # Valid positions are invariant to left/interior padding.
        positions = (~full_mask).long().cumsum(dim=1) - 1
        if positions.max() >= self.position.encoding.shape[1]:
            raise ValueError("minute sequence exceeds positional capacity")
        tokens = tokens + self.scale_embedding + self.position.encoding[0, positions].to(tokens.dtype)
        tokens = tokens.masked_fill(full_mask.unsqueeze(-1), 0)
        encoded = self.output_norm(self.transformer(tokens, src_key_padding_mask=full_mask))
        return encoded.masked_fill(full_mask.unsqueeze(-1), 0), full_mask

    def forward(self, market: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # The public target path accepts market and padding only.
        return self.encode_tokens(market, mask)[0][:, 0]


class PeriodSequenceEncoder(nn.Module):
    def __init__(self, market_dim: int, context_dim: int, hidden: int, layers: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.market_dim, self.context_dim = market_dim, context_dim
        self.gru = nn.GRU(market_dim + context_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.projection = nn.Linear(hidden, dim)
        self.scale_embedding = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.scale_embedding, std=0.02)

    def forward(self, market: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if market.shape[:2] != context.shape[:2] or market.shape[-1] != self.market_dim or context.shape[-1] != self.context_dim:
            raise ValueError("period market/context dimensions do not match config")
        values = torch.cat((market, context), dim=-1)
        mask = padding_mask(values, mask, allow_empty=True)
        if values.shape[1] == 0:
            return values.new_zeros((values.shape[0], 0, self.projection.out_features)), mask
        values = values.masked_fill(mask.unsqueeze(-1), 0)
        lengths = (~mask).sum(dim=1)
        order = mask.to(torch.int64).argsort(dim=1, stable=True)
        packed_values = values.gather(1, order.unsqueeze(-1).expand_as(values))
        packed = pack_padded_sequence(packed_values, lengths.clamp_min(1).detach().cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        sequence, _ = pad_packed_sequence(output, batch_first=True, total_length=values.shape[1])
        sequence = torch.zeros_like(sequence).scatter(1, order.unsqueeze(-1).expand_as(sequence), sequence)
        tokens = self.projection(sequence) + self.scale_embedding
        return tokens.masked_fill(mask.unsqueeze(-1), 0), mask


class CrossScaleInteractionBlock(nn.Module):
    """One complete State←T, T←State round; feedback is shared across scales."""

    def __init__(self, config: dict, *, feedback: bool = True) -> None:
        super().__init__()
        self.has_feedback = feedback
        dim, heads, dropout = config["d_model"], config["num_heads"], config["dropout"]
        self.state_query_norm = nn.LayerNorm(dim)
        self.token_key_norm = nn.LayerNorm(dim)
        self.state_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.state_ffn_norm = nn.LayerNorm(dim)
        self.state_ffn = self._ffn(dim, config["ffn_dim"], dropout)
        if feedback:
            self.token_query_norm = nn.LayerNorm(dim)
            self.state_key_norm = nn.LayerNorm(dim)
            self.feedback_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.feedback_ffn_norm = nn.LayerNorm(dim)
            self.feedback_ffn = self._ffn(dim, config["ffn_dim"], dropout)

    @staticmethod
    def _ffn(dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, state: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.token_key_norm(tokens)
        state = state + self.state_attention(self.state_query_norm(state), keys, keys,
                                             key_padding_mask=mask, need_weights=False)[0]
        state = state + self.state_ffn(self.state_ffn_norm(state))
        if self.has_feedback:
            keys = self.state_key_norm(state)
            tokens = tokens + self.feedback_attention(self.token_query_norm(tokens), keys, keys, need_weights=False)[0]
            tokens = tokens + self.feedback_ffn(self.feedback_ffn_norm(tokens))
        return state, tokens.masked_fill(mask.unsqueeze(-1), 0)


class CrossScaleStateEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.context_dim = config["minute_context_dim"]
        self.minute_market = MinuteSequenceCore(config)
        self.minute_context_projection = (
            nn.Linear(self.context_dim, config["d_model"], bias=False) if self.context_dim else None
        )
        for scale in ("daily", "weekly"):
            setattr(self, scale, PeriodSequenceEncoder(
                config[f"{scale}_market_dim"], config[f"{scale}_context_dim"],
                config[f"{scale}_gru_hidden"], config[f"{scale}_gru_layers"], config["d_model"], config["dropout"],
            ))
        self.state_tokens = nn.Parameter(torch.empty(1, config["num_state_tokens"], config["d_model"]))
        nn.init.normal_(self.state_tokens, std=0.02)
        rounds = config["cross_scale_rounds"]
        self.blocks = nn.ModuleList(
            CrossScaleInteractionBlock(config, feedback=index + 1 < rounds)
            for index in range(rounds)
        )
        self.belief_head = nn.Sequential(nn.LayerNorm(config["d_model"]), nn.Linear(config["d_model"], config["belief_dim"]))

    def forward(self, batch: dict, *, return_intermediates: bool = False):
        context = batch["minute_context"]
        if context.shape[:2] != batch["minute_market"].shape[:2] or context.shape[-1] != self.context_dim:
            raise ValueError("minute context dimensions do not match config")
        minute_mask = padding_mask(context, batch.get("minute_padding_mask"), allow_empty=False)
        context_tokens = (self.minute_context_projection(context.masked_fill(minute_mask.unsqueeze(-1), 0))
                          if self.minute_context_projection is not None else None)
        minute, minute_mask = self.minute_market.encode_tokens(batch["minute_market"], minute_mask, context_tokens=context_tokens)
        daily, daily_mask = self.daily(batch["daily_market"], batch["daily_context"], batch.get("daily_padding_mask"))
        weekly, weekly_mask = self.weekly(batch["weekly_market"], batch["weekly_context"], batch.get("weekly_padding_mask"))
        tokens = torch.cat((minute, daily, weekly), dim=1)
        mask = torch.cat((minute_mask, daily_mask, weekly_mask), dim=1)
        state = self.state_tokens.expand(tokens.shape[0], -1, -1)
        intermediates = None
        if return_intermediates:
            intermediates = {"minute_local_tokens": minute, "daily_local_tokens": daily, "weekly_local_tokens": weekly,
                             "token_lengths": (minute.shape[1], daily.shape[1], weekly.shape[1]), "token_padding_mask": mask}
        for index, block in enumerate(self.blocks, 1):
            state, tokens = block(state, tokens, mask)
            if intermediates is not None:
                intermediates[f"state_tokens_after_round_{index}"] = state
                intermediates[f"tokens_after_feedback_round_{index}"] = tokens
        # Zero rounds is a deliberate debug bypass, not a scientific baseline.
        belief = self.belief_head(state[:, 0] if self.blocks else minute[:, 0])
        if intermediates is not None:
            intermediates["final_belief"] = belief
            return belief, intermediates
        return belief
