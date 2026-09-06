from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from market_jepa.model.jepa import Predictor

from .config import validate_model_config
from .encoders import (
    BeliefEncoder, CommodityMemoryEncoder, ContractLifecycleEncoder,
    MinuteHigherScaleConditioner, MinuteLocalEncoder,
)


class MarketJEPAV11(nn.Module):
    design_version = "1.1"

    def __init__(self, config: dict, horizons: tuple[int, ...] = (16, 64, 256), *, debug: bool = False) -> None:
        super().__init__()
        validate_model_config(config, debug=debug)
        if tuple(horizons) != (16, 64, 256):
            raise ValueError("V1.1 predictors require H16/H64/H256")
        self.architecture_config = deepcopy(config)
        self.horizons = tuple(horizons)
        self.commodity_memory = CommodityMemoryEncoder(config)
        self.contract_lifecycle = ContractLifecycleEncoder(config)
        self.minute_local = MinuteLocalEncoder(config)
        self.minute_conditioner = MinuteHigherScaleConditioner(config)
        self.belief_encoder = BeliefEncoder(config)
        self.target_minute = deepcopy(self.minute_local.market_core).requires_grad_(False).eval()
        self.predictors = nn.ModuleDict({
            str(horizon): Predictor(config["belief_dim"], config["predictor_hidden"], config["dropout"])
            for horizon in self.horizons
        })
        self.assert_no_identity_parameters()

    def assert_no_identity_parameters(self) -> None:
        forbidden = ("commodity_embedding", "symbol_embedding", "symbol_id", "instrument_embedding")
        if any(any(word in name.lower() for word in forbidden) for name in self.state_dict()):
            raise RuntimeError("V1.1 state contains a forbidden identity parameter")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_minute.eval()
        return self

    def forward(
        self, *,
        minute_market: torch.Tensor, minute_context: torch.Tensor, minute_mask: torch.Tensor,
        minute_imc_validity: torch.Tensor,
        daily_market: torch.Tensor, daily_context: torch.Tensor, daily_mask: torch.Tensor,
        daily_imc_validity: torch.Tensor,
        current_weekly_market: torch.Tensor, current_weekly_context: torch.Tensor,
        current_weekly_mask: torch.Tensor, current_weekly_imc_validity: torch.Tensor,
        history_weekly_market: torch.Tensor, history_weekly_context: torch.Tensor,
        history_weekly_mask: torch.Tensor, history_weekly_imc_validity: torch.Tensor,
        history_weekly_contract_boundary: torch.Tensor,
        target_minute_market: dict[int, torch.Tensor] | None = None,
        target_minute_imc_validity: dict[int, torch.Tensor] | None = None,
        target_minute_mask: dict[int, torch.Tensor] | None = None,
        return_intermediates: bool = False,
    ) -> dict:
        commodity, history_tokens, _ = self.commodity_memory(
            history_weekly_market, history_weekly_context, history_weekly_mask,
            history_weekly_imc_validity, history_weekly_contract_boundary,
        )
        contract, daily_tokens, weekly_tokens, contract_gates = self.contract_lifecycle(
            commodity, daily_market, daily_context, daily_mask, daily_imc_validity,
            current_weekly_market, current_weekly_context, current_weekly_mask,
            current_weekly_imc_validity,
        )
        minute_local = self.minute_local(minute_market, minute_context, minute_mask, minute_imc_validity)
        minute_conditioned = self.minute_conditioner(minute_local, minute_mask, commodity, contract)
        belief, belief_tokens, belief_gates = self.belief_encoder(
            minute_conditioned, minute_mask, contract, commodity
        )
        output = {
            "z_market": belief,
            "predictions": {h: self.predictors[str(h)](belief) for h in self.horizons},
        }
        supplied = (target_minute_market, target_minute_imc_validity, target_minute_mask)
        if any(value is not None for value in supplied):
            if any(value is None for value in supplied):
                raise ValueError("target market, validity and mask must be supplied together")
            if set(target_minute_market) != set(self.horizons) or set(target_minute_imc_validity) != set(self.horizons) or set(target_minute_mask) != set(self.horizons):
                raise ValueError("target horizon keys differ from H16/H64/H256")
            self.target_minute.eval()
            with torch.no_grad():
                output["targets"] = {
                    h: self.target_minute(target_minute_market[h], target_minute_imc_validity[h], target_minute_mask[h])
                    for h in self.horizons
                }
        if return_intermediates:
            # Attention weights are deliberately not interpreted as causal importance;
            # scientific conclusions use controlled interventions.
            output["intermediates"] = {
                "commodity_state": commodity, "contract_state": contract,
                "history_weekly_tokens": history_tokens, "daily_tokens": daily_tokens,
                "current_weekly_tokens": weekly_tokens,
                "minute_local_tokens": minute_local,
                "minute_conditioned_tokens": minute_conditioned,
                "belief_tokens": belief_tokens, "final_belief": belief,
                "source_gate_values": {"contract": contract_gates, "belief": belief_gates},
            }
        return output

    @torch.no_grad()
    def update_target(self, tau: float = 0.996) -> None:
        if not 0 <= tau <= 1:
            raise ValueError("EMA tau must be in [0,1]")
        online = dict(self.minute_local.market_core.named_parameters())
        target = dict(self.target_minute.named_parameters())
        if online.keys() != target.keys():
            raise RuntimeError("online/target minute parameter schemas differ")
        for name, value in target.items():
            value.mul_(tau).add_(online[name], alpha=1 - tau)
        online_buffers = dict(self.minute_local.market_core.named_buffers())
        target_buffers = dict(self.target_minute.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError("online/target minute buffer schemas differ")
        for name, value in target_buffers.items():
            value.copy_(online_buffers[name])

    def optimizer_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("target_minute.") and parameter.requires_grad:
                yield parameter
