from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from market_jepa.model.jepa import Predictor
from market_jepa_v1_1.encoders import CommodityMemoryEncoder, MinuteLocalEncoder
from market_jepa_v1_1.model import MarketJEPAV11
from market_jepa_v1_1.training import V11Trainer

from .structure import remove_sources
from .protocol import control_semantic_hashes, digest


class LateFusionControl(MarketJEPAV11):
    """Independent M/D/Wc/Wh compression, then one MLP late fusion.

    Three independent memory compressors use four tokens each; every scale is
    reduced to one vector before any cross-scale interaction. The minute/EMA
    architecture and horizon predictors exactly match Full S.
    """

    def __init__(self, config, *, debug=False):
        super().__init__(config, debug=debug)
        if not debug and self.model_size != "S":
            raise ValueError("first matched LateFusion control is S only")
        del self.contract_lifecycle, self.minute_conditioner, self.belief_encoder
        self.daily_memory = CommodityMemoryEncoder(self._memory_config(config, "daily"))
        self.weekly_memory = CommodityMemoryEncoder(self._memory_config(config, "current_weekly"))
        self.daily_memory.tokenizer.boundary_projection = None
        self.weekly_memory.tokenizer.boundary_projection = None
        dim = config["d_model"]
        self.fusion = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 6 * dim),
                                    nn.GELU(), nn.Dropout(config["dropout"]), nn.Linear(6 * dim, dim))
        self.evaluation_variant = "LateFusion"

    @staticmethod
    def _memory_config(config, source):
        result = deepcopy(config)
        for suffix in ("market_dim", "context_dim", "capacity"):
            result["history_weekly_" + suffix] = config[source + "_" + suffix]
        return result

    def forward(self, *, return_intermediates=False, **batch):
        compressed = []
        for source, encoder in (("daily", self.daily_memory), ("current_weekly", self.weekly_memory),
                                ("history_weekly", self.commodity_memory)):
            mask = batch[source + "_mask"]
            boundary = batch["history_weekly_contract_boundary"] if source == "history_weekly" else None
            state, _, _ = encoder(batch[source + "_market"], batch[source + "_context"], mask,
                                   batch[source + "_imc_validity"], boundary)
            compressed.append(state.mean(1))
        context = batch["minute_context"].masked_fill(batch["minute_mask"].unsqueeze(-1), 0)
        if not torch.isfinite(context).all():
            raise ValueError("valid minute context contains NaN/Inf")
        encoded, _ = self.minute_local.market_core.encode_tokens(
            batch["minute_market"], batch["minute_imc_validity"], batch["minute_mask"],
            additive_context=self.minute_local.context_projection(context))
        minute, minute_vector = encoded[:, 1:], encoded[:, 0]
        belief = self.fusion(torch.cat([minute_vector, *compressed], -1))
        result = {"z_market": belief, "predictions": {h: self.predictors[str(h)](belief) for h in self.horizons}}
        if "target_minute_market" in batch and batch["target_minute_market"] is not None:
            with torch.no_grad():
                result["targets"] = {h: self.target_minute(batch["target_minute_market"][h],
                    batch["target_minute_imc_validity"][h], batch["target_minute_mask"][h]) for h in self.horizons}
        if return_intermediates:
            result["intermediates"] = {"minute_local_tokens": minute, "scale_vectors": compressed,
                                       "minute_vector": minute_vector, "final_belief": belief}
        return result


class SourceControl(MarketJEPAV11):
    def __init__(self, config, variant, *, debug=False):
        super().__init__(config, debug=debug)
        if variant not in {"MinuteOnly", "NoHistoricalWeekly"}:
            raise ValueError("unsupported retrain control")
        self.evaluation_variant = variant
        # Permanently unavailable read/tokenizer paths are not intended trainables.
        inactive = [self.commodity_memory.tokenizer, self.commodity_memory.read]
        if variant == "MinuteOnly":
            inactive += [self.contract_lifecycle.daily, self.contract_lifecycle.weekly,
                         self.contract_lifecycle.daily_read, self.contract_lifecycle.weekly_read]
            self.contract_lifecycle.source_logits.requires_grad_(False)
        for module in inactive:
            module.requires_grad_(False)

    def forward(self, **batch):
        return super().forward(**remove_sources(batch, self.evaluation_variant))


def control_model(config, variant, *, debug=False):
    if variant == "LateFusion":
        return LateFusionControl(config, debug=debug)
    return SourceControl(config, variant, debug=debug)


class ControlTrainer(V11Trainer):
    def __init__(self, *args, control_protocol, **kwargs):
        self.control_protocol = deepcopy(control_protocol)
        self.control_provenance = {
            "reference_checkpoint_sha256": control_protocol["reference_checkpoint_sha256"],
            "reference_sampler_num_samples": control_protocol["reference_sampler_num_samples"],
            "control_protocol_sha256": digest(control_protocol),
        }
        self.control_semantics = control_semantic_hashes()
        super().__init__(*args, **kwargs)

    def _state(self, epoch):
        if self.control_semantics != control_semantic_hashes():
            raise ValueError("control implementation changed during training")
        state = super()._state(epoch)
        state["evaluation_variant"] = self.model.evaluation_variant
        state["control_protocol"] = self.control_protocol
        state["control_provenance"] = self.control_provenance
        state["control_semantic_hashes"] = self.control_semantics
        return state

    def resume(self, state):
        if state.get("control_provenance") != self.control_provenance or state.get("control_semantic_hashes") != self.control_semantics:
            raise ValueError("control provenance/semantic mismatch")
        if state.get("evaluation_variant") != self.model.evaluation_variant:
            raise ValueError("control checkpoint variant mismatch")
        super().resume(state)
