from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from market_jepa.model.jepa import jepa_loss
from market_jepa_v1_1.capacity import load_capacity_configs, parameter_report
from market_jepa_v1_1.config import (
    DAILY_CONTEXT_FEATURES, IMC_FEATURES, MINUTE_CONTEXT_FEATURES,
    WEEKLY_CONTEXT_FEATURES, validate_v11_config,
)
from market_jepa_v1_1.encoders import (
    BeliefEncoder, CommodityMemoryEncoder, ContractLifecycleEncoder,
    MinuteHigherScaleConditioner, MinuteLocalEncoder,
)
from market_jepa_v1_1.development import parameter_counts as development_parameter_counts
from market_jepa_v1_1.formal_training import validate_formal_training_config
from market_jepa_v1_1.model import MarketJEPAV11
from market_jepa_v1_1.training import assert_all_trainable_gradients


EXPECTED_COUNTS = {
    "S": (8_262_661, 3_164_928, 11_427_589),
    "M": (22_086_917, 10_655_616, 32_742_533),
    "L": (45_518_853, 25_230_848, 70_749_701),
    "XL": (102_291_461, 56_720_640, 159_012_101),
}


def _inputs(batch: int = 1, *, targets: bool = False) -> dict:
    generator = torch.Generator().manual_seed(818)
    result = {}
    sources = {
        "minute": (4, len(MINUTE_CONTEXT_FEATURES)),
        "daily": (3, len(DAILY_CONTEXT_FEATURES)),
        "current_weekly": (2, len(WEEKLY_CONTEXT_FEATURES)),
        "history_weekly": (3, len(WEEKLY_CONTEXT_FEATURES)),
    }
    for source, (length, context_dim) in sources.items():
        result[f"{source}_market"] = torch.randn(batch, length, len(IMC_FEATURES), generator=generator)
        result[f"{source}_context"] = torch.randn(batch, length, context_dim, generator=generator)
        result[f"{source}_mask"] = torch.zeros(batch, length, dtype=torch.bool)
        result[f"{source}_imc_validity"] = torch.ones(batch, length, len(IMC_FEATURES), dtype=torch.bool)
    result["history_weekly_contract_boundary"] = torch.zeros(batch, 3)
    result["history_weekly_contract_boundary"][:, 0] = 1
    if targets:
        result["target_minute_market"] = {
            horizon: torch.randn(batch, 2, len(IMC_FEATURES), generator=generator)
            for horizon in (16, 64, 256)
        }
        result["target_minute_imc_validity"] = {
            horizon: torch.ones(batch, 2, len(IMC_FEATURES), dtype=torch.bool)
            for horizon in (16, 64, 256)
        }
        result["target_minute_mask"] = {
            horizon: torch.zeros(batch, 2, dtype=torch.bool) for horizon in (16, 64, 256)
        }
    return result


@pytest.fixture(scope="module")
def capacity_configs():
    return load_capacity_configs()


def test_v11_capacity_exact_parameter_counts(capacity_configs):
    for size, config in capacity_configs.items():
        report = parameter_report(MarketJEPAV11(config["model"]))
        assert (
            report["trainable_parameters"], report["ema_target_parameters"],
            report["total_resident_parameters"],
        ) == EXPECTED_COUNTS[size]
    assert 95_000_000 <= EXPECTED_COUNTS["XL"][0] <= 110_000_000
    assert development_parameter_counts(capacity_configs["XL"])["v1_1_trainable"] == EXPECTED_COUNTS["XL"][0]


def test_v11_all_sizes_same_frozen_semantics(capacity_configs):
    fixed_model = (
        "minute_market_dim", "minute_context_dim", "daily_market_dim", "daily_context_dim",
        "current_weekly_market_dim", "current_weekly_context_dim", "history_weekly_market_dim",
        "history_weekly_context_dim", "minute_capacity", "daily_capacity",
        "current_weekly_capacity", "history_weekly_capacity", "commodity_state_tokens",
        "contract_state_tokens", "belief_tokens", "dropout", "commodity_embedding",
    )
    baseline = capacity_configs["S"]
    for config in capacity_configs.values():
        validate_v11_config(config)
        assert {name: config["model"][name] for name in fixed_model} == {
            name: baseline["model"][name] for name in fixed_model
        }
        assert config["data"] == baseline["data"]
        assert config["history_week"] == baseline["history_week"]
        assert config["model"]["commodity_embedding"] is False
        assert config["model"]["commodity_state_tokens"] == 4
        assert config["model"]["contract_state_tokens"] == 4
        assert config["model"]["belief_tokens"] == 8
        assert config["model"]["predictor_hidden"] == 2 * config["model"]["d_model"]
        assert config["training"]["batch_size"] * config["training"]["gradient_accumulation"] == 128
        validate_formal_training_config(config)


def test_v11_all_sizes_same_information_flow_and_forward(capacity_configs):
    expected_modules = (
        CommodityMemoryEncoder, ContractLifecycleEncoder, MinuteLocalEncoder,
        MinuteHigherScaleConditioner, BeliefEncoder,
    )
    for size, config in capacity_configs.items():
        model = MarketJEPAV11(config["model"]).eval()
        assert isinstance(model.commodity_memory, expected_modules[0])
        assert isinstance(model.contract_lifecycle, expected_modules[1])
        assert isinstance(model.minute_local, expected_modules[2])
        assert isinstance(model.minute_conditioner, expected_modules[3])
        assert isinstance(model.belief_encoder, expected_modules[4])
        with torch.inference_mode():
            output = model(**_inputs(targets=True), return_intermediates=True)
        dim = config["model"]["d_model"]
        assert output["z_market"].shape == (1, dim)
        assert all(value.shape == (1, dim) for value in output["predictions"].values())
        assert all(value.shape == (1, dim) for value in output["targets"].values())
        middle = output["intermediates"]
        assert middle["commodity_state"].shape == (1, 4, dim)
        assert middle["contract_state"].shape == (1, 4, dim)
        assert middle["minute_conditioned_tokens"].shape == (1, 4, dim)
        assert middle["belief_tokens"].shape == (1, 8, dim)
        assert not any("commodity_embedding" in name for name in model.state_dict())


def test_v11_s_state_dict_schema_is_legacy_compatible(capacity_configs):
    model = MarketJEPAV11(capacity_configs["S"]["model"])
    assert len(model.state_dict()) == 241
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 8_262_661
    assert model.state_dict()["belief_encoder.output.1.weight"].shape == (256, 256)
    assert model.state_dict()["minute_local.market_core.transformer.layers.3.linear1.weight"].shape == (1024, 256)


def test_v11_gradient_checkpointing_forward_backward_and_gradients():
    config = deepcopy(load_capacity_configs()["S"])
    config["profile"] = "debug"
    config["model"].update(
        d_model=16, belief_dim=16, num_heads=4, ffn_dim=32, minute_layers=2,
        commodity_state_tokens=2, contract_state_tokens=2, belief_tokens=2,
        predictor_hidden=32, dropout=0.0,
    )
    plain = MarketJEPAV11(config["model"], debug=True).train()
    checked = deepcopy(plain).train()
    plain.set_gradient_checkpointing(False)
    checked.set_gradient_checkpointing(True)
    inputs = _inputs(batch=2, targets=True)
    plain_output = plain(**inputs)
    checked_output = checked(**inputs)
    torch.testing.assert_close(checked_output["z_market"], plain_output["z_market"], rtol=0, atol=0)
    for horizon in (16, 64, 256):
        torch.testing.assert_close(
            checked_output["predictions"][horizon], plain_output["predictions"][horizon], rtol=0, atol=0,
        )
    loss, _ = jepa_loss(checked_output)
    loss.backward()
    connectivity = assert_all_trainable_gradients(checked)
    assert connectivity["missing"] == []
    assert connectivity["nonfinite"] == []
    assert checked.target_minute.gradient_checkpointing is False
