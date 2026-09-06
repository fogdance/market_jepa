from __future__ import annotations

import inspect
from copy import deepcopy

import pytest
import torch

from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.model import jepa_loss
from market_jepa_v1 import DEFAULT_V1_CONFIG, MarketJEPAV1, load_v1_config, validate_v1_config
from market_jepa_v1.data import adapt_market_batch
from market_jepa_v1.development import V0_MANIFEST_BEFORE, parameter_counts, synthetic_batch


@pytest.fixture
def v1_config():
    value = deepcopy(DEFAULT_V1_CONFIG)
    value["profile"] = "debug"
    value["model"].update(d_model=32, belief_dim=32, num_heads=4, ffn_dim=64,
        minute_layers=1, daily_gru_hidden=16, weekly_gru_hidden=16, predictor_hidden=64, dropout=0.0)
    return value


@pytest.fixture
def v1_model(v1_config):
    torch.manual_seed(31)
    return MarketJEPAV1(v1_config["model"], debug=True).eval()


@pytest.fixture
def v1_batch(v1_config):
    torch.manual_seed(32)
    return synthetic_batch(v1_config["model"], 2, minute_length=8, daily_length=4, weekly_length=3)


def test_market_jepa_v1_shapes():
    config = load_v1_config("configs/v1/market_jepa_v1.yaml")
    assert config == DEFAULT_V1_CONFIG
    model = MarketJEPAV1(config["model"]).eval()
    with torch.no_grad():
        output = model(synthetic_batch(config["model"], 2, minute_length=16), return_intermediates=True)
    assert output["z_market"].shape == (2, 256)
    debug = output["intermediates"]
    assert debug["minute_local_tokens"].shape == (2, 17, 256)
    assert debug["daily_local_tokens"].shape == (2, 32, 256)
    assert debug["weekly_local_tokens"].shape == (2, 8, 256)
    for index in (1, 2):
        assert debug[f"state_tokens_after_round_{index}"].shape == (2, 8, 256)
        assert debug[f"tokens_after_feedback_round_{index}"].shape == (2, 57, 256)
    for h in (16, 64, 256):
        assert output["targets"][h].shape == output["predictions"][h].shape == (2, 256)
    assert debug["final_belief"] is output["z_market"]


@pytest.mark.parametrize("dimensions", [(7, 7, 7), (7, 9, 11)])
def test_market_jepa_v1_independent_imc_dimensions(v1_config, dimensions):
    config = v1_config["model"]
    config.update(minute_market_dim=dimensions[0], daily_market_dim=dimensions[1], weekly_market_dim=dimensions[2],
                  minute_context_dim=3, daily_context_dim=2, weekly_context_dim=0)
    model = MarketJEPAV1(config, debug=True).eval()
    output = model(synthetic_batch(config, 2, minute_length=6))
    loss, _ = jepa_loss(output)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.online.daily.gru.input_size == dimensions[1] + 2
    assert model.online.weekly.gru.input_size == dimensions[2]


def assert_changes_belief(v1_model, v1_batch, field):
    other = deepcopy(v1_batch)
    other[field][:, 0] += torch.linspace(-3, 3, other[field].shape[-1])
    with torch.no_grad():
        first = v1_model(v1_batch, return_intermediates=True)
        second = v1_model(other, return_intermediates=True)
    assert not torch.allclose(first["z_market"], second["z_market"], atol=1e-6, rtol=1e-5)
    if field == "minute_context":
        assert not torch.allclose(first["intermediates"]["minute_local_tokens"], second["intermediates"]["minute_local_tokens"])
    else:
        torch.testing.assert_close(first["intermediates"]["minute_local_tokens"], second["intermediates"]["minute_local_tokens"], rtol=0, atol=0)


def test_market_jepa_v1_daily_changes_belief(v1_model, v1_batch):
    assert_changes_belief(v1_model, v1_batch, "daily_market")


def test_market_jepa_v1_weekly_changes_belief(v1_model, v1_batch):
    assert_changes_belief(v1_model, v1_batch, "weekly_market")


def test_market_jepa_v1_context_changes_online(v1_model, v1_batch):
    assert_changes_belief(v1_model, v1_batch, "minute_context")


def test_market_jepa_v1_no_future_context_target(v1_model, v1_batch, v1_config):
    # Emulate an upstream collated batch carrying future context metadata.
    raw = {"minute_market": v1_batch["minute_market"], "minute_context": v1_batch["minute_context"],
           "daily": torch.cat((v1_batch["daily_market"], v1_batch["daily_context"]), -1),
           "weekly": torch.cat((v1_batch["weekly_market"], v1_batch["weekly_context"]), -1),
           "daily_lengths": torch.tensor([4, 3]), "weekly_lengths": torch.tensor([3, 2]),
           "targets": v1_batch["targets"], "future_context": {h: torch.randn(2, h, 5) for h in (16, 64, 256)},
           "future_daily_context": torch.randn(2, 4, 1), "future_weekly_context": torch.randn(2, 2, 1)}
    other = deepcopy(raw)
    for h in other["future_context"]:
        other["future_context"][h] += 1000
    other["future_daily_context"] *= -100
    other["future_weekly_context"] *= -100
    other["minute_context"] += 5
    other["daily"] += 7
    other["weekly"] -= 9
    v1_model.train()  # Target must remain deterministic even while online is training.
    with torch.no_grad():
        first = v1_model(adapt_market_batch(raw, v1_config["model"]))
        second = v1_model(adapt_market_batch(other, v1_config["model"]))
    for h in v1_model.horizons:
        torch.testing.assert_close(first["targets"][h], second["targets"][h], rtol=0, atol=0)
    assert not torch.equal(first["z_market"], second["z_market"])
    assert "context" not in str(inspect.signature(v1_model.target_minute.forward))
    assert not any("context" in key for key in v1_model.target_minute.state_dict())
    with pytest.raises(TypeError):
        v1_model.target_minute(v1_batch["targets"][16], context=other["future_context"][16])


@pytest.mark.parametrize("scale", ["daily", "weekly"])
def test_market_jepa_v1_cross_scale_feedback(v1_model, v1_batch, scale):
    before_second_read = []
    handle = v1_model.online.blocks[1].register_forward_pre_hook(
        lambda _module, inputs: before_second_read.append(inputs[1].detach().clone()))
    other = deepcopy(v1_batch)
    other[f"{scale}_market"][:, 0] += 3
    try:
        with torch.no_grad():
            first = v1_model(v1_batch, return_intermediates=True)
            second = v1_model(other, return_intermediates=True)
    finally:
        handle.remove()
    n = first["intermediates"]["token_lengths"][0]
    torch.testing.assert_close(first["intermediates"]["minute_local_tokens"], second["intermediates"]["minute_local_tokens"], rtol=0, atol=0)
    assert not torch.allclose(before_second_read[0][:, 1:n-1], before_second_read[1][:, 1:n-1])
    torch.testing.assert_close(before_second_read[0], first["intermediates"]["tokens_after_feedback_round_1"], rtol=0, atol=0)
    # Cutting the first feedback removes the high-scale influence on minute tokens.
    def cut_feedback(_module, inputs, outputs):
        return outputs[0], inputs[1]
    handle = v1_model.online.blocks[0].register_forward_hook(cut_feedback)
    try:
        with torch.no_grad():
            cut_first = v1_model(v1_batch, return_intermediates=True)
            cut_second = v1_model(other, return_intermediates=True)
    finally:
        handle.remove()
    torch.testing.assert_close(cut_first["intermediates"]["tokens_after_feedback_round_1"][:, :n],
                               cut_second["intermediates"]["tokens_after_feedback_round_1"][:, :n], rtol=0, atol=0)


def test_market_jepa_v1_gradients(v1_model, v1_batch):
    output = v1_model(v1_batch, return_intermediates=True)
    debug = output["intermediates"]
    for key in ("minute_local_tokens", "daily_local_tokens", "weekly_local_tokens"):
        debug[key].retain_grad()
    output["z_market"].sum().backward()
    grads = [debug["minute_local_tokens"].grad[:, 1:-1], debug["daily_local_tokens"].grad[:, :-1],
             debug["weekly_local_tokens"].grad[:, :-1], v1_model.online.minute_context_projection.weight.grad,
             v1_model.online.state_tokens.grad]
    for grad in grads:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert (v1_model.online.state_tokens.grad.abs().sum(-1) > 0).all()
    for scale in ("daily", "weekly"):
        mask = v1_batch[f"{scale}_padding_mask"]
        grad = debug[f"{scale}_local_tokens"].grad
        assert (grad[~mask].abs().sum(-1) > 0).all()


def test_market_jepa_v1_terminal_feedback_removed_and_all_parameters_connected(v1_model, v1_batch):
    output = v1_model(v1_batch)
    loss, _ = jepa_loss(output)
    loss.backward()
    terminal = v1_model.online.blocks[1]
    assert not terminal.has_feedback and not hasattr(terminal, "feedback_attention")
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in v1_model.optimizer_parameters())


def test_market_jepa_v1_zero_rounds_is_explicit_debug(v1_config, v1_batch):
    v1_config["model"]["cross_scale_rounds"] = 0
    with pytest.raises(ValueError, match="formal V1"):
        MarketJEPAV1(v1_config["model"])
    model = MarketJEPAV1(v1_config["model"], debug=True).eval()
    assert len(model.online.blocks) == 0
    output = model(v1_batch, return_intermediates=True)
    assert not any("round_" in key for key in output["intermediates"])
    with torch.no_grad():
        expected = model.online.belief_head(model.online.minute_market.encode_tokens(
            v1_batch["minute_market"], v1_batch["minute_padding_mask"],
            context_tokens=model.online.minute_context_projection(v1_batch["minute_context"].masked_fill(v1_batch["minute_padding_mask"].unsqueeze(-1), 0)))[0][:, 0])
    torch.testing.assert_close(output["z_market"], expected)


def test_market_jepa_v1_no_commodity_embedding(v1_model, v1_batch):
    forbidden = ("commodity", "symbol_id", "instrument_embedding")
    for text in [*v1_model.state_dict(), str(inspect.signature(MarketJEPAV1)), str(inspect.signature(v1_model.forward))]:
        assert not any(key in text for key in forbidden)
    for name in forbidden:
        with pytest.raises(ValueError, match="unexpected V1"):
            v1_model({**v1_batch, name: torch.zeros(2)})
    config = deepcopy(DEFAULT_V1_CONFIG)
    config["model"]["use_commodity_embedding"] = True
    with pytest.raises(ValueError, match="forbids"):
        validate_v1_config(config)


def test_market_jepa_v1_ema_frozen(v1_model, v1_batch):
    v1_model.train()
    assert not v1_model.target_minute.training
    loss, _ = jepa_loss(v1_model(v1_batch))
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in v1_model.target_minute.parameters())
    optimizer_ids = {id(p) for p in v1_model.optimizer_parameters()}
    assert not optimizer_ids.intersection(id(p) for p in v1_model.target_minute.parameters())
    before = {k: v.clone() for k, v in v1_model.target_minute.named_parameters()}
    with torch.no_grad():
        for p in v1_model.online.minute_market.parameters():
            p.add_(1.0)
        v1_model.online.minute_context_projection.weight.add_(100)
    v1_model.update_target(0.996)
    for key, target in v1_model.target_minute.named_parameters():
        torch.testing.assert_close(target, before[key] + 0.004)
    with torch.no_grad():
        v1_model.eval()
        for h, expected in v1_model.encode_persistence(v1_batch).items():
            torch.testing.assert_close(expected, v1_model.target_minute(v1_batch["persistence"][h]))


@pytest.mark.parametrize("scale", ["minute", "daily", "weekly"])
def test_market_jepa_v1_padding_invariance(v1_model, v1_batch, scale):
    # Place padding before, between and after valid tokens, with NaN padding values.
    base = {key: ({h: value[:1].clone() for h, value in val.items()} if isinstance(val, dict) else val[:1].clone()) for key, val in v1_batch.items()}
    length = base[f"{scale}_market"].shape[1]
    expanded = deepcopy(base)
    for kind in ("market", "context"):
        source = base[f"{scale}_{kind}"]
        values = torch.full((1, length * 2 + 1, source.shape[-1]), float("nan"))
        values[:, 1::2] = source
        expanded[f"{scale}_{kind}"] = values
    mask = torch.ones(1, length * 2 + 1, dtype=torch.bool)
    mask[:, 1::2] = False
    expanded[f"{scale}_padding_mask"] = mask
    with torch.no_grad():
        original, padded = v1_model(base), v1_model(expanded, return_intermediates=True)
    torch.testing.assert_close(original["z_market"], padded["z_market"], atol=2e-6, rtol=2e-5)
    debug = padded["intermediates"]
    assert (debug["tokens_after_feedback_round_2"][debug["token_padding_mask"]] == 0).all()
    for kind in ("market", "context"):
        expanded[f"{scale}_{kind}"].requires_grad_()
    v1_model(expanded)["z_market"].sum().backward()
    for kind in ("market", "context"):
        grad = expanded[f"{scale}_{kind}"].grad
        assert torch.isfinite(grad).all()
        assert (grad[mask] == 0).all()


def test_market_jepa_v1_empty_periods_and_invalid_inputs(v1_model, v1_batch):
    for scale in ("daily", "weekly"):
        v1_batch[f"{scale}_padding_mask"][:] = True
        v1_batch[f"{scale}_market"][:] = float("nan")
    assert torch.isfinite(v1_model(v1_batch)["z_market"]).all()
    for scale in ("daily", "weekly"):
        for key in (f"{scale}_market", f"{scale}_context", f"{scale}_padding_mask"):
            v1_batch[key] = v1_batch[key][:, :0]
    assert torch.isfinite(v1_model(v1_batch)["z_market"]).all()
    v1_batch["minute_padding_mask"][:] = True
    with pytest.raises(ValueError, match="at least one"):
        v1_model(v1_batch)


def test_market_jepa_v1_target_padding(v1_model):
    market = torch.randn(2, 5, 14)
    padded = torch.cat((market, torch.full((2, 3, 14), float("nan"))), dim=1)
    mask = torch.tensor([[False] * 5 + [True] * 3] * 2)
    with torch.no_grad():
        torch.testing.assert_close(v1_model.target_minute(market), v1_model.target_minute(padded, mask), atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="bool"):
        v1_model.target_minute(market, torch.zeros(2, 5))


def test_market_jepa_v1_rejects_wrong_target_horizon(v1_model, v1_batch):
    v1_batch["targets"][16] = v1_batch["targets"][16][:, :15]
    with pytest.raises(ValueError, match="H16 target"):
        v1_model(v1_batch)
    v1_batch["persistence"][64] = v1_batch["persistence"][64][:1]
    with pytest.raises(ValueError, match="H64 persistence"):
        v1_model.encode_persistence(v1_batch)


def test_market_jepa_v1_debug_does_not_change_normal_forward(v1_model, v1_batch):
    with torch.no_grad():
        normal = v1_model(v1_batch)
        debug = v1_model(v1_batch, return_intermediates=True)
    assert "intermediates" not in normal
    torch.testing.assert_close(normal["z_market"], debug["z_market"], rtol=0, atol=0)
    assert not any("intermediate" in key for key in vars(v1_model.online))


def test_market_jepa_v0_regression():
    assert manifest_sha256(implementation_manifest()) == V0_MANIFEST_BEFORE
    counts = parameter_counts(DEFAULT_V1_CONFIG)
    assert counts["v0_trainable"] == 4_676_512
    assert counts["v0_ema"] == 3_163_648
    assert counts["v1_trainable"] < 3 * counts["v0_trainable"]
    assert sum(counts["v1_components"].values()) + counts["v1_state_tokens"] + counts["v1_predictors"] == counts["v1_trainable"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires accessible CUDA")
def test_market_jepa_v1_cuda_calibration_preserves_weights_and_rng():
    from market_jepa.train.trainer import configure_determinism, capture_rng_state, _move
    from market_jepa_v1.development import calibrate_smoke_scaler
    configure_determinism(42)
    config = DEFAULT_V1_CONFIG["model"]
    model = MarketJEPAV1(config).cuda().train()
    batch = _move(synthetic_batch(config, 2), torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.optimizer_parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = capture_rng_state(include_cuda=True)
    scaler, attempts = calibrate_smoke_scaler(model, batch, optimizer, torch.device("cuda"), amp=True)
    assert attempts[-1]["gradients_finite"] and 1 <= len(attempts) <= 8
    assert scaler.get_scale() == attempts[-1]["loss_scale"]
    assert not optimizer.state and all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    after = capture_rng_state(include_cuda=True)
    assert torch.equal(rng["torch_cpu"], after["torch_cpu"])
    assert all(torch.equal(a, b) for a, b in zip(rng["torch_cuda"], after["torch_cuda"], strict=True))
