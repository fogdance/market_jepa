from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.model import MarketJEPA, jepa_loss


def small_model() -> MarketJEPA:
    config = deepcopy(DEFAULT_CONFIG["model"])
    config.update(
        minute_d_model=16,
        latent_dim=16,
        minute_layers=1,
        minute_heads=4,
        minute_ffn_dim=32,
        time_hidden=4,
        daily_hidden=8,
        weekly_hidden=8,
        recurrent_layers=1,
        fusion_hidden=32,
        predictor_hidden=32,
        dropout=0.0,
    )
    return MarketJEPA(14, 5, [2, 4], config)


def fake_batch() -> dict:
    return {
        "minute_market": torch.randn(3, 8, 14),
        "minute_context": torch.randn(3, 8, 5),
        "daily": torch.randn(3, 4, 15),
        "daily_lengths": torch.tensor([4, 3, 2]),
        "weekly": torch.randn(3, 2, 15),
        "weekly_lengths": torch.tensor([2, 2, 1]),
        "targets": {2: torch.randn(3, 2, 14), 4: torch.randn(3, 4, 14)},
        "persistence": {2: torch.randn(3, 2, 14), 4: torch.randn(3, 4, 14)},
    }


def test_online_and_target_market_encoders_are_isomorphic() -> None:
    model = small_model()
    assert type(model.online.minute_market) is type(model.target_minute)
    online = model.online.minute_market.state_dict()
    target = model.target_minute.state_dict()
    assert online.keys() == target.keys()
    assert all(online[key].shape == target[key].shape for key in online)
    assert all(torch.equal(online[key], target[key]) for key in online)
    with pytest.raises(TypeError):
        model.target_minute(torch.randn(2, 4, 14), torch.randn(2, 4, 5))


def test_target_has_no_gradient_and_is_not_optimized() -> None:
    model = small_model()
    output = model(fake_batch())
    loss, metrics = jepa_loss(output)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.target_minute.parameters())
    optimizer_ids = {id(parameter) for parameter in model.optimizer_parameters()}
    assert not optimizer_ids.intersection(id(parameter) for parameter in model.target_minute.parameters())
    assert output["predictions"][2].shape == output["targets"][2].shape == (3, 16)
    assert torch.isfinite(metrics["total_loss"])


def test_ema_updates_only_from_isomorphic_market_encoder() -> None:
    model = small_model()
    first_name, online_parameter = next(iter(model.online.minute_market.named_parameters()))
    target_parameter = dict(model.target_minute.named_parameters())[first_name]
    before = target_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(2.0)
    model.update_target(0.75)
    assert torch.allclose(target_parameter, before + 0.5)


def test_inference_mode_matches_no_grad_outputs_exactly() -> None:
    model = small_model().eval()
    batch = fake_batch()
    with torch.no_grad():
        expected = model(batch)
    with torch.inference_mode():
        actual = model(batch)
    torch.testing.assert_close(actual["z_market"], expected["z_market"], rtol=0, atol=0)
    for horizon in model.horizons:
        torch.testing.assert_close(
            actual["predictions"][horizon],
            expected["predictions"][horizon],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual["targets"][horizon], expected["targets"][horizon], rtol=0, atol=0
        )
