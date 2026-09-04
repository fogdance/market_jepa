from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from export_market_latents import VALIDATION_EXPORT_SPLITS, export_split
from market_jepa.data import CONTEXT_FEATURES, MARKET_FEATURES, MarketDataset
from market_jepa.model import MarketJEPA


def test_export_preserves_timestamp_provenance_and_hashes(
    tmp_path, causal_data, causal_config
) -> None:
    model_config = deepcopy(causal_config["model"])
    model_config.update(
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
    model = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), causal_config["data"]["horizons"], model_config
    )
    original_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    dataset = MarketDataset(causal_data, causal_config, "train", indices=np.asarray([4]))
    path = export_split(
        model,
        dataset,
        tmp_path / "train.npz",
        torch.device("cpu"),
        batch_size=1,
        provenance={"checkpoint_sha256": "checkpoint", "source_sha256": "source"},
    )
    with np.load(path, allow_pickle=False) as exported:
        assert exported["timestamp_ns"][0] == causal_data.minute.iloc[4]["timestamp"].value
        assert exported["daily_partial_source_max"][0] == 4
        assert exported["weekly_partial_source_max"][0] == 4
        assert exported["checkpoint_sha256"][0] == "checkpoint"
        assert exported["source_sha256"][0] == "source"
        assert exported["symbol"][0] == "JM"
        assert exported["series_id"][0] == "8Y_DCE_JM2601"
        assert exported["z_target_h1"].shape == (1, 16)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original_parameters[name]), name


def test_validation_export_surface_excludes_test() -> None:
    assert VALIDATION_EXPORT_SPLITS == ("train", "validation")
