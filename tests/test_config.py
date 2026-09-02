from copy import deepcopy

import pytest

from market_jepa.config import DEFAULT_CONFIG, FROZEN_EXPERIMENTS, load_config, validate_config


def test_frozen_yaml_matches_protocol() -> None:
    assert load_config("configs/market_jepa_v0.yaml") == DEFAULT_CONFIG


def test_formal_protocol_rejects_research_freedom() -> None:
    changed = deepcopy(DEFAULT_CONFIG)
    changed["training"]["seed"] = 7
    with pytest.raises(ValueError, match="frozen"):
        validate_config(changed, smoke=False)


def test_only_preregistered_ablation_pairs_are_accepted() -> None:
    for experiment_id, ablation in FROZEN_EXPERIMENTS.items():
        config = deepcopy(DEFAULT_CONFIG)
        config["experiment_id"] = experiment_id
        config["model"]["ablation"] = ablation
        validate_config(config, smoke=False)
    mismatched = deepcopy(DEFAULT_CONFIG)
    mismatched["model"]["ablation"] = "minute"
    with pytest.raises(ValueError, match="frozen"):
        validate_config(mismatched, smoke=False)
