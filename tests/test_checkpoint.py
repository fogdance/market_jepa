from pathlib import Path
import random

import numpy as np
import torch

from market_jepa.train.checkpoint import load_checkpoint, save_checkpoint
from market_jepa.train.trainer import capture_rng_state, configure_determinism, restore_rng_state


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    state = {"tensor": torch.arange(5), "nested": {"value": 3}, "design_version": "0.6.1"}
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    assert len(digest) == 64
    assert torch.equal(loaded["tensor"], state["tensor"])
    assert loaded["nested"] == state["nested"]


def test_rng_state_round_trip() -> None:
    configure_determinism(42)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.random()
    np.random.random()
    torch.rand(3)
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
