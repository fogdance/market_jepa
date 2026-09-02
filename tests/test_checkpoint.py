from pathlib import Path

import torch

from market_jepa.train.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    state = {"tensor": torch.arange(5), "nested": {"value": 3}, "design_version": "0.6.0"}
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    assert len(digest) == 64
    assert torch.equal(loaded["tensor"], state["tensor"])
    assert loaded["nested"] == state["nested"]
