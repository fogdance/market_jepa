from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

from .config import TRAIN_COMMODITIES
from .dataset import V11ContractDataset


class HierarchicalCommodityContractSampler(Sampler[int]):
    """Deterministic uniform Commodity -> Contract -> Anchor sampling."""

    def __init__(self, dataset: V11ContractDataset, num_samples: int, seed: int) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset, self.num_samples, self.seed = dataset, int(num_samples), int(seed)
        self.epoch = 0
        hierarchy = dataset.hierarchy
        self.commodities = tuple(c for c in TRAIN_COMMODITIES if c in hierarchy)
        if tuple(self.commodities) != TRAIN_COMMODITIES:
            missing = set(TRAIN_COMMODITIES) - set(self.commodities)
            raise ValueError(f"balanced sampler is missing train commodities: {sorted(missing)}")
        self.episodes = {commodity: tuple(hierarchy[commodity]) for commodity in self.commodities}
        if any(not values for values in self.episodes.values()):
            raise ValueError("every train commodity requires a valid episode")

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for _ in range(self.num_samples):
            commodity = self.commodities[int(rng.integers(len(self.commodities)))]
            episode_index = self.episodes[commodity][int(rng.integers(len(self.episodes[commodity])))]
            count = len(self.dataset.episode_arrays[episode_index].anchors)
            local_index = int(rng.integers(count))
            yield self.dataset.global_index(episode_index, local_index)

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch, "num_samples": self.num_samples}

    def load_state_dict(self, state: dict[str, int]) -> None:
        expected = {"seed": self.seed, "num_samples": self.num_samples}
        if any(int(state[name]) != value for name, value in expected.items()):
            raise ValueError("hierarchical sampler state/config mismatch")
        self.set_epoch(int(state["epoch"]))
