from __future__ import annotations

from typing import Iterable

import numpy as np
from torch.utils.data import Sampler

from .dataset import V11ContractDataset


class HierarchicalCommodityContractSampler(Sampler[int]):
    """Deterministic uniform Commodity -> Contract -> Anchor sampling."""

    def __init__(
        self, dataset: V11ContractDataset, num_samples: int, seed: int,
        commodities: Iterable[str] | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset, self.num_samples, self.seed = dataset, int(num_samples), int(seed)
        self.epoch = 0
        hierarchy = dataset.hierarchy
        configured = tuple(commodities) if commodities is not None else dataset.train_commodities
        self.commodities = tuple(c for c in configured if c in hierarchy)
        if self.commodities != configured:
            missing = set(configured) - set(self.commodities)
            raise ValueError(f"balanced sampler is missing train commodities: {sorted(missing)}")
        self.episodes = {commodity: tuple(hierarchy[commodity]) for commodity in self.commodities}
        if any(not values for values in self.episodes.values()):
            raise ValueError("every train commodity requires a valid episode")
        for commodity, episode_indices in self.episodes.items():
            eligible = set(dataset.eligible_contracts_by_commodity[commodity])
            if any(dataset.episode_arrays[index].episode.key not in eligible for index in episode_indices):
                raise ValueError("sampler hierarchy contains a history-ineligible contract")

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
