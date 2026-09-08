from __future__ import annotations

from typing import Iterable
import hashlib
import json

import numpy as np
from torch.utils.data import Sampler

from .dataset import V11ContractDataset


class HierarchicalCommodityContractSampler(Sampler[int]):
    """Deterministic uniform Commodity -> Contract -> Anchor sampling."""

    def __init__(
        self, dataset: V11ContractDataset, num_samples: int, seed: int,
        commodities: Iterable[str] | None = None, sampling_population_sha256: str = "",
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset, self.num_samples, self.seed = dataset, int(num_samples), int(seed)
        self.sampling_cycle = 0
        self.start_offset = 0
        self.sampling_population_sha256 = str(sampling_population_sha256)
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
        return self.num_samples - self.start_offset

    def configure_cycle(self, cycle: int, num_samples: int, start_offset: int = 0) -> None:
        if cycle < 0 or num_samples <= 0 or not 0 <= start_offset < num_samples:
            raise ValueError("invalid sampling cycle/budget/offset")
        self.sampling_cycle = int(cycle)
        self.num_samples = int(num_samples)
        self.start_offset = int(start_offset)

    def set_epoch(self, epoch: int) -> None:
        """Legacy test compatibility; fixed-budget training calls configure_cycle."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.sampling_cycle = int(epoch)
        self.start_offset = 0

    def _draw(self, rng) -> int:
        commodity = self.commodities[int(rng.integers(len(self.commodities)))]
        episode_index = self.episodes[commodity][int(rng.integers(len(self.episodes[commodity])))]
        count = len(self.dataset.episode_arrays[episode_index].anchors)
        local_index = int(rng.integers(count))
        return self.dataset.global_index(episode_index, local_index)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.sampling_cycle]))
        for _ in range(self.start_offset):
            self._draw(rng)
        for _ in range(self.start_offset, self.num_samples):
            yield self._draw(rng)

    def state_dict(self) -> dict[str, int | str]:
        return {"seed": self.seed, "sampling_cycle": self.sampling_cycle,
                "num_samples": self.num_samples, "start_offset": self.start_offset,
                "sampling_population_sha256": self.sampling_population_sha256}

    def load_state_dict(self, state: dict[str, int | str]) -> None:
        if "epoch" in state:
            raise ValueError("legacy epoch sampler state cannot resume fixed_sample_budget_v1")
        expected = {"seed": self.seed, "sampling_population_sha256": self.sampling_population_sha256}
        if any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("hierarchical sampler state/config mismatch")
        self.configure_cycle(int(state["sampling_cycle"]), int(state["num_samples"]), int(state["start_offset"]))


def sampling_population_sha256(dataset: V11ContractDataset, data_manifest_sha256: str) -> str:
    config = dataset.config
    commodities = list(dataset.train_commodities)
    hierarchy = dataset.hierarchy
    population = []
    for commodity in commodities:
        episodes = []
        for index in hierarchy[commodity]:
            arrays = dataset.episode_arrays[index]
            episode = arrays.episode
            anchors = np.asarray(arrays.anchors, dtype=np.int64)
            episodes.append({
                "key": [episode.commodity, episode.contract_uid, int(episode.episode_id)],
                "anchor_count": int(len(anchors)),
                "anchor_index_sha256": hashlib.sha256(anchors.tobytes(order="C")).hexdigest(),
            })
        # Preserve the exact hierarchy order used by the sampler. Re-sorting here
        # could hide a change that maps the same RNG draw to a different episode.
        population.append({"commodity": commodity, "episodes": episodes})
    payload = {
        "data_manifest_sha256": str(data_manifest_sha256),
        "ordered_effective_train_commodities": commodities,
        "population": population,
        "history_week": config["history_week"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
