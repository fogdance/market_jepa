from __future__ import annotations

import numpy as np
import torch


class LatentAccumulator:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.sum = torch.zeros(dimension, dtype=torch.float64)
        self.cross = torch.zeros(dimension, dimension, dtype=torch.float64)

    def update(self, value: torch.Tensor) -> None:
        cpu = value.detach().to(device="cpu", dtype=torch.float64)
        self.count += len(cpu)
        self.sum += cpu.sum(dim=0)
        self.cross += cpu.T @ cpu

    def metrics(self, std_threshold: float) -> dict[str, float | bool]:
        if self.count == 0:
            raise ValueError("no latent samples accumulated")
        mean = self.sum / self.count
        covariance = self.cross / self.count - torch.outer(mean, mean)
        covariance = (covariance + covariance.T) * 0.5
        variance = torch.diagonal(covariance).clamp_min(0.0)
        std = variance.sqrt()
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        eigen_sum = eigenvalues.sum()
        if eigen_sum <= 0:
            effective_rank = 0.0
        else:
            probability = eigenvalues / eigen_sum
            nonzero = probability > 0
            entropy = -(probability[nonzero] * probability[nonzero].log()).sum()
            effective_rank = float(entropy.exp())
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        return {
            "mean_std": float(std.mean()),
            "effective_rank": effective_rank,
            "low_std_fraction": float((std < std_threshold).to(torch.float64).mean()),
            "covariance_offdiag_rms": float(torch.sqrt(torch.mean(off_diagonal.square()))),
            "collapsed": bool(float(std.mean()) < std_threshold),
        }


class MetricAverage:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, torch.Tensor | float], batch_size: int) -> None:
        self.count += batch_size
        for name, value in metrics.items():
            number = float(value.item() if isinstance(value, torch.Tensor) else value)
            self.total[name] = self.total.get(name, 0.0) + number * batch_size

    def result(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("no metrics accumulated")
        return {name: value / self.count for name, value in self.total.items()}
