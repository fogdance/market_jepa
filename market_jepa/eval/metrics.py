from __future__ import annotations

from typing import Any

import numpy as np


def cosine_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return 1.0 - numerator / np.maximum(denominator, 1e-12)


def block_bootstrap(
    effects: np.ndarray, blocks: np.ndarray, samples: int, seed: int
) -> dict[str, float]:
    effects = np.asarray(effects, dtype=np.float64)
    blocks = np.asarray(blocks)
    unique, inverse = np.unique(blocks, return_inverse=True)
    block_means = np.asarray([effects[inverse == index].mean() for index in range(len(unique))])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(block_means), size=(samples, len(block_means)))
    distribution = block_means[draws].mean(axis=1)
    return {
        "effect": float(block_means.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "blocks": int(len(unique)),
    }


def block_shuffle_indices(timestamp_ns: np.ndarray, seed: int) -> np.ndarray:
    timestamps = timestamp_ns.astype("datetime64[ns]")
    # Keep the full year-month. Grouping only by month number would mix, for
    # example, January 2018 and January 2022 and make the control too easy.
    months = timestamps.astype("datetime64[M]").astype(int)
    hours = (timestamps.astype("datetime64[h]").astype(int) % 24)
    sessions = hours >= 18
    result = np.arange(len(timestamps))
    rng = np.random.default_rng(seed)
    for month in range(12):
        for session in (False, True):
            group = np.flatnonzero((months == month) & (sessions == session))
            if len(group) > 1:
                shift = int(rng.integers(1, len(group)))
                result[group] = np.roll(group, shift)
    return result


def nearest_indices(
    reference: np.ndarray,
    query: np.ndarray,
    reference_timestamp: np.ndarray,
    query_timestamp: np.ndarray,
    k: int,
    chunk_size: int = 64,
) -> np.ndarray:
    if k > len(reference):
        raise ValueError("k exceeds reference sample count")
    if np.max(reference_timestamp) >= np.min(query_timestamp):
        # The V0 split makes all train rows earlier; fail instead of silently using future neighbors.
        raise ValueError("kNN reference contains a query-time-or-future row")
    ref = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    qry = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    result = np.empty((len(query), k), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        similarity = qry[start:stop] @ ref.T
        candidates = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        row = np.arange(stop - start)[:, None]
        order = np.argsort(-similarity[row, candidates], axis=1)
        result[start:stop] = candidates[row, order]
    return result


def outcome_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q25": float(np.quantile(flat, 0.25)),
        "q75": float(np.quantile(flat, 0.75)),
        "q95": float(np.quantile(flat, 0.95)),
        "positive_probability": float((flat > 0).mean()),
        "negative_probability": float((flat < 0).mean()),
    }


class Ridge:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x_mean = x.mean(0)
        self.x_std = x.std(0)
        self.x_std[self.x_std < 1e-12] = 1.0
        self.y_mean = y.mean(0)
        standardized = (x - self.x_mean) / self.x_std
        centered_y = y - self.y_mean
        if len(x) <= x.shape[1]:
            kernel = standardized @ standardized.T
            self.dual = np.linalg.solve(
                kernel + self.alpha * np.eye(len(kernel)), centered_y
            )
            self.train_x = standardized
            self.weight = None
        else:
            gram = standardized.T @ standardized
            self.weight = np.linalg.solve(
                gram + self.alpha * np.eye(gram.shape[0]), standardized.T @ centered_y
            )
            self.dual = None
            self.train_x = None
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        if self.weight is not None:
            return standardized @ self.weight + self.y_mean
        return standardized @ self.train_x.T @ self.dual + self.y_mean
