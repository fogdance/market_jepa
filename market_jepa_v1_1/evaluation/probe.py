from __future__ import annotations

import numpy as np

from .protocol import ALPHAS, HORIZONS, SOURCES, WINDOWS


def future_outcomes(bars, anchor, horizon):
    """Exact V0 scalar arithmetic/order, using one real contract's OHLCVOI."""
    if anchor < 0 or anchor + horizon >= len(bars):
        raise ValueError("incomplete future horizon")
    close = float(bars[anchor, 3])
    future = bars[anchor + 1:anchor + horizon + 1]
    returns = np.diff(np.log(bars[anchor:anchor + horizon + 1, 3].astype(np.float64)))
    return np.asarray([float(future[-1, 3] / close - 1),
                       float((future[:, 1] / close - 1).max()),
                       float((future[:, 2] / close - 1).min()),
                       float(np.sqrt(np.mean(returns ** 2)))], dtype=np.float32)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def context_features(sample):
    # Current causal contexts, including progress/main status; no market channels.
    parts = []
    for source in SOURCES:
        context = _array(sample[source + "_context"])
        mask = _array(sample[source + "_mask"])
        parts.append(context[~mask][-1] if (~mask).any() else np.zeros(context.shape[1]))
        parts.append(np.asarray([float((~mask).any())]))
    return np.concatenate(parts)


def multiscale_summary(sample):
    parts = []
    for source, windows in zip(SOURCES, WINDOWS):
        values = _array(sample[source + "_market"])
        valid = _array(sample[source + "_imc_validity"])
        mask = _array(sample[source + "_mask"])
        for window in windows:
            indices = np.flatnonzero(~mask)[-window:]
            for channel in range(values.shape[1]):
                selected = values[indices, channel][valid[indices, channel]]
                parts.extend(([selected[-1], selected.mean(), selected.std(), selected.min(), selected.max()]
                              if len(selected) else [0.] * 5) + [len(selected) / window])
            if source == "history_weekly":
                # Explicit boundary/count information survives fixed summarization.
                boundary = _array(sample["history_weekly_contract_boundary"])[indices]
                parts.extend([float(boundary.sum()), len(indices) / window])
    return np.r_[parts, context_features(sample)].astype(np.float64)


def minute_raw24(sample, bars, anchor):
    """Raw, unscaled statistics restricted to the recipient's 512-minute input."""
    parts = []
    start = max(0, anchor - 511)
    for k in (4, 16, 64, 256):
        selected = bars[max(start, anchor - k + 1):anchor + 1]
        path = bars[max(start, anchor - k):anchor + 1]
        close = path[:, 3]
        logret = np.diff(np.log(close))
        oi = path[:, 5]
        valid_oi = np.isfinite(oi[[0, -1]]).all() and oi[0] > 0 and oi[-1] > 0
        # Q20 uses only preceding bars; missing baseline remains neutral.
        q = []
        for position in range(max(start, anchor - k + 1), anchor + 1):
            prior = bars[max(0, position - 20):position, 4]
            if len(prior) == 20 and np.isfinite(prior).all() and np.median(prior) > 0:
                q.append(bars[position, 4] / np.median(prior))
        parts += [float(np.log(close[-1] / close[0])), float(np.sqrt(np.mean(logret ** 2))) if len(logret) else 0.,
                  float(np.log(selected[:, 1].max() / selected[:, 2].min())),
                  float(np.log(oi[-1] / oi[0])) if valid_oi else 0.,
                  float(np.mean(q)) if q else 0., float(np.max(q)) if q else 0.]
    return np.r_[parts, context_features(sample)]


def standardizer(x):
    mean, std = x.mean(0), x.std(0)
    return mean, np.where(std < 1e-12, 1., std)


def _ridge(x, y, alpha):
    # Center intercept only; feature normalization was frozen on ProbeFit.
    xm, ym = x.mean(0), y.mean(0)
    a, b = x - xm, y - ym
    if len(a) < a.shape[1]:
        w = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(len(a)), b)
    else:
        w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ b)
    return w, ym - xm @ w


def fit_probe(x, y, fit, dev):
    if not np.any(fit) or not np.any(dev):
        raise ValueError("empty ProbeFit/ProbeDev")
    xm, xs = standardizer(x[fit]); ym, ys = standardizer(y[fit])
    sx, sy = (x - xm) / xs, (y - ym) / ys
    # One spectral factorization per fit population, shared by every alpha/outcome.
    # This is the same Ridge solution, avoiding 84 repeated cubic solves.
    def path(a, b):
        am, bm = a.mean(0), b.mean(0)
        a, b = a - am, b - bm
        if len(a) < a.shape[1]:
            eigen, vectors = np.linalg.eigh(a @ a.T)
            left, right = a.T @ vectors, vectors.T @ b
        else:
            eigen, vectors = np.linalg.eigh(a.T @ a)
            left, right = vectors, vectors.T @ (a.T @ b)
        eigen = np.maximum(eigen, 0)
        solutions = []
        for alpha in ALPHAS:
            w = left @ (right / (eigen[:, None] + alpha))
            solutions.append((w, bm - am @ w))
        return solutions
    candidates = path(sx[fit], sy[fit])
    scores = np.stack([np.square(sx[dev] @ w + b - sy[dev]).mean(0) for w, b in candidates])
    selected = np.argmin(scores, axis=0)
    refitted = path(sx[fit | dev], sy[fit | dev])
    weights = [refitted[a][0][:, j] for j, a in enumerate(selected)]
    biases = [float(refitted[a][1][j]) for j, a in enumerate(selected)]
    chosen = [ALPHAS[a] for a in selected]
    return {"x_mean": xm.tolist(), "x_std": xs.tolist(), "y_mean": ym.tolist(), "y_std": ys.tolist(),
            "weights": np.stack(weights, 1).tolist(), "bias": biases, "alphas": chosen}


def predict_probe(probe, x):
    prediction = ((x - np.asarray(probe["x_mean"])) / np.asarray(probe["x_std"])) @ np.asarray(probe["weights"]) + probe["bias"]
    return prediction * probe["y_std"] + probe["y_mean"]
