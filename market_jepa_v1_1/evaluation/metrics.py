from __future__ import annotations

import numpy as np

from .protocol import HORIZONS, PROTOCOL


def cosine_loss(prediction, target):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("invalid latent arrays")
    norms = np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1)
    if (norms <= 1e-12).any():
        raise ValueError("degenerate latent: cosine undefined")
    return np.clip(1 - (p * t).sum(-1) / norms, 0, 2)


def _ratio(numerator, denominator):
    if np.any(denominator <= 1e-12):
        raise ValueError("degenerate baseline: Gain/Skill undefined")
    return 1 - numerator / denominator


def bootstrap(numerator, denominator, records, *, difference=False, replicates=2000, seed=42, return_draws=False):
    """Paired contract-day draws, equal commodity means, ratio recomputed per draw.

    Within commodity, observations retain their sample weight; days are sampling
    clusters, not an alternative equal-day estimand. Input columns may be horizons
    or the twelve standardized outcome squared errors.
    """
    n, d = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if n.ndim == 1:
        n, d = n[:, None], d[:, None]
    if n.shape != d.shape or len(n) != len(records) or not len(n):
        raise ValueError("bootstrap shape/empty population")
    if not np.isfinite(n).all() or not np.isfinite(d).all():
        raise ValueError("nonfinite bootstrap observations")
    rng = np.random.default_rng(seed)
    observed_n, observed_d, sampled_n, sampled_d = [], [], [], []
    blocks_report = {}
    for commodity in sorted({r["commodity"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["commodity"] == commodity]
        keys = [(records[i]["contract_uid"], records[i]["trading_date"]) for i in indices]
        unique = sorted(set(keys)); lookup = {k: i for i, k in enumerate(unique)}
        inverse = np.asarray([lookup[k] for k in keys])
        counts = np.bincount(inverse, minlength=len(unique))
        sums_n = np.zeros((len(unique), n.shape[1])); sums_d = np.zeros_like(sums_n)
        np.add.at(sums_n, inverse, n[indices]); np.add.at(sums_d, inverse, d[indices])
        observed_n.append(n[indices].mean(0)); observed_d.append(d[indices].mean(0))
        draw_n = np.empty((replicates, n.shape[1])); draw_d = np.empty_like(draw_n)
        for j in range(replicates):
            draw = rng.integers(len(unique), size=len(unique))
            total = counts[draw].sum()
            draw_n[j] = sums_n[draw].sum(0) / total
            draw_d[j] = sums_d[draw].sum(0) / total
        sampled_n.append(draw_n); sampled_d.append(draw_d)
        blocks_report[commodity] = len(unique)
    on, od = np.mean(observed_n, axis=0), np.mean(observed_d, axis=0)
    sn, sd = np.mean(sampled_n, axis=0), np.mean(sampled_d, axis=0)
    function = (lambda a, b: a - b) if difference else _ratio
    point, draws = function(on, od), function(sn, sd)
    all_point, all_draws = function(on.mean(), od.mean()), function(sn.mean(1), sd.mean(1))
    result = {"point": point.tolist(), "ci95_low": np.quantile(draws, .025, axis=0).tolist(),
              "ci95_high": np.quantile(draws, .975, axis=0).tolist(),
              "all": {"point": float(all_point), "ci95_low": float(np.quantile(all_draws, .025)),
                      "ci95_high": float(np.quantile(all_draws, .975))},
              "blocks": blocks_report, "replicates": replicates, "seed": seed}
    width = n.shape[1] // 3
    if width * 3 == n.shape[1]:
        result["horizons"] = {}
        for i, h in enumerate(HORIZONS):
            sl = slice(i * width, (i + 1) * width)
            hp = function(on[sl].mean(), od[sl].mean())
            hd = function(sn[:, sl].mean(1), sd[:, sl].mean(1))
            result["horizons"][str(h)] = {"point": float(hp), "ci95_low": float(np.quantile(hd, .025)),
                                            "ci95_high": float(np.quantile(hd, .975))}
    if return_draws:
        result["draws_all"] = all_draws
        result["draws_horizons"] = {
            str(h): function(sn[:, i * width:(i + 1) * width].mean(1), sd[:, i * width:(i + 1) * width].mean(1))
            for i, h in enumerate(HORIZONS)
        }
    return result


def predictive_status(report):
    value = report["all"]
    if value["point"] <= 0:
        return "FAIL"
    horizons = report.get("horizons", {})
    if value["ci95_low"] > 0 and all(v["point"] >= 0 for v in horizons.values()):
        return "PASS"
    return "PARTIAL"
