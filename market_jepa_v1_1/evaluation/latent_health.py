"""Numerical collapse checks; low effective rank is a warning, not a quality gate."""
import numpy as np


def latent_health(values, *, minimum_samples=3, epsilon=1e-12):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("latent health requires N x d")
    n, d = x.shape
    report = {"sample_count": n, "dimension": d, "minimum_samples": minimum_samples,
              "numerical_epsilon": epsilon, "relative_rank_threshold": 1e-6,
              "finite_fraction": float(np.isfinite(x).mean()) if x.size else 0.0}
    if n < minimum_samples or not np.isfinite(x).all():
        return {**report, "status": "BLOCKED_TRIVIAL_COLLAPSE", "reason": "insufficient_samples_or_nonfinite"}
    centered = x - x.mean(0)
    eigen = np.linalg.svd(centered, compute_uv=False) ** 2 / (n - 1)
    total = float(eigen.sum())
    rank = int((eigen > eigen.max() * 1e-6).sum())
    p = eigen[eigen > 0] / total if total else np.array([])
    effective = float(np.exp(-(p * np.log(p)).sum())) if total else 0.0
    std = x.std(0, ddof=1)
    def pairwise(v):
        if len(v) > 4096:
            v = v[np.random.default_rng(42).choice(len(v), 4096, replace=False)]
        norms = np.linalg.norm(v, axis=1)
        v = v[norms > epsilon] / norms[norms > epsilon, None]
        if len(v) < 2:
            return None
        # Sum of off-diagonal dot products without allocating an N x N matrix.
        return float((np.square(v.sum(0)).sum() - np.square(v).sum()) / (len(v) * (len(v) - 1)))
    raw_cos = pairwise(x)
    report.update(mean_vector_norm=float(np.linalg.norm(x.mean(0))),
                  per_dimension_std=dict(zip(("min", "p10", "median", "mean", "max"),
                      map(float, (std.min(), np.quantile(std, .1), np.median(std), std.mean(), std.max())))),
                  centered_total_variance=total, covariance_eigenvalues=eigen.tolist(),
                  numerical_rank=rank, effective_rank=effective, effective_rank_ratio=effective / d,
                  participation_ratio=float(total ** 2 / np.square(eigen).sum()) if total else 0.,
                  mean_raw_pairwise_cosine=raw_cos, mean_centered_pairwise_cosine=pairwise(centered))
    report["warnings"] = (["low_effective_rank_ratio"] if effective / d < .05 else []) + (
        ["high_raw_pairwise_cosine"] if raw_cos is not None and raw_cos > .99 else [])
    report["status"] = "BLOCKED_TRIVIAL_COLLAPSE" if total <= epsilon or rank < 2 else "PASS"
    return report
