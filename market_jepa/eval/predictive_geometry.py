from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .metrics import nearest_indices
from .pipeline import _knn_report, _load, _strip_arrays, _validate_export


def _diagnostic_knn(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    train_state: np.ndarray,
    validation_state: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    horizon = 64
    k = 50
    neighbors = nearest_indices(
        train_state,
        validation_state,
        train["timestamp_ns"],
        validation["timestamp_ns"],
        k,
    )
    result = _knn_report(
        train,
        validation,
        horizon,
        [k],
        neighbors,
        bootstrap_samples,
        seed,
        bootstrap_seed=seed,
    )[str(k)]
    bootstrap = result["improvement"]
    return {
        "horizon": horizon,
        "k": k,
        "latent_error": result["nearest_outcome_mse"],
        "random_error": result["random_outcome_mse"],
        "effect": bootstrap["effect"],
        "ci95": [bootstrap["ci95_low"], bootstrap["ci95_high"]],
        "pass": bootstrap["ci95_low"] > 0,
    }


def evaluate_predictive_geometry(
    input_dir: str | Path,
    config: dict[str, Any],
    z_market_result: dict[str, Any],
    expected_checkpoint_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare predictor-output geometries using Train and Validation only."""

    horizon = 64
    k = 50
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["seed"])
    if horizon not in config["data"]["horizons"] or k not in config["evaluation"]["ks"]:
        raise ValueError("predictive geometry diagnostic requires H64 and K=50")
    if bootstrap_samples != 10_000 or seed != 42:
        raise ValueError("diagnostic requires 10000 bootstraps with seed 42")

    directory = Path(input_dir)
    metadata = {
        "split",
        "design_version",
        "ablation",
        "symbol",
        "series_id",
        "checkpoint_sha256",
        "source_sha256",
        "timestamp_ns",
        "trading_day_ns",
        "outcomes_h64",
    }
    predictions = {"z_prediction_h16", "z_prediction_h64", "z_prediction_h256"}
    train = _load(directory / "train.npz", metadata | predictions)
    validation = _load(directory / "validation.npz", metadata | predictions)
    for split_name, data in (("train", train), ("validation", validation)):
        _validate_export(
            split_name,
            data,
            config,
            expected_checkpoint_sha256,
            expected_source_sha256,
        )

    p64 = _diagnostic_knn(
        train,
        validation,
        train["z_prediction_h64"],
        validation["z_prediction_h64"],
        bootstrap_samples,
        seed,
    )
    train_multi = np.concatenate(
        [train[f"z_prediction_h{value}"] for value in (16, 64, 256)], axis=1
    )
    validation_multi = np.concatenate(
        [validation[f"z_prediction_h{value}"] for value in (16, 64, 256)], axis=1
    )
    multi = _diagnostic_knn(
        train,
        validation,
        train_multi,
        validation_multi,
        bootstrap_samples,
        seed,
    )

    baseline_random_error = float(z_market_result["random_error"])
    if not (
        np.isclose(p64["random_error"], baseline_random_error, rtol=0.0, atol=1e-12)
        and np.isclose(multi["random_error"], baseline_random_error, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("diagnostic did not reuse the frozen random baseline")

    if not z_market_result["pass"] and p64["pass"] and multi["pass"]:
        interpretation = (
            "Predictive geometry may form mainly in predictor output rather than internal "
            "Z_market; this supports Internal Belief != Predictive State."
        )
    elif not z_market_result["pass"] and not p64["pass"] and not multi["pass"]:
        interpretation = (
            "The predictor outputs also fail to form the required future-distribution "
            "geometry; a future study must examine the predictive-state objective or target "
            "representation rather than only relocating the state."
        )
    else:
        interpretation = (
            "The diagnostic is mixed; it does not establish either pre-registered mechanism "
            "pattern. The formal V0 conclusion remains unchanged."
        )

    return _strip_arrays(
        {
            "diagnostic": "Predictive Geometry Diagnostic",
            "design_version": "0.6.2",
            "formal_v0_decision": "VALIDATION_NO_GO",
            "z_market_knn": dict(z_market_result),
            "p64_knn": p64,
            "multi_horizon_predictive_state_knn": multi,
            "interpretation": interpretation,
            "test_consumed": False,
        }
    )
