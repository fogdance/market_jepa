from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    Ridge,
    block_bootstrap,
    block_shuffle_indices,
    cosine_error,
    nearest_indices,
    outcome_summary,
)


OUTCOME_NAMES = ("return", "mfe", "mae", "realized_volatility")


def _load(path: Path, names: set[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        selected = loaded.files if names is None else names
        missing = set(selected).difference(loaded.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
        return {name: loaded[name] for name in selected}


def _date_range(data: dict[str, np.ndarray]) -> dict[str, str | int]:
    timestamps = data["timestamp_ns"].astype("datetime64[ns]")
    return {"start": str(timestamps.min()), "end": str(timestamps.max()), "samples": len(timestamps)}


def _future_report(
    train: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    horizon: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    target = data[f"z_target_h{horizon}"]
    prediction = data[f"z_prediction_h{horizon}"]
    persistence = data[f"z_persistence_h{horizon}"]
    mean_target = train[f"z_target_h{horizon}"].mean(axis=0, keepdims=True)
    learned_error = cosine_error(prediction, target)
    persistence_error = cosine_error(persistence, target)
    mean_error = cosine_error(np.repeat(mean_target, len(target), axis=0), target)
    shuffle = block_shuffle_indices(data["timestamp_ns"], seed)
    shuffle_error = cosine_error(prediction, target[shuffle])
    shuffle_valid = shuffle != np.arange(len(shuffle))
    effect = persistence_error - learned_error
    return {
        "learned_cosine_error": float(learned_error.mean()),
        "learned_mse": float(np.mean((prediction - target) ** 2)),
        "persistence_cosine_error": float(persistence_error.mean()),
        "train_mean_cosine_error": float(mean_error.mean()),
        "block_shuffle_cosine_error": (
            float(shuffle_error[shuffle_valid].mean()) if shuffle_valid.any() else None
        ),
        "block_shuffle_samples": int(shuffle_valid.sum()),
        "persistence_improvement": block_bootstrap(
            effect, data["trading_day_ns"], bootstrap_samples, seed
        ),
        "paired_effects": effect,
    }


def _knn_report(
    train: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    horizon: int,
    ks: list[int],
    neighbor_order: np.ndarray,
    bootstrap_samples: int,
    seed: int,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    train_outcome = train[f"outcomes_h{horizon}"].astype(np.float64)
    query_outcome = data[f"outcomes_h{horizon}"].astype(np.float64)
    scale = train_outcome.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_scaled = (train_outcome - train_outcome.mean(0)) / scale
    query_scaled = (query_outcome - train_outcome.mean(0)) / scale
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for k in ks:
        if k > len(train_outcome):
            continue
        neighbor = neighbor_order[:, :k]
        random_neighbor = np.stack(
            [rng.choice(len(train_outcome), size=k, replace=False) for _ in range(len(query_outcome))]
        )
        predicted = train_scaled[neighbor].mean(axis=1)
        random_predicted = train_scaled[random_neighbor].mean(axis=1)
        nearest_error = np.mean((query_scaled - predicted) ** 2, axis=1)
        random_error = np.mean((query_scaled - random_predicted) ** 2, axis=1)
        summaries = {
            name: {
                "nearest": outcome_summary(train_outcome[neighbor, index]),
                "random": outcome_summary(train_outcome[random_neighbor, index]),
            }
            for index, name in enumerate(OUTCOME_NAMES)
        }
        result[str(k)] = {
            "nearest_outcome_mse": float(nearest_error.mean()),
            "random_outcome_mse": float(random_error.mean()),
            "improvement": block_bootstrap(
                random_error - nearest_error,
                data["trading_day_ns"],
                bootstrap_samples,
                seed + k if bootstrap_seed is None else bootstrap_seed,
            ),
            "outcome_distributions": summaries,
            "paired_effects": random_error - nearest_error,
        }
    return result


def _probe_report(
    train: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    horizon: int,
    alpha: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    columns = [0, 3]
    y_train = train[f"outcomes_h{horizon}"][:, columns]
    y = data[f"outcomes_h{horizon}"][:, columns]
    latent_prediction = Ridge(alpha).fit(train["z_market"], y_train).predict(data["z_market"])
    raw_prediction = Ridge(alpha).fit(train["raw_features"], y_train).predict(data["raw_features"])
    variance = y_train.var(axis=0)
    variance[variance < 1e-12] = 1.0
    latent_error_by_target = np.mean((latent_prediction - y) ** 2, axis=0)
    raw_error_by_target = np.mean((raw_prediction - y) ** 2, axis=0)
    latent_row = np.mean((latent_prediction - y) ** 2 / variance, axis=1)
    raw_row = np.mean((raw_prediction - y) ** 2 / variance, axis=1)
    return {
        "latent_mse": {
            "return": float(latent_error_by_target[0]),
            "realized_volatility": float(latent_error_by_target[1]),
        },
        "raw_mse": {
            "return": float(raw_error_by_target[0]),
            "realized_volatility": float(raw_error_by_target[1]),
        },
        "latent_normalized_mse": float(latent_row.mean()),
        "raw_normalized_mse": float(raw_row.mean()),
        "normalized_mse_improvement": block_bootstrap(
            raw_row - latent_row, data["trading_day_ns"], bootstrap_samples, seed
        ),
        "paired_effects": raw_row - latent_row,
    }


def _strip_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_arrays(item) for key, item in value.items() if key != "paired_effects"}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_export(
    split_name: str,
    data: dict[str, np.ndarray],
    config: dict[str, Any],
    expected_checkpoint_sha256: str | None,
    expected_source_sha256: str | None,
) -> None:
    if set(data["split"].astype(str)) != {split_name}:
        raise ValueError(f"{split_name}.npz contains a different split label")
    if str(data["design_version"][0]) != config["design_version"]:
        raise ValueError(f"{split_name}.npz design version differs from checkpoint")
    if str(data["ablation"][0]) != config["model"]["ablation"]:
        raise ValueError(f"{split_name}.npz ablation differs from checkpoint")
    if set(data["symbol"].astype(str)) != {config["data"]["symbol"]}:
        raise ValueError(f"{split_name}.npz symbol differs from checkpoint")
    if set(data["series_id"].astype(str)) != {config["data"]["series_id"]}:
        raise ValueError(f"{split_name}.npz series id differs from checkpoint")
    if expected_checkpoint_sha256 is not None and (
        "checkpoint_sha256" not in data
        or str(data["checkpoint_sha256"][0]) != expected_checkpoint_sha256
    ):
        raise ValueError(f"{split_name}.npz checkpoint hash differs")
    if expected_source_sha256 is not None and (
        "source_sha256" not in data
        or str(data["source_sha256"][0]) != expected_source_sha256
    ):
        raise ValueError(f"{split_name}.npz source hash differs")


def evaluate_validation_exports(
    input_dir: str | Path,
    config: dict[str, Any],
    expected_checkpoint_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Run only the three frozen H64 Validation gates; never load Test."""

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
    }
    train = _load(
        directory / "train.npz",
        metadata | {"z_market", "outcomes_h64", "raw_features"},
    )
    validation = _load(
        directory / "validation.npz",
        metadata
        | {
            "z_market",
            "z_target_h64",
            "z_prediction_h64",
            "z_persistence_h64",
            "outcomes_h64",
            "raw_features",
        },
    )
    for split_name, data in (("train", train), ("validation", validation)):
        _validate_export(
            split_name,
            data,
            config,
            expected_checkpoint_sha256,
            expected_source_sha256,
        )

    horizon = 64
    k = 50
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["evaluation"]["seed"])
    if horizon not in config["data"]["horizons"]:
        raise ValueError("Validation protocol requires H64")
    if k not in config["evaluation"]["ks"]:
        raise ValueError("Validation protocol requires K=50")
    if float(config["evaluation"]["ridge_alpha"]) != 1.0:
        raise ValueError("Validation protocol requires Ridge alpha=1.0")
    if bootstrap_samples != 10_000 or seed != 42:
        raise ValueError("Validation protocol requires 10000 bootstraps with seed 42")

    target = validation[f"z_target_h{horizon}"]
    learned_error = cosine_error(validation[f"z_prediction_h{horizon}"], target)
    persistence_error = cosine_error(validation[f"z_persistence_h{horizon}"], target)
    prediction_bootstrap = block_bootstrap(
        persistence_error - learned_error,
        validation["trading_day_ns"],
        bootstrap_samples,
        seed,
    )

    neighbor_order = nearest_indices(
        train["z_market"],
        validation["z_market"],
        train["timestamp_ns"],
        validation["timestamp_ns"],
        k,
    )
    knn = _knn_report(
        train,
        validation,
        horizon,
        [k],
        neighbor_order,
        bootstrap_samples,
        seed,
        bootstrap_seed=seed,
    )[str(k)]
    probe = _probe_report(
        train,
        validation,
        horizon,
        float(config["evaluation"]["ridge_alpha"]),
        bootstrap_samples,
        seed,
    )

    prediction_pass = prediction_bootstrap["ci95_low"] > 0
    knn_pass = knn["improvement"]["ci95_low"] > 0
    probe_pass = probe["normalized_mse_improvement"]["ci95_low"] > 0
    return _strip_arrays(
        {
            "design_version": "0.6.2",
            "training_design_version": str(train["design_version"][0]),
            "test_consumed": False,
            "ranges": {
                "train": _date_range(train),
                "validation": _date_range(validation),
            },
            "prediction": {
                "horizon": horizon,
                "jepa_error": float(learned_error.mean()),
                "persistence_error": float(persistence_error.mean()),
                "effect": prediction_bootstrap["effect"],
                "ci95": [
                    prediction_bootstrap["ci95_low"],
                    prediction_bootstrap["ci95_high"],
                ],
                "pass": prediction_pass,
            },
            "knn": {
                "horizon": horizon,
                "k": k,
                "latent_error": knn["nearest_outcome_mse"],
                "random_error": knn["random_outcome_mse"],
                "effect": knn["improvement"]["effect"],
                "ci95": [
                    knn["improvement"]["ci95_low"],
                    knn["improvement"]["ci95_high"],
                ],
                "pass": knn_pass,
            },
            "probe": {
                "horizon": horizon,
                "targets": ["return", "realized_volatility"],
                "latent_mse": probe["latent_normalized_mse"],
                "raw_mse": probe["raw_normalized_mse"],
                "effect": probe["normalized_mse_improvement"]["effect"],
                "ci95": [
                    probe["normalized_mse_improvement"]["ci95_low"],
                    probe["normalized_mse_improvement"]["ci95_high"],
                ],
                "pass": probe_pass,
            },
            "validation_go": prediction_pass and knn_pass and probe_pass,
        }
    )


def evaluate_exports(
    input_dir: str | Path,
    config: dict[str, Any],
    expected_checkpoint_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    directory = Path(input_dir)
    splits = {name: _load(directory / f"{name}.npz") for name in ("train", "validation", "test")}
    for split_name, data in splits.items():
        _validate_export(
            split_name,
            data,
            config,
            expected_checkpoint_sha256,
            expected_source_sha256,
        )
    train = splits["train"]
    horizons = sorted(
        int(name.removeprefix("outcomes_h")) for name in train if name.startswith("outcomes_h")
    )
    report: dict[str, Any] = {
        "design_version": str(train["design_version"][0]),
        "ablation": str(train["ablation"][0]),
        "ranges": {name: _date_range(data) for name, data in splits.items()},
        "future_prediction": {},
        "knn": {},
        "linear_probe": {},
    }
    bootstrap_samples = config["evaluation"]["bootstrap_samples"]
    seed = config["evaluation"]["seed"]
    for split_name, data in splits.items():
        report["future_prediction"][split_name] = {
            str(horizon): _future_report(train, data, horizon, bootstrap_samples, seed + horizon)
            for horizon in horizons
        }
    for split_index, split_name in enumerate(("validation", "test")):
        data = splits[split_name]
        valid_ks = [k for k in config["evaluation"]["ks"] if k <= len(train["z_market"])]
        if not valid_ks:
            raise ValueError("no configured k fits the train reference set")
        # Neighbor identities depend only on Z_market, not on the outcome
        # horizon. Compute max-K once per split and reuse its ordered prefixes.
        neighbor_order = nearest_indices(
            train["z_market"],
            data["z_market"],
            train["timestamp_ns"],
            data["timestamp_ns"],
            max(valid_ks),
        )
        report["knn"][split_name] = {
            str(horizon): _knn_report(
                train,
                data,
                horizon,
                valid_ks,
                neighbor_order,
                bootstrap_samples,
                seed + split_index * 1000 + horizon,
            )
            for horizon in horizons
        }
        probe_horizons = [16, 64] if {16, 64}.issubset(horizons) else horizons[:2]
        report["linear_probe"][split_name] = {
            str(horizon): _probe_report(
                train,
                data,
                horizon,
                config["evaluation"]["ridge_alpha"],
                bootstrap_samples,
                seed + split_index * 1000 + horizon,
            )
            for horizon in probe_horizons
        }

    primary_horizon = 64 if 64 in horizons else horizons[len(horizons) // 2]
    available_ks = [k for k in config["evaluation"]["ks"] if str(k) in report["knn"]["validation"][str(primary_horizon)]]
    primary_k = 50 if 50 in available_ks else max(available_ks)
    validation_effects = {
        "prediction": report["future_prediction"]["validation"][str(primary_horizon)]["persistence_improvement"],
        "knn": report["knn"]["validation"][str(primary_horizon)][str(primary_k)]["improvement"],
        "probe": report["linear_probe"]["validation"][str(primary_horizon)]["normalized_mse_improvement"],
    }
    test_effects = {
        "prediction": report["future_prediction"]["test"][str(primary_horizon)]["persistence_improvement"],
        "knn": report["knn"]["test"][str(primary_horizon)][str(primary_k)]["improvement"],
        "probe": report["linear_probe"]["test"][str(primary_horizon)]["normalized_mse_improvement"],
    }
    validation_pass = all(value["ci95_low"] > 0 for value in validation_effects.values())
    test_direction = all(value["effect"] > 0 for value in test_effects.values())
    report["go_no_go"] = {
        "primary_horizon": primary_horizon,
        "primary_k": primary_k,
        "validation": validation_effects,
        "test": test_effects,
        "decision": "GO" if validation_pass and test_direction else "NO-GO/INCONCLUSIVE",
    }
    return _strip_arrays(report)


def save_evaluation(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
