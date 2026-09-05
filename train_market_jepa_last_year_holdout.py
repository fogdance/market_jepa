from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.data import (
    CONTEXT_FEATURES,
    MARKET_FEATURES,
    MarketDataset,
    build_split_indices,
    collate_market_batch,
    prepare_market_data,
    run_preflight,
)
from market_jepa.data.preflight import file_sha256
from market_jepa.eval.metrics import block_bootstrap, cosine_error
from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.model import MarketJEPA
from market_jepa.train import Trainer, configure_determinism, load_checkpoint
from market_jepa.train.checkpoint import sha256
from v0_jepa_price_oi_volume_audit import (
    OI_FEATURES,
    VOLUME_FEATURES,
    _donor_frame,
    _matching_quality,
    _prediction_errors,
    _sample_indices,
    marginal_summaries,
    match_donors,
)


EXPERIMENT_ID = "market_jepa_last_year_holdout_2025"
CONFIG_PATH = Path("configs/market_jepa_last_year_holdout_2025.yaml")
OUTPUT_DIR = Path("artifacts/evaluation/jepa_last_year_holdout_2025")
CHECKPOINT_DIR = Path("artifacts/checkpoints") / EXPERIMENT_ID
HORIZONS = (16, 64, 256)
SCOPES: tuple[int | str, ...] = (*HORIZONS, "avg")
STRUCTURAL_EFFECTS = {
    "oi_use": ("no_oi", "original"),
    "volume_use": ("no_volume", "original"),
    "ov_use": ("price_only", "original"),
    "price_x_ov": ("price_ov_relation_broken", "original"),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def expected_config() -> dict[str, Any]:
    expected = deepcopy(DEFAULT_CONFIG)
    expected["experiment_id"] = EXPERIMENT_ID
    expected["data"]["splits"] = {
        "train": ["2018-01-02", "2024-12-31"],
        "validation": ["2025-01-01", "2025-06-30"],
        "test": ["2025-07-01", "2025-12-02"],
    }
    expected["evaluation"]["output_dir"] = str(OUTPUT_DIR)
    expected["benchmark"] = {
        "checkpoint_selection": "fixed_budget_final",
        "evaluate_validation_during_training": False,
        "structural_samples_per_split": 8192,
        "structural_sample_seed": 4501,
        "structural_batch_size": 64,
        "bootstrap_block": "iso_trading_week",
        "test_authorized": True,
    }
    return expected


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config != expected_config():
        raise ValueError("last-year holdout config differs from the frozen protocol")
    return config


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def split_integrity(data: Any, config: dict[str, Any]) -> dict[str, Any]:
    frame = data.minute
    days = frame["trading_day"].to_numpy(dtype="datetime64[ns]")
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
    horizon = max(config["data"]["horizons"])
    report: dict[str, Any] = {}
    populations: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        start, end = config["data"]["splits"][split]
        indices = build_split_indices(data, config, split)
        if len(indices) == 0:
            raise AssertionError(f"{split} has no eligible anchors")
        start64 = np.datetime64(start)
        end64 = np.datetime64(end)
        target_indices = indices + horizon
        if days[indices].min() < start64 or days[indices].max() > end64:
            raise AssertionError(f"{split} anchor trading-day boundary failed")
        if days[target_indices].max() > end64:
            raise AssertionError(f"{split} future trading-day boundary failed")
        if timestamps[target_indices].max() >= end64 + np.timedelta64(1, "D"):
            raise AssertionError(f"{split} future timestamp boundary failed")
        populations[split] = indices
        report[split] = {
            "samples": len(indices),
            "anchor_min_trading_day": str(days[indices].min().astype("datetime64[D]")),
            "anchor_max_trading_day": str(days[indices].max().astype("datetime64[D]")),
            "target_max_trading_day": str(days[target_indices].max().astype("datetime64[D]")),
            "target_max_timestamp": str(timestamps[target_indices].max()),
        }
    if np.intersect1d(populations["validation"], populations["test"]).size:
        raise AssertionError("Validation and Test populations overlap")
    return report


def prepare_nested_data(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    fit_range = tuple(config["data"]["splits"]["train"])
    data = prepare_market_data(
        config,
        normalizer_fit_range=fit_range,
        max_trading_day="2025-12-02",
    )
    end = pd.Timestamp(fit_range[1])
    minute_fit_mask = data.minute["trading_day"].between(*map(pd.Timestamp, fit_range))
    boundary_index = int(np.flatnonzero(data.minute["trading_day"].le(end).to_numpy())[-1])
    completed_sources: dict[str, int] = {}
    for name, frame in (
        ("daily", data.daily_completed),
        ("weekly", data.weekly_completed),
    ):
        selected = frame[frame["trading_day"].between(*map(pd.Timestamp, fit_range))]
        maximum_source = int(selected["source_max_index"].max())
        if maximum_source > boundary_index:
            raise AssertionError(f"2025 source entered {name} normalizer fit population")
        completed_sources[name] = maximum_source
    metadata = {
        "normalizer_fit_range": list(fit_range),
        "normalizer_minute_rows": int(minute_fit_mask.sum()),
        "normalizer_source_boundary_index": boundary_index,
        "completed_source_max_indices": completed_sources,
        "train_eligible_anchors": len(build_split_indices(data, config, "train")),
        "normalizer_sha256": hashlib.sha256(
            json.dumps(data.normalizers.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return data, metadata


def _prediction_result(
    model: MarketJEPA,
    dataset: MarketDataset,
    device: torch.device,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, dict[int | str, np.ndarray]]]:
    loader = DataLoader(
        dataset,
        batch_size=dataset.config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_market_batch,
        pin_memory=device.type == "cuda",
    )
    errors: dict[str, dict[int | str, list[np.ndarray]]] = {
        "jepa": {horizon: [] for horizon in HORIZONS},
        "persistence": {horizon: [] for horizon in HORIZONS},
    }
    model.eval()
    with torch.inference_mode():
        for batch_number, host_batch in enumerate(loader, start=1):
            batch = _move(host_batch, device)
            output = model(batch)
            persistence = model.encode_persistence(batch)
            for horizon in HORIZONS:
                target = output["targets"][horizon].float().cpu().numpy()
                prediction = output["predictions"][horizon].float().cpu().numpy()
                persisted = persistence[horizon].float().cpu().numpy()
                errors["jepa"][horizon].append(cosine_error(prediction, target))
                errors["persistence"][horizon].append(cosine_error(persisted, target))
            if batch_number == 1 or batch_number % 256 == 0 or batch_number == len(loader):
                print(
                    json.dumps(
                        {
                            "phase": "prediction_evaluation",
                            "split": dataset.split,
                            "batches_complete": batch_number,
                            "batches_total": len(loader),
                        }
                    ),
                    flush=True,
                )
    final: dict[str, dict[int | str, np.ndarray]] = {"jepa": {}, "persistence": {}}
    for kind in final:
        for horizon in HORIZONS:
            final[kind][horizon] = np.concatenate(errors[kind][horizon])
        final[kind]["avg"] = np.column_stack(
            [final[kind][horizon] for horizon in HORIZONS]
        ).mean(axis=1)
    blocks = dataset.data.minute.iloc[dataset.indices]["iso_key"].to_numpy(dtype=np.int64)
    scopes: dict[str, Any] = {}
    for scope in SCOPES:
        effect = final["persistence"][scope] - final["jepa"][scope]
        bootstrap = block_bootstrap(effect, blocks, bootstrap_samples, bootstrap_seed)
        scopes[str(scope).upper()] = {
            "jepa_error": float(final["jepa"][scope].mean()),
            "persistence_error": float(final["persistence"][scope].mean()),
            "effect": bootstrap["effect"],
            "ci95": [bootstrap["ci95_low"], bootstrap["ci95_high"]],
            "n": len(effect),
            "weeks": bootstrap["blocks"],
        }
    return {"split": dataset.split, "scopes": scopes}, final


def _structural_result(
    model: MarketJEPA,
    dataset: MarketDataset,
    device: torch.device,
    config: dict[str, Any],
    sampled: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    population = dataset.indices
    population_summary, names = marginal_summaries(dataset.data.minute_market, population)
    base_summary, base_names = marginal_summaries(dataset.data.minute_market, sampled)
    if names != base_names:
        raise AssertionError("donor summary schema drift")
    donors = match_donors(
        dataset.data.minute,
        population,
        sampled,
        population_summary,
        base_summary,
        int(config["benchmark"]["structural_sample_seed"]),
    )
    if not np.isin(donors.joint, population).all():
        raise AssertionError(f"{dataset.split} donor escaped its population")
    sampled_dataset = MarketDataset(dataset.data, config, dataset.split, sampled)
    errors = _prediction_errors(
        model,
        sampled_dataset,
        donors,
        device,
        int(config["benchmark"]["structural_batch_size"]),
    )
    blocks = dataset.data.minute.iloc[sampled]["iso_key"].to_numpy(dtype=np.int64)
    effects: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for effect_name, (left, right) in STRUCTURAL_EFFECTS.items():
        effects[effect_name] = {}
        for scope in SCOPES:
            paired = errors[left][scope] - errors[right][scope]
            bootstrap = block_bootstrap(
                paired,
                blocks,
                int(config["evaluation"]["bootstrap_samples"]),
                int(config["evaluation"]["seed"]),
            )
            value = {
                "effect": bootstrap["effect"],
                "ci95": [bootstrap["ci95_low"], bootstrap["ci95_high"]],
                "n": len(paired),
                "weeks": bootstrap["blocks"],
                "significant_positive": bootstrap["ci95_low"] > 0,
            }
            effects[effect_name][str(scope).upper()] = value
            rows.append(
                {
                    "population": dataset.split,
                    "metric_family": "structural",
                    "effect_name": effect_name,
                    "horizon": str(scope).upper(),
                    **{key: item for key, item in value.items() if key != "ci95"},
                    "ci95_low": value["ci95"][0],
                    "ci95_high": value["ci95"][1],
                }
            )
    quality = _matching_quality(
        dataset.split,
        names,
        base_summary,
        population_summary,
        population,
        donors,
    )
    matching = {
        "median_matched_distance": float(np.median(donors.joint_distance)),
        "p90_matched_distance": float(np.quantile(donors.joint_distance, 0.9)),
        "median_random_distance": float(np.median(donors.random_distance)),
        "p90_random_distance": float(np.quantile(donors.random_distance, 0.9)),
        "matched_better_than_random": bool(
            np.median(donors.joint_distance) < np.median(donors.random_distance)
        ),
    }
    condition_errors = {
        condition: {
            str(scope).upper(): float(errors[condition][scope].mean()) for scope in SCOPES
        }
        for condition in ("original", "no_oi", "no_volume", "price_only", "price_ov_relation_broken")
    }
    report = {
        "split": dataset.split,
        "samples": len(sampled),
        "population_samples": len(population),
        "effects": effects,
        "condition_mean_errors": condition_errors,
        "matching": matching,
    }
    donor_frame = _donor_frame(dataset.split, dataset.data.minute, sampled, donors)
    return report, donor_frame, quality, rows


def _prediction_bootstrap_rows(split: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "population": split,
            "metric_family": "prediction",
            "effect_name": "jepa_vs_persistence",
            "horizon": scope,
            "effect": values["effect"],
            "ci95_low": values["ci95"][0],
            "ci95_high": values["ci95"][1],
            "n": values["n"],
            "weeks": values["weeks"],
            "significant_positive": values["ci95"][0] > 0,
        }
        for scope, values in report["scopes"].items()
    ]


def _classify(test_prediction: dict[str, Any], test_structural: dict[str, Any]) -> dict[str, Any]:
    prediction_pass = test_prediction["scopes"]["AVG"]["ci95"][0] > 0
    structural_positive = {
        name: values["AVG"]["effect"] > 0
        for name, values in test_structural["effects"].items()
    }
    if not prediction_pass:
        code = "A"
        text = "JEPA无法在最后一年保持future prediction能力。"
    elif not all(structural_positive[name] for name in ("oi_use", "volume_use", "ov_use")):
        code = "B"
        text = "JEPA在最后一年仍有prediction能力，但OI/Volume structural information没有稳定保留。"
    elif not structural_positive["price_x_ov"]:
        code = "C"
        text = "JEPA在最后一年仍利用OI/Volume，但Price×(OI,Volume)联合关系没有稳定泛化。"
    else:
        code = "D"
        text = "JEPA在最后一年仍保持prediction能力，同时OI/Volume以及Price×(OI,Volume)联合结构也能泛化到完全未见的2025。"
    return {
        "code": code,
        "text": text,
        "prediction_pass": prediction_pass,
        "structural_point_estimates_positive": structural_positive,
        "structural_category_rule": "point_estimate_gt_zero_as_frozen_in_section_16",
    }


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    validation_prediction = summary["prediction"]["validation"]["scopes"]
    test_prediction = summary["prediction"]["test"]["scopes"]
    validation_structural = summary["structural"]["validation"]["effects"]
    test_structural = summary["structural"]["test"]["effects"]
    lines = [
        "# JEPA Last-Year Holdout Benchmark — 2018–2024 Train / 2025 Validation+Test",
        "",
        f"- Formal checkpoint: epoch {summary['checkpoint']['epoch']} `last.pt`",
        f"- Checkpoint SHA-256: `{summary['checkpoint']['sha256']}`",
        "- Checkpoint selection: fixed 50-epoch endpoint; Validation was not evaluated during training",
        "- Test consumed: `true`",
        "",
        "## Prediction: JEPA vs Persistence",
        "",
        "| Split | Scope | JEPA error | Persistence error | Effect | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in (("Validation 2025H1", validation_prediction), ("Test 2025H2", test_prediction)):
        for scope in ("H16", "H64", "H256", "AVG"):
            item = values[scope]
            lines.append(
                f"| {split} | {scope} | {item['jepa_error']:.8g} | {item['persistence_error']:.8g} | "
                f"{item['effect']:.8g} | [{item['ci95'][0]:.8g}, {item['ci95'][1]:.8g}] |"
            )
    lines.extend(
        [
            "",
            "## Structural effects",
            "",
            "Positive means the intervention worsened prediction.",
            "",
            "| Split | Effect | Scope | Estimate | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split, values in (("Validation 2025H1", validation_structural), ("Test 2025H2", test_structural)):
        for effect_name in ("oi_use", "volume_use", "ov_use", "price_x_ov"):
            for scope in ("H16", "H64", "H256", "AVG"):
                item = values[effect_name][scope]
                lines.append(
                    f"| {split} | {effect_name} | {scope} | {item['effect']:.8g} | "
                    f"[{item['ci95'][0]:.8g}, {item['ci95'][1]:.8g}] |"
                )
    lines.extend(["", "## Required answers", ""])
    vp = validation_prediction["AVG"]
    tp = test_prediction["AVG"]
    lines.append(
        f"1. 2025H1 prediction effect={vp['effect']:.8g}, CI={vp['ci95']}; "
        + ("PASS." if vp["ci95"][0] > 0 else "FAIL.")
    )
    lines.append(
        f"2. 2025H2 prediction effect={tp['effect']:.8g}, CI={tp['ci95']}; "
        + ("LAST_YEAR_PREDICTION_PASS." if summary["classification"]["prediction_pass"] else "LAST_YEAR_PREDICTION_FAIL.")
    )
    horizon_status = [
        f"{scope}={'PASS' if test_prediction[scope]['ci95'][0] > 0 else 'FAIL'}"
        for scope in ("H16", "H64", "H256")
    ]
    lines.append("3. 2025H2 horizons: " + ", ".join(horizon_status) + ".")
    for number, effect_name, label in (
        (4, "oi_use", "OI"),
        (5, "volume_use", "Volume"),
        (6, "ov_use", "OI+Volume beyond Price"),
    ):
        value = test_structural[effect_name]["AVG"]
        lines.append(
            f"{number}. {label}: effect={value['effect']:.8g}, CI={value['ci95']}, "
            f"significant_positive={value['significant_positive']}."
        )
    relation = test_structural["price_x_ov"]
    lines.append(
        "7. Price×(OI,Volume) Test effects: "
        + ", ".join(
            f"{scope}={relation[scope]['effect']:.6g} CI={relation[scope]['ci95']}"
            for scope in ("H16", "H64", "H256")
        )
        + "."
    )
    lines.append(
        "8. Validation→Test AVG effect changes: "
        f"prediction={tp['effect'] - vp['effect']:.6g}, "
        + ", ".join(
            f"{name}={test_structural[name]['AVG']['effect'] - validation_structural[name]['AVG']['effect']:.6g}"
            for name in ("oi_use", "volume_use", "ov_use", "price_x_ov")
        )
        + "."
    )
    lines.append(
        f"9. Final Category **{summary['classification']['code']}** — {summary['classification']['text']}"
    )
    lines.append("10. **2025 test_consumed = true**")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _runtime_options(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run(
    config_path: Path,
    device: torch.device,
    runtime_path: Path,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    config = load_benchmark_config(config_path)
    configure_determinism(int(config["training"]["seed"]))
    data, normalizer_metadata = prepare_nested_data(config)
    boundaries = split_integrity(data, config)
    source_hash = file_sha256(config["data"]["csv_path"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "design_version": "0.6.1",
        "config": config,
        "config_sha256": sha256(config_path),
        "source_sha256": source_hash,
        "split_integrity": boundaries,
        "normalizer_fit": normalizer_metadata,
        "checkpoint_selection": "fixed_budget_final_epoch49",
        "validation_during_training": False,
        "test_authorized": True,
        "test_consumed": False,
    }
    _write_json(OUTPUT_DIR / "protocol.json", protocol)

    last_path = CHECKPOINT_DIR / "last.pt"
    best_path = CHECKPOINT_DIR / "best.pt"
    if resume_path is None and last_path.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {last_path}; use --resume")
    if best_path.exists():
        raise RuntimeError("formal fixed-budget run must not contain best.pt")
    report, json_path, csv_path = run_preflight(data, config, Path("artifacts") / EXPERIMENT_ID / "preflight")
    train = MarketDataset(data, config, "train")
    validation = MarketDataset(data, config, "validation")
    model = MarketJEPA(
        len(MARKET_FEATURES),
        len(CONTEXT_FEATURES),
        config["data"]["horizons"],
        config["model"],
    )
    trainer = Trainer(
        model,
        config,
        train,
        validation,
        source_sha256=source_hash,
        preflight_metadata={"json": str(json_path), "csv": str(csv_path), "source": report["source"]},
        device=device,
        runtime_options=_runtime_options(runtime_path),
        checkpoint_selection="fixed_budget_final",
        evaluate_validation_during_training=False,
    )
    if resume_path is not None:
        trainer.resume(load_checkpoint(resume_path))
    history = trainer.fit()
    pd.json_normalize(history, sep=".").to_csv(OUTPUT_DIR / "train_history.csv", index=False)
    if best_path.exists():
        raise AssertionError("best.pt was created despite fixed-budget selection")
    checkpoint = load_checkpoint(last_path, map_location="cpu")
    if checkpoint["epoch"] != 49 or len(checkpoint["history"]) != 50:
        raise AssertionError("formal checkpoint is not the completed epoch49 endpoint")
    if checkpoint["checkpoint_selection"] != "fixed_budget_final":
        raise AssertionError("checkpoint selection metadata drift")

    frozen = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), config["data"]["horizons"], config["model"]
    )
    frozen.load_state_dict(checkpoint["model"])
    frozen.requires_grad_(False).eval().to(device)
    weight_hash = _model_hash(frozen)

    prediction: dict[str, Any] = {}
    structural: dict[str, Any] = {}
    donor_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    sample_rng = np.random.Generator(
        np.random.PCG64(int(config["benchmark"]["structural_sample_seed"]))
    )
    donor_sets: dict[str, set[int]] = {}

    # The frozen order is Validation first, then the same weights are used once for Test.
    for split in ("validation", "test"):
        dataset = MarketDataset(data, config, split)
        prediction[split], _ = _prediction_result(
            frozen,
            dataset,
            device,
            int(config["evaluation"]["bootstrap_samples"]),
            int(config["evaluation"]["seed"]),
        )
        _write_json(OUTPUT_DIR / f"prediction_{split}.json", prediction[split])
        bootstrap_rows.extend(_prediction_bootstrap_rows(split, prediction[split]))
        sampled = _sample_indices(
            dataset.indices,
            int(config["benchmark"]["structural_samples_per_split"]),
            sample_rng,
        )
        structural[split], donor_frame, quality, rows = _structural_result(
            frozen, dataset, device, config, sampled
        )
        _write_json(OUTPUT_DIR / f"structural_{split}.json", structural[split])
        donor_frames.append(donor_frame)
        quality_frames.append(quality)
        bootstrap_rows.extend(rows)
        donor_sets[split] = set()
        for column in (
            "joint_donor_anchor",
            "independent_donor_anchor",
            "random_donor_anchor",
        ):
            donor_sets[split].update(donor_frame[column].astype(int))
        if _model_hash(frozen) != weight_hash:
            raise AssertionError("frozen model weights changed during evaluation")

    validation_population = set(build_split_indices(data, config, "validation").tolist())
    test_population = set(build_split_indices(data, config, "test").tolist())
    if donor_sets["validation"] & test_population or donor_sets["test"] & validation_population:
        raise AssertionError("Validation/Test donor pools overlap")
    pd.concat(donor_frames, ignore_index=True).to_csv(OUTPUT_DIR / "donor_pairs.csv", index=False)
    pd.concat(quality_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "donor_matching_quality.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(OUTPUT_DIR / "bootstrap_effects.csv", index=False)

    classification = _classify(prediction["test"], structural["test"])
    protocol.update(
        {
            "checkpoint": {
                "path": str(last_path),
                "sha256": sha256(last_path),
                "epoch": checkpoint["epoch"],
                "model_state_sha256": weight_hash,
                "implementation_sha256": checkpoint["implementation_sha256"],
            },
            "evaluation_model_state_sha256_before_after": [weight_hash, _model_hash(frozen)],
            "implementation_manifest_sha256_current": manifest_sha256(implementation_manifest()),
            "test_consumed": True,
        }
    )
    _write_json(OUTPUT_DIR / "protocol.json", protocol)
    summary = {
        "experiment": "JEPA Last-Year Holdout Benchmark — 2018–2024 Train / 2025 Validation+Test",
        "checkpoint": protocol["checkpoint"],
        "prediction": prediction,
        "structural": structural,
        "validation_to_test": {
            "prediction_avg_effect_change": prediction["test"]["scopes"]["AVG"]["effect"]
            - prediction["validation"]["scopes"]["AVG"]["effect"],
            "structural_avg_effect_changes": {
                name: structural["test"]["effects"][name]["AVG"]["effect"]
                - structural["validation"]["effects"][name]["AVG"]["effect"]
                for name in STRUCTURAL_EFFECTS
            },
        },
        "classification": classification,
        "test_consumed": True,
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_summary(summary, OUTPUT_DIR / "summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal 2018–2024 → 2025 JEPA holdout benchmark")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--runtime-config", type=Path, default=Path("artifacts/performance/selected_runtime.json"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    summary = run(args.config, torch.device(args.device), args.runtime_config, args.resume)
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "test_consumed": summary["test_consumed"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
