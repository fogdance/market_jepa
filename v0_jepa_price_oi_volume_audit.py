from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from market_jepa.data import (
    CONTEXT_FEATURES,
    MARKET_FEATURES,
    MarketDataset,
    NormalizerBundle,
    build_split_indices,
    collate_market_batch,
    prepare_market_data,
)
from market_jepa.eval.metrics import block_bootstrap, cosine_error
from market_jepa.model import MarketJEPA
from market_jepa.train.checkpoint import load_checkpoint, sha256


CHECKPOINT = Path("artifacts/checkpoints/market_jepa_v0_default/last.pt")
OUTPUT_DIR = Path("artifacts/evaluation/v0_jepa_price_oi_volume_audit")
FORMAL_SAMPLES = 8192
SAMPLE_SEED = 4501
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
MAX_TRADING_DAY = "2024-12-31"
HORIZONS = (16, 64, 256)
LOOKBACKS = (16, 64, 256, 512)

PRICE_FEATURES = (
    "log_open",
    "log_high",
    "log_low",
    "log_close",
    "close_log_return",
    "open_to_prev_close",
    "high_to_prev_close",
    "low_to_prev_close",
    "normalized_range",
    "realized_volatility",
)
VOLUME_FEATURES = ("log1p_volume", "volume_log_change")
OI_FEATURES = ("log1p_open_interest", "open_interest_log_change")

CONDITIONS = (
    "original",
    "no_oi",
    "no_volume",
    "price_only",
    "oi_relation_broken",
    "volume_relation_broken",
    "price_ov_relation_broken",
    "all_relations_broken",
)

EFFECTS = {
    "oi_use": ("no_oi", "original"),
    "volume_use": ("no_volume", "original"),
    "ov_use": ("price_only", "original"),
    "price_x_oi": ("oi_relation_broken", "original"),
    "price_x_volume": ("volume_relation_broken", "original"),
    "price_x_ov": ("price_ov_relation_broken", "original"),
    "oi_x_volume_additional": ("all_relations_broken", "price_ov_relation_broken"),
}


@dataclass(frozen=True)
class Donors:
    joint: np.ndarray
    independent: np.ndarray
    random: np.ndarray
    joint_distance: np.ndarray
    independent_distance: np.ndarray
    random_distance: np.ndarray


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


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _feature_indices(names: tuple[str, ...]) -> list[int]:
    return [MARKET_FEATURES.index(name) for name in names]


def _sample_indices(population: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    population = np.asarray(population, dtype=np.int64)
    if len(population) <= count:
        return population.copy()
    return np.sort(rng.choice(population, size=count, replace=False))


def _window_statistics(
    values: np.ndarray, anchors: np.ndarray, lookbacks: tuple[int, ...], prefix: str,
    operations: tuple[str, ...],
) -> tuple[list[np.ndarray], list[str]]:
    values64 = np.asarray(values, dtype=np.float64)
    total = np.concatenate([[0.0], np.cumsum(values64, dtype=np.float64)])
    total_sq = np.concatenate([[0.0], np.cumsum(values64 * values64, dtype=np.float64)])
    columns: list[np.ndarray] = []
    names: list[str] = []
    stops = anchors + 1
    for lookback in lookbacks:
        starts = stops - lookback
        sums = total[stops] - total[starts]
        means = sums / lookback
        variances = np.maximum((total_sq[stops] - total_sq[starts]) / lookback - means**2, 0.0)
        available = {
            "mean": means,
            "std": np.sqrt(variances),
            "sum": sums,
        }
        for operation in operations:
            columns.append(available[operation])
            names.append(f"{prefix}_{operation}_{lookback}")
    return columns, names


def marginal_summaries(minute_market: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, list[str]]:
    oi_change = minute_market[:, MARKET_FEATURES.index("open_interest_log_change")]
    volume_level = minute_market[:, MARKET_FEATURES.index("log1p_volume")]
    volume_change = minute_market[:, MARKET_FEATURES.index("volume_log_change")]
    columns: list[np.ndarray] = []
    names: list[str] = []
    for values, prefix, operations in (
        (oi_change, "oi_change", ("mean", "std", "sum")),
        (volume_level, "volume_level", ("mean", "std")),
        (volume_change, "volume_change", ("mean", "std")),
    ):
        new_columns, new_names = _window_statistics(values, anchors, LOOKBACKS, prefix, operations)
        columns.extend(new_columns)
        names.extend(new_names)
    columns.append(minute_market[anchors, MARKET_FEATURES.index("log1p_open_interest")])
    names.append("oi_level_current")
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite donor matching summary")
    return result, names


def _temporal_metadata(frame: pd.DataFrame, anchors: np.ndarray) -> tuple[np.ndarray, ...]:
    rows = frame.iloc[anchors]
    timestamps = rows["timestamp"]
    weekday = rows["trading_day"].dt.weekday.to_numpy(dtype=np.int8)
    night = (timestamps.dt.hour.to_numpy(dtype=np.int16) >= 18)
    minute_of_day = (
        timestamps.dt.hour.to_numpy(dtype=np.int16) * 60
        + timestamps.dt.minute.to_numpy(dtype=np.int16)
    )
    iso_week = rows["iso_key"].to_numpy(dtype=np.int64)
    return weekday, night, minute_of_day, iso_week


def match_donors(
    frame: pd.DataFrame,
    population: np.ndarray,
    bases: np.ndarray,
    population_summary: np.ndarray,
    base_summary: np.ndarray,
    seed: int,
) -> Donors:
    mean = population_summary.mean(axis=0, dtype=np.float64)
    std = population_summary.std(axis=0, dtype=np.float64)
    std[std < 1e-12] = 1.0
    candidates_z = (population_summary - mean) / std
    bases_z = (base_summary - mean) / std
    candidate_weekday, candidate_night, candidate_tod, candidate_week = _temporal_metadata(
        frame, population
    )
    base_weekday, base_night, base_tod, base_week = _temporal_metadata(frame, bases)
    groups: dict[tuple[int, bool], np.ndarray] = {}
    for weekday in range(5):
        for night in (False, True):
            groups[(weekday, night)] = np.flatnonzero(
                (candidate_weekday == weekday) & (candidate_night == night)
            )

    joint = np.empty(len(bases), dtype=np.int64)
    independent = np.empty(len(bases), dtype=np.int64)
    random = np.empty(len(bases), dtype=np.int64)
    joint_distance = np.empty(len(bases), dtype=np.float64)
    independent_distance = np.empty(len(bases), dtype=np.float64)
    random_distance = np.empty(len(bases), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))

    for row in range(len(bases)):
        group = groups[(int(base_weekday[row]), bool(base_night[row]))]
        eligible = group[
            (np.abs(candidate_tod[group] - base_tod[row]) <= 30)
            & (candidate_week[group] != base_week[row])
        ]
        if len(eligible) < 2:
            raise RuntimeError(f"no eligible donor pair for anchor {int(bases[row])}")
        squared = np.sum((candidates_z[eligible] - bases_z[row]) ** 2, axis=1)
        # population/build_split_indices are sorted; argmin therefore resolves exact ties
        # to the smaller Train/Development anchor index.
        joint_position = int(np.argmin(squared))
        j_candidate_position = int(eligible[joint_position])
        j_week = candidate_week[j_candidate_position]

        eligible_k_mask = candidate_week[eligible] != j_week
        eligible_k = eligible[eligible_k_mask]
        squared_k = squared[eligible_k_mask]
        if len(eligible_k) == 0:
            raise RuntimeError(f"no independent-week donor for anchor {int(bases[row])}")
        k_position = int(np.argmin(squared_k))
        k_candidate_position = int(eligible_k[k_position])
        random_candidate_position = int(eligible[int(rng.integers(0, len(eligible)))])

        joint[row] = population[j_candidate_position]
        independent[row] = population[k_candidate_position]
        random[row] = population[random_candidate_position]
        joint_distance[row] = math.sqrt(float(squared[joint_position]))
        independent_distance[row] = math.sqrt(float(squared_k[k_position]))
        random_distance[row] = float(np.linalg.norm(candidates_z[random_candidate_position] - bases_z[row]))

    if np.any(joint == independent):
        raise AssertionError("joint and independent donor anchors overlap")
    joint_weekday, joint_night, joint_tod, joint_week = _temporal_metadata(frame, joint)
    independent_weekday, independent_night, independent_tod, independent_week = _temporal_metadata(
        frame, independent
    )
    for donor_weekday, donor_night, donor_tod, label in (
        (joint_weekday, joint_night, joint_tod, "joint"),
        (independent_weekday, independent_night, independent_tod, "independent"),
    ):
        if np.any(donor_weekday != base_weekday):
            raise AssertionError(f"{label} donor weekday mismatch")
        if np.any(donor_night != base_night):
            raise AssertionError(f"{label} donor session mismatch")
        if np.any(np.abs(donor_tod - base_tod) > 30):
            raise AssertionError(f"{label} donor time-of-day mismatch")
    if np.any(joint_week == base_week) or np.any(independent_week == base_week):
        raise AssertionError("donors must differ from the base ISO week")
    if np.any(joint_week == independent_week):
        raise AssertionError("joint and independent donors must come from different ISO weeks")
    return Donors(joint, independent, random, joint_distance, independent_distance, random_distance)


def _zero_features(value: torch.Tensor, indices: list[int]) -> torch.Tensor:
    result = value.clone()
    result[..., indices] = 0.0
    return result


def relation_interventions(
    original: torch.Tensor,
    joint_paths: torch.Tensor,
    independent_paths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    oi = _feature_indices(OI_FEATURES)
    volume = _feature_indices(VOLUME_FEATURES)
    oi_broken = original.clone()
    oi_broken[..., oi] = joint_paths[..., oi]
    volume_broken = original.clone()
    volume_broken[..., volume] = joint_paths[..., volume]
    joint_broken = original.clone()
    joint_broken[..., oi] = joint_paths[..., oi]
    joint_broken[..., volume] = joint_paths[..., volume]
    all_broken = original.clone()
    all_broken[..., oi] = joint_paths[..., oi]
    all_broken[..., volume] = independent_paths[..., volume]
    return {
        "oi_relation_broken": oi_broken,
        "volume_relation_broken": volume_broken,
        "price_ov_relation_broken": joint_broken,
        "all_relations_broken": all_broken,
    }


def _shared_components(model: MarketJEPA, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    online = model.online
    minute = online.minute_market(batch["minute_market"])
    context = online.minute_context(batch["minute_context"])
    daily = online.daily(batch["daily"], batch["daily_lengths"])
    weekly = online.weekly(batch["weekly"], batch["weekly_lengths"])
    return minute, context, daily, weekly


def _predict_from_components(
    model: MarketJEPA,
    minute: torch.Tensor,
    context: torch.Tensor,
    daily: torch.Tensor,
    weekly: torch.Tensor,
) -> dict[int, torch.Tensor]:
    z_market = model.online.fusion(torch.cat([minute, context, daily, weekly], dim=-1))
    return {horizon: model.predictors[str(horizon)](z_market) for horizon in HORIZONS}


def _predict_removal(
    model: MarketJEPA,
    batch: dict[str, Any],
    indices: list[int],
    context: torch.Tensor,
) -> dict[int, torch.Tensor]:
    online = model.online
    minute = online.minute_market(_zero_features(batch["minute_market"], indices))
    daily = online.daily(_zero_features(batch["daily"], indices), batch["daily_lengths"])
    weekly = online.weekly(_zero_features(batch["weekly"], indices), batch["weekly_lengths"])
    return _predict_from_components(model, minute, context, daily, weekly)


def _prediction_errors(
    model: MarketJEPA,
    dataset: MarketDataset,
    donors: Donors,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[int | str, np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_market_batch,
        pin_memory=device.type == "cuda",
    )
    result: dict[str, dict[int | str, list[np.ndarray]]] = {
        condition: {horizon: [] for horizon in HORIZONS} for condition in CONDITIONS
    }
    oi = _feature_indices(OI_FEATURES)
    volume = _feature_indices(VOLUME_FEATURES)
    ov = sorted(oi + volume)
    context_length = dataset.context_length
    model.eval()
    checked_decomposition = False
    offset = 0
    started = time.monotonic()

    with torch.inference_mode():
        for batch_number, host_batch in enumerate(loader, start=1):
            size = len(host_batch["anchor_index"])
            joint_paths = np.stack(
                [
                    dataset.data.minute_market[anchor - context_length + 1 : anchor + 1]
                    for anchor in donors.joint[offset : offset + size]
                ]
            )
            independent_paths = np.stack(
                [
                    dataset.data.minute_market[anchor - context_length + 1 : anchor + 1]
                    for anchor in donors.independent[offset : offset + size]
                ]
            )
            batch = _move(host_batch, device)
            joint_tensor = torch.from_numpy(joint_paths).to(device, non_blocking=device.type == "cuda")
            independent_tensor = torch.from_numpy(independent_paths).to(
                device, non_blocking=device.type == "cuda"
            )
            # Formal V0 latent export/evaluation runs eval-mode FP32 forward and
            # applies eval.metrics.cosine_error afterwards. Keep that exact path.
            minute, context, daily, weekly = _shared_components(model, batch)
            predictions: dict[str, dict[int, torch.Tensor]] = {
                "original": _predict_from_components(model, minute, context, daily, weekly),
                "no_oi": _predict_removal(model, batch, oi, context),
                "no_volume": _predict_removal(model, batch, volume, context),
                "price_only": _predict_removal(model, batch, ov, context),
            }
            for name, altered_minute in relation_interventions(
                batch["minute_market"], joint_tensor, independent_tensor
            ).items():
                altered_encoding = model.online.minute_market(altered_minute)
                predictions[name] = _predict_from_components(
                    model, altered_encoding, context, daily, weekly
                )
            targets = {
                horizon: model.target_minute(batch["targets"][horizon])
                for horizon in HORIZONS
            }
            if not checked_decomposition:
                reference = model(batch)
                for horizon in HORIZONS:
                    torch.testing.assert_close(
                        predictions["original"][horizon],
                        reference["predictions"][horizon],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    torch.testing.assert_close(
                        targets[horizon],
                        reference["targets"][horizon],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                checked_decomposition = True

            target_arrays = {
                horizon: targets[horizon].float().cpu().numpy() for horizon in HORIZONS
            }
            for condition in CONDITIONS:
                for horizon in HORIZONS:
                    prediction = predictions[condition][horizon].float().cpu().numpy()
                    result[condition][horizon].append(cosine_error(prediction, target_arrays[horizon]))
            offset += size
            if batch_number == 1 or batch_number % 16 == 0 or offset == len(dataset):
                elapsed = time.monotonic() - started
                rate = offset / max(elapsed, 1e-9)
                remaining = (len(dataset) - offset) / max(rate, 1e-9)
                print(
                    json.dumps(
                        {
                            "phase": "inference",
                            "split": dataset.split,
                            "completed": offset,
                            "total": len(dataset),
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": round(remaining, 1),
                        }
                    ),
                    flush=True,
                )

    final: dict[str, dict[int | str, np.ndarray]] = {}
    for condition, horizon_values in result.items():
        final[condition] = {
            horizon: np.concatenate(horizon_values[horizon]) for horizon in HORIZONS
        }
        final[condition]["avg"] = np.mean(
            np.column_stack([final[condition][horizon] for horizon in HORIZONS]), axis=1
        )
    return final


def _matching_quality(
    population_name: str,
    names: list[str],
    base_summary: np.ndarray,
    population_summary: np.ndarray,
    population: np.ndarray,
    donors: Donors,
) -> pd.DataFrame:
    mean = population_summary.mean(axis=0, dtype=np.float64)
    std = population_summary.std(axis=0, dtype=np.float64)
    std[std < 1e-12] = 1.0
    anchor_to_position = {int(anchor): position for position, anchor in enumerate(population)}
    base_z = (base_summary - mean) / std
    rows: list[dict[str, Any]] = []
    for donor_type, donor_anchors in (
        ("joint", donors.joint),
        ("independent", donors.independent),
    ):
        matched = np.stack([population_summary[anchor_to_position[int(a)]] for a in donor_anchors])
        random = np.stack([population_summary[anchor_to_position[int(a)]] for a in donors.random])
        matched_diff = np.abs((matched - mean) / std - base_z)
        random_diff = np.abs((random - mean) / std - base_z)
        for column, name in enumerate(names):
            rows.append(
                {
                    "population": population_name,
                    "donor_type": donor_type,
                    "summary_dimension": name,
                    "matched_median_abs_z_diff": float(np.median(matched_diff[:, column])),
                    "matched_p90_abs_z_diff": float(np.quantile(matched_diff[:, column], 0.9)),
                    "random_median_abs_z_diff": float(np.median(random_diff[:, column])),
                    "random_p90_abs_z_diff": float(np.quantile(random_diff[:, column], 0.9)),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_rows(
    population_name: str,
    errors: dict[str, dict[int | str, np.ndarray]],
    blocks: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for effect_name, (left, right) in EFFECTS.items():
        for horizon in (*HORIZONS, "avg"):
            effect = errors[left][horizon] - errors[right][horizon]
            bootstrap = block_bootstrap(effect, blocks, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            rows.append(
                {
                    "population": population_name,
                    "effect_name": effect_name,
                    "horizon": str(horizon).upper(),
                    "left_condition": left,
                    "right_condition": right,
                    "effect": bootstrap["effect"],
                    "ci95_low": bootstrap["ci95_low"],
                    "ci95_high": bootstrap["ci95_high"],
                    "n": len(effect),
                    "weeks": bootstrap["blocks"],
                }
            )
    return rows


def _per_sample_frame(
    population_name: str,
    frame: pd.DataFrame,
    anchors: np.ndarray,
    errors: dict[str, dict[int | str, np.ndarray]],
) -> pd.DataFrame:
    rows = frame.iloc[anchors]
    payload: dict[str, Any] = {
        "population": np.repeat(population_name, len(anchors)),
        "anchor_index": anchors,
        "timestamp": rows["timestamp"].to_numpy(),
        "trading_day": rows["trading_day"].to_numpy(),
        "iso_week": rows["iso_key"].to_numpy(dtype=np.int64),
    }
    for condition in CONDITIONS:
        for horizon in (*HORIZONS, "avg"):
            payload[f"{condition}_error_{str(horizon).lower()}"] = errors[condition][horizon]
    return pd.DataFrame(payload)


def _donor_frame(
    population_name: str,
    frame: pd.DataFrame,
    bases: np.ndarray,
    donors: Donors,
) -> pd.DataFrame:
    def values(anchors: np.ndarray, column: str) -> np.ndarray:
        return frame.iloc[anchors][column].to_numpy()

    return pd.DataFrame(
        {
            "population": np.repeat(population_name, len(bases)),
            "base_anchor": bases,
            "base_timestamp": values(bases, "timestamp"),
            "base_iso_week": values(bases, "iso_key"),
            "joint_donor_anchor": donors.joint,
            "joint_donor_timestamp": values(donors.joint, "timestamp"),
            "joint_donor_iso_week": values(donors.joint, "iso_key"),
            "joint_matching_distance": donors.joint_distance,
            "independent_donor_anchor": donors.independent,
            "independent_donor_timestamp": values(donors.independent, "timestamp"),
            "independent_donor_iso_week": values(donors.independent, "iso_key"),
            "independent_matching_distance": donors.independent_distance,
            "random_donor_anchor": donors.random,
            "random_matching_distance": donors.random_distance,
        }
    )


def _lookup_effect(bootstrap: pd.DataFrame, population: str, effect: str) -> dict[str, Any]:
    row = bootstrap.loc[
        (bootstrap["population"] == population)
        & (bootstrap["effect_name"] == effect)
        & (bootstrap["horizon"] == "AVG")
    ].iloc[0]
    return {
        "effect": float(row["effect"]),
        "ci95": [float(row["ci95_low"]), float(row["ci95_high"])],
        "positive": bool(row["ci95_low"] > 0),
    }


def _classification(bootstrap: pd.DataFrame) -> tuple[str, str]:
    development_joint = _lookup_effect(bootstrap, "development", "price_x_ov")
    train_joint = _lookup_effect(bootstrap, "train", "price_x_ov")
    development_use = any(
        _lookup_effect(bootstrap, "development", name)["positive"]
        for name in ("oi_use", "volume_use", "ov_use")
    )
    train_use = any(
        _lookup_effect(bootstrap, "train", name)["positive"]
        for name in ("oi_use", "volume_use", "ov_use")
    )
    if development_joint["positive"]:
        return "D", "JEPA learned Price×OI×Volume joint structure that generalizes OOS."
    if train_joint["positive"]:
        return "C", "JEPA learned Price×OI×Volume joint structure in Train, but it did not generalize OOS."
    if development_use or train_use:
        return "B", "JEPA uses OI/Volume information, but this audit finds no significant joint-relation evidence."
    return "A", "This audit finds no significant evidence that JEPA uses OI/Volume information."


def _answer(effect: dict[str, Any], positive: str, negative: str) -> str:
    status = positive if effect["positive"] else negative
    return (
        f"{status} AVG effect={effect['effect']:.8g}, "
        f"95% CI=[{effect['ci95'][0]:.8g}, {effect['ci95'][1]:.8g}]."
    )


def _write_report(summary: dict[str, Any], path: Path) -> None:
    answers = summary["answers"]
    lines = [
        "# V0 JEPA Price × OI × Volume Learned-Structure Audit",
        "",
        f"- Checkpoint: `{summary['checkpoint']['path']}`",
        f"- Epoch: {summary['checkpoint']['epoch']}",
        f"- Dataset SHA-256: `{summary['checkpoint']['source_sha256']}`",
        f"- Fixed samples: Train={summary['samples']['train']}, Development={summary['samples']['development']}",
        f"- Test consumed: `{str(summary['test_consumed']).lower()}`",
        "",
        "## Primary AVG effects",
        "",
        "| Population | Effect | Estimate | 95% CI | Significant positive |",
        "|---|---:|---:|---:|---:|",
    ]
    for population in ("train", "development"):
        for effect_name, values in summary["primary_avg_effects"][population].items():
            lines.append(
                f"| {population} | {effect_name} | {values['effect']:.8g} | "
                f"[{values['ci95'][0]:.8g}, {values['ci95'][1]:.8g}] | {values['positive']} |"
            )
    lines.extend(
        [
            "",
            "## Donor matching sanity check",
            "",
            "| Population | Donor | Median matched distance | P90 matched | Median random | P90 random |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["matching_distance"]:
        lines.append(
            f"| {row['population']} | {row['donor_type']} | {row['matched_median']:.6g} | "
            f"{row['matched_p90']:.6g} | {row['random_median']:.6g} | {row['random_p90']:.6g} |"
        )
    quality_confirmed = all(row["matched_better_than_random"] for row in summary["matching_distance"])
    lines.append("")
    lines.append(
        "Matching-quality check: "
        + ("confirmed; every matched median is below its random median." if quality_confirmed else "not confirmed.")
    )
    lines.extend(["", "## Direct answers", ""])
    for number, answer in enumerate(answers, start=1):
        lines.append(f"{number}. {answer}")
    lines.extend(
        [
            "",
            f"Final classification: **{summary['classification']['code']}** — {summary['classification']['text']}",
            "",
            "This is a fixed diagnostic of the existing V0 JEPA. It does not modify V0 or expand the claim beyond the four requested categories.",
            "",
            "2025 Test was not loaded, exported, or evaluated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(checkpoint_path: Path, output_dir: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("epoch") != 49:
        raise RuntimeError(f"formal audit requires epoch49 last.pt; found epoch={checkpoint.get('epoch')}")
    config = checkpoint["config"]
    if tuple(int(value) for value in config["data"]["horizons"]) != HORIZONS:
        raise RuntimeError("checkpoint horizons do not match formal audit")
    if config["data"]["splits"]["train"] != ["2018-01-02", "2022-12-31"]:
        raise RuntimeError("checkpoint Train split does not match formal audit")
    if config["data"]["splits"]["validation"] != ["2023-01-01", "2024-12-31"]:
        raise RuntimeError("checkpoint Development split does not match formal audit")

    normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
    data = prepare_market_data(config, normalizers=normalizers, max_trading_day=MAX_TRADING_DAY)
    loaded_max = pd.Timestamp(data.minute["trading_day"].max())
    if loaded_max > pd.Timestamp(MAX_TRADING_DAY):
        raise AssertionError("2025 Test data was loaded")
    train_population = build_split_indices(data, config, "train")
    development_population = build_split_indices(data, config, "validation")
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    sampled = {
        "train": _sample_indices(train_population, FORMAL_SAMPLES, rng),
        "development": _sample_indices(development_population, FORMAL_SAMPLES, rng),
    }
    populations = {"train": train_population, "development": development_population}

    model = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), list(HORIZONS), config["model"]
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_errors: dict[str, dict[str, dict[int | str, np.ndarray]]] = {}
    all_donors: dict[str, Donors] = {}
    per_sample_frames: list[pd.DataFrame] = []
    donor_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    summary_names: list[str] | None = None

    for population_name in ("train", "development"):
        print(json.dumps({"phase": "donor_summaries", "population": population_name}), flush=True)
        population_summary, names = marginal_summaries(data.minute_market, populations[population_name])
        base_summary, base_names = marginal_summaries(data.minute_market, sampled[population_name])
        if names != base_names:
            raise AssertionError("matching summary schemas differ")
        summary_names = names
        print(json.dumps({"phase": "donor_matching", "population": population_name}), flush=True)
        donors = match_donors(
            data.minute,
            populations[population_name],
            sampled[population_name],
            population_summary,
            base_summary,
            SAMPLE_SEED,
        )
        all_donors[population_name] = donors
        quality_frames.append(
            _matching_quality(
                population_name,
                names,
                base_summary,
                population_summary,
                populations[population_name],
                donors,
            )
        )
        donor_frames.append(_donor_frame(population_name, data.minute, sampled[population_name], donors))
        split = "train" if population_name == "train" else "validation"
        dataset = MarketDataset(data, config, split, sampled[population_name])
        errors = _prediction_errors(model, dataset, donors, device, batch_size)
        all_errors[population_name] = errors
        per_sample_frames.append(
            _per_sample_frame(population_name, data.minute, sampled[population_name], errors)
        )
        blocks = data.minute.iloc[sampled[population_name]]["iso_key"].to_numpy(dtype=np.int64)
        bootstrap_rows.extend(_bootstrap_rows(population_name, errors, blocks))

    per_sample = pd.concat(per_sample_frames, ignore_index=True)
    donor_pairs = pd.concat(donor_frames, ignore_index=True)
    matching_quality = pd.concat(quality_frames, ignore_index=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    per_sample.to_csv(output_dir / "per_sample_errors.csv", index=False)
    donor_pairs.to_csv(output_dir / "donor_pairs.csv", index=False)
    matching_quality.to_csv(output_dir / "donor_matching_quality.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_effects.csv", index=False)

    primary = {
        population: {
            effect_name: _lookup_effect(bootstrap, population, effect_name)
            for effect_name in EFFECTS
        }
        for population in ("train", "development")
    }
    matching_distance: list[dict[str, Any]] = []
    for population in ("train", "development"):
        donors = all_donors[population]
        for donor_type, matched in (
            ("joint", donors.joint_distance),
            ("independent", donors.independent_distance),
        ):
            matching_distance.append(
                {
                    "population": population,
                    "donor_type": donor_type,
                    "matched_median": float(np.median(matched)),
                    "matched_p90": float(np.quantile(matched, 0.9)),
                    "random_median": float(np.median(donors.random_distance)),
                    "random_p90": float(np.quantile(donors.random_distance, 0.9)),
                    "median_ratio_to_random": float(
                        np.median(matched) / np.median(donors.random_distance)
                    ),
                    "matched_better_than_random": bool(
                        np.median(matched) < np.median(donors.random_distance)
                    ),
                }
            )
    code, classification = _classification(bootstrap)
    development = primary["development"]
    answers = [
        _answer(development["oi_use"], "Yes: removing OI significantly worsened prediction.", "No significant OOS evidence that removing OI worsened prediction."),
        _answer(development["volume_use"], "Yes: removing Volume significantly worsened prediction.", "No significant OOS evidence that removing Volume worsened prediction."),
        _answer(development["ov_use"], "Yes: OI+Volume significantly improved on Price-only history.", "No significant OOS evidence that OI+Volume improved on Price-only history."),
        _answer(development["price_x_oi"], "Yes: breaking Price↔OI significantly worsened prediction.", "No significant OOS harm from breaking Price↔OI."),
        _answer(development["price_x_volume"], "Yes: breaking Price↔Volume significantly worsened prediction.", "No significant OOS harm from breaking Price↔Volume."),
        _answer(development["price_x_ov"], "Yes: breaking Price↔(OI,Volume) while preserving donor OI×Volume significantly worsened prediction.", "No significant OOS harm from the primary matched relation destruction."),
        f"Category {code}: {classification}",
    ]
    condition_means = {
        population: {
            condition: {
                str(horizon).upper(): float(all_errors[population][condition][horizon].mean())
                for horizon in (*HORIZONS, "avg")
            }
            for condition in CONDITIONS
        }
        for population in ("train", "development")
    }
    summary = {
        "audit": "V0 JEPA Price × OI × Volume Learned-Structure Audit",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "epoch": checkpoint["epoch"],
            "design_version": checkpoint.get("design_version"),
            "experiment_id": config.get("experiment_id"),
            "source_sha256": checkpoint["source_sha256"],
            "implementation_sha256": checkpoint.get("implementation_sha256"),
        },
        "samples": {name: len(values) for name, values in sampled.items()},
        "sampling": {
            "seed": SAMPLE_SEED,
            "rng": "NumPy Generator(PCG64), one stream: Train then Development",
            "without_replacement": True,
            "anchor_indices": {name: values.tolist() for name, values in sampled.items()},
        },
        "feature_groups": {
            "price": list(PRICE_FEATURES),
            "volume": list(VOLUME_FEATURES),
            "open_interest": list(OI_FEATURES),
        },
        "donor_summary_names": summary_names,
        "condition_mean_cosine_errors": condition_means,
        "primary_avg_effects": primary,
        "matching_distance": matching_distance,
        "classification": {"code": code, "text": classification},
        "answers": answers,
        "max_loaded_trading_day": str(loaded_max.date()),
        "test_consumed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(summary, output_dir / "summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit learned Price×OI×Volume structure in V0 JEPA")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    summary = run(args.checkpoint, args.output_dir, torch.device(args.device), args.batch_size)
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "test_consumed": summary["test_consumed"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
