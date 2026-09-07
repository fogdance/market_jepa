from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from market_jepa.implementation import manifest_sha256
from market_jepa.train.checkpoint import sha256
from market_jepa.train.trainer import configure_determinism

from .checkpoint import (
    load_v11_checkpoint, model_from_checkpoint, v11_implementation_manifest,
)
from .config import PROFILE_TRAINING, load_v11_config, model_size_from_config, validate_v11_config
from .dataset import V11ContractDataset, V11DataStore, fit_v11_shared_scaler
from .development import (
    audit_contract_lineages, audit_full_production_bars,
    audit_history_week_eligibility, parameter_counts,
)
from .model import MarketJEPAV11
from .training import V11Trainer
from .wandb_logging import V11WandbLogger, read_persisted_run_id


FORMAL_OUTPUT = Path("artifacts/training/v1_1_formal")
FORMAL_DATA_ROOT = Path("/data/jepa/v1_1_raw")
DEFAULT_FORMAL_SCALER_ANCHORS_PER_COMMODITY = 32
EPOCH_FIELDS = (
    "epoch", "train_loss", "prediction_loss_h16", "prediction_loss_h64",
    "prediction_loss_h256", "learning_rate", "global_step",
    "optimizer_steps_this_epoch", "skipped_amp_steps", "elapsed_seconds",
    "peak_vram_allocated", "peak_vram_reserved",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _git_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    return commit, dirty


def validate_formal_training_config(config: dict[str, Any]) -> None:
    validate_v11_config(config)
    if Path(config["data"]["root"]).resolve() != FORMAL_DATA_ROOT.resolve():
        raise ValueError(f"formal V1.1 data root must be {FORMAL_DATA_ROOT}")
    history = config["history_week"]
    if (
        history["years"] != 3 or history["capacity"] != 156
        or history["require_full_history"] is not True
        or history["series_mode"] != "same_delivery_month"
    ):
        raise ValueError("formal V1.1 requires default 3-year same-lineage history with capacity 156")
    effective_history = {
        commodity: int(history["commodity_years"].get(commodity, history["years"]))
        for commodity in config["data"]["train_commodities"]
    }
    frozen_current = {"FG": 3, "SA": 3, "JM": 3, "SH": 2, "SP": 3}
    mismatched_history = {
        commodity: effective_history[commodity]
        for commodity, expected in frozen_current.items()
        if commodity in effective_history and effective_history[commodity] != expected
    }
    if mismatched_history:
        raise ValueError(f"formal V1.1 frozen commodity history mismatch: {mismatched_history}")
    model_size = model_size_from_config(config["model"])
    profile_training = PROFILE_TRAINING[model_size]
    expected_training = {
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "batch_size": profile_training["batch_size"],
        "gradient_accumulation": profile_training["gradient_accumulation"],
        "warmup_ratio": 0.05,
        "scheduler": "cosine",
        "gradient_clip_norm": 1.0,
        "ema_tau": 0.996,
        "amp": True,
        "amp_dtype": "bfloat16",
        "num_workers": 8,
        "gradient_checkpointing": profile_training["gradient_checkpointing"],
    }
    mismatches = {
        name: {"expected": expected, "actual": config["training"].get(name)}
        for name, expected in expected_training.items()
        if config["training"].get(name) != expected
    }
    if mismatches:
        raise ValueError(f"formal V1.1 training configuration mismatch: {mismatches}")


def _save_epoch_outputs(output: Path, history: list[dict]) -> None:
    write_json(output / "history.json", history)
    temporary = output / "epoch_metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_FIELDS)
        writer.writeheader()
        for record in history:
            writer.writerow({name: record.get(name) for name in EPOCH_FIELDS})
    temporary.replace(output / "epoch_metrics.csv")


def _append_log(path: Path, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _formal_config(config_path: str | Path, output: Path) -> dict[str, Any]:
    config = deepcopy(load_v11_config(config_path))
    config["experiment_id"] = "market_jepa_v1_1_formal"
    config["training"]["checkpoint_dir"] = str(output / "checkpoints")
    validate_formal_training_config(config)
    return config


def _scaler_selection(dataset: V11ContractDataset, anchors_per_commodity: int) -> dict[str, int]:
    return {
        commodity: min(
            anchors_per_commodity,
            sum(
                len(dataset.episode_arrays[index].anchors)
                for index in dataset.hierarchy[commodity]
            ),
        )
        for commodity in dataset.train_commodities
    }


def _filter_train_commodities_by_history(
    config: dict[str, Any], eligibility: dict[str, Any],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    requested = tuple(config["data"]["train_commodities"])
    removed = [
        {
            "commodity": commodity,
            "reason": eligibility["commodities"][commodity]["reason"],
            "total_contract_count": eligibility["commodities"][commodity]["total_contract_count"],
            "filtered_contract_count": eligibility["commodities"][commodity][
                "filtered_insufficient_history_count"
            ],
        }
        for commodity in requested
        if eligibility["commodities"][commodity]["eligible_contract_count"] == 0
    ]
    removed_names = {record["commodity"] for record in removed}
    effective = tuple(commodity for commodity in requested if commodity not in removed_names)
    if not effective:
        raise RuntimeError("history-week filtering removed every configured Train commodity")
    config["data"]["train_commodities"] = list(effective)
    allowed_overrides = {*effective, str(config["data"]["held_out_commodity"])}
    config["history_week"]["commodity_years"] = {
        commodity: years
        for commodity, years in config["history_week"]["commodity_years"].items()
        if commodity in allowed_overrides
    }
    validate_v11_config(config)
    return effective, removed


def _filter_train_commodities_by_bar_audit(
    config: dict[str, Any], audit: dict[str, Any],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    requested = tuple(config["data"]["train_commodities"])
    failed = set(audit["failed_commodities"])
    removed = [
        {"commodity": commodity, "audit": audit["commodities"][commodity]}
        for commodity in requested if commodity in failed
    ]
    effective = tuple(commodity for commodity in requested if commodity not in failed)
    if not effective:
        raise RuntimeError("full-bar filtering removed every history-eligible Train commodity")
    config["data"]["train_commodities"] = list(effective)
    allowed_overrides = {*effective, str(config["data"]["held_out_commodity"])}
    config["history_week"]["commodity_years"] = {
        commodity: years
        for commodity, years in config["history_week"]["commodity_years"].items()
        if commodity in allowed_overrides
    }
    validate_v11_config(config)
    return effective, removed


def _bar_audit_failure_description(record: dict[str, Any]) -> str:
    audit = record["audit"]
    reasons = [
        f"{name}={audit[name]}"
        for name in (
            "invalid_price_rows", "negative_or_invalid_volume_rows", "invalid_ohlc_rows",
        )
        if audit[name]
    ]
    if audit["used_contracts_missing_daily_or_weekly"]:
        reasons.append(
            "missing_daily_or_weekly="
            f"{len(audit['used_contracts_missing_daily_or_weekly'])}"
        )
    daily_mismatches = sum(audit["daily_field_mismatches"].values())
    weekly_mismatches = sum(audit["weekly_field_mismatches"].values())
    if daily_mismatches:
        reasons.append(f"daily_field_mismatches={daily_mismatches}")
    if weekly_mismatches:
        reasons.append(f"weekly_field_mismatches={weekly_mismatches}")
    if not reasons:
        reasons.append("cache_coverage_or_parity_failure")
    return f"{record['commodity']}({','.join(reasons)})"


def _wandb_run_name(config: dict[str, Any], trainable_params: int) -> str:
    configured = config["logging"]["wandb"].get("run_name")
    if configured:
        return str(configured)
    size = model_size_from_config(config["model"])
    millions = trainable_params / 1_000_000
    return (
        f"v1.1-{size}-{millions:.2f}M-"
        f"{int(config['training']['max_epochs'])}E-seed{int(config['training']['seed'])}"
    )


def _wandb_metadata(
    config: dict[str, Any], counts: dict[str, Any], protocol: dict[str, Any],
) -> dict[str, Any]:
    model = config["model"]
    training = config["training"]
    return {
        "design_version": "V1.1",
        "model_size": model_size_from_config(model),
        "trainable_params": int(counts["v1_1_trainable"]),
        "ema_params": int(counts["v1_1_ema"]),
        "d_model": int(model["d_model"]),
        "heads": int(model["num_heads"]),
        "ffn_dim": int(model["ffn_dim"]),
        "minute_layers": int(model["minute_layers"]),
        "commodity_state_tokens": int(model["commodity_state_tokens"]),
        "contract_state_tokens": int(model["contract_state_tokens"]),
        "belief_tokens": int(model["belief_tokens"]),
        "minute_capacity": int(model["minute_capacity"]),
        "daily_capacity": int(model["daily_capacity"]),
        "current_weekly_capacity": int(model["current_weekly_capacity"]),
        "historical_weekly_capacity": int(model["history_weekly_capacity"]),
        "train_commodities": list(protocol["train_commodities"]),
        "heldout_commodity": protocol["held_out_commodity"],
        "history_years": dict(protocol["history_years"]),
        "eligible_contracts_by_commodity": dict(protocol["eligible_contract_count"]),
        "anchors_by_commodity": dict(protocol["valid_anchors"]),
        "total_anchors": int(protocol["total_train_anchors"]),
        "sampler": protocol["sampler"],
        "batch_size": int(training["batch_size"]),
        "gradient_accumulation": int(training["gradient_accumulation"]),
        "effective_batch_size": int(training["batch_size"] * training["gradient_accumulation"]),
        "max_epochs": int(training["max_epochs"]),
        "seed": int(training["seed"]),
        "optimizer": training["optimizer"],
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip": float(training["gradient_clip_norm"]),
        "amp": bool(training["amp"]),
        "amp_dtype": training["amp_dtype"],
        "ema_tau": float(training["ema_tau"]),
        "checkpoint_policy": protocol["checkpoint_policy"],
        "git_commit": protocol["git_commit"],
        "data_manifest_sha256": protocol["data_build_manifest_sha256"],
        "implementation_sha256": protocol["implementation_sha256"],
    }


def _wandb_tags(config: dict[str, Any], trainable_params: int) -> list[str]:
    dtype = str(config["training"]["amp_dtype"])
    precision = {"bfloat16": "BF16", "float16": "FP16"}.get(dtype, dtype.upper())
    return [
        "formal",
        "v1.1",
        model_size_from_config(config["model"]),
        f"{round(trainable_params / 1_000_000)}M",
        precision,
    ]


def _final_summary_markdown(summary: dict[str, Any]) -> str:
    answers = summary["answers"]
    lines = ["# Market-JEPA V1.1 formal training", "", f"Status: `{summary['status']}`", ""]
    lines.extend(f"{index}. {answer}" for index, answer in enumerate(answers, 1))
    lines.extend(("", f"Held-out {summary['held_out_commodity']} evaluation was not run.", ""))
    return "\n".join(lines)


def write_failure_summary(output: Path, error: BaseException) -> None:
    summary = {
        "status": "V1_1_FORMAL_TRAINING_FAIL",
        "error": f"{type(error).__name__}: {error}",
    }
    write_json(output / "final_summary.json", summary)
    _atomic_text(
        output / "final_summary.md",
        "# Market-JEPA V1.1 formal training\n\n"
        "Status: `V1_1_FORMAL_TRAINING_FAIL`\n\n"
        f"Error: `{summary['error']}`\n",
    )
    _append_log(output / "training.log", f"training failed error={summary['error']}")


def run_formal_training(
    *, config_path: str | Path, output: Path = FORMAL_OUTPUT,
    resume: str | Path | None = None,
    scaler_anchors_per_commodity: int = DEFAULT_FORMAL_SCALER_ANCHORS_PER_COMMODITY,
) -> dict[str, Any]:
    if scaler_anchors_per_commodity <= 0:
        raise ValueError("scaler anchors per commodity must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.log"
    config = _formal_config(config_path, output)
    model_size = model_size_from_config(config["model"])
    max_epochs = int(config["training"]["max_epochs"])
    final_epoch = max_epochs - 1
    checkpoint_path = Path(config["training"]["checkpoint_dir"]) / "last.pt"
    if resume is None and checkpoint_path.exists():
        raise FileExistsError(f"formal checkpoint already exists; use --resume {checkpoint_path}")
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(resume)
    if not torch.cuda.is_available():
        raise RuntimeError("formal V1.1 requires CUDA because AMP=true; CUDA_NOT_AVAILABLE")

    data_root = Path(config["data"]["root"])
    build_manifest = data_root / "build_manifest.json"
    if not build_manifest.is_file():
        raise FileNotFoundError(build_manifest)
    data_manifest_sha256 = sha256(build_manifest)
    git_commit, git_dirty = _git_state()
    implementation = v11_implementation_manifest()
    implementation_sha256 = manifest_sha256(implementation)
    requested_train_commodities = tuple(config["data"]["train_commodities"])
    held_out_commodity = str(config["data"]["held_out_commodity"])
    _append_log(
        log_path,
        "formal hard-gate audit started; market data scope="
        f"{','.join(requested_train_commodities)}",
    )

    eligibility = audit_history_week_eligibility(
        data_root, config, output, episode_role="train",
    )
    train_commodities, removed_commodities = _filter_train_commodities_by_history(
        config, eligibility,
    )
    removed_description = ",".join(
        f"{record['commodity']}({record['reason']})" for record in removed_commodities
    ) or "none"
    ineligible_contract_count = sum(
        not record["eligible"] for record in eligibility["contracts"]
    )
    _append_log(
        log_path,
        "history-week filter removed commodities="
        f"{removed_description}; ineligible_contracts={ineligible_contract_count}; "
        f"remaining={len(train_commodities)}",
    )
    history_filter = {
        "requested_train_commodities": list(requested_train_commodities),
        "effective_train_commodities": list(train_commodities),
        "removed_commodities": removed_commodities,
        "ineligible_contract_count": ineligible_contract_count,
    }
    write_json(output / "history_week_filter.json", history_filter)

    full_bars = audit_full_production_bars(
        data_root, train_commodities, raise_on_failure=False,
    )
    train_commodities, bar_removed_commodities = _filter_train_commodities_by_bar_audit(
        config, full_bars,
    )
    bar_removed_description = ",".join(
        _bar_audit_failure_description(record) for record in bar_removed_commodities
    ) or "none"
    _append_log(
        log_path,
        "full-bar filter removed commodities="
        f"{bar_removed_description}; remaining={len(train_commodities)}",
    )
    bar_filter = {
        "requested_train_commodities": history_filter["effective_train_commodities"],
        "effective_train_commodities": list(train_commodities),
        "removed_commodities": bar_removed_commodities,
    }
    write_json(output / "full_bar_filter.json", bar_filter)

    config_yaml = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    snapshot = output / "config_snapshot.yaml"
    if resume is not None and snapshot.exists() and snapshot.read_text(encoding="utf-8") != config_yaml:
        raise ValueError("cannot resume with a changed formal config snapshot")
    _atomic_text(snapshot, config_yaml)
    config_sha256 = sha256(snapshot)

    lineage = audit_contract_lineages(data_root, output, commodities=train_commodities)
    write_json(output / "lineage_audit.json", lineage)
    hard_gate_pass = lineage["status"] == "PASS"
    data_audit = {
        "status": "PASS" if hard_gate_pass else "FAIL",
        "root": str(data_root),
        "market_data_commodities_opened": list(requested_train_commodities),
        "history_week_filter": history_filter,
        "full_bar_filter": bar_filter,
        "held_out_commodity": held_out_commodity,
        "held_out_bar_files_opened": False,
        "lineage": lineage,
        "history_week_eligibility": eligibility,
        "full_production_bar_audit": full_bars,
    }
    write_json(output / "data_audit.json", data_audit)
    if not hard_gate_pass:
        raise RuntimeError("formal production data hard gate failed")
    _append_log(log_path, "formal hard-gate audit PASS")

    store = V11DataStore.from_directory(
        data_root, train_commodities, episode_role="train",
    )
    unscaled_train = V11ContractDataset(store, config, role="train", scaler=None)
    included_keys = {arrays.episode.key for arrays in unscaled_train.episode_arrays}
    eligible_keys = {
        (record["commodity"], record["contract_uid"], int(record["episode_id"]))
        for record in eligibility["contracts"]
        if record["eligible"] and record["commodity"] in train_commodities
    }
    missing_eligible = sorted(eligible_keys - included_keys)
    if missing_eligible:
        raise RuntimeError(f"eligible contracts failed dataset construction: {missing_eligible[:5]}")
    scaler_selection = _scaler_selection(unscaled_train, scaler_anchors_per_commodity)
    shared_scaler = fit_v11_shared_scaler(unscaled_train, scaler_anchors_per_commodity)
    if set(shared_scaler.fitted_commodities) != set(train_commodities):
        raise RuntimeError("shared scaler fitting population differs from configured Train commodities")
    write_json(output / "shared_imc_scaler.json", shared_scaler.to_dict())
    train_dataset = V11ContractDataset(store, config, role="train", scaler=shared_scaler)
    if len(train_dataset) != len(unscaled_train):
        raise RuntimeError("scaled dataset population differs from unscaled fitting dataset")

    anchors_by_commodity = {
        commodity: sum(
            len(train_dataset.episode_arrays[index].anchors)
            for index in train_dataset.hierarchy[commodity]
        )
        for commodity in train_commodities
    }
    eligible_counts = {
        commodity: eligibility["commodities"][commodity]["eligible_contract_count"]
        for commodity in train_commodities
    }
    filtered_counts = {
        commodity: eligibility["commodities"][commodity]["filtered_insufficient_history_count"]
        for commodity in train_commodities
    }
    counts = parameter_counts(config)
    write_json(output / "parameter_counts.json", counts)
    protocol = {
        "design_version": "1.1",
        "model_scale": f"V1.1-{model_size}",
        "git_commit": git_commit,
        "git_tracked_worktree_dirty_at_start": git_dirty,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "data_build_manifest_sha256": data_manifest_sha256,
        "requested_train_commodities": list(requested_train_commodities),
        "history_eligible_train_commodities": history_filter["effective_train_commodities"],
        "train_commodities": list(train_commodities),
        "held_out_commodity": held_out_commodity,
        "history_years": {
            commodity: int(config["history_week"]["commodity_years"].get(
                commodity, config["history_week"]["years"],
            ))
            for commodity in train_commodities
        },
        "history_series_mode": "same_delivery_month",
        "eligible_contract_count": eligible_counts,
        "filtered_contract_count": filtered_counts,
        "valid_anchors": anchors_by_commodity,
        "total_train_anchors": len(train_dataset),
        "sampler": "Commodity -> Eligible Contract -> Anchor",
        "samples_per_epoch": len(train_dataset),
        "batch_size": config["training"]["batch_size"],
        "gradient_accumulation": config["training"]["gradient_accumulation"],
        "amp": config["training"]["amp"],
        "amp_dtype": config["training"].get("amp_dtype", "float16"),
        "epochs": config["training"]["max_epochs"],
        "seed": config["training"]["seed"],
        "num_workers": config["training"]["num_workers"],
        "shared_scaler_fitting_population": list(train_commodities),
        "scaler_anchors_per_commodity": scaler_anchors_per_commodity,
        "scaler_selected_anchor_count": scaler_selection,
        "scaler_total_selected_anchors": sum(scaler_selection.values()),
        "scaler_source_valid_counts": shared_scaler.source_counts,
        "checkpoint_policy": "fixed_budget_final",
        "official_endpoint": f"epoch {final_epoch} last.pt",
        "HELD_OUT_READ_DURING_TRAINING": False,
        "RB_READ_DURING_TRAINING": False if held_out_commodity == "RB" else None,
        "wandb": {
            "enabled": config["logging"]["wandb"]["enabled"],
            "mode": config["logging"]["wandb"]["mode"],
            "project": config["logging"]["wandb"]["project"],
            "group": config["logging"]["wandb"]["group"],
            "log_every_optimizer_steps": config["logging"]["wandb"]["log_every_optimizer_steps"],
            "diagnostic_only": True,
        },
    }
    write_json(output / "protocol.json", protocol)
    checkpoint_policy = {
        "selection": "fixed_budget_final",
        "official_checkpoint": str(checkpoint_path),
        "official_epoch": final_epoch,
        "validation_dataset": False,
        "validation_can_select_checkpoint": False,
        "early_stopping": False,
    }
    write_json(output / "checkpoint_policy.json", checkpoint_policy)
    _append_log(
        log_path,
        f"dataset ready: anchors={len(train_dataset)} eligible_contracts={eligible_counts}",
    )

    configure_determinism(int(config["training"]["seed"]))
    model = MarketJEPAV11(config["model"])
    wandb_logger = V11WandbLogger(
        config["logging"]["wandb"], output,
        warning=lambda message: _append_log(log_path, f"WARNING {message}"),
        info=lambda message: _append_log(log_path, message),
    )
    trainer = V11Trainer(
        model, config, train_dataset, torch.device("cuda"),
        validation_dataset=None, samples_per_epoch=len(train_dataset),
        data_manifest_sha256=data_manifest_sha256, step_logger=wandb_logger,
    )
    resumed = resume is not None
    resume_state = None
    if resumed:
        resume_state = load_v11_checkpoint(Path(resume))
        trainer.resume(resume_state)
        _append_log(log_path, f"resume accepted: {resume}; start_epoch={trainer.start_epoch}")

    resume_run_id = (
        resume_state.get("wandb_run_id") if resume_state is not None else None
    ) or (read_persisted_run_id(output) if resumed else None)
    run_name = _wandb_run_name(config, int(counts["v1_1_trainable"]))
    run_id = wandb_logger.init(
        _wandb_metadata(config, counts, protocol),
        run_name=run_name,
        tags=_wandb_tags(config, int(counts["v1_1_trainable"])),
        resume_run_id=resume_run_id,
        resumed=resumed,
    )
    if run_id:
        trainer.wandb_run_id = run_id

    def epoch_complete(record: dict) -> None:
        # Local files are the source of truth and are committed before W&B mirrors them.
        _save_epoch_outputs(output, trainer.history)
        _append_log(
            log_path,
            "epoch={epoch} train_loss={train_loss:.8f} h16={prediction_loss_h16:.8f} "
            "h64={prediction_loss_h64:.8f} h256={prediction_loss_h256:.8f} "
            "global_step={global_step} elapsed_seconds={elapsed_seconds:.3f}".format(**record),
        )
        wandb_logger.log_epoch(record, samples_per_epoch=len(train_dataset))

    torch.cuda.reset_peak_memory_stats()
    invocation_started = time.perf_counter()
    try:
        history = trainer.fit(epoch_callback=epoch_complete)
    except BaseException as error:
        wandb_logger.fail(error)
        raise
    torch.cuda.synchronize()
    invocation_elapsed = time.perf_counter() - invocation_started
    _save_epoch_outputs(output, history)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = load_v11_checkpoint(checkpoint_path)
    restored = model_from_checkpoint(state)
    del restored
    checkpoint_roundtrip = state["history"] == history
    checkpoint_sha256 = sha256(checkpoint_path)

    finite_metrics = all(
        math.isfinite(float(record[field]))
        for record in history
        for field in ("train_loss", "prediction_loss_h16", "prediction_loss_h64", "prediction_loss_h256")
    )
    gradient_ok = bool(
        state.get("gradient_connectivity")
        and not state["gradient_connectivity"]["missing"]
        and not state["gradient_connectivity"]["nonfinite"]
    )
    completed = (
        len(history) == max_epochs and history[-1]["epoch"] == final_epoch
        and int(state["epoch"]) == final_epoch and finite_metrics and gradient_ok
        and state["checkpoint_selection"] == "fixed_budget_final"
        and state["data_manifest_sha256"] == data_manifest_sha256
        and state["v11_implementation_sha256"] == implementation_sha256
        and state["shared_imc_scaler"] == shared_scaler.to_dict()
        and checkpoint_roundtrip
    )
    peak_allocated = max(int(record["peak_vram_allocated"] or 0) for record in history)
    peak_reserved = max(int(record["peak_vram_reserved"] or 0) for record in history)
    total_elapsed = sum(float(record["elapsed_seconds"]) for record in history)
    final = history[-1]
    resource_usage = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "peak_vram_allocated": peak_allocated,
        "peak_vram_reserved": peak_reserved,
        "total_epoch_elapsed_seconds": total_elapsed,
        "this_invocation_elapsed_seconds": invocation_elapsed,
    }
    write_json(output / "resource_usage.json", resource_usage)
    answers = [
        f"All {max_epochs} epochs complete: {'YES' if len(history) == max_epochs else 'NO'}.",
        f"Final checkpoint is epoch{final_epoch} last.pt: "
        f"{'YES' if state['epoch'] == final_epoch else 'NO'}.",
        f"Resume occurred: {'YES' if resumed else 'NO'}.",
        f"Final global_step: {state['global_step']}.",
        f"Train commodities: {list(train_commodities)}.",
        f"Held-out {held_out_commodity} read during training: NO.",
        f"Shared scaler fit only configured Train commodities {list(train_commodities)}: YES.",
        f"Eligible contracts: {eligible_counts}.",
        f"Valid anchors: {anchors_by_commodity}.",
        f"History requirements by commodity: {protocol['history_years']}.",
        f"Explicit history overrides: {config['history_week']['commodity_years']}.",
        "Sampler is Commodity -> Eligible Contract -> Anchor: YES.",
        f"Final train loss: {final['train_loss']}.",
        f"Final H16/H64/H256 losses: {final['prediction_loss_h16']}/{final['prediction_loss_h64']}/{final['prediction_loss_h256']}.",
        f"NaN/Inf occurred: {'NO' if finite_metrics else 'YES'}.",
        f"AMP skipped optimizer steps: {state.get('skipped_optimizer_steps', 0)}.",
        f"All intended trainable parameters have finite gradients: {'YES' if gradient_ok else 'NO'}.",
        f"Daily truncation count: {state['daily_truncation_count']}.",
        f"Peak VRAM allocated/reserved: {peak_allocated}/{peak_reserved} bytes.",
        f"Total epoch elapsed time: {total_elapsed} seconds.",
        f"Checkpoint SHA256: {checkpoint_sha256}.",
        f"Data manifest SHA256: {data_manifest_sha256}.",
        f"Implementation SHA256: {implementation_sha256}.",
    ]
    summary = {
        "status": "V1_1_FORMAL_TRAINING_PASS" if completed else "V1_1_FORMAL_TRAINING_FAIL",
        "final_epoch": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "resumed": resumed,
        "total_elapsed_seconds": total_elapsed,
        "final_metrics": {name: final[name] for name in (
            "train_loss", "prediction_loss_h16", "prediction_loss_h64", "prediction_loss_h256",
        )},
        "eligible_contracts": eligible_counts,
        "anchors": anchors_by_commodity,
        "peak_vram_allocated": peak_allocated,
        "peak_vram_reserved": peak_reserved,
        "daily_truncation_count": int(state["daily_truncation_count"]),
        "skipped_optimizer_steps": int(state.get("skipped_optimizer_steps", 0)),
        "held_out_commodity": held_out_commodity,
        "held_out_read": False,
        "held_out_RB_read": False if held_out_commodity == "RB" else None,
        "checkpoint_roundtrip": "PASS" if checkpoint_roundtrip else "FAIL",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "implementation_sha256": implementation_sha256,
        "answers": answers,
    }
    write_json(output / "final_summary.json", summary)
    _atomic_text(output / "final_summary.md", _final_summary_markdown(summary))
    _append_log(log_path, f"training finished status={summary['status']}")
    wandb_logger.finish({
        "status": summary["status"],
        "final_epoch": summary["final_epoch"],
        "final_global_step": summary["global_step"],
        "final_loss": final["train_loss"],
        "final_h16_loss": final["prediction_loss_h16"],
        "final_h64_loss": final["prediction_loss_h64"],
        "final_h256_loss": final["prediction_loss_h256"],
        "total_training_seconds": total_elapsed,
        "peak_vram_gb": peak_allocated / (1024 ** 3),
        "checkpoint_sha256": checkpoint_sha256,
    })
    return summary
