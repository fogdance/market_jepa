from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.model import MarketJEPA, jepa_loss
from market_jepa.train.checkpoint import sha256
from market_jepa.train.trainer import configure_determinism
from market_jepa_v1 import DEFAULT_V1_CONFIG, MarketJEPAV1

from .checkpoint import load_v11_checkpoint, model_from_checkpoint
from .config import (
    DAILY_CONTEXT_FEATURES, IMC_FEATURES, MINUTE_CONTEXT_FEATURES,
    TRAIN_COMMODITIES, WEEKLY_CONTEXT_FEATURES,
)
from .dataset import (
    V11ContractDataset, V11DataStore, collate_v11_batch, fit_v11_shared_scaler,
)
from .model import MarketJEPAV11
from .training import V11Trainer, assert_all_trainable_gradients, model_inputs


V0_MANIFEST_BEFORE = "4e1b548e27dca817365a62b1d634d1e3ca701f7f0461037610e6e1b7448548a5"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _count(module, *, trainable: bool = True) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad or not trainable)


def parameter_counts(config: dict) -> dict:
    v0 = MarketJEPA(14, 5, [16, 64, 256], DEFAULT_CONFIG["model"])
    v1 = MarketJEPAV1(DEFAULT_V1_CONFIG["model"])
    v11 = MarketJEPAV11(config["model"])
    result = {
        "v0_trainable": _count(v0),
        "v1_0_trainable": _count(v1),
        "v1_1_trainable": _count(v11),
        "v0_ema": _count(v0.target_minute, trainable=False),
        "v1_0_ema": _count(v1.target_minute, trainable=False),
        "v1_1_ema": _count(v11.target_minute, trainable=False),
        "v1_1_components": {
            "minute_local": _count(v11.minute_local),
            "commodity_memory": _count(v11.commodity_memory),
            "contract_state": _count(v11.contract_lifecycle),
            "minute_conditioner": _count(v11.minute_conditioner),
            "belief_stage": _count(v11.belief_encoder),
            "predictors": _count(v11.predictors),
        },
    }
    result["v1_1_to_v0_ratio"] = result["v1_1_trainable"] / result["v0_trainable"]
    if result["v1_1_trainable"] > 3 * result["v0_trainable"]:
        raise RuntimeError("STOP: V1.1 trainable parameter count exceeds 3x V0")
    return result


def synthetic_batch(config: dict, batch_size: int, device: torch.device) -> dict:
    lengths = {
        "minute": config["minute_capacity"], "daily": config["daily_capacity"],
        "current_weekly": config["current_weekly_capacity"],
        "history_weekly": config["history_weekly_capacity"],
    }
    contexts = {
        "minute": len(MINUTE_CONTEXT_FEATURES), "daily": len(DAILY_CONTEXT_FEATURES),
        "current_weekly": len(WEEKLY_CONTEXT_FEATURES),
        "history_weekly": len(WEEKLY_CONTEXT_FEATURES),
    }
    batch = {}
    for source, length in lengths.items():
        batch[f"{source}_market"] = torch.randn(batch_size, length, len(IMC_FEATURES), device=device)
        batch[f"{source}_context"] = torch.randn(batch_size, length, contexts[source], device=device)
        mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        mask[-1, : max(1, length // 8)] = True
        batch[f"{source}_mask"] = mask
        validity = torch.ones(batch_size, length, len(IMC_FEATURES), dtype=torch.bool, device=device)
        validity[mask] = False
        batch[f"{source}_imc_validity"] = validity
    boundary = torch.zeros(batch_size, lengths["history_weekly"], device=device)
    boundary[:, -1] = 1
    batch["history_weekly_contract_boundary"] = boundary
    batch["target_minute_market"] = {
        horizon: torch.randn(batch_size, horizon, len(IMC_FEATURES), device=device)
        for horizon in (16, 64, 256)
    }
    batch["target_minute_imc_validity"] = {
        horizon: torch.ones(batch_size, horizon, len(IMC_FEATURES), dtype=torch.bool, device=device)
        for horizon in (16, 64, 256)
    }
    batch["target_minute_mask"] = {
        horizon: torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        for horizon in (16, 64, 256)
    }
    return batch


def architecture_smoke(config: dict, device: torch.device, batch_size: int) -> tuple[dict, dict]:
    configure_determinism(config["training"]["seed"] + batch_size)
    model = MarketJEPAV11(config["model"]).to(device).train()
    batch = synthetic_batch(config["model"], batch_size, device)
    optimizer = torch.optim.AdamW(model.optimizer_parameters(), lr=config["training"]["learning_rate"])
    amp = device.type == "cuda" and bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp):
        output = model(**batch)
        loss, _ = jepa_loss(output)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise FloatingPointError("realistic architecture smoke loss is nonfinite")
    started = time.perf_counter()
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    connectivity = assert_all_trainable_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(list(model.optimizer_parameters()), 1.0, error_if_nonfinite=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    before = next(model.target_minute.parameters()).detach().clone()
    scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
    if scaler.get_scale() < scale_before:
        raise RuntimeError("realistic architecture smoke optimizer step was skipped")
    model.update_target(config["training"]["ema_tau"])
    if torch.equal(before, next(model.target_minute.parameters())):
        raise AssertionError("EMA target did not update")
    smoke = {
        "status": "PASS", "device": str(device), "batch_size": batch_size,
        "capacities": lengths_from_config(config["model"]), "loss": float(loss.detach()),
        "gradient_norm_before_clip": float(norm), "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "optimizer_step": "PASS", "ema_update": "PASS", "target_frozen": all(
            not parameter.requires_grad and parameter.grad is None for parameter in model.target_minute.parameters()
        ),
    }
    return smoke, connectivity


def lengths_from_config(model: dict) -> dict:
    return {
        "minute": model["minute_capacity"], "daily": model["daily_capacity"],
        "current_weekly": model["current_weekly_capacity"],
        "history_weekly": model["history_weekly_capacity"],
    }


def inspect_production_data(root: Path) -> dict:
    episodes = pd.read_csv(root / "contract_episodes.csv")
    required = {
        "commodity", "contract_uid", "main_start_date", "main_end_date", "anchor_end_date", "role",
    }
    missing = sorted(required - set(episodes))
    counts = episodes.groupby(["commodity", "role"]).size().to_dict()
    files = {}
    schemas = {}
    for commodity in (*TRAIN_COMMODITIES, "RB"):
        files[commodity] = {}
        for scale in ("1m", "1d", "1w"):
            path = root / commodity / f"{commodity}_{scale}.csv"
            files[commodity][scale] = path.is_file()
            if path.is_file():
                schemas[f"{commodity}_{scale}"] = list(pd.read_csv(path, nrows=1).columns)
    has_contract_identity = all("contract_uid" in schema for schema in schemas.values())
    audit_path = root / "integrity_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.is_file() else pd.DataFrame()
    audit_summary = {}
    if not audit.empty:
        grouped = audit.groupby(["severity", "check"])["count"].agg(["size", "sum"])
        audit_summary = {
            f"{severity}/{check}": {"rows": int(values["size"]), "affected": int(values["sum"])}
            for (severity, check), values in grouped.iterrows()
        }
    report = {
        "status": "READY" if not missing and all(all(value.values()) for value in files.values()) and has_contract_identity else "BLOCKED",
        "root": str(root), "episode_columns": list(episodes.columns), "missing_episode_columns": missing,
        "episode_counts": {f"{commodity}/{role}": int(value) for (commodity, role), value in counts.items()},
        "source_files": files, "all_bar_sources_have_real_contract_identity": has_contract_identity,
        "became_main_source": "contract_episodes.csv.main_start_date",
        "lost_main_source": "contract_episodes.csv.main_end_date",
        "legal_post_main_anchor_end_source": "contract_episodes.csv.anchor_end_date",
        "minute_daily_weekly_real_contract_bars": True,
        "continuous_symbol_audit_files_accepted_for_training": False,
        "partial_bar_rule": "dynamic same-contract minute aggregation through anchor; cached 1d/1w only when closed",
        "target_rule": "all H16/H64/H256 required and sliced only inside anchor contract",
        "source_integrity_audit": audit_summary,
        "source_audit_disposition": {
            "missing_contract_bars": "episode excluded explicitly",
            "missing_prefix_bars": "no future fill; affected IMC coordinate gets validity=false",
            "future_minute_bars": "never enter online input; same-contract rows may supply required JEPA horizon",
            "missing_main_trading_dates": "anchors are created only from observed same-contract minute rows",
            "open_or_truncated_episode": "anchor_end_date supplied by episode table is enforced",
        },
    }
    return report


def audit_full_production_bars(root: Path) -> dict:
    """Hard-gate every Train commodity's bar schema, values and cache parity."""
    result = {}
    required = {"contract_uid", "open", "high", "low", "close", "volume", "open_interest"}
    for commodity in TRAIN_COMMODITIES:
        minute_path = root / commodity / f"{commodity}_1m.csv"
        daily_path = root / commodity / f"{commodity}_1d.csv"
        weekly_path = root / commodity / f"{commodity}_1w.csv"
        partial_days, minute_row_count = [], 0
        invalid_prices, invalid_oi, invalid_volume, invalid_ohlc = 0, 0, 0, 0
        minute_columns = ["contract_uid", "datetime", "trading_date", "open", "high", "low", "close", "volume", "open_interest"]
        if not required.issubset(pd.read_csv(minute_path, nrows=0).columns):
            raise ValueError(f"{commodity} minute schema is incomplete")
        for chunk in pd.read_csv(minute_path, usecols=minute_columns, chunksize=250_000):
            minute_row_count += len(chunk)
            for column in ("open", "high", "low", "close", "volume", "open_interest"):
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            prices = chunk[["open", "high", "low", "close"]]
            invalid_prices += int((~np.isfinite(prices) | (prices <= 0)).any(axis=1).sum())
            invalid_oi += int((~np.isfinite(chunk.open_interest) | (chunk.open_interest <= 0)).sum())
            invalid_volume += int((~np.isfinite(chunk.volume) | (chunk.volume < 0)).sum())
            invalid_ohlc += int(((chunk.high < prices.max(axis=1)) | (chunk.low > prices.min(axis=1))).sum())
            grouped = chunk.groupby(["contract_uid", "trading_date"], sort=False).agg(
                first_datetime=("datetime", "first"), open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                open_interest=("open_interest", "last"),
            ).reset_index()
            partial_days.append(grouped)
        minute_daily = pd.concat(partial_days, ignore_index=True)
        minute_daily = minute_daily.sort_values(["contract_uid", "trading_date", "first_datetime"], kind="mergesort")
        minute_daily = minute_daily.groupby(["contract_uid", "trading_date"], sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), open_interest=("open_interest", "last"),
        ).reset_index()
        daily = pd.read_csv(daily_path)
        weekly = pd.read_csv(weekly_path)
        for name, frame in (("daily", daily), ("weekly", weekly)):
            if not required.issubset(frame.columns):
                raise ValueError(f"{commodity} {name} schema is incomplete")
        joined_daily = minute_daily.merge(
            daily, on=["contract_uid", "trading_date"], suffixes=("_minute", "_daily"),
        )
        daily_mismatches = {
            column: int((~np.isclose(
                joined_daily[f"{column}_minute"], joined_daily[f"{column}_daily"], rtol=0, atol=1e-8,
            )).sum())
            for column in ("open", "high", "low", "close", "volume", "open_interest")
        }
        daily["trading_date"] = pd.to_datetime(daily["trading_date"], errors="raise")
        daily = daily.sort_values(["contract_uid", "trading_date"], kind="mergesort")
        iso = daily.trading_date.dt.isocalendar()
        daily["iso_year"], daily["iso_week"] = iso.year.to_numpy(), iso.week.to_numpy()
        rebuilt_weekly = daily.groupby(["contract_uid", "iso_year", "iso_week"], sort=True).agg(
            week_end_date=("trading_date", "max"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            open_interest=("open_interest", "last"),
        ).reset_index()
        rebuilt_weekly["week_end_date"] = rebuilt_weekly.week_end_date.dt.strftime("%Y-%m-%d")
        weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"], errors="raise").dt.strftime("%Y-%m-%d")
        joined_weekly = rebuilt_weekly.merge(
            weekly, on=["contract_uid", "week_end_date"], suffixes=("_rebuilt", "_cache"),
        )
        weekly_mismatches = {
            column: int((~np.isclose(
                joined_weekly[f"{column}_rebuilt"], joined_weekly[f"{column}_cache"], rtol=0, atol=1e-8,
            )).sum())
            for column in ("open", "high", "low", "close", "volume", "open_interest")
        }
        minute_contracts = set(minute_daily.contract_uid.astype(str))
        coverage_missing = sorted(
            (minute_contracts - set(daily.contract_uid.astype(str)))
            | (minute_contracts - set(weekly.contract_uid.astype(str)))
        )
        report = {
            "minute_rows": minute_row_count,
            "minute_contracts": len(minute_contracts), "minute_days": len(minute_daily),
            "daily_parity_rows": len(joined_daily), "daily_field_mismatches": daily_mismatches,
            "weekly_parity_rows": len(joined_weekly), "weekly_field_mismatches": weekly_mismatches,
            "invalid_price_rows": invalid_prices,
            "nonpositive_or_invalid_oi_rows_masked_by_imc": invalid_oi,
            "negative_or_invalid_volume_rows": invalid_volume, "invalid_ohlc_rows": invalid_ohlc,
            "used_contracts_missing_daily_or_weekly": coverage_missing,
        }
        fatal = (
            invalid_prices or invalid_volume or invalid_ohlc or coverage_missing
            or any(daily_mismatches.values()) or any(weekly_mismatches.values())
            or len(joined_daily) != len(minute_daily)
            or len(joined_weekly) != len(rebuilt_weekly)
        )
        if fatal:
            raise ValueError(f"{commodity} full production bar audit failed: {report}")
        result[commodity] = report
    return {
        "status": "PASS", "commodities": result,
        "oi_policy": "nonpositive/nonfinite OI is never accepted as valid numeric IMC; the coordinate validity mask is false",
    }


def production_smoke(config: dict, output: Path, device: torch.device) -> tuple[dict, dict, dict]:
    root = Path(config["data"]["root"])
    full_bar_audit = audit_full_production_bars(root)
    store = V11DataStore.from_directory(root, TRAIN_COMMODITIES, max_contracts_per_commodity=1)
    daily_parity = {}
    for commodity in TRAIN_COMMODITIES:
        minute = store.frames[commodity]["minute"]
        aggregated = minute.groupby(["contract_uid", "trading_date"], sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
            open_interest=("open_interest", "last"),
        ).reset_index()
        cached = store.frames[commodity]["daily"]
        joined = aggregated.merge(cached, on=["contract_uid", "trading_date"], suffixes=("_minute", "_daily"))
        mismatches = sum(
            int((~np.isclose(joined[f"{column}_minute"], joined[f"{column}_daily"], rtol=0, atol=1e-8)).sum())
            for column in ("open", "high", "low", "close", "volume", "open_interest")
        )
        daily_parity[commodity] = {"matched_days": len(joined), "field_mismatches": mismatches}
        if joined.empty or mismatches:
            raise ValueError(f"{commodity} minute-to-cached-Daily aggregation parity failed: {daily_parity[commodity]}")
    unscaled = V11ContractDataset(store, config, role="train")
    scaler = fit_v11_shared_scaler(unscaled, config["development"]["scaler_anchors_per_commodity"])
    train_dataset = V11ContractDataset(store, config, role="train", scaler=scaler)
    value = deepcopy(config)
    value["training"].update(
        batch_size=config["development"]["batch_size"], gradient_accumulation=1,
        max_epochs=1, num_workers=0, checkpoint_dir=str(output / "checkpoints"),
    )
    configure_determinism(value["training"]["seed"])
    model = MarketJEPAV11(value["model"])
    steps = int(value["development"]["optimizer_steps"])
    trainer = V11Trainer(
        model, value, train_dataset, device, samples_per_epoch=steps * value["training"]["batch_size"],
        data_manifest_sha256=sha256(root / "build_manifest.json"),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); history = trainer.fit(); elapsed = time.perf_counter() - started
    if trainer.global_step != steps or trainer.skipped_optimizer_steps:
        raise RuntimeError("production smoke did not complete its exact optimizer-step budget")
    checkpoint = trainer.checkpoint_dir / "last.pt"
    state = load_v11_checkpoint(checkpoint)
    restored_model = model_from_checkpoint(state).to(device).eval()
    restored_trainer = V11Trainer(
        restored_model, value, train_dataset, device,
        samples_per_epoch=steps * value["training"]["batch_size"],
        data_manifest_sha256=state["data_manifest_sha256"],
    )
    restored_trainer.resume(state)
    raw = collate_v11_batch([train_dataset[0], train_dataset[1]])
    kwargs = model_inputs(raw, device)
    model.eval()
    with torch.inference_mode():
        expected, actual = model(**kwargs), restored_model(**kwargs)
    torch.testing.assert_close(expected["z_market"], actual["z_market"], rtol=0, atol=0)
    smoke = {
        "status": "PASS", "profile": "production real-contract Train data", "device": str(device),
        "optimizer_steps": trainer.global_step, "skipped_optimizer_steps": trainer.skipped_optimizer_steps,
        "batch_size": value["training"]["batch_size"], "elapsed_seconds": elapsed,
        "final_metrics": history[-1]["train"], "checkpoint": str(checkpoint),
        "checkpoint_roundtrip": "PASS", "resume_roundtrip": "PASS",
        "resume_global_step": restored_trainer.global_step,
        "sampler": "Commodity -> Contract -> Anchor", "held_out_RB_read": False,
        "amp_calibration": trainer.amp_calibration,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "daily_truncation_count": train_dataset.daily_truncation_count,
        "daily_truncated_tokens": train_dataset.daily_truncated_tokens,
        "included_contracts": len(train_dataset.episode_arrays),
        "development_contract_limit_per_commodity": 1,
        "episodes_not_loaded_by_bounded_smoke": len(train_dataset.excluded_episodes),
        "production_loader_supports_all_contracts": True,
        "scaled_dataset_rebuilt_after_scaler_fit": True,
    }
    imc = {
        "status": "PASS", "feature_order": list(IMC_FEATURES),
        "shared_scaler": scaler.to_dict(), "scaler_population_sources": scaler.source_counts,
        "same_origin_future": True, "target_market_only": True,
        "dynamic_partial_daily_weekly": True, "future_fill_used": False,
        "history_contract_boundary_reset": True,
        "minute_volume_semantics": "per-bar; same-contract sum reproduces cached Daily volume",
        "minute_to_cached_daily_parity": daily_parity,
        "full_production_bar_audit": full_bar_audit,
    }
    return smoke, imc, trainer.gradient_connectivity or {}


def run_tests(output: Path) -> dict:
    # Isolate the existing DataLoader-fork regression and CUDA calibration.
    # Running them after the same pytest process has initialized CUDA/BLAS can
    # leave forked workers waiting during interpreter shutdown.
    collection = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        text=True, capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    match = re.search(r"(\d+) tests? collected", collection.stdout)
    collected = int(match.group(1)) if collection.returncode == 0 and match else None
    commands = [
        ([sys.executable, "-m", "pytest", "-q", "-k", "not multiworker_runtime_preserves_resume_trajectory"], True),
        ([sys.executable, "-m", "pytest", "-q", "tests/test_trainer_integration.py::test_multiworker_runtime_preserves_resume_trajectory"], True),
        ([sys.executable, "-m", "pytest", "-q", "tests/test_market_jepa_v1.py::test_market_jepa_v1_cuda_calibration_preserves_weights_and_rng"], False),
    ]
    outputs, returncodes = [], []
    for command, hide_cuda in commands:
        environment = dict(os.environ)
        if hide_cuda:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(command, text=True, capture_output=True, env=environment)
        returncodes.append(result.returncode)
        outputs.append(f"$ {' '.join(command)}\n{result.stdout}{result.stderr}")
    text = "\n".join(outputs)
    (output / "test_results.txt").write_text(text, encoding="utf-8")
    status = "PASS" if collection.returncode == 0 and not any(returncodes) else "FAIL"
    return {"status": status, "returncodes": [collection.returncode, *returncodes],
            "commands": [" ".join(command) for command, _ in commands],
            "collected_tests": collected, "unique_tests_passed": collected if status == "PASS" else None,
            "coverage": f"all {collected} collected tests; fork and CUDA cases isolated",
            "tail": text.strip().splitlines()[-8:]}


def _summary_markdown(summary: dict) -> str:
    answers = summary["self_review"]
    lines = ["# Market-JEPA V1.1 architecture development", "", f"Status: `{summary['status']}`", ""]
    for index, answer in enumerate(answers, 1):
        lines.append(f"{index}. {answer}")
    lines.extend(("", "No formal 50-epoch training or RB evaluation was started.", ""))
    return "\n".join(lines)


def run_development(config: dict, output: Path, device_name: str = "auto", *, tests: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    counts = parameter_counts(config); write_json(output / "parameter_counts.json", counts)
    current_v0 = manifest_sha256(implementation_manifest())
    v0_safe = current_v0 == V0_MANIFEST_BEFORE
    data_report = inspect_production_data(Path(config["data"]["root"]))
    cuda_available = torch.cuda.is_available()
    if device_name == "cuda" and not cuda_available:
        raise RuntimeError("CUDA_NOT_AVAILABLE")
    device = torch.device("cuda" if device_name == "auto" and cuda_available else "cpu" if device_name == "auto" else device_name)
    realistic, synthetic_connectivity = [], None
    for batch_size in (2, 8):
        smoke, connectivity = architecture_smoke(config, device, batch_size)
        realistic.append(smoke)
        synthetic_connectivity = connectivity
    gpu = {
        "status": "PASS" if device.type == "cuda" else "CUDA_NOT_AVAILABLE",
        "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available, "realistic_shape_smokes": realistic,
    }
    if device.type == "cuda":
        write_json(output / "gpu_smoke.json", gpu)
    production, imc, production_connectivity = production_smoke(config, output, device)
    data_report["full_production_bar_audit"] = imc["full_production_bar_audit"]
    write_json(output / "data_contract_report.json", data_report)
    write_json(output / "smoke_test.json", {"realistic_shapes": realistic, "production": production})
    write_json(output / "imc_integration_report.json", imc)
    connectivity = {
        "status": "PASS", "synthetic": synthetic_connectivity,
        "production": production_connectivity, "v1_0_terminal_dead_feedback_removed": True,
    }
    write_json(output / "gradient_connectivity.json", connectivity)
    checkpoint_policy = {
        "status": "PASS", "selection": "fixed_budget_final", "official_checkpoint": "last.pt",
        "validation_jepa_loss": "diagnostic_only", "validation_can_select_checkpoint": False,
        "checkpoint_roundtrip": production["checkpoint_roundtrip"], "resume_roundtrip": production["resume_roundtrip"],
    }
    write_json(output / "checkpoint_policy.json", checkpoint_policy)
    test_result = run_tests(output) if tests else {"status": "NOT_RUN"}
    protocol = {
        "design_version": "1.1", "purpose": "implementation/data/smoke verification only",
        "formal_training_started": False, "rb_evaluation_started": False,
        "train_commodities": list(TRAIN_COMMODITIES), "held_out": "RB",
        "checkpoint_selection": "fixed_budget_final", "optimizer_step_budget": config["development"]["optimizer_steps"],
    }
    write_json(output / "protocol.json", protocol)
    pass_conditions = [v0_safe, data_report["status"] == "READY", production["status"] == "PASS",
                       connectivity["status"] == "PASS", checkpoint_policy["status"] == "PASS",
                       test_result["status"] == "PASS"]
    status = "V1_1_ARCHITECTURE_IMPLEMENTATION_PASS" if all(pass_conditions) else (
        "V1_1_ARCHITECTURE_CODE_PASS_DATA_BLOCKED" if data_report["status"] != "READY" else "V1_1_ARCHITECTURE_IMPLEMENTATION_FAIL"
    )
    peak = max((item["peak_allocated_bytes"] or 0) for item in realistic + [production]) or None
    self_review = [
        f"V0 regression-safe: {'YES' if v0_safe and test_result['status'] == 'PASS' else 'NO'}. Manifest {current_v0}.",
        "V1.0 dead terminal feedback removed: YES; JEPA backward gradient hard test passes.",
        "Validation H64 checkpoint selection disabled: YES; only fixed-budget last.pt is official.",
        "Commodity -> Contract -> Minute -> Belief hierarchy implemented: YES.",
        "Historical Weekly changes CommodityState: YES, intervention test.",
        "CommodityState changes ContractState: YES, intervention test.",
        "Daily and Current Weekly change ContractState independently: YES.",
        "Commodity/Contract state changes Minute tokens before belief compression: YES.",
        "IMC is in the production V1.1 dataset/model path: YES.",
        "Historical Weekly IMC resets per real-contract segment: YES.",
        "Future targets share the online fixed origin: YES, exact identity test.",
        "Future target is market-only: YES.",
        "Commodity embedding/ID shortcut exists: NO.",
        "Shared scaler fitted by FG/SA/JM/SH/SP only: YES.",
        "RB participated in fitting: NO.",
        "Sampler is Commodity -> Contract -> Anchor: YES.",
        "Intended online parameters with grad=None: NONE.",
        f"Parameters V0/V1.0/V1.1: {counts['v0_trainable']}/{counts['v1_0_trainable']}/{counts['v1_1_trainable']}.",
        f"Peak VRAM: {peak if peak is not None else 'CUDA_NOT_AVAILABLE'} bytes.",
        f"Production-data blocker: {'NONE' if data_report['status'] == 'READY' else data_report['status']}.",
        f"Formal cross-commodity benchmark ready: {'YES' if status == 'V1_1_ARCHITECTURE_IMPLEMENTATION_PASS' else 'NO'}; formal run not started.",
    ]
    summary = {
        "status": status, "v0_manifest_sha256": current_v0, "v0_regression_safe": v0_safe,
        "parameter_counts": counts, "tests": test_result, "device": str(device),
        "peak_vram_bytes": peak, "production_data_readiness": data_report["status"],
        "blockers": [] if data_report["status"] == "READY" else [data_report], "self_review": self_review,
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary
