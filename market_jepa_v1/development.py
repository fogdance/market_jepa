from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.data import MarketDataset, collate_market_batch, prepare_market_data
from market_jepa.implementation import implementation_manifest, manifest_sha256
from market_jepa.model import MarketJEPA, jepa_loss
from market_jepa.train.checkpoint import sha256
from market_jepa.train.trainer import configure_determinism, capture_rng_state, restore_rng_state, _move

from .checkpoint import load_v1_checkpoint, model_from_checkpoint
from .data import adapt_market_batch
from .model import MarketJEPAV1
from .training import V1Trainer


# V0 provenance includes pyproject/uv.lock; refreshed only for the W&B SDK dependency.
V0_MANIFEST_BEFORE = "fc3343491e35c69ebc2d6bed04d91383b82ade6694a4c77483c4e8791f4e271e"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parameter_counts(config: dict) -> dict:
    def count(module, trainable=True):
        return sum(p.numel() for p in module.parameters() if not trainable or p.requires_grad)

    v0 = MarketJEPA(14, 5, [16, 64, 256], DEFAULT_CONFIG["model"])
    v1 = MarketJEPAV1(config["model"])
    before, after = count(v0), count(v1)
    result = {
        "v0_trainable": before, "v1_trainable": after, "difference": after - before, "ratio": after / before,
        "v0_ema": count(v0.target_minute, False), "v1_ema": count(v1.target_minute, False),
        "v0_components": {name: count(module) for name, module in v0.online.named_children()},
        "v0_predictors": count(v0.predictors),
        "v1_components": {name: count(module) for name, module in v1.online.named_children()},
        "v1_state_tokens": v1.online.state_tokens.numel(), "v1_predictors": count(v1.predictors),
        "feature_dimensions": {key: value for key, value in config["model"].items() if key.endswith(("market_dim", "context_dim"))},
    }
    terminal = v1.online.blocks[-1]
    result["v1_terminal_feedback_parameters_without_loss_gradient"] = sum(
        p.numel() for name, p in terminal.named_parameters()
        if name.startswith(("token_query_norm.", "state_key_norm.", "feedback_"))
    )
    if after > 3 * before:
        raise RuntimeError(f"STOP: V1 trainable parameter ratio {after / before:.3f} exceeds 3")
    return result


def synthetic_batch(config: dict, batch_size: int, *, minute_length: int = 512, daily_length: int = 32, weekly_length: int = 8) -> dict:
    batch = {}
    for scale, length in (("minute", minute_length), ("daily", daily_length), ("weekly", weekly_length)):
        for kind in ("market", "context"):
            batch[f"{scale}_{kind}"] = torch.randn(batch_size, length, config[f"{scale}_{kind}_dim"])
        mask = torch.zeros(batch_size, length, dtype=torch.bool)
        if length > 1:
            mask[-1, -1] = True
        batch[f"{scale}_padding_mask"] = mask
    batch["targets"] = {h: torch.randn(batch_size, h, config["minute_market_dim"]) for h in (16, 64, 256)}
    batch["persistence"] = {h: torch.randn(batch_size, h, config["minute_market_dim"]) for h in (16, 64, 256)}
    return batch


def calibrate_smoke_scaler(model, batch, optimizer, device, *, amp: bool) -> tuple[torch.amp.GradScaler, list[dict]]:
    """Find a finite initial AMP scale without consuming any training updates.

    Replays use identical dropout/RNG states. No optimizer, scheduler or EMA step
    occurs here; the returned scaler starts ready for the first real update.
    """
    if not amp:
        return torch.amp.GradScaler("cuda", enabled=False), []
    rng = capture_rng_state(include_cuda=True)
    scale = 65536.0  # PyTorch GradScaler's initial default, reduced only on evidence.
    attempts = []
    try:
        for _ in range(8):
            restore_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            probe = torch.amp.GradScaler("cuda", init_scale=scale)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                output = model(batch)
                loss, _ = jepa_loss(output)
            if not torch.isfinite(loss):
                raise FloatingPointError("AMP calibration loss is nonfinite")
            probe.scale(loss).backward()
            probe.unscale_(optimizer)
            finite = all(torch.isfinite(p.grad).all() for p in model.optimizer_parameters() if p.grad is not None)
            attempts.append({"loss_scale": scale, "gradients_finite": bool(finite)})
            if finite:
                return torch.amp.GradScaler("cuda", init_scale=scale), attempts
            scale *= 0.5
        raise FloatingPointError(f"AMP calibration exhausted eight attempts: {attempts}")
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(rng)


def synthetic_smoke(config: dict, device: torch.device, batch_size: int) -> dict:
    configure_determinism(config["training"]["seed"])
    model = MarketJEPAV1(config["model"]).to(device).train()
    batch = _move(synthetic_batch(config["model"], batch_size), device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.optimizer_parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"], betas=tuple(training["betas"]), eps=training["eps"])
    amp = device.type == "cuda" and training["amp"]
    scaler, calibration = calibrate_smoke_scaler(model, batch, optimizer, device, amp=amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp):
        output = model(batch)
        loss, _ = jepa_loss(output)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise FloatingPointError("synthetic JEPA loss is nonfinite")
    started = time.perf_counter()
    scaler.scale(loss).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(list(model.optimizer_parameters()), training["gradient_clip_norm"], error_if_nonfinite=True)
    before = next(model.target_minute.parameters()).detach().clone()
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < scale_before:
        raise RuntimeError("synthetic smoke optimizer step skipped by AMP")
    model.update_target(training["ema_tau"])
    expected = before * training["ema_tau"] + next(model.online.minute_market.parameters()).detach() * (1 - training["ema_tau"])
    torch.testing.assert_close(next(model.target_minute.parameters()), expected)
    if not all(not p.requires_grad and p.grad is None for p in model.target_minute.parameters()):
        raise AssertionError("EMA target is not frozen")
    if torch.equal(before, next(model.target_minute.parameters())):
        raise AssertionError("EMA target did not change after optimizer step")
    return {"status": "PASS", "device": str(device), "batch_size": batch_size, "minute_length": 512,
            "daily_length": 32, "weekly_length": 8, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
            "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "ema_updated": True, "target_gradients_none": True,
            "amp_initial_scale_calibration": calibration, "amp_final_scale": scaler.get_scale(),
            "timing_note": "Single development step, includes first-use overhead; other GPU workloads are not controlled."}


def _training_prefix_cutoff(config: dict) -> str:
    """Read only timestamps until the bounded Train prefix is known."""
    start, end = map(pd.Timestamp, config["data"]["splits"]["train"])
    dates = set()
    with Path(config["data"]["csv_path"]).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = pd.Timestamp(row["Date"])
            day = stamp.normalize()
            if day > end:
                break
            if start <= day and stamp.hour < 18:
                if len(dates) == config["development"]["train_days"] and day not in dates:
                    break
                dates.add(day)
    if len(dates) < 2:
        raise ValueError("not enough daytime observations in the training prefix")
    return max(dates).strftime("%Y-%m-%d")


def real_data_smoke(config: dict, device: torch.device, output: Path) -> dict:
    if not Path(config["data"]["csv_path"]).is_file():
        return {"status": "UNAVAILABLE", "reason": "configured Train CSV does not exist"}
    value = deepcopy(config)
    cutoff = _training_prefix_cutoff(value)
    value["training"].update(batch_size=value["development"]["batch_size"], gradient_accumulation=1,
                             max_epochs=1, checkpoint_dir=str(output / "checkpoints"))
    configure_determinism(value["training"]["seed"])
    data = prepare_market_data(value, max_trading_day=cutoff)
    train = MarketDataset(data, value, "train")
    needed = value["development"]["optimizer_steps"] * value["training"]["batch_size"]
    if len(train) < needed:
        raise ValueError(f"Train prefix has {len(train)} anchors; {needed} needed for the smoke")
    selection = np.linspace(0, len(train) - 1, needed, dtype=np.int64)
    train = MarketDataset(data, value, "train", train.indices[selection])
    if data.minute["trading_day"].max() > pd.Timestamp(value["data"]["splits"]["train"][1]):
        raise AssertionError("nontraining data materialized")
    if np.any(train.indices + max(value["data"]["horizons"]) >= len(data.minute)):
        raise AssertionError("training targets exceed the bounded prefix")
    model = MarketJEPAV1(value["model"])
    trainer = V1Trainer(model, value, train, source_sha256=sha256(value["data"]["csv_path"]),
                        preflight_metadata={"purpose": "architecture_smoke", "train_prefix_cutoff": cutoff}, device=device)
    calibration_batch = _move(adapt_market_batch(collate_market_batch(
        [train[index] for index in range(value["training"]["batch_size"])]), value["model"]), device)
    trainer.scaler, calibration = calibrate_smoke_scaler(model, calibration_batch, trainer.optimizer, device, amp=trainer.amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history = trainer.fit()
    elapsed = time.perf_counter() - started
    if trainer.global_step != value["development"]["optimizer_steps"] or trainer.skipped_optimizer_steps:
        raise RuntimeError("real smoke did not complete its exact successful optimizer-step budget")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("real smoke produced nonfinite parameters")
    path = trainer.checkpoint_dir / "last.pt"
    state = load_v1_checkpoint(path)
    restored = model_from_checkpoint(state).to(device).eval()
    restored_trainer = V1Trainer(restored, value, train, source_sha256=state["source_sha256"],
                                preflight_metadata=state["preflight"], device=device)
    restored_trainer.resume(state)
    batch = _move(adapt_market_batch(collate_market_batch([train[0], train[1]]), value["model"]), device)
    model.eval()
    with torch.inference_mode():
        expected, actual = model(batch), restored(batch)
    for key in ("z_market",):
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    for key in ("targets", "predictions"):
        for h in model.horizons:
            torch.testing.assert_close(expected[key][h], actual[key][h], rtol=0, atol=0)
    return {"status": "PASS", "device": str(device), "optimizer_steps": trainer.global_step,
            "skipped_optimizer_steps": trainer.skipped_optimizer_steps, "batch_size": value["training"]["batch_size"],
            "amp_initial_scale_calibration": calibration, "amp_final_scale": trainer.scaler.get_scale(),
            "data_profile": "existing V0 upstream features; IMC preprocessing is not implemented by V1",
            "population": "JM Train only", "cutoff": cutoff, "materialized_rows": len(data.minute), "selected_anchors": len(train),
            "normalizer_fit_end": cutoff, "validation_evaluated": False, "held_out_RB_read": False,
            "elapsed_seconds": elapsed, "max_preclip_gradient_norm": trainer.max_preclip_gradient_norm,
            "checkpoint": str(path), "checkpoint_roundtrip": "PASS", "optimizer_resume_step": restored_trainer.global_step,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "final_metrics": history[-1]["train"]}


def run_development(config: dict, output: Path, device_name: str = "auto", *, run_tests: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    counts = parameter_counts(config)
    write_json(output / "parameter_counts.json", counts)
    cuda_available = torch.cuda.is_available()
    cuda_error = None
    if not cuda_available:
        try:
            torch.cuda.init()
        except Exception as error:
            cuda_error = f"{type(error).__name__}: {error}"
    if device_name == "cuda" and not cuda_available:
        raise RuntimeError(f"requested CUDA is unavailable: {cuda_error}")
    device = torch.device("cuda" if device_name == "auto" and cuda_available else "cpu" if device_name == "auto" else device_name)
    smoke = {"torch_version": torch.__version__, "cuda_build_version": torch.version.cuda,
             "cuda_available": cuda_available, "cuda_initialization_error": cuda_error,
             "cuda_smoke_status": "PENDING" if device.type == "cuda" else "NOT_VERIFIED", "synthetic": []}
    write_json(output / "smoke_test.json", smoke)
    for size in (2, 8):
        smoke["synthetic"].append(synthetic_smoke(config, device, size))
        print(f"synthetic B={size} PASS", flush=True)
        write_json(output / "smoke_test.json", smoke)
    if device.type == "cuda":
        smoke["cuda_smoke_status"] = "PASS"
    smoke["real_data"] = real_data_smoke(config, device, output)
    write_json(output / "smoke_test.json", smoke)
    tests = {"status": "NOT_RUN", "command": [sys.executable, "-m", "market_jepa_v1.test_suite", "-q", f"--junitxml={output / 'test_results.xml'}"],
             "worker_wait_timeout_seconds": 15, "assertions_skipped_by_runner": 0}
    if run_tests:
        with (output / "test_results.txt").open("w", encoding="utf-8") as handle:
            try:
                process = subprocess.run(tests["command"], stdout=handle, stderr=subprocess.STDOUT, check=False, timeout=300)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                handle.write("\nTest suite exceeded the 300-second development deadline.\n")
        tests.update(status="PASS" if returncode == 0 else "FAIL", returncode=returncode,
                     output_tail=(output / "test_results.txt").read_text(encoding="utf-8").splitlines()[-10:])
        if (output / "test_results.xml").is_file():
            suites = ET.parse(output / "test_results.xml").getroot().iter("testsuite")
            totals = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
            for suite in suites:
                for name in totals:
                    totals[name] += int(suite.attrib.get(name, 0))
            tests.update(totals)
            tests["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
            cases = list(ET.parse(output / "test_results.xml").getroot().iter("testcase"))
            tests["failed_cases"] = [f"{case.attrib.get('classname')}.{case.attrib.get('name')}" for case in cases
                                     if case.find("failure") is not None or case.find("error") is not None]
            v1_cases = [case for case in cases if case.attrib.get("classname", "").startswith("tests.test_market_jepa_v1")]
            tests["v1_test_count"] = len(v1_cases)
            tests["v1_status"] = "PASS" if v1_cases and all(
                case.find(tag) is None for case in v1_cases for tag in ("failure", "error", "skipped")
            ) else "FAIL"
    v0_unchanged = manifest_sha256(implementation_manifest()) == V0_MANIFEST_BEFORE
    runtime_gate = not cuda_available or smoke["cuda_smoke_status"] == "PASS"
    passed = tests["status"] == "PASS" and v0_unchanged and smoke["real_data"]["status"] == "PASS" and runtime_gate
    summary = {"status": "V1_ARCHITECTURE_IMPLEMENTATION_PASS" if passed else "V1_ARCHITECTURE_IMPLEMENTATION_FAIL",
               "v0_manifest_unchanged": v0_unchanged, "parameter_counts": counts, "tests": tests,
               "cuda_smoke_status": smoke["cuda_smoke_status"], "real_data_smoke_status": smoke["real_data"]["status"],
               "formal_benchmark_started": False, "formal_imc_benchmark_ready": False,
               "limitations": ["IMC-compatible tensor contract; production multiscale IMC preprocessing/benchmark remains upstream.",
                               "Terminal round-2 feedback was removed because it had no path to belief or JEPA loss.",
                               "Attention weights are not causal importance; audits use interventions and gradients."]}
    if smoke["cuda_smoke_status"] != "PASS":
        summary["limitations"].append(f"CUDA smoke not verified in this execution environment: {cuda_error}")
    if tests["status"] != "PASS":
        summary["limitations"].append(f"Full V0 regression gate has not passed; failing cases: {tests.get('failed_cases', [])}. See test_results.txt for runtime evidence.")
    summary["architecture_gates"] = {name: tests.get("v1_status", "NOT_VERIFIED") for name in (
        "market_context_early_interaction", "precompression_cross_scale_interaction", "daily_weekly_minute_feedback",
        "market_only_target", "no_identity_embedding", "gradient_connectivity", "checkpoint_roundtrip")}
    summary["architecture_gates"].update(v0_regression="PASS" if v0_unchanged and tests["status"] == "PASS" else "FAIL",
                                          smoke_training=smoke["real_data"]["status"] if smoke["real_data"]["status"] != "UNAVAILABLE" else "PASS_SYNTHETIC_ONLY")
    write_json(output / "summary.json", summary)
    questions = [
        f"1. V0 preserved: {v0_unchanged}; frozen manifest {V0_MANIFEST_BEFORE}.",
        "2. Market × Context: separate additive projections before the minute Transformer; covered by intervention/gradient tests.",
        "3. All three local token sequences enter two rounds before BELIEF compression; tested.",
        "4. Daily changes minute tokens after first feedback; tested before second State read.",
        "5. Weekly changes minute tokens after first feedback; tested before second State read.",
        "6. Target takes only future minute market and padding; context/period/feedback parameters are outside EMA.",
        "7. No commodity identity input or embedding; adapter removes metadata, direct unknown model inputs are rejected.",
        f"8. Trainable parameters: V0 {counts['v0_trainable']:,}; V1 {counts['v1_trainable']:,}; ratio {counts['ratio']:.6f}.",
        f"9. CUDA smoke: {smoke['cuda_smoke_status']}; peak VRAM: " + (
            str(max(row["peak_allocated_bytes"] for row in smoke["synthetic"])) + " allocated bytes"
            if smoke["cuda_smoke_status"] == "PASS" else "not measured"),
        f"10. Complete test suite: {tests['status']}; see test_results.txt for individual failures/skips and counts.",
        "11. Formal IMC + cross-commodity benchmark: NOT started; upstream integration and any unavailable runtime gates remain prerequisites.",
    ]
    (output / "summary.md").write_text("# V1 architecture development\n\n" + summary["status"] + "\n\n" + "\n\n".join(questions)
        + "\n\n## Verification limits\n\n" + "\n".join(f"- {item}" for item in summary["limitations"]) + "\n", encoding="utf-8")
    return summary
