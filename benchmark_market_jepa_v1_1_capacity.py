"""One bounded full-step CUDA smoke for a V1.1 capacity profile; never trains an epoch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from market_jepa.model.jepa import jepa_loss
from market_jepa_v1_1.capacity import PROFILE_CONFIG_PATHS, parameter_report
from market_jepa_v1_1.config import (
    DAILY_CONTEXT_FEATURES, IMC_FEATURES, MINUTE_CONTEXT_FEATURES,
    WEEKLY_CONTEXT_FEATURES, load_v11_config,
)
from market_jepa_v1_1.formal_training import write_json
from market_jepa_v1_1.model import MarketJEPAV11
from market_jepa_v1_1.training import assert_all_trainable_gradients


def synthetic_formal_batch(batch_size: int, device: torch.device) -> dict:
    generator = torch.Generator().manual_seed(911)
    sources = {
        "minute": (512, len(MINUTE_CONTEXT_FEATURES)),
        "daily": (256, len(DAILY_CONTEXT_FEATURES)),
        "current_weekly": (64, len(WEEKLY_CONTEXT_FEATURES)),
        "history_weekly": (156, len(WEEKLY_CONTEXT_FEATURES)),
    }
    result = {}
    for source, (length, context_dim) in sources.items():
        result[f"{source}_market"] = torch.randn(
            batch_size, length, len(IMC_FEATURES), generator=generator,
        ).to(device)
        result[f"{source}_context"] = torch.randn(
            batch_size, length, context_dim, generator=generator,
        ).to(device)
        result[f"{source}_mask"] = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        result[f"{source}_imc_validity"] = torch.ones(
            batch_size, length, len(IMC_FEATURES), dtype=torch.bool, device=device,
        )
    result["history_weekly_contract_boundary"] = torch.zeros(
        batch_size, 156, device=device,
    )
    result["history_weekly_contract_boundary"][:, 0] = 1
    result["target_minute_market"] = {
        horizon: torch.randn(
            batch_size, horizon, len(IMC_FEATURES), generator=generator,
        ).to(device)
        for horizon in (16, 64, 256)
    }
    result["target_minute_imc_validity"] = {
        horizon: torch.ones(
            batch_size, horizon, len(IMC_FEATURES), dtype=torch.bool, device=device,
        )
        for horizon in (16, 64, 256)
    }
    result["target_minute_mask"] = {
        horizon: torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        for horizon in (16, 64, 256)
    }
    return result


def run_smoke(size: str, batch_size: int) -> dict:
    if not torch.cuda.is_available():
        return {"status": "CUDA_NOT_AVAILABLE", "model_size": size, "micro_batch": batch_size}
    config = load_v11_config(PROFILE_CONFIG_PATHS[size])
    device = torch.device("cuda")
    torch.manual_seed(912)
    torch.cuda.manual_seed_all(912)
    model = MarketJEPAV11(config["model"]).to(device).train()
    model.set_gradient_checkpointing(config["training"]["gradient_checkpointing"])
    optimizer = torch.optim.AdamW(
        model.optimizer_parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    batch = synthetic_formal_batch(batch_size, device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(**batch)
        loss, metrics = jepa_loss(output)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise FloatingPointError("capacity CUDA smoke loss is nonfinite")
    started = time.perf_counter()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    connectivity = assert_all_trainable_gradients(model)
    torch.nn.utils.clip_grad_norm_(
        list(model.optimizer_parameters()), config["training"]["gradient_clip_norm"],
        error_if_nonfinite=True,
    )
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    amp_skipped = scaler.get_scale() < scale_before
    if not amp_skipped:
        model.update_target(config["training"]["ema_tau"])
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - started
    total_seconds = forward_seconds + backward_seconds + optimizer_seconds
    report = parameter_report(model)
    return {
        "status": "PASS",
        "model_size": size,
        "micro_batch": batch_size,
        "effective_batch_target": 128,
        "recommended_gradient_accumulation_if_selected": 128 // batch_size if 128 % batch_size == 0 else None,
        "precision": "torch.float16 autocast + GradScaler",
        "gradient_checkpointing": config["training"]["gradient_checkpointing"],
        "loss": float(loss.detach()),
        "prediction_losses": {
            name: float(value) for name, value in metrics.items() if name.startswith("prediction_loss_h")
        },
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_ema_seconds": optimizer_seconds,
        "total_step_seconds": total_seconds,
        "samples_per_second": batch_size / total_seconds,
        "peak_vram_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved": int(torch.cuda.max_memory_reserved()),
        "gradient_finite": not connectivity["missing"] and not connectivity["nonfinite"],
        "gradient_connectivity": connectivity,
        "amp_skipped_step": bool(amp_skipped),
        "target_requires_grad": any(parameter.requires_grad for parameter in model.target_minute.parameters()),
        "parameter_counts": {
            name: report[name] for name in (
                "trainable_parameters", "ema_target_parameters", "total_resident_parameters",
            )
        },
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=tuple(PROFILE_CONFIG_PATHS), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        result = run_smoke(args.size, args.batch_size)
    except torch.cuda.OutOfMemoryError as error:
        result = {
            "status": "OOM", "model_size": args.size, "micro_batch": args.batch_size,
            "error": str(error),
        }
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] not in {"PASS", "OOM", "CUDA_NOT_AVAILABLE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
