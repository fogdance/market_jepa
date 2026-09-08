"""Formal Market-JEPA V1.1 training entrypoint; never evaluates the configured held-out commodity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from market_jepa_v1_1.formal_training import (
    DEFAULT_FORMAL_SCALER_ANCHORS_PER_COMMODITY,
    FORMAL_OUTPUT,
    run_formal_training,
    write_failure_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1_1/market_jepa_v1_1.yaml")
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--scaler-anchors-per-commodity", type=int,
        default=DEFAULT_FORMAL_SCALER_ANCHORS_PER_COMMODITY,
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    if args.scaler_anchors_per_commodity <= 0:
        parser.error("scaler anchors per commodity must be positive")
    torch.set_num_threads(args.threads)
    try:
        summary = run_formal_training(
            config_path=args.config,
            output=args.output,
            resume=args.resume,
            scaler_anchors_per_commodity=args.scaler_anchors_per_commodity,
        )
    except Exception as error:
        write_failure_summary(args.output, error)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] not in {
        "V1_1_FIXED_SAMPLE_BUDGET_TRAINING_PASS",
        "INTERRUPTED_RECOVERY_CHECKPOINT_SAVED",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
