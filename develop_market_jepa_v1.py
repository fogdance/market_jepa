"""Bounded architecture verification only; never launches a formal experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from market_jepa_v1 import load_v1_config
from market_jepa_v1.development import run_development, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1/market_jepa_v1.yaml")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/v1_architecture_development"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    if (args.output / "summary.json").exists():
        parser.error("development report already exists; supply a new --output to preserve it")
    torch.set_num_threads(args.threads)
    try:
        result = run_development(load_v1_config(args.config), args.output, args.device)
    except Exception as error:
        write_json(args.output / "summary.json", {"status": "V1_ARCHITECTURE_IMPLEMENTATION_FAIL",
                   "error": f"{type(error).__name__}: {error}"})
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "V1_ARCHITECTURE_IMPLEMENTATION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
