from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from market_jepa.config import load_config, smoke_config
from market_jepa.data import CONTEXT_FEATURES, MARKET_FEATURES, MarketDataset, prepare_market_data, run_preflight
from market_jepa.data.preflight import file_sha256
from market_jepa.model import MarketJEPA
from market_jepa.train import Trainer, configure_determinism, load_checkpoint


def _limit(dataset: MarketDataset, maximum: int) -> MarketDataset:
    if len(dataset) <= maximum:
        return dataset
    positions = np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64)
    return MarketDataset(dataset.data, dataset.config, dataset.split, dataset.indices[positions])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market_jepa_v0.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if args.smoke and args.cuda_smoke:
        parser.error("choose only one smoke profile")
    config = load_config(args.config)
    if args.smoke or args.cuda_smoke:
        config = smoke_config(config)
        if args.cuda_smoke:
            config["experiment_id"] = config["experiment_id"].replace("_smoke", "_cuda_smoke")
            config["training"]["amp"] = True
    elif not torch.cuda.is_available():
        raise SystemExit("Formal V0 requires a passing CUDA deterministic smoke; use --smoke for CPU verification")
    if args.cuda_smoke and not torch.cuda.is_available():
        raise SystemExit("CUDA smoke requested but torch.cuda.is_available() is false")
    configure_determinism(config["training"]["seed"])
    data = prepare_market_data(config)
    artifact_root = Path("artifacts") / config["experiment_id"]
    report, json_path, csv_path = run_preflight(data, config, artifact_root / "preflight")
    print(f"preflight_json={json_path} preflight_csv={csv_path}")
    train = MarketDataset(data, config, "train")
    validation = MarketDataset(data, config, "validation")
    if args.smoke or args.cuda_smoke:
        maximum = config["smoke"]["max_samples_per_split"]
        train = _limit(train, max(maximum, config["training"]["batch_size"]))
        validation = _limit(validation, maximum)
    model = MarketJEPA(len(MARKET_FEATURES), len(CONTEXT_FEATURES), config["data"]["horizons"], config["model"])
    trainer = Trainer(
        model,
        config,
        train,
        validation,
        source_sha256=file_sha256(config["data"]["csv_path"]),
        preflight_metadata={"json": str(json_path), "csv": str(csv_path), "source": report["source"]},
        device=torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu"),
    )
    if args.resume:
        trainer.resume(load_checkpoint(args.resume))
    trainer.fit()


if __name__ == "__main__":
    main()
