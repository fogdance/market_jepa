from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from market_jepa.data import CONTEXT_FEATURES, MARKET_FEATURES, MarketDataset, NormalizerBundle, collate_market_batch, prepare_market_data
from market_jepa.data.preflight import file_sha256
from market_jepa.eval.raw_baseline import raw_baseline_features, raw_feature_names
from market_jepa.model import MarketJEPA
from market_jepa.train import load_checkpoint
from market_jepa.train.checkpoint import sha256


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def export_split(
    model: MarketJEPA,
    dataset: MarketDataset,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    provenance: dict[str, str] | None = None,
) -> Path:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_market_batch)
    arrays: dict[str, list[np.ndarray]] = {}

    def append(name: str, value: torch.Tensor | np.ndarray) -> None:
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        arrays.setdefault(name, []).append(array)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            anchors = batch["anchor_index"].numpy()
            device_batch = _move(batch, device)
            output = model(device_batch)
            persistence = model.encode_persistence(device_batch)
            append("anchor_index", anchors)
            append("timestamp_ns", batch["timestamp_ns"].numpy())
            append("trading_day_ns", batch["trading_day_ns"].numpy())
            append("z_market", output["z_market"])
            raw = np.stack(
                [
                    raw_baseline_features(
                        dataset.data,
                        int(anchor),
                        dataset.context_length,
                        model.online.ablation,
                    )
                    for anchor in anchors
                ]
            )
            append("raw_features", raw)
            daily = dataset.data.daily_partial.iloc[anchors]
            weekly = dataset.data.weekly_partial.iloc[anchors]
            append("daily_partial_source_min", daily["source_min_index"].to_numpy())
            append("daily_partial_source_max", daily["source_max_index"].to_numpy())
            append("weekly_partial_source_min", weekly["source_min_index"].to_numpy())
            append("weekly_partial_source_max", weekly["source_max_index"].to_numpy())
            for horizon in model.horizons:
                append(f"z_prediction_h{horizon}", output["predictions"][horizon])
                append(f"z_target_h{horizon}", output["targets"][horizon])
                append(f"z_persistence_h{horizon}", persistence[horizon])
                append(f"outcomes_h{horizon}", batch["outcomes"][horizon].numpy())
    payload = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    payload.update(
        raw_feature_names=np.asarray(raw_feature_names(model.online.ablation)),
        symbol=np.asarray([dataset.config["data"]["symbol"]] * len(dataset)),
        split=np.asarray([dataset.split] * len(dataset)),
        design_version=np.asarray([dataset.config["design_version"]]),
        ablation=np.asarray([model.online.ablation]),
    )
    if provenance:
        payload.update({name: np.asarray([value]) for name, value in provenance.items()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if file_sha256(config["data"]["csv_path"]) != checkpoint["source_sha256"]:
        raise SystemExit("source hash differs from checkpoint")
    normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
    data = prepare_market_data(config, normalizers=normalizers)
    model = MarketJEPA(
        len(MARKET_FEATURES), len(CONTEXT_FEATURES), config["data"]["horizons"], config["model"]
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    output_dir = Path(args.output_dir or f'artifacts/latents/{config["experiment_id"]}')
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    provenance = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_sha256": checkpoint["source_sha256"],
    }
    for split in splits:
        dataset = MarketDataset(data, config, split)
        if config.get("profile") == "smoke":
            maximum = config["smoke"]["max_samples_per_split"]
            if len(dataset) > maximum:
                positions = np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64)
                dataset = MarketDataset(data, config, split, dataset.indices[positions])
        path = export_split(
            model,
            dataset,
            output_dir / f"{split}.npz",
            device,
            config["training"]["batch_size"],
            provenance,
        )
        print(path)


if __name__ == "__main__":
    main()
