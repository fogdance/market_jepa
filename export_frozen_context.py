from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from market_jepa.data import CONTEXT_FEATURES, MARKET_FEATURES, NormalizerBundle, prepare_market_data
from market_jepa.data.dataset import build_split_indices
from market_jepa.data.preflight import file_sha256
from market_jepa.model import MarketJEPA
from market_jepa.train import load_checkpoint
from market_jepa.train.checkpoint import sha256


ALLOWED_SPLITS = ("train", "validation")


class FrozenContextDataset(Dataset[dict[str, Any]]):
    def __init__(self, data, config: dict[str, Any], split: str) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError("V0.7-Frozen context export allows only Train and Validation")
        self.data = data
        self.context_length = int(config["data"]["minute_context_length"])
        self.indices = build_split_indices(data, config, split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        anchor = int(self.indices[item])
        start = anchor - self.context_length + 1
        daily_tokens = int(self.data.daily_position[anchor]) + 1
        weekly_tokens = int(self.data.weekly_position[anchor]) + 1
        progress = np.concatenate(
            [
                self.data.minute_context_raw[anchor],
                np.asarray(
                    [
                        self.data.daily_partial.iloc[anchor]["source_bar_count"],
                        daily_tokens,
                        self.data.weekly_partial.iloc[anchor]["source_bar_count"],
                        weekly_tokens,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return {
            "minute_context": torch.from_numpy(
                self.data.minute_context[start : anchor + 1]
            ),
            "context_progress": torch.from_numpy(progress),
            "anchor_index": anchor,
            "timestamp_ns": int(self.data.minute.iloc[anchor]["timestamp"].value),
            "trading_day_ns": int(self.data.minute.iloc[anchor]["trading_day"].value),
        }


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_context_split(
    model: MarketJEPA,
    dataset: FrozenContextDataset,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    metadata: dict[str, str],
) -> Path:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    before = _state_digest(model)
    model.eval().requires_grad_(False)
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "z_context",
            "context_progress",
            "anchor_index",
            "timestamp_ns",
            "trading_day_ns",
        )
    }
    with torch.inference_mode():
        for batch in loader:
            context = batch["minute_context"].to(device)
            arrays["z_context"].append(
                model.online.minute_context(context).cpu().numpy()
            )
            for name in arrays.keys() - {"z_context"}:
                arrays[name].append(batch[name].numpy())
    after = _state_digest(model)
    if before != after:
        raise RuntimeError("frozen Market-JEPA parameters changed during context export")

    payload = {name: np.concatenate(values) for name, values in arrays.items()}
    payload.update({name: np.asarray([value]) for name, value in metadata.items()})
    payload["jepa_state_sha256"] = np.asarray([before])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen context-only state")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint_path.name != "last.pt" or int(checkpoint.get("epoch", -1)) != 49:
        raise SystemExit("V0.7-Frozen requires epoch 49 last.pt")
    config = checkpoint["config"]
    if file_sha256(config["data"]["csv_path"]) != checkpoint["source_sha256"]:
        raise SystemExit("source hash differs from checkpoint")
    data = prepare_market_data(
        config, normalizers=NormalizerBundle.from_dict(checkpoint["normalizers"])
    )
    model = MarketJEPA(
        len(MARKET_FEATURES),
        len(CONTEXT_FEATURES),
        config["data"]["horizons"],
        config["model"],
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    dataset = FrozenContextDataset(data, config, args.split)
    output = export_context_split(
        model,
        dataset,
        Path(args.output_dir) / f"context_{args.split}.npz",
        device,
        args.batch_size,
        {
            "split": args.split,
            "checkpoint_sha256": sha256(checkpoint_path),
            "source_sha256": checkpoint["source_sha256"],
        },
    )
    print(output)
    print("test_consumed=false")


if __name__ == "__main__":
    main()
