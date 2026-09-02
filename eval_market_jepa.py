from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from market_jepa.eval.pipeline import evaluate_exports, save_evaluation
from market_jepa.train import load_checkpoint
from market_jepa.train.checkpoint import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rerun-label", default=None)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    checkpoint_digest = sha256(args.checkpoint)
    report = evaluate_exports(
        args.latents_dir,
        config,
        expected_checkpoint_sha256=checkpoint_digest,
        expected_source_sha256=checkpoint["source_sha256"],
    )
    output = Path(args.output or str(
        Path(config["evaluation"]["output_dir"]) / config["experiment_id"] / "evaluation.json"
    ))
    manifest_path = output.with_name("evaluation_manifest.json")
    if (output.exists() or manifest_path.exists()) and not args.rerun_label:
        raise SystemExit("evaluation output already exists; provide --rerun-label and a new --output")
    if args.rerun_label and output.exists():
        output = output.with_name(f"{output.stem}_{args.rerun_label}{output.suffix}")
        manifest_path = output.with_name(f"evaluation_manifest_{args.rerun_label}.json")
    path = save_evaluation(report, output)
    manifest = {
        "design_version": config["design_version"],
        "experiment_id": config["experiment_id"],
        "profile": config.get("profile", "formal"),
        "rerun_label": args.rerun_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "source_sha256": checkpoint["source_sha256"],
        "latent_sha256": {
            split: sha256(Path(args.latents_dir) / f"{split}.npz")
            for split in ("train", "validation", "test")
        },
        "evaluation": str(path.resolve()),
        "evaluation_sha256": sha256(path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)
    print(manifest_path)
    print(f"decision={report['go_no_go']['decision']}")


if __name__ == "__main__":
    main()
