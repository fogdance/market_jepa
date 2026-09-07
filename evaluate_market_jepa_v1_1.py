"""Formal V1.1 evaluation commands. RB-Test is an explicit, reviewed one-shot campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from market_jepa_v1_1.evaluation.comparison import compare_runs
from market_jepa_v1_1.evaluation.heldout import freeze, run_heldout
from market_jepa_v1_1.evaluation.runner import evaluate, inventory
from market_jepa_v1_1.evaluation.tracking import log_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--config", required=True, help="resolved training config_snapshot.yaml")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--rb-inventory", action="store_true")
    train = commands.add_parser("train-domain")
    train.add_argument("--checkpoint", required=True); train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    train.add_argument("--device", default="cpu"); train.add_argument("--batch-size", type=int, default=8)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--runs", nargs="+", required=True); comparison.add_argument("--output", required=True)
    comparison.add_argument("--rb-runs", nargs="+")
    frozen = commands.add_parser("freeze-rb")
    frozen.add_argument("--checkpoints", nargs="+", required=True)
    frozen.add_argument("--evaluations", nargs="+", required=True)
    frozen.add_argument("--manifest", required=True); frozen.add_argument("--output", required=True)
    for name in ("rb-dev", "rb-test"):
        rb = commands.add_parser(name)
        rb.add_argument("--freeze", required=True); rb.add_argument("--output", required=True)
        rb.add_argument("--approval"); rb.add_argument("--device", default="cpu")
        rb.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(2)
    if args.command == "manifest":
        result = inventory(yaml.safe_load(Path(args.config).read_text()), args.output, rb=args.rb_inventory)
        print(json.dumps({"sha256": result["sha256"], "anchors": len(result["anchors"])}))
    elif args.command == "train-domain":
        result = evaluate(args.checkpoint, args.manifest, args.output, device=args.device, batch_size=args.batch_size)
        log_evaluation(args.output, result, mode=args.wandb_mode)
    elif args.command == "compare":
        compare_runs(args.runs, args.output, args.rb_runs)
    elif args.command == "freeze-rb":
        freeze(args.checkpoints, args.evaluations, args.manifest, args.output)
    else:
        run_heldout(args.freeze, args.output, partition=args.command.replace("-", "_"),
                    approval=args.approval, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
