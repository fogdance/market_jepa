from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from market_jepa.eval.predictive_geometry import evaluate_predictive_geometry
from market_jepa.train import load_checkpoint
from market_jepa.train.checkpoint import sha256


def diagnostic_markdown(report: dict[str, Any]) -> str:
    lines = ["# Predictive Geometry Diagnostic", ""]
    for label, key in (
        ("Z_market kNN", "z_market_knn"),
        ("P64(Z_market) kNN", "p64_knn"),
        ("Multi-horizon predictive state", "multi_horizon_predictive_state_knn"),
    ):
        metric = report[key]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- effect = {metric['effect']:.10g}",
                f"- 95% CI = [{metric['ci95'][0]:.10g}, {metric['ci95'][1]:.10g}]",
                f"- {'PASS' if metric['pass'] else 'FAIL'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Formal V0 result: **VALIDATION_NO_GO** (unchanged)",
            "",
            "Test consumed = **false**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train+Validation predictive geometry diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation/market_jepa_v0_6_2",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint_path.name != "last.pt" or int(checkpoint.get("epoch", -1)) != 49:
        raise SystemExit("diagnostic requires epoch 49 last.pt")

    validation_summary = json.loads(Path(args.validation_summary).read_text(encoding="utf-8"))
    if validation_summary.get("test_consumed") is not False:
        raise SystemExit("validation summary must state test_consumed=false")
    if validation_summary.get("validation_go") is not False:
        raise SystemExit("diagnostic expects the preserved V0 VALIDATION_NO_GO result")

    checkpoint_digest = sha256(checkpoint_path)
    report = evaluate_predictive_geometry(
        args.latents_dir,
        checkpoint["config"],
        validation_summary["knn"],
        expected_checkpoint_sha256=checkpoint_digest,
        expected_source_sha256=checkpoint["source_sha256"],
    )
    report.update(
        checkpoint_epoch=49,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=checkpoint_digest,
    )

    output_dir = Path(args.output_dir)
    json_path = output_dir / "predictive_geometry_diagnostic.json"
    markdown_path = output_dir / "predictive_geometry_diagnostic.md"
    if json_path.exists() or markdown_path.exists():
        raise SystemExit("diagnostic output already exists; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(diagnostic_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    print("formal_v0_decision=VALIDATION_NO_GO")
    print("test_consumed=false")


if __name__ == "__main__":
    main()
