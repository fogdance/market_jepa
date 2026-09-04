from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from market_jepa.eval.pipeline import evaluate_validation_exports
from market_jepa.train import load_checkpoint
from market_jepa.train.checkpoint import sha256


def validation_report_markdown(report: dict[str, Any]) -> str:
    decision = "VALIDATION_GO" if report["validation_go"] else "VALIDATION_NO_GO"
    rows = []
    for label, key in (("Prediction", "prediction"), ("kNN", "knn"), ("Probe", "probe")):
        metric = report[key]
        rows.append(
            f"| {label} | {metric['effect']:.10g} | "
            f"[{metric['ci95'][0]:.10g}, {metric['ci95'][1]:.10g}] | "
            f"{'PASS' if metric['pass'] else 'FAIL'} |"
        )
    return "\n".join(
        [
            "# Market-JEPA V0.6.2 Validation",
            "",
            f"Checkpoint: `{report['checkpoint_path']}` (epoch {report['checkpoint_epoch']})",
            "",
            "| Gate | Effect | 95% CI | Result |",
            "| --- | ---: | ---: | --- |",
            *rows,
            "",
            f"Overall: **{decision}**",
            "",
            "Test consumed: **false**",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market-JEPA V0.6.2 Train+Validation-only core evaluation"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation/market_jepa_v0_6_2",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint_path.name != "last.pt":
        raise SystemExit("V0.6.2 Validation requires last.pt")
    if int(checkpoint.get("epoch", -1)) != 49:
        raise SystemExit("V0.6.2 Validation requires checkpoint epoch 49")

    checkpoint_digest = sha256(checkpoint_path)
    report = evaluate_validation_exports(
        args.latents_dir,
        checkpoint["config"],
        expected_checkpoint_sha256=checkpoint_digest,
        expected_source_sha256=checkpoint["source_sha256"],
    )
    report.update(
        checkpoint_epoch=49,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=checkpoint_digest,
    )

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "validation_summary.json"
    report_path = output_dir / "validation_report.md"
    if summary_path.exists() or report_path.exists():
        raise SystemExit("Validation output already exists; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(validation_report_markdown(report), encoding="utf-8")

    decision = "VALIDATION_GO" if report["validation_go"] else "VALIDATION_NO_GO"
    print(summary_path)
    print(report_path)
    print(f"decision={decision}")
    print("test_consumed=false")


if __name__ == "__main__":
    main()
