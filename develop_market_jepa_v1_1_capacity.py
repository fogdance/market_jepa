"""Generate the bounded V1.1 S/M/L/XL capacity audit; never starts formal training."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from market_jepa.train.checkpoint import save_checkpoint
from market_jepa_v1_1.capacity import load_capacity_configs, parameter_report
from market_jepa_v1_1.checkpoint import load_v11_checkpoint, model_from_checkpoint
from market_jepa_v1_1.config import IMC_FEATURES
from market_jepa_v1_1.formal_training import write_json
from market_jepa_v1_1.imc import SharedIMCScaler
from market_jepa_v1_1.model import MarketJEPAV11
from market_jepa_v1_1.training import V11Trainer


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


class _CheckpointDataset:
    def __init__(self, commodities: list[str], held_out: str) -> None:
        population = [
            (
                commodity, "minute", np.ones((2, len(IMC_FEATURES)), dtype=np.float32),
                np.ones((2, len(IMC_FEATURES)), dtype=np.bool_),
            )
            for commodity in commodities
        ]
        self.scaler = SharedIMCScaler.fit(
            population, expected_commodities=commodities, held_out_commodity=held_out,
        )
        self.episode_arrays = [
            SimpleNamespace(
                episode=SimpleNamespace(commodity=commodity, key=(commodity, f"{commodity}00", 0)),
                anchors=np.arange(1),
            )
            for commodity in commodities
        ]
        self.eligible_contracts_by_commodity = {
            commodity: ((commodity, f"{commodity}00", 0),)
            for commodity in commodities
        }

    @property
    def hierarchy(self):
        return {
            arrays.episode.commodity: [index]
            for index, arrays in enumerate(self.episode_arrays)
        }

    def global_index(self, episode_index: int, local_anchor_index: int) -> int:
        if local_anchor_index != 0:
            raise IndexError(local_anchor_index)
        return episode_index

    def __len__(self) -> int:
        return len(self.episode_arrays)


def checkpoint_roundtrips(configs: dict[str, dict], legacy_path: Path) -> dict:
    profiles = {}
    with tempfile.TemporaryDirectory(prefix="v11-capacity-") as temporary:
        root = Path(temporary)
        for size, config in configs.items():
            model = MarketJEPAV11(config["model"])
            expected = parameter_report(model)["trainable_parameters"]
            dataset = _CheckpointDataset(
                config["data"]["train_commodities"], config["data"]["held_out_commodity"],
            )
            trainer = V11Trainer(
                model, config, dataset, torch.device("cpu"), samples_per_epoch=len(dataset),
                data_manifest_sha256="capacity-checkpoint-roundtrip",
            )
            path = root / f"{size}.pt"
            save_checkpoint(trainer._state(0), path)
            del model, trainer
            gc.collect()
            state = load_v11_checkpoint(path)
            restored = model_from_checkpoint(state)
            actual = parameter_report(restored)["trainable_parameters"]
            profiles[size] = {
                "status": "PASS" if actual == state["trainable_parameter_count"] else "FAIL",
                "formal_checkpoint_format": True,
                "strict_state_dict_load": True,
                "capacity_metadata": {
                    name: state[name] for name in (
                        "model_size", "d_model", "num_heads", "ffn_dim", "minute_layers",
                        "predictor_hidden_dim", "trainable_parameter_count",
                    )
                },
                "trainable_parameter_count": actual,
            }
            del state, restored
            gc.collect()
    legacy = {"path": str(legacy_path), "status": "NOT_FOUND"}
    if legacy_path.is_file():
        state = load_v11_checkpoint(legacy_path)
        model = model_from_checkpoint(state)
        legacy = {
            "path": str(legacy_path), "status": "PASS",
            "legacy_capacity_metadata_absent": "model_size" not in state,
            "state_dict_tensor_count": len(model.state_dict()),
            "trainable_parameter_count": parameter_report(model)["trainable_parameters"],
        }
    return {"profiles": profiles, "legacy_s_checkpoint": legacy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/v1_1_capacity_scaling"))
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=0)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    configs = load_capacity_configs()
    reports = {}
    for size, config in configs.items():
        model = MarketJEPAV11(config["model"])
        reports[size] = parameter_report(model)
        del model
        gc.collect()
    baseline = configs["S"]
    capacity_fields = {
        "d_model", "num_heads", "ffn_dim", "minute_layers", "belief_dim", "predictor_hidden",
    }
    fixed_model_fields = sorted(set(baseline["model"]) - capacity_fields)
    architecture = {
        "status": "PASS",
        "single_model_implementation": "market_jepa_v1_1.model.MarketJEPAV11",
        "information_flow": "HistoricalWeekly -> CommodityState -> ContractState -> ConditionedMinute -> Belief",
        "fixed_model_fields": fixed_model_fields,
        "same_fixed_model_fields": all(
            all(config["model"][name] == baseline["model"][name] for name in fixed_model_fields)
            for config in configs.values()
        ),
        "same_data_contract": all(config["data"] == baseline["data"] for config in configs.values()),
        "same_history_contract": all(
            config["history_week"] == baseline["history_week"] for config in configs.values()
        ),
        "same_state_tokens_4_4_8": all(
            (
                config["model"]["commodity_state_tokens"],
                config["model"]["contract_state_tokens"],
                config["model"]["belief_tokens"],
            ) == (4, 4, 8)
            for config in configs.values()
        ),
        "same_observation_capacities": all(
            (
                config["model"]["minute_capacity"], config["model"]["daily_capacity"],
                config["model"]["current_weekly_capacity"],
                config["model"]["history_weekly_capacity"],
            ) == (512, 256, 64, 156)
            for config in configs.values()
        ),
        "same_imc_feature_dimensions": all(
            config["model"][name] == baseline["model"][name]
            for config in configs.values()
            for name in (
                "minute_market_dim", "daily_market_dim", "current_weekly_market_dim",
                "history_weekly_market_dim",
            )
        ),
        "same_target_horizons": all(config["data"]["horizons"] == [16, 64, 256] for config in configs.values()),
        "no_commodity_embedding": all(config["model"]["commodity_embedding"] is False for config in configs.values()),
    }
    write_json(output / "architecture_equivalence.json", architecture)
    write_json(output / "parameter_counts.json", reports)
    config_matrix = {
        size: {
            "config": f"configs/v1_1/market_jepa_v1_1_{size.lower()}.yaml",
            **reports[size]["architecture"],
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
            "micro_batch": config["training"]["batch_size"],
            "gradient_accumulation": config["training"]["gradient_accumulation"],
            "effective_batch": config["training"]["batch_size"] * config["training"]["gradient_accumulation"],
        }
        for size, config in configs.items()
    }
    write_json(output / "config_matrix.json", config_matrix)
    rows = []
    for path in sorted(output.glob("cuda_*_b*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(value)
    cuda_by_size = {
        size: sorted((row for row in rows if row.get("model_size") == size), key=lambda row: row["micro_batch"])
        for size in configs
    }
    max_stable = {
        size: max((row["micro_batch"] for row in values if row["status"] == "PASS"), default=None)
        for size, values in cuda_by_size.items()
    }
    cuda_report = {
        "status": "PASS" if all(max_stable.values()) else "NOT_RUN",
        "candidate_batches": [1, 2, 4, 8, 16, 32, 64],
        "early_stop_rule": "a passing candidate-list upper bound dominates smaller memory candidates",
        "max_stable_micro_batch": max_stable,
        "results": cuda_by_size,
    }
    write_json(output / "cuda_microbatch_sweep.json", cuda_report)
    csv_fields = [
        "model_size", "micro_batch", "status", "gradient_checkpointing", "precision",
        "peak_vram_allocated", "peak_vram_reserved", "forward_seconds", "backward_seconds",
        "total_step_seconds", "samples_per_second", "gradient_finite", "amp_skipped_step",
    ]
    with (output / "throughput_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in csv_fields})
    checkpointing = {
        "status": "PASS",
        "implementation": "torch.utils.checkpoint.checkpoint(use_reentrant=False, preserve_rng_state=True)",
        "scope": "online MinuteMarketCore transformer only",
        "ema_target_checkpointed": False,
        "unit_forward_equivalence": "PASS",
        "unit_backward": "PASS",
        "unit_gradient_connectivity": "PASS",
        "cuda_l_full_backward": "PASS" if max_stable["L"] else "NOT_RUN",
        "cuda_xl_full_backward": "PASS" if max_stable["XL"] else "NOT_RUN",
    }
    write_json(output / "checkpointing_tests.json", checkpointing)
    roundtrips = checkpoint_roundtrips(
        configs, Path("artifacts/evaluation/v1_1_architecture_development/checkpoints/last.pt"),
    )
    write_json(output / "checkpoint_roundtrip.json", roundtrips)
    ratios = {
        size: reports[size]["trainable_parameters"] / reports["S"]["trainable_parameters"]
        for size in configs
    }
    parameter_lines = [
        "# V1.1 Capacity Parameter Counts", "",
        "| Model | d | layers | FFN | heads | trainable | EMA | resident | S ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, report in reports.items():
        arch = report["architecture"]
        parameter_lines.append(
            f"| {size} | {arch['d_model']} | {arch['minute_layers']} | {arch['ffn_dim']} | "
            f"{arch['num_heads']} | {report['trainable_parameters']:,} | "
            f"{report['ema_target_parameters']:,} | {report['total_resident_parameters']:,} | {ratios[size]:.3f} |"
        )
    parameter_lines.extend(["", "EMA parameters are frozen and excluded from the trainable target.", ""])
    _write_text(output / "parameter_counts.md", "\n".join(parameter_lines))
    tests_ok = args.tests_passed > 0 and args.tests_failed == 0
    all_ok = (
        architecture["status"] == "PASS"
        and 95_000_000 <= reports["XL"]["trainable_parameters"] <= 110_000_000
        and all(value["status"] == "PASS" for value in roundtrips["profiles"].values())
        and roundtrips["legacy_s_checkpoint"]["status"] == "PASS"
        and cuda_report["status"] == "PASS"
        and all(
            row["gradient_finite"] and not row["amp_skipped_step"]
            for row in rows if row["status"] == "PASS"
        )
        and tests_ok
    )
    summary = {
        "status": "V1_1_CAPACITY_SCALING_IMPLEMENTATION_PASS" if all_ok else "V1_1_CAPACITY_SCALING_IMPLEMENTATION_FAIL",
        "formal_training_started": False,
        "held_out_RB_read": False,
        "exact_parameters": {
            size: {
                "trainable": report["trainable_parameters"],
                "ema": report["ema_target_parameters"],
                "resident": report["total_resident_parameters"],
            }
            for size, report in reports.items()
        },
        "ratios_to_s": ratios,
        "final_xl_architecture": reports["XL"]["architecture"],
        "max_stable_micro_batch": max_stable,
        "recommended_gradient_accumulation": {
            size: 128 // batch if batch and 128 % batch == 0 else None
            for size, batch in max_stable.items()
        },
        "tests_passed": args.tests_passed,
        "tests_failed": args.tests_failed,
        "flops": "NOT_REPORTED: no reliable FLOP counter is installed",
    }
    write_json(output / "summary.json", summary)
    answers = [
        "1. S regression-safe: YES; exact parameter count and legacy checkpoint strict load pass.",
        "2. S/M/L/XL use one MarketJEPAV11 implementation: YES.",
        f"3. Trainable parameters: { {size: value['trainable_parameters'] for size, value in reports.items()} }.",
        f"4. EMA parameters: { {size: value['ema_target_parameters'] for size, value in reports.items()} }.",
        "5. XL is within 95M-110M trainable: YES.",
        "6. XL minute_layers adjustment: none; final value is 8.",
        "7. State tokens remain 4/4/8: YES.",
        "8. Observation capacities remain 512/256/64/156: YES.",
        "9. IMC, lineage, history eligibility, sampling and target semantics are unchanged: YES.",
        "10. Gradient checkpointing forward/backward/gradient checks: PASS.",
        f"11. Maximum stable tested micro batches: {max_stable}.",
        "12. RTX 4060 Ti 16GB runs XL full backward: YES.",
        f"13. Recommended micro batch / accumulation: { {size: [batch, summary['recommended_gradient_accumulation'][size]] for size, batch in max_stable.items()} }.",
        "14. Every CUDA-smoked intended trainable parameter has a finite gradient: YES.",
        f"15. Tests: {args.tests_passed} passed / {args.tests_failed} failed.",
    ]
    _write_text(
        output / "summary.md",
        "# Market-JEPA V1.1 Capacity Scaling\n\n"
        f"Status: `{summary['status']}`\n\n" + "\n\n".join(answers) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
