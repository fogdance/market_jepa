from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import bootstrap
from .protocol import write_json
from .runner import write_csv


def paired_improvement(left_n, left_d, right_n, right_d, records):
    # Identical seed + ordered records couples each resampled contract-day across models.
    left = bootstrap(left_n, left_d, records, return_draws=True)
    right = bootstrap(right_n, right_d, records, return_draws=True)
    result = {}
    for key in ("all", "16", "64", "256"):
        lp = left["all"] if key == "all" else left["horizons"][key]
        rp = right["all"] if key == "all" else right["horizons"][key]
        draws = (left["draws_all"] - right["draws_all"] if key == "all" else
                 left["draws_horizons"][key] - right["draws_horizons"][key])
        result[key] = {"point": lp["point"] - rp["point"],
                       "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975))}
    return result


def comparison_status(variant, gain, skill, rb_gain, rb_skill):
    if rb_gain is None or rb_skill is None:
        return "NOT_EVALUATED_RB_REQUIRED"
    overall = gain["all"]["point"] > 0 and skill["all"]["point"] > 0
    long = any(gain[h]["ci95_low"] > 0 or skill[h]["ci95_low"] > 0 for h in ("64", "256"))
    if variant == "LateFusion":
        if not overall or rb_skill["all"]["ci95_high"] < 0:
            return "FAIL"
        if not long:
            return "PARTIAL"
        # The reviewed text explicitly requires an H16 severe-regression gate,
        # but specifies no quantitative definition. Never silently invent one.
        if gain["16"]["point"] >= 0 and skill["16"]["point"] >= 0:
            return "V1_1_ARCHITECTURE_PASS"
        return "BLOCKED_H16_SEVERE_REGRESSION_THRESHOLD_UNDEFINED"
    if variant == "MinuteOnly":
        return "V1_1_MULTISCALE_PASS" if overall and long and rb_skill["all"]["point"] >= 0 else "NOT_PASS"
    if variant == "NoHistoricalWeekly":
        evidence = long or rb_skill["all"]["ci95_low"] > 0 or rb_gain["all"]["ci95_low"] > 0
        return "V1_1_HISTORY_PASS" if evidence else "NOT_PASS"
    rb_improves = rb_gain["all"]["point"] > 0 and rb_skill["all"]["point"] > 0
    if overall and rb_improves:
        return "V1_1_SCALING_SUPPORTED"
    return "memorization / in-domain scaling" if overall else "SCALING_NOT_SUPPORTED"


def compare_runs(directories, output, rb_directories=None):
    runs = []
    for directory in directories:
        p = Path(directory)
        summary = json.loads((p / "summary.json").read_text())
        protocol = summary["protocol"]
        records = json.loads((p / "paired_records.json").read_text())
        with np.load(p / "paired_errors.npz", allow_pickle=False) as values:
            errors = {key: values[key] for key in values.files}
        runs.append((p, summary, protocol, records, errors))
    reference = runs[0]
    rows, comparisons = [], []
    for run in runs:
        p, summary, protocol, records, errors = run
        for key in ("manifest_sha256", "epochs", "scaler_sha256", "evaluation_code_sha256", "data_manifest_sha256"):
            if protocol[key] != reference[2][key]:
                raise ValueError(f"unmatched comparison: {key}")
        left_training, right_training = dict(protocol["training_protocol"]), dict(reference[2]["training_protocol"])
        for value in (left_training, right_training):
            value["effective_batch"] = value.pop("batch_size") * value.pop("gradient_accumulation")
        if left_training != right_training:
            raise ValueError("unmatched training budget/optimizer/precision")
        if records != reference[3] or not np.array_equal(errors["test_mask"], reference[4]["test_mask"]):
            raise ValueError("comparison anchor alignment differs")
        rows.append({"run": str(p), "model_size": protocol["model_size"], "variant": protocol["variant"],
                     "trainable_params": protocol["trainable_params"],
                     "Gain_ALL": summary["Gain_ALL"]["point"], "Skill_ALL": summary["Skill_ALL"]["point"],
                     "RB_status": summary["RB"]})
        if rb_directories is not None:
            if len(rb_directories) != len(runs):
                raise ValueError("one RB-Test output per Train evaluation required")
            report = json.loads((Path(rb_directories[len(rows) - 1]) / "rb_bootstrap.json").read_text())
            if report["partition"] != "rb_test" or report["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
                raise ValueError("RB-Test checkpoint/split mismatch")
            rows[-1].update(RB_Gain_ALL=report["RB_Gain"]["all"]["point"],
                            RB_Skill_ALL=report["RB_Skill"]["all"]["point"], RB_status=report["status"])
        if run is reference: continue
        test_records = [r for r, flag in zip(records, errors["test_mask"]) if flag]
        a1 = paired_improvement(errors["jepa"], errors["persistence"], reference[4]["jepa"], reference[4]["persistence"], records)
        a2 = paired_improvement(errors["belief_error"], errors["summary_error"],
                                reference[4]["belief_error"], reference[4]["summary_error"], test_records)
        rb_gain, rb_skill = None, None
        if rb_directories is not None:
            if len(rb_directories) != len(runs):
                raise ValueError("one RB-Test output per Train evaluation required")
            rb = []
            for ri in (0, len(rows) - 1):
                directory = Path(rb_directories[ri])
                report = json.loads((directory / "rb_bootstrap.json").read_text())
                if report["partition"] != "rb_test" or report["checkpoint_sha256"] != runs[ri][2]["checkpoint_sha256"]:
                    raise ValueError("RB-Test checkpoint/split mismatch")
                rec = json.loads((directory / "paired_records.json").read_text())
                with np.load(directory / "paired_errors.npz", allow_pickle=False) as data:
                    rb.append((rec, dict(data)))
            if rb[0][0] != rb[1][0]:
                raise ValueError("RB anchors differ")
            rb_gain = paired_improvement(rb[1][1]["jepa"], rb[1][1]["persistence"], rb[0][1]["jepa"], rb[0][1]["persistence"], rb[0][0])
            rb_skill = paired_improvement(rb[1][1]["belief_error"], rb[1][1]["summary_error"], rb[0][1]["belief_error"], rb[0][1]["summary_error"], rb[0][0])
        if protocol["variant"] != "Full":
            if reference[2]["variant"] != "Full" or reference[2]["model_size"] != "S" or protocol["model_size"] != "S":
                raise ValueError("controls must compare to Full S")
            if protocol["variant"] == "LateFusion" and abs(protocol["trainable_params"] / reference[2]["trainable_params"] - 1) > .05:
                raise ValueError("LateFusion parameter mismatch")
            # Controls are reported as Full minus Control.
            def negate(report):
                return {k: {"point": -v["point"], "ci95_low": -v["ci95_high"], "ci95_high": -v["ci95_low"]} for k, v in report.items()}
            a1, a2 = negate(a1), negate(a2)
            rb_gain = None if rb_gain is None else negate(rb_gain)
            rb_skill = None if rb_skill is None else negate(rb_skill)
        elif protocol["trainable_params"] <= reference[2]["trainable_params"]:
            raise ValueError("scaling comparison must order reference small -> larger models")
        comparisons.append({"run": str(p), "relative_to": str(reference[0]), "delta_gain": a1, "delta_skill": a2,
                            "delta_rb_gain": rb_gain, "delta_rb_skill": rb_skill,
                            "direction": "Full-Control" if protocol["variant"] != "Full" else "larger-smaller",
                            "formal_status": comparison_status(protocol["variant"], a1, a2, rb_gain, rb_skill)})
    result = {"rows": rows, "paired_comparisons": comparisons,
              "scaling_status": "SEE_PAIRED_COMPARISONS" if rb_directories else "NOT_EVALUATED_RB_REQUIRED",
              "rb_test_consumed": rb_directories is not None,
              "architecture_gate_blocker": "Reviewed V2 does not quantify H16 severe regression; no invented threshold"}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "scaling_summary.csv", rows)
    write_json(output / "scaling_summary.json", result)
    (output / "scaling_report.md").write_text(json.dumps(result, indent=2) + "\n")
    return result
