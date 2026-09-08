from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import bootstrap
from .protocol import write_json, matched_control_gate
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
        if (not overall or rb_skill["all"]["ci95_high"] < 0
                or any(metric["16"]["ci95_high"] < 0 for metric in (gain, skill))):
            return "FAIL"
        if not long:
            return "PARTIAL"
        return "V1_1_ARCHITECTURE_PASS"
    if variant == "MinuteOnly":
        return "V1_1_MULTISCALE_PASS" if overall and long and rb_skill["all"]["ci95_high"] >= 0 else "NOT_PASS"
    if variant == "NoHistoricalWeekly":
        evidence = long or rb_skill["all"]["ci95_low"] > 0 or rb_gain["all"]["ci95_low"] > 0
        negative_transfer = any(metric["all"]["ci95_high"] < 0 for metric in (gain, skill, rb_gain, rb_skill))
        return "V1_1_HISTORY_PASS" if evidence and not negative_transfer else "NOT_PASS"
    rb_improves = rb_gain["all"]["point"] > 0 and rb_skill["all"]["point"] > 0
    if overall and rb_improves:
        return "CAPACITY_BENEFIT_VS_S"
    return "memorization / in-domain scaling" if overall else "SCALING_NOT_SUPPORTED"


def compare_runs(directories, output, rb_directories=None):
    if not directories:
        raise ValueError("comparison requires evaluation runs")
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
    if reference[2]["variant"] != "Full" or reference[2]["model_size"] != "S":
        raise ValueError("comparison reference must be Full S")
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
            matched_control_gate(protocol, reference[2])
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
                            "formal_status": comparison_status(protocol["variant"], a1, a2, rb_gain, rb_skill),
                            "warnings": ["H16 negative point, uncertain (CI includes zero)"] if any(
                                metric["16"]["point"] < 0 <= metric["16"]["ci95_high"] for metric in (a1, a2)) else []})
    adjacent = []
    full_indices = sorted([i for i, r in enumerate(runs) if r[2]["variant"] == "Full"],
                          key=lambda i: runs[i][2]["trainable_params"])
    for small, large in zip(full_indices, full_indices[1:]):
        left, right = runs[large], runs[small]
        rec = left[3]
        test_records = [r for r, flag in zip(rec, left[4]["test_mask"]) if flag]
        gain = paired_improvement(left[4]["jepa"], left[4]["persistence"], right[4]["jepa"], right[4]["persistence"], rec)
        skill = paired_improvement(left[4]["belief_error"], left[4]["summary_error"], right[4]["belief_error"], right[4]["summary_error"], test_records)
        rb_gain = rb_skill = None
        if rb_directories is not None:
            rb = []
            for i in (large, small):
                directory = Path(rb_directories[i])
                with np.load(directory / "paired_errors.npz", allow_pickle=False) as data:
                    rb.append(dict(data))
            rec = json.loads((Path(rb_directories[large]) / "paired_records.json").read_text())
            rb_gain = paired_improvement(rb[0]["jepa"], rb[0]["persistence"], rb[1]["jepa"], rb[1]["persistence"], rec)
            rb_skill = paired_improvement(rb[0]["belief_error"], rb[0]["summary_error"], rb[1]["belief_error"], rb[1]["summary_error"], rec)
        adjacent.append({"from": right[2]["model_size"], "to": left[2]["model_size"],
                         "delta_gain": gain, "delta_skill": skill, "delta_rb_gain": rb_gain, "delta_rb_skill": rb_skill})
    result = {"rows": rows, "paired_comparisons": comparisons,
              "adjacent_pairs": adjacent,
              "scaling_status": scaling_sequence_status([runs[i][2]["model_size"] for i in full_indices], adjacent),
              "rb_test_consumed": rb_directories is not None,
              "h16_severe_regression": "paired 95% CI upper < 0"}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "scaling_summary.csv", rows)
    write_json(output / "scaling_summary.json", result)
    (output / "scaling_report.md").write_text(json.dumps(result, indent=2) + "\n")
    return result


def scaling_sequence_status(sizes, adjacent):
    if sizes != ["S", "M", "L", "XL"] or len(adjacent) != 3:
        return "INCOMPLETE_SCALING_COHORT"
    if any(p["delta_rb_gain"] is None or p["delta_rb_skill"] is None for p in adjacent):
        return "NOT_EVALUATED_RB_REQUIRED"
    train_up = all(p[k]["all"]["point"] > 0 for p in adjacent for k in ("delta_gain", "delta_skill"))
    rb_up = all(p[k]["all"]["point"] > 0 for p in adjacent for k in ("delta_rb_gain", "delta_rb_skill"))
    if train_up and rb_up:
        return "V1_1_SCALING_SUPPORTED"
    return "IN_DOMAIN_CAPACITY_SCALING_ONLY" if train_up else "SCALING_MIXED"
