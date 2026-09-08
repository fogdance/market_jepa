"""Stage-A strict temporal core audit. Extended attribution and RB are deferred."""
import json
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import load_v11_checkpoint, model_from_checkpoint
from ..imc import SharedIMCScaler
from ..temporal import STAGE_A_CONFIG, isolation_gate, validate_split
from .latent_health import latent_health
from .metrics import bootstrap, cosine_loss
from .protocol import (PROTOCOL, HORIZONS, balanced_select, digest, write_json, file_hash,
                       completed_run_gate, final_checkpoint_gate, semantic_implementation_gate,
                       verify_data_files, evaluation_code_hash)
from .runner import dataset_for, extract, _probe_results, write_csv


def stage_a_status(isolation, a0, a1, a2):
    if isolation != "PASS" or a0 != "PASS":
        return "STAGE_A_BLOCKED"
    if "FAIL" in (a1, a2):
        return "STAGE_A_NO_GO"
    return "STAGE_A_GO" if a1 == a2 == "PASS" else "STAGE_A_INCONCLUSIVE"


def predictive_gate(report):
    aggregate, horizons = report["all"], report["horizons"]
    if aggregate["ci95_high"] <= 0 or any(horizons[h]["ci95_high"] < 0 for h in ("64", "256")):
        return "FAIL"
    if aggregate["ci95_low"] > 0 and any(horizons[h]["ci95_low"] > 0 for h in ("64", "256")):
        return "PASS"
    return "INCONCLUSIVE"


def sanity_predictions(targets, records):
    fit = np.asarray([r["split"] == "probe_fit" for r in records])
    test = np.asarray([r["split"] == "probe_test" for r in records])
    pool = np.flatnonzero(fit)
    if not len(pool) or not test.any():
        raise ValueError("Stage-A requires ProbeFit and ProbeTest")
    mean = targets[fit].mean(0)
    rng = np.random.default_rng(42)
    donors = []
    for i in np.flatnonzero(test):
        key = (records[i]["commodity"], records[i]["contract_uid"], records[i]["trading_date"])
        eligible = [j for j in pool if (records[j]["commodity"], records[j]["contract_uid"], records[j]["trading_date"]) != key]
        donors.append(int(rng.choice(eligible or pool)))
    return mean, targets[donors], donors


def a1_reports(exports, records):
    test = np.asarray([r["split"] == "probe_test" for r in records])
    rr = [r for r, flag in zip(records, test) if flag]
    mean, shuffled, donors = sanity_predictions(exports["target_latents"], records)
    target = exports["target_latents"][test]
    mean_loss = cosine_loss(np.broadcast_to(mean, target.shape), target)
    shuffle_loss = cosine_loss(shuffled, target)
    reports = {name: bootstrap(exports["jepa"][test], baseline, rr) for name, baseline in (
        ("Persistence", exports["persistence"][test]), ("TrainMean", mean_loss), ("BlockShuffle", shuffle_loss))}
    primary = predictive_gate(reports["Persistence"])
    sanity = [reports[k]["all"] for k in ("TrainMean", "BlockShuffle")]
    status = ("FAIL" if primary == "FAIL" or any(v["ci95_high"] <= 0 for v in sanity) else
              "PASS" if primary == "PASS" and all(v["ci95_low"] > 0 for v in sanity) else "INCONCLUSIVE")
    return {"status": status, "reports": reports, "train_mean_vectors": mean.tolist(),
            "shuffle_donor_indices": donors, "shuffle_source_partition": "probe_fit"}, mean_loss, shuffle_loss


def stage_a_manifest(dataset, split):
    validate_split(split)
    records, seen = [], {}
    for commodity, item in split["commodities"].items():
        if not item["formal_stage_a"]:
            continue
        for partition in ("probe_fit", "probe_dev", "probe_test"):
            candidates = {}
            for i, arrays in enumerate(dataset.episode_arrays):
                e = arrays.episode
                if e.commodity != commodity or e.contract_uid not in item[partition + "_contracts"]:
                    continue
                identity = (commodity, e.contract_uid)
                timestamps = arrays.minute_frame.datetime.to_numpy()[arrays.anchors]
                keep = ~np.isin(timestamps, seen.get(identity, np.array([], dtype="datetime64[ns]")))
                seen[identity] = np.union1d(seen.get(identity, timestamps[:0]), timestamps)
                local = np.flatnonzero(keep)
                dates = arrays.minute_frame.trading_date.to_numpy()[arrays.anchors[local]]
                for day in np.unique(dates):
                    indices = dataset.offsets[i] + local[dates == day]
                    candidates.setdefault((e.contract_uid, str(day)), []).extend(indices.tolist())
            selected = balanced_select(candidates, STAGE_A_CONFIG["probe_anchor_cap_per_commodity"])
            if not selected:
                raise ValueError(f"empty Stage-A partition {commodity}/{partition}")
            for index in selected:
                i = int(np.searchsorted(dataset.offsets, index, side="right") - 1)
                arrays = dataset.episode_arrays[i]
                e = arrays.episode
                position = int(arrays.anchors[index - dataset.offsets[i]])
                row = arrays.minute_frame.iloc[position]
                records.append({
                    "commodity": commodity, "contract_uid": e.contract_uid, "episode_id": e.episode_id,
                    "series_key": e.series_key, "trading_date": row.trading_date.date().isoformat(),
                    "anchor_timestamp": row.datetime.isoformat(),
                    "window_start": arrays.minute_frame.iloc[max(0, position - 511)].datetime.isoformat(),
                    "full_minute": position >= 511, "split": partition,
                    **{f"H{h}_valid": True for h in HORIZONS}, "data_manifest_sha256": split["data_manifest_sha256"],
                })
    if not records:
        raise ValueError("no formal Stage-A commodities")
    result = {"protocol": PROTOCOL, "stage_a_temporal_split_sha256": split["sha256"],
              "anchors": records, "train_commodities": list(dataset.train_commodities),
              "rb_inventory": False, "data_manifest_sha256": split["data_manifest_sha256"]}
    result["sha256"] = digest(result)
    return result


def run_stage_a(checkpoint, output, *, device="cpu", batch_size=8):
    output = Path(output)
    if (output / "stage_a_summary.json").exists():
        raise FileExistsError("Stage-A output already exists")
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary = _run(checkpoint, output, device, batch_size)
    except (ValueError, KeyError, FileNotFoundError, FloatingPointError, np.linalg.LinAlgError) as error:
        summary = {"status": "STAGE_A_BLOCKED", "reason": str(error), "RB": "DEFERRED_UNTIL_STAGE_A_GO"}
    write_json(output / "stage_a_summary.json", summary)
    (output / "stage_a_summary.md").write_text("# Stage-A core audit\n\n" + json.dumps(summary, indent=2) + "\n")
    return summary


def _run(checkpoint, output, device, batch_size):
    before = file_hash(checkpoint)
    state = load_v11_checkpoint(checkpoint)
    if not state.get("stage_a_temporal_split_sha256"):
        raise ValueError("STAGE_A_BLOCKED_LEGACY_CHECKPOINT_NO_TEMPORAL_HOLDOUT")
    if state.get("evaluation_variant", "Full") != "Full" or state.get("model_size", "S") != "S":
        raise ValueError("Stage-A requires Full S")
    config = state["v11_config"]
    if config.get("evaluation", {}).get("stage_a_temporal_oos") != STAGE_A_CONFIG:
        raise ValueError("Stage-A config mismatch")
    if (state.get("training_protocol_version") != "fixed_sample_budget_v1" or
            state.get("max_optimizer_steps") != 250000 or state.get("effective_batch_size") != 128 or
            state.get("warmup_optimizer_steps") != 12500 or state.get("target_samples_seen") != 32000000):
        raise ValueError("Stage-A requires the frozen Scheme-A budget")
    if "RB" in config["data"]["train_commodities"]:
        raise ValueError("Stage-A forbids RB reads")
    split_path = Path(checkpoint).resolve().parent.parent / "stage_a_temporal_split.json"
    split = json.loads(split_path.read_text())
    validate_split(split, state["data_manifest_sha256"])
    if split != state["stage_a_temporal_split"]:
        raise ValueError("Stage-A split file/checkpoint mismatch")
    verify_data_files(config["data"]["root"], config["data"]["train_commodities"])
    if file_hash(Path(config["data"]["root"]) / "build_manifest.json") != state["data_manifest_sha256"]:
        raise ValueError("Stage-A current data manifest mismatch")
    semantic_implementation_gate(state)
    completed_run_gate(checkpoint, state)
    scaler = SharedIMCScaler.from_dict(state["shared_imc_scaler"])
    ds = dataset_for(config, scaler, config["data"]["train_commodities"])
    isolation = isolation_gate(state, split, ds)
    manifest = stage_a_manifest(ds, split)
    final_checkpoint_gate(state, manifest)
    del ds
    write_json(output / "evaluation_manifest.json", manifest)
    write_json(output / "stage_a_temporal_split.json", split)
    model = model_from_checkpoint(state).to(device).eval().requires_grad_(False)
    exports, records = extract(model, config, scaler, manifest["anchors"], torch.device(device),
                               batch_size=batch_size, diagnostics=False, export_latents=True, defer_latent_losses=True)
    test = np.asarray([r["split"] == "probe_test" for r in records])
    np.savez_compressed(output / "a0_probe_test_latents.npz", belief=exports["belief"][test],
                        **{f"target_h{h}": exports["target_latents"][test, j] for j, h in enumerate(HORIZONS)})
    write_json(output / "extraction_records.json", records)
    health = {"belief": latent_health(exports["belief"][test]),
              **{f"target_h{h}": latent_health(exports["target_latents"][test, j]) for j, h in enumerate(HORIZONS)}}
    a0 = "PASS" if all(v["status"] == "PASS" for v in health.values()) else "BLOCKED_TRIVIAL_COLLAPSE"
    write_json(output / "a0_latent_health.json", {"status": a0, "representations": health})
    if a0 != "PASS":
        return {"status": "STAGE_A_BLOCKED", "strict_temporal_oos": isolation, "latent_health": a0}
    exports["jepa"] = cosine_loss(exports["prediction_latents"], exports["target_latents"])
    exports["persistence"] = cosine_loss(exports["persistence_latents"], exports["target_latents"])
    a1, mean_loss, shuffle_loss = a1_reports(exports, records)
    write_json(output / "a1_oos_prediction.json", a1)
    test_records = [r for r, flag in zip(records, test) if flag]
    test_jepa, test_persistence = exports["jepa"][test], exports["persistence"][test]
    write_csv(output / "a1_oos_prediction.csv", [
        {**r, "horizon": h, "jepa_loss": float(test_jepa[i, j]),
         "persistence_loss": float(test_persistence[i, j]),
         "mean_loss": float(mean_loss[i, j]), "shuffle_loss": float(shuffle_loss[i, j])}
        for i, r in enumerate(test_records) for j, h in enumerate(HORIZONS)])
    probes, a2, rows, _, _ = _probe_results(exports, records)
    a2_status = predictive_gate(a2)
    write_json(output / "a2_oos_probe.json", {"status": a2_status, "report": a2})
    write_json(output / "a2_probe_hyperparams.json", probes)
    write_csv(output / "a2_probe_results.csv", [r for r in rows if r["representation"] == "belief"])
    write_csv(output / "a2_baselines.csv", [r for r in rows if r["representation"] != "belief"])
    if before != file_hash(checkpoint):
        raise ValueError("checkpoint changed during Stage-A audit")
    status = stage_a_status(isolation, a0, a1["status"], a2_status)
    gain = a1["reports"]["Persistence"]
    return {"question": "Does V1.1 learn a stable future-relevant predictive latent representation?",
            "status": status, "strict_temporal_oos": isolation, "latent_health": a0,
            "A1_future_prediction": a1["status"], "A2_frozen_probe": a2_status,
            "Gain_ALL": gain["all"], "Gain_H64": gain["horizons"]["64"], "Gain_H256": gain["horizons"]["256"],
            "Skill_ALL": a2["all"], "Skill_H64": a2["horizons"]["64"], "Skill_H256": a2["horizons"]["256"],
            "formal_commodities": [c for c, v in split["commodities"].items() if v["formal_stage_a"]],
            "excluded_commodities": split["excluded_formal_commodities"],
            "claim": ("On strictly time-isolated real-contract test episodes, the frozen predictive latent "
                      "has significant future-relevant linear-probe gain over causal history summaries.") if status == "STAGE_A_GO" else "Stage-A predictive representation claim not established",
            "not_claimed": ["identified Predictive State", "persistent HiddenMarketState", "RSSM value", "trading value", "cross-instrument generalization"],
            "V0_comparison": "NOT_REQUIRED_FOR_STAGE_A",
            **{k: "DEFERRED_UNTIL_STAGE_A_GO" for k in ("A3_structure", "controls", "RB", "scaling")},
            "protocol": {"checkpoint_sha256": before, "config_sha256": digest(config),
                         "implementation_sha256": state["v11_implementation_sha256"],
                         "evaluation_code_sha256": evaluation_code_hash(), "scaler_sha256": scaler.checksum,
                         "data_manifest_sha256": state["data_manifest_sha256"], "manifest_sha256": manifest["sha256"],
                         "stage_a_temporal_split_sha256": split["sha256"], "sampling_population_sha256": state["sampling_population_sha256"]}}
