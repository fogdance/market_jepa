from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from market_jepa_v1_1.checkpoint import load_v11_checkpoint, model_from_checkpoint
from market_jepa_v1_1.dataset import V11ContractDataset, V11DataStore, collate_v11_batch
from market_jepa_v1_1.imc import SharedIMCScaler
from market_jepa_v1_1.training import model_inputs

from .controls import control_model
from .metrics import bootstrap, cosine_loss, predictive_status
from .probe import context_features, fit_probe, future_outcomes, minute_raw24, multiscale_summary, predict_probe
from .protocol import HORIZONS, PROTOCOL, build_manifest, digest, evaluation_code_hash, file_hash, final_checkpoint_gate, validate_manifest, write_json, verify_data_files, completed_run_gate, semantic_implementation_gate
from .structure import REMOVALS, donor_manifest, remove_sources, replace_channels


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def dataset_for(config, scaler, commodities, *, rb=False):
    role = "heldout" if rb else "train"
    store = V11DataStore.from_directory(config["data"]["root"], commodities, episode_role=role)
    return V11ContractDataset(store, config, role=role, scaler=scaler)


def inventory(config, output, *, rb=False):
    """Read-only production inventory; RB mode inventories timestamps, no model execution."""
    root = Path(config["data"]["root"])
    if "RB" in config["data"]["train_commodities"] or config["data"]["held_out_commodity"] != "RB":
        raise ValueError("formal evaluation requires RB held out")
    data_hash = file_hash(root / "build_manifest.json")
    commodities = [config["data"]["held_out_commodity"]] if rb else config["data"]["train_commodities"]
    audit = verify_data_files(root, commodities)
    manifests = []
    # Bound memory to one commodity; same dataset semantics and complete episode coverage.
    for commodity in commodities:
        ds = dataset_for(config, None, [commodity], rb=rb)
        manifests.append(build_manifest(ds, data_hash, rb=rb))
    result = dict(manifests[0])
    result["anchors"] = [r for m in manifests for r in m["anchors"]]
    result["excluded_probe_groups"] = {k: v for m in manifests for k, v in m["excluded_probe_groups"].items()}
    result.pop("sha256")
    result["data_integrity"] = audit
    result["sha256"] = digest(result)
    if Path(output).exists():
        previous = json.loads(Path(output).read_text())
        if previous != result:
            raise FileExistsError("manifest already exists with different content; use a new campaign")
        return result
    write_json(output, result)
    return result


def resolve_records(dataset, records):
    lookup = {a.episode.key: i for i, a in enumerate(dataset.episode_arrays)}
    result = []
    for r in records:
        key = (r["commodity"], r["contract_uid"], int(r["episode_id"]))
        if key not in lookup:
            raise ValueError(f"manifest episode unavailable: {key}")
        i = lookup[key]; arrays = dataset.episode_arrays[i]
        times = arrays.minute_frame.datetime.to_numpy(dtype="datetime64[ns]")
        position = int(np.searchsorted(times, np.datetime64(r["anchor_timestamp"])))
        local = int(np.searchsorted(arrays.anchors, position))
        if position >= len(times) or times[position] != np.datetime64(r["anchor_timestamp"]):
            raise ValueError("manifest anchor timestamp unavailable")
        if local >= len(arrays.anchors) or arrays.anchors[local] != position:
            raise ValueError("manifest anchor is no longer eligible")
        expected_start = arrays.minute_frame.iloc[max(0, position - 511)].datetime.isoformat()
        if expected_start != r["window_start"] or str(arrays.minute_frame.iloc[position].trading_date.date()) != r["trading_date"]:
            raise ValueError("manifest window/trading date mismatch")
        result.append((dataset.global_index(i, local), arrays, position))
    return result


def extract(model, config, scaler, records, device, *, batch_size=8, diagnostics=True, rb=False):
    exports = {key: [] for key in ("belief", "jepa", "persistence", "outcomes", "context", "minute_raw24", "summary")}
    if diagnostics:
        exports.update({"belief_" + variant: [] for variant in REMOVALS if variant != "Full"})
        exports.update({"jepa_" + variant: [] for variant in REMOVALS if variant != "Full"})
    ordered_records = []
    for commodity in sorted({r["commodity"] for r in records}):
        subset = [r for r in records if r["commodity"] == commodity]
        ds = dataset_for(config, scaler, [commodity], rb=rb)
        resolved = resolve_records(ds, subset)
        for start in range(0, len(resolved), batch_size):
            items = resolved[start:start + batch_size]
            samples = [ds[index] for index, _, _ in items]
            kwargs = model_inputs(collate_v11_batch(samples), device)
            with torch.inference_mode():
                output = model(**kwargs)
                persistence = horizon_persistence(model, kwargs)
                targets = {h: output["targets"][h].cpu().numpy() for h in HORIZONS}
                exports["belief"].extend(output["z_market"].cpu().numpy())
                exports["jepa"].extend(np.stack([cosine_loss(output["predictions"][h].cpu().numpy(), targets[h]) for h in HORIZONS], 1))
                exports["persistence"].extend(np.stack([cosine_loss(persistence[h].cpu().numpy(), targets[h]) for h in HORIZONS], 1))
                if diagnostics:
                    for variant in REMOVALS:
                        if variant == "Full": continue
                        changed = model(**remove_sources(kwargs, variant))
                        exports["belief_" + variant].extend(changed["z_market"].cpu().numpy())
                        exports["jepa_" + variant].extend(np.stack([cosine_loss(changed["predictions"][h].cpu().numpy(), targets[h]) for h in HORIZONS], 1))
            for sample, (_, arrays, position) in zip(samples, items):
                exports["outcomes"].append(np.concatenate([future_outcomes(arrays.minute_bars, position, h) for h in HORIZONS]))
                exports["context"].append(context_features(sample))
                exports["minute_raw24"].append(minute_raw24(sample, arrays.minute_bars, position))
                exports["summary"].append(multiscale_summary(sample))
        ordered_records.extend(subset)
    return {k: np.asarray(v) for k, v in exports.items()}, ordered_records


def horizon_persistence(model, batch):
    """V0 continuity: past H bars versus future H bars, retaining online IMC origin."""
    return {h: model.target_minute(batch["minute_market"][:, -h:],
                                  batch["minute_imc_validity"][:, -h:],
                                  batch["minute_mask"][:, -h:]) for h in HORIZONS}


def structure_audit(model, config, scaler, records, device, *, batch_size=8):
    mapping = donor_manifest(records)
    errors = {"clean": [], "OI": [], "Volume": [], "Joint": []}
    recipients = []
    for commodity in sorted({records[p["recipient"]]["commodity"] for p in mapping["pairs"]}):
        ds = dataset_for(config, scaler, [commodity])
        pairs = [p for p in mapping["pairs"] if records[p["recipient"]]["commodity"] == commodity]
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            rr = [records[p["recipient"]] for p in chunk]
            dr = [records[p["donor"]] for p in chunk]
            recipient = [ds[item[0]] for item in resolve_records(ds, rr)]
            donor = [ds[item[0]] for item in resolve_records(ds, dr)]
            kwargs = model_inputs(collate_v11_batch(recipient), device)
            with torch.inference_mode():
                clean = model(**kwargs)
                target = {h: clean["targets"][h].cpu().numpy() for h in HORIZONS}
                errors["clean"].extend(np.stack([cosine_loss(clean["predictions"][h].cpu().numpy(), target[h]) for h in HORIZONS], 1))
                for kind in ("OI", "Volume", "Joint"):
                    corrupted = [replace_channels(r, d, kind) for r, d in zip(recipient, donor)]
                    changed = model(**model_inputs(collate_v11_batch(corrupted), device))
                    errors[kind].extend(np.stack([cosine_loss(changed["predictions"][h].cpu().numpy(), target[h]) for h in HORIZONS], 1))
            recipients.extend(rr)
    reports = {kind: bootstrap(errors[kind], errors["clean"], recipients, difference=True)
               for kind in ("OI", "Volume", "Joint")} if recipients else {}
    return mapping, reports


def _probe_results(exports, records):
    fit = np.asarray([r["split"] == "probe_fit" for r in records])
    dev = np.asarray([r["split"] == "probe_dev" for r in records])
    test = np.asarray([r["split"] == "probe_test" for r in records])
    if not test.any():
        raise ValueError("no eligible ProbeTest population")
    y = exports["outcomes"]
    models, predictions = {}, {}
    for key in ("belief", "context", "minute_raw24", "summary"):
        models[key] = fit_probe(exports[key], y, fit, dev)
        predictions[key] = predict_probe(models[key], exports[key][test])
    models["unconditional_mean"] = y[fit].mean(0).tolist()
    predictions["unconditional"] = np.broadcast_to(models["unconditional_mean"], y[test].shape)
    scale = np.asarray(models["belief"]["y_std"])
    errors = {k: np.square((p - y[test]) / scale) for k, p in predictions.items()}
    test_records = [r for r, flag in zip(records, test) if flag]
    report = bootstrap(errors["belief"], errors["summary"], test_records)
    commodities = np.asarray([r["commodity"] for r in test_records])
    groups = [commodities == c for c in sorted(set(commodities))]
    def aggregate(values):
        return float(np.mean([values[group].mean() for group in groups]))
    rows = []
    for key, p in predictions.items():
        for j in range(12):
            rows.append({"representation": key, "horizon": HORIZONS[j // 4],
                         "outcome": ("return", "mfe", "mae", "rv")[j % 4],
                         "aggregation": "equal_commodity_weight",
                         "native_rmse": float(np.sqrt(aggregate(np.square(p[:, j] - y[test, j])))),
                         "native_mae": aggregate(np.abs(p[:, j] - y[test, j])),
                         "standardized_mse": aggregate(errors[key][:, j])})
    return models, report, rows, errors, test


def evaluate(checkpoint, manifest_path, output, *, device="cpu", batch_size=8):
    output = Path(output)
    if (output / "summary.json").exists():
        raise FileExistsError("evaluation output already exists; use a new output directory")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["rb_inventory"]:
        raise ValueError("RB requires separately authorized frozen evaluation")
    before = file_hash(checkpoint)
    state = load_v11_checkpoint(checkpoint)
    final_checkpoint_gate(state, manifest)
    completion = completed_run_gate(checkpoint, state)
    if before != file_hash(checkpoint):
        raise ValueError("checkpoint changed during read")
    config = state["v11_config"]
    data_integrity = verify_data_files(config["data"]["root"], manifest["train_commodities"])
    if file_hash(Path(config["data"]["root"]) / "build_manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("current production build differs from manifest")
    # A newer evaluation implementation may read older endpoints, but the actual
    # model/data/IMC interpretation must still match their recorded source files.
    semantic_implementation_gate(state)
    variant = state.get("evaluation_variant", "Full")
    if variant == "Full":
        model = model_from_checkpoint(state)
    else:
        model = control_model(state["architecture_config"], variant)
        model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    scaler = SharedIMCScaler.from_dict(state["shared_imc_scaler"])
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"checkpoint_sha256": before, "config_sha256": digest(config),
                "implementation_sha256": state["v11_implementation_sha256"],
                "evaluation_code_sha256": evaluation_code_hash(), "scaler_sha256": scaler.checksum,
                "manifest_sha256": manifest["sha256"], "variant": variant,
                "model_size": state.get("model_size", "S"), "epochs": config["training"]["max_epochs"],
                "trainable_params": state.get("trainable_parameter_count"),
                "data_manifest_sha256": state["data_manifest_sha256"],
                "training_protocol": {k: config["training"][k] for k in (
                    "seed", "optimizer", "learning_rate", "weight_decay", "gradient_accumulation",
                    "batch_size", "max_epochs", "ema_tau", "amp", "amp_dtype")},
                "rb_test_consumed": False, "precision": "float32", "protocol": PROTOCOL}
    protocol["completion_gate"] = completion
    protocol["sampler_num_samples"] = state["sampler"]["num_samples"]
    if variant != "Full":
        protocol["control_provenance"] = state["control_provenance"]
    protocol["data_integrity"] = data_integrity
    write_json(output / "evaluation_protocol.json", protocol)
    write_json(output / "evaluation_manifest.json", manifest)
    exports, records = extract(model, config, scaler, manifest["anchors"], torch.device(device),
                               batch_size=batch_size, diagnostics=variant == "Full")
    a1 = bootstrap(exports["jepa"], exports["persistence"], records)
    write_json(output / "a1_bootstrap.json", a1)
    write_csv(output / "a1_jepa_vs_persistence.csv", [
        {**r, "horizon": h, "jepa_loss": float(exports["jepa"][i, j]),
         "persistence_loss": float(exports["persistence"][i, j])}
        for i, r in enumerate(records) for j, h in enumerate(HORIZONS)])
    probes, a2, rows, errors, test = _probe_results(exports, records)
    write_json(output / "a2_probe_hyperparams.json", probes)
    write_json(output / "a2_bootstrap.json", a2)
    write_csv(output / "a2_probe_results.csv", [r for r in rows if r["representation"] == "belief"])
    write_csv(output / "a2_baselines.csv", [r for r in rows if r["representation"] != "belief"])
    donors, a3 = structure_audit(model, config, scaler, records, torch.device(device), batch_size=batch_size)
    write_json(output / "a3_donor_manifest.json", {**donors, "manifest_sha256": manifest["sha256"],
        "anchor_order": records})
    write_json(output / "a3_bootstrap.json", a3)
    write_csv(output / "a3_structure_results.csv", [
        {"intervention": k, "horizon": scope, **values}
        for k, v in a3.items() for scope, values in [("ALL", v["all"]), *v["horizons"].items()]])
    diagnostics = {}
    if variant == "Full":
        for removal in REMOVALS:
            if removal == "Full": continue
            changed = predict_probe(probes["belief"], exports["belief_" + removal][test])
            changed_error = np.square((changed - exports["outcomes"][test]) / probes["belief"]["y_std"])
            diagnostics[removal] = {
                "gain": bootstrap(exports["jepa_" + removal], exports["persistence"], records),
                "skill": bootstrap(changed_error, errors["summary"], [r for r, t in zip(records, test) if t]),
                "interpretation": "OOD diagnostic only; frozen Full probe; not retrain evidence"}
            diagnostics[removal]["delta_Gain_ALL"] = diagnostics[removal]["gain"]["all"]["point"] - a1["all"]["point"]
            diagnostics[removal]["delta_Skill_ALL"] = diagnostics[removal]["skill"]["all"]["point"] - a2["all"]["point"]
            diagnostics[removal]["delta_Gain_H"] = {
                str(h): diagnostics[removal]["gain"]["horizons"][str(h)]["point"] - a1["horizons"][str(h)]["point"] for h in HORIZONS}
            diagnostics[removal]["delta_Skill_H"] = {
                str(h): diagnostics[removal]["skill"]["horizons"][str(h)]["point"] - a2["horizons"][str(h)]["point"] for h in HORIZONS}
    write_json(output / "source_diagnostics.json", diagnostics)
    np.savez_compressed(output / "paired_errors.npz", jepa=exports["jepa"], persistence=exports["persistence"],
                        belief_error=errors["belief"], summary_error=errors["summary"], test_mask=test)
    write_json(output / "paired_records.json", records)
    write_csv(output / "rb_results.csv", [], ["status", "metric", "value"])
    write_json(output / "rb_bootstrap.json", {"status": "NOT_RUN", "rb_test_consumed": False})
    summary = {"A1": predictive_status(a1), "A2": predictive_status(a2),
               "V0_CONTINUITY": "PASS" if predictive_status(a1) == predictive_status(a2) == "PASS" else "NOT_PASS",
               "architecture": "NOT_EVALUATED_CONTROLS_AND_RB_REQUIRED", "multiscale": "NOT_EVALUATED",
               "history": "NOT_EVALUATED", "RB": "NOT_RUN", "rb_test_consumed": False,
               "Gain_ALL": a1["all"], "Skill_ALL": a2["all"], "protocol": protocol,
               "claim": "Predictive evaluation only; transferable/multiscale claim requires controls and RB"}
    joint = a3.get("Joint", {}).get("all")
    summary["structure"] = ("NO_DONOR_POPULATION" if joint is None else
        "STRUCTURE_PASS" if joint["ci95_low"] > 0 else
        "WEAK_EVIDENCE" if joint["point"] > 0 else "NO_EVIDENCE")
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text("# V1.1 evaluation\n\n" + json.dumps(summary, indent=2) + "\n")
    return summary
