from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HORIZONS = (16, 64, 256)
OUTCOMES = ("return", "mfe", "mae", "rv")
SOURCES = ("minute", "daily", "current_weekly", "history_weekly")
WINDOWS = ((16, 64, 256, 512), (5, 20, 60, 256), (4, 13, 26, 64), (13, 52, 104, 156))
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)
PROTOCOL = {
    "version": "reviewed_v2_v0_outcome_override", "seed": 42,
    "probe_split": [0.70, 0.15, 0.15], "rb_dev_fraction": 1 / 3,
    "train_probe_anchors_per_commodity": 4096, "rb_dev_anchors": 32768,
    "rb_test_anchors": 65536, "structure_audit_anchors": 16384,
    "bootstrap_replicates": 2000, "confidence": 0.95,
    "block": ["contract_uid", "trading_date"], "aggregate": "equal_commodity",
    "outcome_definition": "V0 exact: ratio-1 Return/MFE/MAE; sqrt(mean(log-return squared)) RV",
    "outcome_override": "Explicit user decision: preserve V0 exact parity instead of V2 log/sum formulas",
    "alphas": ALPHAS, "primary_probe_baseline": "MultiScaleSummary",
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data_files(root, commodities):
    root = Path(root)
    build = json.loads((root / "build_manifest.json").read_text())
    entries = {entry["path"]: entry for entry in build.get("files", [])}
    required = ["contract_episodes.csv"] + [f"{c}/{c}_{scale}.csv" for c in commodities for scale in ("1m", "1d", "1w")]
    for name in required:
        if name not in entries or file_hash(root / name) != entries[name]["sha256"]:
            raise ValueError(f"production source checksum mismatch: {name}")
    return {"status": "PASS", "files_verified": required, "build_manifest_sha256": file_hash(root / "build_manifest.json")}


def completed_run_gate(checkpoint, state):
    run = Path(checkpoint).resolve().parent.parent
    summary_path = run / "final_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") not in {"V1_1_FORMAL_TRAINING_PASS", "V1_1_CONTROL_TRAINING_PASS"}:
        raise ValueError("training completion/hard-gate summary is not PASS")
    if summary.get("checkpoint_sha256") != file_hash(checkpoint):
        raise ValueError("training summary checkpoint checksum mismatch")
    if state.get("evaluation_variant", "Full") == "Full":
        audit = json.loads((run / "data_audit.json").read_text())
        if audit.get("status") != "PASS" or audit.get("held_out_bar_files_opened") is not False:
            raise ValueError("training data audit or held-out isolation failed")
    return {"summary_sha256": file_hash(summary_path), "status": "PASS"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluation_code_hash():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "market_jepa_v1_1").rglob("*.py"))
    paths += sorted((root / "market_jepa").rglob("*.py"))
    paths += [root / "evaluate_market_jepa_v1_1.py", root / "train_market_jepa_v1_1_control.py"]
    paths += [root / "tests/test_v11_formal_evaluation.py", root / "docs/MARKET_JEPA_V1_1_EVALUATION_DESIGN_REVIEWED_V2.md"]
    return digest({str(p.relative_to(root)): file_hash(p) for p in paths if p.exists()})


def split_episodes(episodes, *, rb=False):
    """Chronological whole episodes; reserve at least one episode per partition."""
    ordered = sorted(episodes, key=lambda e: (e.main_start, e.contract_uid, e.episode_id))
    n = len(ordered)
    if n < (2 if rb else 3):
        return {}, "insufficient_episodes"
    if rb:
        cut = max(1, n // 3)
        labels = ["rb_dev"] * cut + ["rb_test"] * (n - cut)
    else:
        fit = min(max(1, int(n * .70)), n - 2)
        dev = min(max(1, int(n * .15)), n - fit - 1)
        labels = ["probe_fit"] * fit + ["probe_dev"] * dev + ["probe_test"] * (n - fit - dev)
    return {e.key: label for e, label in zip(ordered, labels)}, None


def balanced_select(groups, budget, seed=42):
    """Uniform shuffled round-robin over episodes, without replacement/retries."""
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.asarray(groups[k], dtype=np.int64)).tolist() for k in sorted(groups)]
    selected = []
    active = [i for i, p in enumerate(pools) if p]
    while active and len(selected) < budget:
        for i in rng.permutation(active):
            if len(selected) == budget:
                break
            selected.append(pools[i].pop())
        active = [i for i in active if pools[i]]
    return sorted(selected)


def build_manifest(dataset, data_hash, *, rb=False):
    """Only inventory/sampling: never runs a model or constructs future outcomes."""
    groups = {}
    for arrays in dataset.episode_arrays:
        e = arrays.episode
        groups.setdefault(e.series_key if rb else e.commodity, []).append(e)
    membership, excluded = {}, {}
    for key, episodes in sorted(groups.items()):
        parts, reason = split_episodes(episodes, rb=rb)
        membership.update(parts)
        if reason:
            excluded[key] = reason
    # Re-main episodes may share raw anchors; check whole populations before
    # sampling so a low budget cannot hide split leakage.
    by_contract = {}
    for arrays in dataset.episode_arrays:
        e = arrays.episode
        previous = by_contract.setdefault((e.commodity, e.contract_uid), [])
        for prior in previous:
            if membership.get(e.key) != membership.get(prior.episode.key) and np.intersect1d(arrays.anchors, prior.anchors).size:
                raise ValueError("overlapping re-main episodes cross probe partitions")
        previous.append(arrays)
    candidates, partitions = {}, {}
    for i, arrays in enumerate(dataset.episode_arrays):
        e = arrays.episode
        partition = membership.get(e.key, "a1_only")
        if rb and e.key not in membership:
            continue
        indices = []
        for local, position in enumerate(arrays.anchors):
            index = dataset.global_index(i, local)
            indices.append(index)
            partitions[index] = (i, int(position), partition)
        bucket = partition if rb else e.commodity
        candidates.setdefault(bucket, {})[e.key] = indices
    selected = []
    for key, values in sorted(candidates.items()):
        budget = PROTOCOL[key + "_anchors"] if rb else PROTOCOL["train_probe_anchors_per_commodity"]
        selected += balanced_select(values, budget)
    records = []
    for index in sorted(selected):
        i, position, partition = partitions[index]
        arrays = dataset.episode_arrays[i]
        e = arrays.episode
        row = arrays.minute_frame.iloc[position]
        records.append({
            "commodity": e.commodity, "series_key": e.series_key,
            "contract_uid": e.contract_uid, "episode_id": e.episode_id,
            "anchor_timestamp": row.datetime.isoformat(), "trading_date": row.trading_date.date().isoformat(),
            "window_start": arrays.minute_frame.iloc[max(0, position - 511)].datetime.isoformat(),
            "full_minute": position >= 511, "split": partition,
            "H16_valid": True, "H64_valid": True, "H256_valid": True,
            "data_manifest_sha256": data_hash,
        })
    # Overlapping episodes must never put an identical market anchor in two splits.
    identities = [(r["commodity"], r["contract_uid"], r["anchor_timestamp"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate raw anchor across episodes; episode split requires review")
    result = {"protocol": PROTOCOL, "anchors": records, "excluded_probe_groups": excluded,
              "train_commodities": list(dataset.train_commodities), "held_out": dataset.held_out_commodity,
              "data_manifest_sha256": data_hash, "rb_inventory": rb}
    result["sha256"] = digest(result)
    return result


def validate_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if digest(body) != manifest.get("sha256") or digest(manifest["protocol"]) != digest(PROTOCOL):
        raise ValueError("evaluation manifest/protocol hash mismatch")
    seen = set()
    episode_splits = {}
    for r in manifest["anchors"]:
        key = (r["commodity"], r["contract_uid"], r["anchor_timestamp"])
        if key in seen or not all(r[f"H{h}_valid"] for h in HORIZONS):
            raise ValueError("duplicate/invalid evaluation anchor")
        seen.add(key)
        e = (r["commodity"], r["contract_uid"], r["episode_id"])
        if episode_splits.setdefault(e, r["split"]) != r["split"]:
            raise ValueError("episode crosses evaluation partitions")
        if r["data_manifest_sha256"] != manifest["data_manifest_sha256"]:
            raise ValueError("anchor data hash mismatch")


def final_checkpoint_gate(state, manifest):
    validate_manifest(manifest)
    config = state["v11_config"]
    if state["epoch"] != config["training"]["max_epochs"] - 1:
        raise ValueError("CHECKPOINT_NOT_FINAL")
    if state["checkpoint_selection"] != "fixed_budget_final":
        raise ValueError("checkpoint policy mismatch")
    if state["data_manifest_sha256"] != manifest["data_manifest_sha256"]:
        raise ValueError("checkpoint/evaluation data mismatch")
    train = set(config["data"]["train_commodities"])
    if train != set(manifest["train_commodities"]) or "RB" in train or config["data"]["held_out_commodity"] != "RB":
        raise ValueError("training population mismatch or RB contamination")
    if set(state["shared_imc_scaler"]["fitted_commodities"]) != train:
        raise ValueError("scaler training population mismatch")
    if len(state.get("history", [])) != config["training"]["max_epochs"]:
        raise ValueError("incomplete fixed training budget")
    if [r["epoch"] for r in state["history"]] != list(range(config["training"]["max_epochs"])):
        raise ValueError("checkpoint epoch history is not contiguous")
    connectivity = state.get("gradient_connectivity")
    if not connectivity or connectivity.get("missing") or connectivity.get("nonfinite"):
        raise ValueError("checkpoint gradient connectivity gate failed")
    for record in state["history"]:
        for key in ("train_loss", "prediction_loss_h16", "prediction_loss_h64", "prediction_loss_h256"):
            if key not in record or not np.isfinite(record[key]):
                raise ValueError("checkpoint training metrics missing/nonfinite")


def semantic_implementation_gate(state):
    root = Path(__file__).resolve().parents[2]
    for relative in ("market_jepa_v1_1/model.py", "market_jepa_v1_1/encoders.py",
                     "market_jepa_v1_1/dataset.py", "market_jepa_v1_1/imc.py"):
        expected = state["v11_implementation_manifest"]["files"].get(relative)
        if expected != file_hash(root / relative):
            raise ValueError(f"checkpoint semantic implementation mismatch: {relative}")
