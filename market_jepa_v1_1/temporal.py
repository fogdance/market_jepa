"""Pretraining contract-group holdout. Raw causal memories remain intact."""
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE_A_CONFIG = {
    "enabled": True, "version": "stage_a_temporal_oos_v1",
    "split_unit": "contract_uid", "split": [0.70, 0.15, 0.15],
    "test_policy": "latest_suffix", "min_contract_groups": 3,
    "probe_anchor_cap_per_commodity": 4096,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_temporal_split(dataset, data_hash):
    commodities, excluded = {}, {}
    for commodity in dataset.train_commodities:
        groups = {}
        for arrays in dataset.episode_arrays:
            e = arrays.episode
            if e.commodity == commodity:
                groups[e.contract_uid] = min(groups.get(e.contract_uid, e.main_start), e.main_start)
        ordered = sorted(groups, key=lambda uid: (groups[uid], uid))
        n = len(ordered)
        formal = n >= 3
        fit = min(max(1, int(n * .70)), n - 2) if formal else n
        dev = min(max(1, int(n * .15)), n - fit - 1) if formal else 0
        commodities[commodity] = {
            "probe_fit_contracts": ordered[:fit],
            "probe_dev_contracts": ordered[fit:fit + dev],
            "probe_test_contracts": ordered[fit + dev:],
            "formal_stage_a": formal,
            "contract_main_starts": {uid: groups[uid].isoformat() for uid in ordered},
        }
        if not formal:
            excluded[commodity] = "insufficient_contract_groups"
    result = {"version": STAGE_A_CONFIG["version"], "unit": "commodity_contract_uid",
              "policy": "chronological_70_15_15_latest_suffix_test",
              "data_manifest_sha256": data_hash, "commodities": commodities,
              "excluded_formal_commodities": excluded}
    result["sha256"] = digest(result)
    return result


def validate_split(split, data_hash=None):
    if split.get("sha256") != digest({k: v for k, v in split.items() if k != "sha256"}):
        raise ValueError("Stage-A temporal split checksum mismatch")
    if (split.get("version") != STAGE_A_CONFIG["version"] or split.get("unit") != "commodity_contract_uid"
            or split.get("policy") != "chronological_70_15_15_latest_suffix_test"):
        raise ValueError("Stage-A temporal split protocol mismatch")
    if data_hash is not None and split["data_manifest_sha256"] != data_hash:
        raise ValueError("Stage-A temporal split data mismatch")
    for item in split["commodities"].values():
        parts = [item[k + "_contracts"] for k in ("probe_fit", "probe_dev", "probe_test")]
        flattened = sum(parts, [])
        if len(flattened) != len(set(flattened)):
            raise ValueError("Stage-A contract group crosses partitions")
        ordered = sorted(item["contract_main_starts"], key=lambda uid: (item["contract_main_starts"][uid], uid))
        if flattened != ordered or (item["formal_stage_a"] and not all(parts)):
            raise ValueError("Stage-A split is not chronological/latest suffix")
        n = len(ordered)
        fit = min(max(1, int(n * .70)), n - 2) if n >= 3 else n
        dev = min(max(1, int(n * .15)), n - fit - 1) if n >= 3 else 0
        if item["formal_stage_a"] != (n >= 3) or list(map(len, parts)) != [fit, dev, n - fit - dev]:
            raise ValueError("Stage-A split allocation differs from frozen policy")


def freeze_split(path, split):
    validate_split(split)
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != split:
            raise ValueError("immutable Stage-A temporal split changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(split, handle, indent=2)


def training_view(dataset, split):
    """Restrict anchor episodes; retain original raw store for causal history."""
    validate_split(split)
    view = copy(dataset)
    view.episode_arrays, view.offsets = [], [0]
    for arrays in dataset.episode_arrays:
        e = arrays.episode
        item = split["commodities"][e.commodity]
        if e.contract_uid in item["probe_test_contracts"]:
            continue
        if hasattr(dataset, "history_weekly_rows"):
            history = dataset.history_weekly_rows(e)
            if set(history.contract_uid).intersection(item["probe_test_contracts"]):
                raise ValueError("Stage-A ProbeTest contract appears in earlier causal history; isolation requires review")
        selected = copy(arrays)
        if item["formal_stage_a"]:
            cutoff = min(item["contract_main_starts"][uid] for uid in item["probe_test_contracts"])
            # Re-main/overlapping lifecycles must not train on labels in the test era.
            times = arrays.minute_frame.datetime.to_numpy(dtype="datetime64[ns]")
            selected.anchors = arrays.anchors[times[arrays.anchors + max(dataset.horizons)] < np.datetime64(cutoff)]
        if len(selected.anchors):
            view.episode_arrays.append(selected)
            view.offsets.append(view.offsets[-1] + len(selected.anchors))
    if set(view.hierarchy) != set(dataset.train_commodities):
        raise ValueError("Stage-A temporal restriction leaves an empty training commodity")
    view.stage_a_temporal_split = deepcopy(split)
    return view


def training_contracts(dataset):
    return sorted({(a.episode.commodity, a.episode.contract_uid) for a in dataset.episode_arrays})


def isolation_gate(state, split, dataset):
    validate_split(split, state["data_manifest_sha256"])
    if state.get("stage_a_temporal_split_sha256") != split["sha256"]:
        raise ValueError("Stage-A checkpoint split binding mismatch")
    from .sampler import sampling_population_sha256
    restricted = training_view(dataset, split)
    if sampling_population_sha256(restricted, state["data_manifest_sha256"]) != state["sampling_population_sha256"]:
        raise ValueError("Stage-A sampling population mismatch")
    expected = training_contracts(restricted)
    if [list(x) for x in expected] != state.get("stage_a_training_contracts"):
        raise ValueError("Stage-A sampler contract provenance mismatch")
    tests = {(c, uid) for c, item in split["commodities"].items() for uid in item["probe_test_contracts"]}
    scaler = state.get("stage_a_scaler_selected_anchors")
    if not scaler or tests.intersection(expected) or any((r["commodity"], r["contract_uid"]) in tests for r in scaler):
        raise ValueError("Stage-A ProbeTest sampler/scaler leakage or absent provenance")
    if any((r["commodity"], r["contract_uid"]) not in expected for r in scaler):
        raise ValueError("Stage-A scaler population mismatch")
    episodes = {a.episode.key: a for a in restricted.episode_arrays}
    for r in scaler:
        key = (r["commodity"], r["contract_uid"], int(r["episode_id"]))
        arrays = episodes.get(key)
        if arrays is None or int(r["anchor_position"]) not in arrays.anchors:
            raise ValueError("Stage-A scaler anchor outside restricted population")
    return "PASS"
