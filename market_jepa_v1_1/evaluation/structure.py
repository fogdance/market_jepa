from __future__ import annotations

from copy import deepcopy

import numpy as np

from .protocol import PROTOCOL

REMOVALS = {
    "Full": (), "NoDaily": ("daily",), "NoCurrentWeekly": ("current_weekly",),
    "NoHistoricalWeekly": ("history_weekly",),
    "MinuteOnly": ("daily", "current_weekly", "history_weekly"),
}


def remove_sources(batch, variant):
    result = dict(batch)
    for source in REMOVALS[variant]:
        for suffix in ("market", "context", "imc_validity"):
            result[source + "_" + suffix] = batch[source + "_" + suffix].clone().zero_()
        result[source + "_mask"] = batch[source + "_mask"].clone().fill_(True)
        if source == "history_weekly":
            result["history_weekly_contract_boundary"] = batch["history_weekly_contract_boundary"].clone().zero_()
    return result


def donor_manifest(records, *, budget=PROTOCOL["structure_audit_anchors"], seed=42):
    rng = np.random.default_rng(seed)
    pools = {}
    for i, row in enumerate(records):
        if row["full_minute"]:
            pools.setdefault((row["commodity"], row["series_key"], row["split"]), []).append(i)
    eligible, omitted = [], []
    for i, r in enumerate(records):
        if not r["full_minute"]:
            omitted.append({"recipient": i, "reason": "minute_padding"}); continue
        candidates = []
        for j in pools[(r["commodity"], r["series_key"], r["split"])]:
            d = records[j]
            disjoint = d["anchor_timestamp"] < r["window_start"] or r["anchor_timestamp"] < d["window_start"]
            if disjoint and (d["contract_uid"] != r["contract_uid"] or d["trading_date"] != r["trading_date"]):
                candidates.append(j)
        preferred = [j for j in candidates if records[j]["contract_uid"] != r["contract_uid"]]
        candidates = preferred or candidates
        if candidates:
            eligible.append({"recipient": i, "donor": int(rng.choice(candidates))})
        else:
            omitted.append({"recipient": i, "reason": "no_same_partition_nonoverlapping_donor"})
    # Equal-commodity then round-robin recipient selection, never duplicate recipients.
    groups = {}
    for pair in eligible:
        groups.setdefault(records[pair["recipient"]]["commodity"], []).append(pair)
    for key in groups:
        rng.shuffle(groups[key])
    selected = []
    while any(groups.values()) and len(selected) < budget:
        for key in sorted(groups):
            if groups[key] and len(selected) < budget:
                selected.append(groups[key].pop())
    return {"seed": seed, "pairs": selected, "omitted": omitted,
            "claim": "Price-aligned OI/Volume dependence"}


def replace_channels(recipient, donor, intervention):
    channels = {"OI": (5, 6), "Volume": (7, 8), "Joint": (5, 6, 7, 8)}[intervention]
    result = dict(recipient)
    for key in ("minute_market", "minute_imc_validity"):
        result[key] = recipient[key].clone()
        result[key][..., list(channels)] = donor[key][..., list(channels)]
    return result
