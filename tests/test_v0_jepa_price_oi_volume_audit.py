from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from market_jepa.data.schema import MARKET_FEATURES
from v0_jepa_price_oi_volume_audit import (
    OI_FEATURES,
    PRICE_FEATURES,
    VOLUME_FEATURES,
    _sample_indices,
    marginal_summaries,
    match_donors,
    relation_interventions,
)


def test_feature_groups_are_complete_and_disjoint() -> None:
    grouped = set(PRICE_FEATURES) | set(OI_FEATURES) | set(VOLUME_FEATURES)
    assert grouped == set(MARKET_FEATURES)
    assert set(PRICE_FEATURES).isdisjoint(OI_FEATURES)
    assert set(PRICE_FEATURES).isdisjoint(VOLUME_FEATURES)
    assert set(OI_FEATURES).isdisjoint(VOLUME_FEATURES)


def test_relation_interventions_replace_only_requested_minute_channels() -> None:
    shape = (2, 512, len(MARKET_FEATURES))
    original = torch.zeros(shape)
    joint = torch.ones(shape)
    independent = torch.full(shape, 2.0)
    result = relation_interventions(original, joint, independent)
    oi = [MARKET_FEATURES.index(name) for name in OI_FEATURES]
    volume = [MARKET_FEATURES.index(name) for name in VOLUME_FEATURES]
    other = sorted(set(range(len(MARKET_FEATURES))) - set(oi) - set(volume))

    assert torch.all(result["oi_relation_broken"][..., oi] == 1)
    assert torch.all(result["oi_relation_broken"][..., volume + other] == 0)
    assert torch.all(result["volume_relation_broken"][..., volume] == 1)
    assert torch.all(result["volume_relation_broken"][..., oi + other] == 0)
    assert torch.all(result["price_ov_relation_broken"][..., oi + volume] == 1)
    assert torch.all(result["price_ov_relation_broken"][..., other] == 0)
    assert torch.all(result["all_relations_broken"][..., oi] == 1)
    assert torch.all(result["all_relations_broken"][..., volume] == 2)
    assert torch.all(result["all_relations_broken"][..., other] == 0)


def test_sampling_is_deterministic_without_replacement() -> None:
    population = np.arange(10_000, dtype=np.int64)
    left = _sample_indices(population, 8192, np.random.Generator(np.random.PCG64(4501)))
    right = _sample_indices(population, 8192, np.random.Generator(np.random.PCG64(4501)))
    np.testing.assert_array_equal(left, right)
    assert len(np.unique(left)) == 8192


def test_donor_matching_obeys_time_and_week_constraints() -> None:
    anchors = np.arange(512, 512 + 30, dtype=np.int64)
    minute_market = np.zeros((600, len(MARKET_FEATURES)), dtype=np.float32)
    minute_market[:, MARKET_FEATURES.index("open_interest_log_change")] = np.linspace(0, 1, 600)
    minute_market[:, MARKET_FEATURES.index("log1p_volume")] = np.linspace(1, 2, 600)
    minute_market[:, MARKET_FEATURES.index("volume_log_change")] = np.linspace(2, 3, 600)
    minute_market[:, MARKET_FEATURES.index("log1p_open_interest")] = np.linspace(3, 4, 600)
    summary, _ = marginal_summaries(minute_market, anchors)

    timestamp = pd.to_datetime(
        [
            "2021-01-04 09:00",
            "2021-01-11 09:05",
            "2021-01-18 09:10",
        ]
        * 200
    )[:600]
    frame = pd.DataFrame({"timestamp": timestamp})
    frame["trading_day"] = frame["timestamp"].dt.normalize()
    iso = frame["trading_day"].dt.isocalendar()
    frame["iso_key"] = iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)
    # Align the three eligible weeks to contiguous anchor groups while preserving
    # identical weekday/session and <=30 minute time matching.
    for position, anchor in enumerate(anchors):
        week = position % 3
        frame.loc[anchor, "timestamp"] = pd.Timestamp("2021-01-04 09:00") + pd.Timedelta(
            days=7 * week, minutes=position % 10
        )
        frame.loc[anchor, "trading_day"] = frame.loc[anchor, "timestamp"].normalize()
        iso_value = frame.loc[anchor, "trading_day"].isocalendar()
        frame.loc[anchor, "iso_key"] = int(iso_value.year) * 100 + int(iso_value.week)

    bases = anchors[:3]
    donors = match_donors(frame, anchors, bases, summary, summary[:3], 4501)
    base_rows = frame.iloc[bases]
    joint_rows = frame.iloc[donors.joint]
    independent_rows = frame.iloc[donors.independent]
    assert np.all(base_rows["trading_day"].dt.weekday.to_numpy() == joint_rows["trading_day"].dt.weekday.to_numpy())
    assert np.all(base_rows["iso_key"].to_numpy() != joint_rows["iso_key"].to_numpy())
    assert np.all(joint_rows["iso_key"].to_numpy() != independent_rows["iso_key"].to_numpy())
