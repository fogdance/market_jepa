from __future__ import annotations

import numpy as np

from market_jepa.data.schema import CONTEXT_FEATURES


def test_night_partial_daily_and_day_session_update_same_token(causal_data) -> None:
    data = causal_data
    first_night, night_end, first_day = 2, 3, 4
    night_market, night_context, night_sources = data.daily_snapshot(first_night)
    assert len(night_market) == 2  # completed Friday + partial Monday trading day
    assert data.daily_partial.iloc[first_night]["source_bar_count"] == 1
    assert data.daily_partial.iloc[first_night]["open"] == 103
    assert data.daily_partial.iloc[night_end]["high"] == 105
    assert data.daily_partial.iloc[night_end]["low"] == 101
    assert data.daily_partial.iloc[night_end]["close"] == 104
    assert data.daily_partial.iloc[night_end]["volume"] == 70
    assert data.daily_partial.iloc[night_end]["open_interest"] == 1030
    assert data.daily_partial.iloc[first_day]["source_bar_count"] == 3
    assert len(data.daily_snapshot(first_day)[0]) == len(night_market)
    assert int(night_sources[-1]) == first_night
    assert night_context.shape[1] == 1


def test_partial_high_cannot_see_future_and_weekly_tracks_partial(causal_data) -> None:
    data = causal_data
    anchor = 4
    assert data.daily_partial.iloc[anchor]["high"] == 106
    assert data.minute.iloc[anchor + 1]["high"] == 999
    assert data.weekly_partial.iloc[anchor]["high"] == 106
    assert data.weekly_partial.iloc[anchor]["volume"] == 120
    assert data.weekly_partial.iloc[anchor + 1]["high"] == 999
    assert data.weekly_partial.iloc[anchor + 1]["volume"] == 180


def test_friday_night_starts_next_trading_week(causal_data) -> None:
    data = causal_data
    friday_day_key = int(data.minute.iloc[1]["iso_key"])
    friday_night_key = int(data.minute.iloc[2]["iso_key"])
    monday_day_key = int(data.minute.iloc[4]["iso_key"])
    assert friday_night_key == monday_day_key
    assert friday_night_key != friday_day_key
    assert data.weekly_partial.iloc[2]["source_bar_count"] == 1


def test_every_snapshot_source_is_causal(causal_data) -> None:
    for anchor in range(len(causal_data.minute)):
        _, _, daily_sources = causal_data.daily_snapshot(anchor)
        _, _, weekly_sources = causal_data.weekly_snapshot(anchor)
        assert np.all(daily_sources <= anchor)
        assert np.all(weekly_sources <= anchor)
        assert daily_sources[-1] == anchor
        assert weekly_sources[-1] == anchor


def test_time_observation_uses_five_day_cycle_and_log_gap(causal_data) -> None:
    raw = causal_data.minute_context_raw
    names = list(CONTEXT_FEATURES)
    assert "delta_minutes" not in names
    monday = 2
    assert np.isclose(raw[monday, names.index("day_of_week_sin")], 0.0)
    delta = (causal_data.minute.iloc[2]["timestamp"] - causal_data.minute.iloc[1]["timestamp"]).total_seconds() / 60
    assert np.isclose(raw[2, names.index("log1p_delta_minutes")], np.log1p(delta))
