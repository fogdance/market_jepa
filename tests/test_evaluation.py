from __future__ import annotations

import numpy as np
import pytest

from market_jepa.config import DEFAULT_CONFIG
from market_jepa.eval.metrics import block_bootstrap, block_shuffle_indices, nearest_indices
from market_jepa.eval.pipeline import _future_report, evaluate_exports
from market_jepa.eval.raw_baseline import raw_baseline_features, raw_feature_names


def test_raw_baseline_gets_equal_context_observations(causal_data) -> None:
    anchor = 4
    minute = raw_baseline_features(causal_data, anchor, 2, "minute")
    daily = raw_baseline_features(causal_data, anchor, 2, "minute_daily")
    weekly = raw_baseline_features(causal_data, anchor, 2, "minute_daily_weekly")
    assert len(minute) == len(raw_feature_names("minute"))
    assert len(daily) == len(raw_feature_names("minute_daily"))
    assert len(weekly) == len(raw_feature_names("minute_daily_weekly"))
    minute_names = raw_feature_names("minute")
    for name in (
        "current_time_of_day_sin",
        "current_day_of_week_cos",
        "current_log1p_delta_minutes",
    ):
        assert name in minute_names
    assert "current_daily_source_bar_count" not in minute_names
    assert "current_daily_source_bar_count" in raw_feature_names("minute_daily")
    assert "current_weekly_source_bar_count" in raw_feature_names("minute_daily_weekly")


def test_knn_rejects_query_time_or_future_reference() -> None:
    reference = np.eye(3)
    query = np.eye(3)[:1]
    with pytest.raises(ValueError, match="future"):
        nearest_indices(reference, query, np.array([1, 2, 4]), np.array([4]), k=1)


def test_block_bootstrap_is_reproducible() -> None:
    effects = np.asarray([1.0, 2.0, -1.0, 3.0])
    blocks = np.asarray([1, 1, 2, 3])
    first = block_bootstrap(effects, blocks, samples=100, seed=42)
    second = block_bootstrap(effects, blocks, samples=100, seed=42)
    assert first == second
    assert first["effect"] == pytest.approx(effects.mean())


def test_block_bootstrap_preserves_sample_weighted_estimand() -> None:
    effects = np.concatenate([np.ones(100), np.asarray([-10.0])])
    blocks = np.concatenate([np.zeros(100, dtype=int), np.asarray([1])])
    result = block_bootstrap(effects, blocks, samples=1_000, seed=42)
    assert result["effect"] == pytest.approx(effects.mean())
    assert result["effect"] == pytest.approx(0.8910891089108911)


def test_prediction_effect_equals_difference_of_mean_errors() -> None:
    target = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    prediction = np.asarray([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    persistence = np.asarray([[0.7, 0.3], [1.0, 0.0], [0.5, 0.5]])
    timestamps = np.asarray(
        ["2024-01-02T09:01", "2024-01-02T09:02", "2024-01-03T09:01"],
        dtype="datetime64[ns]",
    ).astype(np.int64)
    train = {"z_target_h1": target}
    data = {
        "z_target_h1": target,
        "z_prediction_h1": prediction,
        "z_persistence_h1": persistence,
        "timestamp_ns": timestamps,
        "trading_day_ns": timestamps.astype("datetime64[ns]").astype("datetime64[D]").astype(np.int64),
    }
    report = _future_report(train, data, horizon=1, bootstrap_samples=100, seed=42)
    assert report["persistence_improvement"]["effect"] == pytest.approx(
        report["persistence_cosine_error"] - report["learned_cosine_error"]
    )


def test_block_shuffle_never_crosses_calendar_month_or_session() -> None:
    timestamps = np.asarray(
        [
            "2023-01-02T09:00",
            "2023-01-03T10:00",
            "2023-01-02T21:00",
            "2023-01-03T22:00",
            "2024-01-02T09:00",
            "2024-01-03T10:00",
        ],
        dtype="datetime64[ns]",
    )
    shuffled = block_shuffle_indices(timestamps.astype(np.int64), seed=42)
    month = timestamps.astype("datetime64[M]")
    session = timestamps.astype("datetime64[h]").astype(int) % 24 >= 18
    assert np.all(month[shuffled] == month)
    assert np.all(session[shuffled] == session)
    assert np.all(shuffled != np.arange(len(shuffled)))
    assert set(shuffled[:2]) == {0, 1}
    assert set(shuffled[4:]) == {4, 5}


def test_evaluation_rejects_latents_from_another_checkpoint(tmp_path) -> None:
    for split in ("train", "validation", "test"):
        np.savez_compressed(
            tmp_path / f"{split}.npz",
            split=np.asarray([split]),
            design_version=np.asarray(["0.6.1"]),
            ablation=np.asarray(["minute_daily_weekly"]),
            symbol=np.asarray(["JM"]),
            series_id=np.asarray(["8Y_DCE_JM2601"]),
            checkpoint_sha256=np.asarray(["wrong"]),
            source_sha256=np.asarray(["source"]),
        )
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        evaluate_exports(
            tmp_path,
            DEFAULT_CONFIG,
            expected_checkpoint_sha256="expected",
            expected_source_sha256="source",
        )
