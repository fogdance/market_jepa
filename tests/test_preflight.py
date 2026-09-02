from market_jepa.data.preflight import build_preflight_report


def test_preflight_has_required_anomalies_and_distributions(causal_data, causal_config) -> None:
    report = build_preflight_report(causal_data, causal_config, top_n=3)
    assert set(report["anomalies"]) == {
        "absolute_1m_log_return",
        "cross_observation_gap",
        "volume_log_jump",
        "open_interest_log_jump",
    }
    for records in report["anomalies"].values():
        assert len(records) <= 3
        assert [record["rank"] for record in records] == list(range(1, len(records) + 1))
    for timeframe in ("daily", "weekly"):
        assert set(report["normalization_distributions"][timeframe]) == {
            "completed",
            "partial_all",
            "partial_early",
            "partial_late",
        }
    assert len(report["source"]["sha256"]) == 64
