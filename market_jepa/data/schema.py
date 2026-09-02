MARKET_FEATURES = (
    "log_open",
    "log_high",
    "log_low",
    "log_close",
    "close_log_return",
    "open_to_prev_close",
    "high_to_prev_close",
    "low_to_prev_close",
    "normalized_range",
    "log1p_volume",
    "volume_log_change",
    "log1p_open_interest",
    "open_interest_log_change",
    "realized_volatility",
)

CONTEXT_FEATURES = (
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "log1p_delta_minutes",
)

PERIOD_CONTEXT_FEATURES = ("source_bar_count",)
RAW_COLUMNS = ("open", "high", "low", "close", "volume", "open_interest")
