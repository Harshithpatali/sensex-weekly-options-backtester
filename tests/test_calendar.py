"""Calendar / expiry tests — no hardcoded weekday assumption."""

import pandas as pd

from sensex_options.calendar import (
    available_expiries,
    previous_trading_session,
    validate_expiry_pair,
)


def test_previous_session_not_weekday_hardcoded():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-05"]),  # Thu Fri Mon
        "close": [1, 2, 3],
    })
    # Expiry on Tuesday 2023-06-06 → previous session is Monday 2023-06-05
    prev = previous_trading_session(index, pd.Timestamp("2023-06-06"))
    assert prev == pd.Timestamp("2023-06-05")


def test_available_expiries_from_data():
    opts = pd.DataFrame({
        "expiry_date": pd.to_datetime(["2023-06-08", "2023-06-15", "2023-06-08"]),
    })
    exp = available_expiries(opts)
    assert len(exp) == 2
    assert exp[0] == pd.Timestamp("2023-06-08")


def test_validate_expiry_requires_contracts():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01", "2023-06-08"]),
        "close": [80000, 80100],
    })
    options = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-06-01"]),
        "expiry_date": pd.to_datetime(["2023-06-08"]),
        "option_type": ["CE"],
        "strike": [80100],
        "close": [100],
    })
    ok, reason, entry, exit_d = validate_expiry_pair(index, options, pd.Timestamp("2023-06-08"))
    assert ok
    assert entry == pd.Timestamp("2023-06-01")
