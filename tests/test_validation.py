"""Validation module tests."""

import pandas as pd

from sensex_options.validation import validate_all


def test_empty_data_reports_errors():
    report = validate_all(pd.DataFrame(), pd.DataFrame())
    assert report["status"] == "incomplete"
    assert any("Index" in e for e in report["errors"])
    assert any("Options" in e for e in report["errors"])


def test_valid_minimal():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01"]),
        "close": [80000.0],
        "open": [80000.0], "high": [80000.0], "low": [80000.0], "volume": [1],
    })
    options = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-06-01"]),
        "expiry_date": pd.to_datetime(["2023-06-08"]),
        "strike": [80100],
        "option_type": ["CE"],
        "close": [100],
        "lot_size": [10],
        "volume": [50],
        "symbol": ["SENSEX"],
    })
    report = validate_all(index, options)
    assert report["status"] == "ok"
    assert report["expiries"] == 1
