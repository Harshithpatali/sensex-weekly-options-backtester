"""Validation module tests."""
import pandas as pd
from sensex_options.validation import validate_all


def test_empty_data_reports_errors():
    report = validate_all(pd.DataFrame(), pd.DataFrame())
    assert report["status"] == "incomplete"
    assert any("Index" in error for error in report["errors"])
    assert any("Options" in error for error in report["errors"])


def test_valid_minimal():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01"]), "close": [80000.0],
        "open": [80000.0], "high": [80000.0], "low": [80000.0], "volume": [1],
    })
    options = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
        "expiry_date": pd.to_datetime(["2023-06-08", "2023-06-08"]),
        "strike": [80100, 80100], "option_type": ["CE", "PE"], "close": [100, 110],
        "lot_size": [10, 10], "volume": [50, 60], "symbol": ["SENSEX", "SENSEX"],
    })
    report = validate_all(index, options)
    assert report["status"] == "ok"
    assert report["expiries"] == 1


def test_missing_required_option_column_is_reported_not_crashed():
    index = pd.DataFrame({"date": pd.to_datetime(["2023-06-01"]), "close": [80000]})
    options = pd.DataFrame({"trade_date": ["2023-06-01"]})
    report = validate_all(index, options)
    assert report["status"] == "incomplete"
    assert any("missing required columns" in error for error in report["errors"])


def test_validation_reports_expiry_and_session_coverage():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"]),
        "close": [100.0, 101.0, 102.0],
    })
    options = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-04", "2024-01-04"]),
        "expiry_date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
        "strike": [100.0, 100.0], "option_type": ["CE", "PE"],
        "close": [5.0, 6.0], "lot_size": [10, 10],
        "volume": [1, 1], "symbol": ["SENSEX", "SENSEX"],
    })
    report = validate_all(index, options)
    assert report["expiry_completeness_pct"] == 100.0
    assert report["option_session_coverage_pct"] == 33.33
    assert report["expiry_coverage"][0]["has_both_sides"] is True
