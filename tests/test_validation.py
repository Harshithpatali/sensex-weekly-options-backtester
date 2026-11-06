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
        "trade_date": pd.to_datetime(["2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05"]),
        "expiry_date": pd.to_datetime(["2024-01-05"] * 4),
        "strike": [100.0, 100.0, 100.0, 100.0], "option_type": ["CE", "PE", "CE", "PE"],
        "close": [5.0, 6.0, 4.0, 3.0], "lot_size": [10, 10, 10, 10],
        "volume": [1, 1, 1, 1], "symbol": ["SENSEX"] * 4,
    })
    report = validate_all(index, options)
    assert report["expiry_completeness_pct"] == 100.0
    assert report["option_session_coverage_pct"] == 100.0
    assert report["expiry_coverage"][0]["complete_for_backtest"] is True


def test_expiry_completeness_requires_usable_same_strike_round_trips():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
        "close": [100.0, 101.0],
    })
    options = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-04"] * 2 + ["2024-01-05"] * 2),
        "expiry_date": pd.to_datetime(["2024-01-05"] * 4),
        "strike": [100.0] * 4,
        "option_type": ["CE", "PE", "CE", "PE"],
        "close": [5.0, 6.0, 0.0, 0.0],
        "lot_size": [10, 10, 10, 10],
        "symbol": ["SENSEX"] * 4,
    })
    report = validate_all(index, options)
    row = report["expiry_coverage"][0]
    assert row["has_usable_call_round_trip"] is True
    assert row["has_usable_put_round_trip"] is True
    assert row["complete_for_backtest"] is True

    options["lot_size"] = float("nan")
    report = validate_all(index, options)
    row = report["expiry_coverage"][0]
    assert row["complete_for_backtest"] is False
    assert report["complete_expiries"] == 0


def test_option_session_coverage_uses_options_archive_date_window():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "close": [100.0, 101.0, 102.0, 103.0],
    })
    options = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
        "expiry_date": pd.to_datetime(["2024-01-04", "2024-01-04"]),
        "strike": [100.0, 100.0],
        "option_type": ["CE", "PE"],
        "close": [5.0, 6.0],
        "lot_size": [10, 10],
        "symbol": ["SENSEX", "SENSEX"],
    })
    report = validate_all(index, options)
    assert report["index_sessions_in_option_range"] == 1
    assert report["option_session_coverage_pct"] == 100.0
