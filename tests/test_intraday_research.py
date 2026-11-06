import pandas as pd

from sensex_options.intraday_research import (
    evaluate_intraday_put_filters,
    intraday_session_features,
    latest_previous_close,
)


def test_previous_close_returns_latest_completed_close_and_change():
    index = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-04", "2026-11-05"]),
        "close": [82000.0, 82500.0],
    })
    result = latest_previous_close(index)
    assert result["status"] == "ok"
    assert result["close"] == 82500.0
    assert result["change"] == 500.0
    assert round(result["change_pct"], 3) == 0.61


def test_intraday_features_and_chronological_filter_validation():
    dates = pd.bdate_range("2026-01-01", periods=20)
    bars = []
    ledger = []
    for i, day in enumerate(dates):
        start = 80000.0 + i * 10
        # Three observed bars per session; some bearish, some bullish.
        direction = -1 if i % 2 == 0 else 1
        closes = [start, start + direction * 20, start + direction * 40]
        for j, close in enumerate(closes):
            bars.append({
                "timestamp": day + pd.Timedelta(hours=9, minutes=15 + j * 5),
                "open": start if j == 0 else closes[j - 1],
                "high": max(start, close) + 5,
                "low": min(start, close) - 5,
                "close": close,
                "volume": 100,
            })
        ledger.append({
            "strategy": "DAILY_ATM_PUT",
            "entry_date": day,
            "net_pnl": 200.0 if i % 3 else -100.0,
        })
    features = intraday_session_features(pd.DataFrame(bars))
    assert len(features) == 20
    assert {"session_return_pct", "close_vs_intraday_mean_pct", "range_pct"}.issubset(features.columns)

    joined, report, _ = evaluate_intraday_put_filters(pd.DataFrame(ledger), pd.DataFrame(bars))
    assert report["status"] == "ok"
    assert report["training_sessions"] == 14
    assert report["validation_sessions"] == 6
    assert report["threshold_fitted_on"] == "training sessions only"
    assert "Not tested" in report["stop_loss_status"]
    assert len(joined) == 20
