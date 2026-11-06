import numpy as np
import pandas as pd

from sensex_options.advanced_signal_research import (
    build_lagged_features,
    build_signal_specs,
    run_advanced_signal_research,
)


def sample_index(periods=180):
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    t = np.arange(periods, dtype=float)
    close = 72000 + 18 * t + 180 * np.sin(t / 5.0) + 70 * np.sin(t / 13.0)
    return pd.DataFrame({
        "date": dates,
        "open": close - 15 * np.sin(t / 3.0),
        "high": close + 100 + 15 * np.sin(t / 2.0),
        "low": close - 100 - 15 * np.cos(t / 2.0),
        "close": close,
        "volume": 1000,
    })


def sample_options(index):
    rows = []
    dates = pd.to_datetime(index["date"])
    expiry = dates.max() + pd.Timedelta(days=30)
    for i, day in enumerate(dates):
        spot = float(index.iloc[i]["close"])
        strike = round(spot / 100) * 100
        for typ in ("CE", "PE"):
            premium = 130 + 15 * np.sin(i / 4) + (i % 7) * 2
            rows.append({
                "trade_date": day,
                "expiry_date": expiry,
                "symbol": "SENSEX",
                "strike": strike,
                "option_type": typ,
                "open": premium,
                "high": premium + 25,
                "low": max(1, premium - 25),
                "close": premium + 3 * np.sin(i / 3),
                "lot_size": 10,
            })
    return pd.DataFrame(rows)


def test_features_use_only_trailing_values_for_channel_boundaries():
    index = sample_index()
    features = build_lagged_features(index)
    i = 40
    expected = index["close"].iloc[i - 10:i].max()
    assert features.loc[i, "prior_close_high_10"] == expected
    assert pd.notna(features.loc[i, "vol20"])
    assert pd.notna(features.loc[i, "rsi14"])


def test_signal_library_contains_trend_reversal_and_regime_rules():
    features = build_lagged_features(sample_index())
    specs = build_signal_specs(features)
    names = set(specs)
    assert any(name.startswith("MOM") for name in names)
    assert any(name.startswith("RSI_REV") for name in names)
    assert any("ADX_GE" in name for name in names)
    assert any("LOWVOL" in name or "HIGHVOL" in name for name in names)
    assert any(name.startswith("CONSENSUS") for name in names)
    assert all(len(signal) == len(features) for signal in specs.values())


def test_advanced_signal_research_returns_chronological_monthly_metrics():
    index = sample_index(180)
    options = sample_options(index)
    ledger, monthly, board, report = run_advanced_signal_research(index, options)

    assert report["status"] in {
        "strict_candidates_found",
        "out_of_sample_candidates_found",
        "no_candidate_met_monthly_target",
    }
    assert report["candidate_rules_tested"] > 20
    assert not ledger.empty
    assert not monthly.empty
    assert not board.empty
    assert set(ledger["option_type"]).issubset({"CE", "PE"})
    assert set(monthly["month_status"]).issubset({"PASS", "FAIL", "INSUFFICIENT_TRADES"})
    assert report["holdout_start"] > report["train_end"]
    assert "train_target_hit_pct" in board.columns
    assert "holdout_target_hit_pct" in board.columns


def test_empty_inputs_are_reported_without_exception():
    ledger, monthly, board, report = run_advanced_signal_research(
        pd.DataFrame(), pd.DataFrame()
    )
    assert ledger.empty and monthly.empty and board.empty
    assert report["status"] == "insufficient_data"
