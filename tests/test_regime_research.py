import numpy as np
import pandas as pd

from sensex_options.regime_research import build_regime_features, run_regime_research


def test_regime_features_are_point_in_time_and_classified():
    dates = pd.bdate_range("2024-01-01", periods=180)
    close = [100.0]
    for i in range(1, len(dates)):
        close.append(close[-1] * (1.002 if i % 3 else 0.999))
    frame = pd.DataFrame({
        "date": dates, "open": close, "high": np.array(close) * 1.01,
        "low": np.array(close) * 0.99, "close": close,
    })
    features = build_regime_features(frame)
    changed = frame.copy()
    changed.loc[150:, ["open", "high", "low", "close"]] *= 2
    other = build_regime_features(changed)
    pd.testing.assert_series_equal(features.loc[:149, "ema_20"], other.loc[:149, "ema_20"])
    assert set(features["regime"].dropna()).issubset(
        {"TREND_UP", "TREND_DOWN", "HIGH_VOL", "LOW_VOL", "RANGE_OR_MIXED"}
    )


def test_regime_research_lags_index_and_splits_chronologically():
    dates = pd.bdate_range("2024-01-01", periods=220)
    close = [100.0]
    for i in range(1, len(dates)):
        drift = 1.002 if (i // 30) % 2 == 0 else 0.998
        close.append(close[-1] * drift)
    index = pd.DataFrame({
        "date": dates, "open": close, "high": np.array(close) * 1.006,
        "low": np.array(close) * 0.994, "close": close,
    })
    ledger = []
    for i in range(1, len(dates) - 1):
        entry = dates[i]
        # create both CE and PE observed outcomes, with no fabricated option prices
        for side in ("CE", "PE"):
            ledger.append({
                "entry_date": entry, "exit_date": dates[i + 1],
                "option_type": side, "net_pnl": 100 if (i % 4 == 0) == (side == "CE") else -80,
            })
    joined, board, report = run_regime_research(index, pd.DataFrame(ledger), minimum_trades=10)
    assert report["status"] == "screen_complete"
    assert not board.empty
    assert report["train_end"] < report["validation_start"]
    row = joined.sort_values("entry_date").iloc[0]
    assert row["signal_date"] < row["entry_date"]
    assert set(board["strategy"].str.split("__").str[0]).issubset(
        {"MOMENTUM_5D", "MOMENTUM_10D", "TREND_EMA_ADX", "TREND_PULLBACK", "MEAN_REVERSION"}
    )
