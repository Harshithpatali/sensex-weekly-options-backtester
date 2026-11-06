"""Business-logic tests for the backtest engine."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from sensex_options.backtest import select_strike, run_backtest, compute_summary
from sensex_options.config import BacktestConfig, CostConfig


def _index(dates, closes):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000_000] * len(dates),
    })


def _opts(rows):
    """rows: list of dicts with trade_date, expiry_date, strike, close, lot_size, option_type, symbol"""
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["option_type"] = df.get("option_type", "CE")
    if "symbol" not in df.columns:
        df["symbol"] = "SENSEX"
    if "volume" not in df.columns:
        df["volume"] = 100
    return df


class TestStrikeSelection:
    def test_smallest_strictly_above(self):
        calls = pd.DataFrame({
            "strike": [80000, 80100, 80200, 80300],
            "close": [200, 150, 120, 90],
            "option_type": ["CE"] * 4,
        })
        row = select_strike(calls, 80125)
        assert row is not None
        assert row["strike"] == 80200

    def test_equal_strike_not_selected(self):
        calls = pd.DataFrame({
            "strike": [80000, 80100, 80200],
            "close": [200, 150, 120],
            "option_type": ["CE"] * 3,
        })
        row = select_strike(calls, 80100)
        assert row is not None
        assert row["strike"] == 80200  # strictly greater

    def test_no_eligible_returns_none(self):
        calls = pd.DataFrame({
            "strike": [80000, 80100],
            "close": [200, 150],
            "option_type": ["CE"] * 2,
        })
        assert select_strike(calls, 80500) is None


class TestCapitalAndLots:
    def test_whole_lots_only_and_no_negative_capital(self):
        # Entry session 2023-06-01, expiry 2023-06-08
        index = _index(
            ["2023-06-01", "2023-06-08"],
            [80125.0, 80500.0],
        )
        options = _opts([
            # entry day
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 120, "lot_size": 10, "option_type": "CE"},
            # exit day
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 150, "lot_size": 10, "option_type": "CE"},
        ])
        cfg = BacktestConfig(initial_capital=10_000)
        result = run_backtest(index, options, cfg)
        trades = result["trades"]
        completed = trades[trades["status"] == "completed"]
        assert len(completed) == 1
        t = completed.iloc[0]
        # cost per lot = 120 * 10 = 1200 → max lots = 8
        assert t["n_lots"] == 8
        assert t["entry_investment"] == 9600
        assert t["gross_pnl"] == 2400  # (150-120)*10*8
        assert result["summary"]["ending_capital"] == 12400
        assert result["summary"]["ending_capital"] >= 0

    def test_insufficient_capital_skips(self):
        index = _index(["2023-06-01", "2023-06-08"], [80125.0, 80500.0])
        options = _opts([
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 500, "lot_size": 10, "option_type": "CE"},
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 600, "lot_size": 10, "option_type": "CE"},
        ])
        cfg = BacktestConfig(initial_capital=1000)  # cannot afford 500*10
        result = run_backtest(index, options, cfg)
        skipped = result["trades"][result["trades"]["status"] == "skipped"]
        assert len(skipped) == 1
        assert skipped.iloc[0]["skip_reason"] == "insufficient_capital"


class TestMissingDataSkips:
    def test_missing_entry_premium_skips(self):
        index = _index(["2023-06-01", "2023-06-08"], [80125.0, 80500.0])
        options = _opts([
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80200, "close": np.nan, "lot_size": 10, "option_type": "CE"},
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 150, "lot_size": 10, "option_type": "CE"},
        ])
        result = run_backtest(index, options, BacktestConfig(initial_capital=10000))
        assert result["trades"].iloc[0]["skip_reason"] == "missing_or_invalid_entry_premium"

    def test_missing_exit_premium_skips(self):
        index = _index(["2023-06-01", "2023-06-08"], [80125.0, 80500.0])
        options = _opts([
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 120, "lot_size": 10, "option_type": "CE"},
            # no exit row for same strike
        ])
        result = run_backtest(index, options, BacktestConfig(initial_capital=10000))
        assert result["trades"].iloc[0]["skip_reason"] == "missing_exit_premium"

    def test_missing_lot_size_skips(self):
        index = _index(["2023-06-01", "2023-06-08"], [80125.0, 80500.0])
        options = _opts([
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 120, "lot_size": np.nan, "option_type": "CE"},
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 150, "lot_size": np.nan, "option_type": "CE"},
        ])
        result = run_backtest(index, options, BacktestConfig(initial_capital=10000))
        assert result["trades"].iloc[0]["skip_reason"] == "missing_historical_lot_size"


class TestSameContractAndNoLookahead:
    def test_entry_exit_same_strike(self):
        index = _index(["2023-06-01", "2023-06-08"], [80125.0, 80500.0])
        options = _opts([
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 120, "lot_size": 10, "option_type": "CE"},
            {"trade_date": "2023-06-01", "expiry_date": "2023-06-08",
             "strike": 80300, "close": 90, "lot_size": 10, "option_type": "CE"},
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 150, "lot_size": 10, "option_type": "CE"},
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80300, "close": 999, "lot_size": 10, "option_type": "CE"},
        ])
        result = run_backtest(index, options, BacktestConfig(initial_capital=10000))
        t = result["trades"][result["trades"]["status"] == "completed"].iloc[0]
        assert t["strike"] == 80200
        assert t["exit_premium"] == 150  # not the 999 of the other strike

    def test_no_future_data_for_entry(self):
        """Entry selection uses only entry-date rows."""
        index = _index(["2023-06-01", "2023-06-08"], [80125.0, 80500.0])
        options = _opts([
            # only future-day data for the chosen strike — should skip
            {"trade_date": "2023-06-08", "expiry_date": "2023-06-08",
             "strike": 80200, "close": 150, "lot_size": 10, "option_type": "CE"},
        ])
        result = run_backtest(index, options, BacktestConfig(initial_capital=10000))
        assert result["trades"].iloc[0]["status"] == "skipped"
        assert "no_call" in result["trades"].iloc[0]["skip_reason"]


class TestNoFabrication:
    def test_empty_options_no_trades(self):
        index = _index(["2023-06-01"], [80000.0])
        result = run_backtest(index, pd.DataFrame(), BacktestConfig())
        assert result["trades"].empty or result["summary"].get("completed_trades", 0) == 0
