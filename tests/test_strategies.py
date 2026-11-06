import pandas as pd

from sensex_options.strategies import run_strategy_suite, period_pnl


def fixture():
    index = pd.DataFrame({"date": ["2024-01-04", "2024-01-05"], "close": [100.0, 100.0]})
    rows = []
    for typ in ("CE", "PE"):
        for strike in (99.0, 100.0, 101.0):
            rows += [
                {"trade_date": "2024-01-04", "expiry_date": "2024-01-05", "symbol": "SENSEX",
                 "strike": strike, "option_type": typ, "close": 5.0, "lot_size": 10},
                {"trade_date": "2024-01-05", "expiry_date": "2024-01-05", "symbol": "SENSEX",
                 "strike": strike, "option_type": typ, "close": 6.0, "lot_size": 10},
            ]
    return index, pd.DataFrame(rows)


def test_strategy_suite_generates_five_baselines():
    index, options = fixture()
    trades, ranking, meta = run_strategy_suite(index, options, capital=10000, brokerage_per_order=0, slippage_points=0)
    assert set(ranking.strategy) == {"ATM_CALL", "ATM_PUT", "OTM_CALL_1", "OTM_PUT_1", "ATM_STRADDLE"}
    assert len(trades) == 5
    assert ranking.net_pnl.notna().all()
    assert {"starting_capital", "ending_capital"}.issubset(trades.columns)
    assert meta["initial_capital_per_strategy"] == 10000


def test_period_pnl_has_daily_weekly_monthly():
    index, options = fixture()
    trades, _, _ = run_strategy_suite(index, options, capital=10000, brokerage_per_order=0, slippage_points=0)
    reports = period_pnl(trades)
    assert set(reports) == {"daily", "weekly", "monthly"}
    assert not reports["daily"].empty


def test_missing_lot_size_does_not_crash():
    index, options = fixture()
    options["lot_size"] = float("nan")
    trades, _, _ = run_strategy_suite(index, options)
    assert trades.empty


def test_straddle_requires_both_legs_marked_on_same_exit_date():
    index, options = fixture()
    options = options[~((options["trade_date"] == "2024-01-05") & (options["option_type"] == "PE"))]
    trades, _, _ = run_strategy_suite(index, options, capital=10000, brokerage_per_order=0, slippage_points=0)
    assert "ATM_STRADDLE" not in set(trades["strategy"])



def test_strategy_capital_compounds_between_expiries():
    dates = ["2024-01-04", "2024-01-05", "2024-01-11", "2024-01-12"]
    index = pd.DataFrame({"date": dates, "close": [100.0] * 4})
    rows = []
    for expiry, entry_day, exit_day in [
        ("2024-01-05", "2024-01-04", "2024-01-05"),
        ("2024-01-12", "2024-01-11", "2024-01-12"),
    ]:
        for option_type in ("CE", "PE"):
            for strike in (99.0, 100.0, 101.0):
                rows.extend([
                    {"trade_date": entry_day, "expiry_date": expiry, "symbol": "SENSEX",
                     "strike": strike, "option_type": option_type, "close": 5.0, "lot_size": 10},
                    {"trade_date": exit_day, "expiry_date": expiry, "symbol": "SENSEX",
                     "strike": strike, "option_type": option_type, "close": 6.0, "lot_size": 10},
                ])
    trades, _, _ = run_strategy_suite(
        index, pd.DataFrame(rows), capital=10000, brokerage_per_order=0, slippage_points=0
    )
    atm_calls = trades[trades.strategy == "ATM_CALL"].sort_values("exit_date")
    assert len(atm_calls) == 2
    assert atm_calls.iloc[0]["ending_capital"] == 12000.0
    assert atm_calls.iloc[1]["starting_capital"] == 12000.0


def test_costed_strategy_suite_charges_both_straddle_legs():
    from sensex_options.config import CostConfig
    index, options = fixture()
    cost = CostConfig(
        brokerage_per_lot=10, stt_pct=0.001, exchange_charges_pct=0.0003,
        gst_pct=0.18, stamp_duty_pct=0.00003, entry_slippage=0.5, exit_slippage=0.5,
    )
    trades, ranking, _ = run_strategy_suite(index, options, capital=10000, cost=cost)
    assert not trades.empty
    straddle = trades[trades.strategy == "ATM_STRADDLE"].iloc[0]
    assert straddle["brokerage"] == 10 * straddle["lots"] * 4
    assert straddle["estimated_costs"] > straddle["brokerage"]
    assert straddle["net_pnl"] < straddle["gross_pnl"]
    assert {"profit_factor", "max_drawdown_pct"}.issubset(ranking.columns)


def test_strategy_suite_does_not_use_stale_pre_expiry_exit_quote():
    index, options = fixture()
    options = options[~((options["trade_date"] == "2024-01-05") & (options["option_type"] == "CE"))]
    trades, _, _ = run_strategy_suite(index, options, brokerage_per_order=0, slippage_points=0)
    assert "ATM_CALL" not in set(trades["strategy"])
    assert "ATM_PUT" in set(trades["strategy"])


def test_zero_expiry_exit_premium_is_a_valid_observed_loss():
    index, options = fixture()
    mask = (
        (options["trade_date"] == "2024-01-05")
        & (options["option_type"] == "CE")
        & (options["strike"] == 100.0)
    )
    options.loc[mask, "close"] = 0.0
    trades, _, _ = run_strategy_suite(index, options, brokerage_per_order=0, slippage_points=0)
    atm_call = trades[trades.strategy == "ATM_CALL"].iloc[0]
    assert atm_call["raw_exit_credit"] == 0.0
    assert atm_call["net_pnl"] < 0
