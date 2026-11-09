import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.daily_strategies import run_daily_strategy_suite


def daily_fixture():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    index = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 102.0, 100.0, 99.0]})
    rows = []
    for day_i, day in enumerate(dates):
        for typ in ("CE", "PE"):
            rows.append({
                "trade_date": day, "expiry_date": dates[-1] + pd.Timedelta(days=2),
                "symbol": f"SENSEX_{typ}", "strike": 101.0, "option_type": typ,
                "close": 10.0 + day_i if typ == "CE" else 12.0 - day_i,
                "lot_size": 10, "volume": 100, "open_interest": 500,
            })
    return index, pd.DataFrame(rows)


def test_daily_suite_runs_calls_puts_and_directional_strategies():
    index, options = daily_fixture()
    cost = CostConfig()
    ledger, ranking, meta = run_daily_strategy_suite(index, options, capital=10000, cost=cost, max_atm_distance_pct=0.02)
    assert set(ranking["strategy"]) == {
        "DAILY_ATM_CALL", "DAILY_ATM_PUT", "DAILY_MOMENTUM", "DAILY_MEAN_REVERSION"
    }
    assert len(ledger[ledger["strategy"] == "DAILY_ATM_CALL"]) == 2
    assert len(ledger[ledger["strategy"] == "DAILY_ATM_PUT"]) == 2
    assert set(ledger["option_type"]) == {"CE", "PE"}
    assert meta["holding_period"].startswith("one SENSEX trading session")


def test_daily_signal_uses_prior_closes_and_exit_is_next_session():
    index, options = daily_fixture()
    ledger, _, _ = run_daily_strategy_suite(index, options, capital=10000, cost=CostConfig(), max_atm_distance_pct=0.02)
    call = ledger[ledger["strategy"] == "DAILY_MOMENTUM"].sort_values("entry_date")
    assert call.iloc[0]["signal_direction"] == "UP"
    assert call.iloc[0]["option_type"] == "CE"
    assert call.iloc[0]["exit_date"] == call.iloc[0]["entry_date"] + pd.offsets.BDay(1)


def test_daily_suite_skips_missing_next_session_quote():
    index, options = daily_fixture()
    entry = pd.Timestamp("2024-01-03")
    exit_day = pd.Timestamp("2024-01-04")
    options = options[~((options["trade_date"] == exit_day) & (options["option_type"] == "CE"))]
    ledger, _, _ = run_daily_strategy_suite(index, options, capital=10000, cost=CostConfig())
    assert not ((ledger["entry_date"] == entry) & (ledger["option_type"] == "CE")).any()



def test_daily_holdout_is_chronological_and_reports_only_test_window():
    from sensex_options.daily_strategies import daily_holdout_evaluation

    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    ledger = pd.DataFrame({
        "strategy": ["DAILY_ATM_CALL"] * 20,
        "entry_date": dates,
        "exit_date": dates + pd.offsets.BDay(1),
        "net_pnl": [100.0, -50.0] * 10,
    })
    result = daily_holdout_evaluation(ledger)
    assert result["status"] == "ok"
    assert result["holdout_sessions"] == 6
    assert result["holdout_start"] > result["train_end"]
    assert result["holdout_metrics"][0]["trades"] == 6



def test_extended_daily_indicators_are_replayed_when_history_is_sufficient():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    closes = list(range(1020, 1000, -1)) + list(range(1001, 1021))
    index = pd.DataFrame({"date": dates, "close": closes})
    rows = []
    expiry = dates[-1] + pd.Timedelta(days=7)
    for i, day in enumerate(dates):
        strike = 1010.0
        for typ in ("CE", "PE"):
            rows.append({
                "trade_date": day, "expiry_date": expiry, "symbol": f"SENSEX_{typ}",
                "strike": strike, "option_type": typ,
                "close": float(20 + i if typ == "CE" else 60 - i * 0.5),
                "lot_size": 10, "volume": 100, "open_interest": 500,
            })
    ledger, board, _ = run_daily_strategy_suite(index, pd.DataFrame(rows), capital=10000, cost=CostConfig(), max_atm_distance_pct=0.03)
    assert {"DAILY_5D_MOMENTUM", "DAILY_EMA_TREND", "DAILY_RSI_REVERSAL"}.issubset(set(board["strategy"]))
    assert {"CE", "PE"}.issubset(set(ledger["option_type"]))


def test_default_daily_research_does_not_apply_ten_thousand_affordability_gate():
    index, options = daily_fixture()
    options["close"] = 1500.0
    research, _, _ = run_daily_strategy_suite(
        index, options, cost=CostConfig(), max_atm_distance_pct=0.02
    )
    constrained, _, _ = run_daily_strategy_suite(
        index, options, capital=10000, cost=CostConfig(), max_atm_distance_pct=0.02
    )
    assert not research.empty
    assert constrained.empty


def test_zero_premium_is_kept_as_observed_expiry_loss():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    index = pd.DataFrame({"date": dates, "close": [100.0, 100.0, 100.0, 100.0]})
    rows = []
    for side in ("CE", "PE"):
        rows.extend([
            {
                "trade_date": dates[2], "expiry_date": dates[3], "symbol": f"SENSEX_{side}",
                "strike": 100.0, "option_type": side, "close": 10.0,
                "lot_size": 10, "volume": 100, "open_interest": 500,
            },
            {
                "trade_date": dates[3], "expiry_date": dates[3], "symbol": f"SENSEX_{side}",
                "strike": 100.0, "option_type": side, "close": 0.0,
                "lot_size": 10, "volume": 0, "open_interest": 0,
            },
        ])
    ledger, _, _ = run_daily_strategy_suite(index, pd.DataFrame(rows), cost=CostConfig())
    call = ledger[ledger["strategy"] == "DAILY_ATM_CALL"]
    assert len(call) == 1
    assert call.iloc[0]["exit_premium"] == 0.0
    assert call.iloc[0]["net_pnl"] < 0
