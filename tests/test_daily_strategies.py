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
                "lot_size": 10,
            })
    return index, pd.DataFrame(rows)


def test_daily_suite_runs_calls_puts_and_directional_strategies():
    index, options = daily_fixture()
    cost = CostConfig()
    ledger, ranking, meta = run_daily_strategy_suite(index, options, capital=10000, cost=cost)
    assert set(ranking["strategy"]) == {
        "DAILY_ATM_CALL", "DAILY_ATM_PUT", "DAILY_MOMENTUM", "DAILY_MEAN_REVERSION"
    }
    assert len(ledger[ledger["strategy"] == "DAILY_ATM_CALL"]) == 2
    assert len(ledger[ledger["strategy"] == "DAILY_ATM_PUT"]) == 2
    assert set(ledger["option_type"]) == {"CE", "PE"}
    assert meta["holding_period"].startswith("one SENSEX trading session")


def test_daily_signal_uses_prior_closes_and_exit_is_next_session():
    index, options = daily_fixture()
    ledger, _, _ = run_daily_strategy_suite(index, options, capital=10000, cost=CostConfig())
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
