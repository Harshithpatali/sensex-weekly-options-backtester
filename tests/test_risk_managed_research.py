import pandas as pd

from sensex_options.risk_managed_research import (
    _simulate_exit,
    run_risk_managed_research,
)


def test_stop_and_target_same_session_assumes_stop_first():
    exit_price, reason = _simulate_exit(
        entry=100.0,
        close=110.0,
        open_price=100.0,
        high=130.0,
        low=70.0,
        stop_pct=0.20,
        target_pct=0.20,
    )
    assert exit_price == 80.0
    assert reason == "STOP_LOSS"


def test_opening_gap_through_stop_uses_observed_open():
    exit_price, reason = _simulate_exit(
        entry=100.0,
        close=90.0,
        open_price=75.0,
        high=105.0,
        low=70.0,
        stop_pct=0.20,
        target_pct=None,
    )
    assert exit_price == 75.0
    assert reason == "STOP_GAP"


def test_risk_managed_research_builds_baseline_and_exit_variants():
    dates = pd.date_range("2024-01-01", periods=14, freq="B")
    expiry = dates[-1] + pd.Timedelta(days=5)
    base_rows = []
    option_rows = []
    for i in range(12):
        entry_date = dates[i]
        exit_date = dates[i + 1]
        base_rows.append({
            "strategy": "DAILY_ATM_CALL",
            "entry_date": entry_date,
            "exit_date": exit_date,
            "expiry_date": expiry,
            "strike": 100.0,
            "symbol": "SENSEX",
            "option_type": "CE",
            "entry_premium": 100.0,
            "exit_premium": 110.0,
            "quantity": 10,
            "lot_size": 10,
            "lots": 1,
            "net_pnl": 90.0,
        })
        option_rows.append({
            "trade_date": exit_date,
            "expiry_date": expiry,
            "strike": 100.0,
            "symbol": "SENSEX",
            "option_type": "CE",
            "open": 100.0,
            "high": 130.0,
            "low": 70.0,
            "close": 110.0,
            "volume": 100, "open_interest": 100,
        })

    monthly, board, report = run_risk_managed_research(
        pd.DataFrame(base_rows),
        pd.DataFrame(option_rows),
        stop_loss_levels=(0.20,),
        take_profit_levels=(0.20,),
    )

    assert report["status"] == "no_candidate_met_monthly_target"
    assert report["base_trade_count"] == 12
    assert report["option_ohlc_coverage_pct"] == 100.0
    assert {"DAILY_ATM_CALL", "DAILY_ATM_CALL__SL20",
            "DAILY_ATM_CALL__TP20", "DAILY_ATM_CALL__SL20_TP20"}.issubset(
                set(board["strategy"])
            )

    stopped = monthly[monthly["strategy"] == "DAILY_ATM_CALL__SL20"]
    combined = monthly[monthly["strategy"] == "DAILY_ATM_CALL__SL20_TP20"]
    assert int(stopped.iloc[0]["trades"]) == 12
    assert stopped.iloc[0]["net_pnl"] < monthly.loc[
        monthly["strategy"] == "DAILY_ATM_CALL", "net_pnl"
    ].iloc[0]
    # Both stop and target were touched, so the conservative stop is used.
    assert stopped.iloc[0]["net_pnl"] == combined.iloc[0]["net_pnl"]


def test_risk_managed_research_refuses_missing_option_ohlc():
    monthly, board, report = run_risk_managed_research(
        pd.DataFrame({"strategy": ["X"]}),
        pd.DataFrame({"trade_date": []}),
    )
    assert monthly.empty
    assert board.empty
    assert report["status"] == "invalid_ledger"
