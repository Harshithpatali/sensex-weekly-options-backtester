import pandas as pd

from sensex_options.capital_controls import apply_capital_controls
from sensex_options.config import CostConfig


def test_rejects_one_lot_when_premium_exceeds_two_percent_risk_budget():
    ledger = pd.DataFrame([{
        "strategy": "CALL", "entry_date": "2024-01-02", "exit_date": "2024-01-03",
        "entry_premium": 100.0, "exit_premium": 150.0, "lot_size": 25,
        "lots": 1, "quantity": 25, "net_pnl": 1200.0,
    }])
    trades, board, report = apply_capital_controls(
        ledger, initial_capital=10_000, risk_per_trade_pct=0.02,
        daily_loss_limit_pct=0.02, cost=CostConfig(),
    )
    assert trades.empty
    assert board.iloc[0]["ending_capital"] == 10_000
    assert board.iloc[0]["trades_rejected_risk_budget"] == 1
    assert report["feasible_trades"] == 0


def test_accepts_trade_within_budget_and_compounds_realized_equity():
    ledger = pd.DataFrame([
        {"strategy": "CALL", "entry_date": "2024-01-02", "exit_date": "2024-01-03",
         "entry_premium": 5.0, "exit_premium": 7.0, "lot_size": 10,
         "lots": 1, "quantity": 10, "net_pnl": 20.0},
        {"strategy": "CALL", "entry_date": "2024-01-04", "exit_date": "2024-01-05",
         "entry_premium": 5.0, "exit_premium": 4.0, "lot_size": 10,
         "lots": 1, "quantity": 10, "net_pnl": -10.0},
    ])
    trades, board, report = apply_capital_controls(
        ledger, initial_capital=10_000, risk_per_trade_pct=0.02,
        daily_loss_limit_pct=0.02, cost=CostConfig(),
    )
    assert len(trades) == 2
    assert trades.iloc[0]["capital_start"] == 10_000
    assert trades.iloc[1]["capital_start"] == trades.iloc[0]["capital_end"]
    assert board.iloc[0]["trades_taken"] == 2
    assert report["candidate_trades"] == 2


def test_rejects_overlapping_positions_in_same_strategy():
    ledger = pd.DataFrame([
        {"strategy": "CALL", "entry_date": "2024-01-02", "exit_date": "2024-01-04",
         "entry_premium": 5.0, "exit_premium": 7.0, "lot_size": 10,
         "lots": 1, "quantity": 10, "net_pnl": 20.0},
        {"strategy": "CALL", "entry_date": "2024-01-03", "exit_date": "2024-01-05",
         "entry_premium": 5.0, "exit_premium": 7.0, "lot_size": 10,
         "lots": 1, "quantity": 10, "net_pnl": 20.0},
    ])
    trades, board, _ = apply_capital_controls(
        ledger, initial_capital=10_000, risk_per_trade_pct=0.02,
        daily_loss_limit_pct=0.02, cost=CostConfig(),
    )
    assert len(trades) == 1
    assert board.iloc[0]["trades_rejected_overlap"] == 1


def test_capital_holdout_is_chronological_and_resets_each_period():
    from sensex_options.capital_controls import evaluate_capital_control_holdout

    dates = pd.bdate_range("2024-01-01", periods=20)
    ledger = pd.DataFrame([{
        "strategy": "CALL", "entry_date": day, "exit_date": day + pd.offsets.BDay(1),
        "entry_premium": 5.0, "exit_premium": 7.0, "lot_size": 10,
        "lots": 1, "quantity": 10, "net_pnl": 20.0,
    } for day in dates])
    board, report = evaluate_capital_control_holdout(
        ledger, initial_capital=10_000, risk_per_trade_pct=0.02,
        daily_loss_limit_pct=0.02, cost=CostConfig(),
    )
    assert report["status"] == "complete"
    assert report["train_end"] < report["validation_start"]
    assert board.iloc[0]["train_ending_capital"] >= 10_000
    assert board.iloc[0]["validation_ending_capital"] >= 10_000
