import pandas as pd

from sensex_options.analytics import strategy_performance_metrics, walk_forward_evaluation


def test_strategy_performance_metrics_calculate_expectancy_profit_factor_and_drawdown():
    trades = pd.DataFrame({
        "strategy": ["ATM_CALL"] * 3,
        "status": ["completed"] * 3,
        "exit_date": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
        "starting_capital": [10000.0, 11000.0, 10500.0],
        "net_pnl": [1000.0, -500.0, 700.0],
    })
    metrics = strategy_performance_metrics(trades, initial_capital=10000.0).iloc[0]
    assert metrics["trades"] == 3
    assert metrics["expectancy"] == 400.0
    assert metrics["profit_factor"] == 3.4
    assert metrics["max_drawdown"] == -500.0
    assert metrics["sample_warning"]


def test_walk_forward_uses_chronological_expiry_windows():
    expiries = pd.date_range("2024-01-04", periods=12, freq="7D")
    trades = pd.DataFrame({
        "strategy": ["ATM_CALL"] * 12,
        "status": ["completed"] * 12,
        "expiry_date": expiries,
        "exit_date": expiries,
        "starting_capital": [10000.0] * 12,
        "net_pnl": [100.0, -50.0] * 6,
    })
    result = walk_forward_evaluation(trades, initial_capital=10000.0, train_expiries=4, test_expiries=2)
    assert result["status"] == "ok"
    assert result["train_expiries"] == 8
    assert result["holdout_expiries"] == 4
    assert len(result["walk_forward_folds"]) == 4
