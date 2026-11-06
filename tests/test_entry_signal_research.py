import pandas as pd

from sensex_options.entry_signal_research import (
    _build_monthly_selector,
    _paired_option_outcomes,
    build_lagged_features,
    run_entry_signal_research,
)


def test_features_are_lagged_completed_session_values():
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    close = [100 + i * 0.2 + (i % 7) for i in range(len(dates))]
    frame = pd.DataFrame({
        "date": dates,
        "open": [v - 0.3 for v in close],
        "high": [v + 1 for v in close],
        "low": [v - 1 for v in close],
        "close": close,
    })
    first = build_lagged_features(frame)
    changed = frame.copy()
    changed.loc[changed.index >= 70, ["open", "high", "low", "close"]] *= 3
    second = build_lagged_features(changed)
    # Changing future observations cannot alter already-completed earlier signals.
    pd.testing.assert_series_equal(
        first.loc[:69, "ema_20"],
        second.loc[:69, "ema_20"],
        check_names=True,
    )
    assert first.loc[69, "date"] == dates[69]


def test_paired_outcomes_require_same_date_expiry_and_strike():
    date = pd.Timestamp("2024-01-03")
    exit_date = pd.Timestamp("2024-01-04")
    expiry = pd.Timestamp("2024-01-12")
    ledger = pd.DataFrame([
        {"strategy": "DAILY_ATM_CALL", "entry_date": date, "exit_date": exit_date,
         "expiry_date": expiry, "strike": 72000, "net_pnl": 100},
        {"strategy": "DAILY_ATM_PUT", "entry_date": date, "exit_date": exit_date,
         "expiry_date": expiry, "strike": 72000, "net_pnl": -50},
        {"strategy": "DAILY_ATM_PUT", "entry_date": date, "exit_date": exit_date,
         "expiry_date": expiry, "strike": 72100, "net_pnl": 90},
    ])
    paired = _paired_option_outcomes(ledger)
    assert len(paired) == 1
    assert paired.iloc[0]["call_pnl"] == 100
    assert paired.iloc[0]["put_pnl"] == -50


def test_rolling_selector_uses_only_prior_complete_months_and_can_stay_cash():
    rows = []
    for month, strategy, pnl in [
        ("2024-01", "A", 100),
        ("2024-01", "B", -100),
        ("2024-02", "A", 100),
        ("2024-02", "B", -100),
        ("2024-03", "A", -100),
        ("2024-03", "B", 100),
        ("2024-04", "A", -100),
        ("2024-04", "B", 100),
    ]:
        rows.append({"month": month, "strategy": strategy, "net_pnl": pnl})
    selector = _build_monthly_selector(pd.DataFrame(rows), ["A", "B"], 2, minimum_prior_trades=2)
    april = selector[selector["month"] == "2024-04"].iloc[0]
    assert april["selected_rule"] == "CASH"


def test_entry_signal_research_returns_out_of_sample_and_monthly_outputs():
    dates = pd.bdate_range("2024-01-01", periods=260)
    close = []
    value = 72000.0
    for i in range(len(dates)):
        value *= 1.001 if (i // 20) % 2 == 0 else 0.999
        close.append(value + (i % 5) * 8)
    index = pd.DataFrame({
        "date": dates,
        "open": [v * 0.999 for v in close],
        "high": [v * 1.002 for v in close],
        "low": [v * 0.998 for v in close],
        "close": close,
        "volume": 0,
    })
    ledger_rows = []
    for i in range(1, len(dates) - 1):
        entry, exit_ = dates[i], dates[i + 1]
        expiry = dates[min(i + 10, len(dates) - 1)]
        common = {
            "entry_date": entry, "exit_date": exit_, "expiry_date": expiry,
            "strike": round(close[i - 1] / 100) * 100, "quantity": 10,
            "lot_size": 10, "lots": 1,
        }
        ledger_rows.append({
            **common, "strategy": "DAILY_ATM_CALL", "net_pnl": 300 if i % 4 < 2 else -100,
        })
        ledger_rows.append({
            **common, "strategy": "DAILY_ATM_PUT", "net_pnl": -100 if i % 4 < 2 else 300,
        })
    trades, board, selector, report = run_entry_signal_research(
        pd.DataFrame(ledger_rows), index
    )
    assert report["status"] in {"candidates_found", "no_candidate_met_research_screen"}
    assert report["candidate_rules_tested"] >= 20
    assert report["training_end"] < report["holdout_start"]
    assert not trades.empty
    assert not board.empty
    assert {"ROLLING_3M_SELECTOR", "ROLLING_6M_SELECTOR"}.issubset(set(selector["strategy"]))
    assert {"train_aggregate_pf", "holdout_aggregate_pf", "candidate_pass"}.issubset(board.columns)
