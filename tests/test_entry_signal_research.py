import numpy as np
import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.entry_signal_research import (
    _build_monthly_selector,
    _paired_option_outcomes,
    _option_horizon_outcomes,
    _option_debit_vertical_outcomes,
    _option_credit_vertical_outcomes,
    _option_chain_market_features,
    _one_position_at_a_time,
    _candidate_signals,
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
    assert pd.notna(first.loc[69, "realized_vol_20"])
    assert pd.notna(first.loc[69, "ret_z_5"])
    assert pd.notna(first.loc[69, "ema_5_20_z"])
    assert pd.notna(first.loc[69, "plus_di_14"])
    signals = _candidate_signals(first)
    assert signals
    signatures = [
        tuple(series.fillna("").astype(str).tolist())
        for series in signals.values()
    ]
    assert len(signatures) == len(set(signatures))


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
            "lot_size": 10, "volume": 100, "open_interest": 100, "lots": 1,
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


def test_option_horizon_outcomes_use_same_contract_for_multiple_holds():
    dates = pd.bdate_range("2024-01-01", periods=18)
    close = [100.0 + i * 0.1 for i in range(len(dates))]
    index = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": [v + 1 for v in close],
        "low": [v - 1 for v in close],
        "close": close,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    option_rows = []
    for i, day in enumerate(dates):
        for option_type in ("CE", "PE"):
            for strike in (99.0, 100.0, 101.0):
                base = 15.0 + (strike - 100.0) * (-2 if option_type == "CE" else 2)
                premium = base + i * (1.0 if option_type == "CE" else 0.7)
                option_rows.append({
                    "trade_date": day,
                    "expiry_date": expiry,
                    "strike": strike,
                    "symbol": "SENSEX",
                    "option_type": option_type,
                    "close": premium,
                    "lot_size": 10, "volume": 100, "open_interest": 100,
                })
    outcomes = _option_horizon_outcomes(
        index, pd.DataFrame(option_rows), holding_periods=(1, 2), risk_pct=0.03
    )
    assert set(outcomes["holding_sessions"].unique()) == {1, 2}
    assert set(outcomes["moneyness"].unique()) == {"ATM", "ITM1", "OTM1", "OTM2"}
    assert set(outcomes["option_family"].unique()) == {
        "LONG_OPTIONS_ATM", "LONG_OPTIONS_ITM1", "LONG_OPTIONS_OTM1", "LONG_OPTIONS_OTM2"
    }

    h1 = outcomes[
        (outcomes["holding_sessions"] == 1)
        & (outcomes["moneyness"] == "ATM")
    ].sort_values("entry_date").iloc[0]
    h2 = outcomes[
        (outcomes["holding_sessions"] == 2)
        & (outcomes["moneyness"] == "ATM")
    ].sort_values("entry_date").iloc[0]
    assert h1["entry_date"] == dates[1]
    assert h1["signal_date"] == dates[0]
    assert h1["exit_date"] == dates[2]
    assert h2["exit_date"] == dates[3]
    assert h1["call_strike"] == 100.0
    assert h1["put_strike"] == 100.0
    assert pd.notna(h1["call_pnl"])
    assert pd.notna(h1["put_pnl"])

    itm = outcomes[
        (outcomes["entry_date"] == dates[1])
        & (outcomes["holding_sessions"] == 1)
        & (outcomes["moneyness"] == "ITM1")
    ].iloc[0]
    otm = outcomes[
        (outcomes["entry_date"] == dates[1])
        & (outcomes["holding_sessions"] == 1)
        & (outcomes["moneyness"] == "OTM1")
    ].iloc[0]
    assert itm["call_strike"] == 99.0 and itm["put_strike"] == 101.0
    assert otm["call_strike"] == 101.0 and otm["put_strike"] == 99.0

def test_option_horizon_outcomes_do_not_invent_missing_option_side():
    dates = pd.bdate_range("2024-01-01", periods=8)
    index = pd.DataFrame({
        "date": dates,
        "open": [100.0] * len(dates),
        "high": [101.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [100.0] * len(dates),
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    option_rows = []
    for day in dates:
        option_rows.append({
            "trade_date": day, "expiry_date": expiry, "strike": 100.0,
            "symbol": "SENSEX", "option_type": "CE", "close": 10.0, "lot_size": 10, "volume": 100, "open_interest": 100,
        })
        if day != dates[3]:
            option_rows.append({
                "trade_date": day, "expiry_date": expiry, "strike": 100.0,
                "symbol": "SENSEX", "option_type": "PE", "close": 10.0, "lot_size": 10, "volume": 100, "open_interest": 100,
            })
    outcomes = _option_horizon_outcomes(
        index, pd.DataFrame(option_rows), holding_periods=(1,)
    )
    row = outcomes[
        (outcomes["entry_date"] == dates[2]) & (outcomes["holding_sessions"] == 1)
    ].iloc[0]
    assert pd.notna(row["call_pnl"])
    assert pd.isna(row["put_pnl"])


def test_one_position_book_skips_overlapping_trades_but_allows_exit_day_reentry():
    dates = pd.bdate_range("2024-01-01", periods=6)
    trades = pd.DataFrame([
        {"entry_date": dates[0], "exit_date": dates[2], "trade": "first"},
        {"entry_date": dates[1], "exit_date": dates[3], "trade": "overlaps_first"},
        {"entry_date": dates[2], "exit_date": dates[4], "trade": "exit_day_reentry"},
        {"entry_date": dates[3], "exit_date": dates[5], "trade": "overlaps_third"},
    ])
    selected, skipped = _one_position_at_a_time(trades)
    assert selected["trade"].tolist() == ["first", "exit_day_reentry"]
    assert skipped == 2


def test_option_chain_features_calculate_pcr_and_option_positioning():
    dates = pd.bdate_range("2024-01-01", periods=35)
    close = [72000 + i * 10 for i in range(len(dates))]
    index = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": [v + 100 for v in close],
        "low": [v - 100 for v in close],
        "close": close,
    })
    expiry = dates[-1] + pd.Timedelta(days=14)
    rows = []
    for i, day in enumerate(dates):
        for strike in (71900, 72000, 72100):
            for option_type in ("CE", "PE"):
                is_put = option_type == "PE"
                rows.append({
                    "trade_date": day,
                    "expiry_date": expiry,
                    "strike": strike,
                    "symbol": "SENSEX",
                    "option_type": option_type,
                    "close": (100 + i if not is_put else 105 - i * 0.5) + abs(strike - 72000) / 100,
                    "volume": 1000 if is_put else 500,
                    "open_interest": (2000 if is_put else 1000) + i * (20 if is_put else 10),
                })
    features = _option_chain_market_features(index, pd.DataFrame(rows))
    latest = features[features["signal_date"] == dates[-1]].iloc[0]
    assert latest["pcr_volume_near_atm"] > 1.0
    assert latest["pcr_oi_near_atm"] > 1.0
    assert pd.notna(latest["call_oi_change_pct"])
    assert pd.notna(latest["put_oi_change_pct"])
    assert "atm_straddle_pct" in features.columns

    # Force the straddle/realised-volatility gate to be selective so the
    # gated rule is behaviourally distinct rather than a duplicate alias.
    features["implied_to_realized_vol_ratio"] = np.where(
        np.arange(len(features)) % 2 == 0, 0.50, 2.00
    )
    index_features = build_lagged_features(index).rename(columns={"date": "signal_date"})
    combined_features = features.merge(
        index_features,
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    signals = _candidate_signals(combined_features)
    assert any(name.startswith("PCR_OI_") for name in signals)
    assert any("CHEAP_STRADDLE" in name for name in signals)
    assert signals
    signatures = [
        tuple(series.fillna("").astype(str).tolist())
        for series in signals.values()
    ]
    assert len(signatures) == len(set(signatures))


def test_call_oi_buildup_rule_survives_when_it_is_behaviorally_distinct():
    dates = pd.bdate_range("2024-01-01", periods=80)
    index = pd.DataFrame({
        "date": dates,
        "open": [100.0] * len(dates),
        "high": [101.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [100.0] * len(dates),
    })
    features = build_lagged_features(index).iloc[[-1]].copy()
    features["call_oi_change_pct"] = 0.15
    features["call_premium_change_pct"] = 0.03
    features["put_oi_change_pct"] = np.nan
    features["put_premium_change_pct"] = np.nan
    features["positioning_score"] = np.nan
    features["pcr_oi_near_atm"] = np.nan
    features["pcr_volume_near_atm"] = np.nan
    features["implied_to_realized_vol_ratio"] = np.nan

    signals = _candidate_signals(features)
    assert any(name.startswith("CALL_OI_BUILDUP") for name in signals)


def test_defined_risk_credit_spreads_use_observed_hedge_and_fit_risk_budget():
    dates = pd.bdate_range("2024-01-01", periods=12)
    close = [72000.0] * len(dates)
    index = pd.DataFrame({
        "date": dates, "open": close, "high": [v + 100 for v in close],
        "low": [v - 100 for v in close], "close": close,
    })
    expiry = dates[-1] + pd.Timedelta(days=10)
    option_rows = []
    for i, day in enumerate(dates):
        for strike in (71900.0, 72000.0, 72100.0):
            for option_type in ("CE", "PE"):
                # A rising index regime: puts cheapen (bull-put credit wins),
                # while calls reprice higher (bear-call credit loses).
                if option_type == "PE":
                    premium = 15.0 - 1.0 * i if strike == 72000 else 5.0 - 0.2 * i
                else:
                    premium = 15.0 + 1.0 * i if strike == 72000 else 5.0 + 0.2 * i
                option_rows.append({
                    "trade_date": day, "expiry_date": expiry, "strike": strike,
                    "symbol": "SENSEX", "option_type": option_type,
                    "close": premium, "lot_size": 10, "volume": 100, "open_interest": 100,
                })
    outcomes = _option_credit_vertical_outcomes(
        index, pd.DataFrame(option_rows), holding_periods=(1,), spread_width_steps=(1,),
        capital=10000.0, risk_pct=1.0,
        cost=CostConfig(),
    )
    assert not outcomes.empty
    strict_risk_outcomes = _option_credit_vertical_outcomes(
        index, pd.DataFrame(option_rows), holding_periods=(1,), spread_width_steps=(1,),
        capital=10000.0, risk_pct=0.02, cost=CostConfig(),
    )
    assert strict_risk_outcomes.empty

    assert set(outcomes["option_family"]) == {"CREDIT_SPREADS_W1"}
    assert outcomes["call_pnl"].notna().any()
    assert outcomes["put_pnl"].notna().any()
    # Bullish CE signals route to bull-put credit spreads; bearish PE signals
    # route to bear-call credit spreads in the shared candidate engine.
    assert float(outcomes.iloc[0]["call_pnl"]) > 0
    assert float(outcomes.iloc[0]["put_pnl"]) < 0


def test_vertical_spreads_reject_quotes_outside_strike_width():
    dates = pd.bdate_range("2024-01-01", periods=5)
    index = pd.DataFrame({
        "date": dates,
        "open": [100.0] * len(dates),
        "high": [101.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [100.0] * len(dates),
    })
    expiry = dates[-1] + pd.Timedelta(days=10)
    option_rows = []
    for day in dates:
        for strike in (99.0, 100.0, 101.0):
            for option_type in ("CE", "PE"):
                # Premium differences are 145 points against only 1 point
                # strike width: these cannot be valid vertical-spread marks.
                premium = 150.0 if strike == 100.0 else 5.0
                option_rows.append({
                    "trade_date": day, "expiry_date": expiry, "strike": strike,
                    "symbol": "SENSEX", "option_type": option_type,
                    "close": premium, "lot_size": 1,
                })
    options = pd.DataFrame(option_rows)
    debit = _option_debit_vertical_outcomes(
        index, options, holding_periods=(1,), spread_width_steps=(1,), cost=CostConfig()
    )
    credit = _option_credit_vertical_outcomes(
        index, options, holding_periods=(1,), spread_width_steps=(1,), cost=CostConfig()
    )
    assert debit.empty
    assert credit.empty


def test_option_horizon_outcomes_reject_illiquid_entry_and_exit_marks():
    dates = pd.bdate_range("2024-02-01", periods=6)
    index = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    rows = []
    for day in dates:
        for side in ("CE", "PE"):
            rows.append({
                "trade_date": day, "expiry_date": expiry, "strike": 100.0,
                "symbol": "SENSEX", "option_type": side, "close": 10.0,
                "lot_size": 10, "volume": 100, "open_interest": 100,
            })
    for row in rows:
        if row["trade_date"] == dates[1] and row["option_type"] == "CE":
            row["volume"] = 1
        if row["trade_date"] == dates[3] and row["option_type"] == "PE":
            row["open_interest"] = 1
    outcomes = _option_horizon_outcomes(index, pd.DataFrame(rows), holding_periods=(1,))
    first = outcomes[outcomes["entry_date"] == dates[1]].iloc[0]
    assert pd.isna(first["call_pnl"])
    assert pd.notna(first["put_pnl"])
    second = outcomes[outcomes["entry_date"] == dates[2]].iloc[0]
    assert pd.notna(second["call_pnl"])
    assert pd.isna(second["put_pnl"])


def test_long_option_outcomes_enforce_two_percent_max_loss_budget():
    dates = pd.bdate_range("2024-03-01", periods=6)
    index = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    rows = []
    for day in dates:
        for side in ("CE", "PE"):
            rows.append({
                "trade_date": day, "expiry_date": expiry, "strike": 100.0,
                "symbol": "SENSEX", "option_type": side, "close": 25.0,
                "lot_size": 10, "volume": 100, "open_interest": 100,
            })
    outcomes = _option_horizon_outcomes(
        index, pd.DataFrame(rows), holding_periods=(1,), moneyness_variants=("ATM",),
        capital=10000.0, risk_pct=0.02,
    )
    assert outcomes.empty


def test_far_otm_contract_variants_select_correct_strikes():
    dates = pd.bdate_range("2024-04-01", periods=6)
    index = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    rows = []
    for day in dates:
        for strike in range(95, 106):
            for side in ("CE", "PE"):
                rows.append({
                    "trade_date": day, "expiry_date": expiry, "strike": float(strike),
                    "symbol": "SENSEX", "option_type": side, "close": 5.0,
                    "lot_size": 10, "volume": 100, "open_interest": 100,
                })
    outcomes = _option_horizon_outcomes(
        index, pd.DataFrame(rows), holding_periods=(1,), moneyness_variants=("OTM5",),
    )
    first = outcomes[outcomes["entry_date"] == dates[1]].iloc[0]
    assert first["call_strike"] == 105.0
    assert first["put_strike"] == 95.0


def test_default_option_outcomes_ignore_capital_risk_budget_but_explicit_budget_applies():
    dates = pd.bdate_range("2024-01-01", periods=8)
    index = pd.DataFrame({"date": dates, "close": [72000.0] * len(dates)})
    expiry = dates[-1] + pd.Timedelta(days=7)
    rows = []
    for day in dates:
        for option_type in ("CE", "PE"):
            rows.append({
                "trade_date": day, "expiry_date": expiry, "strike": 72000.0,
                "symbol": "SENSEX", "option_type": option_type, "close": 100.0,
                "lot_size": 20, "volume": 100, "open_interest": 500,
            })
    options = pd.DataFrame(rows)
    unit_lot = _option_horizon_outcomes(index, options, holding_periods=(1,))
    budgeted = _option_horizon_outcomes(
        index, options, holding_periods=(1,), capital=10000.0, risk_pct=0.02
    )
    assert not unit_lot.empty
    assert budgeted.empty


def test_zero_premium_on_expiry_is_a_valid_long_option_loss():
    dates = pd.bdate_range("2024-01-01", periods=3)
    index = pd.DataFrame({"date": dates, "close": [72000.0, 72000.0, 72000.0]})
    options = pd.DataFrame([
        {
            "trade_date": dates[1], "expiry_date": dates[2], "strike": 72000.0,
            "symbol": "SENSEX", "option_type": side, "close": 10.0,
            "lot_size": 20, "volume": 100, "open_interest": 500,
        }
        for side in ("CE", "PE")
    ] + [
        {
            "trade_date": dates[2], "expiry_date": dates[2], "strike": 72000.0,
            "symbol": "SENSEX", "option_type": side, "close": 0.0,
            "lot_size": 20, "volume": 0, "open_interest": 0,
        }
        for side in ("CE", "PE")
    ])
    outcomes = _option_horizon_outcomes(index, options, holding_periods=(1,), moneyness_variants=("ATM",))
    assert not outcomes.empty
    assert (outcomes["call_pnl"] < 0).all()
    assert (outcomes["put_pnl"] < 0).all()
