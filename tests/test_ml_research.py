import numpy as np
import pandas as pd

from sensex_options.ml_research import (
    build_ml_option_dataset, _selected_trade_metrics, run_ml_option_research, _black_scholes_price,
    _implied_volatility, _black_scholes_greeks, RISK_FREE_RATE,
)


def ml_fixture(periods=100):
    dates = pd.bdate_range("2024-01-01", periods=periods)
    close = []
    value = 72000.0
    for i in range(periods):
        value *= 1.0004 if (i // 8) % 2 == 0 else 0.9996
        value += np.sin(i / 3.0) * 18
        close.append(value)
    index = pd.DataFrame({
        "date": dates,
        "open": [v - 12 for v in close],
        "high": [v + 55 for v in close],
        "low": [v - 55 for v in close],
        "close": close,
    })
    expiry = dates[-1] + pd.Timedelta(days=14)
    rows = []
    strikes = [71800.0, 72000.0, 72200.0]
    for i, day in enumerate(dates):
        spot = close[i]
        for strike in strikes:
            intrinsic_call = max(spot - strike, 0.0)
            intrinsic_put = max(strike - spot, 0.0)
            call_premium = 80.0 + intrinsic_call * 0.08 + (i % 5) * 2.0
            put_premium = 80.0 + intrinsic_put * 0.08 + ((i + 2) % 5) * 2.0
            rows.extend([
                {
                    "trade_date": day, "expiry_date": expiry, "strike": strike,
                    "symbol": "SENSEX", "option_type": "CE", "close": call_premium,
                    "lot_size": 20, "volume": 500, "open_interest": 2000,
                },
                {
                    "trade_date": day, "expiry_date": expiry, "strike": strike,
                    "symbol": "SENSEX", "option_type": "PE", "close": put_premium,
                    "lot_size": 20, "volume": 500, "open_interest": 2000,
                },
            ])
    return index, pd.DataFrame(rows)


def test_ml_dataset_has_paired_observed_sides_and_lagged_features():
    index, options = ml_fixture()
    dataset, features, report = build_ml_option_dataset(index, options)
    assert report["status"] == "ok"
    assert len(dataset) > 100
    assert set(dataset["option_side"]) == {"CE", "PE"}
    assert (pd.to_datetime(dataset["signal_date"]) < pd.to_datetime(dataset["entry_date"])).all()
    assert {"target_win", "target_return_pct", "target_net_pnl"}.issubset(dataset.columns)
    assert len(features) >= 3
    assert not {"target_win", "target_return_pct", "target_net_pnl", "signal_spot", "close", "ema_5", "ema_20", "atr_14", "near_atm_call_oi"}.intersection(features)
    assert not any(name.startswith("donchian_high_") or name.startswith("donchian_low_") for name in features)


def test_ml_strategy_selection_uses_at_most_one_candidate_per_entry_session():
    frame = pd.DataFrame({
        "entry_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        "option_side": ["CE", "PE", "CE"],
        "moneyness": ["ATM", "OTM1", "ATM"],
        "target_net_pnl": [100.0, -50.0, 80.0],
        "target_return_pct": [20.0, -10.0, 15.0],
    })
    result = _selected_trade_metrics(frame, np.array([0.80, 0.90, 0.75]), 0.70)
    selected = result.pop("selected_predictions")
    assert result["trades"] == 2
    assert result["sessions"] == 2
    assert selected["option_side"].tolist() == ["PE", "CE"]


def test_lagged_black_scholes_greeks_recover_known_volatility():
    spot, strike, years, volatility = 72000.0, 72000.0, 30 / 365.25, 0.25
    premium = _black_scholes_price(spot, strike, years, RISK_FREE_RATE, volatility, "CE")
    implied = _implied_volatility(
        premium, spot, strike, years, RISK_FREE_RATE, "CE"
    )
    assert implied is not None
    assert abs(implied - volatility) < 1e-5
    greeks = _black_scholes_greeks(
        spot, strike, years, RISK_FREE_RATE, implied, "CE"
    )
    assert 0 < greeks["lagged_delta"] < 1
    assert greeks["lagged_gamma"] > 0
    assert greeks["lagged_vega_per_1pct"] > 0
    assert "lagged_theta_per_day" in greeks


def test_ml_research_fits_models_and_evaluates_separate_holdout(tmp_path):
    index, options = ml_fixture(periods=100)
    comparison, importance, report = run_ml_option_research(
        index, options, output_dir=tmp_path,
        minimum_validation_trades=2, thresholds=(0.0,),
    )
    assert report["status"] == "holdout_evaluated"
    assert report["split_dates"]["train_end"] < report["split_dates"]["validation_start"]
    assert report["split_dates"]["validation_end"] < report["split_dates"]["holdout_start"]
    assert report["holdout_strategy"]["trades"] > 0
    assert len(comparison) == 3
    assert not importance.empty
    assert (tmp_path / "ml_option_classifier.joblib").exists()
    assert (tmp_path / "ml_option_holdout_predictions.csv.gz").exists()


def test_equal_model_probabilities_use_deterministic_not_realized_return_tiebreak():
    frame = pd.DataFrame({
        "entry_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "option_side": ["PE", "CE"],
        "moneyness": ["ATM", "ATM"],
        "target_net_pnl": [500.0, -100.0],
        "target_return_pct": [50.0, -10.0],
    })
    result = _selected_trade_metrics(frame, np.array([0.8, 0.8]), 0.7)
    selected = result["selected_predictions"]
    assert selected.iloc[0]["option_side"] == "CE"
    assert selected.iloc[0]["target_net_pnl"] == -100.0
