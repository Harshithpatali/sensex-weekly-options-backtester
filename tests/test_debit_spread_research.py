import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.entry_signal_research import _option_debit_vertical_outcomes


def test_debit_vertical_spreads_use_observed_same_expiry_legs():
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = [100.0] * len(dates)
    index = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": [101.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": close,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    quotes = []
    for i, day in enumerate(dates):
        quotes.extend([
            {"trade_date": day, "expiry_date": expiry, "strike": 100.0,
             "symbol": "SENSEX", "option_type": "CE", "close": 20.0 + i, "lot_size": 10, "volume": 100, "open_interest": 100},
            {"trade_date": day, "expiry_date": expiry, "strike": 110.0,
             "symbol": "SENSEX", "option_type": "CE", "close": 12.0 + i * 0.5, "lot_size": 10, "volume": 100, "open_interest": 100},
            {"trade_date": day, "expiry_date": expiry, "strike": 100.0,
             "symbol": "SENSEX", "option_type": "PE", "close": 20.0 + i, "lot_size": 10, "volume": 100, "open_interest": 100},
            {"trade_date": day, "expiry_date": expiry, "strike": 90.0,
             "symbol": "SENSEX", "option_type": "PE", "close": 12.0 + i * 0.5, "lot_size": 10, "volume": 100, "open_interest": 100},
        ])
    result = _option_debit_vertical_outcomes(
        index, pd.DataFrame(quotes), holding_periods=(1,), cost=CostConfig()
    )
    assert not result.empty
    first = result[result["entry_date"] == dates[1]].iloc[0]
    assert first["option_family"] == "DEBIT_SPREADS_W1"
    assert first["call_pnl"] > 0
    assert first["put_pnl"] > 0


def test_debit_vertical_spreads_skip_missing_exit_leg():
    dates = pd.bdate_range("2024-01-01", periods=4)
    index = pd.DataFrame({
        "date": dates, "open": [100.0] * 4, "high": [101.0] * 4,
        "low": [99.0] * 4, "close": [100.0] * 4,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    quotes = []
    for day in dates:
        quotes.extend([
            {"trade_date": day, "expiry_date": expiry, "strike": 100.0,
             "symbol": "SENSEX", "option_type": "CE", "close": 20.0, "lot_size": 10, "volume": 100, "open_interest": 100},
            {"trade_date": day, "expiry_date": expiry, "strike": 110.0,
             "symbol": "SENSEX", "option_type": "CE", "close": 12.0, "lot_size": 10, "volume": 100, "open_interest": 100},
            {"trade_date": day, "expiry_date": expiry, "strike": 100.0,
             "symbol": "SENSEX", "option_type": "PE", "close": 20.0, "lot_size": 10, "volume": 100, "open_interest": 100},
        ])
        if day != dates[2]:
            quotes.append({
                "trade_date": day, "expiry_date": expiry, "strike": 90.0,
                "symbol": "SENSEX", "option_type": "PE", "close": 10.0, "lot_size": 10, "volume": 100, "open_interest": 100,
            })
    result = _option_debit_vertical_outcomes(
        index, pd.DataFrame(quotes), holding_periods=(1,), cost=CostConfig()
    )
    first = result[result["entry_date"] == dates[1]].iloc[0]
    assert pd.notna(first["call_pnl"])
    assert pd.isna(first["put_pnl"])


def test_debit_vertical_spread_grid_evaluates_multiple_available_widths():
    dates = pd.bdate_range("2024-02-01", periods=5)
    close = [100.0] * len(dates)
    index = pd.DataFrame({
        "date": dates, "open": close, "high": [101.0] * 5,
        "low": [99.0] * 5, "close": close,
    })
    expiry = dates[-1] + pd.Timedelta(days=7)
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    quotes = []
    for i, day in enumerate(dates):
        premiums = {
            ("CE", 100.0): 30.0 + i,
            ("CE", 110.0): 22.0 + i * 0.5,
            ("CE", 120.0): 12.0 + i * 0.3,
            ("CE", 130.0): 6.0 + i * 0.2,
            ("PE", 100.0): 30.0 + i,
            ("PE", 90.0): 22.0 + i * 0.5,
            ("PE", 80.0): 12.0 + i * 0.3,
            ("PE", 70.0): 6.0 + i * 0.2,
        }
        for (option_type, strike), premium in premiums.items():
            quotes.append({
                "trade_date": day, "expiry_date": expiry, "strike": strike,
                "symbol": "SENSEX", "option_type": option_type,
                "close": premium, "lot_size": 10, "volume": 100, "open_interest": 100,
            })
    result = _option_debit_vertical_outcomes(
        index, pd.DataFrame(quotes), holding_periods=(1,),
        spread_width_steps=(1, 2, 3), cost=CostConfig(),
    )
    first_entry = result[result["entry_date"] == dates[1]]
    assert set(first_entry["spread_width_steps"].astype(int)) == {1, 2, 3}
    assert set(first_entry["option_family"]) == {
        "DEBIT_SPREADS_W1", "DEBIT_SPREADS_W2", "DEBIT_SPREADS_W3"
    }
    assert first_entry["call_pnl"].notna().all()
    assert first_entry["put_pnl"].notna().all()
