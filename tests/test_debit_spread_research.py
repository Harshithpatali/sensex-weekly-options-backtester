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
             "symbol": "SENSEX", "option_type": "CE", "close": 20.0 + i * 5, "lot_size": 10},
            {"trade_date": day, "expiry_date": expiry, "strike": 110.0,
             "symbol": "SENSEX", "option_type": "CE", "close": 10.0 + i, "lot_size": 10},
            {"trade_date": day, "expiry_date": expiry, "strike": 100.0,
             "symbol": "SENSEX", "option_type": "PE", "close": 20.0 + i * 4, "lot_size": 10},
            {"trade_date": day, "expiry_date": expiry, "strike": 90.0,
             "symbol": "SENSEX", "option_type": "PE", "close": 10.0 + i, "lot_size": 10},
        ])
    result = _option_debit_vertical_outcomes(
        index, pd.DataFrame(quotes), holding_periods=(1,), cost=CostConfig()
    )
    assert not result.empty
    first = result[result["entry_date"] == dates[1]].iloc[0]
    assert first["option_family"] == "DEBIT_SPREADS"
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
             "symbol": "SENSEX", "option_type": "CE", "close": 20.0, "lot_size": 10},
            {"trade_date": day, "expiry_date": expiry, "strike": 110.0,
             "symbol": "SENSEX", "option_type": "CE", "close": 10.0, "lot_size": 10},
            {"trade_date": day, "expiry_date": expiry, "strike": 100.0,
             "symbol": "SENSEX", "option_type": "PE", "close": 20.0, "lot_size": 10},
        ])
        if day != dates[2]:
            quotes.append({
                "trade_date": day, "expiry_date": expiry, "strike": 90.0,
                "symbol": "SENSEX", "option_type": "PE", "close": 10.0, "lot_size": 10,
            })
    result = _option_debit_vertical_outcomes(
        index, pd.DataFrame(quotes), holding_periods=(1,), cost=CostConfig()
    )
    first = result[result["entry_date"] == dates[1]].iloc[0]
    assert pd.notna(first["call_pnl"])
    assert pd.isna(first["put_pnl"])
