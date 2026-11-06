import pandas as pd

from sensex_options.quote_data import load_and_validate_quotes, executable_round_trip_pnl


def quote_fixture():
    return pd.DataFrame([
        {"quote_timestamp": "2024-01-02T09:30:00Z", "expiry_date": "2024-01-11",
         "symbol": "SENSEX", "strike": 70000, "option_type": "CE",
         "bid": 100, "ask": 102, "bid_size": 50, "ask_size": 40},
        {"quote_timestamp": "2024-01-03T09:30:00Z", "expiry_date": "2024-01-11",
         "symbol": "SENSEX", "strike": 70000, "option_type": "CE",
         "bid": 110, "ask": 112, "bid_size": 30, "ask_size": 25},
        {"quote_timestamp": "2024-01-03T09:30:00Z", "expiry_date": "2024-01-11",
         "symbol": "SENSEX", "strike": 70000, "option_type": "PE",
         "bid": 0, "ask": 2, "bid_size": 10, "ask_size": 10},
    ])


def test_quote_loader_rejects_zero_bid_and_calculates_spread():
    quotes, report = load_and_validate_quotes(quote_fixture())
    assert report["input_rows"] == 3
    assert report["valid_rows"] == 2
    assert len(quotes) == 2
    assert (quotes["ask"] >= quotes["bid"]).all()
    assert (quotes["spread_pct"] > 0).all()


def test_executable_round_trip_uses_entry_ask_and_exit_bid():
    quotes, _ = load_and_validate_quotes(quote_fixture())
    trades = pd.DataFrame([{
        "entry_quote_timestamp": "2024-01-02T09:30:00Z",
        "exit_quote_timestamp": "2024-01-03T09:30:00Z",
        "expiry_date": "2024-01-11", "symbol": "SENSEX", "strike": 70000,
        "option_type": "CE", "quantity": 10,
    }])
    priced, report = executable_round_trip_pnl(trades, quotes)
    assert report["fully_quote_matched_trades"] == 1
    assert priced.iloc[0]["executable_entry_ask"] == 102
    assert priced.iloc[0]["executable_exit_bid"] == 110
    assert priced.iloc[0]["executable_pnl_before_fees"] == 80


def test_ohlc_without_bid_ask_is_not_accepted_as_quote_data():
    quotes, report = load_and_validate_quotes(pd.DataFrame([{"close": 100, "volume": 10}]))
    assert quotes.empty
    assert report["status"] == "invalid_schema"
