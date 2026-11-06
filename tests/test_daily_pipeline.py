import pandas as pd

from sensex_options.daily_pipeline import merge_csv, save_entry_signal_trade_ledger


def test_merge_csv_preserves_existing_history_when_no_new_rows(tmp_path):
    path = tmp_path / "history.csv"
    old = pd.DataFrame({
        "trade_date": ["2024-01-01"], "expiry_date": ["2024-01-04"],
        "symbol": ["SENSEX"], "strike": [70000], "option_type": ["CE"], "close": [120.0],
    })
    old.to_csv(path, index=False)

    result = merge_csv(pd.DataFrame(), path, [
        "trade_date", "expiry_date", "symbol", "strike", "option_type"
    ])

    assert len(result) == 1
    assert path.exists()
    assert pd.read_csv(path).iloc[0]["close"] == 120.0


def test_merge_csv_deduplicates_contract_day_and_keeps_latest_value(tmp_path):
    path = tmp_path / "history.csv"
    keys = ["trade_date", "expiry_date", "symbol", "strike", "option_type"]
    old = pd.DataFrame([{
        "trade_date": "2024-01-01", "expiry_date": "2024-01-04",
        "symbol": "SENSEX", "strike": 70000, "option_type": "CE", "close": 100.0,
    }])
    old.to_csv(path, index=False)
    new = pd.DataFrame([{
        "trade_date": pd.Timestamp("2024-01-01"), "expiry_date": pd.Timestamp("2024-01-04"),
        "symbol": "SENSEX", "strike": 70000, "option_type": "CE", "close": 125.0,
    }])

    result = merge_csv(new, path, keys)

    assert len(result) == 1
    assert float(result.iloc[0]["close"]) == 125.0


def test_entry_signal_ledger_is_compressed_and_legacy_csv_removed(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    legacy_path = output_dir / "entry_signal_trade_ledger.csv"
    legacy_path.write_text("old,ledger\\n1,2\\n", encoding="utf-8")
    trades = pd.DataFrame([
        {"strategy": "LONG_H1D_TEST", "entry_date": "2024-01-02", "net_pnl": 125.5},
        {"strategy": "LONG_H1D_TEST", "entry_date": "2024-01-03", "net_pnl": -75.0},
    ])

    saved_path = save_entry_signal_trade_ledger(trades, output_dir)

    assert saved_path.name == "entry_signal_trade_ledger.csv.gz"
    assert saved_path.exists()
    assert not legacy_path.exists()
    restored = pd.read_csv(saved_path, compression="gzip")
    pd.testing.assert_frame_equal(restored, trades)
