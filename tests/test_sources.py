"""Source helper unit tests (no network)."""
from pathlib import Path
import pandas as pd

from sensex_options.sources.contracts import resolve_lot_size
from sensex_options.sources.bse import parse_bse_fo_file, download_bse_options_range


def test_resolve_lot_size_from_row():
    assert resolve_lot_size(pd.Series({"lot_size": 15})) == 15


def test_resolve_lot_size_missing():
    assert resolve_lot_size(pd.Series({"lot_size": float("nan")})) is None


def test_parse_empty_file(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert parse_bse_fo_file(path).empty


def test_parser_keeps_only_sensex_calls_and_puts(tmp_path: Path):
    raw = pd.DataFrame({
        "TradDt": [20261105] * 3, "XpryDt": [20261112] * 3,
        "TckrSymb": ["SENSEX", "SENSEX", "NIFTY"], "StrkPric": [80000, 80000, 25000],
        "OptnTp": ["CE", "PE", "CE"], "OpnPric": [100, 110, 90],
        "HghPric": [120, 130, 100], "LwPric": [80, 90, 70],
        "ClsPric": [105, 115, 95], "TtlTradgVol": [100, 120, 200],
        "OpnIntrst": [1000, 1100, 2000], "NewBrdLotQty": [10, 10, 25],
    })
    path = tmp_path / "bse.csv"
    raw.to_csv(path, index=False)
    parsed = parse_bse_fo_file(path)
    assert set(parsed["symbol"]) == {"SENSEX"}
    assert set(parsed["option_type"]) == {"CE", "PE"}
    assert len(parsed) == 2



def test_no_download_does_not_overwrite_existing_processed_history(tmp_path, monkeypatch):
    import sensex_options.sources.bse as bse
    from sensex_options.config import DEFAULT_PATHS

    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    raw = tmp_path / "raw"
    processed.mkdir()
    reports.mkdir()
    existing_path = processed / "sensex_options.csv"
    original = pd.DataFrame([{
        "trade_date": "2024-01-01", "expiry_date": "2024-01-04",
        "symbol": "SENSEX", "strike": 70000, "option_type": "CE", "close": 123.0,
    }])
    original.to_csv(existing_path, index=False)

    monkeypatch.setattr(DEFAULT_PATHS, "processed", processed)
    monkeypatch.setattr(DEFAULT_PATHS, "reports", reports)
    monkeypatch.setattr(DEFAULT_PATHS, "raw_options", raw)
    monkeypatch.setattr(bse, "download_one_day", lambda *args, **kwargs: (None, "not_found_404"))
    monkeypatch.setattr(bse.time, "sleep", lambda *_: None)

    result = download_bse_options_range("2024-01-02", "2024-01-02", sleep=0)

    assert result.empty
    saved = pd.read_csv(existing_path)
    assert len(saved) == 1
    assert float(saved.iloc[0]["close"]) == 123.0



def test_downloader_does_not_cache_html_as_bhavcopy(tmp_path):
    from datetime import datetime
    from sensex_options.sources.bse import download_one_day

    class Response:
        status_code = 200
        content = b"<!doctype html><html><body>Access denied</body></html>"

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    path, status = download_one_day(
        datetime(2026, 11, 5), tmp_path, session=Session(), retries=1
    )
    assert path is None
    assert "html_instead_of_csv" in status
    assert not list(tmp_path.glob("*.csv"))
