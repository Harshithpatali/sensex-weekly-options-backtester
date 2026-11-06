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
        "TckrSymb": ["SENSEX", "SENSEX", "NIFTY"], "FinInstrmTp": ["IDO", "IDO", "IDO"],
        "StrkPric": [80000, 80000, 25000],
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
    assert set(parsed["instrument_type"]) == {"OPTIDX"}
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



def test_parser_excludes_contracts_expired_before_trade_date(tmp_path: Path):
    raw = pd.DataFrame({
        "TradDt": ["2026-11-05", "2026-11-05"],
        "XpryDt": ["2026-11-12", "2026-10-29"],
        "TckrSymb": ["SENSEX", "SENSEX"],
        "StrkPric": [80000, 80000],
        "OptnTp": ["CE", "PE"],
        "ClsPric": [105, 115],
    })
    path = tmp_path / "bse.csv"
    raw.to_csv(path, index=False)

    parsed = parse_bse_fo_file(path)

    assert len(parsed) == 1
    assert parsed.iloc[0]["trade_date"] == pd.Timestamp("2026-11-05")
    assert parsed.iloc[0]["expiry_date"] == pd.Timestamp("2026-11-12")


def test_load_options_filters_reversed_expiry_history(tmp_path: Path):
    from sensex_options.sources.bse import load_options

    path = tmp_path / "options.csv"
    pd.DataFrame({
        "trade_date": ["2026-11-05", "2026-11-05"],
        "expiry_date": ["2026-11-12", "2026-10-29"],
        "symbol": ["SENSEX", "SENSEX"],
        "strike": [80000, 80000],
        "option_type": ["CE", "PE"],
        "close": [105, 115],
    }).to_csv(path, index=False)

    loaded = load_options(path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["expiry_date"] >= loaded.iloc[0]["trade_date"]

def test_parser_supports_legacy_bse_derivatives_columns(tmp_path: Path):
    raw = pd.DataFrame({
        "INSTRUMENT": ["OPTIDX", "OPTIDX", "OPTIDX"],
        "SYMBOL": ["SENSEX", "SENSEX", "NIFTY"],
        "EXPIRY_DT": ["04-Jul-2024"] * 3,
        "STRIKE_PR": [80000, 80000, 25000],
        "OPTION_TYP": ["CE", "PE", "CE"],
        "OPEN": [100, 110, 90],
        "HIGH": [120, 130, 100],
        "LOW": [80, 90, 70],
        "CLOSE": [105, 115, 95],
        "SETTLE_PR": [106, 116, 96],
        "CONTRACTS": [100, 120, 200],
        "OPEN_INT": [1000, 1100, 2000],
        "CHG_IN_OI": [10, 20, 30],
        "TIMESTAMP": ["04-Jul-2024"] * 3,
    })
    path = tmp_path / "legacy_bse.csv"
    raw.to_csv(path, index=False)

    parsed = parse_bse_fo_file(path)

    assert len(parsed) == 2
    assert set(parsed["symbol"]) == {"SENSEX"}
    assert set(parsed["option_type"]) == {"CE", "PE"}
    assert parsed["trade_date"].eq(pd.Timestamp("2024-07-04")).all()
    assert parsed["expiry_date"].eq(pd.Timestamp("2024-07-04")).all()
    assert parsed["open_interest"].tolist() == [1000, 1100]
    assert parsed["volume"].tolist() == [100, 120]
    assert parsed["lot_size"].tolist() == [10, 10]


def test_parser_supports_legacy_bse_zip_archive(tmp_path: Path):
    import zipfile

    raw = pd.DataFrame({
        "INSTRUMENT": ["OPTIDX"],
        "SYMBOL": ["SENSEX"],
        "EXPIRY_DT": ["01-Aug-2024"],
        "STRIKE_PR": [81000],
        "OPTION_TYP": ["CE"],
        "OPEN": [125],
        "HIGH": [140],
        "LOW": [115],
        "CLOSE": [132],
        "SETTLE_PR": [133],
        "CONTRACTS": [75],
        "OPEN_INT": [900],
        "TIMESTAMP": ["04-Jul-2024"],
    })
    csv_bytes = raw.to_csv(index=False).encode("utf-8")
    path = tmp_path / "BSE_FO_20240704.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bhavcopy.csv", csv_bytes)

    parsed = parse_bse_fo_file(path)

    assert len(parsed) == 1
    assert parsed.iloc[0]["symbol"] == "SENSEX"
    assert parsed.iloc[0]["option_type"] == "CE"
    assert parsed.iloc[0]["strike"] == 81000
    assert parsed.iloc[0]["close"] == 132
    assert parsed.iloc[0]["lot_size"] == 10



def test_parser_supports_actual_spaced_bse_legacy_archive_schema(tmp_path: Path):
    raw = pd.DataFrame({
        "Contract Type": ["IF", "IO", "IO", "IO", "IO"],
        "Symbol": [
            "SENSEX23519FUT", "SENSEX2351963000CE",
            "SENSEX2351961500PE", "SENSEX2470580000CE", "SENSEX2501763000CE",
        ],
        "Expiry": ["19 May 2023", "19 May 2023", "19 May 2023", "05 Jul 2024", "17 Jan 2025"],
        "Strike": [0.0, 63000.0, 61500.0, 80000.0, 63000.0],
        "Option Type": [None, "CALL", "PUT", "CALL", "CALL"],
        "Open": [62175.0, 60.15, 48.35, 186.95, 10.0],
        "High": [63080.0, 92.0, 77.15, 186.95, 15.0],
        "Low": [62073.8, 45.1, 37.4, 0.2, 5.0],
        "Close": [62358.55, 54.4, 65.35, 79996.6, 12.0],
        "L T Date": ["15 May 2023", "15 May 2023", "15 May 2023", "05 Jul 2024", "10 Jan 2025"],
        "Contracts Traded": [454, 84, 107, 43963272, 200],
        "Open Interest": [420, 250, 260, 0, 300],
        "Is Traded": ["Y", "Y", "Y", "Y", "Y"],
    })
    path = tmp_path / "bhavcopy15-05-23.csv"
    raw.to_csv(path, index=False)

    parsed = parse_bse_fo_file(path)

    # Futures are excluded; Calls and Puts are decoded and receive the
    # documented 10-unit legacy SENSEX lot size.
    assert len(parsed) == 4
    options = parsed[parsed["option_type"].isin(["CE", "PE"])]
    assert set(options["option_type"]) == {"CE", "PE"}
    legacy_rows = options[options["trade_date"] <= pd.Timestamp("2024-07-05")]
    future_rows = options[options["trade_date"] > pd.Timestamp("2024-07-05")]
    assert legacy_rows["lot_size"].eq(10).all()
    assert future_rows["lot_size"].isna().all()
    first_day = options[options["trade_date"] == pd.Timestamp("2023-05-15")]
    assert first_day.set_index("option_type")["volume"].to_dict() == {"CE": 84, "PE": 107}
    assert first_day.set_index("option_type")["open_interest"].to_dict() == {"CE": 250, "PE": 260}
    assert len(first_day) == 2
    assert options["instrument_type"].eq("OPTIDX").all()
    # The expiry-day row has an index-level close, not an option premium.
    expiry_row = options[options["trade_date"] == pd.Timestamp("2024-07-05")].iloc[0]
    assert pd.isna(expiry_row["close"])


def test_archive_session_overrides_legacy_last_traded_date(tmp_path: Path):
    raw = pd.DataFrame({
        "Contract Type": ["IO"],
        "Symbol": ["SENSEX2351963000CE"],
        "Expiry": ["19 May 2023"],
        "Strike": [63000.0],
        "Option Type": ["CALL"],
        "Open": [60.15], "High": [92.0], "Low": [45.1], "Close": [54.4],
        "L T Date": ["15 May 2023"],
        "Contracts Traded": [84], "Open Interest": [250],
    })
    path = tmp_path / "legacy.csv"
    raw.to_csv(path, index=False)

    parsed = parse_bse_fo_file(path, trade_date="2023-05-16")

    assert len(parsed) == 1
    assert parsed.iloc[0]["trade_date"] == pd.Timestamp("2023-05-16")
