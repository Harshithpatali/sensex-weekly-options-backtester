"""Source helper unit tests (no network)."""
from pathlib import Path
import pandas as pd

from sensex_options.sources.contracts import resolve_lot_size
from sensex_options.sources.bse import parse_bse_fo_file


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
        "TradDt": ["2026-11-05"] * 3, "XpryDt": ["2026-11-12"] * 3,
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
