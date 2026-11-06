"""Source helper unit tests (no network)."""

from pathlib import Path
import pandas as pd
import tempfile

from sensex_options.sources.contracts import resolve_lot_size
from sensex_options.sources.bse import parse_bse_fo_file


def test_resolve_lot_size_from_row():
    row = pd.Series({"lot_size": 15})
    assert resolve_lot_size(row) == 15


def test_resolve_lot_size_missing():
    row = pd.Series({"lot_size": float("nan")})
    assert resolve_lot_size(row) is None


def test_parse_empty_file(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    df = parse_bse_fo_file(p)
    assert df.empty
