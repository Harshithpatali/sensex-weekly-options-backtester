"""
SENSEX index data via Yahoo Finance (yfinance).
Ticker: ^BSESN
Research source only — verify independently for production use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from sensex_options.config import DEFAULT_PATHS, INDEX_TICKER

logger = logging.getLogger(__name__)


def download_sensex_index(
    start: str,
    end: str,
    raw_dir: Optional[Path] = None,
    processed_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download historical SENSEX daily OHLCV from Yahoo Finance.

    Parameters
    ----------
    start : str
        Inclusive start date YYYY-MM-DD.
    end : str
        End date YYYY-MM-DD. yfinance treats end as exclusive for some versions;
        we request end+1 day internally to maximise coverage and then filter.
    raw_dir : Path
        Directory to store original download.
    processed_path : Path
        Path for normalised CSV.

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, volume
    """
    raw_dir = raw_dir or DEFAULT_PATHS.raw_index
    processed_path = processed_path or DEFAULT_PATHS.processed / "sensex_index.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading SENSEX (^BSESN) from %s to %s via yfinance", start, end)

    # Request a day past end to handle exclusive end behaviour
    end_adj = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ticker = yf.Ticker(INDEX_TICKER)
    df = ticker.history(start=start, end=end_adj, auto_adjust=False)

    if df.empty:
        logger.warning("No index data returned for range %s – %s", start, end)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = df.reset_index()
    # Normalise column names and timezone
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={
        date_col: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Clip to requested inclusive range
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

    raw_path = raw_dir / "sensex_index_yfinance.csv"
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)
    logger.info("Saved %d index rows to %s and %s", len(df), raw_path, processed_path)
    return df


def load_index(path: Optional[Path] = None) -> pd.DataFrame:
    """Load processed SENSEX index data."""
    path = path or DEFAULT_PATHS.processed / "sensex_index.csv"
    if not path.exists():
        logger.warning("Index file not found: %s", path)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)
