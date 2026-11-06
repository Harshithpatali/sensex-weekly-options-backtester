"""
Expiry calendar derived from actual BSE contract data.
Does not hard-code a single weekday as expiry for the entire history.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def available_expiries(options: pd.DataFrame) -> pd.DatetimeIndex:
    """Return sorted unique expiry dates present in options data."""
    if options.empty or "expiry_date" not in options.columns:
        return pd.DatetimeIndex([])
    exp = pd.to_datetime(options["expiry_date"]).dropna().dt.normalize().unique()
    return pd.DatetimeIndex(sorted(exp))


def previous_trading_session(
    index: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """
    Find the latest SENSEX trading session strictly before ref_date.
    """
    if index.empty:
        return None
    dates = pd.to_datetime(index["date"]).dt.normalize()
    prior = dates[dates < ref_date.normalize()]
    if prior.empty:
        return None
    return prior.max()


def sessions_on_or_before(
    index: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """Latest session on or before ref_date."""
    if index.empty:
        return None
    dates = pd.to_datetime(index["date"]).dt.normalize()
    ok = dates[dates <= ref_date.normalize()]
    if ok.empty:
        return None
    return ok.max()


def validate_expiry_pair(
    index: pd.DataFrame,
    options: pd.DataFrame,
    expiry: pd.Timestamp,
) -> Tuple[bool, str, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Validate that an expiry has a usable previous session for entry
    and an expiry-session close for exit.

    Returns
    -------
    (ok, reason, entry_date, exit_date)
    """
    expiry = pd.Timestamp(expiry).normalize()
    entry = previous_trading_session(index, expiry)
    if entry is None:
        return False, "no_previous_trading_session", None, None

    # Exit: prefer actual trading session on expiry date, else last session on/before
    exit_d = sessions_on_or_before(index, expiry)
    if exit_d is None:
        return False, "no_exit_session", entry, None

    # Check options exist for this expiry on entry date
    opts = options[
        (pd.to_datetime(options["expiry_date"]).dt.normalize() == expiry)
        & (pd.to_datetime(options["trade_date"]).dt.normalize() == entry)
        & (options["option_type"] == "CE")
    ]
    if opts.empty:
        return False, "no_call_contracts_on_entry_date", entry, exit_d

    return True, "ok", entry, exit_d


def detect_duplicate_expiries(options: pd.DataFrame) -> List[dict]:
    """Report suspicious duplicate contract keys."""
    if options.empty:
        return []
    keys = ["trade_date", "expiry_date", "strike", "option_type", "symbol"]
    present = [k for k in keys if k in options.columns]
    if len(present) < 4:
        return []
    dup = options.duplicated(subset=present, keep=False)
    if not dup.any():
        return []
    rows = options.loc[dup, present].drop_duplicates().to_dict(orient="records")
    return rows[:100]  # cap
