from __future__ import annotations

"""Strict ingestion and executable-side valuation for timestamped option quotes.

Daily bhavcopy OHLC is not a substitute for bid/ask. This module accepts only
explicit timestamped quotes and uses ask-to-buy / bid-to-sell, never midpoint.
"""

from pathlib import Path

import numpy as np
import pandas as pd


QUOTE_COLUMNS = {
    "quote_timestamp", "expiry_date", "symbol", "strike", "option_type",
    "bid", "ask", "bid_size", "ask_size",
}


def load_and_validate_quotes(
    source: str | Path | pd.DataFrame,
    max_spread_pct: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Load quote snapshots and reject invalid/crossed/empty markets."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    missing = sorted(QUOTE_COLUMNS - set(frame.columns))
    if missing:
        return pd.DataFrame(), {
            "status": "invalid_schema", "missing_columns": missing,
            "note": "Historical OHLC/last-trade files do not satisfy this bid/ask schema.",
        }
    x = frame.copy()
    x["quote_timestamp"] = pd.to_datetime(x["quote_timestamp"], errors="coerce", utc=True)
    x["expiry_date"] = pd.to_datetime(x["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "bid", "ask", "bid_size", "ask_size"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x["option_type"] = x["option_type"].astype(str).str.strip().str.upper()
    x["option_type"] = x["option_type"].replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
    total = len(x)
    x = x.dropna(subset=list(QUOTE_COLUMNS))
    x = x[
        x["symbol"].str.startswith("SENSEX")
        & x["option_type"].isin(["CE", "PE"])
        & (x["strike"] > 0)
        & (x["bid"] > 0)
        & (x["ask"] >= x["bid"])
        & (x["bid_size"] > 0)
        & (x["ask_size"] > 0)
    ].copy()
    x["mid"] = (x["bid"] + x["ask"]) / 2
    x["spread_points"] = x["ask"] - x["bid"]
    x["spread_pct"] = x["spread_points"] / x["mid"] * 100
    x = x[x["spread_pct"] <= max_spread_pct]
    x = x.drop_duplicates(
        ["quote_timestamp", "expiry_date", "symbol", "strike", "option_type"],
        keep="last",
    ).sort_values("quote_timestamp").reset_index(drop=True)
    report = {
        "status": "ok" if not x.empty else "no_valid_quotes",
        "input_rows": total,
        "valid_rows": int(len(x)),
        "rejected_rows": int(total - len(x)),
        "max_spread_pct": max_spread_pct,
        "median_spread_pct": round(float(x["spread_pct"].median()), 4) if not x.empty else None,
        "quote_source_required": True,
        "execution_convention": "buy at ask; sell at bid; never use midpoint as a fill",
    }
    return x, report


def executable_round_trip_pnl(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    quantity_column: str = "quantity",
) -> tuple[pd.DataFrame, dict]:
    """Price trades only when exact entry/exit quote timestamps are supplied."""
    required_trades = {
        "entry_quote_timestamp", "exit_quote_timestamp", "expiry_date",
        "symbol", "strike", "option_type", quantity_column,
    }
    if trades is None or trades.empty or not required_trades.issubset(trades.columns):
        return pd.DataFrame(), {
            "status": "invalid_trade_schema",
            "missing_columns": sorted(required_trades - set(trades.columns)) if trades is not None else sorted(required_trades),
        }
    required_quotes = QUOTE_COLUMNS
    if quotes is None or quotes.empty or not required_quotes.issubset(quotes.columns):
        return pd.DataFrame(), {"status": "missing_validated_quotes"}
    q = quotes.copy()
    q["quote_timestamp"] = pd.to_datetime(q["quote_timestamp"], utc=True, errors="coerce")
    q["expiry_date"] = pd.to_datetime(q["expiry_date"], errors="coerce").dt.normalize()
    t = trades.copy()
    t["entry_quote_timestamp"] = pd.to_datetime(t["entry_quote_timestamp"], utc=True, errors="coerce")
    t["exit_quote_timestamp"] = pd.to_datetime(t["exit_quote_timestamp"], utc=True, errors="coerce")
    t["expiry_date"] = pd.to_datetime(t["expiry_date"], errors="coerce").dt.normalize()
    keys = ["expiry_date", "symbol", "strike", "option_type"]
    entry = q[keys + ["quote_timestamp", "ask"]].rename(
        columns={"quote_timestamp": "entry_quote_timestamp", "ask": "executable_entry_ask"}
    )
    exit_ = q[keys + ["quote_timestamp", "bid"]].rename(
        columns={"quote_timestamp": "exit_quote_timestamp", "bid": "executable_exit_bid"}
    )
    out = t.merge(entry, on=keys + ["entry_quote_timestamp"], how="left", validate="many_to_one")
    out = out.merge(exit_, on=keys + ["exit_quote_timestamp"], how="left", validate="many_to_one")
    out[quantity_column] = pd.to_numeric(out[quantity_column], errors="coerce")
    out["executable_pnl_before_fees"] = (
        out["executable_exit_bid"] - out["executable_entry_ask"]
    ) * out[quantity_column]
    out["quote_complete"] = out[["executable_entry_ask", "executable_exit_bid"]].notna().all(axis=1)
    report = {
        "status": "ok" if out["quote_complete"].any() else "no_exact_quote_matches",
        "trades": int(len(out)),
        "fully_quote_matched_trades": int(out["quote_complete"].sum()),
        "quote_coverage_pct": round(float(out["quote_complete"].mean() * 100), 2),
        "pricing": "entry ask to exit bid; fees still need to be deducted separately",
    }
    return out, report
