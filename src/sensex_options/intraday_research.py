from __future__ import annotations

"""Recent SENSEX intraday context and a strictly chronological put-filter study.

Yahoo intraday bars are index data, not option quotes. They can filter an EOD
option entry but cannot simulate option-premium intraday stop-loss execution.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from sensex_options.config import DEFAULT_PATHS, INDEX_TICKER


def download_sensex_intraday(
    interval: str = "5m",
    period: str = "60d",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Download recent intraday SENSEX bars; Yahoo restricts the lookback."""
    if interval not in {"5m", "15m"}:
        raise ValueError("Supported intraday intervals are 5m and 15m.")
    if period not in {"5d", "1mo", "3mo", "60d"}:
        raise ValueError("Use a recent period no longer than Yahoo's intraday window.")
    # Use explicit dates rather than a "60d" period token, which is not
    # accepted consistently by Yahoo's chart endpoint.
    days = {"5d": 5, "1mo": 30, "3mo": 60, "60d": 60}[period]
    end_date = pd.Timestamp.now(tz="Asia/Kolkata").date()
    start_date = end_date - pd.Timedelta(days=days - 1)
    frame = yf.Ticker(INDEX_TICKER).history(
        start=start_date.isoformat(),
        end=(end_date + pd.Timedelta(days=1)).isoformat(),
        interval=interval, auto_adjust=False, prepost=False
    )
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "date", "open", "high", "low", "close", "volume"])
    frame = frame.reset_index()
    time_col = "Datetime" if "Datetime" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={
        time_col: "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].dt.tz is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    frame["date"] = frame["timestamp"].dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame[(frame["close"] > 0) & (frame["open"] > 0)]
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    path = output_path or (DEFAULT_PATHS.outputs / "sensex_intraday_5m.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def latest_previous_close(index: pd.DataFrame) -> dict:
    """Return the latest completed daily close and its change from the prior close."""
    if index.empty or not {"date", "close"}.issubset(index.columns):
        return {"status": "unavailable"}
    frame = index[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date")
    if frame.empty:
        return {"status": "unavailable"}
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else None
    change = float(latest["close"] - previous["close"]) if previous is not None else None
    change_pct = float(change / previous["close"] * 100) if previous is not None and previous["close"] else None
    return {
        "status": "ok", "date": latest["date"].date().isoformat(),
        "close": float(latest["close"]),
        "change": change, "change_pct": change_pct,
        "label": "Latest completed session close (previous close)",
    }


def intraday_session_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Create end-of-session features known by an EOD entry decision."""
    required = {"timestamp", "open", "high", "low", "close"}
    if bars.empty or not required.issubset(bars.columns):
        return pd.DataFrame()
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].dt.tz is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    frame["date"] = frame["timestamp"].dt.normalize()
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).clip(lower=0)
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    rows = []
    for day, group in frame.sort_values("timestamp").groupby("date"):
        group = group.sort_values("timestamp")
        first, last = group.iloc[0], group.iloc[-1]
        # An index has no meaningful traded volume; use an equal-weighted
        # average of bar typical prices, never label this as VWAP.
        intraday_mean = float(((group["high"] + group["low"] + group["close"]) / 3).mean())
        rows.append({
            "date": day, "bars": int(len(group)),
            "session_return_pct": (float(last["close"]) / float(first["open"]) - 1) * 100,
            "close_vs_intraday_mean_pct": (float(last["close"]) / intraday_mean - 1) * 100 if intraday_mean else float("nan"),
            "range_pct": (float(group["high"].max()) - float(group["low"].min())) / float(first["open"]) * 100,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _metrics(group: pd.DataFrame, name: str) -> dict:
    pnl = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    curve = pnl.cumsum()
    peak = curve.cummax().clip(lower=0)
    drawdown = curve - peak
    positive = pnl[pnl > 0]
    return {
        "rule": name, "trades": int(len(pnl)), "net_pnl": round(float(pnl.sum()), 2),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else None,
        "profit_factor": round(gains / losses, 4) if losses else None,
        "expectancy": round(float(pnl.mean()), 2) if len(pnl) else None,
        "median_trade_pnl": round(float(pnl.median()), 2) if len(pnl) else None,
        "max_drawdown": round(float(drawdown.min()), 2) if len(pnl) else None,
        "best_trade": round(float(pnl.max()), 2) if len(pnl) else None,
        "worst_trade": round(float(pnl.min()), 2) if len(pnl) else None,
        "top3_wins_share_pct": round(float(positive.nlargest(3).sum() / pnl.sum() * 100), 2)
            if pnl.sum() > 0 and len(positive) else None,
    }


def evaluate_intraday_put_filters(
    daily_ledger: pd.DataFrame,
    bars: pd.DataFrame,
    strategy: str = "DAILY_ATM_PUT",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compare predeclared EOD index filters on a later chronological holdout.

    Intraday features use the full session and therefore are only valid for an
    EOD entry. The volatility threshold is fitted on the training segment only.
    """
    features = intraday_session_features(bars)
    if daily_ledger.empty or features.empty:
        return pd.DataFrame(), {"status": "insufficient_data", "note": "Need both daily option trades and recent 5m/15m index bars."}, features
    trades = daily_ledger.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce").dt.normalize()
    trades["net_pnl"] = pd.to_numeric(trades["net_pnl"], errors="coerce")
    trades = trades[(trades["strategy"] == strategy) & trades["entry_date"].notna()].copy()
    joined = trades.merge(features, left_on="entry_date", right_on="date", how="inner")
    joined = joined.dropna(subset=["net_pnl", "session_return_pct", "range_pct"]).sort_values("entry_date")
    dates = sorted(joined["entry_date"].drop_duplicates())
    if len(dates) < 10:
        return joined, {
            "status": "insufficient_data",
            "note": f"Only {len(dates)} matched option sessions. At least 10 are required; more is better.",
            "matched_sessions": len(dates),
        }, features
    split = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    train_dates, test_dates = set(dates[:split]), set(dates[split:])
    train = joined[joined["entry_date"].isin(train_dates)]
    test = joined[joined["entry_date"].isin(test_dates)]
    range_cutoff = float(train["range_pct"].median())
    rules = {
        "Baseline — every ATM Put": pd.Series(True, index=test.index),
        "Bearish session": test["session_return_pct"] < 0,
        "Close below intraday mean": test["close_vs_intraday_mean_pct"] < 0,
        "Bearish + below intraday mean": (test["session_return_pct"] < 0) & (test["close_vs_intraday_mean_pct"] < 0),
        "Bearish + lower volatility": (test["session_return_pct"] < 0) & (test["range_pct"] <= range_cutoff),
        "Bearish + below intraday mean + lower volatility": (
            (test["session_return_pct"] < 0) & (test["close_vs_intraday_mean_pct"] < 0)
            & (test["range_pct"] <= range_cutoff)
        ),
    }
    results = pd.DataFrame([_metrics(test.loc[mask], label) for label, mask in rules.items()])
    report = {
        "status": "ok", "strategy": strategy, "interval_note": "Features aggregated from downloaded 5m or 15m index bars.",
        "matched_sessions": len(dates), "training_sessions": len(train_dates), "validation_sessions": len(test_dates),
        "training_start": dates[0].date().isoformat(), "training_end": dates[split - 1].date().isoformat(),
        "validation_start": dates[split].date().isoformat(), "validation_end": dates[-1].date().isoformat(),
        "training_median_range_pct": round(range_cutoff, 5),
        "threshold_fitted_on": "training sessions only",
        "validation_metrics": results.to_dict("records"),
        "warning": "Recent 60-day intraday overlap is a small sample. Filters are fixed research hypotheses, not optimized trading signals.",
        "stop_loss_status": "Not tested: Yahoo index candles are not option-premium candles. An index move cannot be substituted for an option stop-loss.",
    }
    return joined, report, features
