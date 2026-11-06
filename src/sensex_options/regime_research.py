from __future__ import annotations

"""Predeclared regime-gated index-signal diagnostics for observed SENSEX options.

This module deliberately does not invent option prices. It expects a paired
option outcome ledger generated from observed BSE contracts and evaluates
predeclared index-only filters using only lagged completed-session features.
It is a diagnostic research layer, not a live execution engine.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_regime_features(index: pd.DataFrame) -> pd.DataFrame:
    """Build lagged, point-in-time regime features for each completed session."""
    required = {"date", "close"}
    if index is None or index.empty or not required.issubset(index.columns):
        return pd.DataFrame()
    x = index.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close"):
        if col not in x:
            x[col] = x["close"] if col == "open" else np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = (x.dropna(subset=["date", "close"]).sort_values("date")
         .drop_duplicates("date").reset_index(drop=True))
    close = x["close"]
    ret = close.pct_change()
    x["ret_1"] = ret
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["vol_10"] = ret.rolling(10, min_periods=8).std()
    x["vol_20"] = ret.rolling(20, min_periods=15).std()
    # Percentile ranks use only the trailing history, excluding current value.
    prior_vol = x["vol_20"].shift(1)
    x["vol_percentile_126"] = prior_vol.rolling(126, min_periods=40).rank(pct=True)
    for span in (5, 10, 20, 50):
        x[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    x["trend_gap_5_20"] = x["ema_5"] / x["ema_20"] - 1
    x["trend_gap_20_50"] = x["ema_20"] / x["ema_50"] - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = 100 - 100 / (1 + rs)
    x.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100
    x.loc[(gain == 0) & (loss > 0), "rsi_14"] = 0
    prev = close.shift(1)
    tr = pd.concat([(x["high"] - x["low"]), (x["high"] - prev).abs(),
                    (x["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    up = x["high"].diff()
    down = -x["low"].diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    pdi = 100 * plus.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    x["adx_14"] = dx.rolling(14, min_periods=14).mean()
    x["atr_pct"] = atr / close
    # Regime is descriptive and based on known data only.
    x["regime"] = np.select(
        [
            (x["adx_14"] >= 22) & (x["trend_gap_5_20"] > 0),
            (x["adx_14"] >= 22) & (x["trend_gap_5_20"] < 0),
            x["vol_percentile_126"] >= 0.80,
            x["vol_percentile_126"] <= 0.20,
        ],
        ["TREND_UP", "TREND_DOWN", "HIGH_VOL", "LOW_VOL"],
        default="RANGE_OR_MIXED",
    )
    # Feature timestamp is the close of date; entry decision uses previous row.
    x["signal_date"] = x["date"]
    return x


def _pf(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gp = float(values[values > 0].sum())
    gl = float(-values[values < 0].sum())
    if gl == 0:
        return float("inf") if gp > 0 else None
    return gp / gl


def _stats(group: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(group.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    curve = pnl.cumsum()
    dd = curve - curve.cummax().clip(lower=0)
    return {
        "trades": int(len(pnl)),
        "net_pnl": round(float(pnl.sum()), 2),
        "profit_factor": round(_pf(pnl), 4) if _pf(pnl) is not None and np.isfinite(_pf(pnl)) else _pf(pnl),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else None,
        "expectancy": round(float(pnl.mean()), 2) if len(pnl) else None,
        "median_trade_pnl": round(float(pnl.median()), 2) if len(pnl) else None,
        "max_drawdown": round(float(dd.min()), 2) if len(pnl) else None,
    }


def run_regime_research(
    index: pd.DataFrame,
    option_ledger: pd.DataFrame,
    output_dir: str | Path | None = None,
    minimum_trades: int = 30,
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score fixed regime filters on observed options with strict chronology.

    Ledger contract: entry_date, exit_date, option_type (CE/PE), net_pnl.
    Features from the previous completed index session are joined to entry date.
    The latest segment is validation only; no parameters are optimized on it.
    """
    features = build_regime_features(index)
    required = {"entry_date", "exit_date", "option_type", "net_pnl"}
    if features.empty or option_ledger is None or option_ledger.empty or not required.issubset(option_ledger):
        report = {"status": "insufficient_data", "reason": "Need index OHLC and observed option ledger with entry/exit/type/net_pnl."}
        return pd.DataFrame(), pd.DataFrame(), report

    f = features.copy()
    f["entry_date"] = f["date"].shift(-1)
    # Lag regime/features: only yesterday's completed session is used for today's entry.
    fcols = ["entry_date", "signal_date", "regime", "ret_5", "ret_10",
             "vol_percentile_126", "trend_gap_5_20", "trend_gap_20_50",
             "rsi_14", "adx_14", "atr_pct"]
    f = f[fcols].dropna(subset=["entry_date"])
    t = option_ledger.copy()
    if "strategy" in t.columns:
        # Use one observed baseline book per side; exclude other simultaneous
        # strategy ledgers so the same market session is not counted repeatedly.
        calls = t[t["strategy"] == "DAILY_ATM_CALL"].copy()
        puts = t[t["strategy"] == "DAILY_ATM_PUT"].copy()
        if not calls.empty or not puts.empty:
            calls["option_type"] = "CE"
            puts["option_type"] = "PE"
            t = pd.concat([calls, puts], ignore_index=True)
        else:
            t = t.drop_duplicates(subset=["entry_date", "exit_date", "option_type", "net_pnl"])
    t["entry_date"] = pd.to_datetime(t["entry_date"], errors="coerce").dt.normalize()
    t["exit_date"] = pd.to_datetime(t["exit_date"], errors="coerce").dt.normalize()
    t["option_type"] = t["option_type"].astype(str).str.upper().str.strip()
    t["net_pnl"] = pd.to_numeric(t["net_pnl"], errors="coerce")
    t = t.dropna(subset=["entry_date", "exit_date", "net_pnl"])
    t = t[t["option_type"].isin(["CE", "PE"])]
    joined = t.merge(f, on="entry_date", how="inner", validate="many_to_one")
    if joined.empty:
        return joined, pd.DataFrame(), {"status": "no_matched_trades"}

    dates = sorted(joined["entry_date"].unique())
    if len(dates) < 2:
        return joined, pd.DataFrame(), {"status": "insufficient_sessions", "matched_sessions": len(dates)}
    split = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    validation_start = pd.Timestamp(dates[split])
    # Purge training trades whose realized exit reaches into validation.
    train_cutoff = validation_start - pd.Timedelta(days=1)
    joined["split"] = np.where(
        (joined["entry_date"] <= train_cutoff) & (joined["exit_date"] < validation_start),
        "train",
        np.where(joined["entry_date"] >= validation_start, "validation", "purged"),
    )

    # Predeclared signal rules; no parameter search is performed in validation.
    def direction_rule(row, family):
        if family == "MOMENTUM_5D":
            return "CE" if row["ret_5"] > 0 else ("PE" if row["ret_5"] < 0 else "")
        if family == "MOMENTUM_10D":
            return "CE" if row["ret_10"] > 0 else ("PE" if row["ret_10"] < 0 else "")
        if family == "TREND_EMA_ADX":
            if row["adx_14"] < 22: return ""
            return "CE" if row["trend_gap_5_20"] > 0 else ("PE" if row["trend_gap_5_20"] < 0 else "")
        if family == "TREND_PULLBACK":
            if row["adx_14"] < 18: return ""
            if row["trend_gap_5_20"] > 0 and row["rsi_14"] <= 45: return "CE"
            if row["trend_gap_5_20"] < 0 and row["rsi_14"] >= 55: return "PE"
            return ""
        if family == "MEAN_REVERSION":
            if row["rsi_14"] <= 25: return "CE"
            if row["rsi_14"] >= 75: return "PE"
            return ""
        return ""

    families = ["MOMENTUM_5D", "MOMENTUM_10D", "TREND_EMA_ADX", "TREND_PULLBACK", "MEAN_REVERSION"]
    gates = {
        "ALL_REGIMES": lambda r: True,
        "NOT_HIGH_VOL": lambda r: not (r["regime"] == "HIGH_VOL"),
        "TREND_ONLY": lambda r: r["regime"] in {"TREND_UP", "TREND_DOWN"},
        "LOW_OR_RANGE": lambda r: r["regime"] in {"LOW_VOL", "RANGE_OR_MIXED"},
        "ADX_AT_LEAST_22": lambda r: r["adx_14"] >= 22,
        "VOL_BELOW_80TH": lambda r: r["vol_percentile_126"] < 0.80,
    }
    records = []
    for family in families:
        for gate_name, gate in gates.items():
            selected = joined[
                joined.apply(lambda r: direction_rule(r, family) == r["option_type"] and gate(r), axis=1)
            ].copy()
            if selected.empty:
                continue
            selected["strategy"] = f"{family}__{gate_name}"
            for split_name in ("train", "validation"):
                part = selected[selected["split"] == split_name]
                records.append({"strategy": selected["strategy"].iloc[0], "split": split_name,
                                **_stats(part), "active_months": int(part["exit_date"].dt.to_period("M").nunique())})
    board = pd.DataFrame(records)
    if board.empty:
        report = {"status": "no_candidate_trades"}
    else:
        tr = board[board["split"] == "train"].set_index("strategy")
        va = board[board["split"] == "validation"].set_index("strategy")
        rows = []
        for name in sorted(set(tr.index) | set(va.index)):
            a = tr.loc[name] if name in tr.index else pd.Series(dtype=object)
            b = va.loc[name] if name in va.index else pd.Series(dtype=object)
            rows.append({"strategy": name,
                         "train_trades": int(a.get("trades", 0)), "train_pf": a.get("profit_factor"),
                         "train_net_pnl": a.get("net_pnl", 0), "train_months": int(a.get("active_months", 0)),
                         "validation_trades": int(b.get("trades", 0)), "validation_pf": b.get("profit_factor"),
                         "validation_net_pnl": b.get("net_pnl", 0), "validation_months": int(b.get("active_months", 0)),
                         "eligible_for_prospective_paper": bool(
                             int(a.get("trades", 0)) >= minimum_trades and a.get("net_pnl", 0) > 0
                             and a.get("profit_factor") is not None and float(a.get("profit_factor")) >= 1.20
                             and int(b.get("trades", 0)) >= max(10, minimum_trades // 3)
                             and b.get("net_pnl", 0) > 0
                             and b.get("profit_factor") is not None and float(b.get("profit_factor")) >= 1.20
                         )})
        summary = pd.DataFrame(rows).sort_values(["eligible_for_prospective_paper", "train_pf", "train_net_pnl"],
                                                  ascending=[False, False, False], na_position="last")
        report = {
            "status": "screen_complete",
            "method": "Predeclared 5 signal families x 6 regime gates; features lagged one completed session; split by entry date with exit-date purging at the boundary; observed option ledger only.",
            "train_end": train_cutoff.date().isoformat(),
            "validation_start": validation_start.date().isoformat(),
            "purged_overlap_trades": int((joined["split"] == "purged").sum()),
            "last_entry_date": pd.Timestamp(max(dates)).date().isoformat(),
            "minimum_train_trades": minimum_trades,
            "candidate_count": len(summary),
            "prospective_paper_candidates": int(summary["eligible_for_prospective_paper"].sum()),
            "important": "Index-only bars are not option-premium execution data. No intraday option stops, option VWAP, or live fills are simulated.",
        }
        board = summary
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joined.to_csv(out / "regime_research_joined.csv", index=False)
        board.to_csv(out / "regime_research_leaderboard.csv", index=False)
        (out / "regime_research_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return joined, board, report
