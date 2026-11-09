"""
Statistical, probability, and lightweight ML analysis of backtest results
and SENSEX index returns. Research only — does not generate synthetic option prices.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sensex_options.config import DEFAULT_PATHS

logger = logging.getLogger(__name__)


def trade_statistics(trades: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive statistics on completed trades."""
    c = trades[trades["status"] == "completed"].copy() if "status" in trades.columns else trades
    if c.empty or "net_pnl" not in c.columns:
        return {"n": 0, "note": "no completed trades"}

    pnl = c["net_pnl"].astype(float)
    ret = c["return_pct"].astype(float) if "return_pct" in c.columns else pd.Series(dtype=float)

    out: Dict[str, Any] = {
        "n_completed": int(len(c)),
        "mean_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "skew_pnl": float(pnl.skew()) if len(pnl) > 2 else None,
        "kurtosis_pnl": float(pnl.kurtosis()) if len(pnl) > 3 else None,
        "min_pnl": float(pnl.min()),
        "max_pnl": float(pnl.max()),
        "p25_pnl": float(pnl.quantile(0.25)),
        "p75_pnl": float(pnl.quantile(0.75)),
        "win_rate": float((pnl > 0).mean()),
        "loss_rate": float((pnl < 0).mean()),
    }
    if len(ret):
        out["mean_return_pct"] = float(ret.mean())
        out["std_return_pct"] = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
    return out


def probability_metrics(trades: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """
    Probability-oriented metrics:
      - empirical win probability
      - bootstrap CI for mean PnL and win rate
      - simple Kelly fraction (edge / odds) under empirical mean
    """
    c = trades[trades["status"] == "completed"].copy() if "status" in trades.columns else trades
    if c.empty or "net_pnl" not in c.columns:
        return {"n": 0}

    pnl = c["net_pnl"].astype(float).values
    rng = np.random.default_rng(seed)
    n = len(pnl)
    means = []
    win_rates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnl, size=n, replace=True)
        means.append(sample.mean())
        win_rates.append((sample > 0).mean())
    means = np.array(means)
    win_rates = np.array(win_rates)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    p_win = float((pnl > 0).mean())
    # Kelly for binary-ish: f* = p - (1-p)/b where b = avg_win/avg_loss
    if avg_loss > 0 and avg_win > 0:
        b = avg_win / avg_loss
        kelly = p_win - (1 - p_win) / b
    else:
        kelly = None

    return {
        "n": n,
        "empirical_win_prob": p_win,
        "mean_pnl": float(pnl.mean()),
        "bootstrap_mean_pnl_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "bootstrap_win_rate_ci95": [float(np.percentile(win_rates, 2.5)), float(np.percentile(win_rates, 97.5))],
        "prob_mean_pnl_positive": float((means > 0).mean()),
        "avg_win": avg_win,
        "avg_loss_abs": avg_loss,
        "kelly_fraction_empirical": kelly,
        "note": "Kelly is illustrative only; options payoffs are not binary and capital constraints apply.",
    }


def index_return_stats(index: pd.DataFrame) -> Dict[str, Any]:
    """Basic return distribution stats on SENSEX daily closes."""
    if index.empty or "close" not in index.columns:
        return {}
    s = index.sort_values("date")["close"].astype(float)
    rets = s.pct_change().dropna()
    if rets.empty:
        return {}
    return {
        "n_days": int(len(rets)),
        "mean_daily_return": float(rets.mean()),
        "std_daily_return": float(rets.std(ddof=1)),
        "sharpe_approx_ann": float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if rets.std() else None,
        "skew": float(rets.skew()),
        "kurtosis": float(rets.kurtosis()),
        "pct_up_days": float((rets > 0).mean()),
        "worst_day": float(rets.min()),
        "best_day": float(rets.max()),
    }


def ml_index_regime(index: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Lightweight unsupervised regime labels using rolling return/volatility features
    and k-means (numpy only, no sklearn required).
    Labels are research features, not trading signals.
    """
    out_dir = out_dir or DEFAULT_PATHS.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    if index.empty or len(index) < 60:
        return {"note": "insufficient index history for regime model"}

    df = index.sort_values("date").copy()
    df["ret"] = df["close"].pct_change()
    df["vol20"] = df["ret"].rolling(20).std()
    df["ret5"] = df["close"].pct_change(5)
    feat = df[["ret5", "vol20"]].dropna()
    if len(feat) < 30:
        return {"note": "insufficient features"}

    X = feat.values.astype(float)
    # Standardise
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma

    # Simple k-means k=3
    k = 3
    rng = np.random.default_rng(0)
    centroids = Z[rng.choice(len(Z), size=k, replace=False)]
    for _ in range(25):
        dist = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centroids[j] = Z[labels == j].mean(axis=0)

    feat = feat.copy()
    feat["regime"] = labels
    # Map regimes by mean vol
    vol_by = feat.groupby("regime")["vol20"].mean().sort_values()
    name_map = {int(vol_by.index[0]): "low_vol", int(vol_by.index[1]): "mid_vol", int(vol_by.index[-1]): "high_vol"}
    feat["regime_name"] = feat["regime"].map(name_map)

    path = out_dir / "index_regimes.csv"
    # align back to dates
    aligned = df.loc[feat.index, ["date"]].copy() if "date" in df.columns else feat.reset_index()
    if "date" in df.columns:
        aligned = df.loc[feat.index, ["date", "close"]].copy()
        aligned["regime"] = feat["regime"].values
        aligned["regime_name"] = feat["regime_name"].values
        aligned.to_csv(path, index=False)
    else:
        feat.to_csv(path)

    counts = feat["regime_name"].value_counts().to_dict()
    return {
        "n_labeled_days": int(len(feat)),
        "regime_counts": {str(k): int(v) for k, v in counts.items()},
        "centroids_standardised": centroids.tolist(),
        "output": str(path),
        "method": "kmeans_k3_on_ret5_vol20",
        "disclaimer": "Unsupervised regimes for research only; not predictive signals.",
    }


def ml_trade_classifier_proxy(trades: pd.DataFrame, index: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple logistic-style analysis: does prior 5-day index return correlate with trade win?
    Uses numpy only (sigmoid MLE not full sklearn). Reports correlation and contingency.
    """
    c = trades[trades["status"] == "completed"].copy() if len(trades) and "status" in trades.columns else trades
    if c.empty or index.empty:
        return {"note": "need completed trades and index"}

    idx = index.sort_values("date").copy()
    idx["ret5"] = idx["close"].pct_change(5)
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    c = c.copy()
    c["entry_date"] = pd.to_datetime(c["entry_date"]).dt.normalize()
    merged = c.merge(idx[["date", "ret5"]], left_on="entry_date", right_on="date", how="left")
    merged["win"] = (merged["net_pnl"] > 0).astype(int)
    valid = merged.dropna(subset=["ret5", "win"])
    if len(valid) < 3:
        return {"n": len(valid), "note": "too few overlapping trades"}

    corr = float(valid["ret5"].corr(valid["win"]))
    # Contingency: ret5 > 0 vs win
    up = valid["ret5"] > 0
    tab = pd.crosstab(up, valid["win"])
    return {
        "n": int(len(valid)),
        "corr_prior5d_return_vs_win": corr,
        "contingency_ret5_positive_vs_win": tab.to_dict(),
        "mean_ret5_when_win": float(valid.loc[valid["win"] == 1, "ret5"].mean()) if (valid["win"] == 1).any() else None,
        "mean_ret5_when_loss": float(valid.loc[valid["win"] == 0, "ret5"].mean()) if (valid["win"] == 0).any() else None,
        "disclaimer": "Exploratory association only; not causal or predictive out-of-sample.",
    }


def run_full_analytics(
    trades: pd.DataFrame,
    index: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = out_dir or DEFAULT_PATHS.outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "trade_statistics": trade_statistics(trades),
        "probability": probability_metrics(trades),
        "index_returns": index_return_stats(index),
        "ml_regimes": ml_index_regime(index, out_dir),
        "ml_trade_proxy": ml_trade_classifier_proxy(trades, index),
    }
    path = out_dir / "analytics_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Analytics report -> %s", path)
    result["report_path"] = str(path)
    return result


def strategy_performance_metrics(trades: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """Per-strategy trade metrics. Sharpe is approximate trade-level Sharpe."""
    if trades.empty or "net_pnl" not in trades.columns:
        return pd.DataFrame()
    frame = trades.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].eq("completed")]
    if frame.empty:
        return pd.DataFrame()
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce")
    frame["starting_capital"] = pd.to_numeric(frame.get("starting_capital", initial_capital), errors="coerce")
    frame["trade_return"] = frame["net_pnl"] / frame["starting_capital"].replace(0, np.nan)
    rows = []
    for strategy, group in frame.groupby("strategy"):
        group = group.sort_values("exit_date") if "exit_date" in group else group
        pnl = group["net_pnl"].dropna()
        if "return_pct" in group:
            returns = pd.to_numeric(group["return_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() / 100.0
        else:
            returns = group["trade_return"].replace([np.inf, -np.inf], np.nan).dropna()
        gains, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        # Unit-lot research is not a cash account. Drawdown is measured on an
        # additive premium-return index (base 100), separately from ₹ P&L.
        returns = returns.reset_index(drop=True)
        equity = 100.0 + returns.cumsum() * 100.0
        peak = equity.cummax()
        dd = equity - peak
        std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
        sharpe = float(returns.mean() / std * np.sqrt(52)) if pd.notna(std) and std > 0 else None
        rows.append({
            "strategy": strategy, "trades": int(len(pnl)), "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(gains / losses, 4) if losses > 0 else None,
            "expectancy": round(float(pnl.mean()), 2),
            "sharpe_trade_annualized": round(sharpe, 4) if sharpe is not None else None,
            "max_drawdown": round(float(dd.min()), 2) if len(dd) else 0.0,\n            "drawdown_basis": "additive one-lot premium-return index (base 100)",
            "max_drawdown_pct": round(float((dd / peak.replace(0, np.nan) * 100).min()), 2) if len(dd) else 0.0,
            "sample_warning": "Small samples are not statistically reliable" if len(pnl) < 30 else "",
        })
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def walk_forward_evaluation(trades: pd.DataFrame, initial_capital: float = 10000.0,
                            train_expiries: int = 8, test_expiries: int = 4) -> dict:
    """Chronological 70/30 expiry holdout plus rolling expiry test windows."""
    empty = {"status": "insufficient_data", "note": "Not enough distinct expiry cycles for chronological evaluation."}
    if trades.empty or "expiry_date" not in trades.columns:
        return empty
    frame = trades.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].eq("completed")]
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"], errors="coerce")
    frame = frame.dropna(subset=["expiry_date"]).sort_values(["expiry_date", "strategy"])
    expiries = sorted(frame["expiry_date"].drop_duplicates().tolist())
    if len(expiries) < max(3, train_expiries + 1):
        return {**empty, "distinct_expiries": len(expiries), "required_expiries": train_expiries + 1}
    split = max(1, min(len(expiries) - 1, int(len(expiries) * 0.70)))
    train_dates, test_dates = expiries[:split], expiries[split:]
    holdout = frame[frame["expiry_date"].isin(test_dates)]
    holdout_metrics = strategy_performance_metrics(holdout, initial_capital).to_dict("records")
    folds, start = [], train_expiries
    while start < len(expiries):
        fold_dates = expiries[start:start + test_expiries]
        if not fold_dates: break
        sample = frame[frame["expiry_date"].isin(fold_dates)]
        folds.append({
            "train_expiries_before_test": int(start),
            "test_expiries": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in fold_dates],
            "metrics": strategy_performance_metrics(sample, initial_capital).to_dict("records"),
        })
        start += test_expiries
    return {
        "status": "ok", "distinct_expiries": len(expiries), "train_expiries": len(train_dates),
        "holdout_expiries": len(test_dates),
        "holdout_start": pd.Timestamp(test_dates[0]).strftime("%Y-%m-%d"),
        "holdout_end": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d"),
        "holdout_metrics": holdout_metrics, "walk_forward_folds": folds,
        "methodology": "Chronological 70/30 expiry holdout plus rolling context/test windows. These are fixed rules, not fitted ML models. Position sizing follows the chronological replay. Trade Sharpe uses sqrt(52) annualisation and is not daily mark-to-market.",
    }
