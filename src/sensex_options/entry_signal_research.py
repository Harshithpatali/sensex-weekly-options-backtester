from __future__ import annotations

"""Lagged index-signal research for long SENSEX ATM options.

Each candidate selects CE, PE, or stays in cash using completed index data
available before the option entry session. P&L comes from the paired, observed
BSE one-session ATM call/put ledger; no synthetic option returns are generated.
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

TARGET_PF = 1.20
MIN_MONTH_TRADES = 3
MIN_TRAIN_TRADES = 50
MIN_HOLDOUT_TRADES = 20


def _profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses > 0:
        return gains / losses
    return float("inf") if gains > 0 else None


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.mask((loss == 0) & (gain > 0), 100.0)
    value = value.mask((gain == 0) & (loss > 0), 0.0)
    return value


def build_lagged_features(index: pd.DataFrame) -> pd.DataFrame:
    """Build features through each completed session; caller maps prior row to entry."""
    required = {"date", "close"}
    if index is None or index.empty or not required.issubset(index.columns):
        return pd.DataFrame()
    x = index.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close"):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if x.empty:
        return x

    close = x["close"]
    high = x["high"] if "high" in x else close
    low = x["low"] if "low" in x else close
    open_ = x["open"] if "open" in x else close
    ret = close.pct_change()
    x["ret_1"] = close.pct_change(1)
    for period in (2, 3, 5, 10, 20):
        x[f"ret_{period}"] = close.pct_change(period)

    for span in (5, 10, 20, 30, 50):
        x[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    x["ema_5_20"] = x["ema_5"] / x["ema_20"] - 1
    x["ema_10_30"] = x["ema_10"] / x["ema_30"] - 1
    x["ema_20_50"] = x["ema_20"] / x["ema_50"] - 1
    x["rsi_7"] = _rsi(close, 7)
    x["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    x["macd_hist_pct"] = (macd - macd_signal) / close

    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    x["bb_z"] = (close - mid) / sd.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    x["atr_pct"] = true_range.rolling(14, min_periods=14).mean() / close
    x["atr_pct_median_60"] = x["atr_pct"].shift(1).rolling(60, min_periods=30).median()
    x["atr_pct_q75_60"] = x["atr_pct"].shift(1).rolling(60, min_periods=30).quantile(0.75)

    # ADX-like trend strength calculated from completed OHLC observations.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.rolling(14, min_periods=14).mean()
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx_14"] = dx.rolling(14, min_periods=14).mean()

    # Donchian boundaries exclude the signal session itself.
    for period in (5, 10, 20):
        x[f"donchian_high_{period}"] = high.shift(1).rolling(period, min_periods=period).max()
        x[f"donchian_low_{period}"] = low.shift(1).rolling(period, min_periods=period).min()

    x["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    x["gap_pct"] = open_ / previous_close - 1
    x["range_pct"] = (high - low) / previous_close
    # Compression is today's completed range against its trailing prior range median.
    x["range_median_20"] = x["range_pct"].shift(1).rolling(20, min_periods=15).median()
    x["range_compression"] = x["range_pct"] / x["range_median_20"].replace(0, np.nan)
    return x


def _paired_option_outcomes(daily_ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "entry_date", "exit_date", "expiry_date", "strike", "net_pnl"}
    if daily_ledger is None or daily_ledger.empty or not required.issubset(daily_ledger.columns):
        return pd.DataFrame()
    x = daily_ledger[daily_ledger["strategy"].isin(["DAILY_ATM_CALL", "DAILY_ATM_PUT"])].copy()
    if x.empty:
        return pd.DataFrame()
    for col in ("entry_date", "exit_date", "expiry_date"):
        x[col] = pd.to_datetime(x[col], errors="coerce").dt.normalize()
    x["strike"] = pd.to_numeric(x["strike"], errors="coerce")
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["entry_date", "exit_date", "expiry_date", "strike", "net_pnl"])
    keys = ["entry_date", "exit_date", "expiry_date", "strike"]
    calls = x[x["strategy"] == "DAILY_ATM_CALL"][keys + ["net_pnl"]].rename(columns={"net_pnl": "call_pnl"})
    puts = x[x["strategy"] == "DAILY_ATM_PUT"][keys + ["net_pnl"]].rename(columns={"net_pnl": "put_pnl"})
    paired = calls.merge(puts, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return paired
    paired["month"] = paired["exit_date"].dt.strftime("%Y-%m")
    return paired.sort_values("entry_date").reset_index(drop=True)


def _candidate_signals(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Return a deliberately diverse, predeclared family of signal hypotheses."""
    f = features
    signals: dict[str, pd.Series] = {}

    def direction(condition: pd.Series, valid: pd.Series | None = None) -> pd.Series:
        result = pd.Series("", index=f.index, dtype="object")
        mask = condition.notna()
        if valid is not None:
            mask &= valid.fillna(False)
        result.loc[mask & condition] = "CE"
        result.loc[mask & ~condition] = "PE"
        return result

    for period in (1, 2, 3, 5, 10):
        r = f[f"ret_{period}"]
        signals[f"RET{period}_MOMENTUM"] = direction(r > 0, r.notna())
        signals[f"RET{period}_REVERSAL"] = direction(r < 0, r.notna())

    for fast, slow, label in ((5, 20, "EMA5_20"), (10, 30, "EMA10_30"), (20, 50, "EMA20_50")):
        v = f[f"ema_{fast}"].notna() & f[f"ema_{slow}"].notna() & (f[f"ema_{fast}"] != f[f"ema_{slow}"])
        signals[f"{label}_TREND"] = direction(f[f"ema_{fast}"] > f[f"ema_{slow}"], v)

    for lower, upper in ((25, 75), (30, 70), (35, 65)):
        r = f["rsi_14"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[r <= lower] = "CE"
        sig.loc[r >= upper] = "PE"
        signals[f"RSI{lower}_{upper}_REVERSAL"] = sig

    for lower, upper in ((40, 60), (45, 55)):
        r = f["rsi_14"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[r >= upper] = "CE"
        sig.loc[r <= lower] = "PE"
        signals[f"RSI{lower}_{upper}_TREND"] = sig

    macd_valid = f["macd_hist_pct"].notna() & (f["macd_hist_pct"] != 0)
    signals["MACD_HIST_TREND"] = direction(f["macd_hist_pct"] > 0, macd_valid)

    for threshold in (1.0, 1.5, 2.0):
        z = f["bb_z"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[z <= -threshold] = "CE"
        sig.loc[z >= threshold] = "PE"
        signals[f"BB_Z{str(threshold).replace('.', '_')}_REVERSAL"] = sig

    for period in (5, 10, 20):
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[f["close"] > f[f"donchian_high_{period}"]] = "CE"
        sig.loc[f["close"] < f[f"donchian_low_{period}"]] = "PE"
        signals[f"DONCHIAN{period}_BREAKOUT"] = sig

    bull = f["ema_20"] > f["ema_50"]
    bear = f["ema_20"] < f["ema_50"]
    for lower, upper in ((40, 60), (45, 55)):
        rsi = f["rsi_14"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[bull & (rsi <= lower)] = "CE"
        sig.loc[bear & (rsi >= upper)] = "PE"
        signals[f"EMA20_50_RSI{lower}_{upper}_PULLBACK"] = sig

    # Trend-following only when both independent signals agree.
    ema_bull, ema_bear = f["ema_5"] > f["ema_20"], f["ema_5"] < f["ema_20"]
    macd_bull, macd_bear = f["macd_hist_pct"] > 0, f["macd_hist_pct"] < 0
    sig = pd.Series("", index=f.index, dtype="object")
    sig.loc[ema_bull & macd_bull] = "CE"
    sig.loc[ema_bear & macd_bear] = "PE"
    signals["EMA5_20_MACD_CONFIRM"] = sig

    # Trend + ADX strength regime.
    for minimum_adx in (15, 20, 25):
        valid = (f["adx_14"] >= minimum_adx)
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[valid & ema_bull] = "CE"
        sig.loc[valid & ema_bear] = "PE"
        signals[f"EMA5_20_ADX{minimum_adx}"] = sig

    # Volatility-regime filters use past-only trailing ATR thresholds.
    for label, vol_condition in (
        ("LOW_VOL", f["atr_pct"] <= f["atr_pct_median_60"]),
        ("BELOW_Q75_VOL", f["atr_pct"] <= f["atr_pct_q75_60"]),
        ("HIGH_VOL", f["atr_pct"] > f["atr_pct_median_60"]),
    ):
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[vol_condition & ema_bull] = "CE"
        sig.loc[vol_condition & ema_bear] = "PE"
        signals[f"{label}_EMA5_20"] = sig

    # Multi-horizon consensus; no entry when the horizons disagree.
    for first, second in ((3, 5), (5, 10)):
        a, b = f[f"ret_{first}"], f[f"ret_{second}"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[(a > 0) & (b > 0)] = "CE"
        sig.loc[(a < 0) & (b < 0)] = "PE"
        signals[f"RET{first}_{second}_CONSENSUS"] = sig

    # Breakouts after range compression, with an ATR floor.
    sig = pd.Series("", index=f.index, dtype="object")
    compression = f["range_compression"] <= 0.80
    atr_rising = f["atr_pct"] > f["atr_pct_median_60"]
    sig.loc[compression & atr_rising & (f["close"] > f["donchian_high_10"])] = "CE"
    sig.loc[compression & atr_rising & (f["close"] < f["donchian_low_10"])] = "PE"
    signals["COMPRESSION_VOL_BREAKOUT"] = sig

    # Opening gap continuation and fade, only for meaningful gaps.
    for threshold in (0.003, 0.005):
        gap = f["gap_pct"]
        continuation = pd.Series("", index=f.index, dtype="object")
        continuation.loc[gap >= threshold] = "CE"
        continuation.loc[gap <= -threshold] = "PE"
        fade = pd.Series("", index=f.index, dtype="object")
        fade.loc[gap >= threshold] = "PE"
        fade.loc[gap <= -threshold] = "CE"
        signals[f"GAP{int(threshold * 1000)}_CONTINUATION"] = continuation
        signals[f"GAP{int(threshold * 1000)}_FADE"] = fade

    return signals


def _monthly_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, month), group in frame.groupby(["strategy", "month"], sort=True):
        pnl = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        pf = _profit_factor(pnl)
        trades = len(pnl)
        target_testable = trades >= MIN_MONTH_TRADES
        rows.append({
            "strategy": strategy,
            "month": month,
            "trades": trades,
            "gross_profit": round(gp, 2),
            "gross_loss": round(gl, 2),
            "net_pnl": round(float(pnl.sum()), 2),
            "profit_factor": pf,
            "target_pf": TARGET_PF,
            "target_testable": target_testable,
            "target_met": bool(pf is not None and pf >= TARGET_PF) if target_testable else None,
        })
    return pd.DataFrame(rows)


def _score(frame: pd.DataFrame, split: str) -> dict:
    pnl = pd.to_numeric(frame["net_pnl"], errors="coerce").dropna() if not frame.empty else pd.Series(dtype=float)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = _profit_factor(pnl)
    net = float(pnl.sum())
    monthly = _monthly_stats(frame)
    active = monthly[monthly["target_testable"]] if not monthly.empty else monthly
    met = int(active["target_met"].fillna(False).sum()) if not active.empty else 0
    count = len(active)
    ordered = pnl.reset_index(drop=True)
    curve = 10000.0 + ordered.cumsum()
    peaks = pd.concat([pd.Series([10000.0]), curve], ignore_index=True).cummax().iloc[1:].reset_index(drop=True) if len(curve) else pd.Series(dtype=float)
    dd = (curve.reset_index(drop=True) - peaks) if len(curve) else pd.Series(dtype=float)
    return {
        f"{split}_trades": int(len(pnl)),
        f"{split}_net_pnl": round(net, 2),
        f"{split}_gross_profit": round(gp, 2),
        f"{split}_gross_loss": round(gl, 2),
        f"{split}_aggregate_pf": pf,
        f"{split}_win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else None,
        f"{split}_expectancy": round(float(pnl.mean()), 2) if len(pnl) else None,
        f"{split}_median_trade_pnl": round(float(pnl.median()), 2) if len(pnl) else None,
        f"{split}_max_drawdown": round(float(dd.min()), 2) if len(dd) else None,
        f"{split}_active_months": count,
        f"{split}_target_months": met,
        f"{split}_monthly_target_hit_pct": round(met / count * 100, 2) if count else None,
        f"{split}_positive_months": int((monthly["net_pnl"] > 0).sum()) if not monthly.empty else 0,
        f"{split}_months": len(monthly),
    }


def _build_monthly_selector(
    scored: pd.DataFrame,
    candidates: list[str],
    lookback_months: int,
    minimum_prior_trades: int = 20,
) -> pd.DataFrame:
    """At each month start, select only using the previous N complete calendar months."""
    if scored.empty:
        return pd.DataFrame()
    calendar = sorted(scored["month"].unique())
    rows = []
    for position, month in enumerate(calendar):
        prior_months = set(calendar[max(0, position - lookback_months):position])
        if len(prior_months) < lookback_months:
            rows.append({
                "strategy": f"ROLLING_{lookback_months}M_SELECTOR",
                "month": month,
                "selected_rule": "CASH",
                "selection_prior_months": lookback_months,
                "prior_trades": 0,
                "prior_pf": None,
                "trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "profit_factor": None,
                "target_pf": TARGET_PF,
                "target_testable": False,
                "target_met": None,
            })
            continue
        history = scored[scored["month"].isin(prior_months) & scored["strategy"].isin(candidates)]
        eligible = []
        for strategy, group in history.groupby("strategy"):
            pnl = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
            pf = _profit_factor(pnl)
            if len(pnl) >= minimum_prior_trades and pf is not None and pf >= TARGET_PF:
                eligible.append((float(pf), float(pnl.sum()), str(strategy), len(pnl)))
        if not eligible:
            rows.append({
                "strategy": f"ROLLING_{lookback_months}M_SELECTOR",
                "month": month,
                "selected_rule": "CASH",
                "selection_prior_months": lookback_months,
                "prior_trades": 0,
                "prior_pf": None,
                "trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "profit_factor": None,
                "target_pf": TARGET_PF,
                "target_testable": False,
                "target_met": None,
            })
            continue
        eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
        prior_pf, _, selected, prior_n = eligible[0]
        current = scored[(scored["month"] == month) & (scored["strategy"] == selected)]
        pnl = pd.to_numeric(current["net_pnl"], errors="coerce").dropna()
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        pf = _profit_factor(pnl)
        n = len(pnl)
        rows.append({
            "strategy": f"ROLLING_{lookback_months}M_SELECTOR",
            "month": month,
            "selected_rule": selected,
            "selection_prior_months": lookback_months,
            "prior_trades": prior_n,
            "prior_pf": prior_pf,
            "trades": n,
            "gross_profit": round(gp, 2),
            "gross_loss": round(gl, 2),
            "net_pnl": round(float(pnl.sum()), 2),
            "profit_factor": pf,
            "target_pf": TARGET_PF,
            "target_testable": n >= MIN_MONTH_TRADES,
            "target_met": bool(pf is not None and pf >= TARGET_PF) if n >= MIN_MONTH_TRADES else None,
        })
    return pd.DataFrame(rows)


def run_entry_signal_research(
    daily_ledger: pd.DataFrame,
    index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Test predeclared signals and rolling selectors on paired call/put outcomes."""
    pairs = _paired_option_outcomes(daily_ledger)
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_paired_option_data",
            "note": "Could not pair ATM Call and Put outcomes on the same entry, exit, expiry and strike.",
        }

    features = build_lagged_features(index)
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "missing_index_features",
            "note": "SENSEX OHLC history is required.",
        }
    date_to_position = {date: pos for pos, date in enumerate(features["date"])}
    pairs["signal_date"] = pairs["entry_date"].map(
        lambda entry: features.iloc[date_to_position[entry] - 1]["date"]
        if entry in date_to_position and date_to_position[entry] > 0 else pd.NaT
    )
    pairs = pairs.dropna(subset=["signal_date"]).copy()
    pairs["signal_date"] = pd.to_datetime(pairs["signal_date"]).dt.normalize()
    features = features.set_index("date")
    feature_cols = [col for col in features.columns if col != "date"]
    paired_features = pairs[["signal_date"]].join(features[feature_cols], on="signal_date")
    paired = pd.concat(
        [pairs.reset_index(drop=True), paired_features.drop(columns=["signal_date"]).reset_index(drop=True)],
        axis=1,
    )
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_feature_overlap",
            "note": "No paired option trades overlap the SENSEX feature history.",
        }

    signals = _candidate_signals(paired)
    records = []
    for name, signal in signals.items():
        chosen = paired.copy()
        chosen["signal"] = signal.to_numpy()
        chosen = chosen[chosen["signal"].isin(["CE", "PE"])].copy()
        if chosen.empty:
            continue
        chosen["strategy"] = name
        chosen["net_pnl"] = np.where(
            chosen["signal"] == "CE",
            chosen["call_pnl"],
            chosen["put_pnl"],
        )
        chosen["month"] = chosen["exit_date"].dt.strftime("%Y-%m")
        chosen["option_type"] = chosen["signal"]
        records.append(chosen[[
            "strategy", "entry_date", "exit_date", "signal_date", "month",
            "signal", "option_type", "strike", "net_pnl", "call_pnl", "put_pnl",
        ]])
    trades = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "no_signal_trades",
            "note": "No candidate generated a trade from the observed paired option outcomes.",
        }

    dates = sorted(paired["entry_date"].drop_duplicates())
    split_at = max(1, min(len(dates) - 1, int(len(dates) * 0.70))) if len(dates) >= 10 else max(1, len(dates) - 1)
    train_end = pd.Timestamp(dates[split_at - 1])
    holdout_start = pd.Timestamp(dates[split_at]) if split_at < len(dates) else None
    trades["split"] = np.where(trades["entry_date"] <= train_end, "train", "holdout")

    monthly = _monthly_stats(trades)
    leaderboard_rows = []
    candidate_names = sorted(trades["strategy"].unique())
    for name in candidate_names:
        group = trades[trades["strategy"] == name]
        train = group[group["split"] == "train"]
        holdout = group[group["split"] == "holdout"]
        train_metrics = _score(train, "train")
        holdout_metrics = _score(holdout, "holdout")
        full_metrics = _score(group, "full")
        train_ok = (
            train_metrics["train_trades"] >= MIN_TRAIN_TRADES
            and train_metrics["train_active_months"] >= 6
            and train_metrics["train_net_pnl"] > 0
            and train_metrics["train_aggregate_pf"] is not None
            and train_metrics["train_aggregate_pf"] >= TARGET_PF
            and (train_metrics["train_monthly_target_hit_pct"] or 0) >= 60
        )
        holdout_ok = (
            holdout_metrics["holdout_trades"] >= MIN_HOLDOUT_TRADES
            and holdout_metrics["holdout_active_months"] >= 4
            and holdout_metrics["holdout_net_pnl"] > 0
            and holdout_metrics["holdout_aggregate_pf"] is not None
            and holdout_metrics["holdout_aggregate_pf"] >= TARGET_PF
            and (holdout_metrics["holdout_monthly_target_hit_pct"] or 0) >= 50
        )
        leaderboard_rows.append({
            "strategy": name,
            "full_trades": full_metrics["full_trades"],
            "full_net_pnl": full_metrics["full_net_pnl"],
            "full_aggregate_pf": full_metrics["full_aggregate_pf"],
            "full_monthly_target_hit_pct": full_metrics["full_monthly_target_hit_pct"],
            **train_metrics,
            **holdout_metrics,
            "training_screen_pass": bool(train_ok),
            "holdout_validation_pass": bool(holdout_ok),
            "candidate_pass": bool(train_ok and holdout_ok),
        })

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = leaderboard.sort_values(
        ["candidate_pass", "train_monthly_target_hit_pct", "train_aggregate_pf", "train_net_pnl"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.insert(0, "training_rank", range(1, len(leaderboard) + 1))

    selector_rows = []
    for lookback in (3, 6):
        selector = _build_monthly_selector(trades[trades["split"].isin(["train", "holdout"])], candidate_names, lookback)
        if not selector.empty:
            selector_rows.append(selector)
    selector_monthly = pd.concat(selector_rows, ignore_index=True) if selector_rows else pd.DataFrame()
    if not selector_monthly.empty:
        for selector_name, group in selector_monthly.groupby("strategy"):
            monthly_parts = group.copy()
            monthly_parts["split"] = np.where(monthly_parts["month"] <= train_end.strftime("%Y-%m"), "train", "holdout")
            train_months = monthly_parts[monthly_parts["split"] == "train"]
            holdout_months = monthly_parts[monthly_parts["split"] == "holdout"]
            for split_name, chunk in (("train", train_months), ("holdout", holdout_months)):
                active = chunk[chunk["target_testable"]]
                pass_pct = float(active["target_met"].fillna(False).mean() * 100) if not active.empty else None
                pnl = pd.to_numeric(chunk["net_pnl"], errors="coerce").fillna(0)
                selector_monthly.loc[
                    (selector_monthly["strategy"] == selector_name) & selector_monthly["month"].isin(chunk["month"]),
                    f"{split_name}_target_hit_pct",
                ] = pass_pct

    report = {
        "status": "candidates_found" if bool(leaderboard["candidate_pass"].any()) else "no_candidate_met_research_screen",
        "target_definition": "Monthly gross profit / gross loss >= 1.20 in at least 60% of active training months and 50% of active holdout months; aggregate PF >= 1.20 in both windows.",
        "monthly_target_profit_factor": TARGET_PF,
        "minimum_trades_per_month_for_target_test": MIN_MONTH_TRADES,
        "training_screen": {
            "minimum_trades": MIN_TRAIN_TRADES,
            "minimum_active_months": 6,
            "minimum_target_month_hit_pct": 60,
            "minimum_aggregate_profit_factor": TARGET_PF,
            "net_pnl_must_be_positive": True,
        },
        "holdout_screen": {
            "minimum_trades": MIN_HOLDOUT_TRADES,
            "minimum_active_months": 4,
            "minimum_target_month_hit_pct": 50,
            "minimum_aggregate_profit_factor": TARGET_PF,
            "net_pnl_must_be_positive": True,
        },
        "method": (
            "Predeclared lagged index indicator and regime signals choose between same-date, same-expiry, "
            "same-strike paired ATM Call/Put outcomes. Signals are based only on the last completed index "
            "session before the option entry. Candidate rules are ranked using training only; holdout is "
            "reported separately. Rolling selectors use only prior complete calendar months."
        ),
        "paired_trades": int(len(paired)),
        "paired_start": str(paired["entry_date"].min().date()),
        "paired_end": str(paired["entry_date"].max().date()),
        "distinct_entry_sessions": int(paired["entry_date"].nunique()),
        "candidate_rules_tested": int(len(candidate_names)),
        "training_end": str(train_end.date()),
        "holdout_start": str(holdout_start.date()) if holdout_start is not None else None,
        "holdout_end": str(paired["entry_date"].max().date()),
        "candidate_pass_count": int(leaderboard["candidate_pass"].sum()),
        "top_training_rule": leaderboard.iloc[0].to_dict() if not leaderboard.empty else None,
        "rolling_selector_rules": ["ROLLING_3M_SELECTOR", "ROLLING_6M_SELECTOR"],
        "selector_note": (
            "Rolling selector is a paper-research allocation rule only. It goes to cash when no signal "
            "has at least 20 prior trades and prior-window PF >= 1.20."
        ),
        "caveat": (
            "This is a historical screening experiment, not a live profit guarantee. Option outcomes use "
            "one-lot EOD marks and estimated costs; intraday execution, queue position and slippage may differ. "
            "Repeated research iterations create selection bias, so any promising rule needs a new prospective "
            "paper-trading period before live deployment."
        ),
    }
    return trades, leaderboard, selector_monthly, report
