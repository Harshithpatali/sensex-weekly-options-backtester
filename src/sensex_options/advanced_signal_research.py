from __future__ import annotations

"""Advanced, rule-based SENSEX option signal and market-regime research.

Signals are calculated from the previous completed SENSEX session. A unit-lot
ATM option is entered at the current session's observed EOD close and exited
at the next session's observed EOD close. No future index or option marks enter
the signal. This is a research simulator, not a live-fill model.
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs

MONTHLY_TARGET_PF = 1.20
MIN_MONTHLY_TRADES = 3


def _research_cost() -> CostConfig:
    return CostConfig(
        brokerage_per_lot=20.0,
        stt_pct=0.001,
        exchange_charges_pct=0.0003,
        gst_pct=0.18,
        stamp_duty_pct=0.00003,
        entry_slippage=0.5,
        exit_slippage=0.5,
        spread_points=1.0,
        market_impact_points=0.25,
    )


def build_lagged_features(index: pd.DataFrame) -> pd.DataFrame:
    """Build indicators as of each close; callers use row t-1 for entry t."""
    required = {"date", "close"}
    if index is None or index.empty or not required.issubset(index.columns):
        return pd.DataFrame()
    frame = index.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    close = frame["close"]
    ret1 = close.pct_change()
    frame["ret1"] = ret1
    for horizon in (2, 3, 5, 10, 20):
        frame[f"ret{horizon}"] = close.pct_change(horizon)

    frame["vol10"] = ret1.rolling(10, min_periods=8).std()
    frame["vol20"] = ret1.rolling(20, min_periods=15).std()
    frame["vol60"] = ret1.rolling(60, min_periods=40).std()
    frame["vol_q30"] = frame["vol20"].rolling(126, min_periods=40).quantile(0.30)
    frame["vol_q50"] = frame["vol20"].rolling(126, min_periods=40).quantile(0.50)
    frame["vol_q70"] = frame["vol20"].rolling(126, min_periods=40).quantile(0.70)
    frame["vol_q80"] = frame["vol20"].rolling(126, min_periods=40).quantile(0.80)

    for span in (5, 8, 10, 13, 20, 21, 30, 50):
        frame[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()
    frame["ema_gap_5_20"] = (frame["ema5"] - frame["ema20"]) / close
    frame["ema_gap_8_21"] = (frame["ema8"] - frame["ema21"]) / close
    frame["ema_gap_10_30"] = (frame["ema10"] - frame["ema30"]) / close
    frame["ema_gap_5_50"] = (frame["ema5"] - frame["ema50"]) / close

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    frame["rsi14"] = 100 - 100 / (1 + rs)
    frame.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100.0
    frame.loc[(avg_gain == 0) & (avg_loss > 0), "rsi14"] = 0.0

    # True range and Wilder-style directional movement / ADX.
    previous_close = close.shift(1)
    high = frame["high"]
    low = frame["low"]
    tr = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    atr14 = tr.rolling(14, min_periods=14).mean()
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    frame["atr_pct"] = atr14 / close
    frame["adx14"] = dx.rolling(14, min_periods=14).mean()
    frame["plus_di"] = plus_di
    frame["minus_di"] = minus_di

    # Channel boundaries are strictly historical (shifted by one session).
    for window in (5, 10, 20):
        frame[f"prior_close_high_{window}"] = close.shift(1).rolling(window, min_periods=window).max()
        frame[f"prior_close_low_{window}"] = close.shift(1).rolling(window, min_periods=window).min()
        frame[f"prior_high_{window}"] = high.shift(1).rolling(window, min_periods=window).max()
        frame[f"prior_low_{window}"] = low.shift(1).rolling(window, min_periods=window).min()

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    frame["bb_z20"] = (close - mid) / std.replace(0, np.nan)
    frame["range_pct"] = (high - low) / close
    frame["range_median20"] = frame["range_pct"].rolling(20, min_periods=15).median()
    frame["range_ratio"] = frame["range_pct"] / frame["range_median20"].replace(0, np.nan)

    # All features are known at the session close. No centered or future windows.
    return frame


def _direction(condition_call: pd.Series, condition_put: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [condition_call.fillna(False).to_numpy(), condition_put.fillna(False).to_numpy()],
            ["CE", "PE"],
            default="",
        ),
        index=condition_call.index,
        dtype="object",
    )


def build_signal_specs(features: pd.DataFrame) -> dict[str, pd.Series]:
    """A predeclared family of trend, reversal, breakout and regime-gated rules."""
    if features.empty:
        return {}
    specs: dict[str, pd.Series] = {}

    # 1) Volatility-normalised momentum across multiple horizons.
    for horizon in (2, 3, 5, 10, 20):
        ret = features[f"ret{horizon}"]
        vol = features["vol20"] * np.sqrt(horizon)
        z = ret / vol.replace(0, np.nan)
        for threshold in (0.0, 0.35, 0.70):
            name = f"MOM{horizon}_Z{int(threshold * 100):02d}"
            specs[name] = _direction(z > threshold, z < -threshold)

    # 2) EMA trend strength with volatility-scaled separation thresholds.
    ema_defs = (
        ("EMA5_20", "ema_gap_5_20"),
        ("EMA8_21", "ema_gap_8_21"),
        ("EMA10_30", "ema_gap_10_30"),
        ("EMA5_50", "ema_gap_5_50"),
    )
    for label, gap_column in ema_defs:
        normalised_gap = features[gap_column] / features["vol20"].replace(0, np.nan)
        for threshold in (0.0, 0.15, 0.30):
            name = f"{label}_GAP{int(threshold * 100):02d}"
            specs[name] = _direction(
                normalised_gap > threshold,
                normalised_gap < -threshold,
            )

    # 3) RSI reversal and RSI continuation rules.
    for lower, upper in ((20, 80), (25, 75), (30, 70), (35, 65)):
        specs[f"RSI_REV_{lower}_{upper}"] = _direction(
            features["rsi14"] <= lower,
            features["rsi14"] >= upper,
        )
    for lower, upper in ((45, 55), (40, 60), (35, 65)):
        specs[f"RSI_TREND_{lower}_{upper}"] = _direction(
            features["rsi14"] >= upper,
            features["rsi14"] <= lower,
        )

    # 4) Close-confirmed channel breakouts.
    for window in (5, 10, 20):
        specs[f"CHANNEL_CLOSE_{window}"] = _direction(
            features["close"] > features[f"prior_close_high_{window}"],
            features["close"] < features[f"prior_close_low_{window}"],
        )
        specs[f"CHANNEL_WICK_{window}"] = _direction(
            features["close"] > features[f"prior_high_{window}"],
            features["close"] < features[f"prior_low_{window}"],
        )

    # 5) Bollinger displacement: both continuation and mean-reversion hypotheses.
    for width in (1.25, 1.50, 1.75, 2.00):
        specs[f"BB_REVERT_{int(width * 100)}"] = _direction(
            features["bb_z20"] <= -width,
            features["bb_z20"] >= width,
        )
        specs[f"BB_BREAKOUT_{int(width * 100)}"] = _direction(
            features["bb_z20"] >= width,
            features["bb_z20"] <= -width,
        )

    # 6) Trend regimes: ADX confirms trend rules; low ADX gates reversals.
    core_trends = [
        name for name in specs
        if name.startswith(("MOM", "EMA", "CHANNEL", "BB_BREAKOUT"))
    ]
    for base in core_trends:
        signal = specs[base]
        for threshold in (15, 20, 25):
            specs[f"{base}_ADX_GE{threshold}"] = signal.where(
                features["adx14"] >= threshold, ""
            )
    reversal_bases = [
        name for name in specs
        if name.startswith(("RSI_REV", "BB_REVERT"))
    ]
    for base in reversal_bases:
        signal = specs[base]
        for threshold in (15, 20, 25):
            specs[f"{base}_ADX_LE{threshold}"] = signal.where(
                features["adx14"] <= threshold, ""
            )

    # 7) Volatility regime gating; thresholds are trailing, expanding-in-time quantiles.
    low_vol = features["vol20"] <= features["vol_q30"]
    mid_vol = features["vol20"].between(features["vol_q30"], features["vol_q70"])
    high_vol = features["vol20"] >= features["vol_q70"]
    for base in (
        "MOM3_Z00", "MOM3_Z35", "MOM5_Z00", "MOM5_Z35",
        "MOM10_Z00", "MOM10_Z35", "EMA5_20_GAP00", "EMA8_21_GAP15",
        "CHANNEL_CLOSE_10", "CHANNEL_WICK_10",
    ):
        signal = specs[base]
        for suffix, gate in (("LOWVOL", low_vol), ("MIDVOL", mid_vol), ("HIGHVOL", high_vol)):
            specs[f"{base}_{suffix}"] = signal.where(gate, "")

    # 8) Volatility contraction and directional expansion.
    compressed = features["range_ratio"] <= 0.75
    expanded = features["range_ratio"] >= 1.25
    for base in ("CHANNEL_CLOSE_10", "CHANNEL_CLOSE_20", "MOM3_Z35", "MOM5_Z35"):
        signal = specs[base]
        specs[f"{base}_AFTER_COMPRESSION"] = signal.where(compressed, "")
        specs[f"{base}_RANGE_EXPANSION"] = signal.where(expanded, "")

    # 9) Directional movement confirmation; the DI cross gives a separate trend cue.
    specs["DMI_DIRECTION"] = _direction(
        (features["plus_di"] > features["minus_di"]) & (features["adx14"] >= 15),
        (features["minus_di"] > features["plus_di"]) & (features["adx14"] >= 15),
    )
    specs["DMI_STRONG"] = _direction(
        (features["plus_di"] > features["minus_di"]) & (features["adx14"] >= 25),
        (features["minus_di"] > features["plus_di"]) & (features["adx14"] >= 25),
    )

    # 10) Multi-signal consensus. At least two independent directional votes.
    vote_columns = [
        _direction(features["ret3"] > 0, features["ret3"] < 0),
        _direction(features["ret5"] > 0, features["ret5"] < 0),
        _direction(features["ret10"] > 0, features["ret10"] < 0),
        _direction(features["ema_gap_5_20"] > 0, features["ema_gap_5_20"] < 0),
    ]
    votes = pd.concat(vote_columns, axis=1)
    call_votes = votes.eq("CE").sum(axis=1)
    put_votes = votes.eq("PE").sum(axis=1)
    for minimum in (2, 3, 4):
        specs[f"CONSENSUS_{minimum}_OF4"] = _direction(
            call_votes >= minimum,
            put_votes >= minimum,
        )
    for minimum_adx in (15, 20):
        signal = specs["CONSENSUS_2_OF4"]
        specs[f"CONSENSUS_2_OF4_ADX{minimum_adx}"] = signal.where(
            features["adx14"] >= minimum_adx, ""
        )

    # 11) Price-structure reversal: a fresh close near the opposite side of range.
    close_location = (features["close"] - features["low"]) / (
        features["high"] - features["low"]
    ).replace(0, np.nan)
    specs["CLOSE_LOCATION_REVERSAL"] = _direction(
        (close_location >= 0.80) & (features["ret1"] < 0) & (features["rsi14"] < 45),
        (close_location <= 0.20) & (features["ret1"] > 0) & (features["rsi14"] > 55),
    )
    specs["CLOSE_LOCATION_TREND"] = _direction(
        (close_location >= 0.80) & (features["ret1"] > 0) & (features["adx14"] >= 15),
        (close_location <= 0.20) & (features["ret1"] < 0) & (features["adx14"] >= 15),
    )

    # Remove duplicate rules with identical complete signal series. Preserve first name.
    unique: dict[str, pd.Series] = {}
    signatures: set[tuple[str, ...]] = set()
    for name, series in specs.items():
        signature = tuple(series.fillna("").astype(str).tolist())
        if signature in signatures:
            continue
        signatures.add(signature)
        unique[name] = series.fillna("")
    return unique


def _trade_bucket() -> dict[str, float]:
    return {"trades": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0}


def _accumulate(bucket: dict, pnl: float) -> None:
    bucket["trades"] += 1
    bucket["net_pnl"] += pnl
    if pnl > 0:
        bucket["gross_profit"] += pnl
    elif pnl < 0:
        bucket["gross_loss"] += -pnl


def _pf(gross_profit: float, gross_loss: float) -> float | None:
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return None


def _monthly_row(strategy: str, month: str, bucket: dict) -> dict:
    trades = int(bucket["trades"])
    gp = float(bucket["gross_profit"])
    gl = float(bucket["gross_loss"])
    pf = _pf(gp, gl)
    target_met = None if trades < MIN_MONTHLY_TRADES else bool(pf is not None and pf >= MONTHLY_TARGET_PF)
    return {
        "strategy": strategy,
        "month": month,
        "trades": trades,
        "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2),
        "net_pnl": round(float(bucket["net_pnl"]), 2),
        "profit_factor": pf,
        "target_met": target_met,
        "month_status": (
            "INSUFFICIENT_TRADES" if trades < MIN_MONTHLY_TRADES
            else "PASS" if target_met else "FAIL"
        ),
    }


def _period_score(strategy: str, period: str, period_months: dict) -> dict:
    selected = [
        (month, bucket)
        for (name, split, month), bucket in period_months.items()
        if name == strategy and split == period
    ]
    n = sum(int(bucket["trades"]) for _, bucket in selected)
    gp = sum(float(bucket["gross_profit"]) for _, bucket in selected)
    gl = sum(float(bucket["gross_loss"]) for _, bucket in selected)
    net = sum(float(bucket["net_pnl"]) for _, bucket in selected)
    active = [
        _monthly_row(strategy, month, bucket)
        for month, bucket in selected
        if int(bucket["trades"]) >= MIN_MONTHLY_TRADES
    ]
    passed = sum(row["target_met"] is True for row in active)
    hit_pct = passed / len(active) * 100 if active else None
    return {
        f"{period}_trades": int(n),
        f"{period}_net_pnl": round(net, 2),
        f"{period}_aggregate_pf": _pf(gp, gl),
        f"{period}_gross_profit": round(gp, 2),
        f"{period}_gross_loss": round(gl, 2),
        f"{period}_active_months": len(active),
        f"{period}_target_months": int(passed),
        f"{period}_target_hit_pct": round(hit_pct, 2) if hit_pct is not None else None,
        f"{period}_all_active_months_pass": bool(active and passed == len(active)),
        f"{period}_median_trade_pnl": None,
    }


def run_advanced_signal_research(
    index: pd.DataFrame,
    options: pd.DataFrame,
    capital: float = 10000.0,
    cost: CostConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Replay a predeclared signal/regime family and score a chronological holdout."""
    empty = pd.DataFrame()
    if index is None or index.empty or options is None or options.empty:
        return empty, empty, empty, {"status": "insufficient_data", "note": "Index or option data is empty."}

    features = build_lagged_features(index)
    if len(features) < 100:
        return empty, empty, empty, {"status": "insufficient_data", "note": "At least 100 index sessions are required."}
    specs = build_signal_specs(features)
    if not specs:
        return empty, empty, empty, {"status": "insufficient_signals"}

    cfg = cost or _research_cost()
    idx = features
    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for column in ("strike", "close", "lot_size"):
        opt[column] = pd.to_numeric(opt[column], errors="coerce")
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    if "symbol" not in opt:
        opt["symbol"] = ""
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[
        (opt["close"] >= 0) & (opt["strike"] > 0)
        & opt["option_type"].isin(["CE", "PE"])
    ].sort_values(["trade_date", "expiry_date", "strike", "symbol"])
    by_day = {day: frame for day, frame in opt.groupby("trade_date", sort=False)}

    entry_dates = sorted(idx["date"].iloc[1:-1].drop_duplicates().tolist())
    if len(entry_dates) < 30:
        return empty, empty, empty, {"status": "insufficient_data", "note": "Too few sessions for candidate and holdout evaluation."}
    split_at = max(1, min(len(entry_dates) - 1, int(len(entry_dates) * 0.70)))
    train_end = pd.Timestamp(entry_dates[split_at - 1])
    holdout_start = pd.Timestamp(entry_dates[split_at])

    trades: list[dict] = []
    signal_names = list(specs)
    signal_arrays = {name: series.to_numpy(dtype=object) for name, series in specs.items()}
    rows = idx.to_dict("records")
    for i in range(1, len(rows) - 1):
        signal = rows[i - 1]
        entry = rows[i]
        exit_row = rows[i + 1]
        entry_day = pd.Timestamp(entry["date"])
        exit_day = pd.Timestamp(exit_row["date"])
        signal_day = pd.Timestamp(signal["date"])
        entry_options = by_day.get(entry_day)
        exit_options = by_day.get(exit_day)
        if entry_options is None or exit_options is None:
            continue
        # One nearest expiry and one ATM strike shared by all signals that day.
        eligible = entry_options.loc[
            entry_options["expiry_date"] >= exit_day, "expiry_date"
        ].dropna().unique()
        if len(eligible) == 0:
            continue
        expiry = pd.Timestamp(min(eligible))
        chain = entry_options[entry_options["expiry_date"] == expiry]
        strikes = chain["strike"].dropna().unique()
        if len(strikes) == 0:
            continue
        spot_reference = float(signal["close"])
        atm = min(strikes, key=lambda value: abs(float(value) - spot_reference))
        exit_chain = exit_options[exit_options["expiry_date"] == expiry]

        for strategy in signal_names:
            option_type = signal_arrays[strategy][i - 1]
            if option_type not in ("CE", "PE"):
                continue
            candidates = chain[
                (chain["strike"] == atm) & (chain["option_type"] == option_type)
            ]
            if candidates.empty:
                continue
            selected = candidates.iloc[0]
            entry_premium = float(selected["close"])
            lot_value = selected.get("lot_size")
            if not np.isfinite(entry_premium) or entry_premium <= 0:
                continue
            if pd.isna(lot_value) or float(lot_value) <= 0:
                continue
            lot_size = int(lot_value)
            symbol = str(selected["symbol"])
            exits = exit_chain[
                (exit_chain["strike"] == atm)
                & (exit_chain["option_type"] == option_type)
                & (exit_chain["symbol"] == symbol)
            ]
            if exits.empty:
                continue
            exit_premium = float(exits.iloc[-1]["close"])
            if not np.isfinite(exit_premium) or exit_premium < 0:
                continue
            entry_adjustment = cfg.entry_slippage + cfg.spread_points / 2 + cfg.market_impact_points
            adjusted_entry = entry_premium + entry_adjustment
            entry_turnover = adjusted_entry * lot_size
            entry_fees = (
                cfg.brokerage_per_lot + cfg.other_charges
                + (cfg.brokerage_pct + cfg.exchange_charges_pct) * entry_turnover
                + cfg.gst_pct * (
                    cfg.brokerage_per_lot + cfg.other_charges
                    + (cfg.brokerage_pct + cfg.exchange_charges_pct) * entry_turnover
                )
                + cfg.stamp_duty_pct * entry_turnover
            )
            if entry_turnover + entry_fees > capital:
                continue
            charges = compute_transaction_costs(
                entry_premium, exit_premium, lot_size, 1, cfg
            )
            gross = (exit_premium - entry_premium) * lot_size
            net = gross - float(charges["total_cost"])
            trades.append({
                "strategy": strategy,
                "signal_date": signal_day,
                "entry_date": entry_day,
                "exit_date": exit_day,
                "expiry_date": expiry,
                "option_type": option_type,
                "spot_reference": spot_reference,
                "strike": float(atm),
                "symbol": symbol,
                "entry_premium": entry_premium,
                "exit_premium": exit_premium,
                "lot_size": lot_size,
                "lots": 1,
                "quantity": lot_size,
                "gross_pnl": round(gross, 2),
                "estimated_costs": charges["total_cost"],
                "net_pnl": round(net, 2),
                "return_pct": round(net / max(entry_premium * lot_size, 1e-9) * 100, 4),
                "signal_ret1": signal.get("ret1"),
                "signal_ret3": signal.get("ret3"),
                "signal_ret5": signal.get("ret5"),
                "signal_ret10": signal.get("ret10"),
                "signal_vol20": signal.get("vol20"),
                "signal_rsi14": signal.get("rsi14"),
                "signal_adx14": signal.get("adx14"),
                "signal_range_ratio": signal.get("range_ratio"),
            })

    ledger = pd.DataFrame(trades)
    if ledger.empty:
        return empty, empty, empty, {
            "status": "no_trades",
            "candidate_rules": len(specs),
            "note": "No complete one-lot ATM option round trips were found for any candidate.",
        }

    ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])
    ledger["exit_date"] = pd.to_datetime(ledger["exit_date"])
    ledger["split"] = np.where(ledger["entry_date"] <= train_end, "train", "holdout")
    ledger["month"] = ledger["exit_date"].dt.strftime("%Y-%m")

    monthly_buckets: dict[tuple[str, str], dict] = defaultdict(_trade_bucket)
    split_buckets: dict[tuple[str, str, str], dict] = defaultdict(_trade_bucket)
    for row in ledger.to_dict("records"):
        strategy = str(row["strategy"])
        month = str(row["month"])
        _accumulate(monthly_buckets[(strategy, month)], float(row["net_pnl"]))
        _accumulate(
            split_buckets[(strategy, str(row["split"]), month)],
            float(row["net_pnl"]),
        )

    monthly_rows = [
        _monthly_row(strategy, month, bucket)
        for (strategy, month), bucket in monthly_buckets.items()
    ]
    monthly = pd.DataFrame(monthly_rows).sort_values(
        ["strategy", "month"]
    ).reset_index(drop=True)

    leaderboard_rows = []
    for strategy in sorted(ledger["strategy"].unique()):
        train = _period_score(strategy, "train", split_buckets)
        holdout = _period_score(strategy, "holdout", split_buckets)
        pnl = pd.to_numeric(
            ledger.loc[ledger["strategy"] == strategy, "net_pnl"], errors="coerce"
        )
        median = float(pnl.median()) if len(pnl) else None
        row = {
            "strategy": strategy,
            **train,
            **holdout,
            "full_trades": int(len(pnl)),
            "full_net_pnl": round(float(pnl.sum()), 2),
            "full_median_trade_pnl": round(median, 2) if median is not None else None,
        }
        row["strict_monthly_target_pass"] = bool(
            train["train_trades"] >= 35
            and train["train_active_months"] >= 6
            and train["train_all_active_months_pass"]
            and train["train_aggregate_pf"] is not None
            and train["train_aggregate_pf"] >= MONTHLY_TARGET_PF
            and holdout["holdout_trades"] >= 20
            and holdout["holdout_active_months"] >= 4
            and holdout["holdout_all_active_months_pass"]
            and holdout["holdout_aggregate_pf"] is not None
            and holdout["holdout_aggregate_pf"] >= MONTHLY_TARGET_PF
        )
        row["near_target_development_pass"] = bool(
            train["train_trades"] >= 35
            and train["train_active_months"] >= 6
            and (train["train_target_hit_pct"] or 0) >= 70
            and train["train_aggregate_pf"] is not None
            and train["train_aggregate_pf"] >= MONTHLY_TARGET_PF
            and train["train_net_pnl"] > 0
        )
        row["holdout_validation_pass"] = bool(
            row["near_target_development_pass"]
            and holdout["holdout_trades"] >= 20
            and holdout["holdout_active_months"] >= 4
            and (holdout["holdout_target_hit_pct"] or 0) >= 70
            and holdout["holdout_aggregate_pf"] is not None
            and holdout["holdout_aggregate_pf"] >= MONTHLY_TARGET_PF
            and holdout["holdout_net_pnl"] > 0
        )
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["train_target_hit_pct", "train_aggregate_pf", "train_net_pnl"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.insert(0, "training_rank", range(1, len(leaderboard) + 1))

    strict = leaderboard[leaderboard["strict_monthly_target_pass"]]
    near = leaderboard[leaderboard["holdout_validation_pass"]]
    top = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    report = {
        "status": (
            "strict_candidates_found" if len(strict)
            else "out_of_sample_candidates_found" if len(near)
            else "no_candidate_met_monthly_target"
        ),
        "target_definition": "Monthly gross profit / gross loss >= 1.20 after estimated costs, with at least three trades in that month.",
        "target_profit_factor": MONTHLY_TARGET_PF,
        "minimum_trades_per_active_month": MIN_MONTHLY_TRADES,
        "candidate_rules_tested": int(len(specs)),
        "trade_count": int(len(ledger)),
        "train_start": str(pd.Timestamp(entry_dates[0]).date()),
        "train_end": str(train_end.date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(pd.Timestamp(entry_dates[-1]).date()),
        "training_ranking_method": "Sorted only by training monthly-target hit rate, then training aggregate PF and net P&L. Holdout is not used for ranking.",
        "strict_candidate_count": int(len(strict)),
        "holdout_validation_candidate_count": int(len(near)),
        "strict_candidates": strict["strategy"].astype(str).tolist(),
        "holdout_validation_candidates": near["strategy"].astype(str).tolist(),
        "top_training_ranked_strategy": {
            key: top.get(key) for key in (
                "strategy", "train_trades", "train_net_pnl", "train_aggregate_pf",
                "train_target_hit_pct", "holdout_trades", "holdout_net_pnl",
                "holdout_aggregate_pf", "holdout_target_hit_pct",
                "strict_monthly_target_pass", "holdout_validation_pass",
            )
        } if top else None,
        "execution": "One historical lot; previous-session indicators; entry-session EOD option close; next-session EOD option close; estimated transaction costs reapplied.",
        "warning": (
            "This is a finite predeclared rule search, not a guarantee of future profitability. "
            "The final holdout is a single historical validation window and can be reused only as research after inspection. "
            "EOD bhavcopy marks are not guaranteed executable fills. No rule is promoted unless it passes the declared screens."
        ),
    }
    return ledger, monthly, leaderboard, report
