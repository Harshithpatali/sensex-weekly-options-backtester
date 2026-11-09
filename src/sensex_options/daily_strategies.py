from __future__ import annotations

"""Daily close-to-close long-option research; uses observed BSE marks only."""

import numpy as np
import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs


DAILY_STRATEGIES = {
    "DAILY_ATM_CALL": "Buy ATM Call every eligible session",
    "DAILY_ATM_PUT": "Buy ATM Put every eligible session",
    "DAILY_MOMENTUM": "Buy Call after prior up day; Put after prior down day",
    "DAILY_MEAN_REVERSION": "Buy Put after prior up day; Call after prior down day",
    "DAILY_5D_MOMENTUM": "Buy CE or PE using the prior five-session SENSEX return",
    "DAILY_EMA_TREND": "Buy CE when prior EMA-5 is above EMA-20; otherwise PE",
    "DAILY_RSI_REVERSAL": "Buy CE when prior RSI-14 is below 30; PE when above 70",
}


def run_daily_strategy_suite(
    index: pd.DataFrame,
    options: pd.DataFrame,
    capital: float | None = None,
    cost: CostConfig | None = None,
    min_volume: float = 25.0,
    min_open_interest: float = 50.0,
    max_atm_distance_pct: float = 0.0075,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replay one-session holds using only prior-session direction and observed quotes.

    Entry is the current session EOD close and exit is the next index session EOD
    close. The directional signal is computed from the two index closes before
    entry. Strike selection uses the previous index close. Each strategy has an
    isolated sequential capital curve. This is not an intraday or live-fill model.
    """
    if index.empty or options.empty:
        return pd.DataFrame(), pd.DataFrame(), {"error": "Index or options dataset is empty"}
    cost = cost or CostConfig(
        brokerage_per_lot=20.0, stt_pct=0.001, exchange_charges_pct=0.0003,
        gst_pct=0.18, stamp_duty_pct=0.00003, entry_slippage=0.5,
        exit_slippage=0.5, spread_points=1.0, market_impact_points=0.25,
    )

    idx = index.copy()
    idx["date"] = pd.to_datetime(idx["date"], errors="coerce").dt.normalize()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    idx["ema5"] = idx["close"].ewm(span=5, adjust=False).mean()
    idx["ema20"] = idx["close"].ewm(span=20, adjust=False).mean()
    delta = idx["close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = avg_gain / avg_loss.replace(0, float("nan"))
    idx["rsi14"] = 100 - (100 / (1 + relative_strength))
    idx.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100.0
    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "close", "lot_size", "volume", "open_interest"):
        if col not in opt:
            opt[col] = float("nan")
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    if "symbol" not in opt:
        opt["symbol"] = ""
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[(opt["close"] >= 0) & (opt["strike"] > 0) & opt["option_type"].isin(["CE", "PE"])]
    opt = opt.sort_values(["trade_date", "expiry_date", "strike", "symbol"])
    by_day = {day: frame for day, frame in opt.groupby("trade_date", sort=False)}
    # Unit-lot research sizing: one lot per eligible session, independent of account equity.
    trades: list[dict] = []
    sessions = idx.reset_index(drop=True)

    # i is entry session; signal uses closes at i-2 and i-1, exit is i+1.
    for i in range(2, len(sessions) - 1):
        signal_row = sessions.iloc[i - 1]
        prior_row = sessions.iloc[i - 2]
        entry_row = sessions.iloc[i]
        exit_row = sessions.iloc[i + 1]
        entry_day, exit_day = entry_row["date"], exit_row["date"]
        reference_spot = float(signal_row["close"])
        prior_return = reference_spot / float(prior_row["close"]) - 1.0
        prior_5d_return = (
            reference_spot / float(sessions.iloc[i - 6]["close"]) - 1.0
            if i >= 6 else None
        )
        prior_features = sessions.iloc[i - 1]
        rsi14 = prior_features["rsi14"]
        ema_direction = (
            "CE" if prior_features["ema5"] > prior_features["ema20"]
            else "PE" if prior_features["ema5"] < prior_features["ema20"] else None
        )
        entry_options = by_day.get(entry_day)
        exit_options = by_day.get(exit_day)
        if entry_options is None or exit_options is None:
            continue
        eligible_expiries = sorted(
            entry_options.loc[entry_options["expiry_date"] >= exit_day, "expiry_date"].dropna().unique()
        )
        if not eligible_expiries:
            continue
        expiry = pd.Timestamp(eligible_expiries[0])
        chain = entry_options[entry_options["expiry_date"] == expiry]
        strikes = sorted(chain["strike"].dropna().unique())
        if not strikes:
            continue
        atm = min(strikes, key=lambda value: abs(float(value) - reference_spot))
        directions = {
            "DAILY_ATM_CALL": "CE",
            "DAILY_ATM_PUT": "PE",
            "DAILY_MOMENTUM": "CE" if prior_return > 0 else "PE" if prior_return < 0 else None,
            "DAILY_MEAN_REVERSION": "PE" if prior_return > 0 else "CE" if prior_return < 0 else None,
            "DAILY_5D_MOMENTUM": (
                "CE" if prior_5d_return is not None and prior_5d_return > 0
                else "PE" if prior_5d_return is not None and prior_5d_return < 0 else None
            ),
            "DAILY_EMA_TREND": ema_direction if i >= 20 else None,
            "DAILY_RSI_REVERSAL": (
                "CE" if pd.notna(rsi14) and rsi14 < 30
                else "PE" if pd.notna(rsi14) and rsi14 > 70 else None
            ),
        }
        for strategy, option_type in directions.items():
            if option_type is None:
                continue
            entry_candidates = chain[chain["option_type"] == option_type].copy()
            entry_candidates["atm_distance_pct"] = (
                (entry_candidates["strike"] - reference_spot).abs() / reference_spot
            )
            # Daily volume/OI are observable in bhavcopy; bid/ask quotes are not.
            entry_candidates = entry_candidates[
                (entry_candidates["close"] > 0)
                & (entry_candidates["lot_size"] > 0)
                & (entry_candidates["volume"] >= min_volume)
                & (entry_candidates["open_interest"] >= min_open_interest)
                & (entry_candidates["atm_distance_pct"] <= max_atm_distance_pct)
            ]
            if entry_candidates.empty:
                continue
            selected = entry_candidates.sort_values(
                ["atm_distance_pct", "volume", "open_interest"],
                ascending=[True, False, False],
            ).iloc[0]
            atm = float(selected["strike"])
            entry_premium = float(selected["close"])
            lot_value = selected.get("lot_size")
            if not pd.notna(lot_value) or float(lot_value) <= 0 or entry_premium <= 0:
                continue
            lot_size = int(lot_value)
            symbol = selected["symbol"]
            exit_candidates = exit_options[
                (exit_options["expiry_date"] == expiry)
                & (exit_options["strike"] == atm)
                & (exit_options["option_type"] == option_type)
                & (exit_options["symbol"] == symbol)
            ]
            if exit_candidates.empty:
                continue
            exit_mark = exit_candidates.iloc[-1]
            exit_premium = float(exit_mark["close"])
            if exit_premium > 0:
                if (
                    pd.isna(exit_mark["volume"]) or float(exit_mark["volume"]) < min_volume
                    or pd.isna(exit_mark["open_interest"]) or float(exit_mark["open_interest"]) < min_open_interest
                ):
                    continue
            elif exit_day != expiry:
                # Zero is a valid expiry payoff, not a tradable pre-expiry mark.
                continue
            entry_adjustment = cost.entry_slippage + cost.spread_points / 2 + cost.market_impact_points
            adjusted_entry = entry_premium + entry_adjustment
            entry_turnover = adjusted_entry * lot_size
            entry_fees = (
                cost.brokerage_per_lot
                + cost.other_charges
                + (cost.brokerage_pct + cost.exchange_charges_pct) * entry_turnover
                + cost.gst_pct * (cost.brokerage_per_lot + cost.other_charges
                    + (cost.brokerage_pct + cost.exchange_charges_pct) * entry_turnover)
                + cost.stamp_duty_pct * entry_turnover
            )
            if capital is not None and entry_turnover + entry_fees > float(capital):
                continue
            n_lots = 1
            qty = lot_size
            gross = (exit_premium - entry_premium) * qty
            charges = compute_transaction_costs(
                entry_premium, exit_premium, lot_size, n_lots, cost
            )
            net = gross - charges["total_cost"]
            trades.append({
                "strategy": strategy, "strategy_description": DAILY_STRATEGIES[strategy],
                "entry_date": entry_day, "exit_date": exit_day, "expiry_date": expiry,
                "signal_return_pct": round(prior_return * 100, 5),
                "signal_direction": "UP" if prior_return > 0 else "DOWN" if prior_return < 0 else "FLAT",
                "option_type": option_type, "spot_reference": reference_spot,
                "strike": float(atm), "symbol": symbol, "entry_premium": entry_premium,
                "entry_volume": float(selected["volume"]),
                "entry_open_interest": float(selected["open_interest"]),
                "atm_distance_pct": float(selected["atm_distance_pct"]),
                "exit_premium": exit_premium, "lot_size": lot_size, "lots": n_lots,
                "quantity": qty, "starting_capital": None,
                "ending_capital": None, "gross_pnl": round(gross, 2),
                "estimated_costs": charges["total_cost"], "brokerage": charges["brokerage"],
                "stt": charges["stt"], "exchange_charges": charges["exchange"],
                "gst": charges["gst"], "stamp_duty": charges["stamp_duty"],
                "slippage_cost": charges["slippage"], "spread_cost": charges["spread"],
                "market_impact_cost": charges["market_impact"], "net_pnl": round(net, 2),
                "return_pct": round(net / max(entry_premium * qty, 1e-9) * 100, 4),
                "status": "completed",
            })

    ledger = pd.DataFrame(trades)
    if ledger.empty:
        return ledger, pd.DataFrame(), {"error": "No daily round trips had complete next-session quotes and verified lot sizes."}
    rows = []
    for name, group in ledger.groupby("strategy"):
        pnl = group["net_pnl"].astype(float)
        gains, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        ordered = group.sort_values("exit_date").copy()
        unit_pnl_curve = pnl.reset_index(drop=True).cumsum()
        pnl_peak = unit_pnl_curve.cummax().clip(lower=0)
        drawdown = unit_pnl_curve - pnl_peak
        all_entry_dates = pd.Index(sorted(pd.to_datetime(ledger["entry_date"]).dropna().unique()))
        trade_returns = (
            ordered.assign(entry_date=pd.to_datetime(ordered["entry_date"]))
            .set_index("entry_date")["return_pct"].astype(float) / 100.0
        )
        daily_returns = trade_returns.reindex(all_entry_dates, fill_value=0.0)
        return_index = 100.0 + daily_returns.cumsum() * 100.0
        return_peak = return_index.cummax()
        return_dd_pct = (return_index / return_peak.replace(0, np.nan) - 1.0) * 100.0
        return_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else np.nan
        sharpe = float(daily_returns.mean() / return_std * np.sqrt(252)) if pd.notna(return_std) and return_std > 0 else None
        positive = pnl[pnl > 0]
        negative = pnl[pnl < 0]
        rows.append({
            "strategy": name, "description": DAILY_STRATEGIES[name], "trades": len(group),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(gains / losses, 4) if losses else None,
            "expectancy": round(float(pnl.mean()), 2),
            "median_trade_pnl": round(float(pnl.median()), 2),
            "avg_win": round(float(positive.mean()), 2) if len(positive) else None,
            "avg_loss": round(float(negative.mean()), 2) if len(negative) else None,
            "best_trade": round(float(pnl.max()), 2),
            "worst_trade": round(float(pnl.min()), 2),
            "top3_wins_share_of_net_pct": round(float(positive.nlargest(3).sum() / pnl.sum() * 100), 2) if pnl.sum() > 0 and len(positive) else None,
            "avg_return_pct": round(float(group["return_pct"].mean()), 4),
            "max_drawdown": round(float(drawdown.min()), 2),
            "max_drawdown_pct": round(float(return_dd_pct.min()), 2) if len(return_dd_pct.dropna()) else None,
            "sharpe_trade_annualized": round(sharpe, 4) if sharpe is not None else None,
            "drawdown_basis": "unit-lot cumulative P&L; percentage drawdown from additive premium-return index (base 100)",
            "sample_warning": "Small sample: fewer than 30 daily trades" if len(group) < 30 else "",
        })
    ranking = pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)
    meta = {
        "holding_period": "one SENSEX trading session (entry EOD close to next-session EOD close)",
        "signal_timing": "prior-session return; no same-day signal look-ahead",
        "strike_selection": "liquid contract within 0.75% of prior-session ATM; nearest strike first, then highest volume/OI",
        "liquidity_filter": {
            "minimum_daily_volume": min_volume,
            "minimum_open_interest": min_open_interest,
            "missing_volume_or_oi": "rejected",
            "bid_ask_data": "not available in daily BSE bhavcopy; spread stress assumptions are not observed quotes",
        },
        "expiry_selection": "nearest available expiry not earlier than exit session",
        "position_sizing": "one historical lot per eligible session; fixed unit-lot research, not compounded sizing",
        "capital": "No affordability filter in default research mode; when supplied, capital is used only as a one-lot affordability filter. Each strategy has its own cumulative unit-lot P&L curve",
        "return_basis": "one-lot option-premium returns, additive index base 100; not a capital-weighted account return",
        "execution": "EOD close marks with configurable estimated costs; not guaranteed fills",
        "initial_capital_per_strategy": capital,
        "cost_assumptions": cost.__dict__,
    }
    return ledger, ranking, meta



def daily_holdout_evaluation(ledger: pd.DataFrame) -> dict:
    """Chronological 70/30 daily-session holdout; no strategy fitting or tuning."""
    if ledger.empty or "entry_date" not in ledger:
        return {"status": "insufficient_data", "note": "No daily trades available."}
    frame = ledger.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame = frame.dropna(subset=["entry_date"])
    dates = sorted(frame["entry_date"].drop_duplicates())
    if len(dates) < 10:
        return {
            "status": "insufficient_data",
            "note": "At least 10 distinct entry sessions are required for a chronological holdout.",
            "distinct_entry_sessions": len(dates),
        }
    split = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    holdout_dates = set(dates[split:])
    test = frame[frame["entry_date"].isin(holdout_dates)]
    metrics = []
    for strategy, group in test.groupby("strategy"):
        pnl = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
        gains, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        positive = pnl[pnl > 0]
        negative = pnl[pnl < 0]
        ordered = group.sort_values("entry_date").copy()
        all_dates = pd.Index(sorted(pd.to_datetime(test["entry_date"]).dropna().unique()))
        if "return_pct" in ordered:
            trade_returns = (
                ordered.assign(entry_date=pd.to_datetime(ordered["entry_date"]))
                .set_index("entry_date")["return_pct"].astype(float) / 100.0
            )
        else:
            trade_returns = (
                ordered.assign(entry_date=pd.to_datetime(ordered["entry_date"]))
                .set_index("entry_date")["net_pnl"].astype(float) / 10000.0
            )
        daily_returns = trade_returns.reindex(all_dates, fill_value=0.0)
        return_index = 100.0 + daily_returns.cumsum() * 100.0
        return_peak = return_index.cummax()
        return_dd_pct = (return_index / return_peak.replace(0, np.nan) - 1.0) * 100.0
        return_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else np.nan
        sharpe = float(daily_returns.mean() / return_std * np.sqrt(252)) if pd.notna(return_std) and return_std > 0 else None
        pnl_curve = pnl.cumsum()
        pnl_drawdown = pnl_curve - pnl_curve.cummax().clip(lower=0)
        metrics.append({
            "strategy": strategy, "trades": int(len(pnl)),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else None,
            "profit_factor": round(gains / losses, 4) if losses else None,
            "expectancy": round(float(pnl.mean()), 2) if len(pnl) else None,
            "median_trade_pnl": round(float(pnl.median()), 2) if len(pnl) else None,
            "best_trade": round(float(pnl.max()), 2) if len(pnl) else None,
            "worst_trade": round(float(pnl.min()), 2) if len(pnl) else None,
            "max_drawdown_pnl": round(float(pnl_drawdown.min()), 2) if len(pnl_drawdown) else 0.0,
            "max_drawdown_return_pct": round(float(return_dd_pct.min()), 2) if len(return_dd_pct.dropna()) else None,
            "sharpe_trade_annualized": round(sharpe, 4) if sharpe is not None else None,
            "top3_wins_share_of_net_pct": round(float(positive.nlargest(3).sum() / pnl.sum() * 100), 2) if pnl.sum() > 0 and len(positive) else None,
        })
    return {
        "status": "ok", "method": "chronological 70/30 split by entry session",
        "train_start": str(dates[0].date()), "train_end": str(dates[split - 1].date()),
        "holdout_start": str(dates[split].date()), "holdout_end": str(dates[-1].date()),
        "distinct_entry_sessions": len(dates), "holdout_sessions": len(holdout_dates),
        "holdout_metrics": metrics,
        "warning": "This is a signal-level unit-lot candidate comparison, NOT a ₹10,000 capital-constrained portfolio. Use daily_capital_controlled_leaderboard.csv for cash/risk-feasible results. It is not a guarantee of future performance.",
    }
