from __future__ import annotations

"""Daily close-to-close long-option research; uses observed BSE marks only."""

import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs


DAILY_STRATEGIES = {
    "DAILY_ATM_CALL": "Buy ATM Call every eligible session",
    "DAILY_ATM_PUT": "Buy ATM Put every eligible session",
    "DAILY_MOMENTUM": "Buy Call after prior up day; Put after prior down day",
    "DAILY_MEAN_REVERSION": "Buy Put after prior up day; Call after prior down day",
}


def run_daily_strategy_suite(
    index: pd.DataFrame,
    options: pd.DataFrame,
    capital: float = 10000.0,
    cost: CostConfig | None = None,
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
    idx = idx.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "close", "lot_size"):
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    if "symbol" not in opt:
        opt["symbol"] = ""
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[(opt["close"] >= 0) & (opt["strike"] > 0) & opt["option_type"].isin(["CE", "PE"])]
    opt = opt.sort_values(["trade_date", "expiry_date", "strike", "symbol"])
    by_day = {day: frame for day, frame in opt.groupby("trade_date", sort=False)}
    capital_by_strategy = {name: float(capital) for name in DAILY_STRATEGIES}
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
        }
        for strategy, option_type in directions.items():
            if option_type is None:
                continue
            entry_candidates = chain[(chain["strike"] == atm) & (chain["option_type"] == option_type)]
            if entry_candidates.empty:
                continue
            selected = entry_candidates.iloc[0]
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
            exit_premium = float(exit_candidates.iloc[-1]["close"])
            available = capital_by_strategy[strategy]
            entry_adjustment = cost.entry_slippage + cost.spread_points / 2 + cost.market_impact_points
            exit_adjustment = cost.exit_slippage + cost.spread_points / 2 + cost.market_impact_points
            adjusted_entry = entry_premium + entry_adjustment
            adjusted_exit = max(0.0, exit_premium - exit_adjustment)
            entry_cost_per_lot = adjusted_entry * lot_size
            n_lots = int(available // max(entry_cost_per_lot, 1e-9))
            if n_lots < 1:
                continue
            qty = n_lots * lot_size
            gross = (exit_premium - entry_premium) * qty
            charges = compute_transaction_costs(
                entry_premium, exit_premium, lot_size, n_lots, cost
            )
            net = max(-available, gross - charges["total_cost"])
            ending = max(0.0, available + net)
            capital_by_strategy[strategy] = ending
            trades.append({
                "strategy": strategy, "strategy_description": DAILY_STRATEGIES[strategy],
                "entry_date": entry_day, "exit_date": exit_day, "expiry_date": expiry,
                "signal_return_pct": round(prior_return * 100, 5),
                "signal_direction": "UP" if prior_return > 0 else "DOWN" if prior_return < 0 else "FLAT",
                "option_type": option_type, "spot_reference": reference_spot,
                "strike": float(atm), "symbol": symbol, "entry_premium": entry_premium,
                "exit_premium": exit_premium, "lot_size": lot_size, "lots": n_lots,
                "quantity": qty, "starting_capital": round(available, 2),
                "ending_capital": round(ending, 2), "gross_pnl": round(gross, 2),
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
        rows.append({
            "strategy": name, "description": DAILY_STRATEGIES[name], "trades": len(group),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(gains / losses, 4) if losses else None,
            "expectancy": round(float(pnl.mean()), 2),
            "avg_return_pct": round(float(group["return_pct"].mean()), 4),
            "sample_warning": "Small sample: fewer than 30 daily trades" if len(group) < 30 else "",
        })
    ranking = pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)
    meta = {
        "holding_period": "one SENSEX trading session (entry EOD close to next-session EOD close)",
        "signal_timing": "prior-session return; no same-day signal look-ahead",
        "strike_selection": "nearest ATM using prior-session SENSEX close",
        "expiry_selection": "nearest available expiry not earlier than exit session",
        "capital": "isolated sequential capital curve per strategy; not a shared portfolio",
        "execution": "EOD close marks with configurable estimated costs; not guaranteed fills",
        "initial_capital_per_strategy": capital,
        "cost_assumptions": cost.__dict__,
    }
    return ledger, ranking, meta
