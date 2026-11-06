"""
Backtesting engine for long SENSEX weekly Call strategy.

Strategy (exact):
  For each historical expiry:
    1. Previous SENSEX session before expiry → reference close.
    2. Among Call contracts for that expiry available on the previous session,
       select smallest strike STRICTLY greater than reference close.
    3. Entry premium = that contract's previous-session close.
    4. Exit premium  = same contract's close on expiry session.
    5. Size whole lots only, never exceed capital.
    6. Skip if missing premium, missing lot size, or insufficient capital.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sensex_options.calendar import available_expiries, previous_trading_session, sessions_on_or_before
from sensex_options.config import BacktestConfig, CostConfig
from sensex_options.costs import apply_slippage_to_premiums, compute_transaction_costs
from sensex_options.sources.contracts import resolve_lot_size

logger = logging.getLogger(__name__)


def select_strike(
    calls_on_entry: pd.DataFrame,
    reference_close: float,
) -> Optional[pd.Series]:
    """
    Select the Call with the smallest strike strictly greater than reference_close.
    Returns the row or None.
    """
    if calls_on_entry.empty or reference_close is None or np.isnan(reference_close):
        return None
    eligible = calls_on_entry[calls_on_entry["strike"] > reference_close].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values("strike")
    return eligible.iloc[0]


def run_backtest(
    index: pd.DataFrame,
    options: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, Any]:
    """
    Execute the strategy over available expiries.

    Returns dict with keys:
      trades (DataFrame), equity_curve (DataFrame), summary (dict)
    """
    config = config or BacktestConfig()
    capital = float(config.initial_capital)
    start_cap = capital

    if index.empty or options.empty:
        return {
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "summary": {"error": "missing_index_or_options_data", "starting_capital": start_cap},
        }

    index = index.copy()
    index["date"] = pd.to_datetime(index["date"]).dt.normalize()
    options = options.copy()
    options["trade_date"] = pd.to_datetime(options["trade_date"]).dt.normalize()
    options["expiry_date"] = pd.to_datetime(options["expiry_date"]).dt.normalize()

    # Optional date filter on expiries
    expiries = available_expiries(options)
    if config.start_date:
        expiries = expiries[expiries >= pd.Timestamp(config.start_date)]
    if config.end_date:
        expiries = expiries[expiries <= pd.Timestamp(config.end_date)]

    trades: List[dict] = []
    equity_points: List[dict] = [{"date": None, "capital": capital}]

    for expiry in expiries:
        entry_date = previous_trading_session(index, expiry)
        if entry_date is None:
            trades.append(_skip(expiry, None, None, "no_previous_trading_session", capital))
            continue

        exit_date = sessions_on_or_before(index, expiry)
        if exit_date is None:
            trades.append(_skip(expiry, entry_date, None, "no_exit_session", capital))
            continue

        # SENSEX close on entry session
        idx_row = index[index["date"] == entry_date]
        if idx_row.empty:
            trades.append(_skip(expiry, entry_date, exit_date, "missing_index_close", capital))
            continue
        ref_close = float(idx_row.iloc[0]["close"])

        # Calls for this expiry on entry date only (no look-ahead)
        calls = options[
            (options["expiry_date"] == expiry)
            & (options["trade_date"] == entry_date)
            & (options["option_type"] == "CE")
            & (options["strike"].notna())
        ]
        if calls.empty:
            trades.append(_skip(expiry, entry_date, exit_date, "no_call_contracts_on_entry", capital, ref_close))
            continue

        selected = select_strike(calls, ref_close)
        if selected is None:
            trades.append(_skip(expiry, entry_date, exit_date, "no_strike_above_spot", capital, ref_close))
            continue

        strike = float(selected["strike"])
        entry_prem = selected.get("close")
        if entry_prem is None or (isinstance(entry_prem, float) and np.isnan(entry_prem)) or entry_prem <= 0:
            trades.append(_skip(
                expiry, entry_date, exit_date, "missing_or_invalid_entry_premium",
                capital, ref_close, strike=strike,
            ))
            continue
        entry_prem = float(entry_prem)

        # Exit premium: same contract on exit_date
        exit_rows = options[
            (options["expiry_date"] == expiry)
            & (options["trade_date"] == exit_date)
            & (options["option_type"] == "CE")
            & (options["strike"] == strike)
        ]
        if exit_rows.empty:
            trades.append(_skip(
                expiry, entry_date, exit_date, "missing_exit_premium",
                capital, ref_close, strike=strike, entry_prem=entry_prem,
            ))
            continue
        exit_prem = exit_rows.iloc[0].get("close")
        if exit_prem is None or (isinstance(exit_prem, float) and np.isnan(exit_prem)):
            trades.append(_skip(
                expiry, entry_date, exit_date, "invalid_exit_premium",
                capital, ref_close, strike=strike, entry_prem=entry_prem,
            ))
            continue
        exit_prem = float(exit_prem)

        lot_size = resolve_lot_size(selected)
        if lot_size is None:
            trades.append(_skip(
                expiry, entry_date, exit_date, "missing_historical_lot_size",
                capital, ref_close, strike=strike, entry_prem=entry_prem, exit_prem=exit_prem,
            ))
            continue

        # Apply slippage to premiums for costed prices
        adj_entry, adj_exit = apply_slippage_to_premiums(entry_prem, exit_prem, config.cost)
        cost_per_lot = adj_entry * lot_size
        if cost_per_lot <= 0:
            trades.append(_skip(
                expiry, entry_date, exit_date, "non_positive_cost_per_lot",
                capital, ref_close, strike=strike, entry_prem=entry_prem, exit_prem=exit_prem,
                lot_size=lot_size,
            ))
            continue

        max_by_capital = int(capital // cost_per_lot)
        if config.max_lots is not None:
            max_by_capital = min(max_by_capital, config.max_lots)
        n_lots = max_by_capital
        if n_lots < 1:
            trades.append(_skip(
                expiry, entry_date, exit_date, "insufficient_capital",
                capital, ref_close, strike=strike, entry_prem=entry_prem, exit_prem=exit_prem,
                lot_size=lot_size,
            ))
            continue

        qty = n_lots * lot_size
        entry_investment = adj_entry * qty
        gross_pnl = (adj_exit - adj_entry) * qty
        cost_break = compute_transaction_costs(
            adj_entry, adj_exit, lot_size, n_lots, config.cost,
        )
        # Note: slippage already embedded in adj premiums; cost module still reports it
        net_pnl = gross_pnl - (cost_break["total_cost"] - cost_break["slippage"])
        # because slippage is already in gross via adj prices; avoid double-count
        # Simpler: recompute costs without slippage component for net
        cost_wo_slip = cost_break["total_cost"] - cost_break["slippage"]
        net_pnl = gross_pnl - cost_wo_slip

        ending_capital = capital + net_pnl
        if ending_capital < 0:
            # Should not happen with proper sizing, but guard
            ending_capital = 0.0
            net_pnl = -capital

        symbol = str(selected.get("symbol", f"SENSEX{strike}CE"))
        result = "win" if net_pnl > 0 else ("loss" if net_pnl < 0 else "flat")

        trade = {
            "expiry_date": expiry,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "sensex_ref": ref_close,
            "strike": strike,
            "option_symbol": symbol,
            "entry_premium": entry_prem,
            "exit_premium": exit_prem,
            "adj_entry_premium": adj_entry,
            "adj_exit_premium": adj_exit,
            "lot_size": lot_size,
            "n_lots": n_lots,
            "quantity": qty,
            "entry_investment": round(entry_investment, 2),
            "gross_pnl": round(gross_pnl, 2),
            "transaction_costs": round(cost_wo_slip, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(100.0 * net_pnl / entry_investment, 4) if entry_investment else 0.0,
            "starting_capital": round(capital, 2),
            "ending_capital": round(ending_capital, 2),
            "trade_result": result,
            "skip_reason": "",
            "status": "completed",
        }
        trades.append(trade)
        capital = ending_capital
        equity_points.append({"date": exit_date, "capital": capital})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)
    summary = compute_summary(trades_df, start_cap, capital, config)
    return {"trades": trades_df, "equity_curve": equity_df, "summary": summary}


def _skip(
    expiry,
    entry_date,
    exit_date,
    reason: str,
    capital: float,
    sensex_ref: float = None,
    strike: float = None,
    entry_prem: float = None,
    exit_prem: float = None,
    lot_size: int = None,
) -> dict:
    return {
        "expiry_date": expiry,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "sensex_ref": sensex_ref,
        "strike": strike,
        "option_symbol": None,
        "entry_premium": entry_prem,
        "exit_premium": exit_prem,
        "adj_entry_premium": None,
        "adj_exit_premium": None,
        "lot_size": lot_size,
        "n_lots": 0,
        "quantity": 0,
        "entry_investment": 0.0,
        "gross_pnl": 0.0,
        "transaction_costs": 0.0,
        "net_pnl": 0.0,
        "return_pct": 0.0,
        "starting_capital": round(capital, 2),
        "ending_capital": round(capital, 2),
        "trade_result": "skipped",
        "skip_reason": reason,
        "status": "skipped",
    }


def compute_summary(
    trades: pd.DataFrame,
    start_cap: float,
    end_cap: float,
    config: BacktestConfig,
) -> dict:
    if trades.empty:
        return {
            "starting_capital": start_cap,
            "ending_capital": end_cap,
            "total_trades": 0,
            "completed_trades": 0,
            "skipped_trades": 0,
            "note": "No trades — ensure options data is collected and validated.",
        }

    completed = trades[trades["status"] == "completed"]
    skipped = trades[trades["status"] == "skipped"]
    wins = completed[completed["net_pnl"] > 0]
    losses = completed[completed["net_pnl"] < 0]

    total_gross = float(completed["gross_pnl"].sum()) if len(completed) else 0.0
    total_net = float(completed["net_pnl"].sum()) if len(completed) else 0.0
    avg_win = float(wins["net_pnl"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["net_pnl"].mean()) if len(losses) else 0.0
    max_profit = float(completed["net_pnl"].max()) if len(completed) else 0.0
    max_loss = float(completed["net_pnl"].min()) if len(completed) else 0.0

    # Drawdown from equity
    if len(completed):
        equity = completed["ending_capital"].values
        peak = np.maximum.accumulate(np.concatenate([[start_cap], equity]))
        dd = peak[1:] - equity
        max_dd = float(dd.max()) if len(dd) else 0.0
        max_dd_pct = float((dd / peak[1:]).max() * 100) if len(dd) and peak[1:].min() > 0 else 0.0
    else:
        max_dd = 0.0
        max_dd_pct = 0.0

    gross_wins = float(wins["gross_pnl"].sum()) if len(wins) else 0.0
    gross_losses = abs(float(losses["gross_pnl"].sum())) if len(losses) else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)

    n_comp = len(completed)
    win_rate = 100.0 * len(wins) / n_comp if n_comp else 0.0
    expectancy = total_net / n_comp if n_comp else 0.0

    # Consecutive
    results = completed["trade_result"].tolist() if n_comp else []
    max_cw = max_cl = cur_w = cur_l = 0
    for r in results:
        if r == "win":
            cur_w += 1
            cur_l = 0
            max_cw = max(max_cw, cur_w)
        elif r == "loss":
            cur_l += 1
            cur_w = 0
            max_cl = max(max_cl, cur_l)
        else:
            cur_w = cur_l = 0

    return {
        "starting_capital": start_cap,
        "ending_capital": end_cap,
        "total_net_pnl": round(total_net, 2),
        "total_gross_pnl": round(total_gross, 2),
        "total_trades": len(trades),
        "completed_trades": n_comp,
        "skipped_trades": len(skipped),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "avg_profit_per_win": round(avg_win, 2),
        "avg_loss_per_loss": round(avg_loss, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "expectancy": round(expectancy, 2),
        "max_consecutive_wins": max_cw,
        "max_consecutive_losses": max_cl,
        "cost_assumptions": config.cost.label(),
        "zero_cost_default": config.assume_zero_costs,
    }
