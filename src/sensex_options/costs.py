"""
Transaction cost model.

Rates are configurable because historical brokerage/STT/stamp schedules change.
Default research setting is zero costs; README warns that zero-cost returns are unrealistic.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from sensex_options.config import CostConfig


def compute_transaction_costs(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    n_lots: int,
    cost: CostConfig,
    n_legs: int = 1,
) -> Dict[str, float]:
    """
    Compute total transaction costs for a round-trip.

    Premium turnover per lot (entry) = entry_premium * lot_size
    Premium turnover per lot (exit)  = exit_premium * lot_size

    Returns dict with breakdown and total.
    """
    qty = lot_size * n_lots
    entry_turn = entry_premium * qty
    exit_turn = exit_premium * qty

    n_legs = max(1, int(n_legs))
    brokerage = cost.brokerage_per_lot * n_lots * 2 * n_legs  # entry + exit for each leg
    brokerage += cost.brokerage_pct * (entry_turn + exit_turn)

    # STT typically on sell side of options (premium)
    stt = cost.stt_pct * exit_turn

    exchange = cost.exchange_charges_pct * (entry_turn + exit_turn)
    gst = cost.gst_pct * (brokerage + exchange)
    stamp = cost.stamp_duty_pct * entry_turn
    other = cost.other_charges * n_lots * 2 * n_legs

    # Slippage applied to P&L separately (price adjustment)
    slippage_cost = (
        cost.entry_slippage + cost.exit_slippage + cost.spread_points
        + 2 * cost.market_impact_points
    ) * qty * n_legs

    total = brokerage + stt + exchange + gst + stamp + other + slippage_cost

    return {
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange": round(exchange, 4),
        "gst": round(gst, 4),
        "stamp_duty": round(stamp, 4),
        "other": round(other, 4),
        "slippage": round(slippage_cost, 4),
        "total_cost": round(total, 4),
    }


def apply_slippage_to_premiums(
    entry_premium: float,
    exit_premium: float,
    cost: CostConfig,
) -> tuple[float, float]:
    """Return adjusted entry (higher) and exit (lower) premiums."""
    entry_adjustment = cost.entry_slippage + cost.spread_points / 2 + cost.market_impact_points
    exit_adjustment = cost.exit_slippage + cost.spread_points / 2 + cost.market_impact_points
    adj_entry = entry_premium + entry_adjustment
    adj_exit = max(0.0, exit_premium - exit_adjustment)
    return adj_entry, adj_exit
