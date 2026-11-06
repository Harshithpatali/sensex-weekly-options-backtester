from __future__ import annotations

"""Cash- and risk-constrained replay for independent candidate strategy books.

This module converts unit-lot research candidates into a ₹10,000-account
feasibility ledger. Long options are treated as fully paid premium positions:
maximum loss is premium debit plus estimated entry/round-trip charges. No
fractional lots or leverage are permitted.
"""

import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs


def apply_capital_controls(
    ledger: pd.DataFrame,
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.02,
    daily_loss_limit_pct: float = 0.02,
    cost: CostConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replay each strategy as a separate cash account; reject unaffordable risk.

    Candidate strategies are compared as separate hypothetical ₹10k accounts.
    Only one one-lot position per strategy may be open at a time. The daily
    stop blocks new entries for that strategy on a day after its realized loss
    reaches the daily limit. No margin, fractional lots, or borrowed capital.
    """
    required = {"strategy", "entry_date", "exit_date", "entry_premium",
                "exit_premium", "lot_size", "lots", "quantity", "net_pnl"}
    if ledger is None or ledger.empty or not required.issubset(ledger.columns):
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "invalid_or_empty_ledger",
            "missing_columns": sorted(required - set(ledger.columns)) if ledger is not None else sorted(required),
        }
    if initial_capital <= 0 or not 0 < risk_per_trade_pct <= 1 or not 0 < daily_loss_limit_pct <= 1:
        raise ValueError("Capital and risk percentages must be positive and no greater than 100%.")

    cfg = cost or CostConfig(
        brokerage_per_lot=20.0, stt_pct=0.001, exchange_charges_pct=0.0003,
        gst_pct=0.18, stamp_duty_pct=0.00003, entry_slippage=0.5,
        exit_slippage=0.5, spread_points=1.0, market_impact_points=0.25,
    )
    rows: list[dict] = []
    summary: list[dict] = []
    for strategy, source in ledger.groupby("strategy", sort=True):
        source = source.copy()
        source["entry_date"] = pd.to_datetime(source["entry_date"], errors="coerce").dt.normalize()
        source["exit_date"] = pd.to_datetime(source["exit_date"], errors="coerce").dt.normalize()
        source = source.dropna(subset=["entry_date", "exit_date"]).sort_values(["entry_date", "exit_date"])
        equity = float(initial_capital)
        peak = equity
        last_exit = pd.Timestamp.min
        daily_realized: dict[pd.Timestamp, float] = {}
        accepted = 0
        rejected_risk = rejected_cash = rejected_overlap = rejected_daily_stop = 0
        for trade in source.to_dict("records"):
            entry = float(trade["entry_premium"])
            exit_premium = float(trade["exit_premium"])
            lot = int(trade["lot_size"])
            qty = int(trade["quantity"])
            lots = int(trade["lots"])
            day = pd.Timestamp(trade["entry_date"])
            if entry <= 0 or lot <= 0 or qty != lot * lots or lots != 1:
                rejected_risk += 1
                continue
            if day <= last_exit:
                rejected_overlap += 1
                continue
            day_pnl = daily_realized.get(day, 0.0)
            if day_pnl <= -initial_capital * daily_loss_limit_pct:
                rejected_daily_stop += 1
                continue
            # Long-option worst-case loss is the entire premium debit plus costs.
            # Use the full conservative round-trip cost at a zero-value exit.
            # Dividing round-trip costs by two understates the loss budget and can
            # incorrectly admit a trade near the 2% cap.
            worst_case_costs = compute_transaction_costs(entry, 0.0, lot, lots, cfg)
            worst_case = entry * qty + worst_case_costs["total_cost"]
            risk_budget = equity * risk_per_trade_pct
            if worst_case > risk_budget:
                rejected_risk += 1
                continue
            if worst_case > equity:
                rejected_cash += 1
                continue
            charges = compute_transaction_costs(entry, exit_premium, lot, lots, cfg)
            gross = (exit_premium - entry) * qty
            net = gross - charges["total_cost"]
            start_equity = equity
            equity += net
            peak = max(peak, equity)
            last_exit = pd.Timestamp(trade["exit_date"])
            daily_realized[day] = daily_realized.get(day, 0.0) + net
            accepted += 1
            rows.append({
                **trade,
                "capital_start": round(start_equity, 2),
                "capital_end": round(equity, 2),
                "max_loss_estimate": round(worst_case, 2),
                "risk_budget": round(risk_budget, 2),
                "risk_pct_of_equity": round(worst_case / start_equity * 100, 3),
                "controlled_gross_pnl": round(gross, 2),
                "controlled_costs": charges["total_cost"],
                "controlled_net_pnl": round(net, 2),
                "drawdown_from_peak": round(equity - peak, 2),
                "account_status": "ACTIVE" if equity > 0 else "STOPPED",
            })
            if equity <= 0:
                break
        accepted_frame = pd.DataFrame([r for r in rows if r["strategy"] == strategy])
        pnl = accepted_frame["controlled_net_pnl"] if not accepted_frame.empty else pd.Series(dtype=float)
        gp = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
        gl = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
        summary.append({
            "strategy": strategy,
            "initial_capital": initial_capital,
            "ending_capital": round(equity, 2),
            "net_pnl": round(equity - initial_capital, 2),
            "trades_taken": accepted,
            "trades_rejected_risk_budget": rejected_risk,
            "trades_rejected_cash": rejected_cash,
            "trades_rejected_overlap": rejected_overlap,
            "trades_rejected_daily_loss_limit": rejected_daily_stop,
            "profit_factor": round(gp / gl, 4) if gl else (float("inf") if gp else None),
            "max_drawdown_from_peak": round(float(accepted_frame["drawdown_from_peak"].min()), 2) if not accepted_frame.empty else 0.0,
            "risk_per_trade_pct": risk_per_trade_pct * 100,
            "daily_loss_limit_pct": daily_loss_limit_pct * 100,
            "capital_model": "separate ₹10,000 cash account per candidate strategy; one lot maximum; no leverage",
        })
    controlled = pd.DataFrame(rows)
    ranking = pd.DataFrame(summary).sort_values("net_pnl", ascending=False).reset_index(drop=True)
    report = {
        "status": "complete",
        "initial_capital_per_candidate": initial_capital,
        "risk_per_trade_pct": risk_per_trade_pct * 100,
        "daily_loss_limit_pct": daily_loss_limit_pct * 100,
        "strategies_tested": int(ledger["strategy"].nunique()),
        "candidate_trades": int(len(ledger)),
        "feasible_trades": int(len(controlled)),
        "rejected_trades": int(len(ledger) - len(controlled)),
        "capital_model": "Separate independent ₹10,000 cash account for each strategy candidate; full premium debit is at risk; one lot; no leverage or fractional lots.",
        "warning": "If a one-lot option premium exceeds the per-trade risk budget, the trade is skipped. A zero-trade result means the requested capital and risk limits are incompatible with available contract lot sizes/premiums.",
    }
    return controlled, ranking, report


def evaluate_capital_control_holdout(
    ledger: pd.DataFrame,
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.02,
    daily_loss_limit_pct: float = 0.02,
    train_fraction: float = 0.70,
    cost: CostConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate fixed capital rules on a chronological, independently reset split."""
    if ledger is None or ledger.empty or "entry_date" not in ledger:
        return pd.DataFrame(), {"status": "insufficient_data"}
    dates = sorted(pd.to_datetime(ledger["entry_date"], errors="coerce").dropna().dt.normalize().unique())
    if len(dates) < 2:
        return pd.DataFrame(), {"status": "insufficient_sessions"}
    split = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    validation_start = pd.Timestamp(dates[split])
    train = ledger[pd.to_datetime(ledger["entry_date"]).dt.normalize() < validation_start]
    validation = ledger[pd.to_datetime(ledger["entry_date"]).dt.normalize() >= validation_start]
    train_trades, train_board, train_report = apply_capital_controls(
        train, initial_capital, risk_per_trade_pct, daily_loss_limit_pct, cost
    )
    validation_trades, validation_board, validation_report = apply_capital_controls(
        validation, initial_capital, risk_per_trade_pct, daily_loss_limit_pct, cost
    )
    rows = []
    for strategy in sorted(set(train_board.get("strategy", [])) | set(validation_board.get("strategy", []))):
        tr = train_board[train_board["strategy"] == strategy]
        va = validation_board[validation_board["strategy"] == strategy]
        a = tr.iloc[0].to_dict() if not tr.empty else {}
        b = va.iloc[0].to_dict() if not va.empty else {}
        rows.append({
            "strategy": strategy,
            "train_trades_taken": int(a.get("trades_taken", 0)),
            "train_net_pnl": float(a.get("net_pnl", 0.0)),
            "train_ending_capital": float(a.get("ending_capital", initial_capital)),
            "validation_trades_taken": int(b.get("trades_taken", 0)),
            "validation_net_pnl": float(b.get("net_pnl", 0.0)),
            "validation_ending_capital": float(b.get("ending_capital", initial_capital)),
            "validation_profit_factor": b.get("profit_factor"),
            "both_periods_positive": bool(a.get("net_pnl", 0.0) > 0 and b.get("net_pnl", 0.0) > 0),
        })
    board = pd.DataFrame(rows).sort_values(
        ["both_periods_positive", "validation_net_pnl", "train_net_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True) if rows else pd.DataFrame()
    report = {
        "status": "complete",
        "method": "Fixed rules; chronological 70/30 split by entry session; separate ₹10,000 account reset in each period; no parameters fitted on validation.",
        "train_end": pd.Timestamp(dates[split - 1]).date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": pd.Timestamp(dates[-1]).date().isoformat(),
        "initial_capital_each_period": initial_capital,
        "train_feasible_trades": train_report.get("feasible_trades", 0),
        "validation_feasible_trades": validation_report.get("feasible_trades", 0),
        "train_rejected_trades": train_report.get("rejected_trades", 0),
        "validation_rejected_trades": validation_report.get("rejected_trades", 0),
        "warning": "A period with zero feasible trades is not a strategy pass. The validation segment is independent of the training segment but historical data may still be subject to source and execution limitations.",
    }
    return board, report
