from __future__ import annotations

"""Risk-managed daily option research using observed next-session option OHLC.

Stop/target fills are approximations from daily BSE option candles, not tick data.
When both a stop and target are touched within the same session, the stop is
assumed to occur first (conservative ordering). A stop gap is filled at the
observed session open.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs


STOP_LOSS_LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
TAKE_PROFIT_LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60)
MONTHLY_TARGET_PF = 1.20
MIN_MONTHLY_TRADES = 3


def _default_cost() -> CostConfig:
    # Keep assumptions identical to run_daily_strategy_suite.
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


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _bucket() -> dict:
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "net_pnl": 0.0,
    }


def _add(bucket: dict, pnl: float) -> None:
    bucket["trades"] += 1
    bucket["net_pnl"] += pnl
    if pnl > 0:
        bucket["wins"] += 1
        bucket["gross_profit"] += pnl
    elif pnl < 0:
        bucket["losses"] += 1
        bucket["gross_loss"] += -pnl


def _profit_factor(gross_profit: float, gross_loss: float) -> float | None:
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return None


def _bucket_record(strategy: str, month: str, bucket: dict) -> dict:
    trades = int(bucket["trades"])
    gp = float(bucket["gross_profit"])
    gl = float(bucket["gross_loss"])
    pf = _profit_factor(gp, gl)
    target_met = None if trades < MIN_MONTHLY_TRADES else bool(pf is not None and pf >= MONTHLY_TARGET_PF)
    return {
        "strategy": strategy,
        "month": month,
        "trades": trades,
        "wins": int(bucket["wins"]),
        "losses": int(bucket["losses"]),
        "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2),
        "net_pnl": round(float(bucket["net_pnl"]), 2),
        "profit_factor": pf,
        "target_profit_factor": MONTHLY_TARGET_PF,
        "target_met": target_met,
        "month_status": (
            "INSUFFICIENT_TRADES" if trades < MIN_MONTHLY_TRADES
            else "PASS" if target_met else "FAIL"
        ),
    }


def _period_metrics(strategy: str, period: str, buckets: dict) -> dict:
    selected = [
        (month, bucket)
        for (name, split, month), bucket in buckets.items()
        if name == strategy and split == period
    ]
    trades = sum(int(b["trades"]) for _, b in selected)
    gp = sum(float(b["gross_profit"]) for _, b in selected)
    gl = sum(float(b["gross_loss"]) for _, b in selected)
    net = sum(float(b["net_pnl"]) for _, b in selected)
    pf = _profit_factor(gp, gl)
    active = [
        _bucket_record(strategy, month, bucket)
        for month, bucket in selected
        if int(bucket["trades"]) >= MIN_MONTHLY_TRADES
    ]
    passing = sum(1 for row in active if row["target_met"] is True)
    active_count = len(active)
    return {
        f"{period}_trades": trades,
        f"{period}_net_pnl": round(net, 2),
        f"{period}_gross_profit": round(gp, 2),
        f"{period}_gross_loss": round(gl, 2),
        f"{period}_aggregate_pf": pf,
        f"{period}_active_months": active_count,
        f"{period}_target_months": passing,
        f"{period}_target_hit_pct": (
            round(passing / active_count * 100, 2) if active_count else None
        ),
        f"{period}_all_active_months_pass": bool(active_count and passing == active_count),
    }


def _simulate_exit(
    entry: float,
    close: float,
    open_price: float,
    high: float,
    low: float,
    stop_pct: float | None,
    target_pct: float | None,
) -> tuple[float, str]:
    """Simulate a one-session stop/target using daily OHLC, conservatively."""
    stop_price = entry * (1.0 - stop_pct) if stop_pct is not None else None
    target_price = entry * (1.0 + target_pct) if target_pct is not None else None

    # Opening gaps are observable before the daily high/low path.
    if stop_price is not None and _finite(open_price) and float(open_price) <= stop_price:
        return max(0.0, float(open_price)), "STOP_GAP"
    if target_price is not None and _finite(open_price) and float(open_price) >= target_price:
        return target_price, "TAKE_PROFIT_AT_OPEN"

    stop_hit = stop_price is not None and float(low) <= stop_price
    target_hit = target_price is not None and float(high) >= target_price

    # If both levels are touched intraday, assume the adverse stop happens first.
    if stop_hit:
        return max(0.0, stop_price), "STOP_LOSS"
    if target_hit:
        return target_price, "TAKE_PROFIT"
    return max(0.0, close), "EOD"


def run_risk_managed_research(
    base_ledger: pd.DataFrame,
    options: pd.DataFrame,
    cost: CostConfig | None = None,
    stop_loss_levels: tuple[float, ...] = STOP_LOSS_LEVELS,
    take_profit_levels: tuple[float, ...] = TAKE_PROFIT_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Test baseline, stop-loss-only, take-profit-only, and paired exit rules.

    The 70/30 split is chronological by entry session. Candidate ranking uses
    only the training window. The holdout is reported once for validation.
    """
    if base_ledger is None or base_ledger.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_data",
            "note": "Daily strategy ledger is empty.",
        }
    required_ledger = {
        "strategy", "entry_date", "exit_date", "expiry_date", "strike",
        "symbol", "option_type", "entry_premium", "exit_premium",
        "quantity", "lot_size", "lots", "net_pnl",
    }
    missing_ledger = sorted(required_ledger - set(base_ledger.columns))
    if missing_ledger:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "invalid_ledger",
            "missing_columns": missing_ledger,
        }

    required_ohlc = {
        "trade_date", "expiry_date", "strike", "symbol", "option_type",
        "open", "high", "low",
    }
    if options is None or options.empty or not required_ohlc.issubset(options.columns):
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "missing_option_ohlc",
            "note": "Next-session BSE option open/high/low fields are required; no stop-loss claims were made.",
            "missing_columns": sorted(required_ohlc - set(options.columns if options is not None else [])),
        }

    cfg = cost or _default_cost()
    ledger = base_ledger.copy()
    ledger["entry_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce").dt.normalize()
    ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce").dt.normalize()
    ledger["expiry_date"] = pd.to_datetime(ledger["expiry_date"], errors="coerce").dt.normalize()
    ledger = ledger.dropna(subset=["entry_date", "exit_date", "expiry_date"]).copy()
    if ledger.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_data",
            "note": "No valid trade dates remain in the daily ledger.",
        }

    # Chronological split happens once using unique entry sessions, before rule evaluation.
    entry_dates = sorted(ledger["entry_date"].drop_duplicates().tolist())
    if len(entry_dates) < 10:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_data",
            "note": "At least 10 distinct entry sessions are required for a chronological holdout.",
            "distinct_entry_sessions": len(entry_dates),
        }
    split_at = max(1, min(len(entry_dates) - 1, int(len(entry_dates) * 0.70)))
    train_end = pd.Timestamp(entry_dates[split_at - 1])
    holdout_start = pd.Timestamp(entry_dates[split_at])

    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "open", "high", "low"):
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper()
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike"])
    opt = opt.drop_duplicates(
        subset=["trade_date", "expiry_date", "strike", "symbol", "option_type"],
        keep="last",
    )
    candle_map = {}
    for item in opt.to_dict("records"):
        key = (
            pd.Timestamp(item["trade_date"]).strftime("%Y-%m-%d"),
            pd.Timestamp(item["expiry_date"]).strftime("%Y-%m-%d"),
            float(item["strike"]),
            str(item["symbol"]),
            str(item["option_type"]),
        )
        candle_map[key] = item

    rules: list[tuple[str, float | None, float | None]] = []
    for sl in stop_loss_levels:
        rules.append((f"SL{int(round(sl * 100))}", float(sl), None))
    for tp in take_profit_levels:
        rules.append((f"TP{int(round(tp * 100))}", None, float(tp)))
    for sl in stop_loss_levels:
        for tp in take_profit_levels:
            rules.append((
                f"SL{int(round(sl * 100))}_TP{int(round(tp * 100))}",
                float(sl),
                float(tp),
            ))

    full_month: dict[tuple[str, str], dict] = defaultdict(_bucket)
    split_month: dict[tuple[str, str, str], dict] = defaultdict(_bucket)
    base_counts: dict[str, int] = defaultdict(int)
    covered_counts: dict[str, int] = defaultdict(int)
    variant_names: set[str] = set()
    reason_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for trade in ledger.to_dict("records"):
        base_name = str(trade["strategy"])
        base_counts[base_name] += 1
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        month = exit_date.strftime("%Y-%m")
        period = "train" if entry_date <= train_end else "holdout"

        # Baseline remains available even when OHLC is missing.
        base_pnl = float(trade["net_pnl"])
        variant_names.add(base_name)
        _add(full_month[(base_name, month)], base_pnl)
        _add(split_month[(base_name, period, month)], base_pnl)

        key = (
            exit_date.strftime("%Y-%m-%d"),
            pd.Timestamp(trade["expiry_date"]).strftime("%Y-%m-%d"),
            float(trade["strike"]),
            str(trade["symbol"]).strip().upper(),
            str(trade["option_type"]).strip().upper(),
        )
        candle = candle_map.get(key)
        if candle is None or not all(_finite(candle.get(k)) for k in ("high", "low")):
            continue
        high, low = float(candle["high"]), float(candle["low"])
        if low < 0 or high < low:
            continue
        covered_counts[base_name] += 1
        entry = float(trade["entry_premium"])
        close = float(trade["exit_premium"])
        if not _finite(entry) or entry <= 0 or not _finite(close):
            continue
        open_price = candle.get("open", np.nan)
        quantity = int(trade["quantity"])
        lot_size = int(trade["lot_size"])
        lots = int(trade["lots"])

        for rule_name, sl, tp in rules:
            variant = f"{base_name}__{rule_name}"
            variant_names.add(variant)
            exit_price, reason = _simulate_exit(
                entry, close, open_price, high, low, sl, tp
            )
            gross = (exit_price - entry) * quantity
            charges = compute_transaction_costs(
                entry, exit_price, lot_size, lots, cfg
            )
            net = gross - float(charges["total_cost"])
            _add(full_month[(variant, month)], net)
            _add(split_month[(variant, period, month)], net)
            reason_counts[variant][reason] += 1

    monthly_rows = []
    for (strategy, month), bucket in full_month.items():
        monthly_rows.append(_bucket_record(strategy, month, bucket))
    monthly = pd.DataFrame(monthly_rows)
    if not monthly.empty:
        monthly = monthly.sort_values(["strategy", "month"]).reset_index(drop=True)

    leaderboard_rows = []
    for strategy in variant_names:
        train = _period_metrics(strategy, "train", split_month)
        holdout = _period_metrics(strategy, "holdout", split_month)
        full_rows = monthly[monthly["strategy"] == strategy] if not monthly.empty else pd.DataFrame()
        full_net = float(full_rows["net_pnl"].sum()) if not full_rows.empty else 0.0
        full_gp = float(full_rows["gross_profit"].sum()) if not full_rows.empty else 0.0
        full_gl = float(full_rows["gross_loss"].sum()) if not full_rows.empty else 0.0
        base_name = strategy.split("__", 1)[0]
        base_n = int(base_counts.get(base_name, 0))
        covered_n = int(covered_counts.get(base_name, 0)) if "__" in strategy else base_n
        row = {
            "strategy": strategy,
            "base_strategy": base_name,
            "risk_rule": strategy.split("__", 1)[1] if "__" in strategy else "NO_STOP_BASELINE",
            "full_trades": int(full_rows["trades"].sum()) if not full_rows.empty else 0,
            "full_net_pnl": round(full_net, 2),
            "full_aggregate_pf": _profit_factor(full_gp, full_gl),
            "option_ohlc_coverage_pct": round(covered_n / base_n * 100, 2) if base_n else 0.0,
            **train,
            **holdout,
            "train_all_months_target_met": train["train_all_active_months_pass"],
            "holdout_all_months_target_met": holdout["holdout_all_active_months_pass"],
        }
        row["validation_candidate"] = bool(
            row["option_ohlc_coverage_pct"] >= 95.0
            and train["train_trades"] >= 30
            and train["train_active_months"] >= 4
            and train["train_all_active_months_pass"]
            and train["train_aggregate_pf"] is not None
            and train["train_aggregate_pf"] >= MONTHLY_TARGET_PF
            and holdout["holdout_trades"] >= 15
            and holdout["holdout_active_months"] >= 4
            and holdout["holdout_all_active_months_pass"]
            and holdout["holdout_aggregate_pf"] is not None
            and holdout["holdout_aggregate_pf"] >= MONTHLY_TARGET_PF
        )
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["train_target_hit_pct", "train_aggregate_pf", "train_net_pnl"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        leaderboard.insert(0, "training_rank", range(1, len(leaderboard) + 1))

    approved = leaderboard.loc[leaderboard["validation_candidate"]] if not leaderboard.empty else pd.DataFrame()
    top_train = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    coverage_total = sum(covered_counts.values())
    base_total = sum(base_counts.values())
    report = {
        "status": "validation_candidates_found" if not approved.empty else "no_candidate_met_monthly_target",
        "target_definition": "Monthly gross profit must be at least 1.20 times gross loss (profit factor >= 1.20).",
        "target_profit_factor": MONTHLY_TARGET_PF,
        "minimum_trades_per_month_for_target_test": MIN_MONTHLY_TRADES,
        "minimum_validation_candidate_requirements": {
            "option_ohlc_coverage_pct": 95.0,
            "training_trades": 30,
            "training_active_months": 4,
            "training_all_active_months_must_pass": True,
            "holdout_trades": 15,
            "holdout_active_months": 4,
            "holdout_all_active_months_must_pass": True,
        },
        "method": (
            "Baseline plus stop-loss-only, take-profit-only, and paired stop/target variants. "
            "Signals and entries are unchanged. Stops/targets use next-session BSE option OHLC; "
            "if both stop and target are touched intraday, stop is assumed first. "
            "Stop gaps use observed next-session open. Estimated transaction costs are reapplied."
        ),
        "train_start": str(entry_dates[0].date()),
        "train_end": str(train_end.date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(entry_dates[-1].date()),
        "distinct_entry_sessions": len(entry_dates),
        "base_strategies_tested": int(ledger["strategy"].nunique()),
        "risk_rules_per_base_strategy": len(rules),
        "total_variants_including_baselines": int(len(variant_names)),
        "monthly_rows": int(len(monthly)),
        "base_trade_count": int(len(ledger)),
        "option_ohlc_coverage_pct": round(coverage_total / base_total * 100, 2) if base_total else 0.0,
        "approved_candidate_count": int(len(approved)),
        "top_training_ranked_strategy": {
            "strategy": top_train.get("strategy"),
            "train_net_pnl": top_train.get("train_net_pnl"),
            "train_aggregate_pf": top_train.get("train_aggregate_pf"),
            "train_target_hit_pct": top_train.get("train_target_hit_pct"),
            "holdout_net_pnl": top_train.get("holdout_net_pnl"),
            "holdout_aggregate_pf": top_train.get("holdout_aggregate_pf"),
            "holdout_target_hit_pct": top_train.get("holdout_target_hit_pct"),
        } if top_train else None,
        "approved_candidates": approved["strategy"].astype(str).tolist() if not approved.empty else [],
        "exit_reason_counts_top_ranked": dict(reason_counts.get(str(top_train.get("strategy")), {})) if top_train else {},
        "warning": (
            "A passing backtest is not a guaranteed monthly return. Daily OHLC cannot reveal the exact "
            "intraday order of price extremes; the conservative stop-first rule is applied when both "
            "levels are touched. Results depend on historical data quality and estimated execution costs."
        ),
    }
    return monthly, leaderboard, report
