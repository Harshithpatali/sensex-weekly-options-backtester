"""
Report generation: trade ledger, annual summary, JSON backtest summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sensex_options.config import DEFAULT_PATHS

logger = logging.getLogger(__name__)


def write_backtest_reports(
    result: Dict[str, Any],
    out_dir: Path | None = None,
) -> Dict[str, Path]:
    out_dir = out_dir or DEFAULT_PATHS.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    trades: pd.DataFrame = result.get("trades", pd.DataFrame())
    summary = result.get("summary", {})
    equity = result.get("equity_curve", pd.DataFrame())

    if not trades.empty:
        ledger = out_dir / "trade_ledger.csv"
        trades.to_csv(ledger, index=False)
        paths["trade_ledger"] = ledger

        completed = trades[trades["status"] == "completed"]
        skipped = trades[trades["status"] == "skipped"]
        if len(completed):
            p = out_dir / "completed_trades.csv"
            completed.to_csv(p, index=False)
            paths["completed"] = p
        if len(skipped):
            p = out_dir / "skipped_trades.csv"
            skipped.to_csv(p, index=False)
            paths["skipped"] = p

        # Annual summary
        if len(completed) and "exit_date" in completed.columns:
            tmp = completed.copy()
            tmp["year"] = pd.to_datetime(tmp["exit_date"]).dt.year
            annual = tmp.groupby("year").agg(
                trades=("net_pnl", "count"),
                net_pnl=("net_pnl", "sum"),
                gross_pnl=("gross_pnl", "sum"),
                wins=("net_pnl", lambda s: (s > 0).sum()),
            ).reset_index()
            p = out_dir / "annual_summary.csv"
            annual.to_csv(p, index=False)
            paths["annual"] = p

    if not equity.empty:
        p = out_dir / "equity_curve.csv"
        equity.to_csv(p, index=False)
        paths["equity"] = p

    summary_path = out_dir / "backtest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    paths["summary"] = summary_path

    logger.info("Reports written to %s", out_dir)
    return paths
