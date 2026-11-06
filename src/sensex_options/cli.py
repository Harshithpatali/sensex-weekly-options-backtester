"""
Command-line interface for SENSEX Weekly Options Backtester.

Examples:
  sensex-options collect-index --start 2023-05-15 --end 2026-11-07
  sensex-options collect-options --start 2023-05-15 --end 2026-11-07
  sensex-options build-features
  sensex-options validate
  sensex-options backtest --capital 10000 --cost-per-lot 0
  sensex-options report
  python -m sensex_options.cli validate
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from sensex_options.config import BacktestConfig, CostConfig, DEFAULT_PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sensex_options.cli")


@click.group()
def main() -> None:
    """SENSEX Weekly Options Backtester CLI."""
    DEFAULT_PATHS.ensure()


@main.command("collect-index")
@click.option("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
@click.option("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
def collect_index(start: str, end: str) -> None:
    """Download SENSEX index data from Yahoo Finance."""
    from sensex_options.sources.index import download_sensex_index
    df = download_sensex_index(start, end)
    click.echo(f"Index rows: {len(df)}")


@main.command("collect-options")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--sleep", default=0.8, show_default=True, help="Seconds between downloads")
def collect_options(start: str, end: str, sleep: float) -> None:
    """Download BSE FO bhavcopies and normalize SENSEX Call and Put options."""
    from sensex_options.sources.bse import download_bse_options_range
    df = download_bse_options_range(start, end, sleep=sleep)
    click.echo(f"SENSEX CE/PE option rows: {len(df):,}")


@main.command("build-features")
def build_features_cmd() -> None:
    """Assemble feature summary from processed data."""
    from sensex_options.features import build_features
    s = build_features()
    click.echo(s)


@main.command("validate")
def validate_cmd() -> None:
    """Validate index and options datasets."""
    from sensex_options.sources.index import load_index
    from sensex_options.sources.bse import load_options
    from sensex_options.validation import validate_all, save_validation_report

    index = load_index()
    options = load_options()
    report = validate_all(index, options)
    path = save_validation_report(report)
    click.echo(f"Status: {report.get('status')}")
    click.echo(f"Report: {path}")
    for w in report.get("warnings", []):
        click.echo(f"  WARN: {w}")
    for e in report.get("errors", []):
        click.echo(f"  ERR:  {e}")


@main.command("backtest")
@click.option("--capital", default=10000.0, show_default=True, type=float)
@click.option("--start", default=None, help="Filter expiries >= date")
@click.option("--end", default=None, help="Filter expiries <= date")
@click.option("--cost-per-lot", default=0.0, show_default=True, type=float,
              help="Flat brokerage per lot (entry+exit total via CostConfig)")
@click.option("--max-lots", default=None, type=int)
def backtest_cmd(capital: float, start: str | None, end: str | None, cost_per_lot: float, max_lots: int | None) -> None:
    """Run the long Call weekly strategy backtest."""
    from sensex_options.sources.index import load_index
    from sensex_options.sources.bse import load_options
    from sensex_options.backtest import run_backtest
    from sensex_options.reports import write_backtest_reports

    cost = CostConfig(brokerage_per_lot=cost_per_lot)
    cfg = BacktestConfig(
        initial_capital=capital,
        start_date=start,
        end_date=end,
        max_lots=max_lots,
        cost=cost,
        assume_zero_costs=(cost_per_lot == 0),
    )
    index = load_index()
    options = load_options()
    result = run_backtest(index, options, cfg)
    paths = write_backtest_reports(result)
    summary = result["summary"]
    click.echo("--- Backtest Summary ---")
    for k, v in summary.items():
        click.echo(f"  {k}: {v}")
    click.echo(f"Reports: {list(paths.values())}")


@main.command("report")
def report_cmd() -> None:
    """Re-emit reports from last backtest outputs if present."""
    out = DEFAULT_PATHS.outputs
    files = list(out.glob("*"))
    if not files:
        click.echo("No output files found. Run backtest first.")
        return
    for f in files:
        click.echo(f"  {f}")


@main.command("analyze")
def analyze_cmd() -> None:
    """Statistical, probability, and lightweight ML analytics on last backtest + index."""
    from sensex_options.sources.index import load_index
    from sensex_options.analytics import run_full_analytics
    import pandas as pd

    index = load_index()
    ledger = DEFAULT_PATHS.outputs / "trade_ledger.csv"
    trades = pd.read_csv(ledger) if ledger.exists() else pd.DataFrame()
    result = run_full_analytics(trades, index)
    click.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
