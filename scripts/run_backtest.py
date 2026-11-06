"""Run the original single-strategy backtest.

Usage:
  python scripts/run_backtest.py --capital 10000
  python scripts/run_backtest.py --capital 10000 --start 2024-01-01 --end 2025-12-31
"""
from __future__ import annotations
import sys
from sensex_options.cli import main as cli

if __name__ == "__main__":
    cli(args=["backtest", *sys.argv[1:]], standalone_mode=False)
