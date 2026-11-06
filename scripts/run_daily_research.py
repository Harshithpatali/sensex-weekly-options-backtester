"""Run incremental collection, validation, strategy replay and P&L exports.

Usage: python scripts/run_daily_research.py
"""
from sensex_options.daily_pipeline import run

if __name__ == "__main__":
    run()
