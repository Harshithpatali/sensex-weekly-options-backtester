"""
Configuration management for SENSEX Weekly Options Backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from sensex_options import DATA_ROOT, PROJECT_ROOT


@dataclass
class CostConfig:
    """Transaction cost assumptions (configurable; historical rates vary)."""

    brokerage_per_lot: float = 0.0          # flat brokerage per lot (entry+exit)
    brokerage_pct: float = 0.0              # % of premium turnover
    stt_pct: float = 0.0                    # Securities Transaction Tax (sell side typical)
    exchange_charges_pct: float = 0.0
    gst_pct: float = 0.0                    # on brokerage + exchange
    stamp_duty_pct: float = 0.0
    other_charges: float = 0.0
    entry_slippage: float = 0.0             # absolute points added to entry premium
    exit_slippage: float = 0.0              # absolute points subtracted from exit premium

    def label(self) -> str:
        return (
            f"brokerage={self.brokerage_per_lot}/lot + {self.brokerage_pct*100:.3f}%, "
            f"STT={self.stt_pct*100:.3f}%, slippage={self.entry_slippage}/{self.exit_slippage}"
        )


@dataclass
class BacktestConfig:
    """Backtest run parameters."""

    initial_capital: float = 10_000.0
    start_date: Optional[str] = None        # YYYY-MM-DD
    end_date: Optional[str] = None
    max_lots: Optional[int] = None          # None = capital limited only
    cost: CostConfig = field(default_factory=CostConfig)
    # Research default: zero costs. User must understand this is unrealistic.
    assume_zero_costs: bool = True


@dataclass
class Paths:
    data_root: Path = DATA_ROOT
    raw_index: Path = DATA_ROOT / "raw" / "index"
    raw_options: Path = DATA_ROOT / "raw" / "options"
    processed: Path = DATA_ROOT / "processed"
    reports: Path = DATA_ROOT / "reports"
    outputs: Path = PROJECT_ROOT / "outputs"

    def ensure(self) -> None:
        for p in (self.raw_index, self.raw_options, self.processed, self.reports, self.outputs):
            p.mkdir(parents=True, exist_ok=True)


DEFAULT_PATHS = Paths()
DEFAULT_BACKTEST = BacktestConfig()


def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# BSE SENSEX options coverage note
SENSEX_OPTIONS_LAUNCH = "2023-05-15"  # approximate relaunch of BSE equity derivatives
INDEX_TICKER = "^BSESN"
