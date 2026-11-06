"""
SENSEX Weekly Options Backtester
Research platform for historical long Call options strategy on SENSEX weeklies.
Uses actual BSE derivatives bhavcopy data. Never fabricates option premiums.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research"

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_INDEX = DATA_ROOT / "raw" / "index"
RAW_OPTIONS = DATA_ROOT / "raw" / "options"
PROCESSED = DATA_ROOT / "processed"
REPORTS = DATA_ROOT / "reports"
OUTPUTS = PROJECT_ROOT / "outputs"
