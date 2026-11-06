"""
Feature / dataset assembly helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sensex_options.config import DEFAULT_PATHS
from sensex_options.sources.index import load_index
from sensex_options.sources.bse import load_options

logger = logging.getLogger(__name__)


def build_features(
    index_path: Path | None = None,
    options_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """
    Assemble processed datasets and write light summary features.
    """
    out_dir = out_dir or DEFAULT_PATHS.processed
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(index_path)
    options = load_options(options_path)

    summary = {
        "index_rows": len(index),
        "options_rows": len(options),
    }
    if not index.empty:
        summary["index_range"] = f"{index['date'].min().date()} → {index['date'].max().date()}"
    if not options.empty:
        summary["options_range"] = (
            f"{pd.to_datetime(options['trade_date']).min().date()} → "
            f"{pd.to_datetime(options['trade_date']).max().date()}"
        )
        if "expiry_date" in options.columns:
            summary["unique_expiries"] = int(options["expiry_date"].nunique())

    summary_path = out_dir / "features_summary.json"
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Features summary written to %s", summary_path)
    return summary
