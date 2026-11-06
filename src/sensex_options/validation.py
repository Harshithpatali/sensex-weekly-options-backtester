"""
Data validation and quality reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sensex_options.calendar import available_expiries, detect_duplicate_expiries
from sensex_options.config import DEFAULT_PATHS

logger = logging.getLogger(__name__)


def validate_all(
    index: pd.DataFrame,
    options: pd.DataFrame,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "index_rows": len(index),
        "options_rows": len(options),
        "warnings": [],
        "errors": [],
    }

    if index.empty:
        report["errors"].append("Index data is empty. Run collect-index.")
    else:
        report["index_start"] = str(index["date"].min().date())
        report["index_end"] = str(index["date"].max().date())
        dups = index.duplicated(subset=["date"]).sum()
        if dups:
            report["warnings"].append(f"Duplicate index dates: {dups}")
        invalid = (index["close"] <= 0).sum()
        if invalid:
            report["warnings"].append(f"Non-positive index closes: {invalid}")

    if options.empty:
        report["errors"].append(
            "Options data is empty. Run collect-options. "
            "BSE SENSEX options coverage starts ~May 2023."
        )
        report["expiries"] = 0
    else:
        report["options_start"] = str(pd.to_datetime(options["trade_date"]).min().date())
        report["options_end"] = str(pd.to_datetime(options["trade_date"]).max().date())
        expiries = available_expiries(options)
        report["expiries"] = len(expiries)
        report["expiry_list_sample"] = [str(d.date()) for d in expiries[:10]]

        missing_close = options["close"].isna().sum()
        if missing_close:
            report["warnings"].append(f"Options rows with missing close: {missing_close}")

        zero_vol = (options["volume"].fillna(0) == 0).sum() if "volume" in options.columns else 0
        if zero_vol:
            report["warnings"].append(f"Zero-volume option rows: {zero_vol}")

        missing_lot = options["lot_size"].isna().sum() if "lot_size" in options.columns else len(options)
        report["missing_lot_size_rows"] = int(missing_lot)
        if missing_lot == len(options):
            report["warnings"].append(
                "No lot_size values present in options data. "
                "Trades will be skipped until lot size metadata is supplied."
            )

        dups = detect_duplicate_expiries(options)
        report["duplicate_contract_samples"] = len(dups)
        if dups:
            report["warnings"].append(f"Possible duplicate contracts: {len(dups)} samples")

        # Premiums that look stale / extreme (heuristic only)
        if "close" in options.columns:
            extreme = options[options["close"] > 50_000]
            if len(extreme):
                report["warnings"].append(f"Extreme premium values (>50000): {len(extreme)}")

    report["status"] = "ok" if not report["errors"] else "incomplete"
    return report


def save_validation_report(report: Dict[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_PATHS.reports / "data_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation report saved to %s", path)
    return path
