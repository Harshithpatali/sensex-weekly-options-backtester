"""Structural and statistical data-quality checks for normalized SENSEX data."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sensex_options.calendar import available_expiries, detect_duplicate_expiries
from sensex_options.config import DEFAULT_PATHS

logger = logging.getLogger(__name__)


def validate_all(index: pd.DataFrame, options: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "index_rows": int(len(index)),
        "options_rows": int(len(options)),
        "warnings": [],
        "errors": [],
    }

    index_required = {"date", "close"}
    options_required = {"trade_date", "expiry_date", "strike", "option_type", "close"}
    missing_index = sorted(index_required - set(index.columns))
    missing_options = sorted(options_required - set(options.columns))

    if index.empty:
        report["errors"].append("Index data is empty. Run collect-index.")
    elif missing_index:
        report["errors"].append(f"Index dataset missing required columns: {missing_index}")
    else:
        idx = index.copy()
        idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
        idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
        invalid_dates = int(idx["date"].isna().sum())
        invalid_closes = int((idx["close"].isna() | (idx["close"] <= 0)).sum())
        if invalid_dates:
            report["errors"].append(f"Invalid index dates: {invalid_dates}")
        if invalid_closes:
            report["warnings"].append(f"Missing or non-positive index closes: {invalid_closes}")
        valid_idx = idx.dropna(subset=["date", "close"]).copy()
        if not valid_idx.empty:
            report["index_start"] = str(valid_idx["date"].min().date())
            report["index_end"] = str(valid_idx["date"].max().date())
        duplicates = int(idx.duplicated(subset=["date"]).sum())
        if duplicates:
            report["warnings"].append(f"Duplicate index dates: {duplicates}")

    if options.empty:
        report["errors"].append(
            "Options data is empty. Run collect-options. BSE SENSEX options coverage starts around May 2023."
        )
        report["expiries"] = 0
    elif missing_options:
        report["errors"].append(f"Options dataset missing required columns: {missing_options}")
    else:
        opt = options.copy()
        opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce")
        opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce")
        opt["strike"] = pd.to_numeric(opt["strike"], errors="coerce")
        opt["close"] = pd.to_numeric(opt["close"], errors="coerce")
        opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()

        invalid_dates = int((opt["trade_date"].isna() | opt["expiry_date"].isna()).sum())
        if invalid_dates:
            report["errors"].append(f"Invalid option trade/expiry dates: {invalid_dates}")
        invalid_strikes = int((opt["strike"].isna() | (opt["strike"] <= 0)).sum())
        if invalid_strikes:
            report["warnings"].append(f"Missing or non-positive option strikes: {invalid_strikes}")
        invalid_types = int((~opt["option_type"].isin(["CE", "PE"])).sum())
        if invalid_types:
            report["warnings"].append(f"Rows with option type other than CE/PE: {invalid_types}")
        invalid_closes = int((opt["close"].isna() | (opt["close"] < 0)).sum())
        if invalid_closes:
            report["warnings"].append(f"Missing or negative option closes: {invalid_closes}")
        reversed_dates = int((opt["expiry_date"] < opt["trade_date"]).sum())
        if reversed_dates:
            report["errors"].append(f"Option expiry precedes trade date: {reversed_dates}")

        valid_dates = opt["trade_date"].dropna()
        if not valid_dates.empty:
            report["options_start"] = str(valid_dates.min().date())
            report["options_end"] = str(valid_dates.max().date())

        expiries = available_expiries(opt)
        report["expiries"] = int(len(expiries))
        report["expiry_list_sample"] = [str(value.date()) for value in expiries[:10]]

        counts = opt["option_type"].value_counts().to_dict()
        report["option_type_counts"] = {str(key): int(value) for key, value in counts.items()}
        for side in ("CE", "PE"):
            if counts.get(side, 0) == 0:
                report["warnings"].append(f"No {side} rows found; strategies requiring both sides may be skipped.")

        if "volume" in opt.columns:
            volume = pd.to_numeric(opt["volume"], errors="coerce")
            zero_volume = int((volume.fillna(0) <= 0).sum())
            if zero_volume:
                report["warnings"].append(f"Zero or missing-volume option rows: {zero_volume}")

        missing_lot = (
            int(pd.to_numeric(opt["lot_size"], errors="coerce").isna().sum())
            if "lot_size" in opt.columns else len(opt)
        )
        report["missing_lot_size_rows"] = missing_lot
        if missing_lot:
            report["warnings"].append(
                f"Missing historical lot size on {missing_lot} rows; trades using those rows will be skipped."
            )

        duplicates = detect_duplicate_expiries(opt)
        report["duplicate_contract_samples"] = len(duplicates)
        if duplicates:
            report["warnings"].append(f"Possible duplicate contract keys: {len(duplicates)} samples")

        # Coverage is measured against observed SENSEX index sessions, not calendar days.
        if not index.empty and {"date", "close"}.issubset(index.columns) and not valid_dates.empty:
            idx_dates = pd.to_datetime(index["date"], errors="coerce").dt.normalize().dropna()
            option_dates = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize().dropna()
            # Compare against the complete observed index history so missing
            # option sessions at the beginning or end of the archive are visible.
            coverage_start, coverage_end = idx_dates.min(), idx_dates.max()
            expected_sessions = sorted(set(idx_dates[(idx_dates >= coverage_start) & (idx_dates <= coverage_end)]))
            observed_sessions = set(option_dates.unique())
            missing_sessions = [d.strftime("%Y-%m-%d") for d in expected_sessions if d not in observed_sessions]
            report["option_trading_days"] = int(option_dates.nunique())
            report["index_sessions_in_option_range"] = int(len(expected_sessions))
            report["option_session_coverage_pct"] = round(
                100.0 * (len(expected_sessions) - len(missing_sessions)) / len(expected_sessions), 2
            ) if expected_sessions else None
            report["missing_option_sessions"] = missing_sessions[:100]
            if missing_sessions:
                report["warnings"].append(
                    f"Missing option bhavcopy on {len(missing_sessions)} SENSEX index sessions in the covered date range."
                )

        expiry_rows = []
        for expiry in sorted(expiries):
            exp_frame = opt[opt["expiry_date"].dt.normalize().eq(expiry)]
            calls = exp_frame[exp_frame["option_type"].eq("CE")]
            puts = exp_frame[exp_frame["option_type"].eq("PE")]
            trade_days = sorted(exp_frame["trade_date"].dropna().dt.normalize().unique())
            expiry_rows.append({
                "expiry_date": pd.Timestamp(expiry).strftime("%Y-%m-%d"),
                "trade_days": int(len(trade_days)),
                "call_rows": int(len(calls)),
                "put_rows": int(len(puts)),
                "call_contracts": int(calls["strike"].nunique()),
                "put_contracts": int(puts["strike"].nunique()),
                "has_both_sides": bool(len(calls) and len(puts)),
                "first_observation": pd.Timestamp(trade_days[0]).strftime("%Y-%m-%d") if trade_days else None,
                "last_observation": pd.Timestamp(trade_days[-1]).strftime("%Y-%m-%d") if trade_days else None,
            })
        report["expiry_coverage"] = expiry_rows[:250]
        report["expiries_with_both_sides"] = int(sum(row["has_both_sides"] for row in expiry_rows))
        report["expiry_completeness_pct"] = round(
            100.0 * report["expiries_with_both_sides"] / len(expiry_rows), 2
        ) if expiry_rows else 0.0

        if "symbol" in opt.columns:
            non_sensex = int((~opt["symbol"].astype(str).str.upper().str.startswith("SENSEX")).sum())
            if non_sensex:
                report["warnings"].append(f"Non-SENSEX symbols found in options dataset: {non_sensex}")

    report["status"] = "incomplete" if report["errors"] else ("warning" if report["warnings"] else "ok")
    return report


def save_validation_report(report: dict[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_PATHS.reports / "data_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Validation report saved to %s", path)
    return path
