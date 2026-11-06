"""Parse raw BSE bhavcopies, merge normalized options, summarize and validate.

Place official BSE FO .CSV or .ZIP files in data/raw/options/ and run:
    python scripts/preprocess_data.py

The script preserves previously processed history and never invents option prices.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from sensex_options.config import DEFAULT_PATHS
from sensex_options.sources.bse import load_options, parse_bse_fo_file
from sensex_options.sources.index import load_index
from sensex_options.features import build_features
from sensex_options.validation import validate_all, save_validation_report

KEYS = ["trade_date", "expiry_date", "symbol", "strike", "option_type"]

def parse_raw_options() -> pd.DataFrame:
    raw_dir = DEFAULT_PATHS.raw_options
    files = sorted([*raw_dir.glob("*.csv"), *raw_dir.glob("*.CSV"), *raw_dir.glob("*.zip"), *raw_dir.glob("*.ZIP")])
    parsed = []
    for path in files:
        frame = parse_bse_fo_file(path)
        if frame.empty:
            print(f"SKIP (no rows parsed): {path.name}")
            continue
        frame = frame[
            frame["symbol"].astype(str).str.contains("SENSEX", case=False, na=False)
            & frame["option_type"].isin(["CE", "PE"])
            & frame["strike"].notna()
            & (frame["strike"] > 0)
        ].copy()
        if not frame.empty:
            parsed.append(frame)
            print(f"PARSED {path.name}: {len(frame):,} SENSEX option rows")
    return pd.concat(parsed, ignore_index=True) if parsed else pd.DataFrame()

def main() -> None:
    DEFAULT_PATHS.ensure()
    new_rows = parse_raw_options()
    existing = load_options()
    frames = [x for x in (existing, new_rows) if not x.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.normalize()
        combined["expiry_date"] = pd.to_datetime(combined["expiry_date"], errors="coerce").dt.normalize()
        combined = combined.dropna(subset=["trade_date", "expiry_date", "strike"])
        combined = combined.drop_duplicates(subset=KEYS, keep="last").sort_values(KEYS).reset_index(drop=True)
        output = DEFAULT_PATHS.processed / "sensex_options.csv"
        combined.to_csv(output, index=False)
        print(f"Processed options saved: {output} ({len(combined):,} rows)")
    else:
        print("No processed options found. Download data or add raw BSE bhavcopies first.")

    summary = build_features()
    index, options = load_index(), load_options()
    report = validate_all(index, options)
    report_path = save_validation_report(report)
    print("\nPREPROCESSING SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Validation status: {report.get('status')}")
    print(f"Validation report: {report_path}")
    if report.get("errors"):
        raise SystemExit("Preprocessing stopped: data-quality errors found. Inspect the report.")

if __name__ == "__main__":
    main()
