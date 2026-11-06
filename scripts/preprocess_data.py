"""Build the processed dataset summary and run data-quality checks.

Usage: python scripts/preprocess_data.py
"""
from __future__ import annotations
import json
from sensex_options.config import DEFAULT_PATHS
from sensex_options.features import build_features
from sensex_options.sources.index import load_index
from sensex_options.sources.bse import load_options
from sensex_options.validation import validate_all, save_validation_report

def main() -> None:
    DEFAULT_PATHS.ensure()
    summary = build_features()
    index, options = load_index(), load_options()
    report = validate_all(index, options)
    report_path = save_validation_report(report)
    print("PREPROCESSING SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Validation status: {report.get('status')}")
    print(f"Validation report: {report_path}")
    if report.get("errors"):
        raise SystemExit("Preprocessing stopped: data-quality errors found. Inspect the report.")

if __name__ == "__main__":
    main()
