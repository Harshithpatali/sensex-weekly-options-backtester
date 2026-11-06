"""Run data validation independently of preprocessing.

Usage: python scripts/validate_data.py
"""
from sensex_options.config import DEFAULT_PATHS
from sensex_options.sources.index import load_index
from sensex_options.sources.bse import load_options
from sensex_options.validation import validate_all, save_validation_report

def main() -> None:
    DEFAULT_PATHS.ensure()
    report = validate_all(load_index(), load_options())
    path = save_validation_report(report)
    print(f"Status: {report.get('status')} | report: {path}")
    for warning in report.get("warnings", []):
        print("WARNING:", warning)
    for error in report.get("errors", []):
        print("ERROR:", error)
    if report.get("errors"):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
