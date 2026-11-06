"""Download SENSEX index and BSE FO options history.

Usage:
  python scripts/download_data.py --start 2023-05-15 --end 2026-11-06
  python scripts/download_data.py --start 2024-01-01 --end 2024-12-31 --index-only
"""
from __future__ import annotations
import argparse
import logging
from sensex_options.config import DEFAULT_PATHS, SENSEX_OPTIONS_LAUNCH
from sensex_options.sources.index import download_sensex_index
from sensex_options.sources.bse import download_bse_options_range

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=SENSEX_OPTIONS_LAUNCH)
    parser.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.8, help="Delay between BSE requests")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--options-only", action="store_true")
    args = parser.parse_args()
    if args.index_only and args.options_only:
        parser.error("Choose at most one of --index-only or --options-only")
    DEFAULT_PATHS.ensure()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.options_only:
        index = download_sensex_index(args.start, args.end)
        print(f"Index download complete: {len(index):,} rows")
    if not args.index_only:
        options = download_bse_options_range(args.start, args.end, sleep=args.sleep)
        print(f"BSE options download complete: {len(options):,} rows")
        if not options.empty and "option_type" in options:
            print("Option types:", options["option_type"].value_counts(dropna=False).to_dict())

if __name__ == "__main__":
    main()
