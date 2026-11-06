from __future__ import annotations

"""Incremental daily collection + validation + strategy replay.

GitHub Actions invokes this at 06:30 Asia/Kolkata on weekdays. The pipeline
preserves existing history, collects both option sides, and exports research
reports. It never places orders or trains an ML model.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sensex_options.config import DEFAULT_PATHS, SENSEX_OPTIONS_LAUNCH
from sensex_options.sources.bse import download_one_day, parse_bse_fo_file, load_options
from sensex_options.sources.index import download_sensex_index, load_index
from sensex_options.validation import validate_all, save_validation_report
from sensex_options.strategies import run_strategy_suite, period_pnl
from sensex_options.analytics import strategy_performance_metrics, walk_forward_evaluation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("daily_pipeline")

OPTION_KEYS = ["trade_date", "expiry_date", "symbol", "strike", "option_type"]


def merge_csv(new: pd.DataFrame, path: Path, keys: list[str]) -> pd.DataFrame:
    """Merge new observations with CSV or gzip history without discarding old rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = Path(str(path) + ".gz")
    if path.exists():
        old = pd.read_csv(path)
    elif compressed.exists():
        old = pd.read_csv(compressed, compression="gzip")
    else:
        old = pd.DataFrame()

    frames = [frame for frame in (old, new) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for key in ("date", "trade_date", "expiry_date"):
        if key in combined.columns and key in keys:
            combined[key] = pd.to_datetime(combined[key], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=[key for key in keys if key in combined.columns])
    combined = combined.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def run() -> dict:
    DEFAULT_PATHS.ensure()

    # GitHub schedule is 01:00 UTC (06:30 India). Use UTC explicitly so the
    # target date is deterministic regardless of the runner's local timezone.
    today_utc = datetime.now(timezone.utc).date()
    target = today_utc - timedelta(days=1)

    # Retry a short rolling window to cover weekends, exchange holidays, and
    # temporary publication delays without re-downloading the entire archive.
    downloaded_days = []
    for offset in range(4, 0, -1):
        day_date = target - timedelta(days=offset - 1)
        day = datetime.combine(day_date, datetime.min.time())
        raw_path, status = download_one_day(day, DEFAULT_PATHS.raw_options)
        log.info("BSE download %s: %s", day_date, status)
        if raw_path is None:
            continue
        daily = parse_bse_fo_file(raw_path, day)
        if daily.empty:
            log.warning("No valid SENSEX option rows parsed for %s", day_date)
            continue
        daily = daily[
            daily["symbol"].astype(str).str.startswith("SENSEX", na=False)
            & daily["option_type"].isin(["CE", "PE"])
        ].copy()
        if not daily.empty:
            merge_csv(
                daily,
                DEFAULT_PATHS.processed / "sensex_options.csv",
                OPTION_KEYS,
            )
            downloaded_days.append(day_date.isoformat())

    # Refresh only a recent overlap. The bootstrap workflow obtains the full
    # index history; the overlap handles late corrections without daily full
    # history downloads or replacement of the accumulated file.
    refresh_start = max(
        pd.Timestamp(SENSEX_OPTIONS_LAUNCH).date(),
        target - timedelta(days=14),
    )
    fresh_index = download_sensex_index(
        refresh_start.isoformat(),
        target.isoformat(),
        processed_path=DEFAULT_PATHS.processed / "index_refresh.csv",
    )
    if not fresh_index.empty:
        merge_csv(
            fresh_index,
            DEFAULT_PATHS.processed / "sensex_index.csv",
            ["date"],
        )

    index = load_index()
    options = load_options()
    quality = validate_all(index, options)
    validation_path = save_validation_report(quality)
    if quality.get("errors"):
        raise RuntimeError(
            "Daily data validation failed; historical files were preserved. "
            + json.dumps(quality["errors"])
        )

    trades, ranking, meta = run_strategy_suite(index, options)
    if trades.empty or ranking.empty:
        raise RuntimeError(
            meta.get("error", "No complete strategy trades; reports were not overwritten.")
        )

    output_dir = DEFAULT_PATHS.outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "strategy_trade_ledger.csv", index=False)
    ranking.to_csv(output_dir / "strategy_leaderboard.csv", index=False)

    periods = period_pnl(trades)
    for period, frame in periods.items():
        frame.to_csv(output_dir / f"{period}_strategy_pnl.csv", index=False)

    performance = strategy_performance_metrics(trades)
    if not performance.empty:
        performance.to_csv(output_dir / "strategy_performance_metrics.csv", index=False)
    walk_forward = walk_forward_evaluation(trades)
    with open(output_dir / "walk_forward_report.json", "w", encoding="utf-8") as handle:
        json.dump(walk_forward, handle, indent=2, default=str)

    run_info = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "target_data_date": target.isoformat(),
        "downloaded_option_dates": downloaded_days,
        "index_rows": int(len(index)),
        "option_rows": int(len(options)),
        "data_quality": quality,
        "strategy_run": meta,
        "strategy_count": int(len(ranking)),
        "performance_metrics": performance.to_dict("records") if not performance.empty else [],
        "walk_forward_status": walk_forward.get("status"),
        "pnl_semantics": "realized P&L grouped by trade exit date; not daily mark-to-market",
        "validation_report": str(validation_path),
    }
    with open(output_dir / "daily_run.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, default=str)

    # Compress accumulated history after a successful replay. Never delete the
    # only readable copy before the compressed replacement exists.
    options_path = DEFAULT_PATHS.processed / "sensex_options.csv"
    if options_path.exists():
        compressed_path = Path(str(options_path) + ".gz")
        options.to_csv(compressed_path, index=False, compression="gzip")
        if compressed_path.exists() and compressed_path.stat().st_size > 0:
            options_path.unlink()

    log.info("Daily strategy leaderboard:\n%s", ranking.to_string(index=False))
    return run_info


if __name__ == "__main__":
    run()
