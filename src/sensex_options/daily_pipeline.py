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
from sensex_options.daily_strategies import run_daily_strategy_suite, daily_holdout_evaluation
from sensex_options.intraday_research import download_sensex_intraday, evaluate_intraday_put_filters
from sensex_options.risk_managed_research import run_risk_managed_research
from sensex_options.entry_signal_research import run_entry_signal_research
from sensex_options.regime_research import run_regime_research

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("daily_pipeline")

OPTION_KEYS = ["trade_date", "expiry_date", "symbol", "strike", "option_type"]


def save_entry_signal_trade_ledger(trades: pd.DataFrame, output_dir: Path) -> Path:
    """Persist the expanded signal ledger losslessly below GitHub's 100 MB limit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = output_dir / "entry_signal_trade_ledger.csv"
    gzip_path = output_dir / "entry_signal_trade_ledger.csv.gz"
    trades.to_csv(gzip_path, index=False, compression="gzip")
    if legacy_path.exists():
        legacy_path.unlink()
    return gzip_path


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

    # Daily one-session strategies are a separate experiment from expiry trades.
    daily_trades, daily_ranking, daily_meta = run_daily_strategy_suite(index, options)
    daily_holdout = daily_holdout_evaluation(daily_trades)
    if not daily_trades.empty:
        daily_trades.to_csv(output_dir / "daily_strategy_trade_ledger.csv", index=False)
        daily_ranking.to_csv(output_dir / "daily_strategy_leaderboard.csv", index=False)
        with open(output_dir / "daily_strategy_holdout.json", "w", encoding="utf-8") as handle:
            json.dump(daily_holdout, handle, indent=2, default=str)
        daily_periods = daily_trades.copy()
        daily_periods["day"] = pd.to_datetime(daily_periods["exit_date"]).dt.strftime("%Y-%m-%d")
        daily_periods["week"] = pd.to_datetime(daily_periods["exit_date"]).dt.to_period("W-SUN").astype(str)
        daily_periods["month"] = pd.to_datetime(daily_periods["exit_date"]).dt.to_period("M").astype(str)
        for period in ("day", "week", "month"):
            daily_periods.groupby([period, "strategy"], as_index=False)["net_pnl"].sum().to_csv(
                output_dir / f"daily_options_{period}_pnl.csv", index=False
            )


    # Additional predeclared regime-gated analysis against observed option outcomes.
    regime_research_status = {"status": "not_run"}
    if not daily_trades.empty:
        regime_joined, regime_board, regime_report = run_regime_research(
            index, daily_trades, output_dir=output_dir, minimum_trades=30
        )
        regime_research_status = {
            "status": regime_report.get("status"),
            "candidates": int(regime_report.get("candidate_count", 0)),
            "prospective_paper_candidates": int(regime_report.get("prospective_paper_candidates", 0)),
        }

    # Risk controls are researched from the next-session option's own daily OHLC.
    # The dashboard only reads these persisted reports; it never runs this grid.
    risk_research_status = {"status": "not_run"}
    if not daily_trades.empty:
        risk_monthly, risk_board, risk_report = run_risk_managed_research(
            daily_trades, options
        )
        risk_monthly.to_csv(output_dir / "risk_managed_monthly.csv", index=False)
        risk_board.to_csv(output_dir / "risk_managed_leaderboard.csv", index=False)
        with open(output_dir / "risk_managed_research.json", "w", encoding="utf-8") as handle:
            json.dump(risk_report, handle, indent=2, default=str)
        risk_research_status = {
            "status": risk_report.get("status"),
            "variants": risk_report.get("total_variants_including_baselines", 0),
            "option_ohlc_coverage_pct": risk_report.get("option_ohlc_coverage_pct"),
            "approved_candidate_count": risk_report.get("approved_candidate_count", 0),
            "top_training_ranked_strategy": risk_report.get("top_training_ranked_strategy"),
        }

    # Explore alternative lagged entry signals and regime filters.
    # All signals use the last completed index session before the option entry.
    entry_signal_status = {"status": "not_run"}
    if not daily_trades.empty:
        signal_trades, signal_board, signal_selector_monthly, signal_report = run_entry_signal_research(
            daily_trades, index, options=options, holding_periods=(1, 2, 3, 5)
        )
        # Keep the complete ledger but compress it below GitHub's 100 MB file limit.
        save_entry_signal_trade_ledger(signal_trades, output_dir)
        signal_board.to_csv(output_dir / "entry_signal_leaderboard.csv", index=False)
        signal_selector_monthly.to_csv(output_dir / "entry_signal_selector_monthly.csv", index=False)
        signal_monthly_parts = []
        if not signal_report.get("status", "").startswith("insufficient") and not signal_trades.empty:
            from sensex_options.entry_signal_research import _monthly_stats
            candidate_monthly = _monthly_stats(signal_trades)
            if not candidate_monthly.empty:
                signal_monthly_parts.append(candidate_monthly)
        if not signal_selector_monthly.empty:
            signal_monthly_parts.append(signal_selector_monthly)
        if signal_monthly_parts:
            pd.concat(signal_monthly_parts, ignore_index=True, sort=False).to_csv(
                output_dir / "entry_signal_monthly.csv", index=False
            )
        else:
            pd.DataFrame(columns=[
                "strategy", "month", "trades", "gross_profit", "gross_loss",
                "net_pnl", "profit_factor", "target_pf", "target_testable", "target_met",
            ]).to_csv(output_dir / "entry_signal_monthly.csv", index=False)
        with open(output_dir / "entry_signal_research.json", "w", encoding="utf-8") as handle:
            json.dump(signal_report, handle, indent=2, default=str)
        entry_signal_status = {
            "status": signal_report.get("status"),
            "option_outcome_sessions": signal_report.get("option_outcome_sessions", 0),
            "candidate_rules_tested": signal_report.get("candidate_rules_tested", 0),
            "holding_periods_tested": signal_report.get("holding_periods_tested", []),
            "instrument_families_tested": signal_report.get("instrument_families_tested", []),
            "moneyness_variants_tested": signal_report.get("moneyness_variants_tested", []),
            "position_sizing": signal_report.get("position_sizing"),
            "overlapping_candidate_entries_skipped": signal_report.get("overlapping_candidate_entries_skipped", 0),
            "candidate_pass_count": signal_report.get("candidate_pass_count", 0),
            "training_end": signal_report.get("training_end"),
            "holding_periods_tested": signal_report.get("holding_periods_tested", []),
            "holdout_start": signal_report.get("holdout_start"),
            "top_training_rule": (
                {
                    "strategy": signal_report.get("top_training_rule", {}).get("strategy"),
                    "train_net_pnl": signal_report.get("top_training_rule", {}).get("train_net_pnl"),
                    "train_aggregate_pf": signal_report.get("top_training_rule", {}).get("train_aggregate_pf"),
                    "holdout_net_pnl": signal_report.get("top_training_rule", {}).get("holdout_net_pnl"),
                    "holdout_aggregate_pf": signal_report.get("top_training_rule", {}).get("holdout_aggregate_pf"),
                } if signal_report.get("top_training_rule") else None
            ),
            "rolling_selector_summary": signal_report.get("rolling_selector_summary", []),
        }

    # Intraday SENSEX context is collected by the scheduled workflow, never by Streamlit.
    # Yahoo's recent-window limit means this is a rolling 60-calendar-day dataset.
    intraday_status = {"status": "not_run"}
    try:
        intraday_bars = download_sensex_intraday(
            interval="5m", period="60d",
            output_path=output_dir / "sensex_intraday_5m.csv",
        )
        if not intraday_bars.empty and not daily_trades.empty:
            _, intraday_report, intraday_features = evaluate_intraday_put_filters(daily_trades, intraday_bars)
            intraday_features.to_csv(output_dir / "sensex_intraday_session_features.csv", index=False)
            with open(output_dir / "daily_put_intraday_validation.json", "w", encoding="utf-8") as handle:
                json.dump(intraday_report, handle, indent=2, default=str)
            if intraday_report.get("status") == "ok":
                pd.DataFrame(intraday_report.get("validation_metrics", [])).to_csv(
                    output_dir / "daily_put_intraday_validation.csv", index=False
                )
            intraday_status = {
                "status": intraday_report.get("status"),
                "bars": int(len(intraday_bars)),
                "matched_sessions": intraday_report.get("matched_sessions", 0),
                "validation_start": intraday_report.get("validation_start"),
                "validation_end": intraday_report.get("validation_end"),
                "note": intraday_report.get("note"),
            }
        else:
            pd.DataFrame(columns=["timestamp", "date", "open", "high", "low", "close", "volume"]).to_csv(
                output_dir / "sensex_intraday_5m.csv", index=False
            )
            pd.DataFrame().to_csv(output_dir / "sensex_intraday_session_features.csv", index=False)
            empty_report = {
                "status": "insufficient_data",
                "note": "Intraday download empty or daily option ledger unavailable.",
                "stop_loss_status": "Not tested: index intraday candles cannot validate option-premium stop-loss execution.",
            }
            with open(output_dir / "daily_put_intraday_validation.json", "w", encoding="utf-8") as handle:
                json.dump(empty_report, handle, indent=2)
            pd.DataFrame(columns=["rule", "trades", "net_pnl", "profit_factor", "expectancy", "median_trade_pnl", "max_drawdown"]).to_csv(
                output_dir / "daily_put_intraday_validation.csv", index=False
            )
            intraday_status = {
                "status": "insufficient_data",
                "bars": int(len(intraday_bars)),
                "note": empty_report["note"],
            }
    except Exception as exc:
        # Do not lose the daily BSE collection/replay if Yahoo intraday is temporarily unavailable.
        log.exception("Intraday SENSEX refresh failed")
        intraday_status = {"status": "download_error", "error": str(exc)}

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
        "daily_strategy_count": int(len(daily_ranking)),
        "daily_strategy_trades": int(len(daily_trades)),
        "daily_holdout_status": daily_holdout.get("status"),
        "risk_managed_research": risk_research_status,
        "regime_research": regime_research_status,
        "entry_signal_research": entry_signal_status,
        "intraday_research": intraday_status,
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
