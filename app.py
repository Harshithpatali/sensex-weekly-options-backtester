"""
SENSEX Weekly Options Backtester — Streamlit Dashboard
Professional dark financial-terminal style UI.
Does NOT download data automatically on startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sensex_options.config import BacktestConfig, CostConfig, DEFAULT_PATHS, SENSEX_OPTIONS_LAUNCH
from sensex_options.sources.index import download_sensex_index, load_index
from sensex_options.sources.bse import download_bse_options_range, load_options
from sensex_options.validation import validate_all, save_validation_report
from sensex_options.backtest import run_backtest
from sensex_options.reports import write_backtest_reports
from sensex_options.features import build_features
from sensex_options.strategies import run_strategy_suite, period_pnl
from sensex_options.analytics import strategy_performance_metrics, walk_forward_evaluation

st.set_page_config(
    page_title="SENSEX Weekly Options Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark terminal CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #00d4aa !important; }
    .metric-card {
        background: #1a1f2e; border: 1px solid #2a3142; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.5rem;
    }
    div[data-testid="stMetricValue"] { color: #00d4aa; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Overview",
    "Data Collection",
    "Data Validation",
    "Backtest Configuration",
    "Performance Dashboard",
    "Trade Ledger",
    "Reports",
    "Strategy Leaderboard",
    "Daily / Weekly / Monthly P&L",
    "Research & Walk-Forward",
]


def sidebar():
    st.sidebar.title("SENSEX Weekly Options")
    st.sidebar.caption("Research backtester · actual BSE data only")
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.markdown("---")
    st.sidebar.info(
        "⚠️ Research tool only. Not live trading. "
        "Zero-cost defaults are unrealistic. "
        "Never fabricates option premiums."
    )
    return page


def page_overview():
    st.title("SENSEX Weekly Options Backtester")
    st.markdown(
        "Historical long **Call** strategy on SENSEX weeklies using **actual** "
        "BSE derivatives bhavcopy premiums. Initial capital default: **₹10,000**."
    )
    DEFAULT_PATHS.ensure()
    index = load_index()
    options = load_options()
    report = validate_all(index, options)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial Capital (default)", "₹10,000")
    c2.metric("Index rows", report.get("index_rows", 0))
    c3.metric("Option rows", report.get("options_rows", 0))
    c4.metric("Available expiries", report.get("expiries", 0))
    c5, c6 = st.columns(2)
    c5.metric("Option-session coverage", f"{report.get('option_session_coverage_pct', 0) or 0:.1f}%")
    c6.metric("Expiries with CE + PE", f"{report.get('expiry_completeness_pct', 0) or 0:.1f}%")

    st.subheader("Data coverage")
    st.write(f"**Index:** {report.get('index_start', '—')} → {report.get('index_end', '—')}")
    st.write(f"**Options:** {report.get('options_start', '—')} → {report.get('options_end', '—')}")
    st.write(f"**Validation status:** `{report.get('status')}`")
    st.write(f"**SENSEX options approximate launch:** {SENSEX_OPTIONS_LAUNCH}")
    if report.get("warnings"):
        st.warning("\n".join(report["warnings"]))
    if report.get("errors"):
        st.error("\n".join(report["errors"]))


def page_data_collection():
    st.title("Data Collection")
    st.markdown(
        "Download data **on demand**. Yahoo Finance for SENSEX index; "
        "official BSE FO bhavcopy for options. Holidays and missing files are recorded."
    )
    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=pd.Timestamp(SENSEX_OPTIONS_LAUNCH).date())
    end = col2.date_input("End date", value=date.today())
    start_s, end_s = str(start), str(end)

    if st.button("Download SENSEX index (yfinance)", type="primary"):
        with st.spinner("Downloading index..."):
            try:
                df = download_sensex_index(start_s, end_s)
                st.success(f"Downloaded {len(df)} index rows.")
                st.dataframe(df.tail(10))
            except Exception as e:
                st.error(f"Index download failed: {e}")

    if st.button("Download BSE options bhavcopy"):
        st.info(
            "This may take a long time (one HTTP request per trading day). "
            "Progress is shown in the terminal if launched from CLI. "
            "Failed/holiday days are logged in the manifest."
        )
        with st.spinner("Downloading BSE FO files (this can take minutes)..."):
            try:
                df = download_bse_options_range(start_s, end_s, sleep=0.6)
                st.success(f"Parsed {len(df):,} SENSEX Call and Put option rows.")
                if not df.empty:
                    st.dataframe(df.head(20))
                manifest = DEFAULT_PATHS.reports / "options_collection_manifest.csv"
                if manifest.exists():
                    st.subheader("Collection manifest (sample)")
                    st.dataframe(pd.read_csv(manifest).tail(30))
            except Exception as e:
                st.error(f"Options download failed: {e}")
                st.markdown(
                    "**Manual fallback:** Place official BSE FO bhavcopy CSV/ZIP files "
                    f"into `{DEFAULT_PATHS.raw_options}` and re-run collection or parse via CLI."
                )


def page_validation():
    st.title("Data Validation")
    index = load_index()
    options = load_options()
    report = validate_all(index, options)
    save_validation_report(report)
    st.json(report)
    if not index.empty:
        st.subheader("Index sample")
        st.dataframe(index.tail(5))
    if not options.empty:
        st.subheader("Options sample")
        st.dataframe(options.head(10))
    if report.get("expiry_coverage"):
        st.subheader("Expiry completeness")
        coverage = pd.DataFrame(report["expiry_coverage"])
        st.dataframe(coverage, use_container_width=True)
        fig = px.bar(coverage, x="expiry_date", y=["call_rows", "put_rows"],
                     barmode="group", title="CE / PE observations by expiry")
        fig.update_layout(template="plotly_dark", xaxis_title="Expiry", yaxis_title="Option rows")
        st.plotly_chart(fig, use_container_width=True)


def page_backtest_config():
    st.title("Backtest Configuration")
    capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=10000.0, step=1000.0)
    c1, c2 = st.columns(2)
    start = c1.date_input("Start (expiry filter)", value=None)
    end = c2.date_input("End (expiry filter)", value=None)
    st.subheader("Transaction costs (research defaults = zero)")
    brokerage = st.number_input("Brokerage per lot (₹, round-trip flat)", min_value=0.0, value=0.0)
    stt = st.number_input("STT % on exit premium turnover", min_value=0.0, value=0.0, format="%.4f")
    entry_slip = st.number_input("Entry slippage (points)", min_value=0.0, value=0.0)
    exit_slip = st.number_input("Exit slippage (points)", min_value=0.0, value=0.0)
    max_lots = st.number_input("Max lots (0 = unlimited by capital)", min_value=0, value=0)

    if st.button("Run Backtest", type="primary"):
        cost = CostConfig(
            brokerage_per_lot=brokerage,
            stt_pct=stt / 100.0 if stt > 1 else stt,
            entry_slippage=entry_slip,
            exit_slippage=exit_slip,
        )
        cfg = BacktestConfig(
            initial_capital=capital,
            start_date=str(start) if start else None,
            end_date=str(end) if end else None,
            max_lots=max_lots if max_lots > 0 else None,
            cost=cost,
            assume_zero_costs=(brokerage == 0 and stt == 0 and entry_slip == 0 and exit_slip == 0),
        )
        with st.spinner("Running backtest..."):
            index = load_index()
            options = load_options()
            result = run_backtest(index, options, cfg)
            write_backtest_reports(result)
            st.session_state["bt_result"] = result
            st.success("Backtest complete. See Performance Dashboard and Trade Ledger.")
            st.json(result["summary"])


def page_performance():
    st.title("Performance Dashboard")
    result = st.session_state.get("bt_result")
    if result is None:
        # try load from disk
        summary_path = DEFAULT_PATHS.outputs / "backtest_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            trades = pd.read_csv(DEFAULT_PATHS.outputs / "trade_ledger.csv") if (DEFAULT_PATHS.outputs / "trade_ledger.csv").exists() else pd.DataFrame()
            equity = pd.read_csv(DEFAULT_PATHS.outputs / "equity_curve.csv") if (DEFAULT_PATHS.outputs / "equity_curve.csv").exists() else pd.DataFrame()
            result = {"summary": summary, "trades": trades, "equity_curve": equity}
        else:
            st.info("Run a backtest first.")
            return

    summary = result["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net P&L", f"₹{summary.get('total_net_pnl', 0):,.2f}")
    c2.metric("Gross P&L", f"₹{summary.get('total_gross_pnl', 0):,.2f}")
    c3.metric("Win rate", f"{summary.get('win_rate_pct', 0):.1f}%")
    c4.metric("Final capital", f"₹{summary.get('ending_capital', 0):,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Profit factor", str(summary.get("profit_factor", "—")))
    c6.metric("Max drawdown", f"₹{summary.get('max_drawdown', 0):,.2f}")
    c7.metric("Completed trades", summary.get("completed_trades", 0))
    c8.metric("Skipped trades", summary.get("skipped_trades", 0))

    equity = result.get("equity_curve", pd.DataFrame()).copy()
    if not equity.empty and "capital" in equity.columns:
        if "date" in equity.columns:
            equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
            equity = equity.dropna(subset=["date"]).sort_values("date")
        if not equity.empty and "date" in equity.columns:
            fig = px.line(equity, x="date", y="capital", markers=True, title="Backtest equity curve")
        else:
            fig = px.line(equity, y="capital", markers=True, title="Backtest equity curve")
        fig.update_layout(template="plotly_dark", yaxis_title="Capital (₹)")
        st.plotly_chart(fig, use_container_width=True)

        if not equity.empty:
            equity["peak"] = equity["capital"].cummax()
            equity["drawdown"] = equity["capital"] - equity["peak"]
            fig = px.area(equity, x="date" if "date" in equity.columns else equity.index,
                          y="drawdown", title="Equity drawdown")
            fig.update_layout(template="plotly_dark", yaxis_title="Drawdown (₹)")
            st.plotly_chart(fig, use_container_width=True)

    trades_for_charts = result.get("trades", pd.DataFrame()).copy()
    if not trades_for_charts.empty and "status" in trades_for_charts.columns:
        completed = trades_for_charts[trades_for_charts["status"] == "completed"].copy()
        if not completed.empty:
            completed["net_pnl"] = pd.to_numeric(completed["net_pnl"], errors="coerce")
            completed["exit_date"] = pd.to_datetime(completed["exit_date"], errors="coerce")
            daily = completed.dropna(subset=["exit_date"]).groupby("exit_date", as_index=False)["net_pnl"].sum()
            fig = px.bar(daily, x="exit_date", y="net_pnl", title="Realized daily P&L")
            fig.update_layout(template="plotly_dark", yaxis_title="Net P&L (₹)")
            st.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(completed, x="net_pnl", nbins=30, title="Trade P&L distribution")
            fig.update_layout(template="plotly_dark", xaxis_title="Net P&L per trade (₹)")
            st.plotly_chart(fig, use_container_width=True)

    annual_path = DEFAULT_PATHS.outputs / "annual_summary.csv"
    if annual_path.exists():
        st.subheader("Yearly performance")
        st.dataframe(pd.read_csv(annual_path))


def page_ledger():
    st.title("Trade Ledger")
    path = DEFAULT_PATHS.outputs / "trade_ledger.csv"
    if not path.exists():
        st.info("No trade ledger yet. Run a backtest.")
        return
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download trade ledger CSV", df.to_csv(index=False), "trade_ledger.csv", "text/csv")


def page_reports():
    st.title("Reports")
    out = DEFAULT_PATHS.outputs
    files = list(out.glob("*")) if out.exists() else []
    if not files:
        st.info("No reports generated yet.")
        return
    for f in files:
        st.write(f"**{f.name}** — {f.stat().st_size} bytes")
        if f.suffix == ".csv":
            st.download_button(f"Download {f.name}", f.read_bytes(), f.name, "text/csv", key=f.name)
        elif f.suffix == ".json":
            st.download_button(f"Download {f.name}", f.read_bytes(), f.name, "application/json", key=f.name)

    val = DEFAULT_PATHS.reports / "data_validation.json"
    if val.exists():
        st.download_button("Download validation report", val.read_bytes(), "data_validation.json", key="val")



def page_strategy_leaderboard():
    st.title("Strategy Leaderboard")
    st.caption("Historical simulated ranking by net P&L. Ranking is descriptive, not a forecast or a live recommendation.")
    path = DEFAULT_PATHS.outputs / "strategy_leaderboard.csv"
    ledger_path = DEFAULT_PATHS.outputs / "strategy_trade_ledger.csv"
    if not path.exists():
        st.info("No strategy replay is available yet. Run the historical bootstrap workflow first.")
        return

    board = pd.read_csv(path)
    metrics_path = DEFAULT_PATHS.outputs / "strategy_performance_metrics.csv"
    if metrics_path.exists():
        performance = pd.read_csv(metrics_path)
        extra = [column for column in ("strategy", "expectancy", "sharpe_trade_annualized") if column in performance.columns]
        if len(extra) > 1:
            board = board.merge(performance[extra], on="strategy", how="left", suffixes=("", "_risk"))
    if board.empty:
        st.warning("The leaderboard file exists but contains no completed strategy results.")
        return

    top = board.sort_values("net_pnl", ascending=False).head(5)
    c1, c2, c3, c4 = st.columns(4)
    best = top.iloc[0]
    c1.metric("Top historical strategy", str(best["strategy"]))
    c2.metric("Historical net P&L", f"₹{float(best['net_pnl']):,.2f}")
    c3.metric("Win rate", f"{float(best.get('win_rate_pct', 0)):.1f}%")
    c4.metric("Maximum drawdown", f"₹{abs(float(best.get('max_drawdown', 0))):,.2f}")

    st.subheader("Top 5 by historical net P&L")
    st.dataframe(top, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top.sort_values("net_pnl"), x="net_pnl", y="strategy", orientation="h",
                     title="Net P&L by strategy", text_auto=".2s")
        fig.update_layout(template="plotly_dark", xaxis_title="Net P&L (₹)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "max_drawdown" in top.columns:
            risk = top.copy()
            risk["absolute_drawdown"] = risk["max_drawdown"].abs()
            fig = px.bar(risk.sort_values("absolute_drawdown"), x="absolute_drawdown", y="strategy",
                         orientation="h", title="Historical maximum drawdown", text_auto=".2s")
            fig.update_layout(template="plotly_dark", xaxis_title="Drawdown (₹)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        if not ledger.empty:
            ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce")
            ledger["net_pnl"] = pd.to_numeric(ledger["net_pnl"], errors="coerce").fillna(0)
            st.subheader("Strategy performance over time")
            selected = st.selectbox("Choose strategy", sorted(ledger["strategy"].dropna().unique()))
            view = ledger[ledger["strategy"] == selected].sort_values("exit_date").copy()
            view["cumulative_pnl"] = view["net_pnl"].cumsum()
            fig = px.line(view, x="exit_date", y="cumulative_pnl", markers=True,
                          title=f"{selected}: cumulative realized P&L")
            fig.update_layout(template="plotly_dark", xaxis_title="Exit date", yaxis_title="Cumulative P&L (₹)")
            st.plotly_chart(fig, use_container_width=True)
            if {"starting_capital", "ending_capital"}.issubset(view.columns):
                fig = px.line(view, x="exit_date", y="ending_capital", markers=True,
                              title=f"{selected}: isolated strategy capital curve")
                fig.update_layout(template="plotly_dark", xaxis_title="Exit date", yaxis_title="Capital (₹)")
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Trade ledger")
            st.dataframe(view.sort_values("exit_date", ascending=False), use_container_width=True)
            st.download_button("Download selected strategy trades", view.to_csv(index=False),
                               f"{selected.lower()}_trades.csv", "text/csv")
    st.info("Each strategy is simulated with its own sequential capital curve. These are not five simultaneous strategies sharing one ₹10,000 portfolio.")


def page_period_pnl():
    st.title("Daily / Weekly / Monthly Strategy P&L")
    st.caption("Realized P&L grouped by trade exit date. This is not mark-to-market P&L for open positions.")
    periods = [
        ("Daily", "daily_strategy_pnl.csv", "day"),
        ("Weekly", "weekly_strategy_pnl.csv", "week"),
        ("Monthly", "monthly_strategy_pnl.csv", "month"),
    ]
    tabs = st.tabs([item[0] for item in periods])
    for tab, (label, filename, key) in zip(tabs, periods):
        with tab:
            path = DEFAULT_PATHS.outputs / filename
            if not path.exists():
                st.info(f"{filename} will appear after a successful strategy replay.")
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                st.info("No realized trades in this period.")
                continue
            frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0)
            strategies = ["All strategies"] + sorted(frame["strategy"].dropna().unique().tolist())
            selected = st.selectbox(f"{label} strategy filter", strategies, key=f"{key}_strategy_filter")
            view = frame if selected == "All strategies" else frame[frame["strategy"] == selected]
            if selected == "All strategies":
                chart = view.groupby(key, as_index=False)["net_pnl"].sum()
            else:
                chart = view.groupby(key, as_index=False)["net_pnl"].sum()
            chart = chart.sort_values(key)
            fig = px.bar(chart, x=key, y="net_pnl", title=f"{label} realized P&L — {selected}")
            fig.update_layout(template="plotly_dark", xaxis_title=label, yaxis_title="Net P&L (₹)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view.sort_values(key, ascending=False), use_container_width=True)
            st.download_button(f"Download {label.lower()} P&L CSV", view.to_csv(index=False),
                               filename, "text/csv", key=f"{key}_download")



def page_research_lab():
    st.title("Research & Walk-Forward")
    st.caption("Cost-aware comparison, chronological holdout and rolling expiry evaluation. Fixed rule strategies are not trained or optimized.")
    index = load_index()
    options = load_options()
    if index.empty or options.empty:
        st.warning("Historical index and options data are required. Run the historical bootstrap workflow first.")
        return
    validation = validate_all(index, options)
    st.subheader("Historical coverage")
    a, b, c, d = st.columns(4)
    a.metric("Option rows", f"{len(options):,}")
    b.metric("Available expiry cycles", validation.get("expiries", 0))
    c.metric("Session coverage", f"{validation.get('option_session_coverage_pct', 0) or 0:.1f}%")
    d.metric("Complete matured expiries", f"{validation.get('expiry_completeness_pct', 0) or 0:.1f}%")
    st.caption(f"Matured expiries: {validation.get('completed_expiry_count', 0)} · Complete for backtest: {validation.get('complete_expiries', 0)} · Pending future expiries: {validation.get('pending_expiries', 0)}")
    if validation.get("completed_expiry_count", 0) < 30:
        st.warning("Fewer than 30 matured expiry cycles are available. Backfill the full history before relying on the statistics.")
    st.markdown("[Open the historical bootstrap workflow](https://github.com/Harshithpatali/sensex-weekly-options-backtester/actions/workflows/bootstrap-history.yml)")
    if validation.get("missing_option_sessions"):
        with st.expander("Missing option sessions"):
            st.write(", ".join(validation["missing_option_sessions"]))

    st.subheader("Execution and cost assumptions")
    x1, x2, x3 = st.columns(3)
    capital = x1.number_input("Capital per strategy (₹)", min_value=1000.0, value=10000.0, step=1000.0, key="research_capital")
    brokerage = x2.number_input("Brokerage (₹ / lot / order)", min_value=0.0, value=20.0, step=5.0, key="research_brokerage")
    brokerage_pct = x3.number_input("Brokerage (% turnover)", min_value=0.0, value=0.0, format="%.4f", key="research_brokerage_pct")
    y1, y2, y3 = st.columns(3)
    stt = y1.number_input("STT (% sell premium turnover)", min_value=0.0, value=0.10, format="%.4f", key="research_stt")
    exchange = y2.number_input("Exchange charges (% turnover)", min_value=0.0, value=0.03, format="%.4f", key="research_exchange")
    gst = y3.number_input("GST (% brokerage + exchange)", min_value=0.0, value=18.0, format="%.2f", key="research_gst")
    z1, z2, z3 = st.columns(3)
    stamp = z1.number_input("Stamp duty (% buy turnover)", min_value=0.0, value=0.003, format="%.4f", key="research_stamp")
    entry_slip = z2.number_input("Entry slippage (points / leg)", min_value=0.0, value=0.5, step=0.1, key="research_entry_slip")
    exit_slip = z3.number_input("Exit slippage (points / leg)", min_value=0.0, value=0.5, step=0.1, key="research_exit_slip")
    other = st.number_input("Other charges (₹ / lot / order)", min_value=0.0, value=0.0, key="research_other")
    e1, e2 = st.columns(2)
    spread = e1.number_input("Full bid-ask spread (points / leg)", min_value=0.0, value=1.0, step=0.1, key="research_spread")
    impact = e2.number_input("Market impact (points / side / leg)", min_value=0.0, value=0.25, step=0.05, key="research_impact")
    t1, t2 = st.columns(2)
    train_expiries = t1.number_input("Walk-forward context expiries", min_value=2, max_value=100, value=8, step=1)
    test_expiries = t2.number_input("Test expiries per fold", min_value=1, max_value=52, value=4, step=1)
    st.caption("Rates are editable research assumptions, not broker quotes or tax advice. Verify the current BSE and broker tariff before relying on net P&L. Spread is charged half on entry and half on exit; slippage and market impact are applied adversely to each leg on both sides. BSE EOD files do not contain historical order-book quotes, so these are user-controlled execution proxies.")
    if st.button("Run cost-aware strategy research", type="primary"):
        cost = CostConfig(
            brokerage_per_lot=brokerage, brokerage_pct=brokerage_pct / 100.0,
            stt_pct=stt / 100.0, exchange_charges_pct=exchange / 100.0,
            gst_pct=gst / 100.0, stamp_duty_pct=stamp / 100.0,
            other_charges=other, entry_slippage=entry_slip, exit_slippage=exit_slip,
            spread_points=spread, market_impact_points=impact,
        )
        with st.spinner("Replaying all available expiry cycles..."):
            trades, ranking, meta = run_strategy_suite(index, options, capital=capital, cost=cost)
            if trades.empty:
                st.error(meta.get("error", "No complete trades were produced."))
                return
            metrics = strategy_performance_metrics(trades, capital)
            walk = walk_forward_evaluation(trades, capital, int(train_expiries), int(test_expiries))
            DEFAULT_PATHS.outputs.mkdir(parents=True, exist_ok=True)
            trades.to_csv(DEFAULT_PATHS.outputs / "strategy_trade_ledger.csv", index=False)
            ranking.to_csv(DEFAULT_PATHS.outputs / "strategy_leaderboard.csv", index=False)
            metrics.to_csv(DEFAULT_PATHS.outputs / "strategy_performance_metrics.csv", index=False)
            for period_name, period_frame in period_pnl(trades).items():
                period_frame.to_csv(DEFAULT_PATHS.outputs / f"{period_name}_strategy_pnl.csv", index=False)
            with open(DEFAULT_PATHS.outputs / "walk_forward_report.json", "w", encoding="utf-8") as handle:
                json.dump(walk, handle, indent=2, default=str)
            st.session_state["research_trades"] = trades
            st.session_state["research_metrics"] = metrics
            st.session_state["research_walk_forward"] = walk
            st.success(f"Completed {len(trades):,} strategy trades across {trades['expiry_date'].nunique()} expiry cycles.")
            if trades["expiry_date"].nunique() < 30:
                st.warning("Small sample: these results are not statistically reliable. Backfill additional historical expiries.")
            st.json(meta)

    metrics = st.session_state.get("research_metrics")
    trades = st.session_state.get("research_trades")
    walk = st.session_state.get("research_walk_forward")
    metrics_path = DEFAULT_PATHS.outputs / "strategy_performance_metrics.csv"
    walk_path = DEFAULT_PATHS.outputs / "walk_forward_report.json"
    if metrics is None and metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    if walk is None and walk_path.exists():
        with open(walk_path, encoding="utf-8") as handle:
            walk = json.load(handle)
    if metrics is not None and not metrics.empty:
        st.subheader("Strategy performance comparison")
        st.dataframe(metrics, use_container_width=True)
        st.caption("Sharpe is based on trade-level returns annualised at 52 expiry cycles/year. It is approximate, not a daily mark-to-market Sharpe ratio.")
        fig = px.bar(metrics, x="strategy", y="expectancy", title="Net expectancy per completed trade")
        fig.update_layout(template="plotly_dark", yaxis_title="Expectancy (₹)")
        st.plotly_chart(fig, use_container_width=True)
    if walk:
        st.subheader("Out-of-sample and walk-forward results")
        st.write(f"**Status:** {walk.get('status', 'not run')}")
        if walk.get("status") == "ok":
            st.write(f"**Chronological holdout:** {walk.get('holdout_start')} to {walk.get('holdout_end')} ({walk.get('holdout_expiries')} expiry cycles)")
            st.dataframe(pd.DataFrame(walk.get("holdout_metrics", [])), use_container_width=True)
            for i, fold in enumerate(walk.get("walk_forward_folds", []), start=1):
                with st.expander(f"Walk-forward fold {i}: {', '.join(fold['test_expiries'])}"):
                    st.dataframe(pd.DataFrame(fold.get("metrics", [])), use_container_width=True)
        else:
            st.info(walk.get("note", "Insufficient expiry history for walk-forward testing."))
        st.caption(walk.get("methodology", ""))
    if trades is None and (DEFAULT_PATHS.outputs / "strategy_trade_ledger.csv").exists():
        trades = pd.read_csv(DEFAULT_PATHS.outputs / "strategy_trade_ledger.csv")
    if trades is not None and not trades.empty:
        st.subheader("Research trade ledger")
        st.dataframe(trades, use_container_width=True)
        st.download_button("Download costed strategy ledger", trades.to_csv(index=False), "strategy_trade_ledger.csv", "text/csv")


def main():
    page = sidebar()
    if page == "Overview":
        page_overview()
    elif page == "Data Collection":
        page_data_collection()
    elif page == "Data Validation":
        page_validation()
    elif page == "Backtest Configuration":
        page_backtest_config()
    elif page == "Performance Dashboard":
        page_performance()
    elif page == "Trade Ledger":
        page_ledger()
    elif page == "Reports":
        page_reports()
    elif page == "Strategy Leaderboard":
        page_strategy_leaderboard()
    elif page == "Daily / Weekly / Monthly P&L":
        page_period_pnl()
    elif page == "Research & Walk-Forward":
        page_research_lab()


if __name__ == "__main__":
    main()
