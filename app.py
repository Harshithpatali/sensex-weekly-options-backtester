from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sensex_options.config import BacktestConfig, CostConfig, DEFAULT_PATHS, SENSEX_OPTIONS_LAUNCH
from sensex_options.sources.index import load_index
from sensex_options.sources.bse import load_options
from sensex_options.validation import validate_all, save_validation_report
from sensex_options.backtest import run_backtest
from sensex_options.reports import write_backtest_reports
from sensex_options.features import build_features
from sensex_options.strategies import run_strategy_suite, period_pnl
from sensex_options.analytics import strategy_performance_metrics, walk_forward_evaluation
from sensex_options.daily_strategies import run_daily_strategy_suite, daily_holdout_evaluation
from sensex_options.intraday_research import (
    latest_previous_close,
)

st.set_page_config(
    page_title="SENSEX Weekly Options Backtester",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Quant-Terminal Styling CSS
st.markdown(
    """
    <style>
    /* Main Theme & Background */
    .stApp { background-color: #07090e; color: #d1d5db; font-family: 'Inter', -apple-system, sans-serif; }
    
    /* Typography */
    h1, h2, h3, h4 { color: #f3f4f6 !important; font-weight: 600; letter-spacing: -0.025em; }
    h1 { font-size: 1.85rem !important; border-bottom: 1px solid #1f293d; padding-bottom: 0.5rem; }
    h2 { font-size: 1.35rem !important; margin-top: 1.5rem !important; }
    
    /* Sidebar Styling */
    div[data-testid="stSidebar"] { background-color: #0b0f19; border-right: 1px solid #1f293d; }
    div[data-testid="stSidebar"] .stRadio label { color: #9ca3af; font-size: 0.9rem; }
    div[data-testid="stSidebar"] .stRadio label:hover { color: #00d4aa; }
    
    /* Metrics Container Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f172a 0%, #0b0f19 100%);
        border: 1px solid #1f293d;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricValue"] { color: #00d4aa !important; font-family: 'JetBrains Mono', monospace; font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Tables & Dataframes */
    div[data-testid="stDataFrame"] { border: 1px solid #1f293d; border-radius: 6px; overflow: hidden; }
    
    /* Buttons & Inputs */
    .stButton>button {
        background: linear-gradient(135deg, #00d4aa 0%, #00a884 100%);
        color: #07090e;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 0 12px rgba(0, 212, 170, 0.4);
    }
    
    /* Alerts and Warnings */
    div.stAlert { background-color: #0f172a; border: 1px solid #1f293d; color: #d1d5db; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Overview",
    "Data Validation",
    "Backtest Configuration",
    "Performance Dashboard",
    "Trade Ledger",
    "Reports",
    "Strategy Leaderboard",
    "Daily / Weekly / Monthly P&L",
    "Daily Options Research",
    "Research & Walk-Forward",
]


def sidebar():
    st.sidebar.title("⚡ SENSEX Options")
    st.sidebar.caption("Institutional Quant Research Terminal")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "⚠️ **Research Terminal Only**\n\n"
        "• Simulated historical execution.\n"
        "• Zero-cost assumptions are unrealistic.\n"
        "• Never fabricates option premiums."
    )
    return page


def page_overview():
    st.title("SENSEX Weekly Options Backtester")
    st.markdown(
        "Quantitative research platform for **daily Call and Put trades** and expiry-based SENSEX options strategies "
        "driven by observed BSE derivatives bhavcopy data. Initial capital default: **₹10,000 per isolated strategy**."
    )
    DEFAULT_PATHS.ensure()
    index = load_index()
    options = load_options()
    report = validate_all(index, options)
    close_info = latest_previous_close(index)
    if close_info.get("status") == "ok":
        p1, p2, p3 = st.columns(3)
        p1.metric("SENSEX Previous Close", f"₹{close_info['close']:,.2f}", help=f"Latest completed session: {close_info['date']}")
        p2.metric("Previous Session Change", f"{close_info['change']:+,.2f}" if close_info.get("change") is not None else "—")
        p3.metric("Change (%)", f"{close_info['change_pct']:+.2f}%" if close_info.get("change_pct") is not None else "—")
        st.caption(f"Previous close date: {close_info['date']} · Source: saved daily ^BSESN history (Yahoo Finance).")

    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial Capital", "₹10,000")
    c2.metric("Index Rows", f"{report.get('index_rows', 0):,}")
    c3.metric("Option Rows", f"{report.get('options_rows', 0):,}")
    c4.metric("Available Expiries", report.get("expiries", 0))
    
    c5, c6 = st.columns(2)
    c5.metric("Option-Session Coverage", f"{report.get('option_session_coverage_pct', 0) or 0:.1f}%")
    c6.metric("Complete Matured Expiries", f"{report.get('expiry_completeness_pct', 0) or 0:.1f}%")
    st.caption(f"Matured: {report.get('completed_expiry_count', 0)} · Complete: {report.get('complete_expiries', 0)} · Pending: {report.get('pending_expiries', 0)}")

    st.subheader("Data Coverage Timeline")
    st.write(f"**Index Window:** `{report.get('index_start', '—')}` → `{report.get('index_end', '—')}`")
    st.write(f"**Options Window:** `{report.get('options_start', '—')}` → `{report.get('options_end', '—')}`")
    st.write(f"**Validation Status:** `{report.get('status')}`")
    st.write(f"**SENSEX Options Approximate Launch:** `{SENSEX_OPTIONS_LAUNCH}`")
    if report.get("warnings"):
        st.warning("\n".join(report["warnings"]))
    if report.get("errors"):
        st.error("\n".join(report["errors"]))


def page_validation():
    st.title("Data Validation & Diagnostics")
    index = load_index()
    options = load_options()
    report = validate_all(index, options)
    save_validation_report(report)
    
    with st.expander("View Full JSON Report", expanded=False):
        st.json(report)
        
    if not index.empty:
        st.subheader("Index Data Sample")
        st.dataframe(index.tail(5), use_container_width=True)
    if not options.empty:
        st.subheader("Options Data Sample")
        st.dataframe(options.head(10), use_container_width=True)
    if report.get("expiry_coverage"):
        st.subheader("Expiry Completeness Distribution")
        coverage = pd.DataFrame(report["expiry_coverage"])
        st.dataframe(coverage, use_container_width=True)
        fig = px.bar(coverage, x="expiry_date", y=["call_rows", "put_rows"],
                     barmode="group", title="CE / PE Observations by Expiry")
        fig.update_layout(template="plotly_dark", xaxis_title="Expiry Date", yaxis_title="Option Rows", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def page_backtest_config():
    st.title("Backtest Configuration Engine")
    capital = st.number_input("Initial Capital (₹)", min_value=1000.0, value=10000.0, step=1000.0)
    c1, c2 = st.columns(2)
    start = c1.date_input("Start Expiry Filter", value=None)
    end = c2.date_input("End Expiry Filter", value=None)
    
    st.subheader("Transaction Costs & Slippage (Research Defaults = Zero)")
    brokerage = st.number_input("Brokerage per Lot per Order (₹)", min_value=0.0, value=0.0)
    stt = st.number_input("STT % on Exit Premium Turnover", min_value=0.0, value=0.0, format="%.4f")
    entry_slip = st.number_input("Entry Slippage (Points)", min_value=0.0, value=0.0)
    exit_slip = st.number_input("Exit Slippage (Points)", min_value=0.0, value=0.0)
    max_lots = st.number_input("Max Lots (0 = Unlimited by Capital)", min_value=0, value=0)

    if st.button("Execute Backtest Simulation", type="primary"):
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
        with st.spinner("Running high-performance backtest simulation..."):
            index = load_index()
            options = load_options()
            result = run_backtest(index, options, cfg)
            write_backtest_reports(result)
            st.session_state["bt_result"] = result
            st.success("Backtest simulation complete. Review metrics in Performance Dashboard.")
            st.json(result["summary"])


def page_performance():
    st.title("Performance & Risk Dashboard")
    result = st.session_state.get("bt_result")
    if result is None:
        summary_path = DEFAULT_PATHS.outputs / "backtest_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            trades = pd.read_csv(DEFAULT_PATHS.outputs / "trade_ledger.csv") if (DEFAULT_PATHS.outputs / "trade_ledger.csv").exists() else pd.DataFrame()
            equity = pd.read_csv(DEFAULT_PATHS.outputs / "equity_curve.csv") if (DEFAULT_PATHS.outputs / "equity_curve.csv").exists() else pd.DataFrame()
            result = {"summary": summary, "trades": trades, "equity_curve": equity}
        else:
            st.info("No active backtest results found. Configure and execute a backtest first.")
            return

    summary = result["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net P&L", f"₹{summary.get('total_net_pnl', 0):,.2f}")
    c2.metric("Gross P&L", f"₹{summary.get('total_gross_pnl', 0):,.2f}")
    c3.metric("Win Rate", f"{summary.get('win_rate_pct', 0):.1f}%")
    c4.metric("Ending Capital", f"₹{summary.get('ending_capital', 0):,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Profit Factor", str(summary.get("profit_factor", "—")))
    c6.metric("Max Drawdown", f"₹{summary.get('max_drawdown', 0):,.2f}")
    c7.metric("Completed Trades", summary.get("completed_trades", 0))
    c8.metric("Skipped Trades", summary.get("skipped_trades", 0))

    equity = result.get("equity_curve", pd.DataFrame()).copy()
    if not equity.empty and "capital" in equity.columns:
        if "date" in equity.columns:
            equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
            equity = equity.dropna(subset=["date"]).sort_values("date")
        if not equity.empty and "date" in equity.columns:
            fig = px.line(equity, x="date", y="capital", markers=True, title="Backtest Equity Curve")
        else:
            fig = px.line(equity, y="capital", markers=True, title="Backtest Equity Curve")
        fig.update_layout(template="plotly_dark", yaxis_title="Capital (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if not equity.empty:
            equity["peak"] = equity["capital"].cummax()
            equity["drawdown"] = equity["capital"] - equity["peak"]
            fig = px.area(equity, x="date" if "date" in equity.columns else equity.index,
                          y="drawdown", title="Strategy Equity Drawdown Profile")
            fig.update_layout(template="plotly_dark", yaxis_title="Drawdown (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    trades_for_charts = result.get("trades", pd.DataFrame()).copy()
    if not trades_for_charts.empty and "status" in trades_for_charts.columns:
        completed = trades_for_charts[trades_for_charts["status"] == "completed"].copy()
        if not completed.empty:
            completed["net_pnl"] = pd.to_numeric(completed["net_pnl"], errors="coerce")
            completed["exit_date"] = pd.to_datetime(completed["exit_date"], errors="coerce")
            daily = completed.dropna(subset=["exit_date"]).groupby("exit_date", as_index=False)["net_pnl"].sum()
            fig = px.bar(daily, x="exit_date", y="net_pnl", title="Realized Daily P&L Distribution")
            fig.update_layout(template="plotly_dark", yaxis_title="Net P&L (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(completed, x="net_pnl", nbins=30, title="Trade P&L Spread Histogram")
            fig.update_layout(template="plotly_dark", xaxis_title="Net P&L per Trade (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    annual_path = DEFAULT_PATHS.outputs / "annual_summary.csv"
    if annual_path.exists():
        st.subheader("Yearly Aggregate Performance")
        st.dataframe(pd.read_csv(annual_path), use_container_width=True)


def page_ledger():
    st.title("Execution Trade Ledger")
    path = DEFAULT_PATHS.outputs / "trade_ledger.csv"
    if not path.exists():
        st.info("No trade ledger records found. Run a backtest first.")
        return
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download Trade Ledger CSV", df.to_csv(index=False), "trade_ledger.csv", "text/csv")


def page_reports():
    st.title("Generated Diagnostic Reports")
    out = DEFAULT_PATHS.outputs
    files = list(out.glob("*")) if out.exists() else []
    if not files:
        st.info("No reports generated yet.")
        return
    for f in files:
        col_r1, col_r2 = st.columns([4, 1])
        col_r1.write(f"**{f.name}** — `{f.stat().st_size:,} bytes`")
        if f.suffix == ".csv":
            col_r2.download_button("Download CSV", f.read_bytes(), f.name, "text/csv", key=f.name)
        elif f.suffix == ".json":
            col_r2.download_button("Download JSON", f.read_bytes(), f.name, "application/json", key=f.name)

    val = DEFAULT_PATHS.reports / "data_validation.json"
    if val.exists():
        st.download_button("📥 Download Full Validation Report", val.read_bytes(), "data_validation.json", key="val")


def page_strategy_leaderboard():
    st.title("Strategy Leaderboard Matrix")
    st.caption("Historical simulated ranking by net P&L. Ranking is descriptive, not a forecast or live trading recommendation.")
    path = DEFAULT_PATHS.outputs / "strategy_leaderboard.csv"
    ledger_path = DEFAULT_PATHS.outputs / "strategy_trade_ledger.csv"
    if not path.exists():
        st.info("No strategy replay data available yet. Execute the historical bootstrap workflow.")
        return

    board = pd.read_csv(path)
    metrics_path = DEFAULT_PATHS.outputs / "strategy_performance_metrics.csv"
    if metrics_path.exists():
        performance = pd.read_csv(metrics_path)
        extra = [column for column in ("strategy", "expectancy", "sharpe_trade_annualized") if column in performance.columns]
        if len(extra) > 1:
            board = board.merge(performance[extra], on="strategy", how="left", suffixes=("", "_risk"))
    if board.empty:
        st.warning("Leaderboard file contains no records.")
        return

    top = board.sort_values("net_pnl", ascending=False).head(5)
    c1, c2, c3, c4 = st.columns(4)
    best = top.iloc[0]
    c1.metric("Top Strategy", str(best["strategy"]))
    c2.metric("Net P&L", f"₹{float(best['net_pnl']):,.2f}")
    c3.metric("Win Rate", f"{float(best.get('win_rate_pct', 0)):.1f}%")
    c4.metric("Max Drawdown", f"₹{abs(float(best.get('max_drawdown', 0))):,.2f}")

    st.subheader("Top Strategies by Net P&L")
    st.dataframe(top, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top.sort_values("net_pnl"), x="net_pnl", y="strategy", orientation="h",
                     title="Net P&L by Strategy", text_auto=".2s")
        fig.update_layout(template="plotly_dark", xaxis_title="Net P&L (₹)", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "max_drawdown" in top.columns:
            risk = top.copy()
            risk["absolute_drawdown"] = risk["max_drawdown"].abs()
            fig = px.bar(risk.sort_values("absolute_drawdown"), x="absolute_drawdown", y="strategy",
                         orientation="h", title="Historical Max Drawdown", text_auto=".2s")
            fig.update_layout(template="plotly_dark", xaxis_title="Drawdown (₹)", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        if not ledger.empty:
            ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce")
            ledger["net_pnl"] = pd.to_numeric(ledger["net_pnl"], errors="coerce").fillna(0)
            st.subheader("Granular Strategy Timeline Analysis")
            selected = st.selectbox("Select Strategy Profile", sorted(ledger["strategy"].dropna().unique()))
            view = ledger[ledger["strategy"] == selected].sort_values("exit_date").copy()
            view["cumulative_pnl"] = view["net_pnl"].cumsum()
            fig = px.line(view, x="exit_date", y="cumulative_pnl", markers=True,
                          title=f"{selected}: Cumulative Realized P&L")
            fig.update_layout(template="plotly_dark", xaxis_title="Exit Date", yaxis_title="Cumulative P&L (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            if {"starting_capital", "ending_capital"}.issubset(view.columns):
                fig = px.line(view, x="exit_date", y="ending_capital", markers=True,
                              title=f"{selected}: Isolated Strategy Capital Curve")
                fig.update_layout(template="plotly_dark", xaxis_title="Exit Date", yaxis_title="Capital (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view.sort_values("exit_date", ascending=False), use_container_width=True)
            st.download_button("📥 Download Selected Strategy Trades", view.to_csv(index=False),
                               f"{selected.lower()}_trades.csv", "text/csv")


def page_period_pnl():
    st.title("Periodic Realized P&L Breakdown")
    st.caption("Realized P&L grouped by trade exit date (Daily / Weekly / Monthly).")
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
                st.info(f"{filename} will populate after running a strategy execution sweep.")
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                st.info("No realized trades recorded in this period.")
                continue
            frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0)
            strategies = ["All strategies"] + sorted(frame["strategy"].dropna().unique().tolist())
            selected = st.selectbox(f"{label} Strategy Filter", strategies, key=f"{key}_strategy_filter")
            view = frame if selected == "All strategies" else frame[frame["strategy"] == selected]
            chart = view.groupby(key, as_index=False)["net_pnl"].sum().sort_values(key)
            fig = px.bar(chart, x=key, y="net_pnl", title=f"{label} Realized P&L — {selected}")
            fig.update_layout(template="plotly_dark", xaxis_title=label, yaxis_title="Net P&L (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view.sort_values(key, ascending=False), use_container_width=True)
            st.download_button(f"📥 Download {label} P&L CSV", view.to_csv(index=False),
                               filename, "text/csv", key=f"{key}_download")


def page_daily_options_research():
    st.title("Daily Options Research Laboratory")
    st.caption("Session-to-session options experiments: enter at EOD close, exit at the next session's EOD close.")
    close_info = latest_previous_close(load_index())
    if close_info.get("status") == "ok":
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("SENSEX Previous Close", f"₹{close_info['close']:,.2f}")
        pc2.metric("Previous Session Change", f"{close_info['change']:+,.2f}" if close_info.get("change") is not None else "—")
        pc3.metric("Change (%)", f"{close_info['change_pct']:+.2f}%" if close_info.get("change_pct") is not None else "—")
        
    out = DEFAULT_PATHS.outputs
    ledger_path = out / "daily_strategy_trade_ledger.csv"
    board_path = out / "daily_strategy_leaderboard.csv"
    holdout_path = out / "daily_strategy_holdout.json"

    if st.button("🚀 Re-run Daily Options Strategies", type="primary"):
        index_data, option_data = load_index(), load_options()
        with st.spinner("Testing daily Call, Put, and Momentum strategies..."):
            ledger, board, meta = run_daily_strategy_suite(index_data, option_data)
            if ledger.empty:
                st.error(meta.get("error", "No valid contract round trips found."))
            else:
                out.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(ledger_path, index=False)
                board.to_csv(board_path, index=False)
                holdout = daily_holdout_evaluation(ledger)
                with open(holdout_path, "w", encoding="utf-8") as handle:
                    json.dump(holdout, handle, indent=2, default=str)
                st.success(f"Executed {len(ledger):,} daily trades successfully.")

    if not ledger_path.exists() or not board_path.exists():
        st.warning("Daily strategy reports not found. Run the research runner workflow.")
        return

    board = pd.read_csv(board_path)
    ledger = pd.read_csv(ledger_path)
    if board.empty or ledger.empty:
        st.warning("Report files contain no completed trades.")
        return
        
    ledger["entry_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")
    ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce")
    ledger["net_pnl"] = pd.to_numeric(ledger["net_pnl"], errors="coerce").fillna(0)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily Strategies", f"{len(board)}")
    c2.metric("Total Trades", f"{len(ledger):,}")
    c3.metric("Entry Sessions", f"{ledger['entry_date'].nunique():,}")
    best_daily = board.sort_values("net_pnl", ascending=False).iloc[0]
    c4.metric("Top Unit Strategy P&L", f"₹{float(best_daily.net_pnl):,.2f}")

    st.subheader("Strategy Performance Comparison Matrix")
    st.dataframe(board, use_container_width=True, hide_index=True)
    
    left, right = st.columns(2)
    pnl_fig = px.bar(board.sort_values("net_pnl"), x="net_pnl", y="strategy", orientation="h",
                     title="Net P&L After Estimated Costs", text_auto=".2s")
    pnl_fig.update_layout(template="plotly_dark", xaxis_title="Net P&L (₹)", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    left.plotly_chart(pnl_fig, use_container_width=True)
    
    exp_fig = px.bar(board.sort_values("expectancy"), x="expectancy", y="strategy", orientation="h",
                     title="Expectancy per Completed Trade", text_auto=".2s")
    exp_fig.update_layout(template="plotly_dark", xaxis_title="Expectancy (₹)", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    right.plotly_chart(exp_fig, use_container_width=True)


def page_research_lab():
    st.title("Research & Walk-Forward Optimization Lab")
    st.caption("Cost-aware strategy evaluation, rolling walk-forward analysis, and out-of-sample testing.")
    index = load_index()
    options = load_options()
    if index.empty or options.empty:
        st.warning("Historical data missing. Run historical bootstrap workflow.")
        return
    validation = validate_all(index, options)
    
    st.subheader("Historical Coverage Summary")
    a, b, c, d = st.columns(4)
    a.metric("Option Rows", f"{len(options):,}")
    b.metric("Expiry Cycles", validation.get("expiries", 0))
    c.metric("Session Coverage", f"{validation.get('option_session_coverage_pct', 0) or 0:.1f}%")
    d.metric("Complete Expiries", f"{validation.get('expiry_completeness_pct', 0) or 0:.1f}%")

    st.subheader("Execution Cost Parameters")
    x1, x2, x3 = st.columns(3)
    capital = x1.number_input("Capital per Strategy (₹)", min_value=1000.0, value=10000.0, step=1000.0, key="research_capital")
    brokerage = x2.number_input("Brokerage (₹ / lot / order)", min_value=0.0, value=20.0, step=5.0, key="research_brokerage")
    brokerage_pct = x3.number_input("Brokerage (% Turnover)", min_value=0.0, value=0.0, format="%.4f", key="research_brokerage_pct")
    
    y1, y2, y3 = st.columns(3)
    stt = y1.number_input("STT (% Sell Turnover)", min_value=0.0, value=0.10, format="%.4f", key="research_stt")
    exchange = y2.number_input("Exchange Charges (%)", min_value=0.0, value=0.03, format="%.4f", key="research_exchange")
    gst = y3.number_input("GST (%)", min_value=0.0, value=18.0, format="%.2f", key="research_gst")

    if st.button("🧪 Run Cost-Aware Research Suite", type="primary"):
        cost = CostConfig(
            brokerage_per_lot=brokerage, brokerage_pct=brokerage_pct / 100.0,
            stt_pct=stt / 100.0, exchange_charges_pct=exchange / 100.0,
            gst_pct=gst / 100.0, entry_slippage=0.5, exit_slippage=0.5,
        )
        with st.spinner("Executing complete strategy evaluation..."):
            trades, ranking, meta = run_strategy_suite(index, options, capital=capital, cost=cost)
            if trades.empty:
                st.error(meta.get("error", "No trades produced."))
            else:
                metrics = strategy_performance_metrics(trades, capital)
                DEFAULT_PATHS.outputs.mkdir(parents=True, exist_ok=True)
                trades.to_csv(DEFAULT_PATHS.outputs / "strategy_trade_ledger.csv", index=False)
                ranking.to_csv(DEFAULT_PATHS.outputs / "strategy_leaderboard.csv", index=False)
                metrics.to_csv(DEFAULT_PATHS.outputs / "strategy_performance_metrics.csv", index=False)
                st.success(f"Successfully processed {len(trades):,} strategy records.")
                st.json(meta)

    metrics_path = DEFAULT_PATHS.outputs / "strategy_performance_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.subheader("Strategy Performance Metrics")
        st.dataframe(metrics, use_container_width=True)


def main():
    page = sidebar()
    if page == "Overview":
        page_overview()
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
    elif page == "Daily Options Research":
        page_daily_options_research()
    elif page == "Research & Walk-Forward":
        page_research_lab()


if __name__ == "__main__":
    main()
