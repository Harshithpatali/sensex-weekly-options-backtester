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


def page_backtest_config():
    st.title("Backtest Configuration")
    capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=10000.0, step=1000.0)
    c1, c2 = st.columns(2)
    start = c1.date_input("Start (expiry filter)", value=None)
    end = c2.date_input("End (expiry filter)", value=None)
    st.subheader("Transaction costs (research defaults = zero)")
    brokerage = st.number_input("Brokerage per lot (₹, round-trip flat)", value=0.0)
    stt = st.number_input("STT % on exit premium turnover", value=0.0, format="%.4f")
    entry_slip = st.number_input("Entry slippage (points)", value=0.0)
    exit_slip = st.number_input("Exit slippage (points)", value=0.0)
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

    equity = result.get("equity_curve", pd.DataFrame())
    if not equity.empty and "capital" in equity.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=equity["capital"],
            mode="lines+markers",
            name="Equity",
            line=dict(color="#00d4aa"),
        ))
        fig.update_layout(
            title="Equity Curve",
            template="plotly_dark",
            height=400,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1f2e",
        )
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


if __name__ == "__main__":
    main()
