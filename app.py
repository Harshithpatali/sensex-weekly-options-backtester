"""
SENSEX Weekly Options Backtester — Streamlit Dashboard
Modern quant-terminal UI · Dark financial aesthetic · Research only
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
from sensex_options.sources.index import load_index
from sensex_options.sources.bse import load_options
from sensex_options.validation import validate_all, save_validation_report
from sensex_options.backtest import run_backtest
from sensex_options.reports import write_backtest_reports
from sensex_options.features import build_features
from sensex_options.strategies import run_strategy_suite, period_pnl
from sensex_options.analytics import strategy_performance_metrics, walk_forward_evaluation
from sensex_options.daily_strategies import run_daily_strategy_suite, daily_holdout_evaluation
from sensex_options.intraday_research import latest_previous_close

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENSEX Weekly Options | Quant Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "SENSEX Weekly Options Research Backtester · Not live trading",
    },
)

# ─────────────────────────────────────────────────────────────
# Modern Quant Terminal CSS — fixed contrast, no white top bar
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Force dark base ---------- */
    html, body, [class*="css"], .stApp {
        background-color: #0a0c12 !important;
        color: #e8eaed !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    .stApp {
        background: linear-gradient(180deg, #0a0c12 0%, #0e1117 45%, #0a0c12 100%) !important;
    }

    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px;
    }

    /* ---------- Kill the white top bar / header ---------- */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        border: none !important;
    }
    header[data-testid="stHeader"] > div {
        background: transparent !important;
    }
    /* Hide Streamlit toolbar / Deploy button area */
    div[data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    .stDeployButton, [data-testid="stDecoration"] {
        display: none !important;
    }
    /* Remove extra top padding that creates white strip */
    .stApp > header {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] + div {
        background: transparent !important;
    }

    /* ---------- Typography (high contrast) ---------- */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f2f5 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 1.75rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.3rem !important; margin-top: 1.6rem !important; color: #f0f2f5 !important; }
    h3 { font-size: 1.05rem !important; color: #00d4aa !important; }

    p, span, label, .stMarkdown, .stCaption, div[data-testid="stMarkdownContainer"] {
        color: #e8eaed !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9aa0a6 !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #0c0e14 !important;
        border-right: 1px solid #1e2433 !important;
    }
    [data-testid="stSidebar"] * {
        color: #e8eaed !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.92rem !important;
        color: #e8eaed !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #9aa0a6 !important;
        font-size: 0.8rem !important;
    }

    /* ---------- Metrics (high visibility) ---------- */
    div[data-testid="stMetric"] {
        background: #12151e !important;
        border: 1px solid #1e2433 !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #00d4aa !important;
        font-family: "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace !important;
        font-size: 1.3rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #9aa0a6 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricDelta"] {
        font-family: "SF Mono", Consolas, monospace !important;
    }

    /* ---------- Cards ---------- */
    .quant-card {
        background: #12151e !important;
        border: 1px solid #1e2433 !important;
        border-radius: 12px !important;
        padding: 1.15rem 1.3rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        color: #e8eaed !important;
    }
    .quant-card-header {
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #9aa0a6 !important;
        margin-bottom: 0.4rem !important;
    }
    .quant-accent {
        color: #00d4aa !important;
        font-family: "SF Mono", Consolas, monospace !important;
    }
    .quant-muted {
        color: #9aa0a6 !important;
        font-size: 0.85rem !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border: 1px solid #2a3142 !important;
        color: #e8eaed !important;
        background: #1a1f2e !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #00d4aa !important;
        color: #00d4aa !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00b894 0%, #00d4aa 100%) !important;
        color: #0a0c12 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00d4aa 0%, #00e6b8 100%) !important;
        box-shadow: 0 0 18px rgba(0, 212, 170, 0.3) !important;
        color: #0a0c12 !important;
    }

    /* ---------- DataFrames / Tables (critical fix) ---------- */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #12151e !important;
        border: 1px solid #1e2433 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    /* Table header */
    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrame"] th {
        background-color: #1a1f2e !important;
        color: #00d4aa !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #2a3142 !important;
    }
    /* Table body */
    [data-testid="stDataFrame"] tbody td,
    [data-testid="stDataFrame"] td {
        background-color: #12151e !important;
        color: #e8eaed !important;
        border-bottom: 1px solid #1e2433 !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        background-color: #1a1f2e !important;
    }
    /* Force any remaining white cells */
    div[data-testid="stDataFrame"] div[data-testid="stTable"] {
        background: #12151e !important;
    }
    /* Streamlit dataframe internal */
    .stDataFrame, .dataframe {
        background-color: #12151e !important;
        color: #e8eaed !important;
    }
    .stDataFrame th {
        background: #1a1f2e !important;
        color: #00d4aa !important;
    }
    .stDataFrame td {
        background: #12151e !important;
        color: #e8eaed !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0c0e14 !important;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #9aa0a6 !important;
        font-weight: 500 !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1f2e !important;
        color: #00d4aa !important;
    }

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] {
        background: #12151e !important;
        border: 1px solid #1e2433 !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #e8eaed !important;
    }

    /* ---------- Alerts ---------- */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid #1e2433 !important;
        color: #e8eaed !important;
    }

    /* ---------- Inputs ---------- */
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stDateInput input {
        background-color: #0c0e14 !important;
        border: 1px solid #2a3142 !important;
        border-radius: 8px !important;
        color: #e8eaed !important;
    }
    .stSelectbox [data-baseweb="select"] span {
        color: #e8eaed !important;
    }
    /* Dropdown menu */
    [data-baseweb="popover"] {
        background-color: #12151e !important;
        border: 1px solid #2a3142 !important;
    }
    [data-baseweb="menu"] li {
        color: #e8eaed !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: #1a1f2e !important;
    }

    /* ---------- Radio / Checkbox ---------- */
    .stRadio label, .stCheckbox label {
        color: #e8eaed !important;
    }

    /* ---------- Download buttons ---------- */
    .stDownloadButton > button {
        background: #1a1f2e !important;
        color: #e8eaed !important;
        border: 1px solid #2a3142 !important;
    }
    .stDownloadButton > button:hover {
        border-color: #00d4aa !important;
        color: #00d4aa !important;
    }

    /* ---------- Dividers ---------- */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #1e2433, transparent) !important;
        margin: 1.4rem 0 !important;
    }

    /* ---------- Plotly ---------- */
    .js-plotly-plot, .plotly {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0a0c12; }
    ::-webkit-scrollbar-thumb { background: #2a3142; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a4558; }

    /* ---------- Footer ---------- */
    .quant-footer {
        text-align: center;
        color: #5f6368 !important;
        font-size: 0.75rem;
        margin-top: 2.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #1e2433;
    }

    /* ---------- Info / Warning text contrast ---------- */
    .stAlert p, .stAlert span {
        color: #e8eaed !important;
    }

    /* ---------- Fix any remaining white backgrounds ---------- */
    .stApp > div:first-child {
        background: transparent !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Plotly dark quant theme helper
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,21,30,0.7)",
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#e8eaed", size=12),
    margin=dict(l=40, r=30, t=50, b=40),
    xaxis=dict(gridcolor="#1e2433", zerolinecolor="#2a3142", showline=False, color="#9aa0a6"),
    yaxis=dict(gridcolor="#1e2433", zerolinecolor="#2a3142", showline=False, color="#9aa0a6"),
    colorway=["#00d4aa", "#00b4d8", "#7b68ee", "#f72585", "#ff9f1c", "#4cc9f0"],
    title_font=dict(color="#f0f2f5", size=14),
)

def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────
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

def sidebar() -> str:
    st.sidebar.markdown(
        """
        <div style="padding: 0.4rem 0 1rem 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #00d4aa; letter-spacing: -0.02em;">
                ◈ SENSEX OPTIONS
            </div>
            <div style="font-size: 0.7rem; color: #5f6368; margin-top: 3px; letter-spacing: 0.05em; text-transform: uppercase;">
                Quant Research Terminal
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="background:#12151e; border:1px solid #1e2433; border-radius:10px; padding:0.85rem 1rem; font-size:0.78rem; color:#9aa0a6; line-height:1.5;">
            <strong style="color:#00d4aa;">Research only</strong><br>
            Not live trading. Zero-cost defaults are unrealistic.
            Never fabricates option premiums. Uses observed BSE bhavcopy only.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page

# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def section_header(title: str, subtitle: str | None = None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

# ─────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────
def page_overview():
    st.markdown(
        """
        <div style="margin-bottom: 1.4rem;">
            <h1 style="margin-bottom:0.25rem; color:#f0f2f5;">SENSEX Weekly Options</h1>
            <p style="color:#9aa0a6; font-size:0.95rem; margin:0;">
                Research platform for daily Call / Put trades and expiry-based strategies using observed BSE derivatives bhavcopy premiums.
                Default capital: <span style="color:#00d4aa; font-family:monospace;">₹10,000</span> per isolated strategy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    DEFAULT_PATHS.ensure()
    index = load_index()
    options = load_options()
    report = validate_all(index, options)
    close_info = latest_previous_close(index)

    if close_info.get("status") == "ok":
        p1, p2, p3 = st.columns(3)
        p1.metric("SENSEX Previous Close", f"₹{close_info['close']:,.2f}", help=f"Latest completed session: {close_info['date']}")
        p2.metric(
            "Previous Session Change",
            f"{close_info['change']:+,.2f}" if close_info.get("change") is not None else "—",
        )
        p3.metric(
            "Change (%)",
            f"{close_info['change_pct']:+.2f}%" if close_info.get("change_pct") is not None else "—",
        )
        st.caption(f"Previous close date: {close_info['date']} · Source: saved daily ^BSESN history (Yahoo Finance).")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial Capital (default)", "₹10,000")
    c2.metric("Index rows", f"{report.get('index_rows', 0):,}")
    c3.metric("Option rows", f"{report.get('options_rows', 0):,}")
    c4.metric("Available expiries", report.get("expiries", 0))

    c5, c6 = st.columns(2)
    c5.metric("Option-session coverage", f"{report.get('option_session_coverage_pct', 0) or 0:.1f}%")
    c6.metric("Complete matured expiries", f"{report.get('expiry_completeness_pct', 0) or 0:.1f}%")

    st.caption(
        f"Matured: {report.get('completed_expiry_count', 0)} · "
        f"Complete: {report.get('complete_expiries', 0)} · "
        f"Pending: {report.get('pending_expiries', 0)}"
    )

    st.markdown("---")
    section_header("Data coverage")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="quant-card">
                <div class="quant-card-header">Index history</div>
                <div style="font-size:1.05rem; color:#e8eaed;">{report.get('index_start', '—')} → {report.get('index_end', '—')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="quant-card">
                <div class="quant-card-header">Options history</div>
                <div style="font-size:1.05rem; color:#e8eaed;">{report.get('options_start', '—')} → {report.get('options_end', '—')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="quant-card">
            <div class="quant-card-header">Validation status</div>
            <span class="quant-accent" style="font-size:1.1rem;">{report.get('status')}</span>
            <div class="quant-muted" style="margin-top:0.4rem;">SENSEX options approximate launch: {SENSEX_OPTIONS_LAUNCH}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report.get("warnings"):
        st.warning("\n".join(report["warnings"]))
    if report.get("errors"):
        st.error("\n".join(report["errors"]))

def page_validation():
    section_header("Data Validation", "Full integrity checks on index and option series")
    index = load_index()
    options = load_options()
    report = validate_all(index, options)
    save_validation_report(report)

    with st.expander("Full validation report (JSON)", expanded=False):
        st.json(report)

    if not index.empty:
        st.subheader("Index sample (latest 5)")
        st.dataframe(index.tail(5), use_container_width=True)

    if not options.empty:
        st.subheader("Options sample (first 10)")
        st.dataframe(options.head(10), use_container_width=True)

    if report.get("expiry_coverage"):
        st.subheader("Expiry completeness")
        coverage = pd.DataFrame(report["expiry_coverage"])
        st.dataframe(coverage, use_container_width=True)
        fig = px.bar(
            coverage,
            x="expiry_date",
            y=["call_rows", "put_rows"],
            barmode="group",
            title="CE / PE observations by expiry",
            color_discrete_sequence=["#00d4aa", "#00b4d8"],
        )
        apply_plotly_theme(fig)
        fig.update_layout(xaxis_title="Expiry", yaxis_title="Option rows")
        st.plotly_chart(fig, use_container_width=True)

def page_backtest_config():
    section_header("Backtest Configuration", "Single-strategy expiry-cycle engine")
    capital = st.number_input("Initial capital (₹)", min_value=1000.0, value=10000.0, step=1000.0)
    c1, c2 = st.columns(2)
    start = c1.date_input("Start (expiry filter)", value=None)
    end = c2.date_input("End (expiry filter)", value=None)

    st.subheader("Transaction costs (research defaults = zero)")
    brokerage = st.number_input("Brokerage per lot per order (₹)", min_value=0.0, value=0.0)
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
        with st.spinner("Running backtest…"):
            index = load_index()
            options = load_options()
            result = run_backtest(index, options, cfg)
            write_backtest_reports(result)
            st.session_state["bt_result"] = result
            st.success("Backtest complete. See Performance Dashboard and Trade Ledger.")
            st.json(result["summary"])

def page_performance():
    section_header("Performance Dashboard", "Equity curve, drawdown and realized P&L distribution")
    result = st.session_state.get("bt_result")
    if result is None:
        summary_path = DEFAULT_PATHS.outputs / "backtest_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            trades = (
                pd.read_csv(DEFAULT_PATHS.outputs / "trade_ledger.csv")
                if (DEFAULT_PATHS.outputs / "trade_ledger.csv").exists()
                else pd.DataFrame()
            )
            equity = (
                pd.read_csv(DEFAULT_PATHS.outputs / "equity_curve.csv")
                if (DEFAULT_PATHS.outputs / "equity_curve.csv").exists()
                else pd.DataFrame()
            )
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
        apply_plotly_theme(fig)
        fig.update_layout(yaxis_title="Capital (₹)")
        st.plotly_chart(fig, use_container_width=True)

        if not equity.empty:
            equity["peak"] = equity["capital"].cummax()
            equity["drawdown"] = equity["capital"] - equity["peak"]
            fig = px.area(
                equity,
                x="date" if "date" in equity.columns else equity.index,
                y="drawdown",
                title="Equity drawdown",
            )
            apply_plotly_theme(fig)
            fig.update_traces(fillcolor="rgba(247, 37, 133, 0.25)", line_color="#f72585")
            fig.update_layout(yaxis_title="Drawdown (₹)")
            st.plotly_chart(fig, use_container_width=True)

    trades_for_charts = result.get("trades", pd.DataFrame()).copy()
    if not trades_for_charts.empty and "status" in trades_for_charts.columns:
        completed = trades_for_charts[trades_for_charts["status"] == "completed"].copy()
        if not completed.empty:
            completed["net_pnl"] = pd.to_numeric(completed["net_pnl"], errors="coerce")
            completed["exit_date"] = pd.to_datetime(completed["exit_date"], errors="coerce")
            daily = completed.dropna(subset=["exit_date"]).groupby("exit_date", as_index=False)["net_pnl"].sum()
            fig = px.bar(daily, x="exit_date", y="net_pnl", title="Realized daily P&L")
            apply_plotly_theme(fig)
            fig.update_layout(yaxis_title="Net P&L (₹)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(completed, x="net_pnl", nbins=30, title="Trade P&L distribution")
            apply_plotly_theme(fig)
            fig.update_layout(xaxis_title="Net P&L per trade (₹)")
            st.plotly_chart(fig, use_container_width=True)

    annual_path = DEFAULT_PATHS.outputs / "annual_summary.csv"
    if annual_path.exists():
        st.subheader("Yearly performance")
        st.dataframe(pd.read_csv(annual_path), use_container_width=True)

def page_ledger():
    section_header("Trade Ledger", "Every completed and skipped trade from the last backtest")
    path = DEFAULT_PATHS.outputs / "trade_ledger.csv"
    if not path.exists():
        st.info("No trade ledger yet. Run a backtest.")
        return
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download trade ledger CSV",
        df.to_csv(index=False),
        "trade_ledger.csv",
        "text/csv",
    )

def page_reports():
    section_header("Reports", "All generated artefacts available for download")
    out = DEFAULT_PATHS.outputs
    files = list(out.glob("*")) if out.exists() else []
    if not files:
        st.info("No reports generated yet.")
        return
    for f in sorted(files, key=lambda x: x.name):
        st.write(f"**{f.name}** — {f.stat().st_size:,} bytes")
        if f.suffix == ".csv":
            st.download_button(f"Download {f.name}", f.read_bytes(), f.name, "text/csv", key=f.name)
        elif f.suffix == ".json":
            st.download_button(f"Download {f.name}", f.read_bytes(), f.name, "application/json", key=f.name)

    val = DEFAULT_PATHS.reports / "data_validation.json"
    if val.exists():
        st.download_button(
            "Download validation report",
            val.read_bytes(),
            "data_validation.json",
            key="val",
        )

def page_strategy_leaderboard():
    section_header(
        "Strategy Leaderboard",
        "Historical simulated ranking by net P&L. Ranking is descriptive, not a forecast or live recommendation.",
    )
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
        fig = px.bar(
            top.sort_values("net_pnl"),
            x="net_pnl",
            y="strategy",
            orientation="h",
            title="Net P&L by strategy",
            text_auto=".2s",
            color_discrete_sequence=["#00d4aa"],
        )
        apply_plotly_theme(fig)
        fig.update_layout(xaxis_title="Net P&L (₹)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "max_drawdown" in top.columns:
            risk = top.copy()
            risk["absolute_drawdown"] = risk["max_drawdown"].abs()
            fig = px.bar(
                risk.sort_values("absolute_drawdown"),
                x="absolute_drawdown",
                y="strategy",
                orientation="h",
                title="Historical maximum drawdown",
                text_auto=".2s",
                color_discrete_sequence=["#f72585"],
            )
            apply_plotly_theme(fig)
            fig.update_layout(xaxis_title="Drawdown (₹)", yaxis_title="")
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
            fig = px.line(
                view,
                x="exit_date",
                y="cumulative_pnl",
                markers=True,
                title=f"{selected}: cumulative realized P&L",
            )
            apply_plotly_theme(fig)
            fig.update_layout(xaxis_title="Exit date", yaxis_title="Cumulative P&L (₹)")
            st.plotly_chart(fig, use_container_width=True)

            if {"starting_capital", "ending_capital"}.issubset(view.columns):
                fig = px.line(
                    view,
                    x="exit_date",
                    y="ending_capital",
                    markers=True,
                    title=f"{selected}: isolated strategy capital curve",
                )
                apply_plotly_theme(fig)
                fig.update_layout(xaxis_title="Exit date", yaxis_title="Capital (₹)")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trade ledger")
            st.dataframe(view.sort_values("exit_date", ascending=False), use_container_width=True)
            st.download_button(
                "Download selected strategy trades",
                view.to_csv(index=False),
                f"{selected.lower()}_trades.csv",
                "text/csv",
            )

    st.info(
        "Each strategy is simulated with its own sequential capital curve. "
        "These are not five simultaneous strategies sharing one ₹10,000 portfolio."
    )

def page_period_pnl():
    section_header(
        "Daily / Weekly / Monthly Strategy P&L",
        "Realized P&L grouped by trade exit date. This is not mark-to-market P&L for open positions.",
    )
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
            chart = view.groupby(key, as_index=False)["net_pnl"].sum().sort_values(key)
            fig = px.bar(chart, x=key, y="net_pnl", title=f"{label} realized P&L — {selected}")
            apply_plotly_theme(fig)
            fig.update_layout(xaxis_title=label, yaxis_title="Net P&L (₹)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view.sort_values(key, ascending=False), use_container_width=True)
            st.download_button(
                f"Download {label.lower()} P&L CSV",
                view.to_csv(index=False),
                filename,
                "text/csv",
                key=f"{key}_download",
            )

def page_daily_options_research():
    section_header(
        "Daily Options Research",
        "One-session options experiments: enter at one session's EOD close and exit at the next session's EOD close. Calls and Puts tested independently.",
    )
    st.info(
        "This page is separate from the expiry-only leaderboard. Signals use the previous session's index move; "
        "the ATM strike is selected using the previous session's SENSEX close. EOD marks are estimates, not guaranteed fills."
    )

    close_info = latest_previous_close(load_index())
    if close_info.get("status") == "ok":
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("SENSEX Previous Close", f"₹{close_info['close']:,.2f}")
        pc2.metric(
            "Previous Session Change",
            f"{close_info['change']:+,.2f}" if close_info.get("change") is not None else "—",
        )
        pc3.metric(
            "Change (%)",
            f"{close_info['change_pct']:+.2f}%" if close_info.get("change_pct") is not None else "—",
        )
        st.caption(f"Last completed SENSEX session: {close_info['date']}. This is not a live quote.")

    out = DEFAULT_PATHS.outputs
    ledger_path = out / "daily_strategy_trade_ledger.csv"
    board_path = out / "daily_strategy_leaderboard.csv"
    holdout_path = out / "daily_strategy_holdout.json"

    if st.button("Re-run daily strategies using available history", type="primary"):
        index_data, option_data = load_index(), load_options()
        with st.spinner("Testing daily Call, Put, momentum and mean-reversion strategies…"):
            ledger, board, meta = run_daily_strategy_suite(index_data, option_data)
            if ledger.empty:
                st.error(meta.get("error", "No valid next-session contract round trips were found."))
            else:
                out.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(ledger_path, index=False)
                board.to_csv(board_path, index=False)
                holdout = daily_holdout_evaluation(ledger)
                with open(holdout_path, "w", encoding="utf-8") as handle:
                    json.dump(holdout, handle, indent=2, default=str)
                st.success(
                    f"Completed {len(ledger):,} daily trades across {ledger['entry_date'].nunique():,} entry sessions."
                )
                st.json(meta)

    if not ledger_path.exists() or not board_path.exists():
        st.warning(
            "Daily strategy reports are not available yet. "
            "Run the historical bootstrap and the daily research workflow."
        )
        st.markdown(
            "[Open daily research workflow](https://github.com/Harshithpatali/sensex-weekly-options-backtester/actions/workflows/daily-research.yml)"
        )
        return

    board = pd.read_csv(board_path)
    ledger = pd.read_csv(ledger_path)
    if board.empty or ledger.empty:
        st.warning("The daily report files exist but contain no completed trades.")
        return

    ledger["entry_date"] = pd.to_datetime(ledger["entry_date"], errors="coerce")
    ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce")
    ledger["net_pnl"] = pd.to_numeric(ledger["net_pnl"], errors="coerce").fillna(0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily strategies", f"{len(board)}")
    c2.metric("Completed daily trades", f"{len(ledger):,}")
    c3.metric("Entry sessions", f"{ledger['entry_date'].nunique():,}")
    best_daily = board.sort_values("net_pnl", ascending=False).iloc[0]
    c4.metric("Top unit-lot strategy P&L", f"₹{float(best_daily.net_pnl):,.2f}")

    st.subheader("Strategy definitions")
    definitions = pd.DataFrame(
        [
            {"Strategy": "Daily ATM Call", "Rule": "Buy nearest ATM CE every eligible session"},
            {"Strategy": "Daily ATM Put", "Rule": "Buy nearest ATM PE every eligible session"},
            {"Strategy": "Daily Momentum", "Rule": "Buy CE after a positive prior-session move; PE after a negative move"},
            {"Strategy": "Daily Mean Reversion", "Rule": "Buy PE after a positive prior-session move; CE after a negative move"},
            {"Strategy": "5-Day Momentum", "Rule": "Buy CE or PE using the prior five-session SENSEX return"},
            {"Strategy": "EMA Trend", "Rule": "Buy CE when prior EMA-5 is above EMA-20; otherwise PE"},
            {"Strategy": "RSI Reversal", "Rule": "Buy CE when prior RSI-14 is below 30; PE when above 70"},
        ]
    )
    st.dataframe(definitions, use_container_width=True, hide_index=True)
    st.caption(
        "Each strategy is tested at one lot per eligible session. "
        "₹10,000 is only the one-lot affordability filter; cumulative unit-lot P&L is not a compounded portfolio return."
    )

    st.subheader("Daily strategy comparison")
    st.dataframe(board, use_container_width=True, hide_index=True)

    if {"net_pnl", "median_trade_pnl", "top3_wins_share_of_net_pct"}.issubset(board.columns):
        concentration = board[
            (pd.to_numeric(board["net_pnl"], errors="coerce") > 0)
            & (
                (pd.to_numeric(board["median_trade_pnl"], errors="coerce") < 0)
                | (pd.to_numeric(board["top3_wins_share_of_net_pct"], errors="coerce") > 50)
            )
        ]
        if not concentration.empty:
            flagged = ", ".join(concentration["strategy"].astype(str))
            st.warning(
                f"Tail-risk warning: {flagged} has positive total P&L but a negative median trade "
                "or more than half of net profit from its three largest wins. "
                "Do not treat the headline P&L as consistent profitability."
            )

    left, right = st.columns(2)
    pnl_fig = px.bar(
        board.sort_values("net_pnl"),
        x="net_pnl",
        y="strategy",
        orientation="h",
        title="Net P&L after estimated costs",
        text_auto=".2s",
        color_discrete_sequence=["#00d4aa"],
    )
    apply_plotly_theme(pnl_fig)
    pnl_fig.update_layout(xaxis_title="Net P&L (₹)", yaxis_title="")
    left.plotly_chart(pnl_fig, use_container_width=True)

    exp_fig = px.bar(
        board.sort_values("expectancy"),
        x="expectancy",
        y="strategy",
        orientation="h",
        title="Expectancy per completed trade",
        text_auto=".2s",
        color_discrete_sequence=["#00b4d8"],
    )
    apply_plotly_theme(exp_fig)
    exp_fig.update_layout(xaxis_title="Expectancy (₹)", yaxis_title="")
    right.plotly_chart(exp_fig, use_container_width=True)

    risk_left, risk_right = st.columns(2)
    if "max_drawdown_pct" in board.columns:
        dd_fig = px.bar(
            board.sort_values("max_drawdown_pct"),
            x="max_drawdown_pct",
            y="strategy",
            orientation="h",
            title="Maximum drawdown (%)",
            color_discrete_sequence=["#f72585"],
        )
        apply_plotly_theme(dd_fig)
        dd_fig.update_layout(xaxis_title="Drawdown (%)", yaxis_title="")
        risk_left.plotly_chart(dd_fig, use_container_width=True)
    if "profit_factor" in board.columns:
        pf_fig = px.bar(
            board.sort_values("profit_factor", na_position="first"),
            x="profit_factor",
            y="strategy",
            orientation="h",
            title="Profit factor",
            color_discrete_sequence=["#7b68ee"],
        )
        pf_fig.add_vline(x=1.0, line_dash="dash", annotation_text="Break-even")
        apply_plotly_theme(pf_fig)
        pf_fig.update_layout(xaxis_title="Gross profit / gross loss", yaxis_title="")
        risk_right.plotly_chart(pf_fig, use_container_width=True)

    if holdout_path.exists():
        with open(holdout_path, encoding="utf-8") as handle:
            holdout = json.load(handle)
        st.subheader("Chronological out-of-sample holdout")
        st.write(f"**Status:** {holdout.get('status', 'not run')}")
        if holdout.get("status") == "ok":
            st.write(f"Training window: {holdout.get('train_start')} to {holdout.get('train_end')}")
            st.write(
                f"Holdout window: {holdout.get('holdout_start')} to {holdout.get('holdout_end')} · "
                f"{holdout.get('holdout_sessions')} sessions"
            )
            holdout_frame = pd.DataFrame(holdout.get("holdout_metrics", []))
            if not holdout_frame.empty:
                st.dataframe(holdout_frame, use_container_width=True, hide_index=True)
            st.caption(holdout.get("warning", ""))
        else:
            st.warning(holdout.get("note", "Not enough daily sessions for holdout testing."))

    st.subheader("Intraday filters — Daily ATM Put")
    st.caption(
        "Read-only results from the scheduled GitHub Actions data pipeline. "
        "The dashboard does not download or collect market data."
    )
    intraday_path = out / "sensex_intraday_5m.csv"
    filter_report_path = out / "daily_put_intraday_validation.json"
    filter_metrics_path = out / "daily_put_intraday_validation.csv"
    intraday_run = {}
    daily_run_path = out / "daily_run.json"
    if daily_run_path.exists():
        try:
            with open(daily_run_path, encoding="utf-8") as handle:
                intraday_run = json.load(handle).get("intraday_research", {})
        except (OSError, json.JSONDecodeError):
            intraday_run = {}

    status = intraday_run.get("status", "not_available")
    if status == "ok":
        m1, m2, m3 = st.columns(3)
        m1.metric("Saved 5-minute bars", f"{intraday_run.get('bars', 0):,}")
        m2.metric("Matched option sessions", f"{intraday_run.get('matched_sessions', 0):,}")
        m3.metric(
            "Validation period",
            f"{intraday_run.get('validation_start', '—')} → {intraday_run.get('validation_end', '—')}",
        )
    elif status == "insufficient_data":
        st.warning(intraday_run.get("note", "The scheduled run did not have enough overlapping data for validation."))
    elif status == "download_error":
        st.warning("The latest scheduled intraday download failed. Previously saved reports, if present, remain available.")
        if intraday_run.get("error"):
            st.caption(str(intraday_run["error"]))
    else:
        st.info("No saved intraday research output yet. It will appear after the scheduled data workflow completes.")

    if intraday_path.exists():
        bars_saved = pd.read_csv(intraday_path)
        if not bars_saved.empty:
            st.caption(
                f"Saved intraday dataset: {len(bars_saved):,} rows · "
                f"{bars_saved['timestamp'].min()} to {bars_saved['timestamp'].max()}"
            )
            with st.expander("View saved intraday SENSEX candles"):
                st.dataframe(bars_saved.tail(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Download saved SENSEX intraday data",
                intraday_path.read_bytes(),
                "sensex_intraday_5m.csv",
                "text/csv",
            )

    if filter_report_path.exists():
        try:
            with open(filter_report_path, encoding="utf-8") as handle:
                filter_report = json.load(handle)
            st.subheader("Saved intraday filter validation")
            if filter_report.get("status") == "ok":
                st.caption(
                    f"Training: {filter_report.get('training_start')} → {filter_report.get('training_end')} · "
                    f"Validation: {filter_report.get('validation_start')} → {filter_report.get('validation_end')} · "
                    f"Volatility threshold learned from training only: {filter_report.get('training_median_range_pct')}% range"
                )
                if filter_metrics_path.exists():
                    st.dataframe(pd.read_csv(filter_metrics_path), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download saved filter validation CSV",
                        filter_metrics_path.read_bytes(),
                        "daily_put_intraday_validation.csv",
                        "text/csv",
                    )
                st.caption(filter_report.get("warning", ""))
                st.warning(filter_report.get("stop_loss_status", ""))
            else:
                st.info(filter_report.get("note", "No valid validation result was saved in the latest workflow run."))
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Saved intraday validation report could not be read: {exc}")
    else:
        st.caption("The intraday validation report has not been generated by the scheduled workflow yet.")

    st.subheader("Daily trade ledger")
    choices = ["All strategies"] + sorted(ledger["strategy"].dropna().unique().tolist())
    selected = st.selectbox("Filter daily strategy", choices, key="daily_options_strategy_filter")
    view = ledger if selected == "All strategies" else ledger[ledger["strategy"] == selected]
    chart = view.groupby("exit_date", as_index=False)["net_pnl"].sum().sort_values("exit_date")
    if not chart.empty:
        chart["cumulative_net_pnl"] = chart["net_pnl"].cumsum()
        curve = px.line(
            chart,
            x="exit_date",
            y="cumulative_net_pnl",
            markers=True,
            title=f"Cumulative realized P&L — {selected}",
        )
        apply_plotly_theme(curve)
        curve.update_layout(xaxis_title="Exit session", yaxis_title="Cumulative net P&L (₹)")
        st.plotly_chart(curve, use_container_width=True)
    st.dataframe(view.sort_values("entry_date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "Download daily options trade ledger",
        view.to_csv(index=False),
        "daily_options_trade_ledger.csv",
        "text/csv",
    )

def page_research_lab():
    section_header(
        "Research & Walk-Forward",
        "Cost-aware comparison, chronological holdout and rolling expiry evaluation. Fixed rule strategies are not trained or optimized.",
    )
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
    st.caption(
        f"Matured expiries: {validation.get('completed_expiry_count', 0)} · "
        f"Complete for backtest: {validation.get('complete_expiries', 0)} · "
        f"Pending future expiries: {validation.get('pending_expiries', 0)}"
    )

    if validation.get("completed_expiry_count", 0) < 30:
        st.warning("Fewer than 30 matured expiry cycles are available. Backfill the full history before relying on the statistics.")

    st.markdown(
        "[Open the historical bootstrap workflow](https://github.com/Harshithpatali/sensex-weekly-options-backtester/actions/workflows/bootstrap-history.yml)"
    )

    if validation.get("missing_option_sessions"):
        with st.expander("Missing option sessions"):
            st.write(", ".join(validation["missing_option_sessions"]))

    expiry_rows = validation.get("expiry_coverage", [])
    if expiry_rows:
        with st.expander("Latest expiry-by-expiry completeness diagnostics"):
            expiry_frame = pd.DataFrame(expiry_rows)
            if "complete_for_backtest" in expiry_frame.columns:
                expiry_frame["coverage_status"] = expiry_frame["complete_for_backtest"].map(
                    {True: "Complete", False: "Incomplete"}
                )
                coverage_filter = st.selectbox(
                    "Expiry coverage filter",
                    ["All", "Complete", "Incomplete"],
                    key="expiry_coverage_filter",
                )
                if coverage_filter != "All":
                    expiry_frame = expiry_frame[expiry_frame["coverage_status"] == coverage_filter]
            st.dataframe(expiry_frame, use_container_width=True, hide_index=True)
            st.caption(
                "An expiry is complete only when both CE and PE have at least one "
                "same-strike round trip with a positive entry premium, observed exit "
                "mark and verified lot size."
            )

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

    st.caption(
        "Rates are editable research assumptions, not broker quotes or tax advice. "
        "Verify the current BSE and broker tariff before relying on net P&L. "
        "Spread is charged half on entry and half on exit; slippage and market impact are applied adversely to each leg on both sides. "
        "BSE EOD files do not contain historical order-book quotes, so these are user-controlled execution proxies."
    )

    if st.button("Run cost-aware strategy research", type="primary"):
        cost = CostConfig(
            brokerage_per_lot=brokerage,
            brokerage_pct=brokerage_pct / 100.0,
            stt_pct=stt / 100.0,
            exchange_charges_pct=exchange / 100.0,
            gst_pct=gst / 100.0,
            stamp_duty_pct=stamp / 100.0,
            other_charges=other,
            entry_slippage=entry_slip,
            exit_slippage=exit_slip,
            spread_points=spread,
            market_impact_points=impact,
        )
        with st.spinner("Replaying all available expiry cycles…"):
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

    daily_run_path = DEFAULT_PATHS.outputs / "daily_run.json"
    if daily_run_path.exists():
        try:
            with open(daily_run_path, encoding="utf-8") as handle:
                daily_run = json.load(handle)
            st.subheader("Latest automated research run")
            run_cols = st.columns(4)
            run_cols[0].metric("Last refresh (UTC)", str(daily_run.get("run_time_utc", "—"))[:19].replace("T", " "))
            run_cols[1].metric("Historical option rows", f"{daily_run.get('option_rows', 0):,}")
            run_cols[2].metric("Expiry cycles", daily_run.get("data_quality", {}).get("expiries", 0))
            run_cols[3].metric("Walk-forward status", daily_run.get("walk_forward_status", "—"))
        except (OSError, json.JSONDecodeError):
            st.warning("The latest automated research summary could not be read.")

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
        st.caption(
            "Sharpe is based on trade-level returns annualised at 52 expiry cycles/year. "
            "It is approximate, not a daily mark-to-market Sharpe ratio."
        )
        chart_left, chart_right = st.columns(2)
        fig = px.bar(metrics, x="strategy", y="expectancy", title="Net expectancy per completed trade")
        apply_plotly_theme(fig)
        fig.update_layout(yaxis_title="Expectancy (₹)")
        chart_left.plotly_chart(fig, use_container_width=True)

        drawdown = metrics.dropna(subset=["max_drawdown_pct"])
        dd_fig = px.bar(
            drawdown,
            x="strategy",
            y="max_drawdown_pct",
            title="Maximum drawdown (%) — lower magnitude is better",
        )
        apply_plotly_theme(dd_fig)
        dd_fig.update_layout(yaxis_title="Maximum drawdown (%)")
        chart_right.plotly_chart(dd_fig, use_container_width=True)

        if "profit_factor" in metrics.columns:
            pf = metrics.dropna(subset=["profit_factor"])
            if not pf.empty:
                pf_fig = px.bar(pf, x="strategy", y="profit_factor", title="Profit factor")
                apply_plotly_theme(pf_fig)
                pf_fig.update_layout(yaxis_title="Gross profit / gross loss")
                st.plotly_chart(pf_fig, use_container_width=True)

    if walk:
        st.subheader("Out-of-sample and walk-forward results")
        st.write(f"**Status:** {walk.get('status', 'not run')}")
        if walk.get("status") == "ok":
            st.write(
                f"**Chronological holdout:** {walk.get('holdout_start')} to {walk.get('holdout_end')} "
                f"({walk.get('holdout_expiries')} expiry cycles)"
            )
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
        st.download_button(
            "Download costed strategy ledger",
            trades.to_csv(index=False),
            "strategy_trade_ledger.csv",
            "text/csv",
        )

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
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

    st.markdown(
        """
        <div class="quant-footer">
            SENSEX Weekly Options · Research Backtester · Not investment advice · Data: BSE bhavcopy + Yahoo Finance
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
