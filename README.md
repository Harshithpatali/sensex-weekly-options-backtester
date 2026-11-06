# SENSEX Weekly Options Backtester

Research platform for historical SENSEX index and BSE options data, a transparent single-strategy backtester, and a five-strategy comparative research suite.

- **Initial capital (default):** ₹10,000  
- Uses **actual** historical SENSEX index data (Yahoo Finance) and **actual** BSE equity-derivatives bhavcopy option premiums.  
- **Never fabricates** option prices, lot sizes, or successful trades.  
- Missing data → trade is **skipped** with an explicit reason.

> **Disclaimer:** This is a research tool, not a live trading system. Past simulated performance does not guarantee future results. Closing prices are not guaranteed fills. Transaction costs and slippage are configurable assumptions. You are solely responsible for any trading decisions.

---

## Core backtest strategy rules

The original single-strategy backtest uses the following rules for each **actual** historical SENSEX options expiry present in BSE data:

1. Identify the previous SENSEX trading session before expiry.  
2. Take the SENSEX **closing price** on that session as the reference.  
3. From Call contracts for that expiry available **on the entry session only**, select the **smallest strike strictly greater** than the reference close.  
4. **Entry premium** = that contract’s previous-session close.  
5. Hold until the actual expiry date.  
6. **Exit premium** = the **same** contract’s close on the expiry session (or last available session on/before expiry).  
7. Size in **whole lots only**, never exceeding available capital.  
8. Update capital after each completed trade.  
9. Skip if: missing entry/exit premium, missing historical lot size, insufficient capital, or no eligible strike.

### P&L (per completed trade)

\[
\text{Gross P\&L} = (P_{\text{exit}} - P_{\text{entry}}) \times \text{lot size} \times N_{\text{lots}}
\]

\[
\text{Net P\&L} = \text{Gross P\&L} - \text{transaction costs}
\]

(Slippage can be modelled by adjusting entry/exit premiums.)

---

## Project structure

```
sensex-weekly-options-backtester/
├── app.py                 # Streamlit dashboard
├── scripts/               # Direct, beginner-friendly pipeline entry points
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── validate_data.py
│   ├── run_backtest.py
│   └── run_daily_research.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── src/sensex_options/
│   ├── config.py
│   ├── calendar.py
│   ├── features.py
│   ├── daily_pipeline.py
│   ├── strategies.py
│   ├── analytics.py
│   ├── validation.py
│   ├── backtest.py
│   ├── costs.py
│   ├── reports.py
│   ├── cli.py
│   └── sources/
│       ├── index.py       # yfinance ^BSESN
│       ├── bse.py         # BSE FO bhavcopy
│       └── contracts.py   # lot size resolution
├── tests/
├── data/
│   ├── raw/index/
│   ├── raw/options/
│   ├── processed/
│   └── reports/
└── outputs/
```

---

## Installation

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
streamlit run app.py
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or activate via cmd:

```cmd
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
streamlit run app.py
```

---

## Data collection

**Do not** expect automatic downloads on every app start. Collect explicitly.

### Index (Yahoo Finance – research source)

```bash
sensex-options collect-index --start 2023-05-15 --end 2026-11-05
```

Yahoo Finance end handling can be exclusive in some versions; the downloader requests an extra day and clips to your inclusive range. Independently verify prices for serious research.

### Options (BSE official FO bhavcopy)

```bash
sensex-options collect-options --start 2023-05-15 --end 2026-11-05
```

Endpoints implemented:

| Era | URL pattern |
|-----|-------------|
| UDiFF (≥ 2024-07-08) | `https://www.bseindia.com/download/BhavCopy/Derivative/BhavCopy_BSE_FO_0_0_0_{YYYYMMDD}_F_0000.CSV` |
| Legacy | `https://www.bseindia.com/download/BhavCopy/Derivative/bhavcopy{DD-MM-YY}.zip` |

BSE equity derivatives (including SENSEX options) were relaunched around **May 2023**. Coverage before that is expected to be empty or incomplete.

**Manual fallback:** If HTTP downloads fail (rate limits, schema changes, network), place official BSE FO CSV/ZIP files into `data/raw/options/` and re-run collection / parsing.

A collection manifest is written to `data/reports/options_collection_manifest.csv`.

### One-command Python scripts (recommended)

The repository now includes explicit runnable Python files under `scripts/`. Run these from the repository root after installation:

```bash
# 1. Download index + both CE and PE options
python scripts/download_data.py --start 2023-05-15 --end 2026-11-06

# 2. Build the processed-data summary and validate data quality
python scripts/preprocess_data.py

# 3. Run validation again independently (optional)
python scripts/validate_data.py

# 4. Run the original single-strategy backtest
python scripts/run_backtest.py --capital 10000

# 5. Run incremental daily collection + five-strategy research suite
python scripts/run_daily_research.py
```

Use `--index-only` or `--options-only` with `download_data.py` to collect just one source. Dates are inclusive. BSE holidays and unavailable bhavcopies are recorded in the collection manifest rather than treated as successful downloads.

The preprocessing script parses any official BSE FO CSV/ZIP files placed in `data/raw/options/`, filters SENSEX CE/PE contracts, normalizes dates and numeric fields through the shared parser, merges with existing processed history, removes duplicate contract/day rows, writes `data/processed/sensex_options.csv` and `features_summary.json`, then validates. It does not fabricate or impute option premiums and exits non-zero if validation reports errors.

### Build & validate (CLI alternative)

```bash
sensex-options build-features
sensex-options validate
```

---

## Backtesting

```bash
sensex-options backtest --capital 10000 --cost-per-lot 0
```

Optional:

```bash
sensex-options backtest --capital 10000 --start 2024-01-01 --end 2025-12-31 --max-lots 5
```

Reports land in `outputs/`:

- `trade_ledger.csv`
- `completed_trades.csv`
- `skipped_trades.csv`
- `annual_summary.csv`
- `equity_curve.csv`
- `backtest_summary.json`

```bash
sensex-options report
```

### CLI via module

```bash
python -m sensex_options.cli validate
python -m sensex_options.cli backtest --capital 10000
```

---

## Streamlit dashboard

```bash
streamlit run app.py
```

Pages:

1. **Overview** – coverage and validation status  
2. **Data Collection** – on-demand downloads  
3. **Data Validation** – quality report  
4. **Backtest Configuration** – capital, costs, run  
5. **Performance Dashboard** – metrics + equity curve (Plotly)  
6. **Trade Ledger** – every completed and skipped trade  
7. **Reports** – downloadable CSV/JSON  
8. **Strategy Leaderboard** – historical strategy ranking  
9. **Daily / Weekly / Monthly P&L** – realized expiry-trade P&L  
10. **Research & Walk-Forward** – editable execution costs, risk metrics, chronological holdout and rolling expiry tests  

Dark financial-terminal theme.

---

## Tests

```bash
pytest -q
```

Tests cover strike selection, whole-lot sizing, capital floor, missing premium/lot skips, same-contract entry/exit, no look-ahead, and cost application. They use synthetic **fixtures** only for unit logic; they do **not** claim real-market profitability.

---

## Historical data limitations

- Yahoo Finance index data is convenient for research; verify against official BSE index series when accuracy matters.  
- BSE FO file formats changed (legacy ZIP → UDiFF CSV). The parser maps both where possible.  
- Lot size may be absent from some historical files. **Trades are skipped** rather than inventing a lot size.  
- Expiry weekdays can change with exchange rules and holidays; expiries are taken from **contract data**, not a fixed weekday.  
- Illiquid / zero-volume contracts are flagged in validation.  
- Closing premiums are not guaranteed executable prices.

## Transaction-cost limitations

The original single-strategy Backtest Configuration page retains zero-cost defaults for compatibility and must not be used for realistic net-return claims without entering costs. The comparative strategy suite and scheduled daily replay use illustrative non-zero assumptions for brokerage, STT, exchange charges, GST, stamp duty, spread, slippage and market impact. Review and edit those assumptions in the Streamlit **Research & Walk-Forward** page. Historical rate schedules vary; these are estimates, not broker quotes or tax advice.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| Empty options data | Check date range ≥ 2023-05-15; inspect manifest; try manual bhavcopy drop-in |
| All trades skipped for lot size | Source files lack lot size — supply verified metadata or extend `contracts.py` carefully |
| yfinance empty | Network / ticker; retry; verify `^BSESN` |
| Streamlit import errors | `pip install -e .` from project root with venv active |
| PowerShell activation blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## GitHub readiness

- `.gitignore` excludes venvs, caches, large market data, secrets.  
- GitHub Actions workflow runs `pytest` on push/PR.  
- Do **not** commit `data/raw` or `data/processed` market files.

---

## License / risk

MIT for the code. Market data remains subject to exchange terms.  
**Financial risk disclaimer:** Options can expire worthless. Simulated results can differ materially from live trading. This software provides no investment advice.


## Automated daily research workflow

GitHub Actions is scheduled for **06:30 Asia/Kolkata (01:00 UTC), Monday–Friday**. It attempts to fetch the previous calendar day's BSE FO bhavcopy, refreshes SENSEX index history, validates the accumulated datasets, replays five transparent option-buying baselines, and publishes daily/weekly/monthly realized simulated P&L plus a ranked leaderboard.

Strategies: ATM call, ATM put, one-step OTM call, one-step OTM put, and ATM long straddle. The leaderboard ranks the complete available historical replay by net P&L (not a prediction of tomorrow's best strategy). P&L is simulated from daily closing marks, not live fills. Estimated costs include editable brokerage, STT, exchange charges, GST, stamp duty, other charges, bid/ask spread proxy, slippage and market-impact proxy. Funding, margin and guaranteed execution are not modelled.

**Important setup:** this workflow needs an initial historical dataset to be useful. The first scheduled run only adds the latest available day; it does not backfill all history. Run a historical BSE data bootstrap separately before relying on rankings. The workflow commits processed index/options history and reports to this repository so the Streamlit app can read them; this increases repository size over time. GitHub rejects individual files at 100 MB. For long-term production, move market history to object storage or a database and keep only compact reports in Git.

The daily workflow is research/replay, not machine-learning model training and not live order execution. It never places trades. If BSE has not published a file yet or the date is a holiday, inspect the Actions log and rerun after publication.


## Research model assumptions and report semantics

- The original single-call backtest and the five-strategy comparison are separate simulations.
- The five baselines are ATM call, ATM put, one-strike-step OTM call, one-strike-step OTM put, and ATM long straddle.
- Each baseline has an isolated capital curve, compounded sequentially from the configured starting capital. The five strategies are **not** a single portfolio sharing the same capital.
- Strategy exits require a quote for each required contract on the same selected exit session. Missing legs are skipped; stale prices from different dates are not mixed.
- Daily, weekly, and monthly strategy P&L files are **realized P&L grouped by trade exit date**. They are not mark-to-market valuations of positions still open.
- The strategy replay uses EOD close marks and approximate brokerage/slippage. It does not model bid/ask spread, queue position, market impact, margin, full statutory charges, taxes, or live order execution.
- Historical lot size is taken from source data. Missing lot sizes are not guessed; affected trades are skipped.
- A historical leaderboard ranks the observed sample. It is not a forecast and does not select a live strategy automatically.
- GitHub Actions CI validates code with unit tests. Those tests use synthetic fixtures and do not establish that a strategy is profitable on real-market data.


## Cost-aware research and walk-forward evaluation

The Streamlit **Research & Walk-Forward** page runs the five fixed-rule baselines across all available expiry cycles. It exposes editable assumptions for brokerage per lot/order, brokerage as a turnover percentage, STT, exchange charges, GST, stamp duty, other charges, full bid/ask spread, adverse entry/exit slippage and market impact. The ledger reports the cost components and subtracts estimated charges from raw gross P&L. Rates are editable research assumptions, not broker quotes or tax advice; verify the current BSE and broker tariff. The spread input is charged half on entry and half on exit; slippage and market impact are applied adversely on both sides. Because daily BSE bhavcopies do not provide historical order-book quotes, spread and impact are explicit execution proxies rather than observed fills.

The page reports per-strategy trade count, net P&L, win rate, profit factor, expectancy, maximum drawdown and drawdown percentage. It also reports an approximate trade-return Sharpe annualised at 52 expiry cycles/year. This is **not** a daily mark-to-market Sharpe ratio. The chronological 70/30 expiry holdout and rolling expiry-window evaluation are suitable for fixed-rule comparison; the strategies are not fitted models and no parameters are learned from the context window.

Data validation now reports option-session coverage against SENSEX index sessions and CE/PE observations for each expiry.

### Historical bootstrap is required

The scheduled daily workflow only adds recent sessions; it does not create a multi-year dataset. Run **Actions → Bootstrap SENSEX historical dataset → Run workflow** with the default range (2023-05-15 through the latest available date) to backfill the archive. This can take time because it requests one BSE FO bhavcopy per weekday. Inspect the collection manifest for holidays, unavailable dates and download failures. Do not interpret strategy metrics as reliable until several years and many expiry cycles are present.

The Streamlit app is deployed from this repository's `main` branch. The UI and research-engine changes are committed together; the hosted app updates after its GitHub deployment completes.
