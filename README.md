# SENSEX Weekly Options Backtester

Research backtesting platform for a **long SENSEX weekly Call options** strategy.

- **Initial capital (default):** ₹10,000  
- Uses **actual** historical SENSEX index data (Yahoo Finance) and **actual** BSE equity-derivatives bhavcopy option premiums.  
- **Never fabricates** option prices, lot sizes, or successful trades.  
- Missing data → trade is **skipped** with an explicit reason.

> **Disclaimer:** This is a research tool, not a live trading system. Past simulated performance does not guarantee future results. Closing prices are not guaranteed fills. Transaction costs and slippage are configurable assumptions. You are solely responsible for any trading decisions.

---

## Strategy rules

For each **actual** historical SENSEX options expiry present in BSE data:

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
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── src/sensex_options/
│   ├── config.py
│   ├── calendar.py
│   ├── features.py
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
py -3.11 -m venv .venv
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
python3.11 -m venv .venv
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

### Build & validate

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

Default research configuration uses **zero** brokerage/STT/slippage. This is **unrealistic**. Configure costs in the Streamlit form or via `CostConfig` / CLI flags. Historical rate schedules change; assumptions must be labelled.

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

Strategies: ATM call, ATM put, one-step OTM call, one-step OTM put, and ATM long straddle. The leaderboard ranks the complete available historical replay by net P&L (not a prediction of tomorrow's best strategy). P&L is simulated from daily closing marks, not live fills. Estimated costs in this first version include configurable brokerage and slippage only; statutory charges, spread, impact, funding, margin and taxes are not fully modelled.

**Important setup:** this workflow needs an initial historical dataset to be useful. The first scheduled run only adds the latest available day; it does not backfill all history. Run a historical BSE data bootstrap separately before relying on rankings. The workflow commits processed CSV history and reports to this repository so the Streamlit app can read them; this increases repository size over time. For long-term production, move market history to object storage or a database and keep only compact reports in Git.

The daily workflow is research/replay, not machine-learning model training and not live order execution. It never places trades. If BSE has not published a file yet or the date is a holiday, inspect the Actions log and rerun after publication.
