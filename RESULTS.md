# Sample run results (generated 2026-11-06)

Research sample only. **Not** investment advice. Zero transaction costs.

## Data collected

| Source | Coverage | Rows |
|--------|----------|------|
| SENSEX index (`^BSESN` via yfinance) | 2023-05-15 → 2026-11-05 | 858 |
| BSE SENSEX Call options (FO bhavcopy) | 2024-07-08 → 2024-10-31 | 9,070 |
| Distinct expiries in options sample | — | 23 |

Index and processed options are under `data/processed/`. Raw FO files under `data/raw/options/` (some holiday downloads are HTML error pages and were skipped).

## Backtest (₹10,000 initial, zero costs)

| Metric | Value |
|--------|-------|
| Total expiries considered | 23 |
| Completed trades | 5 |
| Skipped | 18 (12 insufficient capital after drawdowns, 6 no calls on entry) |
| Winning / Losing | 2 / 3 |
| Win rate | 40% |
| Total net P&L | **₹ −9,938** |
| Ending capital | **₹ 62** |
| Max drawdown | ₹ 20,079.5 (≈ 99.7% of peak) |
| Profit factor | 0.67 |
| Expectancy / trade | ₹ −1,988 |

Strategy **lost money** in this Jul–Oct 2024 sample under the defined rules. Do not extrapolate.

## Statistics (completed trades)

- Mean PnL ≈ −1,988; median negative  
- Std PnL high relative to sample size (n=5)  
- Mean return % is unstable with small n  

## Probability (bootstrap, 2000 resamples)

- Empirical win prob = 0.40  
- 95% CI mean PnL ≈ [−11,714 , +7,739] (includes zero)  
- Prob(mean PnL > 0) ≈ 0.36  
- Empirical Kelly fraction ≈ **−0.20** (negative edge under sample)

## Index return stats (full downloaded index)

- Mean daily return ≈ 0.031%  
- Ann. Sharpe (approx) ≈ 0.61  
- Up-day frequency ≈ 52.2%  

## ML / unsupervised (research features only)

1. **K-means regimes** (k=3 on 5-day return + 20-day vol): low / mid / high vol labels saved to `outputs/index_regimes.csv`. Not trading signals.  
2. **Trade proxy**: correlation of prior 5-day index return vs win flag on n=5 trades ≈ −0.79 (exploratory; tiny sample).  

Full JSON: `outputs/analytics_report.json`.

## How to reproduce

```bash
export PYTHONPATH=src
python -m sensex_options.cli collect-index --start 2023-05-15 --end 2026-11-05
python -m sensex_options.cli collect-options --start 2024-07-01 --end 2024-10-31
python -m sensex_options.cli validate
python -m sensex_options.cli backtest --capital 10000 --cost-per-lot 0
python -m sensex_options.cli analyze
pytest -q
```

## Caveats

- Expiry-day BSE closes sometimes equal the index; parser sanitises using last / settle / intrinsic.  
- Lot size taken from `NewBrdLotQty` when present.  
- Sample period is short; many trades skipped after capital was nearly wiped.  
- Zero costs are unrealistic.  
