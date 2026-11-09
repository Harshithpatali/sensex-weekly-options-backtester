Regenerate daily strategy results with trade-distribution and concentration diagnostics.
Intraday SENSEX candles and Daily ATM Put filter validation are collected and saved by the scheduled workflow; the dashboard only reads persisted outputs.
Add option OHLC stop-loss and take-profit grid with a monthly PF>=1.20 screen; only display saved workflow artifacts.
Run alternative lagged entry signals, multi-horizon momentum, trend, reversal, breakout, volatility and regime research; persist monthly validation results for dashboard display.
Test alternative lagged entry signals across 1, 2, 3 and 5 session holds using observed BSE options OHLC; save monthly train and chronological holdout results.
Re-run multi-horizon entry signals with train/holdout separated by realized exit month to prevent multi-session position leakage.
Compare lagged signals on long ATM options and defined-risk ATM debit vertical spreads for 1, 2, 3 and 5 session holding periods with exit-month holdout validation.
Run strict multi-horizon entry research on long options and ATM debit spreads with one position at a time; require every active month to have profit factor >= 1.20 in both training and holdout.
Compare one, two and three available strike-width ATM debit vertical spreads under strict monthly PF 1.20 validation and sequential one-position allocation across all holding periods.
Expand strategy research with option-chain positioning, put/call OI and volume ratios, moneyness variants, volatility-normalized signals, DMI filters, and defined-risk credit spreads. Re-run chronological monthly PF validation.
Test ATM, one-strike ITM and one-strike OTM long options with lagged volatility/trend/RSI/MACD/candle-rejection gap-fade filters; preserve multi-width debit spread tests and strict monthly PF>=1.20 validation.
Run the latest validated option-flow, moneyness, debit/credit spread and regime-signal research with the corrected bullish-put/bearish-call credit-spread mapping and vertical payoff bounds.
Replay with exact-duplicate signal sequences removed; keep chronological monthly PF validation and save all result tables.
Run full option-flow, moneyness, volatility-regime and debit/credit spread signal grid with compressed trade-ledger persistence and read-only dashboard outputs.

Apply liquidity-aware option selection and strict ₹10,000 cash/risk feasibility controls. Rebuild all strategy results, regime filters, risk-managed variants, and validation artifacts from the accumulated BSE options archive. Do not report signal-level unit-lot P&L as account-level returns.

Re-run against the latest main commit: persist the independent chronological ₹10,000 capital-controlled holdout and the liquidity-filtered daily strategy ledger; update dashboard artifacts.

research mode: capital-independent one-lot strategy outcomes; optional capital sizing remains a separate report
