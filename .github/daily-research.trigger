Regenerate daily strategy results with trade-distribution and concentration diagnostics.
Intraday SENSEX candles and Daily ATM Put filter validation are collected and saved by the scheduled workflow; the dashboard only reads persisted outputs.
Add option OHLC stop-loss and take-profit grid with a monthly PF>=1.20 screen; only display saved workflow artifacts.
Run alternative lagged entry signals, multi-horizon momentum, trend, reversal, breakout, volatility and regime research; persist monthly validation results for dashboard display.
Test alternative lagged entry signals across 1, 2, 3 and 5 session holds using observed BSE options OHLC; save monthly train and chronological holdout results.
Re-run multi-horizon entry signals with train/holdout separated by realized exit month to prevent multi-session position leakage.
Compare lagged signals on long ATM options and defined-risk ATM debit vertical spreads for 1, 2, 3 and 5 session holding periods with exit-month holdout validation.
Run strict multi-horizon entry research on long options and ATM debit spreads with one position at a time; require every active month to have profit factor >= 1.20 in both training and holdout.
Compare one, two and three available strike-width ATM debit vertical spreads under strict monthly PF 1.20 validation and sequential one-position allocation across all holding periods.
