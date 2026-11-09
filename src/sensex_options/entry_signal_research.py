from __future__ import annotations

"""Lagged index-signal research for long SENSEX ATM options.

Each candidate selects CE, PE, or stays in cash using completed index data
available before the option entry session. P&L comes from the paired, observed
BSE one-session ATM call/put ledger; no synthetic option returns are generated.
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from sensex_options.config import CostConfig
from sensex_options.costs import compute_transaction_costs

TARGET_PF = 1.20
MIN_MONTH_TRADES = 3
MIN_TRAIN_TRADES = 50
MIN_HOLDOUT_TRADES = 20


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses > 0:
        return gains / losses
    return float("inf") if gains > 0 else None


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.mask((loss == 0) & (gain > 0), 100.0)
    value = value.mask((gain == 0) & (loss > 0), 0.0)
    return value


def build_lagged_features(index: pd.DataFrame) -> pd.DataFrame:
    """Build features through each completed session; caller maps prior row to entry."""
    required = {"date", "close"}
    if index is None or index.empty or not required.issubset(index.columns):
        return pd.DataFrame()
    x = index.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for col in ("open", "high", "low", "close"):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if x.empty:
        return x

    close = x["close"]
    high = x["high"] if "high" in x else close
    low = x["low"] if "low" in x else close
    open_ = x["open"] if "open" in x else close
    ret = close.pct_change()
    x["ret_1"] = close.pct_change(1)
    x["realized_vol_20"] = ret.rolling(20, min_periods=15).std()
    for period in (2, 3, 5, 10, 20):
        x[f"ret_{period}"] = close.pct_change(period)
        x[f"ret_z_{period}"] = (
            x[f"ret_{period}"] /
            (x["realized_vol_20"] * np.sqrt(period)).replace(0, np.nan)
        )

    for span in (5, 10, 20, 30, 50):
        x[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    x["ema_5_20"] = x["ema_5"] / x["ema_20"] - 1
    x["ema_10_30"] = x["ema_10"] / x["ema_30"] - 1
    x["ema_20_50"] = x["ema_20"] / x["ema_50"] - 1
    for gap_name in ("ema_5_20", "ema_10_30", "ema_20_50"):
        x[f"{gap_name}_z"] = (
            x[gap_name] / x["realized_vol_20"].replace(0, np.nan)
        )
    x["rsi_7"] = _rsi(close, 7)
    x["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    x["macd_hist_pct"] = (macd - macd_signal) / close

    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    x["bb_z"] = (close - mid) / sd.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    x["atr_pct"] = true_range.rolling(14, min_periods=14).mean() / close
    x["atr_pct_median_60"] = x["atr_pct"].shift(1).rolling(60, min_periods=30).median()
    x["atr_pct_q75_60"] = x["atr_pct"].shift(1).rolling(60, min_periods=30).quantile(0.75)

    # ADX-like trend strength calculated from completed OHLC observations.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.rolling(14, min_periods=14).mean()
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["plus_di_14"] = plus_di
    x["minus_di_14"] = minus_di
    x["adx_14"] = dx.rolling(14, min_periods=14).mean()

    # Donchian boundaries exclude the signal session itself.
    for period in (5, 10, 20):
        x[f"donchian_high_{period}"] = high.shift(1).rolling(period, min_periods=period).max()
        x[f"donchian_low_{period}"] = low.shift(1).rolling(period, min_periods=period).min()

    x["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    x["gap_pct"] = open_ / previous_close - 1
    x["range_pct"] = (high - low) / previous_close
    # Compression is today's completed range against its trailing prior range median.
    x["range_median_20"] = x["range_pct"].shift(1).rolling(20, min_periods=15).median()
    x["range_compression"] = x["range_pct"] / x["range_median_20"].replace(0, np.nan)
    return x


def _default_signal_cost() -> CostConfig:
    return CostConfig(
        brokerage_per_lot=20.0,
        stt_pct=0.001,
        exchange_charges_pct=0.0003,
        gst_pct=0.18,
        stamp_duty_pct=0.00003,
        entry_slippage=0.5,
        exit_slippage=0.5,
        spread_points=1.0,
        market_impact_points=0.25,
    )


def _filter_liquid_option_rows(
    options: pd.DataFrame,
    min_volume: float = 25.0,
    min_open_interest: float = 50.0,
) -> pd.DataFrame:
    """Keep only contracts with observable, non-zero EOD activity and OI.

    BSE bhavcopy has no bid/ask. Requiring both volume and OI on entry and
    exit days is a conservative stale-mark screen, not proof of executable fills.
    Missing liquidity fields are rejected rather than silently treated as liquid.
    """
    required = {"volume", "open_interest", "close"}
    if options is None or options.empty or not required.issubset(options.columns):
        return pd.DataFrame(columns=list(options.columns) if options is not None else [])
    out = options.copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[
        (out["close"] >= 0)
        & (
            ((out["volume"] >= float(min_volume)) & (out["open_interest"] >= float(min_open_interest)))
            | (out["close"] == 0)
        )
    ].copy()


def _is_liquid_option_row(row, min_volume: float = 25.0, min_open_interest: float = 50.0) -> bool:
    close = row.get("close")
    volume = row.get("volume")
    open_interest = row.get("open_interest")
    return (
        _finite(close) and float(close) > 0
        and _finite(volume) and float(volume) >= min_volume
        and _finite(open_interest) and float(open_interest) >= min_open_interest
    )


def _option_horizon_outcomes(
    index: pd.DataFrame,
    options: pd.DataFrame,
    holding_periods: tuple[int, ...] = (1, 2, 3, 5),
    moneyness_variants: tuple[str, ...] = ("ATM", "ITM1", "OTM1", "OTM2", "OTM3", "OTM5"),
    capital: float = 10000.0,
    risk_pct: float | None = None,
    cost: CostConfig | None = None,
) -> pd.DataFrame:
    """Build observed outcomes for ATM, ITM1, and OTM1/2/3/5 contracts.

    CE ITM uses lower strikes and CE OTM uses higher strikes; PE is reversed.
    Entry uses the previous completed SENSEX close, and the exact selected
    contract is marked at the requested future session close. Full premium
    plus estimated entry fees must fit the configured per-trade risk budget.
    """
    if index is None or index.empty or options is None or options.empty:
        return pd.DataFrame()
    if not {"date", "close"}.issubset(index.columns):
        return pd.DataFrame()
    required = {"trade_date", "expiry_date", "strike", "option_type", "close", "lot_size"}
    if not required.issubset(options.columns):
        return pd.DataFrame()

    cfg = cost or _default_signal_cost()
    idx = index.copy()
    idx["date"] = pd.to_datetime(idx["date"], errors="coerce").dt.normalize()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "close", "lot_size"):
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper() if "symbol" in opt else "SENSEX"
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[
        (opt["strike"] > 0)
        & (opt["close"] >= 0)
        & (opt["option_type"].isin(["CE", "PE"]))
    ].copy()
    # Preserve observed zero-premium expiry payoffs. Entry liquidity and
    # non-expiry exit liquidity are validated independently below.
    opt = opt.sort_values(["trade_date", "expiry_date", "strike", "symbol"])
    by_day = {day: frame for day, frame in opt.groupby("trade_date", sort=False)}
    sessions = idx.reset_index(drop=True)
    rows = []
    valid_moneyness = tuple(v for v in moneyness_variants if v in {"ATM", "ITM1", "OTM1", "OTM2", "OTM3", "OTM5"})
    entry_adjustment = cfg.entry_slippage + cfg.spread_points / 2 + cfg.market_impact_points

    for hold in sorted(set(int(h) for h in holding_periods if int(h) > 0)):
        for entry_pos in range(1, len(sessions) - hold):
            signal_row = sessions.iloc[entry_pos - 1]
            entry_row = sessions.iloc[entry_pos]
            exit_row = sessions.iloc[entry_pos + hold]
            entry_day = pd.Timestamp(entry_row["date"])
            exit_day = pd.Timestamp(exit_row["date"])
            reference_spot = float(signal_row["close"])
            entry_options = by_day.get(entry_day)
            exit_options = by_day.get(exit_day)
            if entry_options is None or exit_options is None or entry_options.empty or exit_options.empty:
                continue

            eligible_expiries = sorted(
                entry_options.loc[entry_options["expiry_date"] >= exit_day, "expiry_date"].dropna().unique()
            )
            if not eligible_expiries:
                continue
            expiry = pd.Timestamp(eligible_expiries[0])
            expiry_sessions = sessions.loc[sessions["date"] <= expiry, "date"]
            if expiry_sessions.empty:
                continue
            last_expiry_session = pd.Timestamp(expiry_sessions.max())
            chain = entry_options[entry_options["expiry_date"] == expiry]
            strikes = sorted(pd.to_numeric(chain["strike"], errors="coerce").dropna().unique())
            if not strikes:
                continue
            atm_index = min(range(len(strikes)), key=lambda i: abs(float(strikes[i]) - reference_spot))
            atm = float(strikes[atm_index])

            for moneyness in valid_moneyness:
                if moneyness == "ATM":
                    side_strikes = {"CE": atm, "PE": atm}
                elif moneyness == "ITM1":
                    side_strikes = {
                        "CE": float(strikes[atm_index - 1]) if atm_index > 0 else None,
                        "PE": float(strikes[atm_index + 1]) if atm_index + 1 < len(strikes) else None,
                    }
                elif moneyness == "OTM1":
                    side_strikes = {
                        "CE": float(strikes[atm_index + 1]) if atm_index + 1 < len(strikes) else None,
                        "PE": float(strikes[atm_index - 1]) if atm_index > 0 else None,
                    }
                else:
                    offset = int(moneyness.replace("OTM", ""))
                    side_strikes = {
                        "CE": float(strikes[atm_index + offset]) if atm_index + offset < len(strikes) else None,
                        "PE": float(strikes[atm_index - offset]) if atm_index - offset >= 0 else None,
                    }

                for option_type in ("CE", "PE"):
                    selected_strike = side_strikes.get(option_type)
                    if selected_strike is None:
                        continue
                    entry_candidates = chain[
                        (chain["strike"] == selected_strike) & (chain["option_type"] == option_type)
                    ]
                    if entry_candidates.empty:
                        continue
                    selected = entry_candidates.iloc[0]
                    entry_premium = float(selected["close"])
                    entry_volume = selected.get("volume")
                    entry_oi = selected.get("open_interest")
                    if (
                        not _finite(entry_volume) or float(entry_volume) < 25
                        or not _finite(entry_oi) or float(entry_oi) < 50
                    ):
                        continue
                    lot_value = selected.get("lot_size")
                    symbol = str(selected.get("symbol", "SENSEX")).strip().upper()
                    if not _finite(entry_premium) or entry_premium <= 0 or not _finite(lot_value) or float(lot_value) <= 0:
                        continue
                    lot_size = int(lot_value)
                    adjusted_entry = entry_premium + entry_adjustment
                    entry_turnover = adjusted_entry * lot_size
                    entry_fees = (
                        cfg.brokerage_per_lot + cfg.other_charges
                        + (cfg.brokerage_pct + cfg.exchange_charges_pct) * entry_turnover
                        + cfg.gst_pct * (
                            cfg.brokerage_per_lot + cfg.other_charges
                            + (cfg.brokerage_pct + cfg.exchange_charges_pct) * entry_turnover
                        )
                        + cfg.stamp_duty_pct * entry_turnover
                    )
                    # Long options can lose the full premium. Include the full
                    # conservative round-trip cost at a zero-value exit.
                    worst_case_costs = compute_transaction_costs(
                        entry_premium, 0.0, lot_size, 1, cfg
                    )
                    worst_case_loss = entry_premium * lot_size + worst_case_costs["total_cost"]
                    if risk_pct is not None and worst_case_loss > capital * risk_pct:
                        continue

                    exit_candidates = exit_options[
                        (exit_options["expiry_date"] == expiry)
                        & (exit_options["strike"] == selected_strike)
                        & (exit_options["option_type"] == option_type)
                        & (exit_options["symbol"] == symbol)
                    ]
                    if exit_candidates.empty:
                        continue
                    exit_row = exit_candidates.iloc[-1]
                    exit_premium = float(exit_row["close"])
                    if not _finite(exit_premium) or exit_premium < 0:
                        continue
                    if exit_premium > 0:
                        exit_volume = exit_row.get("volume")
                        exit_oi = exit_row.get("open_interest")
                        if (
                            not _finite(exit_volume) or float(exit_volume) < 25
                            or not _finite(exit_oi) or float(exit_oi) < 50
                        ):
                            continue
                    elif exit_day != last_expiry_session:
                        continue
                    quantity = lot_size
                    gross = (exit_premium - entry_premium) * quantity
                    charges = compute_transaction_costs(entry_premium, exit_premium, lot_size, 1, cfg)
                    net = gross - float(charges["total_cost"])
                    rows.append({
                        "entry_date": entry_day,
                        "exit_date": exit_day,
                        "signal_date": pd.Timestamp(signal_row["date"]),
                        "holding_sessions": hold,
                        "expiry_date": expiry,
                        "atm_reference_strike": atm,
                        "selected_strike": selected_strike,
                        "moneyness": moneyness,
                        "option_family": f"LONG_OPTIONS_{moneyness}",
                        "symbol": symbol,
                        "option_type": option_type,
                        "entry_premium": entry_premium,
                        "exit_premium": exit_premium,
                        "lot_size": lot_size,
                        "quantity": quantity,
                        "net_pnl": round(net, 2),
                        "return_pct": round(net / (entry_premium * lot_size) * 100, 5),
                    })

    outcomes = pd.DataFrame(rows)
    if outcomes.empty:
        return outcomes
    keys = [
        "entry_date", "exit_date", "holding_sessions", "expiry_date",
        "atm_reference_strike", "moneyness", "option_family",
    ]
    value_columns = ["selected_strike", "net_pnl", "entry_premium", "return_pct", "lot_size"]
    calls = outcomes[outcomes["option_type"] == "CE"][keys + value_columns].rename(
        columns={
            "selected_strike": "call_strike", "net_pnl": "call_pnl",
            "entry_premium": "call_entry_premium", "return_pct": "call_return_pct",
            "lot_size": "call_lot_size",
        }
    )
    puts = outcomes[outcomes["option_type"] == "PE"][keys + value_columns].rename(
        columns={
            "selected_strike": "put_strike", "net_pnl": "put_pnl",
            "entry_premium": "put_entry_premium", "return_pct": "put_return_pct",
            "lot_size": "put_lot_size",
        }
    )
    combined = calls.merge(puts, on=keys, how="outer", validate="one_to_one")
    combined["month"] = pd.to_datetime(combined["exit_date"]).dt.strftime("%Y-%m")
    combined["signal_date"] = pd.to_datetime(combined["entry_date"]).map(
        dict(zip(sessions["date"], sessions["date"].shift(1)))
    )
    return combined.dropna(subset=["signal_date"]).sort_values(
        ["entry_date", "holding_sessions", "moneyness"]
    ).reset_index(drop=True)


def _debit_vertical_costs(
    entry_long: float,
    entry_short: float,
    exit_long: float,
    exit_short: float,
    quantity: int,
    cost: CostConfig,
) -> dict[str, float]:
    """Estimate charges for four orders across a defined-risk debit spread."""
    turnover = quantity * (entry_long + entry_short + exit_long + exit_short)
    brokerage = cost.brokerage_per_lot * 4 + cost.brokerage_pct * turnover
    exchange = cost.exchange_charges_pct * turnover
    gst = cost.gst_pct * (brokerage + exchange)
    # Sell-side STT: short leg opened and long leg closed.
    stt = cost.stt_pct * quantity * (entry_short + exit_long)
    # Buy-side stamp duty: long leg opened and short leg closed.
    stamp = cost.stamp_duty_pct * quantity * (entry_long + exit_short)
    other = cost.other_charges * 4
    slippage = (cost.entry_slippage + cost.exit_slippage) * quantity * 2
    spread = cost.spread_points * quantity * 2
    market_impact = cost.market_impact_points * quantity * 4
    total = brokerage + exchange + gst + stt + stamp + other + slippage + spread + market_impact
    return {
        "brokerage": round(brokerage, 4),
        "exchange": round(exchange, 4),
        "gst": round(gst, 4),
        "stt": round(stt, 4),
        "stamp_duty": round(stamp, 4),
        "other": round(other, 4),
        "slippage": round(slippage, 4),
        "spread": round(spread, 4),
        "market_impact": round(market_impact, 4),
        "total_cost": round(total, 4),
    }


def _debit_vertical_entry_fees(
    entry_long: float,
    entry_short: float,
    quantity: int,
    cost: CostConfig,
) -> float:
    """Entry-only fees and conservative execution costs for both spread legs."""
    turnover = quantity * (entry_long + entry_short)
    brokerage = cost.brokerage_per_lot * 2 + cost.brokerage_pct * turnover
    exchange = cost.exchange_charges_pct * turnover
    gst = cost.gst_pct * (brokerage + exchange)
    stt = cost.stt_pct * quantity * entry_short
    stamp = cost.stamp_duty_pct * quantity * entry_long
    other = cost.other_charges * 2
    slippage = cost.entry_slippage * quantity * 2
    spread = cost.spread_points * quantity
    market_impact = cost.market_impact_points * quantity * 2
    return brokerage + exchange + gst + stt + stamp + other + slippage + spread + market_impact


def _option_debit_vertical_outcomes(
    index: pd.DataFrame,
    options: pd.DataFrame,
    holding_periods: tuple[int, ...] = (1, 2, 3, 5),
    spread_width_steps: tuple[int, ...] = (1, 2, 3),
    capital: float = 10000.0,
    risk_pct: float | None = None,
    cost: CostConfig | None = None,
) -> pd.DataFrame:
    """Generate ATM debit vertical outcomes from observed BSE option candles.

    Bull call spread: long ATM CE, short a higher CE strike. Bear put spread:
    long ATM PE, short a lower PE strike. Both legs must share expiry and lot
    size, be liquid at entry/exit, and fit the per-trade max-loss budget.
    """
    if index is None or index.empty or options is None or options.empty:
        return pd.DataFrame()
    if not {"date", "close"}.issubset(index.columns):
        return pd.DataFrame()
    required = {"trade_date", "expiry_date", "strike", "option_type", "close", "lot_size"}
    if not required.issubset(options.columns):
        return pd.DataFrame()

    cfg = cost or _default_signal_cost()
    idx = index.copy()
    idx["date"] = pd.to_datetime(idx["date"], errors="coerce").dt.normalize()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "close", "lot_size"):
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper() if "symbol" in opt else "SENSEX"
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[
        (opt["strike"] > 0)
        & (opt["close"] >= 0)
        & (opt["option_type"].isin(["CE", "PE"]))
    ].sort_values(["trade_date", "expiry_date", "strike", "symbol"])
    # Both legs must be liquid on entry and exit dates.
    opt = _filter_liquid_option_rows(opt)
    by_day = {day: frame for day, frame in opt.groupby("trade_date", sort=False)}
    sessions = idx.reset_index(drop=True)
    rows = []

    for hold in sorted(set(int(h) for h in holding_periods if int(h) > 0)):
        for entry_pos in range(1, len(sessions) - hold):
            signal_row = sessions.iloc[entry_pos - 1]
            entry_row = sessions.iloc[entry_pos]
            exit_row = sessions.iloc[entry_pos + hold]
            entry_day = pd.Timestamp(entry_row["date"])
            exit_day = pd.Timestamp(exit_row["date"])
            reference_spot = float(signal_row["close"])
            entry_options = by_day.get(entry_day)
            exit_options = by_day.get(exit_day)
            if entry_options is None or exit_options is None or entry_options.empty or exit_options.empty:
                continue

            eligible_expiries = sorted(
                entry_options.loc[entry_options["expiry_date"] >= exit_day, "expiry_date"].dropna().unique()
            )
            if not eligible_expiries:
                continue
            expiry = pd.Timestamp(eligible_expiries[0])
            expiry_sessions = sessions.loc[sessions["date"] <= expiry, "date"]
            if expiry_sessions.empty:
                continue
            last_expiry_session = pd.Timestamp(expiry_sessions.max())
            chain = entry_options[entry_options["expiry_date"] == expiry]
            strikes = sorted(pd.to_numeric(chain["strike"], errors="coerce").dropna().unique())
            if len(strikes) < 2:
                continue
            atm = float(min(strikes, key=lambda strike: abs(float(strike) - reference_spot)))
            higher = [float(strike) for strike in strikes if float(strike) > atm]
            lower = [float(strike) for strike in strikes if float(strike) < atm]
            for width in sorted(set(int(w) for w in spread_width_steps if int(w) > 0)):
                short_call_strike = higher[width - 1] if len(higher) >= width else None
                short_put_strike = lower[-width] if len(lower) >= width else None
                spread_specs = (
                    ("CE", atm, short_call_strike, "BULL_CALL_DEBIT_SPREAD"),
                    ("PE", atm, short_put_strike, "BEAR_PUT_DEBIT_SPREAD"),
                )
                result_by_direction = {}
                for option_type, long_strike, short_strike, label in spread_specs:
                    if short_strike is None:
                        continue
                    long_entry_rows = chain[
                        (chain["strike"] == long_strike) & (chain["option_type"] == option_type)
                    ]
                    short_entry_rows = chain[
                        (chain["strike"] == short_strike) & (chain["option_type"] == option_type)
                    ]
                    if long_entry_rows.empty or short_entry_rows.empty:
                        continue
                    long_entry = long_entry_rows.iloc[0]
                    short_entry = short_entry_rows.iloc[0]
                    if not _is_liquid_option_row(long_entry) or not _is_liquid_option_row(short_entry):
                        continue
                    long_symbol = str(long_entry.get("symbol", "SENSEX")).strip().upper()
                    short_symbol = str(short_entry.get("symbol", "SENSEX")).strip().upper()
                    long_lot = long_entry.get("lot_size")
                    short_lot = short_entry.get("lot_size")
                    if (
                        not _finite(long_lot) or not _finite(short_lot)
                        or int(long_lot) <= 0 or int(short_lot) != int(long_lot)
                        or long_symbol != short_symbol
                    ):
                        continue
                    lot_size = int(long_lot)
                    entry_long_premium = float(long_entry["close"])
                    entry_short_premium = float(short_entry["close"])
                    spread_width = abs(float(long_strike) - float(short_strike))
                    entry_debit = entry_long_premium - entry_short_premium
                    if (
                        not _finite(entry_debit)
                        or entry_debit <= 0
                        or entry_debit >= spread_width
                    ):
                        continue

                    long_exit_rows = exit_options[
                        (exit_options["expiry_date"] == expiry)
                        & (exit_options["strike"] == long_strike)
                        & (exit_options["option_type"] == option_type)
                        & (exit_options["symbol"] == long_symbol)
                    ]
                    short_exit_rows = exit_options[
                        (exit_options["expiry_date"] == expiry)
                        & (exit_options["strike"] == short_strike)
                        & (exit_options["option_type"] == option_type)
                        & (exit_options["symbol"] == short_symbol)
                    ]
                    if long_exit_rows.empty or short_exit_rows.empty:
                        continue
                    long_exit = long_exit_rows.iloc[-1]
                    short_exit = short_exit_rows.iloc[-1]
                    exit_long_premium = float(long_exit["close"])
                    exit_short_premium = float(short_exit["close"])
                    if (
                        not _finite(exit_long_premium) or not _finite(exit_short_premium)
                        or exit_long_premium < 0 or exit_short_premium < 0
                    ):
                        continue
                    if (
                        (exit_long_premium > 0 and not _is_liquid_option_row(long_exit))
                        or (exit_short_premium > 0 and not _is_liquid_option_row(short_exit))
                        or ((exit_long_premium == 0 or exit_short_premium == 0) and exit_day != last_expiry_session)
                    ):
                        continue
                    exit_credit = exit_long_premium - exit_short_premium
                    if exit_credit < 0 or exit_credit > spread_width:
                        # A vertical spread cannot be worth less than zero or more than its width.
                        continue

                    quantity = lot_size
                    entry_fees = _debit_vertical_entry_fees(
                        entry_long_premium, entry_short_premium, quantity, cfg
                    )
                    # Debit spread max loss is the debit plus full round-trip
                    # costs under a worthless-expiry exit.
                    worst_case_costs = _debit_vertical_costs(
                        entry_long_premium, entry_short_premium, 0.0, 0.0,
                        quantity, cfg,
                    )
                    worst_case_loss = entry_debit * quantity + worst_case_costs["total_cost"]
                    if risk_pct is not None and worst_case_loss > capital * risk_pct:
                        continue
                    charges = _debit_vertical_costs(
                        entry_long_premium, entry_short_premium,
                        exit_long_premium, exit_short_premium,
                        quantity, cfg,
                    )
                    net = (exit_credit - entry_debit) * quantity - charges["total_cost"]
                    result_by_direction[label] = round(net, 2)

                if result_by_direction:
                    rows.append({
                        "entry_date": entry_day,
                        "exit_date": exit_day,
                        "signal_date": pd.Timestamp(signal_row["date"]),
                        "holding_sessions": hold,
                        "expiry_date": expiry,
                        "strike": atm,
                        "month": exit_day.strftime("%Y-%m"),
                        "option_family": f"DEBIT_SPREADS_W{width}",
                        "spread_width_steps": width,
                        "call_pnl": result_by_direction.get("BULL_CALL_DEBIT_SPREAD", np.nan),
                        "put_pnl": result_by_direction.get("BEAR_PUT_DEBIT_SPREAD", np.nan),
                    })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["entry_date", "holding_sessions"]
    ).reset_index(drop=True)



def _option_chain_market_features(
    index: pd.DataFrame,
    options: pd.DataFrame,
) -> pd.DataFrame:
    """Compute past-only option-chain PCR, OI build-up and straddle/RV features.

    Values are calculated from the option bhavcopy for the completed signal
    session and the same expiry selected for the eventual trade. OI/premium
    changes use the previous available observation of the exact contract.
    """
    required = {"date", "close"}
    if index is None or index.empty or not required.issubset(index.columns):
        return pd.DataFrame()
    if options is None or options.empty:
        return pd.DataFrame()

    idx = index.copy()
    idx["date"] = pd.to_datetime(idx["date"], errors="coerce").dt.normalize()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    idx["realized_vol20"] = idx["close"].pct_change().rolling(20, min_periods=15).std()
    spot_by_date = dict(zip(idx["date"], idx["close"]))
    vol_by_date = dict(zip(idx["date"], idx["realized_vol20"]))

    opt = options.copy()
    needed = {"trade_date", "expiry_date", "strike", "option_type", "close"}
    if not needed.issubset(opt.columns):
        return pd.DataFrame()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "close", "volume", "open_interest"):
        if col not in opt.columns:
            opt[col] = np.nan
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper() if "symbol" in opt else "SENSEX"
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[
        (opt["strike"] > 0)
        & (opt["close"] >= 0)
        & opt["option_type"].isin(["CE", "PE"])
    ].drop_duplicates(
        ["trade_date", "expiry_date", "strike", "symbol", "option_type"],
        keep="last",
    ).sort_values(["expiry_date", "strike", "symbol", "option_type", "trade_date"])

    contract_keys = ["expiry_date", "strike", "symbol", "option_type"]
    grouped = opt.groupby(contract_keys, sort=False)
    opt["previous_oi"] = grouped["open_interest"].shift(1)
    opt["previous_close"] = grouped["close"].shift(1)
    opt["oi_change"] = opt["open_interest"] - opt["previous_oi"]
    opt["oi_change_pct"] = opt["oi_change"] / opt["previous_oi"].replace(0, np.nan)
    opt["premium_change_pct"] = (
        opt["close"] / opt["previous_close"].replace(0, np.nan) - 1
    )

    rows = []
    for (trade_date, expiry), chain in opt.groupby(["trade_date", "expiry_date"], sort=False):
        spot = spot_by_date.get(trade_date)
        if not _finite(spot) or float(spot) <= 0:
            continue
        strikes = sorted(chain["strike"].dropna().unique())
        if not strikes:
            continue
        atm = float(min(strikes, key=lambda value: abs(float(value) - float(spot))))
        atm_pos = strikes.index(atm)
        near_strikes = set(float(value) for value in strikes[max(0, atm_pos - 2):atm_pos + 3])
        near = chain[chain["strike"].isin(near_strikes)]
        side_stats = {}
        for option_type in ("CE", "PE"):
            side = near[near["option_type"] == option_type]
            if side.empty:
                side_stats[option_type] = {
                    "volume": np.nan, "oi": np.nan, "previous_oi": np.nan,
                    "oi_change": np.nan, "oi_change_pct": np.nan,
                    "premium_change_pct": np.nan,
                }
                continue
            volume = side["volume"].sum(min_count=1)
            oi = side["open_interest"].sum(min_count=1)
            previous_oi = side["previous_oi"].sum(min_count=1)
            oi_change = side["oi_change"].sum(min_count=1)
            side_stats[option_type] = {
                "volume": float(volume) if _finite(volume) else np.nan,
                "oi": float(oi) if _finite(oi) else np.nan,
                "previous_oi": float(previous_oi) if _finite(previous_oi) else np.nan,
                "oi_change": float(oi_change) if _finite(oi_change) else np.nan,
                "oi_change_pct": (
                    float(oi_change / previous_oi)
                    if _finite(oi_change) and _finite(previous_oi) and float(previous_oi) > 0
                    else np.nan
                ),
                "premium_change_pct": (
                    float(side["premium_change_pct"].median())
                    if side["premium_change_pct"].notna().any() else np.nan
                ),
            }

        atm_rows = chain[chain["strike"] == atm]
        call_atm = atm_rows[atm_rows["option_type"] == "CE"]
        put_atm = atm_rows[atm_rows["option_type"] == "PE"]
        call_premium = float(call_atm.iloc[0]["close"]) if not call_atm.empty else np.nan
        put_premium = float(put_atm.iloc[0]["close"]) if not put_atm.empty else np.nan
        call_volume = side_stats["CE"]["volume"]
        put_volume = side_stats["PE"]["volume"]
        call_oi = side_stats["CE"]["oi"]
        put_oi = side_stats["PE"]["oi"]
        pcr_volume = (
            put_volume / call_volume
            if _finite(put_volume) and _finite(call_volume) and call_volume > 0 else np.nan
        )
        pcr_oi = (
            put_oi / call_oi
            if _finite(put_oi) and _finite(call_oi) and call_oi > 0 else np.nan
        )
        straddle_pct = (
            (call_premium + put_premium) / float(spot)
            if _finite(call_premium) and _finite(put_premium) else np.nan
        )
        tenor_sessions = max((pd.Timestamp(expiry) - pd.Timestamp(trade_date)).days * 5 / 7, 1.0)
        implied_daily_move = (
            straddle_pct / np.sqrt(2 * tenor_sessions / np.pi)
            if _finite(straddle_pct) else np.nan
        )
        realized_vol = vol_by_date.get(trade_date)
        implied_rv_ratio = (
            implied_daily_move / float(realized_vol)
            if _finite(implied_daily_move) and _finite(realized_vol) and float(realized_vol) > 0
            else np.nan
        )
        call_change = side_stats["CE"]["oi_change_pct"]
        put_change = side_stats["PE"]["oi_change_pct"]
        call_premium_ret = side_stats["CE"]["premium_change_pct"]
        put_premium_ret = side_stats["PE"]["premium_change_pct"]
        # A rising premium with rising OI is classified as long build-up;
        # falling premium with rising OI as writer build-up. Direction is a
        # hypothesis, not proof of buyer/seller identity from OI alone.
        call_build = (
            float(np.sign(call_change) * np.sign(call_premium_ret))
            if _finite(call_change) and _finite(call_premium_ret) else np.nan
        )
        put_build = (
            float(-np.sign(put_change) * np.sign(put_premium_ret))
            if _finite(put_change) and _finite(put_premium_ret) else np.nan
        )
        rows.append({
            "signal_date": pd.Timestamp(trade_date),
            "expiry_date": pd.Timestamp(expiry),
            "pcr_volume_near_atm": pcr_volume,
            "pcr_oi_near_atm": pcr_oi,
            "call_oi_change_pct": call_change,
            "put_oi_change_pct": put_change,
            "call_premium_change_pct": call_premium_ret,
            "put_premium_change_pct": put_premium_ret,
            "call_build_up_score": call_build,
            "put_build_up_score": put_build,
            "positioning_score": (
                call_build + put_build
                if _finite(call_build) and _finite(put_build) else np.nan
            ),
            "atm_straddle_pct": straddle_pct,
            "implied_daily_move_proxy": implied_daily_move,
            "implied_to_realized_vol_ratio": implied_rv_ratio,
            "near_atm_call_oi": call_oi,
            "near_atm_put_oi": put_oi,
            "near_atm_call_volume": call_volume,
            "near_atm_put_volume": put_volume,
        })
    return pd.DataFrame(rows)


def _option_credit_vertical_outcomes(
    index: pd.DataFrame,
    options: pd.DataFrame,
    holding_periods: tuple[int, ...] = (1, 2, 3, 5),
    spread_width_steps: tuple[int, ...] = (1, 2, 3),
    capital: float = 10000.0,
    risk_pct: float | None = None,
    cost: CostConfig | None = None,
) -> pd.DataFrame:
    """Build defined-risk bull-put and bear-call credit spreads from real quotes.

    Risk is bounded by observed strike distance less entry credit. A position is
    skipped when estimated maximum loss plus entry fees exceeds the configured
    percentage of capital. This is not a broker margin quote.
    """
    if index is None or index.empty or options is None or options.empty:
        return pd.DataFrame()
    required = {"date", "close"}
    opt_required = {"trade_date", "expiry_date", "strike", "option_type", "close", "lot_size"}
    if not required.issubset(index.columns) or not opt_required.issubset(options.columns):
        return pd.DataFrame()
    cfg = cost or _default_signal_cost()
    idx = index.copy()
    idx["date"] = pd.to_datetime(idx["date"], errors="coerce").dt.normalize()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    opt = options.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"], errors="coerce").dt.normalize()
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"], errors="coerce").dt.normalize()
    for col in ("strike", "close", "lot_size"):
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["symbol"] = opt["symbol"].astype(str).str.strip().str.upper() if "symbol" in opt else "SENSEX"
    opt["option_type"] = opt["option_type"].astype(str).str.strip().str.upper()
    opt = opt.dropna(subset=["trade_date", "expiry_date", "strike", "close"])
    opt = opt[
        (opt["strike"] > 0) & (opt["close"] >= 0)
        & opt["option_type"].isin(["CE", "PE"])
    ].sort_values(["trade_date", "expiry_date", "strike", "symbol"])
    # Both legs must be liquid on entry and exit dates.
    opt = _filter_liquid_option_rows(opt)
    by_day = {day: frame for day, frame in opt.groupby("trade_date", sort=False)}
    rows = []

    for hold in sorted(set(int(h) for h in holding_periods if int(h) > 0)):
        for entry_pos in range(1, len(idx) - hold):
            signal_row = idx.iloc[entry_pos - 1]
            entry_row = idx.iloc[entry_pos]
            exit_row = idx.iloc[entry_pos + hold]
            entry_day = pd.Timestamp(entry_row["date"])
            exit_day = pd.Timestamp(exit_row["date"])
            entry_options = by_day.get(entry_day)
            exit_options = by_day.get(exit_day)
            if entry_options is None or exit_options is None:
                continue
            eligible = entry_options.loc[
                entry_options["expiry_date"] >= exit_day, "expiry_date"
            ].dropna().unique()
            if not len(eligible):
                continue
            expiry = pd.Timestamp(min(eligible))
            expiry_sessions = idx.loc[idx["date"] <= expiry, "date"]
            if expiry_sessions.empty:
                continue
            last_expiry_session = pd.Timestamp(expiry_sessions.max())
            chain = entry_options[entry_options["expiry_date"] == expiry]
            strikes = sorted(pd.to_numeric(chain["strike"], errors="coerce").dropna().unique())
            if len(strikes) < 2:
                continue
            atm = float(min(strikes, key=lambda value: abs(float(value) - float(signal_row["close"]))))
            atm_pos = strikes.index(atm)
            higher = [float(v) for v in strikes[atm_pos + 1:]]
            lower = [float(v) for v in reversed(strikes[:atm_pos])]
            exit_chain = exit_options[exit_options["expiry_date"] == expiry]
            for width in sorted(set(int(w) for w in spread_width_steps if int(w) > 0)):
                spread_specs = (
                    # Bullish: sell ATM PE, buy lower-strike PE hedge.
                    ("PE", atm, lower[width - 1] if len(lower) >= width else None, "BULL_PUT_CREDIT"),
                    # Bearish: sell ATM CE, buy higher-strike CE hedge.
                    ("CE", atm, higher[width - 1] if len(higher) >= width else None, "BEAR_CALL_CREDIT"),
                )
                pnl_by_direction = {}
                for option_type, short_strike, long_strike, label in spread_specs:
                    if long_strike is None:
                        continue
                    short_entries = chain[
                        (chain["strike"] == short_strike) & (chain["option_type"] == option_type)
                    ]
                    long_entries = chain[
                        (chain["strike"] == long_strike) & (chain["option_type"] == option_type)
                    ]
                    if short_entries.empty or long_entries.empty:
                        continue
                    short_entry, long_entry = short_entries.iloc[0], long_entries.iloc[0]
                    if not _is_liquid_option_row(short_entry) or not _is_liquid_option_row(long_entry):
                        continue
                    short_symbol = str(short_entry.get("symbol", "SENSEX")).strip().upper()
                    long_symbol = str(long_entry.get("symbol", "SENSEX")).strip().upper()
                    short_lot, long_lot = short_entry.get("lot_size"), long_entry.get("lot_size")
                    if (
                        not _finite(short_lot) or not _finite(long_lot)
                        or int(short_lot) <= 0 or int(short_lot) != int(long_lot)
                        or short_symbol != long_symbol
                    ):
                        continue
                    lot_size = int(short_lot)
                    short_entry_premium = float(short_entry["close"])
                    long_entry_premium = float(long_entry["close"])
                    spread_width = abs(float(short_strike) - float(long_strike))
                    entry_credit = short_entry_premium - long_entry_premium
                    if (
                        not _finite(entry_credit)
                        or entry_credit <= 0
                        or entry_credit >= spread_width
                    ):
                        continue
                    short_exits = exit_chain[
                        (exit_chain["strike"] == short_strike)
                        & (exit_chain["option_type"] == option_type)
                        & (exit_chain["symbol"] == short_symbol)
                    ]
                    long_exits = exit_chain[
                        (exit_chain["strike"] == long_strike)
                        & (exit_chain["option_type"] == option_type)
                        & (exit_chain["symbol"] == long_symbol)
                    ]
                    if short_exits.empty or long_exits.empty:
                        continue
                    short_exit, long_exit = short_exits.iloc[-1], long_exits.iloc[-1]
                    short_exit_premium = float(short_exit["close"])
                    long_exit_premium = float(long_exit["close"])
                    if (
                        not _finite(short_exit_premium) or not _finite(long_exit_premium)
                        or short_exit_premium < 0 or long_exit_premium < 0
                    ):
                        continue
                    if (
                        (short_exit_premium > 0 and not _is_liquid_option_row(short_exit))
                        or (long_exit_premium > 0 and not _is_liquid_option_row(long_exit))
                        or ((short_exit_premium == 0 or long_exit_premium == 0) and exit_day != last_expiry_session)
                    ):
                        continue
                    exit_debit = short_exit_premium - long_exit_premium
                    if exit_debit < 0 or exit_debit > spread_width:
                        # A vertical spread cannot have a liability greater than its strike width.
                        continue
                    quantity = lot_size
                    entry_fees = _debit_vertical_entry_fees(
                        long_entry_premium, short_entry_premium, quantity, cfg
                    )
                    max_loss = (abs(float(short_strike) - float(long_strike)) - entry_credit) * quantity
                    # Credit spread max loss is width less credit plus full
                    # round-trip costs at the theoretical max-loss exit.
                    worst_case_costs = _debit_vertical_costs(
                        long_entry_premium, short_entry_premium, 0.0,
                        spread_width, quantity, cfg,
                    )
                    worst_case_loss = max_loss + worst_case_costs["total_cost"]
                    if max_loss <= 0 or (risk_pct is not None and worst_case_loss > capital * risk_pct):
                        continue
                    charges = _debit_vertical_costs(
                        long_entry_premium, short_entry_premium,
                        long_exit_premium, short_exit_premium,
                        quantity, cfg,
                    )
                    net = (entry_credit - exit_debit) * quantity - charges["total_cost"]
                    pnl_by_direction[label] = round(net, 2)
                if pnl_by_direction:
                    rows.append({
                        "entry_date": entry_day,
                        "exit_date": exit_day,
                        "signal_date": pd.Timestamp(signal_row["date"]),
                        "holding_sessions": hold,
                        "expiry_date": expiry,
                        "strike": atm,
                        "month": exit_day.strftime("%Y-%m"),
                        "option_family": f"CREDIT_SPREADS_W{width}",
                        "spread_width_steps": width,
                        # CE signal means bullish exposure: use bull-put credit.
                        # PE signal means bearish exposure: use bear-call credit.
                        "call_pnl": pnl_by_direction.get("BULL_PUT_CREDIT", np.nan),
                        "put_pnl": pnl_by_direction.get("BEAR_CALL_CREDIT", np.nan),
                    })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["entry_date", "holding_sessions"]
    ).reset_index(drop=True)

def _paired_option_outcomes(daily_ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "entry_date", "exit_date", "expiry_date", "strike", "net_pnl"}
    if daily_ledger is None or daily_ledger.empty or not required.issubset(daily_ledger.columns):
        return pd.DataFrame()
    x = daily_ledger[daily_ledger["strategy"].isin(["DAILY_ATM_CALL", "DAILY_ATM_PUT"])].copy()
    if x.empty:
        return pd.DataFrame()
    for col in ("entry_date", "exit_date", "expiry_date"):
        x[col] = pd.to_datetime(x[col], errors="coerce").dt.normalize()
    x["strike"] = pd.to_numeric(x["strike"], errors="coerce")
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["entry_date", "exit_date", "expiry_date", "strike", "net_pnl"])
    keys = ["entry_date", "exit_date", "expiry_date", "strike"]
    calls = x[x["strategy"] == "DAILY_ATM_CALL"][keys + ["net_pnl"]].rename(columns={"net_pnl": "call_pnl"})
    puts = x[x["strategy"] == "DAILY_ATM_PUT"][keys + ["net_pnl"]].rename(columns={"net_pnl": "put_pnl"})
    paired = calls.merge(puts, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return paired
    paired["month"] = paired["exit_date"].dt.strftime("%Y-%m")
    return paired.sort_values("entry_date").reset_index(drop=True)


def _candidate_signals(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Return a deliberately diverse, predeclared family of signal hypotheses."""
    f = features
    signals: dict[str, pd.Series] = {}

    def direction(condition: pd.Series, valid: pd.Series | None = None) -> pd.Series:
        result = pd.Series("", index=f.index, dtype="object")
        mask = condition.notna()
        if valid is not None:
            mask &= valid.fillna(False)
        result.loc[mask & condition] = "CE"
        result.loc[mask & ~condition] = "PE"
        return result

    for period in (1, 2, 3, 5, 10):
        r = f[f"ret_{period}"]
        signals[f"RET{period}_MOMENTUM"] = direction(r > 0, r.notna())
        signals[f"RET{period}_REVERSAL"] = direction(r < 0, r.notna())

    # Volatility-normalised momentum thresholds avoid treating a tiny quiet
    # move as equivalent to a large move in a high-volatility regime.
    for period in (2, 3, 5, 10, 20):
        z = f[f"ret_z_{period}"]
        for threshold in (0.25, 0.50, 0.75, 1.00):
            sig = pd.Series("", index=f.index, dtype="object")
            sig.loc[z >= threshold] = "CE"
            sig.loc[z <= -threshold] = "PE"
            signals[f"RET{period}_Z{int(threshold * 100):03d}_MOMENTUM"] = sig

    for fast, slow, label in ((5, 20, "EMA5_20"), (10, 30, "EMA10_30"), (20, 50, "EMA20_50")):
        v = f[f"ema_{fast}"].notna() & f[f"ema_{slow}"].notna() & (f[f"ema_{fast}"] != f[f"ema_{slow}"])
        signals[f"{label}_TREND"] = direction(f[f"ema_{fast}"] > f[f"ema_{slow}"], v)
        gap_z = f[f"ema_{fast}_{slow}_z"]
        for threshold in (0.05, 0.15, 0.30):
            sig = pd.Series("", index=f.index, dtype="object")
            sig.loc[gap_z >= threshold] = "CE"
            sig.loc[gap_z <= -threshold] = "PE"
            signals[f"{label}_NORMALISED_GAP{int(threshold * 100):02d}"] = sig

    for lower, upper in ((25, 75), (30, 70), (35, 65)):
        r = f["rsi_14"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[r <= lower] = "CE"
        sig.loc[r >= upper] = "PE"
        signals[f"RSI{lower}_{upper}_REVERSAL"] = sig

    for lower, upper in ((40, 60), (45, 55)):
        r = f["rsi_14"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[r >= upper] = "CE"
        sig.loc[r <= lower] = "PE"
        signals[f"RSI{lower}_{upper}_TREND"] = sig

    macd_valid = f["macd_hist_pct"].notna() & (f["macd_hist_pct"] != 0)
    signals["MACD_HIST_TREND"] = direction(f["macd_hist_pct"] > 0, macd_valid)

    for threshold in (1.0, 1.5, 2.0):
        z = f["bb_z"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[z <= -threshold] = "CE"
        sig.loc[z >= threshold] = "PE"
        signals[f"BB_Z{str(threshold).replace('.', '_')}_REVERSAL"] = sig

    for period in (5, 10, 20):
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[f["close"] > f[f"donchian_high_{period}"]] = "CE"
        sig.loc[f["close"] < f[f"donchian_low_{period}"]] = "PE"
        signals[f"DONCHIAN{period}_BREAKOUT"] = sig

    bull = f["ema_20"] > f["ema_50"]
    bear = f["ema_20"] < f["ema_50"]
    for lower, upper in ((40, 60), (45, 55)):
        rsi = f["rsi_14"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[bull & (rsi <= lower)] = "CE"
        sig.loc[bear & (rsi >= upper)] = "PE"
        signals[f"EMA20_50_RSI{lower}_{upper}_PULLBACK"] = sig

    # Trend-following only when both independent signals agree.
    ema_bull, ema_bear = f["ema_5"] > f["ema_20"], f["ema_5"] < f["ema_20"]
    macd_bull, macd_bear = f["macd_hist_pct"] > 0, f["macd_hist_pct"] < 0
    sig = pd.Series("", index=f.index, dtype="object")
    sig.loc[ema_bull & macd_bull] = "CE"
    sig.loc[ema_bear & macd_bear] = "PE"
    signals["EMA5_20_MACD_CONFIRM"] = sig

    # Trend + ADX strength regime.
    for minimum_adx in (15, 20, 25):
        valid = (f["adx_14"] >= minimum_adx)
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[valid & ema_bull] = "CE"
        sig.loc[valid & ema_bear] = "PE"
        signals[f"EMA5_20_ADX{minimum_adx}"] = sig

    # Directional movement confirmation is separate from EMA crossover.
    dmi_valid = f["plus_di_14"].notna() & f["minus_di_14"].notna()
    signals["DMI14_DIRECTION"] = direction(
        f["plus_di_14"] > f["minus_di_14"], dmi_valid & (f["plus_di_14"] != f["minus_di_14"])
    )
    for minimum_adx in (15, 20, 25):
        valid = dmi_valid & (f["adx_14"] >= minimum_adx)
        signals[f"DMI14_ADX{minimum_adx}"] = direction(
            f["plus_di_14"] > f["minus_di_14"],
            valid & (f["plus_di_14"] != f["minus_di_14"]),
        )

    # Volatility-regime filters use past-only trailing ATR thresholds.
    for label, vol_condition in (
        ("LOW_VOL", f["atr_pct"] <= f["atr_pct_median_60"]),
        ("BELOW_Q75_VOL", f["atr_pct"] <= f["atr_pct_q75_60"]),
        ("HIGH_VOL", f["atr_pct"] > f["atr_pct_median_60"]),
    ):
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[vol_condition & ema_bull] = "CE"
        sig.loc[vol_condition & ema_bear] = "PE"
        signals[f"{label}_EMA5_20"] = sig

    # Multi-horizon consensus; no entry when the horizons disagree.
    for first, second in ((3, 5), (5, 10)):
        a, b = f[f"ret_{first}"], f[f"ret_{second}"]
        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[(a > 0) & (b > 0)] = "CE"
        sig.loc[(a < 0) & (b < 0)] = "PE"
        signals[f"RET{first}_{second}_CONSENSUS"] = sig

    # Breakouts after range compression, with an ATR floor.
    sig = pd.Series("", index=f.index, dtype="object")
    compression = f["range_compression"] <= 0.80
    atr_rising = f["atr_pct"] > f["atr_pct_median_60"]
    sig.loc[compression & atr_rising & (f["close"] > f["donchian_high_10"])] = "CE"
    sig.loc[compression & atr_rising & (f["close"] < f["donchian_low_10"])] = "PE"
    signals["COMPRESSION_VOL_BREAKOUT"] = sig

    # Opening gap continuation and fade, only for meaningful gaps.
    for threshold in (0.003, 0.005):
        gap = f["gap_pct"]
        continuation = pd.Series("", index=f.index, dtype="object")
        continuation.loc[gap >= threshold] = "CE"
        continuation.loc[gap <= -threshold] = "PE"
        fade = pd.Series("", index=f.index, dtype="object")
        fade.loc[gap >= threshold] = "PE"
        fade.loc[gap <= -threshold] = "CE"
        signals[f"GAP{int(threshold * 1000)}_CONTINUATION"] = continuation
        signals[f"GAP{int(threshold * 1000)}_FADE"] = fade

    # Conditional gap fades require independent prior-session context.
    gap3 = f["gap_pct"]
    gap_fade = pd.Series("", index=f.index, dtype="object")
    gap_fade.loc[gap3 >= 0.003] = "PE"
    gap_fade.loc[gap3 <= -0.003] = "CE"
    for label, mask in (
        ("GAP3_FADE_LOW_VOL", f["atr_pct"] <= f["atr_pct_median_60"]),
        ("GAP3_FADE_BELOW_Q75_VOL", f["atr_pct"] <= f["atr_pct_q75_60"]),
        ("GAP3_FADE_HIGH_VOL", f["atr_pct"] > f["atr_pct_median_60"]),
        ("GAP3_FADE_LOW_ADX", f["adx_14"] < 20),
        ("GAP3_FADE_TREND_ADX", f["adx_14"] >= 20),
        ("GAP3_FADE_RSI_EXTREME", (f["rsi_14"] >= 60) | (f["rsi_14"] <= 40)),
        ("GAP3_FADE_MACD_OPPOSES_GAP", ((gap3 > 0) & (f["macd_hist_pct"] < 0)) | ((gap3 < 0) & (f["macd_hist_pct"] > 0))),
        ("GAP3_FADE_REJECTION_CLOSE", ((gap3 > 0) & (f["close_location"] <= 0.35)) | ((gap3 < 0) & (f["close_location"] >= 0.65))),
        ("GAP3_FADE_BELOW_EMA20", ((gap3 > 0) & (f["close"] < f["ema_20"])) | ((gap3 < 0) & (f["close"] > f["ema_20"]))),
        ("GAP3_FADE_LOW_VOL_LOW_ADX", (f["atr_pct"] <= f["atr_pct_median_60"]) & (f["adx_14"] < 20)),
        ("GAP3_FADE_REJECTION_LOW_ADX", (((gap3 > 0) & (f["close_location"] <= 0.35)) | ((gap3 < 0) & (f["close_location"] >= 0.65))) & (f["adx_14"] < 20)),
    ):
        filtered = pd.Series("", index=f.index, dtype="object")
        valid = mask.fillna(False)
        filtered.loc[valid] = gap_fade.loc[valid]
        signals[label] = filtered

    # Option-chain positioning hypotheses. PCR is ambiguous (put demand versus
    # put writing), so both directional interpretations are tested explicitly.
    if "pcr_oi_near_atm" in f.columns:
        pcr = f["pcr_oi_near_atm"]
        for threshold in (0.80, 0.90, 1.20, 1.50):
            high = pcr >= max(threshold, 1.0 / threshold)
            low = pcr <= min(threshold, 1.0 / threshold)
            for label, call_when_high in (
                ("PUT_HEAVY_SUPPORT", True),
                ("PUT_HEAVY_DEMAND", False),
            ):
                sig = pd.Series("", index=f.index, dtype="object")
                if call_when_high:
                    sig.loc[high] = "CE"
                    sig.loc[low] = "PE"
                else:
                    sig.loc[high] = "PE"
                    sig.loc[low] = "CE"
                signals[f"PCR_OI_{int(threshold * 100)}_{label}"] = sig
    if "pcr_volume_near_atm" in f.columns:
        pcr = f["pcr_volume_near_atm"]
        for threshold in (0.80, 0.90, 1.20, 1.50):
            high = pcr >= max(threshold, 1.0 / threshold)
            low = pcr <= min(threshold, 1.0 / threshold)
            for label, call_when_high in (
                ("PUT_HEAVY_SUPPORT", True),
                ("PUT_HEAVY_DEMAND", False),
            ):
                sig = pd.Series("", index=f.index, dtype="object")
                if call_when_high:
                    sig.loc[high] = "CE"
                    sig.loc[low] = "PE"
                else:
                    sig.loc[high] = "PE"
                    sig.loc[low] = "CE"
                signals[f"PCR_VOL_{int(threshold * 100)}_{label}"] = sig

    # Price/OI build-up patterns; OI changes alone do not reveal who initiated
    # a position, so price direction is required as a second confirmation.
    for oi_threshold in (0.02, 0.05, 0.10):
        call_oi_up = f.get("call_oi_change_pct", pd.Series(np.nan, index=f.index)) >= oi_threshold
        put_oi_up = f.get("put_oi_change_pct", pd.Series(np.nan, index=f.index)) >= oi_threshold
        call_premium = f.get("call_premium_change_pct", pd.Series(np.nan, index=f.index))
        put_premium = f.get("put_premium_change_pct", pd.Series(np.nan, index=f.index))
        call_up = call_premium >= 0.01
        call_down = call_premium <= -0.01
        put_up = put_premium >= 0.01
        put_down = put_premium <= -0.01

        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[call_oi_up & put_oi_up & call_up & put_down] = "CE"
        sig.loc[call_oi_up & put_oi_up & call_down & put_up] = "PE"
        signals[f"OI_BUILDUP_BOTH_{int(oi_threshold * 100)}"] = sig

        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[call_oi_up & call_up] = "CE"
        sig.loc[call_oi_up & call_down] = "PE"
        signals[f"CALL_OI_BUILDUP_{int(oi_threshold * 100)}"] = sig

        sig = pd.Series("", index=f.index, dtype="object")
        sig.loc[put_oi_up & put_down] = "CE"
        sig.loc[put_oi_up & put_up] = "PE"
        signals[f"PUT_OI_BUILDUP_{int(oi_threshold * 100)}"] = sig

    if "positioning_score" in f.columns:
        score = f["positioning_score"]
        for threshold in (0.0, 0.5, 1.0):
            sig = pd.Series("", index=f.index, dtype="object")
            sig.loc[score > threshold] = "CE"
            sig.loc[score < -threshold] = "PE"
            signals[f"POSITIONING_SCORE_{int(threshold * 10)}"] = sig

    # Use the ATM straddle-implied daily move proxy as a premium-cost regime.
    # A low implied/realised ratio gates long-option entries; all gates are
    # lagged because these values come from the previous completed session.
    if "implied_to_realized_vol_ratio" in f.columns:
        ratio = f["implied_to_realized_vol_ratio"]
        for base in (
            "RET3_MOMENTUM", "RET5_MOMENTUM", "EMA5_20_TREND",
            "EMA5_20_ADX20", "RET3_5_CONSENSUS", "DONCHIAN10_BREAKOUT",
            "RSI35_65_REVERSAL",
        ):
            base_signal = signals[base]
            for threshold in (0.90, 1.10, 1.30):
                signals[f"{base}_CHEAP_STRADDLE{int(threshold * 100)}"] = base_signal.where(
                    ratio <= threshold, ""
                )
                signals[f"{base}_EXPENSIVE_STRADDLE{int(threshold * 100)}"] = base_signal.where(
                    ratio >= threshold, ""
                )

    # Remove exact duplicates: different threshold labels can collapse to
    # identical CE/PE/flat sequences on a finite historical sample. They are
    # the same executable rule and must not be counted as separate discoveries.
    unique_signals = {}
    seen_signatures = set()
    for name, series in signals.items():
        normalized = series.fillna("").astype(str)
        signature = tuple(normalized.tolist())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_signals[name] = normalized
    return unique_signals


def _one_position_at_a_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sequential fixed-one-lot book; do not open a new trade before prior exit."""
    if frame.empty:
        return frame.copy(), 0
    ordered = frame.sort_values(["entry_date", "exit_date"], kind="stable")
    keep = []
    last_exit = None
    for idx, row in ordered.iterrows():
        entry = pd.Timestamp(row["entry_date"])
        exit_ = pd.Timestamp(row["exit_date"])
        if last_exit is None or entry >= last_exit:
            keep.append(idx)
            last_exit = exit_
    selected = ordered.loc[keep].copy()
    return selected, int(len(frame) - len(selected))


def _monthly_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized monthly gross profit/loss summary for all candidate rules."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame[["strategy", "month", "net_pnl"]].copy()
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["strategy", "month", "net_pnl"])
    if x.empty:
        return pd.DataFrame()
    x["gross_profit_component"] = x["net_pnl"].clip(lower=0)
    x["gross_loss_component"] = (-x["net_pnl"]).clip(lower=0)
    grouped = x.groupby(["strategy", "month"], sort=True).agg(
        trades=("net_pnl", "count"),
        gross_profit=("gross_profit_component", "sum"),
        gross_loss=("gross_loss_component", "sum"),
        net_pnl=("net_pnl", "sum"),
    ).reset_index()
    grouped["profit_factor"] = np.where(
        grouped["gross_loss"] > 0,
        grouped["gross_profit"] / grouped["gross_loss"],
        np.where(grouped["gross_profit"] > 0, np.inf, np.nan),
    )
    grouped["target_pf"] = TARGET_PF
    grouped["target_testable"] = grouped["trades"] >= MIN_MONTH_TRADES
    grouped["target_met"] = [
        (bool(pf >= TARGET_PF) if testable and _finite(pf) else
         bool(np.isinf(pf)) if testable and isinstance(pf, (float, np.floating)) and np.isinf(pf) else
         None)
        for pf, testable in zip(grouped["profit_factor"], grouped["target_testable"])
    ]
    grouped["gross_profit"] = grouped["gross_profit"].round(2)
    grouped["gross_loss"] = grouped["gross_loss"].round(2)
    grouped["net_pnl"] = grouped["net_pnl"].round(2)
    return grouped


def _empty_score(split: str) -> dict:
    return {
        f"{split}_trades": 0,
        f"{split}_net_pnl": 0.0,
        f"{split}_gross_profit": 0.0,
        f"{split}_gross_loss": 0.0,
        f"{split}_aggregate_pf": None,
        f"{split}_win_rate_pct": None,
        f"{split}_expectancy": None,
        f"{split}_median_trade_pnl": None,
        f"{split}_max_drawdown": None,
        f"{split}_active_months": 0,
        f"{split}_target_months": 0,
        f"{split}_monthly_target_hit_pct": None,
        f"{split}_positive_months": 0,
        f"{split}_months": 0,
    }


def _score_many(frame: pd.DataFrame, split: str) -> dict[str, dict]:
    """Compute all per-candidate metrics with a small number of vectorized groupbys."""
    empty = _empty_score(split)
    if frame is None or frame.empty:
        return {}

    x = frame[["strategy", "month", "entry_date", "exit_date", "net_pnl"]].copy()
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["strategy", "month", "entry_date", "exit_date", "net_pnl"])
    if x.empty:
        return {}
    x["gross_profit_component"] = x["net_pnl"].clip(lower=0)
    x["gross_loss_component"] = (-x["net_pnl"]).clip(lower=0)
    x["win_component"] = (x["net_pnl"] > 0).astype(float)

    aggregate = x.groupby("strategy", sort=False).agg(
        trades=("net_pnl", "count"),
        net_pnl=("net_pnl", "sum"),
        gross_profit=("gross_profit_component", "sum"),
        gross_loss=("gross_loss_component", "sum"),
        win_rate=("win_component", "mean"),
        expectancy=("net_pnl", "mean"),
        median_trade_pnl=("net_pnl", "median"),
    )

    ordered = x.sort_values(["strategy", "exit_date", "entry_date"], kind="stable").copy()
    ordered["equity"] = 10000.0 + ordered.groupby("strategy", sort=False)["net_pnl"].cumsum()
    ordered["peak"] = ordered["equity"].clip(lower=10000.0).groupby(
        ordered["strategy"], sort=False
    ).cummax()
    ordered["drawdown"] = ordered["equity"] - ordered["peak"]
    drawdown = ordered.groupby("strategy", sort=False)["drawdown"].min()
    aggregate["max_drawdown"] = drawdown

    monthly = _monthly_stats(x)
    if not monthly.empty:
        monthly["active_flag"] = monthly["target_testable"].astype(int)
        monthly["target_pass_flag"] = (
            monthly["target_met"].fillna(False).astype(int)
        )
        monthly["positive_month_flag"] = (monthly["net_pnl"] > 0).astype(int)
        monthly_summary = monthly.groupby("strategy", sort=False).agg(
            active_months=("active_flag", "sum"),
            target_months=("target_pass_flag", "sum"),
            positive_months=("positive_month_flag", "sum"),
            months=("month", "count"),
        )
        aggregate = aggregate.join(monthly_summary, how="left")

    aggregate["active_months"] = aggregate.get("active_months", 0)
    aggregate["target_months"] = aggregate.get("target_months", 0)
    aggregate["positive_months"] = aggregate.get("positive_months", 0)
    aggregate["months"] = aggregate.get("months", 0)
    result = {}
    for strategy, row in aggregate.iterrows():
        gp, gl = float(row["gross_profit"]), float(row["gross_loss"])
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else None)
        active_months = int(row["active_months"])
        passed = int(row["target_months"])
        result[str(strategy)] = {
            f"{split}_trades": int(row["trades"]),
            f"{split}_net_pnl": round(float(row["net_pnl"]), 2),
            f"{split}_gross_profit": round(gp, 2),
            f"{split}_gross_loss": round(gl, 2),
            f"{split}_aggregate_pf": pf,
            f"{split}_win_rate_pct": round(float(row["win_rate"]) * 100, 2),
            f"{split}_expectancy": round(float(row["expectancy"]), 2),
            f"{split}_median_trade_pnl": round(float(row["median_trade_pnl"]), 2),
            f"{split}_max_drawdown": round(float(row["max_drawdown"]), 2),
            f"{split}_active_months": active_months,
            f"{split}_target_months": passed,
            f"{split}_monthly_target_hit_pct": (
                round(passed / active_months * 100, 2) if active_months else None
            ),
            f"{split}_positive_months": int(row["positive_months"]),
            f"{split}_months": int(row["months"]),
        }
    return result


def _score(frame: pd.DataFrame, split: str) -> dict:
    """Single-rule reference implementation retained for unit-test comparability."""
    if frame is None or frame.empty:
        return _empty_score(split)
    result = _score_many(frame, split)
    return result.get(str(frame["strategy"].iloc[0]), _empty_score(split))


def _build_monthly_selector(
    scored: pd.DataFrame,
    candidates: list[str],
    lookback_months: int,
    minimum_prior_trades: int = 20,
) -> pd.DataFrame:
    """At each month start, select only using the previous N complete calendar months."""
    if scored.empty:
        return pd.DataFrame()
    calendar = sorted(scored["month"].unique())
    rows = []
    for position, month in enumerate(calendar):
        prior_months = set(calendar[max(0, position - lookback_months):position])
        if len(prior_months) < lookback_months:
            rows.append({
                "strategy": f"ROLLING_{lookback_months}M_SELECTOR",
                "month": month,
                "selected_rule": "CASH",
                "selection_prior_months": lookback_months,
                "prior_trades": 0,
                "prior_pf": None,
                "trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "profit_factor": None,
                "target_pf": TARGET_PF,
                "target_testable": False,
                "target_met": None,
            })
            continue
        history = scored.loc[
            scored["month"].isin(prior_months) & scored["strategy"].isin(candidates),
            ["strategy", "net_pnl"],
        ].copy()
        history["net_pnl"] = pd.to_numeric(history["net_pnl"], errors="coerce")
        history = history.dropna(subset=["net_pnl"])
        history["gross_profit_component"] = history["net_pnl"].clip(lower=0)
        history["gross_loss_component"] = (-history["net_pnl"]).clip(lower=0)
        summary = history.groupby("strategy", sort=False).agg(
            prior_trades=("net_pnl", "count"),
            prior_net=("net_pnl", "sum"),
            prior_gross_profit=("gross_profit_component", "sum"),
            prior_gross_loss=("gross_loss_component", "sum"),
        )
        summary = summary[summary["prior_trades"] >= minimum_prior_trades]
        summary["prior_pf"] = np.where(
            summary["prior_gross_loss"] > 0,
            summary["prior_gross_profit"] / summary["prior_gross_loss"],
            np.where(summary["prior_gross_profit"] > 0, np.inf, np.nan),
        )
        eligible_frame = summary[summary["prior_pf"] >= TARGET_PF]
        eligible = [
            (float(row.prior_pf), float(row.prior_net), str(strategy), int(row.prior_trades))
            for strategy, row in eligible_frame.iterrows()
        ]
        if not eligible:
            rows.append({
                "strategy": f"ROLLING_{lookback_months}M_SELECTOR",
                "month": month,
                "selected_rule": "CASH",
                "selection_prior_months": lookback_months,
                "prior_trades": 0,
                "prior_pf": None,
                "trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "profit_factor": None,
                "target_pf": TARGET_PF,
                "target_testable": False,
                "target_met": None,
            })
            continue
        eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
        prior_pf, _, selected, prior_n = eligible[0]
        current = scored[(scored["month"] == month) & (scored["strategy"] == selected)]
        pnl = pd.to_numeric(current["net_pnl"], errors="coerce").dropna()
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        pf = _profit_factor(pnl)
        n = len(pnl)
        rows.append({
            "strategy": f"ROLLING_{lookback_months}M_SELECTOR",
            "month": month,
            "selected_rule": selected,
            "selection_prior_months": lookback_months,
            "prior_trades": prior_n,
            "prior_pf": prior_pf,
            "trades": n,
            "gross_profit": round(gp, 2),
            "gross_loss": round(gl, 2),
            "net_pnl": round(float(pnl.sum()), 2),
            "profit_factor": pf,
            "target_pf": TARGET_PF,
            "target_testable": n >= MIN_MONTH_TRADES,
            "target_met": bool(pf is not None and pf >= TARGET_PF) if n >= MIN_MONTH_TRADES else None,
        })
    return pd.DataFrame(rows)


def run_entry_signal_research(
    daily_ledger: pd.DataFrame,
    index: pd.DataFrame,
    options: pd.DataFrame | None = None,
    holding_periods: tuple[int, ...] = (1, 2, 3, 5),
    capital: float = 10000.0,
    risk_per_trade_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Test predeclared signals across one-, two-, three- and five-session holds."""
    if options is not None:
        long_outcomes = _option_horizon_outcomes(
            index, options, holding_periods=holding_periods, capital=capital,
            risk_pct=risk_per_trade_pct,
        )
        debit_outcomes = _option_debit_vertical_outcomes(
            index, options, holding_periods=holding_periods, capital=capital,
            risk_pct=risk_per_trade_pct,
        )
        credit_outcomes = _option_credit_vertical_outcomes(
            index, options, holding_periods=holding_periods, capital=capital,
            risk_pct=risk_per_trade_pct,
        )
        frames = [
            frame for frame in (long_outcomes, debit_outcomes, credit_outcomes)
            if frame is not None and not frame.empty
        ]
        pairs = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    else:
        pairs = _paired_option_outcomes(daily_ledger)
        if not pairs.empty:
            pairs["holding_sessions"] = 1
            pairs["option_family"] = "LONG_OPTIONS"
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_paired_option_data",
            "note": "Could not pair ATM Call and Put outcomes on the same entry, exit, expiry and strike.",
        }

    features = build_lagged_features(index)
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "missing_index_features",
            "note": "SENSEX OHLC history is required.",
        }
    date_to_position = {date: pos for pos, date in enumerate(features["date"])}
    pairs["signal_date"] = pairs["entry_date"].map(
        lambda entry: features.iloc[date_to_position[entry] - 1]["date"]
        if entry in date_to_position and date_to_position[entry] > 0 else pd.NaT
    )
    pairs = pairs.dropna(subset=["signal_date"]).copy()
    pairs["signal_date"] = pd.to_datetime(pairs["signal_date"]).dt.normalize()
    features = features.set_index("date")
    feature_cols = [col for col in features.columns if col != "date"]
    paired_features = pairs[["signal_date"]].join(features[feature_cols], on="signal_date")
    paired = pd.concat(
        [pairs.reset_index(drop=True), paired_features.drop(columns=["signal_date"]).reset_index(drop=True)],
        axis=1,
    )
    if options is not None and not options.empty:
        option_features = _option_chain_market_features(index, options)
        if not option_features.empty:
            paired["expiry_date"] = pd.to_datetime(paired["expiry_date"], errors="coerce").dt.normalize()
            option_features["expiry_date"] = pd.to_datetime(
                option_features["expiry_date"], errors="coerce"
            ).dt.normalize()
            paired = paired.merge(
                option_features,
                on=["signal_date", "expiry_date"],
                how="left",
                validate="many_to_one",
            )
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_feature_overlap",
            "note": "No paired option trades overlap the SENSEX feature history.",
        }

    signals = _candidate_signals(paired)
    records = []
    # Group expensive outcome frames once; reuse them for each predeclared signal.
    horizon_groups = list(paired.groupby(["option_family", "holding_sessions"], sort=True))
    for name, signal in signals.items():
        for (family, holding), horizon in horizon_groups:
            chosen = horizon.copy()
            chosen["signal"] = signal.loc[chosen.index].to_numpy()
            chosen = chosen[chosen["signal"].isin(["CE", "PE"])].copy()
            if chosen.empty:
                continue
            if str(family).startswith("LONG_OPTIONS"):
                prefix = str(family).replace("LONG_OPTIONS", "LONG")
            elif str(family).startswith("DEBIT_SPREADS"):
                prefix = str(family).replace("DEBIT_SPREADS", "DEBIT")
            else:
                prefix = str(family).replace("CREDIT_SPREADS", "CREDIT")
            chosen["strategy"] = f"{prefix}_H{int(holding)}D_{name}"
            chosen["net_pnl"] = np.where(
                chosen["signal"] == "CE",
                chosen["call_pnl"],
                chosen["put_pnl"],
            )
            chosen = chosen.dropna(subset=["net_pnl"])
            raw_trade_count = len(chosen)
            chosen, overlap_skipped = _one_position_at_a_time(chosen)
            if chosen.empty:
                continue
            chosen["overlap_skipped_trades_for_rule"] = overlap_skipped
            if chosen.empty:
                continue
            chosen["month"] = chosen["exit_date"].dt.strftime("%Y-%m")
            chosen["option_type"] = chosen["signal"]
            chosen["holding_sessions"] = int(holding)
            if {"call_strike", "put_strike"}.issubset(chosen.columns):
                chosen["strike"] = np.where(
                    chosen["signal"] == "CE",
                    chosen["call_strike"],
                    chosen["put_strike"],
                )
            elif "strike" not in chosen:
                chosen["strike"] = chosen.get("atm_reference_strike")
            records.append(chosen[[
                "strategy", "option_family", "holding_sessions", "entry_date", "exit_date",
                "signal_date", "month", "signal", "option_type", "strike",
                "net_pnl", "call_pnl", "put_pnl",
                "overlap_skipped_trades_for_rule",
            ]])
    trades = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "no_signal_trades",
            "note": "No candidate generated a trade from the observed paired option outcomes.",
        }

    exit_months = sorted(paired["month"].dropna().astype(str).unique())
    if len(exit_months) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_months",
            "note": "At least two distinct realized exit months are required for a month-aligned holdout.",
        }
    split_month_at = max(1, min(len(exit_months) - 1, int(len(exit_months) * 0.70)))
    train_month_end = exit_months[split_month_at - 1]
    train_exit_dates = paired.loc[
        paired["month"] <= train_month_end, "exit_date"
    ]
    train_end = pd.Timestamp(train_exit_dates.max())
    holdout_exit_dates = paired.loc[
        paired["month"] > train_month_end, "exit_date"
    ]
    holdout_start = pd.Timestamp(holdout_exit_dates.min()) if not holdout_exit_dates.empty else None
    trades["split"] = np.where(trades["month"] <= train_month_end, "train", "holdout")

    monthly = _monthly_stats(trades)
    leaderboard_rows = []
    candidate_names = sorted(trades["strategy"].unique())
    split_scores = {
        "train": _score_many(trades[trades["split"] == "train"], "train"),
        "holdout": _score_many(trades[trades["split"] == "holdout"], "holdout"),
        "full": _score_many(trades, "full"),
    }
    empty_train = _empty_score("train")
    empty_holdout = _empty_score("holdout")
    empty_full = _empty_score("full")
    for name in candidate_names:
        train_metrics = split_scores["train"].get(str(name), empty_train)
        holdout_metrics = split_scores["holdout"].get(str(name), empty_holdout)
        full_metrics = split_scores["full"].get(str(name), empty_full)
        train_ok = (
            train_metrics["train_trades"] >= MIN_TRAIN_TRADES
            and train_metrics["train_active_months"] >= 6
            and train_metrics["train_net_pnl"] > 0
            and train_metrics["train_aggregate_pf"] is not None
            and train_metrics["train_aggregate_pf"] >= TARGET_PF
            and train_metrics["train_active_months"] > 0
            and train_metrics["train_target_months"] == train_metrics["train_active_months"]
        )
        holdout_ok = (
            holdout_metrics["holdout_trades"] >= MIN_HOLDOUT_TRADES
            and holdout_metrics["holdout_active_months"] >= 4
            and holdout_metrics["holdout_net_pnl"] > 0
            and holdout_metrics["holdout_aggregate_pf"] is not None
            and holdout_metrics["holdout_aggregate_pf"] >= TARGET_PF
            and holdout_metrics["holdout_active_months"] > 0
            and holdout_metrics["holdout_target_months"] == holdout_metrics["holdout_active_months"]
        )
        leaderboard_rows.append({
            "strategy": name,
            "full_trades": full_metrics["full_trades"],
            "full_net_pnl": full_metrics["full_net_pnl"],
            "full_aggregate_pf": full_metrics["full_aggregate_pf"],
            "full_monthly_target_hit_pct": full_metrics["full_monthly_target_hit_pct"],
            **train_metrics,
            **holdout_metrics,
            "training_screen_pass": bool(train_ok),
            "holdout_validation_pass": bool(holdout_ok),
            "candidate_pass": bool(train_ok and holdout_ok),
        })

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = leaderboard.sort_values(
        ["candidate_pass", "train_monthly_target_hit_pct", "train_aggregate_pf", "train_net_pnl"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.insert(0, "training_rank", range(1, len(leaderboard) + 1))

    selector_rows = []
    for lookback in (3, 6):
        selector = _build_monthly_selector(trades[trades["split"].isin(["train", "holdout"])], candidate_names, lookback)
        if not selector.empty:
            selector_rows.append(selector)
    selector_monthly = pd.concat(selector_rows, ignore_index=True) if selector_rows else pd.DataFrame()
    selector_summary = []
    if not selector_monthly.empty:
        for selector_name, group in selector_monthly.groupby("strategy"):
            monthly_parts = group.copy()
            monthly_parts["split"] = np.where(monthly_parts["month"] <= train_month_end, "train", "holdout")
            summary = {"strategy": selector_name}
            for split_name in ("train", "holdout"):
                chunk = monthly_parts[monthly_parts["split"] == split_name]
                active = chunk[chunk["target_testable"]]
                gp = float(pd.to_numeric(chunk["gross_profit"], errors="coerce").fillna(0).sum())
                gl = float(pd.to_numeric(chunk["gross_loss"], errors="coerce").fillna(0).sum())
                net = float(pd.to_numeric(chunk["net_pnl"], errors="coerce").fillna(0).sum())
                trades_n = int(pd.to_numeric(chunk["trades"], errors="coerce").fillna(0).sum())
                hit_pct = (
                    float(active["target_met"].fillna(False).mean() * 100)
                    if not active.empty else None
                )
                summary.update({
                    f"{split_name}_trades": trades_n,
                    f"{split_name}_net_pnl": round(net, 2),
                    f"{split_name}_aggregate_pf": gp / gl if gl > 0 else (float("inf") if gp > 0 else None),
                    f"{split_name}_active_months": int(len(active)),
                    f"{split_name}_monthly_target_hit_pct": round(hit_pct, 2) if hit_pct is not None else None,
                    f"{split_name}_positive_months": int((pd.to_numeric(chunk["net_pnl"], errors="coerce") > 0).sum()),
                })
            selector_summary.append(summary)

    report = {
        "status": "candidates_found" if bool(leaderboard["candidate_pass"].any()) else "no_candidate_met_research_screen",
        "target_definition": "Every active month (at least 3 trades) must have gross profit / gross loss >= 1.20 in both training and holdout; aggregate PF must also be >= 1.20 in both windows.",
        "monthly_target_profit_factor": TARGET_PF,
        "minimum_trades_per_month_for_target_test": MIN_MONTH_TRADES,
        "training_screen": {
            "minimum_trades": MIN_TRAIN_TRADES,
            "minimum_active_months": 6,
            "minimum_target_month_hit_pct": 100,
            "minimum_aggregate_profit_factor": TARGET_PF,
            "net_pnl_must_be_positive": True,
        },
        "holdout_screen": {
            "minimum_trades": MIN_HOLDOUT_TRADES,
            "minimum_active_months": 4,
            "minimum_target_month_hit_pct": 100,
            "minimum_aggregate_profit_factor": TARGET_PF,
            "net_pnl_must_be_positive": True,
        },
        "method": (
            "Chronological training/holdout split is aligned to realized exit months so a multi-session "
            "position cannot spill across the validation boundary. Predeclared lagged index indicator and "
            "regime signals choose between same-date, same-expiry, "
            "Call/Put outcomes at ATM, one-strike ITM and one-strike OTM moneyness, plus "
            "bounded-risk ATM debit and credit vertical spread outcomes "
            "at one-, two- and three-available-strike widths for one-, two-, three- and five-session holds. "
            "Long-option candidates include ATM, one-step ITM, and one-, two-, three- and five-step OTM contracts. "
            "Both spread types require liquid observed legs at entry and exit, reject impossible marks beyond "
            "their strike width. Default research evaluates one historical lot without an account-level affordability or risk-budget gate; an optional risk_per_trade_pct can enable that separate filter. Option-chain PCR, OI build-up and ATM straddle/realised-volatility "
            "proxies are aligned to the prior completed signal session. "
            "Option outcomes are rebuilt from observed BSE daily candles, with the exact same contract "
            "held through the selected exit session. A missing side is not fabricated; that direction is "
            "skipped when its observed quote is unavailable. Signals use only the last completed index "
            "session before entry. Candidate rules are ranked using training only; holdout is separate. "
            "Rolling selectors use only prior complete calendar months. Gap-fade variants are gated by "
            "lagged volatility, ADX, RSI, MACD, EMA and completed-candle rejection conditions."
        ),
        "option_outcome_sessions": int(len(paired)),
        "sample_start": str(paired["entry_date"].min().date()),
        "sample_end": str(paired["entry_date"].max().date()),
        "distinct_entry_sessions": int(paired["entry_date"].nunique()),
        "holding_periods_tested": sorted(paired["holding_sessions"].dropna().astype(int).unique().tolist()),
        "instrument_families_tested": sorted(paired["option_family"].dropna().astype(str).unique().tolist()),
        "moneyness_variants_tested": sorted(
            str(value) for value in paired.get("moneyness", pd.Series(dtype=object)).dropna().unique()
        ),
        "spread_width_steps_tested": sorted({
            int(str(family).split("_W")[-1])
            for family in paired["option_family"].dropna().astype(str).unique()
            if str(family).startswith(("DEBIT_SPREADS_W", "CREDIT_SPREADS_W"))
        }),
        "option_market_features": [
            "near-ATM put/call volume and open-interest ratios",
            "call/put open-interest changes matched to premium changes",
            "ATM straddle premium versus trailing realised-volatility proxy",
        ] if options is not None else [],
        "initial_capital": float(capital),
        "capital_is_sizing_constraint": risk_per_trade_pct is not None,
        "risk_per_trade_pct": float(risk_per_trade_pct) * 100 if risk_per_trade_pct is not None else None,
        "maximum_risk_per_trade": round(float(capital) * float(risk_per_trade_pct), 2) if risk_per_trade_pct is not None else None,
        "credit_spread_risk_budget": (
            f"Defined-risk credit spreads require maximum theoretical loss plus entry fees <= "
            f"₹{float(capital) * float(risk_per_trade_pct):.2f} ({float(risk_per_trade_pct) * 100:.2f}% of capital); "
            "this is a conservative theoretical loss estimate, not a broker margin quote."
        ) if risk_per_trade_pct is not None else (
            "No ₹10,000 affordability or 2% risk cap is applied in signal research; each observed strategy outcome uses one historical lot."
        ),
        "candidate_rules_tested": int(len(candidate_names)),
        "position_sizing": (
            f"One lot per candidate, one open position at a time; maximum theoretical loss including "
            f"entry fees is capped at {float(risk_per_trade_pct) * 100:.2f}% of ₹{float(capital):.0f} "
            "for long options and defined-risk spreads."
        ) if risk_per_trade_pct is not None else (
            "One historical lot per candidate outcome; no account affordability filter or capital-based position sizing."
        ),
        "overlapping_candidate_entries_skipped": int(
            pd.to_numeric(
                trades[["strategy", "overlap_skipped_trades_for_rule"]]
                .drop_duplicates("strategy")["overlap_skipped_trades_for_rule"],
                errors="coerce",
            ).fillna(0).sum()
        ) if not trades.empty else 0,
        "training_start": str(paired["entry_date"].min().date()),
        "training_end": str(train_end.date()),
        "holdout_start": str(holdout_start.date()) if holdout_start is not None else None,
        "holdout_end": str(paired["exit_date"].max().date()),
        "candidate_pass_count": int(leaderboard["candidate_pass"].sum()),
        "top_training_rule": leaderboard.iloc[0].to_dict() if not leaderboard.empty else None,
        "rolling_selector_rules": ["ROLLING_3M_SELECTOR", "ROLLING_6M_SELECTOR"],
        "rolling_selector_summary": selector_summary,
        "training_month_end": train_month_end,
        "selector_note": (
            "Rolling selector is a diagnostic paper-research rule, not a shared-capital portfolio: "
            "strategy switches across month boundaries may overlap positions. It goes to cash when no signal "
            "has at least 20 prior trades and prior-window PF >= 1.20."
        ),
        "caveat": (
            "This is a historical screening experiment, not a live profit guarantee. Option outcomes use "
            "one-lot EOD marks and estimated costs; intraday execution, queue position and slippage may differ. "
            "Repeated research iterations create selection bias, so any promising rule needs a new prospective "
            "paper-trading period before live deployment."
        ),
    }
    return trades, leaderboard, selector_monthly, report
