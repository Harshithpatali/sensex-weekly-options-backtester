from __future__ import annotations

"""Chronological ML research for observed one-session SENSEX option outcomes.

The target is derived only from matched BSE option candles. Features are lagged
to the last completed session before entry. The module does not create option
prices or model executable bid/ask fills.
"""

import json
from math import erf, exp, log, pi, sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sensex_options.config import DEFAULT_PATHS
from sensex_options.entry_signal_research import (
    _option_chain_market_features,
    _option_horizon_outcomes,
    build_lagged_features,
)



RISK_FREE_RATE = 0.065


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _black_scholes_price(
    spot: float, strike: float, years: float, rate: float, volatility: float, option_side: str
) -> float:
    if min(spot, strike, years, volatility) <= 0:
        return float("nan")
    root_t = sqrt(years)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility * volatility) * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discounted_strike = strike * exp(-rate * years)
    if option_side == "CE":
        return spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return discounted_strike * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def _implied_volatility(
    market_price: float, spot: float, strike: float, years: float,
    rate: float, option_side: str,
) -> float | None:
    """Bisection IV estimate; None when the observed premium violates model bounds."""
    if not all(np.isfinite(v) for v in (market_price, spot, strike, years)):
        return None
    if market_price <= 0 or spot <= 0 or strike <= 0 or years <= 0:
        return None
    discounted_strike = strike * exp(-rate * years)
    intrinsic = max(spot - discounted_strike, 0.0) if option_side == "CE" else max(discounted_strike - spot, 0.0)
    upper = spot if option_side == "CE" else discounted_strike
    if market_price < intrinsic - 1e-6 or market_price >= upper:
        return None
    low, high = 1e-4, 5.0
    for _ in range(60):
        mid = (low + high) / 2.0
        model_price = _black_scholes_price(spot, strike, years, rate, mid, option_side)
        if not np.isfinite(model_price):
            return None
        if model_price > market_price:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def _black_scholes_greeks(
    spot: float, strike: float, years: float, rate: float,
    volatility: float, option_side: str,
) -> dict[str, float]:
    """Approximate European Greeks; theta is per calendar day and vega per 1 vol point."""
    root_t = sqrt(years)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility * volatility) * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    density = _normal_pdf(d1)
    gamma = density / (spot * volatility * root_t)
    vega = spot * density * root_t / 100.0
    if option_side == "CE":
        delta = _normal_cdf(d1)
        theta = (
            -spot * density * volatility / (2.0 * root_t)
            - rate * strike * exp(-rate * years) * _normal_cdf(d2)
        ) / 365.0
    else:
        delta = _normal_cdf(d1) - 1.0
        theta = (
            -spot * density * volatility / (2.0 * root_t)
            + rate * strike * exp(-rate * years) * _normal_cdf(-d2)
        ) / 365.0
    return {
        "lagged_iv": volatility,
        "lagged_delta": delta,
        "lagged_gamma": gamma,
        "lagged_theta_per_day": theta,
        "lagged_vega_per_1pct": vega,
    }


def _lagged_greek_features(dataset: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    """Estimate Greeks from prior-session closes only; missing marks remain missing."""
    opt = options.copy()
    required = {"trade_date", "expiry_date", "strike", "option_type", "close"}
    if opt.empty or not required.issubset(opt.columns):
        return pd.DataFrame(index=dataset.index)
    for col in ("trade_date", "expiry_date"):
        opt[col] = pd.to_datetime(opt[col], errors="coerce").dt.normalize()
    for col in ("strike", "close"):
        opt[col] = pd.to_numeric(opt[col], errors="coerce")
    opt["option_type"] = opt["option_type"].astype(str).str.upper().str.strip()
    opt = opt.dropna(subset=list(required))
    opt = opt[(opt["close"] > 0) & (opt["strike"] > 0)]
    opt = opt.sort_values("symbol" if "symbol" in opt.columns else "trade_date")
    opt = opt.drop_duplicates(
        ["trade_date", "expiry_date", "strike", "option_type"], keep="last"
    )
    price_lookup = {
        (row.trade_date, row.expiry_date, float(row.strike), row.option_type): float(row.close)
        for row in opt.itertuples()
    }
    rows = []
    for row in dataset.itertuples():
        signal_date = pd.Timestamp(row.signal_date).normalize()
        expiry_date = pd.Timestamp(row.expiry_date).normalize()
        strike = float(row.selected_strike)
        side = str(row.option_side)
        premium = price_lookup.get((signal_date, expiry_date, strike, side))
        spot = float(row.signal_spot)
        years = max((expiry_date - signal_date).days, 1) / 365.25
        iv = _implied_volatility(
            premium, spot, strike, years, RISK_FREE_RATE, side
        ) if premium is not None else None
        if iv is None:
            rows.append({})
        else:
            rows.append(_black_scholes_greeks(
                spot, strike, years, RISK_FREE_RATE, iv, side
            ))
    return pd.DataFrame(rows, index=dataset.index)


def build_ml_option_dataset(
    index: pd.DataFrame,
    options: pd.DataFrame,
    holding_period: int = 1,
    moneyness_variants: tuple[str, ...] = ("ATM", "ITM1", "OTM1"),
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Create paired CE/PE observations with strictly lagged predictors."""
    outcomes = _option_horizon_outcomes(
        index,
        options,
        holding_periods=(holding_period,),
        moneyness_variants=moneyness_variants,
        risk_pct=None,
    )
    if outcomes.empty:
        return pd.DataFrame(), [], {"status": "insufficient_option_outcomes"}

    lagged = build_lagged_features(index)
    if lagged.empty:
        return pd.DataFrame(), [], {"status": "missing_index_features"}
    lagged["date"] = pd.to_datetime(lagged["date"], errors="coerce").dt.normalize()
    outcomes["signal_date"] = pd.to_datetime(outcomes["signal_date"], errors="coerce").dt.normalize()
    outcomes["entry_date"] = pd.to_datetime(outcomes["entry_date"], errors="coerce").dt.normalize()
    outcomes["exit_date"] = pd.to_datetime(outcomes["exit_date"], errors="coerce").dt.normalize()
    outcomes["expiry_date"] = pd.to_datetime(outcomes["expiry_date"], errors="coerce").dt.normalize()

    index_feature_cols = [
        c for c in lagged.columns
        if c not in {"date", "open", "high", "low", "close", "volume"}
        and pd.api.types.is_numeric_dtype(lagged[c])
    ]
    lagged_lookup = lagged[["date", "close"] + index_feature_cols].rename(
        columns={"date": "signal_date", "close": "signal_spot"}
    )
    paired = outcomes.merge(
        lagged_lookup, on="signal_date", how="inner", validate="many_to_one"
    )

    option_feature_cols: list[str] = []
    chain_features = _option_chain_market_features(index, options)
    if not chain_features.empty:
        chain_features["signal_date"] = pd.to_datetime(
            chain_features["signal_date"], errors="coerce"
        ).dt.normalize()
        chain_features["expiry_date"] = pd.to_datetime(
            chain_features["expiry_date"], errors="coerce"
        ).dt.normalize()
        option_feature_cols = [
            c for c in chain_features.columns
            if c not in {"signal_date", "expiry_date"}
            and pd.api.types.is_numeric_dtype(chain_features[c])
        ]
        paired = paired.merge(
            chain_features[["signal_date", "expiry_date"] + option_feature_cols],
            on=["signal_date", "expiry_date"],
            how="left",
            validate="many_to_one",
        )

    rows: list[pd.DataFrame] = []
    for side in ("call", "put"):
        pnl_col = f"{side}_pnl"
        strike_col = f"{side}_strike"
        premium_col = f"{side}_entry_premium"
        return_col = f"{side}_return_pct"
        if pnl_col not in paired or strike_col not in paired:
            continue
        frame = paired.dropna(subset=[pnl_col, strike_col]).copy()
        if frame.empty:
            continue
        frame["option_type_call"] = 1.0 if side == "call" else 0.0
        frame["moneyness"] = frame["moneyness"].astype(str)
        frame["dte_calendar_days"] = (
            frame["expiry_date"] - frame["entry_date"]
        ).dt.days.astype(float)
        reference_spot = pd.to_numeric(frame["signal_spot"], errors="coerce")
        frame["moneyness_distance_pct"] = (
            pd.to_numeric(frame[strike_col], errors="coerce") / reference_spot - 1.0
        )
        frame["target_net_pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce")
        if return_col in frame:
            frame["target_return_pct"] = pd.to_numeric(frame[return_col], errors="coerce")
        elif premium_col in frame:
            premium = pd.to_numeric(frame[premium_col], errors="coerce")
            frame["target_return_pct"] = (
                frame["target_net_pnl"] / (premium * pd.to_numeric(frame["lot_size"], errors="coerce")) * 100
            )
        else:
            frame["target_return_pct"] = np.nan
        frame["target_win"] = (frame["target_net_pnl"] > 0).astype(int)
        frame["selected_strike"] = pd.to_numeric(frame[strike_col], errors="coerce")
        frame["option_side"] = "CE" if side == "call" else "PE"
        rows.append(frame)

    if not rows:
        return pd.DataFrame(), [], {"status": "no_paired_sides"}
    dataset = pd.concat(rows, ignore_index=True, sort=False)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=["entry_date", "exit_date", "target_net_pnl", "target_return_pct"])
    dataset = dataset.reset_index(drop=True)
    greek_features = _lagged_greek_features(dataset, options)
    dataset = pd.concat([dataset, greek_features.reset_index(drop=True)], axis=1)
    dummies = pd.get_dummies(dataset["moneyness"], prefix="moneyness", dtype=float)
    dataset = pd.concat([dataset, dummies.reset_index(drop=True)], axis=1)

    feature_cols = (
        index_feature_cols
        + option_feature_cols
        + ["option_type_call", "dte_calendar_days", "moneyness_distance_pct",
           "lagged_iv", "lagged_delta", "lagged_gamma", "lagged_theta_per_day", "lagged_vega_per_1pct"]
        + list(dummies.columns)
    )
    feature_cols = list(dict.fromkeys(c for c in feature_cols if c in dataset.columns))
    # Remove fields that can encode the outcome, execution price, or future session.
    forbidden = {
        "target_net_pnl", "target_return_pct", "target_win", "call_pnl", "put_pnl",
        "call_entry_premium", "put_entry_premium", "call_return_pct", "put_return_pct",
        "entry_premium", "exit_premium", "net_pnl", "gross_pnl", "return_pct",
        "entry_date", "exit_date", "expiry_date", "signal_date", "selected_strike",
        "call_strike", "put_strike", "strike", "symbol", "option_type", "option_side",
        "moneyness", "month", "option_family", "holding_sessions",
    }
    feature_cols = [c for c in feature_cols if c not in forbidden]
    dataset = dataset.sort_values(["entry_date", "option_side", "moneyness"]).reset_index(drop=True)
    report = {
        "status": "ok",
        "rows": int(len(dataset)),
        "entry_sessions": int(dataset["entry_date"].nunique()),
        "date_start": str(dataset["entry_date"].min().date()),
        "date_end": str(dataset["entry_date"].max().date()),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "holding_sessions": int(holding_period),
        "moneyness_variants": sorted(dataset["moneyness"].unique().tolist()),
        "side_counts": dataset["option_side"].value_counts().to_dict(),
        "target": "net option P&L after estimated costs > 0",
        "signal_timing": "all index, option-chain and model-derived Greek predictors are from the prior completed SENSEX session",
        "greeks": "Black-Scholes approximations from prior-session option close and index close; 6.5% risk-free rate, no dividend yield; missing/invalid IV remains null",
    }
    return dataset, feature_cols, report


def _classifier_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return {"n": 0}
    result: dict[str, Any] = {
        "n": int(len(y)),
        "positive_rate_pct": round(float(y.mean() * 100), 3),
        "accuracy_at_0_5": round(float(accuracy_score(y, p >= 0.5)), 4),
        "precision_at_0_5": round(float(precision_score(y, p >= 0.5, zero_division=0)), 4),
        "recall_at_0_5": round(float(recall_score(y, p >= 0.5, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 5),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        result["pr_auc"] = round(float(average_precision_score(y, p)), 4)
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def _selected_trade_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    scored = frame.copy()
    scored["predicted_win_probability"] = probability
    eligible = scored[scored["predicted_win_probability"] >= threshold]
    selected = (
        eligible.sort_values(
            ["entry_date", "predicted_win_probability", "target_return_pct"],
            ascending=[True, False, False],
        )
        .drop_duplicates("entry_date", keep="first")
        .sort_values("entry_date")
    )
    pnl = pd.to_numeric(selected["target_net_pnl"], errors="coerce").dropna()
    returns = pd.to_numeric(selected["target_return_pct"], errors="coerce").dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    cumulative = pnl.cumsum()
    peak = cumulative.cummax().clip(lower=0)
    drawdown = cumulative - peak
    all_sessions = pd.Index(sorted(pd.to_datetime(frame["entry_date"]).dropna().unique()))
    daily_returns = (
        selected.assign(entry_date=pd.to_datetime(selected["entry_date"]))
        .set_index("entry_date")["target_return_pct"].astype(float) / 100.0
    ).reindex(all_sessions, fill_value=0.0)
    return_index = 100.0 + daily_returns.cumsum() * 100.0
    return_peak = return_index.cummax().clip(lower=100.0)
    return_drawdown_pct = (return_index / return_peak.replace(0, np.nan) - 1.0) * 100.0
    return_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else np.nan
    sharpe = (
        float(daily_returns.mean() / return_std * np.sqrt(252))
        if np.isfinite(return_std) and return_std > 0 else None
    )
    return {
        "threshold": float(threshold),
        "trades": int(len(selected)),
        "sessions": int(selected["entry_date"].nunique()),
        "net_pnl": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (None if gross_profit == 0 else "inf"),
        "expectancy_pnl": round(float(pnl.mean()), 2) if len(pnl) else None,
        "expectancy_return_pct": round(float(returns.mean()), 4) if len(returns) else None,
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else None,
        "median_return_pct": round(float(returns.median()), 4) if len(returns) else None,
        "max_drawdown_pnl": round(float(drawdown.min()), 2) if len(drawdown) else 0.0,
        "max_drawdown_return_pct": round(float(return_drawdown_pct.min()), 3) if len(return_drawdown_pct.dropna()) else None,
        "sharpe_annualized": round(sharpe, 4) if sharpe is not None else None,
        "selected_predictions": selected,
    }


def run_ml_option_research(
    index: pd.DataFrame,
    options: pd.DataFrame,
    output_dir: Path | None = None,
    minimum_validation_trades: int = 20,
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train candidate classifiers, select on validation, then evaluate once on holdout."""
    output_dir = output_dir or DEFAULT_PATHS.outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, feature_cols, dataset_report = build_ml_option_dataset(index, options)
    if dataset.empty or len(dataset) < 100 or len(feature_cols) < 3:
        report = {
            "status": "insufficient_data",
            "dataset": dataset_report,
            "note": "At least 100 matched option observations and three usable predictors are required.",
        }
        return pd.DataFrame(), pd.DataFrame(), report

    dates = sorted(pd.to_datetime(dataset["entry_date"]).dt.normalize().unique())
    if len(dates) < 60:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_sessions",
            "dataset": dataset_report,
            "distinct_entry_sessions": int(len(dates)),
        }
    train_cut = max(1, int(len(dates) * 0.70))
    validation_cut = max(train_cut + 1, int(len(dates) * 0.85))
    if validation_cut >= len(dates):
        validation_cut = len(dates) - 1
    validation_start = pd.Timestamp(dates[train_cut])
    holdout_start = pd.Timestamp(dates[validation_cut])
    train = dataset[pd.to_datetime(dataset["exit_date"]) < validation_start].copy()
    validation = dataset[
        (pd.to_datetime(dataset["entry_date"]) >= validation_start)
        & (pd.to_datetime(dataset["exit_date"]) < holdout_start)
    ].copy()
    holdout = dataset[pd.to_datetime(dataset["entry_date"]) >= holdout_start].copy()
    if min(len(train), len(validation), len(holdout)) < 20 or train["target_win"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_split_samples",
            "dataset": dataset_report,
            "split_rows": {"train": len(train), "validation": len(validation), "holdout": len(holdout)},
            "split_dates": {
                "train_end": str(train["entry_date"].max().date()) if not train.empty else None,
                "validation_start": validation_start.date().isoformat(),
                "holdout_start": holdout_start.date().isoformat(),
            },
        }

    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train["target_win"].astype(int)
    x_val = validation[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_val = validation["target_win"].astype(int)
    x_holdout = holdout[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_holdout = holdout["target_win"].astype(int)

    candidates = {
        "LOGISTIC_REGRESSION": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ]),
        "RANDOM_FOREST": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=10,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )),
        ]),
        "HIST_GRADIENT_BOOSTING": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=42,
            )),
        ]),
    }

    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    validation_trade_rows: dict[str, dict[str, Any]] = {}
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        probability = estimator.predict_proba(x_val)[:, 1]
        fitted[name] = estimator
        validation_probabilities[name] = probability
        classification = _classifier_metrics(y_val, probability)
        threshold_results = []
        for threshold in thresholds:
            metrics = _selected_trade_metrics(validation, probability, threshold)
            if metrics["trades"] >= minimum_validation_trades:
                threshold_results.append(metrics)
        if threshold_results:
            threshold_results.sort(
                key=lambda item: (
                    float(item["expectancy_return_pct"] or -1e9),
                    float(item["profit_factor"]) if isinstance(item["profit_factor"], (int, float)) else 1e9,
                    int(item["trades"]),
                ),
                reverse=True,
            )
            chosen = threshold_results[0]
            validation_trade_rows[name] = chosen
        else:
            chosen = None
            validation_trade_rows[name] = {}
        comparison_rows.append({
            "model": name,
            **{f"validation_{key}": value for key, value in classification.items()},
            "validation_selected_trades": chosen["trades"] if chosen else 0,
            "validation_selected_expectancy_return_pct": chosen["expectancy_return_pct"] if chosen else None,
            "validation_selected_profit_factor": chosen["profit_factor"] if chosen else None,
            "validation_threshold": chosen["threshold"] if chosen else None,
            "validation_selection_status": "eligible" if chosen else "not_enough_validation_trades",
        })

    eligible_models = [
        row for row in comparison_rows
        if row["validation_selection_status"] == "eligible"
        and row.get("validation_selected_expectancy_return_pct") is not None
    ]
    if not eligible_models:
        return pd.DataFrame(comparison_rows), pd.DataFrame(), {
            "status": "no_model_selected",
            "dataset": dataset_report,
            "split_dates": {
                "train_start": str(train["entry_date"].min().date()),
                "train_end": str(train["entry_date"].max().date()),
                "validation_start": validation_start.date().isoformat(),
                "validation_end": str(validation["entry_date"].max().date()),
                "holdout_start": holdout_start.date().isoformat(),
                "holdout_end": str(holdout["entry_date"].max().date()),
            },
            "minimum_validation_trades": minimum_validation_trades,
            "note": "No classifier generated enough validation trades at the predeclared probability thresholds; no holdout strategy was forced.",
        }

    eligible_models.sort(
        key=lambda row: (
            float(row["validation_selected_expectancy_return_pct"]),
            float(row.get("validation_pr_auc") or 0.0),
        ),
        reverse=True,
    )
    selected_name = str(eligible_models[0]["model"])
    selected_threshold = float(eligible_models[0]["validation_threshold"])

    development = pd.concat([train, validation], ignore_index=True)
    final_model = candidates[selected_name]
    final_model.fit(development[feature_cols].apply(pd.to_numeric, errors="coerce"), development["target_win"].astype(int))
    holdout_probability = final_model.predict_proba(x_holdout)[:, 1]
    holdout_classification = _classifier_metrics(y_holdout, holdout_probability)
    holdout_trading = _selected_trade_metrics(holdout, holdout_probability, selected_threshold)
    selected_predictions = holdout_trading.pop("selected_predictions")
    prediction_output = holdout.copy()
    prediction_output["predicted_win_probability"] = holdout_probability
    selected_keys = set(
        zip(
            selected_predictions["entry_date"],
            selected_predictions["option_side"],
            selected_predictions["moneyness"],
        )
    )
    prediction_output["selected_for_strategy"] = [
        (row.entry_date, row.option_side, row.moneyness) in selected_keys
        for row in prediction_output.itertuples()
    ]

    importance_rows = []
    try:
        # Feature importance is measured on the validation window, never on holdout.
        importance_model = fitted[selected_name]
        permutation = permutation_importance(
            importance_model,
            x_val,
            y_val,
            scoring="average_precision",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        importance_rows = [
            {"feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
            for feature, mean, std in zip(feature_cols, permutation.importances_mean, permutation.importances_std)
        ]
    except Exception as exc:
        importance_rows = [{"feature": "importance_error", "importance_mean": 0.0, "importance_std": 0.0, "note": str(exc)}]
    importance = pd.DataFrame(importance_rows).sort_values("importance_mean", ascending=False)

    model_path = output_dir / "ml_option_classifier.joblib"
    joblib.dump({
        "model": final_model,
        "feature_columns": feature_cols,
        "selected_model": selected_name,
        "probability_threshold": selected_threshold,
        "target": "net option P&L after estimated costs > 0",
        "feature_timing": "prior completed SENSEX session",
    }, model_path)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_selection_status", "validation_selected_expectancy_return_pct", "validation_pr_auc"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    report = {
        "status": "holdout_evaluated",
        "dataset": dataset_report,
        "model_candidates": list(candidates),
        "selected_model": selected_name,
        "selected_probability_threshold": selected_threshold,
        "selection_rule": (
            "Choose the classifier with the highest validation one-trade-per-session expectancy return, "
            "subject to at least the configured number of validation trades. The threshold is selected "
            "from the predeclared grid. The final holdout is not used for model or threshold selection."
        ),
        "minimum_validation_trades": minimum_validation_trades,
        "split_dates": {
            "train_start": str(train["entry_date"].min().date()),
            "train_end": str(train["entry_date"].max().date()),
            "validation_start": validation_start.date().isoformat(),
            "validation_end": str(validation["entry_date"].max().date()),
            "holdout_start": holdout_start.date().isoformat(),
            "holdout_end": str(holdout["entry_date"].max().date()),
        },
        "split_rows": {"train": int(len(train)), "validation": int(len(validation)), "holdout": int(len(holdout))},
        "holdout_classification": holdout_classification,
        "holdout_strategy": holdout_trading,
        "holdout_selection": "At most one highest-probability candidate per entry session; one-session holding period; no overlapping positions.",
        "feature_importance_window": "validation",
        "model_artifact": str(model_path),
        "execution_caveat": (
            "BSE bhavcopy contains daily OHLC, volume and open interest, not timestamped bid/ask quotes. "
            "The target uses observed EOD close-to-close premiums with conservative estimated costs; "
            "it is not a quote-level executable return."
        ),
        "research_warning": (
            "Model selection across several candidates creates selection bias. Treat the holdout as one "
            "historical diagnostic, not a guarantee. A genuinely prospective paper period is needed to "
            "test stability after research choices are frozen."
        ),
    }
    prediction_output.to_csv(output_dir / "ml_option_holdout_predictions.csv.gz", index=False, compression="gzip")
    importance.to_csv(output_dir / "ml_option_feature_importance.csv", index=False)
    comparison.to_csv(output_dir / "ml_option_model_comparison.csv", index=False)
    with open(output_dir / "ml_option_research.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return comparison, importance, report
