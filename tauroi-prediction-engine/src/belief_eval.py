"""
belief_eval.py — Belief Model Evaluation
==========================================
Walk-forward evaluation of the hybrid belief model using high-frequency
Kalshi price data.  No settlement data required — every price observation
is a testable prediction.

Metrics:
    1. Information Coefficient (IC) — rank correlation of signal vs next return
    2. QLIKE Loss — evaluates volatility forecasts
    3. Long-Short Spread — return spread between top and bottom quintile
    4. Mean-Reversion Capture — how much of post-jump reversals are captured
    5. Sharpe Ratio — risk-adjusted annualised return
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.belief_model import (
    BeliefCalibration,
    calibrate_belief,
    forecast_variance,
    kalman_filter,
)
from src.utils import get_logger

logger = get_logger("tauroi.belief_eval")

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EVAL_DIR = _PROJECT_ROOT / "data" / "eval"


# ── Walk-Forward Engine ──────────────────────────────────────────────────────

def _walk_forward_signals(
    df: pd.DataFrame,
    train_window: int = 200,
    step: int = 1,
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward calibration producing one-step-ahead predictions.

    For each evaluation step t:
        1. Calibrate on data[t - train_window : t]
        2. Predict return over [t : t + forecast_horizon]
        3. Record signal, predicted variance, actual return

    Returns a DataFrame with columns:
        timestamp, signal, predicted_var, actual_return, actual_var
    """
    n = len(df)
    if n < train_window + forecast_horizon + 10:
        logger.warning(
            "Not enough data for walk-forward (%d rows, need %d+)",
            n, train_window + forecast_horizon + 10,
        )
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    logits = df["logit"].values
    timestamps = df["timestamp"].values
    prices = df["mid_price"].values

    # Estimate dt in years
    if n > 1:
        ts_int = pd.to_datetime(timestamps).astype(np.int64) // 10**9
        dt_sec = float(np.median(np.diff(ts_int)))
        dt_sec = max(dt_sec, 1.0)
    else:
        dt_sec = 86400.0
    dt_years = dt_sec / (365.25 * 86400)

    for t in range(train_window, n - forecast_horizon, step):
        train_slice = df.iloc[t - train_window : t]
        cal = calibrate_belief(train_slice, ticker="eval")

        # Signal: deviation of current log-odds from filtered mean
        # Positive => price above trend => expect mean-reversion down
        x_hat = kalman_filter(train_slice["logit"].values)
        current_x = logits[t]
        filtered_x = x_hat[-1]
        signal = -(current_x - filtered_x)  # mean-reversion direction

        # Predicted variance
        pred_var = forecast_variance(cal, forecast_horizon, dt_years)

        # Actual next-period return (in log-odds space)
        actual_return = logits[t + forecast_horizon] - logits[t]
        actual_var = actual_return ** 2  # squared return as realized variance

        records.append({
            "timestamp": timestamps[t],
            "price": prices[t],
            "signal": signal,
            "predicted_var": pred_var,
            "actual_return": actual_return,
            "actual_var": actual_var,
            "sigma_b": cal.sigma_b,
            "jump_intensity": cal.jump_intensity,
            "mean_reversion": cal.mean_reversion,
        })

    return pd.DataFrame(records)


# ── Metric Calculations ─────────────────────────────────────────────────────

def _information_coefficient(signals: pd.DataFrame) -> float:
    """Rank correlation between signal and next-period return (Spearman)."""
    if signals.empty or len(signals) < 10:
        return 0.0
    rho, _ = sp_stats.spearmanr(signals["signal"], signals["actual_return"])
    return float(rho) if not np.isnan(rho) else 0.0


def _qlike_loss(signals: pd.DataFrame) -> float:
    """
    QLIKE loss for volatility forecasts.

    QLIKE = mean(rv / fv - log(rv / fv) - 1)
    where rv = realised variance, fv = forecast variance.
    Lower is better. A perfect forecast scores 0.
    """
    if signals.empty or len(signals) < 10:
        return float("inf")

    rv = signals["actual_var"].values
    fv = signals["predicted_var"].values

    # Regularise to avoid division by zero
    rv = np.clip(rv, 1e-10, None)
    fv = np.clip(fv, 1e-10, None)

    ratio = rv / fv
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    return qlike


def _long_short_spread(signals: pd.DataFrame) -> Dict[str, float]:
    """
    Average return of top-quintile minus bottom-quintile signals.

    Positive spread means the signal has directional value.
    """
    if signals.empty or len(signals) < 20:
        return {"spread": 0.0, "long_return": 0.0, "short_return": 0.0}

    q20 = signals["signal"].quantile(0.20)
    q80 = signals["signal"].quantile(0.80)

    # Top quintile = strongest "buy" signal; bottom = strongest "sell"
    longs = signals[signals["signal"] >= q80]
    shorts = signals[signals["signal"] <= q20]

    long_ret = float(longs["actual_return"].mean()) if len(longs) > 0 else 0.0
    short_ret = float(shorts["actual_return"].mean()) if len(shorts) > 0 else 0.0

    return {
        "spread": long_ret - short_ret,
        "long_return": long_ret,
        "short_return": short_ret,
    }


def _mean_reversion_capture(signals: pd.DataFrame, threshold_z: float = 1.5) -> Dict[str, float]:
    """
    After large moves (|return| > threshold * sigma), measure how much
    of the subsequent reversal the model's signal captures.
    """
    if signals.empty or len(signals) < 30:
        return {"capture_rate": 0.0, "n_events": 0}

    ret = signals["actual_return"].values
    sigma = float(np.std(ret))
    if sigma < 1e-8:
        return {"capture_rate": 0.0, "n_events": 0}

    threshold = threshold_z * sigma
    big_moves = np.where(np.abs(ret) > threshold)[0]

    if len(big_moves) == 0:
        return {"capture_rate": 0.0, "n_events": 0}

    captures = 0
    total = 0
    for idx in big_moves:
        if idx + 1 >= len(ret):
            continue
        move_dir = np.sign(ret[idx])
        next_ret = ret[idx + 1]
        # Reversion = next return has opposite sign
        if move_dir * next_ret < 0:
            signal_at_event = signals["signal"].iloc[idx + 1]
            # Did the signal agree with the reversion direction?
            if signal_at_event * next_ret > 0:
                captures += 1
            total += 1
        else:
            total += 1

    capture_rate = captures / total if total > 0 else 0.0
    return {"capture_rate": capture_rate, "n_events": total}


def _sharpe_ratio(signals: pd.DataFrame) -> float:
    """
    Annualised Sharpe ratio of a simple signal-following strategy.

    Strategy: go long proportional to signal. PnL = signal * actual_return.
    """
    if signals.empty or len(signals) < 20:
        return 0.0

    # Normalise signal to [-1, 1] range
    sig = signals["signal"].values
    sig_max = np.abs(sig).max()
    if sig_max > 0:
        sig = sig / sig_max

    pnl = sig * signals["actual_return"].values

    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl))
    if std_pnl < 1e-10:
        return 0.0

    # Estimate observations per year for annualisation
    if len(signals) > 1:
        ts = pd.to_datetime(signals["timestamp"].values)
        total_days = (ts.max() - ts.min()).total_seconds() / 86400
        if total_days > 0:
            obs_per_year = len(signals) / total_days * 365.25
        else:
            obs_per_year = 252
    else:
        obs_per_year = 252

    sharpe = mean_pnl / std_pnl * math.sqrt(obs_per_year)
    return float(sharpe)


def _calibration_summary(signals: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics of the calibrated parameters across walk-forward."""
    if signals.empty:
        return {}
    return {
        "sigma_b_mean": float(signals["sigma_b"].mean()),
        "sigma_b_std": float(signals["sigma_b"].std()),
        "jump_intensity_mean": float(signals["jump_intensity"].mean()),
        "mean_reversion_mean": float(signals["mean_reversion"].mean()),
    }


# ── Per-Ticker Evaluation ───────────────────────────────────────────────────

def evaluate_ticker(
    df: pd.DataFrame,
    ticker: str = "",
    train_window: int = 200,
    forecast_horizon: int = 1,
) -> Dict[str, Any]:
    """Full evaluation for a single ticker."""
    logger.info("Evaluating %s (%d observations)...", ticker, len(df))

    signals = _walk_forward_signals(
        df, train_window=train_window, forecast_horizon=forecast_horizon,
    )

    if signals.empty:
        logger.warning("No walk-forward signals for %s", ticker)
        return {"ticker": ticker, "n_obs": len(df), "status": "insufficient_data"}

    ic = _information_coefficient(signals)
    qlike = _qlike_loss(signals)
    ls = _long_short_spread(signals)
    mr = _mean_reversion_capture(signals)
    sharpe = _sharpe_ratio(signals)
    cal_summary = _calibration_summary(signals)

    result = {
        "ticker": ticker,
        "n_obs": len(df),
        "n_eval_steps": len(signals),
        "ic": ic,
        "qlike": qlike,
        "long_short_spread": ls["spread"],
        "long_return_avg": ls["long_return"],
        "short_return_avg": ls["short_return"],
        "mean_rev_capture_rate": mr["capture_rate"],
        "mean_rev_events": mr["n_events"],
        "sharpe": sharpe,
        **cal_summary,
    }

    logger.info(
        "  %s — IC=%.4f  QLIKE=%.4f  LS=%.4f  MR_cap=%.2f  Sharpe=%.2f",
        ticker, ic, qlike, ls["spread"], mr["capture_rate"], sharpe,
    )
    return result


# ── Aggregate Evaluation ────────────────────────────────────────────────────

def run_evaluation(
    hf_data: Dict[str, pd.DataFrame],
    fundamental_rows: List[Dict[str, Any]] | None = None,
    train_window: int = 200,
    forecast_horizon: int = 1,
    min_obs: int = 250,
) -> Dict[str, Any]:
    """
    Run walk-forward evaluation across all tickers with sufficient data.

    Parameters
    ----------
    hf_data : dict[str, DataFrame]
        Per-ticker HF price data.
    fundamental_rows : list[dict], optional
        Fundamental model scan results (for future cross-model comparison).
    train_window : int
        Calibration window size.
    forecast_horizon : int
        Steps ahead to forecast.
    min_obs : int
        Skip tickers with fewer than this many observations.

    Returns
    -------
    dict
        Aggregate metrics and per-ticker breakdowns.
    """
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    eligible = {
        tkr: df for tkr, df in hf_data.items() if len(df) >= min_obs
    }

    logger.info(
        "Evaluating %d tickers with >= %d observations (of %d total)",
        len(eligible), min_obs, len(hf_data),
    )

    if not eligible:
        logger.warning("No tickers with enough data (need >= %d obs)", min_obs)
        return {
            "status": "insufficient_data",
            "n_tickers_total": len(hf_data),
            "n_tickers_eligible": 0,
        }

    per_ticker: List[Dict[str, Any]] = []
    for ticker, df in eligible.items():
        result = evaluate_ticker(
            df, ticker, train_window=train_window,
            forecast_horizon=forecast_horizon,
        )
        per_ticker.append(result)

    # Aggregate (only from tickers with valid results)
    valid = [r for r in per_ticker if "ic" in r]
    if not valid:
        return {"status": "all_failed", "per_ticker": per_ticker}

    ic_values = [r["ic"] for r in valid]
    sharpe_values = [r["sharpe"] for r in valid]
    qlike_values = [r["qlike"] for r in valid if r["qlike"] != float("inf")]
    ls_values = [r["long_short_spread"] for r in valid]
    mr_values = [r["mean_rev_capture_rate"] for r in valid if r["mean_rev_events"] > 0]

    aggregate = {
        "status": "ok",
        "n_tickers_total": len(hf_data),
        "n_tickers_eligible": len(eligible),
        "n_tickers_evaluated": len(valid),
        "total_eval_steps": sum(r.get("n_eval_steps", 0) for r in valid),

        # IC
        "ic_mean": float(np.mean(ic_values)),
        "ic_median": float(np.median(ic_values)),
        "ic_std": float(np.std(ic_values)),
        "ic_positive_pct": float(np.mean(np.array(ic_values) > 0)),

        # Volatility forecasting
        "qlike_mean": float(np.mean(qlike_values)) if qlike_values else float("inf"),
        "qlike_median": float(np.median(qlike_values)) if qlike_values else float("inf"),

        # Long-short
        "ls_spread_mean": float(np.mean(ls_values)),
        "ls_spread_median": float(np.median(ls_values)),

        # Mean-reversion
        "mean_rev_capture_mean": float(np.mean(mr_values)) if mr_values else 0.0,

        # Risk-adjusted
        "sharpe_mean": float(np.mean(sharpe_values)),
        "sharpe_median": float(np.median(sharpe_values)),
        "sharpe_positive_pct": float(np.mean(np.array(sharpe_values) > 0)),
    }

    # Save results
    output_path = _EVAL_DIR / "belief_model_eval.json"
    full_output = {"aggregate": aggregate, "per_ticker": per_ticker}
    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    logger.info("Evaluation saved to %s", output_path)

    return aggregate
