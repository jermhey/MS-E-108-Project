"""
hybrid_signal.py — Signal Generators (Pure Belief + Hybrid)
============================================================
Two modes:

**Pure belief** (recommended — ``generate_pure_belief_signals``):
    Uses only Kalshi price data.  Fair value = Kalman-filtered mid-price
    in probability space.  Dynamic spread from belief-vol.  Mean-reversion
    overlay after large moves.

**Hybrid** (``generate_hybrid_signals``):
    Blends the fundamental MC-OU fair-value anchor (30 %) with the
    market price (70 %) and overlays belief-vol spreads.  Our backtests
    show this *under-performs* the pure model at every horizon.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.belief_model import BeliefCalibration, forecast_variance
from src.utils import get_logger

logger = get_logger("tauroi.hybrid_signal")

# Blending weight: how much to trust the fundamental model vs market
FUNDAMENTAL_WEIGHT = 0.30

# Mean-reversion thresholds (in units of local sigma)
MEAN_REV_TRIGGER_Z = 1.5    # trigger reversion signal at 1.5σ
MEAN_REV_BOOST = 1.5        # size multiplier when mean-reversion agrees

# Baseline sigma_b for spread scaling (approximate median from EDA)
BASELINE_SIGMA_B = 0.80

# Minimum / maximum dynamic spread in cents
# Backtest (mm_backtest.ipynb) shows net profitability requires ≥8c
# half-spread after Kalshi's 1.5c/fill maker fee.
MIN_SPREAD_CENTS = 5
MAX_SPREAD_CENTS = 15
BASE_SPREAD_CENTS = 8


def _compute_dynamic_spread(cal: BeliefCalibration) -> int:
    """
    Compute dynamic half-spread in cents from belief-vol.

    Wider when sigma_b is high (volatile / uncertain), narrower when low.
    """
    multiplier = cal.spread_multiplier(BASELINE_SIGMA_B)
    spread = BASE_SPREAD_CENTS * multiplier
    return int(np.clip(round(spread), MIN_SPREAD_CENTS, MAX_SPREAD_CENTS))


def _detect_mean_reversion(
    cal: BeliefCalibration,
    hf_df: pd.DataFrame,
    lookback: int = 5,
) -> Dict[str, Any]:
    """
    Detect if a recent large move should trigger a mean-reversion signal.

    Returns a dict with:
        triggered: bool
        direction: "buy" | "sell" | None
        z_score: float  (how many sigmas the recent move was)
    """
    result: Dict[str, Any] = {
        "triggered": False, "direction": None, "z_score": 0.0,
    }

    if cal.mean_reversion >= -0.05:
        return result

    if hf_df.empty or len(hf_df) < lookback + 2:
        return result

    logits = hf_df["logit"].values
    recent_inc = logits[-1] - logits[-lookback]

    # Use the belief model's sigma_b to normalize
    # Convert annual sigma_b to the lookback-period scale
    # Approximate: dt per step from data
    if len(hf_df) > 1:
        ts = hf_df["timestamp"].values
        dt_sec = float(np.median(np.diff(ts.astype(np.int64) // 10**9)))
        dt_sec = max(dt_sec, 1.0)
    else:
        dt_sec = 86400.0

    dt_years = dt_sec / (365.25 * 86400)
    sigma_period = cal.sigma_b * math.sqrt(dt_years * lookback)

    if sigma_period < 1e-6:
        return result

    z = recent_inc / sigma_period
    result["z_score"] = float(z)

    if abs(z) > MEAN_REV_TRIGGER_Z:
        result["triggered"] = True
        # Fade the move: if price went UP (z > 0), signal is SELL; vice versa
        result["direction"] = "sell" if z > 0 else "buy"

    return result


def generate_hybrid_signals(
    fundamental_rows: List[Dict[str, Any]],
    belief_cals: Dict[str, BeliefCalibration],
    hf_data: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    """
    Produce scan_rows that the OrderExecutor can consume directly.

    For each artist/contract in *fundamental_rows*, blends the
    fundamental fair value with market data and attaches dynamic
    spread and mean-reversion overlays from the belief model.

    Parameters
    ----------
    fundamental_rows : list[dict]
        Output from the existing MC-OU scan pipeline.
    belief_cals : dict[str, BeliefCalibration]
        Per-ticker belief calibrations.
    hf_data : dict[str, pd.DataFrame]
        Per-ticker high-frequency price data.

    Returns
    -------
    list[dict]
        Enhanced scan_rows with added keys:
            dynamic_spread_cents, mean_rev_signal, hybrid_fair_value,
            belief_sigma_b, belief_jump_intensity
    """
    output: List[Dict[str, Any]] = []

    for row in fundamental_rows:
        ticker = row.get("ticker")
        if not ticker:
            output.append(row)
            continue

        fundamental_fv = row.get("fair_value", 0.5)
        market_price = row.get("market_price", 0.5)

        cal = belief_cals.get(ticker)
        hf = hf_data.get(ticker)

        if cal is None:
            # No belief model — pass through the fundamental signal unchanged
            row["dynamic_spread_cents"] = BASE_SPREAD_CENTS
            row["hybrid_fair_value"] = fundamental_fv
            output.append(row)
            continue

        # 1. Blended mid-price: weighted average of fundamental and market
        blended_fv = (
            FUNDAMENTAL_WEIGHT * fundamental_fv
            + (1.0 - FUNDAMENTAL_WEIGHT) * market_price
        )
        blended_fv = float(np.clip(blended_fv, 0.01, 0.99))

        # 2. Dynamic spread from belief volatility
        dyn_spread = _compute_dynamic_spread(cal)

        # 3. Mean-reversion overlay
        mr = _detect_mean_reversion(cal, hf) if hf is not None else {
            "triggered": False, "direction": None, "z_score": 0.0,
        }

        # Adjust fair value for mean-reversion
        # If price recently spiked up and we expect reversion, shade our
        # fair value slightly lower (making us more likely to sell into
        # the spike and buy after it reverts).
        if mr["triggered"]:
            shade = abs(mr["z_score"]) * 0.01  # 1c per sigma of overshoot
            shade = min(shade, 0.05)
            if mr["direction"] == "buy":
                blended_fv += shade
            else:
                blended_fv -= shade
            blended_fv = float(np.clip(blended_fv, 0.01, 0.99))

        # Compute edge from the hybrid fair value
        edge = blended_fv - market_price

        # Determine signal
        edge_threshold = dyn_spread / 100.0  # spread in decimal
        if edge > edge_threshold:
            signal = "BUY"
        elif edge < -edge_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Build enhanced row
        enhanced = dict(row)
        enhanced.update({
            "hybrid_fair_value": round(blended_fv, 4),
            "fair_value": round(blended_fv, 4),
            "edge": round(edge, 4),
            "signal": signal,
            "dynamic_spread_cents": dyn_spread,
            "mean_rev_triggered": mr["triggered"],
            "mean_rev_direction": mr["direction"],
            "mean_rev_z": round(mr["z_score"], 2),
            "belief_sigma_b": round(cal.sigma_b, 4),
            "belief_jump_intensity": round(cal.jump_intensity, 1),
            "belief_jump_std": round(cal.jump_std, 4),
            "belief_mean_reversion": round(cal.mean_reversion, 4),
            "belief_rn_drift": round(cal.rn_drift, 6),
        })

        # Conviction boost when mean-reversion and fundamental agree
        if mr["triggered"] and signal != "HOLD":
            if (mr["direction"] == "buy" and signal == "BUY") or \
               (mr["direction"] == "sell" and signal == "SELL"):
                enhanced["conviction_boost"] = MEAN_REV_BOOST
            else:
                enhanced["conviction_boost"] = 1.0
        else:
            enhanced["conviction_boost"] = 1.0

        output.append(enhanced)

    # Sort by absolute edge descending
    output.sort(key=lambda r: abs(r.get("edge", 0)), reverse=True)

    logger.info(
        "Hybrid signals: %d rows, %d with belief model, %d mean-rev triggered",
        len(output),
        sum(1 for r in output if r.get("belief_sigma_b") is not None),
        sum(1 for r in output if r.get("mean_rev_triggered")),
    )
    return output


# ── Pure Belief Model Signals ────────────────────────────────────────────────

def generate_pure_belief_signals(
    belief_cals: Dict[str, BeliefCalibration],
    hf_data: Dict[str, pd.DataFrame],
    market_tickers: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate scan_rows purely from the belief model (no fundamental input).

    Fair value = Kalman-filtered mid-price (sigmoid of filtered log-odds).
    This is the "Toward Black-Scholes for Prediction Markets" approach
    and out-performs the hybrid at every horizon in our backtests.

    Parameters
    ----------
    belief_cals : dict[str, BeliefCalibration]
        Per-ticker calibrations from ``calibrate_belief()``.
    hf_data : dict[str, pd.DataFrame]
        Per-ticker HF price data (tick-level or 1-min candles).
    market_tickers : list[str], optional
        If given, only produce signals for these tickers.

    Returns
    -------
    list[dict]
        scan_rows format consumable by ``OrderExecutor``.
    """
    from src.belief_model import kalman_filter as _kf, _sigmoid

    tickers = market_tickers if market_tickers else list(belief_cals.keys())
    output: List[Dict[str, Any]] = []

    for ticker in tickers:
        cal = belief_cals.get(ticker)
        hf = hf_data.get(ticker)
        if cal is None or hf is None or hf.empty:
            continue

        logits = hf["logit"].values
        prices = hf["mid_price"].values

        # Kalman-filtered fair value in probability space
        x_hat = _kf(logits)
        fair_logit = float(x_hat[-1])
        fair_price = float(_sigmoid(fair_logit))
        fair_price = float(np.clip(fair_price, 0.01, 0.99))

        market_price = float(prices[-1])

        # Edge = fair - market
        edge = fair_price - market_price

        # Dynamic spread from belief-vol
        dyn_spread = _compute_dynamic_spread(cal)

        # Mean-reversion overlay
        mr = _detect_mean_reversion(cal, hf)

        # Shade fair value for mean-reversion
        if mr["triggered"]:
            shade = abs(mr["z_score"]) * 0.01
            shade = min(shade, 0.05)
            if mr["direction"] == "buy":
                fair_price += shade
            else:
                fair_price -= shade
            fair_price = float(np.clip(fair_price, 0.01, 0.99))
            edge = fair_price - market_price

        # Signal
        edge_threshold = dyn_spread / 100.0
        if edge > edge_threshold:
            signal = "BUY"
        elif edge < -edge_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        row: Dict[str, Any] = {
            "ticker": ticker,
            "fair_value": round(fair_price, 4),
            "market_price": round(market_price, 4),
            "edge": round(edge, 4),
            "signal": signal,
            "dynamic_spread_cents": dyn_spread,
            "mean_rev_triggered": mr["triggered"],
            "mean_rev_direction": mr["direction"],
            "mean_rev_z": round(mr["z_score"], 2),
            "belief_sigma_b": round(cal.sigma_b, 4),
            "belief_jump_intensity": round(cal.jump_intensity, 1),
            "belief_jump_std": round(cal.jump_std, 4),
            "belief_mean_reversion": round(cal.mean_reversion, 4),
            "belief_rn_drift": round(cal.rn_drift, 6),
            "conviction_boost": MEAN_REV_BOOST if (
                mr["triggered"] and signal != "HOLD"
                and ((mr["direction"] == "buy" and signal == "BUY")
                     or (mr["direction"] == "sell" and signal == "SELL"))
            ) else 1.0,
            "model": "pure_belief",
        }
        output.append(row)

    output.sort(key=lambda r: abs(r.get("edge", 0)), reverse=True)

    logger.info(
        "Pure belief signals: %d tickers, %d mean-rev triggered, "
        "avg spread %dc",
        len(output),
        sum(1 for r in output if r.get("mean_rev_triggered")),
        int(np.mean([r["dynamic_spread_cents"] for r in output])) if output else 0,
    )
    return output
