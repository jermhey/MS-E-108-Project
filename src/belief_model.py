"""
belief_model.py — Logit Jump-Diffusion Belief Model
=====================================================
Implements the core pricing kernel from "Toward Black-Scholes for
Prediction Markets" (arXiv:2510.15205), adapted for Kalshi KXTOPMONTHLY
contracts.

The model treats the traded probability p_t as a Q-martingale, maps it
to log-odds x_t = logit(p_t), and calibrates a jump-diffusion:

    dx = mu(t,x) dt  +  sigma_b dW  +  J dN

where the drift mu is pinned by the martingale constraint, sigma_b is
the belief volatility, and (J, N) is a compound-Poisson jump process.

Calibration uses an EM algorithm to separate diffusion from jumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("tauroi.belief_model")


# ── Logit helpers ────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """S(x) = 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sigmoid_prime(x: np.ndarray | float) -> np.ndarray | float:
    """S'(x) = p(1-p)"""
    s = _sigmoid(x)
    return s * (1.0 - s)


def _sigmoid_double_prime(x: np.ndarray | float) -> np.ndarray | float:
    """S''(x) = p(1-p)(1-2p)"""
    s = _sigmoid(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class BeliefCalibration:
    """Calibrated belief-model parameters for a single contract."""

    ticker: str

    # Core kernel parameters
    sigma_b: float          # belief volatility (annualised, in log-odds space)
    jump_intensity: float   # Poisson arrival rate (annualised)
    jump_std: float         # log-normal std of jump sizes in log-odds

    # Price dynamics
    mean_reversion: float   # lag-1 autocorrelation of Δx (negative = reverts)
    vol_of_vol: float       # persistence of volatility (autocorr of |Δx|)

    # Current state
    last_logit: float       # most recent filtered log-odds
    last_price: float       # most recent mid-price
    n_observations: int     # number of data points used

    # Belief-vol surface: sigma_b estimates bucketed by (dte, moneyness)
    sigma_b_surface: Dict[str, float] = field(default_factory=dict)

    # Risk-neutral drift at current state
    rn_drift: float = 0.0

    def spread_multiplier(self, base_sigma: float | None = None) -> float:
        """
        Ratio of current belief-vol to a baseline.  Use this to scale
        the market-making half-spread dynamically.
        """
        base = base_sigma if base_sigma is not None else 0.8
        if base <= 0:
            return 1.0
        return max(0.3, min(self.sigma_b / base, 4.0))


# ── Kalman Filter ────────────────────────────────────────────────────────────

def kalman_filter(
    logits: np.ndarray,
    process_var_init: float | None = None,
    meas_var: float = 0.01,
) -> np.ndarray:
    """
    Simple scalar Kalman filter on log-odds to denoise mid-prices.

    Uses a local-level model:  x_{t+1} = x_t + w_t,  y_t = x_t + v_t
    where w_t ~ N(0, Q) and v_t ~ N(0, R).

    Q is estimated from a rolling variance of raw increments.
    R reflects microstructure noise (tick-size quantisation).
    """
    n = len(logits)
    if n < 3:
        return logits.copy()

    raw_inc = np.diff(logits)
    Q = float(np.nanmedian(raw_inc ** 2)) if process_var_init is None else process_var_init
    Q = max(Q, 1e-8)
    R = meas_var

    x_hat = np.empty(n)
    x_hat[0] = logits[0]
    P = Q

    for t in range(1, n):
        # Predict
        x_pred = x_hat[t - 1]
        P_pred = P + Q

        # Update
        K = P_pred / (P_pred + R)
        x_hat[t] = x_pred + K * (logits[t] - x_pred)
        P = (1 - K) * P_pred

    return x_hat


# ── EM Calibration ───────────────────────────────────────────────────────────

def _em_step(
    increments: np.ndarray,
    dt: float,
    sigma_sq: float,
    lam: float,
    s_j_sq: float,
    n_iter: int = 6,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Expectation-Maximisation to separate diffusion from jumps.

    Models each increment as a mixture:
        - Diffusion: N(0, sigma^2 * dt)
        - Jump:      N(0, s_J^2)

    Returns (sigma_sq, lambda, s_J_sq, gamma) where gamma[t] is the
    posterior probability that increment t was a jump.
    """
    n = len(increments)
    if n < 5:
        return sigma_sq, lam, s_j_sq, np.zeros(n)

    gamma = np.zeros(n)

    for _ in range(n_iter):
        # Variance of each component
        var_d = max(sigma_sq * dt, 1e-12)
        var_j = max(s_j_sq, 1e-12)

        # E-step: posterior probability of jump
        log_p_diff = -0.5 * (increments ** 2 / var_d + np.log(var_d))
        log_p_jump = -0.5 * (increments ** 2 / var_j + np.log(var_j))

        p_mix = lam * dt  # prior prob of jump per step
        p_mix = np.clip(p_mix, 1e-6, 1 - 1e-6)

        log_w_diff = np.log(1 - p_mix) + log_p_diff
        log_w_jump = np.log(p_mix) + log_p_jump
        log_max = np.maximum(log_w_diff, log_w_jump)
        log_denom = log_max + np.log(
            np.exp(log_w_diff - log_max) + np.exp(log_w_jump - log_max)
        )
        gamma = np.exp(log_w_jump - log_denom)
        gamma = np.clip(gamma, 0, 1)

        # M-step
        w_diff = 1.0 - gamma
        w_jump = gamma

        sum_w_diff = w_diff.sum()
        sum_w_jump = w_jump.sum()

        if sum_w_diff > 1:
            sigma_sq = float((w_diff * increments ** 2).sum() / (sum_w_diff * dt))
        sigma_sq = max(sigma_sq, 1e-8)

        if sum_w_jump > 0.5:
            s_j_sq = float((w_jump * increments ** 2).sum() / sum_w_jump)
        s_j_sq = max(s_j_sq, sigma_sq * dt * 4)

        lam = float(sum_w_jump / (n * dt))
        lam = np.clip(lam, 0.1, 500.0)

    return sigma_sq, lam, s_j_sq, gamma


# ── Risk-Neutral Drift ──────────────────────────────────────────────────────

def _rn_drift(
    x: float,
    sigma_b_sq: float,
    lam: float,
    s_j: float,
    n_mc: int = 500,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute the risk-neutral drift mu(t, x) that makes p_t = S(x_t) a
    Q-martingale.

    From arXiv:2510.15205 Eq. 3:
        mu = -[0.5 * S''(x) * sigma_b^2 + jump_compensation] / S'(x)
    """
    sp = _sigmoid_prime(x)
    if abs(sp) < 1e-10:
        return 0.0

    spp = _sigmoid_double_prime(x)
    diffusion_term = 0.5 * spp * sigma_b_sq

    # Jump compensation via Monte Carlo
    if rng is None:
        rng = np.random.default_rng(42)

    z_samples = rng.normal(0, s_j, size=n_mc)
    s_x = _sigmoid(x)
    s_xz = _sigmoid(x + z_samples)
    sp_x = sp

    # Truncation function chi(z) = z * 1{|z|<1}
    chi = z_samples * (np.abs(z_samples) < 1.0)
    jump_comp = lam * float(np.mean(s_xz - s_x - sp_x * chi))

    mu = -(diffusion_term + jump_comp) / sp
    return float(np.clip(mu, -5.0, 5.0))


# ── Rolling Calibration ─────────────────────────────────────────────────────

def calibrate_belief(
    series: pd.DataFrame,
    ticker: str = "",
    window: int | None = None,
) -> BeliefCalibration:
    """
    Calibrate the logit jump-diffusion from a HF price series.

    Parameters
    ----------
    series : DataFrame
        Must have columns: timestamp, mid_price, logit.
        Can be tick-level or candlestick data.
    ticker : str
        Contract identifier (for labelling).
    window : int or None
        Rolling window for EM calibration.  None = use all data.

    Returns
    -------
    BeliefCalibration
    """
    if series.empty or len(series) < 10:
        logger.warning("Too few observations for %s (%d)", ticker, len(series))
        return BeliefCalibration(
            ticker=ticker, sigma_b=1.0, jump_intensity=12.0,
            jump_std=0.5, mean_reversion=0.0, vol_of_vol=0.0,
            last_logit=0.0, last_price=0.5, n_observations=len(series),
        )

    logits = series["logit"].values.astype(np.float64)
    timestamps = series["timestamp"].values

    # Kalman filter to denoise
    x_hat = kalman_filter(logits)

    # Compute increments and dt
    increments = np.diff(x_hat)

    if len(timestamps) > 1:
        ts_arr = np.array(
            pd.to_datetime(timestamps).astype(np.int64) // 10**9,
            dtype=np.float64,
        )
        ts_seconds = np.diff(ts_arr)
        ts_seconds[ts_seconds <= 0] = 1.0
        median_dt_seconds = float(np.median(ts_seconds))
    else:
        median_dt_seconds = 86400.0

    dt_years = median_dt_seconds / (365.25 * 86400)
    dt_years = max(dt_years, 1e-8)

    # Use the most recent window for calibration
    if window is not None and len(increments) > window:
        cal_inc = increments[-window:]
        cal_logits = x_hat[-(window + 1):]
    else:
        cal_inc = increments
        cal_logits = x_hat

    # Initial parameter estimates
    var_inc = float(np.var(cal_inc))
    sigma_sq_init = max(var_inc / dt_years, 0.01)
    s_j_sq_init = var_inc * 4.0
    lam_init = 12.0

    sigma_sq, lam, s_j_sq, gamma = _em_step(
        cal_inc, dt_years, sigma_sq_init, lam_init, s_j_sq_init,
    )

    sigma_b = math.sqrt(max(sigma_sq, 0.0))
    jump_std = math.sqrt(max(s_j_sq, 0.0))

    # Mean-reversion: lag-1 autocorrelation of increments
    if len(cal_inc) > 5:
        mean_rev = float(pd.Series(cal_inc).autocorr(lag=1))
        if np.isnan(mean_rev):
            mean_rev = 0.0
    else:
        mean_rev = 0.0

    # Volatility clustering: autocorrelation of |increments|
    if len(cal_inc) > 5:
        vol_of_vol = float(pd.Series(np.abs(cal_inc)).autocorr(lag=1))
        if np.isnan(vol_of_vol):
            vol_of_vol = 0.0
    else:
        vol_of_vol = 0.0

    # Risk-neutral drift at current state
    last_x = float(x_hat[-1])
    rn = _rn_drift(last_x, sigma_sq, lam, jump_std)

    # Belief-vol surface: bucket by approximate DTE and price level
    surface = _build_surface(series, x_hat, dt_years)

    last_p = float(series["mid_price"].iloc[-1])

    cal = BeliefCalibration(
        ticker=ticker,
        sigma_b=sigma_b,
        jump_intensity=lam,
        jump_std=jump_std,
        mean_reversion=mean_rev,
        vol_of_vol=vol_of_vol,
        last_logit=last_x,
        last_price=last_p,
        n_observations=len(series),
        sigma_b_surface=surface,
        rn_drift=rn,
    )

    logger.info(
        "Belief calibration %s: σ_b=%.3f  λ=%.1f  s_J=%.3f  "
        "mean_rev=%.3f  n=%d",
        ticker, sigma_b, lam, jump_std, mean_rev, len(series),
    )
    return cal


def _build_surface(
    series: pd.DataFrame,
    x_hat: np.ndarray,
    dt_years: float,
) -> Dict[str, float]:
    """
    Build a coarse belief-vol surface indexed by price bucket.

    For each bucket, estimate local sigma_b from the increments
    that fall in that bucket.
    """
    surface: Dict[str, float] = {}
    if len(x_hat) < 20:
        return surface

    prices = series["mid_price"].values
    increments = np.diff(x_hat)
    bucket_prices = prices[:-1]

    bins = [(0, 0.10), (0.10, 0.25), (0.25, 0.50),
            (0.50, 0.75), (0.75, 0.90), (0.90, 1.0)]

    for lo, hi in bins:
        mask = (bucket_prices >= lo) & (bucket_prices < hi)
        if mask.sum() < 5:
            continue
        local_var = float(np.var(increments[mask]))
        local_sigma = math.sqrt(max(local_var / dt_years, 0.0))
        key = f"p_{lo:.2f}_{hi:.2f}"
        surface[key] = local_sigma

    return surface


# ── Variance Forecasting ────────────────────────────────────────────────────

def forecast_variance(
    cal: BeliefCalibration,
    horizon_steps: int,
    dt_years: float,
) -> float:
    """
    Forecast realized variance of log-odds over the next *horizon_steps*.

    V_hat = sigma_b^2 * dt * H  +  lambda * dt * H * s_J^2
    """
    diffusion_var = cal.sigma_b ** 2 * dt_years * horizon_steps
    jump_var = cal.jump_intensity * dt_years * horizon_steps * cal.jump_std ** 2
    return diffusion_var + jump_var


# ── Batch Calibration ───────────────────────────────────────────────────────

def calibrate_all(
    hf_data: Dict[str, pd.DataFrame],
    window: int | None = None,
) -> Dict[str, BeliefCalibration]:
    """Run belief calibration on every ticker in the HF dataset."""
    results: Dict[str, BeliefCalibration] = {}
    for ticker, df in hf_data.items():
        results[ticker] = calibrate_belief(df, ticker=ticker, window=window)
    logger.info("Calibrated %d belief models", len(results))
    return results
