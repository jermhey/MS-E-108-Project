"""
as_detector.py — Adverse-Selection Detector via Logit Jump-Diffusion
=====================================================================
Implements the Dalen (2025) RN-JD calibration pipeline on Kalshi tick
data to produce a real-time adverse-selection score.

Three signal layers:
  A) Jump posterior γ_t from rolling EM calibration
  B) Trade arrival burst detector
  C) Composite AS score combining A + B

Reference: "Toward Black-Scholes for Prediction Markets" (arXiv:2510.15205)
           Sections 3.1-3.2 (kernel), 5.1-5.2 (calibration pipeline)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("tauroi.as_detector")


# ── Logit helpers (matching the paper's notation) ────────────────────────────

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """S(x) = 1 / (1 + exp(-x)), the logistic sigmoid."""
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
class ASEvent:
    """A single flagged adverse-selection event."""
    timestamp: pd.Timestamp
    ticker: str
    score: float           # composite AS score in [0, 1]
    gamma: float           # jump posterior from EM
    burst_flag: bool       # whether arrival-rate anomaly fired
    price_at_flag: float   # mid-price when flagged
    logit_at_flag: float   # log-odds when flagged
    sigma_b_local: float   # local belief-vol estimate
    lambda_local: float    # local jump intensity
    detail: str = ""


@dataclass
class CalibrationSnapshot:
    """Rolling calibration state at a point in time."""
    sigma_b_sq: float      # diffusion variance (per unit time)
    lam: float             # jump intensity (per unit time)
    s_j_sq: float          # jump variance
    gamma: np.ndarray      # posterior jump probabilities for the window
    rn_drift: float        # risk-neutral drift at current state
    x_filtered: np.ndarray # Kalman-filtered log-odds


# ── Heteroskedastic Kalman Filter (Dalen Section 5.1) ───────────────────────

def kalman_filter_hf(
    logits: np.ndarray,
    dt_seconds: np.ndarray,
    meas_var: float = 0.005,
) -> np.ndarray:
    """
    Heteroskedastic Kalman filter on log-odds.

    Unlike the basic version in belief_model.py, this scales the
    process noise Q proportionally to dt between observations,
    matching the paper's recommendation for irregularly-spaced trades.

    Parameters
    ----------
    logits : array
        Raw log-odds from trade prices.
    dt_seconds : array
        Time gap (seconds) between consecutive observations.
        Length = len(logits) - 1.
    meas_var : float
        Measurement noise variance (tick-size quantisation noise).
        For 1-cent ticks on logit scale, ~0.005 is reasonable.
    """
    n = len(logits)
    if n < 3:
        return logits.copy()

    raw_inc = np.diff(logits)
    base_Q_per_sec = float(np.nanmedian(raw_inc ** 2 / np.maximum(dt_seconds, 1.0)))
    base_Q_per_sec = max(base_Q_per_sec, 1e-10)

    x_hat = np.empty(n)
    x_hat[0] = logits[0]
    P = base_Q_per_sec * float(np.median(dt_seconds))

    for t in range(1, n):
        dt = max(float(dt_seconds[t - 1]), 0.1)
        Q = base_Q_per_sec * dt

        x_pred = x_hat[t - 1]
        P_pred = P + Q

        K = P_pred / (P_pred + meas_var)
        x_hat[t] = x_pred + K * (logits[t] - x_pred)
        P = (1 - K) * P_pred

    return x_hat


# ── Rolling EM Calibration (Dalen Section 5.2) ──────────────────────────────

def rolling_em_calibrate(
    x_filtered: np.ndarray,
    dt_seconds: np.ndarray,
    window: int = 200,
    n_em_iter: int = 8,
    min_window: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling-window EM to separate diffusion from jumps at each time step.

    Returns time series of:
      sigma_b_sq[t] : local diffusion variance (per second)
      lam[t]        : local jump intensity (per second)
      s_j_sq[t]     : local jump variance
      gamma[t]      : posterior probability that trade t was a jump
    """
    increments = np.diff(x_filtered)
    n = len(increments)

    sigma_b_sq_ts = np.full(n, np.nan)
    lam_ts = np.full(n, np.nan)
    s_j_sq_ts = np.full(n, np.nan)
    gamma_ts = np.full(n, np.nan)

    # Global initialisation from first pass
    median_dt = float(np.median(dt_seconds))
    global_var = float(np.var(increments))
    sigma_sq_init = max(global_var / median_dt, 1e-8)
    s_j_sq_init = global_var * 4.0
    lam_init = 0.01  # ~1 jump per 100 seconds initially

    sigma_sq = sigma_sq_init
    lam = lam_init
    s_j_sq = s_j_sq_init

    for t in range(n):
        start = max(0, t - window + 1)
        if t - start < min_window:
            # Not enough data yet; use expanding window with global params
            win_inc = increments[:t + 1]
            win_dt = dt_seconds[:t + 1]
        else:
            win_inc = increments[start:t + 1]
            win_dt = dt_seconds[start:t + 1]

        w_len = len(win_inc)
        if w_len < 5:
            sigma_b_sq_ts[t] = sigma_sq
            lam_ts[t] = lam
            s_j_sq_ts[t] = s_j_sq
            gamma_ts[t] = 0.0
            continue

        sigma_sq_w, lam_w, s_j_sq_w, gamma_w = _em_step_hetero(
            win_inc, win_dt, sigma_sq, lam, s_j_sq, n_iter=n_em_iter,
        )

        sigma_b_sq_ts[t] = sigma_sq_w
        lam_ts[t] = lam_w
        s_j_sq_ts[t] = s_j_sq_w
        gamma_ts[t] = gamma_w[-1]  # posterior for the current observation

        # Carry forward as warm start for next window
        sigma_sq = sigma_sq_w
        lam = lam_w
        s_j_sq = s_j_sq_w

    return sigma_b_sq_ts, lam_ts, s_j_sq_ts, gamma_ts


def _em_step_hetero(
    increments: np.ndarray,
    dt_seconds: np.ndarray,
    sigma_sq: float,
    lam: float,
    s_j_sq: float,
    n_iter: int = 8,
    min_jump_logit: float = 0.15,
) -> Tuple[float, float, float, np.ndarray]:
    """
    EM separating diffusion from jumps with heterogeneous time steps.

    Each increment Δx_t has:
      - Diffusion component: N(0, σ² · dt_t)
      - Jump component: N(0, s_J²)     (occurs with prob λ · dt_t)

    For Kalshi's 1-cent tick markets, `min_jump_logit` prevents
    normal tick-to-tick moves from being classified as jumps.
    On the logit scale, a 2-cent move near p=0.5 is ~0.08, and near
    p=0.2 is ~0.13.  We set the floor at 0.15 to ensure only moves
    of 3+ cents in the active zone are considered jumps.
    """
    n = len(increments)
    if n < 5:
        return sigma_sq, lam, s_j_sq, np.zeros(n)

    gamma = np.zeros(n)
    dt = np.maximum(dt_seconds, 0.1)

    min_s_j_sq = min_jump_logit ** 2

    for _ in range(n_iter):
        var_d = np.maximum(sigma_sq * dt, 1e-12)
        var_j = max(s_j_sq, min_s_j_sq)

        # E-step
        log_p_diff = -0.5 * (increments ** 2 / var_d + np.log(var_d))
        log_p_jump = -0.5 * (increments ** 2 / var_j + np.log(var_j))

        p_mix = np.clip(lam * dt, 1e-6, 1 - 1e-6)

        log_w_diff = np.log(1 - p_mix) + log_p_diff
        log_w_jump = np.log(p_mix) + log_p_jump
        log_max = np.maximum(log_w_diff, log_w_jump)
        log_denom = log_max + np.log(
            np.exp(log_w_diff - log_max) + np.exp(log_w_jump - log_max)
        )
        gamma = np.clip(np.exp(log_w_jump - log_denom), 0, 1)

        # M-step
        w_diff = 1.0 - gamma
        w_jump = gamma

        sum_w_diff = w_diff.sum()
        sum_w_jump = w_jump.sum()

        if sum_w_diff > 1:
            sigma_sq = float(
                (w_diff * increments ** 2).sum()
                / (w_diff * dt).sum()
            )
        sigma_sq = max(sigma_sq, 1e-10)

        if sum_w_jump > 0.5:
            s_j_sq = float((w_jump * increments ** 2).sum() / sum_w_jump)
        s_j_sq = max(s_j_sq, min_s_j_sq)

        lam = float(sum_w_jump / dt.sum())
        lam = np.clip(lam, 1e-6, 0.1)  # cap at 1 jump per 10 seconds

    return sigma_sq, lam, s_j_sq, gamma


# ── Risk-Neutral Drift Enforcement (Dalen Eq. 3 / Eq. 14) ──────────────────

def compute_rn_drift(
    x: float,
    sigma_b_sq: float,
    lam: float,
    s_j: float,
    n_mc: int = 600,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute μ(t,x) so that p_t = S(x_t) is a Q-martingale.

    μ = -[½ S''(x) σ_b² + λ E[S(x+Z) - S(x) - S'(x)χ(Z)]] / S'(x)
    """
    sp = _sigmoid_prime(x)
    if abs(sp) < 1e-10:
        return 0.0

    spp = _sigmoid_double_prime(x)
    diffusion_term = 0.5 * spp * sigma_b_sq

    if rng is None:
        rng = np.random.default_rng(42)

    z = rng.normal(0, max(s_j, 1e-6), size=n_mc)
    s_x = _sigmoid(x)
    s_xz = _sigmoid(x + z)
    chi = z * (np.abs(z) < 1.0)
    jump_comp = lam * float(np.mean(s_xz - s_x - sp * chi))

    mu = -(diffusion_term + jump_comp) / sp
    return float(np.clip(mu, -5.0, 5.0))


# ── Burst Detector (Trade Arrival Clustering) ───────────────────────────────

def detect_bursts(
    timestamps: np.ndarray,
    short_window: int = 10,
    long_window: int = 500,
    burst_multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect abnormal clustering of trade arrivals.

    Compares a short-term arrival rate (last `short_window` trades) to a
    long-term baseline (last `long_window` trades).  When the short-term
    rate exceeds `burst_multiplier` times the long-term rate, flags a burst.

    This two-timescale approach avoids the adaptation problem where a
    single EMA drifts with the burst and never flags it.

    Returns
    -------
    burst_flags : bool array, length n
        True at each trade that occurs during a burst.
    arrival_rate_ratio : float array, length n
        Short-term rate / long-term rate. Values > burst_multiplier
        are flagged.
    """
    ts = pd.to_datetime(timestamps)
    dt_sec = np.diff(ts).astype("timedelta64[ms]").astype(float) / 1000.0
    dt_sec = np.maximum(dt_sec, 0.01)

    n = len(timestamps)
    rate_ratio = np.ones(n)
    burst_flags = np.zeros(n, dtype=bool)

    if n < long_window + 5:
        return burst_flags, rate_ratio

    for i in range(long_window, n - 1):
        # Short window: average ITI over last `short_window` trades
        short_start = max(0, i - short_window)
        short_iti = np.median(dt_sec[short_start:i + 1])

        # Long window: average ITI over last `long_window` trades
        long_start = max(0, i - long_window)
        long_iti = np.median(dt_sec[long_start:i + 1])

        if short_iti > 0 and long_iti > 0:
            # ratio > 1 means trades arriving faster than baseline
            rate_ratio[i + 1] = long_iti / short_iti
        elif short_iti <= 0.1:
            rate_ratio[i + 1] = burst_multiplier + 1

    burst_flags = rate_ratio > burst_multiplier
    return burst_flags, rate_ratio


# ── Composite Adverse Selection Score ────────────────────────────────────────

@dataclass
class ASResult:
    """Full adverse-selection analysis for one ticker."""
    ticker: str
    timestamps: np.ndarray
    prices: np.ndarray
    x_filtered: np.ndarray

    # Signal A: jump detection
    gamma: np.ndarray          # jump posterior at each trade
    sigma_b_sq: np.ndarray     # rolling belief-vol²
    lam: np.ndarray            # rolling jump intensity
    s_j_sq: np.ndarray         # rolling jump variance

    # Signal B: burst detection
    burst_flags: np.ndarray
    arrival_rate_ratio: np.ndarray

    # Composite
    as_score: np.ndarray       # combined adverse-selection score [0, 1]

    # Calibration metadata
    alpha_weight: float        # weight on gamma vs burst
    gamma_threshold: float
    burst_threshold: float

    @property
    def events(self) -> pd.DataFrame:
        """Return a DataFrame of flagged AS events."""
        mask = self.as_score > 0.5
        if not mask.any():
            return pd.DataFrame()
        return pd.DataFrame({
            "timestamp": self.timestamps[mask],
            "price": self.prices[mask],
            "as_score": self.as_score[mask],
            "gamma": self.gamma[mask],
            "burst": self.burst_flags[mask],
            "sigma_b": np.sqrt(self.sigma_b_sq[mask]),
            "lambda": self.lam[mask],
        })

    @property
    def calibration_summary(self) -> Dict:
        """Summary statistics of the calibration."""
        valid = ~np.isnan(self.sigma_b_sq)
        return {
            "ticker": self.ticker,
            "n_trades": len(self.timestamps),
            "n_events_flagged": int((self.as_score > 0.5).sum()),
            "pct_flagged": float((self.as_score > 0.5).mean() * 100),
            "mean_sigma_b": float(np.sqrt(np.nanmean(self.sigma_b_sq))),
            "mean_lambda": float(np.nanmean(self.lam)),
            "mean_s_j": float(np.sqrt(np.nanmean(self.s_j_sq))),
            "n_jumps_detected": int((self.gamma > 0.5).sum()),
            "n_bursts_detected": int(self.burst_flags.sum()),
            "mean_as_score": float(np.nanmean(self.as_score)),
        }


def run_as_detection(
    df: pd.DataFrame,
    ticker: str = "",
    em_window: int = 200,
    em_iterations: int = 8,
    burst_short_window: int = 10,
    burst_long_window: int = 500,
    burst_multiplier: float = 3.0,
    alpha: float = 0.7,
    gamma_threshold: float = 0.6,
    burst_threshold: float = 3.0,
) -> ASResult:
    """
    Full adverse-selection detection pipeline for one ticker.

    Parameters
    ----------
    df : DataFrame
        Must have columns: timestamp, mid_price, logit.
    ticker : str
        Contract identifier.
    em_window : int
        Rolling window size for EM calibration (in trades).
    em_iterations : int
        EM iterations per window position.
    burst_short_window : int
        Short window for arrival-rate estimation (in trades).
    burst_long_window : int
        Long window baseline for arrival rate (in trades).
    burst_multiplier : float
        Threshold for flagging arrival-rate anomalies.
    alpha : float
        Weight on gamma (vs burst) in composite score.
        AS = alpha * gamma + (1 - alpha) * burst_indicator
    gamma_threshold : float
        Gamma values above this contribute to AS score.
    burst_threshold : float
        Rate ratio above this triggers burst flag.
    """
    timestamps = pd.to_datetime(df["timestamp"].values)
    prices = df["mid_price"].values.astype(np.float64)
    logits = df["logit"].values.astype(np.float64)
    n = len(df)

    # Compute irregular time steps
    ts_epoch = timestamps.astype(np.int64) // 10**9
    dt_seconds = np.diff(ts_epoch).astype(np.float64)
    dt_seconds = np.maximum(dt_seconds, 0.1)

    # --- Signal A: Kalman filter + rolling EM ---
    x_filtered = kalman_filter_hf(logits, dt_seconds)

    sigma_b_sq, lam, s_j_sq, gamma_raw = rolling_em_calibrate(
        x_filtered, dt_seconds,
        window=em_window,
        n_em_iter=em_iterations,
    )

    # Pad gamma to match trade array length (increments are n-1)
    gamma = np.zeros(n)
    gamma[1:] = gamma_raw

    sigma_b_sq_full = np.zeros(n)
    sigma_b_sq_full[1:] = sigma_b_sq
    sigma_b_sq_full[0] = sigma_b_sq[0] if len(sigma_b_sq) > 0 else 0

    lam_full = np.zeros(n)
    lam_full[1:] = lam
    lam_full[0] = lam[0] if len(lam) > 0 else 0

    s_j_sq_full = np.zeros(n)
    s_j_sq_full[1:] = s_j_sq
    s_j_sq_full[0] = s_j_sq[0] if len(s_j_sq) > 0 else 0

    # --- Signal B: Burst detection ---
    burst_flags, rate_ratio = detect_bursts(
        timestamps,
        short_window=burst_short_window,
        long_window=burst_long_window,
        burst_multiplier=burst_multiplier,
    )

    # --- Composite score ---
    gamma_signal = np.clip((gamma - gamma_threshold) / (1 - gamma_threshold), 0, 1)
    burst_signal = (rate_ratio > burst_threshold).astype(float)
    as_score = np.clip(alpha * gamma_signal + (1 - alpha) * burst_signal, 0, 1)

    logger.info(
        "AS detection %s: %d trades, %d jumps (γ>0.5), %d bursts, "
        "%d flagged (score>0.5), mean σ_b=%.4f, mean λ=%.5f",
        ticker, n, int((gamma > 0.5).sum()), int(burst_flags.sum()),
        int((as_score > 0.5).sum()),
        float(np.sqrt(np.nanmean(sigma_b_sq))),
        float(np.nanmean(lam)),
    )

    return ASResult(
        ticker=ticker,
        timestamps=timestamps.values,
        prices=prices,
        x_filtered=x_filtered,
        gamma=gamma,
        sigma_b_sq=sigma_b_sq_full,
        lam=lam_full,
        s_j_sq=s_j_sq_full,
        burst_flags=burst_flags,
        arrival_rate_ratio=rate_ratio,
        as_score=as_score,
        alpha_weight=alpha,
        gamma_threshold=gamma_threshold,
        burst_threshold=burst_threshold,
    )


def run_all_tickers(
    hf_data: Dict[str, pd.DataFrame],
    **kwargs,
) -> Dict[str, ASResult]:
    """Run AS detection on every ticker in the dataset."""
    results = {}
    for ticker, df in hf_data.items():
        if len(df) < 50:
            logger.warning("Skipping %s — only %d trades", ticker, len(df))
            continue
        results[ticker] = run_as_detection(df, ticker=ticker, **kwargs)
    return results
