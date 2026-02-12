"""
calibration.py — Artist-Specific Model Calibrator
====================================================
Reads the processed ``model_ready.csv`` and derives the empirical parameters
that feed the Jump-Diffusion pricing engine:

    sigma   — annualised base volatility of Spotify monthly listeners
    jump_beta — lagged correlation between TikTok activity and listener growth
    best_lag  — the lag (1, 2, or 3 days) that maximises jump_beta
    tiktok_p95 — 95th-percentile of tiktok_sound_posts_change (normalisation cap)
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("tauroi.calibrator")


# ── calibration result ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationResult:
    """Immutable container for calibrated model parameters."""

    sigma: float          # annualised base volatility
    jump_beta: float      # best lagged correlation (TikTok → listeners)
    best_lag: int         # optimal lag in days (1, 2, or 3)
    tiktok_p95: float     # 95th percentile of tiktok_sound_posts_change
    vol_gamma: float      # conditional vol sensitivity (TikTok → vol)

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "sigma": self.sigma,
            "jump_beta": self.jump_beta,
            "best_lag": self.best_lag,
            "tiktok_p95": self.tiktok_p95,
            "vol_gamma": self.vol_gamma,
        }


# ── calibrator ──────────────────────────────────────────────────────────────

class Calibrator:
    """
    Derive empirical pricing parameters from processed Chartmetric data.

    Parameters
    ----------
    csv_path : path to ``model_ready.csv``

    Example
    -------
    >>> cal = Calibrator("data/processed/model_ready.csv")
    >>> result = cal.run()
    >>> result.sigma
    0.42...
    """

    def __init__(self, csv_path: str | pathlib.Path) -> None:
        self.csv_path = pathlib.Path(csv_path)
        self.df = self._load()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Calibrator":
        """
        Create a Calibrator from an in-memory DataFrame (for backtesting).

        Bypasses file I/O — useful when the backtest needs to re-calibrate
        on rolling windows without writing temporary CSVs.
        """
        instance = object.__new__(cls)
        instance.csv_path = pathlib.Path("(in-memory)")
        instance.df = df.sort_values("Date").reset_index(drop=True)

        required = {"spotify_monthly_listeners", "tiktok_sound_posts_change"}
        missing = required - set(instance.df.columns)
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        return instance

    # ── data loading ────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        """Load the processed CSV and validate required columns."""
        df = pd.read_csv(self.csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        required = {"spotify_monthly_listeners", "tiktok_sound_posts_change"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        logger.info(
            "Loaded %d rows (%s → %s)",
            len(df),
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d"),
        )
        return df

    # ── sigma: base volatility ──────────────────────────────────────────

    def compute_sigma(self, window: int = 7) -> float:
        """
        Realized Volatility (sigma).

        Computes annualised volatility via two independent estimators and
        takes the **maximum** — the more conservative (wider distribution)
        estimate is always safer for a pricing model.

        **Estimator 1 — Sample Std of Returns:**
            Standard deviation of ALL daily percent-changes, annualised.
            This is the textbook realized-vol estimator and naturally
            captures fat tails (e.g. the +7.9% single-day jump on Feb 10).

        **Estimator 2 — Range-Implied Vol Floor:**
            ``(max - min) / min / sqrt(N_days)`` annualised.
            If the price moved 23% over 91 days, vol cannot be lower than
            what that range implies.  Acts as a floor to prevent the model
            from ever producing "0% probability" for moves that clearly
            happen in the data.

        The old method (mean of 7-day rolling stds) is logged for reference
        but no longer used as the primary estimator — it averages away the
        tails that matter most for jump-diffusion pricing.
        """
        listeners = self.df["spotify_monthly_listeners"].astype(float)
        pct_change = listeners.pct_change().dropna()

        if pct_change.empty:
            logger.warning("Not enough data to compute sigma")
            return 0.20  # sensible fallback

        # ── Estimator 1: Sample std of all daily returns ─────────────
        sigma_daily_sample = float(pct_change.std())
        sigma_sample = sigma_daily_sample * math.sqrt(365)

        # ── Estimator 2: Range-implied vol floor ─────────────────────
        price_min = float(listeners.min())
        price_max = float(listeners.max())
        n_days = max(len(listeners) - 1, 1)

        if price_min > 0:
            total_range_pct = (price_max - price_min) / price_min
            sigma_range_daily = total_range_pct / math.sqrt(n_days)
            sigma_range = sigma_range_daily * math.sqrt(365)
        else:
            sigma_range = 0.0

        # ── Old method (logged for comparison) ───────────────────────
        rolling_std = pct_change.rolling(window=window).std().dropna()
        sigma_rolling_mean = (
            float(rolling_std.mean()) * math.sqrt(365)
            if not rolling_std.empty else 0.0
        )

        # ── Take the max ─────────────────────────────────────────────
        sigma_annual = max(sigma_sample, sigma_range)

        logger.info(
            "sigma estimators — sample=%.4f | range_floor=%.4f | "
            "rolling_mean=%.4f (old) → using %.4f",
            sigma_sample, sigma_range, sigma_rolling_mean, sigma_annual,
        )
        return sigma_annual

    # ── jump_beta: TikTok → listener correlation ────────────────────────

    def compute_jump_beta(self, lags: list[int] | None = None) -> tuple[float, int]:
        """
        Jump Sensitivity (beta).

        Correlate ``tiktok_sound_posts_change`` (lagged by 1, 2, 3 days)
        against the daily percent-change of ``spotify_monthly_listeners``.

        Returns ``(best_correlation, best_lag)``.
        """
        if lags is None:
            lags = [1, 2, 3]

        listeners_pct = (
            self.df["spotify_monthly_listeners"]
            .astype(float)
            .pct_change()
        )
        tiktok = self.df["tiktok_sound_posts_change"].astype(float)

        best_corr = 0.0
        best_lag = lags[0]

        for lag in lags:
            tiktok_lagged = tiktok.shift(lag)
            # Build a clean frame (drop NaN rows for this pair)
            pair = pd.DataFrame({
                "listeners_pct": listeners_pct,
                "tiktok_lagged": tiktok_lagged,
            }).dropna()

            if len(pair) < 5:
                logger.warning("Lag %d: too few data points (%d)", lag, len(pair))
                continue

            corr = float(pair["listeners_pct"].corr(pair["tiktok_lagged"]))
            logger.info("Lag %d: correlation = %.6f", lag, corr)

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        logger.info(
            "Best jump_beta = %.6f at lag = %d days",
            best_corr,
            best_lag,
        )
        return best_corr, best_lag

    # ── vol_gamma: conditional volatility sensitivity ───────────────────

    def compute_vol_gamma(self) -> float:
        """
        Conditional Volatility Sensitivity (gamma).

        Quantifies **how much** TikTok activity inflates the volatility
        of Spotify listener changes.  Virality is a spectrum: a "warm"
        trend (velocity = 0.7) should widen the uncertainty band more
        than a dead period (velocity = 0.1), but less than mega-viral
        (velocity = 1.0).

        Method
        ------
        1.  Compute squared daily returns of ``spotify_monthly_listeners``
            — these are a non-parametric **variance proxy** (no windowing
            or smoothing needed).

        2.  Normalise ``tiktok_sound_posts_change`` to [0, 1] using the
            P95 cap, then lag by 1 day (today's TikTok → tomorrow's vol).

        3.  OLS regression:  ``r_t^2 = alpha + beta * velocity_{t-1}``

            *  ``alpha`` ≈ baseline daily variance (when TikTok is quiet).
            *  ``beta``  ≈ additional daily variance per unit velocity.

        4.  Linearise to vol-space:

            ``sigma(v) = sqrt(alpha + beta*v)``
            ``        ≈ sqrt(alpha) + beta / (2*sqrt(alpha)) * v``

            So ``gamma_daily = beta / (2 * sqrt(alpha))``

        5.  Annualise:  ``gamma = gamma_daily * sqrt(365)``

        The result plugs directly into the pricing engine:

            ``sigma_adj = sigma_base + gamma * normalised_velocity``

        Returns
        -------
        float
            Annualised vol-gamma (additional annualised vol at full velocity).
        """
        listeners = self.df["spotify_monthly_listeners"].astype(float)
        daily_return = listeners.pct_change().dropna()
        squared_return = daily_return ** 2  # variance proxy

        tiktok = self.df["tiktok_sound_posts_change"].astype(float).clip(lower=0)
        p95 = float(tiktok.quantile(0.95))
        if p95 <= 0:
            logger.warning("TikTok P95 is 0 — cannot compute vol_gamma")
            return 0.0
        norm_velocity = (tiktok / p95).clip(upper=1.0)

        # Lag velocity by 1 day (today's TikTok → tomorrow's vol)
        velocity_lagged = norm_velocity.shift(1)

        pair = pd.DataFrame({
            "var_proxy": squared_return,
            "velocity": velocity_lagged,
        }).dropna()

        if len(pair) < 10:
            logger.warning("Too few data points (%d) to compute vol_gamma", len(pair))
            return 0.0

        # ── OLS: var_proxy = alpha + beta * velocity ─────────────────
        x = pair["velocity"].values
        y = pair["var_proxy"].values
        x_mean, y_mean = x.mean(), y.mean()

        ss_xx = float(np.sum((x - x_mean) ** 2))
        if ss_xx == 0:
            return 0.0

        beta = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
        alpha = y_mean - beta * x_mean

        # ── Linearise from variance-space to vol-space ───────────────
        # sigma(v) ≈ sqrt(alpha) + beta/(2*sqrt(alpha)) * v
        # so gamma_daily = beta / (2 * sqrt(alpha))
        if alpha <= 0:
            # Fallback: use sample variance as alpha
            alpha = float(daily_return.std() ** 2)

        if alpha > 0:
            gamma_daily = beta / (2 * math.sqrt(alpha))
        else:
            gamma_daily = 0.0

        # Floor at 0 — TikTok activity should never *decrease* vol
        gamma_daily = max(gamma_daily, 0.0)
        gamma_annual = gamma_daily * math.sqrt(365)

        # Diagnostics
        corr = float(pair["var_proxy"].corr(pair["velocity"]))
        logger.info(
            "Vol gamma — OLS: alpha=%.2e, beta=%.2e | "
            "corr(r^2, velocity)=%.4f | gamma_daily=%.6f | gamma_annual=%.4f",
            alpha, beta, corr, gamma_daily, gamma_annual,
        )
        return gamma_annual

    # ── normalisation stats ─────────────────────────────────────────────

    def compute_tiktok_p95(self) -> float:
        """
        Return the 95th percentile of ``tiktok_sound_posts_change``.

        Used as the saturation cap when normalising raw TikTok counts
        into the [0, 1] velocity score.
        """
        positive_only = self.df["tiktok_sound_posts_change"].clip(lower=0)
        p95 = float(positive_only.quantile(0.95))
        logger.info("TikTok P95 normalisation cap = %s", f"{p95:,.0f}")
        return p95

    # ── orchestrator ────────────────────────────────────────────────────

    def run(self) -> CalibrationResult:
        """
        Run the full calibration pipeline and return a CalibrationResult.
        """
        logger.info("Running calibration on %s", self.csv_path.name)

        sigma = self.compute_sigma()
        jump_beta, best_lag = self.compute_jump_beta()
        tiktok_p95 = self.compute_tiktok_p95()

        raw_gamma = self.compute_vol_gamma()

        # ── Parameter Shrinkage ──────────────────────────────────
        # Raw OLS on ~90 days overfits to recent spikes.  Damping
        # factor of 0.5 shrinks gamma toward zero — a Bayesian
        # "meet-in-the-middle" between the data and our prior that
        # vol should stay moderate for a global superstar.
        DAMPING_FACTOR = 0.5
        vol_gamma = raw_gamma * DAMPING_FACTOR
        logger.info(
            "Shrinkage Applied: Raw Gamma %.2f%% -> Damped Gamma %.2f%%",
            raw_gamma * 100, vol_gamma * 100,
        )

        result = CalibrationResult(
            sigma=sigma,
            jump_beta=jump_beta,
            best_lag=best_lag,
            tiktok_p95=tiktok_p95,
            vol_gamma=vol_gamma,
        )

        logger.info("Calibration complete: %s", result.to_dict())
        return result
