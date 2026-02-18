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
    theta: float = 0.1    # OU mean-reversion speed (annualised)
    jump_intensity: float = 12.0  # annualised Poisson jump arrival rate
    jump_std: float = 0.04        # log-normal std of jump sizes
    trend: float = 0.0    # annualised growth rate (EWMA of recent log-returns)

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "sigma": self.sigma,
            "jump_beta": self.jump_beta,
            "best_lag": self.best_lag,
            "tiktok_p95": self.tiktok_p95,
            "vol_gamma": self.vol_gamma,
            "theta": self.theta,
            "jump_intensity": self.jump_intensity,
            "jump_std": self.jump_std,
            "trend": self.trend,
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

    # Minimum credible annualised volatility.  Prevents sigma from
    # collapsing to near-zero on ultra-stable artists, which would make
    # the OU simulation deterministic.  5% is a conservative floor —
    # even the stickiest superstar has ≥ 5% annualised listener churn.
    _SIGMA_FLOOR: float = 0.05

    def compute_sigma(self, window: int = 7) -> float:
        """
        Realized Volatility (sigma).

        Uses the **sample standard deviation of daily returns**, annualised.
        This is the textbook realized-vol estimator.

        The old "range-implied floor" (max-min / min / sqrt(N)) is logged
        for reference but **no longer used** — over multi-year history the
        min-to-max range reflects structural growth, not short-term vol,
        and inflates sigma by 10x+.

        A modest absolute floor of 5% prevents degenerate simulations.
        """
        listeners = self.df["spotify_monthly_listeners"].astype(float)
        pct_change = listeners.pct_change().dropna()

        if pct_change.empty:
            logger.warning("Not enough data to compute sigma — using conservative floor")
            return self._SIGMA_FLOOR  # 0.05: assume stable, not chaotic

        # ── Primary: Sample std of all daily returns ─────────────────
        sigma_daily_sample = float(pct_change.std())
        sigma_sample = sigma_daily_sample * math.sqrt(365)

        # ── Reference only: Range-implied (logged, NOT used) ─────────
        price_min = float(listeners.min())
        price_max = float(listeners.max())
        n_days = max(len(listeners) - 1, 1)

        if price_min > 0:
            total_range_pct = (price_max - price_min) / price_min
            sigma_range_daily = total_range_pct / math.sqrt(n_days)
            sigma_range = sigma_range_daily * math.sqrt(365)
        else:
            sigma_range = 0.0

        # ── Apply floor ──────────────────────────────────────────────
        sigma_annual = max(sigma_sample, self._SIGMA_FLOOR)

        logger.info(
            "sigma estimators — sample=%.4f | range=%.4f (ref only) "
            "| floor=%.4f → using %.4f",
            sigma_sample, sigma_range, self._SIGMA_FLOOR, sigma_annual,
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

    # ── theta: mean-reversion speed ────────────────────────────────────

    def compute_theta(self) -> float:
        """
        Mean-Reversion Speed (theta) for the Ornstein-Uhlenbeck model.

        Listener counts are "sticky" — an artist at 100M today is
        overwhelmingly likely to be near 100M tomorrow.  Theta quantifies
        how strongly the process pulls back toward its long-term mean.

        Method
        ------
        The raw lag-1 autocorrelation of *levels* is nearly 1.0 for any
        growing artist, making theta ≈ 0 (i.e. a random walk).  This is
        because autocorrelation on non-stationary data captures the trend,
        not the mean-reversion.

        Instead we use an OLS regression approach on **detrended** data:

        1.  Remove the linear trend from the listener series to obtain
            residuals (the "deviation from trend").
        2.  Regress residual(t) on residual(t-1):
                residual(t) = a + b * residual(t-1) + epsilon
            For a discrete OU: b = exp(-theta * dt).
        3.  Invert: theta = -ln(b) / dt   (annualised).
        4.  Clamp to [0.01, 10.0] to avoid degenerate values.

        Returns
        -------
        float
            Annualised mean-reversion speed.  Higher = stickier.
        """
        listeners = self.df["spotify_monthly_listeners"].astype(float).dropna()

        if len(listeners) < 30:
            logger.warning("Too few data points for theta — using default 0.1")
            return 0.1

        # ── Detrend: remove linear growth to isolate mean-reverting component
        x_idx = np.arange(len(listeners), dtype=float)
        coeffs = np.polyfit(x_idx, listeners.values, 1)
        trend = np.polyval(coeffs, x_idx)
        residuals = pd.Series(listeners.values - trend, index=listeners.index)

        # ── Lag-1 autocorrelation of detrended residuals
        rho = float(residuals.autocorr(lag=1))

        if np.isnan(rho) or rho <= 0 or rho >= 1.0:
            # Fallback: try raw levels (short windows may already be ~stationary)
            rho_raw = float(listeners.autocorr(lag=1))
            if not np.isnan(rho_raw) and 0 < rho_raw < 1.0:
                rho = rho_raw
            else:
                logger.info(
                    "Autocorrelation invalid (detrended rho=%.4f, raw=%.4f) "
                    "— using default 0.1",
                    rho, rho_raw if not np.isnan(rho_raw) else float("nan"),
                )
                return 0.1

        dt = 1.0 / 365.0
        theta = -math.log(rho) / dt

        # Clamp to reasonable range
        theta = float(np.clip(theta, 0.01, 10.0))

        logger.info(
            "OU theta — detrended rho=%.6f | theta_annual=%.4f "
            "(half-life ≈ %.1f days)",
            rho, theta, math.log(2) / theta * 365 if theta > 0 else float("inf"),
        )
        return theta

    # ── trend: recent growth trajectory ────────────────────────────────

    def compute_trend(self, window: int = 7) -> float:
        """
        Recent Growth Trajectory (annualised).

        Uses an **exponentially-weighted mean** of daily log-returns
        over the last ``window`` days.  EWMA gives more weight to the
        most recent days, capturing acceleration/deceleration.

        This is the single most important input for the trend-projected
        OU model: it tells the simulation *where the artist is heading*,
        not just where they are now.

        Returns
        -------
        float
            Annualised growth rate.  Positive = gaining listeners.
            Clamped to [-2.0, 2.0] to prevent extrapolation artefacts.
        """
        listeners = self.df["spotify_monthly_listeners"].astype(float).dropna()

        if len(listeners) < 3:
            return 0.0

        # Use the most recent `window+1` observations
        recent = listeners.tail(window + 1)
        log_rets = np.log(recent / recent.shift(1)).dropna()
        log_rets = log_rets[np.isfinite(log_rets)]

        if len(log_rets) < 2:
            return 0.0

        # EWMA with span = window gives ~86% weight to last `window` days
        ewma_daily = float(log_rets.ewm(span=window).mean().iloc[-1])
        trend_annual = ewma_daily * 365

        # Clamp to prevent absurd extrapolation
        trend_annual = float(np.clip(trend_annual, -2.0, 2.0))

        logger.info(
            "Trend — EWMA(7d) daily=%.6f | annualised=%.4f (%.2f%%/day)",
            ewma_daily, trend_annual, ewma_daily * 100,
        )
        return trend_annual

    # ── jump calibration ────────────────────────────────────────────────

    def compute_jump_params(self) -> tuple[float, float]:
        """
        Calibrate jump intensity (λ) and jump size (σ_J) from historical data.

        Method
        ------
        1.  Compute daily log-returns of ``spotify_monthly_listeners``.
        2.  Estimate "normal" diffusion volatility via the **median absolute
            deviation** (MAD) — robust to jump contamination.
        3.  Flag any day where |log-return| > 3 × MAD-sigma as a "jump day".
        4.  Count jump days → annualise → ``jump_intensity``.
        5.  Measure the std of detected jump log-returns → ``jump_std``.

        Returns
        -------
        (jump_intensity, jump_std) : tuple[float, float]
            jump_intensity: annualised Poisson arrival rate.
            jump_std: log-normal std of jump sizes.
        """
        listeners = self.df["spotify_monthly_listeners"].astype(float).dropna()

        if len(listeners) < 30:
            logger.warning("Too few data points for jump calibration — using defaults")
            return 12.0, 0.04

        # Daily log returns (filter out zero/negative for safety)
        valid = listeners[listeners > 0]
        log_returns = np.log(valid / valid.shift(1)).dropna()
        log_returns = log_returns[np.isfinite(log_returns)]

        if len(log_returns) < 20:
            logger.warning("Too few valid log returns — using default jump params")
            return 12.0, 0.04

        # Robust volatility via MAD (resistant to jump contamination)
        median_ret = float(np.median(log_returns))
        mad = float(np.median(np.abs(log_returns - median_ret)))
        sigma_robust = mad * 1.4826  # MAD → σ conversion for normal dist

        if sigma_robust <= 0:
            logger.warning("MAD-sigma is zero — using default jump params")
            return 12.0, 0.04

        # Flag jumps: |log-return| > 3 × robust_sigma
        threshold = 3.0 * sigma_robust
        jump_mask = np.abs(log_returns) > threshold
        jump_returns = log_returns[jump_mask]
        n_jumps = int(jump_mask.sum())
        n_days = len(log_returns)

        # Annualise jump intensity
        if n_days > 0 and n_jumps > 0:
            daily_intensity = n_jumps / n_days
            jump_intensity = daily_intensity * 365.0
        else:
            jump_intensity = 6.0  # mild default: ~0.5/month

        # Jump std: std of detected jump absolute log-returns
        if n_jumps >= 3:
            jump_std = float(np.std(np.abs(jump_returns)))
        elif n_jumps > 0:
            jump_std = float(np.mean(np.abs(jump_returns)))
        else:
            # No detected jumps — use 2× normal vol as a prior
            jump_std = sigma_robust * 2.0

        # Clamp to reasonable ranges
        jump_intensity = float(np.clip(jump_intensity, 1.0, 120.0))
        jump_std = float(np.clip(jump_std, 0.005, 0.30))

        logger.info(
            "Jump calibration — %d jumps in %d days (%.1f/yr) | "
            "σ_robust=%.6f | threshold=%.6f | jump_std=%.4f",
            n_jumps, n_days, jump_intensity,
            sigma_robust, threshold, jump_std,
        )
        return jump_intensity, jump_std

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

        theta = self.compute_theta()

        jump_intensity, jump_std = self.compute_jump_params()

        trend = self.compute_trend()

        result = CalibrationResult(
            sigma=sigma,
            jump_beta=jump_beta,
            best_lag=best_lag,
            tiktok_p95=tiktok_p95,
            vol_gamma=vol_gamma,
            theta=theta,
            jump_intensity=jump_intensity,
            jump_std=jump_std,
            trend=trend,
        )

        logger.info("Calibration complete: %s", result.to_dict())
        return result
