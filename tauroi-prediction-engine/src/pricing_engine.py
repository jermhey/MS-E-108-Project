"""
pricing_engine.py — Modified Merton Jump-Diffusion Pricing Model
=================================================================
Prices a **binary call option** on a prediction-market settlement value
(e.g. "Will Bad Bunny exceed 100 M monthly listeners by March 1?").

Two operating modes:

1.  **Default (Phase 1)** — uses hardcoded ``ModelParams`` and an additive
    nowcast:  ``S = S_official + velocity * correlation_factor``

2.  **Calibrated (Phase 3)** — accepts empirical ``sigma`` and ``jump_beta``
    from the ``Calibrator`` and uses a multiplicative nowcast:
        ``S_adj = S_current * (1 + normalised_velocity * jump_beta)``

The fair value of the binary call is always:

    P = e^{-rT} * N(d_2)

where

    d_2 = [ln(S / K) + (r - 0.5 * sigma_adj^2) * T] / (sigma_adj * sqrt(T))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ── tunable model parameters ───────────────────────────────────────────────

@dataclass
class ModelParams:
    """Encapsulates every knob the quant team can tune."""

    risk_free_rate: float = 0.05        # r  — annualised risk-free rate
    time_to_expiry: float = 1 / 365     # T  — default 1 day (fraction of year)
    sigma_base: float = 0.20            # base annualised vol of listener churn
    jump_threshold: float = 0.80        # TikTok velocity that triggers jump
    jump_multiplier: float = 3.0        # vol multiplier under jump regime
    correlation_factor: float = 3_000_000  # maps velocity to listener delta


# ── core engine ─────────────────────────────────────────────────────────────

class JumpDiffusionPricer:
    """
    Prices binary call options using the Modified Merton Jump-Diffusion Model.

    Supports both default (hardcoded) and calibrated (data-driven) modes.
    """

    def __init__(self, params: ModelParams | None = None) -> None:
        self.params = params or ModelParams()

    # ──────────────────────────────────────────────────────────────────────
    #  Phase 1 — default methods (backward-compatible with existing tests)
    # ──────────────────────────────────────────────────────────────────────

    def nowcast_spot(
        self,
        spot_official: float,
        tiktok_velocity: float,
    ) -> float:
        """
        Additive nowcast (Phase 1).

            S_now = Spotify_Last_Official + (TikTok_Velocity * Correlation_Factor)
        """
        return spot_official + (tiktok_velocity * self.params.correlation_factor)

    def adjusted_sigma(self, tiktok_velocity: float) -> float:
        """
        Return the volatility parameter, regime-switched on jump detection.

            if velocity > threshold:  sigma_adj = sigma_base * jump_multiplier
            else:                     sigma_adj = sigma_base
        """
        if tiktok_velocity > self.params.jump_threshold:
            return self.params.sigma_base * self.params.jump_multiplier
        return self.params.sigma_base

    def d2(
        self,
        spot: float,
        strike: float,
        sigma: float,
        r: float | None = None,
        T: float | None = None,
    ) -> float:
        """
        Compute d2 of the modified Black-Scholes formula.

            d2 = [ln(S/K) + (r - 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        """
        r = r if r is not None else self.params.risk_free_rate
        T = T if T is not None else self.params.time_to_expiry

        if T <= 0:
            raise ValueError(f"Time to expiry must be positive, got {T}")
        if spot <= 0 or strike <= 0:
            raise ValueError("Spot and strike must be positive")

        numerator = math.log(spot / strike) + (r - 0.5 * sigma ** 2) * T
        denominator = sigma * math.sqrt(T)
        return numerator / denominator

    def fair_value(
        self,
        spot_official: float,
        tiktok_velocity: float,
        strike: float,
        r: float | None = None,
        T: float | None = None,
    ) -> float:
        """
        End-to-end pricing (Phase 1 defaults).

            Fair_Price = e^{-rT} * N(d2)

        Returns a probability in [0, 1].
        """
        r = r if r is not None else self.params.risk_free_rate
        T = T if T is not None else self.params.time_to_expiry

        S = self.nowcast_spot(spot_official, tiktok_velocity)
        sigma = self.adjusted_sigma(tiktok_velocity)
        d2_val = self.d2(S, strike, sigma, r, T)

        discount = math.exp(-r * T)
        return float(discount * norm.cdf(d2_val))

    def compute_edge(
        self,
        spot_official: float,
        tiktok_velocity: float,
        strike: float,
        market_price: float,
    ) -> float:
        """
        Edge = Fair_Value − Market_Price.

        Positive edge => market is cheap (BUY signal).
        Negative edge => market is rich  (SELL signal).
        """
        fv = self.fair_value(spot_official, tiktok_velocity, strike)
        return fv - market_price

    # ──────────────────────────────────────────────────────────────────────
    #  Phase 3 — calibrated methods (data-driven sigma & jump_beta)
    # ──────────────────────────────────────────────────────────────────────

    def nowcast_spot_calibrated(
        self,
        spot_official: float,
        normalized_velocity: float,
        jump_beta: float,
    ) -> float:
        """
        Multiplicative nowcast using empirical jump sensitivity.

            S_adj = S_current * (1 + normalised_tiktok_velocity * jump_beta)

        If TikTok is going viral today, the "True" spot price is higher
        than what the stale Spotify number shows.
        """
        return spot_official * (1.0 + normalized_velocity * jump_beta)

    def adjusted_sigma_calibrated(
        self,
        sigma_base: float,
        normalized_velocity: float,
        vol_gamma: float = 0.0,
        event_impact_score: float = 1.0,
    ) -> float:
        """
        Three-layer conditional volatility.

        **Layer 1 — Continuous TikTok boost:**

            sigma_cond = sigma_base + (vol_gamma * normalised_velocity)

        **Layer 2 — Event-driven multiplier:**

            sigma_event = sigma_cond * event_impact_score

        where ``event_impact_score`` is 1.0 (quiet), 1.1 (concert),
        2.0 (award show / Super Bowl), or 3.0 (album drop).  This
        forces the model to price higher *before* the viral spike
        even occurs — capturing the "implied volatility" of the
        upcoming event.

        **Layer 3 — Safety rails:**

            floor at sigma_base, cap at 1.0 (100% annualised)
        """
        sigma_cond = sigma_base + vol_gamma * normalized_velocity
        sigma_event = sigma_cond * event_impact_score
        sigma_event = max(sigma_event, sigma_base)   # floor
        sigma_event = min(sigma_event, 1.0)           # cap at 100%
        return sigma_event

    def fair_value_calibrated(
        self,
        spot_official: float,
        normalized_velocity: float,
        strike: float,
        sigma: float,
        jump_beta: float,
        vol_gamma: float = 0.0,
        event_impact_score: float = 1.0,
        r: float | None = None,
        T: float | None = None,
    ) -> float:
        """
        End-to-end calibrated pricing.

        Parameters
        ----------
        spot_official : float
            Latest official Spotify monthly listener count.
        normalized_velocity : float
            TikTok velocity normalised to [0, 1].
        strike : float
            Binary option strike (e.g. 100_000_000).
        sigma : float
            Annualised base volatility from Calibrator.
        jump_beta : float
            Lagged TikTok→Listener correlation from Calibrator.
        vol_gamma : float
            Conditional vol sensitivity from Calibrator.
        event_impact_score : float
            Event-driven vol multiplier (1.0 = quiet, up to 3.0).
        r : float, optional
            Risk-free rate override.
        T : float, optional
            Time-to-expiry override (fraction of year).

        Returns
        -------
        float
            Fair value probability ∈ [0, 1].
        """
        r = r if r is not None else self.params.risk_free_rate
        T = T if T is not None else self.params.time_to_expiry

        S = self.nowcast_spot_calibrated(spot_official, normalized_velocity, jump_beta)
        sigma_adj = self.adjusted_sigma_calibrated(
            sigma, normalized_velocity, vol_gamma, event_impact_score,
        )
        d2_val = self.d2(S, strike, sigma_adj, r, T)

        discount = math.exp(-r * T)
        return float(discount * norm.cdf(d2_val))

    def compute_edge_calibrated(
        self,
        spot_official: float,
        normalized_velocity: float,
        strike: float,
        sigma: float,
        jump_beta: float,
        market_price: float,
        r: float | None = None,
        T: float | None = None,
    ) -> float:
        """
        Calibrated edge = Fair_Value_Calibrated − Market_Price.
        """
        fv = self.fair_value_calibrated(
            spot_official, normalized_velocity, strike,
            sigma, jump_beta, r, T,
        )
        return fv - market_price

    # ──────────────────────────────────────────────────────────────────────
    #  Implied Volatility Solver
    # ──────────────────────────────────────────────────────────────────────

    def implied_vol(
        self,
        market_price: float,
        spot: float,
        strike: float,
        r: float | None = None,
        T: float | None = None,
    ) -> float | None:
        """
        Back out the implied volatility from an observed market price.

        Given the binary call formula  P = e^{-rT} * N(d2),
        find sigma such that the model price equals ``market_price``.

        **Key insight for binary options:**  Unlike vanilla options, the
        binary call price is NOT monotonic in sigma.  It has a single
        peak at ``sigma* = sqrt(-2(ln(S/K) + rT) / T)`` when the option
        is out-of-the-money.  We search the *lower* branch [lo, sigma*]
        because the lower-vol solution represents the minimum uncertainty
        needed to justify the market price — a more informative answer.

        Returns
        -------
        float or None
            Annualised implied vol, or None if no solution exists.
        """
        r = r if r is not None else self.params.risk_free_rate
        T = T if T is not None else self.params.time_to_expiry

        if market_price <= 0 or market_price >= 1:
            return None
        if spot <= 0 or strike <= 0 or T <= 0:
            return None

        discount = math.exp(-r * T)
        sqrt_T = math.sqrt(T)

        def _model_price(sigma: float) -> float:
            d2_val = self.d2(spot, strike, sigma, r, T)
            return discount * norm.cdf(d2_val)

        def _objective(sigma: float) -> float:
            return _model_price(sigma) - market_price

        lo = 0.01

        # ── Determine the search upper bound ────────────────────────
        log_moneyness = math.log(spot / strike)

        if log_moneyness >= 0:
            # In-the-money: price decreases monotonically with sigma.
            # Simple search from low vol to high vol.
            hi = 20.0
        else:
            # Out-of-the-money: price has a single peak at sigma*.
            # sigma* = sqrt(-2(ln(S/K) + rT) / T)
            sigma_peak_sq = -2 * (log_moneyness + r * T) / T
            if sigma_peak_sq <= 0:
                hi = 20.0
            else:
                sigma_peak = math.sqrt(sigma_peak_sq)

                # Check that the max achievable price exceeds the market
                d2_peak = -sigma_peak * sqrt_T
                max_price = discount * norm.cdf(d2_peak)
                if max_price < market_price:
                    return None  # market price exceeds theoretical max

                hi = sigma_peak  # search the lower (rising) branch

        try:
            iv = brentq(_objective, lo, hi, xtol=1e-6, maxiter=200)
            return float(iv)
        except ValueError:
            return None
