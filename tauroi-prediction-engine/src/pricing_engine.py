"""
pricing_engine.py — Pricing Models for Prediction Markets
============================================================
Two engines for pricing prediction-market contracts on Spotify listeners:

**Engine 1: JumpDiffusionPricer (legacy / binary-call mode)**
    Closed-form Black-Scholes binary call: P = e^{-rT} * N(d2).
    Best for single-artist threshold contracts ("Will X exceed 100M?").
    Retained for backward compatibility with unit tests.

**Engine 2: MonteCarloOU (Winner-Take-All mode)**
    Ornstein-Uhlenbeck (mean-reverting) Monte Carlo simulation.
    Runs 10,000 paths for ALL artists simultaneously; at expiry,
    the artist with max listeners wins.

    P(artist_i) = #{paths where i has max listeners} / n_paths

    This naturally produces probabilities that sum to 1.0 —
    no post-hoc normalization needed.

    The OU process models listener counts as "sticky":
        dX = theta * (mu - X) * dt + sigma_abs * dW

    where mu is adjusted for TikTok velocity via jump_beta:
        mu_i = spot_i * (1 + beta * velocity_i)
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
    sigma_base: float = 0.05            # base annualised vol of listener churn (conservative)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Engine 2: Ornstein-Uhlenbeck Monte Carlo (Winner-Take-All markets)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OUArtistInput:
    """Per-artist inputs for the Monte Carlo simulation."""

    name: str
    listeners: float            # current Spotify monthly listeners (spot)
    sigma: float                # annualised percentage volatility (e.g. 1.55)
    norm_velocity: float = 0.0  # TikTok velocity normalised to [0, 1]
    trend: float = 0.0          # annualised absolute drift (listeners/year)
                                # computed from recent growth + leading indicators


@dataclass
class OUArtistResult:
    """Per-artist outputs from the Monte Carlo simulation."""

    name: str
    probability: float      # win probability ∈ [0, 1], sum across artists = 1.0
    win_count: int           # raw path wins out of n_paths
    avg_final: float         # mean listener count at expiry across all paths
    p5_final: float          # 5th percentile of final listener count
    p95_final: float         # 95th percentile of final listener count
    spot: float              # input spot (for reference)
    mu: float                # long-term mean used in the simulation


class MonteCarloOU:
    """
    Ornstein-Uhlenbeck Monte Carlo simulator for Winner-Take-All markets.

    Simulates **all** artists simultaneously.  At expiry, the artist with
    the most listeners wins.  The win count across paths IS the probability
    — no normalization hack needed.

    The OU process keeps listener counts "sticky":

        X_{t+dt} = X_t + theta * (mu_i - X_t) * dt + sigma_abs_i * sqrt(dt) * Z

    where:
        theta     — mean-reversion speed (higher = stickier)
        mu_i      — long-term mean, = spot * (1 + beta * velocity)
        sigma_abs — absolute vol = sigma_pct * spot

    Parameters
    ----------
    theta : float
        Annualised mean-reversion speed.  0.1 = slow pull.
        Calibrated from lag-1 autocorrelation if available.
    n_paths : int
        Number of Monte Carlo paths (default 10,000).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        theta: float = 0.1,
        n_paths: int = 10_000,
        seed: int | None = 42,
    ) -> None:
        self.theta = theta
        self.n_paths = n_paths
        self.seed = seed

    # ── Analytical contention filter ────────────────────────────────
    # Before the expensive MC simulation, use the OU model's own
    # closed-form solution to pre-screen artists.  If an artist has
    # negligible analytical probability of beating the leader 1-on-1,
    # their paths are made deterministic (sigma → 0) so probability
    # naturally concentrates on real contenders.
    #
    # For OU process starting at X_0 with long-term mean mu:
    #   E[X(T)]   = mu + (X_0 - mu) * exp(-theta*T)
    #   Var[X(T)] = sigma_abs^2 / (2*theta) * (1 - exp(-2*theta*T))
    #
    # P(artist_i beats leader) = Φ( E[D] / sqrt(Var_i + Var_leader) )
    # where D = X_i(T) - X_leader(T).
    #
    # This is fully data-driven — no manual tuning parameters beyond
    # the natural threshold min_contention_prob (default 0.1%).

    @staticmethod
    def _analytical_contention_mask(
        spots: np.ndarray,
        mu: np.ndarray,
        sigma_abs: np.ndarray,
        theta: float,
        T: float,
        trends: np.ndarray | None = None,
        min_prob: float = 0.001,
    ) -> np.ndarray:
        """
        Return a boolean mask: True = contender, False = out of contention.

        Uses the OU closed-form to compute each artist's pairwise
        probability of beating the current leader at time T, accounting
        for both volatility AND trend (momentum drift).

        Parameters
        ----------
        trends : array or None
            Per-artist annualised absolute drift (listeners/year).
            Included in the expected value at T so that a fast-growing
            artist isn't filtered out prematurely.
        min_prob : float
            Minimum analytical P(beats leader) to remain a contender.
            0.001 (0.1%) is a conservative default — only filters
            artists with essentially zero chance.
        """
        n = len(spots)
        if trends is None:
            trends = np.zeros(n)

        # Leader = artist with highest EXPECTED value at T
        # (not just highest current spot — a surging underdog may
        # have the highest expected value despite a lower spot)
        exp_theta_T = math.exp(-theta * T)
        exp_2theta_T = math.exp(-2 * theta * T)

        # OU expected values at T (with trend drift)
        # For trending OU: E[X(T)] ≈ OU_expectation + trend * T
        E_T = mu + (spots - mu) * exp_theta_T + trends * T

        # OU variances at T (avoid division by zero if theta ≈ 0)
        if theta > 1e-8:
            var_coeff = (1.0 - exp_2theta_T) / (2.0 * theta)
        else:
            var_coeff = T  # degenerate case: pure diffusion
        Var_T = sigma_abs ** 2 * var_coeff

        # Leader by expected value at T
        leader_idx = int(np.argmax(E_T))

        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if i == leader_idx:
                continue
            mean_diff = E_T[i] - E_T[leader_idx]
            std_diff = math.sqrt(max(Var_T[i] + Var_T[leader_idx], 1e-20))
            p_beat = float(norm.cdf(mean_diff / std_diff))
            if p_beat < min_prob:
                mask[i] = False

        return mask

    def simulate_wta(
        self,
        artists: list[OUArtistInput],
        T: float,
        jump_beta: float = 0.0,
        min_contention_prob: float = 0.001,
    ) -> list[OUArtistResult]:
        """
        Run the full WTA Monte Carlo simulation.

        Parameters
        ----------
        artists : list[OUArtistInput]
            Per-artist spot, sigma, and velocity.
        T : float
            Time to expiry in years (e.g. 15/365).
        jump_beta : float
            TikTok → listener lagged correlation.  Shifts the
            long-term mean upward for viral artists.
        min_contention_prob : float
            Minimum analytical P(beats leader) for an artist to remain
            in contention.  Artists below this threshold get
            deterministic paths (sigma → 0), so their probability
            naturally flows to real contenders.
            Set to 0 to disable (all artists keep full sigma).
            Default 0.001 (0.1%) — conservative, only prunes artists
            with essentially no chance.

        Returns
        -------
        list[OUArtistResult]
            One result per artist, sorted by probability descending.
            Probabilities sum to 1.0 by construction.
        """
        n_artists = len(artists)
        n_steps = max(int(T * 365), 1)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)

        rng = np.random.default_rng(self.seed)

        # ── Extract per-artist parameter vectors ──────────────────────
        spots = np.array(
            [a.listeners for a in artists], dtype=np.float64,
        )
        sigmas_pct = np.array(
            [a.sigma for a in artists], dtype=np.float64,
        )
        velocities = np.array(
            [a.norm_velocity for a in artists], dtype=np.float64,
        )
        trends = np.array(
            [a.trend for a in artists], dtype=np.float64,
        )

        # Long-term mean: current spot adjusted for TikTok momentum
        mu = spots * (1.0 + jump_beta * velocities)

        # Convert percentage vol to absolute vol
        # sigma_abs = annualised_pct_vol * spot
        sigma_abs = sigmas_pct * spots

        # ── Analytical contention filter ─────────────────────────────
        # Use the model's own closed-form OU solution to identify
        # artists with negligible chance of winning.  Their sigma is
        # zeroed → deterministic paths → probability concentrates on
        # real contenders.  Fully data-driven, no manual tuning.
        # The filter accounts for both volatility AND trend.
        if min_contention_prob > 0:
            contender_mask = self._analytical_contention_mask(
                spots, mu, sigma_abs, self.theta, T, trends,
                min_prob=min_contention_prob,
            )
            n_contenders = int(contender_mask.sum())
            # Zero out sigma AND trend for non-contenders
            sigma_abs = np.where(contender_mask, sigma_abs, 0.0)
            trends = np.where(contender_mask, trends, 0.0)
        else:
            n_contenders = n_artists

        # ── Initialise paths: shape (n_paths, n_artists) ─────────────
        X = np.tile(spots, (self.n_paths, 1))

        # ── Simulate (trending OU process) ───────────────────────────
        # dX = theta*(mu - X)*dt + trend*dt + sigma_abs*sqrt(dt)*dW
        #
        # The trend term adds an independent directional drift on top
        # of mean-reversion.  This is critical: theta is slow (~0.055)
        # so mu-adjustment alone barely moves the needle over 12 days.
        # The trend directly encodes observed momentum (listener growth,
        # TikTok virality) as a per-artist drift rate.
        for _ in range(n_steps):
            Z = rng.standard_normal((self.n_paths, n_artists))
            drift_mr = self.theta * (mu - X) * dt
            drift_momentum = trends * dt
            diffusion = sigma_abs * sqrt_dt * Z
            X = X + drift_mr + drift_momentum + diffusion
            np.maximum(X, 0, out=X)  # listeners can't go negative

        # ── Count wins ───────────────────────────────────────────────
        winners = np.argmax(X, axis=1)                # (n_paths,)
        win_counts = np.bincount(winners, minlength=n_artists)
        probs = win_counts / self.n_paths

        # ── Build results ────────────────────────────────────────────
        results = []
        for i, artist in enumerate(artists):
            results.append(OUArtistResult(
                name=artist.name,
                probability=float(probs[i]),
                win_count=int(win_counts[i]),
                avg_final=float(X[:, i].mean()),
                p5_final=float(np.percentile(X[:, i], 5)),
                p95_final=float(np.percentile(X[:, i], 95)),
                spot=float(spots[i]),
                mu=float(mu[i]),
            ))

        # Sort by probability descending
        results.sort(key=lambda r: r.probability, reverse=True)
        return results
