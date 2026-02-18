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
    jump_intensity: float | None = None  # per-artist Poisson rate (None = global)
    jump_std: float | None = None        # per-artist log-normal std (None = global)
    event_impact_score: float = 1.0      # scales jump params (1.0=quiet, 3.0=album)
    trend: float = 0.0         # annualised growth rate from recent history (EWMA)
    momentum: float = 0.0      # composite cross-platform momentum score [-1, 1]


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
    Ornstein-Uhlenbeck Monte Carlo simulator for Winner-Take-All markets,
    extended with **Merton-style jump diffusion** and **Laplace smoothing**.

    Simulates **all** artists simultaneously.  At expiry, the artist with
    the most listeners wins.

    The OU-Jump process:

        dX = theta * (mu - X) * dt + sigma_abs * dW + X * J * dN

    where:
        theta      — mean-reversion speed (higher = stickier)
        mu_i       — long-term mean, = spot * (1 + beta * velocity)
        sigma_abs  — absolute vol = sigma_pct * X (level-proportional)
        dN         — Poisson jump arrival (intensity = jump_intensity)
        J          — log-normal jump size (mean 0, std = jump_std)

    The jump component models discrete shocks (album drops, viral events,
    playlist additions) that the continuous diffusion cannot capture.
    Without jumps, artists far behind the leader get exactly 0% —
    mathematically correct but ignores tail risk.

    **Laplace smoothing** applies a Bayesian pseudocount to prevent any
    artist from receiving exactly 0% or 100%, reflecting genuine model
    uncertainty.

    Parameters
    ----------
    theta : float
        Annualised mean-reversion speed.  0.1 = slow pull.
    n_paths : int
        Number of Monte Carlo paths (default 10,000).
    seed : int or None
        Random seed for reproducibility.
    jump_intensity : float
        Annualised Poisson arrival rate for jumps (default 12.0 =
        ~1 jump per month per artist).  Set to 0 to disable jumps.
    jump_std : float
        Log-normal standard deviation of jump sizes (default 0.04 =
        typical jump of ±4-8% of current level).
    laplace_alpha : float
        Pseudocount for Laplace smoothing (default 25).  Each artist
        gets ``alpha`` phantom wins added, ensuring a probability
        floor of roughly ``alpha / (n_paths + n_artists * alpha)``.
        Set to 0 to disable smoothing.
    """

    def __init__(
        self,
        theta: float = 0.1,
        n_paths: int = 10_000,
        seed: int | None = None,
        jump_intensity: float = 12.0,
        jump_std: float = 0.04,
        laplace_alpha: float = 25.0,
    ) -> None:
        self.theta = theta
        self.n_paths = n_paths
        self.seed = seed
        self.jump_intensity = jump_intensity
        self.jump_std = jump_std
        self.laplace_alpha = laplace_alpha

    def simulate_wta(
        self,
        artists: list[OUArtistInput],
        T: float,
        jump_beta: float = 0.0,
    ) -> list[OUArtistResult]:
        """
        Run the full WTA Monte Carlo simulation with jump diffusion,
        **trend-projected drift**, **competition-aware volatility**,
        and **cross-artist correlation**.

        Key improvements over the snapshot-only model:

        1.  **Trend-projected OU mean**: Instead of mean-reverting to
            today's spot, the OU target is the trend-projected level
            at expiry: ``mu_i = spot_i * exp(trend_i * T)``.
            This gives the model "memory" of recent trajectory.

        2.  **Competition-aware volatility**: When the gap between
            the top-2 artists is < 5%, sigma is boosted for all
            artists, reflecting the genuine uncertainty in tight races.
            Boost factor ranges from 1.0 (at 5% gap) to 3.0 (at 0%).

        3.  **Cross-artist correlation**: The top-3 contenders (by
            current listeners) have negatively correlated Brownian
            shocks (rho = -0.15), modelling the zero-sum nature of
            playlist slots and listener attention.

        Parameters
        ----------
        artists : list[OUArtistInput]
            Per-artist spot, sigma, velocity, and trend.
        T : float
            Time to expiry in years (e.g. 15/365).
        jump_beta : float
            TikTok → listener lagged correlation.

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
        momentums = np.array(
            [a.momentum for a in artists], dtype=np.float64,
        )

        # ── Multi-factor drift: trend + TikTok velocity + momentum ───
        # The momentum score ([-1, 1]) captures cross-platform signals
        # that aren't in the listener trend: spotify_popularity changes,
        # follower acceleration, wikipedia spikes, playlist reach, etc.
        #
        # MOMENTUM_SCALE controls the max drift boost from momentum.
        # 0.03 = ±3% terminal value adjustment for max momentum.
        MOMENTUM_SCALE = 0.03

        trend_projection = np.exp(trends * T)
        momentum_boost = 1.0 + momentums * MOMENTUM_SCALE
        mu = spots * trend_projection * (1.0 + jump_beta * velocities) * momentum_boost

        # ── Momentum-aware volatility ────────────────────────────────
        # Strong momentum (positive or negative) means the artist is in
        # flux — sigma should be slightly higher.  This prevents the
        # model from being overconfident about trending artists.
        MOMENTUM_VOL_BOOST = 0.5  # ±50% vol boost for max |momentum|
        vol_factor = 1.0 + MOMENTUM_VOL_BOOST * np.abs(momentums)
        sigmas_pct = sigmas_pct * vol_factor

        # ── Competition-aware adjustments ────────────────────────────
        # When the top-2 artists are within 8% of each other, two
        # adjustments fire:
        #
        # 1. VOLATILITY BOOST: sigma is scaled up so the diffusion
        #    cone properly reflects the uncertainty in tight races.
        #    Boost factor: 1.0x at 8% gap → 4.0x at 0% gap.
        #
        # 2. THETA REDUCTION: mean-reversion is weakened so the OU
        #    process becomes more random-walk-like.  A tight race is
        #    inherently unstable — the "stickiness" assumption that
        #    works for individual artists is wrong when two are
        #    neck-and-neck.
        #    Theta factor: 1.0x at 8% gap → 0.1x at 0% gap.
        theta_eff = self.theta
        if n_artists >= 2:
            sorted_spots = np.sort(spots)[::-1]
            gap_pct = (
                (sorted_spots[0] - sorted_spots[1]) / sorted_spots[0]
                if sorted_spots[0] > 0 else 1.0
            )
            COMPETITION_THRESHOLD = 0.08  # 8% gap
            if gap_pct < COMPETITION_THRESHOLD:
                closeness = (
                    COMPETITION_THRESHOLD - gap_pct
                ) / COMPETITION_THRESHOLD  # 0 at threshold, 1 at 0% gap

                # Sigma boost: 1.0x → 4.0x
                sigma_boost = 1.0 + 3.0 * closeness
                sigmas_pct = sigmas_pct * sigma_boost

                # Theta reduction: 1.0x → 0.1x (near random walk)
                theta_eff = self.theta * max(1.0 - 0.9 * closeness, 0.1)

        # ── Cross-artist correlation matrix ──────────────────────────
        # Top contenders compete for the same playlist slots and
        # listener attention — when one gains, another tends to lose.
        # Model this with modest negative correlation (rho = -0.15)
        # between the top-3 contenders.
        CROSS_CORR = -0.15
        corr = np.eye(n_artists)
        if n_artists >= 2:
            sorted_idx = np.argsort(-spots)
            n_top = min(3, n_artists)
            for i in range(n_top):
                for j in range(i + 1, n_top):
                    corr[sorted_idx[i], sorted_idx[j]] = CROSS_CORR
                    corr[sorted_idx[j], sorted_idx[i]] = CROSS_CORR

        # Cholesky factorization for correlated draws
        try:
            L = np.linalg.cholesky(corr)
            use_corr = True
        except np.linalg.LinAlgError:
            use_corr = False

        # ── Per-artist jump parameters ───────────────────────────────
        per_artist_intensity = np.array([
            (a.jump_intensity if a.jump_intensity is not None
             else self.jump_intensity) * a.event_impact_score
            for a in artists
        ], dtype=np.float64)

        per_artist_jump_std = np.array([
            (a.jump_std if a.jump_std is not None
             else self.jump_std)
            * (1.0 + 0.5 * (a.event_impact_score - 1.0))
            for a in artists
        ], dtype=np.float64)

        jump_prob_per_step = per_artist_intensity * dt
        use_jumps = bool(np.any(per_artist_intensity > 0)
                         and np.any(per_artist_jump_std > 0))

        jump_prob_bcast = jump_prob_per_step[np.newaxis, :]
        jump_std_bcast = per_artist_jump_std[np.newaxis, :]

        # ── Initialise paths: shape (n_paths, n_artists) ─────────────
        X = np.tile(spots, (self.n_paths, 1))

        # ── Simulate ─────────────────────────────────────────────────
        for _ in range(n_steps):
            Z_raw = rng.standard_normal((self.n_paths, n_artists))

            # Apply cross-correlation via Cholesky transform
            Z = Z_raw @ L.T if use_corr else Z_raw

            drift = theta_eff * (mu - X) * dt
            sigma_abs = sigmas_pct * X  # level-proportional volatility
            diffusion = sigma_abs * sqrt_dt * Z
            X = X + drift + diffusion

            # ── Jump component (Merton-style, per-artist) ────────────
            if use_jumps:
                jump_mask = (
                    rng.random((self.n_paths, n_artists)) < jump_prob_bcast
                )
                jump_log = (
                    rng.standard_normal((self.n_paths, n_artists))
                    * jump_std_bcast
                )
                jump_sizes = np.exp(jump_log)
                X = np.where(jump_mask, X * jump_sizes, X)

            np.maximum(X, 0, out=X)  # listeners can't go negative

        # ── Count wins ───────────────────────────────────────────────
        winners = np.argmax(X, axis=1)                # (n_paths,)
        win_counts = np.bincount(winners, minlength=n_artists)

        # ── Laplace smoothing ────────────────────────────────────────
        alpha = self.laplace_alpha
        if alpha > 0:
            smoothed_counts = win_counts.astype(np.float64) + alpha
            probs = smoothed_counts / (self.n_paths + n_artists * alpha)
        else:
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
