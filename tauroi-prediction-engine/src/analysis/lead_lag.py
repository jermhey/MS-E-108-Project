"""
lead_lag.py — Lead-Lag Alpha Analysis (Enhanced)
===================================================
Answers the core market microstructure questions:

1.  **Do we have predictive power?**
    Cross-correlation + Granger causality between model FV and Kalshi price.

2.  **What is the lead-lag?**
    Optimal lag (positive = model leads) with confidence intervals.

3.  **Alpha Decay: How long until things shift on you?**
    Measures how quickly the edge decays after a signal fires.
    Half-life of the cross-correlation envelope.

4.  **Taker vs Maker: Do we have alpha to take or only make ability?**
    Compares the model edge against typical Kalshi bid-ask spreads.
    If edge > spread → taker alpha.  If edge < spread → maker only.

5.  **Does the market converge to our price (book closing)?**
    Tracks whether market prices move toward our fair value after
    a signal (convergence) or away from it (divergence).

6.  **Speed edge: How big is it?**
    Quantifies the information advantage in hours and in cents.

Methodology
-----------
-  Fetch Kalshi daily OHLC candlesticks for the target market.
-  Reconstruct a walk-forward model fair-value time series by
   replaying the pricing engine on historical Chartmetric data.
-  Inner-join on date.
-  Cross-correlation at lags k = -max_lag … +max_lag.
-  Granger causality F-test.
-  Alpha decay analysis via rolling-window edge tracking.
-  Spread analysis for taker/maker classification.
"""

from __future__ import annotations

import datetime
import math
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.calibration import Calibrator, CalibrationResult
from src.pricing_engine import (
    JumpDiffusionPricer,
    ModelParams,
    MonteCarloOU,
    OUArtistInput,
)
from src.utils import get_logger

logger = get_logger("tauroi.lead_lag")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═════════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AlphaDecayResult:
    """Quantifies how quickly the model's edge decays after firing."""

    half_life_days: float          # days until edge halves
    full_decay_days: float         # days until edge is < 1 cent
    decay_curve: Dict[int, float]  # {days_after_signal: avg_residual_edge}
    n_signals: int                 # number of signal events analysed


@dataclass
class TakerMakerResult:
    """Classifies whether the edge is taker-grade or maker-only."""

    avg_edge_cents: float          # average |edge| in cents
    avg_spread_cents: float        # average Kalshi bid-ask spread in cents
    edge_to_spread_ratio: float    # edge / spread  (>1 = taker, <1 = maker)
    taker_pct: float               # % of signals where edge > spread
    classification: str            # "TAKER", "MAKER", or "NO_EDGE"
    edge_after_fees_cents: float   # edge minus Kalshi taker fee (~1c or 7%)


@dataclass
class BookClosingResult:
    """Measures whether the market converges toward or away from our price."""

    convergence_rate: float        # fraction of signals where market moved toward us
    avg_convergence_cents: float   # avg cents the market moved toward our FV
    avg_divergence_cents: float    # avg cents the market moved away
    net_convergence_cents: float   # signed: positive = market comes to us
    closes_book: bool              # True if convergence > 60%


@dataclass
class SpeedEdgeResult:
    """Quantifies the information advantage in time and money."""

    lead_hours: float              # estimated hours of information advantage
    lead_cents: float              # average cents of mispricing during lead window
    lead_cents_p25: float          # 25th percentile (conservative edge)
    lead_cents_p75: float          # 75th percentile (optimistic edge)
    n_lead_events: int             # number of events where we visibly led


@dataclass
class LeadLagResult:
    """Comprehensive result container for the full analysis."""

    # Core lead-lag
    optimal_lag: int
    optimal_corr: float
    lag_unit: str
    cross_corr: Dict[int, float]
    granger_f_stat: float | None
    granger_p_value: float | None
    n_observations: int
    model_series: pd.Series
    market_series: pd.Series

    # Enhanced analysis
    alpha_decay: AlphaDecayResult | None = None
    taker_maker: TakerMakerResult | None = None
    book_closing: BookClosingResult | None = None
    speed_edge: SpeedEdgeResult | None = None


# ═════════════════════════════════════════════════════════════════════════════
#  Core Analyzer
# ═════════════════════════════════════════════════════════════════════════════

class LeadLagAnalyzer:
    """
    Comprehensive lead-lag and alpha analysis between the Tauroi model
    signal and the Kalshi market price.
    """

    def __init__(
        self,
        kalshi_client: Any,
        cm_client: Any,
        calibration: CalibrationResult,
    ) -> None:
        self.kalshi = kalshi_client
        self.cm = cm_client
        self.cal = calibration
        self.pricer = JumpDiffusionPricer()

    # ── Step 1: Fetch Kalshi market price history ────────────────────────

    def fetch_market_prices(
        self,
        ticker: str,
        days: int = 60,
        interval_minutes: int = 1440,
    ) -> pd.DataFrame:
        """
        Fetch daily candlestick data from Kalshi.

        Returns a DataFrame indexed by date with columns:
            open, high, low, close, volume, spread
        Prices are converted from cents to [0, 1] probability.
        """
        now = int(_time.time())
        start = now - (days * 86400)

        try:
            candles = self.kalshi.get_market_candlesticks(
                ticker=ticker,
                start_ts=start,
                end_ts=now,
                period_interval=interval_minutes,
            )
        except Exception as exc:
            logger.warning("Candlestick fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if not candles:
            logger.warning("No candlestick data returned for %s", ticker)
            return pd.DataFrame()

        rows = []
        for c in candles:
            ts = c.get("end_period_ts", 0)
            if ts == 0:
                continue
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            o = c.get("open", 0) / 100.0
            h = c.get("high", 0) / 100.0
            lo = c.get("low", 0) / 100.0
            cl = c.get("close", 0) / 100.0
            rows.append({
                "date": dt.date(),
                "open": o,
                "high": h,
                "low": lo,
                "close": cl,
                "volume": c.get("volume", 0),
                "spread": h - lo,  # intraday range as spread proxy
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.groupby("date").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": "mean",
        }).sort_index()

        logger.info(
            "Market prices: %d days (%s -> %s)",
            len(df), df.index.min(), df.index.max(),
        )
        return df

    # ── Step 1b: Fallback — use trade history ────────────────────────────

    def fetch_trade_prices(
        self,
        ticker: str,
        max_pages: int = 10,
    ) -> pd.DataFrame:
        """
        Fallback: reconstruct daily prices from individual trades
        if the candlestick endpoint is unavailable.
        """
        all_trades: list[Dict[str, Any]] = []
        cursor = None

        for _ in range(max_pages):
            try:
                trades, cursor = self.kalshi.get_market_trades(
                    ticker=ticker, limit=1000, cursor=cursor,
                )
            except Exception as exc:
                logger.warning("Trade fetch failed for %s: %s", ticker, exc)
                break
            all_trades.extend(trades)
            if not cursor or not trades:
                break

        if not all_trades:
            return pd.DataFrame()

        rows = []
        for t in all_trades:
            ts_str = t.get("created_time", "")
            price = t.get("yes_price", t.get("price", 0))
            if not ts_str:
                continue
            try:
                dt = datetime.datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                )
                rows.append({
                    "date": dt.date(),
                    "price": price / 100.0 if price > 1 else price,
                })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        daily = df.groupby("date")["price"].agg(
            open="first", high="max", low="min", close="last",
        )
        daily["volume"] = df.groupby("date")["price"].count()
        daily["spread"] = daily["high"] - daily["low"]
        daily = daily.sort_index()

        logger.info(
            "Trade-derived prices: %d days (%s -> %s)",
            len(daily), daily.index.min(), daily.index.max(),
        )
        return daily

    # ── Step 2: Reconstruct model fair-value series ──────────────────────

    def reconstruct_model_series(
        self,
        artist_data: pd.DataFrame,
        strike: float,
        expiry_date: datetime.date,
        sigma: float | None = None,
    ) -> pd.Series:
        """
        Walk-forward reconstruction: for each historical date, compute
        what our model would have said given the data available that day.
        """
        sigma = sigma or self.cal.sigma
        beta = self.cal.jump_beta
        gamma = self.cal.vol_gamma
        p95 = self.cal.tiktok_p95

        df = artist_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        results = {}

        for date, row in df.iterrows():
            listeners = row.get("spotify_monthly_listeners", 0)
            if pd.isna(listeners) or listeners <= 0:
                continue

            days_to_expiry = (expiry_date - date.date()).days
            if days_to_expiry <= 0:
                continue
            T = days_to_expiry / 365.0

            raw_tiktok = row.get("tiktok_sound_posts_change", 0)
            if pd.isna(raw_tiktok):
                raw_tiktok = 0
            norm_v = min(max(raw_tiktok, 0) / p95, 1.0) if p95 > 0 else 0.0

            try:
                fv = self.pricer.fair_value_calibrated(
                    spot_official=float(listeners),
                    normalized_velocity=norm_v,
                    strike=strike,
                    sigma=sigma,
                    jump_beta=beta,
                    vol_gamma=gamma,
                    event_impact_score=1.0,
                    T=T,
                )
                results[date.date()] = fv
            except (ValueError, ZeroDivisionError):
                continue

        series = pd.Series(results, name="model_fv")
        series.index.name = "date"
        logger.info(
            "Reconstructed %d daily model fair values (%s -> %s)",
            len(series),
            series.index.min() if len(series) else "—",
            series.index.max() if len(series) else "—",
        )
        return series

    # ── Step 2b: WTA-aware model reconstruction ────────────────────────

    def reconstruct_wta_model_series(
        self,
        target_artist_name: str,
        target_data: pd.DataFrame,
        expiry_date: datetime.date,
        sigma: float | None = None,
        lookback_days: int = 90,
    ) -> pd.Series:
        """
        Walk-forward WTA reconstruction using the MC-OU engine.

        For each historical date (within ``lookback_days`` of expiry),
        loads all cached competitor listener counts and runs a mini
        Monte Carlo simulation to compute the target artist's win
        probability — the correct pricing for a Winner-Take-All market.

        Falls back to ``reconstruct_model_series`` (binary pricer) if
        competitor cache data is unavailable.

        Parameters
        ----------
        target_artist_name : str
            The artist whose win probability we're tracking.
        target_data : pd.DataFrame
            Historical data for the target artist (must have Date,
            spotify_monthly_listeners, tiktok_sound_posts_change).
        expiry_date : datetime.date
            Market expiry date.
        sigma : float, optional
            Override base sigma.
        lookback_days : int
            Only reconstruct this many days before expiry (default 90).
            Keeps runtime reasonable while covering the relevant period.
        """
        import pathlib

        sigma = sigma or self.cal.sigma
        beta = self.cal.jump_beta
        p95 = self.cal.tiktok_p95
        theta = self.cal.theta

        # ── Load competitor data from Parquet cache ──────────────────
        cache_dir = pathlib.Path(__file__).resolve().parents[2] / "cache"
        manifest_path = cache_dir / "_scan_manifest.json"

        competitor_histories: Dict[str, pd.DataFrame] = {}

        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)

            for cm_id_str, meta in manifest.items():
                name = meta.get("name", "")
                if name.lower() == target_artist_name.lower():
                    continue  # skip the target artist itself

                parquet_path = cache_dir / f"scan_{cm_id_str}.parquet"
                if parquet_path.exists():
                    try:
                        cdf = pd.read_parquet(parquet_path)
                        if "Date" in cdf.columns and "spotify_monthly_listeners" in cdf.columns:
                            cdf["Date"] = pd.to_datetime(cdf["Date"])
                            cdf = cdf.sort_values("Date").set_index("Date")
                            competitor_histories[name] = cdf
                    except Exception as exc:
                        logger.warning("Failed to load cache for %s: %s", name, exc)

        if not competitor_histories:
            logger.warning(
                "No competitor cache data found — falling back to binary pricer"
            )
            return self.reconstruct_model_series(
                target_data, strike=100_000_000, expiry_date=expiry_date, sigma=sigma,
            )

        logger.info(
            "WTA reconstruction: %d competitors loaded (%s)",
            len(competitor_histories),
            ", ".join(competitor_histories.keys()),
        )

        # ── Prepare target data ──────────────────────────────────────
        df = target_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        # Limit to lookback window before expiry
        earliest = pd.Timestamp(expiry_date) - pd.Timedelta(days=lookback_days)
        df = df.loc[earliest:]

        logger.info(
            "WTA reconstruction window: %d days (%s → expiry %s)",
            len(df), earliest.date(), expiry_date,
        )

        # Use fewer paths for reconstruction (speed vs accuracy tradeoff)
        mc = MonteCarloOU(
            theta=theta, n_paths=2_000, seed=42,
            jump_intensity=self.cal.jump_intensity,
            jump_std=self.cal.jump_std,
            laplace_alpha=25.0,
        )

        results: Dict[datetime.date, float] = {}
        n_computed = 0

        for date, row in df.iterrows():
            listeners = row.get("spotify_monthly_listeners", 0)
            if pd.isna(listeners) or listeners <= 0:
                continue

            days_to_expiry = (expiry_date - date.date()).days
            if days_to_expiry <= 0:
                continue
            T = days_to_expiry / 365.0

            raw_tiktok = row.get("tiktok_sound_posts_change", 0)
            if pd.isna(raw_tiktok):
                raw_tiktok = 0
            norm_v = min(max(raw_tiktok, 0) / p95, 1.0) if p95 > 0 else 0.0

            # Build artist inputs for this date
            mc_inputs = [
                OUArtistInput(
                    name=target_artist_name,
                    listeners=float(listeners),
                    sigma=sigma,
                    norm_velocity=norm_v,
                ),
            ]

            # Add each competitor's listener count as of this date
            for comp_name, comp_df in competitor_histories.items():
                valid = comp_df.loc[:date]
                if valid.empty:
                    continue
                comp_listeners = float(valid["spotify_monthly_listeners"].iloc[-1])
                if pd.isna(comp_listeners) or comp_listeners <= 0:
                    continue

                comp_sigma = sigma
                mc_inputs.append(OUArtistInput(
                    name=comp_name,
                    listeners=comp_listeners,
                    sigma=comp_sigma,
                    norm_velocity=0.0,
                ))

            if len(mc_inputs) < 2:
                continue

            try:
                mc_results = mc.simulate_wta(mc_inputs, T=T, jump_beta=beta)
                our = next(
                    (r for r in mc_results if r.name == target_artist_name), None,
                )
                if our is not None:
                    results[date.date()] = our.probability
                    n_computed += 1
            except Exception:
                continue

        series = pd.Series(results, name="model_fv")
        series.index.name = "date"
        logger.info(
            "WTA reconstructed %d daily model fair values (%s -> %s)",
            len(series),
            series.index.min() if len(series) else "—",
            series.index.max() if len(series) else "—",
        )
        return series

    # ── Step 3: Cross-Correlation ────────────────────────────────────────

    @staticmethod
    def cross_correlation(
        model: pd.Series,
        market: pd.Series,
        max_lag: int = 10,
    ) -> Dict[int, float]:
        """
        Compute cross-correlation between model and market at various lags.

        corr(model(t), market(t + k))  for k in [-max_lag, +max_lag]

        Positive optimal lag -> model LEADS market by k periods.
        Negative optimal lag -> model LAGS market by k periods.
        """
        merged = pd.DataFrame({
            "model": model,
            "market": market,
        }).dropna()

        if len(merged) < 5:
            logger.warning(
                "Only %d overlapping observations — cross-correlation unreliable",
                len(merged),
            )

        results = {}
        for k in range(-max_lag, max_lag + 1):
            m_shifted = merged["market"].shift(-k)

            pair = pd.DataFrame({
                "model": merged["model"],
                "market_shifted": m_shifted,
            }).dropna()

            if len(pair) < 3:
                results[k] = float("nan")
                continue

            corr = float(pair["model"].corr(pair["market_shifted"]))
            results[k] = corr

        return results

    # ── Step 4: Granger Causality ────────────────────────────────────────

    @staticmethod
    def granger_test(
        model: pd.Series,
        market: pd.Series,
        max_lag: int = 3,
    ) -> tuple[float | None, float | None]:
        """
        Simple Granger causality test: does the model signal improve
        prediction of market price changes?

        Null:  delta_P(t) = alpha + Sum beta_i * delta_P(t-i)
        Alt:   delta_P(t) = alpha + Sum beta_i * delta_P(t-i) + Sum gamma_j * Model(t-j)

        Returns (F_statistic, p_value) or (None, None) if insufficient data.
        """
        from scipy.stats import f as f_dist

        merged = pd.DataFrame({
            "model": model,
            "market": market,
        }).dropna().sort_index()

        if len(merged) < max_lag + 5:
            return None, None

        delta_p = merged["market"].diff().dropna()

        n = len(delta_p)
        if n <= 2 * max_lag + 2:
            return None, None

        X_r_cols = []
        for i in range(1, max_lag + 1):
            X_r_cols.append(delta_p.shift(i).rename(f"dp_lag{i}"))
        X_r = pd.concat(X_r_cols, axis=1).dropna()

        X_u_cols = list(X_r_cols)
        for j in range(1, max_lag + 1):
            X_u_cols.append(
                merged["model"].reindex(delta_p.index).shift(j).rename(f"model_lag{j}")
            )
        X_u = pd.concat(X_u_cols, axis=1).dropna()

        common_idx = X_u.index.intersection(X_r.index).intersection(delta_p.index)
        if len(common_idx) < 2 * max_lag + 2:
            return None, None

        y = delta_p.loc[common_idx].values
        X_r_mat = X_r.loc[common_idx].values
        X_u_mat = X_u.loc[common_idx].values
        n_obs = len(y)

        ones = np.ones((n_obs, 1))
        X_r_mat = np.hstack([ones, X_r_mat])
        X_u_mat = np.hstack([ones, X_u_mat])

        def _rss(X: np.ndarray, y: np.ndarray) -> float:
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                return float(np.sum(residuals ** 2))
            except np.linalg.LinAlgError:
                return float("inf")

        rss_r = _rss(X_r_mat, y)
        rss_u = _rss(X_u_mat, y)

        p_r = X_r_mat.shape[1]
        p_u = X_u_mat.shape[1]
        q = p_u - p_r

        if q <= 0 or rss_u <= 0 or n_obs <= p_u:
            return None, None

        f_stat = ((rss_r - rss_u) / q) / (rss_u / (n_obs - p_u))
        p_value = 1.0 - float(f_dist.cdf(f_stat, q, n_obs - p_u))

        return float(f_stat), float(p_value)

    # ═════════════════════════════════════════════════════════════════════
    #  ENHANCED ANALYSIS: Alpha Decay
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def analyze_alpha_decay(
        model: pd.Series,
        market: pd.Series,
        edge_threshold: float = 0.03,
        max_horizon: int = 10,
    ) -> AlphaDecayResult:
        """
        Measure how quickly the model's edge decays after a signal fires.

        For each day where |model - market| > threshold (a "signal"),
        track how the residual edge (model_t - market_{t+k}) evolves
        over the next ``max_horizon`` days.

        The half-life tells you how long you have before the market
        catches up to your information.
        """
        merged = pd.DataFrame({
            "model": model,
            "market": market,
        }).dropna().sort_index()

        if len(merged) < 5:
            return AlphaDecayResult(
                half_life_days=0.0, full_decay_days=0.0,
                decay_curve={}, n_signals=0,
            )

        edge = merged["model"] - merged["market"]
        signal_mask = edge.abs() > edge_threshold
        signal_dates = merged.index[signal_mask]
        n_signals = len(signal_dates)

        if n_signals == 0:
            return AlphaDecayResult(
                half_life_days=0.0, full_decay_days=0.0,
                decay_curve={0: 0.0}, n_signals=0,
            )

        # Track edge decay for each signal
        decay_buckets: Dict[int, list[float]] = {
            k: [] for k in range(max_horizon + 1)
        }

        for sig_date in signal_dates:
            initial_edge = float(edge.loc[sig_date])
            if abs(initial_edge) < 1e-6:
                continue

            sig_idx = merged.index.get_loc(sig_date)
            for k in range(max_horizon + 1):
                future_idx = sig_idx + k
                if future_idx >= len(merged):
                    break
                future_date = merged.index[future_idx]
                residual = float(
                    merged["model"].iloc[sig_idx] - merged["market"].iloc[future_idx]
                )
                # Normalise by initial edge direction
                normalised = residual / initial_edge if initial_edge != 0 else 0
                decay_buckets[k].append(normalised)

        # Average decay curve
        decay_curve = {}
        for k in range(max_horizon + 1):
            if decay_buckets[k]:
                decay_curve[k] = float(np.mean(decay_buckets[k]))
            else:
                decay_curve[k] = float("nan")

        # Compute half-life: first k where average normalised edge < 0.5
        half_life = float(max_horizon)
        for k in range(1, max_horizon + 1):
            if k in decay_curve and not math.isnan(decay_curve[k]):
                if decay_curve[k] < 0.5:
                    # Linear interpolation
                    prev = decay_curve.get(k - 1, 1.0)
                    if not math.isnan(prev) and prev > decay_curve[k]:
                        fraction = (prev - 0.5) / (prev - decay_curve[k])
                        half_life = (k - 1) + fraction
                    else:
                        half_life = float(k)
                    break

        # Full decay: first k where average normalised edge < 0.01 (1 cent)
        full_decay = float(max_horizon)
        for k in range(1, max_horizon + 1):
            if k in decay_curve and not math.isnan(decay_curve[k]):
                if abs(decay_curve[k]) < 0.01:
                    full_decay = float(k)
                    break

        return AlphaDecayResult(
            half_life_days=half_life,
            full_decay_days=full_decay,
            decay_curve=decay_curve,
            n_signals=n_signals,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  ENHANCED ANALYSIS: Taker vs Maker
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def analyze_taker_maker(
        model: pd.Series,
        market: pd.Series,
        market_df: pd.DataFrame,
        kalshi_fee_pct: float = 0.07,
    ) -> TakerMakerResult:
        """
        Determine if the edge is large enough to take (cross the spread)
        or if it only supports passive market-making.

        Kalshi's fee structure: typically ~7% of contract price (1c min).
        Spread is estimated from the daily high-low range.

        Taker:  |edge| > spread + fees  →  cross immediately
        Maker:  |edge| < spread         →  post limit orders
        """
        merged = pd.DataFrame({
            "model": model,
            "market": market,
        }).dropna().sort_index()

        if len(merged) < 3:
            return TakerMakerResult(
                avg_edge_cents=0.0, avg_spread_cents=0.0,
                edge_to_spread_ratio=0.0, taker_pct=0.0,
                classification="NO_EDGE", edge_after_fees_cents=0.0,
            )

        edge = (merged["model"] - merged["market"]).abs()
        edge_cents = edge * 100.0

        # Spread proxy: use high-low if available, else default 2c
        if "spread" in market_df.columns and not market_df["spread"].isna().all():
            common = merged.index.intersection(market_df.index)
            if len(common) > 0:
                spreads = market_df.loc[common, "spread"] * 100.0
                avg_spread = float(spreads.mean())
            else:
                avg_spread = 2.0
        else:
            avg_spread = 2.0  # default 2-cent spread

        avg_edge = float(edge_cents.mean())

        # Fee: Kalshi charges ~7% of contract price (min 1c per side)
        avg_market = float(merged["market"].mean()) * 100.0
        fee_per_side = max(avg_market * kalshi_fee_pct, 1.0)

        edge_after_fees = avg_edge - fee_per_side
        ratio = avg_edge / avg_spread if avg_spread > 0 else 0.0

        # Per-signal taker classification
        taker_count = 0
        total = len(merged)
        for idx in merged.index:
            e = float(edge_cents.loc[idx])
            s = avg_spread
            if e > s + fee_per_side:
                taker_count += 1
        taker_pct = taker_count / total if total > 0 else 0.0

        if ratio > 1.5 and taker_pct > 0.3:
            classification = "TAKER"
        elif ratio > 0.5:
            classification = "MAKER"
        else:
            classification = "NO_EDGE"

        return TakerMakerResult(
            avg_edge_cents=avg_edge,
            avg_spread_cents=avg_spread,
            edge_to_spread_ratio=ratio,
            taker_pct=taker_pct,
            classification=classification,
            edge_after_fees_cents=edge_after_fees,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  ENHANCED ANALYSIS: Book Closing
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def analyze_book_closing(
        model: pd.Series,
        market: pd.Series,
        edge_threshold: float = 0.03,
        horizon: int = 3,
    ) -> BookClosingResult:
        """
        Does the market converge toward our fair value after a signal?

        For each signal day (|edge| > threshold):
        - Look at market price ``horizon`` days later.
        - If the market moved toward our FV → convergence (book closing).
        - If it moved away → divergence.
        """
        merged = pd.DataFrame({
            "model": model,
            "market": market,
        }).dropna().sort_index()

        if len(merged) < horizon + 3:
            return BookClosingResult(
                convergence_rate=0.0, avg_convergence_cents=0.0,
                avg_divergence_cents=0.0, net_convergence_cents=0.0,
                closes_book=False,
            )

        edge = merged["model"] - merged["market"]
        signal_mask = edge.abs() > edge_threshold
        signal_idxs = [
            i for i, m in enumerate(signal_mask) if m
        ]

        convergences = []
        divergences = []

        for i in signal_idxs:
            if i + horizon >= len(merged):
                continue

            fv_at_signal = float(merged["model"].iloc[i])
            mkt_at_signal = float(merged["market"].iloc[i])
            mkt_later = float(merged["market"].iloc[i + horizon])

            initial_gap = abs(fv_at_signal - mkt_at_signal)
            later_gap = abs(fv_at_signal - mkt_later)

            # Positive = market moved toward us
            movement = (initial_gap - later_gap) * 100.0  # in cents

            if movement > 0:
                convergences.append(movement)
            else:
                divergences.append(abs(movement))

        total = len(convergences) + len(divergences)
        if total == 0:
            return BookClosingResult(
                convergence_rate=0.0, avg_convergence_cents=0.0,
                avg_divergence_cents=0.0, net_convergence_cents=0.0,
                closes_book=False,
            )

        conv_rate = len(convergences) / total
        avg_conv = float(np.mean(convergences)) if convergences else 0.0
        avg_div = float(np.mean(divergences)) if divergences else 0.0
        net = avg_conv * conv_rate - avg_div * (1 - conv_rate)

        return BookClosingResult(
            convergence_rate=conv_rate,
            avg_convergence_cents=avg_conv,
            avg_divergence_cents=avg_div,
            net_convergence_cents=net,
            closes_book=conv_rate > 0.60,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  ENHANCED ANALYSIS: Speed Edge
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def analyze_speed_edge(
        model: pd.Series,
        market: pd.Series,
        optimal_lag: int,
    ) -> SpeedEdgeResult:
        """
        Quantify the information advantage in time and money.

        The optimal lag tells us HOW MANY days we lead.
        The edge during that window tells us HOW MUCH.
        """
        merged = pd.DataFrame({
            "model": model,
            "market": market,
        }).dropna().sort_index()

        if len(merged) < 5 or optimal_lag <= 0:
            return SpeedEdgeResult(
                lead_hours=0.0, lead_cents=0.0,
                lead_cents_p25=0.0, lead_cents_p75=0.0,
                n_lead_events=0,
            )

        # For each day, compute the edge we had vs the market
        # that day, and check if the market caught up ``optimal_lag``
        # days later.
        lead_edges = []
        for i in range(len(merged) - optimal_lag):
            model_today = float(merged["model"].iloc[i])
            market_today = float(merged["market"].iloc[i])
            market_future = float(merged["market"].iloc[i + optimal_lag])

            edge_today = abs(model_today - market_today) * 100.0

            # Did the market move toward our prediction?
            direction = 1.0 if model_today > market_today else -1.0
            market_move = (market_future - market_today) * direction * 100.0

            if market_move > 0 and edge_today > 1.0:
                lead_edges.append(edge_today)

        if not lead_edges:
            return SpeedEdgeResult(
                lead_hours=optimal_lag * 24.0,
                lead_cents=0.0,
                lead_cents_p25=0.0,
                lead_cents_p75=0.0,
                n_lead_events=0,
            )

        arr = np.array(lead_edges)
        return SpeedEdgeResult(
            lead_hours=optimal_lag * 24.0,
            lead_cents=float(arr.mean()),
            lead_cents_p25=float(np.percentile(arr, 25)),
            lead_cents_p75=float(np.percentile(arr, 75)),
            n_lead_events=len(lead_edges),
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Visualization
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _ascii_sparkline(values: List[float], width: int = 50) -> str:
        """Render a simple ASCII sparkline."""
        if not values:
            return ""
        v_min, v_max = min(values), max(values)
        spread = v_max - v_min if v_max != v_min else 1.0
        blocks = " ▁▂▃▄▅▆▇█"
        return "".join(
            blocks[min(int((v - v_min) / spread * 8), 8)] for v in values
        )

    @staticmethod
    def plot_lead_lag(
        result: LeadLagResult,
        output_path: str = "lead_lag.png",
    ) -> str | None:
        """
        Generate a comprehensive 4-panel lead_lag.png:
        1. Model FV vs Market Price over time
        2. Cross-correlation bar chart
        3. Alpha decay curve
        4. Scatter plot (model vs market at optimal lag)
        """
        if not HAS_MPL:
            logger.info("matplotlib not installed — skipping PNG generation")
            return None

        n_panels = 4 if result.alpha_decay and result.alpha_decay.n_signals > 0 else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels))
        fig.suptitle("Tauroi Lead-Lag Alpha Analysis", fontsize=14, fontweight="bold")

        # ── Panel 1: Time series ──────────────────────────────────────
        ax1 = axes[0]
        model = result.model_series
        market = result.market_series
        common = model.index.intersection(market.index)
        if len(common) > 0:
            ax1.plot(common, model.loc[common], "b-o", markersize=3, label="Model FV")
            ax1.plot(common, market.loc[common], "r-s", markersize=3, label="Kalshi Close")
            ax1.fill_between(
                common,
                model.loc[common],
                market.loc[common],
                alpha=0.15,
                color="green",
                label="Edge",
            )
            ax1.set_ylabel("Probability")
            ax1.set_title("Model Fair Value vs Kalshi Market Price")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # ── Panel 2: Cross-correlation ────────────────────────────────
        ax2 = axes[1]
        lags = sorted(result.cross_corr.keys())
        corrs = [result.cross_corr[k] for k in lags]
        colors = ["green" if k == result.optimal_lag else "steelblue" for k in lags]
        ax2.bar(lags, corrs, color=colors, edgecolor="black", linewidth=0.3)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axvline(result.optimal_lag, color="green", linestyle="--", alpha=0.7,
                     label=f"Optimal lag = {result.optimal_lag:+d} {result.lag_unit}")
        ax2.set_xlabel(f"Lag ({result.lag_unit})")
        ax2.set_ylabel("Correlation")
        ax2.set_title("Cross-Correlation: Model(t) vs Market(t + k)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: Alpha decay (if available) ──────────────────────
        panel_idx = 2
        if result.alpha_decay and result.alpha_decay.n_signals > 0:
            ax3 = axes[panel_idx]
            dc = result.alpha_decay.decay_curve
            days = sorted(dc.keys())
            vals = [dc[d] for d in days]
            ax3.plot(days, vals, "b-o", markersize=5, linewidth=2)
            ax3.axhline(0.5, color="orange", linestyle="--", alpha=0.7,
                         label=f"Half-life = {result.alpha_decay.half_life_days:.1f} days")
            ax3.axhline(0, color="black", linewidth=0.5)
            ax3.set_xlabel("Days After Signal")
            ax3.set_ylabel("Normalised Edge (1.0 = initial)")
            ax3.set_title(
                f"Alpha Decay Curve ({result.alpha_decay.n_signals} signals)"
            )
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.2, 1.2)
            panel_idx += 1

        # ── Panel 4: Scatter at optimal lag ──────────────────────────
        ax4 = axes[panel_idx]
        k = result.optimal_lag
        if len(common) > abs(k) + 2:
            merged = pd.DataFrame({
                "model": model.loc[common],
                "market": market.loc[common],
            }).dropna()
            shifted_market = merged["market"].shift(-k).dropna()
            valid = merged["model"].loc[shifted_market.index]
            if len(valid) > 2:
                ax4.scatter(valid, shifted_market, alpha=0.6, edgecolors="black", linewidth=0.3)
                z = np.polyfit(valid.values, shifted_market.values, 1)
                x_line = np.linspace(valid.min(), valid.max(), 50)
                ax4.plot(x_line, z[0] * x_line + z[1], "r--", alpha=0.7)
                ax4.set_xlabel("Model Fair Value (t)")
                ax4.set_ylabel(f"Market Price (t + {k})")
                ax4.set_title(
                    f"Scatter at Optimal Lag ({k:+d} {result.lag_unit}) — "
                    f"r = {result.optimal_corr:.3f}"
                )
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved lead-lag chart -> %s", output_path)
        return output_path

    # ═════════════════════════════════════════════════════════════════════
    #  Multi-month chained market data
    # ═════════════════════════════════════════════════════════════════════

    def fetch_chained_monthly_prices(
        self,
        monthly_markets: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Fetch and chain trade data from multiple monthly WTA markets.

        Each month's trades are fetched separately and concatenated into
        a single time series.  Where months overlap (last day of one
        month = first day of next), the later market's price is used.

        Returns a DataFrame indexed by date with columns:
            open, high, low, close, volume, spread, source_ticker
        """
        all_dfs: List[pd.DataFrame] = []

        for mm in monthly_markets:
            ticker = mm.get("market_ticker", "")
            if not ticker:
                continue

            df = self.fetch_trade_prices(ticker, max_pages=10)
            if df.empty:
                df = self.fetch_market_prices(ticker, days=120)
            if df.empty:
                continue

            df["source_ticker"] = ticker
            all_dfs.append(df)
            logger.info(
                "  Chained %s: %d days (%s → %s)",
                ticker, len(df),
                df.index.min(), df.index.max(),
            )

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs)
        # On overlapping dates, prefer the later (more recent) market
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        logger.info(
            "Chained monthly prices: %d total days (%s → %s) from %d markets",
            len(combined), combined.index.min(), combined.index.max(),
            len(all_dfs),
        )
        return combined

    def reconstruct_chained_wta_series(
        self,
        target_artist_name: str,
        target_data: pd.DataFrame,
        monthly_markets: List[Dict[str, Any]],
        sigma: float | None = None,
    ) -> pd.Series:
        """
        Walk-forward WTA reconstruction across multiple monthly markets.

        For each month, uses that month's expiry date to compute
        the model's win probability.  This correctly accounts for the
        different time-to-expiry as the market transitions between months.
        """
        sigma = sigma or self.cal.sigma
        beta = self.cal.jump_beta
        p95 = self.cal.tiktok_p95
        theta = self.cal.theta

        # Load competitor histories from cache
        import pathlib
        import json as _json

        cache_dir = pathlib.Path(__file__).resolve().parents[2] / "cache"
        manifest_path = cache_dir / "_scan_manifest.json"

        competitor_histories: Dict[str, pd.DataFrame] = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = _json.load(f)
            for cm_id_str, meta in manifest.items():
                name = meta.get("name", "")
                if name.lower() == target_artist_name.lower():
                    continue
                parquet_path = cache_dir / f"scan_{cm_id_str}.parquet"
                if parquet_path.exists():
                    try:
                        cdf = pd.read_parquet(parquet_path)
                        if "Date" in cdf.columns and "spotify_monthly_listeners" in cdf.columns:
                            cdf["Date"] = pd.to_datetime(cdf["Date"])
                            cdf = cdf.sort_values("Date").set_index("Date")
                            competitor_histories[name] = cdf
                    except Exception as exc:
                        logger.warning("Failed to load cache for %s: %s", name, exc)

        if not competitor_histories:
            logger.warning("No competitor cache — cannot do WTA reconstruction")
            return pd.Series(dtype=float)

        logger.info(
            "Chained WTA reconstruction: %d competitors (%s)",
            len(competitor_histories),
            ", ".join(competitor_histories.keys()),
        )

        # Prepare target data
        df = target_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        # Build a mapping: for each calendar date, which month's expiry applies?
        date_to_expiry: Dict[datetime.date, datetime.date] = {}
        for mm in monthly_markets:
            expiry = mm.get("expiry_date")
            if expiry is None:
                continue
            # This month's market is relevant for dates up to and including expiry
            # Assume trading starts ~45 days before expiry
            start = expiry - datetime.timedelta(days=45)
            d = start
            while d <= expiry:
                # Later months override earlier ones on overlap
                date_to_expiry[d] = expiry
                d += datetime.timedelta(days=1)

        if not date_to_expiry:
            return pd.Series(dtype=float)

        # MC engine — use seed for reproducibility across dates,
        # but vary by date for realistic variation
        mc_base = MonteCarloOU(
            theta=theta, n_paths=2_000,
            jump_intensity=self.cal.jump_intensity,
            jump_std=self.cal.jump_std,
            laplace_alpha=25.0,
        )

        results: Dict[datetime.date, float] = {}
        dates_to_compute = sorted(date_to_expiry.keys())

        for d in dates_to_compute:
            if d not in df.index:
                # Try nearest prior date
                prior = df.loc[:pd.Timestamp(d)]
                if prior.empty:
                    continue
                row = prior.iloc[-1]
            else:
                row = df.loc[pd.Timestamp(d)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

            listeners = row.get("spotify_monthly_listeners", 0)
            if pd.isna(listeners) or listeners <= 0:
                continue

            expiry = date_to_expiry[d]
            days_to_expiry = (expiry - d).days
            if days_to_expiry <= 0:
                continue
            T = days_to_expiry / 365.0

            raw_tiktok = row.get("tiktok_sound_posts_change", 0)
            if pd.isna(raw_tiktok):
                raw_tiktok = 0
            norm_v = min(max(raw_tiktok, 0) / p95, 1.0) if p95 > 0 else 0.0

            mc_inputs = [
                OUArtistInput(
                    name=target_artist_name,
                    listeners=float(listeners),
                    sigma=sigma,
                    norm_velocity=norm_v,
                ),
            ]

            for comp_name, comp_df in competitor_histories.items():
                valid = comp_df.loc[:pd.Timestamp(d)]
                if valid.empty:
                    continue
                comp_listeners = float(valid["spotify_monthly_listeners"].iloc[-1])
                if pd.isna(comp_listeners) or comp_listeners <= 0:
                    continue
                mc_inputs.append(OUArtistInput(
                    name=comp_name,
                    listeners=comp_listeners,
                    sigma=sigma,
                    norm_velocity=0.0,
                ))

            if len(mc_inputs) < 2:
                continue

            # Use date-based seed for reproducibility per date
            mc_base.seed = hash(d.isoformat()) % (2**31)
            try:
                mc_results = mc_base.simulate_wta(mc_inputs, T=T, jump_beta=beta)
                our = next(
                    (r for r in mc_results if r.name == target_artist_name), None,
                )
                if our is not None:
                    results[d] = our.probability
            except Exception:
                continue

        series = pd.Series(results, name="model_fv")
        series.index.name = "date"
        logger.info(
            "Chained WTA reconstructed %d daily model fair values (%s → %s)",
            len(series),
            series.index.min() if len(series) else "—",
            series.index.max() if len(series) else "—",
        )
        return series

    # ═════════════════════════════════════════════════════════════════════
    #  Orchestrator
    # ═════════════════════════════════════════════════════════════════════

    def run(
        self,
        ticker: str,
        artist_name: str,
        artist_data: pd.DataFrame,
        strike: float,
        expiry_date: datetime.date,
        sigma: float | None = None,
        max_lag: int = 10,
        output_path: str = "lead_lag.png",
        market_type: str = "binary",
        historical_tickers: Optional[List[str]] = None,
        monthly_markets: Optional[List[Dict[str, Any]]] = None,
    ) -> LeadLagResult | None:
        """
        Full lead-lag analysis pipeline with enhanced market microstructure
        diagnostics.

        Parameters
        ----------
        market_type : str
            "winner_take_all" or "binary".  WTA uses MC-OU reconstruction.
        historical_tickers : list of str, optional
            Additional settled market tickers to include for longer history.
        monthly_markets : list of dict, optional
            Past monthly WTA markets from ``find_monthly_listener_markets()``.
            When provided, uses chained multi-month analysis instead of
            single-market mode.
        """
        print()
        print("  " + "=" * 72)
        print("  LEAD-LAG ALPHA ANALYSIS (Enhanced)")
        print(f"  Ticker: {ticker}  |  Artist: {artist_name}")
        print(f"  Market Type: {market_type.upper().replace('_', ' ')}")
        print("  " + "=" * 72)

        # ── 1. Market prices ──────────────────────────────────────────
        if monthly_markets and len(monthly_markets) > 1:
            # Chained multi-month mode
            print(f"\n  [1/6] Fetching chained monthly market data ({len(monthly_markets)} months)...")
            for mm in monthly_markets:
                print(f"    {mm['market_ticker']:35s}  vol={mm.get('volume', 0):>8,}  result={mm.get('result', '?')}")
            market_df = self.fetch_chained_monthly_prices(monthly_markets)
        else:
            print("\n  [1/6] Fetching Kalshi market price history...")
            market_df = self.fetch_market_prices(ticker, days=90)

            if market_df.empty:
                print("  -> No candlestick data. Trying trade history fallback...")
                market_df = self.fetch_trade_prices(ticker)

        if market_df.empty:
            print("  ERROR: No market price history available.")
            print("         (Market may be too new or have zero volume)")
            return None

        print(f"  -> {len(market_df)} daily price observations total")
        if not market_df.empty:
            print(f"     Range: {market_df.index.min()} → {market_df.index.max()}")
        market_close = market_df["close"]

        # ── 2. Model reconstruction ───────────────────────────────────
        if market_type == "winner_take_all" and monthly_markets and len(monthly_markets) > 1:
            print("  [2/6] Reconstructing chained WTA model (MC-OU per month)...")
            model_series = self.reconstruct_chained_wta_series(
                target_artist_name=artist_name,
                target_data=artist_data,
                monthly_markets=monthly_markets,
                sigma=sigma,
            )
        elif market_type == "winner_take_all":
            print("  [2/6] Reconstructing WTA model fair values (MC-OU)...")
            model_series = self.reconstruct_wta_model_series(
                target_artist_name=artist_name,
                target_data=artist_data,
                expiry_date=expiry_date,
                sigma=sigma,
            )
        else:
            print("  [2/6] Reconstructing walk-forward model fair values...")
            model_series = self.reconstruct_model_series(
                artist_data, strike, expiry_date, sigma,
            )

        if model_series.empty:
            print("  ERROR: Could not reconstruct model values.")
            return None

        print(f"  -> {len(model_series)} daily model observations")

        # ── 3. Cross-correlation ──────────────────────────────────────
        print(f"  [3/6] Computing cross-correlation (lags -{max_lag}..+{max_lag} days)...")
        xcorr = self.cross_correlation(model_series, market_close, max_lag)

        valid_lags = {k: v for k, v in xcorr.items() if not math.isnan(v)}
        if not valid_lags:
            print("  ERROR: No valid cross-correlations (insufficient overlap).")
            return None

        optimal_lag = max(valid_lags, key=lambda k: abs(valid_lags[k]))
        optimal_corr = valid_lags[optimal_lag]

        n_overlap = len(
            model_series.index.intersection(market_close.index)
        )

        # ── 4. Granger causality ──────────────────────────────────────
        print("  [4/6] Running Granger causality test...")
        f_stat, p_value = self.granger_test(model_series, market_close)

        # ── 5. Enhanced analysis ──────────────────────────────────────
        print("  [5/6] Running enhanced alpha analysis...")

        alpha_decay = self.analyze_alpha_decay(
            model_series, market_close,
            edge_threshold=0.03, max_horizon=max_lag,
        )

        taker_maker = self.analyze_taker_maker(
            model_series, market_close, market_df,
        )

        book_closing = self.analyze_book_closing(
            model_series, market_close,
            edge_threshold=0.03, horizon=3,
        )

        speed_edge = self.analyze_speed_edge(
            model_series, market_close, optimal_lag,
        )

        # ── Build result ──────────────────────────────────────────────
        result = LeadLagResult(
            optimal_lag=optimal_lag,
            optimal_corr=optimal_corr,
            lag_unit="days",
            cross_corr=xcorr,
            granger_f_stat=f_stat,
            granger_p_value=p_value,
            n_observations=n_overlap,
            model_series=model_series,
            market_series=market_close,
            alpha_decay=alpha_decay,
            taker_maker=taker_maker,
            book_closing=book_closing,
            speed_edge=speed_edge,
        )

        # ── 6. Print & visualize ──────────────────────────────────────
        print("  [6/6] Generating report and charts...")
        self._print_report(result, artist_name)
        self.plot_lead_lag(result, output_path)

        return result

    # ═════════════════════════════════════════════════════════════════════
    #  Report Printer
    # ═════════════════════════════════════════════════════════════════════

    def _print_report(self, result: LeadLagResult, artist_name: str) -> None:
        """Print a comprehensive console report answering all questions."""
        w = 72
        print()
        print("  " + "=" * w)
        print("  RESULTS — Full Market Microstructure Analysis")
        print("  " + "=" * w)

        # ── Q1: Lead-Lag ──────────────────────────────────────────────
        print()
        print("  [Q1] WHAT IS THE LEAD-LAG?")
        print("  " + "-" * w)
        lag = result.optimal_lag
        if lag > 0:
            headline = (
                f"Model LEADS market by {lag} {result.lag_unit} "
                f"(r = {result.optimal_corr:+.3f})"
            )
        elif lag < 0:
            headline = (
                f"Model LAGS market by {abs(lag)} {result.lag_unit} "
                f"(r = {result.optimal_corr:+.3f})"
            )
        else:
            headline = (
                f"Model and market move in SYNC "
                f"(r = {result.optimal_corr:+.3f})"
            )
        print(f"  >>> {headline}")
        print(f"  Overlapping observations: {result.n_observations}")

        # Cross-correlation table
        print(f"\n  Cross-Correlation Table (model(t) vs market(t+k)):")
        print("  " + "-" * 50)
        print(f"  {'Lag (days)':>12s}  {'Correlation':>12s}  {'':>10s}")
        print("  " + "-" * 50)
        for k in sorted(result.cross_corr.keys()):
            corr = result.cross_corr[k]
            if math.isnan(corr):
                bar = "  n/a"
            else:
                bar_len = int(abs(corr) * 20)
                bar_char = "+" if corr > 0 else "-"
                bar = " " + bar_char * bar_len
            marker = " <-- BEST" if k == result.optimal_lag else ""
            print(f"  {k:>+12d}  {corr:>12.4f}  {bar}{marker}")
        print("  " + "-" * 50)

        # Sparkline
        lags_sorted = sorted(result.cross_corr.keys())
        corr_values = [result.cross_corr[k] for k in lags_sorted]
        valid_values = [v for v in corr_values if not math.isnan(v)]
        if valid_values:
            spark = self._ascii_sparkline(valid_values)
            print(f"\n  Sparkline:  [{spark}]")
            print(f"              lag {min(lags_sorted):+d} -> {max(lags_sorted):+d}")

        # ── Q2: Granger (Predictive Power) ────────────────────────────
        print()
        print("  [Q2] DO WE HAVE PREDICTIVE POWER? (Granger Causality)")
        print("  " + "-" * w)
        if result.granger_f_stat is not None:
            sig = "YES" if result.granger_p_value < 0.05 else "NO"
            stars = ""
            if result.granger_p_value < 0.01:
                stars = " ***"
            elif result.granger_p_value < 0.05:
                stars = " **"
            elif result.granger_p_value < 0.10:
                stars = " *"
            print(f"  F-statistic:        {result.granger_f_stat:.3f}")
            print(f"  p-value:            {result.granger_p_value:.4f}{stars}")
            print(f"  Significant at 5%?  {sig}")
            if sig == "YES":
                print("  >>> Our signal Granger-causes market price changes")
            else:
                print("  >>> Cannot reject null: model adds no predictive power beyond own lags")
        else:
            print("  Insufficient data for Granger test")

        # ── Q3: Alpha Decay (How long until things shift?) ────────────
        print()
        print("  [Q3] HOW LONG UNTIL THINGS SHIFT ON YOU? (Alpha Decay)")
        print("  " + "-" * w)
        ad = result.alpha_decay
        if ad and ad.n_signals > 0:
            print(f"  Signal events analysed:   {ad.n_signals}")
            print(f"  Edge half-life:           {ad.half_life_days:.1f} days")
            print(f"  Full decay (to < 1c):     {ad.full_decay_days:.0f} days")
            print()
            print("  Decay Curve (normalised edge vs days after signal):")
            for k in sorted(ad.decay_curve.keys()):
                v = ad.decay_curve[k]
                if not math.isnan(v):
                    bar = "#" * max(int(v * 30), 0)
                    print(f"    day +{k:>2d}:  {v:>5.2f}  |{bar}")
            if ad.half_life_days <= 1.5:
                print("  >>> FAST DECAY — edge disappears within ~1 day, must act immediately")
            elif ad.half_life_days <= 3:
                print("  >>> MODERATE DECAY — ~2-3 day window to capture edge")
            else:
                print(f"  >>> SLOW DECAY — {ad.half_life_days:.0f}-day window (sustainable alpha)")
        else:
            print("  No signal events detected (edge never exceeded threshold)")

        # ── Q4: Taker vs Maker ────────────────────────────────────────
        print()
        print("  [Q4] TAKER OR MAKER? (Edge vs Spread Analysis)")
        print("  " + "-" * w)
        tm = result.taker_maker
        if tm:
            print(f"  Average |edge|:           {tm.avg_edge_cents:.1f}c")
            print(f"  Average spread:           {tm.avg_spread_cents:.1f}c")
            print(f"  Edge-to-spread ratio:     {tm.edge_to_spread_ratio:.2f}x")
            print(f"  Edge after fees:          {tm.edge_after_fees_cents:+.1f}c")
            print(f"  Taker-grade signals:      {tm.taker_pct:.0%}")
            print(f"  Classification:           {tm.classification}")
            if tm.classification == "TAKER":
                print("  >>> TAKER ALPHA — edge is large enough to cross the spread and pay fees")
            elif tm.classification == "MAKER":
                print("  >>> MAKER ONLY — post limit orders at model price, collect the spread")
            else:
                print("  >>> NO ACTIONABLE EDGE — edge does not cover transaction costs")

        # ── Q5: Book Closing ──────────────────────────────────────────
        print()
        print("  [Q5] DOES IT CLOSE THE BOOK? (Market Convergence)")
        print("  " + "-" * w)
        bc = result.book_closing
        if bc:
            print(f"  Convergence rate:         {bc.convergence_rate:.0%} of signals")
            print(f"  Avg convergence:          +{bc.avg_convergence_cents:.1f}c toward our FV")
            print(f"  Avg divergence:           -{bc.avg_divergence_cents:.1f}c away from FV")
            print(f"  Net convergence:          {bc.net_convergence_cents:+.1f}c")
            print(f"  Closes book?              {'YES' if bc.closes_book else 'NO'}")
            if bc.closes_book:
                print("  >>> YES — market converges to our price (>60% of signals)")
                print("  >>> This means our model is discovering true value")
            else:
                print("  >>> NO — market does not consistently move toward our price")
                print("  >>> Edge may be noise or the market may have information we lack")

        # ── Q6: Speed Edge ────────────────────────────────────────────
        print()
        print("  [Q6] SPEED EDGE: HOW BIG IS IT?")
        print("  " + "-" * w)
        se = result.speed_edge
        if se and se.n_lead_events > 0:
            print(f"  Information lead:         {se.lead_hours:.0f} hours ({se.lead_hours/24:.1f} days)")
            print(f"  Avg mispricing:           {se.lead_cents:.1f}c (during lead window)")
            print(f"  Conservative edge (P25):  {se.lead_cents_p25:.1f}c")
            print(f"  Optimistic edge (P75):    {se.lead_cents_p75:.1f}c")
            print(f"  Lead events observed:     {se.n_lead_events}")
            if se.lead_hours > 24:
                print(f"  >>> {se.lead_hours/24:.0f}-DAY speed advantage "
                      f"averaging {se.lead_cents:.0f}c per signal")
            else:
                print(f"  >>> {se.lead_hours:.0f}-HOUR speed advantage "
                      f"averaging {se.lead_cents:.1f}c per signal")
        elif lag > 0:
            print(f"  Model leads by {lag} day(s) but no profitable lead events detected")
            print("  >>> Timing edge exists but may not translate to P&L")
        else:
            print("  No speed edge detected (model does not lead market)")

        # ── Summary Verdict ───────────────────────────────────────────
        print()
        print("  " + "=" * w)
        print("  SUMMARY VERDICT")
        print("  " + "=" * w)

        has_predictive_power = (
            result.granger_p_value is not None and result.granger_p_value < 0.10
        )
        has_lead = lag > 0
        has_convergence = bc.closes_book if bc else False
        has_taker = tm.classification == "TAKER" if tm else False
        has_maker = tm.classification == "MAKER" if tm else False
        has_speed = se.n_lead_events > 0 if se else False

        checks = [
            ("Predictive power (Granger)", has_predictive_power),
            ("Timing lead (model leads market)", has_lead),
            ("Book closing (market converges)", has_convergence),
            ("Speed edge (profitable lead)", has_speed),
            ("Taker-grade edge (crosses spread)", has_taker),
            ("Maker ability (within spread)", has_taker or has_maker),
        ]

        for label, passed in checks:
            icon = "PASS" if passed else "FAIL"
            print(f"  [{icon}] {label}")

        passing = sum(1 for _, p in checks if p)
        total = len(checks)
        if passing >= 5:
            print(f"\n  >>> STRONG ALPHA — {passing}/{total} checks passed")
            print("  >>> Recommend aggressive taker strategy")
        elif passing >= 3:
            print(f"\n  >>> MODERATE ALPHA — {passing}/{total} checks passed")
            print("  >>> Recommend maker strategy with selective takes")
        elif passing >= 1:
            print(f"\n  >>> WEAK ALPHA — {passing}/{total} checks passed")
            print("  >>> Recommend passive maker only, tighten model")
        else:
            print(f"\n  >>> NO ALPHA — {passing}/{total} checks passed")
            print("  >>> Model does not generate actionable edge")

        print("  " + "=" * w)

        if HAS_MPL:
            print(f"\n  Chart saved to lead_lag.png")
        else:
            print(f"\n  (Install matplotlib for chart: pip install matplotlib)")
