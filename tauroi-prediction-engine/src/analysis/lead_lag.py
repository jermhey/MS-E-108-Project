"""
lead_lag.py — Lead-Lag Alpha Analysis
=======================================
Answers the question: **"Do we have an edge in speed?"**

Compares our model's theoretical fair-value time series against
Kalshi's actual market-price history to determine whether our signal
leads, lags, or moves in sync with the market.

Methodology
-----------
1.  Fetch Kalshi daily OHLC candlesticks for the target market.
2.  Reconstruct a walk-forward model fair-value time series by
    replaying the pricing engine on historical Chartmetric data.
3.  Inner-join on date.
4.  Cross-correlation at lags k = -max_lag … +max_lag:
        corr(model(t), market(t + k))
    Positive optimal lag → our model **leads** the market.
5.  (Optional) Granger causality F-test:
        Does knowing model(t-1…t-k) improve the forecast of
        ΔMarket(t) beyond its own lagged values?

Output
------
- Console report with optimal lag, cross-correlation table, and
  ASCII sparkline.
- ``lead_lag.png`` visualisation (if matplotlib is available).
"""

from __future__ import annotations

import datetime
import math
import time as _time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.calibration import Calibrator, CalibrationResult
from src.pricing_engine import JumpDiffusionPricer, ModelParams
from src.utils import get_logger

logger = get_logger("tauroi.lead_lag")

# Try importing matplotlib (optional — falls back to ASCII)
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
class LeadLagResult:
    """Immutable result container for the lead-lag analysis."""

    optimal_lag: int              # +k means model leads by k periods
    optimal_corr: float           # cross-correlation at the optimal lag
    lag_unit: str                 # "days" or "hours"
    cross_corr: Dict[int, float]  # {lag: correlation}
    granger_f_stat: float | None  # F-statistic (None if not computed)
    granger_p_value: float | None # p-value (None if not computed)
    n_observations: int           # number of overlapping data points
    model_series: pd.Series       # model fair values (date-indexed)
    market_series: pd.Series      # market close prices (date-indexed)


# ═════════════════════════════════════════════════════════════════════════════
#  Core Analyzer
# ═════════════════════════════════════════════════════════════════════════════

class LeadLagAnalyzer:
    """
    Computes the lead-lag relationship between the Tauroi model signal
    and the Kalshi market price.

    Parameters
    ----------
    kalshi_client : KalshiClient
        Authenticated Kalshi API client.
    cm_client : ChartmetricClient
        Authenticated Chartmetric API client.
    calibration : CalibrationResult
        Pre-computed calibration parameters (sigma, beta, gamma, etc.).
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
            open, high, low, close, volume
        Prices are converted from cents to [0, 1] probability.
        """
        now = int(_time.time())
        start = now - (days * 86400)

        candles = self.kalshi.get_market_candlesticks(
            ticker=ticker,
            start_ts=start,
            end_ts=now,
            period_interval=interval_minutes,
        )

        if not candles:
            logger.warning("No candlestick data returned for %s", ticker)
            return pd.DataFrame()

        rows = []
        for c in candles:
            ts = c.get("end_period_ts", 0)
            if ts == 0:
                continue
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            rows.append({
                "date": dt.date(),
                "open": c.get("open", 0) / 100.0,
                "high": c.get("high", 0) / 100.0,
                "low": c.get("low", 0) / 100.0,
                "close": c.get("close", 0) / 100.0,
                "volume": c.get("volume", 0),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Aggregate to daily if multiple candles per day
        df = df.groupby("date").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).sort_index()

        logger.info(
            "Market prices: %d days (%s → %s)",
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
            trades, cursor = self.kalshi.get_market_trades(
                ticker=ticker, limit=1000, cursor=cursor,
            )
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
        daily = daily.sort_index()

        logger.info(
            "Trade-derived prices: %d days (%s → %s)",
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

        Parameters
        ----------
        artist_data : DataFrame
            Must have columns: Date, spotify_monthly_listeners,
            and optionally tiktok_sound_posts_change.
        strike : float
            Binary option strike price.
        expiry_date : date
            Market expiry date (used to compute T for each day).
        sigma : float, optional
            Override base sigma (defaults to calibration result).

        Returns
        -------
        pd.Series
            Date-indexed series of model fair values ∈ [0, 1].
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

            # Compute time to expiry as of this date
            days_to_expiry = (expiry_date - date.date()).days
            if days_to_expiry <= 0:
                continue
            T = days_to_expiry / 365.0

            # TikTok velocity
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
            "Reconstructed %d daily model fair values (%s → %s)",
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

        Positive optimal lag → model LEADS market by k periods.
        Negative optimal lag → model LAGS market by k periods.
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
            if k >= 0:
                m_shifted = merged["market"].shift(-k)
            else:
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

        Null:  ΔP(t) = α + Σ β_i · ΔP(t-i)
        Alt:   ΔP(t) = α + Σ β_i · ΔP(t-i) + Σ γ_j · Model(t-j)

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

        # Build design matrices
        n = len(delta_p)
        if n <= 2 * max_lag + 2:
            return None, None

        # Restricted model: only lagged ΔP
        X_r_cols = []
        for i in range(1, max_lag + 1):
            X_r_cols.append(delta_p.shift(i).rename(f"dp_lag{i}"))
        X_r = pd.concat(X_r_cols, axis=1).dropna()

        # Unrestricted model: lagged ΔP + lagged Model
        X_u_cols = list(X_r_cols)
        for j in range(1, max_lag + 1):
            X_u_cols.append(
                merged["model"].reindex(delta_p.index).shift(j).rename(f"model_lag{j}")
            )
        X_u = pd.concat(X_u_cols, axis=1).dropna()

        # Align y with X_u index
        common_idx = X_u.index.intersection(X_r.index).intersection(delta_p.index)
        if len(common_idx) < 2 * max_lag + 2:
            return None, None

        y = delta_p.loc[common_idx].values
        X_r_mat = X_r.loc[common_idx].values
        X_u_mat = X_u.loc[common_idx].values
        n_obs = len(y)

        # Add intercept
        ones = np.ones((n_obs, 1))
        X_r_mat = np.hstack([ones, X_r_mat])
        X_u_mat = np.hstack([ones, X_u_mat])

        # OLS: RSS
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
        q = p_u - p_r  # number of additional regressors

        if q <= 0 or rss_u <= 0 or n_obs <= p_u:
            return None, None

        # F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - p_u))
        f_stat = ((rss_r - rss_u) / q) / (rss_u / (n_obs - p_u))
        p_value = 1.0 - float(f_dist.cdf(f_stat, q, n_obs - p_u))

        return float(f_stat), float(p_value)

    # ── Step 5: Visualization ────────────────────────────────────────────

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
        Generate a lead_lag.png with three subplots:
        1. Model FV vs Market Price over time
        2. Cross-correlation bar chart
        3. Scatter plot (model vs market at optimal lag)

        Returns the path if saved, None if matplotlib is unavailable.
        """
        if not HAS_MPL:
            logger.info("matplotlib not installed — skipping PNG generation")
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Tauroi Lead-Lag Alpha Analysis", fontsize=14, fontweight="bold")

        # ── Panel 1: Time series ──────────────────────────────────────
        ax1 = axes[0]
        model = result.model_series
        market = result.market_series
        common = model.index.intersection(market.index)
        if len(common) > 0:
            ax1.plot(common, model.loc[common], "b-o", markersize=3, label="Model FV")
            ax1.plot(common, market.loc[common], "r-s", markersize=3, label="Kalshi Close")
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

        # ── Panel 3: Scatter at optimal lag ───────────────────────────
        ax3 = axes[2]
        k = result.optimal_lag
        if len(common) > abs(k) + 2:
            merged = pd.DataFrame({
                "model": model.loc[common],
                "market": market.loc[common],
            }).dropna()
            shifted_market = merged["market"].shift(-k).dropna()
            valid = merged["model"].loc[shifted_market.index]
            if len(valid) > 2:
                ax3.scatter(valid, shifted_market, alpha=0.6, edgecolors="black", linewidth=0.3)
                # Fit line
                z = np.polyfit(valid.values, shifted_market.values, 1)
                x_line = np.linspace(valid.min(), valid.max(), 50)
                ax3.plot(x_line, z[0] * x_line + z[1], "r--", alpha=0.7)
                ax3.set_xlabel("Model Fair Value (t)")
                ax3.set_ylabel(f"Market Price (t + {k})")
                ax3.set_title(
                    f"Scatter at Optimal Lag ({k:+d} {result.lag_unit}) — "
                    f"r = {result.optimal_corr:.3f}"
                )
                ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved lead-lag chart → %s", output_path)
        return output_path

    # ── Orchestrator ─────────────────────────────────────────────────────

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
    ) -> LeadLagResult | None:
        """
        Full lead-lag analysis pipeline.

        Parameters
        ----------
        ticker : str
            Kalshi market ticker for the specific artist contract.
        artist_name : str
            Display name.
        artist_data : DataFrame
            Historical Chartmetric data (Date, spotify_monthly_listeners, etc.).
        strike : float
            Binary option strike price.
        expiry_date : date
            Market expiry date.
        sigma : float, optional
            Override base volatility.
        max_lag : int
            Maximum lag (in candle periods) to test.
        output_path : str
            Where to save the PNG chart.

        Returns
        -------
        LeadLagResult or None
        """
        print()
        print("  " + "=" * 72)
        print("  LEAD-LAG ALPHA ANALYSIS")
        print(f"  Ticker: {ticker}  |  Artist: {artist_name}")
        print("  " + "=" * 72)

        # ── 1. Market prices ──────────────────────────────────────────
        print("\n  [1/4] Fetching Kalshi market price history...")
        market_df = self.fetch_market_prices(ticker, days=60)

        if market_df.empty:
            print("  -> No candlestick data. Trying trade history fallback...")
            market_df = self.fetch_trade_prices(ticker)

        if market_df.empty:
            print("  ERROR: No market price history available.")
            print("         (Market may be too new or have zero volume)")
            return None

        print(f"  -> {len(market_df)} daily price observations")
        market_close = market_df["close"]

        # ── 2. Model reconstruction ───────────────────────────────────
        print("  [2/4] Reconstructing walk-forward model fair values...")
        model_series = self.reconstruct_model_series(
            artist_data, strike, expiry_date, sigma,
        )

        if model_series.empty:
            print("  ERROR: Could not reconstruct model values.")
            return None

        print(f"  -> {len(model_series)} daily model observations")

        # ── 3. Cross-correlation ──────────────────────────────────────
        print(f"  [3/4] Computing cross-correlation (lags -{ max_lag}..+{max_lag} days)...")
        xcorr = self.cross_correlation(model_series, market_close, max_lag)

        # Find optimal lag (highest absolute correlation)
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
        print("  [4/4] Running Granger causality test...")
        f_stat, p_value = self.granger_test(model_series, market_close)

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
        )

        # ── Print report ──────────────────────────────────────────────
        self._print_report(result, artist_name)

        # ── Save chart ────────────────────────────────────────────────
        self.plot_lead_lag(result, output_path)

        return result

    # ── Report Printer ───────────────────────────────────────────────────

    def _print_report(self, result: LeadLagResult, artist_name: str) -> None:
        """Print a formatted console report."""
        print()
        print("  " + "-" * 72)
        print("  RESULTS")
        print("  " + "-" * 72)

        # Headline
        lag = result.optimal_lag
        if lag > 0:
            headline = (
                f"Model LEADS market by {lag} {result.lag_unit} "
                f"(r = {result.optimal_corr:+.3f})"
            )
            verdict = "ALPHA DETECTED — Our signal predicts market movement"
        elif lag < 0:
            headline = (
                f"Model LAGS market by {abs(lag)} {result.lag_unit} "
                f"(r = {result.optimal_corr:+.3f})"
            )
            verdict = "NO EDGE — Market moves before our signal"
        else:
            headline = (
                f"Model and market move in SYNC "
                f"(r = {result.optimal_corr:+.3f})"
            )
            verdict = "REAL-TIME PARITY — No timing advantage"

        print(f"\n  >>> {headline}")
        print(f"  >>> {verdict}")
        print(f"\n  Overlapping observations: {result.n_observations}")

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

        # ASCII sparkline
        lags_sorted = sorted(result.cross_corr.keys())
        corr_values = [result.cross_corr[k] for k in lags_sorted]
        valid_values = [v for v in corr_values if not math.isnan(v)]
        if valid_values:
            spark = self._ascii_sparkline(valid_values)
            print(f"\n  Sparkline:  [{spark}]")
            print(f"              lag {min(lags_sorted):+d} {'→':>1s} {max(lags_sorted):+d}")

        # Granger causality
        print(f"\n  Granger Causality Test:")
        if result.granger_f_stat is not None:
            sig = "YES" if result.granger_p_value < 0.05 else "NO"
            stars = ""
            if result.granger_p_value < 0.01:
                stars = " ***"
            elif result.granger_p_value < 0.05:
                stars = " **"
            elif result.granger_p_value < 0.10:
                stars = " *"

            print(f"    F-statistic:  {result.granger_f_stat:.3f}")
            print(f"    p-value:      {result.granger_p_value:.4f}{stars}")
            print(f"    Significant at 5%?  {sig}")
            if sig == "YES":
                print("    -> Our signal Granger-causes market price changes")
            else:
                print("    -> Cannot reject null: model adds no predictive power")
        else:
            print("    -> Insufficient data for Granger test")

        # Chart note
        if HAS_MPL:
            print(f"\n  Chart saved to lead_lag.png")
        else:
            print(f"\n  (Install matplotlib for chart: pip install matplotlib)")

        print("\n  " + "=" * 72)
