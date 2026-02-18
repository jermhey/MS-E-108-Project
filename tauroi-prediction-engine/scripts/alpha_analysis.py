#!/usr/bin/env python3
"""
alpha_analysis.py — Comprehensive Alpha & Lead-Lag Analysis
=============================================================
Answers the six core trading questions by comparing the trend-aware
MC-OU model against **real Kalshi trade prices** across historical months.

Questions answered:
  1. Does the model have predictive power? (Granger causality)
  2. What is the lead-lag? (Cross-correlation)
  3. Taker or maker alpha? (Edge vs spread)
  4. Does it close the book? (Market convergence)
  5. How long until the edge decays? (Alpha half-life)
  6. Speed edge — how far ahead? (Hours & cents)

All model reconstruction uses NO look-ahead: on day d, only data
through day d feeds calibration and simulation.
"""
from __future__ import annotations

import sys
import pathlib
import json
import time
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import Calibrator, CalibrationResult
from src.pricing_engine import MonteCarloOU, OUArtistInput
from src.chartmetric_client import compute_momentum
from src.config import load_settings

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

CACHE_FILES = {
    "Bruno Mars":    (3501,  PROJECT_ROOT / "cache" / "scan_3501.parquet"),
    "The Weeknd":    (3852,  PROJECT_ROOT / "cache" / "scan_3852.parquet"),
    "Bad Bunny":     (214945, PROJECT_ROOT / "cache" / "scan_214945.parquet"),
    "Taylor Swift":  (2762,  PROJECT_ROOT / "cache" / "scan_2762.parquet"),
    "Billie Eilish": (5596,  PROJECT_ROOT / "cache" / "scan_5596.parquet"),
}

# Correct Kalshi tickers discovered via event API
KALSHI_TICKERS = {
    "2025-06": {
        "event": "KXTOPMONTHLY-25JUN",
        "Bruno Mars": "KXTOPMONTHLY-25JUN-BMAR",
        "The Weeknd": "KXTOPMONTHLY-25JUN-TWEE",
        "Taylor Swift": "KXTOPMONTHLY-25JUN-TSWI",
    },
    "2025-07": {
        "event": "KXTOPMONTHLY-25JUL",
        "Bruno Mars": "KXTOPMONTHLY-25JUL-BRU",
        "The Weeknd": "KXTOPMONTHLY-25JUL-WEE",
        "Taylor Swift": "KXTOPMONTHLY-25JUL-TAY",
    },
    "2025-08": {
        "event": "KXTOPMONTHLY-25AUG",
        "Bruno Mars": "KXTOPMONTHLY-25AUG-BRU",
        "The Weeknd": "KXTOPMONTHLY-25AUG-WEE",
        "Taylor Swift": "KXTOPMONTHLY-25AUG-TAY",
    },
    "2025-09": {
        "event": "KXTOPMONTHLY-25SEP",
        "The Weeknd": "KXTOPMONTHLY-25SEP-WEE",
        "Bruno Mars": "KXTOPMONTHLY-25SEP-BRU",
    },
    "2025-10": {
        "event": "KXTOPMONTHLY-25OCT",
        "The Weeknd": "KXTOPMONTHLY-25OCT-WEE",
        "Taylor Swift": "KXTOPMONTHLY-25OCT-TAY",
        "Bad Bunny": "KXTOPMONTHLY-25OCT-BAD",
    },
    "2025-11": {
        "event": "KXTOPMONTHLY-25NOV",
        "The Weeknd": "KXTOPMONTHLY-25NOV-WEE",
        "Taylor Swift": "KXTOPMONTHLY-25NOV-TAY",
        "Billie Eilish": "KXTOPMONTHLY-25NOV-BIL",
    },
    "2025-12": {
        "event": "KXTOPMONTHLY-25DEC",
        "The Weeknd": "KXTOPMONTHLY-25DEC-WEE",
        "Bruno Mars": "KXTOPMONTHLY-25DEC-BRU",
        "Taylor Swift": "KXTOPMONTHLY-25DEC-TAY",
        "Bad Bunny": "KXTOPMONTHLY-25DEC-BAD",
        "Billie Eilish": "KXTOPMONTHLY-25DEC-BIL",
    },
    "2026-01": {
        "event": "KXTOPMONTHLY-26JAN",
        "Bruno Mars": "KXTOPMONTHLY-26JAN-BRU",
        "The Weeknd": "KXTOPMONTHLY-26JAN-WEE",
        "Taylor Swift": "KXTOPMONTHLY-26JAN-TAY",
        "Bad Bunny": "KXTOPMONTHLY-26JAN-BAD",
        "Billie Eilish": "KXTOPMONTHLY-26JAN-BIL",
    },
}

ACTUAL_WINNERS = {
    "2025-06": "Bruno Mars",
    "2025-07": "Bruno Mars",
    "2025-08": "Bruno Mars",
    "2025-09": "The Weeknd",
    "2025-10": "The Weeknd",
    "2025-11": "The Weeknd",
    "2025-12": None,  # Ariana Grande won — NOT in our model! All 5 settle at 0.
    "2026-01": "Bruno Mars",
}

# Dec 2025: Ariana Grande won. This is a structural blind spot for our model.
DEC_2025_REAL_WINNER = "Ariana Grande"

TARGET_MONTHS = [
    "2025-06", "2025-07", "2025-08", "2025-09",
    "2025-10", "2025-11", "2025-12", "2026-01",
]

CALIBRATION_LOOKBACK = 90
MC_PATHS = 10_000
MC_SEED_BASE = 42


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data():
    all_dfs, combined, tiktok_map = {}, pd.DataFrame(), {}
    for name, (cm_id, path) in CACHE_FILES.items():
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date").asfreq("D").ffill()
        all_dfs[name] = df
        combined[name] = df["spotify_monthly_listeners"].astype(float)
        tc = "tiktok_sound_posts_change"
        tiktok_map[name] = df[tc].fillna(0).astype(float) if tc in df.columns else pd.Series(0.0, index=df.index)
    combined = combined.sort_index()
    return all_dfs, combined, tiktok_map


# ═════════════════════════════════════════════════════════════════════════════
#  KALSHI PRICE FETCHING (via trade aggregation)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_all_trades(kalshi, ticker: str, max_pages: int = 20) -> List[Dict]:
    """Paginate through all trades for a ticker."""
    all_trades = []
    cursor = None
    for _ in range(max_pages):
        trades, cursor = kalshi.get_market_trades(ticker, limit=1000, cursor=cursor)
        all_trades.extend(trades)
        if not cursor or not trades:
            break
        time.sleep(0.1)
    return all_trades


def trades_to_daily(trades: List[Dict]) -> pd.DataFrame:
    """Convert raw trades into daily OHLC + VWAP."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        dt = pd.Timestamp(t.get("created_time", "")).tz_localize(None).normalize()
        price = t.get("yes_price", t.get("price", 0))
        vol = t.get("count", t.get("volume", 1))
        rows.append({"Date": dt, "price": price / 100.0, "volume": vol})
    df = pd.DataFrame(rows)
    daily = df.groupby("Date").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        vwap=("price", lambda x: np.average(x, weights=df.loc[x.index, "volume"])),
        volume=("volume", "sum"),
    ).sort_index()
    return daily


def fetch_kalshi_daily_prices(
    year_month: str,
) -> Dict[str, pd.DataFrame]:
    """Fetch real Kalshi daily prices for all artists in a given month."""
    from src.kalshi_client import KalshiClient
    settings = load_settings()
    kalshi = KalshiClient(settings=settings)

    tickers = KALSHI_TICKERS.get(year_month, {})
    results = {}

    for artist, ticker in tickers.items():
        if artist == "event":
            continue
        trades = fetch_all_trades(kalshi, ticker)
        if trades:
            daily = trades_to_daily(trades)
            if len(daily) >= 3:
                results[artist] = daily
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL RECONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_artist(artist_df, as_of, lookback=CALIBRATION_LOOKBACK):
    start = as_of - pd.Timedelta(days=lookback)
    window = artist_df.loc[start:as_of].copy()
    if len(window) < 30:
        return None
    cal_df = pd.DataFrame({
        "Date": window.index,
        "spotify_monthly_listeners": window["spotify_monthly_listeners"].values,
        "tiktok_sound_posts_change": window.get(
            "tiktok_sound_posts_change", pd.Series(0.0, index=window.index)
        ).values,
    })
    try:
        return Calibrator.from_dataframe(cal_df).run()
    except Exception:
        return None


def compute_trend(listeners, as_of):
    hist = listeners.loc[:as_of].dropna()
    if len(hist) < 8:
        return 0.0
    recent = hist.tail(8)
    lr = np.log(recent / recent.shift(1)).dropna()
    lr = lr[np.isfinite(lr)]
    if len(lr) < 2:
        return 0.0
    return float(np.clip(float(lr.ewm(span=7).mean().iloc[-1]) * 365, -2.0, 2.0))


def reconstruct_month(
    year_month: str,
    combined: pd.DataFrame,
    all_dfs: Dict[str, pd.DataFrame],
    tiktok_map: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Day-by-day model reconstruction with multi-factor momentum."""
    ym = pd.Timestamp(f"{year_month}-01")
    month_end = ym + pd.offsets.MonthEnd(0)
    names = sorted(CACHE_FILES.keys())
    month_data = combined.loc[ym:month_end].dropna(how="all")
    month_dates = month_data.index
    last_cal = {}
    records = []

    for idx, day in enumerate(month_dates):
        days_remaining = (month_end - day).days
        T_years = max(days_remaining, 1) / 365.0

        if idx == 0 or idx % 7 == 0 or not last_cal:
            for name in names:
                cal = calibrate_artist(all_dfs[name], day)
                if cal is not None:
                    last_cal[name] = cal

        ou_inputs = []
        global_theta, global_jbeta, global_ji, global_js = 0.1, 0.0, 12.0, 0.04

        for name in names:
            cal = last_cal.get(name)
            val = combined.at[day, name] if day in combined.index else np.nan
            if np.isnan(val):
                before = combined[name].loc[:day].dropna()
                val = before.iloc[-1] if len(before) > 0 else 0

            if cal is not None:
                sigma = cal.sigma
                norm_vel = 0.0
                tv = tiktok_map[name].get(day, 0) if name in tiktok_map else 0
                if cal.tiktok_p95 > 0 and tv > 0:
                    norm_vel = min(tv / cal.tiktok_p95, 1.0)
                trend = cal.trend if cal.trend != 0 else compute_trend(combined[name], day)
                global_theta = cal.theta
                global_jbeta = cal.jump_beta
                global_ji = cal.jump_intensity
                global_js = cal.jump_std
            else:
                sigma, norm_vel = 0.20, 0.0
                trend = compute_trend(combined[name], day)

            # Compute multi-factor momentum from all cached features
            momentum = compute_momentum(all_dfs[name], as_of=day)

            ou_inputs.append(OUArtistInput(
                name=name, listeners=val, sigma=sigma,
                norm_velocity=norm_vel, trend=trend,
                event_impact_score=1.0, momentum=momentum,
            ))

        mc = MonteCarloOU(
            theta=global_theta, n_paths=MC_PATHS,
            seed=MC_SEED_BASE + idx, jump_intensity=global_ji,
            jump_std=global_js, laplace_alpha=25.0,
        )
        results = mc.simulate_wta(ou_inputs, T=T_years, jump_beta=global_jbeta)
        row = {"Date": day, "days_remaining": days_remaining}
        for r in results:
            row[r.name] = r.probability
        records.append(row)

    return pd.DataFrame(records).set_index("Date")


# ═════════════════════════════════════════════════════════════════════════════
#  STATISTICAL TOOLS
# ═════════════════════════════════════════════════════════════════════════════

def cross_correlation(model_s: pd.Series, market_s: pd.Series, max_lag: int = 7):
    aligned = pd.DataFrame({"model": model_s, "market": market_s}).dropna()
    if len(aligned) < 10:
        return {}
    m = aligned["model"].values
    k = aligned["market"].values
    m_n = (m - m.mean()) / (m.std() + 1e-12)
    k_n = (k - k.mean()) / (k.std() + 1e-12)
    n = len(m_n)
    cc = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            cc[lag] = float(np.mean(m_n[:n-lag] * k_n[lag:])) if lag < n else 0
        else:
            cc[lag] = float(np.mean(m_n[-lag:] * k_n[:n+lag])) if -lag < n else 0
    return cc


def granger_test(model_s: pd.Series, market_s: pd.Series, max_lag: int = 2):
    aligned = pd.DataFrame({"model": model_s, "market": market_s}).dropna()
    if len(aligned) < max_lag + 8:
        return None, None

    dy = aligned["market"].diff().dropna().values
    dx = aligned["model"].diff().dropna().values
    n = min(len(dy), len(dx))
    dy, dx = dy[:n], dx[:n]
    if n < max_lag + 5:
        return None, None

    Y = dy[max_lag:]
    X_r = np.column_stack([np.ones(len(Y))] + [dy[max_lag-i-1:n-i-1] for i in range(max_lag)])
    X_u = np.column_stack([X_r] + [dx[max_lag-i-1:n-i-1] for i in range(max_lag)])

    try:
        beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        ssr_r = float(np.sum((Y - X_r @ beta_r) ** 2))
        ssr_u = float(np.sum((Y - X_u @ beta_u) ** 2))
        df1 = max_lag
        df2 = len(Y) - X_u.shape[1]
        if df2 <= 0 or ssr_u <= 1e-15:
            return None, None
        F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p = 1.0 - sp_stats.f.cdf(max(F, 0), df1, df2)
        return round(float(F), 3), round(float(p), 5)
    except Exception:
        return None, None


def analyze_alpha_decay(model_s, market_s, edge_threshold=0.05, max_h=10):
    aligned = pd.DataFrame({"model": model_s, "market": market_s}).dropna()
    if len(aligned) < 15:
        return None
    edge = aligned["model"] - aligned["market"]
    signals = edge[edge.abs() > edge_threshold]
    if len(signals) < 3:
        return None
    decay = {h: [] for h in range(max_h + 1)}
    for date, se in signals.items():
        idx = aligned.index.get_loc(date)
        sign = 1 if se > 0 else -1
        for h in range(max_h + 1):
            if idx + h < len(aligned):
                move = aligned["market"].iloc[idx + h] - aligned["market"].iloc[idx]
                decay[h].append(sign * move)
    curve = {h: float(np.mean(vs)) for h, vs in decay.items() if vs}
    peak = max(curve.values()) if curve else 0
    half_life = max_h
    for h in sorted(curve.keys()):
        if h > 0 and curve[h] < peak / 2:
            half_life = h
            break
    return {
        "n_signals": len(signals),
        "avg_signal_edge": float(signals.abs().mean()),
        "curve": curve,
        "half_life": half_life,
        "peak_cents": round(peak * 100, 1),
    }


def analyze_book_closing(model_s, market_s, edge_threshold=0.05, horizon=3):
    aligned = pd.DataFrame({"model": model_s, "market": market_s}).dropna()
    if len(aligned) < 10:
        return None
    edge = aligned["model"] - aligned["market"]
    signals = edge[edge.abs() > edge_threshold]
    if len(signals) < 3:
        return None
    cases = []
    for date, se in signals.items():
        idx = aligned.index.get_loc(date)
        if idx + horizon < len(aligned):
            future = abs(aligned["model"].iloc[idx + horizon] - aligned["market"].iloc[idx + horizon])
            init = abs(se)
            cases.append({"init": init, "final": future, "converged": future < init})
    if not cases:
        return None
    df = pd.DataFrame(cases)
    return {
        "n": len(df),
        "rate": float(df["converged"].mean()),
        "net_cents": float((df["init"] - df["final"]).mean() * 100),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  TRADING SIMULATION (PnL)
# ═════════════════════════════════════════════════════════════════════════════

def simulate_trading(model_s, market_s, is_winner: bool, fee_pct=0.07):
    """
    Simulate a simple strategy: if model says P(win) > market price + threshold, buy.
    If model says P(win) < market price - threshold, sell.
    Mark-to-market at settlement (100 if winner, 0 if not).
    """
    aligned = pd.DataFrame({"model": model_s, "market": market_s}).dropna()
    if len(aligned) < 5:
        return None

    settlement = 1.0 if is_winner else 0.0
    trades = []

    for date, row in aligned.iterrows():
        edge = row["model"] - row["market"]
        if abs(edge) > 0.05:  # 5 cent threshold
            direction = "BUY" if edge > 0 else "SELL"
            entry_price = row["market"]
            if direction == "BUY":
                pnl = settlement - entry_price - fee_pct / 100
            else:
                pnl = entry_price - settlement - fee_pct / 100
            trades.append({
                "date": date,
                "direction": direction,
                "entry": entry_price,
                "edge": edge,
                "pnl": pnl,
            })

    if not trades:
        return None
    df = pd.DataFrame(trades)
    return {
        "n_trades": len(df),
        "avg_pnl": float(df["pnl"].mean()),
        "total_pnl_cents": float(df["pnl"].sum() * 100),
        "win_rate": float((df["pnl"] > 0).mean()),
        "avg_edge": float(df["edge"].abs().mean()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  TAUROI PREDICTION ENGINE — COMPREHENSIVE ALPHA ANALYSIS")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[1/4] Loading artist data from cache...")
    all_dfs, combined, tiktok_map = load_all_data()
    print(f"  Loaded {len(all_dfs)} artists, date range: "
          f"{combined.index[0].date()} to {combined.index[-1].date()}")

    # ── Reconstruct model ────────────────────────────────────────────
    print("\n[2/4] Reconstructing model fair values (day-by-day, no look-ahead)...")
    all_model = {}
    for ym in TARGET_MONTHS:
        print(f"  {ym}...", end="", flush=True)
        model_df = reconstruct_month(ym, combined, all_dfs, tiktok_map)
        all_model[ym] = model_df
        winner = ACTUAL_WINNERS[ym]
        fp = model_df.iloc[-1].get(winner, 0)
        print(f" {len(model_df)}d, P({winner})={fp:.1%}")

    # ── Fetch Kalshi prices ──────────────────────────────────────────
    print("\n[3/4] Fetching real Kalshi trade prices...")
    all_kalshi = {}
    total_contracts = 0
    for ym in TARGET_MONTHS:
        prices = fetch_kalshi_daily_prices(ym)
        all_kalshi[ym] = prices
        if prices:
            artists = ", ".join(f"{a}({len(d)}d)" for a, d in prices.items())
            total_contracts += len(prices)
            print(f"  {ym}: {artists}")
        else:
            print(f"  {ym}: no trades found")
    print(f"  Total: {total_contracts} artist-month contracts with real prices")

    # ── Build paired series ──────────────────────────────────────────
    print("\n[4/4] Aligning model vs market prices...")
    pairs = []
    for ym in TARGET_MONTHS:
        model_df = all_model[ym]
        kalshi = all_kalshi.get(ym, {})
        winner = ACTUAL_WINNERS[ym]  # None for Dec 2025

        for artist in CACHE_FILES.keys():
            if artist not in model_df.columns:
                continue
            model_prob = model_df[artist]

            # Use real Kalshi prices where available
            if artist in kalshi:
                kdf = kalshi[artist]
                market_prob = kdf["vwap"].reindex(model_prob.index).ffill().bfill()
                source = "kalshi"
            else:
                continue  # Skip artists without real market data

            pairs.append({
                "model": model_prob,
                "market": market_prob,
                "artist": artist,
                "month": ym,
                "is_winner": (winner is not None and artist == winner),
                "source": source,
            })

    print(f"  Built {len(pairs)} artist-month pairs with REAL Kalshi prices")
    if len(pairs) == 0:
        print("\n  WARNING: No real Kalshi data available. Cannot run analysis.")
        return

    # Separate winner and non-winner pairs
    winner_pairs = [p for p in pairs if p["is_winner"]]
    nonwinner_pairs = [p for p in pairs if not p["is_winner"]]

    # ══════════════════════════════════════════════════════════════════
    #  Q1: PREDICTIVE POWER
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Q1: DOES THE MODEL HAVE PREDICTIVE POWER?")
    print(f"{'═' * 80}\n")

    print("  Granger Causality: Do model changes predict future market changes?")
    print(f"  (p < 0.05 → significant, p < 0.01 → strong)\n")
    print(f"  {'Month':<8s}  {'Artist':<16s}  {'Winner?':<8s}  {'F-stat':>8s}  {'p-value':>8s}  {'Verdict'}")
    print("  " + "-" * 70)

    sig_count, total_tests = 0, 0
    for p in pairs:
        F, pv = granger_test(p["model"], p["market"], max_lag=2)
        if F is not None:
            total_tests += 1
            verdict = "PREDICTIVE ***" if pv < 0.01 else ("PREDICTIVE *" if pv < 0.05 else ("marginal" if pv < 0.10 else "no signal"))
            if pv < 0.05:
                sig_count += 1
            w_str = "YES" if p["is_winner"] else "no"
            print(f"  {p['month']:<8s}  {p['artist']:<16s}  {w_str:<8s}  {F:>8.2f}  {pv:>8.5f}  {verdict}")

    print(f"\n  Result: {sig_count}/{total_tests} contracts show Granger causality (p < 0.05)")

    # ══════════════════════════════════════════════════════════════════
    #  Q2: LEAD-LAG
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Q2: WHAT IS THE LEAD-LAG?")
    print(f"{'═' * 80}\n")

    # Cross-correlation on individual pairs
    print("  Per-contract cross-correlation (peak lag shown):")
    print(f"  {'Month':<8s}  {'Artist':<16s}  {'Best Lag':>8s}  {'r':>7s}  {'Interpretation'}")
    print("  " + "-" * 60)

    all_best_lags = []
    for p in pairs:
        cc = cross_correlation(p["model"], p["market"], max_lag=5)
        if cc:
            best = max(cc, key=cc.get)
            interp = (f"model leads {best}d" if best > 0
                      else (f"market leads {abs(best)}d" if best < 0 else "same day"))
            print(f"  {p['month']:<8s}  {p['artist']:<16s}  {best:>+6d}d  {cc[best]:>+7.3f}  {interp}")
            all_best_lags.append(best)

    if all_best_lags:
        avg_lag = np.mean(all_best_lags)
        print(f"\n  Average optimal lag across all contracts: {avg_lag:+.1f} days")

    # Aggregate cross-correlation across all winner pairs
    if winner_pairs:
        all_model_w = pd.concat([p["model"] for p in winner_pairs])
        all_market_w = pd.concat([p["market"] for p in winner_pairs])
        cc_agg = cross_correlation(all_model_w, all_market_w, max_lag=5)
        if cc_agg:
            print(f"\n  Aggregate cross-correlation (WINNER contracts only):")
            print(f"  {'Lag':>6s}  {'Corr':>8s}  {'':>30s}")
            print("  " + "-" * 50)
            for lag in sorted(cc_agg.keys()):
                bar = "█" * max(int(abs(cc_agg[lag]) * 30), 0)
                label = "←model leads" if lag > 0 else ("market leads→" if lag < 0 else "  same day  ")
                print(f"  {lag:>+4d}d   {cc_agg[lag]:>+7.4f}  {bar:<25s}  {label}")
            best_agg = max(cc_agg, key=cc_agg.get)
            print(f"\n  Peak at lag {best_agg:+d}d (r = {cc_agg[best_agg]:.4f})")

    # ══════════════════════════════════════════════════════════════════
    #  Q3: TAKER OR MAKER?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Q3: DO WE HAVE TAKER ALPHA OR ONLY MAKER ABILITY?")
    print(f"{'═' * 80}\n")

    # Per-contract edge analysis
    KALSHI_FEE = 0.07  # $0.07 per contract

    print(f"  {'Month':<8s}  {'Artist':<16s}  {'Win?':<6s}  {'Mean|Edge|':>10s}  "
          f"{'Med|Edge|':>10s}  {'P75':>8s}  {'%>5c':>6s}  {'%>10c':>6s}")
    print("  " + "-" * 80)

    all_edges = []
    for p in pairs:
        aligned = pd.DataFrame({"model": p["model"], "market": p["market"]}).dropna()
        if len(aligned) < 3:
            continue
        edge = (aligned["model"] - aligned["market"]).abs()
        all_edges.extend(edge.values)
        w = "YES" if p["is_winner"] else "no"
        print(f"  {p['month']:<8s}  {p['artist']:<16s}  {w:<6s}  "
              f"{edge.mean()*100:>9.1f}c  {edge.median()*100:>9.1f}c  "
              f"{np.percentile(edge, 75)*100:>7.1f}c  "
              f"{(edge > 0.05).mean():>5.0%}  {(edge > 0.10).mean():>5.0%}")

    if all_edges:
        edges = np.array(all_edges)
        print("  " + "-" * 80)
        print(f"  {'ALL':<8s}  {'':<16s}  {'':<6s}  "
              f"{edges.mean()*100:>9.1f}c  {np.median(edges)*100:>9.1f}c  "
              f"{np.percentile(edges, 75)*100:>7.1f}c  "
              f"{(edges > 0.05).mean():>5.0%}  {(edges > 0.10).mean():>5.0%}")

        # Spreads from Kalshi data
        spreads = []
        for p in pairs:
            if p["source"] == "kalshi":
                for ym_data in all_kalshi.get(p["month"], {}).values():
                    if "high" in ym_data.columns and "low" in ym_data.columns:
                        s = (ym_data["high"] - ym_data["low"]).mean()
                        spreads.append(s)
        avg_spread = np.mean(spreads) if spreads else 0.05
        cross_cost = avg_spread + KALSHI_FEE

        print(f"\n  Avg daily spread (from trades): {avg_spread*100:.1f}c")
        print(f"  Kalshi fee:                     {KALSHI_FEE*100:.1f}c")
        print(f"  Total cross-spread cost:        {cross_cost*100:.1f}c")
        print(f"\n  % observations with |edge| > spread:        {(edges > avg_spread).mean():.0%}")
        print(f"  % observations with |edge| > spread+fee:    {(edges > cross_cost).mean():.0%}")

        if (edges > cross_cost).mean() > 0.30:
            print(f"\n  VERDICT: **TAKER ALPHA** — edge exceeds all costs "
                  f"{(edges > cross_cost).mean():.0%} of observations")
        elif (edges > avg_spread).mean() > 0.30:
            print(f"\n  VERDICT: **MAKER ALPHA** — profitable as limit-order strategy")
        else:
            print(f"\n  VERDICT: Edge assessment depends on individual contracts (see table)")

    # ══════════════════════════════════════════════════════════════════
    #  Q4: BOOK CLOSING
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Q4: DOES THE MARKET CONVERGE TO OUR FAIR VALUE? (Book Closing)")
    print(f"{'═' * 80}\n")

    print(f"  {'Month':<8s}  {'Artist':<16s}  {'Signals':>8s}  {'Conv%':>6s}  {'Net':>8s}  {'Closes?'}")
    print("  " + "-" * 60)

    all_bc = []
    for p in pairs:
        bc = analyze_book_closing(p["model"], p["market"], edge_threshold=0.05, horizon=3)
        if bc:
            closes = "YES" if bc["rate"] > 0.55 else "no"
            print(f"  {p['month']:<8s}  {p['artist']:<16s}  {bc['n']:>8d}  "
                  f"{bc['rate']:>5.0%}  {bc['net_cents']:>+7.1f}c  {closes}")
            all_bc.append(bc)

    if all_bc:
        avg_rate = np.mean([b["rate"] for b in all_bc])
        avg_net = np.mean([b["net_cents"] for b in all_bc])
        print("  " + "-" * 60)
        print(f"  {'OVERALL':<8s}  {'':<16s}  {sum(b['n'] for b in all_bc):>8d}  "
              f"{avg_rate:>5.0%}  {avg_net:>+7.1f}c")
        print()
        if avg_rate > 0.55:
            print(f"  VERDICT: **CLOSES THE BOOK** — market converges {avg_rate:.0%} of the time")
            print(f"           Average convergence: {avg_net:+.1f}c over 3 days")
        else:
            print(f"  VERDICT: **DOES NOT RELIABLY CLOSE** — {avg_rate:.0%} convergence")

    # ══════════════════════════════════════════════════════════════════
    #  Q5: ALPHA DECAY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Q5: HOW LONG DO WE HAVE? (Alpha Decay)")
    print(f"{'═' * 80}\n")

    print(f"  {'Month':<8s}  {'Artist':<16s}  {'Signals':>8s}  {'Avg Edge':>9s}  {'Half-Life':>10s}")
    print("  " + "-" * 60)

    for p in pairs:
        ad = analyze_alpha_decay(p["model"], p["market"], edge_threshold=0.05, max_h=7)
        if ad:
            print(f"  {p['month']:<8s}  {p['artist']:<16s}  {ad['n_signals']:>8d}  "
                  f"{ad['avg_signal_edge']*100:>8.1f}c  {ad['half_life']:>8d}d")

    # Aggregate decay
    if winner_pairs:
        agg_ad = analyze_alpha_decay(
            pd.concat([p["model"] for p in winner_pairs]),
            pd.concat([p["market"] for p in winner_pairs]),
            edge_threshold=0.05, max_h=7,
        )
        if agg_ad:
            print("  " + "-" * 60)
            print(f"  {'WINNERS':<8s}  {'aggregate':<16s}  {agg_ad['n_signals']:>8d}  "
                  f"{agg_ad['avg_signal_edge']*100:>8.1f}c  {agg_ad['half_life']:>8d}d")
            print(f"\n  Decay curve (avg PnL per signal, winner contracts):")
            for h, pnl in sorted(agg_ad["curve"].items()):
                bar = "█" * max(int(pnl * 300), 0)
                print(f"    h={h}d: {pnl*100:>+6.1f}c  {bar}")
            print(f"\n  VERDICT: Alpha half-life = {agg_ad['half_life']} day(s)")
            if agg_ad["half_life"] <= 2:
                print(f"           Must trade within 1-2 days of signal")
            elif agg_ad["half_life"] <= 5:
                print(f"           3-5 day capture window")
            else:
                print(f"           Persistent edge — comfortable multi-day hold")

    # ══════════════════════════════════════════════════════════════════
    #  Q6: SPEED EDGE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Q6: DO WE HAVE A SPEED EDGE? HOW BIG?")
    print(f"{'═' * 80}\n")

    print("  When does model identify winner vs when does market price it in?")
    print(f"  (Dec 2025 excluded — Ariana Grande won, not in our model)\n")
    print(f"  {'Month':<8s}  {'Winner':<16s}  {'Model>50%':>10s}  {'Market>50%':>10s}  "
          f"{'Lead':>6s}  {'Edge@Signal':>12s}")
    print("  " + "-" * 75)

    lead_days, lead_cents = [], []
    for p in winner_pairs:
        model_s = p["model"]
        market_s = p["market"]
        ym = p["month"]
        month_end = pd.Timestamp(f"{ym}-01") + pd.offsets.MonthEnd(0)

        # Find first day model > 50%
        model_first = None
        for d, v in model_s.items():
            if v > 0.50:
                model_first = d
                break

        # Find first day market > 50%
        market_first = None
        aligned = pd.DataFrame({"model": model_s, "market": market_s}).dropna()
        for d in aligned.index:
            if aligned.at[d, "market"] > 0.50:
                market_first = d
                break

        if model_first is not None and market_first is not None:
            lead = (market_first - model_first).days
            lead_days.append(lead)
            edge_at = model_s.get(model_first, 0.5) - market_s.get(model_first, 0.5) if model_first in market_s.index else 0
            lead_cents.append(edge_at)
            m_str = f"{(month_end - model_first).days}d left"
            k_str = f"{(month_end - market_first).days}d left"
            l_str = f"+{lead}d" if lead > 0 else f"{lead}d"
            e_str = f"{edge_at*100:+.1f}c"
        elif model_first is not None:
            m_str = f"{(month_end - model_first).days}d left"
            k_str = "never >50%"
            l_str = "+∞"
            e_str = "N/A"
        else:
            m_str = "never >50%"
            k_str = "never >50%" if market_first is None else f"{(month_end - market_first).days}d left"
            l_str = "N/A"
            e_str = "N/A"

        print(f"  {ym:<8s}  {p['artist']:<16s}  {m_str:>10s}  {k_str:>10s}  "
              f"{l_str:>6s}  {e_str:>12s}")

    if lead_days:
        avg_lead = np.mean(lead_days)
        avg_edge = np.mean(lead_cents) * 100 if lead_cents else 0
        print("  " + "-" * 75)
        print(f"  Average: model leads by {avg_lead:+.1f} days, edge at signal = {avg_edge:+.1f}c")
        print()
        if avg_lead > 0.5:
            print(f"  VERDICT: **SPEED EDGE EXISTS** — {avg_lead:.1f} day(s) ahead on average")
        elif avg_lead > -0.5:
            print(f"  VERDICT: **ROUGHLY SIMULTANEOUS** — model and market ~same timing")
        else:
            print(f"  VERDICT: **MARKET LEADS** — market prices information faster")

    # ══════════════════════════════════════════════════════════════════
    #  TRADING PnL SIMULATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  BONUS: SIMULATED TRADING PnL")
    print(f"{'═' * 80}\n")

    print(f"  Strategy: Trade when |edge| > 5c (BUY if model > market, SELL if model < market)")
    print(f"  Settlement: 100c if winner, 0c if loser. Fee: 7c round-trip.\n")

    print(f"  {'Month':<8s}  {'Artist':<16s}  {'Win?':<6s}  {'Trades':>7s}  "
          f"{'Win%':>6s}  {'Avg PnL':>8s}  {'Total PnL':>10s}")
    print("  " + "-" * 70)

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    for p in pairs:
        sim = simulate_trading(p["model"], p["market"], p["is_winner"])
        if sim:
            w = "YES" if p["is_winner"] else "no"
            print(f"  {p['month']:<8s}  {p['artist']:<16s}  {w:<6s}  "
                  f"{sim['n_trades']:>7d}  {sim['win_rate']:>5.0%}  "
                  f"{sim['avg_pnl']*100:>+7.1f}c  {sim['total_pnl_cents']:>+9.1f}c")
            total_pnl += sim["total_pnl_cents"]
            total_trades += sim["n_trades"]
            total_wins += int(sim["win_rate"] * sim["n_trades"])

    if total_trades > 0:
        print("  " + "-" * 70)
        print(f"  {'TOTAL':<8s}  {'':<16s}  {'':<6s}  {total_trades:>7d}  "
              f"{total_wins/total_trades:>5.0%}  "
              f"{total_pnl/total_trades:>+7.1f}c  {total_pnl:>+9.1f}c")

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  OVERALL ALPHA ASSESSMENT SUMMARY")
    print(f"{'═' * 80}\n")

    print(f"  1. Predictive Power:  {sig_count}/{total_tests} contracts show "
          f"Granger causality at p<0.05")

    if all_best_lags:
        print(f"  2. Lead-Lag:          Avg optimal lag = {avg_lag:+.1f} days")

    if all_edges:
        print(f"  3. Taker/Maker:       Mean |edge| = {edges.mean()*100:.1f}c, "
              f"{(edges > cross_cost).mean():.0%} exceed full cost")

    if all_bc:
        print(f"  4. Book Closing:      {avg_rate:.0%} convergence rate "
              f"({'YES' if avg_rate > 0.55 else 'NO'})")

    if winner_pairs and agg_ad:
        print(f"  5. Alpha Decay:       Half-life = {agg_ad['half_life']} day(s)")

    if lead_days:
        print(f"  6. Speed Edge:        {avg_lead:+.1f} days, {avg_edge:+.1f}c at signal")

    if total_trades > 0:
        print(f"\n  Trading PnL:          {total_pnl:+.0f}c across {total_trades} trades "
              f"({total_wins/total_trades:.0%} win rate)")

    # ── Structural risk: December 2025 blind spot ────────────────────
    print(f"\n  ⚠  STRUCTURAL RISK: December 2025 was won by Ariana Grande,")
    print(f"     who was NOT in our 5-artist model. All December contracts")
    print(f"     for our tracked artists settled at $0. The model assigned")

    dec_model = all_model.get("2025-12")
    if dec_model is not None:
        last_day = dec_model.iloc[-1]
        for name in ["The Weeknd", "Bruno Mars"]:
            if name in last_day:
                print(f"     {name}: {last_day[name]:.1%} on last day → settled at 0%")
    print(f"     This represents a catastrophic tail risk from incomplete")
    print(f"     artist coverage. Must expand to ALL Kalshi-listed artists.")


if __name__ == "__main__":
    main()
