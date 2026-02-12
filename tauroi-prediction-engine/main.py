#!/usr/bin/env python3
"""
main.py — Tauroi Prediction Engine: Entry Point
=================================================
Phase 7: Cloud-Ready Production.

All data is fetched from the Chartmetric API — no local CSV files required.
Designed to run as a GitHub Action, Docker container, or standalone cron job.

Modes
-----
    python main.py                         # Single-artist deep dive (from .env)
    python main.py --scan                  # Full market scan — price ALL artists
    python main.py --artist "Taylor Swift" --artist-id 2762  # Override artist
    python main.py --dry-run               # No order execution
    python main.py --loop --interval 900   # Continuous: re-run every 15 min

Signal Persistence
------------------
Every run appends a JSON record to ``signals/signal_log.jsonl``.
This file is cumulative and safe to commit to git for audit trails.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import time as _time
import sys
import traceback
from typing import Any, Dict, Optional

import pandas as pd

from src.calibration import Calibrator, CalibrationResult
from src.config import load_settings, Settings
from src.pricing_engine import JumpDiffusionPricer, ModelParams
from src.utils import get_logger, format_signal


# ── configuration ───────────────────────────────────────────────────────────

EDGE_THRESHOLD = 0.05
DEFAULT_STRIKE = 100_000_000
SIGNAL_LOG_DIR = pathlib.Path(__file__).resolve().parent / "signals"

logger = get_logger("tauroi.main")

# ARTIST_NAME is set dynamically from .env / CLI at runtime.
# Modules reference this global; it is updated by main() before any run.
ARTIST_NAME: str = "Bad Bunny"


# ── helpers ─────────────────────────────────────────────────────────────────

def normalise_tiktok(raw_change: float, cap: float) -> float:
    """
    Normalise a raw ``tiktok_sound_posts_change`` value to [0, 1].

    Negative changes (cooldown) map to 0 — no viral signal.
    Values above ``cap`` saturate at 1.0.
    """
    if raw_change <= 0 or cap <= 0:
        return 0.0
    return min(raw_change / cap, 1.0)


def compute_time_to_expiry(close_time_str: str | None) -> float:
    """
    Compute time-to-expiry in years from an ISO close-time string.

    Falls back to 1/365 (one day) if the string is unparseable.
    """
    if not close_time_str:
        return 1 / 365
    try:
        close_dt = datetime.datetime.fromisoformat(
            close_time_str.replace("Z", "+00:00")
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        delta_days = max((close_dt - now).total_seconds() / 86400, 0.5)
        return delta_days / 365
    except Exception:
        return 1 / 365


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

def append_signal_log(record: Dict[str, Any]) -> pathlib.Path:
    """
    Append a single JSON signal record to ``signals/signal_log.jsonl``.

    The file is append-only (one JSON object per line).  Safe for
    concurrent writes and trivially parseable:

        ``pd.read_json("signals/signal_log.jsonl", lines=True)``
    """
    SIGNAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SIGNAL_LOG_DIR / "signal_log.jsonl"

    record["logged_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("Signal logged → %s", log_path)
    return log_path


# ═════════════════════════════════════════════════════════════════════════════
#  COMPETITIVE LANDSCAPE
# ═════════════════════════════════════════════════════════════════════════════

def _print_competitive_landscape(
    competitor_data: list[Dict[str, Any]],
    kalshi_competitors: list[Dict[str, Any]],
    our_artist: str,
    our_listeners: float,
) -> None:
    """
    Print a rich table showing every competitor's real listener count,
    7-day trend, and Kalshi market price — all in one view.
    """
    # Build a lookup for Kalshi prices by competitor name
    kalshi_lookup: Dict[str, int] = {}
    for kc in kalshi_competitors:
        k_name = kc.get("name", "")
        k_last = kc.get("last_price", 0) or 0
        kalshi_lookup[k_name] = k_last

    # Merge our own artist into the table for context
    rows: list[tuple[str, int, float | None, int | None, bool]] = []

    # Add competitors (already sorted desc by listeners)
    for cd in competitor_data:
        rows.append((
            cd["name"],
            cd["listeners"],
            cd.get("change_pct"),
            kalshi_lookup.get(cd["name"]),
            False,
        ))

    # Insert our artist in sorted position
    inserted = False
    sorted_rows: list[tuple[str, int, float | None, int | None, bool]] = []
    for row in rows:
        if not inserted and our_listeners >= row[1]:
            sorted_rows.append((our_artist, int(our_listeners), None, None, True))
            inserted = True
        sorted_rows.append(row)
    if not inserted:
        sorted_rows.append((our_artist, int(our_listeners), None, None, True))

    # Print table
    print()
    print("  COMPETITIVE LANDSCAPE (Live Chartmetric Data)")
    print("  " + "-" * 62)
    print(f"  {'Artist':<22s}  {'Listeners':>15s}  {'7d Chg':>8s}  {'Kalshi':>8s}")
    print("  " + "-" * 62)

    for name, listeners, chg_pct, kalshi_price, is_us in sorted_rows:
        chg_str = f"{chg_pct:+.1f}%" if chg_pct is not None else "—"
        kalshi_str = f"{kalshi_price}c" if kalshi_price is not None else "—"
        marker = " <-- US" if is_us else ""
        print(
            f"  {name:<22s}  {listeners:>15,d}  {chg_str:>8s}  {kalshi_str:>8s}{marker}"
        )

    print("  " + "-" * 62)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA SOURCES
# ═════════════════════════════════════════════════════════════════════════════

def load_data_from_api(settings: Settings) -> pd.DataFrame:
    """
    Fetch **full historical** artist metrics from the Chartmetric API.

    Fetches Spotify Monthly Listeners + TikTok Sound Posts since 2019-01-01
    (paginated through yearly windows), plus Album/Single release history
    to compute ``event_impact_score``.

    Falls back to local Parquet cache if the API is rate-limited.

    Returns a DataFrame with columns the Calibrator and PricingEngine expect:
        Date, spotify_monthly_listeners, tiktok_sound_posts_change,
        event_impact_score
    """
    from src.chartmetric_client import ChartmetricClient, ChartmetricAPIError

    logger.info("=" * 64)
    logger.info("DATA SOURCE — Chartmetric API (full history)")
    logger.info("=" * 64)

    client = ChartmetricClient(settings=settings)
    client.check_connection()

    try:
        df = client.get_artist_metrics(since="2019-01-01")
    except (ChartmetricAPIError, Exception) as exc:
        # ── Fallback: use local scan cache for the primary artist ────
        artist_id = settings.chartmetric_artist_id
        cm_id = int(artist_id) if artist_id and artist_id.isdigit() else None

        if cm_id is not None:
            cached_df = client._load_stale_cache(cm_id)
            if cached_df is not None and not cached_df.empty:
                logger.warning(
                    "API unavailable (%s) — using CACHED scan data "
                    "(%d rows) for calibration",
                    str(exc)[:80], len(cached_df),
                )
                # Ensure columns the Calibrator expects
                if "event_impact_score" not in cached_df.columns:
                    cached_df["event_impact_score"] = 1.0
                if "tiktok_sound_posts_change" not in cached_df.columns:
                    cached_df["tiktok_sound_posts_change"] = 0.0
                client._last_release_count = 0
                return cached_df, client

        raise  # No cache available — propagate the original error

    return df, client



# ═════════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

def run_calibration_from_df(df: pd.DataFrame) -> CalibrationResult:
    """
    Run calibration directly from an in-memory DataFrame.

    Works identically for both API-sourced and CSV-sourced data.
    """
    logger.info("=" * 64)
    logger.info("CALIBRATION — %s", ARTIST_NAME)
    logger.info("=" * 64)

    calibrator = Calibrator.from_dataframe(df)
    result = calibrator.run()

    # Pretty-print calibration results
    print()
    print("  " + "=" * 52)
    print(f"  CALIBRATION RESULTS — {ARTIST_NAME}")
    print("  " + "=" * 52)
    print(f"  Base Volatility (sigma)      : {result.sigma:>11.2%}  (annualised)")
    print(f"  Vol Gamma (gamma)            : {result.vol_gamma:>11.2%}  (conditional)")
    print(f"  Jump Sensitivity (beta)      : {result.jump_beta:>12.6f}")
    print(f"  Best TikTok Lag              : {result.best_lag:>12d}  days")
    print(f"  TikTok P95 (norm cap)        : {result.tiktok_p95:>12,.0f}")
    print("  " + "=" * 52)
    print(f"  sigma at velocity=0.0        : {result.sigma:>11.2%}")
    print(f"  sigma at velocity=0.5        : {result.sigma + result.vol_gamma * 0.5:>11.2%}")
    print(f"  sigma at velocity=1.0        : {result.sigma + result.vol_gamma:>11.2%}")
    print("  " + "=" * 52)
    print()

    return result



# ═════════════════════════════════════════════════════════════════════════════
#  KALSHI MARKET DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def fetch_kalshi_market(
    settings: Settings,
    artist_name: str = ARTIST_NAME,
) -> Optional[Dict[str, Any]]:
    """
    Two-Stage Discovery: find a streaming-specific Kalshi market for the artist.

    Returns an enriched dict with keys:
        target_contract, event_title, event_ticker, market_type, competitors
    or None if no market is found.
    """
    logger.info("=" * 64)
    logger.info("KALSHI — Two-Stage Market Discovery")
    logger.info("=" * 64)

    from src.kalshi_client import KalshiClient, KalshiAuthError

    try:
        client = KalshiClient(settings=settings)
        client.check_connection()

        result, rejected = client.find_active_listener_market(artist_name)

        print()
        if result is not None:
            contract = result["target_contract"]
            ticker = contract.get("ticker", "UNKNOWN")
            option = contract.get("yes_sub_title", contract.get("subtitle", "—"))
            yes_bid = contract.get("yes_bid", 0)
            yes_ask = contract.get("yes_ask", 0)
            last = contract.get("last_price", 0)
            vol = contract.get("volume", 0)
            rules = contract.get("rules_primary", "")
            expiry = contract.get("close_time", contract.get("expiration_time", "?"))
            market_type = result["market_type"]
            competitors = result["competitors"]

            print(f"  LOCKED TARGET: {ticker}")
            print(f"  Market:  {result['event_title']}")
            print(f"  Option:  {option}")
            print(f"  Type:    {market_type.upper().replace('_', ' ')}")
            print(f"  Rules:   {rules[:120]}")
            print(f"  Bid/Ask: {yes_bid}c / {yes_ask}c  |  Last: {last}c  |  Vol: {vol:,}")
            print(f"  Closes:  {expiry}")

            if competitors:
                print()
                print(f"  COMPETITORS ({len(competitors)}):")
                for c in competitors:
                    c_bid = c.get("yes_bid", 0) or 0
                    c_ask = c.get("yes_ask", 0) or 0
                    c_last = c.get("last_price", 0) or 0
                    c_vol = c.get("volume", 0) or 0
                    print(
                        f"    {c['name']:<25s}  "
                        f"Bid/Ask: {c_bid}c/{c_ask}c  "
                        f"Last: {c_last}c  "
                        f"Vol: {c_vol:,}"
                    )
        else:
            print(
                f"  No streaming-specific market found for {artist_name}. "
                f"(Found {rejected} other event(s), but filtered them out for safety)."
            )
        print()

        return result

    except KalshiAuthError:
        logger.error("Kalshi auth failed — proceeding without live market")
        print(f"\n  Kalshi authentication failed — skipping market lookup.\n")
        return None
    except Exception as exc:
        logger.warning("Kalshi unavailable (%s) — proceeding without live market", exc)
        print(f"\n  Kalshi unavailable: {exc}\n")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_real_strategy(
    df: pd.DataFrame,
    cal: CalibrationResult,
    market_result: Optional[Dict[str, Any]],
    strike: float,
    competitor_data: list[Dict[str, Any]] | None = None,
    competitor_listeners: float | None = None,
    dry_run: bool = False,
) -> None:
    """
    Execute the pricing engine on the latest real data point.

    Handles two market types:
    - **binary**: Standard binary option — uses ``strike`` as-is.
    - **winner_take_all**: Relative market — the "strike" is the leading
      competitor's **live** listener count (fetched from Chartmetric API).

    Parameters
    ----------
    competitor_data : list of dict, optional
        Live competitor listener data from ``ChartmetricClient.get_competitors_data()``.
        Each dict has: name, cm_id, listeners, change_pct.
    competitor_listeners : float, optional
        Manual override for the leader's listener count (takes precedence).
    """
    logger.info("=" * 64)
    logger.info("STRATEGY ENGINE (mode=%s)",
                "DRY-RUN" if dry_run else "LIVE")
    logger.info("=" * 64)

    # ── 1. Latest data point ────────────────────────────────────────────
    latest = df.sort_values("Date").iloc[-1]
    date_str = str(latest["Date"])[:10]
    spot_official = float(latest["spotify_monthly_listeners"])
    tiktok_raw = float(latest.get("tiktok_sound_posts_change", 0) or 0)
    normalized_velocity = normalise_tiktok(tiktok_raw, cal.tiktok_p95)

    # ── 2. Detect market type & adjust strike ───────────────────────────
    market_type = "binary"
    kalshi_competitors: list[Dict[str, Any]] = []
    leader_name = None
    effective_strike = strike

    if market_result is not None:
        market_type = market_result.get("market_type", "binary")
        kalshi_competitors = market_result.get("competitors", [])

    if market_type == "winner_take_all":
        print()
        print("  " + "*" * 60)
        print("  WINNER-TAKE-ALL Market — Competitive Analysis")
        print("  " + "*" * 60)

        # ── 2a. Competitive Landscape Table ─────────────────────────
        if competitor_data:
            _print_competitive_landscape(
                competitor_data, kalshi_competitors,
                our_artist=ARTIST_NAME,
                our_listeners=spot_official,
            )
            # Leader = competitor with the most listeners
            leader = competitor_data[0]  # already sorted desc
            leader_name = leader["name"]
            effective_strike = float(leader["listeners"])
            gap = spot_official - effective_strike
            gap_pct = gap / effective_strike * 100 if effective_strike > 0 else 0

            print()
            print(f"  LEADER:  {leader_name} ({effective_strike:,.0f} listeners)")
            if gap >= 0:
                print(f"  GAP:     {ARTIST_NAME} is {gap:,.0f} AHEAD (+{gap_pct:.1f}%)")
            else:
                print(f"  GAP:     {ARTIST_NAME} is {abs(gap):,.0f} BEHIND ({gap_pct:.1f}%)")
            print(f"  STRIKE:  Using {leader_name} @ {effective_strike:,.0f} as effective strike")

        elif competitor_listeners is not None and competitor_listeners > 0:
            # Manual override
            effective_strike = competitor_listeners
            print(f"  Using manual --competitor-listeners as strike: {effective_strike:,.0f}")

        else:
            if kalshi_competitors:
                leader_name = kalshi_competitors[0]["name"]
                leader_price = kalshi_competitors[0].get("last_price", 0) or 0
                comp_names = [c["name"] for c in kalshi_competitors]
                print(f"  Competitors: {', '.join(comp_names)}")
                print(f"  Market Leader: {leader_name} ({leader_price}c)")
            print(f"\n  Could not fetch competitor data — using default strike: {effective_strike:,.0f}")

        print("  " + "*" * 60)

    # ── 2b. Event-Driven Volatility (3-day lookahead) ──────────────────
    latest_date = pd.Timestamp(latest["Date"])
    lookahead_end = latest_date + pd.Timedelta(days=3)
    event_window = df[
        (df["Date"] >= latest_date) & (df["Date"] <= lookahead_end)
    ].copy() if "event_impact_score" in df.columns else pd.DataFrame()

    if not event_window.empty and "event_impact_score" in event_window.columns:
        event_impact_score = float(event_window["event_impact_score"].max())
        peak_row = event_window.loc[event_window["event_impact_score"].idxmax()]
        if event_impact_score >= 3.0:
            event_name = str(peak_row.get("releases", ""))[:80] or "Album/Single Drop"
        elif event_impact_score >= 2.0:
            event_name = str(peak_row.get("events", ""))[:80] or "Major Event"
        elif event_impact_score > 1.0:
            event_name = str(peak_row.get("events", ""))[:80] or "Concert/Appearance"
        else:
            event_name = None
    else:
        event_impact_score = 1.0
        event_name = None

    if event_name and event_impact_score > 1.0:
        print()
        print(f"  MAJOR EVENT DETECTED: {event_name}")
        print(f"  Boosting Volatility by {event_impact_score:.1f}x.")
        print()

    # ── 3. Market price (live or theoretical) ───────────────────────────
    if market_result is not None:
        contract = market_result["target_contract"]
        yes_bid = contract.get("yes_bid", 0) or 0
        yes_ask = contract.get("yes_ask", 0) or 0
        last = contract.get("last_price", 0) or 0

        if yes_bid and yes_ask:
            mid_cents = (yes_bid + yes_ask) / 2.0
        else:
            mid_cents = float(last)

        market_price = mid_cents / 100.0
        market_ticker = contract.get("ticker", "UNKNOWN")
        market_source = f"LIVE ({market_ticker})"

        close_time = contract.get(
            "close_time", contract.get("expiration_time"),
        )
        T = compute_time_to_expiry(close_time)
    else:
        market_price = 0.50
        market_source = "THEORETICAL (no live contract found)"
        T = 1 / 365

    # ── 4. Pricing ──────────────────────────────────────────────────────
    pricer = JumpDiffusionPricer()

    spot_nowcast = pricer.nowcast_spot_calibrated(
        spot_official, normalized_velocity, cal.jump_beta,
    )
    sigma_adj = pricer.adjusted_sigma_calibrated(
        cal.sigma, normalized_velocity, cal.vol_gamma, event_impact_score,
    )
    fair_value = pricer.fair_value_calibrated(
        spot_official=spot_official,
        normalized_velocity=normalized_velocity,
        strike=effective_strike,
        sigma=cal.sigma,
        jump_beta=cal.jump_beta,
        vol_gamma=cal.vol_gamma,
        event_impact_score=event_impact_score,
        T=T,
    )
    edge = fair_value - market_price
    signal = format_signal(edge, threshold=EDGE_THRESHOLD)

    # ── 5. Implied Volatility (what does the MARKET think?) ──────────
    implied_sigma = pricer.implied_vol(
        market_price=market_price,
        spot=spot_nowcast,
        strike=effective_strike,
        T=T,
    )
    iv_str = f"{implied_sigma:.1%}" if implied_sigma is not None else "N/A"
    vol_ratio = (
        f"{implied_sigma / sigma_adj:.1f}x"
        if implied_sigma is not None and sigma_adj > 0
        else "N/A"
    )

    # ── 6. Output ───────────────────────────────────────────────────────
    itm = spot_nowcast > effective_strike
    strike_label = (
        f"Competitor ({leader_name})" if market_type == "winner_take_all" and leader_name
        else "Strike (K)"
    )

    if fair_value >= 0.0001:
        fv_str = f"{fair_value:>18.4f}"
        fv_pct = f"{fair_value:.1%}"
    elif fair_value > 0:
        fv_str = f"{fair_value:>18.2e}"
        fv_pct = f"{fair_value:.2e}"
    else:
        fv_str = f"{'~0':>18s}"
        fv_pct = "~0%"

    print()
    print("  " + "=" * 62)
    print(f"  TAUROI REAL-DATA ENGINE — {ARTIST_NAME}")
    print(f"  Date: {date_str}  |  T = {T * 365:.1f} days to expiry")
    print(f"  Market Type: {market_type.upper().replace('_', ' ')}")
    print("  " + "=" * 62)
    print()
    print("  ┌──────────────────────────────────┬──────────────────────┐")
    print("  │  METRIC                           │  VALUE               │")
    print("  ├──────────────────────────────────┼──────────────────────┤")
    print(f"  │  Official Spot (Spotify)          │  {spot_official:>18,.0f}  │")
    print(f"  │  Nowcasted Spot (S_adj)           │  {spot_nowcast:>18,.0f}  │")
    print(f"  │  {strike_label:<33s} │  {effective_strike:>18,.0f}  │")
    print(f"  │  Spot vs Strike                   │  {'IN-THE-MONEY' if itm else 'OUT-OF-MONEY':>18s}  │")
    print("  ├──────────────────────────────────┼──────────────────────┤")
    print(f"  │  TikTok Velocity (EMA-3)          │  {tiktok_raw:>+18,.0f}  │")
    print(f"  │  TikTok Normalised Velocity       │  {normalized_velocity:>18.4f}  │")
    print(f"  │  Vol Gamma (conditional)          │  {cal.vol_gamma:>17.2%}  │")
    print(f"  │  Gamma Contribution               │  {cal.vol_gamma * normalized_velocity:>+17.2%}  │")
    event_label = f"{event_impact_score:.1f}x" + (f" ({event_name[:20]})" if event_name else "")
    print(f"  │  Event Impact                     │  {event_label:>18s}  │")
    print("  ├──────────────────────────────────┼──────────────────────┤")
    print(f"  │  Calibrated sigma (base)          │  {cal.sigma:>17.2%}  │")
    print(f"  │  Adjusted sigma (final)           │  {sigma_adj:>17.2%}  │")
    print(f"  │  Market Implied sigma             │  {iv_str:>18s}  │")
    print(f"  │  Implied / Model Vol Ratio        │  {vol_ratio:>18s}  │")
    print(f"  │  Jump beta (sensitivity)          │  {cal.jump_beta:>18.6f}  │")
    print("  ├──────────────────────────────────┼──────────────────────┤")
    print(f"  │  Model Fair Value (P)             │  {fv_str}  │")
    print(f"  │  Market Price (M)                 │  {market_price:>18.4f}  │")
    print(f"  │  Edge (P - M)                     │  {edge:>+18.4f}  │")
    print("  ├──────────────────────────────────┼──────────────────────┤")
    print(f"  │  Market Source                    │  {market_source:>18s}  │")
    print("  └──────────────────────────────────┴──────────────────────┘")
    print()
    print("  " + "-" * 62)

    if signal == "BUY":
        print(f"  >>> SIGNAL: BUY  (edge {edge:+.4f} > +{EDGE_THRESHOLD})")
        print(f"  >>> Market is CHEAP — model says {fv_pct}, market offers {market_price:.1%}")
    elif signal == "SELL":
        print(f"  >>> SIGNAL: SELL (edge {edge:+.4f} < -{EDGE_THRESHOLD})")
        print(f"  >>> Market is RICH — model says {fv_pct}, market offers {market_price:.1%}")
    else:
        print(f"  >>> SIGNAL: HOLD (|edge| = {abs(edge):.4f} < {EDGE_THRESHOLD})")
        print(f"  >>> No actionable edge — model and market roughly agree")

    print("  " + "-" * 62)

    if dry_run:
        logger.info("DRY-RUN complete — no orders placed.")
    else:
        logger.info("LIVE mode — order execution would happen here.")

    print()

    # ── 7. Persist signal to JSONL log ───────────────────────────────
    signal_record = {
        "artist": ARTIST_NAME,
        "date": date_str,
        "spot_official": int(spot_official),
        "spot_nowcast": int(spot_nowcast),
        "strike": int(effective_strike),
        "leader": leader_name,
        "market_type": market_type,
        "fair_value": round(fair_value, 6),
        "market_price": round(market_price, 4),
        "edge": round(edge, 6),
        "signal": signal,
        "sigma_base": round(cal.sigma, 6),
        "sigma_adj": round(sigma_adj, 6),
        "tiktok_velocity": round(tiktok_raw, 2),
        "norm_velocity": round(normalized_velocity, 4),
        "event_impact": round(event_impact_score, 1),
        "ticker": market_ticker if market_result else None,
        "T_days": round(T * 365, 1),
        "dry_run": dry_run,
    }
    append_signal_log(signal_record)

    return signal_record


# ═════════════════════════════════════════════════════════════════════════════
#  FULL MARKET SCAN
# ═════════════════════════════════════════════════════════════════════════════

def run_market_scan(
    cal: CalibrationResult,
    market_result: Dict[str, Any],
    all_artists: list[Dict[str, Any]],
    dry_run: bool = False,
) -> list[Dict[str, Any]]:
    """
    Price **every** contract in a Winner-Take-All market using
    **per-artist** data from identical sources.

    Each artist is priced with:
    - Their own Spotify listeners (spot)
    - Their own sigma (volatility calibrated from their own 90-day history)
    - Their own TikTok velocity (normalised per-artist)
    - Shared gamma/beta from the primary artist (calibration anchor)
    - Strike = max(all other artists' listeners)

    Parameters
    ----------
    cal : CalibrationResult
        Calibration from the primary artist — provides shared gamma/beta.
    market_result : dict
        Kalshi market data.
    all_artists : list of dict
        Per-artist enriched data from ``get_all_artists_scan_data()``.
        Each dict has: name, listeners, change_pct, sigma, norm_velocity,
        tiktok_velocity, spotify_followers, spotify_popularity, data_points.
    dry_run : bool
        If True, signals are informational only.
    """
    contract = market_result["target_contract"]
    close_time = contract.get("close_time", contract.get("expiration_time"))
    T = compute_time_to_expiry(close_time)
    event_title = market_result.get("event_title", "Unknown Market")

    # Build a Kalshi price lookup — merge target + competitors
    kalshi_lookup: Dict[str, Dict[str, Any]] = {}

    target_name = contract.get("yes_sub_title", contract.get("subtitle", ""))
    if target_name:
        kalshi_lookup[target_name] = contract

    for c in market_result.get("competitors", []):
        kalshi_lookup[c["name"]] = c

    pricer = JumpDiffusionPricer()
    scan_rows: list[Dict[str, Any]] = []

    for artist in all_artists:
        name = artist["name"]
        spot = float(artist["listeners"])
        change_pct = artist.get("change_pct")
        artist_sigma = float(artist.get("sigma", cal.sigma))
        norm_velocity = float(artist.get("norm_velocity", 0.0))

        # Strike = max listeners among all OTHER artists
        others_max = max(
            (a["listeners"] for a in all_artists if a["name"] != name),
            default=spot,
        )
        effective_strike = float(others_max)

        # Per-artist pricing: their OWN sigma + velocity, shared gamma/beta
        fair_value = pricer.fair_value_calibrated(
            spot_official=spot,
            normalized_velocity=norm_velocity,
            strike=effective_strike,
            sigma=artist_sigma,
            jump_beta=cal.jump_beta,
            vol_gamma=cal.vol_gamma,
            event_impact_score=1.0,
            T=T,
        )

        # Market price from Kalshi
        kc = kalshi_lookup.get(name, {})
        yes_bid = kc.get("yes_bid", 0) or 0
        yes_ask = kc.get("yes_ask", 0) or 0
        last = kc.get("last_price", 0) or 0

        if yes_bid and yes_ask:
            market_price = (yes_bid + yes_ask) / 2.0 / 100.0
        elif last:
            market_price = float(last) / 100.0
        else:
            market_price = 0.01  # default floor

        edge = fair_value - market_price
        signal = format_signal(edge, threshold=EDGE_THRESHOLD)

        scan_rows.append({
            "name": name,
            "listeners": int(spot),
            "change_pct": change_pct,
            "sigma": artist_sigma,
            "norm_velocity": norm_velocity,
            "tiktok_velocity": artist.get("tiktok_velocity", 0),
            "spotify_followers": artist.get("spotify_followers", 0),
            "spotify_popularity": artist.get("spotify_popularity", 0),
            "youtube_views": artist.get("youtube_views", 0),
            "instagram_followers": artist.get("instagram_followers", 0),
            "deezer_fans": artist.get("deezer_fans", 0),
            "wikipedia_views": artist.get("wikipedia_views", 0),
            "playlist_reach": artist.get("playlist_reach", 0),
            "num_playlists": artist.get("num_playlists", 0),
            "strike": int(effective_strike),
            "fair_value": round(fair_value, 4),
            "market_price": round(market_price, 4),
            "edge": round(edge, 4),
            "signal": signal,
            "kalshi_last": last,
            "ticker": kc.get("ticker"),
            "data_points": artist.get("data_points", 0),
            "history_start": artist.get("history_start"),
            "history_end": artist.get("history_end"),
        })

    # Sort by edge descending (best opportunities first)
    scan_rows.sort(key=lambda r: r["edge"], reverse=True)

    # ── Print the scan table ──────────────────────────────────────────
    print()
    print("  " + "=" * 100)
    print(f"  FULL MARKET SCAN — Per-Artist Calibration")
    print(f"  {event_title}")
    print(f"  T = {T * 365:.1f} days to expiry  |  Gamma (shared) = {cal.vol_gamma:.2%}  |  Beta (shared) = {cal.jump_beta:.4f}")
    print("  " + "=" * 100)
    print()
    print(f"  {'Artist':<22s} {'Listeners':>14s} {'7d Chg':>8s}"
          f" {'Sigma':>8s} {'TikTok':>8s}"
          f" {'Model':>8s} {'Kalshi':>8s} {'Edge':>8s}  {'Signal':<6s}")
    print("  " + "-" * 100)

    for row in scan_rows:
        chg = f"{row['change_pct']:+.1f}%" if row["change_pct"] is not None else "—"
        sigma_str = f"{row['sigma']:.0%}"
        vel_str = f"{row['norm_velocity']:.2f}" if row["norm_velocity"] > 0 else "—"
        fv_pct = f"{row['fair_value']:.1%}"
        mkt = f"{row['kalshi_last']}c" if row["kalshi_last"] else "—"
        edge_str = f"{row['edge']:+.1%}"
        star = " *" if row["signal"] == "BUY" else ""
        print(
            f"  {row['name']:<22s} {row['listeners']:>14,d} {chg:>8s}"
            f" {sigma_str:>8s} {vel_str:>8s}"
            f" {fv_pct:>8s} {mkt:>8s} {edge_str:>8s}  {row['signal']:<6s}{star}"
        )

    print("  " + "-" * 100)

    # Highlight the best opportunity
    buys = [r for r in scan_rows if r["signal"] == "BUY"]
    if buys:
        best = buys[0]
        print()
        if len(buys) == 1:
            print(f"  BEST EDGE:  {best['name']} ({best['edge']:+.1%})")
            print(f"  >>> BUY at {best['kalshi_last']}c — model says {best['fair_value']:.1%}")
        else:
            print(f"  ACTIONABLE BUYS ({len(buys)}):")
            for b in buys:
                print(
                    f"    {b['name']:<22s}  edge {b['edge']:+.1%}"
                    f"  (model {b['fair_value']:.1%} vs {b['kalshi_last']}c)"
                )
    else:
        print()
        print("  No actionable edges — all contracts fairly priced.")

    print("  " + "=" * 100)

    # ── Data consistency report ───────────────────────────────────────
    print()
    print("  DATA CONSISTENCY REPORT (identical sources per artist)")
    print("  " + "-" * 130)
    print(
        f"  {'Artist':<22s} {'Rows':>6s} {'Range':>19s}  {'Sigma':>7s}"
        f"  {'Followers':>12s} {'Pop':>4s}"
        f"  {'YouTube':>12s} {'Insta':>12s} {'Deezer':>10s} {'Wikipedia':>10s}"
    )
    print("  " + "-" * 130)
    for row in sorted(scan_rows, key=lambda r: r["listeners"], reverse=True):
        fol = f"{row.get('spotify_followers', 0):,}" if row.get("spotify_followers") else "—"
        pop = str(row.get("spotify_popularity", 0)) if row.get("spotify_popularity") else "—"
        yt = f"{row.get('youtube_views', 0):,}" if row.get("youtube_views") else "—"
        ig = f"{row.get('instagram_followers', 0):,}" if row.get("instagram_followers") else "—"
        dz = f"{row.get('deezer_fans', 0):,}" if row.get("deezer_fans") else "—"
        wiki = f"{row.get('wikipedia_views', 0):,}" if row.get("wikipedia_views") else "—"
        hist = f"{row.get('history_start', '?')}→{row.get('history_end', '?')}"
        print(
            f"  {row['name']:<22s} {row['data_points']:>6d} {hist:>19s}"
            f"  {row['sigma']:>6.1%}  {fol:>12s} {pop:>4s}"
            f"  {yt:>12s} {ig:>12s} {dz:>10s} {wiki:>10s}"
        )
    print("  " + "-" * 130)
    print()

    # ── Persist each signal ───────────────────────────────────────────
    for row in scan_rows:
        append_signal_log({
            "mode": "scan",
            "artist": row["name"],
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "spot_official": row["listeners"],
            "strike": row["strike"],
            "market_type": "winner_take_all",
            "fair_value": row["fair_value"],
            "market_price": row["market_price"],
            "edge": row["edge"],
            "signal": row["signal"],
            "sigma": round(row["sigma"], 6),
            "norm_velocity": row["norm_velocity"],
            "spotify_followers": row.get("spotify_followers", 0),
            "spotify_popularity": row.get("spotify_popularity", 0),
            "youtube_views": row.get("youtube_views", 0),
            "instagram_followers": row.get("instagram_followers", 0),
            "deezer_fans": row.get("deezer_fans", 0),
            "wikipedia_views": row.get("wikipedia_views", 0),
            "playlist_reach": row.get("playlist_reach", 0),
            "data_points": row.get("data_points", 0),
            "history_start": row.get("history_start"),
            "history_end": row.get("history_end"),
            "ticker": row.get("ticker"),
            "T_days": round(T * 365, 1),
            "dry_run": dry_run,
        })

    return scan_rows


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def print_live_dashboard(
    df: pd.DataFrame,
    cal: CalibrationResult,
    data_source: str,
    release_count: int = 0,
) -> None:
    """Print a compact live-status dashboard before the full strategy table."""
    sorted_df = df.sort_values("Date")
    latest = sorted_df.iloc[-1]
    spot = float(latest["spotify_monthly_listeners"])
    tiktok_raw = float(latest.get("tiktok_sound_posts_change", 0) or 0)
    norm_vel = normalise_tiktok(tiktok_raw, cal.tiktok_p95)

    start_date = sorted_df["Date"].iloc[0]
    end_date = sorted_df["Date"].iloc[-1]
    start_str = str(start_date)[:10]
    end_str = str(end_date)[:10]

    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    range_str = f"{start_str} to {end_str}"

    print()
    print("  " + "+" * 58)
    print("  |   TAUROI LIVE DASHBOARD                               |")
    print("  " + "+" * 58)
    print(f"  |   Timestamp:       {now_str:<36s}|")
    print(f"  |   Data Source:     {data_source:<36s}|")
    print(f"  |   Artist:          {ARTIST_NAME:<36s}|")
    print(f"  |   API Status:      {'Connected':<36s}|")
    print(f"  |   API Data Loaded: {range_str:<36s}|")
    print(f"  |   Total Rows:      {len(df):<36d}|")
    print(f"  |   Releases Found:  {release_count:<36d}|")
    print(f"  |   Live Spot:       {spot:>34,.0f} |")
    print(f"  |   Live Velocity:   {tiktok_raw:>+34,.0f} |")
    print(f"  |   Norm. Velocity:  {norm_vel:>34.4f} |")
    print(f"  |   Cal. sigma:      {cal.sigma:>33.2%} |")
    print(f"  |   Cal. vol_gamma:  {cal.vol_gamma:>33.2%} |")
    print("  " + "+" * 58)
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run(
    dry_run: bool = False,
    data_only: bool = False,
    scan: bool = False,
    strike: float = DEFAULT_STRIKE,
    competitor_listeners: float | None = None,
    artist_name: str | None = None,
    artist_id: str | None = None,
    no_cache: bool = False,
) -> None:
    """
    Full pipeline: fetch API data -> calibrate -> market -> strategy.

    Parameters
    ----------
    dry_run : bool
        If True, log signals without placing orders.
    data_only : bool
        If True, only fetch data & calibrate (skip Kalshi & strategy).
    scan : bool
        If True, discover the market and price EVERY contract (full scan).
    strike : float
        Binary option strike price.
    competitor_listeners : float, optional
        Competitor's listeners (for WTA markets).
    artist_name : str, optional
        Override the artist to analyse (defaults to ARTIST_NAME from config).
    artist_id : str, optional
        Override the Chartmetric artist ID (defaults to .env).
    no_cache : bool
        If True, skip the local Parquet cache and force a full API fetch.
    """
    global ARTIST_NAME

    need_secrets = not data_only
    settings = load_settings(require_secrets=need_secrets)

    # ── Resolve artist identity ──────────────────────────────────────
    if artist_name:
        ARTIST_NAME = artist_name
    else:
        ARTIST_NAME = settings.artist_name

    effective_artist_id = artist_id or settings.chartmetric_artist_id

    # Patch settings with overrides so the CM client uses the right ID
    if artist_id:
        # Build a new Settings with the overridden ID
        settings = Settings(
            kalshi_access_key=settings.kalshi_access_key,
            kalshi_api_secret=settings.kalshi_api_secret,
            chartmetric_refresh_token=settings.chartmetric_refresh_token,
            chartmetric_artist_id=effective_artist_id,
            artist_name=ARTIST_NAME,
        )

    # ── Step 1: Load Data from Chartmetric API ───────────────────────
    df, cm_client = load_data_from_api(settings)
    data_source = "Chartmetric API (full history)"
    release_count = getattr(cm_client, "_last_release_count", 0)

    # ── Step 2: Calibration (from in-memory DataFrame) ───────────────
    cal = run_calibration_from_df(df)

    # ── Step 3: Live Dashboard ───────────────────────────────────────
    print_live_dashboard(df, cal, data_source, release_count=release_count)

    if data_only:
        logger.info("--data-only set — skipping Kalshi & strategy.")
        return

    # ── Step 4: Kalshi Market Lookup ─────────────────────────────────
    market_result = fetch_kalshi_market(settings)

    if market_result is None:
        logger.warning("No market found — cannot proceed with strategy.")
        return

    # ── Step 5: Branch — SCAN or SINGLE-ARTIST ───────────────────────
    if scan:
        return _run_scan_pipeline(
            cm_client, cal, market_result, df, dry_run, no_cache,
        )
    else:
        return _run_single_artist_pipeline(
            cm_client, cal, market_result, df, strike,
            competitor_listeners, dry_run,
        )


def _run_scan_pipeline(
    cm_client: Any,
    cal: CalibrationResult,
    market_result: Dict[str, Any],
    df: pd.DataFrame,
    dry_run: bool,
    no_cache: bool = False,
) -> list[Dict[str, Any]]:
    """
    Full market scan with **per-artist consistent data**.

    For every artist in the Kalshi market:
    1.  Resolve name → Chartmetric ID (via search).
    2.  Fetch **full history** (since 2019) for Spotify, TikTok,
        YouTube, Shazam, Instagram, and Deezer.
    3.  Use local Parquet cache to avoid re-fetching (6h TTL).
    4.  Compute per-artist sigma from their own listener history.
    5.  Compute per-artist TikTok velocity.
    6.  Price using per-artist parameters + shared gamma/beta.
    """
    from src.chartmetric_client import DEFAULT_HISTORY_START

    # Collect all artist names from the market
    contract = market_result["target_contract"]
    target_name = contract.get("yes_sub_title", contract.get("subtitle", ""))
    comp_names = [c["name"] for c in market_result.get("competitors", [])]

    # Build a deduplicated list of ALL artist names in the market
    all_names: list[str] = []
    if target_name:
        all_names.append(target_name)
    for cn in comp_names:
        if cn not in all_names:
            all_names.append(cn)

    cache_status = "OFF (--no-cache)" if no_cache else "ON (6h TTL)"
    logger.info(
        "SCAN MODE — %d artists × full history (since %s) | cache=%s",
        len(all_names), DEFAULT_HISTORY_START, cache_status,
    )
    print(f"\n  Fetching full-history data for {len(all_names)} artists...")
    print(f"  Sources: Spotify (listeners/followers/popularity), TikTok, YouTube, Shazam, Instagram, Deezer")
    print(f"  Cache: {cache_status}")
    print("  (Identical sources for every artist — no calibration bias)\n")

    # Fetch consistent per-artist data with caching
    all_artists = cm_client.get_all_artists_scan_data(
        all_names,
        since=DEFAULT_HISTORY_START,
        use_cache=not no_cache,
    )

    if not all_artists:
        logger.error("Could not resolve any artists from Chartmetric.")
        return []

    cached = sum(1 for a in all_artists if a.get("data_points", 0) > 365)
    print(f"\n  Resolved {len(all_artists)} / {len(all_names)} artists "
          f"({cached} with deep history)\n")

    # Run the full scan with per-artist parameters
    scan_results = run_market_scan(
        cal=cal,
        market_result=market_result,
        all_artists=all_artists,
        dry_run=dry_run,
    )

    return scan_results


def _run_single_artist_pipeline(
    cm_client: Any,
    cal: CalibrationResult,
    market_result: Dict[str, Any],
    df: pd.DataFrame,
    strike: float,
    competitor_listeners: float | None,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Original single-artist deep dive with full calibration + TikTok signal.
    """
    # Competitor intelligence (auto-fetch for WTA markets)
    competitor_data: list[Dict[str, Any]] | None = None
    if (
        market_result.get("market_type") == "winner_take_all"
        and competitor_listeners is None
    ):
        comp_names = [c["name"] for c in market_result.get("competitors", [])]
        if comp_names:
            competitor_data = cm_client.get_competitors_data(comp_names)

    # Context-aware strategy engine
    signal_record = run_real_strategy(
        df, cal, market_result, strike,
        competitor_data=competitor_data,
        competitor_listeners=competitor_listeners,
        dry_run=dry_run,
    )

    return signal_record


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tauroi Prediction Engine — Cloud-Ready Production",
    )

    # ── Mode switches ────────────────────────────────────────────────
    parser.add_argument(
        "--scan",
        action="store_true",
        default=False,
        help="Full market scan — price EVERY contract in the market.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log signals without placing orders.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        default=False,
        help="Only run data loading + calibration (no Kalshi / strategy).",
    )

    # ── Artist overrides ─────────────────────────────────────────────
    parser.add_argument(
        "--artist",
        type=str,
        default=None,
        help=(
            "Artist name for market discovery & display "
            "(overrides ARTIST_NAME in .env). Example: --artist 'Taylor Swift'"
        ),
    )
    parser.add_argument(
        "--artist-id",
        type=str,
        default=None,
        help=(
            "Chartmetric numeric artist ID for full-history calibration "
            "(overrides CHARTMETRIC_ARTIST_ID in .env). "
            "Example: --artist-id 2762"
        ),
    )

    # ── Pricing overrides ────────────────────────────────────────────
    parser.add_argument(
        "--strike",
        type=float,
        default=DEFAULT_STRIKE,
        help=f"Binary option strike price (default: {DEFAULT_STRIKE:,.0f}).",
    )
    parser.add_argument(
        "--competitor-listeners",
        type=float,
        default=None,
        help=(
            "Current monthly listeners for the leading competitor. "
            "Used as the effective strike in Winner-Take-All markets. "
            "Example: --competitor-listeners 120e6"
        ),
    )

    # ── Cache control ────────────────────────────────────────────────
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help=(
            "Force a full API re-fetch, ignoring the local Parquet cache. "
            "By default, scan data is cached for 6 hours in cache/."
        ),
    )

    # ── Loop / scheduling ────────────────────────────────────────────
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Run continuously, re-executing every --interval seconds.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Seconds between runs when --loop is set (default: 900 = 15 min).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    def _single_run() -> None:
        """Execute one full pipeline cycle with top-level error handling."""
        try:
            run(
                dry_run=args.dry_run,
                data_only=args.data_only,
                scan=args.scan,
                strike=args.strike,
                competitor_listeners=args.competitor_listeners,
                artist_name=args.artist,
                artist_id=args.artist_id,
                no_cache=args.no_cache,
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.error("Pipeline failed:\n%s", traceback.format_exc())
            append_signal_log({
                "artist": ARTIST_NAME,
                "signal": "ERROR",
                "error": traceback.format_exc()[-500:],
            })
            if not args.loop:
                sys.exit(1)

    if args.loop:
        logger.info(
            "LOOP MODE — running every %d seconds (Ctrl+C to stop)",
            args.interval,
        )
        while True:
            _single_run()
            logger.info(
                "Next run in %d seconds...", args.interval,
            )
            _time.sleep(args.interval)
    else:
        _single_run()


if __name__ == "__main__":
    main()
