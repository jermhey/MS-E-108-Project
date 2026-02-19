"""
belief_data.py — High-Frequency Kalshi Data Fetcher
=====================================================
Fetches tick-level trades and 1-minute candlesticks for KXTOPMONTHLY
contracts, with Parquet caching to avoid redundant API calls.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.kalshi_client import KalshiClient
from src.utils import get_logger

logger = get_logger("tauroi.belief_data")

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "cache" / "kalshi_hf"

PRICE_CLIP_LO = 0.01
PRICE_CLIP_HI = 0.99


def _ensure_cache_dir() -> pathlib.Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(ticker: str, kind: str) -> pathlib.Path:
    safe = ticker.replace("/", "_")
    return _ensure_cache_dir() / f"{safe}_{kind}.parquet"


# ── Tick-Level Trades ────────────────────────────────────────────────────────

def fetch_trades(
    client: KalshiClient,
    ticker: str,
    max_pages: int = 30,
    delay: float = 0.12,
) -> pd.DataFrame:
    """Paginate through all trades for *ticker* and return a DataFrame."""
    all_trades: List[Dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(max_pages):
        trades, cursor = client.get_market_trades(ticker, limit=1000, cursor=cursor)
        all_trades.extend(trades)
        if not cursor or not trades:
            break
        time.sleep(delay)

    if not all_trades:
        return pd.DataFrame(columns=["timestamp", "mid_price", "volume"])

    rows = []
    for t in all_trades:
        ts_str = t.get("created_time", "")
        price_cents = t.get("yes_price", t.get("price", 0))
        vol = t.get("count", t.get("volume", 1))
        if not ts_str or price_cents == 0:
            continue
        rows.append({
            "timestamp": pd.Timestamp(ts_str).tz_localize(None),
            "mid_price": price_cents / 100.0,
            "volume": int(vol),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["mid_price"] = df["mid_price"].clip(PRICE_CLIP_LO, PRICE_CLIP_HI)
    df["logit"] = np.log(df["mid_price"] / (1.0 - df["mid_price"]))
    return df


# ── 1-Minute Candlesticks ───────────────────────────────────────────────────

def fetch_candles_1m(
    client: KalshiClient,
    ticker: str,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Fetch 1-minute candlesticks for the last *lookback_days*."""
    now_ts = int(time.time())
    start_ts = now_ts - lookback_days * 86400

    # Kalshi may limit the window per request; chunk into 7-day windows.
    frames: List[pd.DataFrame] = []
    chunk = 7 * 86400

    t = start_ts
    while t < now_ts:
        end = min(t + chunk, now_ts)
        try:
            candles = client.get_market_candlesticks(
                ticker, start_ts=t, end_ts=end, period_interval=1,
            )
        except Exception as exc:
            logger.warning("1-min candle fetch failed for %s: %s", ticker, exc)
            candles = []

        if candles:
            rows = []
            for c in candles:
                ts = c.get("end_period_ts", 0)
                close = c.get("close", 0)
                vol = c.get("volume", 0)
                if ts == 0 or close == 0:
                    continue
                rows.append({
                    "timestamp": pd.Timestamp(ts, unit="s"),
                    "mid_price": close / 100.0,
                    "volume": int(vol),
                })
            if rows:
                frames.append(pd.DataFrame(rows))

        t = end
        time.sleep(0.12)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "mid_price", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["mid_price"] = df["mid_price"].clip(PRICE_CLIP_LO, PRICE_CLIP_HI)
    df["logit"] = np.log(df["mid_price"] / (1.0 - df["mid_price"]))
    return df


# ── Unified Fetcher with Caching ────────────────────────────────────────────

def fetch_hf_data(
    client: KalshiClient,
    ticker: str,
    prefer_trades: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch the highest-frequency data available for *ticker*.

    Tries tick-level trades first (finest granularity).  Falls back to
    1-minute candles if trade data is sparse.  Caches results as Parquet.
    """
    kind = "trades" if prefer_trades else "candles1m"
    cache = _cache_path(ticker, kind)

    cached: pd.DataFrame | None = None
    if use_cache and cache.exists():
        try:
            cached = pd.read_parquet(cache)
            logger.info("Cache hit for %s (%s): %d rows", ticker, kind, len(cached))
        except Exception:
            cached = None

    if prefer_trades:
        fresh = fetch_trades(client, ticker)
    else:
        fresh = fetch_candles_1m(client, ticker)

    if cached is not None and not fresh.empty:
        df = pd.concat([cached, fresh], ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    elif cached is not None:
        df = cached
    else:
        df = fresh

    if not df.empty and use_cache:
        df.to_parquet(cache, index=False)

    logger.info("HF data for %s: %d rows (%s)", ticker, len(df), kind)
    return df


# ── Discover & Fetch All Active Contracts ───────────────────────────────────

def discover_tickers(client: KalshiClient) -> List[Dict[str, Any]]:
    """Return all KXTOPMONTHLY market tickers (open + recently settled)."""
    tickers: List[Dict[str, Any]] = []

    for status in ("open", "settled"):
        events = client.get_all_events(status=status, series_ticker="KXTOPMONTHLY")
        for ev in events:
            event_ticker = ev.get("event_ticker", "")
            markets = ev.get("markets", [])
            if not markets:
                # Events endpoint may not include nested markets;
                # fetch the single event with nested markets.
                try:
                    full_ev = client.get_event(event_ticker)
                    markets = full_ev.get("markets", [])
                except Exception as exc:
                    logger.warning("Could not fetch markets for %s: %s", event_ticker, exc)
                    continue

            for mkt in markets:
                t = mkt.get("ticker", "")
                if t:
                    tickers.append({
                        "ticker": t,
                        "event_ticker": event_ticker,
                        "title": mkt.get("title", mkt.get("subtitle", "")),
                        "status": status,
                        "close_time": mkt.get("close_time", ""),
                    })

    logger.info("Discovered %d KXTOPMONTHLY tickers", len(tickers))
    return tickers


def fetch_all_hf_data(
    client: KalshiClient,
    max_tickers: int = 100,
    prefer_trades: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Discover all KXTOPMONTHLY contracts and fetch HF data for each.

    Returns a dict keyed by ticker.
    """
    infos = discover_tickers(client)
    results: Dict[str, pd.DataFrame] = {}

    for i, info in enumerate(infos[:max_tickers]):
        ticker = info["ticker"]
        logger.info(
            "[%d/%d] Fetching HF data for %s …",
            i + 1, len(infos), ticker,
        )
        try:
            df = fetch_hf_data(client, ticker, prefer_trades=prefer_trades)
            if not df.empty:
                results[ticker] = df
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
        time.sleep(0.15)

    logger.info(
        "Fetched HF data for %d / %d tickers",
        len(results), len(infos),
    )
    return results
