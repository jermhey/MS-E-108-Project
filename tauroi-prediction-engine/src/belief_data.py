"""
belief_data.py — High-Frequency Kalshi Data Fetcher
=====================================================
Fetches tick-level trades and 1-minute candlesticks for KXTOPMONTHLY
contracts, with Parquet caching and **incremental fetching** to avoid
redundant API calls on tight polling loops.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.kalshi_client import KalshiClient
from src.utils import get_logger

logger = get_logger("tauroi.belief_data")

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "cache" / "kalshi_hf"

PRICE_CLIP_LO = 0.01
PRICE_CLIP_HI = 0.99

MIN_TRADES_FOR_SIGNAL = 15


def _ensure_cache_dir() -> pathlib.Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(ticker: str, kind: str) -> pathlib.Path:
    safe = ticker.replace("/", "_")
    return _ensure_cache_dir() / f"{safe}_{kind}.parquet"


def _parse_trades(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw API trade dicts into a clean DataFrame."""
    rows = []
    for t in raw_trades:
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

    if not rows:
        return pd.DataFrame(columns=["timestamp", "mid_price", "volume", "logit"])

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["mid_price"] = df["mid_price"].clip(PRICE_CLIP_LO, PRICE_CLIP_HI)
    df["logit"] = np.log(df["mid_price"] / (1.0 - df["mid_price"]))
    return df


# ── Tick-Level Trades ────────────────────────────────────────────────────────

def fetch_trades(
    client: KalshiClient,
    ticker: str,
    max_pages: int = 30,
    delay: float = 0.12,
    min_ts: int | None = None,
) -> pd.DataFrame:
    """
    Paginate through trades for *ticker*.

    If *min_ts* is given, only fetches trades after that Unix timestamp,
    enabling fast incremental updates.
    """
    all_trades: List[Dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(max_pages):
        trades, cursor = client.get_market_trades(
            ticker, limit=1000, cursor=cursor, min_ts=min_ts,
        )
        all_trades.extend(trades)
        if not cursor or not trades:
            break
        time.sleep(delay)

    return _parse_trades(all_trades)


# ── 1-Minute Candlesticks ───────────────────────────────────────────────────

def fetch_candles_1m(
    client: KalshiClient,
    ticker: str,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Fetch 1-minute candlesticks for the last *lookback_days*."""
    now_ts = int(time.time())
    start_ts = now_ts - lookback_days * 86400

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


# ── Unified Fetcher with Incremental Caching ────────────────────────────────

def fetch_hf_data(
    client: KalshiClient,
    ticker: str,
    prefer_trades: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch HF data with **incremental updates**.

    On the first call (cold cache), fetches everything.  On subsequent
    calls, only fetches trades newer than the latest cached timestamp,
    then merges.  This keeps each loop iteration fast (~1 API call per
    ticker instead of 10-20).
    """
    kind = "trades" if prefer_trades else "candles1m"
    cache = _cache_path(ticker, kind)

    cached: pd.DataFrame | None = None
    if use_cache and cache.exists():
        try:
            cached = pd.read_parquet(cache)
        except Exception:
            cached = None

    if prefer_trades:
        min_ts: int | None = None
        if cached is not None and not cached.empty:
            latest = pd.to_datetime(cached["timestamp"]).max()
            min_ts = int(latest.timestamp()) + 1
            logger.debug(
                "Incremental fetch for %s: trades after %s",
                ticker, latest.isoformat(),
            )

        fresh = fetch_trades(client, ticker, min_ts=min_ts)
    else:
        fresh = fetch_candles_1m(client, ticker)

    if cached is not None and not fresh.empty:
        df = pd.concat([cached, fresh], ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    elif cached is not None:
        df = cached
    else:
        df = fresh

    # Recompute logit in case it was missing from old cache
    if not df.empty and "logit" not in df.columns:
        df["mid_price"] = df["mid_price"].clip(PRICE_CLIP_LO, PRICE_CLIP_HI)
        df["logit"] = np.log(df["mid_price"] / (1.0 - df["mid_price"]))

    if not df.empty and use_cache:
        df.to_parquet(cache, index=False)

    n_new = len(fresh) if not fresh.empty else 0
    n_total = len(df)
    if n_new > 0:
        logger.info("HF %s: %d new trades, %d total", ticker, n_new, n_total)
    else:
        logger.info("HF %s: no new trades (%d cached)", ticker, n_total)

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
