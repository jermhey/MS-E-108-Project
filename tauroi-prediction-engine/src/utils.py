"""
utils.py — Shared helpers for Tauroi Prediction Engine
=======================================================
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


# ── structured logger ───────────────────────────────────────────────────────

def get_logger(name: str = "tauroi", level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently-formatted logger.

    Format: ``[2026-02-10 08:15:23 UTC] [INFO] module — message``
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s UTC] [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        formatter.converter = lambda *_: datetime.now(timezone.utc).timetuple()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# ── normalisation helpers ───────────────────────────────────────────────────

def normalise_velocity(raw_posts: int, max_posts: int = 100_000) -> float:
    """
    Map an absolute TikTok post count to the [0, 1] velocity score.

    Parameters
    ----------
    raw_posts : int
        Absolute number of new TikTok posts featuring the track.
    max_posts : int
        Saturation cap — values above this map to 1.0.

    Returns
    -------
    float
        Normalised velocity ∈ [0.0, 1.0].
    """
    if raw_posts < 0:
        raise ValueError(f"raw_posts cannot be negative, got {raw_posts}")
    return min(raw_posts / max_posts, 1.0)


def kalshi_maker_fee(count: int, price: float) -> float:
    """
    Kalshi maker fee per the official schedule.

    Formula: 0.0175 * count * P * (1 - P)

    Returns fee in **dollars** (not cents).
    """
    p = max(0.01, min(0.99, price))
    return 0.0175 * count * p * (1.0 - p)


def kalshi_taker_fee(count: int, price: float) -> float:
    """
    Kalshi taker fee per the official schedule.

    Formula: 0.07 * count * P * (1 - P)

    Returns fee in **dollars** (not cents).
    """
    p = max(0.01, min(0.99, price))
    return 0.07 * count * p * (1.0 - p)


def format_signal(edge: float, threshold: float = 0.05) -> str:
    """
    Convert a numerical edge into a human-readable trading signal.

    Returns one of ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
    """
    if edge > threshold:
        return "BUY"
    elif edge < -threshold:
        return "SELL"
    return "HOLD"


def format_edge_report(
    artist_id: str,
    fair_value: float,
    market_price: float,
    edge: float,
    signal: str,
    velocity: float,
    sigma: float,
    spot_nowcast: float,
) -> str:
    """Return a multi-line report string suitable for logging / stdout."""
    lines = [
        "=" * 64,
        f"  TAUROI EDGE REPORT — {artist_id}",
        "=" * 64,
        f"  Nowcast Spot (S)      : {spot_nowcast:>15,.0f}",
        f"  TikTok Velocity       : {velocity:>15.4f}",
        f"  Adjusted Sigma        : {sigma:>15.4f}",
        f"  Model Fair Value (P)  : {fair_value:>15.4f}",
        f"  Kalshi Market Price   : {market_price:>15.4f}",
        f"  Edge (P − M)          : {edge:>+15.4f}",
        "-" * 64,
        f"  >>> SIGNAL: {signal}",
        "=" * 64,
    ]
    return "\n".join(lines)
