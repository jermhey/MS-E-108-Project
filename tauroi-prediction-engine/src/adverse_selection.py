"""
adverse_selection.py — Adverse-Selection Detection & Quote Pulling
===================================================================
Detects when an informed trader is likely moving the market so the
market-maker can **pull** its resting quotes before accumulating
toxic inventory.

Two detection layers:

**Reactive (price-action)**
    A sudden large move in the Kalshi logit series (z-score > threshold)
    signals that someone with private information just traded.  When
    triggered we cancel all resting orders for the affected ticker and
    wait for the move to settle before requoting.

**Predictive (Spotify monitor)**
    Checks the public Spotify Web API for leading indicators of
    listener-count changes:
      - New single / album releases (biggest mover)
      - Popularity-score jumps (updates faster than monthly listeners)
    When a release or popularity spike is detected we pre-emptively
    widen or pull quotes on that artist's contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("tauroi.adverse_selection")

# ── Reactive Detection (price action) ────────────────────────────────────────

Z_SCORE_PULL_THRESHOLD = 2.5   # pull quotes at 2.5σ
Z_SCORE_WIDEN_THRESHOLD = 1.8  # widen spread at 1.8σ
WIDEN_MULTIPLIER = 2.0         # double the half-spread when widening


@dataclass
class AdverseSignal:
    """Result of adverse-selection check for a single ticker."""
    ticker: str
    triggered: bool        # True if any threshold was crossed
    action: str            # "pull" | "widen" | "none"
    z_score: float         # magnitude of the recent move
    lookback: int          # number of observations used
    detail: str = ""       # human-readable explanation


def detect_from_price_action(
    hf_df: pd.DataFrame,
    sigma_b: float,
    ticker: str = "",
    lookback: int = 5,
) -> AdverseSignal:
    """
    Detect adverse selection from recent Kalshi price moves.

    Compares the most recent *lookback*-period return in log-odds
    to the belief volatility.  A z-score above the threshold
    indicates likely informed trading.
    """
    default = AdverseSignal(
        ticker=ticker, triggered=False, action="none",
        z_score=0.0, lookback=lookback,
    )

    if hf_df is None or hf_df.empty or len(hf_df) < lookback + 2:
        return default

    logits = hf_df["logit"].values
    recent_move = logits[-1] - logits[-lookback]

    if len(hf_df) > 1:
        ts = hf_df["timestamp"].values
        dt_sec = float(np.median(np.diff(ts.astype(np.int64) // 10**9)))
        dt_sec = max(dt_sec, 1.0)
    else:
        dt_sec = 86400.0

    dt_years = dt_sec / (365.25 * 86400)
    sigma_period = sigma_b * math.sqrt(dt_years * lookback)

    if sigma_period < 1e-6:
        return default

    z = recent_move / sigma_period

    if abs(z) >= Z_SCORE_PULL_THRESHOLD:
        return AdverseSignal(
            ticker=ticker, triggered=True, action="pull",
            z_score=float(z), lookback=lookback,
            detail=f"|z|={abs(z):.1f} >= {Z_SCORE_PULL_THRESHOLD} — pull all quotes",
        )
    elif abs(z) >= Z_SCORE_WIDEN_THRESHOLD:
        return AdverseSignal(
            ticker=ticker, triggered=True, action="widen",
            z_score=float(z), lookback=lookback,
            detail=f"|z|={abs(z):.1f} >= {Z_SCORE_WIDEN_THRESHOLD} — widen spread",
        )

    return default


# ── Predictive Detection (Spotify monitor) ───────────────────────────────────

# Mapping from Kalshi ticker suffix to Spotify artist ID.
# Extend as needed; these cover the main KXTOPMONTHLY artists.
TICKER_TO_SPOTIFY_ID: Dict[str, str] = {
    "BRU": "0du5cEVh5yTK9QJze8zA0C",   # Bruno Mars
    "BAD": "4q3ewBCX7sLwd24euuV69X",   # Bad Bunny
    "TAY": "06HL4z0CvFAxyc27GXpf02",   # Taylor Swift
    "WEE": "1Xyo4u8uXC1ZmMpatF05PJ",   # The Weeknd
    "DRA": "3TVXtAsR1Inumwj472S9r4",   # Drake
    "BIL": "6qqNVTkY8uBg9cP3Jd7DAH",   # Billie Eilish
    "ARI": "66CXWjxzNUsdJxJ2JdwvnR",   # Ariana Grande
    "JUC": "2RdwBSPQiwcmiDo9kixcl8",   # Juice WRLD
    "EDM": "6eUKZXaKkcviH0Ku9w2n3V",   # Ed Sheeran
    "RIH": "5pKCCKE2ajJHZ9KAiaK11H",   # Rihanna
}


@dataclass
class SpotifyAlert:
    """Alert from the Spotify monitor for a single artist."""
    ticker_suffix: str
    artist_name: str
    alert_type: str        # "new_release" | "popularity_spike" | "none"
    detail: str = ""


def check_spotify_releases(
    spotify_token: str | None = None,
) -> list[SpotifyAlert]:
    """
    Check the Spotify Web API for new releases by tracked artists.

    Uses the public ``/artists/{id}/albums`` endpoint.  If no token
    is provided, this is a no-op (returns empty list).
    """
    if not spotify_token:
        return []

    import datetime
    alerts: list[SpotifyAlert] = []
    headers = {"Authorization": f"Bearer {spotify_token}"}

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — Spotify monitor disabled")
        return []

    today = datetime.date.today()
    lookback = today - datetime.timedelta(days=3)

    for suffix, artist_id in TICKER_TO_SPOTIFY_ID.items():
        try:
            resp = requests.get(
                f"https://api.spotify.com/v1/artists/{artist_id}/albums",
                headers=headers,
                params={"include_groups": "single,album", "limit": 5},
                timeout=5,
            )
            if resp.status_code != 200:
                continue

            for album in resp.json().get("items", []):
                release_str = album.get("release_date", "")
                if len(release_str) >= 10:
                    try:
                        rd = datetime.date.fromisoformat(release_str[:10])
                    except ValueError:
                        continue
                    if rd >= lookback:
                        alerts.append(SpotifyAlert(
                            ticker_suffix=suffix,
                            artist_name=album.get("artists", [{}])[0].get("name", "?"),
                            alert_type="new_release",
                            detail=f"{album.get('name', '?')} released {release_str}",
                        ))
        except Exception as exc:
            logger.debug("Spotify check failed for %s: %s", suffix, exc)

    return alerts
