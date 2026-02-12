"""
data_loader.py — Mock Data Layer
==================================
Provides deterministic mock snapshots for tests and Phase 1 dry-run.

All live / backtest data is now fetched from the Chartmetric API
(see ``src/chartmetric_client.py``).  This module exists solely
for the unit-test fixtures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from src.utils import get_logger

logger = get_logger("tauroi.data_loader")


# ═════════════════════════════════════════════════════════════════════════════
#  MOCK DATA — deterministic fixtures (for tests / Phase 1 dry-run)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ArtistMarketSnapshot:
    """Single point-in-time observation for one artist-market pair."""

    artist_id: str
    spotify_listeners_daily: int
    playlist_reach_delta: int
    tiktok_track_posts_velocity: float  # normalised 0.0 – 1.0
    kalshi_strike_price: int
    kalshi_market_price: float          # 0.00 – 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.tiktok_track_posts_velocity <= 1.0:
            raise ValueError(
                f"tiktok_track_posts_velocity must be in [0, 1], "
                f"got {self.tiktok_track_posts_velocity}"
            )
        if not 0.0 <= self.kalshi_market_price <= 0.99:
            raise ValueError(
                f"kalshi_market_price must be in [0, 0.99], "
                f"got {self.kalshi_market_price}"
            )


MOCK_SNAPSHOTS: List[ArtistMarketSnapshot] = [
    # Scenario A — Low social velocity, market fairly priced
    ArtistMarketSnapshot(
        artist_id="taylor-swift-001",
        spotify_listeners_daily=102_000_000,
        playlist_reach_delta=150_000,
        tiktok_track_posts_velocity=0.35,
        kalshi_strike_price=105_000_000,
        kalshi_market_price=0.40,
    ),
    # Scenario B — High social velocity (viral), market has NOT repriced
    ArtistMarketSnapshot(
        artist_id="taylor-swift-001",
        spotify_listeners_daily=102_000_000,
        playlist_reach_delta=800_000,
        tiktok_track_posts_velocity=0.92,
        kalshi_strike_price=105_000_000,
        kalshi_market_price=0.42,
    ),
    # Scenario C — Post-viral cooldown, market still elevated
    ArtistMarketSnapshot(
        artist_id="taylor-swift-001",
        spotify_listeners_daily=106_000_000,
        playlist_reach_delta=-50_000,
        tiktok_track_posts_velocity=0.15,
        kalshi_strike_price=105_000_000,
        kalshi_market_price=0.80,
    ),
]


def load_mock_snapshots() -> List[ArtistMarketSnapshot]:
    """Return the deterministic fixture set (test-safe)."""
    return list(MOCK_SNAPSHOTS)


def generate_random_snapshot(
    artist_id: str = "taylor-swift-001",
    strike: int = 105_000_000,
    seed: int | None = None,
) -> ArtistMarketSnapshot:
    """Generate a single randomised snapshot — useful for Monte-Carlo sims."""
    rng = random.Random(seed)
    base_listeners = rng.randint(95_000_000, 110_000_000)
    velocity = round(rng.random(), 4)
    return ArtistMarketSnapshot(
        artist_id=artist_id,
        spotify_listeners_daily=base_listeners,
        playlist_reach_delta=rng.randint(-200_000, 1_000_000),
        tiktok_track_posts_velocity=velocity,
        kalshi_strike_price=strike,
        kalshi_market_price=round(rng.uniform(0.10, 0.90), 2),
    )


