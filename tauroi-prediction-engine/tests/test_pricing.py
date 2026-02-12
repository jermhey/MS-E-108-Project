"""
test_pricing.py — Unit Tests for the Jump-Diffusion Pricing Engine
===================================================================
Core invariant under test:
    **Higher TikTok velocity ⟹ higher fair value** (all else equal).

This is the fundamental economic thesis: social virality is a leading
indicator of the settlement number, so the model must monotonically
increase its probability estimate as velocity rises.
"""

from __future__ import annotations

import math
import unittest

from src.data_loader import (
    ArtistMarketSnapshot,
    load_mock_snapshots,
    generate_random_snapshot,
)
from src.pricing_engine import JumpDiffusionPricer, ModelParams
from src.utils import normalise_velocity, format_signal


class TestNowcastSpot(unittest.TestCase):
    """Verify S_now = S_official + velocity * correlation_factor."""

    def setUp(self) -> None:
        self.pricer = JumpDiffusionPricer()

    def test_zero_velocity_returns_official(self):
        S = self.pricer.nowcast_spot(100_000_000, 0.0)
        self.assertEqual(S, 100_000_000)

    def test_positive_velocity_increases_spot(self):
        S = self.pricer.nowcast_spot(100_000_000, 0.5)
        self.assertGreater(S, 100_000_000)

    def test_nowcast_formula_exact(self):
        params = ModelParams(correlation_factor=2_000_000)
        pricer = JumpDiffusionPricer(params)
        S = pricer.nowcast_spot(100_000_000, 0.75)
        self.assertAlmostEqual(S, 100_000_000 + 0.75 * 2_000_000)


class TestAdjustedSigma(unittest.TestCase):
    """Verify jump-regime volatility switching."""

    def setUp(self) -> None:
        self.pricer = JumpDiffusionPricer()

    def test_below_threshold_returns_base(self):
        sigma = self.pricer.adjusted_sigma(0.5)
        self.assertEqual(sigma, self.pricer.params.sigma_base)

    def test_above_threshold_returns_jump(self):
        sigma = self.pricer.adjusted_sigma(0.9)
        expected = self.pricer.params.sigma_base * self.pricer.params.jump_multiplier
        self.assertAlmostEqual(sigma, expected)

    def test_at_threshold_returns_base(self):
        """Boundary: velocity == threshold should NOT trigger jump."""
        sigma = self.pricer.adjusted_sigma(0.80)
        self.assertEqual(sigma, self.pricer.params.sigma_base)


class TestFairValue(unittest.TestCase):
    """Core thesis: higher velocity ⟹ higher price (monotonicity)."""

    def setUp(self) -> None:
        self.pricer = JumpDiffusionPricer()
        self.spot = 102_000_000
        self.strike = 105_000_000

    def test_high_velocity_increases_fair_value(self):
        """THE key test — proves our alpha hypothesis in code."""
        fv_low = self.pricer.fair_value(self.spot, 0.30, self.strike)
        fv_high = self.pricer.fair_value(self.spot, 0.92, self.strike)
        self.assertGreater(
            fv_high,
            fv_low,
            "Fair value must increase with higher TikTok velocity",
        )

    def test_fair_value_in_unit_interval(self):
        """Binary option price must be a probability ∈ [0, 1]."""
        for v in [0.0, 0.25, 0.50, 0.75, 0.99]:
            fv = self.pricer.fair_value(self.spot, v, self.strike)
            self.assertGreaterEqual(fv, 0.0)
            self.assertLessEqual(fv, 1.0)

    def test_deep_in_the_money(self):
        """Spot >> Strike ⟹ fair value ≈ 1."""
        fv = self.pricer.fair_value(120_000_000, 0.5, 100_000_000)
        self.assertGreater(fv, 0.90)

    def test_deep_out_of_the_money(self):
        """Spot << Strike ⟹ fair value ≈ 0."""
        fv = self.pricer.fair_value(80_000_000, 0.1, 105_000_000)
        self.assertLess(fv, 0.10)

    def test_discount_factor_applied(self):
        """With T → 0 the discount is negligible, but still present."""
        params = ModelParams(time_to_expiry=1 / 365, risk_free_rate=0.05)
        pricer = JumpDiffusionPricer(params)
        fv = pricer.fair_value(110_000_000, 0.5, 100_000_000)
        # Discount over one day is tiny but should still pull price below 1.0
        self.assertLess(fv, 1.0)


class TestEdge(unittest.TestCase):
    """Verify edge = fair_value − market_price."""

    def setUp(self) -> None:
        self.pricer = JumpDiffusionPricer()

    def test_positive_edge(self):
        edge = self.pricer.compute_edge(
            spot_official=110_000_000,
            tiktok_velocity=0.5,
            strike=105_000_000,
            market_price=0.10,
        )
        self.assertGreater(edge, 0.0)

    def test_negative_edge(self):
        edge = self.pricer.compute_edge(
            spot_official=90_000_000,
            tiktok_velocity=0.1,
            strike=105_000_000,
            market_price=0.90,
        )
        self.assertLess(edge, 0.0)


class TestUtils(unittest.TestCase):
    """Test helper functions."""

    def test_normalise_velocity_zero(self):
        self.assertAlmostEqual(normalise_velocity(0), 0.0)

    def test_normalise_velocity_saturates(self):
        self.assertAlmostEqual(normalise_velocity(200_000), 1.0)

    def test_normalise_velocity_midpoint(self):
        self.assertAlmostEqual(normalise_velocity(50_000), 0.5)

    def test_format_signal_buy(self):
        self.assertEqual(format_signal(0.10), "BUY")

    def test_format_signal_sell(self):
        self.assertEqual(format_signal(-0.10), "SELL")

    def test_format_signal_hold(self):
        self.assertEqual(format_signal(0.02), "HOLD")


class TestDataLoader(unittest.TestCase):
    """Smoke-test the mock data layer."""

    def test_mock_snapshots_not_empty(self):
        snapshots = load_mock_snapshots()
        self.assertGreater(len(snapshots), 0)

    def test_snapshot_schema(self):
        snap = load_mock_snapshots()[0]
        self.assertIsInstance(snap.artist_id, str)
        self.assertIsInstance(snap.spotify_listeners_daily, int)
        self.assertIsInstance(snap.tiktok_track_posts_velocity, float)

    def test_random_snapshot_deterministic(self):
        a = generate_random_snapshot(seed=42)
        b = generate_random_snapshot(seed=42)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
