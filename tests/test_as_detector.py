"""
test_as_detector.py — Smoke Tests for the Adverse Selection Pipeline
=====================================================================
Verifies core invariants of the Kalman filter, EM calibration, and
burst detection on synthetic data (no Kalshi API required).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.as_detector import (
    kalman_filter_hf,
    rolling_em_calibrate,
    detect_bursts,
    run_as_detection,
)


def _make_synthetic_trades(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic trade DataFrame with realistic structure."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="2min")
    prices = 50 + np.cumsum(rng.normal(0, 0.3, n))
    prices = np.clip(prices, 2, 98).round(0).astype(int)
    p = prices / 100.0
    logits = np.log(p / (1 - p))
    return pd.DataFrame(
        {"timestamp": ts, "mid_price": prices, "volume": rng.integers(1, 5, n), "logit": logits}
    )


class TestKalmanFilter(unittest.TestCase):
    """Kalman filter should produce a smoothed series with lower variance."""

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(0)
        raw = np.cumsum(rng.normal(0, 0.1, 300))
        noisy = raw + rng.normal(0, 0.5, 300)
        dt = np.full(299, 120.0)
        filtered = kalman_filter_hf(noisy, dt, meas_var=0.25)
        self.assertLess(np.var(np.diff(filtered)), np.var(np.diff(noisy)))

    def test_output_length_matches_input(self):
        logits = np.random.randn(100)
        dt = np.full(99, 60.0)
        filtered = kalman_filter_hf(logits, dt)
        self.assertEqual(len(filtered), len(logits))


class TestBurstDetection(unittest.TestCase):
    """Burst detector should flag periods of unusually fast arrivals."""

    def _seconds_to_datetimes(self, seconds: np.ndarray) -> np.ndarray:
        base = pd.Timestamp("2026-01-01")
        return np.array([base + pd.Timedelta(s, "s") for s in seconds])

    def test_flags_fast_cluster(self):
        rng = np.random.default_rng(1)
        slow = np.cumsum(rng.exponential(120, 400))
        fast = slow[-1] + np.cumsum(rng.exponential(3, 50))
        slow2 = fast[-1] + np.cumsum(rng.exponential(120, 400))
        raw_ts = np.concatenate([slow, fast, slow2])
        timestamps = self._seconds_to_datetimes(raw_ts)

        burst_flags, _ = detect_bursts(
            timestamps, short_window=8, long_window=200, burst_multiplier=2.0
        )
        fast_start = 400
        fast_end = 450
        self.assertTrue(
            burst_flags[fast_start:fast_end].mean() > 0.3,
            "Burst detector should flag the fast-arrival cluster",
        )

    def test_no_bursts_in_uniform_arrivals(self):
        raw_ts = np.arange(0, 600000, 60.0)
        timestamps = self._seconds_to_datetimes(raw_ts)
        burst_flags, _ = detect_bursts(timestamps)
        self.assertLess(
            burst_flags.mean(), 0.05, "Uniform arrivals should have near-zero burst rate"
        )


class TestRunASDetection(unittest.TestCase):
    """End-to-end smoke test on synthetic data."""

    def test_returns_valid_result(self):
        df = _make_synthetic_trades(400)
        result = run_as_detection(df, ticker="SYNTHETIC", em_window=80)
        self.assertEqual(len(result.as_score), len(df))
        self.assertTrue(np.all(result.as_score >= 0))
        self.assertTrue(np.all(result.as_score <= 1))
        self.assertIsInstance(result.events, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
