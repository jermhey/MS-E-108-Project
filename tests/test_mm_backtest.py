"""
test_mm_backtest.py — Canonical MM Backtest Regression Tests
=============================================================
Verifies the lagged AS gating rule and the PnL decomposition
used by the canonical market-making backtester.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.as_detector import ASResult
from src.mm_backtest import backtest_mm
from src.utils import kalshi_maker_fee


def _make_as_result(prices: list[float], as_scores: list[float]) -> ASResult:
    n = len(prices)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min").values
    arr = np.asarray(prices, dtype=float)
    zeros = np.zeros(n, dtype=float)
    return ASResult(
        ticker="SYNTHETIC",
        timestamps=timestamps,
        prices=arr,
        x_filtered=zeros.copy(),
        gamma=zeros.copy(),
        sigma_b_sq=zeros.copy(),
        lam=zeros.copy(),
        s_j_sq=zeros.copy(),
        burst_flags=np.zeros(n, dtype=bool),
        arrival_rate_ratio=np.ones(n, dtype=float),
        as_score=np.asarray(as_scores, dtype=float),
        alpha_weight=0.7,
        gamma_threshold=0.6,
        burst_threshold=3.0,
    )


class TestMMBacktest(unittest.TestCase):
    def test_lagged_signal_avoids_same_trade_lookahead(self):
        # Trade 1 moves up and is scored toxic only after it happens.
        # With a one-trade lag, trade 1 still fills; trade 2 can be blocked.
        as_result = _make_as_result(
            prices=[0.50, 0.52, 0.50, 0.50],
            as_scores=[0.0, 1.0, 0.0, 0.0],
        )

        lagged = backtest_mm(
            as_result,
            strategy="as_informed",
            as_threshold=0.5,
            half_spread_cents=1.0,
            mtm_horizon_trades=1,
            decision_lag_trades=1,
        )
        unlagged = backtest_mm(
            as_result,
            strategy="as_informed",
            as_threshold=0.5,
            half_spread_cents=1.0,
            mtm_horizon_trades=1,
            decision_lag_trades=0,
        )

        self.assertEqual(lagged.n_fills, 1)
        self.assertEqual(unlagged.n_fills, 1)
        self.assertEqual([f.idx for f in lagged.fills], [1])
        self.assertEqual([f.idx for f in unlagged.fills], [2])
        self.assertGreaterEqual(lagged.n_pulled, 1)

    def test_net_pnl_does_not_double_count_spread(self):
        # Sell at 51c off a 50c mid, then horizon price returns to 50c.
        # Net should be 1c spread minus fee, not 2c minus fee.
        as_result = _make_as_result(
            prices=[0.50, 0.52],
            as_scores=[0.0, 0.0],
        )

        result = backtest_mm(
            as_result,
            strategy="naive",
            half_spread_cents=1.0,
            mtm_horizon_trades=None,
            mtm_horizon_minutes=5.0,
            decision_lag_trades=1,
            settlement_price=0.50,
        )

        expected_fee_cents = kalshi_maker_fee(1, 0.51) * 100
        expected_net = 1.0 - expected_fee_cents

        self.assertEqual(result.n_fills, 1)
        self.assertAlmostEqual(result.gross_spread, 1.0, places=6)
        self.assertAlmostEqual(result.mtm_pnl, 0.0, places=6)
        self.assertAlmostEqual(result.net_pnl, expected_net, places=6)

    def test_include_fees_toggle(self):
        as_result = _make_as_result(
            prices=[0.50, 0.52],
            as_scores=[0.0, 0.0],
        )
        with_fees = backtest_mm(
            as_result,
            strategy="naive",
            half_spread_cents=1.0,
            mtm_horizon_trades=None,
            mtm_horizon_minutes=5.0,
            decision_lag_trades=1,
            settlement_price=0.50,
            include_fees=True,
        )
        no_fees = backtest_mm(
            as_result,
            strategy="naive",
            half_spread_cents=1.0,
            mtm_horizon_trades=None,
            mtm_horizon_minutes=5.0,
            decision_lag_trades=1,
            settlement_price=0.50,
            include_fees=False,
        )

        expected_fee_cents = kalshi_maker_fee(1, 0.51) * 100
        self.assertAlmostEqual(no_fees.total_fees, 0.0, places=6)
        self.assertAlmostEqual(no_fees.net_pnl - with_fees.net_pnl, expected_fee_cents, places=6)


if __name__ == "__main__":
    unittest.main()
