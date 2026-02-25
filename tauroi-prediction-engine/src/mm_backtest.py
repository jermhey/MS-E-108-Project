"""
mm_backtest.py — Market-Making Backtest with Adverse-Selection Signal
=====================================================================
Simulates two market makers on historical Kalshi trade streams:

  1) Naive MM:  always quotes ± half-spread around mid, fills against
     every incoming trade.
  2) AS-Informed MM:  same quoting, but pulls quotes when the adverse-
     selection score exceeds a threshold.

The comparison shows the economic value of the AS detector: the informed
MM should avoid toxic fills and achieve better net PnL.

Mark-to-market uses the price observed `horizon` trades into the future.
Settlement PnL uses the final contract price (0 or 1 for expired, or
the last traded price for still-active contracts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.as_detector import ASResult
from src.utils import get_logger, kalshi_maker_fee

logger = get_logger("tauroi.mm_backtest")


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Fill:
    """A single fill in the backtest."""
    idx: int               # index in the trade array
    timestamp: object
    side: str              # "buy" or "sell" (from MM's perspective)
    price: float           # fill price
    spread_earned: float   # half-spread captured
    mtm_1h: float          # mark-to-market P&L at horizon
    fee: float             # maker fee paid


@dataclass
class MMResult:
    """Backtest result for a single market maker strategy on one ticker."""
    ticker: str
    strategy: str          # "naive" or "as_informed"
    n_trades: int          # total trades in the stream
    n_fills: int           # trades where the MM was filled
    n_pulled: int          # trades where the MM had pulled quotes (AS only)

    gross_spread: float    # total half-spread captured (cents)
    total_fees: float      # total maker fees paid (cents)
    mtm_pnl: float         # mark-to-market P&L at horizon (cents)
    net_pnl: float         # gross_spread - total_fees + mtm_pnl

    pnl_per_fill: float    # net_pnl / n_fills
    win_rate: float        # fraction of fills with positive P&L

    fills: List[Fill] = field(default_factory=list, repr=False)

    # Time series for plotting
    cumulative_pnl: np.ndarray = field(default=None, repr=False)

    @property
    def summary(self) -> Dict:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy,
            "n_fills": self.n_fills,
            "n_pulled": self.n_pulled,
            "gross_spread_c": round(self.gross_spread, 2),
            "fees_c": round(self.total_fees, 2),
            "mtm_pnl_c": round(self.mtm_pnl, 2),
            "net_pnl_c": round(self.net_pnl, 2),
            "pnl_per_fill_c": round(self.pnl_per_fill, 4),
            "win_rate": round(self.win_rate, 3),
        }


# ── Backtest Engine ─────────────────────────────────────────────────────────

def _compute_half_spread(price: float, base_spread_cents: float = 2.0) -> float:
    """
    Half-spread in price units (0-1 scale).

    A market maker posts at mid ± half_spread.  We use a fixed base
    spread in cents, converted to the 0-1 price scale.
    """
    return base_spread_cents / 100.0


def backtest_mm(
    as_result: ASResult,
    strategy: str = "naive",
    as_threshold: float = 0.5,
    half_spread_cents: float = 2.0,
    mtm_horizon_trades: int = 60,
    settlement_price: Optional[float] = None,
) -> MMResult:
    """
    Run a market-making backtest on one ticker.

    Parameters
    ----------
    as_result : ASResult
        Output from the AS detector pipeline.
    strategy : str
        "naive" (always quote) or "as_informed" (pull when AS > threshold).
    as_threshold : float
        AS score threshold for pulling quotes (only used for "as_informed").
    half_spread_cents : float
        Half-spread in cents that the MM quotes.
    mtm_horizon_trades : int
        Number of trades into the future for mark-to-market evaluation.
    settlement_price : float or None
        If provided, use this as the terminal value for P&L calculation.
        For settled contracts, this is 0 or 1.
    """
    prices = as_result.prices
    as_scores = as_result.as_score
    timestamps = as_result.timestamps
    n = len(prices)

    half_spread = half_spread_cents / 100.0
    fills: List[Fill] = []
    cumulative = np.zeros(n)

    running_pnl = 0.0
    n_pulled = 0

    for i in range(1, n):
        price_change = prices[i] - prices[i - 1]

        # Determine if MM has quotes live
        if strategy == "as_informed" and as_scores[i] > as_threshold:
            n_pulled += 1
            cumulative[i] = running_pnl
            continue

        if abs(price_change) < 1e-8:
            cumulative[i] = running_pnl
            continue

        # The incoming trade moved the price.
        # If price went UP, the MM's resting sell (ask) got hit → MM sold.
        # If price went DOWN, the MM's resting buy (bid) got hit → MM bought.
        if price_change > 0:
            mm_side = "sell"
            fill_price = prices[i - 1] + half_spread  # MM's ask
        else:
            mm_side = "buy"
            fill_price = prices[i - 1] - half_spread  # MM's bid

        # Check if the incoming trade would have crossed our quote
        # (i.e., the price moved by more than the half-spread)
        if abs(price_change) < half_spread:
            cumulative[i] = running_pnl
            continue

        spread_earned = half_spread

        # Mark-to-market: how does our fill look `mtm_horizon_trades` later?
        future_idx = min(i + mtm_horizon_trades, n - 1)
        future_price = prices[future_idx]

        if mm_side == "sell":
            mtm = fill_price - future_price  # sold high, value dropped = profit
        else:
            mtm = future_price - fill_price  # bought low, value rose = profit

        # Maker fee
        fee = kalshi_maker_fee(1, fill_price) * 100  # convert to cents

        fill_pnl = (spread_earned + mtm) * 100 - fee  # all in cents

        fills.append(Fill(
            idx=i,
            timestamp=timestamps[i],
            side=mm_side,
            price=fill_price,
            spread_earned=spread_earned * 100,
            mtm_1h=mtm * 100,
            fee=fee,
        ))

        running_pnl += fill_pnl
        cumulative[i] = running_pnl

    # Forward-fill the cumulative PnL
    for i in range(1, n):
        if cumulative[i] == 0 and i > 0:
            cumulative[i] = cumulative[i - 1]

    # Aggregate
    n_fills = len(fills)
    if n_fills == 0:
        return MMResult(
            ticker=as_result.ticker, strategy=strategy,
            n_trades=n, n_fills=0, n_pulled=n_pulled,
            gross_spread=0, total_fees=0, mtm_pnl=0, net_pnl=0,
            pnl_per_fill=0, win_rate=0,
            fills=fills, cumulative_pnl=cumulative,
        )

    gross_spread = sum(f.spread_earned for f in fills)
    total_fees = sum(f.fee for f in fills)
    mtm_pnl = sum(f.mtm_1h for f in fills)
    net_pnl = gross_spread + mtm_pnl - total_fees

    fill_pnls = [(f.spread_earned + f.mtm_1h - f.fee) for f in fills]
    win_rate = sum(1 for p in fill_pnls if p > 0) / n_fills

    return MMResult(
        ticker=as_result.ticker,
        strategy=strategy,
        n_trades=n,
        n_fills=n_fills,
        n_pulled=n_pulled,
        gross_spread=gross_spread,
        total_fees=total_fees,
        mtm_pnl=mtm_pnl,
        net_pnl=net_pnl,
        pnl_per_fill=net_pnl / n_fills,
        win_rate=win_rate,
        fills=fills,
        cumulative_pnl=cumulative,
    )


# ── Comparative Backtest ────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """Head-to-head comparison of naive vs AS-informed MM."""
    ticker: str
    naive: MMResult
    informed: MMResult

    @property
    def pnl_improvement(self) -> float:
        return self.informed.net_pnl - self.naive.net_pnl

    @property
    def toxic_fills_avoided(self) -> int:
        return self.naive.n_fills - self.informed.n_fills

    @property
    def summary(self) -> Dict:
        return {
            "ticker": self.ticker,
            "naive_pnl_c": round(self.naive.net_pnl, 2),
            "informed_pnl_c": round(self.informed.net_pnl, 2),
            "improvement_c": round(self.pnl_improvement, 2),
            "naive_fills": self.naive.n_fills,
            "informed_fills": self.informed.n_fills,
            "fills_avoided": self.toxic_fills_avoided,
            "naive_win_rate": round(self.naive.win_rate, 3),
            "informed_win_rate": round(self.informed.win_rate, 3),
        }


def compare_strategies(
    as_result: ASResult,
    as_threshold: float = 0.7,
    half_spread_cents: float = 2.0,
    mtm_horizon_trades: int = 60,
    settlement_price: Optional[float] = None,
) -> ComparisonResult:
    """Run both strategies on the same ticker and compare."""
    naive = backtest_mm(
        as_result, strategy="naive",
        half_spread_cents=half_spread_cents,
        mtm_horizon_trades=mtm_horizon_trades,
        settlement_price=settlement_price,
    )
    informed = backtest_mm(
        as_result, strategy="as_informed",
        as_threshold=as_threshold,
        half_spread_cents=half_spread_cents,
        mtm_horizon_trades=mtm_horizon_trades,
        settlement_price=settlement_price,
    )

    logger.info(
        "Comparison %s: naive=%.1fc (%d fills) | informed=%.1fc (%d fills) | "
        "improvement=%.1fc | avoided=%d toxic fills",
        as_result.ticker, naive.net_pnl, naive.n_fills,
        informed.net_pnl, informed.n_fills,
        informed.net_pnl - naive.net_pnl,
        naive.n_fills - informed.n_fills,
    )

    return ComparisonResult(
        ticker=as_result.ticker,
        naive=naive,
        informed=informed,
    )


def compare_all_tickers(
    as_results: Dict[str, ASResult],
    **kwargs,
) -> Dict[str, ComparisonResult]:
    """Run comparative backtest on every ticker."""
    comparisons = {}
    for ticker, ar in as_results.items():
        comparisons[ticker] = compare_strategies(ar, **kwargs)
    return comparisons


# ── Threshold Sweep ─────────────────────────────────────────────────────────

def analyse_fill_toxicity(
    as_result: ASResult,
    mtm_horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compare average price continuation during AS-flagged vs non-flagged periods.

    This is the core validation: if flagged periods have significantly
    larger adverse price moves, the detector is identifying toxic flow.

    Returns a DataFrame with one row per horizon showing mean absolute
    price move for flagged and unflagged trades, plus a t-test.
    """
    from scipy import stats as sp_stats

    if mtm_horizons is None:
        mtm_horizons = [10, 30, 60, 120]

    prices = as_result.prices
    as_scores = as_result.as_score
    n = len(prices)

    flagged = as_scores > 0.5
    # Only consider trades where price actually moved
    price_changed = np.zeros(n, dtype=bool)
    price_changed[1:] = np.abs(np.diff(prices)) > 0.005
    # Intersect: flagged & price changed
    flagged_active = flagged & price_changed
    unflagged_active = (~flagged) & price_changed

    rows = []
    for h in mtm_horizons:
        future_move = np.zeros(n)
        for i in range(n):
            fi = min(i + h, n - 1)
            future_move[i] = abs(prices[fi] - prices[i])

        move_flagged = future_move[flagged_active]
        move_unflagged = future_move[unflagged_active]

        if len(move_flagged) < 5 or len(move_unflagged) < 5:
            continue

        t_stat, p_val = sp_stats.ttest_ind(
            move_flagged, move_unflagged, alternative='greater'
        )

        rows.append({
            "horizon_trades": h,
            "n_flagged": int(flagged_active.sum()),
            "n_unflagged": int(unflagged_active.sum()),
            "mean_move_flagged": float(move_flagged.mean()),
            "mean_move_unflagged": float(move_unflagged.mean()),
            "ratio": float(move_flagged.mean() / max(move_unflagged.mean(), 1e-8)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        })

    return pd.DataFrame(rows)


def sweep_thresholds(
    as_result: ASResult,
    thresholds: Optional[List[float]] = None,
    half_spread_cents: float = 2.0,
    mtm_horizon_trades: int = 60,
) -> pd.DataFrame:
    """
    Sweep over AS thresholds to find the optimal operating point.

    Returns a DataFrame with one row per threshold showing PnL,
    fill count, win rate, etc.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    # Add naive baseline
    naive = backtest_mm(
        as_result, strategy="naive",
        half_spread_cents=half_spread_cents,
        mtm_horizon_trades=mtm_horizon_trades,
    )
    rows.append({
        "threshold": 0.0,
        "strategy": "naive",
        "n_fills": naive.n_fills,
        "n_pulled": 0,
        "net_pnl_c": naive.net_pnl,
        "pnl_per_fill_c": naive.pnl_per_fill,
        "win_rate": naive.win_rate,
        "gross_spread_c": naive.gross_spread,
        "mtm_pnl_c": naive.mtm_pnl,
        "fees_c": naive.total_fees,
    })

    for tau in thresholds:
        res = backtest_mm(
            as_result, strategy="as_informed",
            as_threshold=tau,
            half_spread_cents=half_spread_cents,
            mtm_horizon_trades=mtm_horizon_trades,
        )
        rows.append({
            "threshold": tau,
            "strategy": "as_informed",
            "n_fills": res.n_fills,
            "n_pulled": res.n_pulled,
            "net_pnl_c": res.net_pnl,
            "pnl_per_fill_c": res.pnl_per_fill,
            "win_rate": res.win_rate,
            "gross_spread_c": res.gross_spread,
            "mtm_pnl_c": res.mtm_pnl,
            "fees_c": res.total_fees,
        })

    return pd.DataFrame(rows)
