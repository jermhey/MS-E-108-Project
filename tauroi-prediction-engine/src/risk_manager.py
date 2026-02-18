"""
risk_manager.py — Pre-Trade Risk Validation
=============================================
Pure validation layer that sits between the signal generator and the
order executor.  Every proposed trade must pass ALL checks before any
API call is made.

All limits have safe defaults and can be overridden via constructor.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.utils import get_logger

logger = get_logger("tauroi.risk")


@dataclass
class TradeProposal:
    """A proposed trade to be validated by the RiskManager."""
    ticker: str
    side: str           # "yes" or "no"
    action: str         # "buy" or "sell"
    count: int          # number of contracts
    price_cents: int    # limit price in cents (1-99)
    edge_cents: float   # model edge in cents
    fair_value: float   # model fair value (0-1)
    market_price: float # current market price (0-1)


@dataclass
class RiskDecision:
    """Result of a risk check."""
    approved: bool
    reason: str
    proposal: TradeProposal


class RiskManager:
    """
    Pre-trade risk gate.  Validates every proposed trade against
    hard limits before it touches the Kalshi API.

    Parameters
    ----------
    max_order_size : int
        Maximum contracts per single order.
    max_position_per_ticker : int
        Maximum contracts held for any single ticker.
    max_total_exposure_pct : float
        Maximum % of account balance at risk across all positions.
    max_daily_loss_pct : float
        Circuit breaker: halt trading if daily realised + unrealised
        loss exceeds this % of the day's starting balance.
    max_edge_cents : float
        Sanity check: reject if |edge| exceeds this (likely bad data).
    min_balance_floor_cents : int
        Never trade if available balance drops below this (in cents).
    """

    def __init__(
        self,
        max_order_size: int = 25,
        max_position_per_ticker: int = 50,
        max_total_exposure_pct: float = 0.30,
        max_daily_loss_pct: float = 0.10,
        max_edge_cents: float = 50.0,
        min_balance_floor_cents: int = 1000,  # $10
    ) -> None:
        self.max_order_size = max_order_size
        self.max_position_per_ticker = max_position_per_ticker
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_edge_cents = max_edge_cents
        self.min_balance_floor_cents = min_balance_floor_cents

        # Daily P&L tracking — reset via reset_daily_pnl()
        self._day_start_balance_cents: int = 0
        self._realised_pnl_cents: int = 0
        self._circuit_breaker_tripped: bool = False
        self._today: str = ""

    # ── Daily P&L Management ──────────────────────────────────────────

    def reset_daily_pnl(self, balance_cents: int) -> None:
        """Call at the start of each trading day (or on startup)."""
        today = datetime.date.today().isoformat()
        if self._today != today:
            self._today = today
            self._day_start_balance_cents = balance_cents
            self._realised_pnl_cents = 0
            self._circuit_breaker_tripped = False
            logger.info(
                "Daily P&L reset — start balance: %d cents ($%.2f)",
                balance_cents, balance_cents / 100,
            )

    def record_fill(self, pnl_cents: int) -> None:
        """Record realised P&L from a fill."""
        self._realised_pnl_cents += pnl_cents

    @property
    def daily_loss_cents(self) -> int:
        """Current daily realised loss (negative = loss)."""
        return self._realised_pnl_cents

    # ── Core Validation ───────────────────────────────────────────────

    def check(
        self,
        proposal: TradeProposal,
        balance_cents: int,
        positions: List[Dict[str, Any]],
    ) -> RiskDecision:
        """
        Run ALL pre-trade checks.  Returns a RiskDecision.

        Parameters
        ----------
        proposal : TradeProposal
            The trade to validate.
        balance_cents : int
            Current available cash in cents from Kalshi.
        positions : list of dict
            Current open positions (from ``get_all_positions()``).
        """
        # 0. Circuit breaker
        if self._circuit_breaker_tripped:
            return RiskDecision(
                approved=False,
                reason="CIRCUIT BREAKER: daily loss limit hit, trading halted",
                proposal=proposal,
            )

        # 1. Balance floor
        if balance_cents < self.min_balance_floor_cents:
            return RiskDecision(
                approved=False,
                reason=f"Balance {balance_cents}c < floor {self.min_balance_floor_cents}c",
                proposal=proposal,
            )

        # 2. Order size
        if proposal.count > self.max_order_size:
            return RiskDecision(
                approved=False,
                reason=f"Order size {proposal.count} > max {self.max_order_size}",
                proposal=proposal,
            )
        if proposal.count < 1:
            return RiskDecision(
                approved=False,
                reason=f"Order size must be >= 1, got {proposal.count}",
                proposal=proposal,
            )

        # 3. Price sanity
        if not 1 <= proposal.price_cents <= 99:
            return RiskDecision(
                approved=False,
                reason=f"Price {proposal.price_cents}c outside 1-99 range",
                proposal=proposal,
            )

        # 4. Edge sanity — reject if edge is implausibly large
        if abs(proposal.edge_cents) > self.max_edge_cents:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Edge {proposal.edge_cents:.1f}c exceeds sanity limit "
                    f"{self.max_edge_cents}c (likely bad data)"
                ),
                proposal=proposal,
            )

        # 5. Position limit per ticker
        current_position = self._ticker_position_count(
            positions, proposal.ticker,
        )
        resulting_position = current_position + proposal.count
        if resulting_position > self.max_position_per_ticker:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Resulting position {resulting_position} > max "
                    f"{self.max_position_per_ticker} for {proposal.ticker}"
                ),
                proposal=proposal,
            )

        # 6. Order cost vs available balance
        order_cost_cents = proposal.count * proposal.price_cents
        if proposal.action == "buy" and order_cost_cents > balance_cents:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Order cost {order_cost_cents}c > balance {balance_cents}c"
                ),
                proposal=proposal,
            )

        # 7. Total portfolio exposure
        total_exposure = self._total_exposure_cents(positions)
        new_exposure = total_exposure + order_cost_cents
        max_exposure = int(balance_cents / self.max_total_exposure_pct) if self.max_total_exposure_pct < 1.0 else balance_cents
        exposure_limit = int(
            (balance_cents + total_exposure) * self.max_total_exposure_pct
        )
        if new_exposure > exposure_limit:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Total exposure {new_exposure}c would exceed "
                    f"{self.max_total_exposure_pct:.0%} limit ({exposure_limit}c)"
                ),
                proposal=proposal,
            )

        # 8. Daily loss circuit breaker
        if self._day_start_balance_cents > 0:
            max_loss = int(
                self._day_start_balance_cents * self.max_daily_loss_pct,
            )
            if self._realised_pnl_cents < -max_loss:
                self._circuit_breaker_tripped = True
                logger.warning(
                    "CIRCUIT BREAKER TRIPPED: daily loss %dc exceeds %dc limit",
                    abs(self._realised_pnl_cents), max_loss,
                )
                return RiskDecision(
                    approved=False,
                    reason=(
                        f"CIRCUIT BREAKER: daily loss {self._realised_pnl_cents}c "
                        f"exceeds {max_loss}c limit"
                    ),
                    proposal=proposal,
                )

        # All checks passed
        logger.info(
            "RISK APPROVED: %s %s %s ×%d @%dc (edge %.1fc)",
            proposal.action, proposal.side, proposal.ticker,
            proposal.count, proposal.price_cents, proposal.edge_cents,
        )
        return RiskDecision(
            approved=True,
            reason="all checks passed",
            proposal=proposal,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ticker_position_count(
        positions: List[Dict[str, Any]],
        ticker: str,
    ) -> int:
        """Sum of contracts held for a specific ticker."""
        total = 0
        for p in positions:
            if p.get("ticker") == ticker:
                yes_count = abs(p.get("position", 0))
                total += yes_count
        return total

    @staticmethod
    def _total_exposure_cents(positions: List[Dict[str, Any]]) -> int:
        """Approximate total capital at risk across all positions."""
        total = 0
        for p in positions:
            count = abs(p.get("position", 0))
            # Worst case cost per contract is the entry price
            market_exposure = p.get("market_exposure", 0) or 0
            if market_exposure:
                total += abs(market_exposure)
            else:
                total += count * 50  # conservative estimate: 50c average
        return total

    def summary(self) -> Dict[str, Any]:
        """Current risk state for logging/display."""
        return {
            "max_order_size": self.max_order_size,
            "max_position_per_ticker": self.max_position_per_ticker,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_edge_cents": self.max_edge_cents,
            "min_balance_floor_cents": self.min_balance_floor_cents,
            "circuit_breaker_tripped": self._circuit_breaker_tripped,
            "daily_realised_pnl_cents": self._realised_pnl_cents,
            "day_start_balance_cents": self._day_start_balance_cents,
        }
