"""
executor.py — Order Execution Engine
======================================
Translates trading signals into Kalshi limit orders, manages the
order lifecycle (place, amend, cancel stale), and tracks fills.

Supports two modes:
  - **Directional**: one-sided orders based on edge sign.
  - **Market-Making**: two-sided quoting around fair value to earn
    the bid-ask spread with maker fees.

This module ties together the KalshiClient, RiskManager, and
PositionStore into a coherent execution pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.kalshi_client import KalshiClient, KalshiAPIError
from src.risk_manager import RiskManager, TradeProposal, RiskDecision
from src.position_store import PositionStore
from src.utils import get_logger

logger = get_logger("tauroi.executor")

# Maximum drift in cents between a resting order's price and the
# current model fair value before we cancel and replace.
STALE_ORDER_DRIFT_CENTS = 3

# Default half-spread for market-making mode (cents).
# Backtest shows net profitability at ≥8c after 1.5c/fill maker fee.
DEFAULT_MM_HALF_SPREAD_CENTS = 8


class OrderExecutor:
    """
    Converts model signals into live Kalshi orders with full risk
    management and position tracking.

    Parameters
    ----------
    kalshi : KalshiClient
        Authenticated Kalshi API client.
    risk : RiskManager
        Pre-trade risk validation.
    store : PositionStore
        Disk-persisted position and order state.
    base_unit_pct : float
        Base position size as % of available cash (default 2%).
    max_conviction_units : int
        Maximum conviction multiplier (default 3 = 6% max).
    market_making : bool
        If True, place two-sided quotes around fair value instead of
        directional orders (default False).
    mm_half_spread_cents : int
        Half-spread for market-making mode (default 8c).
    """

    def __init__(
        self,
        kalshi: KalshiClient,
        risk: RiskManager,
        store: PositionStore,
        base_unit_pct: float = 0.02,
        max_conviction_units: int = 3,
        market_making: bool = False,
        mm_half_spread_cents: int = DEFAULT_MM_HALF_SPREAD_CENTS,
    ) -> None:
        self.kalshi = kalshi
        self.risk = risk
        self.store = store
        self.base_unit_pct = base_unit_pct
        self.max_conviction_units = max_conviction_units
        self.market_making = market_making
        self.mm_half_spread_cents = mm_half_spread_cents

        self._balance_cents: int = 0
        self._positions: List[Dict[str, Any]] = []

    # ── Startup ───────────────────────────────────────────────────────

    def startup(self) -> Dict[str, Any]:
        """
        Initialise executor state from Kalshi API.

        Call this once on startup before processing any signals.
        Returns a summary dict.
        """
        # Fetch live state
        balance_data = self.kalshi.get_balance()
        self._balance_cents = balance_data.get("balance", 0)

        self._positions = self.kalshi.get_all_positions()
        api_orders = self.kalshi.get_all_resting_orders()

        # Reconcile local store against API
        changes = self.store.reconcile(self._positions, api_orders)

        # Reset daily P&L in risk manager
        self.risk.reset_daily_pnl(self._balance_cents)

        summary = {
            "balance_cents": self._balance_cents,
            "balance_dollars": self._balance_cents / 100,
            "open_positions": len(self._positions),
            "resting_orders": len(api_orders),
            "reconciliation": changes,
            "risk_limits": self.risk.summary(),
        }

        logger.info(
            "Executor startup: $%.2f balance, %d positions, %d orders",
            summary["balance_dollars"],
            summary["open_positions"],
            summary["resting_orders"],
        )
        return summary

    def refresh_state(self) -> None:
        """Refresh balance and positions from Kalshi (call each cycle)."""
        try:
            balance_data = self.kalshi.get_balance()
            self._balance_cents = balance_data.get("balance", 0)
            self._positions = self.kalshi.get_all_positions()
        except KalshiAPIError as exc:
            logger.warning("Failed to refresh state: %s", exc)

    # ── Position Sizing ───────────────────────────────────────────────

    def compute_order_size(
        self,
        edge: float,
        price_cents: int,
        edge_threshold: float = 0.05,
    ) -> int:
        """
        Risk-adjusted position sizing, adapted from Backtester.

        Returns the number of contracts to order.
        """
        if self._balance_cents <= 0 or price_cents <= 0:
            return 0

        base_unit_dollars = (self._balance_cents / 100) * self.base_unit_pct
        abs_edge = abs(edge)

        if abs_edge > 0.25:
            units = self.max_conviction_units
        elif abs_edge > 0.15:
            units = 2
        elif abs_edge > edge_threshold:
            units = 1
        else:
            return 0

        dollar_amount = base_unit_dollars * units
        contracts = int(dollar_amount / (price_cents / 100))
        return max(contracts, 1) if contracts > 0 else 0

    # ── Core Execution ────────────────────────────────────────────────

    def process_signal(
        self,
        ticker: str,
        fair_value: float,
        market_price: float,
        edge: float,
        edge_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Process a single trading signal through the full pipeline:
        sizing → risk check → order placement.

        Parameters
        ----------
        ticker : str
            Kalshi market ticker.
        fair_value : float
            Model fair value (0-1 probability).
        market_price : float
            Current market mid-price (0-1).
        edge : float
            fair_value - market_price.
        edge_threshold : float
            Minimum |edge| to trade.

        Returns
        -------
        dict with keys: action, status, reason, order (if placed)
        """
        result: Dict[str, Any] = {
            "ticker": ticker,
            "fair_value": fair_value,
            "market_price": market_price,
            "edge": edge,
            "action": "none",
            "status": "skipped",
            "reason": "",
        }

        if not ticker:
            result["reason"] = "no ticker"
            return result

        abs_edge = abs(edge)
        if abs_edge < edge_threshold:
            result["reason"] = f"|edge| {abs_edge:.4f} < threshold {edge_threshold}"
            return result

        # Determine order parameters
        if edge > 0:
            # Model thinks it's cheap → BUY YES
            action = "buy"
            side = "yes"
            price_cents = max(1, min(99, int(round(fair_value * 100))))
        else:
            # Model thinks it's expensive → SELL YES (buy NO)
            action = "buy"
            side = "no"
            price_cents = max(1, min(99, int(round((1.0 - fair_value) * 100))))

        count = self.compute_order_size(edge, price_cents, edge_threshold)
        if count < 1:
            result["reason"] = "position size too small"
            return result

        edge_cents = edge * 100

        # Check for existing resting order on this ticker
        existing = self.store.get_resting_order_for_ticker(ticker, side)
        if existing:
            existing_price = existing.get("price_cents", 0)
            drift = abs(price_cents - existing_price)
            if drift <= STALE_ORDER_DRIFT_CENTS:
                result["action"] = "hold_existing"
                result["status"] = "skipped"
                result["reason"] = (
                    f"resting order exists @{existing_price}c, "
                    f"drift {drift}c <= {STALE_ORDER_DRIFT_CENTS}c"
                )
                return result
            else:
                # Cancel stale order and replace
                oid = existing.get("order_id", "")
                try:
                    self.kalshi.cancel_order(oid)
                    self.store.remove_order(oid, reason="stale_replaced")
                    logger.info(
                        "Cancelled stale order %s (drift %dc)", oid, drift,
                    )
                except KalshiAPIError as exc:
                    logger.warning("Failed to cancel stale order %s: %s", oid, exc)

        # Build proposal and run through risk gate
        proposal = TradeProposal(
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            price_cents=price_cents,
            edge_cents=edge_cents,
            fair_value=fair_value,
            market_price=market_price,
        )

        decision = self.risk.check(
            proposal, self._balance_cents, self._positions,
        )

        if not decision.approved:
            result["action"] = f"{action}_{side}"
            result["status"] = "rejected"
            result["reason"] = decision.reason
            logger.warning(
                "RISK REJECTED: %s %s %s ×%d @%dc — %s",
                action, side, ticker, count, price_cents, decision.reason,
            )
            self.store.append_log({
                "event": "risk_rejected",
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "price_cents": price_cents,
                "reason": decision.reason,
            })
            return result

        # Place the order
        try:
            order = self.kalshi.place_order(
                ticker=ticker,
                side=side,
                action=action,
                count=count,
                price_cents=price_cents,
                post_only=True,
            )

            self.store.track_order(order, ticker_override=ticker)

            result["action"] = f"{action}_{side}"
            result["status"] = "placed"
            result["reason"] = f"order {order.get('order_id', '?')}"
            result["order"] = order

            logger.info(
                "ORDER PLACED: %s %s %s ×%d @%dc (edge %.1fc) → %s",
                action, side, ticker, count, price_cents,
                edge_cents, order.get("order_id", "?"),
            )

        except KalshiAPIError as exc:
            result["action"] = f"{action}_{side}"
            result["status"] = "error"
            result["reason"] = str(exc)[:200]
            logger.error(
                "ORDER FAILED: %s %s %s ×%d @%dc — %s",
                action, side, ticker, count, price_cents, exc,
            )
            self.store.append_log({
                "event": "order_error",
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "price_cents": price_cents,
                "error": str(exc)[:200],
            })

        return result

    # ── Market-Making Execution ──────────────────────────────────────

    def process_market_making_signal(
        self,
        ticker: str,
        fair_value: float,
    ) -> List[Dict[str, Any]]:
        """
        Place two-sided quotes around the model fair value.

        Posts a resting BUY-YES at ``fair - half_spread`` and a
        resting BUY-NO (sell YES) at ``fair + half_spread``, both
        with ``post_only=True`` for maker fees.

        Returns a list of result dicts (one per side attempted).
        """
        results: List[Dict[str, Any]] = []
        hs = self.mm_half_spread_cents
        fair_cents = int(round(fair_value * 100))

        bid_cents = max(1, fair_cents - hs)
        ask_cents = min(99, fair_cents + hs)

        if bid_cents >= ask_cents:
            results.append({
                "ticker": ticker, "action": "mm_skip", "status": "skipped",
                "reason": f"spread collapsed: bid={bid_cents}c ask={ask_cents}c",
            })
            return results

        base_size = max(1, int(
            (self._balance_cents / 100) * self.base_unit_pct / (fair_cents / 100)
        )) if self._balance_cents > 0 and fair_cents > 0 else 1

        for side, action, price_cents, label in [
            ("yes", "buy", bid_cents, "mm_bid"),
            ("no", "buy", 100 - ask_cents, "mm_ask"),
        ]:
            existing = self.store.get_resting_order_for_ticker(ticker, side)
            if existing:
                drift = abs(price_cents - existing.get("price_cents", 0))
                if drift <= STALE_ORDER_DRIFT_CENTS:
                    results.append({
                        "ticker": ticker, "action": label, "status": "skipped",
                        "reason": f"resting {side} order still fresh (drift {drift}c)",
                    })
                    continue
                oid = existing.get("order_id", "")
                try:
                    self.kalshi.cancel_order(oid)
                    self.store.remove_order(oid, reason="mm_requote")
                except KalshiAPIError as exc:
                    logger.warning("MM cancel failed %s: %s", oid, exc)

            proposal = TradeProposal(
                ticker=ticker, side=side, action=action,
                count=base_size, price_cents=price_cents,
                edge_cents=hs, fair_value=fair_value,
                market_price=fair_value,
            )
            decision = self.risk.check(proposal, self._balance_cents, self._positions)

            if not decision.approved:
                results.append({
                    "ticker": ticker, "action": label, "status": "rejected",
                    "reason": decision.reason,
                })
                continue

            try:
                order = self.kalshi.place_order(
                    ticker=ticker, side=side, action=action,
                    count=base_size, price_cents=price_cents, post_only=True,
                )
                self.store.track_order(
                    order,
                    ticker_override=ticker,
                    price_cents_override=price_cents,
                )
                results.append({
                    "ticker": ticker, "action": label, "status": "placed",
                    "reason": f"order {order.get('order_id', '?')}",
                    "order": order,
                })
                logger.info(
                    "MM ORDER: %s %s %s ×%d @%dc",
                    action, side, ticker, base_size, price_cents,
                )
            except KalshiAPIError as exc:
                results.append({
                    "ticker": ticker, "action": label, "status": "error",
                    "reason": str(exc)[:200],
                })
                logger.error("MM ORDER FAILED: %s %s %s — %s", action, side, ticker, exc)

        return results

    # ── Batch Signal Processing ───────────────────────────────────────

    def process_scan_signals(
        self,
        scan_rows: List[Dict[str, Any]],
        edge_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Process all signals from a market scan.

        In **market-making** mode, places two-sided quotes on every
        ticker regardless of edge direction.

        In **directional** mode, only acts on signals with |edge| > threshold.
        """
        self.refresh_state()

        results: List[Dict[str, Any]] = []
        active_tickers: set[str] = set()

        for row in scan_rows:
            ticker = row.get("ticker")
            if not ticker:
                continue

            fair_value = row.get("fair_value", 0)
            edge = row.get("edge", 0)
            market_price = row.get("market_price", 0)
            signal = row.get("signal", "HOLD")

            active_tickers.add(ticker)

            if self.market_making:
                # Use per-ticker dynamic spread if the hybrid model provided one
                dynamic_hs = row.get("dynamic_spread_cents")
                if dynamic_hs is not None:
                    saved = self.mm_half_spread_cents
                    self.mm_half_spread_cents = int(dynamic_hs)
                    mm_results = self.process_market_making_signal(ticker, fair_value)
                    self.mm_half_spread_cents = saved
                else:
                    mm_results = self.process_market_making_signal(ticker, fair_value)
                results.extend(mm_results)

                # Also place directional order if edge is large
                if abs(edge) >= edge_threshold:
                    result = self.process_signal(
                        ticker=ticker, fair_value=fair_value,
                        market_price=market_price, edge=edge,
                        edge_threshold=edge_threshold,
                    )
                    results.append(result)
            else:
                if signal == "HOLD":
                    continue
                result = self.process_signal(
                    ticker=ticker, fair_value=fair_value,
                    market_price=market_price, edge=edge,
                    edge_threshold=edge_threshold,
                )
                results.append(result)

        self._cleanup_stale_orders(active_tickers, scan_rows)
        return results

    def _cleanup_stale_orders(
        self,
        active_tickers: set[str],
        scan_rows: List[Dict[str, Any]],
    ) -> None:
        """Cancel resting orders where the signal is now HOLD."""
        signal_by_ticker = {r["ticker"]: r.get("signal", "HOLD") for r in scan_rows if r.get("ticker")}

        for oid, order in list(self.store.resting_orders.items()):
            ticker = order.get("ticker", "")
            current_signal = signal_by_ticker.get(ticker, "HOLD")

            if current_signal == "HOLD":
                try:
                    self.kalshi.cancel_order(oid)
                    self.store.remove_order(oid, reason="signal_flipped")
                    logger.info(
                        "Cancelled order %s — signal flipped to HOLD", oid,
                    )
                except KalshiAPIError as exc:
                    logger.warning("Failed to cancel %s: %s", oid, exc)

    # ── Kill Switch ───────────────────────────────────────────────────

    def kill_all(self) -> Dict[str, Any]:
        """
        Emergency shutdown: cancel all orders, close all positions.

        Returns a summary of actions taken.
        """
        summary: Dict[str, Any] = {
            "orders_cancelled": 0,
            "positions_closed": 0,
            "errors": [],
        }

        # 1. Cancel all resting orders
        logger.warning("KILL SWITCH: cancelling all orders...")
        try:
            n = self.kalshi.cancel_all_orders()
            summary["orders_cancelled"] = n
        except KalshiAPIError as exc:
            summary["errors"].append(f"cancel_all: {exc}")

        # Clear local order tracking
        for oid in list(self.store.resting_orders.keys()):
            self.store.remove_order(oid, reason="kill_switch")

        # 2. Close all positions by placing aggressive limit orders
        logger.warning("KILL SWITCH: closing all positions...")
        positions = self.kalshi.get_all_positions()

        for p in positions:
            ticker = p.get("ticker", "")
            pos_count = p.get("position", 0)
            if pos_count == 0 or not ticker:
                continue

            # If long YES (positive), sell YES at 1c (guaranteed fill)
            # If short YES (negative), buy YES at 99c (guaranteed fill)
            if pos_count > 0:
                side, action, price = "yes", "sell", 1
                count = pos_count
            else:
                side, action, price = "yes", "buy", 99
                count = abs(pos_count)

            try:
                self.kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    action=action,
                    count=count,
                    price_cents=price,
                    post_only=False,  # cross the spread to guarantee fill
                )
                summary["positions_closed"] += 1
                logger.warning(
                    "KILL: %s %s %s ×%d @%dc", action, side, ticker, count, price,
                )
            except KalshiAPIError as exc:
                summary["errors"].append(f"close {ticker}: {exc}")
                logger.error("KILL failed for %s: %s", ticker, exc)

        # Clear local positions
        for ticker in list(self.store.positions.keys()):
            self.store.remove_position(ticker)

        self.store.append_log({
            "event": "kill_switch",
            "summary": summary,
        })

        logger.warning("KILL SWITCH complete: %s", summary)
        return summary
