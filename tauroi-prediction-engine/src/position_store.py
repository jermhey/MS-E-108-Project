"""
position_store.py — Disk-Persisted Position & Order Tracking
=============================================================
Tracks open positions, resting orders, and an append-only order log
in the ``state/`` directory.  Designed to survive restarts and
reconcile against the Kalshi API on startup.
"""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger("tauroi.positions")

_STATE_DIR = pathlib.Path(__file__).resolve().parent.parent / "state"


class PositionStore:
    """
    Local position and order state, persisted to JSON files.

    Files managed:
        ``state/positions.json``  — current positions per ticker
        ``state/orders.json``     — resting orders we placed
        ``state/order_log.jsonl`` — append-only audit trail
        ``state/daily_pnl.json``  — daily P&L tracking
    """

    def __init__(self, state_dir: pathlib.Path | None = None) -> None:
        self._dir = state_dir or _STATE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._positions_path = self._dir / "positions.json"
        self._orders_path = self._dir / "orders.json"
        self._log_path = self._dir / "order_log.jsonl"
        self._pnl_path = self._dir / "daily_pnl.json"

        self._positions: Dict[str, Dict[str, Any]] = self._load_json(
            self._positions_path, {},
        )
        self._orders: Dict[str, Dict[str, Any]] = self._load_json(
            self._orders_path, {},
        )

    # ── Persistence Helpers ───────────────────────────────────────────

    @staticmethod
    def _load_json(path: pathlib.Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return default
        return default

    def _save_positions(self) -> None:
        self._positions_path.write_text(
            json.dumps(self._positions, indent=2, default=str),
        )

    def _save_orders(self) -> None:
        self._orders_path.write_text(
            json.dumps(self._orders, indent=2, default=str),
        )

    def append_log(self, record: Dict[str, Any]) -> None:
        """Append a record to the order audit log."""
        record["logged_at"] = datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat()
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Position Management ───────────────────────────────────────────

    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        """Current positions keyed by ticker."""
        return dict(self._positions)

    def get_position(self, ticker: str) -> Dict[str, Any] | None:
        return self._positions.get(ticker)

    def update_position(
        self,
        ticker: str,
        side: str,
        count: int,
        avg_price_cents: float,
    ) -> None:
        """Set or update the local position for a ticker."""
        self._positions[ticker] = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "avg_price_cents": round(avg_price_cents, 2),
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
        }
        self._save_positions()

    def remove_position(self, ticker: str) -> None:
        """Remove a position (e.g. after full close or settlement)."""
        self._positions.pop(ticker, None)
        self._save_positions()

    def record_fill(
        self,
        ticker: str,
        side: str,
        action: str,
        fill_count: int,
        fill_price_cents: int,
    ) -> None:
        """
        Update position state after a fill.

        For BUY: increases position.  For SELL: decreases position.
        Tracks weighted average entry price.
        """
        existing = self._positions.get(ticker)

        if action == "buy":
            if existing and existing["side"] == side:
                old_count = existing["count"]
                old_avg = existing["avg_price_cents"]
                new_count = old_count + fill_count
                new_avg = (
                    (old_avg * old_count + fill_price_cents * fill_count)
                    / new_count
                )
                self.update_position(ticker, side, new_count, new_avg)
            else:
                self.update_position(ticker, side, fill_count, fill_price_cents)
        elif action == "sell":
            if existing:
                new_count = existing["count"] - fill_count
                if new_count <= 0:
                    self.remove_position(ticker)
                else:
                    self.update_position(
                        ticker, side, new_count, existing["avg_price_cents"],
                    )

        self.append_log({
            "event": "fill",
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": fill_count,
            "price_cents": fill_price_cents,
        })

    # ── Order Tracking ────────────────────────────────────────────────

    @property
    def resting_orders(self) -> Dict[str, Dict[str, Any]]:
        """Resting orders keyed by order_id."""
        return dict(self._orders)

    def track_order(
        self,
        order: Dict[str, Any],
        *,
        ticker_override: str | None = None,
        price_cents_override: int | None = None,
    ) -> None:
        """
        Start tracking a newly placed order.

        Use ticker_override so we always have the correct ticker for lookups
        even if the API response omits or uses a different field name.
        """
        oid = order.get("order_id", "")
        ticker = (
            ticker_override
            if ticker_override
            else order.get("ticker") or order.get("contract_ticker") or ""
        )
        price_cents = (
            price_cents_override
            if price_cents_override is not None
            else order.get("yes_price", 0)
        )
        self._orders[oid] = {
            "order_id": oid,
            "ticker": ticker,
            "side": order.get("side", ""),
            "action": order.get("action", ""),
            "price_cents": price_cents,
            "count": order.get("remaining_count", order.get("initial_count", 0)),
            "status": order.get("status", "resting"),
            "placed_at": datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
        }
        self._save_orders()

        self.append_log({
            "event": "order_placed",
            "order_id": oid,
            "ticker": ticker,
            "side": order.get("side"),
            "action": order.get("action"),
            "price_cents": price_cents,
            "count": order.get("initial_count"),
        })

    def remove_order(self, order_id: str, reason: str = "cancelled") -> None:
        """Stop tracking an order (cancelled, filled, or expired)."""
        removed = self._orders.pop(order_id, None)
        self._save_orders()
        if removed:
            self.append_log({
                "event": f"order_{reason}",
                "order_id": order_id,
                "ticker": removed.get("ticker"),
            })

    def get_resting_order_for_ticker(
        self,
        ticker: str,
        side: str | None = None,
    ) -> Dict[str, Any] | None:
        """Find a resting order we placed for a given ticker."""
        for o in self._orders.values():
            if o.get("ticker") == ticker:
                if side is None or o.get("side") == side:
                    return o
        return None

    # ── Reconciliation ────────────────────────────────────────────────

    def reconcile(
        self,
        api_positions: List[Dict[str, Any]],
        api_orders: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Reconcile local state against the Kalshi API.

        Fixes drift from fills/settlements that happened while offline.

        Returns a summary of changes made.
        """
        changes: Dict[str, Any] = {
            "positions_added": 0,
            "positions_removed": 0,
            "positions_updated": 0,
            "orders_removed": 0,
        }

        # -- Reconcile positions --
        api_pos_by_ticker: Dict[str, Dict[str, Any]] = {}
        for p in api_positions:
            ticker = p.get("ticker", "")
            pos_count = p.get("position", 0)
            if ticker and pos_count != 0:
                api_pos_by_ticker[ticker] = p

        # Remove local positions that no longer exist on Kalshi
        for ticker in list(self._positions.keys()):
            if ticker not in api_pos_by_ticker:
                self.remove_position(ticker)
                changes["positions_removed"] += 1
                logger.info("Reconcile: removed local position %s", ticker)

        # Add/update positions from Kalshi
        for ticker, p in api_pos_by_ticker.items():
            pos_count = abs(p.get("position", 0))
            side = "yes" if p.get("position", 0) > 0 else "no"
            local = self._positions.get(ticker)

            if local is None:
                avg_price = p.get("market_exposure", 0) / pos_count if pos_count else 50
                self.update_position(ticker, side, pos_count, avg_price)
                changes["positions_added"] += 1
                logger.info("Reconcile: added position %s ×%d", ticker, pos_count)
            elif local["count"] != pos_count:
                self.update_position(
                    ticker, side, pos_count, local["avg_price_cents"],
                )
                changes["positions_updated"] += 1
                logger.info(
                    "Reconcile: updated %s %d→%d",
                    ticker, local["count"], pos_count,
                )

        # -- Reconcile orders: remove local orders gone from API, and merge
        # API orders into local store so ticker/side/price are correct for lookups
        api_order_ids = set()
        for o in api_orders:
            oid = o.get("order_id")
            if not oid:
                continue
            api_order_ids.add(oid)
            ticker = o.get("ticker") or o.get("contract_ticker") or o.get("market_ticker") or ""
            yes_price = o.get("yes_price", 0)
            if isinstance(yes_price, float):
                yes_price = int(round(yes_price * 100)) if yes_price <= 1 else int(yes_price)
            self._orders[oid] = {
                "order_id": oid,
                "ticker": ticker,
                "side": o.get("side", ""),
                "action": o.get("action", ""),
                "price_cents": yes_price,
                "count": o.get("remaining_count", o.get("initial_count", 0)),
                "status": o.get("status", "resting"),
                "placed_at": self._orders.get(oid, {}).get("placed_at", ""),
            }
        self._save_orders()

        for oid in list(self._orders.keys()):
            if oid not in api_order_ids:
                self.remove_order(oid, reason="reconciled_gone")
                changes["orders_removed"] += 1
                logger.info("Reconcile: removed stale order %s", oid)

        logger.info("Reconciliation complete: %s", changes)
        return changes

    # ── Daily P&L ─────────────────────────────────────────────────────

    def get_daily_pnl(self) -> Dict[str, Any]:
        """Load today's P&L tracking."""
        data = self._load_json(self._pnl_path, {})
        today = datetime.date.today().isoformat()
        return data.get(today, {"realised_cents": 0, "trades": 0})

    def save_daily_pnl(self, realised_cents: int, trades: int) -> None:
        """Persist today's P&L."""
        data = self._load_json(self._pnl_path, {})
        today = datetime.date.today().isoformat()
        data[today] = {"realised_cents": realised_cents, "trades": trades}
        self._pnl_path.write_text(json.dumps(data, indent=2))
