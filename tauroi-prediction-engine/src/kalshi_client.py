"""
kalshi_client.py — Kalshi Exchange API Client
===============================================
Lightweight ``requests``-based client for the Kalshi v2 REST API.

Authentication uses **RSA-signed** requests per Kalshi's API key spec:
    - Header ``KALSHI-ACCESS-KEY``:       your API key id
    - Header ``KALSHI-ACCESS-TIMESTAMP``: current epoch-ms
    - Header ``KALSHI-ACCESS-SIGNATURE``: RSA-PSS signature of
      ``<timestamp><method><path>`` (and body for POST/PUT)

References
----------
- Kalshi API docs: https://trading-api.readme.io/reference
"""

from __future__ import annotations

import base64
import hashlib
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from src.config import Settings, load_settings
from src.utils import get_logger

logger = get_logger("tauroi.kalshi")


class KalshiAuthError(Exception):
    """Raised when Kalshi authentication fails."""


class KalshiAPIError(Exception):
    """Raised on non-2xx responses from the Kalshi API."""


class KalshiClient:
    """
    Minimal Kalshi v2 API client.

    Parameters
    ----------
    settings : Settings, optional
        If not supplied, credentials are loaded from ``.env`` automatically.
    use_demo : bool
        If True, point at the Kalshi demo environment.

    Example
    -------
    >>> client = KalshiClient()
    >>> client.get_market_status("KXBTC-26FEB14-T97250")
    {'ticker': 'KXBTC-26FEB14-T97250', 'status': 'active', ...}
    """

    def __init__(
        self,
        settings: Settings | None = None,
        use_demo: bool = False,
        timeout: int = 15,
    ) -> None:
        self._settings = settings or load_settings(require_secrets=True)
        self._base_url = (
            self._settings.kalshi_demo_url if use_demo
            else self._settings.kalshi_base_url
        )
        self._timeout = timeout
        self._session = requests.Session()
        self._api_key = self._settings.kalshi_access_key
        self._api_secret = self._settings.kalshi_api_secret
        self._authenticated = False

    # ── authentication ──────────────────────────────────────────────────

    def _full_path(self, path: str) -> str:
        """
        Return the full URL path used for signing.

        Kalshi requires the signature message to include the full path
        starting from ``/trade-api/v2/...``, not just the resource suffix.
        """
        full_url = f"{self._base_url}{path}"
        return urlparse(full_url).path

    def _build_auth_headers(
        self,
        method: str,
        path: str,
    ) -> Dict[str, str]:
        """
        Build Kalshi auth headers.

        Kalshi v2 requires an RSA-PSS signature over:
            ``<timestamp_ms><HTTP_METHOD></trade-api/v2/...>``

        The path in the signature must be the **full URL path** (e.g.
        ``/trade-api/v2/portfolio/balance``), not just the resource
        suffix we pass around internally.
        """
        timestamp_ms = str(int(time.time() * 1000))
        sign_path = self._full_path(path)

        headers = {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # If the secret looks like a PEM private key, use RSA-PSS signing.
        if self._api_secret.startswith("-----BEGIN"):
            signature = self._rsa_sign(timestamp_ms, method.upper(), sign_path)
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
        else:
            # Fallback: HMAC for non-PEM secret formats.
            msg = f"{timestamp_ms}{method.upper()}{sign_path}"
            import hmac
            sig = hmac.new(
                self._api_secret.encode(),
                msg.encode(),
                hashlib.sha256,
            ).digest()
            headers["KALSHI-ACCESS-SIGNATURE"] = base64.b64encode(sig).decode()

        return headers

    def _rsa_sign(
        self,
        timestamp: str,
        method: str,
        path: str,
    ) -> str:
        """
        Produce an RSA-PSS signature using the ``cryptography`` library.

        The message is: ``<timestamp><METHOD><path>``
        """
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError:
            raise ImportError(
                "RSA signing requires the 'cryptography' package. "
                "Install via: pip install cryptography"
            )

        private_key = serialization.load_pem_private_key(
            self._api_secret.encode(),
            password=None,
        )
        message = f"{timestamp}{method}{path}".encode()
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,   # 32 bytes (SHA-256)
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    # ── HTTP helpers ────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] | None = None,
        json_body: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Execute an authenticated request and return the JSON body."""
        url = f"{self._base_url}{path}"
        headers = self._build_auth_headers(method, path)

        resp = self._session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=self._timeout,
        )

        if resp.status_code == 401:
            raise KalshiAuthError(
                f"Authentication failed ({resp.status_code}): {resp.text}"
            )
        if not resp.ok:
            raise KalshiAPIError(
                f"Kalshi API error {resp.status_code}: {resp.text}"
            )

        return resp.json()

    # ── public API methods ──────────────────────────────────────────────

    def check_connection(self) -> bool:
        """
        Verify that we can reach the Kalshi API and our credentials work.

        Hits the exchange status endpoint (public, no auth required)
        then attempts an authenticated call.

        Returns True if both succeed.
        """
        try:
            # 1. Public health check — exchange schedule
            url = f"{self._base_url}/exchange/schedule"
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            logger.info("Kalshi exchange reachable — status OK")

            # 2. Authenticated check — fetch account balance
            self._request("GET", "/portfolio/balance")
            self._authenticated = True
            logger.info("Kalshi authentication successful")
            return True

        except KalshiAuthError:
            logger.error("Kalshi authentication FAILED — check your API keys")
            raise
        except requests.RequestException as exc:
            logger.error("Cannot reach Kalshi API: %s", exc)
            raise

    def get_market_status(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch full market details for a given ticker.

        Endpoint: GET /markets/{ticker}

        Returns the JSON response dict which includes:
            ticker, title, status, yes_price, no_price, volume, etc.
        """
        path = f"/markets/{ticker}"
        data = self._request("GET", path)
        market = data.get("market", data)

        logger.info(
            "Market %s — status=%s | yes=%.2f | no=%.2f",
            market.get("ticker", ticker),
            market.get("status", "unknown"),
            market.get("yes_price", 0) / 100 if isinstance(market.get("yes_price"), int) else market.get("yes_price", 0),
            market.get("no_price", 0) / 100 if isinstance(market.get("no_price"), int) else market.get("no_price", 0),
        )
        return market

    def get_markets(
        self,
        limit: int = 25,
        status: str = "open",
        series_ticker: str | None = None,
        cursor: str | None = None,
    ) -> tuple[list[Dict[str, Any]], str | None]:
        """
        List markets with optional filters.

        Endpoint: GET /markets

        Returns (markets_list, next_cursor).
        """
        params: Dict[str, Any] = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor

        data = self._request("GET", "/markets", params=params)
        markets = data.get("markets", [])
        next_cursor = data.get("cursor", None)
        logger.info("Fetched %d markets (status=%s)", len(markets), status)
        return markets, next_cursor

    def get_all_markets(
        self,
        status: str = "open",
        max_pages: int = 10,
    ) -> list[Dict[str, Any]]:
        """
        Paginate through all open markets (up to ``max_pages`` pages of 200).
        """
        all_markets: list[Dict[str, Any]] = []
        cursor: str | None = None

        for page in range(max_pages):
            batch, cursor = self.get_markets(
                limit=200, status=status, cursor=cursor,
            )
            all_markets.extend(batch)
            if not cursor or not batch:
                break

        logger.info("Total markets fetched: %d (pages=%d)", len(all_markets), page + 1)
        return all_markets

    # ── Two-Stage Market Discovery ──────────────────────────────────────

    # Streaming-context whitelist — an event MUST contain at least one of
    # these keywords (case-insensitive) to be considered streaming-related.
    STREAMING_KEYWORDS = [
        "spotify", "listener", "stream", "top artist",
        "monthly listener", "most listened", "top song",
        "top album",
    ]

    def get_events(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
        with_nested_markets: bool = False,
    ) -> tuple[list[Dict[str, Any]], str | None]:
        """Fetch events (groups of related markets)."""
        params: Dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if with_nested_markets:
            params["with_nested_markets"] = "true"

        data = self._request("GET", "/events", params=params)
        events = data.get("events", [])
        next_cursor = data.get("cursor", None)
        return events, next_cursor

    def get_all_events(
        self,
        status: str = "open",
        max_pages: int = 25,
    ) -> list[Dict[str, Any]]:
        """Paginate through all open events."""
        all_events: list[Dict[str, Any]] = []
        cursor: str | None = None

        for page in range(max_pages):
            batch, cursor = self.get_events(
                status=status, limit=200, cursor=cursor,
            )
            all_events.extend(batch)
            if not cursor or not batch:
                break

        logger.info("Total events fetched: %d", len(all_events))
        return all_events

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """Fetch a single event with its nested markets."""
        data = self._request(
            "GET",
            f"/events/{event_ticker}",
            params={"with_nested_markets": "true"},
        )
        return data.get("event", data)

    def find_active_listener_market(
        self,
        artist_name: str,
    ) -> tuple[Optional[Dict[str, Any]], int]:
        """
        Two-Stage Discovery: find a streaming-specific Kalshi market for
        an artist, identify the market type, and extract competitors.

        Pipeline
        --------
        **Step 1 — Event Discovery:**
            Paginate through ALL open events.  Filter to events whose
            title or event_ticker contains a streaming keyword.

        **Step 2 — Artist Match:**
            For each streaming event, fetch its nested markets.
            Kalshi stores the artist name in ``yes_sub_title`` or
            ``custom_strike.Artist``, not in ``subtitle``.

        **Step 3 — Context Filter:**
            Reject events that mention the artist but are not
            streaming-related (Grammy, dating, etc.).

        **Step 4 — Sort by Expiry:**
            Prefer "monthly" events.  Among equals, pick nearest expiry.

        **Step 5 — Holistic Discovery:**
            Classify market as ``winner_take_all`` (multiple sibling
            artist options) or ``binary`` (single threshold contract).
            Extract all competitors from the parent event.

        Returns
        -------
        (enriched_result | None, rejected_count)
            enriched_result is a dict with keys:
                ``target_contract`` — full market dict with live prices
                ``event_title``     — parent event question
                ``event_ticker``    — parent event ticker
                ``market_type``     — "winner_take_all" or "binary"
                ``competitors``     — list of dicts with sibling artist info
        """
        artist_lower = artist_name.lower()

        # ── Step 1: Fetch all events & filter to streaming context ──────
        logger.info("Step 1 — Fetching all open events...")
        all_events = self.get_all_events(status="open")

        streaming_events: list[Dict[str, Any]] = []
        for e in all_events:
            title = e.get("title", "").lower()
            ticker = e.get("event_ticker", "").lower()
            blob = f"{title} {ticker}"
            if any(kw in blob for kw in self.STREAMING_KEYWORDS):
                streaming_events.append(e)

        logger.info(
            "Step 1 — %d streaming events out of %d total",
            len(streaming_events), len(all_events),
        )

        # ── Step 2 & 3: Search for the artist inside streaming events ───
        artist_candidates: list[Dict[str, Any]] = []
        rejected = 0

        for e in streaming_events:
            event_ticker = e.get("event_ticker", "")
            event_title = e.get("title", "")

            try:
                full_event = self.get_event(event_ticker)
            except Exception:
                continue

            markets = full_event.get("markets", [])

            for m in markets:
                yes_sub = m.get("yes_sub_title", "").lower()
                no_sub = m.get("no_sub_title", "").lower()
                custom = m.get("custom_strike", {})
                custom_artist = (
                    custom.get("Artist", "").lower()
                    if isinstance(custom, dict) else ""
                )
                ticker_suffix = m.get("ticker", "").lower()

                searchable = f"{yes_sub} {no_sub} {custom_artist} {ticker_suffix}"

                if artist_lower in searchable:
                    logger.info(
                        "  Checking Market: %s (Option: %s)",
                        event_title,
                        m.get("yes_sub_title", m.get("subtitle", "—")),
                    )
                    logger.info("    -> PASS (artist + streaming context)")
                    artist_candidates.append({
                        "event": full_event,
                        "market": m,
                    })

        # Count non-streaming rejects
        non_streaming = [e for e in all_events if e not in streaming_events]
        for e in non_streaming:
            title = e.get("title", "").lower()
            ticker = e.get("event_ticker", "").lower()
            if artist_lower in f"{title} {ticker}":
                logger.info(
                    "  Checking Event: %s — REJECT (no streaming keywords)",
                    e.get("title", "?"),
                )
                rejected += 1

        logger.info(
            "Step 2/3 — %d streaming match(es), %d non-streaming rejected",
            len(artist_candidates), rejected,
        )

        if not artist_candidates:
            return None, rejected

        # ── Step 4: Sort & select ───────────────────────────────────────
        def _sort_key(entry: Dict[str, Any]) -> tuple[int, str]:
            title = entry["event"].get("title", "").lower()
            priority = 0 if "monthly" in title else 1
            expiry = entry["market"].get(
                "expiration_time",
                entry["market"].get("close_time", "9999"),
            )
            return (priority, str(expiry))

        artist_candidates.sort(key=_sort_key)
        best_entry = artist_candidates[0]
        best_market_ticker = best_entry["market"].get("ticker", "")
        parent_event = best_entry["event"]

        logger.info(
            "Step 4 — LOCKED: %s | %s | option=%s",
            best_market_ticker,
            parent_event.get("title", "?"),
            best_entry["market"].get("yes_sub_title", "—"),
        )

        # ── Step 5: Holistic Discovery — competitors & market type ──────
        sibling_markets = parent_event.get("markets", [])
        competitors: list[Dict[str, Any]] = []

        for m in sibling_markets:
            sib_artist = (
                m.get("yes_sub_title", "")
                or m.get("custom_strike", {}).get("Artist", "")
            )
            sib_ticker = m.get("ticker", "")

            if sib_ticker == best_market_ticker:
                continue  # skip the target artist itself

            if sib_artist:
                competitors.append({
                    "name": sib_artist,
                    "ticker": sib_ticker,
                    "yes_bid": m.get("yes_bid", 0) or 0,
                    "yes_ask": m.get("yes_ask", 0) or 0,
                    "last_price": m.get("last_price", 0) or 0,
                    "volume": m.get("volume", 0) or 0,
                })

        # Classify: if there are sibling options it's "Winner Take All"
        if len(competitors) > 0:
            market_type = "winner_take_all"
        else:
            market_type = "binary"

        logger.info(
            "Step 5 — market_type=%s | %d competitor(s): %s",
            market_type,
            len(competitors),
            [c["name"] for c in competitors],
        )

        # Re-fetch via get_market_status for live bid/ask prices
        live_market = self.get_market_status(best_market_ticker)

        enriched = {
            "target_contract": live_market,
            "event_title": parent_event.get("title", "?"),
            "event_ticker": parent_event.get("event_ticker", "?"),
            "market_type": market_type,
            "competitors": sorted(
                competitors,
                key=lambda c: c.get("last_price", 0),
                reverse=True,
            ),
        }

        return enriched, rejected

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated
