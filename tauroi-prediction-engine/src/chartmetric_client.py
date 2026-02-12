"""
chartmetric_client.py — Chartmetric REST API Client
=====================================================
Fetches **full historical** artist metrics (Spotify Monthly Listeners,
TikTok Sound Posts, Album/Single Releases) from the Chartmetric API
and converts them into a pandas DataFrame ready for Calibration and
the Pricing Engine.

History Depth
-------------
The Chartmetric ``/stat`` endpoint caps each request at **365 days**.
To get full history (since 2019), this client automatically **paginates**
through yearly windows, concatenates the results, and de-duplicates.

Authentication
--------------
Chartmetric uses a **refresh-token → access-token** flow:

1.  POST ``/api/token`` with ``{"refreshtoken": "<token>"}``
2.  Response: ``{"token": "<short-lived access token>", ...}``
3.  Subsequent requests use ``Authorization: Bearer <token>``
4.  Access tokens expire (~1 hour); the client auto-refreshes.

Endpoints Used
--------------
- ``GET /api/artist/{id}/stat/{source}``  — fan-metric time-series
- ``GET /api/artist/{id}/albums``         — album / single release history
"""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.config import Settings, load_settings
from src.utils import get_logger

logger = get_logger("tauroi.chartmetric")

# ── Constants ────────────────────────────────────────────────────────────────

# Chartmetric enforces a 365-day max per stat request.
_MAX_WINDOW_DAYS = 350  # stay safely under 365

# Default history start date — 6+ years of data for proper vol calibration.
DEFAULT_HISTORY_START = "2019-01-01"


class ChartmetricAuthError(Exception):
    """Raised when token exchange fails."""


class ChartmetricAPIError(Exception):
    """Raised on unexpected API responses."""


class ChartmetricClient:
    """
    Chartmetric REST API client with automatic token refresh
    and paginated full-history fetching.

    Parameters
    ----------
    settings : Settings, optional
        If not supplied, loaded from ``.env`` automatically.
    """

    _TOKEN_URL = "/api/token"

    def __init__(
        self,
        settings: Settings | None = None,
        timeout: int = 30,
    ) -> None:
        self._settings = settings or load_settings(require_secrets=False)
        self._base_url = self._settings.chartmetric_base_url
        self._refresh_token = self._settings.chartmetric_refresh_token
        self._artist_id = self._settings.chartmetric_artist_id
        self._timeout = timeout
        self._session = requests.Session()

        # Token state
        self._access_token: str = ""
        self._token_expires_at: float = 0.0  # epoch seconds

        if not self._refresh_token:
            raise ChartmetricAuthError(
                "Missing CHARTMETRIC_REFRESH_TOKEN in .env. "
                "Request API access at https://app.chartmetric.com."
            )

    # ═════════════════════════════════════════════════════════════════════
    #  Authentication
    # ═════════════════════════════════════════════════════════════════════

    def _ensure_token(self) -> None:
        """Refresh the access token if expired or missing."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_access_token()

    def _refresh_access_token(self) -> None:
        """Exchange the long-lived refresh token for a short-lived access token."""
        url = f"{self._base_url}{self._TOKEN_URL}"
        payload = {"refreshtoken": self._refresh_token}

        logger.info("Refreshing Chartmetric access token...")

        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            raise ChartmetricAuthError(f"Cannot reach Chartmetric API: {exc}")

        if resp.status_code != 200:
            raise ChartmetricAuthError(
                f"Token exchange failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        self._access_token = data.get("token", "")
        expires_in = data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in

        if not self._access_token:
            raise ChartmetricAuthError(
                "Token exchange returned empty token. "
                "Verify your CHARTMETRIC_REFRESH_TOKEN."
            )

        logger.info(
            "Chartmetric token acquired (expires in %ds)", expires_in,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  HTTP
    # ═════════════════════════════════════════════════════════════════════

    def _request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """Execute an authenticated request with auto-refresh & exponential backoff."""
        self._ensure_token()
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

        last_error = ""
        for attempt in range(1 + retries):
            if attempt > 0:
                backoff = min(2 ** attempt, 30)
                logger.info("Retry %d/%d — waiting %ds...", attempt, retries, backoff)
                time.sleep(backoff)

            resp = self._session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=self._timeout,
            )

            if resp.status_code == 401:
                logger.warning("401 received — refreshing token and retrying")
                self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                continue

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                logger.warning("Rate limited (429) — sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                last_error = resp.text[:300]
                logger.warning(
                    "Server error %d on %s %s (attempt %d/%d)",
                    resp.status_code, method, path, attempt + 1, retries + 1,
                )
                continue

            if not resp.ok:
                raise ChartmetricAPIError(
                    f"Chartmetric API error {resp.status_code} "
                    f"for {method} {path}: {resp.text[:300]}"
                )

            return resp.json()

        raise ChartmetricAPIError(
            f"Failed after {retries + 1} attempts for {method} {path}: {last_error}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Connection Check
    # ═════════════════════════════════════════════════════════════════════

    def check_connection(self) -> bool:
        """Verify that the refresh token is valid and we can reach the API."""
        self._refresh_access_token()
        logger.info("Chartmetric connection verified")
        return True

    # ═════════════════════════════════════════════════════════════════════
    #  Artist ID Handling
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_numeric_id(artist_id: str) -> bool:
        """Check if the artist_id is a Chartmetric numeric ID."""
        return artist_id.isdigit()

    # ═════════════════════════════════════════════════════════════════════
    #  Low-Level: Single-Window Stat Fetch
    # ═════════════════════════════════════════════════════════════════════

    def _get_artist_stat_window(
        self,
        artist_id: str,
        source: str,
        *,
        field: str | None = None,
        since: str,
        until: str,
        is_domain_id: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a fan-metric time-series for a single ≤365-day window.

        Returns a list of dicts (each has ``timestp`` + value fields).
        """
        params: Dict[str, Any] = {
            "since": since,
            "until": until,
        }
        if field:
            params["field"] = field
        if is_domain_id:
            params["isDomainId"] = "true"

        path = f"/api/artist/{artist_id}/stat/{source}"
        data = self._request("GET", path, params=params)

        # Chartmetric wraps the payload in "obj" (list or dict)
        obj = data.get("obj", data)

        if isinstance(obj, dict):
            for key in (field, source, "data", "series"):
                if key and key in obj and isinstance(obj[key], list):
                    return obj[key]
            for v in obj.values():
                if isinstance(v, list):
                    return v
            return []

        if isinstance(obj, list):
            return obj

        logger.warning("Unexpected response format for %s/%s: %s",
                        source, field, type(obj))
        return []

    # ═════════════════════════════════════════════════════════════════════
    #  Paginated Full-History Stat Fetch
    # ═════════════════════════════════════════════════════════════════════

    def get_artist_stat(
        self,
        artist_id: str,
        source: str,
        *,
        field: str | None = None,
        since: str | None = None,
        until: str | None = None,
        is_domain_id: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a fan-metric time-series, **automatically paginating**
        through yearly windows to cover the full date range.

        The Chartmetric ``/stat`` endpoint caps each request at 365 days.
        This method breaks ``[since, until]`` into ≤350-day windows,
        fetches each one, and concatenates the results.

        Parameters
        ----------
        artist_id : str
            Chartmetric numeric ID (preferred) or platform domain ID.
        source : str
            Platform: ``spotify``, ``tiktok``, ``youtube``, etc.
        field : str, optional
            Specific metric field (e.g. ``listeners``).
        since : str, optional
            Start date ``YYYY-MM-DD`` (default: ``2019-01-01``).
        until : str, optional
            End date ``YYYY-MM-DD`` (default: today).
        is_domain_id : bool
            If True, ``artist_id`` is a platform domain ID.

        Returns
        -------
        list of dict
        """
        today = datetime.date.today()
        if until is None:
            until = today.isoformat()
        if since is None:
            since = DEFAULT_HISTORY_START

        start = datetime.date.fromisoformat(since)
        end = datetime.date.fromisoformat(until)

        # Single window — no pagination needed
        if (end - start).days <= _MAX_WINDOW_DAYS:
            return self._get_artist_stat_window(
                artist_id, source, field=field,
                since=since, until=until,
                is_domain_id=is_domain_id,
            )

        # ── Paginate through yearly windows ──────────────────────────
        all_rows: List[Dict[str, Any]] = []
        window_start = start
        window_num = 0

        while window_start < end:
            window_end = min(
                window_start + datetime.timedelta(days=_MAX_WINDOW_DAYS),
                end,
            )
            window_num += 1

            logger.info(
                "  [%s] Window %d: %s → %s",
                source, window_num,
                window_start.isoformat(), window_end.isoformat(),
            )

            try:
                rows = self._get_artist_stat_window(
                    artist_id, source, field=field,
                    since=window_start.isoformat(),
                    until=window_end.isoformat(),
                    is_domain_id=is_domain_id,
                )
                all_rows.extend(rows)
                logger.info(
                    "  [%s] Window %d: %d rows", source, window_num, len(rows),
                )
            except ChartmetricAPIError as exc:
                logger.warning(
                    "  [%s] Window %d failed: %s — skipping",
                    source, window_num, exc,
                )

            # Rate-limit courtesy pause between windows
            time.sleep(1.0)
            window_start = window_end + datetime.timedelta(days=1)

        logger.info(
            "[%s] Pagination complete — %d total rows across %d windows",
            source, len(all_rows), window_num,
        )
        return all_rows

    # ═════════════════════════════════════════════════════════════════════
    #  Album / Single Releases
    # ═════════════════════════════════════════════════════════════════════

    def get_artist_releases(
        self,
        artist_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch the artist's album and single release history.

        Endpoint: ``GET /api/artist/{id}/albums``

        Returns a DataFrame with columns:
            release_date (datetime), title (str), type (str: album/single/etc.)
        """
        if artist_id is None:
            artist_id = self._artist_id
        if not artist_id or not self._is_numeric_id(artist_id):
            logger.warning("Releases endpoint requires numeric CM ID — skipping")
            return pd.DataFrame(columns=["release_date", "title", "type"])

        logger.info("Fetching album/single releases for CM#%s...", artist_id)

        all_albums: List[Dict[str, Any]] = []
        offset = 0
        limit = 100

        # Paginate through all albums
        while True:
            path = f"/api/artist/{artist_id}/albums"
            try:
                data = self._request("GET", path, params={
                    "limit": limit,
                    "offset": offset,
                })
            except ChartmetricAPIError as exc:
                logger.warning("Albums fetch failed: %s", exc)
                break

            obj = data.get("obj", data)
            if isinstance(obj, list):
                batch = obj
            elif isinstance(obj, dict):
                # Response might be {"obj": [...]}} or {"obj": {"data": [...]}}
                batch = obj.get("data", obj.get("albums", []))
                if not isinstance(batch, list):
                    batch = []
            else:
                batch = []

            if not batch:
                break

            all_albums.extend(batch)
            logger.info("  Albums page offset=%d: %d items", offset, len(batch))

            if len(batch) < limit:
                break  # last page

            offset += limit
            time.sleep(0.5)

        logger.info("Total releases fetched: %d", len(all_albums))

        if not all_albums:
            return pd.DataFrame(columns=["release_date", "title", "type"])

        # Parse into DataFrame
        records = []
        for album in all_albums:
            # Try multiple date field names
            date_str = (
                album.get("release_date")
                or album.get("released")
                or album.get("release_dates", {}).get("spotify", "")
                if isinstance(album.get("release_dates"), dict) else
                album.get("release_date", "")
            )
            title = album.get("name", album.get("title", "Unknown"))
            album_type = album.get("type", album.get("album_type", "album"))

            if not date_str:
                continue

            try:
                dt = pd.Timestamp(date_str).normalize()
                records.append({
                    "release_date": dt,
                    "title": str(title),
                    "type": str(album_type).lower(),
                })
            except Exception:
                continue

        if not records:
            return pd.DataFrame(columns=["release_date", "title", "type"])

        releases_df = pd.DataFrame(records)
        releases_df = (
            releases_df
            .sort_values("release_date")
            .drop_duplicates(subset=["release_date", "title"], keep="last")
            .reset_index(drop=True)
        )

        logger.info(
            "Releases parsed: %d (earliest: %s, latest: %s)",
            len(releases_df),
            releases_df["release_date"].min().strftime("%Y-%m-%d"),
            releases_df["release_date"].max().strftime("%Y-%m-%d"),
        )
        return releases_df

    # ═════════════════════════════════════════════════════════════════════
    #  Competitor Intelligence
    # ═════════════════════════════════════════════════════════════════════

    def search_artist(self, name: str) -> int | None:
        """
        Search Chartmetric for an artist by name.

        Returns the Chartmetric **numeric ID** of the best match, or
        ``None`` if the search fails or returns no results.
        """
        # Try the general search endpoint first
        for search_path, parse_fn in [
            ("/api/search", self._parse_search_general),
            ("/api/artist/search", self._parse_search_artist),
        ]:
            try:
                data = self._request(
                    "GET", search_path,
                    params={"q": name, "type": "artists", "limit": 5},
                )
                artists = parse_fn(data)
                if artists:
                    return self._best_match(name, artists)
            except ChartmetricAPIError:
                continue

        logger.warning("Artist search found no results for '%s'", name)
        return None

    @staticmethod
    def _parse_search_general(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse ``GET /api/search`` response."""
        obj = data.get("obj", data)
        if isinstance(obj, dict):
            return obj.get("artists", [])
        if isinstance(obj, list):
            return obj
        return []

    @staticmethod
    def _parse_search_artist(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse ``GET /api/artist/search`` response."""
        obj = data.get("obj", data)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
        return []

    @staticmethod
    def _best_match(name: str, artists: List[Dict[str, Any]]) -> int | None:
        """Pick the artist whose name matches most closely."""
        name_lower = name.lower().strip()
        for a in artists:
            if str(a.get("name", "")).lower().strip() == name_lower:
                return int(a["id"])
        # Fallback: first result
        if artists:
            first = artists[0]
            logger.info(
                "Approximate match for '%s' → '%s' (CM#%s)",
                name, first.get("name"), first.get("id"),
            )
            return int(first["id"])
        return None

    def get_artist_latest_listeners(self, artist_id: int) -> Dict[str, Any] | None:
        """
        Fetch the *latest* Spotify Monthly Listeners for an artist.

        Returns ``{"listeners": int, "change_pct": float | None}``
        (7-day percentage change) or ``None`` on failure.
        """
        today = datetime.date.today()
        since = (today - datetime.timedelta(days=14)).isoformat()
        until = today.isoformat()

        rows: List[Dict[str, Any]] = []
        for field in ("listeners", None):
            try:
                rows = self._get_artist_stat_window(
                    str(artist_id), "spotify",
                    field=field, since=since, until=until,
                    is_domain_id=False,
                )
            except ChartmetricAPIError:
                continue
            if rows:
                break

        if not rows:
            return None

        df = self._parse_timeseries(rows, "listeners")
        if df.empty:
            return None

        df = df.sort_values("Date")
        latest = float(df["listeners"].iloc[-1])

        # 7-day change
        change_pct = None
        if len(df) >= 7:
            old = float(df["listeners"].iloc[-7])
            if old > 0:
                change_pct = round((latest - old) / old * 100, 2)

        return {"listeners": int(latest), "change_pct": change_pct}

    def get_competitors_data(
        self,
        competitor_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Resolve competitor names to Chartmetric IDs and fetch their
        current Spotify Monthly Listeners.

        Returns a sorted (by listeners desc) list of dicts::

            [{"name": str, "cm_id": int, "listeners": int,
              "change_pct": float | None}, ...]

        Competitors that cannot be resolved or fetched are omitted.
        """
        logger.info("=" * 64)
        logger.info(
            "COMPETITOR INTELLIGENCE — Resolving %d competitors",
            len(competitor_names),
        )
        logger.info("=" * 64)

        results: List[Dict[str, Any]] = []

        for name in competitor_names:
            cm_id = self.search_artist(name)
            if cm_id is None:
                logger.warning("  Could not resolve '%s' — skipping", name)
                continue

            logger.info("  %s → CM#%d — fetching listeners...", name, cm_id)
            stats = self.get_artist_latest_listeners(cm_id)

            if stats is None:
                logger.warning("  %s (CM#%d) — no listener data", name, cm_id)
                continue

            results.append({
                "name": name,
                "cm_id": cm_id,
                "listeners": stats["listeners"],
                "change_pct": stats["change_pct"],
            })

            chg_str = (
                f"{stats['change_pct']:+.1f}%"
                if stats["change_pct"] is not None else "N/A"
            )
            logger.info(
                "  %s — %s listeners (%s 7d)",
                name, f"{stats['listeners']:,}", chg_str,
            )

            # Rate-limit courtesy between lookups
            time.sleep(0.5)

        # Sort descending by listeners (leader first)
        results.sort(key=lambda r: r["listeners"], reverse=True)

        logger.info(
            "Competitor data fetched: %d/%d resolved",
            len(results), len(competitor_names),
        )
        return results

    # ═════════════════════════════════════════════════════════════════════
    #  High-Level: Get Full-History Model-Ready Metrics
    # ═════════════════════════════════════════════════════════════════════

    def get_artist_metrics(
        self,
        artist_id: str | None = None,
        since: str = DEFAULT_HISTORY_START,
    ) -> pd.DataFrame:
        """
        Fetch **full historical** Spotify Monthly Listeners, TikTok Sound
        Posts, and Release Events, then merge them into a single DataFrame
        matching the schema the Calibrator and PricingEngine expect.

        The output DataFrame has columns:
            Date, spotify_monthly_listeners, tiktok_sound_posts_change,
            event_impact_score

        Parameters
        ----------
        artist_id : str, optional
            Chartmetric numeric ID (default: ``CHARTMETRIC_ARTIST_ID``).
        since : str
            Start date for history (default: ``'2019-01-01'``).

        Returns
        -------
        pd.DataFrame
        """
        if artist_id is None:
            artist_id = self._artist_id
        if not artist_id:
            raise ValueError(
                "No artist_id provided and CHARTMETRIC_ARTIST_ID not set in .env. "
                "Set it to the numeric Chartmetric ID "
                "(from app.chartmetric.com/artist?id=<ID>)."
            )

        use_domain_id = not self._is_numeric_id(artist_id)
        id_label = f"Spotify:{artist_id}" if use_domain_id else f"CM#{artist_id}"
        until = datetime.date.today().isoformat()

        # ══════════════════════════════════════════════════════════════
        #  1. Spotify Monthly Listeners (paginated full history)
        # ══════════════════════════════════════════════════════════════
        logger.info(
            "Fetching Spotify listeners for %s (%s → %s)...",
            id_label, since, until,
        )
        spotify_raw = self.get_artist_stat(
            artist_id, "spotify",
            field="listeners",
            since=since, until=until,
            is_domain_id=use_domain_id,
        )

        # If field-specific fetch returns empty, retry without field filter
        if not spotify_raw:
            logger.info("Retrying Spotify stat without field filter...")
            spotify_raw = self.get_artist_stat(
                artist_id, "spotify",
                since=since, until=until,
                is_domain_id=use_domain_id,
            )

        spotify_df = self._parse_timeseries(spotify_raw, "spotify_monthly_listeners")
        logger.info("Spotify: %d total data points", len(spotify_df))

        # ══════════════════════════════════════════════════════════════
        #  2. TikTok Sound Posts (paginated full history)
        # ══════════════════════════════════════════════════════════════
        logger.info(
            "Fetching TikTok sound posts for %s (%s → %s)...",
            id_label, since, until,
        )

        tiktok_raw: List[Dict[str, Any]] = []
        if use_domain_id:
            logger.warning(
                "TikTok stat requires numeric CM ID — skipping. "
                "Set CHARTMETRIC_ARTIST_ID=<numeric> for TikTok data."
            )
        else:
            try:
                tiktok_raw = self.get_artist_stat(
                    artist_id, "tiktok",
                    since=since, until=until,
                    is_domain_id=False,
                )
            except ChartmetricAPIError as exc:
                logger.warning("TikTok fetch failed: %s — velocity will be 0", exc)

        tiktok_df = self._parse_timeseries(tiktok_raw, "tiktok_sound_posts_cumulative")
        logger.info("TikTok:  %d total data points", len(tiktok_df))

        # ══════════════════════════════════════════════════════════════
        #  3. Album / Single Releases
        # ══════════════════════════════════════════════════════════════
        releases_df = self.get_artist_releases(artist_id)

        # ══════════════════════════════════════════════════════════════
        #  4. Merge on Date
        # ══════════════════════════════════════════════════════════════
        if spotify_df.empty:
            raise ChartmetricAPIError(
                "No Spotify listener data returned from Chartmetric API. "
                "Check your CHARTMETRIC_ARTIST_ID and API access."
            )

        merged = spotify_df.copy()

        if not tiktok_df.empty:
            merged = pd.merge(merged, tiktok_df, on="Date", how="left")
        else:
            logger.warning("No TikTok data — velocity will be 0.")
            merged["tiktok_sound_posts_cumulative"] = np.nan

        merged = merged.sort_values("Date").reset_index(drop=True)

        # ══════════════════════════════════════════════════════════════
        #  5. Compute TikTok velocity (3-day EMA over full history)
        # ══════════════════════════════════════════════════════════════
        merged = self._compute_velocity(merged)

        # ══════════════════════════════════════════════════════════════
        #  6. Compute event_impact_score from releases
        # ══════════════════════════════════════════════════════════════
        merged = self._apply_release_events(merged, releases_df)

        # ── Store release count for dashboard reporting ──────────────
        self._last_release_count = len(releases_df)

        logger.info(
            "API data ready — %d rows (%s → %s) | %d releases",
            len(merged),
            merged["Date"].min().strftime("%Y-%m-%d") if not merged.empty else "?",
            merged["Date"].max().strftime("%Y-%m-%d") if not merged.empty else "?",
            len(releases_df),
        )

        return merged

    # ─────────────────────────────────────────────────────────────────────
    #  Internal Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_timeseries(
        raw: List[Dict[str, Any]],
        value_col: str,
    ) -> pd.DataFrame:
        """
        Convert a Chartmetric stat response to a DataFrame.

        Handles multiple response shapes:
        - ``[{"timestp": "2026-02-10", "value": 123}, ...]``
        - ``[{"timestp": "2026-02-10", "listeners": 123, ...}, ...]``
        - ``[{"date": "2026-02-10", "value": 123}, ...]``
        """
        if not raw:
            return pd.DataFrame(columns=["Date", value_col])

        records = []
        for item in raw:
            dt_str = (
                item.get("timestp")
                or item.get("date")
                or item.get("timestamp")
                or ""
            )
            if not dt_str:
                continue

            val = item.get("value")
            if val is None:
                for k, v in item.items():
                    if k not in ("timestp", "date", "timestamp", "id") and isinstance(v, (int, float)):
                        val = v
                        break

            if val is None:
                continue

            try:
                dt = pd.Timestamp(dt_str).normalize()
                records.append({"Date": dt, value_col: float(val)})
            except Exception:
                continue

        if not records:
            return pd.DataFrame(columns=["Date", value_col])

        df = pd.DataFrame(records)
        df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        return df.reset_index(drop=True)

    @staticmethod
    def _compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive TikTok velocity from cumulative sound posts.

        1.  Daily diff from cumulative.
        2.  Clamp negatives to 0.
        3.  Cap 3-sigma outliers at rolling median.
        4.  Apply 3-day EMA.
        """
        if "tiktok_sound_posts_cumulative" not in df.columns:
            df["tiktok_sound_posts_change"] = 0.0
            return df

        cumul = df["tiktok_sound_posts_cumulative"].astype(float)

        if cumul.isna().all():
            df["tiktok_sound_posts_change"] = 0.0
            return df

        raw_diff = cumul.diff().fillna(0)
        clamped = raw_diff.clip(lower=0)

        mean_val = clamped.mean()
        std_val = clamped.std()
        if std_val > 0:
            threshold = mean_val + 3 * std_val
            rolling_med = clamped.rolling(window=7, min_periods=1).median()
            outlier_mask = clamped > threshold
            clamped = clamped.where(~outlier_mask, rolling_med)

        ema = clamped.ewm(span=3, adjust=False).mean()
        df["tiktok_sound_posts_change"] = ema

        logger.info(
            "TikTok velocity (EMA-3) — min=%.0f, max=%.0f, mean=%.0f",
            ema.min(), ema.max(), ema.mean(),
        )
        return df

    @staticmethod
    def _apply_release_events(
        df: pd.DataFrame,
        releases_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute ``event_impact_score`` by matching release dates.

        Scoring:
        - Date matches a Release (album/single): **3.0**
        - Default:                                **1.0**

        A 3-day forward window is applied so the day *before* and *after*
        a release also get a mild boost (2.0) to capture anticipation
        and immediate aftermath.
        """
        df["event_impact_score"] = 1.0

        if releases_df.empty or "release_date" not in releases_df.columns:
            logger.info("No releases — event_impact_score defaults to 1.0")
            return df

        release_dates = set(releases_df["release_date"].dt.normalize())

        # Build a set of +/- 1 day around each release for the 2.0 halo
        halo_dates: set = set()
        for rd in release_dates:
            halo_dates.add(rd - pd.Timedelta(days=1))
            halo_dates.add(rd + pd.Timedelta(days=1))
        halo_dates -= release_dates  # don't downgrade exact matches

        exact_mask = df["Date"].isin(release_dates)
        halo_mask = df["Date"].isin(halo_dates)

        df.loc[exact_mask, "event_impact_score"] = 3.0
        df.loc[halo_mask, "event_impact_score"] = 2.0

        n_exact = int(exact_mask.sum())
        n_halo = int(halo_mask.sum())
        logger.info(
            "Event scoring — release days=%d (3.0x) | halo days=%d (2.0x) | neutral=%d (1.0x)",
            n_exact, n_halo, len(df) - n_exact - n_halo,
        )
        return df
