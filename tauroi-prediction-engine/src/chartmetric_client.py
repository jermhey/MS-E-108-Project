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
import json
import pathlib
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

# ── Local cache ───────────────────────────────────────────────────────────────
_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache"
_DEFAULT_CACHE_TTL_HOURS = 6


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
        timeout: int = 60,
    ) -> None:
        self._settings = settings or load_settings(require_secrets=False)
        self._base_url = self._settings.chartmetric_base_url
        self._refresh_token = self._settings.chartmetric_refresh_token
        self._artist_id = self._settings.chartmetric_artist_id
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36 "
                "TauroiPredictionEngine/1.0"
            ),
        })

        # Cache state
        self._cache_dir = _CACHE_DIR
        self._cache_ttl_hours = _DEFAULT_CACHE_TTL_HOURS
        self._disabled_scan_sources: set[str] = set()

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

            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                last_error = str(exc)[:500]
                logger.error(
                    "[API ERROR] Network failure on %s %s (attempt %d/%d): %s",
                    method, path, attempt + 1, retries + 1, last_error,
                )
                continue

            if resp.status_code == 401:
                logger.warning(
                    "[API ERROR] 401 Unauthorized on %s %s — "
                    "refreshing token and retrying. Body: %s",
                    method, path, resp.text[:500],
                )
                self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                continue

            if resp.status_code == 429:
                try:
                    retry_after = int(float(
                        resp.headers.get("Retry-After", "5"),
                    ))
                except (ValueError, TypeError):
                    retry_after = 60

                logger.warning(
                    "[API ERROR] 429 Rate Limited on %s %s — "
                    "Retry-After: %ds | Body: %s",
                    method, path, retry_after, resp.text[:500],
                )

                if retry_after > 120:
                    raise ChartmetricAPIError(
                        f"Rate limited (429) with {retry_after}s backoff — "
                        f"aborting {method} {path}. Try again later or use cache."
                    )

                time.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                last_error = resp.text[:500]
                logger.error(
                    "[API ERROR] Server %d on %s %s (attempt %d/%d)\n"
                    "  Body: %s\n"
                    "  Headers: %s",
                    resp.status_code, method, path, attempt + 1, retries + 1,
                    last_error,
                    dict(resp.headers),
                )
                continue

            if not resp.ok:
                logger.error(
                    "[API ERROR] Client %d on %s %s\n"
                    "  Body: %s\n"
                    "  Params: %s",
                    resp.status_code, method, path,
                    resp.text[:500],
                    params,
                )
                raise ChartmetricAPIError(
                    f"Chartmetric API error {resp.status_code} "
                    f"for {method} {path}: {resp.text[:500]}"
                )

            return resp.json()

        raise ChartmetricAPIError(
            f"Failed after {retries + 1} attempts for {method} {path}: {last_error}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Connection Check
    # ═════════════════════════════════════════════════════════════════════

    def check_connection(self) -> bool:
        """
        Verify that the refresh token is valid and we can reach the API.

        If the token refresh itself is rate-limited (429), log a warning
        but do NOT crash — the caller (get_artist_metrics) will fall back
        to the local cache.
        """
        try:
            self._refresh_access_token()
            logger.info("Chartmetric connection verified")
            return True
        except ChartmetricAPIError as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "rate" in err_msg or "401" in err_msg:
                logger.warning(
                    "Chartmetric API rate-limited during auth — "
                    "will attempt cache fallback for data."
                )
                return False
            raise

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
        verbose: bool = True,
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
        verbose : bool
            If True, log per-window progress at INFO level; else DEBUG.

        Returns
        -------
        list of dict
        """
        _log = logger.info if verbose else logger.debug

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

            _log(
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
                _log(
                    "  [%s] Window %d: %d rows", source, window_num, len(rows),
                )
            except ChartmetricAPIError as exc:
                exc_str = str(exc)
                if "Rate limited" in exc_str or "429" in exc_str:
                    _log(
                        "  [%s] Rate limited — aborting pagination",
                        source,
                    )
                    break  # Don't waste calls on remaining windows
                logger.warning(
                    "  [%s] Window %d failed: %s — skipping",
                    source, window_num, exc,
                )

            # Rate-limit courtesy pause between windows
            time.sleep(0.5)
            window_start = window_end + datetime.timedelta(days=1)

        _log(
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
    #  Local Parquet Cache
    # ═════════════════════════════════════════════════════════════════════

    def _cache_path(self, cm_id: int) -> pathlib.Path:
        """Path to the per-artist Parquet cache file."""
        return self._cache_dir / f"scan_{cm_id}.parquet"

    def _manifest_path(self) -> pathlib.Path:
        return self._cache_dir / "_scan_manifest.json"

    def _load_manifest(self) -> Dict[str, Any]:
        mp = self._manifest_path()
        if mp.exists():
            try:
                return json.loads(mp.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2, default=str),
        )

    def _try_load_fresh_cache(
        self, cm_id: int, name: str,
    ) -> pd.DataFrame | None:
        """Return cached DataFrame if it exists AND is within the TTL."""
        manifest = self._load_manifest()
        entry = manifest.get(str(cm_id))
        if not entry:
            return None

        try:
            last_updated = datetime.datetime.fromisoformat(
                entry["last_updated_utc"],
            )
        except (ValueError, KeyError):
            return None

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        age_hours = (now_utc - last_updated).total_seconds() / 3600

        if age_hours > self._cache_ttl_hours:
            return None  # stale

        cp = self._cache_path(cm_id)
        if not cp.exists():
            return None

        try:
            df = pd.read_parquet(cp)
            logger.info(
                "  %s — CACHE HIT (%d rows, %.1fh old, %s → %s)",
                name, len(df), age_hours,
                entry.get("since", "?"), entry.get("until", "?"),
            )
            return df
        except Exception as exc:
            logger.warning("Cache read failed for CM#%d: %s", cm_id, exc)
            return None

    def _load_stale_cache(self, cm_id: int) -> pd.DataFrame | None:
        """Load cached DataFrame regardless of freshness (for delta merge)."""
        cp = self._cache_path(cm_id)
        if not cp.exists():
            return None
        try:
            return pd.read_parquet(cp)
        except Exception:
            return None

    def _save_to_cache(
        self, cm_id: int, name: str, df: pd.DataFrame,
    ) -> None:
        """Persist a merged DataFrame to Parquet and update the manifest."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cp = self._cache_path(cm_id)

        # Ensure all columns are Parquet-compatible
        save_df = df.copy()
        for col in save_df.columns:
            if save_df[col].dtype == object:
                try:
                    save_df[col] = pd.to_numeric(save_df[col], errors="coerce")
                except Exception:
                    save_df = save_df.drop(columns=[col])

        save_df.to_parquet(cp, index=False)

        manifest = self._load_manifest()
        manifest[str(cm_id)] = {
            "name": name,
            "last_updated_utc": datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
            "rows": len(df),
            "since": str(df["Date"].min())[:10] if not df.empty else "?",
            "until": str(df["Date"].max())[:10] if not df.empty else "?",
        }
        self._save_manifest(manifest)
        logger.debug("  Cached %d rows → %s", len(df), cp.name)

    # ═════════════════════════════════════════════════════════════════════
    #  Market-Wide Scan Data (Consistent Per-Artist, Full History)
    # ═════════════════════════════════════════════════════════════════════

    # Sources fetched per artist — (api_source, field, column_name, required)
    # Valid Chartmetric stat sources (from API):
    #   spotify, deezer, facebook, twitter, instagram, youtube_channel,
    #   youtube_artist, wikipedia, bandsintown, soundcloud, tiktok,
    #   twitch, line, melon, bilibili, snap
    _SCAN_SOURCES: List[tuple] = [
        ("spotify",          "listeners",   "spotify_monthly_listeners",   True),
        ("spotify",          "followers",   "spotify_followers",           False),
        ("spotify",          "popularity",  "spotify_popularity",          False),
        ("tiktok",           None,          "tiktok_sound_posts_cumulative", False),
        ("youtube_channel",  None,          "youtube_channel_views",       False),
        ("instagram",        "followers",   "instagram_followers",         False),
        ("deezer",           "fans",        "deezer_fans",                 False),
        ("soundcloud",       None,          "soundcloud_followers",        False),
        ("wikipedia",        None,          "wikipedia_views",             False),
    ]

    def get_artist_scan_data(
        self,
        cm_id: int,
        name: str,
        since: str = DEFAULT_HISTORY_START,
        use_cache: bool = True,
        sigma_window: int | None = None,
    ) -> Dict[str, Any] | None:
        """
        Fetch a **full-history, multi-source** data package for one artist.

        With caching enabled (default), data is stored in
        ``cache/scan_{cm_id}.parquet`` and only delta-fetched when stale.

        Sources fetched (all from Chartmetric ``/stat`` endpoint):

        1. Spotify listeners (primary — **required**)
        2. Spotify followers
        3. Spotify popularity
        4. TikTok sound posts
        5. YouTube channel views
        6. Shazam count
        7. Instagram followers
        8. Deezer fans

        Returns
        -------
        dict or None
            Enriched artist dict with per-artist sigma, velocity, and
            latest values for every source.
        """
        artist_id = str(cm_id)
        today = datetime.date.today()
        until = today.isoformat()

        # ── 1. Cache check (fresh → return immediately) ──────────────
        if use_cache:
            fresh_df = self._try_load_fresh_cache(cm_id, name)
            if fresh_df is not None:
                return self._build_scan_result(
                    fresh_df, cm_id, name, sigma_window=sigma_window,
                )

        # ── 2. Determine fetch range ─────────────────────────────────
        stale_cache = self._load_stale_cache(cm_id) if use_cache else None

        if stale_cache is not None and not stale_cache.empty:
            # Delta: fetch last 30 days and merge with cached history
            fetch_since = (today - datetime.timedelta(days=30)).isoformat()
            logger.info(
                "  %s — cache stale → delta %s → %s",
                name, fetch_since, until,
            )
        else:
            fetch_since = since
            stale_cache = None

        # ── 3. Fetch every source ────────────────────────────────────
        source_dfs: Dict[str, pd.DataFrame] = {}

        for source, field, col_name, required in self._SCAN_SOURCES:
            source_key = f"{source}/{field or 'default'}"

            # Skip sources disabled earlier in this session
            if source_key in self._disabled_scan_sources:
                continue

            try:
                # Use a single-window probe first for non-required sources
                # to avoid wasting 8 paginated calls on unavailable sources
                if not required:
                    probe_since = (today - datetime.timedelta(days=14)).isoformat()
                    try:
                        probe = self._get_artist_stat_window(
                            artist_id, source, field=field,
                            since=probe_since, until=until,
                            is_domain_id=False,
                        )
                    except ChartmetricAPIError:
                        self._disabled_scan_sources.add(source_key)
                        logger.info(
                            "  Source %s unavailable — disabled for session",
                            source_key,
                        )
                        continue

                    if not probe:
                        # Empty but no error — source exists but no data
                        # Still fetch full range (older data may exist)
                        pass

                raw = self.get_artist_stat(
                    artist_id, source, field=field,
                    since=fetch_since, until=until,
                    is_domain_id=False, verbose=False,
                )
                df = self._parse_timeseries(raw, col_name)
                if not df.empty:
                    source_dfs[col_name] = df
                elif required:
                    # Retry without field filter for required sources
                    raw = self.get_artist_stat(
                        artist_id, source,
                        since=fetch_since, until=until,
                        is_domain_id=False, verbose=False,
                    )
                    df = self._parse_timeseries(raw, col_name)
                    if not df.empty:
                        source_dfs[col_name] = df
                    else:
                        logger.warning("  %s — no data for %s", name, col_name)
                        return None
            except ChartmetricAPIError as exc:
                if required:
                    logger.warning(
                        "  %s — required source %s failed: %s",
                        name, source, exc,
                    )
                    return None
                exc_str = str(exc)
                if any(code in exc_str for code in ("403", "404", "400")):
                    self._disabled_scan_sources.add(source_key)
                    logger.info(
                        "  Source %s unavailable — disabled for session",
                        source_key,
                    )
                else:
                    logger.debug(
                        "  %s — %s unavailable: %s",
                        name, source_key, exc,
                    )

        if "spotify_monthly_listeners" not in source_dfs:
            logger.warning("  %s — no Spotify listener data", name)
            return None

        # ── 4. Merge all sources on Date ─────────────────────────────
        def _strip_tz(frame: pd.DataFrame) -> pd.DataFrame:
            if not frame.empty and "Date" in frame.columns:
                dt_col = pd.to_datetime(frame["Date"])
                if hasattr(dt_col.dt, "tz") and dt_col.dt.tz is not None:
                    dt_col = dt_col.dt.tz_localize(None)
                frame["Date"] = dt_col
            return frame

        merged = _strip_tz(source_dfs["spotify_monthly_listeners"].copy())
        for col_name, sdf in source_dfs.items():
            if col_name == "spotify_monthly_listeners":
                continue
            sdf = _strip_tz(sdf.copy())
            if not sdf.empty:
                merged = pd.merge(merged, sdf, on="Date", how="left")

        # Ensure all expected columns exist
        for _, _, col_name, _ in self._SCAN_SOURCES:
            if col_name not in merged.columns:
                merged[col_name] = np.nan

        merged = merged.sort_values("Date").reset_index(drop=True)

        # ── 5. Merge with stale cache (delta) ────────────────────────
        if stale_cache is not None:
            stale_cache = _strip_tz(stale_cache.copy())
            for col in merged.columns:
                if col not in stale_cache.columns:
                    stale_cache[col] = np.nan
            old_rows = stale_cache[~stale_cache["Date"].isin(merged["Date"])]
            merged = pd.concat([old_rows, merged], ignore_index=True)
            merged = (
                merged.sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="last")
                .reset_index(drop=True)
            )

        # ── 6. Compute TikTok velocity ───────────────────────────────
        merged = self._compute_velocity(merged)

        # ── 7. Save to cache ─────────────────────────────────────────
        if use_cache:
            self._save_to_cache(cm_id, name, merged)

        return self._build_scan_result(
            merged, cm_id, name, sigma_window=sigma_window,
        )

    # Default windows for multi-window sigma estimation.
    # Uses the MAX sigma across several recent time scales to
    # automatically capture event risk (album drops, viral moments)
    # without including multi-year structural growth.
    _SIGMA_WINDOWS = [90, 180, 365]
    _SIGMA_FLOOR = 0.05   # minimum credible annualised vol
    _SIGMA_CAP = 1.00     # maximum credible annualised vol (100%)
                          # anything above is data corruption, not real volatility

    def _build_scan_result(
        self,
        merged: pd.DataFrame,
        cm_id: int,
        name: str,
        sigma_window: int | None = None,
    ) -> Dict[str, Any] | None:
        """Build the enriched per-artist result dict from a merged DataFrame.

        Parameters
        ----------
        sigma_window : int or None
            If set, use only the last ``sigma_window`` days for sigma.
            If None (default), use **multi-window estimation**: compute
            sigma over 90d, 180d, and 365d windows and take the max.
            This automatically captures event risk at any time scale
            without including multi-year structural growth.
        """
        import math

        if merged.empty or "spotify_monthly_listeners" not in merged.columns:
            return None

        # Recompute velocity if missing (e.g., from cache load)
        if "tiktok_sound_posts_change" not in merged.columns:
            merged = self._compute_velocity(merged)

        # ── Per-artist sigma (multi-window, no range floor) ──────────
        # Compute realized vol over several recent windows and take the
        # MAX.  This adapts to each artist's recent dynamics: if they
        # had an album drop 4 months ago, the 180d window catches it.
        # If they've been quiet, the 365d window provides a baseline.
        # No manual tuning — the data decides.
        listeners = merged["spotify_monthly_listeners"].astype(float)

        if sigma_window is not None:
            # Explicit single window
            windows = [sigma_window]
        else:
            # Multi-window: use all windows that fit the data
            windows = [w for w in self._SIGMA_WINDOWS if w < len(listeners)]
            if not windows:
                windows = [len(listeners)]

        best_sigma = self._SIGMA_FLOOR
        for w in windows:
            recent = listeners.iloc[-w:]
            pct_change = recent.pct_change().dropna()
            if len(pct_change) >= 5:
                s = float(pct_change.std()) * math.sqrt(365)
                if s > best_sigma:
                    best_sigma = s

        sigma = max(best_sigma, self._SIGMA_FLOOR)
        sigma = min(sigma, self._SIGMA_CAP)  # guard against data corruption

        # ── Latest values (from the last row of the merged DF) ───────
        latest = merged.iloc[-1]

        def _safe_int(val: Any, default: int = 0) -> int:
            """Convert to int, treating NaN/None as default."""
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        def _safe_float(val: Any, default: float = 0.0) -> float:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        latest_listeners = _safe_int(latest.get("spotify_monthly_listeners"))
        latest_velocity = _safe_float(latest.get("tiktok_sound_posts_change"))
        latest_followers = _safe_int(latest.get("spotify_followers"))
        latest_popularity = _safe_int(latest.get("spotify_popularity"))
        latest_youtube = _safe_int(latest.get("youtube_channel_views"))
        latest_instagram = _safe_int(latest.get("instagram_followers"))
        latest_deezer = _safe_int(latest.get("deezer_fans"))
        latest_soundcloud = _safe_int(latest.get("soundcloud_followers"))
        latest_wikipedia = _safe_int(latest.get("wikipedia_views"))

        # TikTok P95 for normalisation
        tiktok_col = merged["tiktok_sound_posts_change"].clip(lower=0)
        tiktok_p95 = (
            float(tiktok_col.quantile(0.95))
            if not tiktok_col.isna().all()
            else 0.0
        )
        norm_velocity = (
            min(latest_velocity / tiktok_p95, 1.0)
            if tiktok_p95 > 0 and latest_velocity > 0
            else 0.0
        )

        # 7-day change
        change_pct = None
        if len(merged) >= 7:
            old = float(merged.iloc[-7]["spotify_monthly_listeners"])
            if old > 0:
                change_pct = round(
                    (latest_listeners - old) / old * 100, 2,
                )

        # ── Listener trend (14-day OLS slope) ────────────────────────
        # More robust than point-to-point change_pct: fits a line
        # through the last 14 days of daily listeners.  The slope
        # (listeners gained per day) directly feeds the MC drift term.
        listener_trend_daily = 0.0  # listeners/day
        trend_window = min(14, len(listeners))
        if trend_window >= 5:
            recent_trend = listeners.iloc[-trend_window:].dropna()
            if len(recent_trend) >= 5:
                x = np.arange(len(recent_trend), dtype=np.float64)
                y = recent_trend.values.astype(np.float64)
                # OLS slope: Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
                x_mean, y_mean = x.mean(), y.mean()
                ss_xx = float(np.sum((x - x_mean) ** 2))
                if ss_xx > 0:
                    listener_trend_daily = float(
                        np.sum((x - x_mean) * (y - y_mean)) / ss_xx
                    )

        # Playlist reach (best effort)
        playlist_data = self.get_artist_playlist_reach(cm_id)

        return {
            "name": name,
            "cm_id": cm_id,
            "df": merged,
            "listeners": latest_listeners,
            "change_pct": change_pct,
            "listener_trend_daily": round(listener_trend_daily, 1),
            "sigma": round(sigma, 6),
            "tiktok_velocity": latest_velocity,
            "tiktok_p95": tiktok_p95,
            "norm_velocity": round(norm_velocity, 4),
            "spotify_followers": latest_followers,
            "spotify_popularity": latest_popularity,
            "youtube_views": latest_youtube,
            "instagram_followers": latest_instagram,
            "deezer_fans": latest_deezer,
            "soundcloud_followers": latest_soundcloud,
            "wikipedia_views": latest_wikipedia,
            "playlist_reach": playlist_data["total_reach"],
            "num_playlists": playlist_data["num_playlists"],
            "editorial_reach": playlist_data["editorial_reach"],
            "data_points": len(merged),
            "history_start": str(merged["Date"].min())[:10],
            "history_end": str(merged["Date"].max())[:10],
        }

    def get_all_artists_scan_data(
        self,
        artist_names: List[str],
        since: str = DEFAULT_HISTORY_START,
        use_cache: bool = True,
        sigma_window: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolve and fetch **consistent** multi-source data for every
        artist in a market, using **full historical data** and local
        Parquet caching.

        For each artist:
        1.  Resolve name → Chartmetric numeric ID (via search).
        2.  Check the local cache — if fresh, return immediately.
        3.  Otherwise fetch full history (since 2019) across all sources.
        4.  Compute per-artist sigma and velocity from own data.

        Returns a list sorted by listeners descending.
        """
        n_artists = len(artist_names)
        logger.info("=" * 64)
        logger.info(
            "FULL MARKET SCAN DATA — %d artists (since %s, cache=%s)",
            n_artists, since, "ON" if use_cache else "OFF",
        )
        logger.info("=" * 64)

        # Build a reverse lookup from manifest: name → cm_id
        manifest = self._load_manifest()
        _name_to_cmid: Dict[str, int] = {}
        for cmid_str, entry in manifest.items():
            cached_name = entry.get("name", "")
            if cached_name:
                _name_to_cmid[cached_name.lower().strip()] = int(cmid_str)

        results: List[Dict[str, Any]] = []

        for i, name in enumerate(artist_names, 1):
            # ── Try cached CM ID first (avoids search API when rate-limited)
            cm_id = _name_to_cmid.get(name.lower().strip())
            if cm_id is not None:
                logger.info(
                    "  [%d/%d] %s → CM#%d (from cache manifest)",
                    i, n_artists, name, cm_id,
                )
            else:
                try:
                    cm_id = self.search_artist(name)
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ) as exc:
                    logger.warning(
                        "  [%d/%d] Timeout searching for '%s': %s — skipping",
                        i, n_artists, name, str(exc)[:80],
                    )
                    continue
                if cm_id is None:
                    logger.warning("  Could not resolve '%s' — skipping", name)
                    continue
                logger.info(
                    "  [%d/%d] %s → CM#%d",
                    i, n_artists, name, cm_id,
                )

            try:
                data = self.get_artist_scan_data(
                    cm_id, name, since=since, use_cache=use_cache,
                    sigma_window=sigma_window,
                )
            except ChartmetricAPIError as exc:
                logger.warning(
                    "  %s — API error (likely rate limit): %s — skipping",
                    name, str(exc)[:120],
                )
                continue
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                logger.warning(
                    "  %s — network timeout: %s — skipping",
                    name, str(exc)[:80],
                )
                continue

            if data is None:
                logger.warning("  %s — no data returned, skipping", name)
                continue

            results.append(data)

            chg = (
                f"{data['change_pct']:+.1f}%"
                if data["change_pct"] is not None
                else "—"
            )
            logger.info(
                "  %s — %s listeners (%s 7d) | sigma=%.1f%% | %d rows (%s→%s)",
                name,
                f"{data['listeners']:,}",
                chg,
                data["sigma"] * 100,
                data["data_points"],
                data.get("history_start", "?"),
                data.get("history_end", "?"),
            )

            time.sleep(0.3)

        results.sort(key=lambda r: r["listeners"], reverse=True)

        logger.info(
            "Scan data complete: %d/%d artists | cache=%s",
            len(results), n_artists, "ON" if use_cache else "OFF",
        )
        return results

    # ═════════════════════════════════════════════════════════════════════
    #  Spotify Playlist Reach
    # ═════════════════════════════════════════════════════════════════════

    # Class-level flag: skip playlist calls if the endpoint is unavailable
    _playlist_endpoint_available: bool = True

    def get_artist_playlist_reach(
        self,
        cm_id: int,
    ) -> Dict[str, Any]:
        """
        Fetch the artist's **current** Spotify playlist placements and
        compute total playlist reach (sum of all playlist follower counts).

        Endpoint: ``GET /api/artist/{id}/playlists/spotify/current``

        If the endpoint returns 401/403 (trial tier), it is disabled for
        the rest of this session to avoid burning API retries.

        Returns
        -------
        dict
            Keys: ``total_reach`` (int), ``num_playlists`` (int),
            ``editorial_reach`` (int), ``num_editorial`` (int).
        """
        result = {
            "total_reach": 0,
            "num_playlists": 0,
            "editorial_reach": 0,
            "num_editorial": 0,
        }

        # Skip if we've already detected the endpoint is unavailable
        if not self._playlist_endpoint_available:
            return result

        try:
            # Use a single-shot request (no retries) to fail fast
            self._ensure_token()
            url = f"{self._base_url}/api/artist/{cm_id}/playlists/spotify/current"
            resp = self._session.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json",
                },
                params={
                    "editorial": "true",
                    "indie": "true",
                    "majorCurator": "true",
                    "limit": 100,
                    "offset": 0,
                },
                timeout=self._timeout,
            )

            if resp.status_code in (401, 403):
                logger.info(
                    "Playlist endpoint unavailable (%d) — disabling for this session",
                    resp.status_code,
                )
                self._playlist_endpoint_available = False
                return result

            if not resp.ok:
                return result

            data = resp.json()
            obj = data.get("obj", data)
            playlists = obj if isinstance(obj, list) else []

            for pl in playlists:
                playlist_info = pl.get("playlist", pl) if isinstance(pl, dict) else {}
                followers = playlist_info.get("followers", 0) or 0
                is_editorial = playlist_info.get("editorial", False)

                result["total_reach"] += int(followers)
                result["num_playlists"] += 1

                if is_editorial:
                    result["editorial_reach"] += int(followers)
                    result["num_editorial"] += 1

        except Exception as exc:
            logger.debug("Playlist fetch for CM#%d failed: %s", cm_id, exc)

        return result

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

        **Resilience:** If the API returns a 429 (rate limit) or 401,
        this method falls back to the local Parquet scan cache. Only
        raises if *both* the API fails AND no cache exists.

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

        # ── Attempt the full API fetch, fall back to cache on 429/401 ─
        try:
            return self._fetch_artist_metrics_from_api(artist_id, since)
        except ChartmetricAPIError as exc:
            err_msg = str(exc).lower()
            is_rate_limit = "429" in err_msg or "rate" in err_msg or "401" in err_msg
            if not is_rate_limit:
                raise  # non-rate-limit error — propagate immediately

            logger.warning(
                "API Rate Limited (429/401). Attempting cache fallback..."
            )
            cm_id = int(artist_id) if self._is_numeric_id(artist_id) else None
            if cm_id is not None:
                cached_df = self._load_stale_cache(cm_id)
                if cached_df is not None and not cached_df.empty:
                    # Ensure the columns the Calibrator/PricingEngine expect
                    if "event_impact_score" not in cached_df.columns:
                        cached_df["event_impact_score"] = 1.0
                    if "tiktok_sound_posts_change" not in cached_df.columns:
                        cached_df["tiktok_sound_posts_change"] = 0.0
                    self._last_release_count = 0
                    logger.warning(
                        "Using committed cache/ file as backup "
                        "(%d rows, %s -> %s)",
                        len(cached_df),
                        cached_df["Date"].min() if "Date" in cached_df.columns else "?",
                        cached_df["Date"].max() if "Date" in cached_df.columns else "?",
                    )
                    return cached_df

            # No cache available either — re-raise the original API error
            raise ChartmetricAPIError(
                f"API rate-limited AND no local cache found for artist {artist_id}. "
                f"Original error: {exc}"
            ) from exc

    def _fetch_artist_metrics_from_api(
        self,
        artist_id: str,
        since: str = DEFAULT_HISTORY_START,
    ) -> pd.DataFrame:
        """
        Internal: perform the actual API fetch for get_artist_metrics.
        Separated so the public method can wrap it with cache fallback.
        """
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

    # Mapping from our column names to known Chartmetric response keys.
    # When the API returns named keys instead of "value", this ensures
    # we always extract the RIGHT metric — never conflating followers
    # with listeners, etc.
    _FIELD_HINTS: Dict[str, List[str]] = {
        "spotify_monthly_listeners": ["listeners", "monthly_listeners", "value"],
        "spotify_followers": ["followers", "value"],
        "spotify_popularity": ["popularity", "value"],
        "tiktok_sound_posts_cumulative": ["value"],
        "youtube_channel_views": ["views", "channel_views", "daily_views", "value"],
        "youtube_channel_subscribers": ["subscribers", "value"],
        "instagram_followers": ["followers", "value"],
        "deezer_fans": ["fans", "value"],
        "soundcloud_followers": ["followers", "value"],
        "wikipedia_views": ["views", "value"],
    }

    @classmethod
    def _parse_timeseries(
        cls,
        raw: List[Dict[str, Any]],
        value_col: str,
    ) -> pd.DataFrame:
        """
        Convert a Chartmetric stat response to a DataFrame.

        Handles multiple response shapes:
        - ``[{"timestp": "2026-02-10", "value": 123}, ...]``
        - ``[{"timestp": "2026-02-10", "listeners": 123, ...}, ...]``
        - ``[{"date": "2026-02-10", "value": 123}, ...]``

        To avoid conflating different metrics (e.g. followers vs listeners),
        the method first checks ``_FIELD_HINTS`` for the expected key names
        before falling back to the first numeric field.
        """
        if not raw:
            return pd.DataFrame(columns=["Date", value_col])

        # Ordered list of keys to look for, specific to this metric
        hint_keys = cls._FIELD_HINTS.get(value_col, ["value"])

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

            # 1. Try hint keys in priority order (most specific first)
            val = None
            for key in hint_keys:
                if key in item and isinstance(item[key], (int, float)):
                    val = item[key]
                    break

            # 2. Last-resort fallback: first numeric field
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
