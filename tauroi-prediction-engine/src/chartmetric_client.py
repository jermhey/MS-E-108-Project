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

# ═════════════════════════════════════════════════════════════════════════════
#  Composite Momentum Signal
# ═════════════════════════════════════════════════════════════════════════════

def compute_momentum(
    df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    window: int = 7,
) -> float:
    """
    Compute a composite cross-platform momentum score from cached features.

    Combines growth rates across ALL available signals into a single
    normalised score in [-1, 1].  This score captures information that
    the raw listener spot and TikTok velocity miss:

    - spotify_popularity (Spotify's own algorithmic activity score)
    - spotify_followers growth rate
    - instagram_followers growth rate
    - youtube_channel_views growth rate
    - wikipedia_views spike detection
    - deezer_fans growth rate

    Each signal is converted to a z-score-like value, then combined
    with empirical weights.  The result is clipped to [-1, 1].

    Parameters
    ----------
    df : DataFrame
        Cached artist data with Date index or column.
        Must contain ``spotify_monthly_listeners``; all other columns
        are optional and contribute if present.
    as_of : Timestamp or None
        Use data up to this date only (for backtesting).  If None,
        uses the full DataFrame.
    window : int
        Lookback window in days for growth rate computation (default 7).

    Returns
    -------
    float
        Momentum score in [-1, 1].  Positive = cross-platform tailwind.
    """
    if df.empty:
        return 0.0

    # Ensure Date is usable
    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("Date")
    df = df.sort_index()

    if as_of is not None:
        df = df.loc[:as_of]

    if len(df) < window + 1:
        return 0.0

    recent = df.tail(window + 1)

    def _growth_rate(col: str) -> float:
        """7-day growth rate for a column, or 0 if not available."""
        if col not in recent.columns:
            return 0.0
        vals = recent[col].ffill().dropna()
        if len(vals) < 2:
            return 0.0
        old = float(vals.iloc[0])
        new = float(vals.iloc[-1])
        if old <= 0 or np.isnan(old) or np.isnan(new):
            return 0.0
        return (new - old) / old

    def _level_change(col: str) -> float:
        """Absolute change in a level metric (like popularity 0-100)."""
        if col not in recent.columns:
            return 0.0
        vals = recent[col].ffill().dropna()
        if len(vals) < 2:
            return 0.0
        return float(vals.iloc[-1]) - float(vals.iloc[0])

    # ── Individual signal scores ─────────────────────────────────────
    # Each signal gets a normalised score: positive = bullish momentum

    scores = []
    weights = []

    # 1. Spotify popularity change (0-100 scale; +3 in a week is huge)
    pop_change = _level_change("spotify_popularity")
    pop_score = np.clip(pop_change / 5.0, -1.0, 1.0)
    scores.append(pop_score)
    weights.append(3.0)  # highest weight — Spotify's own signal

    # 2. Spotify followers growth rate
    fol_growth = _growth_rate("spotify_followers")
    fol_score = np.clip(fol_growth / 0.02, -1.0, 1.0)  # 2%/wk = max
    scores.append(fol_score)
    weights.append(2.0)

    # 3. Listener growth rate (already captured in trend, lower weight)
    lis_growth = _growth_rate("spotify_monthly_listeners")
    lis_score = np.clip(lis_growth / 0.03, -1.0, 1.0)  # 3%/wk = max
    scores.append(lis_score)
    weights.append(1.0)

    # 4. Instagram followers growth
    ig_growth = _growth_rate("instagram_followers")
    ig_score = np.clip(ig_growth / 0.01, -1.0, 1.0)  # 1%/wk = max
    scores.append(ig_score)
    weights.append(1.5)

    # 5. YouTube views growth
    yt_growth = _growth_rate("youtube_channel_views")
    yt_score = np.clip(yt_growth / 0.05, -1.0, 1.0)  # 5%/wk = max
    scores.append(yt_score)
    weights.append(1.0)

    # 6. Wikipedia views spike (highly indicative of cultural moments)
    wiki_growth = _growth_rate("wikipedia_views")
    wiki_score = np.clip(wiki_growth / 0.50, -1.0, 1.0)  # 50%/wk = max
    scores.append(wiki_score)
    weights.append(2.0)

    # 7. Deezer fans growth (cross-platform streaming confirmation)
    dz_growth = _growth_rate("deezer_fans")
    dz_score = np.clip(dz_growth / 0.02, -1.0, 1.0)  # 2%/wk = max
    scores.append(dz_score)
    weights.append(0.5)

    # 8. TikTok velocity trend (already used, but include for momentum)
    if "tiktok_sound_posts_change" in recent.columns:
        tt = recent["tiktok_sound_posts_change"].fillna(0)
        if len(tt) >= 4:
            recent_avg = float(tt.tail(3).mean())
            older_avg = float(tt.head(3).mean())
            tt_accel = (recent_avg - older_avg) / max(older_avg, 1.0)
            tt_score = np.clip(tt_accel / 0.50, -1.0, 1.0)
        else:
            tt_score = 0.0
    else:
        tt_score = 0.0
    scores.append(tt_score)
    weights.append(1.0)

    # ── Weighted combination ─────────────────────────────────────────
    scores = np.array(scores)
    weights = np.array(weights)
    composite = float(np.average(scores, weights=weights))
    return float(np.clip(composite, -1.0, 1.0))


# ── Seed list: top global Spotify artists by monthly listeners ────────────────
# This provides the starting point for data-first discovery.
# Expanded dynamically from Kalshi market data and cache.
TOP_GLOBAL_ARTISTS_SEED: List[Dict[str, Any]] = [
    {"name": "Bruno Mars",        "cm_id": 3501},
    {"name": "The Weeknd",        "cm_id": 3852},
    {"name": "Taylor Swift",      "cm_id": 2762},
    {"name": "Billie Eilish",     "cm_id": 5596},
    {"name": "Bad Bunny",         "cm_id": 214945},
    {"name": "Ariana Grande",     "cm_id": 795},
    {"name": "Ed Sheeran",        "cm_id": 1280},
    {"name": "Drake",             "cm_id": 1352},
    {"name": "Lady Gaga",         "cm_id": 773},
    {"name": "Sabrina Carpenter", "cm_id": None},
    {"name": "Post Malone",       "cm_id": None},
    {"name": "Rihanna",           "cm_id": None},
    {"name": "Dua Lipa",          "cm_id": None},
    {"name": "Coldplay",          "cm_id": None},
    {"name": "Eminem",            "cm_id": None},
    {"name": "Travis Scott",      "cm_id": None},
    {"name": "SZA",               "cm_id": None},
    {"name": "Kendrick Lamar",    "cm_id": None},
    {"name": "Justin Bieber",     "cm_id": None},
    {"name": "Peso Pluma",        "cm_id": None},
    {"name": "Olivia Rodrigo",    "cm_id": None},
    {"name": "Doja Cat",          "cm_id": None},
    {"name": "Harry Styles",      "cm_id": None},
    {"name": "Tyler, The Creator", "cm_id": None},
    {"name": "Lana Del Rey",      "cm_id": None},
    {"name": "Shakira",           "cm_id": None},
    {"name": "Imagine Dragons",   "cm_id": None},
    {"name": "Miley Cyrus",       "cm_id": None},
    {"name": "Kanye West",        "cm_id": None},
    {"name": "BTS",               "cm_id": None},
]


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

        # Cache state
        self._cache_dir = _CACHE_DIR
        self._cache_ttl_hours = _DEFAULT_CACHE_TTL_HOURS
        self._disabled_scan_sources: set[str] = set()

        # Token state
        self._access_token: str = ""
        self._token_expires_at: float = 0.0  # epoch seconds

        # Global rate-limit state: once ANY call gets a 429, stop ALL
        # API calls until the reset time has passed.  This prevents
        # burning dozens of wasted 429s in the dashboard when iterating
        # across artists / sources.
        self._rate_limit_until: float = 0.0  # epoch seconds

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

    @property
    def is_rate_limited(self) -> bool:
        """True if the API is still within a known rate-limit window."""
        return time.time() < self._rate_limit_until

    @property
    def rate_limit_seconds_left(self) -> float:
        """Seconds remaining until the rate limit resets (0 if not limited)."""
        return max(0.0, self._rate_limit_until - time.time())

    def _request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """Execute an authenticated request with auto-refresh & exponential backoff."""
        # ── Global rate-limit gate: skip the call entirely ─────────
        if self.is_rate_limited:
            mins = self.rate_limit_seconds_left / 60
            raise ChartmetricAPIError(
                f"Rate limited — {mins:.0f}m remaining. "
                f"Skipping {method} {path} (no API call made)."
            )

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
                # ── Parse the actual reset window ─────────────────
                try:
                    retry_after = int(float(
                        resp.headers.get("Retry-After", "60"),
                    ))
                except (ValueError, TypeError):
                    retry_after = 60

                # Also check X-RateLimit-Reset (epoch timestamp)
                try:
                    reset_epoch = float(resp.headers.get("X-RateLimit-Reset", "0"))
                    if reset_epoch > time.time():
                        retry_after = max(retry_after, int(reset_epoch - time.time()))
                except (ValueError, TypeError):
                    pass

                # Set the global gate so NO further calls are wasted
                self._rate_limit_until = time.time() + retry_after
                reset_dt = datetime.datetime.fromtimestamp(
                    self._rate_limit_until,
                ).strftime("%H:%M:%S")
                logger.warning(
                    "Rate limited (429) — ALL API calls paused until %s "
                    "(%d min). Falling back to cache.",
                    reset_dt, retry_after // 60,
                )

                raise ChartmetricAPIError(
                    f"Rate limited (429) — resets at {reset_dt} "
                    f"({retry_after // 60}m). {method} {path}"
                )

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

    # Track rate-limit state for search to avoid wasting calls
    _search_rate_limited: bool = False

    def search_artist(self, name: str) -> int | None:
        """
        Search Chartmetric for an artist by name.

        Returns the Chartmetric **numeric ID** of the best match, or
        ``None`` if the search fails or returns no results.

        When rate-limited, skips the API call entirely and returns None
        to avoid burning remaining quota.
        """
        if self._search_rate_limited:
            return None

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
            except ChartmetricAPIError as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "rate" in exc_str:
                    logger.warning(
                        "Search API rate-limited — disabling for this session"
                    )
                    self._search_rate_limited = True
                    return None
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
                return self._build_scan_result(fresh_df, cm_id, name)

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
        api_failed_required = False

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
                        api_failed_required = True
            except ChartmetricAPIError as exc:
                if required:
                    logger.warning(
                        "  %s — required source %s failed: %s",
                        name, source, exc,
                    )
                    api_failed_required = True
                    break  # stop wasting API calls
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

        # ── Fallback to stale cache when API fails ────────────────
        if (api_failed_required or "spotify_monthly_listeners" not in source_dfs):
            if stale_cache is not None and not stale_cache.empty:
                logger.warning(
                    "  %s — API failed but stale cache available "
                    "(%d rows) — using cached data",
                    name, len(stale_cache),
                )
                # Update TTL so we don't retry for a while
                if use_cache:
                    self._save_to_cache(cm_id, name, stale_cache)
                return self._build_scan_result(stale_cache, cm_id, name)
            logger.warning("  %s — no Spotify data and no cache", name)
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

        return self._build_scan_result(merged, cm_id, name)

    def _build_scan_result(
        self,
        merged: pd.DataFrame,
        cm_id: int,
        name: str,
    ) -> Dict[str, Any] | None:
        """Build the enriched per-artist result dict from a merged DataFrame."""
        import math

        if merged.empty or "spotify_monthly_listeners" not in merged.columns:
            return None

        # Recompute velocity if missing (e.g., from cache load)
        if "tiktok_sound_posts_change" not in merged.columns:
            merged = self._compute_velocity(merged)

        # ── Per-artist sigma ─────────────────────────────────────────
        listeners = merged["spotify_monthly_listeners"].astype(float)
        pct_change = listeners.pct_change().dropna()

        # Use the same estimator as Calibrator.compute_sigma():
        # sample std of daily returns, annualised, with a 5% floor.
        # The old range-implied estimator inflated sigma by 10x+ on
        # multi-year history (it captured structural growth, not vol).
        _SIGMA_FLOOR = 0.05
        if not pct_change.empty and len(pct_change) >= 5:
            sigma_sample = float(pct_change.std()) * math.sqrt(365)
            sigma = max(sigma_sample, _SIGMA_FLOOR)
        else:
            sigma = 0.20  # fallback

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

        # Event impact score (from release calendar if available)
        event_impact = 1.0
        if "event_impact_score" in merged.columns:
            # Check the latest 3 days for any upcoming event
            recent = merged.tail(3)
            event_impact = float(recent["event_impact_score"].max())

        # ── Trend: annualised EWMA growth rate from last 7 days ──────
        # Computed directly from cached data — zero API calls.
        # This gives the MC-OU simulation "memory" of recent trajectory.
        trend = 0.0
        if len(listeners) >= 8:
            recent_listeners = listeners.tail(8)
            log_rets = np.log(
                recent_listeners / recent_listeners.shift(1)
            ).dropna()
            log_rets = log_rets[np.isfinite(log_rets)]
            if len(log_rets) >= 2:
                ewma_daily = float(log_rets.ewm(span=7).mean().iloc[-1])
                trend = float(np.clip(ewma_daily * 365, -2.0, 2.0))

        # ── Composite momentum score from ALL cross-platform signals ──
        momentum = compute_momentum(merged)

        # Playlist reach (best effort)
        playlist_data = self.get_artist_playlist_reach(cm_id)

        return {
            "name": name,
            "cm_id": cm_id,
            "df": merged,
            "listeners": latest_listeners,
            "change_pct": change_pct,
            "sigma": round(sigma, 6),
            "tiktok_velocity": latest_velocity,
            "tiktok_p95": tiktok_p95,
            "norm_velocity": round(norm_velocity, 4),
            "event_impact_score": event_impact,
            "trend": round(trend, 6),
            "momentum": round(momentum, 4),
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
    #  Data-First Discovery: Top Artists by Listeners
    # ═════════════════════════════════════════════════════════════════════

    def discover_top_artists(
        self,
        top_n: int = 15,
        extra_names: List[str] | None = None,
        since: str = DEFAULT_HISTORY_START,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        **Data-first** artist discovery: identify the top artists by
        current Spotify Monthly Listeners, using Chartmetric as the
        sole source of truth.

        This is the inverse of the old flow (which started from Kalshi
        market data).  Now the model discovers the field independently:

        1.  Check the local Parquet cache for previously scanned artists.
        2.  Merge with the global seed list (``TOP_GLOBAL_ARTISTS_SEED``).
        3.  Merge any ``extra_names`` (e.g. from a Kalshi market).
        4.  Resolve names → Chartmetric IDs → fetch full scan data.
        5.  Sort by current monthly listeners descending.
        6.  Return the top ``top_n`` artists with full enriched data.

        The result is a list of artist dicts identical in format to
        ``get_all_artists_scan_data()`` — ready for MC-OU pricing.

        Parameters
        ----------
        top_n : int
            Number of top artists to return (default 15).
        extra_names : list of str, optional
            Additional artist names to include (e.g. from Kalshi).
        since : str
            History start date for full-data fetch.
        use_cache : bool
            Whether to use local Parquet cache (6h TTL).

        Returns
        -------
        list of dict
            Top artists sorted by listeners descending.
        """
        logger.info("=" * 64)
        logger.info("DATA-FIRST DISCOVERY — Top artists by Chartmetric listeners")
        logger.info("=" * 64)

        # ── 1. Build the candidate set from all sources ───────────────
        candidates: Dict[str, int | None] = {}  # name → cm_id or None

        # 1a. From the seed list
        for entry in TOP_GLOBAL_ARTISTS_SEED:
            name = entry["name"]
            cm_id = entry.get("cm_id")
            candidates[name] = cm_id

        # 1b. From the cache manifest (previously scanned artists)
        manifest = self._load_manifest()
        for cm_id_str, meta in manifest.items():
            name = meta.get("name", "")
            if name and name not in candidates:
                candidates[name] = int(cm_id_str)

        # 1c. From extra_names (e.g. Kalshi market artists)
        if extra_names:
            for name in extra_names:
                if name not in candidates:
                    candidates[name] = None

        logger.info(
            "Candidate pool: %d artists (%d seeded, %d cached, %d extra)",
            len(candidates),
            len(TOP_GLOBAL_ARTISTS_SEED),
            len(manifest),
            len(extra_names or []),
        )

        # ── 2. Build name list (put known cm_ids first for efficiency) ─
        ordered: List[tuple[str, int | None]] = sorted(
            candidates.items(),
            key=lambda x: (x[1] is None, x[0]),  # known IDs first
        )

        all_names = [name for name, _ in ordered]

        # ── 3. Fetch enriched data for all candidates ─────────────────
        print(f"\n  Discovering top artists from Chartmetric ({len(all_names)} candidates)...")
        print(f"  Cache: {'ON (6h TTL)' if use_cache else 'OFF'}")

        all_artists = self.get_all_artists_scan_data(
            all_names, since=since, use_cache=use_cache,
        )

        if not all_artists:
            logger.warning("No artists resolved from Chartmetric")
            return []

        # ── 4. Sort by listeners and return top N ─────────────────────
        all_artists.sort(key=lambda a: a["listeners"], reverse=True)
        top = all_artists[:top_n]

        logger.info(
            "Data-first discovery complete: %d/%d resolved, top %d selected",
            len(all_artists), len(all_names), len(top),
        )
        print(f"\n  Top {len(top)} artists by Spotify Monthly Listeners:")
        for i, a in enumerate(top, 1):
            chg = f"{a['change_pct']:+.1f}%" if a.get("change_pct") is not None else "—"
            print(f"    {i:>2d}. {a['name']:<25s}  {a['listeners']:>15,d}  ({chg} 7d)")

        return top

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

        # Winsorize at P99.5 instead of the old mean+3σ hard cap.
        # The old approach replaced genuine viral spikes (the alpha
        # signal) with the rolling median, destroying the information
        # the entire strategy depends on.  P99.5 winsorization
        # compresses only the most extreme glitches (data errors)
        # while preserving real viral events.
        if len(clamped) > 10:
            p995 = float(clamped.quantile(0.995))
            if p995 > 0:
                clamped = clamped.clip(upper=p995)

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
