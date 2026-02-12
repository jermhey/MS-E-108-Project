"""
data_cleaner.py — Chartmetric CSV Cleaning & Merge Pipeline  [DEPRECATED]
===========================================================================
.. deprecated:: Phase 6
    All data is now fetched from the Chartmetric REST API via
    ``src/chartmetric_client.py``.  This module is retained for reference
    only and is no longer imported by any active code path.

Original purpose: ingest raw Chartmetric export CSVs (Playlist Evolution and
Engagement Trends), clean them, engineer the ``total_reach`` feature, and
produce a single model-ready CSV.

Pipeline
--------
1.  **Playlist Evolution**
    - Parse comma-formatted reach columns into integers.
    - Compute ``total_reach = Editorial + User Generated + Algorithmic``.

2.  **Engagement Trends (Bad Bunny)**
    - Keep: Spotify Monthly Listeners, Spotify Popularity Index,
            TikTok Sound Posts (cumulative), YouTube Daily Video Views.
    - Coerce to numeric (some cells are empty).

3.  **Merge** on ``Date`` (inner join).

4.  **Signal Processing — TikTok Velocity (Phase 4)**
    - Recalculate daily diff from cumulative ``TikTok Sound Posts``.
    - Clamp negative values to 0 (database corrections / data refreshes).
    - Replace 3-sigma outliers with rolling median (remove refresh spikes).
    - Apply 3-day Exponential Moving Average (EMA).
    - Output as ``tiktok_sound_posts_change`` for downstream compatibility.

5.  **Output** to ``data/processed/model_ready.csv``.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("tauroi.cleaner")


# ── helpers ─────────────────────────────────────────────────────────────────

def _strip_commas(series: pd.Series) -> pd.Series:
    """Remove thousands-separator commas from a string Series."""
    return series.astype(str).str.replace(",", "", regex=False)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, replacing unparseable values with NaN."""
    return pd.to_numeric(series, errors="coerce")


def _resolve_csv(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    """
    Find exactly one CSV matching *pattern* inside *directory*.

    Raises FileNotFoundError if zero or >1 matches.
    """
    matches = sorted(directory.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {directory}"
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple files match '%s': %s — using most recent.",
            pattern,
            [m.name for m in matches],
        )
    return matches[-1]  # most-recent by name (date-stamped filenames)


# ── stage 1: playlist reach ────────────────────────────────────────────────

def clean_playlist(csv_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load the Spotify Playlist Evolution CSV and compute ``total_reach``.

    Returns a DataFrame indexed by parsed ``Date`` with columns:
        editorial_reach, user_generated_reach, algorithmic_reach, total_reach
    """
    path = _resolve_csv(csv_dir, "*Playlist*Evolution*.csv")
    logger.info("Loading playlist data from %s", path.name)

    df = pd.read_csv(path, dtype=str)

    # Standardise the date column
    df["Date"] = pd.to_datetime(df["Date"].str.strip('"'), format="mixed")

    # The three reach columns — strip commas, coerce to numeric
    reach_cols = {
        "Editorial Playlist Reach": "editorial_reach",
        "User Generated Playlist Total Reach": "user_generated_reach",
        "Algorithmic Playlist Total Reach": "algorithmic_reach",
    }

    for raw_col, clean_col in reach_cols.items():
        df[clean_col] = _to_numeric(_strip_commas(df[raw_col]))

    df["total_reach"] = (
        df["editorial_reach"].fillna(0)
        + df["user_generated_reach"].fillna(0)
        + df["algorithmic_reach"].fillna(0)
    )

    # Preserve Events / Releases columns for event-driven volatility
    for raw_col, clean_col in [("Events", "events"), ("Releases", "releases")]:
        if raw_col in df.columns:
            df[clean_col] = df[raw_col].fillna("")
        else:
            df[clean_col] = ""

    keep = ["Date"] + list(reach_cols.values()) + ["total_reach", "events", "releases"]
    df = df[keep].dropna(subset=["Date"])

    logger.info(
        "Playlist cleaned — %d rows, date range %s → %s",
        len(df),
        df["Date"].min().strftime("%Y-%m-%d"),
        df["Date"].max().strftime("%Y-%m-%d"),
    )
    return df


# ── stage 2: engagement trends ─────────────────────────────────────────────

ENGAGEMENT_KEEP = [
    "Spotify Monthly Listeners",
    "Spotify Popularity Index",
    "TikTok Sound Posts",           # cumulative (source of truth for velocity)
    "TikTok Sound Posts Change",    # raw change (kept for comparison only)
    "YouTube Daily Video Views",
]

ENGAGEMENT_RENAME = {
    "Spotify Monthly Listeners": "spotify_monthly_listeners",
    "Spotify Popularity Index": "spotify_popularity_index",
    "TikTok Sound Posts": "tiktok_sound_posts_cumulative",
    "TikTok Sound Posts Change": "tiktok_sound_posts_change_raw",
    "YouTube Daily Video Views": "youtube_daily_video_views",
}


def clean_engagement(csv_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load the Engagement Trends CSV and extract the key leading indicators.

    Returns a DataFrame indexed by parsed ``Date`` with columns:
        spotify_monthly_listeners, spotify_popularity_index,
        tiktok_sound_posts_change, youtube_daily_video_views
    """
    path = _resolve_csv(csv_dir, "*Engagement*Trends*.csv")
    logger.info("Loading engagement data from %s", path.name)

    df = pd.read_csv(path, dtype=str)

    df["Date"] = pd.to_datetime(df["Date"].str.strip('"'), format="mixed")

    for col in ENGAGEMENT_KEEP:
        if col in df.columns:
            df[col] = _to_numeric(_strip_commas(df[col]))
        else:
            logger.warning("Column '%s' not found in engagement CSV — filling with NaN", col)
            df[col] = float("nan")

    df = df.rename(columns=ENGAGEMENT_RENAME)
    keep = ["Date"] + list(ENGAGEMENT_RENAME.values())
    df = df[keep].dropna(subset=["Date"])

    logger.info(
        "Engagement cleaned — %d rows, date range %s → %s",
        len(df),
        df["Date"].min().strftime("%Y-%m-%d"),
        df["Date"].max().strftime("%Y-%m-%d"),
    )
    return df


# ── stage 3: merge ──────────────────────────────────────────────────────────

def merge_datasets(
    playlist_df: pd.DataFrame,
    engagement_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join playlist and engagement data on Date."""
    merged = pd.merge(playlist_df, engagement_df, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Merged dataset — %d rows, date range %s → %s",
        len(merged),
        merged["Date"].min().strftime("%Y-%m-%d"),
        merged["Date"].max().strftime("%Y-%m-%d"),
    )
    return merged


# ── stage 4: TikTok velocity signal processing ──────────────────────────

def compute_tiktok_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a clean TikTok velocity signal from the cumulative column.

    The raw ``TikTok Sound Posts Change`` from Chartmetric is noisy — it
    contains negative corrections, massive single-day refresh spikes, and
    other database artifacts.  We rebuild the signal from scratch:

    1.  **Recalculate** daily diff from cumulative ``tiktok_sound_posts_cumulative``.
    2.  **Clamp** negatives to 0 (assume data corrections, not real cooldown).
    3.  **Cap outliers** beyond 3 standard deviations at the rolling median
        (removes refresh spikes that don't represent organic virality).
    4.  **Smooth** with a 3-day Exponential Moving Average (EMA).

    The result is stored as ``tiktok_sound_posts_change`` — the column the
    rest of the pipeline (calibration, pricing engine) already expects.
    """
    cumul = df["tiktok_sound_posts_cumulative"].astype(float)

    # ── 1. Recalculate daily change from cumulative ──────────────────
    raw_diff = cumul.diff().fillna(0)

    logger.info(
        "TikTok raw recalculated diff — min=%.0f, max=%.0f, mean=%.0f",
        raw_diff.min(), raw_diff.max(), raw_diff.mean(),
    )

    # ── 2. Clamp negatives to 0 ─────────────────────────────────────
    clamped = raw_diff.clip(lower=0)
    n_negative = int((raw_diff < 0).sum())
    if n_negative:
        logger.info("  Clamped %d negative value(s) to 0", n_negative)

    # ── 3. Replace >3σ outliers with rolling median ──────────────────
    mean_val = clamped.mean()
    std_val = clamped.std()

    if std_val > 0:
        threshold = mean_val + 3 * std_val
        rolling_med = clamped.rolling(window=7, min_periods=1).median()
        outlier_mask = clamped > threshold
        n_outliers = int(outlier_mask.sum())
        if n_outliers:
            clamped = clamped.where(~outlier_mask, rolling_med)
            logger.info(
                "  Outlier threshold=%.0f — replaced %d spike(s) with rolling median",
                threshold, n_outliers,
            )

    # ── 4. Apply 3-day EMA ───────────────────────────────────────────
    ema = clamped.ewm(span=3, adjust=False).mean()

    df["tiktok_sound_posts_change"] = ema

    logger.info(
        "TikTok velocity (cleaned EMA-3) — min=%.0f, max=%.0f, mean=%.0f",
        ema.min(), ema.max(), ema.mean(),
    )

    return df


# ── stage 5: event impact scoring ────────────────────────────────────────────

# Keywords that indicate a *major* industry event (not just a concert)
_MAJOR_EVENT_KEYWORDS = [
    "award", "super bowl", "festival", "grammy", "billboard",
    "vma", "bet award", "coachella", "lollapalooza", "halftime",
]


def compute_event_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse ``events`` and ``releases`` columns and assign a volatility
    multiplier (``event_impact_score``) for each day.

    Scoring hierarchy (highest applicable wins):

    ============  ============================  ========
    Score         Condition                     Rationale
    ============  ============================  ========
    **3.0**       ``releases`` is non-empty      Album / Single drop → massive streaming spike
    **2.0**       ``events`` contains a major    Award show / Super Bowl / Festival —
                  keyword                        global broadcast → millions of new listeners
    **1.0**       (default, incl. concerts)      Standard concerts are local events that
                                                 don't move global listener counts
    ============  ============================  ========

    The score is used downstream as a multiplicative boost on sigma:

        ``sigma_final = sigma_conditional * event_impact_score``
    """
    def _score_row(row: pd.Series) -> float:
        releases = str(row.get("releases", "") or "").strip()
        events = str(row.get("events", "") or "").strip()

        # Releases (Album/Single) → highest impact
        if releases and releases.lower() != "nan":
            return 3.0

        # Major industry event (global broadcast → moves listener counts)
        if events and events.lower() != "nan":
            ev_lower = events.lower()
            for kw in _MAJOR_EVENT_KEYWORDS:
                if kw in ev_lower:
                    return 2.0
            # Standard concerts are local events — they don't move
            # global Spotify monthly listeners, so no vol boost.

        return 1.0

    df["event_impact_score"] = df.apply(_score_row, axis=1)

    n_releases = int((df["event_impact_score"] == 3.0).sum())
    n_major = int((df["event_impact_score"] == 2.0).sum())
    n_neutral = int((df["event_impact_score"] == 1.0).sum())

    logger.info(
        "Event scoring — releases=%d | major_events=%d | neutral=%d",
        n_releases, n_major, n_neutral,
    )
    return df


# ── stage 6: save ───────────────────────────────────────────────────────────

def save_processed(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    filename: str = "model_ready.csv",
) -> pathlib.Path:
    """Write the merged DataFrame to CSV and return the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data → %s (%d rows)", out_path, len(df))
    return out_path


# ── orchestrator ────────────────────────────────────────────────────────────

def run_cleaning_pipeline(
    csv_dir: pathlib.Path | str,
    output_dir: pathlib.Path | str,
) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline.

    Parameters
    ----------
    csv_dir : path to the Chartmetric_CSV folder
    output_dir : path to write model_ready.csv

    Returns
    -------
    pd.DataFrame — the merged, model-ready dataset.
    """
    csv_dir = pathlib.Path(csv_dir)
    output_dir = pathlib.Path(output_dir)

    playlist_df = clean_playlist(csv_dir)
    engagement_df = clean_engagement(csv_dir)
    merged_df = merge_datasets(playlist_df, engagement_df)

    # Phase 4: Robust TikTok velocity signal processing
    merged_df = compute_tiktok_velocity(merged_df)

    # Phase 5: Event-driven volatility scoring
    merged_df = compute_event_impact_score(merged_df)

    save_processed(merged_df, output_dir)

    return merged_df
