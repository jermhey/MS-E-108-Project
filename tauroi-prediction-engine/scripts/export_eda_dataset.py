#!/usr/bin/env python3
"""
Export the full model dataset for EDA.

Aggregates all cached artist time series (cache/scan_*.parquet) into a
single CSV and Parquet file with columns:
  artist_id, artist_name, date, spotify_monthly_listeners,
  tiktok_sound_posts_change, event_impact_score, spotify_followers,
  spotify_popularity, youtube_channel_views, instagram_followers,
  deezer_fans, soundcloud_followers, wikipedia_views, ...

Also writes artist_snapshot_<date>.csv: one row per artist with latest
values (useful for cross-artist EDA).

Usage
-----
  # Export from existing cache (run after at least one full scan)
  python scripts/export_eda_dataset.py

  # Refresh cache from API then export (uses Chartmetric API)
  python scripts/export_eda_dataset.py --refresh

Output
------
  data/eda/model_dataset_YYYY-MM-DD.csv
  data/eda/model_dataset_YYYY-MM-DD.parquet
  data/eda/artist_snapshot_YYYY-MM-DD.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date

import pandas as pd

# Project root (parent of scripts/)
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"


def load_manifest() -> dict:
    manifest_path = CACHE_DIR / "_scan_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def aggregate_from_cache() -> pd.DataFrame | None:
    """Load all cache parquets, add artist_id/artist_name, concatenate."""
    manifest = load_manifest()
    if not manifest:
        return None

    frames = []
    for cm_id_str, entry in manifest.items():
        name = entry.get("name", f"Artist_{cm_id_str}")
        parquet_path = CACHE_DIR / f"scan_{cm_id_str}.parquet"
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        df = df.copy()
        df["artist_id"] = int(cm_id_str)
        df["artist_name"] = name
        frames.append(df)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    # Normalise date column for consistent ordering and CSV output
    combined["date"] = pd.to_datetime(combined["Date"]).dt.strftime("%Y-%m-%d")
    combined = combined.drop(columns=["Date"], errors="ignore")
    # Reorder: identity first, then date, then metrics
    id_cols = ["artist_id", "artist_name", "date"]
    rest = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + rest]
    combined = combined.sort_values(["artist_name", "date"]).reset_index(drop=True)
    return combined


def build_snapshot(combined: pd.DataFrame) -> pd.DataFrame:
    """One row per artist: latest value of each metric."""
    if combined.empty:
        return pd.DataFrame()
    # Latest row per artist (by date)
    idx = combined.groupby("artist_id")["date"].idxmax()
    snapshot = combined.loc[idx].copy()
    snapshot = snapshot.rename(columns={"date": "latest_date"})
    snapshot = snapshot.sort_values("artist_name").reset_index(drop=True)
    return snapshot


def run_refresh() -> bool:
    """Run a scan to populate cache, then return True if we got data."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.chartmetric_client import ChartmetricClient, DEFAULT_HISTORY_START
    from src.config import load_settings

    settings = load_settings(require_secrets=True)
    client = ChartmetricClient(settings=settings)
    client.check_connection()
    # Discover top artists (no Kalshi market needed; use empty extra_names)
    artists = client.discover_top_artists(
        top_n=15,
        extra_names=[],
        since=DEFAULT_HISTORY_START,
        use_cache=False,
    )
    if not artists:
        return False
    names = [a["name"] for a in artists]
    results = client.get_all_artists_scan_data(
        names,
        since=DEFAULT_HISTORY_START,
        use_cache=True,
    )
    return len(results) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export aggregated model dataset for EDA (CSV + Parquet).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch latest data from Chartmetric API (populate/refresh cache) then export.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only write CSV files (skip Parquet).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    today = date.today().isoformat()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        print("Refreshing cache from Chartmetric API...")
        if not run_refresh():
            print("Refresh failed or returned no artists. Check API and .env.")
            sys.exit(1)
        print("Cache refreshed.")

    combined = aggregate_from_cache()
    if combined is None or combined.empty:
        print(
            "No cached data found. Run a full scan first:\n"
            "  python main.py --scan --dry-run\n"
            "Then run this script again. Or use --refresh to fetch and export in one go."
        )
        sys.exit(1)

    # Time series export
    csv_path = out_dir / f"model_dataset_{today}.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {combined.shape[0]:,} rows → {csv_path}")

    if not args.csv_only:
        parquet_path = out_dir / f"model_dataset_{today}.parquet"
        # Parquet needs a proper date column
        combined_parquet = combined.copy()
        combined_parquet["date"] = pd.to_datetime(combined_parquet["date"])
        combined_parquet.to_parquet(parquet_path, index=False)
        print(f"Wrote → {parquet_path}")

    # Artist snapshot (latest per artist)
    snapshot = build_snapshot(combined)
    snapshot_path = out_dir / f"artist_snapshot_{today}.csv"
    snapshot.to_csv(snapshot_path, index=False)
    print(f"Wrote {len(snapshot)} artist snapshots → {snapshot_path}")

    print("\nColumns in time series:", list(combined.columns))
    print("Done.")


if __name__ == "__main__":
    main()
