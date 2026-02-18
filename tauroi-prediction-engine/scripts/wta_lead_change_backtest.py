#!/usr/bin/env python3
"""
wta_lead_change_backtest.py — WTA Lead-Change Bias Detector
=============================================================
Replays historical months day-by-day through the full MC-OU WTA engine
to test whether the model rubber-stamps the current daily leader or
correctly assigns uncertainty during lead changes.

Target months:
  - July 2025:  Razor-thin race, lead changed twice, Bruno Mars won by 0.06%
  - Sept 2025:  Clean lead flip — Bruno Mars → The Weeknd mid-month
  - Jan  2026:  Reverse flip — The Weeknd → Bruno Mars

For each day d in the target month, the script:
  1. Calibrates per-artist parameters from the preceding 90 days (no look-ahead)
  2. Runs MonteCarloOU.simulate_wta() with all 5 artists
  3. Records win probabilities for each artist
  4. Compares model predictions to the actual month-end outcome
"""
from __future__ import annotations

import sys
import pathlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import Calibrator, CalibrationResult
from src.pricing_engine import MonteCarloOU, OUArtistInput


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

CACHE_FILES = {
    "Bruno Mars":    PROJECT_ROOT / "cache" / "scan_3501.parquet",
    "The Weeknd":    PROJECT_ROOT / "cache" / "scan_3852.parquet",
    "Bad Bunny":     PROJECT_ROOT / "cache" / "scan_214945.parquet",
    "Taylor Swift":  PROJECT_ROOT / "cache" / "scan_2762.parquet",
    "Billie Eilish": PROJECT_ROOT / "cache" / "scan_5596.parquet",
}

TARGET_MONTHS = [
    ("2025-07", "July 2025: Razor-thin, double lead change (BM wins by 0.06%)"),
    ("2025-09", "Sept 2025: Clean flip — Bruno Mars → The Weeknd"),
    ("2026-01", "Jan 2026: Reverse flip — The Weeknd → Bruno Mars"),
]

CALIBRATION_LOOKBACK = 90   # days of history before the target month
MC_PATHS = 10_000
MC_SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ArtistHistory:
    name: str
    df: pd.DataFrame          # Full history with Date index
    listeners: pd.Series      # spotify_monthly_listeners, Date-indexed
    tiktok: pd.Series         # tiktok_sound_posts_change, Date-indexed


def load_all_artists() -> Dict[str, ArtistHistory]:
    """Load cached parquet files for all artists."""
    artists = {}
    for name, path in CACHE_FILES.items():
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        # Forward-fill small gaps (weekends, missing days)
        df = df.asfreq("D").ffill()

        listeners = df["spotify_monthly_listeners"].astype(float)
        tiktok_col = "tiktok_sound_posts_change"
        if tiktok_col in df.columns:
            tiktok = df[tiktok_col].fillna(0).astype(float)
        else:
            tiktok = pd.Series(0.0, index=df.index)

        artists[name] = ArtistHistory(
            name=name, df=df, listeners=listeners, tiktok=tiktok,
        )
    return artists


# ═════════════════════════════════════════════════════════════════════════════
#  PER-ARTIST CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_artist(
    history: ArtistHistory,
    as_of: pd.Timestamp,
    lookback: int = CALIBRATION_LOOKBACK,
) -> CalibrationResult | None:
    """
    Calibrate model parameters for one artist using data up to `as_of`.
    Uses a rolling window of `lookback` days before `as_of`.
    """
    start = as_of - pd.Timedelta(days=lookback)
    window = history.df.loc[start:as_of].copy()

    if len(window) < 30:
        return None

    # Build the DataFrame the Calibrator expects
    cal_df = pd.DataFrame({
        "Date": window.index,
        "spotify_monthly_listeners": window["spotify_monthly_listeners"].values,
        "tiktok_sound_posts_change": window.get(
            "tiktok_sound_posts_change",
            pd.Series(0.0, index=window.index),
        ).values,
    })

    try:
        calibrator = Calibrator.from_dataframe(cal_df)
        return calibrator.run()
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  DAY-BY-DAY WTA SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DailyWTASnapshot:
    date: str
    days_remaining: int
    daily_leader: str             # who actually leads on this day
    daily_leader_listeners: float
    gap_to_second: float          # listener gap (leader - #2)
    gap_pct: float                # gap as % of leader
    probs: Dict[str, float]       # model probability per artist
    model_favourite: str          # highest-probability artist
    model_fav_prob: float         # probability of the favourite
    actual_winner: str            # month-end winner (known ex-post)
    model_correct: bool           # does model favourite == actual winner?


def run_month_backtest(
    artists: Dict[str, ArtistHistory],
    year_month: str,
    description: str,
) -> List[DailyWTASnapshot]:
    """
    Run the WTA model day-by-day through a target month.
    """
    ym = pd.Timestamp(f"{year_month}-01")
    month_end = ym + pd.offsets.MonthEnd(0)

    # Build combined listener series for the month
    names = sorted(artists.keys())
    combined = pd.DataFrame()
    for name in names:
        combined[name] = artists[name].listeners

    combined = combined.sort_index()

    # Determine the actual month-end winner
    eom_row = combined.loc[:month_end].iloc[-1]
    actual_winner = eom_row.idxmax()

    print(f"\n{'='*72}")
    print(f"  {description}")
    print(f"  Actual winner: {actual_winner} "
          f"({eom_row[actual_winner]:,.0f} listeners)")
    print(f"{'='*72}")

    # Get all days in the target month where we have data
    month_dates = combined.loc[ym:month_end].index

    snapshots: List[DailyWTASnapshot] = []
    last_cal: Dict[str, CalibrationResult] = {}

    for day in month_dates:
        days_remaining = (month_end - day).days
        T_years = max(days_remaining, 1) / 365.0

        # Current listener counts
        day_listeners = {}
        for name in names:
            val = artists[name].listeners.get(day, np.nan)
            if np.isnan(val):
                # Use last known value
                before = artists[name].listeners.loc[:day].dropna()
                val = before.iloc[-1] if len(before) > 0 else 0
            day_listeners[name] = val

        # Daily leader
        sorted_artists = sorted(
            day_listeners.items(), key=lambda x: x[1], reverse=True,
        )
        daily_leader = sorted_artists[0][0]
        daily_leader_listeners = sorted_artists[0][1]
        second_listeners = sorted_artists[1][1] if len(sorted_artists) > 1 else 0
        gap = daily_leader_listeners - second_listeners
        gap_pct = gap / daily_leader_listeners * 100 if daily_leader_listeners > 0 else 0

        # Calibrate each artist (every 7 days or first day)
        day_idx = (day - ym).days
        if day_idx == 0 or day_idx % 7 == 0 or not last_cal:
            for name in names:
                cal = calibrate_artist(artists[name], day)
                if cal is not None:
                    last_cal[name] = cal

        # Build OUArtistInput for each artist
        ou_inputs = []
        global_jump_intensity = 12.0
        global_jump_std = 0.04
        global_theta = 0.1
        global_jump_beta = 0.0

        for name in names:
            cal = last_cal.get(name)
            listeners = day_listeners[name]

            if cal is not None:
                sigma = cal.sigma
                jump_beta = cal.jump_beta
                norm_vel = 0.0
                tiktok_val = artists[name].tiktok.get(day, 0)
                if cal.tiktok_p95 > 0 and tiktok_val > 0:
                    norm_vel = min(tiktok_val / cal.tiktok_p95, 1.0)
                ji = cal.jump_intensity
                js = cal.jump_std
                theta = cal.theta
                trend = cal.trend
                global_theta = theta
                global_jump_beta = jump_beta
                global_jump_intensity = ji
                global_jump_std = js
            else:
                sigma = 0.20
                norm_vel = 0.0
                ji = None
                js = None
                trend = 0.0

            # Fallback trend: compute directly from listener history
            # if calibration didn't provide one (e.g., early days)
            if trend == 0.0 and name in artists:
                hist = artists[name].listeners.loc[:day].dropna()
                if len(hist) >= 8:
                    recent = hist.tail(8)
                    log_rets = np.log(
                        recent / recent.shift(1)
                    ).dropna()
                    log_rets = log_rets[np.isfinite(log_rets)]
                    if len(log_rets) >= 2:
                        ewma_daily = float(
                            log_rets.ewm(span=7).mean().iloc[-1]
                        )
                        trend = float(
                            np.clip(ewma_daily * 365, -2.0, 2.0)
                        )

            ou_inputs.append(OUArtistInput(
                name=name,
                listeners=listeners,
                sigma=sigma,
                norm_velocity=norm_vel,
                jump_intensity=ji,
                jump_std=js,
                event_impact_score=1.0,
                trend=trend,
            ))

        # Run MC-OU WTA simulation
        mc = MonteCarloOU(
            theta=global_theta,
            n_paths=MC_PATHS,
            seed=MC_SEED + day_idx,   # different seed each day for diversity
            jump_intensity=global_jump_intensity,
            jump_std=global_jump_std,
            laplace_alpha=25.0,
        )

        results = mc.simulate_wta(
            artists=ou_inputs,
            T=T_years,
            jump_beta=global_jump_beta,
        )

        # Extract probabilities
        probs = {r.name: r.probability for r in results}
        model_fav = max(probs, key=probs.get)
        model_fav_prob = probs[model_fav]

        snap = DailyWTASnapshot(
            date=day.strftime("%Y-%m-%d"),
            days_remaining=days_remaining,
            daily_leader=daily_leader,
            daily_leader_listeners=daily_leader_listeners,
            gap_to_second=gap,
            gap_pct=gap_pct,
            probs=probs,
            model_favourite=model_fav,
            model_fav_prob=model_fav_prob,
            actual_winner=actual_winner,
            model_correct=(model_fav == actual_winner),
        )
        snapshots.append(snap)

    return snapshots


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS & REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def analyze_month(
    snapshots: List[DailyWTASnapshot],
    description: str,
) -> Dict[str, Any]:
    """Analyze model behavior during a month and report findings."""

    actual_winner = snapshots[0].actual_winner
    names = sorted(snapshots[0].probs.keys())

    # Focus on the top two (by end-of-month)
    final_probs = snapshots[-1].probs
    sorted_by_prob = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    top_two = [actual_winner]
    for name, _ in sorted_by_prob:
        if name != actual_winner and len(top_two) < 2:
            top_two.append(name)

    print(f"\n{'─'*72}")
    print(f"  ANALYSIS: {description}")
    print(f"  Actual winner: {actual_winner}")
    print(f"{'─'*72}\n")

    # Table header
    header = (
        f"  {'Date':>10s}  {'Days':>4s}  {'Daily Leader':>15s}  "
        f"{'Gap%':>5s}  "
    )
    for name in top_two:
        short = name.split()[-1][:8]  # last name, truncated
        header += f"{'P('+short+')':>10s}  "
    header += f"{'Model Fav':>15s}  {'Correct':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Day-by-day rows
    model_correct_count = 0
    leader_bias_count = 0      # model fav == daily leader
    leader_matches_winner = 0  # daily leader == actual winner

    for snap in snapshots:
        row = (
            f"  {snap.date:>10s}  {snap.days_remaining:>4d}  "
            f"{snap.daily_leader:>15s}  {snap.gap_pct:>5.1f}  "
        )
        for name in top_two:
            p = snap.probs.get(name, 0)
            row += f"{p:>9.1%}   "

        correct_str = "  YES" if snap.model_correct else "   NO"
        row += f"{snap.model_favourite:>15s}  {correct_str:>7s}"
        print(row)

        if snap.model_correct:
            model_correct_count += 1
        if snap.model_favourite == snap.daily_leader:
            leader_bias_count += 1
        if snap.daily_leader == actual_winner:
            leader_matches_winner += 1

    n = len(snapshots)
    print()
    print(f"  {'─'*60}")
    print(f"  Model predicted correct winner: "
          f"{model_correct_count}/{n} days ({model_correct_count/n:.0%})")
    print(f"  Model favourite == daily leader: "
          f"{leader_bias_count}/{n} days ({leader_bias_count/n:.0%})  "
          f"← LEADER BIAS metric")
    print(f"  Daily leader == actual winner: "
          f"{leader_matches_winner}/{n} days ({leader_matches_winner/n:.0%})  "
          f"← baseline")

    # Key question: when the lead changed, how fast did the model adapt?
    # Find the first day where the daily leader flipped
    leader_changes = []
    for i in range(1, len(snapshots)):
        if snapshots[i].daily_leader != snapshots[i-1].daily_leader:
            leader_changes.append(i)

    if leader_changes:
        print(f"\n  Lead changes detected on days: "
              f"{[snapshots[i].date for i in leader_changes]}")

        for idx in leader_changes:
            prev = snapshots[idx - 1]
            curr = snapshots[idx]
            print(f"\n  LEAD CHANGE: {prev.daily_leader} → {curr.daily_leader} "
                  f"on {curr.date}")
            print(f"    Day before: P({prev.daily_leader}) = "
                  f"{prev.probs[prev.daily_leader]:.1%}, "
                  f"P({curr.daily_leader}) = "
                  f"{prev.probs[curr.daily_leader]:.1%}")
            print(f"    Day of flip: P({prev.daily_leader}) = "
                  f"{curr.probs[prev.daily_leader]:.1%}, "
                  f"P({curr.daily_leader}) = "
                  f"{curr.probs[curr.daily_leader]:.1%}")

            # How many days until model favourite matched?
            days_to_adapt = None
            for j in range(idx, len(snapshots)):
                if snapshots[j].model_favourite == curr.daily_leader:
                    days_to_adapt = j - idx
                    break
            if days_to_adapt is not None:
                print(f"    Model adapted after {days_to_adapt} day(s)")
            else:
                print(f"    Model NEVER adapted to new leader")
    else:
        print(f"\n  No intra-month lead changes detected")

    # Probability evolution for the eventual winner
    winner_probs = [snap.probs[actual_winner] for snap in snapshots]
    print(f"\n  P({actual_winner}) evolution:")
    print(f"    Start of month: {winner_probs[0]:.1%}")
    print(f"    Mid-month:      {winner_probs[len(winner_probs)//2]:.1%}")
    print(f"    End of month:   {winner_probs[-1]:.1%}")
    print(f"    Min:            {min(winner_probs):.1%} "
          f"(day {snapshots[winner_probs.index(min(winner_probs))].date})")
    print(f"    Max:            {max(winner_probs):.1%} "
          f"(day {snapshots[winner_probs.index(max(winner_probs))].date})")

    # ── Brier Score ──────────────────────────────────────────────────
    # Measures probability calibration: (predicted_prob - outcome)^2
    # For each day, outcome = 1 if actual_winner is the model's favourite
    # We use P(actual_winner) directly: Brier = mean((P(winner) - 1)^2)
    brier_scores = [(p - 1.0) ** 2 for p in winner_probs]
    brier_score = np.mean(brier_scores)

    # ── Max Overconfidence ───────────────────────────────────────────
    # Maximum probability ever assigned to an artist who turned out to
    # lose.  Lower = better.
    max_wrong_conf = 0.0
    for snap in snapshots:
        for artist_name, prob in snap.probs.items():
            if artist_name != actual_winner and prob > max_wrong_conf:
                max_wrong_conf = prob

    # ── Adaptation Speed ─────────────────────────────────────────────
    adapt_days = []
    for idx in leader_changes:
        curr = snapshots[idx]
        for j in range(idx, len(snapshots)):
            if snapshots[j].model_favourite == curr.daily_leader:
                adapt_days.append(j - idx)
                break
        else:
            adapt_days.append(len(snapshots) - idx)
    avg_adapt = np.mean(adapt_days) if adapt_days else 0.0

    print(f"\n  Brier score: {brier_score:.4f}")
    print(f"  Max wrong confidence: {max_wrong_conf:.1%}")

    return {
        "description": description,
        "actual_winner": actual_winner,
        "model_correct_pct": model_correct_count / n,
        "leader_bias_pct": leader_bias_count / n,
        "leader_changes": len(leader_changes),
        "winner_prob_start": winner_probs[0],
        "winner_prob_end": winner_probs[-1],
        "winner_prob_min": min(winner_probs),
        "winner_prob_max": max(winner_probs),
        "brier_score": brier_score,
        "max_wrong_confidence": max_wrong_conf,
        "days_to_adapt_avg": avg_adapt,
    }


def print_bias_verdict(all_results: List[Dict[str, Any]]) -> None:
    """Print the overall bias verdict with calibration-focused metrics."""
    print(f"\n{'═'*72}")
    print(f"  MODEL CALIBRATION VERDICT")
    print(f"{'═'*72}\n")

    avg_correct = np.mean([r["model_correct_pct"] for r in all_results])
    avg_brier = np.mean([r.get("brier_score", 0.25) for r in all_results])
    avg_overconf = np.mean([
        r.get("max_wrong_confidence", 0.5) for r in all_results
    ])

    for r in all_results:
        print(f"  {r['description'][:55]:55s}")
        print(f"    Correct winner:     {r['model_correct_pct']:.0%}  |  "
              f"Lead changes: {r['leader_changes']}")
        print(f"    P(winner) range:    {r['winner_prob_min']:.1%} → "
              f"{r['winner_prob_max']:.1%}")
        brier = r.get("brier_score", "N/A")
        overconf = r.get("max_wrong_confidence", "N/A")
        adapt = r.get("days_to_adapt_avg", "N/A")
        if isinstance(brier, float):
            print(f"    Brier score:        {brier:.4f}  "
                  f"(lower = better calibrated, perfect = 0)")
        if isinstance(overconf, float):
            print(f"    Max wrong conf:     {overconf:.1%}  "
                  f"(max P given to eventual loser)")
        if isinstance(adapt, (int, float)):
            print(f"    Avg adapt speed:    {adapt:.1f} day(s) after lead change")
        print()

    print(f"  Average Brier score:        {avg_brier:.4f}")
    print(f"  Average correct-winner:     {avg_correct:.0%}")
    print(f"  Average max overconfidence:  {avg_overconf:.1%}")
    print()

    if avg_brier < 0.15:
        print("  VERDICT: **WELL CALIBRATED** — model assigns appropriate uncertainty")
        print("           to competitive races and adjusts quickly to lead changes.")
    elif avg_brier < 0.22:
        print("  VERDICT: **REASONABLY CALIBRATED** — model shows meaningful")
        print("           differentiation but could improve on competitive races.")
    elif avg_brier < 0.30:
        print("  VERDICT: **MODERATELY CALIBRATED** — model assigns some uncertainty")
        print("           but is still overconfident in some scenarios.")
    else:
        print("  VERDICT: **POORLY CALIBRATED** — model needs further tuning")
        print("           to properly price uncertainty in competitive races.")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading artist data from cache...")
    artists = load_all_artists()
    print(f"Loaded {len(artists)} artists")

    all_results = []

    for year_month, description in TARGET_MONTHS:
        snapshots = run_month_backtest(artists, year_month, description)
        result = analyze_month(snapshots, description)
        all_results.append(result)

    print_bias_verdict(all_results)


if __name__ == "__main__":
    main()
