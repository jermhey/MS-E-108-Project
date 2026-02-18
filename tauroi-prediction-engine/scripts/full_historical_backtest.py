#!/usr/bin/env python3
"""
full_historical_backtest.py — Multi-Month WTA Calibration Backtest
====================================================================
Replays the MC-OU WTA model across ALL historical Kalshi market months
(June 2025 → January 2026) with strict no-look-ahead discipline.

For each day d in each target month:
  1. Calibrate per-artist params from the PRECEDING 90 days only.
  2. Compute 7-day EWMA trend from data ending on day d.
  3. Run MonteCarloOU.simulate_wta() with all 5 artists.
  4. Record win probability for each artist.

Outputs:
  - Per-month summary (winner, model pick, days to correct call)
  - Confidence calibration table (reliability diagram data)
  - Overall Brier score, accuracy, and speed metrics
"""
from __future__ import annotations

import sys
import pathlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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

# Every completed month in the Kalshi KXTOPMONTHLY range
TARGET_MONTHS = [
    "2025-06", "2025-07", "2025-08", "2025-09",
    "2025-10", "2025-11", "2025-12", "2026-01",
]

CALIBRATION_LOOKBACK = 90
MC_PATHS = 10_000
MC_SEED_BASE = 42


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data() -> Tuple[
    Dict[str, pd.DataFrame],    # full artist DataFrames
    pd.DataFrame,               # combined listeners (Date-indexed)
    Dict[str, pd.Series],       # tiktok per artist
]:
    all_dfs = {}
    combined = pd.DataFrame()
    tiktok_map = {}

    for name, path in CACHE_FILES.items():
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df.asfreq("D").ffill()
        all_dfs[name] = df

        combined[name] = df["spotify_monthly_listeners"].astype(float)

        tiktok_col = "tiktok_sound_posts_change"
        if tiktok_col in df.columns:
            tiktok_map[name] = df[tiktok_col].fillna(0).astype(float)
        else:
            tiktok_map[name] = pd.Series(0.0, index=df.index)

    combined = combined.sort_index()
    return all_dfs, combined, tiktok_map


# ═════════════════════════════════════════════════════════════════════════════
#  PER-ARTIST CALIBRATION (no look-ahead)
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_artist(
    artist_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = CALIBRATION_LOOKBACK,
) -> CalibrationResult | None:
    start = as_of - pd.Timedelta(days=lookback)
    window = artist_df.loc[start:as_of].copy()
    if len(window) < 30:
        return None

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


def compute_trend_from_series(listeners: pd.Series, as_of: pd.Timestamp) -> float:
    """Compute 7-day EWMA trend from data ending on as_of."""
    hist = listeners.loc[:as_of].dropna()
    if len(hist) < 8:
        return 0.0
    recent = hist.tail(8)
    log_rets = np.log(recent / recent.shift(1)).dropna()
    log_rets = log_rets[np.isfinite(log_rets)]
    if len(log_rets) < 2:
        return 0.0
    ewma_daily = float(log_rets.ewm(span=7).mean().iloc[-1])
    return float(np.clip(ewma_daily * 365, -2.0, 2.0))


# ═════════════════════════════════════════════════════════════════════════════
#  SINGLE-DAY WTA SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DayResult:
    date: str
    days_remaining: int
    probs: Dict[str, float]
    daily_leader: str
    daily_leader_listeners: float
    gap_pct: float
    model_pick: str
    model_pick_prob: float


def simulate_day(
    combined: pd.DataFrame,
    all_dfs: Dict[str, pd.DataFrame],
    tiktok_map: Dict[str, pd.Series],
    day: pd.Timestamp,
    month_end: pd.Timestamp,
    last_cal: Dict[str, CalibrationResult],
    recalibrate: bool,
    seed_offset: int,
) -> DayResult:
    names = sorted(CACHE_FILES.keys())
    days_remaining = (month_end - day).days
    T_years = max(days_remaining, 1) / 365.0

    # Recalibrate if needed
    if recalibrate:
        for name in names:
            cal = calibrate_artist(all_dfs[name], day)
            if cal is not None:
                last_cal[name] = cal

    # Get listeners for this day
    day_listeners = {}
    for name in names:
        val = combined.at[day, name] if day in combined.index else np.nan
        if np.isnan(val):
            before = combined[name].loc[:day].dropna()
            val = before.iloc[-1] if len(before) > 0 else 0
        day_listeners[name] = val

    # Daily leader
    sorted_a = sorted(day_listeners.items(), key=lambda x: x[1], reverse=True)
    daily_leader = sorted_a[0][0]
    daily_leader_listeners = sorted_a[0][1]
    second_listeners = sorted_a[1][1] if len(sorted_a) > 1 else 0
    gap = daily_leader_listeners - second_listeners
    gap_pct = gap / daily_leader_listeners * 100 if daily_leader_listeners > 0 else 0

    # Build OUArtistInput
    ou_inputs = []
    global_theta = 0.1
    global_jbeta = 0.0
    global_ji = 12.0
    global_js = 0.04

    for name in names:
        cal = last_cal.get(name)
        listeners = day_listeners[name]

        if cal is not None:
            sigma = cal.sigma
            norm_vel = 0.0
            tiktok_val = tiktok_map[name].get(day, 0) if name in tiktok_map else 0
            if cal.tiktok_p95 > 0 and tiktok_val > 0:
                norm_vel = min(tiktok_val / cal.tiktok_p95, 1.0)
            trend = cal.trend
            global_theta = cal.theta
            global_jbeta = cal.jump_beta
            global_ji = cal.jump_intensity
            global_js = cal.jump_std
        else:
            sigma = 0.20
            norm_vel = 0.0
            trend = 0.0

        # Fallback trend from raw listener data
        if trend == 0.0:
            trend = compute_trend_from_series(combined[name], day)

        ou_inputs.append(OUArtistInput(
            name=name,
            listeners=listeners,
            sigma=sigma,
            norm_velocity=norm_vel,
            trend=trend,
            event_impact_score=1.0,
        ))

    mc = MonteCarloOU(
        theta=global_theta,
        n_paths=MC_PATHS,
        seed=MC_SEED_BASE + seed_offset,
        jump_intensity=global_ji,
        jump_std=global_js,
        laplace_alpha=25.0,
    )
    results = mc.simulate_wta(ou_inputs, T=T_years, jump_beta=global_jbeta)

    probs = {r.name: r.probability for r in results}
    model_pick = max(probs, key=probs.get)

    return DayResult(
        date=day.strftime("%Y-%m-%d"),
        days_remaining=days_remaining,
        probs=probs,
        daily_leader=daily_leader,
        daily_leader_listeners=daily_leader_listeners,
        gap_pct=gap_pct,
        model_pick=model_pick,
        model_pick_prob=probs[model_pick],
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MONTH BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MonthResult:
    year_month: str
    actual_winner: str
    actual_winner_listeners: float
    second_place: str
    gap_pct: float
    lead_changes: int
    days: int
    day_results: List[DayResult]
    # Derived
    model_correct_days: int = 0
    first_correct_day: int = -1    # day index when model first picked winner
    brier_score: float = 0.0
    avg_winner_prob: float = 0.0
    final_winner_prob: float = 0.0
    max_loser_prob: float = 0.0    # max prob assigned to any non-winner


def run_month(
    year_month: str,
    combined: pd.DataFrame,
    all_dfs: Dict[str, pd.DataFrame],
    tiktok_map: Dict[str, pd.Series],
) -> MonthResult:
    ym = pd.Timestamp(f"{year_month}-01")
    month_end = ym + pd.offsets.MonthEnd(0)
    names = sorted(CACHE_FILES.keys())

    # Actual winner
    eom_row = combined.loc[:month_end].iloc[-1]
    actual_winner = eom_row.idxmax()
    actual_winner_listeners = eom_row[actual_winner]
    sorted_eom = eom_row.sort_values(ascending=False)
    second_place = sorted_eom.index[1]
    gap = sorted_eom.iloc[0] - sorted_eom.iloc[1]
    gap_pct = gap / sorted_eom.iloc[0] * 100 if sorted_eom.iloc[0] > 0 else 0

    # Count lead changes
    month_data = combined.loc[ym:month_end].dropna(how="all")
    daily_leaders = month_data.apply(lambda r: r.idxmax(), axis=1)
    lead_changes = sum(
        1 for i in range(1, len(daily_leaders))
        if daily_leaders.iloc[i] != daily_leaders.iloc[i - 1]
    )

    month_dates = month_data.index
    last_cal: Dict[str, CalibrationResult] = {}
    day_results: List[DayResult] = []

    for idx, day in enumerate(month_dates):
        recalibrate = (idx == 0 or idx % 7 == 0 or not last_cal)
        dr = simulate_day(
            combined, all_dfs, tiktok_map,
            day, month_end, last_cal,
            recalibrate=recalibrate,
            seed_offset=idx,
        )
        day_results.append(dr)

    # Compute derived metrics
    mr = MonthResult(
        year_month=year_month,
        actual_winner=actual_winner,
        actual_winner_listeners=actual_winner_listeners,
        second_place=second_place,
        gap_pct=gap_pct,
        lead_changes=lead_changes,
        days=len(day_results),
        day_results=day_results,
    )

    winner_probs = [dr.probs[actual_winner] for dr in day_results]
    mr.avg_winner_prob = float(np.mean(winner_probs))
    mr.final_winner_prob = winner_probs[-1]
    mr.brier_score = float(np.mean([(p - 1.0) ** 2 for p in winner_probs]))

    correct_days = [dr.model_pick == actual_winner for dr in day_results]
    mr.model_correct_days = sum(correct_days)

    # First correct day
    for i, c in enumerate(correct_days):
        if c:
            mr.first_correct_day = i
            break

    # Max loser prob
    for dr in day_results:
        for artist_name, prob in dr.probs.items():
            if artist_name != actual_winner and prob > mr.max_loser_prob:
                mr.max_loser_prob = prob

    return mr


# ═════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def print_month_detail(mr: MonthResult) -> None:
    """Print detailed day-by-day for a single month."""
    top_two = [mr.actual_winner, mr.second_place]
    short_names = {n: n.split()[-1][:8] for n in top_two}

    print(f"\n  {'Date':>10s}  {'D':>2s}  {'Leader':>15s}  {'Gap%':>5s}", end="")
    for n in top_two:
        print(f"  {'P('+short_names[n]+')':>10s}", end="")
    print(f"  {'Pick':>15s}  {'OK':>3s}")
    print("  " + "-" * 82)

    for dr in mr.day_results:
        correct = "Y" if dr.model_pick == mr.actual_winner else " "
        print(f"  {dr.date:>10s}  {dr.days_remaining:>2d}  "
              f"{dr.daily_leader:>15s}  {dr.gap_pct:>5.1f}", end="")
        for n in top_two:
            p = dr.probs.get(n, 0)
            print(f"  {p:>9.1%} ", end="")
        print(f"  {dr.model_pick:>15s}  {correct:>3s}")


def print_summary_table(all_months: List[MonthResult]) -> None:
    """Print the month-by-month summary."""
    print(f"\n{'═'*100}")
    print(f"  MONTH-BY-MONTH PERFORMANCE (No Time Leakage)")
    print(f"{'═'*100}\n")

    print(f"  {'Month':<8s}  {'Winner':<16s}  {'#2':<16s}  "
          f"{'Gap%':>5s}  {'Chg':>3s}  "
          f"{'Correct':>7s}  {'1st OK':>6s}  "
          f"{'AvgP(W)':>8s}  {'FinalP':>7s}  "
          f"{'Brier':>6s}  {'MaxErr':>7s}")
    print("  " + "-" * 96)

    for mr in all_months:
        first_ok = f"d{mr.first_correct_day}" if mr.first_correct_day >= 0 else "never"
        print(
            f"  {mr.year_month:<8s}  {mr.actual_winner:<16s}  "
            f"{mr.second_place:<16s}  "
            f"{mr.gap_pct:>5.1f}  {mr.lead_changes:>3d}  "
            f"{mr.model_correct_days:>3d}/{mr.days:<3d}  {first_ok:>6s}  "
            f"{mr.avg_winner_prob:>7.1%}  {mr.final_winner_prob:>6.1%}  "
            f"{mr.brier_score:>6.4f}  {mr.max_loser_prob:>6.1%}"
        )

    # Totals
    total_days = sum(mr.days for mr in all_months)
    total_correct = sum(mr.model_correct_days for mr in all_months)
    avg_brier = np.mean([mr.brier_score for mr in all_months])
    avg_winner_p = np.mean([mr.avg_winner_prob for mr in all_months])
    final_correct = sum(
        1 for mr in all_months
        if mr.day_results[-1].model_pick == mr.actual_winner
    )

    print("  " + "-" * 96)
    print(f"  {'TOTAL':<8s}  {'':<16s}  {'':<16s}  "
          f"{'':>5s}  {'':>3s}  "
          f"{total_correct:>3d}/{total_days:<3d}  {'':>6s}  "
          f"{avg_winner_p:>7.1%}  {'':>7s}  "
          f"{avg_brier:>6.4f}  {'':>7s}")
    print(f"\n  Final-day correct pick: {final_correct}/{len(all_months)} months "
          f"({final_correct/len(all_months):.0%})")


def print_confidence_calibration(all_months: List[MonthResult]) -> None:
    """
    Bin all daily predictions by confidence level and measure actual win rate.
    This is the core calibration / reliability table.
    """
    print(f"\n{'═'*100}")
    print(f"  CONFIDENCE CALIBRATION TABLE")
    print(f"  (When model says P=X%, how often does that artist actually win?)")
    print(f"{'═'*100}\n")

    # Collect all (predicted_prob, actual_outcome) pairs
    # For each day, for each artist: prob, and whether they won the month
    records = []
    for mr in all_months:
        for dr in mr.day_results:
            for artist_name, prob in dr.probs.items():
                won = 1 if artist_name == mr.actual_winner else 0
                records.append({
                    "month": mr.year_month,
                    "date": dr.date,
                    "artist": artist_name,
                    "predicted_prob": prob,
                    "actual_won": won,
                    "days_remaining": dr.days_remaining,
                })

    df = pd.DataFrame(records)

    # Bin by predicted probability
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]
    labels = [
        "0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%",
    ]
    df["bin"] = pd.cut(df["predicted_prob"], bins=bins, labels=labels, right=False)

    print(f"  {'Predicted':>10s}  {'Obs':>5s}  {'Actual':>7s}  {'Calibr':>7s}  {'Assess':>12s}")
    print("  " + "-" * 55)

    for label in labels:
        subset = df[df["bin"] == label]
        n = len(subset)
        if n == 0:
            continue
        actual_rate = subset["actual_won"].mean()
        # Midpoint of bin
        bin_idx = labels.index(label)
        bin_lo = bins[bin_idx]
        bin_hi = bins[bin_idx + 1]
        midpoint = (bin_lo + bin_hi) / 2

        # Assessment
        diff = actual_rate - midpoint
        if abs(diff) < 0.10:
            assess = "CALIBRATED"
        elif diff > 0:
            assess = "UNDER-CONF"
        else:
            assess = "OVER-CONF"

        print(f"  {label:>10s}  {n:>5d}  {actual_rate:>6.1%}  "
              f"{'  ' if abs(diff) < 0.10 else '⚠ '}{diff:>+5.1%}  "
              f"{assess:>12s}")

    # Overall Brier score (all artist-day pairs)
    overall_brier = float(
        np.mean((df["predicted_prob"] - df["actual_won"]) ** 2)
    )
    print(f"\n  Overall Brier score (all artist-days): {overall_brier:.4f}")
    print(f"  (Perfect = 0.0000, Naive baseline = 0.2000 for 5-artist field)")


def print_time_analysis(all_months: List[MonthResult]) -> None:
    """How early does the model correctly identify the winner?"""
    print(f"\n{'═'*100}")
    print(f"  SPEED ANALYSIS: How early does the model identify the winner?")
    print(f"{'═'*100}\n")

    # For each month, find days remaining when model first gives winner >50%
    print(f"  {'Month':<8s}  {'Winner':<16s}  "
          f"{'1st Pick':>8s}  {'1st >50%':>8s}  {'1st >60%':>8s}  "
          f"{'1st >70%':>8s}  {'1st >80%':>8s}  {'Final P':>8s}")
    print("  " + "-" * 85)

    speed_data = []
    for mr in all_months:
        winner_probs = [
            (dr.days_remaining, dr.probs[mr.actual_winner])
            for dr in mr.day_results
        ]

        first_pick = None
        thresholds = {50: None, 60: None, 70: None, 80: None}

        for days_rem, prob in winner_probs:
            if first_pick is None and max(
                mr.day_results[mr.days - 1 - days_rem].probs,
                key=mr.day_results[mr.days - 1 - days_rem].probs.get,
            ) == mr.actual_winner:
                # Find the first day where model pick == winner
                pass  # handled below

        # Simpler approach
        for i, dr in enumerate(mr.day_results):
            if first_pick is None and dr.model_pick == mr.actual_winner:
                first_pick = dr.days_remaining

            wp = dr.probs[mr.actual_winner]
            for t in [50, 60, 70, 80]:
                thresh = t / 100
                if thresholds[t] is None and wp >= thresh:
                    thresholds[t] = dr.days_remaining

        fp_str = f"{first_pick}d" if first_pick is not None else "never"
        t50 = f"{thresholds[50]}d" if thresholds[50] is not None else "never"
        t60 = f"{thresholds[60]}d" if thresholds[60] is not None else "never"
        t70 = f"{thresholds[70]}d" if thresholds[70] is not None else "never"
        t80 = f"{thresholds[80]}d" if thresholds[80] is not None else "never"

        print(
            f"  {mr.year_month:<8s}  {mr.actual_winner:<16s}  "
            f"{fp_str:>8s}  {t50:>8s}  {t60:>8s}  "
            f"{t70:>8s}  {t80:>8s}  {mr.final_winner_prob:>7.1%}"
        )

        speed_data.append({
            "first_pick_days": first_pick,
            "t50": thresholds[50],
            "t60": thresholds[60],
            "t70": thresholds[70],
            "t80": thresholds[80],
        })

    # Averages (exclude "never")
    pick_days = [s["first_pick_days"] for s in speed_data if s["first_pick_days"] is not None]
    t50_days = [s["t50"] for s in speed_data if s["t50"] is not None]
    t60_days = [s["t60"] for s in speed_data if s["t60"] is not None]

    print("  " + "-" * 85)
    if pick_days:
        print(f"  Average days remaining at first correct pick: {np.mean(pick_days):.1f}")
    if t50_days:
        print(f"  Average days remaining at first >50% for winner: {np.mean(t50_days):.1f}")
    if t60_days:
        print(f"  Average days remaining at first >60% for winner: {np.mean(t60_days):.1f}")


def print_race_type_breakdown(all_months: List[MonthResult]) -> None:
    """Performance split by race competitiveness."""
    print(f"\n{'═'*100}")
    print(f"  PERFORMANCE BY RACE TYPE")
    print(f"{'═'*100}\n")

    tight = [mr for mr in all_months if mr.gap_pct < 3.0]
    medium = [mr for mr in all_months if 3.0 <= mr.gap_pct < 6.0]
    runaway = [mr for mr in all_months if mr.gap_pct >= 6.0]

    for label, group in [
        ("TIGHT (<3% gap)", tight),
        ("MEDIUM (3-6% gap)", medium),
        ("RUNAWAY (>6% gap)", runaway),
    ]:
        if not group:
            print(f"  {label}: no months")
            continue

        months_str = ", ".join(mr.year_month for mr in group)
        avg_brier = np.mean([mr.brier_score for mr in group])
        avg_winner_p = np.mean([mr.avg_winner_prob for mr in group])
        final_correct = sum(
            1 for mr in group
            if mr.day_results[-1].model_pick == mr.actual_winner
        )
        total_correct = sum(mr.model_correct_days for mr in group)
        total_days = sum(mr.days for mr in group)

        print(f"  {label}")
        print(f"    Months: {months_str}")
        print(f"    Avg Brier:      {avg_brier:.4f}")
        print(f"    Avg P(winner):  {avg_winner_p:.1%}")
        print(f"    Final-day pick: {final_correct}/{len(group)} correct")
        print(f"    Day-by-day:     {total_correct}/{total_days} correct "
              f"({total_correct/total_days:.0%})")
        print()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading artist data from cache...")
    all_dfs, combined, tiktok_map = load_all_data()
    print(f"Loaded {len(all_dfs)} artists\n")

    all_months: List[MonthResult] = []

    for ym in TARGET_MONTHS:
        print(f"  Backtesting {ym}...", end="", flush=True)
        mr = run_month(ym, combined, all_dfs, tiktok_map)
        all_months.append(mr)
        ok = "✓" if mr.day_results[-1].model_pick == mr.actual_winner else "✗"
        print(f"  {ok} Winner: {mr.actual_winner} ({mr.gap_pct:.1f}% gap) "
              f"| Brier: {mr.brier_score:.4f} "
              f"| Correct {mr.model_correct_days}/{mr.days}d")

    # ── Summary table ────────────────────────────────────────────────
    print_summary_table(all_months)

    # ── Detailed per-month (only for months with lead changes) ───────
    interesting = [mr for mr in all_months if mr.lead_changes > 0]
    if interesting:
        print(f"\n{'═'*100}")
        print(f"  DETAILED DAY-BY-DAY (months with lead changes only)")
        print(f"{'═'*100}")
        for mr in interesting:
            print(f"\n  {mr.year_month}: {mr.actual_winner} wins "
                  f"(gap {mr.gap_pct:.1f}%, {mr.lead_changes} lead change(s))")
            print_month_detail(mr)

    # ── Confidence calibration ───────────────────────────────────────
    print_confidence_calibration(all_months)

    # ── Speed analysis ───────────────────────────────────────────────
    print_time_analysis(all_months)

    # ── Race type breakdown ──────────────────────────────────────────
    print_race_type_breakdown(all_months)


if __name__ == "__main__":
    main()
