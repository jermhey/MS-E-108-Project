#!/usr/bin/env python3
"""
run_backtest.py — Backtest Runner & Report Generator
=====================================================
Fetches historical data from the Chartmetric API, runs the historical
backtest, executes stress tests, and writes a timestamped plain-text report.

Usage
-----
    python run_backtest.py                         # full backtest (last 90 days)
    python run_backtest.py --days 180              # last 180 days
    python run_backtest.py --strike 95e6           # custom strike
    python run_backtest.py --capital 50000         # custom starting capital
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import pandas as pd

from src.config import load_settings
from src.chartmetric_client import ChartmetricClient
from src.backtester import (
    Backtester,
    compute_kpis,
    generate_report,
    run_stress_tests,
)
from src.utils import get_logger

logger = get_logger("tauroi.run_backtest")


# ── configuration ────────────────────────────────────────────────────────────

DEFAULT_STRIKE = 95_000_000
DEFAULT_DAYS = 90
DEFAULT_CAPITAL = 10_000


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tauroi Prediction Engine — Backtester",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Number of trailing days to backtest (default: {DEFAULT_DAYS}).",
    )
    parser.add_argument(
        "--strike", type=float, default=DEFAULT_STRIKE,
        help=f"Binary option strike (default: {DEFAULT_STRIKE:,.0f}).",
    )
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL,
        help=f"Initial capital (default: ${DEFAULT_CAPITAL:,.0f}).",
    )
    args = parser.parse_args(argv)

    # ── 1. Fetch data from Chartmetric API ─────────────────────────
    settings = load_settings(require_secrets=False)
    logger.info("=" * 64)
    logger.info("STEP 1 — Fetching Historical Data (Chartmetric API)")
    logger.info("=" * 64)

    client = ChartmetricClient(settings=settings)
    client.check_connection()
    df = client.get_artist_metrics(since="2019-01-01")

    # Sort and trim to requested window
    df = df.sort_values("Date").reset_index(drop=True)
    if args.days and args.days > 0:
        df = df.tail(args.days).reset_index(drop=True)
        logger.info(
            "Trimmed to last %d days (%s → %s)",
            len(df),
            str(df["Date"].iloc[0])[:10],
            str(df["Date"].iloc[-1])[:10],
        )

    # Drop rows where Spotify listeners are NaN (early history)
    df = df.dropna(subset=["spotify_monthly_listeners"]).reset_index(drop=True)
    logger.info("Backtest dataset: %d rows", len(df))

    if len(df) < 30:
        print(f"\n  ERROR: Only {len(df)} valid rows — need at least 30.\n")
        sys.exit(1)

    # ── 2. Main Backtest ─────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("STEP 2 — Main Backtest (strike=%.0f, capital=$%.0f)",
                args.strike, args.capital)
    logger.info("=" * 64)

    bt = Backtester(
        initial_capital=args.capital,
        strike=args.strike,
        edge_threshold=0.05,
        fee_pct=0.01,
        min_calibration_days=21,
        recalibration_interval=7,
    )
    result = bt.run(df, label="Main Backtest")
    kpis = compute_kpis(result)

    # ── 3. Stress Tests ──────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("STEP 3 — Stress Tests")
    logger.info("=" * 64)

    stress = run_stress_tests(
        df, strike=args.strike, initial_capital=args.capital,
    )

    # ── 4. Report ────────────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("STEP 4 — Generating Report")
    logger.info("=" * 64)

    report_text = generate_report(result, kpis, stress)

    # Write to file
    today = datetime.date.today().isoformat()
    out_dir = pathlib.Path("backtest_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"backtest_results_{today}.txt"
    out_path.write_text(report_text)

    # Also print to stdout
    print()
    print(report_text)
    print()
    print(f"  Report saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
