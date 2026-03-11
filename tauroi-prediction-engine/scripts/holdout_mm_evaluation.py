#!/usr/bin/env python3
"""
Canonical holdout profitability evaluation for the AS-informed market maker.

This script is the presentation-safe backtest entrypoint for the repo:
- uses the canonical EM/Kalman detector from `src.as_detector`
- uses the canonical fee-aware MM engine from `src.mm_backtest`
- removes same-trade lookahead via one-trade signal lag
- tunes the AS threshold out-of-sample (train -> validation -> test)
- reports uncertainty with event-level bootstrap intervals

Important:
- This is still a simplified trade-print fill model, not a production order-book simulator.
- Use these outputs for cautious profitability claims, not realized-live PnL claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(BASE))

from src.as_detector import ASResult, run_all_tickers
from src.mm_backtest import compare_all_tickers, compare_strategies, sweep_thresholds


CACHE_DIRS = {
    "spotify": BASE / "cache" / "kalshi_hf",
    "weather": BASE / "cache" / "kalshi_hf_weather",
    "election": BASE / "cache" / "kalshi_hf_election",
    "mlb": BASE / "cache" / "kalshi_hf_mlb",
}

BACKTEST_KW = {
    "half_spread_cents": 1.0,
    "mtm_horizon_minutes": 5.0,
    "mtm_horizon_trades": None,
    "decision_lag_trades": 1,
    "include_fees": True,
}

DETECTOR_KW = {
    "em_window": 200,
    "em_iterations": 8,
}

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8]  # coarser grid for faster tune; expand to [0.3..0.9] for full sweep
MIN_TRADES = 100


def ensure_volume(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        df = df.copy()
        df["volume"] = df.get("count", df.get("quantity", 1))
    return df


def load_dataset(dataset: str, min_trades: int = MIN_TRADES) -> dict[str, pd.DataFrame]:
    cache_dir = CACHE_DIRS[dataset]
    data: dict[str, pd.DataFrame] = {}
    for f in sorted(cache_dir.glob("*_trades.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        ts_col = "timestamp" if "timestamp" in df.columns else "created_time"
        if ts_col != "timestamp":
            df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = ensure_volume(df)
        if "mid_price" not in df.columns:
            df["mid_price"] = df.get("yes_price", df.get("price", 50)) / 100.0
        p = df["mid_price"].clip(0.01, 0.99)
        df["mid_price"] = p
        if "logit" not in df.columns:
            df["logit"] = np.log(p / (1.0 - p))
        if len(df) >= min_trades:
            data[f.stem.replace("_trades", "")] = df
    return data


def event_key_for_ticker(dataset: str, ticker: str) -> str:
    parts = ticker.split("-")
    if dataset in {"spotify", "weather", "election"}:
        return "-".join(parts[:2])
    if dataset == "mlb":
        return ticker.rsplit("-", 1)[0]
    raise ValueError(f"Unknown dataset: {dataset}")


def split_counts(n_events: int) -> tuple[int, int, int]:
    if n_events < 3:
        raise ValueError("Need at least 3 events for train/validation/test split")
    train = max(1, int(np.floor(n_events * 0.6)))
    val = max(1, int(np.floor(n_events * 0.2)))
    test = n_events - train - val
    if test < 1:
        if train > val:
            train -= 1
        else:
            val -= 1
        test = 1
    if train < 1:
        train = 1
    return train, val, test


def chronological_event_split(
    dataset: str,
    data: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    rows = []
    for ticker, df in data.items():
        rows.append({
            "ticker": ticker,
            "event_key": event_key_for_ticker(dataset, ticker),
            "start_ts": pd.to_datetime(df["timestamp"]).min(),
        })
    meta = pd.DataFrame(rows)
    events = (
        meta.groupby("event_key", as_index=False)["start_ts"]
        .min()
        .sort_values(["start_ts", "event_key"])
        .reset_index(drop=True)
    )
    train_n, val_n, test_n = split_counts(len(events))
    train_keys = set(events.iloc[:train_n]["event_key"])
    val_keys = set(events.iloc[train_n:train_n + val_n]["event_key"])
    test_keys = set(events.iloc[train_n + val_n:]["event_key"])

    def subset(keys: set[str]) -> dict[str, pd.DataFrame]:
        return {
            ticker: df for ticker, df in data.items()
            if event_key_for_ticker(dataset, ticker) in keys
        }

    return subset(train_keys), subset(val_keys), subset(test_keys), events


def aggregate_threshold_sweep(
    as_results: Dict[str, ASResult],
    thresholds: Iterable[float],
    backtest_kw: dict | None = None,
) -> pd.DataFrame:
    if backtest_kw is None:
        backtest_kw = BACKTEST_KW
    rows = []
    for ticker, ar in as_results.items():
        sweep_df = sweep_thresholds(ar, thresholds=list(thresholds), **backtest_kw)
        sweep_df["ticker"] = ticker
        rows.append(sweep_df)
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)

    naive_row = full[full["strategy"] == "naive"].agg({
        "net_pnl_c": "sum",
        "n_fills": "sum",
        "n_pulled": "sum",
        "gross_spread_c": "sum",
        "mtm_pnl_c": "sum",
        "fees_c": "sum",
    })
    naive_total = float(naive_row["net_pnl_c"])

    informed = (
        full[full["strategy"] == "as_informed"]
        .groupby("threshold", as_index=False)
        .agg({
            "net_pnl_c": "sum",
            "n_fills": "sum",
            "n_pulled": "sum",
            "gross_spread_c": "sum",
            "mtm_pnl_c": "sum",
            "fees_c": "sum",
        })
        .rename(columns={
            "net_pnl_c": "informed_pnl_c",
            "n_fills": "informed_fills",
            "n_pulled": "pulled_trades",
            "gross_spread_c": "gross_spread_c",
            "mtm_pnl_c": "mtm_pnl_c",
            "fees_c": "fees_c",
        })
    )
    informed["naive_pnl_c"] = naive_total
    informed["improvement_c"] = informed["informed_pnl_c"] - naive_total
    return informed.sort_values("threshold").reset_index(drop=True)


def choose_threshold(
    train_as: Dict[str, ASResult],
    val_as: Dict[str, ASResult],
    backtest_kw: dict | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    train_eval = aggregate_threshold_sweep(train_as, THRESHOLDS, backtest_kw=backtest_kw)
    top_thresholds = (
        train_eval.sort_values(["improvement_c", "informed_pnl_c"], ascending=False)
        .head(min(3, len(train_eval)))["threshold"]
        .tolist()
    )
    val_eval = aggregate_threshold_sweep(val_as, top_thresholds, backtest_kw=backtest_kw)
    best = val_eval.sort_values(["improvement_c", "informed_pnl_c"], ascending=False).iloc[0]
    return float(best["threshold"]), train_eval, val_eval


def comparison_rows(comparisons: Dict[str, object], dataset: str) -> pd.DataFrame:
    rows = []
    for ticker, comp in comparisons.items():
        rows.append({
            "dataset": dataset,
            "ticker": ticker,
            "event_key": event_key_for_ticker(dataset, ticker),
            "naive_pnl_c": comp.naive.net_pnl,
            "informed_pnl_c": comp.informed.net_pnl,
            "improvement_c": comp.pnl_improvement,
            "naive_fills": comp.naive.n_fills,
            "informed_fills": comp.informed.n_fills,
            "fills_avoided": comp.toxic_fills_avoided,
            "naive_fees_c": comp.naive.total_fees,
            "informed_fees_c": comp.informed.total_fees,
        })
    return pd.DataFrame(rows)


def bootstrap_event_ci(
    ticker_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    event_df = (
        ticker_df.groupby("event_key", as_index=False)["improvement_c"]
        .sum()
    )
    values = event_df["improvement_c"].to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        draws[b] = values[idx].sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    beat_rate = float(np.mean(draws > 0))
    return float(lo), float(hi), beat_rate


def compute_future_price_minutes(
    timestamps: np.ndarray,
    prices: np.ndarray,
    horizon_minutes: float,
) -> np.ndarray:
    ts_ns = pd.to_datetime(timestamps).astype("int64")
    targets = ts_ns + int(round(horizon_minutes * 60.0 * 1_000_000_000))
    future_idx = np.searchsorted(ts_ns, targets, side="left")
    out = np.full(len(timestamps), np.nan)
    for i, idx in enumerate(future_idx):
        if idx < len(prices):
            out[i] = float(prices[idx])
    return out


def toxicity_summary(
    as_results: Dict[str, ASResult],
    threshold: float,
) -> tuple[float, float, float]:
    flagged_signed = []
    unflagged_signed = []
    half_spread = BACKTEST_KW["half_spread_cents"] / 100.0
    lag = BACKTEST_KW["decision_lag_trades"]
    horizon_minutes = float(BACKTEST_KW["mtm_horizon_minutes"])

    for ar in as_results.values():
        prices = ar.prices
        timestamps = ar.timestamps
        future_prices = compute_future_price_minutes(timestamps, prices, horizon_minutes)

        for i in range(1, len(prices)):
            price_change = prices[i] - prices[i - 1]
            if abs(price_change) < half_spread:
                continue
            signal_idx = max(0, i - lag)
            flagged = ar.as_score[signal_idx] > threshold
            future_price = future_prices[i]
            if np.isnan(future_price):
                continue
            reference_mid = prices[i - 1]
            if price_change > 0:
                mtm_c = (reference_mid - future_price) * 100
            else:
                mtm_c = (future_price - reference_mid) * 100
            signed_adverse_c = -mtm_c  # positive = adverse, negative = favorable
            if flagged:
                flagged_signed.append(signed_adverse_c)
            else:
                unflagged_signed.append(signed_adverse_c)

    flagged_mean = float(np.mean(flagged_signed)) if flagged_signed else np.nan
    unflagged_mean = float(np.mean(unflagged_signed)) if unflagged_signed else np.nan
    ratio = flagged_mean / unflagged_mean if unflagged_signed and abs(unflagged_mean) > 1e-8 else np.nan
    return flagged_mean, unflagged_mean, ratio


def evaluate_market(
    dataset: str,
    bootstrap_draws: int = 1000,
    include_fees: bool = True,
    alpha: float = 0.7,
) -> dict:
    data = load_dataset(dataset)
    train, val, test, events = chronological_event_split(dataset, data)
    backtest_kw = dict(BACKTEST_KW)
    backtest_kw["include_fees"] = include_fees
    detector_kw = {**DETECTOR_KW, "alpha": alpha}

    train_as = run_all_tickers(train, **detector_kw)
    val_as = run_all_tickers(val, **detector_kw)
    test_as = run_all_tickers(test, **detector_kw)

    chosen_tau, train_eval, val_eval = choose_threshold(train_as, val_as, backtest_kw=backtest_kw)
    test_comparisons = compare_all_tickers(test_as, as_threshold=chosen_tau, **backtest_kw)
    ticker_df = comparison_rows(test_comparisons, dataset)

    naive_pnl = float(ticker_df["naive_pnl_c"].sum())
    informed_pnl = float(ticker_df["informed_pnl_c"].sum())
    improvement = float(ticker_df["improvement_c"].sum())
    event_ci_lo, event_ci_hi, bootstrap_beat_rate = bootstrap_event_ci(
        ticker_df, n_boot=bootstrap_draws
    )
    flagged_adv, unflagged_adv, adv_ratio = toxicity_summary(test_as, chosen_tau)

    return {
        "dataset": dataset,
        "n_tickers": len(data),
        "n_events": len(events),
        "train_events": len({event_key_for_ticker(dataset, t) for t in train}),
        "val_events": len({event_key_for_ticker(dataset, t) for t in val}),
        "test_events": len({event_key_for_ticker(dataset, t) for t in test}),
        "train_tickers": len(train),
        "val_tickers": len(val),
        "test_tickers": len(test),
        "chosen_threshold": chosen_tau,
        "naive_test_pnl_c": naive_pnl,
        "informed_test_pnl_c": informed_pnl,
        "improvement_test_c": improvement,
        "ci95_lo_c": event_ci_lo,
        "ci95_hi_c": event_ci_hi,
        "bootstrap_beat_rate": bootstrap_beat_rate,
        "ticker_beat_rate": float((ticker_df["improvement_c"] > 0).mean()) if len(ticker_df) else np.nan,
        "flagged_signed_adverse_c": flagged_adv,
        "unflagged_signed_adverse_c": unflagged_adv,
        "adverse_ratio": adv_ratio,
        "test_total_fills_naive": int(ticker_df["naive_fills"].sum()),
        "test_total_fills_informed": int(ticker_df["informed_fills"].sum()),
        "test_fills_avoided": int(ticker_df["fills_avoided"].sum()),
        "test_fees_naive_c": float(ticker_df["naive_fees_c"].sum()),
        "test_fees_informed_c": float(ticker_df["informed_fees_c"].sum()),
        "include_fees": include_fees,
        "alpha": alpha,
        "train_eval": train_eval,
        "val_eval": val_eval,
        "ticker_df": ticker_df,
    }


def tune_alpha_global(
    markets: list[str],
    include_fees: bool = True,
) -> tuple[float, dict]:
    """
    Tune alpha globally across markets: for each alpha, run detector with that
    alpha, tune tau per market, sum validation improvement across markets.
    Return best alpha and per-market results for that alpha.
    """
    best_alpha = 0.7
    best_total_val_improvement = -np.inf
    per_alpha_val: dict[float, float] = {}

    for alpha in ALPHA_GRID:
        total_val = 0.0
        for dataset in markets:
            data = load_dataset(dataset)
            train, val, test, events = chronological_event_split(dataset, data)
            if not train or not val:
                continue
            backtest_kw = dict(BACKTEST_KW)
            backtest_kw["include_fees"] = include_fees
            detector_kw = {**DETECTOR_KW, "alpha": alpha}
            train_as = run_all_tickers(train, **detector_kw)
            val_as = run_all_tickers(val, **detector_kw)
            tau, _, val_eval = choose_threshold(train_as, val_as, backtest_kw=backtest_kw)
            best_row = val_eval.sort_values(["improvement_c", "informed_pnl_c"], ascending=False).iloc[0]
            total_val += float(best_row["improvement_c"])
        per_alpha_val[alpha] = total_val
        if total_val > best_total_val_improvement:
            best_total_val_improvement = total_val
            best_alpha = alpha

    return best_alpha, per_alpha_val


def print_market_summary(result: dict) -> None:
    print(f"\n=== {result['dataset'].upper()} ===")
    print(
        f"Events train/val/test = {result['train_events']}/{result['val_events']}/{result['test_events']} | "
        f"Tickers train/val/test = {result['train_tickers']}/{result['val_tickers']}/{result['test_tickers']}"
    )
    print(f"Chosen AS threshold from holdout tuning: tau = {result['chosen_threshold']:.1f}")
    print(
        f"Test PnL: naive={result['naive_test_pnl_c']:,.1f}c | "
        f"informed={result['informed_test_pnl_c']:,.1f}c | "
        f"delta={result['improvement_test_c']:+,.1f}c"
    )
    print(f"Fees included: {result['include_fees']}")
    print(
        f"95% event-bootstrap CI for delta: "
        f"[{result['ci95_lo_c']:,.1f}c, {result['ci95_hi_c']:,.1f}c] | "
        f"P(delta > 0)≈{result['bootstrap_beat_rate']:.3f}"
    )
    print(
        f"Ticker beat rate={result['ticker_beat_rate']:.3f} | "
        f"fills avoided={result['test_fills_avoided']:,}"
    )
    print(
        f"Signed adverse drift (5m): flagged={result['flagged_signed_adverse_c']:.3f}c | "
        f"unflagged={result['unflagged_signed_adverse_c']:.3f}c | "
        f"ratio={result['adverse_ratio']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical holdout MM profitability evaluation")
    parser.add_argument(
        "--market",
        choices=["spotify", "weather", "election", "mlb", "all"],
        default="all",
        help="Market family to evaluate. Default: all.",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=500,
        help="Number of event-level bootstrap draws for uncertainty intervals.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the cross-market summary table as CSV.",
    )
    parser.add_argument(
        "--no-fees",
        action="store_true",
        help="Run the holdout backtest assuming zero maker fees.",
    )
    parser.add_argument(
        "--tune-alpha",
        action="store_true",
        help="Tune alpha globally on validation, then compare tuned vs fixed alpha=0.7 on test.",
    )
    args = parser.parse_args()

    markets = ["spotify", "weather", "election", "mlb"] if args.market == "all" else [args.market]
    include_fees = not args.no_fees

    if args.tune_alpha:
        print("Tuning alpha globally on validation...")
        best_alpha, per_alpha_val = tune_alpha_global(markets, include_fees=include_fees)
        print(f"Alpha grid validation improvement: {per_alpha_val}")
        print(f"Best alpha = {best_alpha}")
        print()
        results_tuned = [
            evaluate_market(m, bootstrap_draws=args.bootstrap_draws, include_fees=include_fees, alpha=best_alpha)
            for m in markets
        ]
        results_baseline = [
            evaluate_market(m, bootstrap_draws=args.bootstrap_draws, include_fees=include_fees, alpha=0.7)
            for m in markets
        ]
        print("=" * 80)
        print("COMPARISON: Tuned alpha vs Fixed alpha=0.7 (test improvement, cents)")
        print("=" * 80)
        for i, m in enumerate(markets):
            rt, rb = results_tuned[i], results_baseline[i]
            print(
                f"{m:12}  tuned(α={best_alpha}): {rt['improvement_test_c']:>8.2f}c  "
                f"baseline(α=0.7): {rb['improvement_test_c']:>8.2f}c  "
                f"delta: {rt['improvement_test_c'] - rb['improvement_test_c']:>+.2f}c"
            )
        total_tuned = sum(r["improvement_test_c"] for r in results_tuned)
        total_baseline = sum(r["improvement_test_c"] for r in results_baseline)
        print(f"\nTotal test improvement: tuned={total_tuned:.2f}c  baseline={total_baseline:.2f}c  delta={total_tuned - total_baseline:+.2f}c")
        results = results_tuned
    else:
        results = [
            evaluate_market(m, bootstrap_draws=args.bootstrap_draws, include_fees=include_fees)
            for m in markets
        ]

    for res in results:
        print_market_summary(res)

    summary = pd.DataFrame([{
        "market": r["dataset"],
        "test_tickers": r["test_tickers"],
        "test_events": r["test_events"],
        "chosen_tau": r["chosen_threshold"],
        "naive_test_pnl_c": r["naive_test_pnl_c"],
        "informed_test_pnl_c": r["informed_test_pnl_c"],
        "improvement_test_c": r["improvement_test_c"],
        "ci95_lo_c": r["ci95_lo_c"],
        "ci95_hi_c": r["ci95_hi_c"],
        "bootstrap_beat_rate": r["bootstrap_beat_rate"],
        "ticker_beat_rate": r["ticker_beat_rate"],
        "include_fees": r["include_fees"],
        "flagged_signed_adverse_c": r["flagged_signed_adverse_c"],
        "unflagged_signed_adverse_c": r["unflagged_signed_adverse_c"],
        "adverse_ratio": r["adverse_ratio"],
        "fills_avoided": r["test_fills_avoided"],
    } for r in results])

    print("\n" + "=" * 118)
    fee_label = "fees included" if not args.no_fees else "no fees"
    print(f"CANONICAL HOLDOUT COMPARISON (lagged signal, 1c half-spread, 5m horizon, {fee_label})")
    print("=" * 118)
    print(summary.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(f"\nSaved summary CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
