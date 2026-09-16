#!/usr/bin/env python3
"""
Heuristic AS Parameter Sweep & Slide Stats.

PART 1: Parameter sweep over naive AS rules (price jump + volume multiplier).
        Simulates a simple Market Maker; finds (jump_cents, vol_mult) that maximize
        a simplified 5-minute PnL proxy.

PART 2: Using optimal params, computes 5-min post-trade adverse drift for
        flagged vs unflagged trades, ratio, t-test, and a presentation bar chart.

Designed as a sensitivity study for slide work. This is NOT the canonical
profitability backtest; use `scripts/holdout_mm_evaluation.py` for the
lagged-signal, fee-aware holdout evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Optional: matplotlib for chart
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Market caches: thick (weather, MLB) vs thin (Spotify/KXTOPMONTHLY, election)
CACHE_DIRS = {
    "spotify": BASE / "cache" / "kalshi_hf",           # KXTOPMONTHLY — thin
    "weather": BASE / "cache" / "kalshi_hf_weather",   # thick
    "election": BASE / "cache" / "kalshi_hf_election", # thin
    "mlb": BASE / "cache" / "kalshi_hf_mlb",          # baseball — thick
}

SPREAD_CENTS = 1.0
POST_TRADE_MINUTES = 5
ROLLING_VOL_WINDOW = 50  # baseline = mean volume over previous 50 TRADES (not time)
MIN_TRADES = 100  # skip tickers with too few trades

# This script uses a NAIVE rule only (no Kalman, no EM). It is a heuristic
# sensitivity study, not the canonical profitability backtest.


def ensure_volume(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        df = df.copy()
        df["volume"] = df.get("count", df.get("quantity", 1))
    if df["volume"].dtype == float and (df["volume"] < 0.5).any():
        df = df.copy()
        df["volume"] = df["volume"].clip(lower=0.1)
    return df


def load_dataset(dataset: str, min_trades: int = MIN_TRADES) -> dict[str, pd.DataFrame]:
    cache_dir = CACHE_DIRS.get(dataset)
    if not cache_dir or not cache_dir.exists():
        raise FileNotFoundError(f"Cache not found: {cache_dir}")
    data = {}
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
        df["mid_price"] = df["mid_price"].clip(0.01, 0.99)
        if len(df) >= min_trades:
            tkr = f.stem.replace("_trades", "")
            data[tkr] = df
    return data


def naive_as_flag(
    prices: np.ndarray,
    volumes: np.ndarray,
    jump_threshold_cents: float,
    vol_multiplier: float,
    rolling_window: int = ROLLING_VOL_WINDOW,
) -> np.ndarray:
    """
    Naive AS rule: flag trade i if
      |price[i] - price[i-1]| > jump_threshold_cents  AND
      volume[i] > vol_multiplier * rolling_avg_volume (before i).
    Returns boolean array of length n; index 0 is False (no prior price).
    """
    n = len(prices)
    flagged = np.zeros(n, dtype=bool)
    if n < 2:
        return flagged
    price_cents = np.asarray(prices, dtype=float) * 100
    jump_cents = np.abs(np.diff(price_cents))
    vol = np.maximum(np.asarray(volumes, dtype=float), 0.1)
    # Rolling avg of volume *before* current trade: at i use mean(vol[start:i])
    roll_prev = np.full(n, np.nan)
    roll_prev[0] = vol[0]
    for i in range(1, n):
        start = max(0, i - rolling_window)
        roll_prev[i] = float(np.mean(vol[start:i]))
    roll_prev = np.maximum(roll_prev, 1e-6)
    for i in range(1, n):
        if jump_cents[i - 1] > jump_threshold_cents and vol[i] > vol_multiplier * roll_prev[i]:
            flagged[i] = True
    return flagged


def price_5min_later(
    timestamps: np.ndarray,
    prices: np.ndarray,
    trade_idx: int,
) -> float | None:
    """Return mid_price at first trade at or after trade_time + 5 minutes, or None if past end."""
    t0 = pd.Timestamp(timestamps[trade_idx])
    t_target = t0 + pd.Timedelta(minutes=POST_TRADE_MINUTES)
    for j in range(trade_idx + 1, len(timestamps)):
        if pd.Timestamp(timestamps[j]) >= t_target:
            return float(prices[j])
    return None


def compute_price_5min_later_array(timestamps: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Precompute price at t+5min for each index. NaN where not available. O(n) pass."""
    n = len(timestamps)
    out = np.full(n, np.nan)
    j = 0
    for i in range(n):
        t_target = pd.Timestamp(timestamps[i]) + pd.Timedelta(minutes=POST_TRADE_MINUTES)
        while j < n and (j <= i or pd.Timestamp(timestamps[j]) < t_target):
            j += 1
        if j < n:
            out[i] = float(prices[j])
    return out


def simulate_mm_one_ticker(
    df: pd.DataFrame,
    jump_threshold_cents: float,
    vol_multiplier: float,
    spread_cents: float = SPREAD_CENTS,
    price_5min_arr: np.ndarray | None = None,
) -> tuple[float, int, int, float, float, np.ndarray, np.ndarray]:
    """
    Simulate MM on one ticker with naive AS rule.
    Returns: (total_pnl_cents, n_fills, n_pulled, spreads_cents, toxic_loss_cents, flagged_array, adverse_moves).
    MM earns spread_cents per fill; loses (adverse 5-min move in cents) on toxic fills.
    """
    timestamps = df["timestamp"].values
    prices = df["mid_price"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    if len(volumes) != len(prices):
        volumes = np.full_like(prices, 1.0)
    n = len(prices)
    flagged = naive_as_flag(prices, volumes, jump_threshold_cents, vol_multiplier)
    if price_5min_arr is None:
        price_5min_arr = compute_price_5min_later_array(timestamps, prices)

    total_pnl = 0.0
    total_spread = 0.0
    total_toxic = 0.0  # sum of adverse moves only (for reporting)
    n_fills = 0
    adverse_moves = []
    half_spread = spread_cents / 100.0 / 2.0

    for i in range(1, n):
        price_change = prices[i] - prices[i - 1]
        if abs(price_change) < 1e-8:
            continue
        if flagged[i]:
            continue
        if price_change > 0:
            mm_side = "sell"
            fill_price = prices[i - 1] + half_spread
        else:
            mm_side = "buy"
            fill_price = prices[i - 1] - half_spread
        if abs(price_change) < half_spread:
            continue
        total_spread += spread_cents
        n_fills += 1
        p5 = price_5min_arr[i] if i < len(price_5min_arr) and not np.isnan(price_5min_arr[i]) else None
        if p5 is not None:
            # Mark-to-market: profit when price moves in our favor after the fill
            if mm_side == "sell":
                mtm_cents = (fill_price - p5) * 100   # we sold; profit if price drops
            else:
                mtm_cents = (p5 - fill_price) * 100   # we bought; profit if price rises
            total_pnl += spread_cents + mtm_cents
            adverse_cents = max(0.0, -mtm_cents)  # adverse = loss (negative mtm)
            total_toxic += adverse_cents
            adverse_moves.append(adverse_cents)
        else:
            total_pnl += spread_cents  # no 5-min data: just spread
            adverse_moves.append(np.nan)

    return (
        total_pnl,
        n_fills,
        int(flagged.sum()),
        total_spread,
        total_toxic,
        flagged,
        np.array(adverse_moves) if adverse_moves else np.array([]),
    )


def parameter_sweep(
    data: dict[str, pd.DataFrame],
    jump_cents_list: list[float],
    vol_mult_list: list[float],
) -> pd.DataFrame:
    """Run MM simulation for every (jump_cents, vol_mult) across all tickers; return results."""
    price_5min_cache: dict[str, np.ndarray] = {}
    for ticker, df in data.items():
        ts = df["timestamp"].values
        pr = df["mid_price"].values.astype(float)
        price_5min_cache[ticker] = compute_price_5min_later_array(ts, pr)

    rows = []
    for jump_cents in jump_cents_list:
        for vol_mult in vol_mult_list:
            total_pnl = 0.0
            total_fills = 0
            total_pulled = 0
            total_spread = 0.0
            total_toxic = 0.0
            for ticker, df in data.items():
                pnl, n_fills, n_pulled, spread, toxic, _, _ = simulate_mm_one_ticker(
                    df, jump_cents, vol_mult, price_5min_arr=price_5min_cache[ticker]
                )
                total_pnl += pnl
                total_fills += n_fills
                total_pulled += n_pulled
                total_spread += spread
                total_toxic += toxic
            rows.append({
                "jump_cents": jump_cents,
                "vol_mult": vol_mult,
                "total_pnl_cents": total_pnl,
                "n_fills": total_fills,
                "n_pulled": total_pulled,
                "spreads_cents": total_spread,
                "toxic_loss_cents": total_toxic,
            })
    return pd.DataFrame(rows)


def run_naive_baseline(data: dict[str, pd.DataFrame]) -> dict:
    """Run MM that never flags (always quotes). Returns spreads, toxic loss, PnL, n_fills."""
    price_5min_cache: dict[str, np.ndarray] = {}
    for ticker, df in data.items():
        ts = df["timestamp"].values
        pr = df["mid_price"].values.astype(float)
        price_5min_cache[ticker] = compute_price_5min_later_array(ts, pr)
    total_pnl = 0.0
    total_spread = 0.0
    total_toxic = 0.0
    total_fills = 0
    for ticker, df in data.items():
        pnl, n_fills, _, spread, toxic, _, _ = simulate_mm_one_ticker(
            df, jump_threshold_cents=999.0, vol_multiplier=999.0,
            price_5min_arr=price_5min_cache[ticker],
        )
        total_pnl += pnl
        total_fills += n_fills
        total_spread += spread
        total_toxic += toxic
    return {
        "total_pnl_cents": total_pnl,
        "spreads_cents": total_spread,
        "toxic_loss_cents": total_toxic,
        "n_fills": total_fills,
    }


def compute_adverse_drift_all_trades(
    df: pd.DataFrame,
    flagged: np.ndarray,
    spread_cents: float = SPREAD_CENTS,
    price_5min_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every trade where price moved (potential fill), compute the 5-min adverse move
    that would have occurred (from MM's perspective). Return arrays for flagged vs unflagged.
    """
    timestamps = df["timestamp"].values
    prices = df["mid_price"].values.astype(float)
    n = len(prices)
    half_spread = spread_cents / 100.0 / 2.0
    if price_5min_arr is None:
        price_5min_arr = compute_price_5min_later_array(timestamps, prices)
    adverse_flagged = []
    adverse_unflagged = []
    for i in range(1, n):
        price_change = prices[i] - prices[i - 1]
        if abs(price_change) < 1e-8:
            continue
        if price_change > 0:
            mm_side = "sell"
            fill_price = prices[i - 1] + half_spread
        else:
            mm_side = "buy"
            fill_price = prices[i - 1] - half_spread
        if abs(price_change) < half_spread:
            continue
        p5 = price_5min_arr[i] if i < len(price_5min_arr) and not np.isnan(price_5min_arr[i]) else None
        if p5 is None:
            continue
        if mm_side == "sell":
            adverse = (p5 - fill_price) * 100
        else:
            adverse = (fill_price - p5) * 100
        adverse = max(0.0, adverse)
        if flagged[i]:
            adverse_flagged.append(adverse)
        else:
            adverse_unflagged.append(adverse)
    return (
        np.array(adverse_flagged) if adverse_flagged else np.array([], dtype=float),
        np.array(adverse_unflagged) if adverse_unflagged else np.array([], dtype=float),
        flagged,
    )


def run_slide_stats(
    data: dict[str, pd.DataFrame],
    jump_cents: float,
    vol_mult: float,
) -> dict:
    """Compute slide stats and optionally plot. Returns dict of stats."""
    all_adv_flagged = []
    all_adv_unflagged = []
    for ticker, df in data.items():
        timestamps = df["timestamp"].values
        prices = df["mid_price"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        if len(volumes) != len(prices):
            volumes = np.full_like(prices, 1.0)
        flagged = naive_as_flag(prices, volumes, jump_cents, vol_mult)
        price_5min_arr = compute_price_5min_later_array(timestamps, prices)
        adv_f, adv_u, _ = compute_adverse_drift_all_trades(
            df, flagged, price_5min_arr=price_5min_arr
        )
        all_adv_flagged.extend(adv_f.tolist())
        all_adv_unflagged.extend(adv_u.tolist())
    adv_flagged = np.array(all_adv_flagged)
    adv_unflagged = np.array(all_adv_unflagged)
    mean_flagged = float(np.mean(adv_flagged)) if len(adv_flagged) > 0 else 0.0
    mean_unflagged = float(np.mean(adv_unflagged)) if len(adv_unflagged) > 0 else 0.0
    ratio = mean_flagged / mean_unflagged if mean_unflagged > 1e-8 else 0.0
    from scipy import stats as sp_stats
    if len(adv_flagged) >= 5 and len(adv_unflagged) >= 5:
        t_stat, p_value = sp_stats.ttest_ind(adv_flagged, adv_unflagged, alternative="greater")
        p_value = float(p_value)
    else:
        t_stat, p_value = np.nan, np.nan
    return {
        "mean_adverse_flagged_cents": mean_flagged,
        "mean_adverse_unflagged_cents": mean_unflagged,
        "ratio": ratio,
        "n_flagged": len(adv_flagged),
        "n_unflagged": len(adv_unflagged),
        "t_stat": t_stat,
        "p_value": p_value,
        "adv_flagged": adv_flagged,
        "adv_unflagged": adv_unflagged,
    }


def plot_slide_chart(stats: dict, output_path: Path) -> None:
    """Bar chart: Average Adverse Move (cents) — Unflagged vs Flagged. Dark theme."""
    if not HAS_MPL:
        print("matplotlib not available; skipping chart.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    labels = ["Unflagged Trades", "Flagged Trades"]
    means = [stats["mean_adverse_unflagged_cents"], stats["mean_adverse_flagged_cents"]]
    colors = ["#666666", "#39FF14"]  # gray, neon green
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Average Adverse Move (cents)", color="#e0e0e0", fontsize=11)
    ax.tick_params(colors="#a0a0a0")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.grid(True, alpha=0.2, color="#444")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{m:.2f}¢",
                ha="center", va="bottom", color="#e0e0e0", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a", edgecolor="none")
    plt.close()
    print(f"Saved chart to {output_path}")


def _run_all_markets() -> None:
    """Run sweep on spotify, weather, election, mlb and print comparison (thick vs thin)."""
    markets = ["spotify", "weather", "election", "mlb"]
    results = []
    for name in markets:
        print(f"Running {name}...", end=" ", flush=True)
        rec = run_one_market(name)
        if rec is None:
            print("no data / cache missing")
            continue
        results.append(rec)
        print(f"{rec['n_tickers']} tickers, {rec['n_trades']:,} trades, "
              f"naive={rec['naive_pnl_cents']:,.0f}¢ optimal={rec['optimal_pnl_cents']:,.0f}¢ "
              f"(+{rec['improvement_cents']:,.0f}¢)")

    if not results:
        print("No market data found.")
        return

    print("\n" + "=" * 90)
    print("CROSS-MARKET COMPARISON: Naive MM (always quote) vs AS-informed MM (optimal 2–5¢ / 2–5×)")
    print("=" * 90)
    print(f"{'Market':<12} {'Tickers':>8} {'Trades':>10} {'Naive PnL (¢)':>14} {'Optimal PnL (¢)':>16} {'Opt params':>12} {'Improvement (¢)':>14}")
    print("-" * 90)
    for r in results:
        params = f"{r['opt_jump']:.0f}¢/{r['opt_vol_mult']:.0f}×"
        print(f"{r['dataset']:<12} {r['n_tickers']:>8} {r['n_trades']:>10,} {r['naive_pnl_cents']:>14,.0f} "
              f"{r['optimal_pnl_cents']:>16,.0f} {params:>12} {r['improvement_cents']:>+14,.0f}")
    print("-" * 90)
    print("Thick markets: weather, mlb. Thin markets: spotify (KXTOPMONTHLY), election.")
    print("=" * 90)


def run_one_market(dataset: str) -> dict | None:
    """Run sweep + baseline for one market. Returns summary dict or None if no data."""
    try:
        data = load_dataset(dataset)
    except FileNotFoundError:
        return None
    if not data:
        return None
    n_tickers = len(data)
    n_trades = sum(len(d) for d in data.values())
    naive = run_naive_baseline(data)
    sweep = parameter_sweep(data, [2, 3, 4, 5], [2.0, 3.0, 4.0, 5.0])
    best = sweep.loc[sweep["total_pnl_cents"].idxmax()]
    return {
        "dataset": dataset,
        "n_tickers": n_tickers,
        "n_trades": n_trades,
        "naive_pnl_cents": naive["total_pnl_cents"],
        "optimal_pnl_cents": best["total_pnl_cents"],
        "opt_jump": float(best["jump_cents"]),
        "opt_vol_mult": float(best["vol_mult"]),
        "improvement_cents": float(best["total_pnl_cents"] - naive["total_pnl_cents"]),
    }


def main():
    parser = argparse.ArgumentParser(description="AS parameter sweep and slide stats")
    parser.add_argument(
        "--dataset",
        choices=["spotify", "weather", "election", "mlb"],
        default="weather",
        help="Dataset: spotify (KXTOPMONTHLY/thin), weather (thick), election (thin), mlb (thick)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all four markets and print comparison table (naive vs optimal PnL)",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=BASE / "figures" / "as_slide_chart.png",
        help="Output path for bar chart",
    )
    parser.add_argument("--no-chart", action="store_true", help="Skip generating bar chart")
    args = parser.parse_args()

    if args.all:
        _run_all_markets()
        return

    print("Loading dataset:", args.dataset)
    data = load_dataset(args.dataset)
    if not data:
        print("No data loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(data)} tickers, {sum(len(d) for d in data.values()):,} total trades\n")

    # ---- PART 1: Parameter sweep ----
    jump_list = [2, 3, 4, 5]
    vol_list = [2.0, 3.0, 4.0, 5.0]
    print("PART 1: Parameter sweep (Market Maker simulation)")
    print("  Price jump threshold (cents):", jump_list)
    print("  Volume multiplier (x rolling avg):", vol_list)
    print()
    sweep = parameter_sweep(data, jump_list, vol_list)
    best_row = sweep.loc[sweep["total_pnl_cents"].idxmax()]
    opt_jump = float(best_row["jump_cents"])
    opt_vol = float(best_row["vol_mult"])

    # Naive baseline: never flag (always quote)
    naive = run_naive_baseline(data)
    print("--- BASELINE: Never flag (always quote) ---")
    print(f"  Spreads captured:  {naive['spreads_cents']:,.0f} cents  ({naive['n_fills']:,} fills × 1¢)")
    print(f"  Toxic losses:       {naive['toxic_loss_cents']:,.0f} cents  (5-min adverse move on fills)")
    print(f"  Total PnL:          {naive['total_pnl_cents']:,.0f} cents  (= spreads - toxic)")
    print()

    print("--- OPTIMAL parameters (max Total PnL) ---")
    print(f"  Price jump threshold: {opt_jump} cents")
    print(f"  Volume multiplier:    {opt_vol}x")
    print(f"  Spreads captured:     {best_row['spreads_cents']:,.0f} cents  ({int(best_row['n_fills']):,} fills)")
    print(f"  Toxic losses:        {best_row['toxic_loss_cents']:,.0f} cents")
    print(f"  Total PnL:           {best_row['total_pnl_cents']:,.0f} cents")
    print(f"  Trades we sat out:   {int(best_row['n_pulled']):,} (flagged — no fill, no loss)")
    print()

    improvement = best_row["total_pnl_cents"] - naive["total_pnl_cents"]
    print("--- INTERPRETATION ---")
    print("  Total PnL = Spreads + Mark-to-market (5 min later). On each fill we earn 1¢ spread;")
    print("  then we add (fill_price vs price_5min): profit when price moves in our favor, loss when adverse.")
    print("  When we FLAG we do not quote → no fill, no PnL from that trade.")
    print("  Optimal params flag so we skip the worst adverse fills; we give up some spread but avoid")
    print("  large losses. Stricter (5¢/5×) = flag less = more fills, more exposure to toxic moves.")
    print(f"  Optimal vs never-flag: PnL improves by {improvement:,.0f} cents.")
    print("  (Trade data only; no historical order book.)")
    print()

    print("Full sweep table:")
    print(sweep.to_string(index=False))
    print()

    # ---- PART 2: Slide stats with optimal params ----
    print("PART 2: Adverse selection stats (optimal params)")
    stats = run_slide_stats(data, opt_jump, opt_vol)
    print(f"  Flagged trades:   n = {stats['n_flagged']}, mean adverse move = {stats['mean_adverse_flagged_cents']:.3f} cents")
    print(f"  Unflagged trades: n = {stats['n_unflagged']}, mean adverse move = {stats['mean_adverse_unflagged_cents']:.3f} cents")
    print(f"  Ratio: Flagged trades had {stats['ratio']:.2f}x larger adverse moves than unflagged trades.")
    print(f"  T-test (flagged > unflagged): t = {stats['t_stat']:.4f}, p = {stats['p_value']:.4e}")
    print()

    if HAS_MPL and not args.no_chart:
        plot_slide_chart(stats, args.chart)


if __name__ == "__main__":
    main()
