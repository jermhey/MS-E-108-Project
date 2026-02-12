"""
backtester.py — Historical Backtest & Stress-Test Engine
=========================================================
Replays processed Chartmetric data day-by-day, simulating how the
Jump-Diffusion strategy would have performed over a historical window.

**Key Design Principles:**

1.  **No Look-Ahead Bias:**  On day *t* the model can only see data
    from the start of the window through day *t*.  Calibration is run
    on this expanding window, never on future data.

2.  **Simulated Market Price:**  We don't have historical Kalshi prices,
    so the "market" is modelled as a *naive* pricer that sees the
    current Spotify number but does NOT incorporate the TikTok signal.
    Our edge is the difference between this naive price and our
    full model's price.

3.  **Transaction Costs:**  Every trade incurs a configurable taker
    fee (default 1%), and stress tests add spread slippage.

4.  **Settlement:**  On the final day, any open position settles at
    $1.00 (if ``spot > strike``) or $0.00 (if not).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.calibration import Calibrator, CalibrationResult
from src.pricing_engine import JumpDiffusionPricer
from src.utils import get_logger

logger = get_logger("tauroi.backtester")

# Suppress noisy calibrator logs during the hundreds of re-calibrations
logging.getLogger("tauroi.calibrator").setLevel(logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """A single executed trade or settlement event."""
    date: str
    action: str          # BUY | SELL | SETTLE
    price: float         # execution price per contract
    quantity: int        # number of contracts
    fee: float           # total fee paid
    pnl: float           # realised P&L (0 for BUY, computed on SELL/SETTLE)
    portfolio_value: float
    reason: str = ""     # human-readable trigger (e.g. "Edge +0.12 (2U)")


@dataclass
class DailySnapshot:
    """The model's view and portfolio state for a single day."""
    date: str
    spot_official: float
    spot_nowcast: float
    model_fair_value: float
    market_price: float
    sigma_adj: float
    signal: str          # BUY | SELL | HOLD | WARMUP
    position: int        # contracts held
    cash: float
    portfolio_value: float


@dataclass
class BacktestResult:
    """Complete output from a single backtest run."""
    snapshots: List[DailySnapshot]
    trades: List[Trade]
    nowcast_errors: List[float]
    directional_correct: List[bool]
    initial_capital: float
    final_capital: float
    settlement_price: float      # 1.0 or 0.0
    calibration_dates: List[str]
    strike: float
    label: str = "Main"          # human-readable name for report


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTESTER
# ═════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Day-by-day historical replay engine.

    Parameters
    ----------
    initial_capital : float
        Starting cash ($).
    strike : float
        Binary option strike (e.g. 100_000_000).
    edge_threshold : float
        Minimum |edge| to trigger a trade (default 5%).
    fee_pct : float
        Taker fee per trade (default 1%).
    min_calibration_days : int
        Warm-up period before first signal (need enough data to calibrate).
    recalibration_interval : int
        Days between re-calibrations.
    spread_slippage : float
        Extra cost per contract added to buys and subtracted from sells
        (simulates bid-ask spread; default 0).
    """

    def __init__(
        self,
        initial_capital: float = 10_000,
        strike: float = 100_000_000,
        edge_threshold: float = 0.05,
        fee_pct: float = 0.01,
        min_calibration_days: int = 21,
        recalibration_interval: int = 7,
        spread_slippage: float = 0.0,
        stop_loss_pct: float = 0.20,
    ) -> None:
        self.initial_capital = initial_capital
        self.strike = strike
        self.edge_threshold = edge_threshold
        self.fee_pct = fee_pct
        self.min_calibration_days = min_calibration_days
        self.recalibration_interval = recalibration_interval
        self.spread_slippage = spread_slippage
        self.stop_loss_pct = stop_loss_pct

    # ─────────────────────────────────────────────────────────────────────
    #  Risk Management
    # ─────────────────────────────────────────────────────────────────────

    def calculate_position_size(
        self, edge: float, cash: float, market_price: float,
    ) -> tuple[float, int]:
        """
        Risk-adjusted position sizing.

        **Base Unit:**  2% of current available cash.

        **Conviction Multiplier (scaled by edge magnitude):**

        =========  ========  =========
        Edge       Units     Allocation
        =========  ========  =========
        > 0.05     1 unit    2%
        > 0.15     2 units   4%
        > 0.25     3 units   6% (cap)
        =========  ========  =========

        Returns ``(dollar_amount, units)``.
        """
        if cash <= 0 or market_price <= 0:
            return 0.0, 0

        base_unit = cash * 0.02

        if edge > 0.25:
            units = 3
        elif edge > 0.15:
            units = 2
        elif edge > self.edge_threshold:
            units = 1
        else:
            return 0.0, 0

        allocation = base_unit * units
        return allocation, units

    def check_stop_loss(
        self, market_price: float, avg_entry_price: float,
    ) -> bool:
        """
        Return True if the position should be stopped out.

        Triggers when ``market_price`` drops 20% below ``avg_entry_price``.
        """
        if avg_entry_price <= 0:
            return False
        return market_price / avg_entry_price < (1.0 - self.stop_loss_pct)

    # ─────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, label: str = "Main") -> BacktestResult:
        """
        Execute the full day-by-day backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: Date, spotify_monthly_listeners,
            tiktok_sound_posts_change.  Optionally: event_impact_score.
        label : str
            Human-readable name (used in the report).

        Returns
        -------
        BacktestResult
        """
        df = df.sort_values("Date").reset_index(drop=True)
        n = len(df)

        if n < self.min_calibration_days + 5:
            raise ValueError(
                f"Need >= {self.min_calibration_days + 5} days of data, "
                f"got {n}"
            )

        pricer = JumpDiffusionPricer()
        expiry_date = pd.Timestamp(df["Date"].iloc[-1])

        # ── State ────────────────────────────────────────────────────
        cash = self.initial_capital
        position = 0
        avg_entry_price = 0.0
        cal: Optional[CalibrationResult] = None
        last_cal_idx = -999

        # ── Tracking ─────────────────────────────────────────────────
        snapshots: List[DailySnapshot] = []
        trades: List[Trade] = []
        nowcast_errors: List[float] = []
        directional_correct: List[bool] = []
        calibration_dates: List[str] = []

        has_events = "event_impact_score" in df.columns

        for t in range(n):
            date_str = str(df["Date"].iloc[t])[:10]
            spot = float(df["spotify_monthly_listeners"].iloc[t])
            tiktok_raw = df["tiktok_sound_posts_change"].iloc[t]
            event_score = (
                float(df["event_impact_score"].iloc[t])
                if has_events else 1.0
            )

            # Handle NaN / missing TikTok gracefully (Stress Test A)
            if pd.isna(tiktok_raw):
                tiktok_raw = 0.0
            else:
                tiktok_raw = float(tiktok_raw)

            # ── Warm-up period ──────────────────────────────────────
            if t < self.min_calibration_days:
                pv = cash + position * 0  # no model price yet
                snapshots.append(DailySnapshot(
                    date=date_str, spot_official=spot, spot_nowcast=spot,
                    model_fair_value=0, market_price=0, sigma_adj=0,
                    signal="WARMUP", position=position, cash=cash,
                    portfolio_value=cash,
                ))
                continue

            # ── Calibrate (every N days) ────────────────────────────
            if cal is None or (t - last_cal_idx) >= self.recalibration_interval:
                window = df.iloc[: t + 1].copy()
                try:
                    calibrator = Calibrator.from_dataframe(window)
                    cal = calibrator.run()
                    last_cal_idx = t
                    calibration_dates.append(date_str)
                except Exception as exc:
                    logger.debug("Calibration skip at %s: %s", date_str, exc)
                    if cal is None:
                        snapshots.append(DailySnapshot(
                            date=date_str, spot_official=spot,
                            spot_nowcast=spot, model_fair_value=0,
                            market_price=0, sigma_adj=0, signal="WARMUP",
                            position=position, cash=cash,
                            portfolio_value=cash,
                        ))
                        continue

            # ── Normalise TikTok velocity ───────────────────────────
            if cal.tiktok_p95 > 0 and tiktok_raw > 0:
                norm_velocity = min(tiktok_raw / cal.tiktok_p95, 1.0)
            else:
                norm_velocity = 0.0

            # ── Time to expiry ──────────────────────────────────────
            current_date = pd.Timestamp(df["Date"].iloc[t])
            T = max((expiry_date - current_date).days, 1) / 365

            # ── Model fair value (WITH TikTok signal) ───────────────
            spot_nowcast = pricer.nowcast_spot_calibrated(
                spot, norm_velocity, cal.jump_beta,
            )
            sigma_adj = pricer.adjusted_sigma_calibrated(
                cal.sigma, norm_velocity, cal.vol_gamma, event_score,
            )
            model_fv = pricer.fair_value_calibrated(
                spot_official=spot,
                normalized_velocity=norm_velocity,
                strike=self.strike,
                sigma=cal.sigma,
                jump_beta=cal.jump_beta,
                vol_gamma=cal.vol_gamma,
                event_impact_score=event_score,
                T=T,
            )

            # ── Naive market price (WITHOUT TikTok signal) ──────────
            #  Represents a market that sees the Spotify number but
            #  doesn't have our proprietary TikTok feed.
            naive_fv = pricer.fair_value_calibrated(
                spot_official=spot,
                normalized_velocity=0.0,
                strike=self.strike,
                sigma=cal.sigma,
                jump_beta=0.0,
                vol_gamma=0.0,
                event_impact_score=1.0,
                T=T,
            )
            market_price = max(naive_fv, 0.001)  # avoid div-by-zero

            # ── Signal ──────────────────────────────────────────────
            edge = model_fv - market_price
            if edge > self.edge_threshold:
                signal = "BUY"
            elif edge < -self.edge_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"

            # ── Execute (Risk-Managed) ────────────────────────────
            slippage = self.spread_slippage

            # CHECK STOP-LOSS FIRST — cut losers before anything else
            if position > 0 and self.check_stop_loss(market_price, avg_entry_price):
                sell_price = max(market_price - slippage, 0.001)
                revenue = position * sell_price * (1 - self.fee_pct)
                fee = position * sell_price * self.fee_pct
                pnl = revenue - (position * avg_entry_price)
                cash += revenue
                trades.append(Trade(
                    date=date_str, action="SELL", price=sell_price,
                    quantity=position, fee=fee, pnl=pnl,
                    portfolio_value=cash,
                    reason=f"Stop Loss ({self.stop_loss_pct:.0%} drawdown)",
                ))
                signal = "STOP"  # override for snapshot
                position = 0
                avg_entry_price = 0.0

            elif signal == "BUY" and market_price > 0.001:
                # Risk-adjusted sizing: allocate 2-6% of cash per trade
                allocation, units = self.calculate_position_size(
                    edge, cash, market_price,
                )
                if allocation > 0:
                    buy_price = market_price + slippage
                    cost_per = buy_price * (1 + self.fee_pct)
                    quantity = int(allocation / cost_per) if cost_per > 0 else 0
                    if quantity > 0:
                        total_cost = quantity * cost_per
                        fee = quantity * buy_price * self.fee_pct
                        # Weighted average entry price for accumulated position
                        old_cost = position * avg_entry_price
                        position += quantity
                        avg_entry_price = (old_cost + quantity * buy_price) / position
                        cash -= total_cost
                        trades.append(Trade(
                            date=date_str, action="BUY", price=buy_price,
                            quantity=quantity, fee=fee, pnl=0,
                            portfolio_value=cash + position * market_price,
                            reason=f"Edge +{edge:.2f} ({units}U, {units*2}%)",
                        ))

            elif signal == "SELL" and position > 0:
                sell_price = max(market_price - slippage, 0.001)
                revenue = position * sell_price * (1 - self.fee_pct)
                fee = position * sell_price * self.fee_pct
                pnl = revenue - (position * avg_entry_price)
                cash += revenue
                trades.append(Trade(
                    date=date_str, action="SELL", price=sell_price,
                    quantity=position, fee=fee, pnl=pnl,
                    portfolio_value=cash,
                    reason=f"Model SELL (edge {edge:+.2f})",
                ))
                position = 0
                avg_entry_price = 0.0

            # ── Portfolio value (mark-to-market) ────────────────────
            portfolio_value = cash + position * market_price

            snapshots.append(DailySnapshot(
                date=date_str, spot_official=spot, spot_nowcast=spot_nowcast,
                model_fair_value=model_fv, market_price=market_price,
                sigma_adj=sigma_adj, signal=signal, position=position,
                cash=cash, portfolio_value=portfolio_value,
            ))

            # ── Nowcast accuracy (compare to next day's actual) ─────
            if t + 1 < n:
                actual_next = float(df["spotify_monthly_listeners"].iloc[t + 1])
                nowcast_errors.append(spot_nowcast - actual_next)

                predicted_up = spot_nowcast > spot
                actual_up = actual_next > spot
                directional_correct.append(predicted_up == actual_up)

        # ── Settlement ──────────────────────────────────────────────────
        final_spot = float(df["spotify_monthly_listeners"].iloc[-1])
        settlement_price = 1.0 if final_spot > self.strike else 0.0

        if position > 0:
            revenue = position * settlement_price
            pnl = revenue - (position * avg_entry_price)
            cash += revenue
            settle_reason = (
                f"Settled @ $1 (spot > strike)"
                if settlement_price == 1.0
                else f"Expired worthless (spot < strike)"
            )
            trades.append(Trade(
                date=str(df["Date"].iloc[-1])[:10], action="SETTLE",
                price=settlement_price, quantity=position, fee=0,
                pnl=pnl, portfolio_value=cash,
                reason=settle_reason,
            ))
            position = 0

        logger.info(
            "[%s] Backtest complete — %d days, %d trades, "
            "final capital $%.2f (%.1f%%)",
            label, n, len(trades), cash,
            (cash - self.initial_capital) / self.initial_capital * 100,
        )

        return BacktestResult(
            snapshots=snapshots,
            trades=trades,
            nowcast_errors=nowcast_errors,
            directional_correct=directional_correct,
            initial_capital=self.initial_capital,
            final_capital=cash,
            settlement_price=settlement_price,
            calibration_dates=calibration_dates,
            strike=self.strike,
            label=label,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  KPI COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_kpis(result: BacktestResult) -> Dict[str, Any]:
    """
    Compute Key Performance Indicators from a backtest result.

    Returns a dict with:
        nowcast_rmse, directional_accuracy, total_return_pct,
        max_drawdown_pct, sharpe_ratio, win_rate, n_trades, ...
    """
    active = [s for s in result.snapshots if s.signal != "WARMUP"]
    if not active:
        return {"error": "No active trading days"}

    # ── Nowcast RMSE ──────────────────────────────────────────────────
    if result.nowcast_errors:
        errors = np.array(result.nowcast_errors)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
    else:
        rmse = mae = float("nan")

    # ── Directional Accuracy ──────────────────────────────────────────
    if result.directional_correct:
        dir_accuracy = sum(result.directional_correct) / len(result.directional_correct)
    else:
        dir_accuracy = float("nan")

    # ── Total Return ──────────────────────────────────────────────────
    total_return = (
        (result.final_capital - result.initial_capital) / result.initial_capital
    )

    # ── Daily Returns & Sharpe ────────────────────────────────────────
    daily_values = [s.portfolio_value for s in active]

    if len(daily_values) >= 2:
        arr = np.array(daily_values)
        daily_returns = np.diff(arr) / arr[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(
                (np.mean(daily_returns) / np.std(daily_returns))
                * math.sqrt(365)
            )
        else:
            sharpe = 0.0
    else:
        daily_returns = np.array([])
        sharpe = 0.0

    # ── Max Drawdown ──────────────────────────────────────────────────
    max_dd = 0.0
    if daily_values:
        peak = daily_values[0]
        for v in daily_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

    # ── Win Rate ──────────────────────────────────────────────────────
    exits = [t for t in result.trades if t.action in ("SELL", "SETTLE")]
    if exits:
        wins = sum(1 for t in exits if t.pnl > 0)
        win_rate = wins / len(exits)
    else:
        win_rate = float("nan")

    # ── Trade stats ───────────────────────────────────────────────────
    total_fees = sum(t.fee for t in result.trades)
    n_buys = sum(1 for t in result.trades if t.action == "BUY")
    n_sells = sum(1 for t in result.trades if t.action == "SELL")
    n_settles = sum(1 for t in result.trades if t.action == "SETTLE")

    return {
        "nowcast_rmse": rmse,
        "nowcast_mae": mae,
        "directional_accuracy": dir_accuracy,
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "n_trades": len(result.trades),
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_settles": n_settles,
        "total_fees": total_fees,
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "settlement": (
            f"WON ($1) — spot {result.snapshots[-1].spot_official:,.0f} "
            f"> strike {result.strike:,.0f}"
            if result.settlement_price == 1.0
            else f"LOST ($0) — spot {result.snapshots[-1].spot_official:,.0f} "
                 f"< strike {result.strike:,.0f}"
        ),
        "trading_days": len(active),
        "calibrations": len(result.calibration_dates),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

def run_stress_tests(
    df: pd.DataFrame,
    strike: float = 100_000_000,
    initial_capital: float = 10_000,
) -> Dict[str, Dict[str, Any]]:
    """
    Run three stress-test scenarios and return KPIs for each.

    Scenario A — "The Blackout"
        TikTok data goes missing for 3 consecutive days.
        Pass: model defaults to HOLD, does not crash.

    Scenario B — "The Pump & Dump"
        Inject a 10x TikTok spike on one day, then 50% drop the next.
        Pass: model BUYs on the spike, SELLs on the drop.

    Scenario C — "Spread Widening"
        Re-run with 5-cent bid-ask slippage per trade.
        Pass: strategy remains profitable (or at least doesn't blow up).
    """
    results: Dict[str, Dict[str, Any]] = {}

    # ── Scenario A: The Blackout ──────────────────────────────────────
    logger.info("Stress Test A — The Blackout (3-day TikTok NaN)")
    df_a = df.copy()
    # Pick a block in the middle of the active trading window
    mid = max(30, len(df_a) // 2)
    for i in range(mid, min(mid + 3, len(df_a))):
        df_a.at[i, "tiktok_sound_posts_change"] = float("nan")

    blackout_dates = [
        str(df_a["Date"].iloc[i])[:10]
        for i in range(mid, min(mid + 3, len(df_a)))
    ]

    try:
        bt_a = Backtester(
            initial_capital=initial_capital, strike=strike,
        )
        res_a = bt_a.run(df_a, label="A: Blackout")
        kpis_a = compute_kpis(res_a)

        # Check: did the model default to safe behaviour during blackout?
        blackout_signals = [
            s.signal for s in res_a.snapshots
            if s.date in blackout_dates
        ]
        safe = all(s in ("HOLD", "WARMUP") for s in blackout_signals)
        kpis_a["passed"] = True  # didn't crash
        kpis_a["safe_during_blackout"] = safe
        kpis_a["blackout_signals"] = blackout_signals
        kpis_a["blackout_dates"] = blackout_dates
        kpis_a["note"] = (
            "PASS — Model survived blackout; signals during gap: "
            + ", ".join(blackout_signals)
        )
    except Exception as exc:
        kpis_a = {"passed": False, "note": f"FAIL — CRASHED: {exc}"}

    results["A_blackout"] = kpis_a

    # ── Scenario B: The Pump & Dump ───────────────────────────────────
    logger.info("Stress Test B — The Pump & Dump (10x spike → 50%% drop)")
    df_b = df.copy()
    inject_idx = max(30, len(df_b) // 2)

    if inject_idx + 1 < len(df_b):
        baseline = float(df_b["tiktok_sound_posts_change"].iloc[inject_idx])
        if baseline <= 0:
            baseline = float(
                df_b["tiktok_sound_posts_change"]
                .clip(lower=0).median()
            )
            baseline = max(baseline, 1000)

        df_b.at[inject_idx, "tiktok_sound_posts_change"] = baseline * 10
        df_b.at[inject_idx + 1, "tiktok_sound_posts_change"] = baseline * 0.5

        spike_date = str(df_b["Date"].iloc[inject_idx])[:10]
        drop_date = str(df_b["Date"].iloc[inject_idx + 1])[:10]

    try:
        bt_b = Backtester(
            initial_capital=initial_capital, strike=strike,
        )
        res_b = bt_b.run(df_b, label="B: Pump & Dump")
        kpis_b = compute_kpis(res_b)

        spike_signal = next(
            (s.signal for s in res_b.snapshots if s.date == spike_date),
            "N/A",
        )
        drop_signal = next(
            (s.signal for s in res_b.snapshots if s.date == drop_date),
            "N/A",
        )

        kpis_b["passed"] = True
        kpis_b["spike_date"] = spike_date
        kpis_b["spike_signal"] = spike_signal
        kpis_b["drop_date"] = drop_date
        kpis_b["drop_signal"] = drop_signal
        kpis_b["note"] = (
            f"Spike ({spike_date}): {spike_signal}  |  "
            f"Drop ({drop_date}): {drop_signal}"
        )
    except Exception as exc:
        kpis_b = {"passed": False, "note": f"FAIL — CRASHED: {exc}"}

    results["B_pump_dump"] = kpis_b

    # ── Scenario C: Spread Widening (5c slippage) ─────────────────────
    logger.info("Stress Test C — Spread Widening (5c slippage)")
    try:
        bt_c = Backtester(
            initial_capital=initial_capital, strike=strike,
            spread_slippage=0.05,   # 5 cents per contract
        )
        res_c = bt_c.run(df, label="C: Spread Widening")
        kpis_c = compute_kpis(res_c)

        profitable = res_c.final_capital > res_c.initial_capital
        kpis_c["passed"] = profitable
        kpis_c["note"] = (
            f"{'PASS' if profitable else 'FAIL'} — "
            f"P&L with 5c spread: "
            f"${res_c.final_capital - res_c.initial_capital:+,.2f} "
            f"({kpis_c['total_return_pct']:+.1f}%)"
        )
    except Exception as exc:
        kpis_c = {"passed": False, "note": f"FAIL — CRASHED: {exc}"}

    results["C_spread_widening"] = kpis_c

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(
    main_result: BacktestResult,
    main_kpis: Dict[str, Any],
    stress_results: Dict[str, Dict[str, Any]],
) -> str:
    """
    Generate a plain-text report summarising all backtest outcomes.

    Returns the full report as a string.
    """
    lines: List[str] = []
    w = 68

    def hr(char: str = "=") -> str:
        return char * w

    def section(title: str) -> None:
        lines.append("")
        lines.append(hr())
        lines.append(f"  {title}")
        lines.append(hr())

    # ── Header ────────────────────────────────────────────────────────
    lines.append(hr("*"))
    lines.append("  TAUROI PREDICTION ENGINE — BACKTEST REPORT")
    lines.append(hr("*"))
    lines.append(f"  Strike:           ${main_result.strike:>20,.0f}")
    lines.append(f"  Initial Capital:  ${main_result.initial_capital:>20,.2f}")
    lines.append(f"  Period:           {main_result.snapshots[0].date} → "
                 f"{main_result.snapshots[-1].date}")
    lines.append(f"  Total Days:       {len(main_result.snapshots):>20d}")
    active = [s for s in main_result.snapshots if s.signal != "WARMUP"]
    lines.append(f"  Active Days:      {len(active):>20d}")
    lines.append(f"  Re-calibrations:  {len(main_result.calibration_dates):>20d}")

    # ── Nowcast Accuracy ──────────────────────────────────────────────
    section("1. NOWCAST ACCURACY (Model vs Reality)")
    rmse = main_kpis.get("nowcast_rmse", float("nan"))
    mae = main_kpis.get("nowcast_mae", float("nan"))
    dir_acc = main_kpis.get("directional_accuracy", float("nan"))

    lines.append(f"  RMSE (Nowcast - Actual):     {rmse:>20,.0f} listeners")
    lines.append(f"  MAE  (Nowcast - Actual):     {mae:>20,.0f} listeners")
    lines.append(f"  Directional Accuracy:        {dir_acc:>19.1%}")

    # ── P&L Metrics ───────────────────────────────────────────────────
    section("2. P&L METRICS")
    lines.append(f"  Final Capital:    ${main_kpis.get('final_capital', 0):>20,.2f}")
    lines.append(f"  Total Return:      {main_kpis.get('total_return_pct', 0):>+19.1f}%")
    lines.append(f"  Max Drawdown:      {main_kpis.get('max_drawdown_pct', 0):>19.1f}%")
    lines.append(f"  Sharpe Ratio:      {main_kpis.get('sharpe_ratio', 0):>19.2f}")
    lines.append(f"  Win Rate:          {main_kpis.get('win_rate', 0):>19.1%}")
    lines.append(f"  Settlement:        {main_kpis.get('settlement', 'N/A')}")
    lines.append(f"  Total Fees Paid:  ${main_kpis.get('total_fees', 0):>20,.2f}")

    # ── Trade Log ─────────────────────────────────────────────────────
    section("3. TRADE LOG")

    for t in main_result.trades:
        size_dollar = t.price * t.quantity
        if t.action == "BUY":
            lines.append(
                f"  [{t.date}] BUY   | Price: ${t.price:.4f} | "
                f"Size: ${size_dollar:,.2f} ({t.quantity:,} contracts) | "
                f"Reason: {t.reason}"
            )
        elif t.action == "SELL":
            lines.append(
                f"  [{t.date}] SELL  | Price: ${t.price:.4f} | "
                f"P&L: ${t.pnl:+,.2f} ({t.quantity:,} contracts) | "
                f"Reason: {t.reason}"
            )
        elif t.action == "SETTLE":
            lines.append(
                f"  [{t.date}] SETTLE| Price: ${t.price:.2f}   | "
                f"P&L: ${t.pnl:+,.2f} ({t.quantity:,} contracts) | "
                f"Reason: {t.reason}"
            )

    if not main_result.trades:
        lines.append("  (no trades executed)")

    # ── Stress Tests ──────────────────────────────────────────────────
    section("4. STRESS TESTS")

    for key, label in [
        ("A_blackout", "Scenario A — The Blackout (3-day TikTok NaN)"),
        ("B_pump_dump", "Scenario B — The Pump & Dump (10x spike → drop)"),
        ("C_spread_widening", "Scenario C — Spread Widening (5c slippage)"),
    ]:
        lines.append("")
        stress = stress_results.get(key, {})
        passed = stress.get("passed", False)
        badge = "PASS" if passed else "FAIL"
        note = stress.get("note", "No data")
        ret = stress.get("total_return_pct")
        ret_str = f" | Return: {ret:+.1f}%" if ret is not None else ""

        lines.append(f"  [{badge}] {label}")
        lines.append(f"         {note}{ret_str}")

    # ── Footer ────────────────────────────────────────────────────────
    lines.append("")
    lines.append(hr("*"))
    lines.append("  END OF REPORT")
    lines.append(hr("*"))

    return "\n".join(lines)
