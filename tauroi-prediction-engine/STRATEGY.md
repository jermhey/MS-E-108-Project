# Tauroi Market-Making Strategy

## Overview

Automated market-maker for Kalshi's **KXTOPMONTHLY** winner-take-all contracts
(monthly Spotify listener counts). The bot quotes two-sided limit orders around
a model-derived fair value, earning the bid-ask spread while managing inventory
risk and defending against adverse selection.

## Fair Value Model

**Logit jump-diffusion with Kalman filtering** — the "Toward Black-Scholes for
Prediction Markets" approach.

1. Fetch tick-level trade history from Kalshi for each contract.
2. Transform prices to log-odds (logit) space where the process is approximately
   Gaussian.
3. Calibrate a jump-diffusion model (drift, volatility, jump intensity/size,
   mean-reversion speed) via walk-forward EM.
4. Run a one-pass Kalman filter to extract a smoothed fair value in probability
   space — strictly causal, no lookahead.

The model updates every 60 seconds using only Kalshi data (no external
dependencies beyond Spotify monitoring).

## Quoting Strategy

### Two-sided quotes with inventory skew

For each active contract, the bot posts:
- **Bid** (BUY YES) at `fair − half_spread − skew`
- **Ask** (BUY NO) at `100 − (fair + half_spread − skew)`

Both orders use `post_only=True` to guarantee maker fees.

### Spread sizing

- **Default half-spread**: 3 cents (6c full spread)
- **Dynamic spread**: scales with belief-volatility (`sigma_b`) — wider when
  the model is less certain, tighter when confident
- **Range**: 2c–8c half-spread depending on regime

### Inventory skew

Quotes shift by **1 cent per contract** of inventory to lean against the
position:

| Position | Effect |
|----------|--------|
| Long 5 YES contracts | Bid drops 5c, ask drops 5c (more eager to sell) |
| Flat | Symmetric quotes |
| Long 5 NO contracts | Bid rises 5c, ask rises 5c (more eager to buy YES) |

When inventory exceeds **±15 contracts**, the side that would increase exposure
is pulled entirely.

## Fee Structure

Kalshi maker fee: `0.0175 × C × P × (1 − P)` per fill.

| Contract price | Fee per contract |
|---------------|-----------------|
| 3c (tail) | 0.05c |
| 50c (coin flip) | 0.44c |
| 85c (favorite) | 0.22c |

Fees are negligible relative to the 6c full spread.

## Adverse Selection Defense

### Layer A — Reactive (price action)

Monitor the z-score of recent price moves in logit space:
- **|z| ≥ 2.5σ**: Pull all quotes for that contract (cancel resting orders,
  wait for the move to settle, requote at new fair value)
- **|z| ≥ 1.8σ**: Double the half-spread to widen quotes defensively

### Layer B — Predictive (Spotify monitoring)

The adverse selection events for KXTOPMONTHLY are:
- New song/album drops (biggest single factor)
- Playlist placements (Today's Top Hits, RapCaviar)
- Viral TikTok moments
- Concert tours / festivals

The bot monitors the Spotify Web API for:
- New releases (`/artists/{id}/albums`) within the last 3 days
- Popularity score jumps

When detected, quotes are widened or pulled before the information is fully
priced into the Kalshi market.

## Risk Controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_daily_loss_pct` | 10% | Circuit breaker: halt trading for the day |
| `max_order_size` | 10 contracts | Maximum per order |
| `max_position_per_ticker` | 50 contracts | Maximum exposure per contract |
| `min_balance_floor_cents` | $10 | Never trade below this balance |
| `STALE_ORDER_DRIFT_CENTS` | 3c | Cancel/replace if fair value drifts >3c |

## Backtest Results

Validated on 15 historical KXTOPMONTHLY tickers (~38k trades total):

| Configuration | Fills | Gross PnL | Fees | Net PnL |
|--------------|-------|-----------|------|---------|
| Full strategy (dynamic + skew + adv pull) | 3,810 | +$4.39 | $3.12 | **+$1.27** |
| Fixed 3c (skew + adv pull) | 5,045 | +$14.07 | $7.85 | **+$6.22** |
| Fixed 5c (skew + adv pull) | 3,909 | +$12.39 | $3.93 | **+$8.46** |
| No inventory skew | 902 | −$1.22 | $2.53 | **−$3.75** |

Key findings:
- Inventory skew is essential — without it, the strategy loses money
- Real fees (~0.08c/fill avg) are negligible vs the old 1.5c assumption
- The model generates consistent alpha vs a random-fair-value baseline

## Deployment

Runs on GitHub Actions with a 5-hour restart cycle:
- **Cron**: every 5 hours (overlapping runs cancel previous via concurrency group)
- **Push to main**: dry-run only (CI verification)
- **Manual dispatch**: live trading

State (positions, resting orders) persists to disk and reconciles with the
Kalshi API on each restart.
