# Tauroi Prediction Engine

**High-Frequency Nowcasting & Arbitrage Engine for Pop-Culture Prediction Markets**

*Tauroi Technologies — MS&E 108 Project*

---

## Overview

Prediction markets on platforms like **Kalshi** let traders bet on real-world outcomes — for example, *"Which artist will have the most Spotify monthly listeners at the end of February?"*

The settlement oracle (Spotify) publishes listener counts on a **24-hour lag** (a step function). During that window, the *true* underlying trajectory is invisible to market participants who rely solely on the official number.

**Our edge: we don't wait.**

We ingest high-frequency social-velocity data (TikTok sound-post volumes, release calendars) from the **Chartmetric API** and build a real-time **"nowcast"** of the settlement number *before* it publishes. We then compare our model's fair value against the **live Kalshi market price** and exploit the spread.

### Key Capabilities

- **Jump-Diffusion pricing model** with conditional volatility and event-driven sigma boosts
- **Full Chartmetric API integration** — 7 years of Spotify + TikTok history, paginated automatically
- **Live Kalshi market discovery** — Two-stage event resolver with WTA competitor extraction
- **Automatic competitor intelligence** — resolves all competitors via Chartmetric search, fetches their live listener counts, and uses the leader as the pricing strike
- **Backtesting engine** with risk-adjusted sizing, stop-loss, and stress tests
- **Signal persistence** — every run logs a structured JSON record for audit and analysis
- **Cloud-ready** — Dockerfile, GitHub Actions CI/CD, loop mode for continuous monitoring

---

## Why Jump-Diffusion over Black-Scholes?

| Property | Black-Scholes (GBM) | Jump-Diffusion (Merton) |
|---|---|---|
| Path continuity | Continuous | Discontinuous jumps allowed |
| Tail behavior | Thin (Gaussian) | **Fat tails** from jump component |
| Viral event modeling | Under-prices tail risk | Captures sudden regime shifts |
| Volatility regime | Single constant sigma | Conditional sigma (TikTok-driven) |

A single viral TikTok trend can add **millions** of listeners in hours. Black-Scholes assigns near-zero probability to these moves. Our model detects the jump regime via TikTok velocity and continuously adjusts volatility:

```
sigma_adj = sigma_base + (gamma * tiktok_velocity)   # Conditional volatility
sigma_final = sigma_adj * event_impact_score          # Event-driven boost
```

---

## Repository Structure

```
tauroi-prediction-engine/
├── .github/workflows/
│   └── quant_pipeline.yml      # CI/CD — tests + live signal every 15 min
├── src/
│   ├── calibration.py          # Sigma, gamma, beta calibration from data
│   ├── chartmetric_client.py   # Chartmetric API (full history + competitor search)
│   ├── config.py               # Secure .env loader
│   ├── data_loader.py          # Mock data for tests
│   ├── kalshi_client.py        # Kalshi API (RSA-PSS auth + market discovery)
│   ├── pricing_engine.py       # Jump-Diffusion pricer + implied vol solver
│   └── utils.py                # Logging, normalisation, signal formatting
├── tests/
│   └── test_pricing.py         # Pricing engine unit tests
├── signals/
│   └── signal_log.jsonl        # Append-only signal audit trail
├── main.py                     # Entry point (single run / loop / GitHub Action)
├── run_backtest.py             # Historical backtester
├── Dockerfile                  # Container image
├── docker-compose.yml          # One-command deployment
├── requirements.txt            # Python dependencies
└── .env.example                # Required secrets template
```

---

## Quick Start

### 1. Clone & configure secrets

```bash
git clone <repo-url> && cd tauroi-prediction-engine
cp .env.example .env
# Edit .env with your API keys (see below)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
# Full live analysis (single run)
python main.py

# Data + calibration only (no Kalshi — faster, no API keys needed)
python main.py --data-only

# Continuous monitoring (re-runs every 15 minutes)
python main.py --loop --interval 900
```

---

## Required Secrets

| Variable | Where to get it | Required for |
|---|---|---|
| `KALSHI_ACCESS_KEY` | [kalshi.com/account/api](https://kalshi.com/account/api) | Live market prices |
| `KALSHI_API_SECRET` | Same (RSA private key, PEM format) | Live market prices |
| `CHARTMETRIC_REFRESH_TOKEN` | [app.chartmetric.com](https://app.chartmetric.com) → Account → API | All data fetching |
| `CHARTMETRIC_ARTIST_ID` | `app.chartmetric.com/artist?id=<THIS>` | Target artist (default: 214945 = Bad Bunny) |

For **GitHub Actions**, add these as repository secrets:
**Settings → Secrets and variables → Actions → New repository secret**

---

## Deployment

### Option A: GitHub Actions (recommended)

The repo ships with a production workflow that:
1. Runs **unit tests** on every push
2. Runs the **full live engine** every 15 minutes (cron)
3. Persists the signal to `signals/signal_log.jsonl` (committed to repo)
4. Uploads the signal log as a **build artifact** (90-day retention)

**Setup:** Push to GitHub, add secrets, enable Actions. That's it.

### Option B: Docker (any server)

```bash
# Build
docker build -t tauroi .

# Single run
docker run --env-file .env -v $(pwd)/signals:/app/signals tauroi

# Continuous loop (15-min interval, auto-restart)
docker compose up tauroi-loop -d
```

### Option C: Any server with Python

```bash
pip install -r requirements.txt
python main.py --loop --interval 900
```

Use `systemd`, `supervisord`, or `tmux` to keep it alive.

---

## Signal Log

Every run appends one JSON line to `signals/signal_log.jsonl`:

```json
{
  "artist": "Bad Bunny",
  "date": "2026-02-12",
  "spot_official": 101419370,
  "spot_nowcast": 117465088,
  "strike": 133758600,
  "leader": "Bruno Mars",
  "market_type": "winner_take_all",
  "fair_value": 0.2319,
  "market_price": 0.19,
  "edge": 0.0419,
  "signal": "HOLD",
  "sigma_adj": 0.9717,
  "ticker": "KXTOPMONTHLY-26FEB-BAD",
  "T_days": 15.9,
  "logged_at": "2026-02-12T19:01:19Z"
}
```

Load for analysis:

```python
import pandas as pd
df = pd.read_json("signals/signal_log.jsonl", lines=True)
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --dry-run                  Log signals without placing orders
  --data-only                Fetch + calibrate only (skip Kalshi)
  --strike FLOAT             Override strike price (default: 100M)
  --competitor-listeners N   Manual override for leader's listeners
  --loop                     Run continuously
  --interval SECONDS         Seconds between loop iterations (default: 900)
```

---

## Team

**Tauroi Technologies** — MS&E 108 Project

---

*Disclaimer: This is an academic research project. No real capital is at risk. All market interactions are read-only.*
