# Tauroi Prediction Engine

**Adverse-Selection Detection & Market-Making for Kalshi Prediction Markets**

*Tauroi Technologies — MS&E 108 Project*

---

## Overview

This system detects **adverse selection events** in Kalshi prediction markets
and uses those signals to protect a market-making strategy from toxic order flow.

The core methodology is grounded in
[**"Toward Black-Scholes for Prediction Markets"** (Dalen, 2025)](https://arxiv.org/abs/2510.15205),
which models prediction-market prices as a **logit jump-diffusion** process.
We calibrate the model on Kalshi's **KXTOPMONTHLY** contracts (monthly Spotify
listener counts) and extract real-time adverse-selection signals that a market
maker can act on.

### Key Capabilities

- **Logit jump-diffusion model** — prices modeled in log-odds space where
  Gaussian assumptions hold, with explicit jump and diffusion components
- **Heteroskedastic Kalman filter** — denoises log-odds series accounting for
  irregular trade arrival times
- **Rolling EM calibration** — extracts belief-volatility (σ_b), jump intensity
  (λ), and jump variance (s_J²) with walk-forward estimation
- **Composite adverse-selection score** — combines jump posterior probability
  (γ_t) with trade-arrival burst detection for robust AS identification
- **Market-making backtest** — compares naive vs. AS-informed quoting with
  real Kalshi maker fees and mark-to-market PnL
- **Live execution engine** — order lifecycle management with inventory skew,
  stale-order replacement, and quote pulling on AS events
- **Cloud-ready** — Dockerfile, GitHub Actions CI/CD

---

## Repository Structure

```
tauroi-prediction-engine/
├── src/
│   ├── as_detector.py        # AS detection: Kalman filter, EM calibration, burst detection
│   ├── mm_backtest.py        # Market-making backtest & fill-toxicity analysis
│   ├── adverse_selection.py  # Reactive AS defense (z-score quote pulling)
│   ├── belief_model.py       # Core belief model (Kalman + EM)
│   ├── belief_data.py        # High-frequency Kalshi data fetcher with caching
│   ├── belief_eval.py        # Walk-forward model evaluation
│   ├── calibration.py        # Parameter calibration from data
│   ├── pricing_engine.py     # Jump-diffusion pricer & Monte Carlo
│   ├── executor.py           # Order execution engine (directional + MM modes)
│   ├── kalshi_client.py      # Kalshi REST API client (RSA-PSS auth)
│   ├── risk_manager.py       # Position limits, circuit breakers
│   ├── position_store.py     # Persistent position tracking
│   ├── config.py             # Settings loader (.env)
│   └── utils.py              # Logging, fee calculations, helpers
├── notebooks/
│   ├── adverse_selection.ipynb   # Full AS analysis: calibration, detection, backtest
│   └── alpha_proof.ipynb         # Belief-model evaluation & alpha attribution
├── cache/
│   └── kalshi_hf/            # Cached tick-level Kalshi trade data (Parquet)
├── tests/
│   └── __init__.py
├── signals/
│   └── signal_log.jsonl      # Append-only signal audit trail
├── main.py                   # Entry point (scan / live / market-making modes)
├── Dockerfile
├── requirements.txt
├── STRATEGY.md               # Detailed market-making strategy documentation
└── .env.example              # Required secrets template
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url> && cd tauroi-prediction-engine
cp .env.example .env
# Edit .env with your Kalshi API keys
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the AS analysis notebook

```bash
jupyter notebook notebooks/adverse_selection.ipynb
```

The notebook walks through the full pipeline:
1. Loads cached tick data from `cache/kalshi_hf/`
2. Calibrates the logit jump-diffusion model via rolling EM
3. Detects adverse-selection events (jump posterior + burst clustering)
4. Validates with fill-toxicity statistics
5. Runs market-making backtest (naive MM vs. AS-informed MM)
6. Sweeps AS-score thresholds to find optimal operating point

### 4. Live trading (optional)

```bash
# Dry run — log signals without placing orders
python main.py --belief --scan --dry-run

# Live market-making
python main.py --live --market-making
```

---

## Adverse Selection Detection

The detection pipeline combines two complementary signals into a composite
**AS score** ∈ [0, 1]:

| Signal | Method | What it captures |
|--------|--------|------------------|
| **Jump posterior (γ_t)** | EM-separated jump component from logit increments | Sudden information-driven price moves |
| **Burst flag** | Two-timescale median inter-trade time comparison | Abnormal clustering of trade arrivals |

**Composite score:** `AS = α · γ_t + (1 − α) · burst_ratio`  (default α = 0.7)

When AS score exceeds the threshold (τ ≈ 0.7), the market maker pulls quotes
to avoid adverse fills.

### Validation

On the 13 most liquid KXTOPMONTHLY contracts (~38k trades):

- Subsequent price moves during AS-flagged periods are **1.2–1.6× larger**
  than during unflagged periods (p < 0.0001)
- AS-informed MM shows improved PnL vs. naive MM on the most liquid contracts
  at τ = 0.7–0.8

---

## Required Secrets

| Variable | Source | Required for |
|----------|--------|--------------|
| `KALSHI_ACCESS_KEY` | [kalshi.com/account/api](https://kalshi.com/account/api) | All Kalshi operations |
| `KALSHI_API_SECRET` | Same (RSA private key, PEM) | All Kalshi operations |

Add these as GitHub repository secrets for CI/CD deployment.

---

## References

- Dalen, S. (2025). *Toward Black-Scholes for Prediction Markets.*
  arXiv:2510.15205. [[link]](https://arxiv.org/abs/2510.15205)

---

*Tauroi Technologies — MS&E 108 Project*
