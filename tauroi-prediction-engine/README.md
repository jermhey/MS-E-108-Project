# Tauroi Prediction Engine

**Adverse-Selection Detection for Kalshi Prediction Markets**

*Tauroi Technologies — MS&E 108 Project*

---

## Overview

This system detects **adverse selection events** in Kalshi prediction markets
and produces a real-time composite score that can be plugged into any
market-making infrastructure.

The core methodology is grounded in
[**"Toward Black-Scholes for Prediction Markets"** (Dalen, 2025)](https://arxiv.org/abs/2510.15205),
which models prediction-market prices as a **logit jump-diffusion** process.
We calibrate the model on Kalshi's **KXTOPMONTHLY** contracts (monthly Spotify
listener counts) and extract adverse-selection signals per trade.

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
- **Kalshi API client** — fetch live tick-level trade data for any contract

---

## Repository Structure

```
tauroi-prediction-engine/
├── src/
│   ├── as_detector.py        # AS detection: Kalman filter, EM calibration, burst detection
│   ├── mm_backtest.py        # Market-making backtest & fill-toxicity analysis
│   ├── belief_model.py       # Core belief model (Kalman + EM)
│   ├── belief_data.py        # High-frequency Kalshi data fetcher with caching
│   ├── kalshi_client.py      # Kalshi REST API client (RSA-PSS auth)
│   ├── config.py             # Settings loader (.env)
│   └── utils.py              # Logging, fee calculations, helpers
├── notebooks/
│   ├── adverse_selection.ipynb   # Full AS analysis: calibration, detection, backtest
│   └── alpha_proof.ipynb         # Belief-model evaluation & alpha attribution
├── cache/
│   └── kalshi_hf/            # Cached tick-level Kalshi trade data (Parquet)
├── tests/
│   └── test_as_detector.py   # Smoke tests for Kalman, burst detection, E2E
├── requirements.txt
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

---

## Integration

The detector is designed to plug into external market-making infrastructure.
Core usage:

```python
from src.as_detector import run_as_detection

result = run_as_detection(df, ticker="KXTOPMONTHLY-26FEB-BAD")

result.as_score      # float array [0, 1] — composite AS score per trade
result.gamma         # float array — jump posterior probability
result.burst_flags   # bool array — trade-arrival burst flags
result.events        # DataFrame of flagged AS events
```

The `as_score` can drive any response: spread widening, quote pulling,
position skewing, or risk monitoring.

---

## Adverse Selection Detection

The detection pipeline combines two complementary signals into a composite
**AS score** ∈ [0, 1]:

| Signal | Method | What it captures |
|--------|--------|------------------|
| **Jump posterior (γ_t)** | EM-separated jump component from logit increments | Sudden information-driven price moves |
| **Burst flag** | Two-timescale median inter-trade time comparison | Abnormal clustering of trade arrivals |

**Composite score:** `AS = α · γ_t + (1 − α) · burst_ratio`  (default α = 0.7)

### Validation

On the 13 most liquid KXTOPMONTHLY contracts (~33k trades):

- Subsequent price moves during AS-flagged periods are **1.5–2x larger**
  than during unflagged periods (p < 0.0001 on liquid contracts)
- Cross-ticker co-jump probability of 40–50% across correlated contracts

---

## Required Secrets

| Variable | Source | Required for |
|----------|--------|--------------|
| `KALSHI_ACCESS_KEY` | [kalshi.com/account/api](https://kalshi.com/account/api) | Fetching trade data |
| `KALSHI_API_SECRET` | Same (RSA private key, PEM) | Fetching trade data |

---

## References

- Dalen, S. (2025). *Toward Black-Scholes for Prediction Markets.*
  arXiv:2510.15205. [[link]](https://arxiv.org/abs/2510.15205)

---

*Tauroi Technologies — MS&E 108 Project*
