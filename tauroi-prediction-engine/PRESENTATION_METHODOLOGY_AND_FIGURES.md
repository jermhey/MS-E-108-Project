# Presentation Methodology & Figures: Q&A Reference

This document explains the methodology used in the presentation, how it differs from earlier approaches, and what each figure shows. Use this to answer audience questions after the presentation.

---

## 1. The Three Processes: Evolution of Our Approach

### Process 1: Naive Heuristic Detector (`as_parameter_sweep.py`)

**What it was:**
- A simple rule-based detector: flag a trade if *both* (1) price moved more than X cents and (2) volume exceeded Y× the rolling average.
- No Kalman filter, no EM calibration—just raw price jumps and volume spikes.
- Used a fixed 50-trade rolling window for the volume baseline.
- Simulated a market maker that captured spread on “normal” trades and took full loss on “toxic” trades; swept jump thresholds (2–5¢) and volume multipliers (2×–5×) *in-sample* to maximize PnL—no holdout, no validation. (Heuristic tuning.)

**Limitations:**
- In-sample evaluation only (no train/validation/test split).
- Same-trade lookahead: the detector used information from the *current* trade to decide whether to flag it—unrealistic in practice.
- No fees, simplified PnL model.
- Not suitable for defensible profitability claims.

---

### Process 2: Earlier Notebook Backtests (`weather_backtest.ipynb`, `election_backtest.ipynb`, etc.)

**What it was:**
- Used the sophisticated detector (Kalman + EM + bursts) and the MM backtest engine.
- Claimed train/test splits, but in practice often evaluated on full data or in-sample.
- **Heuristic threshold tuning:** tau was hardcoded (e.g., 0.9) or chosen by hand, not tuned per market on holdout data.
- PnL model sometimes double-counted spread or used inconsistent MTM horizons.
- Same-trade lookahead: decisions used the AS score of the *current* trade.

**Limitations:**
- Lookahead bias made the informed MM appear to perform *worse* than naive in some runs (paradoxically).
- No systematic, data-driven threshold selection—just heuristic or fixed tau.
- Inconsistent assumptions across notebooks.
- No event-level holdout, no bootstrap uncertainty.

---

### Process 3: Canonical Holdout Evaluation (Current) — **Source of Our Presentation Figures**

**What it is:**
- **Detector:** The full Dalen-style pipeline in `src/as_detector.py`: Kalman smoothing, rolling EM calibration for jump posterior γ, burst detection, and a composite AS score.
- **Backtest:** `src/mm_backtest.py` with lagged signal, unified PnL, and fee toggle.
- **Evaluation:** `scripts/holdout_mm_evaluation.py` with chronological event-level train/validation/test splits, threshold tuning on holdout, and event-level bootstrap confidence intervals.

**Key methodological fixes:**
1. **Data-driven threshold tuning (replacing heuristics):** Instead of hardcoding tau or picking it by hand, we sweep thresholds on *train* data, take the top 3, and choose the best on *validation* data. Each market gets its own tau (e.g., MLB/Weather 0.1, Spotify 0.3, Election 0.2). This replaces the previous heuristic/fixed approach.
2. **Lagged signal:** The MM uses `as_score[i-1]` (from the *previous* trade) to decide whether to quote at trade `i`, removing same-trade lookahead.
3. **Chronological holdout:** Events are sorted by earliest trade timestamp; ~60% train, ~20% validation, ~20% test. Threshold is chosen on validation; test is never touched until final reporting.
4. **Unified PnL:** `Net PnL = Gross Spread + MTM − Fees`, where MTM is signed (favorable moves help; adverse moves hurt). No double-counting.
5. **Time-based MTM:** 5-minute horizon from each fill (configurable). Future price is the first trade price ≥5 minutes after the fill.
6. **Fee toggle:** We report no-fee results for the presentation (institutional maker assumption); the script also supports Kalshi’s maker-fee schedule.
7. **Bootstrap uncertainty:** Event-level resampling to report 95% CI and P(delta > 0) for improvement.

**Tuning summary:** Process 1 used heuristic *detection* (raw price/volume) and heuristic *tuning* (in-sample sweep). Process 2 used sophisticated detection but heuristic/hardcoded *thresholds*. Process 3 uses sophisticated detection and **data-driven, holdout-based threshold selection**—no more guessing tau. (Detector *parameters* like EM window are still fixed; only the *threshold* tau is tuned.)

**Bottom line:** All presentation figures come from this canonical process. We do *not* report numbers from Process 1 or Process 2.

---

### Backtesting Issues Process 3 Fixes vs. Process 1 and 2

**vs. Process 1 (naive heuristic):**

| Issue | Process 1 | Process 3 |
|-------|-----------|-----------|
| Lookahead bias | Uses the current trade’s price move and volume to decide whether to flag it—information that doesn’t exist at decision time. | Uses the *previous* trade’s AS score to decide whether to quote. The decision is based only on past information. |
| In-sample overfitting | Parameters (jump threshold, volume multiplier) are chosen by maximizing PnL on the same data they’re evaluated on. | Train finds candidate taus, validation picks the best, test is held out and never used for tuning. |
| Detection model | Naive rule (price jump + volume spike). | Kalman + EM + burst model that separates jumps from diffusion and adjusts for arrival-rate anomalies. |
| PnL model | Simplified (spread vs. toxic loss). | Full PnL: spread + signed MTM − fees, no double-counting. |
| Uncertainty | Single-point estimates. | Event-level bootstrap gives CIs and P(delta > 0). |

**vs. Process 2 (notebook backtests):**

| Issue | Process 2 | Process 3 |
|-------|-----------|-----------|
| Lookahead bias | Used the current trade’s AS score to decide whether to quote. | Uses a 1-trade lag: `as_score[i-1]` for the decision at trade `i`. |
| Threshold choice | tau hardcoded or chosen by hand (e.g., 0.9). | tau chosen on validation per market. |
| Data leakage | Train/test split often not applied or not chronological. | Chronological event-level split; test events are never seen during tuning. |
| PnL consistency | Spread/MTM sometimes double-counted or inconsistent across notebooks. | Single, consistent PnL definition everywhere. |
| Reproducibility | Logic scattered across notebooks with different assumptions. | One entrypoint (`holdout_mm_evaluation.py`) with fixed assumptions. |

**Overall improvements in Process 3:**
1. **No lookahead** — Decisions use only information available at decision time.
2. **Proper holdout** — Train → validate → test with time-respecting, event-level splits.
3. **Data-driven tuning** — tau per market instead of heuristic/hardcoded values.
4. **Correct PnL** — Spread, MTM, and fees combined correctly, no double-counting.
5. **Uncertainty** — Bootstrap intervals instead of point estimates.
6. **Single methodology** — One canonical script instead of many notebooks.
7. **Auditability** — Clear separation of detector, backtest, and evaluation.

**Why Process 3 is our best evaluation model:**
- **Cautious claims:** We avoid optimistic bias from lookahead, in-sample tuning, and inconsistent PnL.
- **Out-of-sample:** Test data is never used for model or threshold choice; it reflects performance on unseen events.
- **Market-specific thresholds:** Different markets get different taus instead of a single global value.
- **Interpretable and defensible:** Each step (detector, backtest, evaluation) is documented and reproducible.

**Why we’re using it for live deployment:**
- **Same logic, different data:** The detector (Kalman, EM, bursts) and decision rule (quote when `as_score[i-1] <= tau`) are identical in backtest and live.
- **Tuned taus per market:** Tau per market comes from holdout, not from live trading; deployment uses those precomputed values.
- **No hidden assumptions:** The backtest mirrors the live logic (spread, MTM horizon, lag). The main simplification is the fill model (we assume passive fills at quoted prices, no order-book simulation).
- **Risk-aware:** The bootstrap CI reflects variation across events; we know improvement is not guaranteed and can calibrate risk accordingly.

**Bottom line:** Process 3 removes lookahead and overfitting, uses proper holdout and market-specific tuning, and aligns backtest logic with live deployment—so it is both our best evaluation framework and the one we use for going live.

---

## 2. Figure-by-Figure Reference

### Slide 7: The Lie Detector (Detector Pipeline)

**File:** `slide07_detector_pipeline.png` (if present) or a diagram in the slide deck.

**What it shows:**
- Step 1: Ingest trade prices, timestamps, and volume.
- Step 2: Kalman filter smooths log-odds prices to separate signal from tick-noise.
- Step 3: Rolling EM calibration extracts jump posterior γ and burst flags; these combine into the composite AS score.

**Source:** `src/as_detector.py`. The pipeline is: `kalman_filter_hf` → `em_calibrate_window` → `detect_bursts` → composite score.

**Q&A:**  
- *“What’s the Kalman filter doing?”* Smoothing price noise so we don’t overreact to every tick.  
- *“What’s a burst?”* A spike in trade arrival rate relative to recent history (two-timescale EMA to avoid drift).

---

### Slide 8: Catching Them Red-Handed

**What it shows:**
- 1.3M+ historical trades across Spotify, weather, election, and MLB.
- 1.7× larger adverse moves on flagged trades vs. unflagged trades.
- Holdout tests: flagged trades consistently produced worse post-trade moves.

**Source:** `holdout_mm_evaluation.py` → `toxicity_summary()`. For each market, we compute mean signed 5-minute adverse drift for trades flagged vs. unflagged (at chosen tau). The 1.7× is an aggregate ratio across markets; exact ratios vary by market.

**Q&A:**  
- *“How do you define adverse move?”* Change in mid-price from pre-trade to 5 minutes later, signed so positive = move against the MM.  
- *“What’s the p-value?”* A t-test can be run on flagged vs. unflagged drift; the script focuses on the ratio. You can add a t-test if asked.

---

### Slide 9: The Mathematical Tradeoff (Anatomy of the Shield)

**File:** `figures/slide09_anatomy_of_the_shield.png`

**What it shows:**
- Two bars for a **single MLB game** (TOR vs LAD, Oct 27):
  - **Opportunity Cost (gray):** 4,618¢ — spread foregone by pulling quotes on flagged fills.
  - **Toxic Loss Avoided (green):** 22,657¢ — gross adverse 5-minute MTM we would have taken on those same skipped fills.

**Source:** Computed from canonical no-fee holdout: for each trade where the detector fired (at chosen tau), we sum (1) half-spread given up and (2) positive adverse MTM avoided. Only includes the one holdout MLB event `KXMLBGAME-25OCT27TORLAD`.

**Important caveat:** The net PnL improvement for that game is *not* 22,657 − 4,618. We also gave up favorable MTM on some skipped fills. The actual net improvement for that game is ~+2,482¢. The bars illustrate the *gross* tradeoff (cost vs. benefit on the bad side only).

**Q&A:**  
- *“Why only one game?”* To keep the story simple and literal: “In one MLB game, here’s the tradeoff.”  
- *“What about favorable fills we missed?”* Correct—we left those out of the bar chart. The full decomposition would need a third bar for favorable MTM foregone.

---

### Slide 10a: Know Your Market (Market Microstructure Fingerprint)

**File:** `figures/slide10_universal_heuristics_table.png`

**What it shows:**
- A 4×3 matrix: rows = market families (MLB, Weather, Election, Spotify); columns = three metrics.
- **Activity:** Median trades per hour per contract. MLB ~290/hr (very active), Spotify ~0.6/hr (sparse).
- **Move / Trade:** Mean absolute price change per trade in cents. Spotify ~1.71¢ (jumpy), MLB ~0.37¢ (smooth).
- **Toxicity Premium:** Ratio of mean adverse 5-minute drift on flagged trades vs. unflagged trades. MLB 1.63×, Weather 1.13×, etc.

**Source:** `load_dataset` + `evaluate_market` for descriptive stats; toxicity premium from `toxicity_summary()` in holdout eval.

**Q&A:**  
- *“Why these three metrics?”* They summarize liquidity (activity), price behavior (move), and how much worse flagged flow is (toxicity). Together they explain why one threshold cannot fit all markets.

---

### Slide 10b: Tuning Curve

**File:** `figures/slide10_tuning_curve.png`

**What it shows:**
- Spotify validation-set improvement vs. naive quoting (¢) for tau = 0.1, 0.2, 0.3.
- Chosen threshold: tau = 0.3 (highest validation improvement).
- Lower tau = more sensitive (pull quotes more often). Higher tau = looser (quote more).

**Source:** `holdout_mm_evaluation.py` → `choose_threshold()` restricts validation to the top-3 train thresholds; for Spotify those are 0.1, 0.2, 0.3. Validation deltas: 1,489¢, 1,495¢, 1,505¢. We pick 0.3.

**Q&A:**  
- *“Why Spotify?”* Clean tuning story; the curve is monotonic over the candidate range. MLB/Weather use tau=0.1 (more sensitive) and would show a different curve.  
- *“What if we went to tau=0.5?”* For Spotify, higher tau was not in the top-3 train candidates, so we never evaluated it. The tuning is conservative.

---

### Slide 11: Relative Outperformance

**File:** `figures/slide11_relative_outperformance.png`

**What it shows:**
- Bar chart: Improvement vs. naive quoting (¢) for MLB, Spotify, Weather.
- MLB: +2,436¢, Spotify: +3,231¢, Weather: +18,123¢.
- Election is omitted (improvement was +70¢, visually negligible).

**Source:** `holdout_mm_evaluation.py` → `evaluate_market()` with `include_fees=False`. Test-set sum of `informed_pnl_c − naive_pnl_c` per market.

**Q&A:**  
- *“Why no percentages?”* Baselines vary; some markets are near zero or negative. Percentages would be misleading.  
- *“Why no Election?”* The improvement was small (+70¢); we focus on markets where the effect is meaningful.  
- *“Is this live PnL?”* No. This is a simplified backtest (trade-print fill model, no order book). Use for cautious profitability claims, not realized live PnL.

---

## 3. Quick Reference: Key Assumptions

| Parameter              | Value   | Notes                                                |
|------------------------|---------|------------------------------------------------------|
| Half-spread            | 1¢      | Fixed edge per fill                                  |
| MTM horizon            | 5 min   | Time-based, not trade-count-based                    |
| Decision lag           | 1 trade | Use prior trade’s AS score (no lookahead)           |
| Fees                   | Off     | For presentation (institutional maker assumption)   |
| Train/Val/Test split   | ~60/20/20 | Chronological by event                             |
| Threshold grid         | 0.1–0.9 | Top 3 on train evaluated on validation               |

---

## 4. Reproducing the Figures

From the project root:

```bash
cd tauroi-prediction-engine
uv run python scripts/final_presentation_figures.py
```

This regenerates:
- `figures/slide09_anatomy_of_the_shield.png`
- `figures/slide10_universal_heuristics_table.png`
- `figures/slide10_tuning_curve.png`
- `figures/slide11_relative_outperformance.png`

To recompute the underlying data (e.g., after changing detector or backtest logic):

```bash
uv run python scripts/holdout_mm_evaluation.py --market all --no-fees
```

To tune alpha (γ vs burst weight) globally on validation and compare to fixed α=0.7:

```bash
uv run python scripts/holdout_mm_evaluation.py --tune-alpha --no-fees
```

This sweeps α in the grid, picks the best by summed validation improvement across markets, then reports test improvement for tuned vs baseline (α=0.7). Expect ~15–30 min runtime due to detector passes.

---

## 5. Common Q&A One-Liners

- **“How do you avoid overfitting?”** Chronological holdout: we tune on validation and report only on held-out test events. We also use event-level bootstrap for uncertainty.
- **“Why 5 minutes?”** A reasonable horizon for prediction markets where resolution is typically hours to days. Sensitivity to horizon can be studied separately.
- **“Is the detector real-time?”** The pipeline is designed for real-time use (Kalman + rolling EM + bursts). The backtest replays historical trades with lagged scores.
- **“What about market impact?”** The backtest assumes passive fills at our quoted prices. We do not model market impact or adverse selection from our own quote placement.
- **“Can we see the code?”** Yes. Detector: `src/as_detector.py`. Backtest: `src/mm_backtest.py`. Evaluation: `scripts/holdout_mm_evaluation.py`.
