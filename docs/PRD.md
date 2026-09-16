# Product Requirements Document

**Product:** Tauroi Pulse
**Company:** Tauroi Technologies
**Version:** 1.0
**Status:** Draft for build
**Date:** 2026-09-07
**Owner:** Jeremy
**Scope:** Path A — live adverse-selection intelligence. Not a market maker.

---

## 1. Why this document exists

The current repo is a strong MS&E 108 research system: a lagged logit jump-diffusion detector, a holdout backtest, and ~1.3M Kalshi trades. It is not a product. Nobody can open it, add a market, and get a decision they trust.

**Tauroi Pulse** is the smallest thing that *is* a product: a live toxicity radar for prediction-market desks. It watches Kalshi trade streams, publishes an adverse-selection score, and tells an operator (or an external bot) whether to **stay on**, **widen**, or **pull**.

It does **not** place orders. That is Path B. Shipping Pulse first is how the senior project becomes a portfolio piece you can demo, use, and talk about as production software.

---

## 2. Problem

Market makers on Kalshi get run over by informed flow. Prices jump in log-odds, trades cluster, and a resting 1-cent quote becomes a toxic fill. Most independent or semi-systematic desks have no model for this. They either quote blindly or pull by gut.

Academic tools do not help them. Notebooks, parquet caches, and a `tau` sweep are not operable. A desk needs:

1. A score on the *latest* trade, not a full-sample rerun.
2. A recommended action with the reason attached.
3. A record of every decision, so they can argue with the model after the session.
4. A clear “I do not know” state when the feed is stale.

The research already shows flagged trades have ~1.5–1.7× larger adverse 5-minute moves, and that a lagged pull rule can improve simulated maker PnL on holdout data. The product gap is packaging that signal so a human (or another process) can act on it in time.

---

## 3. Product thesis

> Tauroi Pulse is a **quote-risk overlay** for prediction markets. It does not invent a fair price and it does not manage inventory. It answers one question, continuously: *is the flow in this contract toxic enough that you should stop being the bid/ask?*

That is a real job. It is also a honest job. The backtest fill model is simplified; Pulse must never present simulated PnL as live money.

### Positioning

| | We are | We are not |
|---|---|---|
| Category | Real-time adverse-selection intelligence | A market maker, a broker, or a trading bot |
| Buyer | Person or bot that already quotes (or is about to) | A retail picker looking for “locks” |
| Output | Score + action + audit | Fills, inventory, or guaranteed edge |
| Venue (v1) | Kalshi | Polymarket, sportsbooks, crypto perps |

### One-line pitch (portfolio / landing)

**Tauroi Pulse detects toxic flow on Kalshi in real time and tells a market maker when to pull.**

---

## 4. Users

Pulse has one primary user in v1. Do not design for a sales org.

### Primary: Independent or desk operator (“the quoter”)

- Quotes or intends to quote a small set of Kalshi contracts (weather, sports, entertainment, politics).
- Already has or will have their own order logic / another bot.
- Needs a second opinion that is faster and more consistent than watching the tape.
- Will forgive a slightly late score. Will not forgive a silent stale feed or a lookahead lie.

**Jobs to be done**

1. “Show me which of my markets are hot right now.”
2. “Tell me whether I should be quoting this name.”
3. “When you say pull, show me why — jump, burst, or both.”
4. “After the session, let me replay the flags against what actually happened to price.”

### Secondary: External quoting process

- Consumes Pulse over HTTP / WebSocket.
- Treats `recommended_action` as an input to its own spread or cancel logic.
- Needs a stable, versioned schema and predictable staleness semantics.

### Anti-users (do not build for these in v1)

- Multi-tenant SaaS customers with billing and SSO.
- Retail traders looking for directional calls.
- Compliance / official Kalshi market-maker program reporting.
- Anyone who wants Pulse to submit orders.

---

## 5. Goals and non-goals

### Goals (v1)

1. Run continuously against a user-defined Kalshi watchlist (8–20 contracts).
2. Publish a lagged AS score and a discrete action (`quote` / `widen` / `pull`) per market.
3. Give an operator a watchlist + market-detail UI that stays correct on refresh.
4. Persist an append-only decision log.
5. Fail closed: stale, disconnected, or under-warmed markets recommend `pull`.
6. Support a **replay demo** that does not depend on live market hours (portfolio-critical).
7. Expose a small authenticated API another process can subscribe to.

### Non-goals (v1)

- Placing, amending, or canceling orders.
- Inventory, PnL attribution as live money, or capital allocation.
- Order-book / queue simulation.
- Polymarket or any second venue.
- Multi-user accounts, orgs, or billing.
- Mobile app.
- Auto-tuning `tau` in production (use holdout-selected defaults; allow manual override).
- Replacing the research notebooks. They stay for methodology.

---

## 6. Success metrics

### Product (is this actually usable?)

| Metric | Target for v1 |
|---|---|
| Time-to-first-signal after adding a liquid ticker | < 60s once ≥50 trades are in memory (or replay seed) |
| Score freshness | Decision uses only `as_score` from the **previous** trade or previous completed interval |
| End-to-end lag, live mode | p95 < 2s from Kalshi trade visible to Pulse → UI/API update |
| Stale handling | If no trade or heartbeat in 15s on an active market, UI shows STALE and action becomes `pull` |
| Decision completeness | 100% of actions written to the audit log with inputs |
| Operator comprehension | Untrained teammate can explain a `pull` from the market-detail page in < 30s |
| Uptime of a local/demo session | 4-hour sports or weather window without a restart |

### Honesty (do not regress the research)

- Live and replay use the same detector (`src/as_detector.py`) and the same 1-trade lag rule as Process 3.
- UI never labels holdout MTM as “live PnL.”
- Any paper overlay is titled **counterfactual** or **research overlay**, not profit.

### Portfolio (does this look like shipped software?)

A hiring manager should be able to, in 8 minutes:

1. Open the app in replay mode on a known MLB or Spotify episode.
2. See the score spike, the recommended `pull`, and the jump vs burst breakdown.
3. Click a log row and see the exact inputs.
4. Hit a public `/v1/markets/{ticker}/snapshot` and get JSON that matches the screen.
5. Read a README that states what Pulse is, what it is not, and how to run it.

If those five things work, this is no longer a senior project. It is a product demo with a real engine behind it.

---

## 7. Product principles

1. **Fail closed.** Uncertainty is a pull, not a quote.
2. **No lookahead, ever.** The action at trade `i` may use `as_score[i-1]` only.
3. **Server is source of truth.** The browser renders; it does not compute scores or own watchlist state.
4. **Every action is explainable.** Jump posterior, burst flag, tau, and feed age are visible.
5. **Mode is unmistakable.** Replay / paper / live-read are labeled in the chrome at all times.
6. **Narrow and sharp.** One venue, one job, one primary screen plus one detail screen.

---

## 8. Modes

| Mode | Data | Orders | Who uses it |
|---|---|---|---|
| **Replay** | Bundled parquet episode + recorded decisions | None | Demo, interviews, tests |
| **Paper** | Live Kalshi trades (or demo API) | None. Actions are logged as if a quoter followed them | Daily use, validation |
| **Live-read** | Live Kalshi trades | None. Same as paper; name exists so nobody thinks “live” means “sending orders” | Desk use |

v1 ships Replay + Paper. Live-read is the same code path as Paper with a label change. There is no Live-write in this PRD.

Default demo mode is Replay so the portfolio piece works on a plane.

---

## 9. User journeys

### 9.1 First run (portfolio / local)

1. `docker compose up` or `make dev`.
2. App opens in **Replay** on `KXMLBGAME-25OCT17TORSEA-TOR` (or the best bundled toxic episode).
3. Watchlist already has 4–6 names across MLB / Spotify / weather.
4. Operator clicks the row that is red (`pull`).
5. Detail page shows price, Kalman mid, AS score, γ, burst, and a recommended action.
6. They open the audit log, click the flag, and see `as_score=0.81 > tau=0.10`, `decision_lag=1`.

### 9.2 Daily paper use

1. Operator starts Pulse in Paper with their Kalshi keys on the server.
2. They add `KXHIGHLAX-26FEB08-T72` from the add-market dialog.
3. Pulse backfills recent trades from cache/API, warms EM, then goes live.
4. Score crosses tau; row flips to `pull`; optional desktop/browser notification.
5. After the session they filter the log to `pull` and export CSV.

### 9.3 External bot

1. Bot authenticates with a Pulse API token.
2. Subscribes to `ws://.../v1/stream`.
3. On `recommended_action=pull` it cancels its own quotes (bot’s job, not Pulse’s).
4. On `feed_status=stale` it treats that as pull.

---

## 10. Functional requirements

### 10.1 Watchlist

- Persist a watchlist of Kalshi tickers (max 20 in v1).
- Add by ticker search (Kalshi `GET /markets` or a local catalog of recently cached names).
- Remove, pin, and reorder.
- Per-market config: `tau`, `alpha` (optional override), family tag (spotify / weather / mlb / election / other).
- Default `tau` by family from Process 3 holdout: MLB/Weather `0.1`, Spotify `0.3`, Election `0.2`. Operator can override; override is logged.

### 10.2 Ingest

- Paper/live-read: incremental trade fetch via existing `belief_data.fetch_hf_data` / `KalshiClient` (poll ≤1s per active ticker, or batched).
- Replay: read a frozen parquet episode at 1× or 4× speed; pausable.
- Normalize to `timestamp`, `mid_price`, `volume`, `logit`.
- Ignore / mark markets with `< 50` trades as `warming`.

### 10.3 Scoring

- Use `run_as_detection` (or an incremental wrapper that matches it) with Process 3 defaults: `em_window=200`, `em_iterations=8`, `alpha=0.7`, `gamma_threshold=0.6`, `burst_multiplier=3.0`, `min_jump_logit=0.15`.
- Emit per ticker, on each new trade:
  - `as_score`, `gamma`, `burst_flag`, `arrival_rate_ratio`
  - `sigma_b`, `lambda`, `mid`, `logit`
  - `feed_age_ms`, `n_trades`, `warming`
- **Decision lag:** `decision_as_score = previous trade’s as_score`. If only one trade exists, action is `pull` (warming).

### 10.4 Action policy

Discrete action from the lagged score. Keep it simple enough to explain in an interview.

| Condition | Action | Meaning to the quoter |
|---|---|---|
| warming, stale, error, or disconnected | `pull` | Do not quote |
| `decision_as_score > tau` | `pull` | Toxic; leave the book |
| `0.5 * tau < decision_as_score ≤ tau` | `widen` | Elevated; increase spread / cut size |
| `decision_as_score ≤ 0.5 * tau` | `quote` | Clean enough to stay on |

`widen` is advisory in Path A. Pulse does not compute a new spread. It may attach `spread_hint = belief_model.spread_multiplier()` as optional metadata; the UI shows it as a hint, not a live quote.

### 10.5 Audit log

Append-only store (SQLite is enough for v1):

- `id`, `ts`, `mode`, `ticker`
- `mid`, `as_score`, `decision_as_score`, `gamma`, `burst_flag`
- `tau`, `action`, `reason` (short string)
- `feed_age_ms`, `n_trades`

Queryable by ticker, action, and time. Export CSV. Never update a row; corrections are new rows.

### 10.6 Paper overlay (optional panel, not the hero)

A counterfactual: “if you had been a 1-cent naive maker and followed Pulse pulls, here is research-style spread + 5-min MTM on *today’s paper session*.”

Rules:

- Same engine as `mm_backtest.py`.
- Banner: **Research overlay — not live PnL. Trade-print fill model, no order book.**
- Hidden behind a toggle. Default off in Live-read/Paper chrome; on in Replay if it helps the story.
- Do not put a big dollar number on the watchlist.

### 10.7 Health

- Process heartbeat.
- Per-market last trade time, last score time, last error.
- Global banner if Kalshi auth fails.
- Kill switch: **Pause all recommendations** (forces every market to `pull` and freezes new `quote`/`widen`). This is a Pulse kill switch, not an exchange cancel-all.

---

## 11. UI requirements

v1 is a single desktop web app. One operator. Dark, dense, trading-tool — not a marketing site.

### Chrome (always visible)

- Product name + **mode badge** (Replay / Paper / Live-read)
- Connection / feed health
- Kill switch
- Clock + “data as of”

### Screen A — Watchlist (home)

One row per market:

- Ticker, family, last mid, last trade age
- AS score as a compact bar (0–1)
- Action chip: `QUOTE` / `WIDEN` / `PULL` / `WARMING` / `STALE`
- γ and burst dots
- Configured tau

Sort: action severity, then score. Click row → Screen B.

Empty state: “Add a Kalshi ticker or load the MLB replay.”

### Screen B — Market detail

Shared time axis:

1. Mid price + Kalman-filtered implied mid
2. AS score with tau and 0.5·tau guides
3. Burst / jump markers
4. Volume or inter-trade time (small)

Side panel:

- Current action + one-sentence reason
- Latest γ, burst ratio, σ_b, λ
- Config: tau slider (logs the change)
- Last 20 decisions for this ticker

### Screen C — Audit

Table of the log. Filters: ticker, action, time. Click → jumps to the trade on Screen B (replay) or a snapshot inspector (paper).

### Screen D — Settings (minimal)

- Mode
- Kalshi credential status (never display the PEM)
- Poll interval
- Notification toggle
- “Load replay episode”

### Reliability (this is what makes the UI a product)

- Refresh reconstructs the same watchlist, scores, and log from the server.
- WebSocket with REST snapshot fallback every 5s.
- Reconnect banner; no blank charts on drop.
- STALE and WARMING are first-class visuals, not missing data.
- Every mutating control (add market, change tau, kill switch) is idempotent and logged.
- No score computation in the browser.

### What not to build in the UI

- Order entry.
- A fake order book you are not actually on.
- A portfolio of cash and positions.
- User admin.

---

## 12. API (v1)

Base: `/v1`. Auth: bearer token from server env (`PULSE_API_TOKEN`). JSON, UTC timestamps.

### `GET /v1/health`

```json
{
  "status": "ok",
  "mode": "replay",
  "kill_switch": false,
  "kalshi_auth": "skipped",
  "watchlist_size": 6,
  "server_time": "2026-09-07T18:00:00Z"
}
```

### `GET /v1/markets`

Watchlist snapshots. Each item:

```json
{
  "ticker": "KXMLBGAME-25OCT17TORSEA-TOR",
  "family": "mlb",
  "mid": 0.62,
  "as_score": 0.81,
  "decision_as_score": 0.74,
  "gamma": 0.91,
  "burst_flag": true,
  "sigma_b": 0.22,
  "lambda": 0.04,
  "tau": 0.1,
  "action": "pull",
  "reason": "decision_as_score 0.74 > tau 0.10; jump+burst",
  "feed_status": "live",
  "feed_age_ms": 420,
  "n_trades": 1840,
  "updated_at": "2026-09-07T18:00:01.200Z"
}
```

### `GET /v1/markets/{ticker}`

Snapshot + last N points for the detail chart (`points` capped at 2,000).

### `POST /v1/markets`

`{ "ticker": "...", "tau": 0.1 }` — add to watchlist, start warm-up.

### `PATCH /v1/markets/{ticker}`

Update `tau` or pin.

### `DELETE /v1/markets/{ticker}`

### `GET /v1/decisions`

Query: `ticker`, `action`, `since`, `limit`.

### `POST /v1/control/kill`

`{ "enabled": true }` — pause recommendations.

### `WS /v1/stream`

Pushes `market.updated` and `decision.append` events with the same payloads as REST.

**Compatibility rule:** field names stay stable. Additive changes only. Bump to `/v2` if the action policy changes meaning.

---

## 13. Architecture (v1, one box)

```
Kalshi REST  ─┐
Replay parquet┤── Ingest ── Score worker ── State store ── API
              ┘         (lagged AS)      (SQLite+memory)    │
                                                            ├── Web UI
                                                            └── External bot
```

| Component | Responsibility | Build on |
|---|---|---|
| Ingest | Poll or replay trades, normalize, cache parquet | `belief_data.py`, `kalshi_client.py` |
| Score worker | Per-ticker detector, lag, action policy | `as_detector.py` |
| State store | Latest snapshot + audit log | SQLite + in-memory last snapshot |
| API | REST + WS | FastAPI (new) |
| UI | Watchlist, detail, audit | Simple SPA (React or HTMX+server charts). Pick one and stay. |

**Process model:** one Python process is acceptable for v1 (API + background asyncio tasks). Split workers only if polling 20 markets blocks scoring.

**Incremental scoring:** v1 may re-run the detector on a trailing window (e.g. last 2,000 trades) on each new print if p95 stays under 2s. An online Kalman/EM is a v1.1 performance task, not a launch blocker, *provided* the windowed rerun uses the same lag rule and does not peek at the current trade for the action.

**Deploy:** `docker compose` with two services (app + optional static UI) and a volume for SQLite + cache. `.env` for Kalshi + `PULSE_API_TOKEN`. No cloud vendor required for v1.

---

## 14. Non-functional requirements

| Area | Requirement |
|---|---|
| Latency | p95 decision publish < 2s after trade ingest in Paper; Replay is clock-driven |
| Scale | 20 markets, ~few trades/sec aggregate |
| Persistence | Audit log survives restart; snapshots rebuild from last window of trades |
| Security | Secrets only on server; UI never sees PEM; token on all mutating routes |
| Observability | Structured logs: ingest, score, action, errors; `/v1/health` |
| Testing | Unit tests keep existing detector tests. Add: lag invariant, stale→pull, kill switch, API snapshot schema, replay determinism |
| Reproducibility | Replay of a bundled episode produces the same action sequence (golden file) |

---

## 15. Data and model constraints (carry forward from the research)

These are product rules, not footnotes.

1. **Process 3 is the brain.** Do not reintroduce same-trade lookahead (Process 1/2).
2. **Family taus** start from holdout, not from a live session.
3. **Warm-up:** `< 50` trades → no `quote`.
4. **Tick noise:** keep `min_jump_logit=0.15` unless a new holdout says otherwise.
5. **Claims language** in the UI and README:
   - Allowed: “flagged prints showed larger adverse 5-minute moves in holdout.”
   - Forbidden: “Pulse made $X live” or “Pulse is profitable.”

---

## 16. Scope by phase

### Phase 0 — Productize the core (about 3–5 days)

- FastAPI skeleton: health, snapshot, decisions.
- Score worker on cached parquet (no live poll yet).
- Replay clock over one bundled episode.
- Golden-file test: lagged actions match a frozen JSONL.
- README rewritten as a product + research repo.

*Exit:* `curl` a snapshot; replay is deterministic.

### Phase 1 — Operator UI (about 5–7 days)

- Watchlist + detail + audit.
- WebSocket updates.
- Mode badge, stale/warming, kill switch.
- Add/remove ticker against cache.

*Exit:* 8-minute demo journey in §6 works on Replay.

### Phase 2 — Paper / live-read (about 4–6 days)

- Incremental Kalshi poll on the watchlist.
- Warm from cache then tail.
- Auth failure and stale banners.
- CSV export.
- Optional notifications.

*Exit:* 1-hour paper session on 4 live or recently active markets without a crash; log is coherent.

### Phase 3 — Portfolio harden (about 2–3 days)

- Docker compose.
- Seed replay data checked in (or download script + checksum).
- Architecture section in README, screenshots, sample API responses.
- Paper-overlay toggle with the honesty banner.
- One recorded walkthrough (video or GIF).

**v1 launch = end of Phase 3.** That is a viable portfolio piece and a tool you can actually leave running.

### Explicitly later (not in v1)

- Online EM (if windowed rerun is too slow).
- Slack / webhook outbound.
- Auto tau retune on a schedule.
- Polymarket.
- Path B quoting engine.

---

## 17. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Batch `run_as_detection` is too slow live | UI feels lagged; people distrust it | Windowed rerun + lag rule; profile before Phase 2; online EM only if needed |
| Kalshi rate limits | Watchlist of 20 × 1s poll dies | Batch trades endpoint; backoff; fewer markets |
| Quiet markets | Score frozen, looks “broken” | `feed_age_ms` + STALE vs QUIET copy (“no prints, last score still valid until T”) |
| Demo depends on market hours | Portfolio piece fails in an interview | Replay is the default public mode |
| Overlay PnL is misunderstood | Integrity / interview risk | Toggle + banner + never on watchlist |
| Scope creep into OMS | Never ships | This PRD’s non-goals are binding |

---

## 18. Open questions (resolve during Phase 0, not during coding of the UI)

1. **SPA vs server-rendered.** Recommendation: FastAPI + a small React Vite app if you want portfolio polish; HTMX if you want to finish Phase 1 faster. Pick before Phase 1 starts.
2. **Which bundled replay episode?** Prefer one MLB game with a visually obvious toxic burst (already used in `lie_detector_chart`). Confirm the file is redistributable (your own Kalshi cache).
3. **Quiet vs stale.** Proposal: if the market is open and `feed_age > 15s`, `stale` → `pull`. If the market is closed / settled, `closed` → no action, gray row.
4. **Public GitHub.** Replay mode should run without Kalshi keys. Paper mode requires keys. Keep secrets out of the repo (already true).

---

## 19. What “done” looks like

Pulse is v1-complete when all of the following are true:

1. A stranger can clone, `docker compose up`, and see Replay without editing code.
2. Watchlist, detail, and audit agree with `/v1/markets` after a refresh.
3. Actions are lagged and covered by a golden-file test.
4. Stale and kill switch both force `pull`.
5. README states Path A vs Path B in one paragraph.
6. You can do the 8-minute hiring-manager demo without a notebook.

At that point the senior project is the *engine*. Pulse is the *product*.

---

## 20. Appendix — mapping from today’s repo

| Existing asset | Pulse use |
|---|---|
| `src/as_detector.py` | Score worker (do not fork a second detector) |
| `src/mm_backtest.py` | Optional paper overlay only |
| `src/belief_data.py` | Ingest + cache |
| `src/kalshi_client.py` | Paper/live-read transport; order methods stay unused |
| `src/belief_model.py` `spread_multiplier` | Optional hint on `widen` |
| `scripts/holdout_mm_evaluation.py` | Source of default family taus; not a runtime |
| `cache/kalshi_hf_*` | Warm-up + replay seeds |
| Notebooks | Research appendix, not the app |

---

*Tauroi Pulse — Path A PRD. Quoting and inventory are out of scope until a separate Path B PRD.*
