# B3 — Hermes Integration Design: Football Prediction + Paper-Betting Bot ("Punter")

**Author:** Hermes subagent (DeepSeek v4 flash) · **Date:** 6 Oct 2026 · **For:** Rayan
**Deliverable:** a concrete Hermes architecture for a football (soccer) prediction + paper-betting desk,
designed to plug into the existing Iris/uni/dev/hardware/reviewer install without disturbing it.
**Reuses:** the `trader`/`budget` pattern in `H1-hermes-integration.md` (profile + cron + script gate +
Kanban) and the evidence-based "rules decide, LLM explains; limits in code" architecture in
`T2-ai-trading-evidence.md` §5.

Citations are inline. Anything I could not verify against a doc/source page this session is marked **UNVERIFIED**.
Timezone arithmetic below was computed live (Node ICU) on 6 Oct 2026, not estimated.

> **Two bets this design makes explicit.** (1) The probability maths is deterministic and never done by an LLM
> — Dixon-Coles + a margin-removal + a code-enforced Kelly stake. (2) The success metric is **closing-line
> value (CLV)**, not short-run ROI — a betting edge shows up in CLV months before it shows up in profit, and
> ROI over <500 bets is noise. This is the betting-domain version of T2's "benchmark honestly, don't be fooled
> by short windows".

---

## 0. One-page summary

| | **Punter (football prediction + paper betting)** |
|---|---|
| New profile | `punter` — clone of `uni`'s shape (DeepSeek), dedicated secrets + Slack channel |
| Leagues v1 | EPL, Championship (UK sharpest/free data); La Liga, Serie A, Bundesliga, Ligue 1; **A-League** (local, same-timezone) |
| Model | **Dixon-Coles** (time-decayed Poisson + low-score correction) via [`penaltyblog`](https://pypi.org/project/penaltyblog/) v1.12.0; Elo/Pi rating as cross-check |
| Market data | **football-data.co.uk** (free history + closing odds) · **The Odds API** (live book prices) · **Betfair Exchange API** (true closing price → CLV benchmark) · **API-Football** (lineups/injuries) |
| Value logic | `edge = p_model·odds − 1`; only bet if model beats the **de-margined sharp price** and the soft-book price clears a min edge |
| Staking | **fractional Kelly (¼)**, clamped in code (max stake %, max daily exposure, min/max odds) |
| Zero-LLM share | **~92%** — ingest, fit, predict, de-margin, edge, Kelly stake, ledger, settle, CLV, dashboard |
| LLM share | team-news/lineup summary, match preview, weekly narrative, **bounded** news adjustment (A/B only) |
| Main model | DeepSeek `deepseek-flash` for LLM jobs; free local `gemma4` for ad-hoc "explain this pick" |
| Schedule (Sydney) | Weekend kickoff windows **Fri 22:00 → Mon 08:00 AEDT** + midweek UCL; a 15-min **gate tick** decides when to act (DST-proof) |
| Delivery | Slack **`#picks`** (new channel) |
| Hard guardrails | **paper-only — no bet-placement API exists in v1**; stakes clamped in code; keys in `punter` profile `.env`; `state/KILL`; approvals `cron_mode: deny` |
| Est. cost | **$0–$33/mo** data (free tiers likely enough; $30 only if you want historical-odds CLV backtests) + **~$1–2/mo** DeepSeek |
| Build | 4 weeks, Kanban cards for `dev`/`reviewer` (§13) |
| Real-money gate | only after **≥500 settled paper bets** with **statistically positive mean CLV** and acceptable calibration (§14) |

---

## 1. Baseline: what `punter` plugs into

From `H1-hermes-integration.md` and the live audit:

- Profiles: `default` = **Iris** (local gemma, routes work), `uni` (DeepSeek), `dev`/`hardware`/`reviewer` (Claude).
- Slack gateway on `default` (Socket Mode), home channel `#daily-summary`; cron delivers per-job to Slack.
- Cron with **script pre-run gates** (`{"wakeAgent": false|true}`), `--no-agent` zero-LLM jobs, per-job model
  pinning, `--workdir` to root file/terminal in the project dir.
- Kanban wired (`max_in_progress 3`, 30 s dispatch); `bot-routing` skill for Iris hand-offs.
- Multiplexed gateway serves every profile at rest for **$0**.

**Implication:** a fourth worker profile costs nothing at rest. The only recurring spend is LLM tokens on the
few gates that actually fire.

---

## 2. Profile choice — a new `punter` profile (not `trader`, not `uni`)

**Recommendation: a dedicated `punter` profile.** Reasons, all consistent with the `H1` "two profiles not one"
logic:

1. **Secret isolation.** It holds betting-data keys (`ODDS_API_KEY`, `APIFOOTBALL_KEY`, later Betfair keys)
   that the trading/budget profiles must never see. Per-profile `.env` and `terminal.env_passthrough`
   ([secrets](https://hermes-agent.nousresearch.com/docs/user-guide/secrets)) give that boundary for free.
2. **Channel isolation.** Picks land in **`#picks`**, not `#daily-summary` — a wrong/case-mismatched Slack
   target is *silently dropped*
   ([Slack](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/slack)), so keep the target in one place.
3. **Its own workdir** (`C:/Users/Rayan/projects/punter/`) with an `AGENTS.md`, so every cron job inherits the
   desk rules via `--workdir` ([cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)).
4. **Cheap + local front door.** Keep the interactive face on local `gemma4` for "explain this pick" Q&A (free),
   and DeepSeek for the scheduled narrative work — the exact split `H1` uses for `budget`.

```bash
# Clone the cheap DeepSeek shape of `uni` (config/.env/SOUL/skills), NOT history/cron
hermes profile create punter --clone-from uni --description "Football modelling & paper-betting desk: Dixon-Coles picks, Kelly paper ledger, CLV tracking."

# Pin models (verify the exact provider label with `hermes model` inside the profile first)
punter config set model.default gemma4-64k          # interactive Q&A face = local, free
punter config set model.provider custom
punter config set model.base_url http://localhost:11434/v1
punter config set terminal.env_passthrough '["ODDS_API_KEY","APIFOOTBALL_KEY"]'

# Dedicated Slack channel #picks, then: /invite @Hermes Agent   (in Slack)
# Register as a routable bot for Iris (same pattern as uni/dev, from C-bots-profiles.md)
hermes -p punter chat -c "Bot Chat" --create-if-missing
```

Notes:
- `--clone` copies `config.yaml`, `.env`, `SOUL.md`, skills — **create the profile before adding betting keys**,
  then add each key only to `punter`'s `.env` (a cloned `.env` would seed it).
- Put the profile's untrusted-content rule in `SOUL.md` (§9.6).
- **UNVERIFIED** which provider label `hermes cron edit --provider` expects for the local model — read it off
  `hermes model` inside `punter` before pinning any cron job (same caveat as `H1` §2).

---

## 3. Data layer — sources (all checked 6 Oct 2026)

| Source | Use | Cost (verified) | Notes |
|---|---|---|---|
| **football-data.co.uk** | Free CSV history (results + stats + odds incl. **Betfair `BFH/BFD/BFA`** and **closing odds** with `C` suffix, e.g. `B365CH`) | **Free** ([notes.txt](https://www.football-data.co.uk/notes.txt), [England files](https://www.football-data.co.uk/englandm.php)) | The backbone for fitting + honest backtests. England files last updated 24/05/26; multiple seasons available. |
| **The Odds API** | Live/pre-match bookmaker prices for the *soft-book opportunity* and the *sharp reference* | **Free: 500 credits/mo**; 20K = **$30/mo**, 100K = **$59/mo** ([pricing](https://the-odds-api.com/)) | Covers EPL + European + **Aussie books (Sportsbet, TAB, Neds, Ladbrokes)** and **Pinnacle + Betfair** as sharpers ([docs](https://the-odds-api.com/liveapi/guides/v4/)). Historical snapshots (5-min since Sep 2022) are **paid-only** and cost **10 credits/region/market** ([historical](https://the-odds-api.com/historical-odds-data/)). |
| **Betfair Exchange API** | The **true market price** — the CLV benchmark | App key free; historical/streamed data may cost (**UNVERIFIED** — the historical-data intro page failed to load this session) | Betting API (nav, odds/volumes, bet placement) + **Exchange Stream API** for low-latency market data ([Exchange API](https://developer.betfair.com/exchange-api/)). |
| **API-Football** (api-sports) | Fixtures, **lineups**, injuries, in-play/pre-match odds | **Free 100 req/day** incl. line-ups + injuries; Pro **$19/mo** (7,500/day) ([pricing](https://www.api-football.com/pricing)) | Feeds the "lineups ~1h before KO" gate and the LLM team-news summariser. |
| **penaltyblog scrapers** (Understat / Club Elo / FBref) | xG + club-Elo inputs | Free | Ships with `penaltyblog`; xG-based ratings are a **v1.1** upgrade — v1 uses goals + Elo. |
| **Betfair / The Odds API** as *prices* | — | — | Do **not** scrape bookmaker HTML; API-only keeps it deterministic and ToS-clean. |

**Not recommended for v1:** API-Football's own "prediction" endpoint (don't blend an opaque third-party model
into a model you're trying to evaluate); paid in-play feeds (paper desk doesn't need them).

---

## 4. Model per step

**Deterministic core (owns the numbers)** — a `punter-pipeline` Python package in the workdir:

1. **Ingest** (`ingest.py`, zero-LLM): pull the latest `football-data.co.uk` CSVs per league/season, plus
   current-season fixtures + odds from The Odds API; normalise team names to a canonical map
   (`config/teams.json`); insert into SQLite.
2. **Fit** (`fit.py`, zero-LLM): **Dixon-Coles** on ~3–5 seasons with exponential time-decay, per league
   (attack/defence/home-advantage/`rho` params), via `penaltyblog.models`
   ([penaltyblog](https://pypi.org/project/penaltyblog/)). Fit a **Pi/Elo rating** on the same history as an
   independent second opinion (`penaltyblog` ratings); a large divergence flags a data/edge-case, not signal.
3. **Predict** (`predict.py`, zero-LLM): build the scoreline matrix → 1X2, Over/Under, BTTS, Asian handicap
   probabilities.
4. **De-margin** (zero-LLM): remove the bookmaker overround from Pinnacle/Exchange prices to get a **fair
   probability** (`penaltyblog` implied-odds removal — proportional or Shin). This is the "market's true read".
5. **Value** (`value.py`, zero-LLM): for each outcome, `edge_model = p_model·odds − 1` against the soft book,
   and `edge_shift = p_model − p_fair` against the sharp price. **A pick qualifies only if both are positive
   above thresholds** — i.e. the model disagrees with the *sharp* market in the same direction the soft book
   is offering value. This guards against "beating" only a stale soft price.
6. **Stake** (`stake.py`, zero-LLM): fractional Kelly (§7), clamped by `config/limits.json`.
7. **Ledger** (`ledger.py`, zero-LLM): write the paper bet with the price taken and a pending CLV record.
8. **Settle** (`settle.py`, zero-LLM): results from football-data.co.uk / API-Football → P&L, ledger update.
9. **CLV** (`clv.py`, zero-LLM): compare bet price to the **closing sharp price** (§10); the headline metric.

**LLM layer (explains, never sizes)** — DeepSeek, gated:
- **Team-news summariser**: reads injuries/lineups (API-Football) + a couple of web previews → 3–5 bullet
  factual summary in `state/teamnews.md`. No numbers invented (T2 / `percheskowsky` discipline: *all numbers
  computed by script*).
- **Match preview / pick explainer**: renders the pipeline's numbers into readable prose.
- **Weekly review narrative**: reads the ledger + CLV, writes the story + rule suggestions.
- **Bounded news adjustment** (A/B arm only, §11): may shift λ (expected goals) by **≤ ±15%** or veto a pick;
  it can never change stake size or place anything.

**Model-choice rationale (grounded in T2):** FINSABER's bias-mitigated conclusion is to prefer a simple,
regime-aware deterministic core over framework complexity, and the "analyst-not-trader" split is the pattern
that survives; the LLM owns narration, not the arithmetic. Dixon-Coles is the canonical, decades-old
low-parameter football model — few parameters, no overfitting machinery to hide behind
(Dixon & Coles 1997; implemented and documented in `penaltyblog`; the specific Wikipedia page failed to load
this session — **UNVERIFIED** the exact model write-up, but the implementation and its API are confirmed).

---

## 5. Zero-LLM vs LLM split (~92% / 8%)

| Step | Runs as | Cost |
|---|---|---|
| Ingest, canonicalise, SQLite writes | `no_agent` script | $0 |
| Fit Dixon-Coles + Elo, predict, de-margin, edge, Kelly, ledger, settle, CLV | `no_agent` scripts | $0 |
| **Gate** ("is a kickoff within X min? is there an edge?") | pre-run `script` printing `{"wakeAgent": false\|true}` | $0 |
| Team-news summary (only when gate fires) | DeepSeek `deepseek-flash` | ~$0.001/call |
| Preview / A-B bounded adjustment | DeepSeek | small |
| Weekly review narrative | DeepSeek | one call/week |
| Dashboard render + calibration plot | `no_agent` script (matplotlib) | $0 |

Mechanism, verbatim from Hermes docs: `no_agent=True` (`--no-agent --script x.py`) → the script's stdout **is**
the delivered message; exit 0 + empty stdout = silent tick
([cron-script-only](https://hermes-agent.nousresearch.com/docs/guides/cron-script-only)). The **`wakeAgent`
gate** is the cost lever: a pre-run `script` prints `{"wakeAgent": false}` as the **last stdout line** to skip
the LLM, or `{"wakeAgent": true, "context": {...}}` to wake it with data attached
([cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)). On a quiet tick the desk costs
**$0**.

Scripts live under `%LOCALAPPDATA%\hermes\profiles\punter\scripts\` (traversal escapes rejected). **`python3`
is not on PATH on this host** (`python` 3.14.7 is) — use `.py` scripts and let the scheduler pick the
interpreter, or pin a `uv` venv with `--interpreter` (penaltyblog needs pandas/scipy; build it in week 1).

---

## 6. Cron schedule (Sydney time) — kickoff-window aware, DST-proof

**The timezone trap.** Sydney is AEDT (UTC+11) now; the UK is BST (UTC+1) until **25 Oct 2026**, then GMT
(UTC+0). So every European kickoff shifts **+1 h** in Sydney on 25 Oct — exactly the failure mode `H1` §8 flags
for the US market. **Fix: never schedule off a fixed clock for the actual action.** Schedule a frequent **gate
tick** whose script reads real kickoff timestamps from the fixtures table (The Odds API returns `commence_time`
in **UTC**) and decides whether anything is due. The clock only decides *when we look*.

Verified mappings (Node ICU, this session):

| Fixture slot (UK) | Sydney **before 25 Oct** (AEDT, UK=BST) | Sydney **after 25 Oct** (AEDT, UK=GMT) |
|---|---|---|
| Sat 12:30 | **Sat 22:30** | **Sat 23:30** |
| Sat 15:00 (the classic) | **Sun 01:00** | **Sun 02:00** |
| Sat 17:30 | Sun 03:30 | Sun 04:30 |
| Sun 14:00 | Mon 01:00 | Mon 02:00 |
| Sun 16:30 | Mon 03:30 | Mon 04:30 |
| UCL midweek 20:00 | Wed/Thu 07:00 | Wed/Thu 08:00 |
| **A-League** Sat 19:35 local | **Sat 19:35 AEDT** (same tz — no conversion) | same |

| # | Job | Type | Cron (AEDT) | Delivery |
|---|---|---|---|---|
| 1 | `punter-ingest` — pull fixtures + odds + refresh ratings data | `no_agent` script | `0 20 * * 5` (Fri 20:00) + `0 20 * * 2-3` (midweek) | silent unless error |
| 2 | `punter-build` — fit model, project weekend fixtures, write candidate picks (pre-lineup) | gate script; wakes DeepSeek **only** for a preview | `30 20 * * 5` | `#picks` digest, or silent |
| 3 | **`punter-tick`** — every 15 min across the kickoff window; reads `fixtures` for KOs within 65 min / 7 min | gate script (zero-LLM default) | `*/15 22-23 * * 5-6` and `*/15 0-8 * * 6-1` | wakes #4 / #5 |
| 4 | `punter-lineup` — **T-60 min**: fetch confirmed lineups/injuries; DeepSeek writes `teamnews.md`; re-run the edge gate with optional bounded adjustment | DeepSeek, gated by #3 | triggered by #3 | `#picks` "lineups in / final pick" |
| 5 | `punter-snapshot` — **T-5 min**: record pre-kickoff prices (CLV "closing" proxy) | `no_agent` script | triggered by #3 | silent |
| 6 | `punter-settle` — after each window: pull results, settle paper bets, update CLV | `no_agent` script | `0 9 * * 0,1,4` | `#picks` settlement line |
| 7 | `punter-daily-metrics` — refresh dashboard + calibration plot; alert on CLV/ROI/drawdown thresholds | `no_agent` script | `30 9 * * *` | silent unless threshold |
| 8 | `punter-weekly-review` — narrative over the week's ledger + CLV + A/B arms | DeepSeek | `0 18 * * 0` (Sun 18:00) | `#picks` + `reports/YYYY-WW.md` |

```bash
hermes cron create "*/15 22-23 * * 5-6" --no-agent --script punter_tick.py \
  --deliver "slack:#picks" --name punter-tick --workdir "C:/Users/Rayan/projects/punter"
hermes cron create "0 18 * * 0" "Read the week's ledger + clv.csv (attached by the pre-run script). Write the weekly review: CLV trend, ROI vs benchmark, calibration, A/B result, one rule suggestion. Follow the punter-ops skill." \
  --script punter_weekly_context.py --skill punter-ops --deliver slack --name punter-weekly-review \
  --workdir "C:/Users/Rayan/projects/punter"
hermes cron edit punter-weekly-review --provider deepseek --model deepseek-flash --reasoning-effort high
```

Use **5-field** cron expressions — a misformatted schedule silently defaults to one-shot
([cron-troubleshooting](https://hermes-agent.nousresearch.com/docs/guides/cron-troubleshooting)). Cron runs
with `cronjob`/`messaging`/`clarify` toolsets disabled and approvals default to **deny**, so the tick gate
must be pure script.

---

## 7. Value detection & staking (the money logic, all in code)

**Value gate** (qualifies a pick):
```
p_fair   = de_margin(Pinnacle/Exchange decimal odds)      # sharp market's true probability
p_model  = dixon_coles_prob                            # our model
edge_shift = p_model − p_fair                          # must be > +min_shift (e.g. 0.02)
edge_price = p_model · soft_book_odds − 1              # must be > +min_edge  (e.g. 0.03)
qualifies  = edge_shift > min_shift AND edge_price > min_edge
```

**Stake** (`stake.py`, reads `config/limits.json`; the LLM never sees or sets this):
```
f_kelly = (b·p_model − q) / b        # b = odds−1, q = 1−p_model
stake   = clamp( kelly_fraction · f_kelly · bankroll , 0, max_stake_pct·bankroll )
```
- `kelly_fraction` default **0.25** (fractional Kelly cuts variance; full Kelly is reckless with a
  mis-estimated `p`).
- Hard clamps: `max_stake_pct` (2%), `max_daily_exposure_pct`, `max_bets_per_window`, `min_odds` (1.5),
  `max_odds` (6.0 — longshots are where model error bites hardest), `min_edge`.
- Bankroll is **virtual** (paper). Ledger stores stake as a % so a future real bankroll maps proportionally.

**Why fractional + capped:** T2/DXRG show volatility-blind sizing is the classic failure mode; the fix is a
deterministic clamp the model cannot talk its way past, and expected-edge-cost gating so we don't chase
marginal picks into fees.

---

## 8. File / database / ledger layout

Everything under `C:/Users/Rayan/projects/punter/`:

```
AGENTS.md                     # desk rules; loaded by every --workdir cron job
config/limits.json            # staking caps + thresholds (single source of truth; read by stake.py)
config/model.json             # fitted Dixon-Coles params snapshot (versioned each fit)
config/teams.json             # canonical team-name map across sources
data/
  football-data/<league>/<season>.csv     # raw, immutable downloads
  odds/<ts>.json                          # raw The Odds API responses (audit trail)
db/punter.sqlite              # PRIMARY ledger (queryable -> dashboard)
  tables: fixtures, odds_raw, model_probs, picks, bets, results, clv, llm_runs, ab_arms
ledger/bets.csv               # flat export of `bets` (git-diffable, human-readable)
journal/YYYY-MM-DD.jsonl      # append-only audit: every decision, stake, LLM prompt+response, rejects
reports/YYYY-WW.md            # weekly narrative
dash/dashboard.html           # rendered evaluation dashboard (§10)
dash/calibration.png          # reliability diagram
state/teamnews.md             # LLM team-news summary (overwritten per match)
state/signals.json            # current gate context (overwritten)
state/KILL                    # if present, stake.py refuses to write any bet
```

**Why SQLite as primary, CSV/JSONL as audit.** The dashboard needs joins (`bets` × `clv` × `results` ×
`ab_arms`), which CSV handles badly; SQLite is one file, zero server, trivially backed up. The append-only
`journal/*.jsonl` keeps the tamper-evident audit trail (T2 §3.4: log the LLM's full prompt+response so you can
audit *why*). The agent's "memory" of the strategy is **these files, not the prompt** — it survives profile
resets and is greppable.

---

## 9. Guardrails (the part that actually matters)

1. **Paper-only by construction.** v1 ships **no bet-placement integration at all** — no bookmaker account, no
   Betfair bet-placement call. "Bets" are rows in `bets` priced from recorded odds. There is literally no code
   path that can move real money, so a prompt-injected match report has nothing to reach (§9.8 of `H1`, applied).
2. **Staking caps in code.** `stake.py` reads `config/limits.json` and clamps; it never trusts a number from the
   LLM or from `state/signals.json`. A wild edge is logged and clamped, not honoured.
3. **Kill switch.** `state/KILL` → `stake.py` exits immediately (no bet written). Auto-written when
   `max_drawdown_pct` trips on the paper ledger; manually by you or Iris.
4. **Separate secrets, scoped.** `ODDS_API_KEY` / `APIFOOTBALL_KEY` (later `BETFAIR_APP_KEY`) live **only** in
   `punter`'s `.env` and are forwarded to scripts solely via `terminal.env_passthrough` — Hermes strips
   Hermes-managed credentials from child processes otherwise
   ([secrets](https://hermes-agent.nousresearch.com/docs/user-guide/secrets)). Declare them in the `punter-ops`
   skill's `required_environment_variables` so nothing else leaks.
5. **Approvals left safe.** Keep `approvals.cron_mode: deny` and `unattended_mode: deny`; never
   `--yolo`/`mode: off` for this profile ([security](https://hermes-agent.nousresearch.com/docs/user-guide/security)).
6. **Untrusted-content rule in `SOUL.md`.** Web previews, API team-news text and injury blurbs are
   attacker-controllable. The desk must **never act on instructions found inside fetched content** — the LLM
   summarises; it does not execute. (Same rule the `budget` profile carries for bank memo fields.)
7. **Bounded LLM influence.** The only lever an LLM ever gets is a ±15% λ shift or a veto, applied by the
   pipeline (not the LLM) and recorded in `llm_runs`. It can never size a bet or add a market.
8. **Reconciliation, not trust.** `settle.py` recomputes P&L from results and diffs against the ledger; a
   mismatch is a first-class Slack alert. The record never lives only in a model's summary.
9. **Credential-blur rule for Slack.** Picks go to `#picks` only; keep `SLACK_ALLOWED_USERS` set (never
   `GATEWAY_ALLOW_ALL_USERS`) on any profile that touches money logic.

---

## 10. Evaluation dashboard (CLV-first)

`dash/render.py` (zero-LLM, matplotlib + a tiny Jinja/HTML template) rebuilds `dash/dashboard.html` daily.
Metrics, in priority order:

| Metric | Definition | Why it leads |
|---|---|---|
| **CLV (closing-line value)** | `mean( log(bet_odds / closing_sharp_odds) )`; and % of bets with CLV > 0 | **The** leading indicator. Beating the closing sharp price is the most reliable proxy for long-run profitability; it converges far faster than ROI. |
| **ROI / yield** | P&L / turnover, flat-stake **and** Kelly-stake portfolios | Report both; flat-stake isolates selection skill from staking. |
| **Brier score** | `mean((p_pred − outcome)²)` on 1X2 | Probabilistic accuracy. |
| **RPS (Ranked Probability Score)** | ordered-3-way score for H/D/A | Proper scoring rule that respects outcome ordering — the standard football-forecast metric. |
| **Log loss** | `−mean(log p_outcome)` | Penalises confident wrong calls. |
| **Calibration plot** | reliability diagram (predicted-decile vs observed frequency) + Brier decomposition | A well-ranked but badly-calibrated model must be recalibrated before Kelly trusts it. |
| **Drawdown / max exposure** | from the ledger | Feeds the kill-switch. |
| **A/B arms** | A (model-only) vs B (model+LLM) on CLV/RPS/ROI | §11. |

Benchmark = the **closing sharp price** (Pinnacle / Betfair Exchange), not buy-and-hold — in betting the
market *is* the benchmark, and a paper edge that doesn't beat the close is noise. Optional: mirror the
dashboard into the Obsidian vault with the `obsidian` skill, or expose it as a **TUI widget**
([tui-widgets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tui-widgets)); the `hermes-hud`
dashboard plugin is a community option for cost/cron observability (from `C0`).

---

## 11. How the LLM adds value — and how we *measure* it (A/B)

**Hypothesis:** team news (a rested striker, a makeshift back line) is real information that a goals-only
Dixon-Coles fit can't see. The LLM's job is to *read the news* and *boundedly* adjust λ — never to pick sides.

**Design (a genuine experiment, not a vibe):**
- Every qualifying fixture is staked **twice**, identically, by the same `stake.py`:
  - **Arm A (control):** model-only probabilities.
  - **Arm B (treatment):** probabilities after the LLM's bounded λ adjustment (≤ ±15%) or veto.
- Both arms write to `ab_arms`, settle independently, and accumulate CLV/RPS/ROI.
- The LLM's adjustment and its prompt+response are stored in `llm_runs` so we can audit *why* it moved a number.
- **Decision rule (pre-committed):** the B arm earns its keep only if, over a large sample, it shows
  **higher mean CLV or better RPS than A** with the difference distinguishable from noise. If not, drop the
  adjustment and keep the LLM strictly as narrator — the "researches but never trades" precedent (`C0`) and
  T2's analyst-not-trader split are the default posture, and the burden of proof is on the LLM.
- **Guard against the obvious traps:** the LLM must never *invent* an injury or a number (T2/StockBench:
  static competence ≠ decision skill); it quotes the API team-news feed, and the pipeline applies the shift.
  Also watch the DXRG lesson that *prompt framing*, not strategy text, drives behaviour — keep the prompt
  fixed across the experiment so the comparison is clean.

---

## 12. Cost estimate per month

**LLM (DeepSeek `deepseek-flash`, off-peak $0.15/M in, $0.60/M out; peak = 12:00–15:00 & 17:00–21:00 AEDT):**

| Job | calls/mo | est. cost |
|---|---|---|
| Team-news summaries (only on gated ticks) | ~40 | ~$0.10 |
| Previews | ~20 | ~$0.05 |
| Weekly review narrative | 4 | ~$0.05 |
| A/B adjustment calls | ~40 | ~$0.10 |
| **LLM subtotal** | | **≈ $0.30–1.5 / mo** |

Football kickoffs (UK evenings → Sydney late night/early morning) mostly sit **off-peak** — another
"cheapest thing in the house" case, same as `H1` §3.1.

**Data (the real cost):**

| Source | Monthly |
|---|---|
| football-data.co.uk | **$0** |
| API-Football (lineups/injuries, free 100 req/day) | **$0** (→ $19 if polling exceeds the free tier) |
| The Odds API (poll per **window**, 1–2 regions × h2h only) | **$0** on the 500-credit free tier for ~2–3 leagues (→ **$30** if you add historical-odds CLV backtests or broad regions/markets) |
| Betfair Exchange (app key) | **$0**; streamed/historical may cost (**UNVERIFIED**) |

**Bottom line: ~$0–$2/mo** for a free-tier paper desk; **~$30–$40/mo** only if you choose historical-odds
backtesting + API-Football Pro. Comfortably under the existing `deepseek-budget` watchdog threshold.

*Credit arithmetic (to keep the free tier):* The Odds API charges **1 credit per market per region** per
`/odds` call. Polling only at the two gate points per kickoff **window** (not per match), 1 region × h2h, across
~5 weekend windows + 2 midweek = ~14 credits/week ≈ **60 credits/mo** — well inside 500. **UNVERIFIED** the
exact per-call cost for multi-region calls; confirm on the first live request (headers report credits
remaining).

---

## 13. 4-week build plan → Kanban cards (`punter` board, `dev` builds, `reviewer` audits)

```bash
hermes kanban boards create punter --name "Punter desk" --description "Football model + paper-betting ledger"
# dev = Claude Sonnet (code); reviewer = Claude Sonnet (independent audit) — same split as the trading desk
```

**Week 1 — plumbing, no money logic**
- `dev` — **P1** profile `punter` + workdir scaffold (`AGENTS.md`, `config/limits.json`, SQLite schema, `uv` venv with `penaltyblog`), keys in profile `.env`, `#picks` channel. *(accept: `punter` profile answers on Slack with gemma; `penaltyblog` imports)*
- `dev` — **P2** `ingest.py`: football-data.co.uk downloader + canonical team map + SQLite load. *(accept: a full EPL season loads with correct dtypes)*
- `reviewer` — **P3** audit P2 for **point-in-time discipline**: no future rows leak into a fit; verify the as-of timestamp handling. *(T2 §3.6; accept: a written leak/lookahead checklist signed off)*

**Week 2 — the deterministic core**
- `dev` — **P4** `fit.py` + `predict.py`: Dixon-Coles (time-decay) + Elo cross-check → scoreline matrix → 1X2/O-U/BTTS probs. *(accept: backtest Brier/RPS reported on a held-out season)*
- `dev` — **P5** de-margin + `value.py` + `stake.py` with `limits.json` clamps (paper only). *(accept: clamps unit-tested; a 40%-of-bankroll edge is clamped to 2%)*
- `reviewer` — **P6** audit P4/P5: verify the probability maths, margin method, and that no LLM is in the stake path; re-derive Brier/RPS independently.

**Week 3 — the loop + the LLM layer**
- `dev` — **P7** `ledger.py` + `settle.py` + `clv.py` + `punter-ingest`/`punter-build`/`punter-tick`/`punter-snapshot`/`punter-settle` cron jobs. *(accept: a full weekend runs shadow-mode end-to-end with CLV recorded)*
- `dev` — **P8** DeepSeek team-news summariser + bounded λ adjustment + `ab_arms` A/B harness.
- `reviewer` — **P9** audit P8: prompt-injection posture (web content can't size/add markets), the ±15% bound is enforced in the pipeline, and the A/B arms are truly independent.

**Week 4 — evaluation, gating, durability**
- `dev` — **P10** `dash/render.py` (CLV/ROI/RPS/calibration plot) + `punter-daily-metrics` + `punter-weekly-review`.
- `dev` — **P11** real-money **gating report** (§14) generated from the ledger.
- `reviewer` — **P12** final audit: gate criteria are correctly computed; secrets scoped; kill switch works; document the 25-Oct DST behaviour.

```bash
hermes kanban create "P1 punter profile + scaffold"      --assignee dev      --workspace dir:"C:/Users/Rayan/projects/punter"
hermes kanban create "P3 audit ingest point-in-time"     --assignee reviewer --workspace dir:"C:/Users/Rayan/projects/punter"
# ... etc; hard model/strategy design questions -> rare Claude cards only
```

Per `H1`: Kanban is single-host; declare `artifacts` or use `dir:`/`worktree:`; keep secrets/PII out of
card `summary`/`metadata` (run rows are durable).

---

## 14. Gating a future real-money switch

**Default: never.** Real money is a separate project with its own sign-off. It unlocks only when **all** of
these hold, evaluated on the paper ledger (not a narrative):

1. **Sample:** ≥ **500 settled bets** across **≥ 3 months** and **≥ 2 leagues** (CLV needs a large n to
   separate from noise).
2. **Positive CLV, statistically:** mean log-CLV **> 0** with a 95% bootstrap/t confidence interval that
   **excludes 0**, *and* the same sign on the flat-stake and Kelly portfolios.
3. **Calibration:** reliability diagram within tolerance and Brier/RPS no worse than the closing-line
   benchmark's own implied probabilities.
4. **Discipline held:** drawdown never breached a pre-set cap; the kill switch demonstrably fired in a drill.
5. **A/B settled:** the LLM arm either beats the control (keep the adjustment) or is proven neutral (demote the
   LLM to narrator) — no "we'll figure out later".
6. **Then, and only then:** a **manual** flip (never automated), a **separate** live-key secret scope, a
   per-bet and per-day cash cap hard-coded, a hard kill file, and a small initial bankroll. Start on a single
   sharp-friendly market (exchange) where the paper→live gap is smallest.

If any criterion fails, the desk stays paper-only — which is still the point (learn, keep risk disciplined,
produce an auditable track record), per T2's "expectations to hold".

**Legal (Australia) — carry-over from T2 §4, still applicable:** trading **your own** money/wagers is not
"dealing on behalf of a client", so **no AFSL** arises for personal use; the moment you bet/advise **for
others** (publish followable picks as a service) the question changes — **UNVERIFIED** whether a public
picks feed could be "arranging"/"general advice". Recreational punter winnings in Australia are generally not
taxed as income, but professional/gambling-business framing differs — **UNVERIFIED**; confirm with a tax agent.
This note is **not financial, legal, or tax advice**.

---

## 15. Risks, gotchas & open items

- **25 Oct 2026 DST shift** — the single most likely silent breakage. Mitigated by the `punter-tick` gate reading
  real UTC kickoff times, not fixed clocks. Test the tick on both sides of the boundary.
- **Model ≠ market.** Dixon-Coles is a stated-priors model; bookmakers have information you don't. Expect CLV to
  be the *hard* bar, and most picks to show negative CLV until/unless there's a real edge. That's the honest
  baseline (T2 §5).
- **The A/B is the whole LLM thesis.** If it doesn't win, don't hand-wave — demote the LLM to narrator.
- **Free-tier credit burn.** The Odds API poll must be **window-level, not match-level**, and h2h-only, or you
  blow 500 credits fast. Watch the `x-requests-remaining` response header.
- **Betfair key/access** — app-key and historical-data costs are **UNVERIFIED** this session; confirm before
  relying on Exchange closing prices for CLV (Pinnacle-implied close is the fallback).
- **`python3` not on PATH** (host has `python` 3.14.7) — `.py` scripts + a pinned `uv` venv
  (penaltyblog needs pandas/scipy/Cython).
- **Name canonicalisation** across football-data.co.uk / The Odds API / API-Football is a quiet correctness
  risk — get P2's map right or joins silently drop matches.
- **Slack target silent-drop** — a wrong/case-mismatched `#picks` target fails silently; check `last_error`.
- **Cron needs the gateway**, and the kickoff window is overnight Sydney time — decide whether the PC has wake
  timers for the 22:00–08:00 window, or accept that some windows are missed while it's asleep (same
  reliability question `H1` §8 raises for the US session).
- **`penaltyblog` is a live dependency** (v1.12.0, MIT) — pin the version; its scrapers can break on
  upstream site changes.

---

## 16. Sources

Hermes docs (all under `https://hermes-agent.nousresearch.com/docs/`):
[profiles](https://hermes-agent.nousresearch.com/docs/user-guide/profiles) ·
[multi-profile-gateways](https://hermes-agent.nousresearch.com/docs/user-guide/multi-profile-gateways) ·
[cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) ·
[cron-script-only](https://hermes-agent.nousresearch.com/docs/guides/cron-script-only) ·
[cron-troubleshooting](https://hermes-agent.nousresearch.com/docs/guides/cron-troubleshooting) ·
[security](https://hermes-agent.nousresearch.com/docs/user-guide/security) ·
[secrets](https://hermes-agent.nousresearch.com/docs/user-guide/secrets) ·
[Kanban](https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban) ·
[Slack](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/slack) ·
[tool-search](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-search) ·
[tui-widgets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tui-widgets) ·
[llms.txt index](https://hermes-agent.nousresearch.com/docs/llms.txt).

Betting/data sources:
[football-data.co.uk notes](https://www.football-data.co.uk/notes.txt) ·
[football-data.co.uk England files](https://www.football-data.co.uk/englandm.php) ·
[The Odds API](https://the-odds-api.com/) ·
[The Odds API v4 docs](https://the-odds-api.com/liveapi/guides/v4/) ·
[The Odds API historical](https://the-odds-api.com/historical-odds-data/) ·
[Betfair Exchange API](https://developer.betfair.com/exchange-api/) ·
[Betfair developer get-started](https://developer.betfair.com/en/get-started/) ·
[API-Football pricing](https://www.api-football.com/pricing) ·
[penaltyblog (PyPI)](https://pypi.org/project/penaltyblog/) ·
[penaltyblog (GitHub)](https://github.com/martineastwood/penaltyblog) ·
Dixon & Coles (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market* —
**UNVERIFIED** this session (reference page failed to load); the model is implemented in `penaltyblog`.

Evidence/prose reuse: `notes/T2-ai-trading-evidence.md` §3 (design patterns), §5 (recommended architecture),
FINSABER `arXiv:2505.07078`, DXRG `arXiv:2609.05663`, `notes/H1-hermes-integration.md` (profile/cron/Kanban
mechanics), `notes/C0-community-precedents.md` (analyst-not-trader precedent).

All timezone conversions in §6 were computed live with Node ICU on 6 Oct 2026.
