# Implementation plan
Companion to `docs/REPORT.md`. Work is split into Kanban cards for the Hermes **dev** bot (builds) and **reviewer** bot (audits). Each card is one PR-sized piece with a clear "done when". Gates come from REPORT §6.

**Ground rules** (also in `AGENTS.md`): Python 3.13 via `uv` · UTC everywhere · no lookahead (tested) · no LLM in any decision path · paper-only (no order-placement code before Gate 2) · secrets in `.env` · data never committed.

---

## Target layout
```
football-edge/
├── pyproject.toml / uv.lock
├── AGENTS.md  README.md  .env.example  .gitignore
├── config/
│   ├── leagues.toml         # divisions, seasons, which markets are enabled
│   └── limits.toml          # stake caps, kill-switch, commission=0.06, LIVE=false
├── src/fedge/
│   ├── ingest/              # football_data.py, understat.py, betfair_snapshots.py
│   ├── schemas.py           # pandera schemas (matches, odds, snapshots, bets)
│   ├── store.py             # Parquet + DuckDB access, "available_at" discipline
│   ├── teams.py             # team-name normalisation across sources
│   ├── market.py            # de-margining (power, Shin, multiplicative), implied probs
│   ├── features/            # ratings.py (Elo, pi), form.py, xg.py, asof.py
│   ├── models/              # dixon_coles.py, gbm.py, calibrate.py, pool.py
│   ├── backtest/            # walkforward.py, strategy.py, execution.py, stats.py (bootstrap, CLV)
│   ├── paper/               # ledger.py (SQLite), settle.py, risk.py (caps, kill switch)
│   └── report/              # metrics tables, calibration plots, weekly summary
├── scripts/                 # CLI entry points called by Hermes cron (zero-LLM)
├── tests/                   # unit + leakage + determinism tests
├── reports/                 # generated gate reports (committed: markdown + small PNGs)
├── data/                    # gitignored: raw/, parquet/, fedge.duckdb, paper.sqlite
└── docs/                    # REPORT.md, IMPLEMENTATION_PLAN.md, research/
```

---

## Phase 0: Scaffold (½ week)
| # | Card | Done when |
|---|---|---|
| 0.1 | **Project scaffold:** `uv init` (Python 3.13), deps from REPORT §5.4, ruff + pytest, `src/fedge` package, layout above, `config/*.toml` with defaults, `.env.example` | `uv run pytest` passes (smoke test) on Windows; `uv run ruff check` clean |
| 0.2 | **CI:** GitHub Actions on `windows-latest` + `ubuntu-latest`: ruff, pytest | Green badge on README |

## Phase 1: Data (1 week)
| # | Card | Done when |
|---|---|---|
| 1.1 | **football-data.co.uk ingest:** download all divisions in `leagues.toml`, seasons 2012/13→current (follow redirects, polite rate limit, cache raw CSVs); parse to long-format `matches` + `odds` tables (bookmaker, market, side, phase=pre/close) | Parquet + DuckDB populated; pandera schemas pass; re-run is idempotent |
| 1.2 | **Data-quality report:** per league/season coverage of PS/PSC, BFE/BFEC, Avg/Max, xG; flag Pinnacle after 23/07/2025 as `stale` | `reports/data_coverage.md` generated; matches Iris's spot checks (E0 25/26 PSC = 210/380, last 08/01/2026) |
| 1.3 | **Team-name normalisation** across football-data, Understat, Betfair | Mapping table + test that every 2024/25+ fixture maps uniquely |
| 1.4 | **Understat xG ingest** (top-5 leagues, via penaltyblog or soccerdata), cached | xG joined to matches for those leagues; cache prevents re-scraping |

## Phase 2: Market layer + evaluation harness (1 week). Reviewer audits.
| # | Card | Done when |
|---|---|---|
| 2.1 | **De-margining** (`market.py`): power (default), Shin, multiplicative; 1X2 and 2-way (O/U) | Unit tests vs penaltyblog reference values; **A/E ≈ 1.000** sanity report on closing odds per league (Gate 0 B0.1) |
| 2.2 | **Metrics:** RPS, log-loss, Brier, calibration slope, ECE, reliability plot | Tests on hand-computed examples |
| 2.3 | **Walk-forward harness:** season/matchweek folds, as-of feature builder (`asof.py`) that only sees rows with `available_at < bet_time` | Leakage tests: (a) future-row injection detected, (b) shuffled-target collapses edge, (c) **regime-reversal flips sign**, (d) determinism (byte-identical rerun) |
| 2.4 | **Stats:** CLV (log, vs margin-free close), match-clustered bootstrap CI, flat & ¼-Kelly bankroll simulation with 6% commission | Tests; CLV of "bet the close" ≈ 0 |
| 2.5 | **Review card (reviewer):** audit 2.1–2.4 for leakage and maths errors | Reviewer sign-off note in `reports/reviews/` |

## Phase 3: Baseline models (1 week). First evidence.
| # | Card | Done when |
|---|---|---|
| 3.1 | **Dixon-Coles** (penaltyblog), time decay ξ tuned in-fold; outputs 1X2 + O/U 2.5 | Walk-forward RPS/log-loss per league/season vs market in `reports/v1_baseline.md` |
| 3.2 | **Own Elo + pi-ratings** + ordered-logit model | Same report |
| 3.3 | **Log opinion pool** (`pool.py`): weight *w* of model vs de-margined **pre-closing** and **closing** market, out-of-sample, with bootstrap CI | *w* table per league × market in the report. **Decision point:** if *w* ≈ 0 against the pre-closing price everywhere, prioritise timing/steamer strategies (v3) over model work |

## Phase 4: Main model (1½ weeks)
| # | Card | Done when |
|---|---|---|
| 4.1 | **Feature set:** pi/Elo ratings, DC attack/defence, time-weighted form, rest days, promoted flag, xG form (where available); all via `asof.py` | Feature leakage tests pass |
| 4.2 | **LightGBM multiclass** (+ CatBoost comparison; goals-only and xG variants); walk-forward retraining | `reports/v2_main.md`: RPS/log-loss vs DC and market |
| 4.3 | **Calibration:** isotonic, Beta, Venn-Abers chosen in-fold by log-loss | Calibration slope ≈ 1 on holdout; reliability plots |
| 4.4 | **Decorrelation variant** (penalise correlation with market probs) | Included in the pool-weight comparison |

## Phase 5: Strategy backtest → Gate 0 (1 week). Reviewer audits.
| # | Card | Done when |
|---|---|---|
| 5.1 | **Early-line value strategy:** bet at the pre-closing price where pooled edge (after 6% commission) > threshold; threshold, Kelly fraction and league whitelist chosen **in-fold** | Out-of-sample CLV, ROI (flat & ¼-Kelly), bet counts per league/market/season |
| 5.2 | **Baselines on the same ledger:** back-the-favourite, random-with-same-count | Strategy beats both on CLV |
| 5.3 | **Gate 0 report** (`reports/gate0.md`): every B0.x item with evidence, including losing runs | Reviewer sign-off; **you decide** which leagues/markets go to paper |

## Phase 6: Live paper desk (1½ weeks)
| # | Card | Done when |
|---|---|---|
| 6.1 | **Betfair delayed-key client:** login via certs in `.env`, fixture discovery (`listMarketCatalogue`), price snapshots (`listMarketBook`) for whitelisted leagues; confirm football market type strings live | Snapshot job stores T-24h, T-60m and T-1m (close) books with depth |
| 6.2 | **Paper ledger** (SQLite): bet at T-24h snapshot price (only if depth ≥ stake), close capture, settlement from results | End-to-end dry run on one matchweek |
| 6.3 | **Risk module:** caps from `limits.toml`, daily bet cap, drawdown kill-switch file; **no order-placement code** | Kill-switch drill test |
| 6.4 | **Scripts + Hermes cron (zero-LLM):** `fedge-snapshot` (every 15 min, gated on kickoff windows), `fedge-picks` (daily), `fedge-settle` (daily), `fedge-weekly` (Mon) → Slack `#picks` | Jobs registered on the `punter` profile; a Slack digest arrives |
| 6.5 | **Watchdog:** alert if ingest/snapshots fail or a schema check breaks | Failure injected → Slack alert |

## Phase 7: Run the season → Gate 1
- Frozen config; weekly auto-report (CLV with CI, ROI, calibration, *w*).
- Gate 1 check when ≥1,000 settled bets (likely mid-to-late season across 4–6 lower leagues).
- Any change to model/thresholds = new experiment ID, sample resets.

## Stretch (v3, after Gate 0)
Over/under-specific model · steamer/drifter timing (T-24h vs T-60m snapshots) · market-calibrated in-play intensity model (arXiv:2605.16066) · Bayesian hierarchical (PyMC) · player-level/lineup features · conformal stake gating.

---

## Timeline (part-time, around uni)
| Weeks | Phases | Milestone |
|---|---|---|
| 1 | 0 + 1 | Data in DuckDB, coverage report |
| 2 | 2 | Harness + leakage tests (reviewer-audited) |
| 3 | 3 | **First evidence:** baseline vs market, pool weights |
| 4–5 | 4 | Main model + calibration |
| 6 | 5 | **Gate 0** decision |
| 7–8 | 6 | Paper desk live (Betfair delayed key, Slack `#picks`) |
| 9+ | 7 | Season run → Gate 1 |

## Hermes integration
- **Build:** each card → `hermes kanban create "<card>" --assignee dev --body "<card text + links to REPORT/AGENTS>"`; Phase 2, 5 and 6.3 cards also get a reviewer card.
- **Run:** new `punter` profile holding Betfair secrets; all recurring jobs are **zero-LLM scripts** (`no_agent` cron) posting to Slack. An optional DeepSeek weekly narrative (~1¢/week) summarises the weekly report; it never touches decisions.
- **Iris:** "how's the football bot?" → reads the latest `reports/weekly_*.md`.

## Cost
Data free · Betfair delayed key free · Hermes cron free (scripts) · optional DeepSeek ≈ $0.05/month · Claude usage only while dev builds. Live key (£499 under Betfair's global terms; AU fee unconfirmed) only at Gate 2.
