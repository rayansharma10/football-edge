# football-edge ⚽

[![CI](https://github.com/rayansharma10/football-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/rayansharma10/football-edge/actions/workflows/ci.yml)

A football (soccer) match-prediction and **paper-betting** system built as a proper ML pipeline: calibrated statistical/ML models, blended with the betting market, judged by **closing-line value (CLV)** against the Betfair Exchange.

> **Status (2026-10):** pivoted to a **prediction-accuracy tracker** (Premier League + La Liga, winner and scoreline); betting is parked, see "Prediction-accuracy tracker" below.
> **Mode:** paper only. Real money only after the gates in `docs/REPORT.md` §6 are passed and Rayan flips it manually.

## The idea in one paragraph
Good models rarely out-predict the *closing* odds, but markets are less efficient *before* they close. So the system prices each match (Dixon-Coles → LightGBM, calibrated, pooled with the de-margined market), bets early where the price beats our fair probability, and scores every bet by whether it beat the closing price. That's testable on 12+ seasons of free historical odds (football-data.co.uk) before anything goes live.

## Docs
- [`docs/REPORT.md`](docs/REPORT.md): research findings, data reality, recommended ML system, gates
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md): phased Kanban cards, layout, timeline
- [`AGENTS.md`](AGENTS.md): hard rules for anyone (human or agent) writing code here
- [`docs/research/`](docs/research/): six underlying research notes with sources. Note: `B3` predates the decision to remove the LLM from the decision loop; where it conflicts, `REPORT.md` wins.

## Stack
Python 3.13 (uv) · penaltyblog · LightGBM/CatBoost · scikit-learn · DuckDB + Parquet · pandera · betfairlightweight/flumine · Hermes cron + Slack for ops.

*Not financial advice. Gambling involves risk; BetStop (betstop.gov.au) is Australia's national self-exclusion register.*

## Prediction-accuracy tracker (current focus)

**Strategy change:** betting is parked (unlikely to make money). football-edge now tracks how
accurate its **predictions** are for the Premier League (`E0`) and La Liga (`SP1`): for every
fixture it predicts the **winner (W/D/L) and the scoreline**, saves that prediction *before kick-off*
(never edited), saves the real result afterwards, and shows accuracy over time. Bookmaker odds are
not used by any prediction and are not shown anywhere (the old betting code, tests and the
`bets` / `settlements` / `snapshots` tables are kept untouched as an archive; its Hermes cron jobs
are **paused**, not deleted).

### Moving parts

| piece | what it does |
|---|---|
| `scripts/predict_upcoming.py` | refresh football-data + the full-season schedule, predict every upcoming E0/SP1 fixture with the current model, write the display cache and append to `prediction_log` |
| `scripts/update_results.py` | fill `results` with finished matches (schedule scores, cross-checked against the football-data CSVs) |
| `scripts/backfill_retro.py` | leakage-safe retrospective predictions for matches already played (`source='retro'`) |
| `src/fedge/accuracy/` | `store.py` (tables + rules), `metrics.py`, `baselines.py`, `results.py`, `report.py` (the JSON behind the Accuracy tab) |
| `src/fedge/predict/models.py` | the model registry (see below) |
| `scripts/dashboard.py` | read-only dashboard: `/` Upcoming, `/accuracy`, `/archive` (old betting ledger; `/ledger` redirects) |

Hermes cron (all `no-agent`, output saved locally, failures go to Slack): `football-edge predictions`
(`30 11,19 * * *` Sydney), `football-edge results` (`0 16 * * *`) and the hourly
`football-edge catch-up (predict/results)` which re-runs either job if its last success is older
than 12 h / 26 h (the PC is often off before ~11:00). The wrapper is `~/.hermes/scripts/fedge_paper.py`.

### The model (one price-free model for every fixture)

`fedge.predict.models` is a registry keyed by `model_version`; every prediction is produced by
`models.predict(version, ...)` and stored with its version. Version `lgb_xg_dc_v1` is the
non-anchored `lgb_xg` LightGBM 1X2 (ratings, form, xG form, schedule; **no market feature**)
combined with a Dixon-Coles scoreline grid reconciled to that 1X2. Phase 2 (a model bake-off) will
register alternatives with `@models.register("name")` and flip `CURRENT_VERSION`; nothing else
changes. Fixtures weeks away are scored with today's ratings and will shift as results arrive.

**Winner/score consistency:** the predicted score is the most likely score *within the predicted
outcome class*, so score and winner always agree (a 1-1 grid mode inside a home-favourite match is
reported as 1-0). The unconstrained grid mode is logged separately (`pred_mode_score_*`).

### Telemetry: `data/predictions.sqlite`

A **separate database** from the betting ledger (`data/paper.sqlite`): betting is parked, so the
accuracy record must not depend on that file, can be backed up or rebuilt on its own, and the
dashboard can open it `mode=ro` without touching the ledger. Tables:

* `prediction_log` - **append-only** (SQLite triggers abort UPDATE/DELETE) and **frozen**: a row
  with `run_ts >= kickoff_utc` is rejected (trigger + Python check). One row per
  `(match_key, snapshot_kind, run_ts)`. Columns: match/league/teams/kickoff, `run_ts`,
  `hours_before_kickoff`, `model_version`, `source` (`live`/`retro`), `p_home/p_draw/p_away`,
  `xg_home/xg_away`, `pred_outcome`, `pred_score_home/away/p`, `pred_mode_score_home/away`,
  `p_over25`, `p_btts`, `top_scores` (JSON, top 5), `grid` (JSON 8x8: 0-7 goals each side, rows = home).
  No price column exists.
* snapshot kinds: `run` = every logged prediction run (runs more than 14 days before kickoff are
  shown on the page but not logged); **`d7`** = a copy of the *first* run at or after
  `kickoff - 7 days`; **`d3`** = a copy of the *first* run at or after `kickoff - 3 days`. They are
  unique per match and earn no back-fill: if no run fell inside a window (PC off, or the fixture only
  became known later) that snapshot is simply absent. One run can carry both labels (two rows) -
  `hours_before_kickoff` on every row says how early each one really was. **`final`** is derived,
  not stored: view `v_final` = the latest `run` row before kickoff.
* `results` - `match_key, div, home, away, kickoff_utc, home_goals, away_goals, result (H/D/A),
  source, fetched_ts`. Inserted once per match (`INSERT OR IGNORE`); a re-fetch with a different
  score is counted as a conflict and left alone.
* `upcoming` + `meta` - replace-each-run display cache for the Upcoming page.

Results: the full-season schedule feed **fixturedownload.com carries final scores**
(`HomeTeamScore` / `AwayTeamScore`; openfootball carries `score.ft` as a fallback). The football-data
season CSVs (`data/raw/football_data/2026-27/E0.csv`, `SP1.csv`) cross-check it (mismatches are
printed) and fill gaps. Team names reuse `config/team_aliases.csv` + `config/schedule_aliases.csv`;
unmapped names are logged to stderr. A rescheduled match gets a new `match_key` (the key includes
the match date); the old key simply never gets a result.

**Retrospective rows** (`scripts/backfill_retro.py`): matches already played get one prediction each
from a model fitted only on matches kicked off *strictly before* their matchday group's first date
(groups span at most 3 days; `run_ts` is that cut-off, so a Sunday match is predicted with Friday-
morning knowledge). They are labelled *retrospective (not live)* on the page and excluded from the
headline numbers unless you tick "Include retrospective predictions" (or `?include_retro=1` on
`/api/accuracy`). They never get d7/d3 labels.

### Reading the Accuracy tab (`/accuracy`, JSON at `/api/accuracy`)

Headline = the `final` snapshot of each finished match. `n < 100` is flagged *small sample*;
95% ranges are match-clustered bootstrap intervals (1000 resamples, seed 0).

* **Winner hit rate** - the most likely outcome was right. **Brier** (multiclass, 0 best, a
  33/33/33 guess scores 0.667), **log-loss** (1.099 for 33/33/33), **RPS** (ordered H<D<A) - lower is
  better. **Calibration** - when the model says 30%, ~30% should happen. **Confusion matrix** and
  the share of draws predicted vs actual (models rarely call a draw).
* **Exact-score hit rate** - inherently hard: the single most likely score only has about a 10-12%
  chance, so ~1 in 9 is the ceiling. Also top-3 / top-5 (actual score among the 3 / 5 most likely),
  correct goal difference, MAE of expected goals vs goals scored, the mean probability and log
  probability given to the real scoreline, and accuracy + Brier for Over 2.5 and both-teams-score.
* **Baselines on the same matches:** always-home; league-average W/D/L frequencies (and the league's
  most common scoreline); a naive Poisson from each team's previous-season goal rates. A useful model
  must beat these.
* **Splits:** per league, per month, and `d7` vs `d3` vs `final` (does accuracy improve closer to
  kick-off? A like-for-like table restricts to matches that have all three).
* The page also lists recent predictions vs results (winner / exact tick or cross), a calibration
  chart and cumulative accuracy over time. The Upcoming page shows the real result next to our
  prediction for matches that finished in the last 3 days.

### Running it

    uv run python scripts/predict_upcoming.py          # predict + log (about 2-3 min)
    uv run python scripts/update_results.py            # fill results
    uv run python scripts/backfill_retro.py [--max-groups N]   # once, for a populated Accuracy tab
    uv run python scripts/dashboard.py                 # http://127.0.0.1:8765 (--port to change)
    uv run python scripts/predict_upcoming.py --home Arsenal --away Chelsea   # one match

### Archive (paper betting)

The previous system (market-anchored `lgbd_xg`, paper bets, CLV, settlements, weekly report) is
unchanged and still runnable (`scripts/paper_picks.py`, `paper_settle.py`, `paper_weekly.py`), but
its cron jobs are paused and `config/strategy.toml` (gate 0 / mode) was not touched. Its ledger page
is at `/archive` (low-key link in the footer of the Upcoming page); the old `/ledger` URL redirects
there. The remainder of this README describes how the full-season schedule is built, which both the
tracker and the archive use.

#### Whole-season schedule

football-data's `fixtures.csv` only carries the next few days (and often no E0/SP1 rows at all), so the
Upcoming page lists **every remaining match of the season** from a full schedule
(`src/fedge/ingest/schedule.py`):

1. **fixturedownload.com** JSON feed (primary; free, no key, UTC kickoffs, final scores),
2. **openfootball** `football.json` on GitHub (fallback; free, no key),
3. the **last good cache** in `data/raw/schedule/` (flagged STALE on the page when both fail).

Team names are mapped to the football-data names used by the rest of the repo via
`config/team_aliases.csv` plus `config/schedule_aliases.csv`; an unmapped name is logged at ERROR level,
printed to stderr and its fixtures are dropped (add the alias and re-run).
