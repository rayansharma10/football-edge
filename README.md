# football-edge ⚽

[![CI](https://github.com/rayansharma10/football-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/rayansharma10/football-edge/actions/workflows/ci.yml)

A football (soccer) match-prediction and **paper-betting** system built as a proper ML pipeline: calibrated statistical/ML models, blended with the betting market, judged by **closing-line value (CLV)** against the Betfair Exchange.

> **Status:** research complete, implementation not started. See the plan.
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

## Dashboard and scheduling
Read-only local dashboard over `data/paper.sqlite` (opened `mode=ro`, bound to 127.0.0.1 only, no auth):

    uv run python scripts/dashboard.py          # http://127.0.0.1:8765  (--port to change)

Two pages. `/` is the upcoming-fixtures view (model probabilities, expected goals, predicted scoreline and goal markets only — no bookmaker prices). `/ledger` is the paper-trading book: mode/gate status, last picks/settle runs (STALE warning), open and settled bets, cumulative P&L/CLV, top snapshot edges and the latest weekly report. The fixtures page polls `/api/predictions`; the ledger pages poll `/api/summary`, `/api/open_bets`, `/api/settled`, `/api/series`, `/api/snapshots`. Both refresh every 60s.

### Upcoming predictions panel (Premier League + La Liga only)

`uv run python scripts/predict_upcoming.py` scores every upcoming **E0 / SP1** fixture and writes the
`predictions` table of `data/paper.sqlite` (the script is the only writer; the dashboard stays
`mode=ro`). The **main page** lists the upcoming E0/SP1 fixtures grouped by kick-off day
(Australia/Sydney): per match the model's home/draw/away %, expected goals, the single most likely
scoreline with its probability, over/under 2.5 and both-teams-to-score. It shows **no bookmaker
prices at all** - the model may use them internally, but the page displays the model only. The bets,
prices, edge, CLV and P&L live on a second page, `/ledger` (*Paper-trading ledger*), so the fixtures
view stays clean. Both pages show a `last updated` age and a STALE flag after 20h.

The 1X2 **is** the live LightGBM model, which is itself **market-anchored** (its features include the
de-margined snapshot price of the same match). The predicted scoreline is a **Dixon-Coles estimate**:
expected goals per division come from `fedge.models.dixon_coles.fit_dc` (time-decayed, 5-year window,
refit per run), and the score grid is reconciled to the LightGBM 1X2 by solving for the two goal rates
whose grid reproduces the target home/away probabilities (`scipy.optimize.root`); if that does not
converge - or to remove the last of the numerical error - the grid is rescaled by outcome class (IPF)
so the 1X2 matches the model exactly. The predicted score and the goal markets are therefore a model
estimate, not a market price. `/api/predictions` returns the model fields only (`probs`, `xg`,
`top_scores`, `p_over25`, `p_btts`, `long_range`, plus a `schedule` block); the de-vigged price columns and the grid stay in the
`predictions` table and are not served. Two ways to run it by hand:

    uv run python scripts/predict_upcoming.py --home Arsenal --away Chelsea   # single-match lookup
    uv run python scripts/predict_upcoming.py --dry-run --json                # print, write nothing

#### Whole-season schedule and the price-free model

football-data's `fixtures.csv` only carries the next few days (and often no E0/SP1 rows at all), so the
panel lists **every remaining match of the season** from a full schedule (`src/fedge/ingest/schedule.py`):

1. **fixturedownload.com** JSON feed (primary; free, no key, UTC kickoffs),
2. **openfootball** `football.json` on GitHub (fallback; free, no key),
3. the **last good cache** in `data/raw/schedule/` (flagged STALE on the page when both fail).

No API key is needed. (football-data.org's free tier would work with `FOOTBALL_DATA_ORG_KEY`, 10 req/min, but is not required.)
The schedule is re-fetched on every `predict_upcoming.py` / `paper_picks.py` run (`--no-refresh-schedule` /
`--cached-fixtures` reuse the cache). Team names are mapped to the football-data names used by the rest of the repo
via `config/team_aliases.csv` plus `config/schedule_aliases.csv`; an unmapped name is logged at ERROR level,
printed to stderr and its fixtures are dropped (add the alias and re-run). football-data's `fixtures.csv` stays the
source of bookmaker prices: a fixture that has a complete price replaces its schedule twin.

Two 1X2 methods, stored per row in `predictions.model_kind`:

* `lgbd_xg_priced` - fixtures with a price: the market-anchored `lgbd_xg` (unchanged).
* `lgb_xg_price_free` - everything else (weeks away): the non-anchored `lgb_xg` (best price-free 1X2 model in
  `reports/v2_main.md`, log loss 1.0103 vs Elo 1.0164, Dixon-Coles 1.0273), fed the ratings and form **as of today**
  with no market feature. Rest-day features for a far-future match are computed as if it kicked off a week after the
  division's latest played match. These rows carry a subtle *longer-range estimate* note on the page and **will shift
  as results come in**. The scoreline is still the Dixon-Coles grid reconciled to that 1X2 (`predict_match`).

The page has no date cut-off: league chips (all / Premier League / La Liga), a *Next 7 days / Whole season* toggle,
fixtures grouped by day with 10 days shown at a time and a *Show more* button. This is display-only: shadow bets are
still only placed for fixtures with real prices, and the picks digest on stdout is unchanged.

The same refresh runs at the end of `scripts/paper_picks.py` (after the digest, stderr only, failures
never fail the picks job). Fixtures in other divisions are not scored and not shown; the scoring
scope is `desk_divisions` in `config/leagues.toml` (the paper desk and the panel share it, so a
run bets and displays the same leagues).

**Champions League is not supported**: football-data.co.uk publishes no CL results/fixtures/odds, so
there is no training data, no fixture prices and no `predict_match` target for it.

The home PC is often off before ~11:00, so the Hermes cron jobs are: picks `30 11,19 * * *`, settle `0 17 * * *`, weekly `0 12 * * 1`, plus an hourly `5 * * * *` catch-up job that runs picks/settle once if the last success (`data/last_run.json`, written by the wrapper `fedge_paper.py`) is older than 12h / 26h.
