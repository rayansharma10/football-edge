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

Shows mode/gate status, last picks/settle runs (STALE warning), open and settled bets, cumulative P&L/CLV, top snapshot edges and the latest weekly report; it polls `/api/summary`, `/api/open_bets`, `/api/settled`, `/api/series`, `/api/snapshots` every 60s.

The home PC is often off before ~11:00, so the Hermes cron jobs are: picks `30 11,19 * * *`, settle `0 17 * * *`, weekly `0 12 * * 1`, plus an hourly `5 * * * *` catch-up job that runs picks/settle once if the last success (`data/last_run.json`, written by the wrapper `fedge_paper.py`) is older than 12h / 26h.
