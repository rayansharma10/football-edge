# football-edge ⚽

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
