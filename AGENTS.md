# AGENTS.md: rules for coding agents (dev / reviewer bots) and humans

Project: **football-edge**, a football match-prediction + paper-betting ML system. Read `docs/REPORT.md` (why) and `docs/IMPLEMENTATION_PLAN.md` (what, in cards) before working.

## Hard rules
1. **No LLM in any decision path.** Probabilities, edges, stakes and bet selection are produced by deterministic, tested code. LLMs may only write human-readable summaries of finished reports.
2. **Paper only.** Do not write order-placement code (`placeOrders`, flumine live execution) until Gate 2 is signed off by Rayan. `config/limits.toml` keeps `LIVE = false`.
3. **No lookahead.** Every feature is built through `fedge.features.asof` and may only use rows with `available_at < bet_time`. New features need a leakage test. The regime-reversal and determinism tests must stay green.
4. **Walk-forward only.** No random train/test splits anywhere. Thresholds, Kelly fraction, decay and league whitelists are chosen in-fold.
5. **UTC everywhere** (tz-aware datetimes). Convert to Australia/Sydney only for display.
6. **Secrets** live in `.env` (gitignored): Betfair username/password/app key/cert paths, API keys. Never print or log them.
7. **Data is never committed.** `data/` is gitignored. Reports in `reports/` are committed (markdown + small PNGs).
8. **Respect sources:** polite rate limits and caching for football-data.co.uk and Understat; follow redirects (`curl -L`). **Do not scrape** Transfermarkt, Sofascore, FotMob or WhoScored.
9. **Known data facts:** Pinnacle odds in football-data.co.uk are stale from 23/07/2025 (flag, don't use as reference after that); Betfair Exchange closing (`BFEC*`) exists from 2024/25; Club Elo API is down (compute ratings ourselves).
10. **Commission:** model Betfair AU at 6% of net market winnings.

## Tooling
- Python **3.13** via `uv` (`uv sync`, `uv run ...`). No global pip.
- `uv run ruff check . && uv run pytest` must pass before any commit.
- Seeds fixed; re-runs must be byte-identical for the same inputs.
- Windows is the primary runtime (home PC, Hermes cron); keep paths `pathlib`-based.

## Workflow
- One Kanban card ≈ one branch/PR. Update the card's "done when" evidence in the PR description.
- Gate reports go in `reports/gateN.md`, including runs that lost money.
- If a result looks too good (e.g. ROI > 5% or CLV > 3%), assume leakage and investigate before reporting.
