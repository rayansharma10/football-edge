# Review: P6 paper desk (ledger, risk, picks/settle/weekly)

Reviewed: 6ccb3b4 (P6 desk v0) + 171c035 (explicit `[shadow]` bar + whitelist-format fix);
`HEAD == origin/main == 171c035` when the review started.
Scope: `src/fedge/paper/*`, `scripts/paper_picks.py|paper_settle.py|paper_weekly.py`,
`tests/test_paper.py`, `config/limits.toml`, `config/strategy.toml` (`[shadow]`),
`reports/weekly/2026-10-06.md`, plus the P6-touched shared code (`gbm.fit_live`, the ratings
sweep cut change, the `stats` docstring). Reviewer: independent audit; nothing under review was
written by this card.

Method: read the code first, then re-ran the desk on the real data and recomputed the numbers;
ruff + pytest; grep for order-placement/execution code; cross-checked the fixtures parser against
the independently ingested kickoff times; recomputed settlement P&L and log-CLV from the raw odds;
sha256-verified the report stamps; exercised the settle + weekly path end to end on a scratch copy
of the real data (the committed ledger holds 0 bets, so those paths had never run on real data).

## Verdict: CHANGES NEEDED (2 major, 9 minor, 9 nits; no blocker, no leakage path found, no
order-placement code, and the maths that is there is correct).

Item by item against the card:

| card item | result |
|---|---|
| idempotency | **holds.** Bets dedupe on `(match_key, market, selection)` via the unique index and a pre-read; settlements dedupe on `bet_id`; snapshots on `(snapshot_day, match_key, market, selection)`. Verified by re-running the real picker (0 rows added, silent) and the scratch settle (2nd run: "no unsettled bets"). |
| no order-placement / live code | **clean.** `grep -rnE 'place_?order|placeBet|betfairlightweight|flumine|api.betfair' src scripts tests` returns no code hit (only docs and the `pyproject.toml` deps). No `fedge.paper` module imports an HTTP or exchange client; `limits.LIVE = true` raises `RuntimeError` at load (risk.py:48-56). |
| risk caps + kill switch enforced | **enforced, with one design gap** (m3). Stake and daily caps are applied in `plan_bets` (picks.py:137-140) off `risk.stake_units`; `data/KILL` and the realised-drawdown switch zero `remaining` (risk.py:110-139, paper_picks.py:139). The drawdown is computed over **all** settled bets, paper *and* shadow. |
| timezone handling | **holds, independently cross-checked.** UK wall-clock -> `Europe/London` -> UTC (fixtures.py:148/246 -> `uk_local_to_utc`), tz-aware UTC in every ledger column, Sydney only in digests/reports. 21 of the 22 modelled fixtures in today's real file reproduce the *ingest's* independently computed kickoff (date **and** time) exactly; the 1 mismatch is a football-data date change, not a tz bug (m1). Digest check: 2026-10-02 18:30Z -> "Sat 03 Oct 04:30 AEST" is correct (Sydney was still on AEST). |
| settlement / CLV maths | **correct arithmetic, wrong metric for this project** (M1). I reproduced every stored settlement from the raw odds: `closing_price` = the quoted `BFEC*` close, `log_clv = log(price_taken * p_fair)` with `p_fair` the power de-vig of the complete `BFEC` trio, `pnl = stake*(price-1)*(1-commission)` on a win / `-stake` on a loss. All exact. But `p_fair` is combined with the **raw** price while the project's gate variable is **net** CLV. |
| stdout contract | **holds.** Real run `uv run python scripts/paper_picks.py --cached-fixtures --no-refresh`: exit 0, **stdout 0 bytes**, one ops line + one "no upcoming fixture" line on stderr. Scratch settle: digest on stdout, ops line on stderr; second run silent. |

### Independent reproduction (numbers, not claims)

* `uv run ruff check .` -> clean. `uv run pytest -q` -> **158 passed in 86.96s** (matches the handoff).
* Real picker run: `fixtures_total=46, modelled_fixtures=22, modelled_upcoming=0,
  unmodelled_fixtures=24 (EC, SC1, SC2, SC3 - absent from `config/leagues.toml`),
  new_league_fixtures=16`; 0 bets / 0 snapshots. The committed ledger is empty
  (`bets 0, settlements 0, snapshots 0`) and `reports/weekly/2026-10-06.md` matches it.
* Dry run `--dry-run --cached-fixtures --no-refresh`: **44s wall**, 22 fixtures scored, 110
  snapshot rows, 1 shadow bet planned (`SP2 Eldense v Oviedo over @2.76, edge +5.8%, 8.79u`).
  I re-derived the pair from the digest alone: `edge = p*net_odds(2.76,0.06)-1` and
  `stake = 0.25*kelly*1000` are jointly consistent at `p = 0.3987` -> +5.82% and 8.79u. Runtime is
  far inside the 20-min budget (22 fixtures -> 44s; the two live fits dominate, n_train 92872/94213).
* Report stamps verified: `sha256(config/strategy.toml)[:16] = b835bfaf32fd67d9` and
  `sha256(reports/gate0.md)[:16] = a1ad50737a0ab355`, both exactly as printed in the weekly report.
* Real-data settlement replay (scratch `--data-dir` copy of `data/interim/*.parquet`, 4 synthetic
  bets on Sabadell v Andorra and Cordoba v Tenerife): `paper_settle.py` settled 4/4, preferred the
  `BFEC*` close over `AvgC*` as documented, and every stored `pnl` / `closing_price` / `log_clv`
  matched my own recomputation from `odds.parquet` to 1e-9. `paper_weekly.py` on the same ledger
  reproduced the flat ROI (-28.00% on 40u) and the mean log-CLV (-10.13%) exactly.

## MAJOR

### M1. The ledger stores *raw* CLV while the project's decision variable is *net* CLV
`src/fedge/paper/settle.py:116` (`log_clv(float(bet["price_taken"]), p_fair)`) and the module
docstring (settle.py:8-12). The backtest judges the same strategy on `clv_net`
(`src/fedge/backtest/strategy.py:94`; `scripts/run_strategy.py:526`:
`clv_pos = n_sel > 0 and ov["clv_net_lo"] > 0`, which feeds `gate_pass` at :579), `gate0.md`
reports net CLV as the headline, and `stats.py:9` explicitly instructs: "For exchange prices pass
commission-adjusted odds (see `net_odds`)" - the desk trades BFE, an exchange price, and passes the
raw one.
Why it matters: the desk's number is **optimistic by ~3.0 to ~5.1 pp** (my computation: raw-net =
+3.05 pp at price 2.0, +3.90 at 2.76, +4.60 at 4.0, +5.13 at 6.0). That bias is the same order as
the AGENTS "CLV > 3% = assume leakage" alarm and as the Gate 1 bar ("mean log-CLV > 0 with CI
excluding 0"). Measured live on my scratch replay: at `price_taken = 3.00`, raw +39.12% vs net
+35.07% (-4.05 pp). The first live Gate 1 read would be on a metric the backtest never used.
Fix (either, but pick one and label it): store both, as the backtest does -
`log_clv(net_odds(price_taken, commission), p_fair)` into `log_clv` plus a `log_clv_raw` column -
and report the net one in the digest/weekly; or keep raw and state in the weekly caveats *and* in
the Gate 1 definition that the live number is raw and sits ~4 pp below the Gate 0 figure.

### M2. In paper mode an empty Gate 0 whitelist means "every division" instead of "no division"
`src/fedge/paper/picks.py:71-89` (`market_whitelist` returns `[]` for a market whose whitelist list
is empty) and `picks.py:106-109` (`if wl:` -> no division filter at all). The backtest reads the
same config the other way round: `src/fedge/backtest/strategy.py:306-308`
(`whitelisted = fold["div"].isin(cfg["whitelist"])`) -> an empty whitelist selects **nothing**, and
`scripts/run_strategy.py:396-398` writes `qualified = bool(cfg["whitelist"])`, a field the desk
reads past.
Why it matters: it is not hypothetical. `reports/gate0_fold_configs.csv` shows the exact shape in
the committed in-fold history - the `exch` folds 2022/23..2026/27 carry a real threshold (0.02/0.04)
with an **empty** whitelist (n=175-585 history bets), while the `max` folds show the same rule
producing whitelists of 1 to 12 divisions. Today the over-betting is masked only because the other
path into an empty whitelist (`cfg is None`, run_strategy.py:384-388) writes `threshold = 1.0`, an
unrelated sentinel. As soon as Gate 0 passes with a market in the "threshold but no qualifying
division" state - or anyone hand-edits `threshold` - the desk bets that market in every division the
model can score, while the backtest it is supposed to mirror bets none. Nothing in `test_paper.py`
covers this: `test_gate0_whitelist_accepts_the_rendered_pair_format` only exercises non-empty
whitelists.
Fix: in paper mode, treat an empty whitelist (or `qualified = false`) as "matches nothing" - e.g.
return a value that cannot match a `div`, or skip the market in `candidate_bets` - and add a test
using the committed-config shape (`qualified = false`, `threshold = 0.06`, `whitelist = []`) that
asserts zero candidates.

## MINOR

* **m1. The settlement join can silently never fire.** `match_key` is `div|YYYY-MM-DD|home|away`
  built from the *fixture file's* date, but `paper_settle` joins on the *results file's* identity
  (paper_settle.py:137-141, `results.index`). In today's real file 1 of the 22 modelled fixtures
  disagrees: `SP2 Sabadell v Andorra` is `03/10/2026 17:30` in `fixtures.csv` and
  `2026-10-04 20:00` UK in the season CSV (the fixture moved). A bet on it can never settle: the row
  is quietly counted in `skipped` and stays open forever (no void/reschedule path, and a match whose
  result football-data never fills is indistinguishable from "not played yet"). Measured join rate
  today: 21/22 on modelled past fixtures. Fix: fall back to resolving `(div, home, away)` within a
  few days of the stored `date`, mark irreconcilable bets `void`, and surface "open bets older than
  N days" in the weekly report.
* **m2. Kickoff is not re-checked at insert time.** `now` is captured once at `paper_picks.py:76`,
  then the ingest refresh, the feature build and two live model fits run before `insert_bets`
  (paper_picks.py:159-160): 44s measured on 22 fixtures, minutes on a full card. A fixture that
  kicks off inside that window is still `upcoming` (paper_picks.py:90) and gets a bet and a snapshot
  recorded after kickoff. Neither `candidate_bets` nor `plan_bets` guards on kickoff at all - the dry
  run happily plans a 5.8% edge shadow bet on Eldense v Oviedo, a match played four days earlier
  (the real run's `upcoming` filter is the only thing preventing it). That is ledger contamination
  in a system whose whole purpose is timestamped evidence. Fix: pass a fresh `now` (or re-filter
  `kickoff_utc > now`) immediately before planning, and assert in `plan_bets` that every planned bet
  has a future kickoff.
* **m3. The kill switch mixes shadow with paper P&L, and the stake never tracks the bankroll.**
  `risk.realised_drawdown` (risk.py:95-107) is fed `ledger.settled_frame(conn)` - every settled bet,
  any mode - and `daily_state` blocks on it (risk.py:125-131). While Gate 0 is unpassed the entire
  ledger is shadow, so hypothetical losses can freeze the desk and a *paper* drawdown cannot be seen
  separately. Separately, `stake_units` sizes off the constant `BANKROLL0 = 1000` (risk.py:34,89-92)
  although risk.py:11-12 says "fraction of the bankroll", so the desk never compounds and never
  shrinks - while `paper_weekly.metrics` presents a compounding Kelly ROI and max drawdown
  (paper_weekly.py:57-76) that the ledger does not follow. Fix: compute the kill switch on paper
  settlements (show shadow separately), or state in the report that stakes are a fixed 1%-of-initial
  Kelly and label the weekly Kelly columns as counterfactual.
* **m4. The weekly report stamps a config it did not load.** `paper_weekly.py:111-112` hashes
  `ROOT/config/strategy.toml` and `ROOT/reports/gate0.md` while `strategy` comes from `--config`
  (paper_weekly.py:212). The committed report's stamps verify correctly, but any non-default
  `--config` produces a report whose two audit stamps describe different files. Fix: hash the paths
  actually loaded.
* **m5. Two sources of truth for the Kelly fraction.** `config/strategy.toml:9` (in-fold selected) is
  `0.1`; `config/limits.toml:5` (what the desk stakes with) is `0.25`. Whenever the 1% cap is not
  binding the desk stakes 2.5x the fraction the backtest selected, and the weekly report labels it
  `Kelly(0.25)`. Fix: read the strategy value (or assert the two agree at load).
* **m6. `load_strategy` validates keys but not the mode/gate agreement** (picks.py:35-50): a `mode`
  other than `paper`/`shadow` is only caught by the SQLite CHECK after a full model run
  (ledger.py:98), and `gate0_passed = true` with `mode = "shadow"` silently runs the weaker shadow
  rules. `risk.load_limits` likewise accepts out-of-range values (negative caps,
  `kelly_fraction > 1`). Fix: validate both in the loaders and fail fast.
* **m7. A missing Gate 0 report silently degrades the audit trail.** `picks.file_hash` returns `""`
  for a missing file (picks.py:53-56) and `paper_picks.py:136` stores it, so bets can be written with
  an empty `gate0_hash` (the column is NOT NULL but `""` passes). Fix: raise when the Gate 0 report
  or the config is absent.
* **m8. One bad closing price aborts the whole settlement run.** The close path has no `> 1.0` guard
  - `settle.closing_probabilities` (settle.py:63-84) -> `devig` raises
  `ValueError: all decimal odds must be > 1.0` for any row <= 1.0 - whereas the entry path does check
  it (fixtures.py:180). Latent today (min BFE/Avg close price in `odds.parquet` is 1.01; 0 rows <= 1.0
  in 428,036), but a single 0/1.0 quote from football-data would kill settlement for every open bet
  that day instead of skipping one bet. Fix: filter `> 1.0` in `wide_prices` and skip the affected
  match with a warning.
* **m9. SQLite concurrency and migration.** `ledger.connect` (ledger.py:152-161) uses the default
  rollback journal and 5s busy timeout, and `daily_state` reads the day's count before a run that
  ends minutes later: two overlapping cron runs can each hand out the full 20-bet allowance (the
  unique index only prevents the *same* triple). `CREATE TABLE IF NOT EXISTS` also silently keeps an
  older schema - a column added by a later card breaks an in-season ledger with "no such column" and
  no migration path. Fix: WAL + `busy_timeout`, re-check the daily allowance inside the insert
  transaction, and keep a `PRAGMA user_version` schema check.

## NITS

* n1. `fixtures.parse_fixtures` / `parse_new_league` (fixtures.py:135-139, 231-235) treat a
  missing/blank `Time` as **23:59 UK**, which keeps such a fixture "upcoming" for the whole day; the
  docstring only promises to drop rows without a date/home/away. The ingest already carries a
  `time_unknown` flag (matches.parquet) - reuse it, or drop those rows.
* n2. `picks.format_digest` uses `pd.Timestamp(now_utc).tz_convert(SYDNEY)` (picks.py:214) and so
  raises `TypeError` on a naive timestamp, while the module's own `_iso` localizes naive input
  (picks.py:234-239). Make both accept either.
* n3. `ledger.insert_bets` pre-reads the existing triples and then inserts (ledger.py:164-184): a
  duplicate *within* one batch, or a concurrent insert between the read and the write, raises
  `sqlite3.IntegrityError` and aborts the run after snapshots were committed - the docstring promises
  duplicates are skipped. Use `INSERT OR IGNORE` + re-query, or catch per row.
* n4. Nothing tests the "no order-placement code" rule (AGENTS.md 2). A cheap static test scanning
  `src/fedge/paper` + `scripts/paper_*.py` for `placeOrders|placeBet|flumine|betfairlightweight|
  listMarketBook` would lock it in.
* n5. `paper_weekly.fmt` formats every number with an explicit `+` (paper_weekly.py:81-84), so
  drawdowns print as positives: my scratch run reported "Max drawdown: Kelly +2.00% / flat +2.00%"
  for a ledger down 11.2u (the 2% value itself is correct). Give drawdowns their own format.
* n6. `metrics`' "flat" simulation uses `simulate_bankroll`'s default `flat_stake = 10.0`
  (paper_weekly.py:65) and ignores `max_stake_frac`; label it as a fixed 10u counterfactual or feed
  the ledger's actual stakes.
* n7. `fixtures.market_rows` re-indexes the raw CSV frame with `raw.loc[fixtures.index]`
  (fixtures.py:169), relying on `parse_fixtures` having kept `raw`'s positional index. Correct today,
  silently mis-aligning prices if anyone adds a `.reset_index()`. Join on a row id or assert the
  invariant.
* n8. The snapshot claim ("the live CLV of the model's view ... on the whole fixture list",
  ledger.py:15-20) is limited in practice to modelled divisions with a complete book and a future
  kickoff: 24 of today's 46 rows (EC, SC1, SC2, SC3) and the 16 new-league fixtures are counted but
  never snapshotted. Say so in the weekly caveats.
* n9. `wide_prices` silently takes `aggfunc="first"` on duplicate rows (settle.py:59). The ingest
  dedupes on `(match_id, bookmaker, market, selection, phase)` so this is fine today, but the
  assumption is undocumented.

## Phase 6 acceptance items this card does not cover (status, not a defect of the desk code)

`docs/IMPLEMENTATION_PLAN.md` P6 is five cards. What is built here is 6.2 (ledger) and most of 6.3
(risk + no order-placement code), plus the scripts of 6.4. Still missing, and needed before the desk
can produce Gate 1 evidence:

* **6.1 Betfair delayed-key client** - no login, no `listMarketCatalogue`/`listMarketBook`, no
  T-24h/T-60m/T-1m snapshots. The desk trades football-data's `pre` close instead, with no depth and
  an unverified collection time, so 6.2's "only if depth >= stake" is not enforced anywhere. The
  weekly report's own caveat admits the price time is unverified. Gate 1's "scored against the
  Betfair close" currently rests on football-data's `BFEC*` column, which the settle path does use
  correctly.
* **6.4 cron registration + Slack digest** - the scripts have the stdout contract and exist as
  zero-LLM entry points, but no Hermes/no-agent jobs are registered and nothing posts to `#picks`.
* **6.5 watchdog** - no failure alerting for ingest/snapshot/schema breaks.

## What I could not verify

* The real collection time of football-data's `pre` price (inherited P5 assumption). The desk's
  `seen_utc` is the download time, so "T-24h" in `config/strategy.toml` remains an assumption, not a
  measured fact.
* Any *live* bet/snapshot row: the committed ledger holds 0 bets and 0 snapshots because today's
  `fixtures.csv` has no upcoming modelled fixture (its newest entry is 2026-10-05), so the picker's
  insert path, the cap interaction with real data and the snapshot dedup have only been exercised by
  tests and my scratch replay - not by a real desk day.
* The model's live behaviour on a real upcoming card (both live fits' edge distribution, and whether
  the shadow threshold yields a sane number of bets): 0 upcoming fixtures today.
* Depth/liquidity realism: unmeasurable from football-data, as the P5 review already noted.

## Working-tree note

`reports/v1_baseline.md`, `reports/v2_main.md` and `scripts/run_main.py` were modified in this shared
checkout by the concurrently running card t_413828bc while this review ran; this review committed only
`reports/reviews/review_p6.md`. Related: 6ccb3b4 changes the shared sweep cut in
`src/fedge/features/ratings.py` (elo/pi `ko[ptr] <= cut[i]` -> `<`), which shifts every P3-P4
baseline, so the committed `v1_baseline.md` / `v2_main.md` are stale until that regeneration lands.
Coordination, not a desk defect - but the two must not be pushed apart.

## Files changed by this review

* `reports/reviews/review_p6.md` (this file).
