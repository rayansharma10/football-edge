# Review: football-edge, P6 follow-up verification + fresh whole-pipeline audit

Reviewed: HEAD e5176a6 (working tree has one uncommitted change, see N1). Card t_c2d62cb5.
Reviewer: independent, read-only. data/paper.sqlite was copied to the scratch dir before querying.

Scope note: P6 was already reviewed (reports/reviews/review_p6.md, 83d441f). I did not re-list those
findings. I checked that each fix landed, then audited the rest of the pipeline. Betting is parked
(accuracy-tracker pivot), so the live cron jobs are predict and results only; P6 findings below are
therefore latent unless betting resumes.

## Verdict: CHANGES NEEDED (0 blocker, 3 major, 7 minor, 5 nits)
No leakage path found, no order-placement code, and the settlement / CLV / de-vig maths I re-read is
correct. The majors are one ineffective earlier fix, one Gate 0 logic/honesty problem and one
data-integrity risk in the live results job.

## Real test output
- uv run ruff check .  -> "All checks passed!"
- uv run pytest -q     -> 243 passed, 74 warnings in 65.16s (warnings are third-party deprecations:
  starlette/httpx, numpy 2.5 shape setting in penaltyblog, venn_abers all-NaN slice).
- Committed ledger (copy of data/paper.sqlite): bets 0, settlements 0, snapshots 0, user_version 2.
  The settle/CLV path has still never run on a real bet; only tests and the earlier scratch replay.

## Verification of the P6 follow-ups (t_5d021d37 / t_1e550b87 / bf7e10e / a69fd3d)
| old finding | status |
|---|---|
| M1 raw vs net CLV | FIXED. settle.py:118-122 stores net log_clv and log_clv_raw; ledger._migrate recomputes old rows (raw + log(net_odds/price), arithmetic checked). |
| M2 empty whitelist = all divisions | FIXED. picks.candidate_bets skips the market when wl is empty or qualified is false (paper mode). |
| m1 moved fixtures never settle | FIXED (see m3 for a perf nit). paper_settle.match_result falls back to (div, home, away) within 3 days. |
| m2 kickoff re-check at insert | NOT EFFECTIVE, see M1 below. |
| m3 kill switch mixes modes | FIXED. Drawdown scoped to mode; shadow gated by the file switch only. |
| m5 two Kelly fractions | FIXED. risk.effective_limits = min(strategy, limits). |
| m6/m7 loader validation, empty gate0 hash | FIXED (load_strategy mode/gate agreement; gate0_hash raises). |
| m8 bad close quote aborts run | FIXED. settle.wide_prices filters <= 1.0; paper_settle.closing_info guards. |
| m9 / n3 concurrency, migration | FIXED. WAL, busy_timeout 10s, BEGIN IMMEDIATE + INSERT OR IGNORE, in-transaction daily cap, user_version. |
| n1 blank Time -> 23:59 | FIXED in fixtures.py (now 00:00 UK). Ingest still uses 23:59; match_key date is identical either way. |

## MAJOR

### M1. The kickoff re-check in plan_bets uses a stale clock, so the m2 fix does nothing
scripts/paper_picks.py:102 captures now once at start. It is passed unchanged to plan_bets
(paper_picks.py:171), whose guard is cand[kickoff_utc > pd.Timestamp(now_utc)] (picks.py:162-164).
The comment says the guard exists because model fits "can take minutes", but the value compared is
the script-start time, so it equals the upcoming filter earlier in main and can never reject anything
that filter kept. created_utc of every bet is also stamped with the start time, not the insert time.
The unit test (tests/test_paper.py:791) passes only because it hands plan_bets a later now explicitly.
Why it matters: a fixture that kicks off during the 1-5 minute scoring window is still bet and
snapshotted after kickoff, contaminating timestamped evidence.
Fix, in paper_picks.main just before planning:
    t_plan = pd.Timestamp.now(tz="UTC")
    bets = picks.plan_bets(edges, scored, strategy, limits, remaining, t_plan, cfg_hash, gate_hash, experiment)
and use t_plan for the snapshot rows too. Add a test that advances the clock between score_all and
plan_bets. (Latent while betting is parked.)

### M2. Gate 0 can never pass as coded, and several criteria are hard-coded "PASS"
scripts/run_strategy.py:680: gate_pass = all(v[0] == "PASS" for v in gates.values()) and clv_pos.
B0.8 (:662) and B0.9 (:666) are the literal string "PARTIAL", so the conjunction is false for any
data. B0.2 (:628), B0.5 (:644) and B0.10 (:675) are literal "PASS", not computed (B0.5's own text says
"The model does NOT beat the close" and still grades PASS; defensible as "reported", but it is a
label, not a test). reports/gate0.md:607 says "Gate 0 passes only if every item is PASS", which a
reader will take to mean it is achievable.
Why it matters: "Gate 0 NOT passed" currently carries no information about the strategy, since no
result could have produced a pass. mode=paper and the whitelist path in picks.py can never be
exercised by the real config, only by synthetic ones. The negative headline itself (net CLV -3.76%
[-6.10, -1.31], n=216) is real and independent of this; the gate output just should not be read as
evidence.
Fix: either (a) treat PARTIAL as "not demonstrable on history", gate on the computable items only and
list the PARTIAL items as explicit Gate 1 prerequisites, or (b) say in gate0.md that Gate 0 cannot pass
until B0.8/B0.9 change, and compute B0.2/B0.5/B0.10 from data or test results. Add a test with
synthetic all-PASS gates asserting gate_pass is True.

### M3. update_results stores any score the schedule reports, first-write-wins, with no "finished" guard
src/fedge/accuracy/results.py:61-72 emits a row whenever home_goals/away_goals are non-null in the
schedule feed. Nothing checks that kickoff + ~2h45 has passed or that the feed marks the match
finished. store.upsert_results is INSERT OR IGNORE, so a later correct score is only counted as
"conflict" and the stored one is kept. parse_fixturedownload treats HomeTeamScore is not None as
played (ingest/schedule.py:135).
Why it matters: if either feed publishes in-play scores, an unlucky run time (the catch-up wrapper can
fire at any hour) freezes a partial score as the result and silently corrupts the headline accuracy
metrics. I could not verify whether fixturedownload or openfootball ever do this, so severity is
conditional. One fetch during a live match would settle it.
Fix: require kickoff_utc + 3h <= fetched for schedule-sourced rows; on conflict prefer the football-data
value (or overwrite when the stored fetched_ts is earlier than kickoff + 3h). Add a test with a score
present and kickoff 30 minutes ago.

## MINOR

m1. Shadow-desk price fallback contradicts the config. config/strategy.toml says "never Max/Avg" and
Gate 0 found data errors in Max quotes (the guard exists only in the backtest). fixtures.SOURCES =
BFE, Avg, Max (fixtures.py:42). In the cached file 91 of 92 (fixture, market) rows are BFE and 1 is
Avg, so impact is nil today, but a Max fallback with no 1.10x guard would create phantom edges.
Fix: drop Max (or apply the same ratio cap) and make the toml string match the code.

m2. The training table silently excludes any match missing from v2_market_pre_*.parquet. Only
scripts/run_main.py writes it (mtime Oct 6 17:03; matches.parquet is Oct 7). The lgbd_xg live fit on the
paper path drops every match played since the last run_main with no warning, and S1 for calibration is
chosen after that drop (gbm.fit_live: d = d.loc[init notna]). The current price-free live path is not
affected (init_cols is None). Fix before betting resumes: derive market_pre from odds.parquet inside
build_features, or fail when recent played matches lack a prior. (I measured 1341 played matches
without a prior; most are older or unpriced, so this is a latent drift risk, not a current bug.)

m3. paper_settle.match_result rebuilds _by_teams(played) (a pass over ~94k rows) once per unmatched
bet (paper_settle.py:125/147). Harmless at current volume; hoist it out of the loop.

m4. Reschedule handling in the accuracy tracker: match_key embeds the date, so a postponed match gets a
new key. The old key's logged predictions never join to a result and the new key starts a fresh
history (d7/d3 restart). Not biased towards the model, but coverage is silently lower. Log a count of
predictions with no result after kickoff + 7 days.

m5. No void/abandon path in the paper ledger. A bet whose match is postponed past the 3-day tolerance
or abandoned stays open forever and drops out of drawdown and CLV. Add a void settlement and surface
open bets older than N days in the weekly report (carried over from the first P6 review).

m6. predictions.sqlite is not in WAL mode (store.connect sets no pragmas) while paper.sqlite is. A
dashboard read overlapping replace_upcoming's DELETE+INSERT commit can hit SQLITE_BUSY, and
telemetry._rows / app._rows catch every sqlite3.OperationalError and return [] (telemetry.py:50,
app.py:41), so the page shows "no predictions" instead of an error. Fix: PRAGMA journal_mode=WAL in
store.connect and swallow only "no such table".

m7. The dashboard stale alarm is permanently red. app.py:24 STALE_HOURS has picks 14h / settle 30h but
those cron jobs are paused (last_run.json: both last succeeded Oct 6 10:07), so any_stale is true for
the archive page forever. Drop paused jobs from STALE_HOURS or label them "parked".

## NITS
N1. Working tree: src/fedge/paper/model_state.py is modified, but git diff --ignore-cr-at-eol is empty,
i.e. a line-ending-only change (git warns LF->CRLF). risk.py is committed with CRLF. Add a
.gitattributes (* text=auto eol=lf) so Windows editors stop producing noise.
N2. ops/hermes_cron/fedge_paper.py: _record does read-modify-write on last_run.json with no lock
(predict and results crons can overlap and lose an update), and subprocess.TimeoutExpired (1500 s) is
not caught, so a hung job prints a traceback and never records the failed attempt.
N3. settle.market_result for ou25 does not guard NaN goals (NaN > 2.5 is False -> "under"), unlike 1x2.
Unreachable today (results come from the played frame), but cheap to assert.
N4. fixtures.parse_new_league still maps a blank Time to 00:00 silently and has no time_unknown flag.
N5. .gitignore covers *.sqlite but not *.sqlite-wal / -shm outside data/; fine today since all ledgers
live under data/.

## Areas checked with no defect found
- Settlement maths: pnl = stake*(price-1)*(1-c) on a win, -stake on a loss; net log-CLV uses net_odds with
  a power-devigged close; BFE close preferred, Avg fallback, source stored.
- De-vig: power solver (bisection; the arbitrage case c<1 still brackets) and Shin read correctly; rows
  are renormalised to sum to 1.
- Leakage: fixture_features sweeps only rows with kickoff before the fixture (3h embargo inside the
  sweeps) and fixture rows enter no state; fit_live trains on ko < asof - 3h; retro backfill cuts both
  played and table at the group asof and tags rows source=retro; the report headline excludes retro
  and prefers live on duplicates; prediction_log append-only and freeze rule are enforced by triggers,
  and the canonical ISO text timestamps make the string comparison valid.
- Timezone: UK local -> UTC in one place; Sydney only for display.
- Idempotency: bets unique index + INSERT OR IGNORE, settlements by bet_id, snapshots per day,
  prediction_log UNIQUE + d7/d3 partial unique indexes, results INSERT OR IGNORE.
- Dashboard: binds 127.0.0.1, ro connections with query_only, no user-controlled SQL (the only query
  parameter is an int flag), static files from fixed paths.

## What I could not verify
- Whether football-data "pre" prices are really about 24h pre-kickoff (inherited assumption, disclosed
  in gate0.md).
- Live behaviour of the settle path on a real bet (ledger empty), and of the schedule feeds during a
  live match (M3).
- I did not regenerate the walk-forward / Gate 0 numbers (hours of compute). I relied on the leakage and
  strategy tests (all green) and the earlier P2 and P3-P5 reviews.

## Resolution status (card t_95ed72cd, branch fix/review-p6-full)
Fixed: M1, M2, M3, m1, m6, m7, N1, N2 (each with a test; see the branch commits).
Notes: M2 was resolved as option (a): Gate 0 gates on computable items, B0.8/B0.9 are listed as
Gate 1 prerequisites, B0.2/B0.5/B0.10 are computed. reports/gate0.md wording was updated by hand;
the full report was not regenerated (hours of compute). N2: the deployed copy of the cron wrapper
(%LOCALAPPDATA%/hermes/scripts/fedge_paper.py) is separate from ops/hermes_cron/ and must be synced.

Open TODOs (deliberately not done in this card):
- [ ] m2 training table silently drops matches missing from v2_market_pre_*.parquet
- [ ] m3 hoist _by_teams(played) out of the per-bet loop in paper_settle.match_result
- [ ] m4 log count of predictions with no result after kickoff + 7 days (rescheduled matches)
- [ ] m5 void/abandon path for paper bets; surface open bets older than N days in the weekly report
- [ ] N3 guard NaN goals in settle.market_result for ou25
- [ ] N4 parse_new_league: flag blank Time (time_unknown)
- [ ] N5 .gitignore *.sqlite-wal / *.sqlite-shm outside data/
