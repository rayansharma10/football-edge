# Review: Phase 0 + Phase 1 (commits 7eaa6b2, 15ae255, 498f7d1)

Reviewer bot, 2026-10-06. Scope: scaffold, football-data ingest, coverage report, team normalisation, Understat xG.

## Verdict: CHANGES NEEDED
One major lookahead risk and two major data-integrity risks. No blockers. No secrets or data committed.

Run results (my own run):
- uv run ruff check . -> All checks passed
- uv run pytest -> 22 passed in 1.79s (data-dependent tests ran, not skipped, because local data/ exists)

Checked and OK:
- UK local -> UTC uses Europe/London tz_localize (football_data.py:73); BST/GMT and both 2025 transition days tested. Autumn-fold ambiguity resolves to GMT, spring gap shifts forward.
- No data/, .env, .duckdb, parquet or raw files are tracked (git ls-files; .gitignore covers them). No secrets in source. (I did not open the .env files; tooling blocked that.)
- Pinnacle stale flag: real data has 36,465 of 868,613 PS rows flagged; first stale kickoff 2025-07-25, last unflagged 2025-06-01; also covers P->PS over/under and AH, and PSC* closing.
- Real parquet sanity: FTR agrees with FTHG/FTAG on every played row; zero odds rows with available_at > kickoff; 3 null-FTR rows are unplayed/abandoned fixtures (2 G1, 1 F2).
- No order-placement code; LIVE=false. Stub modules are 1-line placeholders.
- Understat cached with 3s delay; football-data 1s delay, redirects followed, cache. No forbidden sources.

## MAJOR

### M1. pre odds available_at = kickoff - 24h is NOT conservative for 2019/20+ (lookahead risk)
src/fedge/ingest/football_data.py:9-12, 38, 269
The docstring says kickoff-24h is a late (conservative) estimate because odds are collected "up to ~3 days before". But docs/research/M2-data-tooling.md:73,364 says the fixtures file (source of pre odds) is refreshed Fridays <=17:00 UK (weekend games) and Tuesdays <=13:00 UK (midweek). Real lead can be far below 24h:
- Saturday 12:30 kickoff: collected Fri <=17:00 = ~19.5h lead, but available_at says Fri 12:30 (up to 4.5h too early).
- Tuesday 19:45 kickoff: collected Tue <=13:00 = ~6.75h lead, but available_at says Mon 19:45 (up to ~17h too early).
Heuristic count on the real data (my calc; assumes the M2 cadence is the true collection time): of 47,332 matches from 2019/20, 18,303 (38.7%) get a pre available_at earlier than the latest plausible collection (Saturday 14,800; Tuesday 3,491; Wednesday 12). A bet_time between those instants would see odds that did not yet exist. 2012/13-2018/19 are safe only by accident because every kickoff there is the 23:59 placeholder.
Fix: drop the flat 24h. Use available_at = the latest scheduled collection (Fri 17:00 / Tue 13:00 UK, converted to UTC) preceding kickoff, or kickoff - 1h until real snapshot timestamps are verified. Update the docstring and replace test_available_at_semantics (it re-implements the formula and cannot fail). Mark as UNVERIFIED in the report.

### M2. Understat cache never refreshes the in-progress season
src/fedge/ingest/understat.py:54-56
The "if path.exists(): return cached" check applies to every season including the current one. data/raw/understat/*/2026.json (2026/27, in progress) was written 2026-10-06 and will be treated as final forever; new matches silently never get xG. Fix: always re-download current_year (as football_data.fetch_one does) or add a max-age check.

### M3. football-data cache can freeze a partial file; a transient bad response writes a permanent .missing marker
src/fedge/ingest/football_data.py:98-131
(a) finished = season != current is evaluated against config at run time. A file downloaded while season X was current is cached forever once config moves current to X+1, without a final refresh. Fix: record a .final marker only when downloaded after season end, otherwise refetch once.
(b) Line 122: a 200 that is not CSV (maintenance page, WAF challenge) writes <div>.missing for finished seasons and is never retried. Only a real 404 should create the marker; treat non-CSV 200 as error.
(c) Line 130 write_bytes is not atomic; a crash leaves a truncated file that is then cached forever. Write to .tmp then replace, and check the header first.

## MINOR

- m1. understat.py:118-125,131: Understat datetime is UTC, not UK local (observed: EPL 2024 opener 2024-08-16 19:00:00 = 20:00 BST); the docstring says "naive". The join compares that UTC date with the football-data UK-local date, which disagrees for kickoffs 23:00-00:00 UTC in BST; the 3-day fallback masks it. Compare UTC to UTC, and for time_unknown rows (all 2012/13-2018/19) take the real kickoff from Understat instead of 23:59.
- m2. understat.py:158: xG available_at = football-data kickoff + 3h. For 2014/15-2018/19 kickoff is the 23:59 placeholder, so availability is ~03:00 next day: safe but crude. The 3h publish delay itself is unmeasured.
- m3. understat.py:65-67: sends cookies beget=begetok and X-Requested-With (mimics the XHR/anti-bot cookie of the site itself). The sleep on line 67 is skipped when requests.get raises, so repeated failures hit the host with no delay. No backoff on 429/5xx. No record of a robots/ToS check. Sleep before the request, back off on 429, note Understat terms as UNVERIFIED.
- m4. No pandera schema for match_xg (other tables validated in store.write_tables). Add one (unique match_id, xg >= 0, tz-aware available_at); assert understat_id is unique across both join passes.
- m5. football_data.py:143: on_bad_lines="skip" drops malformed rows silently. Log the count or fail.
- m6. football_data.py:331-332: pd.concat([]) raises an opaque ValueError if nothing was fetched. Fail with a clear message.
- m7. football_data.py:79-81: match_id hashes raw team names, so a football-data rename orphans xG/odds joins. Hash canonical ids instead, or document. (Dedup at line 200 only logs a warning.)
- m8. football_data.py:318: offline mode reports status "cached" even if the file does not exist.
- m9. football_data.py:249: prices <= 1.0 silently dropped; log a count.
- m10. config/team_aliases.csv: Understat/football-data aliases were auto-derived by date+score voting, not hand-reviewed (per handoff). 99.99% coverage makes a wrong pair unlikely, but spot-check similarly named clubs. betfair column is an empty placeholder.
- m11. reports/data_coverage.md snapshots incomplete 2026/27 data without saying so.

## NITS / test quality

- t1. tests/test_football_data.py:100-107 test_available_at_semantics re-implements the production formula (tautological) and would pass with M1 present. Use explicit expected timestamps for a Saturday 12:30 and a Tuesday 19:45 match.
- t2. test_uk_to_utc_bst_and_gmt: comment at line 66 is wrong; it does not test the ambiguous hour (2025-10-26 01:30) or the non-existent hour (2025-03-30 01:30), which is where ambiguous=/nonexistent= matter.
- t3. No tests for fetch_one (cache hit, 404 marker, HTML-200 marker, current-season refresh, UA/redirects) or _is_csv. M2 and M3 are untested; use a fake session, no network.
- t4. No tests for legacy BbMx/BbAv columns, duplicate-match drop, malformed rows, or a 2012-era file (all fixture rows are 2025/26).
- t5. Real-data Understat/team tests skip when data/ is absent (i.e. in CI on windows+ubuntu). Commit a tiny alias/Understat fixture so mapping is tested in CI.
- t6. tests/test_smoke.py (test_leagues_config) hard-codes 18 divisions and 15 seasons; breaks on legitimate config changes.
- t7. store.py:21-26 f-string table name is safe (fixed tuple); prefer a literal mapping.

## Required fixes before P2 depends on this
1. M1 + t1. This is the one that can create a fake edge.
2. M2.
3. M3 (a-c) + t3.
4. m1/m2 (use Understat UTC kickoff) before xG features are built.

## UNVERIFIED
- True collection timestamps of football-data pre odds (M1 uses the M2-doc cadence; real cadence may differ).
- Understat xG publish delay and its terms of service.
- That football-data Time is UK local for all 18 divisions (GR/TR/NL/BE/PT included).
