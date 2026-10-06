# Review: Phase 2 — market layer, metrics, as-of + walk-forward harness, stats (commit b2b3c29)

Reviewer bot, 2026-10-06. Scope: market.py de-vig, report/metrics.py, features/asof.py,
backtest/walkforward.py, backtest/stats.py, the new tests and reports/devig_sanity.md.
Baseline: b2b3c29 (== origin/main == the tip of the P2 diff a605546+b2b3c29). All line
references are to the COMMITTED files at that commit.

## Verdict: CHANGES NEEDED

No maths defects: I reproduced the de-vig, the metrics and the whole A/E report independently
below. The two majors are leakage-guard gaps — the shipped numbers are correct, but the
gateway promise "fn cannot see the future" (asof.py:51) does not hold for the bet row, and the
embargo guard has no test that can fail (proved by mutation).

My runs at b2b3c29 (clean tree, data/ present):
- uv run ruff check . -> All checks passed
- uv run pytest -> 65 passed in 13.2s (65 collected)
- Clean export of the same commit (git archive b2b3c29, no working-tree edits) -> 63 passed,
  2 skipped (test_teams_understat.py:44 "data/ not ingested", :52 "Understat cache not
  present" — data/ is gitignored)

## Verified independently (not taken on trust)

De-vig. Re-solved both solvers with scipy.optimize.brentq on their own defining equations
(sum(r**c)=1; sum((sqrt(z^2+4(1-z)r^2/S)-z)/(2(1-z)))=1):
- power param matches to <=5e-13 and shin to <=4.6e-13 over 7 worked examples incl.
  [2.7,2.3,4.4], [1.5,4.2,7.5], [1.9,1.95], [1.05,11,21], [1.10,8,15].
- Worked example [2.7,2.3,4.4]: S=1.03242571, power c=1.030913239316 ->
  p=[0.35917110,0.42373075,0.21709815]; Shin z=0.016236442856 ->
  p=[0.35934392,0.42324385,0.21741223]. 2-way [1.9,1.95]: c=1.058640240273,
  Shin z=0.039136570785.
- Invariants on every example: sum p == 1 (<=2.2e-16), 0 < p <= r, c>1 / z>0 for overround.
- multiplicative matches the penaltyblog docstring values [0.35873804, 0.42112726, 0.2201347].
- 2-way Shin is EXACTLY additive removal r - (S-1)/2 (max |d| = 1.4e-16 over 3 cases); the
  test only asserts it at atol=1e-3 (see m8a).
- Degenerate/extreme inputs return finite normalised results with no exception:
  [3.0,3.0,3.0], [1.05,1.05,1.05], [1.001,1.001], [1000,1000], underround [3.5,3.5,3.5].
  See m4/m5 for the two residues of this (zero probabilities for absurd books; undocumented
  underround branch).

Metrics. rps == a brute-force implementation of 1/(k-1)*sum_i<k (cumP-cumO)^2 on a hand case
(0.13375); log_loss/brier match hand values; ece reproduces an independent equal-width 10-bin
count-weighted computation exactly (0.010815072812057307, n=5000); calibration_slope recovers
a known slope on 40k simulated rows (test).

CLV. log_clv(1/p_fair, p_fair) == 0 exactly (<=1.1e-16) for all three de-vig methods, so
betting at the fair close is the fixed point as documented. Raw-close check below (n3).

The A/E report. reports/devig_sanity.md regenerates BYTE-IDENTICALLY (sha256
9289c5df500f93a817fbb5b98e678dfae623930eb58f859f12b60c67f36d10b0) — I ran a copy of
scripts/devig_sanity.py against copies of data/interim/*.parquet in a scratch tree, so the
repo was not touched. I then recomputed A/E with my OWN selection, de-dup, join and a
scipy-brentq power de-vig (no fedge.market, no script import): 114/114 rows of the report
(mult + power for 1X2 PS, 1X2 BFE, OU25 PS) match on n and A/E to 3 dp, 0 mismatches.
Pooled: 1X2 PS n=86305 0.996/1.013/0.994 (power), 0.999/1.009/0.994 (shin), 1.004/1.003/0.992
(mult); 1X2 BFE n=13854 0.992/1.014/0.999 (power); OU25 PS n=39548 1.005/0.995 (all methods).

Which leakage tests actually catch leaks (mutation testing). I copied src/ + tests/ per mutant,
applied one mutation, ran the P2 test files with PYTHONPATH forced onto the mutant tree, and
compared against an unmutated control copy (43 passed, 0 failed, so the harness is clean):

| mutation | caught? |
|---|---|
| asof strict < -> <= (asof.py:31) | YES - test_a_future_row_is_ignored_and_detected |
| build_features bypasses asof (asof.py:56) | YES - same test |
| RPS without the 1/(k-1) factor (metrics.py:33) | YES - both RPS tests |
| CLV ratio inverted (stats.py:26) | YES - both CLV tests |
| bootstrap ignores clusters (stats.py:47) | YES - test_bootstrap_seeded_and_clustered |
| feature eats its own target, +1.0*gd | YES - shuffled-target AND regime-reversal |
| feature eats its own target, +0.05*gd | YES - shuffled-target canary only |
| feature eats its own target, +0.01*gd | NO - 43 passed |
| EMBARGO 3h -> 0 (walkforward.py:16) | NO - 43 passed |
| Shin upper bracket 0.999 -> 0.5 / 10.0 (market.py:84) | NO - 43 passed |

So the non-leak plumbing is well covered; the three uncaught mutants map to M1, M2 and m8d.

Housekeeping. config/limits.toml LIVE=false untouched (git diff f7f9174..HEAD -- config/ empty);
no .env, key, cert, data/, .duckdb or parquet tracked; the diff is 11 files (+1305/-4) and
touches nothing outside the P2 scope; no network calls, no eval/pickle, no order-placement code
in the P2 modules. penaltyblog is a declared runtime dep that P3 also uses in src
(ratings.py:32, dixon_coles.py:30), so no dependency-hygiene problem.

## MAJOR

### M1. build_features hands the bet row (including the target) to the feature function; nothing can detect it
src/fedge/features/asof.py:49-56. The docstring (line 51) claims "fn cannot see the future",
but fn receives the whole bet row — for a real bet frame that row carries FTHG, FTAG, FTR (and,
if odds are joined on, the same match closing prices). assert_no_leak only inspects
available_at of the FEATURE table, so it can never see this leak.
Evidence: a probe fn that returns bet["FTHG"] returns 3.0 for a 3-0 match, with no error and no
warning from any guard. The shuffled-target canary (tests/test_leakage.py:158-166) only fires
above ~0.05*goal-difference — at 0.01*gd every P2 test passes. So AGENTS rule 3 ("new features
need a leakage test") currently rests entirely on each feature author, and this repo own suite
proves it can miss a real target leak.
Fix: narrow what the gateway exposes, e.g.
fn(asof(table, bet[bet_time_col], col), bet.drop(columns=[c for c in OUTCOME if c in bet]))
with OUTCOME = ("FTHG","FTAG","FTR","HTHG","HTAG","HTR"), passing only match_id, home, away,
kickoff_utc, bet_time. Add a test that a target-reading fn cannot be built (or that
build_features output is invariant to permuting the target columns) — that is the one leak class
the suite cannot catch today.

### M2. The embargo guard has no test that can fail
src/fedge/backtest/walkforward.py:16,32,48,57. Mutating EMBARGO from 3h to zero leaves all 43
P2 tests green (verified above), because both fold tests only assert strict train/test ordering,
which still holds without an embargo: train is selected by kickoff < test_start - embargo, so
embargo=0 merely admits training rows whose result was not knowable 3h later (e.g. a match
kicking off 1h before the first test match). The guard is correct today; nothing stops it
regressing silently, and it is the project #1 rule.
Fix: add a fold-invariant test, e.g. for every fold assert
fold.test_start - matches.kickoff_utc.iloc[fold.train_idx].max() >= EMBARGO and
set(train_idx) & set(test_idx) == set(), plus one planted case with a match 1h before test_start
that must be excluded.

## MINOR

- m1. run_walkforward trusts positional indices — walkforward.py:79. Folds hold positional
  train_idx/test_idx; if the frame is reordered or filtered between fold construction and use,
  the indices silently re-point (verified: same folds against a shuffled frame gave train_max
  index 9 / test_min index 0, no error). Fix: store a cheap fingerprint (len + first/last
  match_id) in Fold and assert it in run_walkforward before slicing.
- m2. Commission is double-charged if you follow the module docstring — stats.py:7 tells callers
  to "pass commission-adjusted odds (see net_odds)", but simulate_bankroll already charges
  commission per market on net winnings (stats.py:110). Verified: price 3.0 -> profit 18.80; the
  same bet with net_odds(3.0)=2.88 -> profit 17.672 (commission taken twice). Fix: reword —
  simulate_bankroll/kelly_fraction take RAW prices; net_odds is only for CLV/edge arithmetic
  where no commission is applied downstream.
- m3. Bankroll defaults contradict config/limits.toml — stats.py:72-73 use max_stake_frac=0.05
  and kelly_mult=0.25, while limits.toml has max_stake_frac=0.01 (5x smaller), kelly_fraction=0.25,
  drawdown_kill=0.2, max_bets_per_day=20. Flat mode ignores max_stake_frac entirely and there is
  no kill switch: verified bankroll0=100 with 30 x 50 losers -> final -1400. Gate reports built on
  the defaults would report a risk appetite the config forbids. Fix: read limits.toml in the
  caller (or require explicit args), record the caps used in every gate report, and either cap
  flat mode or state in the docstring that it is an uncapped arithmetic ledger.
- m4. Solvers never check convergence and renormalise anyway — market.py:66 and market.py:92
  divide by the row sum, so a failed solve is masked as a vector that sums to 1. Verified: for
  [1.0001,1.0001,100] power returns c=6931.8 and min(p)=0.0 exactly — a zero probability that
  would give inf log-loss/edge downstream. Fix: compute |sum(p)-1| before normalising and
  raise/warn above ~1e-9; floor p at ~1e-12 or reject booksums above ~2.
- m5. Underround books (booksum < 1, exchange-style) are undocumented — market.py:75-93. Shin
  has no interior root for S < 1, so the bisection degenerates to z~0 and returns p_i = r_i/sqrt(S),
  i.e. probabilities ABOVE raw implied: verified +16.7% at S=0.857 ([3.5,3.5,3.5]), +0.05% at
  S=0.999 (Betfair-like). power (c<1) and multiplicative inflate the same way. Harmless at
  realistic exchange booksums (BFE A/E stays 0.99-1.01) but surprising, and the docstring only
  says an implausible param flags bad odds. Fix: document the S<=1 branch; consider returning
  p = r (or multiplicative) for it.
- m6. Metric input validation — metrics.py:16-22/33: k=1 gives nan plus a RuntimeWarning
  "invalid value encountered in divide" instead of ValueError; an out-of-range y raises
  IndexError (metrics.py:24) rather than ValueError; probs rows that do not sum to 1 are accepted
  silently; calibration_slope does not validate y in {0,1}. Fix: validate k >= 2, 0 <= y < k and
  np.allclose(p.sum(1), 1) in _prep.
- m7. build_features is O(bets x table) — asof.py:54-56 runs a full-table boolean mask per bet:
  2.5 s for 2000 bets x 200k rows, i.e. roughly 2 minutes per feature pass at the ~86k-bet
  walk-forward this repo is heading for. It is the mandated single gateway for every feature, so
  fix it once: sort the table by available_at and np.searchsorted the cut, or bucket bets by time.

- m8. Test nits. (a) test_market.py:53 asserts 2-way Shin == additive at atol=1e-3 although the
  relation is exact (max |d| = 1.4e-16) — tighten to ~1e-12. (b) test_stats_walkforward.py:40-52
  asserts abs(clv.mean()) < 1e-12 for a quantity that is 0 by construction (fair price = inverse
  fair prob) and bounds ROI at 0.05 — tautological, low value. (c) Nothing exercises the
  odds/available_at boundary through asof: the leakage world is purely synthetic, so add a tiny
  test that a row with available_at = kickoff (close) is invisible to a bet at kickoff - 1h while
  a kickoff - 24h (pre) row is visible — that is the exact P1 ingest contract this harness relies
  on. (d) The solver brackets are unspecified by tests (mutants hi=0.5 / hi=10.0 pass and happen
  to be harmless for real booksums; power bracket-doubling at market.py:57-58 is never entered by
  any test). (e) Fold (walkforward.py:19-23) is a frozen dataclass over ndarrays, so comparing two
  folds raises "truth value of an array is ambiguous" — annotate eq=False or store tuples.
- m9. The market oracle is the same library the P3 models use — test_market.py asserts equality
  with penaltyblog.implied.calculate_implied, and penaltyblog is also the runtime dependency behind
  ratings.py:32 / dixon_coles.py:30, so a shared bug would satisfy the oracle. My brentq
  cross-check covers the gap; consider adding one analytic case (Shin == additive exactly for
  2-way) to decouple.

## NITS

- n1. scripts/devig_sanity.py silently drops 1 of 86306 PS close-1x2 match ids
  (a5b15f2a6afd6857, G1 2018/19 Panathinaikos-Olympiakos) because the match has no result in
  matches.parquet (3 such rows in total). Count and print the drop so n is auditable.
- n2. reports/devig_sanity.md never states a tolerance for "calibrated market ~1.000". Worth
  writing the criterion (e.g. |A/E - 1| < 0.02 per selection) and stating that the per-division
  draw outliers (D2 1.058, N1 1.081, I2 1.081 under power) move in the SAME direction under all
  three de-vig methods, so they are a draw-pricing / closing-odds-staleness fact in those
  divisions rather than a de-vig artefact.
- n3. stats.py:1-9: "the raw margined closing price is worse than fair: CLV = -log(booksum)" is
  exact only under multiplicative de-vig (verified max |diff| = 0.398 for power, 0.170 for shin).
  Qualify it.
- n4. implied_1x2 / implied_2way (market.py:104-111) return shape (1,k) for scalar inputs while
  devig(list) returns (k,) — inconsistent for callers.
- n5. asof (asof.py:29) raises a raw AttributeError ("Can only use .dt accessor...") when the
  column is not datetime-like; a ValueError naming the column would be clearer. A non-UTC tz-aware
  bet_time is accepted (the comparison is absolute) — fine, but the docstring says UTC.

## UNVERIFIED assumptions

- Leakage behaviour on REAL matches is untested here: the whole leakage suite is a synthetic
  Poisson world (as its docstring says), and I ran no P3 feature against real data (out of scope).
- The per-league draw A/E outliers (D2/N1/I2) are reproduced but not causally explained.
- My A/E re-run depends on data/interim/*.parquet as produced by P1; it verifies the P2 maths and
  the report, NOT the ingest correctness (that is the P0/P1 review, whose M1 — pre odds
  available_at = kickoff - 24h being too early — would propagate straight through this harness).
- Whether BFEC* closes are true closing prices in thin divisions (G1/I2/SP2) is not re-verified.
- penaltyblog own Shin (scipy ridder) fails on thin BFE 2-way rows where the fedge bisection
  succeeds (parent review note); I did not re-run that specific comparison.
- The exact real collection lead of football-data "pre" odds remains unverified (P1 M1).

## Blockers / environment notes

- data/ is gitignored by design, so a clean checkout cannot run 2 of the tests (skipped, not
  failed). In-repo, with data present, the suite is 65 passed.
- Concurrent writer: during this review the working tree picked up uncommitted changes from
  another card (src/fedge/features/ratings.py, src/fedge/models/dixon_coles.py,
  src/fedge/models/pool.py, src/fedge/report/metrics.py, new tests/test_pool.py). Everything above
  is anchored to b2b3c29; I committed only this report. src/fedge/report/metrics.py is touched by
  both cards — hot spot.

## Review method (reproducible)

All probe scripts were written under the reviewer scratch dir
(%LOCALAPPDATA%/hermes/profiles/reviewer/cache/scratch/rev_p2): verify_math.py, verify_math2.py,
verify_clv.py, verify_misc.py, verify_edge.py, verify_ae.py (+ compare_ae.py), verify_join*.py,
mutate*.py, run_mutants*.sh. Missing deps for the scipy cross-check were already in the project
venv (scipy ships with the declared deps). The A/E copy of the script was pointed at copies of
data/interim/*.parquet, so the repo working tree was never written to by any probe.
