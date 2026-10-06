# Review: P3-P5 (baseline models, main model, strategy backtest, Gate 0)

Reviewed commit: **a59d262** (HEAD == origin/main when the review started). Scope: f3aa748 (P3),
28daf93 (P4), a59d262 (P5) - `src/fedge/models/*`, `src/fedge/features/*`, `src/fedge/backtest/*`,
`scripts/run_baselines.py`, `scripts/run_main.py`, `scripts/run_strategy.py`, `tests/*`,
`reports/v1_baseline.md`, `reports/v2_main.md`, `reports/gate0.md` + its CSVs, `config/strategy.toml`.
Reviewer: independent audit; no part of the code under review was written by this card.

Method: read the code and the three reports against `docs/IMPLEMENTATION_PLAN.md` (P3-P5) and
`docs/research/M3-references-evaluation.md` section 5 (Gate 0); recomputed the P5 headline numbers
straight off the committed ledgers; re-derived CLV, A/E and pool weights from the `data/interim/`
caches with the repo's own modules; re-ran the whole P5 pipeline from the caches; ran ruff/pytest.
Runtime of the re-run: about 2 min (inside the card's 20 min budget).

## Verdict: CHANGES NEEDED - documentation plus one code change. No leakage or maths defect found, and gate0.md's "Gate 0 NOT passed" verdict is trustworthy.

No blocker. I could not construct a leak, find a wrong formula, or find a configuration of this
code that would have produced a different gate verdict. The changes needed are (a) two design facts
about the *headline* scenario that the report does not disclose (M1, M3), and (b) one in-fold
selection rule that has no positivity guard yet is graded PASS (M2). Everything else is minor/nit.

### Trustworthiness of gate0.md's PASS/FAIL claims, item by item

| item | report | my finding |
|---|---|---|
| B0.1 | PASS | **upheld.** Recomputed A/E on `v2_eval_1x2` pre H/D/A = 0.9982/1.0145/0.9902, close 0.9992/1.0094/0.9930 -> max abs dev 0.0145 (report says 0.014). Matches section 8. A weak test by construction (a de-vigged row sums to 1), but it is the test M3 prescribes and it is computed correctly. |
| B0.2 | PASS | **upheld.** `strategy.pooled_by_season` (strategy.py:46-77), `pick_model` (186), `select_market_config` (195), `run_walk_forward` (257) and `gbm.walk_forward` all slice on seasons strictly before the test fold; no random split anywhere (`train_test_split`/`KFold` appear nowhere in `src/` or `scripts/`; the only `shuffle` is a deliberate leakage test, tests/test_leakage.py:270). |
| B0.3 | PASS | **upheld.** `uv run pytest tests/test_leakage.py tests/test_stats_walkforward.py tests/test_strategy.py -q` -> **35 passed in 34s** (matches the report). The regime-reversal test exists (test_leakage.py:167). |
| B0.4 | PASS | **upheld, and stronger than claimed.** Re-running `scripts/run_strategy.py` reproduced `gate0.md`, `gate0_cells.csv`, `gate0_fold_configs.csv`, both ledgers and `config/strategy.toml` **byte-identically** (diff clean), with in-script hashes exch `eae9aad8e603...`, max `b38a7dc60072...`. See m5 for a weakness in the in-script check itself. |
| B0.5 | PASS | **upheld.** Section 6 is per-season and shows the model losing to the de-margined close (1x2 2024/25 pooled 0.2012 vs close 0.2002). |
| B0.6 | FAIL | **upheld.** My own `pool.fit_weight` on the caches: 1x2 lgb_xg close 0.00000 / pre 0.06429, lgbd_xg close **0.08411** / pre 0.45969; ou25 lgb_xg close 0.00000 / pre 0.29054, lgbd_xg close 0.02305 / pre 0.58842. Identical to section 7 and to `config/strategy.toml` (0.459694/0.290542). Two models sit on the w=0 boundary; the one that does not is the market-anchored variant (M3). |
| B0.7 | FAIL | **upheld.** The headline (`exch`) ledger is empty, so it cannot beat any baseline; the FAIL is a statement about the headline rule, and the report says exactly that. The `max` point comparison is in section 5 (m9). |
| B0.8 | PARTIAL | **upheld, and the right call.** 6% AU MBR on net winnings per market (`stats.simulate_bankroll`, stats.py:67-131); raw prices are passed in, so the P2-m2 double-commission trap is *not* re-triggered; stake cap 0.01 matches `limits.toml`. Liquidity/depth genuinely cannot be modelled from football-data. |
| B0.9 | PASS | **qualified - see M2.** The choices really are made from prior seasons only (verified by test and by reading the code), but the threshold rule has no positivity requirement, so it is formally in-fold and substantively a "bet always" rule. |
| B0.10 | PASS | **upheld.** Sections 2-5 include the losing cells, the negative in-fold `hist_clv` values and the no-whitelist variants. |

Headline: **Gate 0 NOT PASSED** is correct and is supported twice over (B0.6 and B0.7 both FAIL, and
the headline net-CLV bar is unmet because n=0).

### Reproduction (independently recomputed, not taken on trust)

From `reports/gate0_ledger_{exch,max}.csv`, with CLV/ROI recomputed from `price`/`p_close`/`won` and a
match-clustered bootstrap (seed 0):

* `exch`, above threshold: n=216, net CLV **-3.764% [-6.105, -1.315]**, raw CLV **+0.970% [-1.330, +3.451]**, flat ROI **-3.428%**, 55 wins, mean price 7.461, mean fair close prob 0.2372 (report: -3.76% [-6.10,-1.31], +0.97%, -3.43%). All 216 rows are 1x2, model `lgbd_xg`, source `BFE`, seasons 2024/25 (158), 2025/26 (1), 2026/27 (57).
* `max`, selected: n=278, net CLV **+0.466% [-0.470, +1.338]**, raw CLV **+3.315% [+2.414, +4.181]**, flat ROI **-1.466%** (report: 0.47% [-0.47,1.34], 3.31%, -1.47%).
* `max`, no whitelist: n=788, net CLV +0.206% [-0.431,+0.811], raw +3.095%, flat ROI **+2.563%** (report 2.56%).
* `max|stored - recomputed|` for `clv_net` is below 4e-6 on every row; the stored value equals `log(net_odds(price,0.06) * p_close)` exactly.
* Where the empty `exch` config comes from: the in-fold model for every `exch` fold is `lgbd_xg`, whose *largest* edge in 2022/23 and 2023/24 is negative (-0.0071, -0.0080), so no bet clears even a 0.02 threshold in those seasons. Section 4 and section 2 agree with the ledger.

## MAJOR

### M1. The headline `exch` scenario is cross-book for 73% of its bets, and the ledger cannot show it
`scripts/run_baselines.py:106-133` (`reference_books`: `order = ["PS","BFE","Avg"] if season <= PS_LAST_SEASON`) fixes the *pool's* market leg and the *CLV denominator* to **Pinnacle** for every season up to and including 2024/25, while `scripts/run_strategy.py:74-87` (`price_tables`) prefers **Betfair** for the *bet price* whenever BFE is present. Verified on the caches: every `v2_eval_1x2` row has `ref_pre == ref_close == PS` for 2016/17-2024/25, and the 158 of 216 `exch` bets in 2024/25 have `source == "BFE"` with `ref_close == PS`.
Why it matters: the headline claim is "bet the Betfair pre-closing price, score against the sharp close"; in 2024/25 it is actually "bet Betfair pre-closing, score against the **Pinnacle** close", and the pool's market probability is Pinnacle's pre-close, not Betfair's. gate0.md:13 and section 10 ("`exch` mixes two sources") only cover the BFE-then-PS sense, so a reader will assume a single venue throughout. The exported ledger has no `ref_pre`/`ref_close` column, so this is not auditable from the committed artefacts either. The verdict does not change (the numbers are negative), but the headline cannot be read as a same-venue CLV.
Fix: export `ref_pre`/`ref_close` per bet in `lcols` (run_strategy.py:588); state in section 1 that for 2024/25 the bet price is Betfair while the pool/CLV reference is Pinnacle; add a same-book scenario (PS pre vs PS close) so the 2017/18-2024/25 headline is internally consistent.

### M2. The in-fold threshold rule has no positivity guard, yet B0.9 is graded PASS
`src/fedge/backtest/strategy.py:180-224` with `:37-40`. `select_market_config` maximises `_score = mean(net CLV) * sqrt(n)` over thresholds and accepts the argmax as long as the subset has at least `MIN_HIST_BETS = 100` rows (`_score` returns `-inf` only for an empty subset), so a negative history score still qualifies. Observed in `gate0_fold_configs.csv`: `exch` folds 2022/23-2024/25 chose threshold 0.02 with `hist_clv = -9.69%`; `max` 1x2 folds 2022/23-2026/27 chose 0.06 with `hist_clv = -2.95%`. The report's own section 4 prints those `hist_clv` values.
Why it matters: B0.9 (gate0.md:493) asserts "parameters chosen in-fold": true literally, but it overstates the discipline, because a rule that selects a threshold which lost money in-sample is not a pre-committed strategy, and it is the reason the strategy fires at all in the folds where it loses. It also means the threshold rule can never itself produce a "no bets" outcome - only the whitelist can.
Fix: require `cfg["hist_clv"] > 0` (or `_score > 0`) before a threshold qualifies and report the folds that then place no bets; or, if the behaviour is deliberate, say so in section 1 and mark B0.9 PARTIAL ("chosen in-fold, no performance floor").

### M3. In every `exch` fold "the model" is the market itself (`lgbd_xg`), and its non-zero pool weight is an artefact of that
`src/fedge/models/gbm.py:15-18, 68-72, 104-130` (`init_score = log p_market`), `strategy.py:186-193` (`pick_model` = lowest pooled log-loss on history), `gate0.md:490` and `:537-541`. Measured on the caches: `lgbd_xg`'s raw 1x2 probabilities against the de-margined pre-closing price have mean absolute difference **0.0063** and `corr(raw_H, pre_H) = 0.9983` (n=66,373). Its pooled probability is therefore `p_market * exp(w*f)` - the market prior keeps weight 1 whatever `w` does - and it wins `pick_model` in *every* `exch` fold precisely because a market-anchored model can never be much worse than the market on pooled log-loss (v2_main.md:25 vs :35: 0.9987 vs 0.9988).
Why it matters: (a) B0.6's row `lgbd_xg 1x2 w=0.084 CI [0.013,0.194]` is the only CI in the report that excludes 0 against the close, and it is not evidence of new information - it is the residual of a model that was handed the price. gate0.md's summary sentence ("about zero for every non-decorrelated model") is technically true and invites the opposite inference. (b) The 216 `exch` bets are the extreme right tail of that residual, selected at a mean price of 7.46; the report never says the strategy is effectively betting a market-residual disagreement on longshots.
Fix: label `lgbd_xg` everywhere as a market-recalibration baseline; read its weight columns as "weight on the residual, given the price it consumed"; state in section 1 that `pick_model` structurally favours it (the market offset gives it a floor in the log-loss race against `lgb_xg`/`cb_xg`).

## MINOR

* m1. **ou25 evaluation window misstated.** `reports/v2_main.md:7` ("Evaluation rows: 66373 (1X2), 46417 (over/under 2.5); ... seasons 2016/17 to 2026/27") and `reports/v1_baseline.md:7-8,31` read as a common window. `data/interim/v2_eval_ou25.parquet` contains only 2019/20-2026/27 (8 seasons, n=46417); the pre-2019/20 ou25 rows have no closing reference book (`v1_baseline_reference_books.csv`: ou25/close/`none`, 126 cells) and are dropped in `run_baselines.build_dataset`. Fix: state the ou25 window separately in both reports.
* m2. **"CI excluding 0" in the pool-weight tables is an in-sample weight.** `reports/v1_baseline.md:13` documents that `w_pooled` and its CI refit the weight on the whole sample - and the report does print `w_expanding` (out-of-sample) beside it, which is honest - but the prose at `v1_baseline.md:490-496` and `v2_main.md:806` ("5 of 36 pre-closing cells ... now have a CI above 0", "the earlier '0 of 36' no longer holds") reads as a significance claim. Of those 5 cells, **3 have expanding-window pooled log-loss worse than the market**: SC0 ou25 0.6815 vs 0.6798, G1 ou25 0.6941 vs 0.6930, SP2 ou25 0.6734 vs 0.6729 (the table's own last two columns). Fix: lead with `w_expanding`, label the CI as in-sample, and note 36 cells x 2 markets of multiple testing.
* m3. **Net CLV is the selection target and the gate bar, but it is not M3's CLV.** `gate0.md:16`, `strategy._score` (180), `ledger_metrics` (349). Folding commission into the CLV numerator (`log_clv(net_odds(price))`) makes the metric price-dependent: at the `exch` ledger's mean price 7.46 the commission term is about -4.7%, so net CLV = -3.76% against raw CLV = +0.97%; for `max` selected, +3.31% raw becomes +0.47% net. Consequence: the in-fold threshold/whitelist/Kelly search optimises an objective that penalises long prices, and `clv_pos` (run_strategy.py:523) tests a stricter bar than M3 P1.2 (mean log-CLV vs the margin-free close). Fix: use raw log-CLV for selection/skill and ROI for money, or justify net CLV explicitly as an additional rather than primary bar.
* m4. **The Max bad-quote guard is an untested post-hoc filter.** `scripts/run_strategy.py:46, 89-96`: rows are dropped when any selection's `Max` price exceeds `1.10x` the BFE/PS price; the constant was picked after a first run showed +13% net CLV on 2024/25+ Max quotes (disclosed at gate0.md:41-43), it drops the whole match when one selection fails, and it removes 6-12% of 1x2 rows a season. No sensitivity check is shown. Fix: sweep 1.05/1.10/1.15/1.20 and report the range; record the constant's provenance in section 10 as well.
* m5. **B0.4's in-script hash is vacuous for the headline scenario.** `scripts/run_strategy.py:151-155` hashes only `bets[selected]`, which is *empty* for `exch`, so `exch=eae9aad8e603...` is the hash of a header-only frame. Fix: include the above-threshold ledger in the hash (non-empty for both scenarios; I verified byte-identical re-runs externally, so the claim itself holds).
* m6. **Separated logistic calibration slopes printed without a flag.** `reports/v1_baseline.md:126` (E2 2026/27 `dc` 1x2 `cal_slope = 48442320576.9402`, n=90) and `:399` (`market_pre` ou25 SP2 2026/27 = 2.2e10, n=78). A reader will read these as calibration diagnostics. Fix: clamp/flag `|slope| > 10` as "not estimable at this n" in `report/metrics.py`.
* m7. **Embargo boundary differs between feature families.** `src/fedge/features/ratings.py:109,145` use `ko[ptr] <= cut[i]` (a row exactly `EMBARGO` old enters the state) while `features/form.py:161` and `features/asof.py:26` are strict (`<`). AGENTS.md rule 3 says `available_at < bet_time`. Impact is one boundary match, but the conventions should be unified (strict `<`).
* m8. **v2_main.md:740-771 counts fitted seasons, not evaluated ones.** Every ou25 row sums to 11 seasons while only 8 are evaluated (m1); `v2_main_fold_info.csv` has 11 ou25 rows per model. Fix: restrict the table to evaluated seasons or say "fitted seasons (3 pre-2019/20 ou25 seasons are fitted but never evaluated)".
* m9. **No CI on the baseline difference.** gate0.md section 5 compares strategy net CLV 0.47% with favourite-on-same-matches 0.14% as a point comparison; only the random baseline gets a distribution (`P(random >= strategy) = 0.000`). With n=278 and a CI of [-0.47, 1.34] on the strategy itself, that gap is not resolvable. Fix: bootstrap the paired difference, or say in the table that the favourite comparison is not a test.

## NITS

* n1. The committed ledgers omit `clv_raw` (`run_strategy.py:588-607`), so the raw-CLV column of the report is not readable off the artefacts (it is recomputable as `log(price*p_close)`).
* n2. `src/fedge/backtest/stats.py:6` still asserts "CLV = -log(booksum)" for the raw margined close. That identity holds for multiplicative de-vig only (the P2 review measured deviations up to 0.398 under power); qualify or drop it.
* n3. `simulate_bankroll` ignores `limits.toml`'s `max_bets_per_day = 20`; with one bet per match-market a busy Saturday can exceed it. Paper-only, but the cap is advertised in the config.
* n4. gate0.md never says in prose that the in-fold **whitelist hurt out-of-sample** in the `max` scenario: selected n=278 flat ROI -1.47% vs no-whitelist n=788 flat ROI +2.56%. The tables carry both; one sentence would strengthen the "no edge" conclusion.
* n5. `config/strategy.toml`'s ou25 block (`threshold = 1.0`, `qualified = false`) is not explained in gate0.md section 11, whose table lists only the 1x2 row; and `history_seasons = "2017/18 to 2026/27"` is the walk-forward fold list, not the in-fold history behind the written config.
* n6. `reports/v2_main.md:807` quotes "~0.0003 per match" for the GBM log-loss gain; section 5's column spans 0.0003-0.0006 (mean 0.00045). Quote the range.

## What I could not verify (settle these before trusting P5 live)

* **The real collection time of football-data's pre-closing price** (P1 M1, restated at v2_main.md:808 and gate0.md:511). The whole P5 timing story rests on `available_at = kickoff - 24h` (`src/fedge/ingest/football_data.py:269`). Settle it by scraping a live week and comparing the file's `Date`/`Time` columns with the price columns; until then "bet at the pre-closing snapshot" is a hypothesis, and a price actually collected later would *inflate* the measured CLV.
* Whether the `BFEC*` columns are true closing prices in thin divisions (G1, I2, SP2) - I did not re-fetch the source files.
* Whether the P3/P4 caches in `data/interim/` were produced by exactly the committed code: I re-ran P5 (which only reads them) but did not re-fit Dixon-Coles, Elo/pi or the GBMs (hours of runtime). The strategy numbers therefore inherit any cache/code mismatch; `uv run python scripts/run_main.py --stage report` and `run_baselines.py --stage report` would settle it.
* The two large P3/P4 reports were read by a second, independent pass whose per-number claims I spot-checked (pool weights, ou25 window, the 5-cell list, the 0.0003 range, the separated slopes). I re-derived the pooled 1x2 weights myself and they match to 4 dp.

## Working-tree note (not a finding about this card)

While this review ran, another card was editing the same working directory (uncommitted:
`src/fedge/models/gbm.py`, `src/fedge/paper/{ledger,risk,settle}.py`, new `src/fedge/paper/*.py`,
`scripts/paper_*.py`, `tests/test_paper.py`). My P5 re-run reproduced the committed artefacts
byte-for-byte with those edits present, so they do not affect the reviewed numbers; this review is
anchored to a59d262 and I committed only this file. `src/fedge/models/gbm.py` is the collision
hotspot (P4 model code, now also touched by the P6 paper card).

## Files changed by this review

* `reports/reviews/review_p3_p5.md` (this file). No code, report or config was modified; the P5
  re-run left every artefact byte-identical.

Review method (reproducible): probes under
`%LOCALAPPDATA%/hermes/profiles/reviewer/cache/scratch/` (`an1.py`: in-fold candidates per season;
`an3.py`: ledger CLV/ROI/bootstrap; `an4.py`: market-embedding distance; `an5/an6.py`: pool weights
and A/E); `uv run ruff check .`; `uv run pytest` (129 passed); `uv run python scripts/run_strategy.py`.
