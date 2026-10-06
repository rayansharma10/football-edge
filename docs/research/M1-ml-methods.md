# M1 — ML modelling methods for football match prediction aimed at beating the market

**For:** Rayan (UNSW computer engineering) — football prediction + paper-betting system, target = positive **closing-line value (CLV)** vs Betfair/Pinnacle closing prices.
**Date:** 6 Oct 2026 · **Scope:** modelling methods, market-aware techniques, league/market selection, a staged stack. No code written, nothing installed.
**Relationship to B1:** `B1-football-models-evidence.md` established *whether* the market can be beaten and catalogued the ecosystem. This note goes a level deeper into **model families, market-aware modelling, and a concrete build plan**. Read B1 first; this one assumes it.

**Convention:** every factual claim carries an inline source URL. Claims I could not verify from a fetched primary source are tagged **[UNVERIFIED]**. Numbers are quoted as they appear in the cited source.

---

## 1. Executive summary / the decision

- **Model class is a second-order lever; the market-aware layer and the data are first-order.** The 2024 survey of the field concludes that gradient-boosted trees (CatBoost) applied to soccer-specific ratings such as pi-ratings are currently the best models *on datasets containing only goals* — but also that "the exact choice of features and the choice of model have only a minor influence on prediction quality" ([arXiv:2403.07669](https://arxiv.org/abs/2403.07669); [arXiv:2408.08331](https://arxiv.org/abs/2408.08331)).
- **A good structural model still does not beat the closing price.** On 19 Serie A seasons (7,220 matches) a tuned Dixon-Coles model reached RPS **0.1972** vs the market's **0.1905**; the market won all 7 test seasons; and the fitted weight on the structural model in a model↔market opinion pool was **exactly 0.000** — a boundary solution, not an artefact ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)).
- **Therefore the winning design is not "predict better", it is "measure incremental information and monetise mispricing":** calibrate, then blend honestly against a de-vigged sharp price, then bet only where your *calibrated* probability plus a margin beats a *soft* price. Selection must be on calibration/log-loss, not accuracy ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021)).
- **Recommended stack:** v1 = Dixon-Coles + Elo baseline vs a de-vigged market benchmark (prove the pipeline); v2 = CatBoost/LightGBM on rating features + xG, calibrated (Venn-Abers/isotonic) and blended with the market via an out-of-sample opinion-pool weight; v3 = xG/shot-distribution and market-calibrated intensity models (Weibull AFT), Bayesian hierarchical uncertainty, player-level features, in-play.
- **Acceptance bar for the whole project:** on a leakage-safe walk-forward backtest, (a) RPS within ~0.005 of the closing market, (b) opinion-pool weight on your model **> 0**, (c) calibration slope ≈ 1, (d) positive CLV on a paper-trading log. If the pool weight is ~0, your model adds nothing and you should be exploiting execution/lag, not modelling ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)).
- **Realistic best market to hunt first:** markets where books invest less effort and prices move — **in-play** (a market-calibrated Weibull model earned **4.5% ROI, Sharpe 5.94 over 17,458 bets**, [arXiv:2605.16066](https://arxiv.org/abs/2605.16066)) and **over/under totals** ([Wheatcroft 2020, IJF](https://doi.org/10.1016/j.ijforecast.2019.11.001)) — rather than EPL 1X2, which is priced to the basis point.

---

## 2. Method comparison (1X2 and goals; headline metrics where reported)

Benchmarks to beat, for orientation: bookmaker RPS ≈ **0.198** and margin ≈ **−5.6%** on the 2023-challenge test set ([arXiv:2501.05873](https://arxiv.org/abs/2501.05873)); market RPS **0.1905** in Serie A ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)). Lower RPS / log-loss is better.

| # | Method | Inputs | Typical RPS / log-loss reported | Pros | Cons | Libraries |
|---|---|---|---|---|---|---|
| 1 | **Elo / Club Elo** | Match results → running rating; goal-difference scaling | Elo ordered-logit ≈ RPS 0.2035 on 2017-challenge test set (see B1 table) — **[UNVERIFIED]** here directly | Trivial to implement, surprisingly strong, no fitting to scorelines | Ignores goals detail, no attack/defence split; needs tuning of K/HFA | `penaltyblog`; Club Elo feed |
| 2 | **Pi-ratings (Constantinou & Fenton 2013)** | Results → home/away attack- & defence-adjusted ratings | Best *feature set* for GBM on goals-only data ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807); [arXiv:2403.07669](https://arxiv.org/abs/2403.07669)) | Captures score discrepancies vs opponents; beats plain Elo as a feature | Ratings-only ceiling; sensitive to update schedule | `penaltyblog` (`PiRating`); method: [DOI 10.1515/jqas-2012-0036](https://doi.org/10.1515/jqas-2012-0036) |
| 3 | **Berrar ratings** | Recent-match attack/defence features → predicted goals | Won the 2017 challenge's exact-score task (avg loss 1.0047 RMSE) ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807)) | Rich recency-aware features; strong for score prediction | Weaker than pi-ratings for W/D/L probability in the 2023 re-run | custom; method: [DOI 10.1007/s10994-018-5747-8](https://doi.org/10.1007/s10994-018-5747-8) |
| 4 | **Maher Poisson (attack/defence)** | Team attack λ and defence μ + home advantage | Comparable to ML for scores ([arXiv:2408.08331](https://arxiv.org/abs/2408.08331)) | Interpretable, fast, well-understood | No low-score correction, no time decay | `penaltyblog`; `statsmodels` |
| 5 | **Dixon-Coles (1997)** | Poisson + low-score (τ) correction + exponential time decay | RPS **0.1972** vs market 0.1905, Serie A ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)); decay ξ=0.0065 (half-life ≈ 1 yr) ([arXiv:2605.16066](https://arxiv.org/abs/2605.16066)) | The academic workhorse; time decay + draw/dependence correction | Single fixed decay; still behind market on RPS | `penaltyblog` (`DixonColesGoalModel`); [DOI 10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) |
| 6 | **Bivariate / zero-inflated Poisson** | Joint goal distribution with score correlation | Used for CL/EL qualification simulation ([arXiv:2508.20075](https://arxiv.org/abs/2508.20075)) | Models goal correlation and 0-0/1-1 excess | Extra parameters; bivariate Poisson only allows non-negative correlation, so zero-inflation is often needed | `penaltyblog` (`BivariatePoisson`, `ZeroInflated`); [Karlis & Ntzoufras 2003, DOI 10.1111/1467-9884.00366](https://doi.org/10.1111/1467-9884.00366) |
| 7 | **Bayesian hierarchical Poisson** | Team effects, league hyper-priors; optionally odds | Reliability "reasonably good… compared to bookmaker odds" in UCL ([arXiv:2106.14345](https://arxiv.org/abs/2106.14345), via score decompositions) | Full uncertainty, shrinkage for promoted/weak teams, mixes with odds | More compute/skill; MCMC can be slow | `penaltyblog` (Bayesian built-ins), `PyMC`, Stan; method: [Baio & Blangiardo 2010, DOI 10.1080/02664760802684177](https://doi.org/10.1080/02664760802684177) |
| 8 | **GBM on ratings (CatBoost/XGBoost/LightGBM)** | pi-ratings / Elo / Berrar features | CatBoost+pi-ratings avg loss **0.2085** (2023 validation, best of its field) ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807)) | State of the art on goals-only data; handles non-linear feature interactions; CatBoost fixes categorical leakage | Still behind bookmaker odds; needs disciplined feature selection | `catboost`, `xgboost`, `lightgbm`, `scikit-learn` |
| 9 | **GBM on engineered features (2017 winners)** | Recency, form, table position, venue %, PageRank | 1st and 2nd in the 2017 challenge were both gradient tree boosting ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807)) | Proven; interpretable importances | 200+ features invites overfitting; needs walk-forward | as #8 |
| 10 | **Deep learning (Transformer/TimesNet/LSTM/GRU)** | Sequences of match features | Best DL candidate **0.2098** (Inception+TE+MLP), behind CatBoost+pi (0.2085) ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807)) | Can learn temporal structure; no manual feature engineering | No consistent edge over GBM; slower; data-hungry | `PyTorch`, `TensorFlow` |
| 11 | **xG / shot-based models** | xG (shot quantity × quality), or match statistics | RPS **0.201** on the 2023 test set (Elo→shot-distribution model) ([arXiv:2501.05873](https://arxiv.org/abs/2501.05873)) | Less noisy target than goals; more signal per match | Needs xG/match-stat feeds; xG itself must be modelled | custom; xG method: [arXiv:2301.13052](https://arxiv.org/abs/2301.13052); stats→results: [Wheatcroft 2021, DOI 10.3233/jsa-200462](https://doi.org/10.3233/jsa-200462) |
| 12 | **Player-level / lineup models** | Plus-minus, player-adjusted xG, aggregated XI ratings | Lineups **did not** improve score prediction in EPL 2020–22 ([arXiv:2210.06327](https://arxiv.org/abs/2210.06327)); plus-minus gives interpretable player value ([arXiv:1706.04943](https://arxiv.org/abs/1706.04943)) | Captures injuries/rotation; long-run player value | Data + identity headaches; evidence that lineups add little to scores | `socceraction`; methods: [arXiv:1706.04943](https://arxiv.org/abs/1706.04943), [arXiv:2511.23072](https://arxiv.org/abs/2511.23072) |
| 13 | **Market-only (de-vigged odds)** | Odds only | Market RPS ≈ 0.1905–0.198 ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505); [arXiv:2501.05873](https://arxiv.org/abs/2501.05873)) | The benchmark; hard to beat; free | Cannot itself create edge — it *is* the price | `penaltyblog` odds→probability; [arXiv:2604.17194](https://arxiv.org/abs/2604.17194) |
| 14 | **Market-calibrated intensity models** | Fit scoring rates to 1X2 **and over/under** prices; + in-play covariate | 70.2% acc vs Betfair 70.6%; in-play RPS 0.1294; **4.5% ROI, Sharpe 5.94, 17,458 bets** ([arXiv:2605.16066](https://arxiv.org/abs/2605.16066)) | Matches the market *and* stays interpretable; monetises in-play | In-play infra; calibration to exchange prices | custom (`lifelines`-style AFT); [arXiv:2605.16066](https://arxiv.org/abs/2605.16066) |
| 15 | **Model↔market blends (opinion pool)** | Model probs + de-vigged market | Fitted pooling weight is the *test* of incremental info; here 0.000 ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)); convex historical+odds mix yields +EV ([arXiv:1802.08848](https://arxiv.org/abs/1802.08848)) | Honest, decomposes calibration vs discrimination | Only as good as the components | custom |

**Why accuracy is the wrong optimisation target.** A model can be highly accurate and useless if it merely reproduces the bookmaker ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021)). Selecting models on **calibration** produced average ROI **+34.69%** vs **−35.17%** when selecting on accuracy (best case +36.93% vs +5.56%) in an NBA study ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021)). And a deliberately **decorrelated** model can profit even when it is worse than the market ([arXiv:2010.12508](https://arxiv.org/abs/2010.12508); [Hubáček et al. 2019, IJF](https://doi.org/10.1016/j.ijforecast.2019.01.001)).

---

## 3. Market-aware techniques (the part that actually matters)

1. **Convert odds to probabilities correctly before doing anything.** Multiplicative normalisation is the common default but assumes equal bettor loss across outcomes, which contradicts the favourite–longshot bias; Shin (two variants) and the power method are alternatives, and a 2026 study over **90,014 matches × 5 bookmakers** proposes an "Odds-Only-Equal-Profitability-Confidence" (OO-EPC) method (best of the odds-only methods) plus a one-parameter Favourite-Longshot-Bias-adjusted GLM (FL-GLM) that beats multinomial/logistic GLMs ([arXiv:2604.17194](https://arxiv.org/abs/2604.17194)). Use OO-EPC (or Shin) for the de-vigged sharp reference, not plain multiplication.

2. **Test for incremental information with an opinion pool, not accuracy.** Fit the weight *w* in a **logarithmic opinion pool** of your model and the de-vigged market, out of sample. If *w* = 0 (monotone-increasing log-loss in *w*), your model adds nothing the closing price hasn't absorbed — exactly what happened to a tuned Dixon-Coles in Serie A (*w* = 0.000) ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)). Note the same paper's sharper point: a shots-on-target variant earned weight **0.35 against the goals model** but **0.000 against the market** — two structural signals can each be informative relative to the other and still both be priced.

3. **Separate calibration from discrimination.** Pitcan (2026) finds the structural model is *better calibrated* than the market on the home-win margin (slope **0.995** vs the market's **1.103**) but *less sharp*; the market's edge is discrimination, which accuracy can't see ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)). Implication: your model may be a legitimate *calibration* anchor even when it can't out-discriminate the market.

4. **Blend historical data with the odds explicitly.** A hierarchical Bayesian Poisson model makes scoring rates a **convex combination** of historical attack/defence parameters and parameters recovered by inverting bookmaker 1X2 odds (via a Skellam), with a Bayesian-estimated mixing weight — the canonical "let the market inform the model" design ([Egidi, Pauli & Torelli 2018, arXiv:1802.08848](https://arxiv.org/abs/1802.08848)). The Skellam/implied-scoring-rate approach originates with [Feng, Polson & Xu 2016](https://arxiv.org/abs/1604.03614).

5. **Calibrate the model, properly.** Methods, cheapest→best-evidenced:
   - **Isotonic / Platt** — classic; but a 2026 benchmark of 21 classifiers found Platt scaling weak and *both Platt and isotonic can degrade proper scoring for strong modern models* ([arXiv:2601.19944](https://arxiv.org/abs/2601.19944)).
   - **Venn-Abers** — guaranteed well-calibrated under exchangeability (built on isotonic regression); achieved the **largest average log-loss reductions**, with Beta calibration close behind ([Venn-Abers theory: arXiv:1211.0025](https://arxiv.org/abs/1211.0025); [benchmark: arXiv:2601.19944](https://arxiv.org/abs/2601.19944)).
   - **Temperature scaling** — standard for neural nets; verify against log-loss before trusting [UNVERIFIED as specific to football].
   - **Conformal / MAPIE** — distribution-free *sets/intervals* for uncertainty-aware staking; a public WC-2026 research repo already fuses Dixon-Coles+Elo+Pi+Weibull+Market+**Conformal** ([AndyDu0921/wc26-predict](https://github.com/AndyDu0921/wc26-predict)). Conformal guarantees are coverage, not calibration of point probabilities — use for staking/uncertainty gating, not as your probability estimator [UNVERIFIED for football-specific gains].

6. **Model the residual vs the market / decorrelate.** Hubáček et al. trained with a loss that *penalises correlation with the bookmaker's odds*, which yielded greater profits than optimising accuracy ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021) reporting [Hubáček et al. 2019, DOI 10.1016/j.ijforecast.2019.01.001](https://doi.org/10.1016/j.ijforecast.2019.01.001)). Formalised in "Beating the market with a bad predictive model": a model *worse* than the market can profit by decorrelating from it to exploit its persistent biases ([arXiv:2010.12508](https://arxiv.org/abs/2010.12508)).

7. **Stake with the estimation error in mind, not textbook Kelly.**
   - An empirical review across horse racing, basketball and football finds the **adaptive variant of fractional Kelly** "a very suitable choice across a wide range of settings" and that risk-control modifications are practically necessary ([Uhrín et al. 2021, arXiv:2107.08827](https://arxiv.org/abs/2107.08827); [DOI 10.1093/imaman/dpaa029](https://doi.org/10.1093/imaman/dpaa029)).
   - Kelly is not automatically conservative: with sampled/estimated distributions the theoretical Kelly bet can be **much smaller** than the empirical one ("too conservative"), and it has real limitations from Taylor approximations and drawdowns ([arXiv:1710.01786](https://arxiv.org/abs/1710.01786); [arXiv:1710.01787](https://arxiv.org/abs/1710.01787)).
   - If your probabilities are uncertain, use **distributionally-robust Kelly** (maximise worst-case log-growth over an uncertainty set of distributions), which is convex/tractable for finite outcomes ([arXiv:1812.10371](https://arxiv.org/abs/1812.10371)).
   - Practical: quarter-to-half Kelly, per-bet cap ≤1–2% of bankroll, flat stakes for *measuring* edge (see B1 §6).

8. **Never feed the market in and then claim credit for matching it.** Pitcan (2026) explicitly flags the design error of supplying bookmaker odds as features and then comparing accuracy with the bookmaker — "the pipeline is discarding" information; incremental information must be measured *against* the price, not with it inside ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)).

---

## 4. Which leagues and markets are most beatable

| Market / league | Evidence | Verdict for a model |
|---|---|---|
| **EPL / top-5 1X2 (closing)** | Market RPS 0.1905 vs best structural 0.1972; market won all 7 seasons ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)); bookmaker odds beat avg-points/goals/Elo models across ~100k matches ([arXiv:2605.16066](https://arxiv.org/abs/2605.16066) citing Wunderlich 2025) | **Do not expect a 1X2 edge**; use as the calibration/benchmark lab |
| **Asian handicap** | Shares the traditional market's inefficiencies but has differences; first published AH model uses ratings + Bayesian networks on 13 EPL seasons ([Constantinou 2020, arXiv:2003.09384](https://arxiv.org/abs/2003.09384); [DOI 10.3233/JSA-200588](https://doi.org/10.3233/JSA-200588)) | Worth exploring as a **second market** (fewer outcomes, no draw) — **[UNVERIFIED]** that it's easier net of margin |
| **Over/under (totals)** | "A profitable model for predicting the over/under market in football" ([Wheatcroft 2020, IJF 36:916–932](https://doi.org/10.1016/j.ijforecast.2019.11.001)); over/under prices also *fix the total-goal rate* and resolve a scoring-rate identifiability problem for 1X2 calibration ([arXiv:2605.16066](https://arxiv.org/abs/2605.16066)) | **Strong candidate** — totals are lower-dimensional and the price gives you the goals distribution |
| **In-play** | Market-calibrated Weibull+PSxG: 70.2% vs Betfair 70.6% acc, **+4.5% ROI, Sharpe 5.94, 17,458 bets** ([arXiv:2605.16066](https://arxiv.org/abs/2605.16066)); but markets do **not** anticipate the first goal ([arXiv:2505.21275](https://arxiv.org/abs/2505.21275)), and first-half goals add nothing once pre-match odds are included ([Klemp et al. via arXiv:2605.16066](https://arxiv.org/abs/2605.16066)) | **Best-documented genuine inefficiency**, but needs real-time infra and stake caps |
| **Early/open lines vs closing** | Opening prices are noisier/biased; the edge is in *predicting the move* ([Nagel 2025, SSRN 5043714](https://doi.org/10.2139/ssrn.5043714)); soft books lag sharp references ([Kaunitz et al. 2017, arXiv:1710.02824](https://arxiv.org/abs/1710.02824)) | Where CLV is *created*: bet early into prices that will shorten |
| **Soft-book vs sharp (Pinnacle) value** | De-vig the sharp price, bet soft price over fair+margin (Kaunitz mechanism) ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824)) | The core **execution** strategy; account limits are the binding constraint |
| **Market disruption windows (e.g. COVID)** | Bookmakers had "problems to adjust the betting odds" after home advantage vanished behind closed doors, "opening opportunities for profitable betting strategies" ([arXiv:2008.05417](https://arxiv.org/abs/2008.05417)) | **Systematic edge class:** bet when a regime change breaks the book's priors |
| **Lower divisions / women's football** | No arXiv evidence found for a model edge; data thinner, limits small | **[UNVERIFIED]** — plausible (less modelling effort → softer prices) but unproven; treat as a hypothesis to test, not a plan |
| **Bookmaker profit source** | Decomposed as a "profit-bias identity": bookmaker profit is the public's prediction error (hold + bias), not superior single-game prediction ([arXiv:2609.06739](https://arxiv.org/abs/2609.06739)) | Bias-based (public/market) mispricing is the exploitable structural feature, not beating the point forecast |

Two more orientation facts: models that exploit mispricing lose edge as data/methods become public ([Mangold & Stübinger 2020, DOI 10.4995/carma2020.2020.11619](https://doi.org/10.4995/carma2020.2020.11619)), and a **neural-network + portfolio-theory** EPL study reported +135.8% over half a season ([arXiv:2307.13807](https://arxiv.org/abs/2307.13807)) and an **XAI + Kelly-Index** study found profit only in a "conservative, high-confidence, easy-match" subset ([arXiv:2211.15734](https://arxiv.org/abs/2211.15734)) — both are small-sample results to treat with caution, not targets.

---

## 5. Recommended staged stack for Rayan

Design principles baked in: **walk-forward by season** (never shuffled CV — chronological segmentation is mandatory in sports prediction, [arXiv:2303.06021](https://arxiv.org/abs/2303.06021)); a **5-year rolling training window with time decay** (the 2023 challenge winner trained on the most recent 5 years, [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)); every stage benchmarked against the **de-vigged market**; success measured by **CLV and calibration**, not accuracy (see B1 §6).

### v1 — Baseline & harness (~1–2 weeks)
- **Data:** `soccerdata` → football-data.co.uk historical results **with closing odds**, plus Club Elo ratings ([soccerdata](https://github.com/probberechts/soccerdata); [football-data.co.uk](https://www.football-data.co.uk/); [Club Elo](http://clubelo.com/)).
- **Models:** (a) **Dixon-Coles** with exponential time decay ξ≈0.0065 (half-life ≈ 1 year) via `penaltyblog`; (b) **Elo → ordered-logit**.
- **Market layer:** de-vig with **Shin/OO-EPC**, not just multiplicative; store bookmaker RPS/log-loss as the benchmark ([arXiv:2604.17194](https://arxiv.org/abs/2604.17194); [arXiv:2501.05873](https://arxiv.org/abs/2501.05873)).
- **Metrics:** RPS, log-loss, Brier, reliability curve, plus a **CLV logger** (record your price and the closing price for every candidate bet — the leading skill indicator per B1, [Nagel 2025](https://doi.org/10.2139/ssrn.5043714)).
- **Acceptance gate:** reproduce market RPS ≈ 0.19–0.20 on a holdout season; Dixon-Coles within **~0.01** RPS of the market ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)); harness provably leakage-free. If it's >0.02 behind, fix data/features before adding model complexity.

### v2 — Main model (~3–8 weeks): rated GBM + calibration + market blend
- **Features:** pi-ratings ([Constantinou & Fenton 2013](https://doi.org/10.1515/jqas-2012-0036)) and Club Elo as the core; Berrar-style recency attack/defence features ([Berrar et al. 2019](https://doi.org/10.1007/s10994-018-5747-8)); xG from Understat/FBref ([Understat](https://understat.com/); [FBref](https://fbref.com/)); recent form (last 6, time-weighted), rest days, promoted/demoted flags, home/away splits. Train a **goals-only** variant first so it's comparable to the published benchmarks, then an **xG-augmented** variant as an ablation ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807); [Wheatcroft 2021](https://doi.org/10.3233/jsa-200462)).
- **Model:** **CatBoost multiclass on pi-ratings** (best on goals-only data, [arXiv:2403.07669](https://arxiv.org/abs/2403.07669); [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)); compare LightGBM/XGBoost; a simple stack of the two.
- **Calibration:** fit on a walk-forward calibration fold; try **isotonic, Venn-Abers, Beta, temperature**; select by log-loss/ECE — expect **Venn-Abers/Beta** to win and isotonic/Platt possibly to hurt ([arXiv:2601.19944](https://arxiv.org/abs/2601.19944); [arXiv:1211.0025](https://arxiv.org/abs/1211.0025)).
- **Market-aware layer (the differentiator):**
  1. de-vigged sharp price as the reference;
  2. **logarithmic opinion pool** with an out-of-sample weight — report *w* as a headline result ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505));
  3. one variant trained with a **decorrelation penalty** vs the bookmaker ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021); [arXiv:2010.12508](https://arxiv.org/abs/2010.12508));
  4. optionally blend historical + implied scoring rates with a Bayesian mixing weight ([arXiv:1802.08848](https://arxiv.org/abs/1802.08848)).
- **Staking:** quarter-Kelly with a ≤1–2% cap; consider distributionally-robust Kelly under probability uncertainty ([arXiv:2107.08827](https://arxiv.org/abs/2107.08827); [arXiv:1812.10371](https://arxiv.org/abs/1812.10371)).
- **Acceptance gate:** RPS ≤ market + **0.005** on walk-forward; **pooling weight > 0**; calibration slope ≈ 1; **positive CLV** on a paper log. If *w* ≈ 0, pivot effort to execution (early lines / soft-vs-sharp) rather than the model.

### v3 — Stretch / edge hunting (quarter 2+)
- **xG/shot-distribution model:** predict shot quantity and quality distributions, simulate matches ([arXiv:2501.05873](https://arxiv.org/abs/2501.05873)); over/under as a target market ([Wheatcroft 2020](https://doi.org/10.1016/j.ijforecast.2019.11.001)).
- **Market-calibrated intensity model:** fit scoring rates to 1X2 **and** over/under prices; add **post-shot xG** as a time-varying covariate for **in-play** — the design that produced 4.5% ROI / Sharpe 5.94 ([arXiv:2605.16066](https://arxiv.org/abs/2605.16066)).
- **Bayesian hierarchical** scoring-rate model in **PyMC/Stan** for uncertainty-aware probabilities ([Baio & Blangiardo 2010](https://doi.org/10.1080/02664760802684177); [Egidi et al. 2018](https://arxiv.org/abs/1802.08848); [PyMC](https://pypi.org/project/pymc/)).
- **Player-level:** plus-minus / player-adjusted xG for team-strength adjustment, accepting that lineups historically add little to score prediction ([arXiv:1706.04943](https://arxiv.org/abs/1706.04943); [arXiv:2301.13052](https://arxiv.org/abs/2301.13052); [arXiv:2511.23072](https://arxiv.org/abs/2511.23072); [arXiv:2210.06327](https://arxiv.org/abs/2210.06327)).
- **Uncertainty-aware staking:** conformal intervals (MAPIE) to gate stake size ([MAPIE](https://pypi.org/project/MAPIE/); [wc26-predict](https://github.com/AndyDu0921/wc26-predict)).
- **Acceptance gate:** a genuinely positive-CLV paper-trading season on an illiquid market (over/under or in-play), with the model's pool weight against the *specific* market price staying > 0.

**Which families to skip** (for a solo build): LLM forecasters (no demonstrated market edge, confidence uncorrelated with accuracy — B1 §2.4), unconstrained deep sequence models (no edge over tuned GBM on goals-only data, [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)), and pure-accuracy selection ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021)).

---

## 6. Evaluation protocol & staking (condensed; full version in B1 §6)

- **Walk-forward only**; chronological splits; no random shuffling ([arXiv:2303.06021](https://arxiv.org/abs/2303.06021)).
- Report **RPS + log-loss + Brier + reliability/discrimination decomposition** ([arXiv:2106.14345](https://arxiv.org/abs/2106.14345)); treat **RPS as primary** for ordinal 1X2, though it has known criticisms for football (see the "case against RPS", [Wheatcroft 2021, JQAS 17(4):273–287](https://doi.org/10.1515/jqas-2019-0089) — cited in [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)).
- Always benchmark against the **de-vigged bookmaker price** ([arXiv:2501.05873](https://arxiv.org/abs/2501.05873); [arXiv:2608.11505](https://arxiv.org/abs/2608.11505)).
- Track **CLV** on paper bets; size the sample before believing any edge (B1 §6.2 shows thousands of bets for t≈2).
- Staking: adaptive fractional / quarter-Kelly with hard caps; robust-Kelly under uncertainty ([arXiv:2107.08827](https://arxiv.org/abs/2107.08827); [arXiv:1812.10371](https://arxiv.org/abs/1812.10371)).

---

## 7. Building blocks (verified 6 Oct 2026)

| Purpose | Tool | Status |
|---|---|---|
| Data scraping (results, odds, xG, Elo, SofaScore, Understat, FBref) | [`probberechts/soccerdata`](https://github.com/probberechts/soccerdata) / PyPI `soccerdata` 1.9.1 ([PyPI](https://pypi.org/project/soccerdata/)) | 2,099★, active |
| Poisson / Dixon-Coles / bivariate / Bayesian / Elo / pi-ratings / odds→prob | [`martineastwood/penaltyblog`](https://github.com/martineastwood/penaltyblog) / PyPI 1.13.1 ([PyPI](https://pypi.org/project/penaltyblog/)) | 230★, MIT, updated 2026-10-05 |
| Event data → SPADL actions / xT | [`socceraction`](https://pypi.org/project/socceraction/) | PyPI 1.5.3 |
| Probabilistic programming (hierarchical Bayes) | [`pymc`](https://pypi.org/project/pymc/) (6.3.2); Stan/CmdStanPy | PyPI |
| Calibration | [`netcal`](https://pypi.org/project/netcal/) 1.4.0, [`venn-abers`](https://pypi.org/project/venn-abers/) 1.5.4, sklearn `CalibratedClassifierCV` | PyPI |
| Conformal / uncertainty | [`MAPIE`](https://pypi.org/project/MAPIE/) 1.5.0 | PyPI |
| Reference end-to-end (leakage-safe walk-forward + paper UI) | [`Reymes/football-match-prediction`](https://github.com/Reymes/football-match-prediction) | 2★, MIT — closest architecture |
| Reference fusion incl. Conformal | [`AndyDu0921/wc26-predict`](https://github.com/AndyDu0921/wc26-predict) | 32★, MIT |
| Kaunitz mispricing reference | [`Lisandro79/BeatTheBookie`](https://github.com/Lisandro79/BeatTheBookie) | (see B1 §5) |

---

## 8. Open questions / what I'd verify next

- **Does an opinion-pool weight > 0 survive on a *soft* book's price (not the sharp close)?** Pitcan tested against the closing price; the exploitable gap is pre-close and cross-book ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505) vs [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)). **[UNVERIFIED]**
- **Are lower divisions / women's leagues actually softer** for a model? No arXiv evidence found in this pass; test empirically. **[UNVERIFIED]**
- **Over/under vs 1X2 head-to-head RPS for *your* features** — Wheatcroft's profitable totals model predates modern xG/GBM; worth a direct backtest ([Wheatcroft 2020](https://doi.org/10.1016/j.ijforecast.2019.11.001)).

---

## 9. References (all fetched/verified 6 Oct 2026)

**Primary papers (arXiv)**
- Yeung, Bunker, Umemoto & Fujii (2023) — https://arxiv.org/abs/2309.14807
- Pitcan (2026) *Does a Structural Model Add Anything to the Closing Price?* — https://arxiv.org/abs/2608.11505
- Clegg, Song & Cartlidge (2026) *A market-calibrated accelerated failure time model for in-play football forecasting* — https://arxiv.org/abs/2605.16066
- Mendes-Neves, Baghoussi, Meireles, Soares & Mendes-Moreira (2025) — https://arxiv.org/abs/2501.05873
- Bunker, Yeung & Fujii (2024) survey — https://arxiv.org/abs/2403.07669
- Fischer & Heuer (2024) ML vs Poisson — https://arxiv.org/abs/2408.08331
- Walsh & Joshi (2024) accuracy vs calibration — https://arxiv.org/abs/2303.06021
- Šourek, Hubáček & Železný (2020) — https://arxiv.org/abs/2010.12508
- Egidi, Pauli & Torelli (2018) — https://arxiv.org/abs/1802.08848
- Constantinou (2020/2022) Asian handicap — https://arxiv.org/abs/2003.09384 · https://doi.org/10.3233/JSA-200588
- Goto, Takeishi & Yairi (2026) odds→probability — https://arxiv.org/abs/2604.17194
- Foulley (2021) score decompositions — https://arxiv.org/abs/2106.14345
- Feng, Polson & Xu (2016) Skellam EPL odds — https://arxiv.org/abs/1604.03614
- Uhrín, Šourek, Hubáček & Železný (2021) betting strategies — https://arxiv.org/abs/2107.08827
- Kelly limitations: https://arxiv.org/abs/1710.01786 · https://arxiv.org/abs/1710.01787 · distributionally-robust https://arxiv.org/abs/1812.10371
- Venn-Abers: https://arxiv.org/abs/1211.0025 · classifier calibration benchmark: https://arxiv.org/abs/2601.19944
- xG: Hewitt & Karakuş (2023) https://arxiv.org/abs/2301.13052 · counterfactual xG (2025) https://arxiv.org/abs/2511.23072
- Player value: plus-minus https://arxiv.org/abs/1706.04943 · lineups https://arxiv.org/abs/2210.06327
- League/market: Stömmer (2023) https://arxiv.org/abs/2303.16648 · Ren & Susnjak (2022) https://arxiv.org/abs/2211.15734 · Jiménez et al. (2023) https://arxiv.org/abs/2307.13807 · Mandadapu (2024) https://arxiv.org/abs/2403.16282 · COVID home advantage https://arxiv.org/abs/2008.05417 · goals anticipation https://arxiv.org/abs/2505.21275 · CL/EL thresholds https://arxiv.org/abs/2508.20075 · systematic review https://arxiv.org/abs/2410.21484

**Peer-reviewed / classical**
- Dixon & Coles (1997) JRSS C 46(2):265–280 — https://doi.org/10.1111/1467-9876.00065
- Karlis & Ntzoufras (2003) JRSS D 52(3):381–393 — https://doi.org/10.1111/1467-9884.00366
- Baio & Blangiardo (2010) J. Applied Statistics 37:253–264 — https://doi.org/10.1080/02664760802684177
- Constantinou & Fenton (2013) JQAS 9(1):37–50 — https://doi.org/10.1515/jqas-2012-0036
- Berrar, Lopes & Dubitzky (2019) Mach. Learn. 108:97–126 — https://doi.org/10.1007/s10994-018-5747-8
- Dubitzky, Lopes, Davis & Berrar (2019) Mach. Learn. 108:9–28 — https://doi.org/10.1007/s10994-018-5726-0
- Hubáček, Šourek & Železný (2019) IJF 35(2):783–796 — https://doi.org/10.1016/j.ijforecast.2019.01.001
- Uhrín et al. (2021) IMA J. Mgmt Math. 32(4):465–489 — https://doi.org/10.1093/imaman/dpaa029
- Wheatcroft (2020) IJF 36:916–932 — https://doi.org/10.1016/j.ijforecast.2019.11.001
- Wheatcroft (2021) J. Sports Analytics 7:77–97 — https://doi.org/10.3233/jsa-200462
- Wheatcroft (2021) "case against RPS" JQAS 17(4):273–287 — https://doi.org/10.1515/jqas-2019-0089
- Graham & Stott (2008) Applied Economics 40:99–109 — https://doi.org/10.1080/00036840701728799
- Mangold & Stübinger (2020) CARMA — https://doi.org/10.4995/carma2020.2020.11619
- Kaunitz, Zhong & Kreiner (2017) — https://arxiv.org/abs/1710.02824
- Nagel (2025) price formation — https://doi.org/10.2139/ssrn.5043714

**Repos / libraries**
- penaltyblog — https://github.com/martineastwood/penaltyblog · https://pypi.org/project/penaltyblog/
- soccerdata — https://github.com/probberechts/soccerdata · https://pypi.org/project/soccerdata/
- socceraction — https://pypi.org/project/socceraction/
- PyMC — https://pypi.org/project/pymc/
- netcal — https://pypi.org/project/netcal/ · venn-abers — https://pypi.org/project/venn-abers/ · MAPIE — https://pypi.org/project/MAPIE/
- Reymes/football-match-prediction — https://github.com/Reymes/football-match-prediction
- AndyDu0921/wc26-predict — https://github.com/AndyDu0921/wc26-predict
- Lisandro79/BeatTheBookie — https://github.com/Lisandro79/BeatTheBookie
- Data: football-data.co.uk — https://www.football-data.co.uk/ · Club Elo — http://clubelo.com/ · Understat — https://understat.com/ · FBref — https://fbref.com/
