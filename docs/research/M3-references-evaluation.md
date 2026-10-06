# M3 — Reference implementations & a rigorous backtest/evaluation protocol

**Author:** Hermes subagent (DeepSeek v4 flash) · **Date:** 6 Oct 2026 · **For:** Rayan (UNSW, Sydney)
**Read first:** `B1-football-models-evidence.md` (does the market beat us) and `B3-hermes-betting-design.md` (the `punter` architecture, CLV-first dashboard, §14 real-money gate). This note **goes deeper on three things B1/B3 did not cover**: (1) a repo-by-repo audit of *evaluation quality* — not just "what it does"; (2) the mechanics of a betting backtest that survives scrutiny (leakage, execution, commission, de-margining); (3) timing/league strategy and a pre-committed **definition-of-done** with machine-checkable gates.
**Convention:** every factual claim carries an inline source URL. Claims I could not verify from a fetched primary source this session are tagged **[UNVERIFIED]**. Repo stats were pulled live via `gh api` on 6 Oct 2026; commit dates are `pushed_at`, which differs from the "last activity" column B1 used (that was GitHub's `updatedAt`, which also moves on stars/forks).

---

## 0. TL;DR

- **There is no open repo that is a complete, honest, CLV-aware football betting system.** The closest are `R1ch1k/betting-backtester` (a *measurement rig*, MIT) and `dawilliucodes/PL-match-predictor-and-betting-backtest` (a *worked example* that reports negative results honestly). Everything with stars is a model, a tutorial, or a scraped-odds toy. **Borrow parts, not a whole system.**
- **The backtest is where projects die, not the model.** Racing Post/city-scale audits show four distinct classes of lookahead and execution assumptions that produce backtests which "could not have been executed" ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).
- **De-margining: use the power method (or Shin) for 1X2, proportional as a transparent baseline.** Clarke (2017) shows the power method "universally outperforms the multiplicative method and outperforms or is comparable to the Shin method" ([DOI 10.11648/j.ajss.20170506.12](https://doi.org/10.11648/j.ajss.20170506.12)).
- **Betfair AU commission is 6% (Market Base Rate) on net winnings per market, sport markets — not 5%** ([Betfair AU commission doc](https://betfair-datascientists.github.io/gettingStarted/commission/)); the 5% figure practitioners quote is the old UK rate ([football-data](https://www.football-data.co.uk/blog/betfair_exchange_prices.php)).
- **Bet price must be the price available at bet time; closing price is a *benchmark*, never a fill.** Buchdahl's EV identity is `EV = odds_taken / (margin-free Pinnacle price at the same timestamp) − 1`, and mis-timestamped cross-book comparisons manufacture fake edge ([football-data](https://www.football-data.co.uk/blog/opening_price_wisdom.php)).
- **Gates (this note's recommendation, tightening B3 §14):** backtest gate → paper gate of **≥1,000 settled bets across ≥2 leagues with a match-clustered bootstrap CI on CLV that excludes 0** → manual small-stakes live gate on the exchange. B3's 500 bets is a floor, not a target — see §5.

---

## 1. Reference implementations — audit table

Legend for "Evaluation quality": **WF** = walk-forward/chronological; **MKT** = benchmarks against de-margined market odds; **CLV** = reports closing-line value; **HON** = states its own limitations / reports failures.

| Repo | ⭐ | Licence | Last push | Language | What it is | Evaluation quality | Reusable parts |
|---|---|---|---|---|---|---|---|
| [martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) | 230 | MIT | 2026-10-05 | Python | Production football analytics: Poisson / bivariate Poisson / Dixon-Coles / hierarchical Bayesian, Elo-Massey-Colley-Pi ratings, **margin removal incl. Shin**, RPS and other metrics, scrapers (Understat/Club Elo/FBref), xT, StatsBomb/Opta connectors ([README](https://github.com/martineastwood/penaltyblog)) | **Model-quality only.** Shipped examples are Colab notebooks; no backtest with commission/CLV is packaged. | **The single best building block** — models, implied-odds removal, RPS metric, ratings, data scrapers, plus a `.claude/skills/penaltyblog/SKILL.md` agent file ([README](https://github.com/martineastwood/penaltyblog)) |
| [betfair-datascientists/predictive-models](https://github.com/betfair-datascientists/predictive-models) | 122 | **none declared** | 2022-02-22 | Jupyter | Betfair Data Scientists' open "end-to-end" models (EPL, AFL, Brownlow, NRL, rugby, World Cup) ([repo](https://github.com/betfair-datascientists/predictive-models)) | **Weak.** EPL folder last committed **2018-08-29** (`gh api .../commits?path=epl`); the staking notebook is a confusion-matrix/accuracy exercise ([05 notebook raw](https://github.com/betfair-datascientists/predictive-models/blob/master/epl/05.%20Analysing%20Predictions%20%26%20Staking%20Strategies.ipynb)); repo disclaimer: "we can't promise that your betting strategy will be profitable" ([README](https://github.com/betfair-datascientists/predictive-models)). No CLV, no market benchmark, no commission. | Feature-engineering patterns (rolling team form), the football-data.co.uk ingestion function — **not** the methodology |
| [betfair-datascientists.github.io](https://github.com/betfair-datascientists/betfair-datascientists.github.io) ("The Automation Hub") | 65 | MIT | 2026-10-05 | Jupyter/MkDocs | Live tutorial site: **Building a Soccer Bot parts I–III**, EPL ML walkthrough, Flumine simulations, market-movement analysis | **Tutorial-grade but the best free end-to-end onramp.** Soccer Bot Part III does **group selections to avoid overfitting** and derives **edge limits** from cumulative-profit-vs-implied-value curves; warns to use realistic stakes ("don't use $1000 stakes if you'd realistically only bet $10") ([Part III](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartIII/)). Simulated bets are placed **10 minutes before kick-off** ([Part II](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartII/)). No CLV/t-stat discipline. | The full pipeline shape: rated prices → Flumine simulation → edge-limit selection → live strategy; `commission.md`, `stakingMethods.md` |
| [betcode-org/flumine](https://github.com/betcode-org/flumine) | 248 | MIT | 2026-10-01 | Python | Betfair API wrapper **with a native simulation/backtest mode** that replays historic stream files ([betfair docs](https://betfair-datascientists.github.io/tutorials/flumineSimulations/)) | **Execution realism, not statistics.** Simulates queue position/partial matching from historic streams. | **The execution layer.** Docs list its honest limits: no market catalogue/runner metadata, no cross-matching (virtual bets), **your own bets do not move the market** (so large stakes/high odds are unrealistic), PRO-level files only, AU/NZ-racing oriented ([flumineSimulations](https://betfair-datascientists.github.io/tutorials/flumineSimulations/)) |
| [Lisandro79/BeatTheBookie](https://github.com/Lisandro79/BeatTheBookie) | 660 | GPL-3.0 | 2021-10-04 | MATLAB/Octave | Reference implementation of the Kaunitz et al. (2017) multi-bookmaker mispricing strategy + dataset ([repo](https://github.com/Lisandro79/BeatTheBookie); method [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)) | **The method is sound and the authors are honest** — the README calls the whole exercise "completely worthless, considering the time spent" once accounts get limited ([README](https://github.com/Lisandro79/BeatTheBookie), summarised in B1 §2.1). Simulation uses closing odds, i.e. it is *not* an executable-price backtest. | The **mispricing-detection logic** (bet the outlier of ~32 books against the consensus) — reimplement in Python rather than porting MATLAB |
| [kochlisGit/ProphitBet-Soccer-Bets-Predictor](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor) | 578 | MIT | 2026-04-16 | Python | Desktop ML app: downloads football-data.co.uk history, engineers team-form features, trains NN/RF/ensembles, predicts upcoming fixtures ([README](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor)) | **Weak on betting honesty.** README describes Cross-Validation + Holdout and a "Profit Balance" metric; **no walk-forward, market benchmark, commission or CLV is described in the README** [UNVERIFIED whether the code contains any]. CV on time-series match data is a leakage risk. | Feature engineering and UI ideas; explicitly **not** its evaluation methodology |
| [georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting) | 809 | MIT | 2026-09-24 | Python | "Collection of sports betting AI tools": **dataloaders** (choose a stats source + an odds source) and **bettors** that wrap any scikit-learn estimator to backtest a strategy and flag upcoming value bets ([README](https://github.com/georgedouzas/sports-betting)) | Framework-grade; whether the packaged backtest is walk-forward with commission is **[UNVERIFIED]** — the README shows the architecture, not a results table | The **dataloader/bettor abstraction** and its explicit "you always know where your data came from" design are worth copying |
| [R1ch1k/betting-backtester](https://github.com/R1ch1k/betting-backtester) | 0 | MIT | 2026-04-19 | Python | "Research-grade backtester for sports-betting strategies on 1X2 markets. Walk-forward evaluation, bootstrap CIs, Betfair-style commission" ([repo](https://github.com/R1ch1k/betting-backtester)) | **Best evaluation design I found — and it is tiny (0 stars).** WF:** `WalkForwardEvaluator` rolling train/test with chained bankroll. **Commission:** `NetWinningsCommission` aggregates per market and charges winners pro-rata (matches Betfair). **Stats:** bootstrap CI **resampling matches, not bets**, to preserve intra-match correlation. **Leakage:** "structural, not conventional" — strategies never hold a forward iterator, plus a regime-reversal test that would flip sign on a leak. **Determinism** verified byte-identical. **HON:** publishes its own failed example (below) and states it models no slippage/latency/partial fills ([README](https://github.com/R1ch1k/betting-backtester)) | **Steal the design wholesale:** the walk-forward harness, match-level bootstrap, per-market commission model, and the explicit "what this is not" section |
| [MaddocksJ27/Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics) | 0 | none | 2026-08-21 | Python | An **audit** of a 583,774-runner horse-racing backtest: "how do you tell a real edge from a leak?" ([README](https://github.com/MaddocksJ27/Backtest-forensics)) | **The best catalogue of backtest sins available.** Leak detection: **four kinds of lookahead, each with a test**; conditioning on a post-race column yields A/E 1.96 — a fabricated edge. Execution: a **volume-weighted average price is not takeable** ("you cannot bet at an average"), and a "maximum-price rule" on exchange SP bets is wrong ("the limit is a *minimum* for backs; no maximum exists") — both produced backtests that could not have been executed. Also: fixed thresholds assume stationarity, which "fails". | **Its checklist is the checklist** in §2.5 — adapt each test to football |
| [dawilliucodes/PL-match-predictor-and-betting-backtest](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest) | 0 | none | 2026-06-12 | Python | Full-stack worked example on ~6,080 EPL matches (2010/11–2025/26): leakage-safe features, expanding-window walk-forward, de-vigged market benchmark, quarter-Kelly, CLV reporting, C++ parameter-sweep engine ([README](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)) | **Honest and instructive.** Reports the model **losing to the market** (log loss 0.9647 vs Pinnacle close 0.9554; Brier 0.5709 vs 0.5654) and concludes "the closing line already prices what public match data can tell you". Chooses parameters walk-forward; settles same-day bets together to avoid intra-day lookahead; compares season vs daily retrain cadence and finds daily more realistic and better. | The **evaluation stack**: ECE/reliability vs market, market-blending `α·model+(1−α)·market`, season-vs-daily cadence comparison, walk-forward *parameter* selection |
| [zakariae-boui/football-prediction-ml](https://github.com/zakariae-boui/football-prediction-ml) | 4 | MIT | 2026-07-13 | Jupyter | RF/XGBoost/SVM with **value betting against closing odds on 6,080 matches** ([repo](https://github.com/zakariae-boui/football-prediction-ml)) | Small; claims value betting vs closing odds — depth **[UNVERIFIED]** | A compact reference for the "bet only when model beats the closing line" loop |
| [probberechts/soccerdata](https://github.com/probberechts/soccerdata) | 2,099 | Apache-2.0 (repo metadata says NOASSERTION) | 2026-09-28 | Python | Scrapers for Club Elo, ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored ([repo](https://github.com/probberechts/soccerdata)) | Data layer only | **The data layer** (also named in B1 §5) |
| [jordantete/OddsHarvester](https://github.com/jordantete/OddsHarvester) | 256 | MIT | 2026-10-05 | Python | Scrapes and processes odds data from oddsportal.com ([repo](https://github.com/jordantete/OddsHarvester)) | Input layer only | Historical **cross-book odds** for building your own sharp-reference and CLV history (ToS [UNVERIFIED]) |
| [sedemmler/WagerBrain](https://github.com/sedemmler/WagerBrain) | 316 | MIT | 2020-05-02 | Python | "The essential math required for sports betting and gambling" ([repo](https://github.com/sedemmler/WagerBrain)) | Utility library, unmaintained | Kelly/staking/odds math to unit-test against |
| [pitcany/seriea-leverage](https://github.com/pitcany/seriea-leverage) | [UNVERIFIED] | [UNVERIFIED] | [UNVERIFIED] | Python | Code + data pipeline for *"Does a Structural Model Add Anything to the Closing Price?"* (19 Serie A seasons, 7,220 matches) ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)) | **Strong methodology, published 2026-08-11.** Formalises "does the model add information the margin-free closing price hasn't absorbed?" as the **fitted weight in a logarithmic opinion pool**; finds the structural model is *better calibrated* on the home-win margin (slope 0.995 vs the market's 1.103) but *less sharp* — "the market's advantage is discrimination rather than honesty", and "value lies not in a better forecast but in what is built on a calibrated one" ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)) | **The opinion-pool test** — adopt this as your "does the model add anything?" check, on top of the usual RPS/log-loss |

**Not in the table but worth knowing:** `opisthokonta/goalmodel` (R) is a long-running Dixon-Coles-family library whose BTTS/scoreline-matrix approach is described on [opisthokonta.net](https://opisthokonta.net/); it is the R-world counterpart to `penaltyblog`.

### 1.1 What the honest examples actually report

Two of the strongest evaluation stories I found are *negative results*:

- **R1ch1k's own xG example:** a Dixon-Coles-lite model with quarter-Kelly and a 2% edge threshold, 4 EPL seasons (2020/21–2023/24, 1,520 matches), 365-day train / 90-day test walk-forward, 5% commission, £1,000 bankroll → **2,233 bets, net −£982.10, yield −5.32%, bootstrap CI [−15.85%, +5.56%]**. Their conclusion: *"four seasons of data aren't enough to pin down the true edge to better than ±10%"*, and the loss points to **miscalibration amplified by Kelly**, not a clean negative edge ([README](https://github.com/R1ch1k/betting-backtester)).
- **dawilliucodes:** 13 out-of-sample EPL seasons — the model loses to the de-vigged closing line on both log loss and Brier, and the durable edge is relocated to **execution** (line-shopping a sharp probability estimate at best-available odds), not superior prediction ([README](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)).

**Take-away for the project:** the reference implementations agree with B1 — the value is in *measurement and execution*, and the tell of a credible repo is that it publishes the run where the model lost.

---

## 2. Backtest / simulation design checklist

### 2.1 Leakage (the #1 killer)

- **Chronological, not random.** "A one-shot fit-then-evaluate on the whole dataset leaks information from the test set into training" ([R1ch1k](https://github.com/R1ch1k/betting-backtester)). Use expanding/rolling walk-forward only.
- **Features carry an as-of timestamp, not just a date.** Every feature for match *t* must be computed from data strictly before *t*'s kickoff (`shift(1)` before any rolling window) ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)). Column-level date stamps, not row-level, are where leaks hide [UNVERIFIED as a citation — this is the standard edge-case].
- **Four classes of lookahead, each with a test** ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)): (i) target/outcome columns leaking into features; (ii) post-event-derived ratings (their "Racing Post Ratings correlate −0.88 with finishing position and are assigned after the race" — the football analogue is a rating that silently incorporates the match being predicted); (iii) **future-vintage data** (a "season-to-date" table pulled *after* the season); (iv) selection/regime leakage from choosing features on the full sample. Their conditioning-on-a-post-race-column test produced an **A/E of 1.96 — a fabricated edge**; a clean de-vigged market gave **A/E 1.0000**.
- **Adopt the regime-reversal test.** Train on one set of team strengths, then feed reversed outcomes; a correct pipeline *loses* and its sign flips — "a lookahead leak would flip the sign" the other way ([R1ch1k](https://github.com/R1ch1k/betting-backtester)).
- **Intra-day leakage is real.** Settle all same-day bets together; never let a late kick-off's result influence an earlier same-day stake ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)).
- **Team-name joins silently drop matches.** Canonicalise across football-data.co.uk / The Odds API / API-Football before any join (B3 §15), and assert row counts before/after every join [UNVERIFIED as a citation — standard practice].

### 2.2 Prices: what you could actually have taken

- **Bet price = the price available at bet time. Closing price = benchmark only.** Using Pinnacle's closing price as the price you "got" is the single most common inflation; the correct use of the close is as the *denominator* of CLV ([football-data](https://www.football-data.co.uk/blog/closing_odds.php), [football-data](https://www.football-data.co.uk/blog/opening_price_wisdom.php)).
- **Timestamps must be aligned across books.** Comparing *opening* prices from different books "is not necessarily comparing like with like" — bet365 opened before Pinnacle in **48 of 50** sampled matches, and in **35** of those bet365's price had already moved by the time Pinnacle opened. Snapping cross-book opening prices to a common wall-clock moment collapsed the apparent edge (ratio sd 0.13 → 0.07; ratios >1.00 fell 29 → 14) ([football-data](https://www.football-data.co.uk/blog/opening_price_wisdom.php)).
- **A price you can't take isn't a price.** A volume-weighted average price is "the average of what others were matched at — you cannot bet at an average"; and on the exchange, an SP "maximum price" rule is a **minimum for backs** ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).
- **Record the timestamp and source of every price**, plus the *available volume* at that price — a 4.0 you can only get £30 matched on is not a 4.0 for a £200 stake (see liquidity, §2.3).

### 2.3 Realistic execution on Betfair

- **Commission (AU): 6% Market Base Rate on sport markets** (NRL 10%; AU racing 8–10%), charged on **net winnings per market**, not per bet — losses in the same market offset ([Betfair AU commission doc](https://betfair-datascientists.github.io/gettingStarted/commission/)). Model it exactly this way: per-market aggregation with pro-rata attribution to winners ([R1ch1k](https://github.com/R1ch1k/betting-backtester)).
- **Commission changes the effective price, and it bites at long odds.** With 5% commission the Betfair exchange's effective overround on a 2016/17 EPL season was **~104.0%** — "marginally above the average bookmaker" — and returns fell from **~97.5%** at prices below 1.25 to **~92%** above 10.0 ([football-data](https://www.football-data.co.uk/blog/betfair_exchange_prices.php)). At 6% the penalty is larger; **a "fair" model edge below ~6% of stake on a winning bet is not an edge on the exchange.**
- **Liquidity, partial fills, market impact.** Flumine's simulation replays historic order books but explicitly does **not** model how your own bets would have moved the market, warns this matters "if you wanted to test a strategy utilising large bet sizes or on high odds selections", and ships only with PRO-level files (AU/NZ-oriented) ([flumineSimulations](https://betfair-datascientists.github.io/tutorials/flumineSimulations/)). No open backtester in the table models slippage or latency ([R1ch1k "What this is not"](https://github.com/R1ch1k/betting-backtester)).
- **Bet at a defined clock offset, consistently.** The Betfair soccer-bot tutorial places simulated bets **10 minutes before kick-off** ([Part II](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartII/)); B3 uses T−60 (lineups) and T−5 (closing snapshot). Pick one policy *before* the backtest and hold it — the policy itself is a parameter that can be overfit.
- **Account limits are part of execution reality, not a footnote.** Soft-book strategies worked on paper until "bookmakers limited our accounts, which forced us to stop our betting completely" ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824)). A soft-book backtest that assumes unlimited stakes has no live analogue [UNVERIFIED as a quantified adjustment].

### 2.4 Statistics — deciding an edge is real

- **Report a bootstrap CI on yield, resampling at the *match* level.** Bets within a match are correlated; "resampling at the bet level would understate variance" ([R1ch1k](https://github.com/R1ch1k/betting-backtester)).
- **Sample size is brutal.** B1 §6.2 computes ~750–8,300 bets for a t≈2 result depending on odds/edge; a 5% edge at 3.00 has a **±8.9% CI on ROI at 1,000 bets** ([B1 §6.2](B1-football-models-evidence.md), computed from `Var = p(1−p)d²`, `μ = pd−1`). R1ch1k's own CI after 2,233 bets was still ±10% ([README](https://github.com/R1ch1k/betting-backtester)).
- **A t-test against the market's expectation is the Buchdahl-style test** for whether a record could arise by chance; he later validated the approximation itself ([football-data](https://www.football-data.co.uk/blog/testing_reliabiity_of_t-test_approximation_for_testing_betting_recrods.php), [football-data](https://www.football-data.co.uk/blog/luck_skill_sports_betting.php)). Use it as a *secondary* check next to the bootstrap.
- **Selection bias in single seasons is huge.** Under full market efficiency there was a **~15% probability of observing a "significant" home-advantage effect in 2 of 14 EPL seasons**; favourites returned only **+0.5% (Serie A)** and **+0.3% (EPL)** over the full period ([Winkelmann et al. 2024, J. Sports Economics](https://journals.sagepub.com/doi/10.1177/15270025231204997)).
- **Calibration ≠ sharpness, and both matter.** A structural model can be *better calibrated* than the market yet *less discriminating*; accuracy alone cannot tell the two apart ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)). Score discrimination (RPS, log loss) **and** calibration (reliability slope, ECE) separately.
- **Kelly amplifies miscalibration into ruin.** A near-zero mean edge with a badly calibrated `p` still lost 98% of the bankroll in the R1ch1k example ([README](https://github.com/R1ch1k/betting-backtester)); Buchdahl: "if you overestimate the advantage … you can easily turn a winning system into an unprofitable one. If in doubt, use a fractional Kelly plan" ([football-data](https://www.football-data.co.uk/blog/kelly_staking.php)).

### 2.5 Common backtest sins (adopt as a pre-commit checklist)

1. **Random or k-fold splits** on time-ordered data ([R1ch1k](https://github.com/R1ch1k/betting-backtester)).
2. **Post-kickoff features** — lineups, in-match xG, or a "rating" computed after the match ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).
3. **Using the closing price as the fill** ([football-data](https://www.football-data.co.uk/blog/closing_odds.php)).
4. **Cross-book opening prices compared without timestamp alignment** ([football-data](https://www.football-data.co.uk/blog/opening_price_wisdom.php)).
5. **Untakeable prices** — market averages, WAPs, max-across-books ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).
6. **Ignoring commission / turnover charge / the AU 6% MBR** ([Betfair AU](https://betfair-datascientists.github.io/gettingStarted/commission/)).
7. **Tuning the edge threshold and Kelly fraction on the whole sample** instead of walk-forward ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)).
8. **Testing many strategies, reporting the winner** (multiple testing; cf. the 15% false-positive rate in [Winkelmann et al. 2024](https://journals.sagepub.com/doi/10.1177/15270025231204997)).
9. **Ignoring liquidity/partial fills/market impact** ([flumine](https://betfair-datascientists.github.io/tutorials/flumineSimulations/)).
10. **Assuming stationarity** — thresholds that worked in one era break; "runners clearing any fixed threshold roughly quadrupled" with no change in the sport ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).
11. **Ignoring account limits** on soft books ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824)).
12. **Bet-level resampling** that understates variance ([R1ch1k](https://github.com/R1ch1k/betting-backtester)).
13. **A backtest that isn't deterministic** — "identical inputs produce byte-identical outputs" is a *tested* invariant, not an assumption ([R1ch1k](https://github.com/R1ch1k/betting-backtester)).
14. **Skipping the sanity check that de-vigged market prices score A/E ≈ 1.0000** on your own data before trusting anything downstream ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).

---

## 3. De-margining (overround removal) — comparison

### 3.1 The methods

| Method | Core assumption | Formula (from the `implied` vignette unless noted) | Needs a solver? | Known defect |
|---|---|---|---|---|
| **Proportional / multiplicative / "basic"** | same *percentage* margin on every raw probability | `p_i = r_i / Σr` | No | "does not account for favorite long-shot bias"; the vignette calls it "the least accurate of the methods in this package" ([CRAN `implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)) |
| **Additive / "difference"** | same *probability points* removed from each outcome | `p_i = r_i − (Σr − 1)/n` | No | "can give negative adjusted probabilities" ([Clarke 2017](https://doi.org/10.11648/j.ajss.20170506.12)); equivalent to Shin for 2 outcomes ([`implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)) |
| **Power / logarithmic** | one exponent maps raw→fair probabilities | `p_i = r_i^(1/k)`, k s.t. `Σp = 1` | Yes (bisection) | "exponent is a transformation parameter, not direct evidence of pricing intent" ([impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/)) |
| **Odds ratio** (Cheung) | one constant odds-ratio links fair and bookmaker probs | `OR = p_i(1−r_i) / r_i(1−p_i)` | Yes | "less common in applied football research" ([impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/)) |
| **Shin (1992/1993)** | a fraction `z` of bettors are insiders; bookmaker maximises profit | — | Yes for 3+ outcomes | "the literal insider interpretation of z is contestable" ([impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/); see [Whelan 2024](https://www.karlwhelan.com/Papers/ShinzNov24.pdf)); "both the normalization and Shin approaches can produce bookmaker probabilities greater than 1 when applied in reverse" ([Clarke 2017](https://doi.org/10.11648/j.ajss.20170506.12)) |
| **Balanced books** | bookmaker minimises worst-case loss | — | Yes | Shin ≡ additive for 2 outcomes ([`implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)) |
| **Margin weights proportional to odds (WPO)** | margin applied proportionally to the *probability* | `p_i = (n − M·O_i)/(n·O_i)` | No | Buchdahl's own method ([`implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)) |
| **Jensen–Shannon distance (Long)** | bookmaker probs are a "noisy" version of true probs | solve for JS distance `D` s.t. `Σp = 1` | Yes | Recent, little independent validation ([`implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)) |

### 3.2 Which is most accurate?

- **Clarke (2017)**, applying the methods to three large bookmaker datasets across three sports: *"the power method universally outperforms the multiplicative method and outperforms or is comparable to the Shin method"* ([DOI 10.11648/j.ajss.20170506.12](https://doi.org/10.11648/j.ajss.20170506.12)). He also notes the additive and Shin methods are **equivalent for two-competitor races**.
- **Štrumbelj (2014)**, *On determining probability forecasts from betting odds* (Int. J. Forecasting) is the standard citation for a head-to-head comparison of these methods on football data ([DOI 10.1016/j.ijforecast.2014.02.008](https://doi.org/10.1016/j.ijforecast.2014.02.008)); **I could not retrieve its abstract or conclusion this session [UNVERIFIED — its exact ranking of methods].**
- **The margin is bigger than the headline number implies.** Whelan & Hegarty (2023) show that because bookmakers set *higher* margins on lower-probability bets, average realised loss rates are "consistently higher than predicted by the conventional calculation" `(Q−1)/Q` ([MPRA 116924](https://mpra.ub.uni-muenchen.de/116924/)). So a naive overround figure understates the true cost of betting longshots — another reason to cap max odds (B3 §7 caps at 6.0).
- **Shin's `z` should not be read literally** as measured insider trading ([Whelan, *On Estimates of Insider Trading in Sports Betting*](https://www.karlwhelan.com/Papers/ShinzNov24.pdf); [impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/)).
- **Method choice moves the answer by less than you'd fear — on balanced markets.** Worked example (odds 2.10/3.50/3.40): home probability ranged 45.09%–45.76% across five methods — a **0.66pp** spread on the favourite and **0.32pp** on the away — but the source stresses this is *model sensitivity*, "not a confidence interval for the true probability", and that differences grow "in lopsided, high-margin and many-outcome markets" ([impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/)).

### 3.3 Recommendation for this project

1. **Primary: power method.** Best-evidenced for football 1X2 ([Clarke 2017](https://doi.org/10.11648/j.ajss.20170506.12)), always stays in [0,1], handles the favourite-longshot bias that proportional ignores ([`implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)).
2. **Cross-check: Shin** on the same market; if the two disagree by more than ~0.5pp on the selection you're betting, treat the pick as *method-dependent* and drop it (this is a pre-committed rule, not a post-hoc one).
3. **Publish proportional alongside** as a transparent baseline ([impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/)) and so a reviewer can reproduce you with one line.
4. **For two-way markets (Asian handicap, O/U)** the choice barely matters — additive ≡ Shin, and power/proportional converge on a near-balanced book ([`implied` vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html), [impliedscore](https://impliedscore.com/bookmaker-margin-removal-methods/)).
5. **Implement all of them and log the parameter** (`k` for power, `z` for Shin, `OR`) per market in `odds_raw`/`model_probs`; the parameters are themselves diagnostics (an implausible `z` or `k` signals bad/stale odds).
6. `penaltyblog` already ships margin removal including Shin ([README](https://github.com/martineastwood/penaltyblog)); the R `implied` package documents all the above if you want an independent cross-check ([vignette](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html)).

---

## 4. League / market selection and timing

### 4.1 Where the edge lives

| Angle | Evidence | Verdict for v1 |
|---|---|---|
| **Beat the closing sharp price (CLV)** | The ratio of the odds you bet to **Pinnacle's margin-free price at the same time** is a "good predictor" of EV; Buchdahl validated predicted vs actual returns for 4 tipsters and made correct forward predictions about their regression ([football-data](https://www.football-data.co.uk/blog/closing_odds.php), [football-data](https://www.football-data.co.uk/blog/closing_odds_2.php)). Rigorous academic proof that CLV *fully* predicts ROI: **[UNVERIFIED]** (also flagged in B1 §4) | **Yes — the headline metric.** But it's a measurement, not a strategy by itself |
| **Opening vs closing** | Pinnacle's **opening** prices are "efficient enough for the purposes of using them as a reasonable measure of assessing 'true' match result probabilities" — Shannon entropy showed "almost no difference in market efficiency between opening and closing" ([football-data](https://www.football-data.co.uk/blog/opening_price_wisdom.php)). Opening lines are not free money | Bet at your fixed policy (T−60); don't assume early = soft |
| **Steamers / drifters** | Prices that shorten "do not shorten enough", prices that lengthen "do not lengthen enough" by closing — the market is not fully efficient; the difference between steamer and drifter level-stakes profitability was statistically convincing ([football-data](https://www.football-data.co.uk/blog/steamers_drifters_revisited.php)). Betting the drift/steam direction pre-closing is where the timing edge is | **Testable angle.** Requires opening *and* pre-close snapshots — a data cost |
| **1X2 vs Asian handicap** | "There is a strong pattern of favourite–longshot bias" in the traditional 1X2 market, whereas the **Asian handicap market "can generate efficient forecasts for the same set of matches"** ([Schnytzer/Davidson? — *Forecasting soccer matches with betting odds: A tale of two markets*, IJF 2024](https://doi.org/10.1016/j.ijforecast.2024.06.013)) | **1X2 is softest; AH is sharper but 2-way (easier de-margin, lower commission drag).** B1/B3 bet 1X2 — that's the consistent choice |
| **Lower leagues / niche markets** | B3 §4 as an *unverified* hypothesis (thinner modelling → softer prices, worse data) | Later expansion only; not v1 |
| **Favourites on soft books** | +0.5% ROI Serie A / +0.3% EPL over a full multi-season window ([Winkelmann et al. 2024](https://journals.sagepub.com/doi/10.1177/15270025231204997)) — real but tiny, and pre-account-limits | A useful *benchmark strategy* the model must beat |
| **The exchange as the venue** | No margin and little FLB on liquid EPL 1X2 — **but** 5% commission pushed effective overround to ~104.0% in 2016/17 (AU MBR is now 6%) ([football-data](https://www.football-data.co.uk/blog/betfair_exchange_prices.php), [Betfair AU](https://betfair-datascientists.github.io/gettingStarted/commission/)) | Exchange is the *honest benchmark and the live venue of first resort*; the commission is the hurdle |

### 4.2 When to bet

- **A fixed, pre-declared policy beats a clever one.** B3 uses T−60 (post-lineups) and T−5 (closing snapshot); the Betfair soccer bot uses T−10 ([B3 §6](B3-hermes-betting-design.md), [Part II](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartII/)). Pick one, hold it across the whole backtest, and record the actual timestamp of every simulated bet.
- **The lineouts window is a real, distinct opportunity** (lineup news lands ~1h before KO; the Betfair docs argue pre-play prices can be inefficient because traders optimise for in-play flexibility) ([Part III](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartIII/)). But it is also where your model is most exposed to information you don't have.
- **Always compare against the price at the same timestamp.** The timestamp-alignment failure in §2.2 ([football-data](https://www.football-data.co.uk/blog/opening_price_wisdom.php)) is *the* timing trap — it manufactures edge out of clock skew.
- **Record the closing price for every bet regardless of when you bet**, so CLV is computable for every pick ([B3 §10](B3-hermes-betting-design.md)).

### 4.3 League strategy

- **v1: EPL + Championship** (deepest free data on football-data.co.uk, incl. closing odds columns; B3 §3) plus, if same-timezone convenience matters, **A-League** ([B3 §0](B3-hermes-betting-design.md)).
- **Add a second and third European league only to reach the ≥2-leagues gate**, not to chase edge; the Winkelmann results show between-league return dispersion is large ([Winkelmann et al. 2024](https://journals.sagepub.com/doi/10.1177/15270025231204997)).
- **Keep the market set to 1X2 + O/U + BTTS** (B3 §4). Every extra market multiplies the multiple-testing problem.

---

## 5. Evaluation protocol & "definition of done"

Three gates, evaluated **in order**, each on evidence written to the ledger (`db/punter.sqlite` + `journal/*.jsonl`, B3 §8). A gate that is not met is a *stop*, not a "continue and watch".

### Gate 0 — Backtest gate (no paper bets until this passes)

| # | Requirement | Acceptance test |
|---|---|---|
| B0.1 | **De-vig sanity** | On the fitted dataset, de-vigged market probabilities score **A/E ≈ 1.000** against outcomes ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)). If not, nothing downstream is trustworthy — fix the data first |
| B0.2 | **Walk-forward only** | Expanding/rolling windows, retrained at the daily or matchweek cadence ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)); no random split anywhere |
| B0.3 | **No lookahead (tested, not asserted)** | The four leak tests pass; the **regime-reversal test** flips the sign ([R1ch1k](https://github.com/R1ch1k/betting-backtester), [Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)) |
| B0.4 | **Determinism** | Byte-identical outputs on re-run with fixed seed ([R1ch1k](https://github.com/R1ch1k/betting-backtester)) |
| B0.5 | **Market benchmark reported** | RPS / log loss / Brier / reliability (slope + ECE) for **model vs de-vigged sharp close**, on every season ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest), [arXiv:2608.11505](https://arxiv.org/abs/2608.11505)) |
| B0.6 | **Model adds information** | The log-opinion-pool weight on your model, given the margin-free closing price, has a 95% CI excluding 0 — otherwise the model is decoration ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)) |
| B0.7 | **Baselines beaten** | Beat "back the favourite" and "back the shortest-priced selection" on the same ledger ([R1ch1k](https://github.com/R1ch1k/betting-backtester), [Winkelmann et al. 2024](https://journals.sagepub.com/doi/10.1177/15270025231204997)) |
| B0.8 | **Execution modelled** | AU 6% MBR per-market commission; a defined bet-timestamp policy; stake sizes within realistic liquidity; no takeable-price violations ([Betfair AU](https://betfair-datascientists.github.io/gettingStarted/commission/), [Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)) |
| B0.9 | **Parameters chosen in-fold** | Edge threshold, Kelly fraction and λ-decay selected walk-forward, never on the full sample ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)) |
| B0.10 | **Honest reporting of failures** | The report includes the run where the strategy lost, with its CI ([R1ch1k README](https://github.com/R1ch1k/betting-backtester)) |

**Note:** passing Gate 0 does **not** require positive ROI. A model can be a *useful, well-calibrated* forecast and still be behind the market ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest), [arXiv:2608.11505](https://arxiv.org/abs/2608.11505)). Gate 0 certifies the *measurement*; Gates 1–2 certify the *edge*.

### Gate 1 — Paper-trading gate (B3 §14, tightened)

Preconditions: Gate 0 passed; the paper desk is running unchanged (no re-tuning mid-flight; any change resets the sample).

| # | Requirement | Acceptance test |
|---|---|---|
| P1.1 | **Sample** | **≥1,000 settled bets**, across **≥2 leagues** and **≥3 months** — a floor, not a target; B1 §6.2 shows a 5% edge at 3.00 still has a ±8.9% ROI CI at n=1,000 ([B1 §6.2](B1-football-models-evidence.md)). Prefer 2,000+; expect ±10% even at 2,233 ([R1ch1k](https://github.com/R1ch1k/betting-backtester)) |
| P1.2 | **CLV is positive and significant** | Mean **log-CLV** (`ln(bet_price / margin-free closing_price)`) **> 0** with a 95% **match-clustered bootstrap CI** (≥10,000 resamples) **excluding 0** ([R1ch1k](https://github.com/R1ch1k/betting-backtester), [football-data](https://www.football-data.co.uk/blog/closing_odds.php)) |
| P1.3 | **Same sign across staking** | Positive on **both** the flat-stake and quarter-Kelly portfolios — a flat-stake edge that vanishes under Kelly means miscalibration, not skill ([B3 §10](B3-hermes-betting-design.md), [football-data](https://www.football-data.co.uk/blog/kelly_staking.php)) |
| P1.4 | **Calibration** | Reliability slope within [0.9, 1.1] and ECE within tolerance; model log loss / RPS no worse than the de-vigged closing benchmark ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest)) |
| P1.5 | **Dependent on method?** | CLV sign unchanged when the de-vig method switches from power to Shin (§3.3 rule) |
| P1.6 | **Robustness** | Positive CLV holds in **each** league separately and in **each** season/half of the sample — no single window carrying it ([Winkelmann et al. 2024](https://journals.sagepub.com/doi/10.1177/15270025231204997)) |
| P1.7 | **Discipline held** | Drawdown never breached the pre-set cap; the kill switch demonstrably fired in a drill ([B3 §9](B3-hermes-betting-design.md)) |
| P1.8 | **A/B settled** | The LLM arm beats the control on CLV/RPS, or is proven neutral and demoted to narrator — no "later" ([B3 §11](B3-hermes-betting-design.md)) |

### Gate 2 — Live-money gate (manual, tiny, reversible)

Only if **all** of P1.1–P1.8 hold. Then: **manual flip, never automated**; a separate live-key secret scope; a per-bet and per-day cash cap hard-coded; a kill file; a small initial bankroll; **start on the exchange** (smallest paper→live gap) ([B3 §14](B3-hermes-betting-design.md)). Continue to compute CLV and ROI against the same benchmarks. Any of these re-opens Gate 1 and halts live staking: mean CLV crossing below 0 over a rolling 250-bet window; calibration slope drifting outside [0.85, 1.15]; drawdown breach ([B3 §9](B3-hermes-betting-design.md)).

### 5.1 What "done" means if the edge never appears

The honest default (B1 §7) is that the desk stays paper-only and the deliverables are the auditable ledger, the leakage-proof evaluation harness, and the calibration/CLV dashboard — all of which are portfolio-grade ML+engineering artefacts regardless of whether a profit ever materialises ([dawilliucodes](https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest), [R1ch1k](https://github.com/R1ch1k/betting-backtester)).

---

## 6. Concrete build deltas vs B3

Small, high-leverage changes to the B3 design based on this note:

1. **Add `backtest.py` modelled on `R1ch1k/betting-backtester`** (walk-forward harness, per-market Betfair commission, match-level bootstrap, regime-reversal + determinism tests) rather than hand-rolling the loop.
2. **De-margin with the power method by default; log `k` and a Shin cross-check per market** (§3.3) — `penaltyblog` ships the primitives.
3. **Add a de-vig A/E ≈ 1.000 sanity check as job #0** before any fit ([Backtest-forensics](https://github.com/MaddocksJ27/Backtest-forensics)).
4. **Add the log-opinion-pool weight test** to the dashboard: does the model add anything to the margin-free close? ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505)).
5. **Set commission to 6% (AU MBR) in `config/limits.json`** instead of a 5% placeholder ([Betfair AU](https://betfair-datascientists.github.io/gettingStarted/commission/)).
6. **Raise the paper gate from 500 to ≥1,000 bets** (B3 §14.1) and add P1.5–P1.6.
7. **Add the steamer/drifter angle as a v1.1 test** — it needs opening + pre-close snapshots, which The Odds API historical tier provides (paid; B3 §3).

---

## 7. References

**Repos (stats via `gh api`, 6 Oct 2026)**
- martineastwood/penaltyblog — https://github.com/martineastwood/penaltyblog
- betfair-datascientists/predictive-models — https://github.com/betfair-datascientists/predictive-models
- betfair-datascientists/betfair-datascientists.github.io (The Automation Hub) — https://github.com/betfair-datascientists/betfair-datascientists.github.io
- betcode-org/flumine — https://github.com/betcode-org/flumine
- Lisandro79/BeatTheBookie — https://github.com/Lisandro79/BeatTheBookie
- kochlisGit/ProphitBet-Soccer-Bets-Predictor — https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor
- georgedouzas/sports-betting — https://github.com/georgedouzas/sports-betting
- R1ch1k/betting-backtester — https://github.com/R1ch1k/betting-backtester
- MaddocksJ27/Backtest-forensics — https://github.com/MaddocksJ27/Backtest-forensics
- dawilliucodes/PL-match-predictor-and-betting-backtest — https://github.com/dawilliucodes/PL-match-predictor-and-betting-backtest
- zakariae-boui/football-prediction-ml — https://github.com/zakariae-boui/football-prediction-ml
- probberechts/soccerdata — https://github.com/probberechts/soccerdata
- jordantete/OddsHarvester — https://github.com/jordantete/OddsHarvester
- sedemmler/WagerBrain — https://github.com/sedemmler/WagerBrain

**Betfair docs / tutorials**
- Building a Soccer Bot Part I — https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartI/
- Building a Soccer Bot Part II — https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartII/
- Building a Soccer Bot Part III — https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartIII/
- EPL Machine Learning (Python) — https://betfair-datascientists.github.io/modelling/EPLmlPython/
- Flumine Simulations — https://betfair-datascientists.github.io/tutorials/flumineSimulations/
- Commission and Charges (AU Market Base Rate = 6% for sport) — https://betfair-datascientists.github.io/gettingStarted/commission/

**De-margining**
- Clarke (2017) Adjusting Bookmaker's Odds to Allow for Overround — https://doi.org/10.11648/j.ajss.20170506.12
- Štrumbelj (2014) On determining probability forecasts from betting odds — https://doi.org/10.1016/j.ijforecast.2014.02.008 *(method ranking [UNVERIFIED] this session)*
- Lindstrøm, `implied` R package vignette (all methods + formulas) — https://cran.r-project.org/web/packages/implied/vignettes/introduction.html
- penaltyblog implied-odds docs — https://penaltyblog.readthedocs.io/en/latest/implied/
- Whelan & Hegarty (2023) Calculating the Bookmaker's Margin — https://mpra.ub.uni-muenchen.de/116924/
- Whelan (2024) On Estimates of Insider Trading in Sports Betting — https://www.karlwhelan.com/Papers/ShinzNov24.pdf
- impliedscore — bookmaker margin-removal methods comparison — https://impliedscore.com/bookmaker-margin-removal-methods/
- Krok Odds — devigging (practitioner, method-per-market-shape) — https://krokodds.com.au/learn/devigging

**CLV / timing / market efficiency (practitioner, football-data.co.uk)**
- Using Pinnacle.com's Closing Line to Predict Profits — https://www.football-data.co.uk/blog/pinnacle_efficiency.php
- Using the Closing Betting Odds to test for a Tipster's Skill — https://www.football-data.co.uk/blog/closing_odds.php
- …Part 2 — https://www.football-data.co.uk/blog/closing_odds_2.php
- Efficiency of Pinnacle's Opening Prices compared to bet365 — https://www.football-data.co.uk/blog/opening_price_wisdom.php
- Steamers and Drifters Revisited — https://www.football-data.co.uk/blog/steamers_drifters_revisited.php
- Testing your Betting Model (WoC raw performance) — https://www.football-data.co.uk/blog/model_testing.php
- What is the true expected profit for the Wisdom of the Crowd system? — https://www.football-data.co.uk/blog/wisdom_of_crowd_betting_system_closing_odds.php
- The Pitfalls of using Kelly Staking — https://www.football-data.co.uk/blog/kelly_staking.php
- How Good are Betfair Exchange Prices Really? — https://www.football-data.co.uk/blog/betfair_exchange_prices.php
- Testing the Reliability of my Betting Record Test Formula — https://www.football-data.co.uk/blog/testing_reliabiity_of_t-test_approximation_for_testing_betting_recrods.php
- Luck Versus Skill in Sports Betting — https://www.football-data.co.uk/blog/luck_skill_sports_betting.php
- Quantifying Predictive Error in Football Models — https://www.football-data.co.uk/blog/football_models_predictive_error.php

**Academic**
- Schnytzer-style two-market study (2024) Forecasting soccer matches with betting odds: A tale of two markets — https://doi.org/10.1016/j.ijforecast.2024.06.013
- Winkelmann, Ötting, Deutscher & Makarewicz (2024) Are Betting Markets Inefficient? — https://journals.sagepub.com/doi/10.1177/15270025231204997
- Pitcan (2026) Does a Structural Model Add Anything to the Closing Price? (Serie A) — https://arxiv.org/abs/2608.11505
- Kaunitz, Zhong & Kreiner (2017) Beating the bookies with their own numbers — https://arxiv.org/abs/1710.02824
- Mandadapu (2024) The Evolution of Football Betting — https://arxiv.org/abs/2403.16282
- Karimov et al. (2025) Domain-Driven Identification of Football Probabilities — https://doi.org/10.3390/math13243976 *(de-vig method comparison; details not extracted this session)*

**Internal**
- B1-football-models-evidence.md (§2 evidence, §5 repo table, §6.2 sample size) — C:/Users/Rayan/hermes-research/projects/notes/B1-football-models-evidence.md
- B3-hermes-betting-design.md (§3 data, §4 model, §6 cron, §7 staking, §10 dashboard, §14 gate) — C:/Users/Rayan/hermes-research/projects/notes/B3-hermes-betting-design.md

**Verified this session (fetched live):** all repo metadata and READMEs in §1; Betfair AU commission = 6% sport MBR; Clarke 2017 abstract; implied-vignette method set; the football-data.co.uk articles quoted; R1ch1k's example numbers; Flumine's documented limitations; The Automation Hub soccer-bot parts I–III.
**Carried from B1/B3 (not re-fetched):** Kaunitz arXiv figures; B1 §6.2 sample-size table; B3 architecture.
**UNVERIFIED markers:** Štrumbelj 2014 method ranking; georgedouzas/sports-betting backtest internals; ProphitBet code contents beyond its README; pitcany/seriea-leverage stars/licence; OddsHarvester ToS; the claim that CLV *fully* predicts long-run ROI.
