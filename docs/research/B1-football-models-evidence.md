# B1 — Can football (soccer) betting be beaten with models? Evidence review

**For:** Rayan (UNSW computer engineering, Sydney) — building an AI/ML match-prediction bot, paper-betting first.
**Date:** 6 Oct 2026 · **Scope:** research-only synthesis (papers, arXiv, GitHub). No bets placed, nothing installed.
**Convention:** every factual claim carries an inline source URL. Claims I could not verify from a fetched primary source are tagged **[UNVERIFIED]**. Numbers quoted are as they appear in the cited source.

---

## 1. Executive summary

- The football betting market is **weakly inefficient, not efficiently beatable at scale**. Academic strategies have produced real but small positive returns in simulation, and the people who achieved them had their **accounts restricted or closed within weeks** once real money was involved. [Kaunitz et al. 2017, arXiv:1710.02824](https://arxiv.org/abs/1710.02824)
- The **bookmaker's own odds are usually a better predictor than any public model.** In the 2017 Soccer Prediction Challenge test set, bookmaker odds scored the best Ranked Probability Score (RPS 0.2020); the best model (XGBoost) scored RPS 0.2063 — close, but still worse. [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)
- The authors of the most famous "we beat the bookies" paper concluded it was **"completely worthless, considering the time spent on the betting and the monetary reward,"** once account restrictions are accounted for. [BeatTheBookie README, github.com/Lisandro79/BeatTheBookie](https://github.com/Lisandro79/BeatTheBookie)
- **Where a real edge plausibly exists** is not "predict matches better" — it's "find soft-book prices that lag a sharp reference line, and stake small." That is an engineering/execution problem (data pipeline, line tracking, selection, staking), not a modelling one.
- **Honest expectation for a student:** treat this as a machine-learning + engineering project with a genuine shot at *small* long-run positive ROI (~1–3%) on soft books, subject to account limits. Do **not** expect a money printer. Positive closing-line value (CLV) is the signal to chase, not win-rate.

---

## 2. Evidence: is the market beatable?

### 2.1 The strongest positive result — and its catch

Kaunitz, Zhong & Kreiner (2017), *"Beating the bookies with their own numbers — and how the online sports betting market is rigged."* [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)

Their method did **not** build a better forecasting model. It exploited the implied probability information already in the odds of ~32 bookmakers (including Interwetten, bwin, bet365, William Hill, Pinnacle Sports, Paddy Power, SBOBET) to detect mispriced games, then bet the outlier price. Reported results:

| Test | Accuracy | Return | Volume |
|---|---|---|---|
| 10-year historical simulation, closing odds (Jan 2005–Jun 2015) | 44.4% | **+3.5%** | 56,435 bets (≈$98,865 at $50/bet) |
| 5 months betting real money | 47% | **+8.5%** (profit $957.50) | 265 bets |

(All figures from the paper body, [arXiv:1710.02824](https://arxiv.org/abs/1710.02824).)

**The catch, in the paper's own words:** "A few months after we began placing bets with real money bookmakers limited our accounts, which forced us to stop our betting completely" ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824)). The companion repo states the authors "would definitely bet on these games if the bookies did not block our accounts" but that the effort is "completely worthless" net of time and reward ([BeatTheBookie README](https://github.com/Lisandro79/BeatTheBookie)).

So: the market was beatable *on paper*, and the bookmakers' response — not the market — was the binding constraint.

### 2.2 The market resizes to absorb any published edge

Mangold & Stübinger (2020), *"Investigating inefficiencies of bookmaker odds in football using machine learning"* (CARMA 2020): "by using simple machine learning models we can systematically outperform the markets belief manifested through the bookmakers odds. **The effect of this inefficiency is diminishing over time**, which indicates that the knowledge that has been derived from and the pure amount of the data is also reflected in the odds in recent times." [DOI: 10.4995/carma2020.2020.11619](https://doi.org/10.4995/carma2020.2020.11619)

Translation: whatever edge a public method finds tends to get priced away once it is public — which is exactly what happens to any edge a student publishes on GitHub.

### 2.3 Bookmaker odds are the benchmark to beat (and they usually win)

**2017 Soccer Prediction Challenge** (the canonical benchmark), test-set results as tabulated in [arXiv:2309.14807](https://arxiv.org/abs/2309.14807):

| Method | RPS (lower is better) | Accuracy |
|---|---|---|
| **Bookmaker odds** | **0.2020** | 0.5194 |
| ELO ordered logit | 0.2035 | 0.5146 |
| ELO + offensive/defensive, ordered logit | 0.2045 | 0.5146 |
| Berrar et al. (2019), rating features | 0.2054 | 0.5194 |
| XGBoost classification (best model on the test set) | 0.2063 | 0.5243 |
| Constantinou (2019), hybrid Bayesian | 0.2083 | 0.5146 |
| Hubáček et al. (2019), pi-ratings + relational | 0.2087 | 0.5388 |
| k-Nearest-Neighbours on rating features (best *during* competition) | 0.2149 | 0.5049 |

A separate 2023-challenge study ([arXiv:2501.05873](https://arxiv.org/abs/2501.05873)) reports the **bookmaker RPS ≈ 0.198** and bookmaker margin ≈ **−5.6%** on its test set, while their own best model reached RPS 0.201. Its betting strategies returned only **+1.1% to +6.7% over baseline**.

Take-away: models cluster within a few thousandths of RPS of the bookmaker, but the bookmaker is consistently *slightly* ahead. Beating it on RPS is possible in isolated cases; beating it *net of margin and after it learns* is the hard part.

### 2.4 Other honest findings

- **High accuracy ≠ profit.** "Wages of wins: could an amateur make money from match outcome predictions?" finds it possible "yet not without its pitfalls… high accuracy does not automatically equal high pay-out," because of which match-ups a model gets right. [arXiv:1702.05982](https://arxiv.org/abs/1702.05982)
- **Model choice matters less than data/features.** "Match predictions in soccer: Machine learning vs. Poisson approaches" concludes the neural network is "comparable to a Poisson model" and "the choice of model is less important than the quality of the data and the associated features." [arXiv:2408.08331](https://arxiv.org/abs/2408.08331)
- **Calibration beats accuracy for betting.** "Machine learning for sports betting: should model selection be based on accuracy or calibration?" argues and shows that selecting models on *calibration* (reliability of probabilities) outperforms selecting on accuracy for the betting problem. [arXiv:2303.06021](https://arxiv.org/abs/2303.06021)
- **A "bad" model can still make money.** "Beating the market with a bad predictive model" proves you can profit with a model that is *worse* at predicting prices than the market, by deliberately decorrelating it from the market to exploit its biases. [arXiv:2010.12508](https://arxiv.org/abs/2010.12508)
- **US markets show inefficiencies too** (NFL/NBA/NCAA/WNBA), using a non-parametric win-probability model to find positive-EV spots. [arXiv:1910.08858](https://arxiv.org/abs/1910.08858)
- **Influencers lose.** A 2026 study tracked 5,467 pre-match slips from three Nigerian tipsters (~$4.8M staked) and found the opposite of the wealth displayed online. [arXiv:2604.08251](https://arxiv.org/abs/2604.08251)
- **Favourite–longshot bias persists** in modern markets: on Polymarket (588M trades), purchases below 10¢ lost 19.3¢ per dollar while purchases ≥90¢ earned 0.83¢. [arXiv:2609.12878](https://arxiv.org/abs/2609.12878)
- **Do LLMs beat the market?** Not demonstrated. The "AI World Cup 2026" benchmark had ten LLM assistants forecast the whole tournament; the best group-stage outcome accuracy was Claude Sonnet 4.6 at 63.89%, and self-reported confidence was uncorrelated with accuracy (r ≈ −0.06). Nobody benchmarked these against closing odds. [arXiv:2608.03416](https://arxiv.org/abs/2608.03416)
- **Bookmaker profit is mostly hold + public bias**, not superior prediction of any single game — decomposed formally as a "profit-bias identity." [arXiv:2609.06739](https://arxiv.org/abs/2609.06739)

---

## 3. Model approaches and expected accuracy

Typical published performance (RPS on 1X2; lower is better; bookmaker ≈ 0.198–0.202 is the bar to clear):

| Approach | What it is | Typical RPS / accuracy | Effort | Notes |
|---|---|---|---|---|
| **Elo / pi-ratings** | Update a team strength from results; convert to probabilities | RPS ≈ 0.203–0.209, acc ≈ 0.51–0.54 [arXiv:2309.14807](https://arxiv.org/abs/2309.14807) | Low | pi-ratings (Constantinou) beat Elo slightly. Strong baseline. |
| **Poisson / bivariate Poisson** | Model goals as Poisson processes | Comparable to ML on goals [arXiv:2408.08331](https://arxiv.org/abs/2408.08331) | Low–med | Interpretable; bivariate variant models score correlation. |
| **Dixon–Coles** | Poisson + low-score correction + time decay | Standard academic workhorse | Low–med | Implemented in `penaltyblog`. [github](https://github.com/martineastwood/penaltyblog) |
| **Gradient boosting (XGBoost/CatBoost/GBM)** | Features → win/draw/loss probabilities | RPS ≈ 0.206–0.213 [arXiv:2309.14807](https://arxiv.org/abs/2309.14807), [arXiv:2501.05873](https://arxiv.org/abs/2501.05873) | Med | Best *model* on the 2017 challenge test set (XGBoost, RPS 0.2063) — still behind bookmaker odds. |
| **Neural nets / deep learning** | MLP / LSTM / hybrids | ~0.21–0.22 RPS in review tables; best LSTM accuracy 43.5% [arXiv:2410.21484](https://arxiv.org/abs/2410.21484) | Med–high | No consistent edge over GBM or Poisson. |
| **xG / shot-based models** | Use expected goals (shot quantity × quality) instead of goals | RPS 0.201 (2023 challenge) [arXiv:2501.05873](https://arxiv.org/abs/2501.05873) | Med | Data/features matter more than model class [arXiv:2408.08331](https://arxiv.org/abs/2408.08331). |
| **Bayesian hierarchical** | Full posteriors over team strengths | RPS ≈ 0.208 [arXiv:2309.14807](https://arxiv.org/abs/2309.14807) | Med–high | Better uncertainty handling; big library support in `penaltyblog`. |
| **LLM-based** | Prompt an LLM to predict | Not benchmarked vs odds; confidence uncorrelated with accuracy [arXiv:2608.03416](https://arxiv.org/abs/2608.03416) | Low | Interesting UX; **no evidence of a betting edge**. |

**Key metric note:** for 1X2 use **RPS** (ranked probability score) as the headline, plus **log loss / Brier** for calibration. Accuracy alone hides whether the model's probabilities are usable for staking. [arXiv:2303.06021](https://arxiv.org/abs/2303.06021), [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)

---

## 4. Where a realistic edge comes from (and the catch for each)

| Source of edge | How it works | Realistic for a student? | Catch |
|---|---|---|---|
| **Value betting vs a sharp reference** | Find soft-book prices longer than a sharp book (Pinnacle) implies; bet only positive-EV | **Yes — most promising** | Soft books limit/close winning accounts fast [arXiv:1710.02824](https://arxiv.org/abs/1710.02824); you are competing with pros for the same stale prices |
| **Early lines vs closing** | Openers are noisier/biased; closing lines are sharp. Bet early if you can predict the drift | Partly | Opening prices are "often noisy and biased" and become accurate only near closing — so the edge is in *predicting the move*, which is hard. [Nagel 2025, SSRN 5043714](https://doi.org/10.2139/ssrn.5043714) |
| **Closing-line value (CLV) tracking** | Beat the closing price; if you consistently do, long-run profit usually follows | **Yes — as a measurement tool** | Widely-held practitioner principle; I found no rigorous paper that *proves* CLV fully predicts ROI — treat as **[UNVERIFIED]** but useful. Supported by closing odds being the most accurate prices [Nagel 2025](https://doi.org/10.2139/ssrn.5043714), [arXiv:1710.02824](https://arxiv.org/abs/1710.02824) |
| **Niche / lower leagues** | Books invest less modelling effort in lower divisions → softer prices | Yes, in principle | Data is thinner/noisier; liquidity and stake limits are small; **[UNVERIFIED]** that edges survive costs |
| **Team news / lineups timing** | Bet when lineups drop but before books adjust | Partly | Requires fast, reliable feeds; edges are seconds-to-minutes and easily arbitraged; live-price reaction documented in [arXiv:2108.00821](https://arxiv.org/abs/2108.00821) |
| **In-play betting** | Exploit slow in-play price adjustment to events | Hard | Fast-moving; books have latency protection and stake caps; needs real-time infra [arXiv:2108.00821](https://arxiv.org/abs/2108.00821), [arXiv:2401.06086](https://arxiv.org/abs/2401.06086) |
| **Player props** | Model individual player markets (shots, goals) | Hard | Separate data + models; thin research base; **[UNVERIFIED]** |
| **Arbitrage** | Back all outcomes across books for guaranteed profit | No | Margins + account limits make it fragile; books detect and cut arbers; **[UNVERIFIED]** |
| **Exploit market-maker biases** | Train a model *decorrelated* from the market to exploit persistent mispricing | Advanced | Requires care and data; proven in principle only [arXiv:2010.12508](https://arxiv.org/abs/2010.12508) |

**Sharp vs soft bookmakers:** the Kaunitz study used ~32 books spanning both soft books (bet365, William Hill, Paddy Power, Ladbrokes…) and the sharp book Pinnacle, and it was the *soft* books' mispricings the strategy fed on — which is why those same books limited the accounts. [arXiv:1710.02824](https://arxiv.org/abs/1710.02824) The general "Pinnacle is sharp, soft books are beatable but restrictive" rule is practitioner consensus — **[UNVERIFIED]** academically as a blanket statement.

---

## 5. Open-source projects (stars / licence / last activity as of 6 Oct 2026)

| Repo | What it is | Language | Stars | Licence | Last activity | Verdict |
|---|---|---|---|---|---|---|
| [martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) | Production football analytics: Poisson, bivariate Poisson, Dixon–Coles, Bayesian/hierarchical, Elo/Massey/Pi ratings, odds→probability (margin removal), scrapers, xT | Python | 230 | MIT | 2026-10-05 | **Best single building block.** Actively maintained, has an agent skill file. |
| [probberechts/soccerdata](https://github.com/probberechts/soccerdata) | Scrape match data from Club Elo, ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored | Python | 2,099 | Apache-2.0 | 2026-10-05 | **Best data layer.** Huge star count, actively maintained. |
| [Lisandro79/BeatTheBookie](https://github.com/Lisandro79/BeatTheBookie) | Reference implementation of the Kaunitz et al. mispricing strategy + dataset | MATLAB/Octave | 660 | GPL-3.0 | 2026-10-06 | Great for reference; MATLAB is awkward for a Python stack. |
| [kochlisGit/ProphitBet-Soccer-Bets-Predictor](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor) | ML soccer-bet predictor (neural nets, random forests, ensembles) | Python | 578 | MIT | 2026-10-01 | Useful as a feature-engineering reference; no honest backtest proof. |
| [mhaythornthwaite/Football_Prediction_Project](https://github.com/mhaythornthwaite/Football_Prediction_Project) | Popular tutorial-grade football prediction project | Python | 311 | MIT | 2026-10-01 | Learning reference only. |
| [Reymes/football-match-prediction](https://github.com/Reymes/football-match-prediction) | Dixon–Coles + Poisson + Elo + LightGBM ensemble, walk-forward backtest, paper-money web UI | Python | 2 | MIT | 2026-09-07 | Small but *architecturally closest* to what Rayan wants (leakage-safe walk-forward + paper UI). |

Gap worth noting: most high-star repos are either **tutorials** or **model-only**; few ship a disciplined, leakage-safe, CLV-aware paper-trading loop end to end ([Reymes/football-match-prediction](https://github.com/Reymes/football-match-prediction) is the closest). That gap is where the student project adds value.

---

## 6. Staking & evaluation

### 6.1 Staking

- **Kelly / fractional Kelly.** The formal optimum is Kelly, but practical reviews find the **adaptive variant of fractional Kelly** "a very suitable choice across a wide range of settings," and that risk-control modifications are practically necessary because Kelly's assumptions don't hold. [arXiv:2107.08827](https://arxiv.org/abs/2107.08827)
- **Recommendation:** quarter- to half-Kelly with a hard per-bet cap (e.g. ≤1–2% of bankroll). Flat staking is fine for *measuring* edge; Kelly-family for *exploiting* it.
- **Bankroll rules:** set aside a fixed, expendable bankroll; never replenish mid-experiment; stop if drawdown exceeds a pre-set rule. (Standard risk practice; **[UNVERIFIED]** as a specific citation.)

### 6.2 How many bets to tell skill from luck

A strategy with a true edge still needs a large sample before results are statistically distinguishable from noise. For unit stakes at decimal odds *d* with true win probability *p*, per-bet return has mean `μ = p·d − 1` and variance `p(1−p)·d²`; you need roughly `N ≈ (t²·variance)/μ²` bets for a t-stat of *t*. **Computed here** (deterministic, not from a paper):

| Odds | True win prob | Edge | Bets for t = 2 (≈95%) |
|---|---|---|---|
| 1.50 | 0.700 | +5% | ~756 |
| 2.00 | 0.530 | +6% | ~1,107 |
| 2.50 | 0.420 | +5% | ~2,436 |
| 3.00 | 0.360 | +8% | ~1,296 |
| 3.00 | 0.350 | +5% | ~3,276 |
| 5.00 | 0.215 | +7.5% | ~3,000 |
| 6.00 | 0.175 | +5% | ~8,316 |

For a 5% edge at odds 3.0, the 95% CI on ROI is **±8.9% at 1,000 bets**, ±5.1% at 3,000 bets, ±4.0% at 5,000 bets — i.e. at 1,000 bets a genuinely profitable strategy still looks like a coin flip. Plan for **thousands** of bets, or a paper-trading period measured in seasons, before believing any edge.

### 6.3 Evaluation protocol (do this, in order)

1. **Walk-forward backtest**, never random splits — train on past, predict future, roll.
2. Report **RPS + log loss + Brier + calibration curve**, not just accuracy. [arXiv:2303.06021](https://arxiv.org/abs/2303.06021)
3. Benchmark against **bookmaker-implied probabilities** (margin-removed) every time. [arXiv:2309.14807](https://arxiv.org/abs/2309.14807)
4. Track **closing-line value** on paper bets — the leading indicator of skill. [Nagel 2025](https://doi.org/10.2139/ssrn.5043714) *(as a profit predictor: [UNVERIFIED])*
5. Compare to a **trivial baseline** (bet the favourite, or the bookmaker's own probabilities) — many models lose to these. [arXiv:1702.05982](https://arxiv.org/abs/1702.05982)
6. **Paper-bet first**, log every bet's price and closing price, for long enough to clear the sample-size bar above.

---

## 7. Recommendation for Rayan

**Verdict: build the project, but build it as an ML/engineering system that measures edge — not as a money printer.**

The most promising, honest version of your idea, given the evidence:

1. **Frame the goal as "positive closing-line value on soft books," not "predict matches better than bookmakers."** Bookmaker odds beat every public model on the canonical benchmark ([arXiv:2309.14807](https://arxiv.org/abs/2309.14807)), so the win condition is finding *stale prices*, not better forecasts.

2. **Stack the proven building blocks instead of writing models from scratch:**
   - Data: [`soccerdata`](https://github.com/probberechts/soccerdata) (Football-Data.co.uk gives historical odds for free) + free xG from Understat/FBref.
   - Models: [`penaltyblog`](https://github.com/martineastwood/penaltyblog) — Dixon–Coles / Bayesian / Pi-ratings / odds→probability with a couple of lines.
   - Your LLM (DeepSeek) is best used for **explaining picks and orchestrating**, not predicting — there's no evidence LLMs beat the market ([arXiv:2608.03416](https://arxiv.org/abs/2608.03416)).

3. **The edge you actually test:** for each fixture, take the margin-removed implied probabilities from a sharp reference (Pinnacle) and compare to soft books. Bet only where the soft price exceeds the fair price by a threshold. This is the Kaunitz mechanism, adapted. [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)

4. **Be disciplined about proof:** walk-forward backtest, RPS/log-loss vs the bookmaker baseline, then a **season-long paper-trading phase** logging CLV. Only after thousands of bets and a clear positive CLV would real (tiny) stakes be justified — and expect account limits if you ever do succeed. [arXiv:1710.02824](https://arxiv.org/abs/1710.02824)

5. **Stake quarter-Kelly, cap 1–2% per bet, fixed bankroll.** [arXiv:2107.08827](https://arxiv.org/abs/2107.08827)

**Honest expectations:**
- Realistic ceiling on soft books, before account limits: **~1–3% long-run ROI**, with high variance and multi-season to prove ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824): 3.5% sim / 8.5% over a lucky 5-month window).
- Most hobby models that *look* profitable are overfit or ignore margin; the published tools that get closest to the bookmaker are still behind it.
- The durable payoffs are the **ML, data-engineering, and evaluation skills** — a genuinely strong portfolio project — plus the (small) chance of a real side income if the execution is sharp and the accounts survive.
- Do not quit anything, do not stake more than expendable money, and never treat a backtest as a promise.

---

## Sources (all fetched 6 Oct 2026)

- Kaunitz, Zhong & Kreiner (2017) — https://arxiv.org/abs/1710.02824
- BeatTheBookie repo README — https://github.com/Lisandro79/BeatTheBookie
- Šourek, Hubáček & Železný (2020) *Beating the market with a bad predictive model* — https://arxiv.org/abs/2010.12508
- Hubáček, Šourek & Železný (2019) *Exploiting sports-betting market using machine learning*, Int. J. Forecasting — https://doi.org/10.1016/j.ijforecast.2019.01.001
- Mangold & Stübinger (2020) CARMA — https://doi.org/10.4995/carma2020.2020.11619
- Berrar, Lopes, Davis & Dubitzky (2018) guest editorial, Machine Learning — https://doi.org/10.1007/s10994-018-5763-8
- 2023 Soccer Prediction Challenge deep-learning paper (RPS table) — https://arxiv.org/abs/2309.14807
- *Forecasting Soccer Matches through Distributions* (2025) — https://arxiv.org/abs/2501.05873
- *Match predictions in soccer: Machine learning vs. Poisson* (2024) — https://arxiv.org/abs/2408.08331
- *A Systematic Review of Machine Learning in Sports Betting* (2024) — https://arxiv.org/abs/2410.21484
- *Machine learning for sports betting: accuracy or calibration?* — https://arxiv.org/abs/2303.06021
- *Optimal sports betting strategies in practice* (2021) — https://arxiv.org/abs/2107.08827
- *Wages of wins: could an amateur make money…* — https://arxiv.org/abs/1702.05982
- *Beating the House: Identifying Inefficiencies…* — https://arxiv.org/abs/1910.08858
- *Forecast Sports Outcomes under Efficient Market Hypothesis* (2026) — https://arxiv.org/abs/2604.17194
- *The Favorite-Longshot Bias in Prediction Markets: Polymarket* (2026) — https://arxiv.org/abs/2609.12878
- *AI World Cup 2026* LLM benchmark (2026) — https://arxiv.org/abs/2608.03416
- *Social Media Sports Betting Influencers* (2026) — https://arxiv.org/abs/2604.08251
- *The profit-bias identity in sports betting* (2026) — https://arxiv.org/abs/2609.06739
- *The reaction to news in live betting* (2021) — https://arxiv.org/abs/2108.00821
- *XGBoost Learning of Dynamic Wager Placement for In-Play Betting* (2024) — https://arxiv.org/abs/2401.06086
- Nagel (2025) *Price Formation Dynamics… Tennis* — https://doi.org/10.2139/ssrn.5043714
- penaltyblog — https://github.com/martineastwood/penaltyblog
- soccerdata — https://github.com/probberechts/soccerdata
- ProphitBet — https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor
- mhaythornthwaite/Football_Prediction_Project — https://github.com/mhaythornthwaite/Football_Prediction_Project
- Reymes/football-match-prediction — https://github.com/Reymes/football-match-prediction
