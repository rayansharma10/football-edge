# football-edge: research report & recommended ML system
*6 Oct 2026. Compiled by Iris from six research notes (`docs/research/`) and Iris's own checks against live data. ✔ = verified by Iris directly against the primary source or live data.*
*Not financial, legal or tax advice.*

---

## 1. The question
Can a student-built ML system find a **genuine, measurable betting edge** in football, starting with paper bets and moving to small real stakes on the Betfair Exchange (AU) only if the edge is proven?

## 2. Executive summary
1. **Don't try to out-predict the closing odds. Measure whether you can beat them in time.**
   - A tuned Dixon-Coles model scored RPS 0.1972 vs the market's 0.1905 over 19 Serie A seasons, lost all 7 test seasons, and got a weight of **exactly 0.000** when pooled with the closing price ([arXiv:2608.11505](https://arxiv.org/abs/2608.11505) ✔ exists). Model class is a second-order lever ([arXiv:2403.07669](https://arxiv.org/abs/2403.07669)).
   - What *is* documented: markets are inefficient **before** they close. The Kaunitz et al. strategy beat bookmakers using the market's own consensus, until they were banned ([arXiv:1710.02824](https://arxiv.org/abs/1710.02824) ✔). Steamers/drifters "move but not enough" (M3 §4).
2. **So the core strategy is "early-line value":** each matchweek, price matches with a calibrated model + market blend; bet early (≈ T-24h) where the price beats our fair probability by a margin; **score every bet by closing-line value (CLV)** against the Betfair close.
3. **The data to test this historically exists and is free** (✔ checked live today): football-data.co.uk records a *pre-closing* snapshot (collected Fri/Tue afternoon UK) **and** the closing price, per bookmaker, for 22 European divisions. That gives ~12 seasons × 22 leagues of "bet early, measure vs close" backtests before risking anything.
4. **Model stack:** v1 Dixon-Coles + pi-rating/Elo baseline → v2 LightGBM on rating/form/xG features, calibrated, then **blended with the de-margined market** via an out-of-sample logarithmic opinion pool → v3 stretch (over/under, in-play, Bayesian). No LLM anywhere in the decision path.
5. **Markets to target:** lower/secondary leagues (Championship, League One, 2. Bundesliga, Eredivisie, etc.) and **over/under 2.5** rather than EPL 1X2, which is priced "to the basis point" (M1 §4). The EPL is kept as the calibration lab.
6. **Gates decide everything:** Gate 0 (leak-proof backtest) → Gate 1 (≥1,000 paper bets, CLV 95% CI > 0) → Gate 2 (manual, tiny live stakes). If the edge never appears, the project still ends with a portfolio-grade, leak-tested forecasting + evaluation system.

**Honest expectation:** most likely outcome is a well-calibrated model that roughly matches the market (a strong engineering result). A real positive-CLV niche is *plausible* in lower leagues / early lines; ROI if found ~1–3% of turnover (B1).

---

## 3. What the evidence says (condensed)
| Finding | Implication | Source |
|---|---|---|
| Bookmaker/market probabilities are the best single forecast (RPS ≈ 0.190–0.202) | Benchmark everything against the de-margined market | M1 §2; arXiv:2608.11505, 2501.05873, 2309.14807 |
| Feature/model choice has "only a minor influence"; CatBoost/LightGBM on **pi-ratings** is best on goals-only data | Don't over-engineer models; invest in data, calibration, evaluation | arXiv:2403.07669, 2309.14807 |
| Selecting on **calibration** beat selecting on accuracy (NBA: +34.7% vs −35.2% ROI) | Optimise log-loss/calibration, never accuracy | arXiv:2303.06021 |
| A model *decorrelated* from the bookmaker can profit even if it's worse overall | Add a decorrelation variant | arXiv:2010.12508; Hubáček 2019 |
| Platt/isotonic can *hurt* strong models; Venn-Abers/Beta calibration did best | Calibrate carefully; compare methods walk-forward | arXiv:2601.19944 |
| Power-method de-margining "universally outperforms" multiplicative; Shin comparable | Power by default, Shin cross-check | M3 §3; Clarke 2017 |
| Lineups added little to score prediction (EPL 2020–22) | Player-level features are v3, not v1 | arXiv:2210.06327 |
| ✔ In-play: market-calibrated Weibull + post-shot xG → **+4.5% ROI, Sharpe 5.94, 17,458 bets** (but only 140 EPL matches) | Promising stretch goal; needs live in-play data | arXiv:2605.16066 |
| No evidence LLMs beat markets | No LLM in the loop | B1; arXiv:2608.03416 ✔ |
| Single-season "findings" are often false positives (~15%) | Require cross-league, cross-season robustness | Winkelmann 2024 (M3) |

---

## 4. Data reality (✔ verified by Iris on 6 Oct 2026)
| Source | What we checked | Result |
|---|---|---|
| football-data.co.uk 2025/26 EPL | Pinnacle closing (`PSCH`) | Filled for **210 / 380** matches, **last on 08/01/2026**, consistent with the site's notice that Pinnacle feeds went stale from 23/07/2025 |
| football-data.co.uk 2026/27 (10 leagues: E1, E2, SC0, D1, D2, I1, SP1, F1, N1, P1) | Betfair Exchange closing (`BFECH`), xG | **BFEC present for ~100% of matches, and new `HxG`/`AxG` columns present for 100%**; no Pinnacle columns at all |
| Championship (E1), 2019/20 → 2025/26 | Coverage per season | Pinnacle closing: full 2019/20–2024/25, half of 2025/26. Betfair Exchange closing: **from 2024/25**. Market-average closing: every season. xG: only from 2026/27 |
| Club Elo API | `/2026-10-01`, `/Fixtures` | **502 error** and **"Fixtures API deactivated"**. Compute our own Elo/pi-ratings (penaltyblog) instead |
| penaltyblog on PyPI | version / wheels | **1.13.1**, Windows wheels for cp310–cp315 |

**Consequences for the design**
- **Historical CLV reference:** Pinnacle closing for ≤ 2024/25 (clean sharp line); **Betfair Exchange closing from 2024/25 on**. The 2024/25 overlap lets us check the two references agree before switching.
- **Historical "bet price":** the pre-closing snapshot (Pinnacle `PSH` pre-2025; `BFEH` from 2024/25; `AvgH`/`MaxH` as soft-book references). Note `Avg*`/`Max*` exclude Pinnacle after 23/07/2025.
- **xG history:** Understat covers top-5 leagues (+RPL) historically; for lower leagues there is no free historical xG before 2026/27, so v2 runs a **goals-only** variant for all leagues and an **xG** variant where available.
- **Live:** Betfair **delayed app key** (free, can place bets, 1–180s delay ✔) for odds snapshots and fixture discovery; football-data.co.uk weekly fixture files as backup. The Odds API free tier (500 credits/mo) is too small for routine polling.

---

## 5. Recommended ML system

### 5.1 Pipeline
```
INGEST (weekly + match-day)
  football-data.co.uk CSVs (22 divisions, 2012/13→now) · Understat xG · Betfair delayed API snapshots
  → validated (pandera) → Parquet + DuckDB, all timestamps UTC, every row tagged with "available_at"
        │
FEATURES (as-of bet time only)
  pi-ratings · Elo (own) · Dixon-Coles attack/defence · time-weighted form · xG form (where available)
  rest days · promoted flag · home/away splits
        │
MODELS
  M0 market:  de-margined odds (power; Shin check)            ← the benchmark
  M1 DC:      Dixon-Coles, time decay ξ≈0.0065 (penaltyblog)  ← baseline (1X2 + O/U 2.5 + scorelines)
  M2 GBM:     LightGBM multiclass on rating/form/xG features  ← main model (CatBoost as comparison)
  CAL:        Venn-Abers / Beta / isotonic, chosen walk-forward by log-loss
  POOL:       log opinion pool of {M1|M2} with M0 (pre-closing price); weight w fitted out-of-sample
        │
DECISION (code only)
  fair prob p* = pooled prob; edge = p* × price − 1 (after 6% commission)
  bet if edge > threshold (chosen in-fold) and league/market is whitelisted
  stake = ¼-Kelly, capped at 1% bankroll/bet, ≤ N bets/day, drawdown kill-switch
        │
LEDGER & EVALUATION
  paper bets (SQLite) → closing snapshot → settlement
  CLV (log, vs margin-free close) · ROI (flat & ¼-Kelly) · RPS · log-loss · calibration slope/ECE · pool weight w
```

### 5.2 Why this shape
- **Pooling with the market** turns "is my model useful?" into a number (*w*). If *w* ≈ 0, the model adds nothing and we stop polishing it (M1 §3, arXiv:2608.11505).
- **Pooling with the *pre-closing* price, scored against the *closing* price**, is exactly the early-line-value hypothesis, and it is testable on 12+ seasons of free data.
- **Calibration over accuracy** and **walk-forward only** are the two most common reasons published betting models fail to replicate (M3 §2).

### 5.3 Staged stack
| Stage | Models | Acceptance test |
|---|---|---|
| **v1 baseline** | M0 market, M1 Dixon-Coles, Elo/pi ordered-logit | De-vig A/E ≈ 1.000; DC within ~0.01 RPS of market; harness leak tests pass |
| **v2 main** | LightGBM (+CatBoost) on ratings/form(+xG), calibrated, pooled | RPS ≤ market + 0.005; **pool weight *w* CI excludes 0** in ≥1 league/market; historical CLV > 0 out-of-sample |
| **v3 stretch** | O/U-specific model; market-calibrated intensity model (in-play); Bayesian hierarchical; steamer/drifter timing; player-level | Positive-CLV paper season on that market |

### 5.4 Tech stack (versions verified in M2)
Python **3.13** via `uv` · `penaltyblog` 1.13.1 · `lightgbm` 4.7 · `catboost` 1.2 · `scikit-learn` 1.9 · `duckdb` 1.5 + Parquet · `pandera` · `betfairlightweight` 2.24 + `flumine` 3.2 · `pytest` · `ruff`. **Skip** MLflow, Selenium/FBref, and any Transfermarkt/Sofascore/FotMob scraping (ToS).

---

## 6. Evaluation protocol: the gates
Full detail: `docs/research/M3-references-evaluation.md` §5.

- **Gate 0, Backtest** (no paper bets until passed): de-vig sanity A/E ≈ 1.000 · walk-forward only · leakage tests incl. regime-reversal · determinism · market benchmark reported per season · **pool weight CI excludes 0** · beats "back the favourite" · 6% commission & bet-time policy modelled · all thresholds chosen in-fold · losing runs reported too.
- **Gate 1, Paper** (no re-tuning mid-run): **≥1,000 settled bets, ≥2 leagues, ≥3 months** · mean log-CLV > 0 with match-clustered bootstrap 95% CI excluding 0 · positive under flat **and** ¼-Kelly · calibration slope 0.9–1.1 · CLV sign unchanged with Shin vs power de-vig · holds per league and per half-season · kill-switch drill done.
- **Gate 2, Live** (manual flip only): Betfair live key (one-off £499 under Betfair's global terms; AU fee unconfirmed) · small separate bankroll · hard caps in code · auto-halt if rolling-250 CLV < 0 or calibration drifts.

---

## 7. Australian practicalities (from B2)
Betfair Exchange explicitly allows bots ✔ · delayed key free ✔ · ~6% commission on net market winnings · corporate bookies restrict winners and have no API · Pinnacle not available in AU · credit-card/crypto deposits banned · hobby winnings generally not taxed (but systematic betting could be argued a business; check if scaling). Paper-only by default, staking caps in code, debit-only funding with a deposit limit.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Lookahead leakage fakes an edge | Mandatory leak tests + regime-reversal test in CI |
| Overfitting thresholds/leagues | In-fold selection; cross-league/season robustness gate |
| Pre-closing snapshot ≠ a price we could actually get (liquidity) | Prefer Betfair `BFE` snapshot where present; cap stakes; live paper uses real Betfair book depth |
| Data source changes (Pinnacle, Club Elo, FBref already broke) | pandera schemas, ingest health checks, Hermes watchdog alert |
| Edge decays / never existed | Gates; "done" still delivers the forecasting + evaluation system |
| Gambling harm | Paper by default; manual live switch; hard caps; BetStop awareness |
