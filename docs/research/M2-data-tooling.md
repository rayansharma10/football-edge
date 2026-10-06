# M2 — Data & Tooling Stack for a Football ML Betting System (Windows 11)

**Prepared for:** Rayan (Sydney, Windows 11 home PC, Python via `uv`, no pip on PATH)
**Date:** 6 October 2026 (AEST, UTC+11)
**Scope:** Verified library + data-source specification for a football (soccer) prediction and paper-betting system, and the ops/storage layout to run it under Hermes cron on Windows.
**Reads after:** `B2-au-betting-practicalities.md` (venues, fees, legality, tax). This note does not repeat that; it goes one level down into versions, wheels, columns, endpoints and repo health.

**Method:** every version/wheel claim below was read from the PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`) on **6 Oct 2026**; repo stats from the GitHub REST API; CSV columns from a live download of a real file. `UNVERIFIED` marks anything I could not confirm against a primary source today.

---

## 0. TL;DR — recommended stack

| Decision | Choice | Why |
|---|---|---|
| **Python** | **3.13** (`uv python install 3.13`) — 3.12 as the fallback | Every package in this document ships a Windows wheel for 3.12 and 3.13; 3.14 also works but has less transitive-dep coverage (see §3) |
| **Model core** | `penaltyblog` (goals/Poisson/Dixon-Coles/Bayesian) + `scikit-learn` calibration + `LightGBM` | penaltyblog gives football-specific models + implied-odds maths out of the box; GBM + calibration covers the feature-based side |
| **Storage** | **Parquet on disk + DuckDB** (SQLite only if you need concurrent writers) | Columnar, single-file, zero-server, works fine on NTFS; DuckDB's one-writer lock is fine for a nightly cron job |
| **Betfair** | `betfairlightweight` + `flumine` (simulation/paper mode) | Both MIT, actively maintained, and `betfairlightweight` has a **built-in historical-download endpoint** (§2.4) |
| **Tracking** | JSON/LP logs + `pandera` schemas; **skip MLflow** initially | A one-model-nightly system does not need a tracking server; add MLflow only if you start sweep-training |
| **Secrets** | `.env` (gitignored) + `python-dotenv` | Betfair creds + app key, Odds API key, API-Football key |
| **Repro** | `pyproject.toml` + `uv.lock` committed | Locked, hash-pinned, no pip needed |

---

## 1. Library table (verified 6 Oct 2026)

"Latest" = `info.version` from PyPI JSON. "Win wheels" = the `win_amd64`/`win32` wheel interpreter tags actually present for that exact version. `py3-none-any`/`py3-none-win_amd64` = one wheel covers all Python 3 versions.

| Package | Latest | Python support | Windows wheel tags | Licence | Last release | Role in this project |
|---|---|---|---|---|---|---|
| **penaltyblog** | **1.13.1** | `>=3.10` | `cp310,cp311,cp312,cp313,cp314,cp315` (win_amd64 **and** win32) | **MIT** | 2026-10-05 | Core models: Poisson, Bivariate Poisson, **Dixon-Coles**, Bayesian + Hierarchical Bayesian goal models; Elo/Massey/Colley/Pi ratings; implied-odds (overround removal); backtest module; scrapers for FBref/Understat/football-data.co.uk/Football Charts/Club Elo; MatchFlow for StatsBomb/Opta JSON ([PyPI](https://pypi.org/project/penaltyblog/), [docs](https://penaltyblog.readthedocs.io/en/latest/models/index.html), [scrapers](https://penaltyblog.readthedocs.io/en/latest/scrapers/index.html)) |
| **soccerdata** | **1.9.1** | `>=3.10,<3.15` | pure-Python `py3-none-any` (no compiled ext) | **Apache-2.0** per PyPI classifier — **but** GitHub API reports `NOASSERTION`/"Other" ([repo](https://github.com/probberechts/soccerdata)) → treat licence as **UNVERIFIED**, read LICENSE before redistribution | 2026-07-24 | Scraper zoo: **ClubElo, ESPN, FBref, FiveThirtyEight, Football-Data.co.uk, Sofascore, SoFIFA, Understat, WhoScored**; uniform DataFrames + local cache ([docs](https://soccerdata.readthedocs.io/en/latest/)) |
| **LightGBM** | **4.7.0** | `>=3.10` | `py3-none-win_amd64` (version-agnostic) | MIT | 2026-07-18 | Primary GBM; fast on CPU, no GPU juggling |
| **CatBoost** | **1.2.10** | not pinned | `cp38…cp314` win_amd64 | Apache-2.0 | 2026-02-18 | Alternative GBM; strong categorical handling, but slower release cadence (Feb 2026) |
| **XGBoost** | **3.4.1** | **`>=3.12`** | `py3-none-win_amd64` | Apache-2.0 | 2026-08-15 | Optional third GBM. **Note: drops Python ≤3.11** |
| **scikit-learn** | **1.9.1** | `>=3.11` | `cp311…cp315` win_amd64 (+`cp314t` free-threaded) | BSD-3-Clause | 2026-09-10 | `CalibratedClassifierCV` (isotonic/Platt), metrics, CV splitters. This is where probability calibration lives |
| **PyMC** | **6.3.2** | `>=3.12` | pure `py3-none-any`, but needs **pytensor 3.3.3** (`cp312,cp313,cp314` win wheels) | Apache-2.0 | 2026-09-08 | Optional: custom hierarchical models. **Also see §3 on Windows toolchain** |
| **betfairlightweight** | **2.24.0** | `>=3.9` | pure `py3-none-any` | **MIT** | 2026-09-04 | Betfair API-NG wrapper: navigation, `listMarketCatalogue`, `listMarketBook`, order placement, streaming, **plus the historic-data download endpoint** (§2.4). 515★, 1 open issue, pushed 2026-09-04 ([repo](https://github.com/betcode-org/betfair)) |
| **flumine** | **3.2.6** | `>=3.10` | pure `py3-none-any` | **MIT** | 2026-10-01 | Event-based betting/trading framework on top of betfairlightweight: strategies, risk, **simulation (historical replay)** and **paper trading**. README states "Tested on Python 3.10, 3.11, 3.12, 3.13 and 3.14" ([repo](https://github.com/betcode-org/flumine)) |
| **polars** | **1.44.2** | `>=3.10` | pure `py3-none-any` | MIT | 2026-09-09 | Fast tabular transforms; optional if you lean on pandas |
| **duckdb** | **1.5.6** | `>=3.10` | `cp310…cp314` win_amd64 (+win_arm64) | MIT | 2026-09-28 | Analytical engine over Parquet; primary query layer |
| **mlflow** | **3.16.1** | `>=3.10` | pure `py3-none-any` | Apache-2.0 | 2026-09-16 | **Defer.** Heavy for a nightly single-model run; add for sweeps |
| **pandera** | **0.34.0** | `>=3.10` | pure `py3-none-any` | MIT | 2026-10-05 | DataFrame schema validation at every ingest boundary — the cheapest defence against silent column drift |
| **numpy** | **2.5.3** | `>=3.12` | `cp312…cp315` win_amd64 | BSD-3 | 2026-09-06 | transitive |
| **pandas** | **3.0.6** | `>=3.11` | `cp311…cp315` win_amd64 | BSD-3 | 2026-09-17 | transitive (soccerdata/many APIs return DataFrames). **pandas 3.x is a major version — check API breaks** |
| **scipy** | **1.18.1** | `>=3.12` | `cp312…cp315` win_amd64 | BSD-3 | 2026-08-21 | transitive (Poisson/optimisation) |
| **pyarrow** | **25.0.1** | `>=3.10` | `cp310…cp314` win_amd64 | Apache-2.0 | 2026-08-10 | Parquet I/O |
| **statsmodels** | **0.15.0** | `>=3.10` | `cp310…cp315` win_amd64 | BSD-3 | 2026-08-27 | Classic Poission GLM / diagnostics if you want them |
| **matplotlib** | **3.11.2** | `>=3.11` | `cp311…cp315` win_amd64 | PSF-based | 2026-09-11 | calibration/EV plots |
| **orjson** | **3.12.0** | `>=3.10` | `cp310…cp315` win_amd64 | Apache-2.0/MIT | 2026-08-14 | speed extra for betfairlightweight (`pip install betfairlightweight[speed]`) |
| **selenium** | **4.50.0** | `>=3.10` | pure `py3-none-any` | Apache-2.0 | 2026-09-30 | needed by soccerdata's **FBref** and **WhoScored** readers (Chrome required) |
| **pytest** | **9.1.1** | `>=3.10` | pure | MIT | 2026-06-19 | tests |
| **python-dotenv** | **1.2.4** | `>=3.10` | pure | BSD-3 | 2026-10-01 | secrets loading |
| **APScheduler** | **3.11.3** | `>=3.8` | pure | MIT | 2026-06-28 | optional in-process scheduling; prefer Hermes cron (§4.2) |
| **cloudscraper** | **1.2.71** | *(none declared)* | pure | MIT | 2023-04-25 | **Stale (last release 2023)** — only if you decide to scrape and hit a WAF; do not make it load-bearing |

### Notes / judgement calls
- **penaltyblog is the single best-value dependency here.** It ships the exact models this project needs (goals-based + Bayesian), all the implied-odds margin maths, a backtest module and its own scrapers — MIT, one maintainer, pushed the day before this note.
- **soccerdata's licence is genuinely ambiguous**: PyPI's classifier says Apache-2.0, GitHub says "Other/NOASSERTION". Don't redistribute derived data without reading the repo LICENSE.
- **XGBoost ≥3.4 requires Python ≥3.12.** If you ever pin 3.11 you lose it. Another reason for 3.13.
- **pandas 3.0.x**: major-version bump landed recently; if you hit breakage, pin `pandas>=2.2,<3`.
- `country_converter` / other scrapers are **not** needed: `soccerdata` already wraps all public sources.

---

## 2. Data sources

### 2.1 Summary table

| Source | What you get | Granularity | Cost / limits | Access method | ToS / licensing notes |
|---|---|---|---|---|---|
| **football-data.co.uk** ([site](https://www.football-data.co.uk/), [notes.txt](https://www.football-data.co.uk/notes.txt), [data.php](https://www.football-data.co.uk/data.php)) | FT/HT results, match stats (shots, SoT, corners, fouls, cards, refs), **pre-closing AND closing 1X2 odds**, over/under 2.5, Asian handicaps; fixtures+odds for upcoming games | One CSV per division per season; **fixtures file refreshed Fri afternoon (weekend) and Tue afternoon (midweek), UK time** | **Free**, no key. Site returns `X-WS-RateLimit-Limit: 1000` header (observed) — be gentle | HTTP GET of CSV; `www.` → bare-domain **302 redirect**, so follow redirects (`curl -L`) | Free for personal use; site describes itself as odds collated from comparison sites. **Not licensed for redistribution** — UNVERIFIED beyond the site's own notes |
| **Understat** ([understat.com](https://understat.com/)) | **xG / xA** per team and player, shot-level xG | Match and season aggregates | Free, no key | **No official API** — HTML scrape (embedded JSON). `soccerdata.Understat` and `penaltyblog` both wrap it | Scraping-adjacent; no published licence. Treat as **UNVERIFIED** for redistribution |
| **Club Elo** ([clubelo.com/API](http://clubelo.com/API)) | Elo ratings per club, daily rankings back to 1939 (pre-1960 provisional), per-club history | Daily CSV, one row per club | Free | `http://api.clubelo.com/YYYY-MM-DD`, `.../CLUBNAME` | Free to use; check terms for redistribution |
| **The Odds API** ([the-odds-api.com](https://the-odds-api.com/), [v4 docs](https://the-odds-api.com/liveapi/guides/v4/), [bookmakers](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)) | Current + **historical** odds: h2h, spreads (handicap), totals (O/U), outrights; plus scores | Live poll; historical snapshots **every 10 min since 6 Jun 2020** | **Free = 500 credits/month**; paid $30/20k … $249/15M. **Cost formula verified**: `/odds` = `markets × regions` credits (e.g. 3 markets × 3 regions = 9); **historical = 10 × markets × regions**; `/sports` and `/events` are free; calls returning empty data are not charged | REST + API key; `regions=au,uk,eu` | Commercial API; free tier includes historical odds |
| **API-Football / API-Sports** ([pricing](https://www.api-football.com/pricing), [docs](https://www.api-football.com/documentation-v3)) | Fixtures, livescore, standings, **lineups**, events, statistics, **pre-match & in-play odds**, injuries/sidelined, predictions, H2H | Per-request; lineups typically ~1h before KO | **Free = 100 req/day, no card**; Pro $19 = 7,500/day; Ultra $29 = 75k; Mega $39 = 150k. Free tier limited in available seasons | REST + API key (`x-apisports-key`) | Commercial API. 100 req/day is ~4 fixtures' worth of full detail — budget carefully |
| **football-data.org** ([coverage](https://www.football-data.org/coverage)) | Fixtures, results, tables, squads, **lineups/subs**, live scores | Per-request | Free tier "forever" on top competitions; paid above | REST + API key | Commercial API; free tier covers EPL/La Liga/Bundesliga/Serie A/Ligue 1/CL etc. |
| **Betfair Historical Data** ([historicdata.betfair.com](https://historicdata.betfair.com/), [Automation Hub](https://betfair-datascientists.github.io/modelling/dataSources/)) | Time-stamped **Exchange prices + volume**, BSP, settlements, as JSON in TARs since 2016, incl. **soccer** | **Basic (free): 1-minute odds intervals, no volume.** Advanced (paid): 1s + volume. Pro (paid): 50 ms + volume | Basic = free; Advanced/Pro = paid. **AU/NZ customers must email `automation@betfair.com.au` before buying** | Official download API; **`betfairlightweight` implements it natively** (§2.4) | Betfair data licence — personal use |
| **Betfair free AU/NZ CSVs** ([same page](https://betfair-datascientists.github.io/modelling/dataSources/)) | Market snapshots by runner (max/min/BSP, overrounds) | Monthly blocks from **Jan 2023**; AU sports leagues from **2020** | Free, no login | CSV download | **Football coverage = A-League only**; European leagues need the paid Stream data |
| **openfootball** ([GitHub](https://github.com/openfootball)) | Results + fixtures as JSON/text, internationals 1872–2026 | Static dumps | Free | Git clone / raw URLs | **CC0-1.0** — the cleanest licence in the set |
| **StatsBomb open data** ([hudl/open-data](https://github.com/hudl/open-data)) | Event-level (passes, shots, pressures) | Selected competitions only | Free | GitHub / StatsBomb API (penaltyblog's MatchFlow can stream it) | Custom/NOASSERTION — read before commercial use |
| **FBref** ([fbref.com](https://fbref.com/en/)) | Team/player season + match stats **including xG** | 100+ competitions | Free (Stathead paid) | `soccerdata.FBref` (needs Chrome/Selenium) or `penaltyblog` | Data churned hard in Jan 2026 (provider pulled advanced data, demanded deletion, then xG reappeared per Sports Reference's own note). **Verify coverage each run — treat as volatile, UNVERIFIED for exact current scope** |
| **Sofascore / FotMob** | Live scores, lineups, stats, some xG; player ratings | Very broad | Free to browse | **No official public API** — scrape only | **Scraping ToS risk: flag, don't build on it.** Specific anti-scraping clauses UNVERIFIED (JS-rendered terms) |
| **Transfermarkt** | Player market values, transfers | Squad/player pages | Free to browse | **No official API** — scrape only | **Explicitly restricts automated access in its terms; commercial scraping is litigated.** Treat as off-limits for anything load-bearing |

### 2.2 football-data.co.uk — exact columns (VERIFIED from a live download)

I downloaded `https://www.football-data.co.uk/mmz4281/2425/E0.csv` (Premier League 2024/25) on 6 Oct 2026: **197,110 bytes, 120 columns**. Follow the `www.` → bare-domain redirect (`curl -L`) or you get an empty body.

Verified column order (numbered) — the ones that matter for modelling:

- **Identity/result:** `Div, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee`
- **Match stats:** `HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR`
- **Pre-closing 1X2 odds, cols 25–48:** `B365H/D/A, BWH/D/A, BFH/D/A, **PSH/PSD/PSA**, WHH/D/A, 1XBH/D/A, MaxH/D/A, AvgH/D/A, **BFEH/BFED/BFEA**`
- **Pre-closing totals/AH, cols 52–72:** `B365>2.5, B365<2.5, P>2.5, P<2.5, Max>2.5/…, Avg>2.5/…, BFE>2.5/…, AHh, B365AHH/AHA, PAHH/PAHA, MaxAHH/AHA, AvgAHH/AHA, BFEAHH/AHA`
- **Closing 1X2 odds, cols 73–99:** `B365CH/D/A, BWCH/D/A, BFCH/D/A, **PSCH/PSCD/PSCA**, WHCH/D/A, 1XBCH/D/A, MaxCH/D/A, AvgCH/D/A, **BFECH/BFECD/BFECA**`
- **Closing totals/AH, cols 100–120:** `B365C>2.5, B365C<2.5, PC>2.5, PC<2.5, MaxC>2.5/…, AvgC>2.5/…, BFEC>2.5/…, AHCh, B365CAHH/AHA, PCAHH/PCAA, MaxCAHH/AHA, AvgCAHH/AHA, BFECAHH/AHA`

Rules from [notes.txt](https://www.football-data.co.uk/notes.txt) and [data.php](https://www.football-data.co.uk/data.php):
- Closing odds = same abbreviation **plus a `C`** (e.g. `B365CH`). Pre-closing odds are collected "after market opening"; closing odds were added from **2019/20**; earlier seasons have pre-closing only.
- **`BFE*` = Betfair Exchange**; `BF*` = Betfair Sportsbook. Use `BFE*`/`BFEC*` for exchange prices.
- **Pinnacle closing 1X2 (`PSC*`) exists back to 2012/13** — but read this:

> **⚠️ Pinnacle degradation (load-bearing).** football-data.co.uk states: *"Since 23/07/2025 Pinnacle's public API for odds delivery has become unreliable meaning their odds are systematically out of date relative to odds for other bookmakers, including both the pre-closing and closing odds. Consequently they should be used with caution… and are no longer being included for the calculation of market average and maximum odds."* ([data.php](https://www.football-data.co.uk/data.php))
> **Consequence:** do **not** use `PSH/PSD/PSA`/`PSCH/PSCD/PSCA` as a clean "sharp closing line" benchmark for anything from mid-2025 onward, and be aware that `AvgC*/MaxC*` no longer include Pinnacle in that window. This invalidates the most common naive "beat the Pinnacle close" backtest design on recent data.

- Coverage: **up to 22 European divisions**, 25 seasons back to **1993/94**; match stats since **2000/01** (all 22 divisions since 2017/18); odds from **up to 10 online bookmakers** back to 2000/01; plus **16 worldwide premier divisions** with FT results + **closing** odds (best/average market price and Pinnacle) back to **2012/13** ([data.php](https://www.football-data.co.uk/data.php)).

### 2.3 Club Elo API — currently broken (VERIFIED 6 Oct 2026)

Documented endpoints ([clubelo.com/API](http://clubelo.com/API)): `api.clubelo.com/YYYY-MM-DD` (full daily ranking), `api.clubelo.com/CLUBNAME` (club history), `api.clubelo.com/Fixtures` (upcoming-match probabilities).

**Live checks today returned failures:**
- `GET http://api.clubelo.com/2026-09-01` → **HTTP 502 Bad Gateway** (repeated 3×)
- `GET http://api.clubelo.com/` → **HTTP 403 Forbidden** ("Access is denied", Microsoft-IIS)
- `GET http://api.clubelo.com/Fixtures` → HTTP 200 but body is literally **"Fixtures API deactivated"**
- `GET http://api.clubelo.com/Arsenal` → empty body
- `https://api.clubelo.com/...` → connection failure

**Consequence:** `soccerdata.ClubElo` and `penaltyblog`'s Club Elo scraper will fail today. **Do not put Club Elo on the critical path.** Snapshot its data when it works, cache it in Parquet, and treat it as a bonus feature source. (Status is UNVERIFIED as to whether this is a permanent shutdown or an outage — retest before relying on it.)

### 2.4 Betfair API + historical data

**Delayed vs Live app key** (from [Automation Hub — app key guide](https://betfair-datascientists.github.io/api/apiappkey/), consistent with B2 §1.4):
- **Delayed ("1.0-Delay") key:** free, variable **1–180 s** delay, **no matched volume**, **has betting functionality**; you must place ≥1 bet through it before live activation is considered.
- **Live ("1.0") key:** needs activation via an AU webform; if you pull *data* without placing corresponding bets, Betfair may **auto-apply a delay** to the live key (stated explicitly on the Automation Hub page).

**API rate limits:** UNVERIFIED. The official page has moved and I could not read it today ([docs.developer.betfair.com](https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/Exchange+API) returns a JS-rendered shell; the old support article 404s). Commonly cited figures (5 requests/sec per app key; a separate cap on concurrent Stream-API connections) are **UNVERIFIED** — confirm in the developer docs before designing a polling loop. **Design defensively regardless:** poll only what you need, reuse the `requests.Session`, and back off on `TOO_MANY_REQUESTS`.

**Market types for football** (standard Betfair `marketType` values): `MATCH_ODDS` (1X2), `OVER_UNDER_25` / `OVER_UNDER_15`… (totals), `ASIAN_HANDICAP`. Event-type id for Soccer is `1`. `betfairlightweight`'s streaming filter uses `event_type_ids=["1"]` ([streaming docs](https://github.com/betcode-org/betfair/blob/master/docs/streaming.md)) — the market-type strings themselves are **UNVERIFIED against a primary Betfair doc today**, verify from the live `listMarketCatalogue` response on first run rather than hard-coding assumptions.

**Historical data via `betfairlightweight` — VERIFIED from source.** `betfairlightweight/endpoints/historic.py` implements a `Historic` endpoint class against `https://historicdata.betfair.com/api/` with:
- `get_my_data()` — what your account has purchased
- `get_collection_options(sport, plan, from_day/month/year, to_..., event_id, market_types_collection, countries_collection, file_type_collection)`
- `get_data_size(...)` — file count + combined size, before you commit
- `get_file_list(...)` — the list of downloadable files
- `download_file(file_path, store_directory)` — streams the TAR to disk in 1 KB chunks

`plan` accepts the literal strings **`Basic Plan` / `Advanced Plan` / `Pro Plan`** — i.e. the free tier is drivable from Python with no manual downloads. Source: [`betfairlightweight/endpoints/historic.py`](https://github.com/betcode-org/betfair/blob/master/betfairlightweight/endpoints/historic.py).

**`flumine` simulation limitations** (from [Flumine Simulations](https://betfair-datascientists.github.io/tutorials/flumineSimulations/)):
- **Market Catalogue is NOT in the historic stream files** → no runner metadata (e.g. team names) from the stream; you must join your own metadata.
- **Cross-matching / virtual bets are NOT included.**
- No modelling of **market impact** — your simulated bets don't move the market, so lightly-traded markets overstate fillability.
- The Automation Hub's human-facing soccer tutorial notes the **simulation steps require Betfair PRO (paid) data**; the modelling/tutorial steps themselves are free ([How to Build a Soccer Bot Part 1](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartI/)).

### 2.5 Lineups, injuries, ratings — and the ToS line you shouldn't cross

| Need | Best free option | Timing | Notes |
|---|---|---|---|
| Confirmed starting XIs | **API-Football `/fixtures/lineups`** (free tier, 100 req/day) or **football-data.org** lineups | Typically **~1 h before KO** (per-league variance is UNVERIFIED) | A pre-match model that consumes lineups must run inside a ~60-min window. That's a *scheduling* constraint, not a data constraint |
| Injuries/suspensions | API-Football `sidelined` + `injuries` endpoints; `soccerdata.WhoScored.read_missing_players()` | Pre-match | WhoScored reader needs Chrome/Selenium and is scrape-fragile |
| xG | Understat (top-6 leagues), FBref (volatile) | Post-match / season | Team & player level |
| Player ratings | **FotMob / Sofascore** ratings exist but there is **no official public API** | Live | ⚠️ **Scraping risk.** Both are ToS-restricted; rate-limit/blocking likely. Do not make ratings a dependency of a bet-sizing path |
| Player market values | **Transfermarkt** | N/A | ⚠️ **Highest ToS risk in this list.** Transfermarkt's terms restrict automated access and it has pursued scrapers. **Recommendation: don't.** If you need player quality, derive it from xG/xA you already have, or use openfootball/StatsBomb event data |

**Rule of thumb:** licensed APIs (football-data.co.uk, The Odds API, API-Football, football-data.org, Betfair) can be load-bearing; scraped sites (Understat, FBref, Sofascore, FotMob, Transfermarkt) should only ever be *soft* features that degrade gracefully to NaN.

---

## 3. Recommended Python version

**Use Python 3.13.** Install with `uv python install 3.13`.

Evidence (all wheels verified on PyPI today):

| Dependency | 3.12 | 3.13 | 3.14 |
|---|---|---|---|
| penaltyblog 1.13.1 | ✅ win | ✅ win | ✅ win |
| soccerdata 1.9.1 (`<3.15`) | ✅ | ✅ | ✅ |
| DuckDB 1.5.6 | ✅ | ✅ | ✅ |
| scikit-learn 1.9.1 | ✅ | ✅ | ✅ |
| CatBoost 1.2.10 | ✅ | ✅ | ✅ |
| pytensor 3.3.3 / PyMC 6.3.2 | ✅ | ✅ | ✅ |
| orjson 3.12.0 (betfairlightweight[speed]) | ✅ | ✅ | ✅ |
| numpy 2.5.3 / scipy 1.18.1 (`>=3.12`) | ✅ | ✅ | ✅ |
| pandas 3.0.6 (`>=3.11`) | ✅ | ✅ | ✅ |
| **XGBoost 3.4.1 (`>=3.12`)** | ✅ | ✅ | ✅ |
| LightGBM 4.7.0 | ✅ | ✅ | ✅ |

**Why 3.13 and not 3.14** (which also technically works): 3.14 is only ~1 year old and the long tail of transitive dependencies you will inevitably add (mlflow's plugin tree, Selenium/browser tooling, lesser-maintained scrapers) still has thinner 3.14 coverage. 3.13 is the widest-supported version in this specific stack. Choose **3.12** only if a dependency of a dependency pins low; avoid **≤3.11** because it loses XGBoost ≥3.4, scikit-learn ≥1.9, pytensor and numpy 2.5.

**Windows-specific cautions**
- `uv venv --python 3.13` then `uv sync` — never mix a `uv` venv with a stray system Python.
- PyMC/pytensor are the only compiled-and-fragile links. pytensor ships its own C backend and needs a working compiler **only** if you use `pytensor`'s C compilation; the wheels work out of the box otherwise. Keep PyMC **optional** (an extra in `pyproject.toml`) so a pytensor build failure can't break the ingest/backtest pipeline.
- `soccerdata.FBref` / `WhoScored` require a **Chrome** binary and Selenium 4.50; `headless=False` can help avoid blocks. Treat these readers as best-effort.
- The earlier note's `F2-westpac-stgeorge.md` is unrelated; ignore.

---

## 4. Storage & ops layout on Windows

### 4.1 Storage model

**Parquet-on-disk as the source of truth + DuckDB as the query engine.** Both verified to have Windows wheels for 3.12–3.14 (pyarrow 25.0.1, duckdb 1.5.6).

```
data/
  raw/            # immutable, as-downloaded (never edited)
    football_data_co_uk/<Season>/<Div>.csv
    understat/<league>/<season>.json
    betfair_historic/<plan>/<YYYY>/<MM>/<event_id>.tar
    odds_api/<utc_date>/<sport_key>.json
  interim/        # parsed, typed, not yet modelled
  processed/      # model-ready feature tables (Parquet, partitioned by league/season)
  football.duckdb # views over raw+processed; NOT the source of truth
  bets/           # append-only paper-bet ledger: JSONL + a Parquet mirror
```

- **Raw is immutable.** Every ingest writes a new file; never overwrite. This gives you free reproducibility and lets you re-derive features when you change your mind.
- **DuckDB caveats on Windows:** a single process holds the write lock on the `.duckdb` file. Serialize your cron jobs (one writer at a time), or write Parquet and let DuckDB read-only. Avoid putting the DB on OneDrive/network drives — file locking on synced folders corrupts it.
- **SQLite** only if you need multiple concurrent writers (e.g. the bot writing bets while a dashboard reads). Otherwise DuckDB is faster for analytics.
- Partition Parquet by `league=…/season=…` — small files, fast scans, trivial incremental ingest.

### 4.2 Scheduling under Hermes cron (Windows)

- Each job = a standalone script invoked as **`uv run python scripts/<name>.py`** with `workdir` set to the project root, so the venv and `uv.lock` are always honoured.
- Suggested jobs (all in **UTC** — see §4.3):
  | Job | Cadence (UTC) | Script |
  |---|---|---|
  | Ingest results + odds | daily, ~12:00 (after FD's UK-morning updates) | `scripts/daily_ingest.py` |
  | Refresh fixtures/odds (Odds API) | daily, cheap markets only | `scripts/refresh_odds.py` |
  | Lineup pull | every 15 min in the pre-KO window | `scripts/lineups.py` |
  | Betfair paper batch (delayed key) | per fixture, T−10 min | `scripts/betfair_poll.py` |
  | Retrain + calibrate | weekly (Tue) | `scripts/nightly_train.py` |
  | Settle + evaluate | daily, ~06:00 | `scripts/settle.py` |
- **Guardrails as code:** a hard per-bet and per-day stake cap, a kill-switch file (`data/HALT`) that every script checks first, and `LIVE=false` by default (carry over B2 §6).
- **Budget the Odds API credits**: `/odds?markets=h2h,totals&regions=au,uk` = **4 credits/call**. Polling that 6×/day for 30 days = 720 credits — **over the free 500/month**. Either narrow to `markets=h2h&regions=au` (1 credit) or move live odds polling to the Betfair delayed key, which is free. Historical snapshots cost **10×**, so pull them deliberately, not in a loop.

### 4.3 Timezones

- **Store everything UTC.** Every ingest timestamp is `datetime.now(tz=timezone.utc)`; every DB/Parquet column is UTC. Kick-off times from API-Football/Odds API are already ISO-8601 UTC (`commence_time` ends in `Z`) — keep them that way.
- **football-data.co.uk is the trap.** `Date` is `dd/mm/yy` and `Time` is **UK local time**, and it is *not* timezone-tagged. Parse with an explicit `Europe/London` conversion, then convert to UTC — otherwise half your 2020s fixtures shift by an hour twice a year.
- **Display in `Australia/Sydney`** (AEDT, UTC+11) via `zoneinfo.ZoneInfo("Australia/Sydney")` — `zoneinfo` is stdlib from 3.9 and works on Windows (it does *not* need `tzdata` on Windows since the OS tz database isn't used; if you ever containerise, add `tzdata`).
- Never do naive datetime arithmetic across a DST boundary. When in doubt: convert to UTC, compute, convert back at the edge.

### 4.4 Secrets, testing, reproducibility

- **Secrets:** `.env` at repo root, **gitignored**; `.env.example` committed with empty keys. Load with `python-dotenv` (`load_dotenv()` at process start). Keys needed: `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, `BETFAIR_APP_KEY`, `BETFAIR_CERTS_DIR` (if used), `ODDS_API_KEY`, `API_FOOTBALL_KEY`. Never log the key values; log only the last 4 chars for debugging.
- **Testing:** `pytest` (9.1.1). The tests that actually pay for themselves:
  1. **Schema tests** — `pandera` DataFrameSchemas asserting every football-data.co.uk column exists with the right dtype (this is what catches FD silently changing a column).
  2. **Leakage tests** — assert that for any training row, every feature column's source timestamp is strictly `< kickoff`.
  3. **Determinism test** — same seed + same input Parquet ⇒ byte-identical model predictions.
  4. **A golden-file test** on a tiny hand-checked fixture (3 matches) end-to-end: ingest → features → predict → stake → settle.
- **Reproducibility:** commit `pyproject.toml`, `uv.lock` and `.python-version`; run everything through `uv run`. Log a `run_id` (UTC timestamp + git SHA) into every output Parquet and every bet record so any number can be traced back.

---

## 5. Sample project layout

```
C:/Users/Rayan/hermes-research/projects/football-bot/
├── pyproject.toml
├── uv.lock
├── .python-version              # 3.13
├── .env                         # gitignored — secrets
├── .env.example
├── .gitignore                   # .env, data/raw/, data/*.duckdb, .venv/
├── README.md
├── data/
│   ├── raw/                     # immutable source drops
│   │   ├── football_data_co_uk/<Season>/<Div>.csv
│   │   ├── understat/<league>/<season>.json
│   │   ├── odds_api/<utc_date>/<sport_key>.json
│   │   └── betfair_historic/<plan>/<YYYY>/<MM>/<event>.tar
│   ├── interim/
│   ├── processed/
│   │   └── features/league=<L>/season=<S>/*.parquet
│   ├── bets/
│   │   ├── paper_bets.jsonl     # append-only ledger
│   │   └── paper_bets.parquet
│   ├── football.duckdb          # views over raw+processed
│   └── HALT                     # presence = kill switch; every script checks it
├── src/football_bot/
│   ├── __init__.py
│   ├── config.py                # paths, TZ constants, .env loading, caps
│   ├── log.py                   # structured UTC-stamped logging
│   ├── ingest/
│   │   ├── football_data.py     # CSV download (follow redirects) + parse
│   │   ├── understat.py
│   │   ├── clubelo.py           # soft-failing; caches; no critical path
│   │   ├── odds_api.py          # credit-aware wrapper
│   │   ├── api_football.py      # fixtures, lineups, injuries
│   │   └── betfair_historic.py  # betfairlightweight Historic endpoint
│   ├── transform/
│   │   ├── schemas.py           # pandera DataFrameSchemas (one per source)
│   │   ├── canonical.py         # team-name/canonical-id mapping
│   │   ├── features.py          # rolling form, Elo, xG, rest days, market-implied
│   │   └── tz.py                # dd/mm/yy + Europe/London -> UTC
│   ├── models/
│   │   ├── dixon_coles.py       # penaltyblog DixonColes (+Bayesian variant)
│   │   ├── gbm.py               # LightGBM over engineered features
│   │   ├── calibrate.py         # sklearn CalibratedClassifierCV / isotonic
│   │   └── registry.py          # save/load artifacts + run_id
│   ├── backtest/
│   │   ├── engine.py            # walk-forward, no-leakage splits
│   │   ├── metrics.py           # log-loss, Brier, RPS, closing-line value
│   │   └── flumine_runner.py    # flumine simulation over historic TARs
│   ├── betting/
│   │   ├── staking.py           # fractional Kelly + hard caps (immutable constants)
│   │   ├── paper_broker.py      # records bets, applies commission, settles
│   │   └── flumine_strategy.py  # live/paper flumine strategy (delayed key)
│   └── report/
│       └── plots.py
├── scripts/
│   ├── daily_ingest.py
│   ├── refresh_odds.py
│   ├── lineups.py
│   ├── betfair_poll.py
│   ├── nightly_train.py
│   └── settle.py
├── notebooks/
│   └── 01_eda_football_data.ipynb
├── tests/
│   ├── test_schemas.py
│   ├── test_no_leakage.py
│   ├── test_determinism.py
│   └── test_end_to_end_small.py
├── artifacts/
│   ├── models/<run_id>/
│   └── runs/<run_id>/metrics.json
└── docs/
    └── sources.md               # link back to this note + B2
```

Bootstrap (nothing installed yet — this is the command set for when you're ready):

```bash
cd C:/Users/Rayan/hermes-research/projects/football-bot
uv python install 3.13
uv init --python 3.13
uv add penaltyblog soccerdata scikit-learn lightgbm pandera \
       betfairlightweight flumine polars duckdb pyarrow python-dotenv requests
uv add --optional bayes pymc          # keep PyMC optional (pytensor build risk)
uv add --dev pytest ruff
uv sync
uv run python scripts/daily_ingest.py --dry-run
```

---

## 6. Risks, gaps and things I could not verify

| # | Item | Status | Action |
|---|---|---|---|
| 1 | **Club Elo API is down / partly deactivated** — 502 on date endpoints, 403 on root, `Fixtures` returns "Fixtures API deactivated" (checked 6 Oct 2026, repeated) | **VERIFIED broken today** | Don't depend on it. Cache snapshots when it works; retest before each season |
| 2 | **Pinnacle odds unreliable since 23/07/2025**; excluded from FD's `Avg*`/`Max*` | **VERIFIED** (football-data.co.uk data.php) | Don't use post-mid-2025 Pinnacle columns as a sharp-line benchmark |
| 3 | Betfair API **rate limits** | **UNVERIFIED** (official page JS-walled; old article 404) | Read developer docs; design with backoff + minimal polling regardless |
| 4 | Exact current **FBref** advanced-stat scope after the Jan 2026 removal/re-add | **UNVERIFIED** (volatile) | Detect per-run: assert xG columns present, else degrade |
| 5 | **soccerdata licence** — PyPI says Apache-2.0, GitHub says NOASSERTION | **Ambiguous** | Read LICENSE before redistributing derived data |
| 6 | Sofascore / FotMob / **Transfermarkt** scraping terms | **UNVERIFIED** (JS terms pages) | Treat as off-limits for load-bearing features; Transfermarkt explicitly discouraged |
| 7 | Lineup release timing per competition (~1 h before KO) | **UNVERIFIED** for exact per-league windows | Measure it empirically for your leagues; schedule polling, don't assume |
| 8 | Betfair `marketType` strings for football (`MATCH_ODDS`, `OVER_UNDER_25`) | Well-known but **UNVERIFIED today** | Confirm from a live `listMarketCatalogue` response on first run |
| 9 | Betfair AU live-key activation fee for AU customers | Refer to B2 §1.4 (still UNVERIFIED) | Email `automation@betfair.com.au` |
| 10 | `cloudscraper` last released 2023, `request` API undocumented | **Stale** | Avoid; don't build scraping on it |

---

## 7. Verification log (what I actually checked, 6 Oct 2026)

- **PyPI JSON API** for: penaltyblog, soccerdata, lightgbm, catboost, xgboost, scikit-learn, pymc, pytensor, betfairlightweight, flumine, polars, duckdb, mlflow, pandera, numpy, pandas, scipy, pyarrow, statsmodels, matplotlib, orjson, selenium, cloudscraper, pytest, python-dotenv, APScheduler, joblib → version, `requires_python`, licence classifier, release date, and **enumerated Windows wheel interpreter tags**.
- **GitHub REST API** for `betcode-org/betfair` (515★, MIT, pushed 2026-09-04, 1 open issue), `betcode-org/flumine` (242★, MIT, pushed 2026-10-01), `martineastwood/penaltyblog` (230★, MIT, pushed 2026-10-05, 1 open issue), `probberechts/soccerdata` (2,099★, licence NOASSERTION, pushed 2026-09-28, 38 open issues).
- **Live HTTP GET** of `https://www.football-data.co.uk/mmz4281/2425/E0.csv` (302 → follow, 197,110 bytes, 120 columns enumerated) and `notes.txt` (column key), `data.php` (coverage + Pinnacle degradation notice), `matches.php` (fixtures cadence: Fridays ≤17:00 UK for weekends, Tuesdays ≤13:00 UK for midweek).
- **Live HTTP checks** of all three Club Elo endpoints (failures as tabulated in §2.3).
- **`betfairlightweight` source** read directly (`endpoints/historic.py`, `enums.py`, docs `streaming.md`, `HISTORY.rst`) to confirm the historic-download API and its `plan` strings.
- **Beta/odds API docs** read directly for the verified credit formula (`markets × regions`; historical `10 ×`; snapshots every 10 min since 2020-06-06).
- **Local environment**: `uv 0.12.13`, system `python` 3.14.7; `uv python list` shows 3.13.15 / 3.12.14 available for download (not installed). **No packages were installed.**

### Sources
- https://pypi.org/pypi/<pkg>/json (all versions above)
- https://github.com/betcode-org/betfair · https://betcode-org.github.io/betfair/ · https://github.com/probberechts/soccerdata · https://github.com/martineastwood/penaltyblog
- https://www.football-data.co.uk/ · https://www.football-data.co.uk/notes.txt · https://www.football-data.co.uk/data.php · https://www.football-data.co.uk/matches.php
- http://clubelo.com/API · http://api.clubelo.com/{YYYY-MM-DD,Fixtures,CLUBNAME}
- https://the-odds-api.com/ · https://the-odds-api.com/liveapi/guides/v4/ · https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
- https://www.api-football.com/pricing · https://www.api-football.com/documentation-v3
- https://historicdata.betfair.com/ · https://betfair-datascientists.github.io/modelling/dataSources/ · https://betfair-datascientists.github.io/api/apiappkey/ · https://betfair-datascientists.github.io/tutorials/flumineSimulations/ · https://betfair-datascientists.github.io/tutorials/goldenRulesOfAutomation/ · https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartI/ · https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/Exchange+API
- https://soccerdata.readthedocs.io/en/latest/ (+ reference/{fbref,understat,clubelo,sofascore,whoscored,matchhistory}.html)
- https://penaltyblog.readthedocs.io/en/latest/models/index.html · /scrapers/index.html
- https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/ · https://github.com/openfootball · https://github.com/hudl/open-data
- Cross-reference: `B2-au-betting-practicalities.md` (venues, fees, legality, responsible gambling)
