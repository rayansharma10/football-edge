# B2 — Australian Practicalities of a Football Betting Bot

**Prepared for:** Rayan (adult student, Sydney; Windows 11; cost-conscious; Python)
**Date:** 6 October 2026
**Scope:** Exchanges vs bookmakers, API access, legality, tax, data sources, and responsible-gambling guardrails for a football (soccer) prediction/paper-betting bot with the option of real money later.
**Not legal or tax advice.** Verify anything load-bearing before you act on it. `UNVERIFIED` marks claims I could not confirm against a primary source.

---

## 1. Betfair Australia exchange & API

### 1.1 Who Betfair AU is, and the fees you actually pay
- Betfair Pty Limited is **licensed and regulated by the Northern Territory Government of Australia** ([betfair.com.au charges page footer](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)).
- **Commission** is charged on **net winnings per market**; if you lose a market you pay no commission. For **sport and international racing markets the Market Base Rate (MBR) is 6%**; NRL is 10%; Australian racing is 8% or 10% by state/code ([Betfair Hub – Commissions and Charges](https://www.betfair.com.au/hub/help/commissions-charges/)).
- **Football (soccer) is therefore 6% MBR** — i.e. ~6% of your net winnings on each market. This is the single dominant running cost of an exchange strategy.

### 1.2 Premium Charge (the "consistent winner" tax)
Betfair AU's revised Premium Charge took effect **6 January 2025** ([Betfair Hub – Premium Charge FAQ](https://www.betfair.com.au/hub/help/commissions-charges/premium-charge-faq/); [Betfair Charges §7](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)):
- An account is only *considered* if its **Last 52 Active Week Gross P&L exceeds £25,000** (AUD equivalent).
- Charge rates: **0% if < £25,000**, **20% if £25,000–£100,000**, **40% if > £100,000** (capped at 40%).
- Additional gates: only after betting in **> 100 markets** and with **lifetime Gross P&L > $0**. A "Buffer" carries forward losses/excess commission.
- AU-relevant nuance: **"Implied Commission" is 3% of market-level losses for ANZ customers** vs 2.5% for others.

**Practical read:** a hobby-scale football bot will essentially never hit the Premium Charge. It matters only if you scale to >A$45–50k (~£25k) rolling 52-week gross profit. ([charge page](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/))

### 1.3 Other Betfair charges
- **Transaction Charge:** only if you place **>5,000 transactions in any hour** — not a concern for a once-a-day football bot ([Charges §5](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)).
- **Turnover Charge:** applies only to **Racing NSW and NRL markets** (3.0% / 2.5% of matched back bets in specific circumstances) — **does not apply to football/soccer** ([Charges §9](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)).
- **Inactive Account Fee:** AUD 10/month after 13 months of no login, refundable on request ([Charges §8](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)).
- **API Access Fee:** `$200 × (Login Days / Month Calendar Days) − Commission paid in prior 12 months`, charged monthly, and it applies **only when you log in using a key with version designation "1.0"** — i.e. the **Live** key (the delayed key is "1.0-DELAY") ([Charges §10](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)). In effect: livish API access can cost up to **$200/month if you generate no commission**, tapering to $0 as commission accrues. **This is a real cost to budget for a live bot.**

### 1.4 App keys: Delayed vs Live
Every Betfair account gets **two keys** on `createDeveloperAppKeys`: a **Delayed** key (active) and a **Live** key (inactive) ([Developer Support – How do I create an Application Key?](https://support.developer.betfair.com/hc/en-us/articles/19192200541468-How-do-I-create-an-Application-Key)).
- **API access is free for development using the Delayed App Key** for private betting customers ([Developer Support – Are there any costs?](https://support.developer.betfair.com/hc/en-us/articles/115003864531-Are-there-any-costs-associated-with-API-access)).
- **Delayed key:** runs on the live production Exchange, gives **delayed price data with a variable 1–180 second delay**, doesn't show matched volume, and **does have betting functionality** ([Developer Support – Delayed or Live key](https://support.developer.betfair.com/hc/en-us/articles/360009638032-When-should-I-use-the-Delayed-or-Live-Application-Key); [AU Automation Hub – app key guide](https://betfair-datascientists.github.io/api/apiappkey/)). You must place **at least one bet through the delayed key** before live activation is considered.
- **Live key:** for real-time transacting. **A one-off, non-refundable £499 activation fee applies** for betting use in the global/UK developer program, and **read-only live access is not permitted** ([Costs article](https://support.developer.betfair.com/hc/en-us/articles/115003864531-Are-there-any-costs-associated-with-API-access); [Activate my Live App Key](https://support.developer.betfair.com/hc/en-us/articles/115003860331-How-do-I-activate-my-Live-App-Key)).
- **Australian customers apply via a different route:** Betfair AU directs you to a webform at [betfair-datascientists.github.io/api/apiappkey/](https://betfair-datascientists.github.io/api/apiappkey/) ("Activate your app key") and to `automation@betfair.com.au`. **Whether AU customers pay the £499 is UNVERIFIED** — the AU page omits the fee.
- **Geo-restriction:** requests from restricted IP regions are blocked; the list is China, France, Germany, Greece, Cyprus, Netherlands, North Korea, India, Iran, Iraq, Singapore, Turkey, UAE, USA. **Australia is not restricted** ([IP regions article](https://support.developer.betfair.com/hc/en-us/articles/28271961503516-Which-IP-regions-are-restricted-from-accessing-the-Betfair-API)).

### 1.5 Is automating bets allowed on Betfair? Yes.
- Betfair AU's own Hub states an automated betting bot is "**completely above board**… thousands of Betfair customers bet for a living using automated betting systems" ([Betfair Hub – Build A Betting Bot](https://www.betfair.com.au/hub/education/build-a-betting-bot/)).
- Betfair AU runs **"The Automation Hub"** (betfair-datascientists.github.io) with a free **"Building a Soccer Bot" Part 1–3** tutorial series, **Flumine simulations**, "Golden Rules of Automation", and app-key/tutorial pages ([Automation Hub](https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartI/), [Flumine Simulations](https://betfair-datascientists.github.io/tutorials/flumineSimulations/), [Golden Rules](https://betfair-datascientists.github.io/tutorials/goldenRulesOfAutomation/)). Caveat: its human-facing tutorials note the *simulation* steps require **Betfair PRO (paid) data** — contact `automation@betfair.com.au`.

### 1.6 Python libraries
- **betfairlightweight** — Betfair API-NG wrapper with streaming. Repo now at [`betcode-org/betfair`](https://github.com/betcode-org/betfair) (license MIT, ~515 stars, last push 4 Sep 2026).
- **flumine** — betting/trading framework built on betfairlightweight. [`betcode-org/flumine`](https://github.com/betcode-org/flumine) (MIT, ~248 stars, last push 1 Oct 2026).
- Betfair AU's own model tutorials/examples: [`betfair-datascientists/predictive-models`](https://github.com/betfair-datascientists/predictive-models) (EPL tutorial uses "publicly available data to model odds and create an automated betting strategy"; repo mostly last pushed 2022).

### 1.7 Betfair historical data
- **Betfair Historic Data site** ([historicdata.betfair.com](https://historicdata.betfair.com/)) sells time-stamped Exchange data since 2016 (prices, volume, BSP, settlements), including soccer, as JSON in TAR files.
- Three tiers ([Automation Hub – Historical Data Sources](https://betfair-datascientists.github.io/modelling/dataSources/)):
  - **Basic — free:** 1-minute odds intervals, no volume.
  - **Advanced — paid:** 1-second odds, volume included.
  - **Pro — paid:** 50-millisecond intervals, volume included.
- **Australian and New Zealand customers should email `automation@betfair.com.au` before purchasing** ([same page](https://betfair-datascientists.github.io/modelling/dataSources/)).
- **Free CSV data:** Betfair also publishes free, no-login historical racing CSVs, and **Sports & Racing CSV files for AU/NZ** (monthly blocks from **January 2023**; select Australian leagues from **2020**: AFL/AFLW, A-League & A-League Women's, BBL/WBBL, NRL, NBL) ([same page](https://betfair-datascientists.github.io/modelling/dataSources/)). Football coverage here is **A-League only**; European leagues live in the paid Stream data.

---

## 2. Australian bookmakers (Sportsbet, TAB, Ladbrokes, bet365 AU…)

- **No public betting APIs** are offered by the corporate bookmakers for retail customers — **UNVERIFIED** as a blanket statement, but I found none, and their model is set-odds/anti-sharp. Betfair is the only major AU venue that both provides an API **and** explicitly sanctions automation ([Betfair Hub](https://www.betfair.com.au/hub/education/build-a-betting-bot/)).
- **Corporate bookmakers restrict or close winning punters.** This is long-documented (ABC News, 2019: operators "simply restrict how much they can bet or close them down altogether" — [abc.net.au](https://www.abc.net.au/news/2019-01-20/gambling-how-bookmakers-stop-winning-bettors/10708394)) and still current: a **Sydney Morning Herald report (Apr 2026)** covers TAB freezing customers' accounts and withholding winnings ([smh.com.au](https://www.smh.com.au/sport/racing/fobbed-off-frozen-out-punters-claim-a-betting-giant-is-denying-them-thousands-of-dollars-20260401-p5zkkh.html)), and the topic has been raised in a parliamentary inquiry into online gambling ([gamingcompliancenews.net](https://gamingcompliancenews.net/industry/australia-looks-into-betting-firms-banning-winning-punters/)).
- **Automated betting on corporate bookmakers is generally prohibited/restricted by their T&Cs** — bots themselves are not illegal under the Interactive Gambling Act 2001, but a bookmaker can void bets or close accounts for automated activity ([matchedbettingaustralia.com](https://matchedbettingaustralia.com/what-is-the-role-of-betting-bots-in-australias-sports-betting/) — secondary source; the **specific T&C clause is UNVERIFIED** because I did not extract it from a bookmaker's terms). Sportsbet's General Rules are at ([helpcentre.sportsbet.com.au](https://helpcentre.sportsbet.com.au/hc/en-us/articles/115004802547-Sportsbet-Rules-Terms-Conditions)).

### 2.1 Minimum Bet Laws (MBLs) — racing only, not football
Government/racing-body rules force licensed bookmakers to accept bets up to a limit from a winning punter — but **only on Australian racing (thoroughbred/harness/greyhound), not on football/soccer.**
- **NSW (Racing NSW, Schedule 1 – Minimum Betting Limits):** bookmakers with ≥ $5m NSW thoroughbred turnover must lay **$2,000 win / $800 place** on **metropolitan** races and **$1,000 / $400** on non-metro; smaller bookmakers **$1,000 / $400** all races. Critically the rule **prohibits** the bookmaker "closing a backer's account", "refusing to open a backer's account", or "placing any restrictions on a backer's account" to dodge the limit — **but only for these races** ([Racing NSW MBL PDF](https://www.racingnsw.com.au/wp-content/uploads/MINIMUM-BETTING-LIMITS-%E2%80%93BOOKMAKERS.pdf); [Racing NSW Minimum Bet page](https://www.racingnsw.com.au/minimum-bet/)).
- **Other states** (VIC, QLD, SA, WA, TAS, ACT) have similar racing MBLs, metro typically **$2,000/$800** ([comparethebookie.com.au – Minimum Bet Laws, reviewed July 2026](https://comparethebookie.com.au/minimum-bet-laws/) — secondary compilation; UNVERIFIED in detail but consistent with the NSW primary source).
- **Bottom line for a football bot:** no minimum-bet protection exists for soccer markets — a corporate bookmaker can limit or close you for winning on football, and automated betting likely breaches their terms. This is the core reason an exchange (Betfair) is the viable real-money venue.

---

## 3. Legality, regulation & tax

### 3.1 Legality of betting & of offshore operators
- Wagering is **legal for individuals in Australia** via Australian-licensed operators. Betfair AU is NT-licensed ([Charges page footer](https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/)).
- The **Interactive Gambling Act 2001** bans **unlicensed offshore operators** from offering services to Australians. **ACMA blocks** illegal sites and pushes them out: **1,518 illegal gambling/affiliate websites blocked** since Nov 2019, and **220+ illegal services have pulled out** since 2017 ([ACMA, 13 Feb 2026](https://www.acma.gov.au/articles/2026-02/latest-illegal-online-gambling-websites-blocked); [ACMA – blocked gambling websites](https://www.acma.gov.au/blocked-gambling-websites); [ACMA quarterly report Apr–Jun 2026 – 187 sites blocked](https://www.acma.gov.au/publications/2026-08/report/action-interactive-gambling-and-gambling-advertising-april-june-2026)).
- **Pinnacle (the archetypal "sharp/bookmaker-friendly" venue) is not available to Australians** — Pinnacle withdrew from the Australian market for lack of a licence and lists Australia among its restricted countries ([bestbitcoincasino.com summary](https://www.bestbitcoincasino.com/news/pinnacle-exits-australia-lack-license/); [arbusers restricted-countries list](https://arbusers.com/pinnacle-restricted-countries-t8922/) — secondary sources, UNVERIFIED against Pinnacle's own page, which I could not fetch). **Practical consequence:** Betfair AU is the realistic low-margin venue that accepts Australian customers.

### 3.2 Credit card & digital-currency ban (2024)
- **From 11 June 2024**, online/telephone wagering operators **cannot accept credit cards**, credit-linked digital wallets, or **digital currency/crypto** for deposits; the ban extends to on-course bookmakers online. It does **not** apply to lotteries ([ACMA – Credit ban](https://www.acma.gov.au/credit-ban)).
- Operators offering credit, or helping you access credit, is illegal; fines for non-compliance up to **$247,500**; the ban is to be **reviewed after June 2026** ([Dept of Infrastructure – Credit card ban](https://www.infrastructure.gov.au/media-communications/media-gambling-laws-regulation/gambling/credit-card-ban)).
- **Implication:** fund the bot's betting account via **debit/bank transfer only**; a credit-card deposit path is not available.

### 3.3 BetStop — national self-exclusion register
- **BetStop – the National Self-Exclusion Register** is a free Australian Government register that blocks you from **all licensed Australian online and phone gambling providers** — they can't let you bet, open new accounts, or send marketing ([betstop.gov.au](https://www.betstop.gov.au/)). Registration ~5 minutes. It is referenced in Betfair AU's own footer.

### 3.4 Tax — is gambling income taxable?
- **For a hobbyist/recreational punter, gambling winnings are generally not ordinary income**, and gambling winnings/losses are **disregarded for CGT** ([ATO – Crypto asset prizes and gambling winnings](https://www.ato.gov.au/individuals-and-families/investments-and-assets/crypto-asset-investments/transactions-acquiring-and-disposing-of-crypto-assets/crypto-asset-prizes-and-gambling-winnings), updated 22 Jun 2026).
- **But if you carry on a *business* of betting/gambling, winnings are assessable.** ATO Taxation Ruling **IT 2655** sets out the criteria ([ATO IT 2655](https://www.ato.gov.au/law/view/document?DocID=ITR/IT2655/NAT/ATO/00001)):
  - whether gambling is conducted in a **"systematic, organised and businesslike way"**;
  - the **volume and size** of the gambling;
  - whether gambling is **related to / part of other businesslike activities** (e.g. horse breeding);
  - whether the gambler's **principal purpose is profit rather than pleasure**.
  - Case law (Evans, Babka, Brajkovich) found "mere punting" rarely a business, and noted gambling involving **significant skill** is more likely to have tax consequences than random events.
- **Risk flag (not advice):** a systematic, code-driven, profit-oriented bot is exactly the profile that *could* be argued to tip a punter toward "carrying on a business". Get professional advice if stakes/volume become material.
- **Local taxes:** AU state/territory governments tax *operators*, not individual punters' winnings (noted in [Betfair Hub – Understanding Commission](https://www.betfair.com.au/hub/education/understanding-commission/)). Betfair's GST treatment of its own services is separate (see [ATO – GST when conducting gambling activities](https://www.ato.gov.au/businesses-and-organisations/gst-excise-and-indirect-taxes/gst/in-detail/your-industry/gst-when-conducting-gambling-activities)).

---

## 4. Data sources for a football bot

| Source | Cost | Data | Coverage | Limits / practical notes | ToS / licensing notes |
|---|---|---|---|---|---|
| **football-data.co.uk** ([site](https://www.football-data.co.uk/), [key/notes](https://www.football-data.co.uk/notes.txt)) | Free | Results, half-time scores, shots/corners/cards, **pre-closing odds from many bookmakers incl. Betfair**, plus **closing odds** (abbrev + "C") | Mainly European leagues (many seasons) | CSV download; no live data | Free for personal use; footnotes warn some old bookmaker columns discontinued |
| **The Odds API** ([home](https://the-odds-api.com/), [sports](https://the-odds-api.com/sports-odds-data/sports-apis.html)) | **Free 500 credits/mo**; paid from **US$30 / 20k credits** | Odds (h2h, spreads, totals, outrights) + scores; **historical odds on paid tiers** | Global soccer + **Aussie bookmakers: Sportsbet, TAB, Neds, Ladbrokes, Betfair, Unibet** | Credit-metered API, key required | Commercial API; all plans give all sports |
| **API-Football / API-Sports** ([pricing](https://www.api-football.com/pricing)) | **Free $0 = 100 req/day**; Pro **$19** = 7,500/day; Ultra $29 = 75k/day; Mega $39 = 150k/day | Fixtures, livescore, standings, **lineups**, events, stats, **pre-match & in-play odds**, predictions, H2H | 1,200+ leagues; free plan limited in available seasons | Per-day + per-minute rate limits | Commercial API; free plan no card required |
| **football-data.org** ([site](https://www.football-data.org/), [coverage](https://www.football-data.org/coverage)) | **Free tier "forever"** on top comps; paid plans for more | Fixtures, results, tables, squads, **lineups/subs**, live scores | Free: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga, Championship, Brazilian Série A, **Champions League, World Cup, Euros** | Request-rate limited on free | Commercial API; free tier is first-party and stable |
| **Understat** ([understat.com](https://understat.com/)) | Free | **xG** for teams & players (shot-quality model) | Top 6 European leagues: EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL | **No official API** — scrape pages / CSV-JSON exports; be gentle | Scraping-adjacent; no published API licence — treat as **UNVERIFIED** for redistribution |
| **Club Elo** ([clubelo.com](http://clubelo.com/)) | Free | Elo ratings for club football (strength → future predictions) | European clubs (deep history) | Provides CSV/JSON endpoints | Free for use; check site terms for redistribution |
| **openfootball** ([GitHub org](https://github.com/openfootball)) | Free | Results, fixtures, league data (structured text/JSON; `football.json`) | England, Germany, Spain, Italy, France, Brazil, Argentina + more; internationals 1872–2026 | Static datasets, **no API key**; not live | **CC0-1.0 public domain** (cleanest licence of the lot) |
| **StatsBomb open data** ([hudl/open-data](https://github.com/hudl/open-data)) | Free | **Event-level data** (passes, shots, pressures) | Selected competitions only (not all leagues/seasons) | GitHub repo; ~3,671★, updated Sep 2026 | Custom/`NOASSERTION` licence — **read the terms before commercial use** |
| **FBref** ([fbref.com](https://fbref.com/en/)) | Free (Stathead paid) | Basic stats huge; historically **advanced stats (xG, progressive carries)** | 100+ competitions | **Advanced data removed Jan 2026** when the provider terminated access and demanded deletion ([Sports Reference, 20 Jan 2026](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/)); site now advertises **"xG from Opta"** and "Progressive Carries are BACK" | Stability **in flux** — the removal plus re-add is documented; exact current scope **UNVERIFIED** |
| **Sofascore / FotMob** ([fotmob.com/terms](https://www.fotmob.com/terms)) | Free to browse | Live scores, lineups, stats, some xG | Very broad | **No official public API**; scraping risks ToS breach / blocking | Specific anti-scraping clauses **UNVERIFIED** (JS ToS not extractable); treat scraping as fragile and legally riskier than the licensed APIs above |

**Lineup timing note:** API-Football and football-data.org both expose **lineups**; confirmed starting XIs typically drop **~1 hour before kick-off**, so a pre-match model that uses lineups must run inside that window (worth verifying per league, **UNVERIFIED** for exact per-competition timing).

---

## 5. Where to bet if real money (exchange vs bookmakers)

| Venue | API? | Limits / bans? | Fees & costs | Verdict for a bot |
|---|---|---|---|---|
| **Betfair Exchange (AU)** | **Yes** — official Exchange API; Delayed key free, Live key needs activation (~£499, AU fee UNVERIFIED) | **Not limited** — you're betting against other users, not a bookmaker's risk book | **6% MBR** on football net winnings; Live-key **API Access Fee** up to $200/mo offset by commission; Premium Charge only >£25k/52wk | **The only sensible real-money venue for automated AU football betting** |
| **Sportsbet / TAB / Ladbrokes / bet365 AU** | **No** public betting API | **Restrict/close winning accounts**; automated betting generally against T&Cs; **no MBL on football** | Margins baked into odds (typically ~5–15% overround); no explicit fee but worse prices | **Avoid for automation** — account likely to be limited or closed |
| **Offshore (e.g. Pinnacle)** | Pinnacle has an API | — | Low margin | **Not available to Australians** — unlicensed; blocked/pulled out ([ACMA](https://www.acma.gov.au/blocked-gambling-websites)) |

---

## 6. Responsible-gambling guardrails worth building in

Keep these practical, not preachy — they double as risk controls:

1. **Hard staking cap in code.** Fixed fractional Kelly (e.g. ¼–½ Kelly) or flat-stake with a per-bet and per-day maximum written as *immutable constants*, plus a hard daily/weekly loss stop that halts the bot. "The LLM proposes, deterministic code acts."
2. **Use BetStop awareness as a design input.** BetStop blocks all licensed AU operators from letting you bet once registered ([betstop.gov.au](https://www.betstop.gov.au/)); a bot that can't log in when self-excluded is behaving correctly. Don't build bypass attempts.
3. **Set the operator's own deposit/bet limits.** Australian licensed wagering providers must offer deposit limits and activity statements — set them low at account level so code bugs can't drain the bankroll.
4. **No credit; debit/bank only.** Credit-card and crypto deposits are banned (11 Jun 2024) — build the funding flow around bank transfer/debit ([ACMA](https://www.acma.gov.au/credit-ban)).
5. **Paper-trade by default; require an explicit flag for live.** Default `LIVE=false`; a real-money run needs a deliberate, logged switch. Validate against **closing-line value / backtest** before switching on.
6. **Log every bet, price, stake and result to disk** with timestamps — needed for honest evaluation, tax discussion, and dispute resolution.
7. **Reality checks & bankroll ceiling.** Cap total bankroll at an amount you're fully prepared to lose; enable session/time reminders. Free support: **1800 858 858**, [gamblinghelponline.org.au](https://gamblinghelponline.org.au/).

---

## 7. Recommendation for Rayan

**Paper-bet on Betfair Exchange data; don't touch corporate bookmakers for automation; keep the stack free until the model proves itself.**

1. **Venue: Betfair Exchange only** for anything automated. It's NT-licensed, explicitly sanctions bots ([Betfair Hub](https://www.betfair.com.au/hub/education/build-a-betting-bot/)), won't limit you for winning, and is reachable from AU IPs. Corporate bookmakers will limit/close you and their T&Cs likely prohibit bots — no MBL protection on football.
2. **Start with the free Delayed App Key** (1–180s delay, betting enabled) + **paper trading**. Only pursue the Live key and the ~£499 activation once a strategy shows positive closing-line value. **Budget for the up-to-$200/month API Access Fee** on the live key (offset by commission).
3. **Data stack (free first):**
   - Historical results **+ closing odds** → **football-data.co.uk**.
   - Model base rate / team strength → **Club Elo** + **openfootball** (CC0) + **football-data.org** free tier.
   - **xG** → **Understat** (top-6 leagues) and, cautiously, **FBref** (advanced stats churned hard in Jan 2026 — verify current coverage each run).
   - Live/market odds → **The Odds API** free tier (500 credits/mo), which covers **Betfair + AU bookmakers**.
   - Fixtures/lineups/odds fallback → **API-Football** free tier (100 req/day).
4. **Follow Betfair's own playbook:** mirror the **"Building a Soccer Bot" Part 1–3** series and the **Golden Rules of Automation** (avoid data leakage, don't overfit, treat too-good backtests as broken, always have a staking plan) ([Automation Hub](https://betfair-datascientists.github.io/tutorials/goldenRulesOfAutomation/)). Build with **betfairlightweight** + **flumine** (both MIT, actively maintained).
5. **Tax/legal posture:** hobby punting winnings are generally not taxable and disregarded for CGT ([ATO](https://www.ato.gov.au/individuals-and-families/investments-and-assets/crypto-asset-investments/transactions-acquiring-and-disposing-of-crypto-assets/crypto-asset-prizes-and-gambling-winnings)), **but** a systematic, profit-driven bot edges toward the "business of gambling" test in [IT 2655](https://www.ato.gov.au/law/view/document?DocID=ITR/IT2655/NAT/ATO/00001). Get advice before scaling. This note is **not legal or tax advice**.
6. **Build the guardrails in from day one** (§6): hard staking/loss caps in code, `LIVE=false` default, deposit limits, debit-only funding, full bet logging, BetStop awareness.

**Cheapest viable first milestone:** free Delayed key + football-data.co.uk + Club Elo + The Odds API free tier → backtest a league-average xG/Elo model on paper → only then consider live.

---

## Sources (URLs used inline above)
- Betfair AU Hub – Commissions and Charges: https://www.betfair.com.au/hub/help/commissions-charges/
- Betfair AU Hub – Premium Charge FAQ: https://www.betfair.com.au/hub/help/commissions-charges/premium-charge-faq/
- Betfair AU Hub – Understanding Commission: https://www.betfair.com.au/hub/education/understanding-commission/
- Betfair AU Hub – Build A Betting Bot: https://www.betfair.com.au/hub/education/build-a-betting-bot/
- Betfair Charges (Terms): https://www.betfair.com/www/GBR/en/aboutUs/Betfair.Charges/
- Betfair Developer Program (support): https://support.developer.betfair.com/hc/en-us
- — Costs: https://support.developer.betfair.com/hc/en-us/articles/115003864531-Are-there-any-costs-associated-with-API-access
- — Create App Key: https://support.developer.betfair.com/hc/en-us/articles/19192200541468-How-do-I-create-an-Application-Key
- — Delayed vs Live key: https://support.developer.betfair.com/hc/en-us/articles/360009638032-When-should-I-use-the-Delayed-or-Live-Application-Key
- — Activate Live key: https://support.developer.betfair.com/hc/en-us/articles/115003860331-How-do-I-activate-my-Live-App-Key
- — Restricted IP regions: https://support.developer.betfair.com/hc/en-us/articles/28271961503516-Which-IP-regions-are-restricted-from-accessing-the-Betfair-API
- Betfair AU Automation Hub: https://betfair-datascientists.github.io/api/apiappkey/ ; https://betfair-datascientists.github.io/modelling/dataSources/ ; https://betfair-datascientists.github.io/tutorials/flumineSimulations/ ; https://betfair-datascientists.github.io/tutorials/goldenRulesOfAutomation/ ; https://betfair-datascientists.github.io/modelling/howToBuildASoccerBotPartI/
- betfairlightweight: https://github.com/betcode-org/betfair · flumine: https://github.com/betcode-org/flumine · Betfair DS models: https://github.com/betfair-datascientists/predictive-models
- ACMA – Credit ban: https://www.acma.gov.au/credit-ban
- Dept of Infrastructure – Credit card ban: https://www.infrastructure.gov.au/media-communications/media-gambling-laws-regulation/gambling/credit-card-ban
- ACMA – Illegal sites blocked (Feb 2026): https://www.acma.gov.au/articles/2026-02/latest-illegal-online-gambling-websites-blocked
- ACMA – Blocked gambling websites: https://www.acma.gov.au/blocked-gambling-websites · Quarterly report Apr–Jun 2026: https://www.acma.gov.au/publications/2026-08/report/action-interactive-gambling-and-gambling-advertising-april-june-2026
- BetStop: https://www.betstop.gov.au/
- ATO – Crypto asset prizes and gambling winnings: https://www.ato.gov.au/individuals-and-families/investments-and-assets/crypto-asset-investments/transactions-acquiring-and-disposing-of-crypto-assets/crypto-asset-prizes-and-gambling-winnings
- ATO – IT 2655 (betting/gambling as a business): https://www.ato.gov.au/law/view/document?DocID=ITR/IT2655/NAT/ATO/00001
- Racing NSW – Minimum Bet + MBL PDF: https://www.racingnsw.com.au/minimum-bet/ ; https://www.racingnsw.com.au/wp-content/uploads/MINIMUM-BETTING-LIMITS-%E2%80%93BOOKMAKERS.pdf
- Compare The Bookie – Minimum Bet Laws by State (secondary): https://comparethebookie.com.au/minimum-bet-laws/
- ABC News – limiting winning bettors: https://www.abc.net.au/news/2019-01-20/gambling-how-bookmakers-stop-winning-bettors/10708394
- SMH – TAB freezing accounts (Apr 2026): https://www.smh.com.au/sport/racing/fobbed-off-frozen-out-punters-claim-a-betting-giant-is-denying-them-thousands-of-dollars-20260401-p5zkkh.html
- Sportsbet Rules & T&Cs: https://helpcentre.sportsbet.com.au/hc/en-us/articles/115004802547-Sportsbet-Rules-Terms-Conditions
- football-data.co.uk: https://www.football-data.co.uk/ ; https://www.football-data.co.uk/notes.txt
- The Odds API: https://the-odds-api.com/ ; https://the-odds-api.com/sports-odds-data/sports-apis.html
- API-Football pricing: https://www.api-football.com/pricing
- football-data.org: https://www.football-data.org/ ; https://www.football-data.org/coverage
- Understat: https://understat.com/ · Club Elo: http://clubelo.com/ · openfootball: https://github.com/openfootball · StatsBomb open data: https://github.com/hudl/open-data
- FBref: https://fbref.com/en/ · Sports Reference data-removal notice: https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/
- FotMob terms: https://www.fotmob.com/terms
