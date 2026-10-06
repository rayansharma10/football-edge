"""Understat team-match xG: cached fetch, parse, join to matches -> `match_xg`.

Uses the same JSON endpoint penaltyblog's scraper uses (getLeagueData), but caches the raw
response under data/raw/understat/<slug>/<year>.json so re-runs never re-scrape.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from fedge.teams import canon_map

log = logging.getLogger(__name__)

BASE = "https://understat.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fedge-research/0.1"
# football-data div -> Understat league slug
LEAGUES = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}
FIRST_YEAR = 2014
XG_DELAY = 3  # hours: available_at = kickoff + 3h


def season_label(year: int) -> str:
    return f"{year}/{(year + 1) % 100:02d}"


def cache_path(raw_dir: Path, slug: str, year: int) -> Path:
    return raw_dir / slug / f"{year}.json"


@dataclass
class FetchStatus:
    slug: str
    year: int
    status: str  # cached | downloaded | missing | error
    detail: str = ""


def fetch_league_season(slug: str, year: int, raw_dir: Path, delay: float = 3.0,
                        offline: bool = False) -> FetchStatus:
    path = cache_path(raw_dir, slug, year)
    if path.exists():
        return FetchStatus(slug, year, "cached")
    if offline:
        return FetchStatus(slug, year, "missing", "offline")
    headers = {
        "User-Agent": UA,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE}/league/{slug}/{year}",
    }
    try:
        r = requests.get(f"{BASE}/getLeagueData/{slug}/{year}", headers=headers,
                         cookies={"beget": "begetok"}, timeout=30)
        time.sleep(delay)
        if r.status_code != 200:
            return FetchStatus(slug, year, "error", f"HTTP {r.status_code}")
        data = r.json()
        if not data.get("dates"):
            return FetchStatus(slug, year, "missing", "no dates")
    except (requests.RequestException, ValueError) as e:
        return FetchStatus(slug, year, "error", repr(e))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return FetchStatus(slug, year, "downloaded")


def parse_fixtures(data: dict, div: str, year: int) -> pd.DataFrame:
    rows = []
    for e in data.get("dates", []):
        if not e.get("isResult"):
            continue
        rows.append({
            "understat_id": str(e["id"]),
            "div": div,
            "season": season_label(year),
            "kickoff_local": e["datetime"],
            "us_home": e["h"]["title"],
            "us_away": e["a"]["title"],
            "goals_home": int(e["goals"]["h"]),
            "goals_away": int(e["goals"]["a"]),
            "xg_home": float(e["xG"]["h"]),
            "xg_away": float(e["xG"]["a"]),
        })
    return pd.DataFrame(rows)


def load_understat(raw_dir: Path, years: list[int] | None = None) -> pd.DataFrame:
    frames = []
    for div, slug in LEAGUES.items():
        for p in sorted((raw_dir / slug).glob("*.json")) if (raw_dir / slug).exists() else []:
            year = int(p.stem)
            if years and year not in years:
                continue
            df = parse_fixtures(json.loads(p.read_text(encoding="utf-8")), div, year)
            if len(df):
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_match_xg(matches: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """Join Understat rows to matches by (div, date, canonical home, canonical away).

    Understat datetimes are naive; the date part is used (matches' date is taken in UK local
    time to be comparable). Unmapped teams are dropped from the join (reported as unmatched).
    """
    cm = canon_map("understat")
    u = us.copy()
    u["home_id"] = u["us_home"].map(cm)
    u["away_id"] = u["us_away"].map(cm)
    u["date"] = pd.to_datetime(u["kickoff_local"]).dt.strftime("%Y-%m-%d")

    fd = canon_map("football_data")
    m = matches[matches["div"].isin(LEAGUES)].copy()
    m["home_id"] = m["home"].map(fd)
    m["away_id"] = m["away"].map(fd)
    m["date"] = m["kickoff_utc"].dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")

    keys = ["div", "date", "home_id", "away_id"]
    cols = ["match_id", *keys, "season", "kickoff_utc", "FTHG", "FTAG"]
    ucols = ["understat_id", "xg_home", "xg_away", "goals_home", "goals_away"]
    j1 = m[cols].merge(u[[*keys, *ucols]], on=keys, how="inner").drop_duplicates(keys)
    j1["match_key"] = "date"

    # Fallback: same fixture (div, season, home, away), Understat date within 3 days AND same
    # full-time score. Covers date disagreements (postponements, late kickoffs). A home/away
    # pair occurs once per league-season, so this is unambiguous.
    rest = m[~m["match_id"].isin(j1["match_id"])]
    fk = ["div", "season", "home_id", "away_id"]
    j2 = rest[cols].merge(u.rename(columns={"date": "date_u"})[[*fk, "date_u", *ucols]],
                          on=fk, how="inner")
    gap = (pd.to_datetime(j2["date_u"]) - pd.to_datetime(j2["date"])).dt.days.abs()
    j2 = j2[(gap <= 3) & (j2["FTHG"] == j2["goals_home"]) & (j2["FTAG"] == j2["goals_away"])]
    j2 = j2.drop_duplicates("match_id")
    j2["match_key"] = "fixture"

    j = pd.concat([j1, j2], ignore_index=True)
    out = pd.DataFrame({
        "match_id": j["match_id"],
        "understat_id": j["understat_id"],
        "xg_home": j["xg_home"].astype(float),
        "xg_away": j["xg_away"].astype(float),
        "match_key": j["match_key"],
        "available_at": j["kickoff_utc"] + pd.Timedelta(hours=XG_DELAY),
    })
    return out.sort_values("match_id").reset_index(drop=True)


def join_rates(matches: pd.DataFrame, match_xg: pd.DataFrame) -> pd.DataFrame:
    m = matches[matches["div"].isin(LEAGUES)
                & matches["FTR"].notna()
                & (matches["season"] >= season_label(FIRST_YEAR))].copy()
    m["joined"] = m["match_id"].isin(set(match_xg["match_id"]))
    g = m.groupby(["div", "season"]).agg(matches=("match_id", "size"), joined=("joined", "sum"))
    g["rate"] = g["joined"] / g["matches"]
    return g.reset_index()


def write_match_xg(match_xg: pd.DataFrame, data_dir: Path) -> None:
    import duckdb

    p = data_dir / "interim" / "match_xg.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    match_xg.to_parquet(p, index=False, compression="zstd")
    con = duckdb.connect(str(data_dir / "fedge.duckdb"))
    try:
        con.execute("CREATE OR REPLACE TABLE match_xg AS SELECT * FROM read_parquet(?)",
                    [p.as_posix()])
    finally:
        con.close()


def ingest(data_dir: Path, delay: float = 3.0, offline: bool = False,
           current_year: int = 2026) -> tuple[pd.DataFrame, list[FetchStatus]]:
    raw = data_dir / "raw" / "understat"
    status = []
    for slug in LEAGUES.values():
        for year in range(FIRST_YEAR, current_year + 1):
            s = fetch_league_season(slug, year, raw, delay, offline)
            log.info("%s %s: %s %s", slug, year, s.status, s.detail)
            status.append(s)
    matches = pd.read_parquet(data_dir / "interim" / "matches.parquet")
    us = load_understat(raw)
    mx = build_match_xg(matches, us)
    write_match_xg(mx, data_dir)
    return mx, status
