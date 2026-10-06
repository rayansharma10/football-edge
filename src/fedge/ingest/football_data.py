"""football-data.co.uk ingest: download, cache, parse to tidy `matches` + long `odds`.

Assumptions (documented, see also docs/research/M2-data-tooling.md):

* ``Date`` / ``Time`` are UK local time (Europe/London), converted to UTC.
  If ``Time`` is missing the kickoff is set to 23:59 UK local on that date and
  ``time_unknown=True``. Late-in-day is deliberately conservative: it can only
  make ``available_at`` later, never earlier, so no lookahead is introduced.
* ``available_at`` for **pre-closing** odds = kickoff - 24h. football-data
  collects them on Friday (weekend fixtures) and Tuesday (midweek) afternoons,
  UK time, i.e. up to ~3 days before kickoff, so kickoff-24h is a conservative
  (late) estimate. UNVERIFIED against true collection timestamps.
* ``available_at`` for **closing** odds = kickoff.
* Pinnacle (``PS``) rows with kickoff >= 2025-07-23 are flagged ``stale=True``
  (football-data says Pinnacle's API has been unreliable since that date).
"""

from __future__ import annotations

import hashlib
import logging
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fedge.schemas import MATCH_STATS

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
USER_AGENT = "Mozilla/5.0 (compatible; football-edge research; personal use)"
PINNACLE_STALE_FROM = pd.Timestamp("2025-07-23", tz="UTC")
PRE_LEAD = pd.Timedelta(hours=24)

# Bookmaker column prefixes per market. Output code for football-data's bare "P" is "PS".
_BK_1X2 = ["B365", "BW", "BF", "BFD", "BMGM", "BV", "CL", "GB", "IW", "LB", "PS", "SB", "SJ",
           "SO", "SY", "VC", "WH", "1XB", "Max", "Avg", "BFE"]
_BK_OU = ["B365", "P", "Max", "Avg", "BFE", "GB", "BW", "IW", "VC", "WH", "SB", "SJ", "SY", "BV",
          "LB", "1XB"]
_BK_AH = ["B365", "P", "Max", "Avg", "BFE", "GB", "BW", "IW", "VC", "WH", "SB", "SJ", "SY", "BV",
          "LB", "1XB"]
# Legacy (<=2018/19) aggregate columns: BbMx -> Max, BbAv -> Avg.
_LEGACY = {"BbMx": "Max", "BbAv": "Avg"}
_OUT_CODE = {"P": "PS"}

_OPTIONAL_TEXT = ["HTR", "Referee"]


# ---------------------------------------------------------------- config / seasons
def load_config(path: Path | str = "config/leagues.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def season_code(season: str) -> str:
    """'2024/25' -> '2425'."""
    a, b = season.split("/")
    return a[2:] + b


# ---------------------------------------------------------------- time handling
def uk_local_to_utc(local: pd.Series) -> pd.Series:
    """Naive UK wall-clock datetimes -> tz-aware UTC.

    Ambiguous autumn-fold times (01:00-02:00 on the last Sunday of October) resolve to GMT;
    non-existent spring-forward times shift forward.
    """
    loc = local.dt.tz_localize(
        "Europe/London", ambiguous=np.zeros(len(local), dtype=bool), nonexistent="shift_forward"
    )
    return loc.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def make_match_id(div: str, date: str, home: str, away: str) -> str:
    key = f"{div}|{date}|{home}|{away}".encode()
    return hashlib.sha1(key).hexdigest()[:16]


# ---------------------------------------------------------------- download
@dataclass
class FetchResult:
    div: str
    season: str
    path: Path | None
    status: str  # cached | downloaded | missing | error


def _is_csv(content: bytes) -> bool:
    head = content.lstrip(b"\xef\xbb\xbf").lstrip()[:200].lower()
    return head.startswith(b"div,") or head.startswith(b"country,")


def fetch_one(div: str, season: str, raw_dir: Path, current: str, session=None,
              delay: float = 1.0) -> FetchResult:
    """Download one division/season CSV into raw_dir, honouring the cache rules.

    Finished seasons are cached forever (including a ``.missing`` marker for 404s);
    the current season is always re-downloaded.
    """
    session = session or requests.Session()
    sdir = raw_dir / season.replace("/", "-")
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{div}.csv"
    marker = sdir / f"{div}.missing"
    finished = season != current
    if finished and path.exists():
        return FetchResult(div, season, path, "cached")
    if finished and marker.exists():
        return FetchResult(div, season, None, "missing")
    url = BASE_URL.format(code=season_code(season), div=div)
    time.sleep(delay)
    try:
        r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        log.warning("%s %s: request failed: %s", div, season, e)
        return FetchResult(div, season, path if path.exists() else None, "error")
    if r.status_code == 404 or (r.status_code == 200 and not _is_csv(r.content)):
        if finished:
            marker.write_text("not available\n")
        log.info("%s %s: not available (HTTP %s)", div, season, r.status_code)
        return FetchResult(div, season, None, "missing")
    if r.status_code != 200:
        log.warning("%s %s: HTTP %s", div, season, r.status_code)
        return FetchResult(div, season, path if path.exists() else None, "error")
    path.write_bytes(r.content)
    return FetchResult(div, season, path, "downloaded")


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a raw CSV: utf-8-sig, falling back to latin-1."""
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    from io import StringIO

    df = pd.read_csv(StringIO(text), dtype=str, on_bad_lines="skip", skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- parsing
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)


def parse_matches(df: pd.DataFrame, season: str, div: str | None = None) -> pd.DataFrame:
    """Raw football-data frame -> tidy `matches` frame (validated by caller)."""
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"], how="any").copy()
    df = df[df["HomeTeam"].str.strip() != ""]
    divs = df["Div"].str.strip() if "Div" in df.columns else pd.Series(div, index=df.index)
    date = pd.to_datetime(df["Date"].str.strip(), format="mixed", dayfirst=True, errors="coerce")
    df, divs, date = df[date.notna()], divs[date.notna()], date[date.notna()]

    t = _col(df, "Time").astype("string").str.strip()
    t_ok = t.str.match(r"^\d{1,2}:\d{2}$").fillna(False).astype(bool)
    hh = pd.to_numeric(t.str.split(":").str[0].where(t_ok), errors="coerce").fillna(23)
    mm = pd.to_numeric(t.str.split(":").str[1].where(t_ok), errors="coerce").fillna(59)
    local = date.dt.normalize() + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")

    home = df["HomeTeam"].str.strip()
    away = df["AwayTeam"].str.strip()
    out = pd.DataFrame(
        {
            "match_id": [
                make_match_id(d, dt.strftime("%Y-%m-%d"), h, a)
                for d, dt, h, a in zip(divs, date, home, away, strict=True)
            ],
            "div": divs.values,
            "season": season,
            "kickoff_utc": uk_local_to_utc(local.reset_index(drop=True)).values,
            "time_unknown": (~t_ok).values,
            "home": home.values,
            "away": away.values,
        },
        index=df.index,
    )
    for c in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        out[c] = _num(_col(df, c)).astype("Int64")
    for c in ["FTR", "HTR"]:
        v = _col(df, c).astype("string").str.strip()
        out[c] = v.astype(object).where(v.isin(["H", "D", "A"]).fillna(False), None)
    ref = _col(df, "Referee").astype("string").str.strip()
    out["Referee"] = ref.astype(object).where(ref.notna() & (ref != ""), None)
    for c in MATCH_STATS:
        out[c] = _num(_col(df, c)).astype("Float64")
    out["HxG"] = _num(_col(df, "HxG")).astype("Float64")
    out["AxG"] = _num(_col(df, "AxG")).astype("Float64")
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    dup = out["match_id"].duplicated(keep="first")
    if dup.any():
        log.warning("%s %s: dropping %d duplicate match_ids", div, season, int(dup.sum()))
        out = out[~dup]
    return out  # indexed by raw-frame row index (used to align odds)


def _odds_specs(columns: set[str]):
    """Yield (source_col, bookmaker, market, selection, phase, line_col)."""
    for bk in _BK_1X2:
        for ph, mid in (("pre", ""), ("close", "C")):
            for s in "HDA":
                c = f"{bk}{mid}{s}"
                if c in columns:
                    yield c, bk, "1x2", s, ph, None
    for bk in _BK_OU:
        for ph, mid in (("pre", ""), ("close", "C")):
            for sel, sym in (("over", ">"), ("under", "<")):
                c = f"{bk}{mid}{sym}2.5"
                if c in columns:
                    yield c, bk, "ou25", sel, ph, None
    for bk in _BK_AH:
        for ph, mid, lc in (("pre", "", "AHh"), ("close", "C", "AHCh")):
            for sel in "HA":
                c = f"{bk}{mid}AH{sel}"
                if c in columns:
                    yield c, bk, "ah", sel, ph, lc
    # legacy Bb* aggregates (only used when the modern column is absent)
    for pfx, bk in _LEGACY.items():
        for s in "HDA":
            if f"{pfx}{s}" in columns and f"{bk}{s}" not in columns:
                yield f"{pfx}{s}", bk, "1x2", s, "pre", None
        for sel, sym in (("over", ">"), ("under", "<")):
            if f"{pfx}{sym}2.5" in columns and f"{bk}{sym}2.5" not in columns:
                yield f"{pfx}{sym}2.5", bk, "ou25", sel, "pre", None
        for sel in "HA":
            if f"{pfx}AH{sel}" in columns and f"{bk}AH{sel}" not in columns:
                yield f"{pfx}AH{sel}", bk, "ah", sel, "pre", "BbAHh"


def parse_odds(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Raw frame + parse_matches() output (indexed by raw row) -> long-format odds."""
    raw = df.loc[matches.index]
    ko = matches
    cols = set(raw.columns)
    parts = []
    for c, bk, mkt, sel, ph, lc in _odds_specs(cols):
        price = _num(raw[c])
        ok = price.notna() & (price > 1.0)
        if not ok.any():
            continue
        line = _num(raw[lc]) if lc and lc in cols else pd.Series(np.nan, index=raw.index)
        part = pd.DataFrame(
            {
                "match_id": ko["match_id"][ok.values].values,
                "kickoff": ko["kickoff_utc"][ok.values].values,
                "bookmaker": _OUT_CODE.get(bk, bk),
                "market": mkt,
                "selection": {"H": "H", "D": "D", "A": "A"}.get(sel, sel),
                "phase": ph,
                "line": line[ok].values if mkt == "ah" else np.nan,
                "price": price[ok].values,
            }
        )
        parts.append(part)
    if not parts:
        return _empty_odds()
    o = pd.concat(parts, ignore_index=True)
    o["kickoff"] = pd.to_datetime(o["kickoff"], utc=True).astype("datetime64[ns, UTC]")
    o["available_at"] = o["kickoff"].where(o["phase"] == "close", o["kickoff"] - PRE_LEAD)
    o["stale"] = (o["bookmaker"] == "PS") & (o["kickoff"] >= PINNACLE_STALE_FROM)
    o = o.drop(columns="kickoff")
    o = o.drop_duplicates(["match_id", "bookmaker", "market", "selection", "phase"])
    return o


def _empty_odds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": pd.Series(dtype=str),
            "bookmaker": pd.Series(dtype=str),
            "market": pd.Series(dtype=str),
            "selection": pd.Series(dtype=str),
            "phase": pd.Series(dtype=str),
            "line": pd.Series(dtype=float),
            "price": pd.Series(dtype=float),
            "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "stale": pd.Series(dtype=bool),
        }
    )


def parse_file(df: pd.DataFrame, season: str, div: str | None = None):
    """Parse a raw frame into (matches, odds)."""
    m = parse_matches(df, season, div)
    o = parse_odds(df, m)
    return m.reset_index(drop=True), o


# ---------------------------------------------------------------- orchestration
def ingest(config_path="config/leagues.toml", data_dir="data", delay: float = 1.0,
           divisions: list[str] | None = None, seasons: list[str] | None = None,
           offline: bool = False):
    """Download (unless offline) + parse everything; write Parquet and DuckDB. Returns summary."""
    from fedge import store

    cfg = load_config(config_path)
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw" / "football_data"
    divs = divisions or cfg["divisions"]
    seas = seasons or cfg["seasons"]["all"]
    current = cfg["seasons"]["current"]
    session = requests.Session()
    m_parts, o_parts, status = [], [], []
    for season in seas:
        for div in divs:
            if offline:
                p = raw_dir / season.replace("/", "-") / f"{div}.csv"
                res = FetchResult(div, season, p if p.exists() else None, "cached")
            else:
                res = fetch_one(div, season, raw_dir, current, session, delay)
            status.append(res)
            if res.path is None:
                continue
            raw = read_raw_csv(res.path)
            if raw.empty or "HomeTeam" not in raw.columns:
                continue
            m, o = parse_file(raw, season, div)
            m_parts.append(m)
            o_parts.append(o)
            log.info("%s %s: %s, %d matches, %d odds", div, season, res.status, len(m), len(o))
    matches = pd.concat(m_parts, ignore_index=True)
    odds = pd.concat(o_parts, ignore_index=True)
    matches = matches.sort_values(["kickoff_utc", "div", "match_id"], kind="stable")
    matches = matches.reset_index(drop=True)
    odds = odds.sort_values(
        ["match_id", "market", "phase", "bookmaker", "selection"], kind="stable"
    ).reset_index(drop=True)
    store.write_tables(matches, odds, data_dir)
    return matches, odds, status

