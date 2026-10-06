"""Full-season fixture schedule for the desk leagues (Premier League + La Liga), display-only.

football-data.co.uk's ``fixtures.csv`` only lists the next few days (and often has no E0/SP1 rows at
all between rounds), so the dashboard's "upcoming" list is fed from a full-season schedule:

1. **fixturedownload.com** JSON feed (primary; free, no key, UTC kickoffs): one file per league.
2. **openfootball** ``football.json`` on GitHub (fallback; free, no key, local-time kickoffs,
   some matches have no time).
3. **Last good cache** under ``data/raw/schedule/`` when both fail (flagged ``stale``).

football-data.org (key ``FOOTBALL_DATA_ORG_KEY``, 10 req/min) was evaluated but is not needed: the
two no-key sources cover the whole season, so no API key is required.

Team names are mapped onto the football-data names used everywhere else in the repo (so the
ratings/features and the ``match_key`` of priced fixtures line up) via the football-data aliases in
``config/team_aliases.csv`` plus ``config/schedule_aliases.csv`` (source-specific spellings). Names
that cannot be mapped are logged loudly and listed in the result; their fixtures are dropped.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fedge.teams import canon_map

log = logging.getLogger("fedge.schedule")

ALIASES_CSV = Path(__file__).resolve().parents[3] / "config" / "schedule_aliases.csv"
UA = "Mozilla/5.0 (compatible; football-edge/0.1; personal research)"
FIXTUREDOWNLOAD = "https://fixturedownload.com/feed/json/{slug}-{year}"
OPENFOOTBALL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
)
FD_SLUG = {"E0": "epl", "SP1": "la-liga"}
OF_CODE = {"E0": "en.1", "SP1": "es.1"}
LOCAL_TZ = {"E0": "Europe/London", "SP1": "Europe/Madrid"}
DEFAULT_LOCAL_TIME = "12:00"  # openfootball matches without a time
RETRIES = 3
RETRY_SLEEP = 2.0
MIN_ROWS = 300  # a 20-team league has 380; far fewer means a truncated/failed feed
COLUMNS = ["div", "home", "away", "kickoff_utc", "round", "played", "home_goals", "away_goals"]


@dataclass
class ScheduleResult:
    """The schedule plus how it was obtained."""

    fixtures: pd.DataFrame
    sources: dict[str, str] = field(default_factory=dict)  # div -> source (or cache:<source>)
    fetched_utc: dict[str, str] = field(default_factory=dict)  # div -> ISO time of the good fetch
    stale: bool = False
    warnings: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)

    @property
    def meta(self) -> dict[str, str]:
        times = sorted(self.fetched_utc.values())
        return {
            "schedule_sources": json.dumps(self.sources, sort_keys=True),
            "schedule_fetched_utc": times[0] if times else "",
            "schedule_stale": "1" if self.stale else "0",
            "schedule_warnings": json.dumps(self.warnings),
            "schedule_unmapped": json.dumps(self.unmapped),
        }


def season_start_year(now=None) -> int:
    ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return ts.year if ts.month >= 7 else ts.year - 1


# ----------------------------------------------------------------------------- name mapping
def load_overrides(path: Path | str = ALIASES_CSV) -> dict[tuple[str, str, str], str]:
    """``{(source, div, name): football_data_name}`` from ``config/schedule_aliases.csv``."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return {
        (s, d, n): fd
        for s, d, n, fd in zip(
            df["source"], df["div"], df["name"], df["football_data"], strict=True
        )
    }


def map_team(name: str, source: str, div: str, overrides=None, known=None) -> str | None:
    """football-data name for a source's team ``name`` (None when it cannot be mapped).

    Order: explicit override, then the name already being a known football-data name.
    """
    overrides = load_overrides() if overrides is None else overrides
    known = set(canon_map("football_data")) if known is None else known
    name = str(name).strip()
    hit = overrides.get((source, div, name))
    if hit is not None:
        return hit
    return name if name in known else None


def _finish(rows: list[dict], source: str, div: str, unmapped: set[str], overrides, known):
    out = []
    for r in rows:
        h = map_team(r["home"], source, div, overrides, known)
        a = map_team(r["away"], source, div, overrides, known)
        for raw, mapped in ((r["home"], h), (r["away"], a)):
            if mapped is None:
                unmapped.add(f"{source}/{div}: {raw}")
        if h is None or a is None:
            continue
        out.append({**r, "home": h, "away": a})
    df = pd.DataFrame(out, columns=COLUMNS)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    for c in ("home_goals", "away_goals"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------------------------------------------------------- parsers
def parse_fixturedownload(payload: list[dict], div: str) -> list[dict]:
    """Raw rows (unmapped names) from a fixturedownload.com JSON feed."""
    rows = []
    for m in payload:
        ts = pd.to_datetime(m.get("DateUtc"), utc=True, errors="coerce")
        h, a = m.get("HomeTeam"), m.get("AwayTeam")
        if pd.isna(ts) or not h or not a:
            continue
        rows.append({
            "div": div, "home": str(h).strip(), "away": str(a).strip(), "kickoff_utc": ts,
            "round": int(m["RoundNumber"]) if m.get("RoundNumber") is not None else None,
            "played": m.get("HomeTeamScore") is not None and m.get("AwayTeamScore") is not None,
            "home_goals": m.get("HomeTeamScore"), "away_goals": m.get("AwayTeamScore"),
        })
    return rows


def parse_openfootball(payload: dict, div: str) -> list[dict]:
    """Raw rows (unmapped names) from an openfootball ``football.json`` season file.

    Kickoffs are local to the league (``LOCAL_TZ``); a missing time becomes 12:00 local.
    """
    tz = LOCAL_TZ[div]
    rows = []
    for m in payload.get("matches", []):
        date, h, a = m.get("date"), m.get("team1"), m.get("team2")
        if not date or not h or not a:
            continue
        t = m.get("time") or DEFAULT_LOCAL_TIME
        try:
            local = pd.Timestamp(f"{date} {t}")
            ts = local.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
            ts = ts.tz_convert("UTC")
        except (ValueError, TypeError):
            continue
        digits = "".join(c for c in str(m.get("round", "")) if c.isdigit())
        ft = (m.get("score") or {}).get("ft")
        has = isinstance(ft, list | tuple) and len(ft) == 2
        rows.append({
            "div": div, "home": str(h).strip(), "away": str(a).strip(), "kickoff_utc": ts,
            "round": int(digits) if digits else None, "played": bool(ft),
            "home_goals": ft[0] if has else None, "away_goals": ft[1] if has else None,
        })
    return rows


# ----------------------------------------------------------------------------- fetch + cache
def _get_json(url: str, session=None, timeout: int = 30):
    import requests

    session = session or requests.Session()
    for attempt in range(RETRIES):
        try:
            r = session.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP * (attempt + 1))


def _cache_path(data_dir, div: str, source: str) -> Path:
    return Path(data_dir) / "raw" / "schedule" / f"{div}_{source}.json"


def _attempts(div: str, year: int):
    return (
        ("fixturedownload", FIXTUREDOWNLOAD.format(slug=FD_SLUG[div], year=year),
         lambda p: parse_fixturedownload(p, div)),
        ("openfootball",
         OPENFOOTBALL.format(season=f"{year}-{str(year + 1)[2:]}", code=OF_CODE[div]),
         lambda p: parse_openfootball(p, div)),
    )


def fetch_division(
    div: str, data_dir="data", session=None, now=None, refresh: bool = True,
    overrides=None, known=None, unmapped: set[str] | None = None,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    """One division's schedule: ``(frame, source, fetched_utc, warnings)``.

    Tries fixturedownload, then openfootball, then the last good cache of either. A fetch is only
    accepted (and cached) when it parses to a plausible full season.
    """
    unmapped = set() if unmapped is None else unmapped
    attempts = _attempts(div, season_start_year(now))
    warnings: list[str] = []
    if refresh:
        for source, url, parse in attempts:
            try:
                payload = _get_json(url, session)
                raw_rows = parse(payload)
                if len(raw_rows) < MIN_ROWS:
                    raise ValueError(f"only {len(raw_rows)} fixtures parsed (expected ~380)")
                df = _finish(raw_rows, source, div, unmapped, overrides, known)
                path = _cache_path(data_dir, div, source)
                path.parent.mkdir(parents=True, exist_ok=True)
                fetched = datetime.now(UTC).isoformat(timespec="seconds")
                path.write_text(json.dumps({"fetched_utc": fetched, "payload": payload}), "utf-8")
                return df, source, fetched, warnings
            except Exception as exc:  # network / format: fall through to the next source
                msg = f"{div}: {source} failed ({type(exc).__name__}: {exc})"
                log.warning("schedule: %s", msg)
                warnings.append(msg)
    best = None  # last good cache, newest first
    for source, _url, parse in attempts:
        p = _cache_path(data_dir, div, source)
        if p.exists():
            blob = json.loads(p.read_text("utf-8"))
            if best is None or blob["fetched_utc"] > best[1]["fetched_utc"]:
                best = (source, blob, parse)
    if best is None:
        why = "; ".join(warnings) if warnings else "refresh off"
        raise RuntimeError(f"{div}: no schedule available (no cache; {why})")
    source, blob, parse = best
    df = _finish(parse(blob["payload"]), source, div, unmapped, overrides, known)
    warnings.append(f"{div}: using cached {source} schedule from {blob['fetched_utc']} (stale)")
    return df, "cache:" + source, blob["fetched_utc"], warnings


def load_schedule(
    divs=("E0", "SP1"), data_dir="data", session=None, now=None, refresh: bool = True,
    delay: float = 1.0,
) -> ScheduleResult:
    """Full-season schedule for ``divs``; never raises (the picks run must keep working).

    A division that cannot be loaded at all is skipped with a warning and ``stale`` is set;
    unmapped team names are logged at ERROR level and listed in ``unmapped``.
    """
    overrides, known = load_overrides(), set(canon_map("football_data"))
    unmapped: set[str] = set()
    res = ScheduleResult(fixtures=pd.DataFrame(columns=COLUMNS))
    frames = []
    for i, div in enumerate(divs):
        if i and refresh:
            time.sleep(delay)
        try:
            df, source, fetched, warns = fetch_division(
                div, data_dir, session, now, refresh, overrides, known, unmapped)
        except Exception as exc:
            res.warnings.append(f"{div}: schedule unavailable ({exc})")
            res.stale = True
            continue
        frames.append(df)
        res.sources[div] = source
        res.fetched_utc[div] = fetched
        res.warnings += warns
        if source.startswith("cache:"):
            res.stale = True
    if frames:
        res.fixtures = pd.concat(frames, ignore_index=True)
    res.unmapped = sorted(unmapped)
    for u in res.unmapped:
        log.error("schedule: UNMAPPED team name %s (add it to config/schedule_aliases.csv)", u)
    return res


def upcoming(sched: pd.DataFrame, now) -> pd.DataFrame:
    """Rows with no result yet and a kickoff after ``now``."""
    now = pd.Timestamp(now)
    return sched.loc[~sched["played"] & (sched["kickoff_utc"] > now)].reset_index(drop=True)


def to_fixtures(sched: pd.DataFrame) -> pd.DataFrame:
    """Schedule rows in the :func:`fedge.paper.fixtures.parse_fixtures` layout (with match_key).

    ``date`` is the UK-local match date (the ``make_match_id`` convention), as for priced fixtures.
    """
    from fedge.ingest.football_data import make_match_id

    cols = ["div", "date", "time", "time_unknown", "home", "away", "kickoff_utc", "match_key"]
    if sched.empty:
        return pd.DataFrame(columns=cols)
    ko = pd.to_datetime(sched["kickoff_utc"], utc=True)
    local = ko.dt.tz_convert("Europe/London")
    out = pd.DataFrame({
        "div": sched["div"].to_numpy(),
        "date": local.dt.strftime("%Y-%m-%d").to_numpy(),
        "time": local.dt.strftime("%H:%M").to_numpy(),
        "time_unknown": False,
        "home": sched["home"].to_numpy(), "away": sched["away"].to_numpy(),
        "kickoff_utc": ko.to_numpy(),
    })
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    out["match_key"] = [
        make_match_id(d, s, h, a)
        for d, s, h, a in zip(out["div"], out["date"], out["home"], out["away"], strict=True)
    ]
    return out


def merge_priced(sched_fixtures: pd.DataFrame, priced: pd.DataFrame) -> pd.DataFrame:
    """Combine schedule fixtures with football-data's priced ones.

    Within a season a (div, home, away) pair occurs once, so it is the join key; a priced fixture
    replaces its schedule twin (its kickoff and ``match_key`` are the ones the price rows use).
    Priced fixtures absent from the schedule are kept.
    """
    if priced.empty:
        return sched_fixtures.reset_index(drop=True)
    cols = list(sched_fixtures.columns)
    pk = set(zip(priced["div"], priced["home"], priced["away"], strict=True))
    mask = [
        k not in pk
        for k in zip(sched_fixtures["div"], sched_fixtures["home"], sched_fixtures["away"],
                     strict=True)
    ]
    out = pd.concat([priced[cols], sched_fixtures.loc[mask]], ignore_index=True)
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    return out.sort_values(["kickoff_utc", "home"], kind="mergesort").reset_index(drop=True)
