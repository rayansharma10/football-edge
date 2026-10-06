"""football-data.co.uk fixture files: download and parse into long-format price snapshots.

Two files, both refreshed roughly Friday and Tuesday afternoon UK time:

``fixtures.csv``
    Upcoming (and just-played) matches of the English/Scottish divisions in ``config/leagues.toml``
    with pre-closing prices and, later, closing prices. Columns: ``Div, Date (dd/mm/yyyy), Time
    (UK local)`` and the usual bookmaker blocks (``B365``, ``BFD``, ``BV``, ``BW``, ``PP``, ``SKB``,
    ``Max``, ``Avg``, ``BFE``) for 1X2, over/under 2.5 and Asian handicaps, in a pre-closing block
    and a closing (``...C...``) block.
``new_league_fixtures.csv``
    The same idea for the 1X2-only "extra" leagues (Argentina, Brazil, Ireland, ...): ``Country,
    League, Date, Time, Home, Away`` and 1X2 prices only. These leagues are **not** in the model's
    training universe, so they are counted and reported but cannot be scored.

Price sources used here, in priority order: **BFE** (Betfair Exchange), then ``Avg`` (market
average), then ``Max`` (best of market). Pinnacle (``PP``) is deliberately excluded: it is stale
from 2025-07-23 (AGENTS.md rule 9). A source only qualifies for a fixture when *every* selection of
that market is quoted, because the de-vig needs the complete book.

The kickoff is ``Date`` + ``Time`` interpreted as Europe/London and converted to UTC by
:func:`fedge.ingest.football_data.uk_local_to_utc`; ``match_key`` is built with the same
``div|YYYY-MM-DD|home|away`` format as :func:`fedge.ingest.football_data.make_match_id`, so ledger
rows join straight onto the ingested matches/odds tables.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from fedge.paper.settle import SELS

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
NEW_FIXTURES_URL = "https://www.football-data.co.uk/new_league_fixtures.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
SOURCES = ("BFE", "Avg", "Max")  # priority order; Pinnacle excluded (stale)

_COLS: dict[str, dict[str, dict[str, str]]] = {
    "BFE": {
        "1x2": {"H": "BFEH", "D": "BFED", "A": "BFEA"},
        "ou25": {"over": "BFE>2.5", "under": "BFE<2.5"},
    },
    "Avg": {
        "1x2": {"H": "AvgH", "D": "AvgD", "A": "AvgA"},
        "ou25": {"over": "Avg>2.5", "under": "Avg<2.5"},
    },
    "Max": {
        "1x2": {"H": "MaxH", "D": "MaxD", "A": "MaxA"},
        "ou25": {"over": "Max>2.5", "under": "Max<2.5"},
    },
}


def column(source: str, market: str, selection: str, phase: str = "pre") -> str:
    """football-data column name for (source, market, selection, phase); ``close`` inserts ``C``."""
    base = _COLS[source][market][selection]
    if phase == "pre":
        return base
    if phase != "close":
        raise ValueError(f"unknown phase {phase!r}")
    return source + "C" + base[len(source) :]


def download(url: str, dest: Path | str, session=None, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` (browser-like UA, follows redirects) and return the path."""
    import requests

    session = session or requests.Session()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=timeout, allow_redirects=True
    )
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def fetch_fixture_files(
    data_dir: Path | str = "data", session=None, delay: float = 1.0, refresh: bool = True
) -> tuple[Path, Path]:
    """Download both fixture files into ``data/raw/fixtures/`` (cached if ``refresh`` is False)."""
    d = Path(data_dir) / "raw" / "fixtures"
    out = []
    for i, (url, name) in enumerate(
        ((FIXTURES_URL, "fixtures.csv"), (NEW_FIXTURES_URL, "new_league_fixtures.csv"))
    ):
        path = d / name
        if refresh or not path.exists():
            if i:
                time.sleep(delay)
            download(url, path, session)
        out.append(path)
    return out[0], out[1]


def read_fixture_csv(path: Path | str) -> pd.DataFrame:
    """Read a fixture CSV as strings (utf-8-sig, latin-1 fallback), stripping column names."""
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    from io import StringIO

    df = pd.read_csv(StringIO(text), dtype=str, skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    """Fixture-level rows: ``div, date, time, home, away, kickoff_utc, match_key``.

    ``date`` is the UK-local match date as ``YYYY-MM-DD`` (the form :func:`make_match_id` expects),
    ``kickoff_utc`` is tz-aware UTC. Rows without a parseable date/home/away are dropped.
    """
    from fedge.ingest.football_data import make_match_id, uk_local_to_utc

    if raw.empty:
        return _empty_fixtures()
    df = raw.copy()
    df["Div"] = df["Div"].astype("string").str.strip()
    dates = df["Date"].astype("string").str.strip()
    local = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    hm = df["HomeTeam"].astype("string").str.strip()
    aw = df["AwayTeam"].astype("string").str.strip()
    ok = local.notna() & hm.notna() & (hm != "") & aw.notna() & (aw != "") & df["Div"].notna()
    df, local, hm, aw = df[ok], local[ok], hm[ok], aw[ok]
    t = df["Time"].astype("string").str.strip()
    t_ok = t.str.match(r"^\d{1,2}:\d{2}$").fillna(False).astype(bool)
    hh = pd.to_numeric(t.str.split(":").str[0].where(t_ok), errors="coerce").fillna(23)
    mm = pd.to_numeric(t.str.split(":").str[1].where(t_ok), errors="coerce").fillna(59)
    local_dt = local.dt.normalize() + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
    date_str = local.dt.strftime("%Y-%m-%d")
    out = pd.DataFrame(
        {
            "div": df["Div"].to_numpy(),
            "date": date_str.to_numpy(),
            "time": t.where(t_ok, "").to_numpy(),
            "home": hm.to_numpy(),
            "away": aw.to_numpy(),
            "kickoff_utc": uk_local_to_utc(local_dt.reset_index(drop=True)).to_numpy(),
        }
    )
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    out["match_key"] = [
        make_match_id(d, s, h, a)
        for d, s, h, a in zip(out["div"], out["date"], out["home"], out["away"], strict=True)
    ]
    return out.reset_index(drop=True)


def market_rows(
    raw: pd.DataFrame, fixtures: pd.DataFrame, phase: str = "pre", markets=("1x2", "ou25")
) -> pd.DataFrame:
    """Long-format price rows: one per (fixture, market, selection) with the best available source.

    Columns ``match_key, market, selection, price, price_source``; a (fixture, market) is skipped
    entirely when no source quotes every selection of that market.
    """
    if fixtures.empty:
        return _empty_rows()
    raw = raw.loc[fixtures.index]
    parts = []
    for market in markets:
        sels = list(SELS[market])
        chosen = pd.Series(index=fixtures.index, dtype=object)
        prices = {}
        for src in SOURCES:
            cols = [column(src, market, s, phase) for s in sels]
            if not set(cols).issubset(raw.columns):
                continue
            vals = raw[cols].apply(pd.to_numeric, errors="coerce")
            complete = vals.notna().all(axis=1) & (vals > 1.0).all(axis=1)
            if not complete.any():
                continue
            fill = complete & chosen.isna()
            if fill.any():
                chosen.loc[fill] = src
                for s, c in zip(sels, cols, strict=True):
                    prices.setdefault(s, pd.Series(float("nan"), index=fixtures.index, dtype=float))
                    prices[s].loc[fill] = vals.loc[fill, c].astype(float)
        keep = chosen.notna()
        if not keep.any():
            continue
        parts.append(_stack(fixtures.loc[keep], market, sels, prices, chosen.loc[keep]))
    if not parts:
        return _empty_rows()
    return pd.concat(parts, ignore_index=True)


def _stack(fix, market: str, sels, prices: dict, source: pd.Series) -> pd.DataFrame:
    """One long block: every selection of one market for the kept fixtures."""
    frames = []
    for s in sels:
        frames.append(
            pd.DataFrame(
                {
                    "match_key": fix["match_key"].to_numpy(),
                    "market": market,
                    "selection": s,
                    "price": prices[s].loc[fix.index].to_numpy(dtype=float),
                    "price_source": source.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def parse_new_league(raw: pd.DataFrame) -> pd.DataFrame:
    """Fixture-level rows of ``new_league_fixtures.csv`` (leagues outside the model's universe)."""
    if raw.empty:
        return pd.DataFrame(
            columns=["country", "league", "date", "home", "away", "kickoff_utc", "has_bfe_1x2"]
        )
    from fedge.ingest.football_data import uk_local_to_utc

    df = raw.copy()
    dates = df["Date"].astype("string").str.strip()
    local = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    hm = df["Home"].astype("string").str.strip()
    aw = df["Away"].astype("string").str.strip()
    ok = local.notna() & hm.notna() & aw.notna()
    df, local, hm, aw = df[ok], local[ok], hm[ok], aw[ok]
    t = df["Time"].astype("string").str.strip()
    t_ok = t.str.match(r"^\d{1,2}:\d{2}$").fillna(False).astype(bool)
    hh = pd.to_numeric(t.str.split(":").str[0].where(t_ok), errors="coerce").fillna(23)
    mm = pd.to_numeric(t.str.split(":").str[1].where(t_ok), errors="coerce").fillna(59)
    local_dt = local.dt.normalize() + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
    has_bfe = (
        df[["BFEH", "BFED", "BFEA"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    out = pd.DataFrame(
        {
            "country": df["Country"].astype("string").str.strip().to_numpy(),
            "league": df["League"].astype("string").str.strip().to_numpy(),
            "date": local.dt.strftime("%Y-%m-%d").to_numpy(),
            "home": hm.to_numpy(),
            "away": aw.to_numpy(),
            "kickoff_utc": uk_local_to_utc(local_dt.reset_index(drop=True)).to_numpy(),
            "has_bfe_1x2": has_bfe.to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def _empty_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["div", "date", "time", "home", "away", "kickoff_utc", "match_key"]
    )


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_key", "market", "selection", "price", "price_source"])
