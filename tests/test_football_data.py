from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fedge.ingest.football_data import (
    make_match_id,
    parse_file,
    read_raw_csv,
    season_code,
    uk_local_to_utc,
)
from fedge.schemas import matches_schema, odds_schema

FIX = Path(__file__).parent / "fixtures" / "football_data_sample.csv"


@pytest.fixture(scope="module")
def parsed():
    raw = read_raw_csv(FIX)
    return parse_file(raw, "2025/26", "E0")


def test_season_code():
    assert season_code("2024/25") == "2425"
    assert season_code("2012/13") == "1213"


def test_uk_to_utc_bst_and_gmt():
    local = pd.Series(
        pd.to_datetime(["2025-08-15 20:00", "2025-12-13 15:00", "2025-03-30 12:00",
                        "2025-03-29 12:00", "2025-10-26 12:00", "2025-10-25 12:00"])
    )
    utc = uk_local_to_utc(local)
    got = [t.strftime("%Y-%m-%d %H:%M") for t in utc]
    assert got == ["2025-08-15 19:00",  # BST (UTC+1)
                   "2025-12-13 15:00",  # GMT
                   "2025-03-30 11:00",  # clocks went forward that morning
                   "2025-03-29 12:00",  # day before: GMT
                   "2025-10-26 12:00",  # clocks went back that morning: GMT
                   "2025-10-25 11:00"]  # BST
    assert str(utc.dt.tz) == "UTC"


def test_parse_matches(parsed):
    m, _ = parsed
    assert len(m) == 4
    matches_schema.validate(m)
    liv = m[m.home == "Liverpool"].iloc[0]
    assert liv.kickoff_utc == pd.Timestamp("2025-08-15 19:00", tz="UTC")
    assert not liv.time_unknown
    assert (liv.FTHG, liv.FTAG, liv.FTR, liv.HTHG) == (4, 2, "H", 1)
    assert liv.HxG == pytest.approx(2.1)
    assert liv.match_id == make_match_id("E0", "2025-08-15", "Liverpool", "Bournemouth")
    ars = m[m.home == "Arsenal"].iloc[0]
    assert ars.kickoff_utc == pd.Timestamp("2025-12-13 15:00", tz="UTC")
    assert pd.isna(ars.HxG)


def test_missing_time_marked_unknown(parsed):
    m, _ = parsed
    ev = m[m.home == "Everton"].iloc[0]
    assert ev.time_unknown
    # conservative: 23:59 UK local (BST on 26 Oct = GMT after clock change -> 23:59 UTC)
    assert ev.kickoff_utc == pd.Timestamp("2025-10-26 23:59", tz="UTC")
    assert m.time_unknown.sum() == 1


def test_match_id_stable_and_unique(parsed):
    m, _ = parsed
    assert m.match_id.is_unique
    a = make_match_id("E0", "2025-08-15", "A", "B")
    assert a == make_match_id("E0", "2025-08-15", "A", "B")
    assert a != make_match_id("E1", "2025-08-15", "A", "B")


def test_odds_long_format(parsed):
    m, o = parsed
    odds_schema.validate(o)
    liv = m[m.home == "Liverpool"].match_id.iloc[0]
    lo = o[o.match_id == liv]

    def price(bk, mkt, sel, ph):
        r = lo[(lo.bookmaker == bk) & (lo.market == mkt) & (lo.selection == sel) & (lo.phase == ph)]
        assert len(r) == 1
        return r.iloc[0]

    assert price("PS", "1x2", "H", "pre").price == 1.3
    assert price("PS", "1x2", "A", "close").price == 9.0
    assert price("Avg", "1x2", "D", "close").price == 6.0
    assert price("PS", "ou25", "over", "pre").price == 1.37
    assert price("PS", "ou25", "under", "close").price == 3.2
    ah_pre, ah_close = price("PS", "ah", "H", "pre"), price("PS", "ah", "H", "close")
    assert (ah_pre.line, ah_pre.price) == (-1.5, 1.9)
    assert (ah_close.line, ah_close.price) == (-1.75, 2.07)


def test_available_at_semantics(parsed):
    m, o = parsed
    ko = m.set_index("match_id").kickoff_utc
    j = o.assign(kickoff=o.match_id.map(ko))
    pre = j[j.phase == "pre"]
    close = j[j.phase == "close"]
    assert (pre.available_at == pre.kickoff - pd.Timedelta(hours=24)).all()
    assert (close.available_at == close.kickoff).all()


def test_pinnacle_stale_flag(parsed):
    m, o = parsed
    ko = m.set_index("match_id").kickoff_utc
    ps = o[o.bookmaker == "PS"]
    assert ps.stale.all()  # every fixture row is after 2025-07-23
    assert not o[o.bookmaker != "PS"].stale.any()
    old = parse_file(
        read_raw_csv(FIX).assign(Date=lambda d: d.Date.str.replace("2025", "2024")), "2024/25", "E0"
    )[1]
    assert not old[old.bookmaker == "PS"].stale.any()
    assert ko.min() > pd.Timestamp("2025-07-23", tz="UTC")


def test_missing_prices_not_emitted(parsed):
    m, o = parsed
    ars = m[m.home == "Arsenal"].match_id.iloc[0]
    sub = o[o.match_id == ars]
    assert not ((sub.bookmaker == "PS") & (sub.phase == "pre")).any()
    assert ((sub.bookmaker == "PS") & (sub.phase == "close")).sum() == 3


def test_latin1_fallback(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes("Div,Date,HomeTeam,AwayTeam\nE0,01/01/2020,Caf\xe9,Bar\n".encode("latin-1"))
    df = read_raw_csv(p)
    assert df.HomeTeam.iloc[0] == "Café"
