from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fedge.ingest.understat import (
    FIRST_YEAR,
    LEAGUES,
    build_match_xg,
    load_understat,
    parse_fixtures,
)
from fedge.teams import SOURCES, canon_map, load_aliases

ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "data" / "interim" / "matches.parquet"
RAW_US = ROOT / "data" / "raw" / "understat"


def test_alias_table_shape():
    df = load_aliases()
    assert list(df.columns) == ["canonical_id", "canonical_name", *SOURCES]
    assert df["canonical_id"].is_unique
    assert (df["canonical_id"] != "").all()


@pytest.mark.parametrize("source", ["football_data", "understat", "betfair"])
def test_each_name_maps_to_one_canonical_id(source):
    cm = canon_map(source)  # raises if a name maps to two ids
    if source != "betfair":  # betfair column is a placeholder until Phase 5
        assert cm


def test_conflicting_alias_raises(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("canonical_id,canonical_name,football_data,understat,betfair\n"
                 "a,A,X,,\nb,B,X,,\n")
    with pytest.raises(ValueError):
        canon_map("football_data", p)


@pytest.mark.skipif(not MATCHES.exists(), reason="data/ not ingested")
def test_every_match_team_maps_uniquely():
    m = pd.read_parquet(MATCHES)
    cm = canon_map("football_data")
    teams = set(m["home"]) | set(m["away"])
    assert not (teams - set(cm)), sorted(teams - set(cm))


@pytest.mark.skipif(not RAW_US.exists(), reason="Understat cache not present")
def test_understat_names_map_all_top5_seasons():
    us = load_understat(RAW_US)
    assert set(us["div"]) == set(LEAGUES)
    assert int(us["season"].str[:4].astype(int).min()) == FIRST_YEAR
    cm = canon_map("understat")
    names = set(us["us_home"]) | set(us["us_away"])
    assert not (names - set(cm)), sorted(names - set(cm))


def _raw(events):
    return {"dates": events}


def _ev(i, dt, h, a, gh, ga, xh, xa, res=True):
    return {"id": i, "isResult": res, "datetime": dt, "h": {"title": h}, "a": {"title": a},
            "goals": {"h": str(gh), "a": str(ga)}, "xG": {"h": str(xh), "a": str(xa)}}


def test_parse_skips_unplayed():
    raw = _raw([_ev(1, "2024-08-17 15:00:00", "Fulham", "Everton", 1, 0, "1.2", "0.4"),
                _ev(2, "2030-01-01 15:00:00", "Fulham", "Everton", 0, 0, "0", "0", res=False)])
    df = parse_fixtures(raw, "E0", 2024)
    assert len(df) == 1 and df.loc[0, "xg_home"] == 1.2 and df.loc[0, "season"] == "2024/25"


def _matches(rows):
    return pd.DataFrame([{
        "match_id": mid, "div": "E0", "season": "2024/25",
        "kickoff_utc": pd.Timestamp(ko, tz="UTC"), "home": h, "away": a,
        "FTHG": gh, "FTAG": ga, "FTR": "H"} for mid, ko, h, a, gh, ga in rows])


def test_join_exact_and_fallback_and_available_at():
    m = _matches([("m1", "2024-08-17 14:00", "Man City", "Chelsea", 2, 1),
                  ("m2", "2024-08-18 14:00", "Wolves", "Arsenal", 0, 0),
                  ("m3", "2024-08-24 14:00", "Man United", "Fulham", 1, 1)])
    raw = _raw([_ev(10, "2024-08-17 15:00:00", "Manchester City", "Chelsea", 2, 1, 2.1, 0.9),
                # one day late: only the fixture fallback can match it
                _ev(11, "2024-08-19 15:00:00", "Wolverhampton Wanderers", "Arsenal", 0, 0, 0.3,
                    1.1),
                # date off but score differs: must not join
                _ev(12, "2024-08-25 15:00:00", "Manchester United", "Fulham", 3, 0, 1.0, 1.0)])
    us = parse_fixtures(raw, "E0", 2024)
    out = build_match_xg(m, us).set_index("match_id")
    assert set(out.index) == {"m1", "m2"}
    assert out.loc["m1", "match_key"] == "date" and out.loc["m2", "match_key"] == "fixture"
    assert out.loc["m1", "xg_home"] == 2.1
    assert out.loc["m1", "available_at"] == pd.Timestamp("2024-08-17 17:00", tz="UTC")
    assert out.index.is_unique
