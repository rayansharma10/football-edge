"""Full-season schedule: parsers, name mapping, fallback/caching, merge, price-free path."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fedge.ingest import schedule as sch
from fedge.paper import model_state
from fedge.predict import upcoming as pred

NOW = pd.Timestamp("2026-10-06T00:00:00Z")


# ----------------------------------------------------------------------------- parsers
def _fd_payload(n=380):
    teams = ["Arsenal", "Man Utd", "Spurs", "Chelsea"]
    out = []
    for i in range(n):
        out.append({
            "MatchNumber": i + 1, "RoundNumber": i // 10 + 1,
            "DateUtc": f"2026-11-{1 + i % 28:02d} 15:00:00Z",
            "HomeTeam": teams[i % 4], "AwayTeam": teams[(i + 1) % 4],
            "HomeTeamScore": 1 if i == 0 else None, "AwayTeamScore": 0 if i == 0 else None,
        })
    return out


def test_parse_fixturedownload_rows_and_played_flag():
    rows = sch.parse_fixturedownload(_fd_payload(5) + [{"DateUtc": None, "HomeTeam": "x"}], "E0")
    assert len(rows) == 5  # the malformed row is dropped
    assert rows[0]["played"] is True and rows[1]["played"] is False
    assert rows[0]["kickoff_utc"] == pd.Timestamp("2026-11-01T15:00:00Z")
    assert rows[0]["round"] == 1


def test_parse_openfootball_local_time_to_utc_and_missing_time():
    payload = {"matches": [
        {"round": "Matchday 3", "date": "2026-10-17", "time": "15:00",
         "team1": "Arsenal FC", "team2": "Chelsea FC"},
        {"round": "Matchday 38", "date": "2027-05-30", "team1": "CA Osasuna",
         "team2": "Valencia CF", "score": {"ft": [1, 0]}},
    ]}
    e0 = sch.parse_openfootball(payload, "E0")
    assert e0[0]["kickoff_utc"] == pd.Timestamp("2026-10-17T14:00:00Z")  # BST = UTC+1
    assert e0[0]["round"] == 3 and e0[0]["played"] is False
    sp = sch.parse_openfootball(payload, "SP1")
    assert sp[1]["kickoff_utc"] == pd.Timestamp("2027-05-30T10:00:00Z")  # 12:00 CEST default
    assert sp[1]["played"] is True


# ----------------------------------------------------------------------------- name mapping
def test_map_team_override_known_and_unmapped():
    ov = {("fixturedownload", "E0", "Man Utd"): "Man United"}
    known = {"Arsenal", "Man United"}
    assert sch.map_team("Man Utd", "fixturedownload", "E0", ov, known) == "Man United"
    assert sch.map_team("Arsenal", "fixturedownload", "E0", ov, known) == "Arsenal"
    assert sch.map_team("Nowhere FC", "fixturedownload", "E0", ov, known) is None


def test_shipped_aliases_map_every_team_of_the_real_feeds():
    """Every committed override points at a real football-data name."""
    from fedge.teams import canon_map

    known = set(canon_map("football_data"))
    ov = sch.load_overrides()
    assert ov, "config/schedule_aliases.csv is empty"
    bad = {k: v for k, v in ov.items() if v not in known}
    assert not bad, f"overrides pointing at unknown football-data names: {bad}"
    assert sch.map_team("Spurs", "fixturedownload", "E0") == "Tottenham"
    assert sch.map_team("R. Racing Club", "fixturedownload", "SP1") == "Santander"
    assert sch.map_team("Real Racing Club de Santander", "openfootball", "SP1") == "Santander"


def test_unmapped_names_are_listed_loudly_and_their_fixtures_dropped(caplog):
    payload = _fd_payload(380)
    payload[5]["HomeTeam"] = "Brand New FC"

    class Sess:
        def get(self, url, **kw):
            return _Resp(payload)

    with caplog.at_level("ERROR", logger="fedge.schedule"):
        res = sch.load_schedule(("E0",), data_dir=_tmp(), session=Sess(), now=NOW, delay=0)
    assert res.unmapped == ["fixturedownload/E0: Brand New FC"]
    assert len(res.fixtures) == 379
    assert any("UNMAPPED" in r.message for r in caplog.records)


# ----------------------------------------------------------------------------- fetch / fallback
class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


_TMP = []


def _tmp():
    import tempfile

    d = tempfile.mkdtemp()
    _TMP.append(d)
    return d


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sch, "RETRY_SLEEP", 0.0)


def _of_payload():
    teams = ["Arsenal FC", "Manchester United FC", "Tottenham Hotspur FC", "Chelsea FC"]
    return {"matches": [
        {"round": f"Matchday {i // 10 + 1}", "date": f"2026-11-{1 + i % 28:02d}", "time": "15:00",
         "team1": teams[i % 4], "team2": teams[(i + 1) % 4]}
        for i in range(380)
    ]}


def test_primary_source_cached_then_used_when_everything_fails():
    d = _tmp()

    class Good:
        def get(self, url, **kw):
            return _Resp(_fd_payload())

    res = sch.load_schedule(("E0",), data_dir=d, session=Good(), now=NOW, delay=0)
    assert res.sources == {"E0": "fixturedownload"} and not res.stale
    assert res.fixtures["home"].isin(["Man United", "Tottenham", "Arsenal", "Chelsea"]).all()
    assert (sch.Path(d) / "raw" / "schedule" / "E0_fixturedownload.json").exists()

    class Dead:
        def get(self, url, **kw):
            raise ConnectionError("offline")

    stale = sch.load_schedule(("E0",), data_dir=d, session=Dead(), now=NOW, delay=0)
    assert stale.stale and stale.sources == {"E0": "cache:fixturedownload"}
    assert len(stale.fixtures) == 380
    assert any("stale" in w for w in stale.warnings)
    assert json.loads(stale.meta["schedule_stale"]) == 1


def test_falls_back_to_openfootball_when_primary_fails():
    class Sess:
        def get(self, url, **kw):
            if "fixturedownload" in url:
                return _Resp(None, 503)
            return _Resp(_of_payload())

    res = sch.load_schedule(("E0",), data_dir=_tmp(), session=Sess(), now=NOW, delay=0)
    assert res.sources == {"E0": "openfootball"} and not res.stale
    assert any("fixturedownload failed" in w for w in res.warnings)
    assert set(res.fixtures["home"]) == {"Arsenal", "Man United", "Tottenham", "Chelsea"}


def test_truncated_feed_is_rejected_not_cached():
    d = _tmp()

    class Sess:
        def get(self, url, **kw):
            return _Resp(_fd_payload(12) if "fixturedownload" in url else _of_payload())

    res = sch.load_schedule(("E0",), data_dir=d, session=Sess(), now=NOW, delay=0)
    assert res.sources == {"E0": "openfootball"}
    assert not (sch.Path(d) / "raw" / "schedule" / "E0_fixturedownload.json").exists()


def test_no_source_and_no_cache_degrades_without_raising():
    class Dead:
        def get(self, url, **kw):
            raise ConnectionError("offline")

    res = sch.load_schedule(("E0",), data_dir=_tmp(), session=Dead(), now=NOW, delay=0)
    assert res.fixtures.empty and res.stale
    assert any("schedule unavailable" in w for w in res.warnings)


# ----------------------------------------------------------------------------- fixtures + merge
def _sched():
    return pd.DataFrame({
        "div": ["E0", "E0", "E0"], "home": ["Arsenal", "Chelsea", "Everton"],
        "away": ["Chelsea", "Arsenal", "Fulham"],
        "kickoff_utc": pd.to_datetime(
            ["2026-10-05T14:00:00Z", "2026-12-20T15:00:00Z", "2026-10-10T14:00:00Z"], utc=True),
        "round": [1, 2, 3], "played": [True, False, False],
    })


def test_upcoming_drops_played_and_past_and_to_fixtures_builds_match_keys():
    up = sch.upcoming(_sched(), NOW + pd.Timedelta(days=1))
    assert list(up["home"]) == ["Everton", "Chelsea"] or set(up["home"]) == {"Everton", "Chelsea"}
    fx = sch.to_fixtures(up)
    from fedge.ingest.football_data import make_match_id

    row = fx.loc[fx["home"] == "Chelsea"].iloc[0]
    assert row["date"] == "2026-12-20"
    assert row["match_key"] == make_match_id("E0", "2026-12-20", "Chelsea", "Arsenal")
    assert fx["kickoff_utc"].dt.tz is not None


def test_merge_priced_replaces_schedule_twin_and_keeps_its_match_key():
    fx = sch.to_fixtures(sch.upcoming(_sched(), NOW))
    priced = fx.loc[fx["home"] == "Everton"].copy()
    priced["match_key"] = "PRICED"
    priced["kickoff_utc"] = priced["kickoff_utc"] + pd.Timedelta(hours=3)
    out = sch.merge_priced(fx, priced)
    assert len(out) == len(fx)
    ev = out.loc[out["home"] == "Everton"].iloc[0]
    assert ev["match_key"] == "PRICED" and ev["kickoff_utc"] == priced["kickoff_utc"].iloc[0]
    assert list(out["kickoff_utc"]) == sorted(out["kickoff_utc"])
    # no priced rows: the schedule stands alone
    assert len(sch.merge_priced(fx, priced.iloc[0:0])) == len(fx)


# ----------------------------------------------------------------------------- price-free path
def _played_e0():
    teams = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    rows = []
    for i in range(24):
        h, a = teams[i % 4], teams[(i + 1 + i // 4) % 4]
        if h == a:
            a = teams[(teams.index(h) + 1) % 4]
        rows.append({
            "match_id": f"m{i}", "div": "E0", "season": "2026/27",
            "kickoff_utc": pd.Timestamp("2026-08-15T14:00:00Z") + pd.Timedelta(days=3 * i),
            "home": h, "away": a, "FTHG": float(i % 3), "FTAG": float((i + 1) % 2), "FTR": "H",
        })
    return pd.DataFrame(rows)


def _future_fixtures(kos):
    teams = [("Arsenal", "Chelsea"), ("Everton", "Fulham"), ("Chelsea", "Everton"),
             ("Fulham", "Arsenal")]
    return pd.DataFrame([
        {"match_key": f"f{i}", "div": "E0", "date": "x", "time": "x", "time_unknown": False,
         "home": teams[i % 4][0], "away": teams[i % 4][1], "kickoff_utc": pd.Timestamp(k)}
        for i, k in enumerate(kos)
    ])


def test_batched_fixture_features_equal_one_at_a_time():
    played = _played_e0()
    codes = model_state.div_code_map(played)
    fx = _future_fixtures(["2027-01-09T15:00:00Z"] * 4)
    batch = model_state.fixture_features(played, fx, codes)
    for i in range(4):
        one = model_state.fixture_features(played, fx.iloc[[i]], codes)
        pd.testing.assert_frame_equal(batch.loc[[f"f{i}"]], one, check_exact=False, rtol=1e-12)


def test_far_future_fixtures_get_a_normal_rest_gap_near_ones_keep_their_date():
    played = _played_e0()
    last = played["kickoff_utc"].max()
    far = _future_fixtures(["2027-05-23T15:00:00Z"])
    capped = model_state.feature_asof_fixtures(far, NOW, played)
    assert capped["kickoff_utc"].iloc[0] == last + pd.Timedelta(days=7)
    assert far["kickoff_utc"].iloc[0] == pd.Timestamp("2027-05-23T15:00:00Z")  # input untouched
    near = _future_fixtures(["2026-10-08T15:00:00Z"])
    assert model_state.feature_asof_fixtures(near, NOW, played)["kickoff_utc"].iloc[0] == near[
        "kickoff_utc"].iloc[0]
    codes = model_state.div_code_map(played)
    uncapped = model_state.fixture_features(played, far, codes)
    cap_x = model_state.fixture_features(played, capped, codes)
    assert uncapped["h_rest"].iloc[0] == 21.0  # the REST_CAP a months-long gap would give
    assert 7.0 <= cap_x["h_rest"].iloc[0] < 21.0  # a normal gap, not the cap


def test_price_free_scoring_passes_columns_in_training_order(monkeypatch, tmp_path):
    """Regression: LightGBM predicts positionally, so X must follow the training column order."""
    played = _played_e0()
    train_cols = model_state.live_feature_columns(
        ["elo_diff", "pi_diff", "ad_att_h", "ad_def_h", "ad_att_a", "ad_def_a", "ad_lam_h",
         "ad_lam_a", "div_code", "h_gf", "a_gf", "h_xf", "a_xf", "h_rest", "a_rest"])
    seen = {}
    feats = pd.DataFrame(index=["m0"], columns=["elo_diff"], dtype=float)
    monkeypatch.setattr(model_state, "build_features", lambda *a, **k: feats)
    monkeypatch.setattr(
        model_state, "training_table",
        lambda *a, **k: pd.DataFrame(columns=[
            "elo_diff", "pi_diff", "ad_att_h", "ad_def_h", "ad_att_a", "ad_def_a", "ad_lam_h",
            "ad_lam_a", "div_code", "h_gf", "a_gf", "h_xf", "a_xf", "h_rest", "a_rest"]),
    )
    monkeypatch.setattr(model_state, "load_market_pre", lambda *a, **k: {})
    monkeypatch.setattr(
        model_state, "fit_live", lambda *a, **k: ("lgb", object(), lambda p: p, {}))

    def fake_predict(model, learner, task, X, init=None):
        seen["cols"] = list(X.columns)
        seen["init"] = init
        return np.full((len(X), 3), 1 / 3)

    monkeypatch.setattr(model_state.gbm, "predict_model", fake_predict)
    fx = _future_fixtures(["2026-12-20T15:00:00Z"])
    out = model_state.score_price_free(tmp_path, played, fx, NOW)
    assert seen["cols"] == train_cols and seen["init"] is None  # no market feature, ever
    assert list(out.columns) == ["H", "D", "A"] and out.iloc[0].sum() == pytest.approx(1.0)
    fixture_order = list(model_state.fixture_features(played, fx).columns)
    assert fixture_order != train_cols  # the two orders really do differ


def test_build_rows_uses_price_free_probs_when_there_is_no_price():
    played = _played_e0()
    fx = _future_fixtures(["2026-12-20T15:00:00Z", "2027-03-01T15:00:00Z"])
    free = pd.DataFrame(
        {"H": [0.5, 0.4], "D": [0.3, 0.3], "A": [0.2, 0.3]}, index=["f0", "f1"])
    rows = pred.build_rows(pd.DataFrame(), fx, played, NOW, NOW, price_free=free)
    assert [r["model_kind"] for r in rows] == [pred.KIND_PRICE_FREE] * 2
    r = rows[0]
    assert r["status"] == "modelled" and r["mkt_home"] is None and r["mkt_draw"] is None
    assert r["p_home"] == pytest.approx(0.5, abs=1e-6)  # DC grid reconciled to the 1X2
    assert json.loads(r["top_scores"]) and "price-free" in json.loads(r["detail"])["model"]
    # without a price-free estimate the fixture is kept visibly unmodelled
    rows = pred.build_rows(pd.DataFrame(), fx, played, NOW, NOW)
    assert {r["status"] for r in rows} == {"not_modelled"}


def test_priced_fixture_keeps_lgbd_xg_and_price_free_only_fills_the_rest():
    played = _played_e0()
    fx = _future_fixtures(["2026-10-10T15:00:00Z", "2027-03-01T15:00:00Z"])
    edges = pd.DataFrame([
        {"match_key": "f0", "market": "1x2", "selection": s, "model_prob": p,
         "market_prob": 0.3, "price": 3.0, "price_source": "BFE", "pooled_prob": p, "edge": 0.0}
        for s, p in (("H", 0.5), ("D", 0.3), ("A", 0.2))
    ])
    free = pd.DataFrame({"H": [0.1, 0.4], "D": [0.1, 0.3], "A": [0.8, 0.3]}, index=["f0", "f1"])
    built = pred.build_rows(edges, fx, played, NOW, NOW, price_free=free)
    rows = {r["match_key"]: r for r in built}
    assert rows["f0"]["model_kind"] == pred.KIND_PRICED
    assert rows["f0"]["p_home"] == pytest.approx(0.5, abs=1e-6)  # not the price-free 0.1
    assert rows["f0"]["mkt_home"] == pytest.approx(0.3)
    assert rows["f1"]["model_kind"] == pred.KIND_PRICE_FREE


def test_write_predictions_stores_kind_and_meta_and_migrates_old_tables(tmp_path):
    import sqlite3

    played = _played_e0()
    fx = _future_fixtures(["2026-12-20T15:00:00Z"])
    free = pd.DataFrame({"H": [0.5], "D": [0.3], "A": [0.2]}, index=["f0"])
    rows = pred.build_rows(pd.DataFrame(), fx, played, NOW, NOW, price_free=free)
    conn = sqlite3.connect(tmp_path / "p.sqlite")
    # an old-format table without model_kind / prediction_meta
    conn.executescript(pred.SCHEMA)
    conn.execute("ALTER TABLE predictions DROP COLUMN model_kind")
    conn.execute("DROP TABLE prediction_meta")
    conn.commit()
    meta = {"schedule_stale": "1", "schedule_warnings": json.dumps(["E0: using cache"])}
    assert pred.write_predictions(conn, rows, meta) == 1
    assert conn.execute("SELECT model_kind FROM predictions").fetchone()[0] == pred.KIND_PRICE_FREE
    got = dict(conn.execute("SELECT key, value FROM prediction_meta").fetchall())
    assert got["schedule_stale"] == "1"
    assert np.isfinite(
        conn.execute("SELECT p_home FROM predictions").fetchone()[0])


def test_price_free_features_carry_xg_like_the_training_table():
    """Regression: without attach_xg every fixture's xg form features are NaN (train/serve skew)."""
    played = _played_e0().assign(HxG=1.4, AxG=1.1)
    fx = _future_fixtures(["2026-12-20T15:00:00Z"])
    codes = model_state.div_code_map(played)
    bare = model_state.fixture_features(played, fx, codes)
    assert bare[["h_xf", "a_xf", "h_xa", "a_xa"]].isna().all().all()
    empty_xg = pd.DataFrame(columns=["match_id", "xg_home", "xg_away"])
    x = model_state.fixture_features(played, fx, codes, empty_xg)
    assert x[["h_xf", "a_xf", "h_xa", "a_xa"]].notna().all().all()
