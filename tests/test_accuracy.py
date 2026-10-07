"""Accuracy tracker: telemetry store rules, score/winner consistency, metrics, report, dashboard."""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fedge.accuracy import baselines as bl
from fedge.accuracy import metrics as mt
from fedge.accuracy import report as rpt
from fedge.accuracy import results as res_mod
from fedge.accuracy import store
from fedge.dashboard.app import create_app
from fedge.predict import models
from fedge.predict.scoreline import consistent_score, predict_match

KO = pd.Timestamp("2026-11-20T15:00:00Z")


def _fx(key="m1", ko=KO, home="Arsenal", away="Chelsea", div="E0"):
    return {"match_key": key, "div": div, "league": "England Premier League", "home": home,
            "away": away, "kickoff_utc": ko}


_REC = predict_match(1.5, 1.1, -0.05, np.array([0.5, 0.25, 0.25]))


def _row(run_ts, key="m1", ko=KO, source="live", rec=None):
    return store.log_row(rec or _REC, _fx(key, ko), run_ts, "test_v1", source)


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "p.sqlite")
    yield c
    c.close()


def test_connect_uses_wal_and_dashboard_rows_only_swallow_missing_tables(conn, tmp_path):
    from fedge.dashboard import app as dash_app
    from fedge.dashboard import telemetry

    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    for rows in (telemetry._rows, dash_app._rows):
        assert rows(conn, "SELECT * FROM no_such_table") == []  # not initialised yet
        with pytest.raises(sqlite3.OperationalError):
            rows(conn, "SELECT FROM WHERE")  # any other failure must surface


def _kinds(conn, key="m1"):
    return conn.execute(
        "SELECT snapshot_kind, run_ts, hours_before_kickoff FROM prediction_log "
        "WHERE match_key = ? ORDER BY snapshot_kind, run_ts", (key,)).fetchall()


# ------------------------------------------------------------------ freeze rule / append-only
def test_freeze_rule_rejects_post_kickoff(conn):
    rows = [_row(KO), _row(KO + pd.Timedelta(minutes=5)), _row(KO - pd.Timedelta(seconds=1))]
    out = store.append_predictions(conn, rows)
    assert out["frozen"] == 2 and out["run"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM prediction_log WHERE snapshot_kind = 'run'").fetchone()[0] == 1
    assert conn.execute(  # nothing at or after kickoff was written, under any label
        "SELECT COUNT(*) FROM prediction_log WHERE run_ts >= kickoff_utc").fetchone()[0] == 0
    # a direct insert (bypassing Python) is refused by the trigger
    bad = {**_row(KO - pd.Timedelta(hours=1)), "run_ts": store.iso(KO), "snapshot_kind": "run"}
    with pytest.raises(sqlite3.DatabaseError, match="freeze"):
        conn.execute(
            f"INSERT INTO prediction_log ({','.join(store.LOG_COLS)}) "
            f"VALUES ({','.join('?' * len(store.LOG_COLS))})",
            tuple(bad[c] for c in store.LOG_COLS))


def test_prediction_log_is_append_only(conn):
    store.append_predictions(conn, [_row(KO - pd.Timedelta(days=2))])
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE prediction_log SET p_home = 0.99")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM prediction_log")


def test_reappending_the_same_run_is_a_noop(conn):
    rows = [_row(KO - pd.Timedelta(days=2))]
    first = store.append_predictions(conn, rows)
    again = store.append_predictions(conn, rows)
    assert first["run"] == 1 and again["run"] == 0 and again["duplicate"] == 1
    assert conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] == 3  # run+d7+d3


def test_runs_too_far_out_are_not_logged(conn):
    out = store.append_predictions(conn, [_row(KO - pd.Timedelta(days=20))])
    assert out["too_early"] == 1 and out["run"] == 0


# ------------------------------------------------------------------ d7 / d3 / final selection
def test_d7_d3_final_selection(conn):
    def at(**kw):
        return KO - pd.Timedelta(**kw)

    runs = [at(days=9), at(days=6, hours=12), at(days=5), at(days=2, hours=21), at(days=1),
            at(hours=2)]
    for r in runs:  # one run at a time, as the cron job would
        store.append_predictions(conn, [_row(r)])
    got = {(k, round(h, 1)) for k, _ts, h in _kinds(conn)}
    # d7 = first run at/after kickoff-7d (the 9-day-out run does not count); d3 = first at -3d
    assert ("d7", 156.0) in got and ("d3", 69.0) in got
    assert [k for k, _, _ in _kinds(conn)].count("d7") == 1
    assert [k for k, _, _ in _kinds(conn)].count("d3") == 1
    assert [k for k, _, _ in _kinds(conn)].count("run") == 6
    final = conn.execute("SELECT run_ts, hours_before_kickoff FROM v_final").fetchall()
    assert len(final) == 1 and final[0][0] == store.iso(at(hours=2)) and final[0][1] == 2.0


def test_d7_and_d3_can_be_the_same_run_and_missing_windows_stay_missing(conn):
    # first prediction only 2 days out: it is both d7 and d3 (two rows), hours stored on both
    store.append_predictions(conn, [_row(KO - pd.Timedelta(days=2))])
    k = _kinds(conn)
    assert sorted(x[0] for x in k) == ["d3", "d7", "run"]
    assert all(x[2] == 48.0 for x in k)
    # a match with no run inside 3 days has d7 but no d3, and nothing is back-filled
    store.append_predictions(conn, [_row(KO - pd.Timedelta(days=5), key="m2")])
    assert sorted(x[0] for x in _kinds(conn, "m2")) == ["d7", "run"]
    # a later run never replaces the d3 row
    store.append_predictions(conn, [_row(KO - pd.Timedelta(days=1))])
    d3 = [x for x in _kinds(conn) if x[0] == "d3"]
    assert len(d3) == 1 and d3[0][1] == store.iso(KO - pd.Timedelta(days=2))


def test_retro_rows_get_no_d7_d3_labels(conn):
    store.append_predictions(conn, [_row(KO - pd.Timedelta(days=1), source="retro")])
    assert [x[0] for x in _kinds(conn)] == ["run"]


# ------------------------------------------------------------------ score / winner consistency
def test_predicted_score_agrees_with_predicted_winner():
    g = np.zeros((8, 8))
    g[1, 1], g[1, 0], g[2, 0], g[0, 1] = 0.30, 0.15, 0.10, 0.10
    out = consistent_score(g, [0.5, 0.25, 0.25])  # home win most likely, but 1-1 is the mode
    assert out["pred_outcome"] == "H" and (out["pred_score_home"], out["pred_score_away"]) == (1, 0)
    assert (out["mode_score_home"], out["mode_score_away"]) == (1, 1)
    out = consistent_score(g, [0.2, 0.5, 0.3])
    assert out["pred_outcome"] == "D" and (out["pred_score_home"], out["pred_score_away"]) == (1, 1)
    out = consistent_score(g, [0.2, 0.3, 0.5])
    assert out["pred_outcome"] == "A" and (out["pred_score_home"], out["pred_score_away"]) == (0, 1)


@pytest.mark.parametrize("lh,la", [(0.6, 2.4), (1.4, 1.2), (2.6, 0.7), (1.0, 1.0)])
def test_predict_match_score_class_matches_outcome(lh, la):
    rec = predict_match(lh, la, -0.05, None)
    h, a = rec["pred_score_home"], rec["pred_score_away"]
    assert store.outcome_of(h, a) == rec["pred_outcome"]
    assert np.asarray(rec["grid_log"]).shape == (8, 8)
    row = store.log_row(rec, _fx(), KO - pd.Timedelta(days=1), "v")
    assert store.outcome_of(row["pred_score_home"], row["pred_score_away"]) == row["pred_outcome"]


def test_logged_prediction_has_no_price_columns():
    assert not [c for c in store.LOG_COLS if "mkt" in c or "odds" in c or "price" in c]


def test_model_registry_is_keyed_by_version():
    assert models.CURRENT_VERSION in models.REGISTRY
    with pytest.raises(KeyError, match="unknown model_version"):
        models.predict("nope", None, None, None, None)


# ------------------------------------------------------------------ metrics vs hand-computed
P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
Y = np.array([0, 2, 1])


def test_winner_metrics_hand_computed():
    pred = np.argmax(P, axis=1)  # [0, 2, 0] (ties -> first)
    assert mt.hit_rate(pred, Y) == pytest.approx(2 / 3)
    assert mt.brier_multiclass(P, Y) == pytest.approx((0.38 + 0.38 + 0.56) / 3)
    assert mt.log_loss(P, Y) == pytest.approx((-math.log(0.5) * 2 - math.log(0.4)) / 3)
    assert mt.rps(P, Y) == pytest.approx((0.145 + 0.145 + 0.10) / 3)
    assert mt.confusion_matrix(pred, Y) == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert mt.draw_predicted_rate(pred) == 0.0
    cal = mt.calibration_table(P, Y, 10)
    assert [c["n"] for c in cal] == [0, 0, 3, 2, 2, 2, 0, 0, 0, 0]
    assert cal[2]["observed"] == 0.0 and cal[4]["observed"] == pytest.approx(0.5)
    assert cal[5]["observed"] == 1.0 and cal[4]["mean_pred"] == pytest.approx(0.4)


def _grids():
    def g(cells):
        a = np.zeros((8, 8))
        for (i, j), p in cells.items():
            a[i, j] = p
        return a

    return np.array([
        g({(1, 0): .3, (0, 0): .25, (1, 1): .2, (2, 0): .1, (0, 1): .05}),
        g({(1, 1): .3, (0, 0): .25, (1, 0): .2, (0, 1): .1, (2, 1): .05}),
        g({(2, 1): .3, (1, 1): .2, (1, 0): .1}),
    ])


def test_scoreline_metrics_hand_computed():
    hg, ag = np.array([1, 0, 3]), np.array([0, 1, 3])
    ph, pa = np.array([1, 1, 2]), np.array([0, 1, 1])
    grids = _grids()
    assert mt.exact_hit_rate(ph, pa, hg, ag) == pytest.approx(1 / 3)
    assert mt.topk_hit_rate(grids, hg, ag, 3) == pytest.approx(1 / 3)  # only match 1
    assert mt.topk_hit_rate(grids, hg, ag, 5) == pytest.approx(2 / 3)  # + match 2 (rank 4)
    assert mt.goal_diff_hit_rate(ph, pa, hg, ag) == pytest.approx(1 / 3)  # predicted gd 1,0,1
    assert mt.mean_log_score(grids, hg, ag) == pytest.approx(
        (math.log(0.3) + math.log(0.1) + math.log(mt.EPS)) / 3)
    assert mt.mean_actual_score_prob(grids, hg, ag) == pytest.approx((0.3 + 0.1 + 0.0) / 3)
    assert mt.mae([1.5, 2.0], [1, 3]) == pytest.approx(0.75)
    acc, brier = mt.binary_accuracy_brier([0.7, 0.4, 0.5], [1, 0, 0])
    assert acc == pytest.approx(2 / 3) and brier == pytest.approx(0.5 / 3)


def test_bootstrap_ci_is_seeded_and_clustered():
    v = [1, 1, 0, 0]
    assert mt.bootstrap_ci(v, ["a", "a", "b", "b"]) == mt.bootstrap_ci(v, ["a", "a", "b", "b"])
    lo, hi = mt.bootstrap_ci(v, ["a", "a", "b", "b"])  # only whole clusters: means in {0, .5, 1}
    assert (lo, hi) == (0.0, 1.0)
    assert mt.bootstrap_ci([1, 1, 1], None) == (1.0, 1.0)
    lo2, hi2 = mt.bootstrap_ci(np.r_[np.ones(60), np.zeros(40)])
    assert lo2 < 0.6 < hi2 and hi2 - lo2 < 0.25


# ------------------------------------------------------------------ baselines
def _history():
    rows = []
    t0 = pd.Timestamp("2025-08-16T15:00Z")
    teams = ["Arsenal", "Chelsea", "Spurs", "Fulham"]
    for k in range(80):
        h, a = teams[k % 4], teams[(k + 1 + k // 4 % 3) % 4]
        if h == a:
            continue
        rows.append({"div": "E0", "home": h, "away": a,
                     "kickoff_utc": t0 + pd.Timedelta(days=3 * k),
                     "FTHG": 2 if h == "Arsenal" else 1, "FTAG": 0 if h == "Arsenal" else 1})
    return pd.DataFrame(rows)


def test_baselines_shapes_and_prev_season_poisson():
    df = pd.DataFrame({"div": ["E0", "E0"], "home": ["Arsenal", "Fulham"],
                       "away": ["Chelsea", "Spurs"],
                       "kickoff_utc": pd.to_datetime(["2026-09-12T15:00Z", "2026-09-12T15:00Z"])})
    out = bl.baseline_frames(df, _history())
    assert set(out) == {"always_home", "league_freq", "prev_season_poisson"}
    for f in out.values():
        assert f["P"].shape == (2, 3)
    assert np.allclose(out["league_freq"]["P"].sum(axis=1), 1)
    assert np.allclose(out["prev_season_poisson"]["P"].sum(axis=1), 1)
    # Arsenal (strong, last season) at home beats Fulham at home in the Poisson baseline
    assert out["prev_season_poisson"]["P"][0, 0] > out["prev_season_poisson"]["P"][1, 0]
    assert out["always_home"]["P"].tolist() == [[1, 0, 0], [1, 0, 0]]


# ------------------------------------------------------------------ results table
def _res(key="m1", hg=2, ag=1, source="fixturedownload"):
    return {"match_key": key, "div": "E0", "home": "Arsenal", "away": "Chelsea",
            "kickoff_utc": store.iso(KO), "home_goals": hg, "away_goals": ag,
            "result": store.outcome_of(hg, ag), "source": source, "fetched_ts": store.iso(KO)}


def test_results_upsert_is_idempotent_and_flags_conflicts(conn):
    assert store.upsert_results(conn, [_res()]) == {"inserted": 1, "unchanged": 0, "conflict": 0}
    assert store.upsert_results(conn, [_res()]) == {"inserted": 0, "unchanged": 1, "conflict": 0}
    assert store.upsert_results(conn, [_res(hg=0)])["conflict"] == 1
    assert conn.execute("SELECT home_goals FROM results").fetchone()[0] == 2  # never changed
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1


def test_build_results_schedule_primary_with_football_data_fallback_and_crosscheck():
    k = pd.Timestamp("2026-08-21T19:00Z")
    sched = pd.DataFrame({
        "div": ["E0", "E0", "E0"], "home": ["Arsenal", "Hull", "Leeds"],
        "away": ["Coventry", "Man United", "Fulham"],
        "kickoff_utc": [k, k + pd.Timedelta(days=1), k + pd.Timedelta(days=2)],
        "round": [1, 1, 1], "played": [True, False, False],
        "home_goals": [3.0, np.nan, np.nan], "away_goals": [0.0, np.nan, np.nan],
    })
    fd = pd.DataFrame({
        "div": ["E0", "E0"], "home": ["Arsenal", "Hull"], "away": ["Coventry", "Man United"],
        "kickoff_utc": [k, k + pd.Timedelta(days=1)], "FTHG": [2, 2], "FTAG": [0, 0],
    })
    rows, warns = res_mod.build_results(
        sched, {"E0": "fixturedownload"}, fd, ["E0"], k + pd.Timedelta(days=5))
    by = {r["home"]: r for r in rows}
    # conflict: the football-data value (2-0) wins over the schedule's 3-0
    assert by["Arsenal"]["home_goals"] == 2 and by["Arsenal"]["source"] == "football_data"
    assert by["Hull"]["source"] == "football_data" and by["Hull"]["result"] == "H"
    assert "Leeds" not in by  # not finished anywhere
    assert any("mismatch" in w for w in warns)  # 3-0 vs 2-0 is reported, schedule kept


def test_build_results_ignores_in_play_schedule_score_but_accepts_after_3h():
    k = pd.Timestamp("2026-08-21T19:00Z")
    sched = pd.DataFrame({
        "div": ["E0"], "home": ["Arsenal"], "away": ["Coventry"], "kickoff_utc": [k],
        "round": [1], "played": [True], "home_goals": [1.0], "away_goals": [0.0],
    })
    src = {"E0": "fixturedownload"}
    live, warns = res_mod.build_results(sched, src, None, ["E0"], k + pd.Timedelta(minutes=30))
    assert live == [] and any("in-play" in w for w in warns)
    short, _ = res_mod.build_results(sched, src, None, ["E0"], k + pd.Timedelta(minutes=179))
    assert short == []
    done, _ = res_mod.build_results(sched, src, None, ["E0"], k + pd.Timedelta(hours=3))
    assert [(r["home_goals"], r["source"]) for r in done] == [(1, "fixturedownload")]
    # a football-data twin is final by definition: used even while the schedule looks in-play
    fd = pd.DataFrame({"div": ["E0"], "home": ["Arsenal"], "away": ["Coventry"],
                       "kickoff_utc": [k], "FTHG": [2], "FTAG": [0]})
    rows, _ = res_mod.build_results(sched, src, fd, ["E0"], k + pd.Timedelta(minutes=30))
    assert [(r["home_goals"], r["source"]) for r in rows] == [(2, "football_data")]


# ------------------------------------------------------------------ report
def _populate(conn, n=6, with_retro=2):
    """``n`` live matches (runs 2 days, 3 hours out) + retro matches (1 day out), all finished."""
    outcomes = [(2, 1), (0, 0), (1, 2), (1, 0), (3, 1), (2, 2)]
    for i in range(n + with_retro):
        ko = KO + pd.Timedelta(days=i)
        key = f"k{i}"
        retro = i >= n
        runs = [ko - pd.Timedelta(days=1)] if retro else [ko - pd.Timedelta(days=2),
                                                          ko - pd.Timedelta(hours=3)]
        for r in runs:
            store.append_predictions(conn, [_row(r, key, ko, "retro" if retro else "live")])
        hg, ag = outcomes[i % len(outcomes)]
        store.upsert_results(conn, [{**_res(key, hg, ag), "kickoff_utc": store.iso(ko)}])


def test_report_counts_kinds_and_retro_toggle(conn):
    _populate(conn)
    frames = rpt.load_frames(conn)
    assert len(frames["final"]) == 8 and len(frames["d7"]) == 6 and len(frames["d3"]) == 6
    live = rpt.build_report(frames, None, include_retro=False, n_boot=50)
    assert live["scored"] == 6 and live["counts"] == {"live": 6, "retro": 2} and not live["empty"]
    both = rpt.build_report(frames, None, include_retro=True, n_boot=50)
    assert both["scored"] == 8
    h = live["headline"]
    assert h["n"] == 6 and h["small_sample"] is True
    assert 0 <= h["winner"]["hit_rate"] <= 1 and h["score"]["exact_hit_rate"] >= 0
    # d7 = the 2-day-out run, final = the 3-hours-before run (different hours reported)
    assert live["by_kind"]["d7"]["mean_hours_before_kickoff"] == pytest.approx(48.0)
    assert live["by_kind"]["final"]["mean_hours_before_kickoff"] == pytest.approx(3.0)
    assert live["by_kind_common"]["final"]["n"] == 6
    assert len(live["cumulative"]) == 6 and len(live["recent"]) == 6
    r0 = live["recent"][0]
    assert r0["winner_ok"] in (True, False) and r0["source"] == "live"


def test_report_with_history_has_baselines(conn):
    _populate(conn)
    rep = rpt.build_report(rpt.load_frames(conn), _history(), n_boot=50)
    b = rep["headline"]["baselines"]
    assert set(b) == {"always_home", "league_freq", "prev_season_poisson"}
    assert "log_loss" not in b["always_home"] and "log_loss" in b["league_freq"]


def test_report_empty_state(conn):
    rep = rpt.build_report(rpt.load_frames(conn), None)
    assert rep["empty"] is True and rep["scored"] == 0 and "headline" not in rep
    # only retro data: live view is empty but says retro exists
    store.append_predictions(conn, [_row(KO - pd.Timedelta(days=1), source="retro")])
    store.upsert_results(conn, [_res()])
    rep = rpt.build_report(rpt.load_frames(conn), None)
    assert rep["empty"] and rep["counts"]["retro"] == 1


# ------------------------------------------------------------------ dashboard
def _root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "strategy.toml").write_text('gate0_passed = false\nmode = "shadow"\n')
    return tmp_path


def test_dashboard_accuracy_empty_and_populated_and_read_only(tmp_path):
    root = _root(tmp_path)
    c = TestClient(create_app(root))
    got = c.get("/api/accuracy").json()
    assert got["empty"] is True and got["scored"] == 0
    assert not (root / "data" / store.DB_NAME).exists()  # the dashboard never creates it
    conn = store.connect(store.default_path(root / "data"))
    _populate(conn)
    conn.commit()
    live = c.get("/api/accuracy").json()
    assert live["scored"] == 6 and live["counts"]["retro"] == 2
    assert c.get("/api/accuracy?include_retro=1").json()["scored"] == 8
    from fedge.dashboard.telemetry import _connect_ro

    ro = _connect_ro(store.default_path(root / "data"))
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO meta VALUES ('a', 'b')")
    ro.close()
    conn.close()


def test_dashboard_predictions_upcoming_and_recent_results(tmp_path):
    root = _root(tmp_path)
    c = TestClient(create_app(root))
    empty = c.get("/api/predictions").json()
    assert empty["rows"] == [] and empty["stale"] is True and "Premier League" in empty["scope"]
    conn = store.connect(store.default_path(root / "data"))
    now = pd.Timestamp.now(tz="UTC")
    far, done = now + pd.Timedelta(days=200), now - pd.Timedelta(days=1)
    rows = []
    for key, ko in (("far", far), ("past", done)):
        row = _row(ko - pd.Timedelta(days=2), key, ko)
        rows.append({**row, "note": None})
    store.replace_upcoming(conn, rows, {"last_run_utc": store.iso(now), "schedule_stale": "0"})
    store.append_predictions(conn, [_row(done - pd.Timedelta(days=1), "past", done)])
    store.upsert_results(conn, [{**_res("past", 2, 0), "kickoff_utc": store.iso(done)}])
    got = c.get("/api/predictions").json()
    assert got["stale"] is False and [r["match_key"] for r in got["rows"]] == ["far"]
    r = got["rows"][0]
    assert r["pred_score"] and r["pred_outcome"] in "HDA" and "long_range" not in r
    assert r["kickoff_local"]
    assert not {k for k in r if "mkt" in k or "market" in k or "odds" in k}
    rec = got["recent_results"]
    assert len(rec) == 1 and rec[0]["actual_score"] == [2, 0] and rec[0]["match_key"] == "past"
    conn.close()


def test_pages_and_archive_redirect(tmp_path):
    c = TestClient(create_app(_root(tmp_path)))
    r = c.get("/ledger", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/archive"
    assert c.get("/archive").status_code == 200 and c.get("/accuracy").status_code == 200
    for page in ("/", "/accuracy"):
        text = c.get(page).text
        for banned in ("odds", "Market", "bookmaker price", "CLV", "longer-range", "long_range"):
            if banned == "bookmaker price":
                continue  # the intro states that no bookmaker prices are used
            assert banned not in text, f"{page} should not mention {banned!r}"
    assert "Archive" in c.get("/archive").text
