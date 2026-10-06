"""Dashboard API: read-only, empty-ledger safe, populated ledger correct."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from fedge.dashboard.app import create_app
from fedge.paper import ledger


def _root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "strategy.toml").write_text(
        'gate0_passed = false\nmode = "shadow"\n[shadow]\nshadow_threshold = 0.03\n'
        '[markets.1x2]\nmodel = "m"\nqualified = false\n'
    )
    return tmp_path


def test_empty_no_db(tmp_path):
    c = TestClient(create_app(_root(tmp_path)))
    s = c.get("/api/summary").json()
    assert s["ledger_exists"] is False and s["gate0_passed"] is False and s["mode"] == "shadow"
    assert s["counts"]["bets"] == 0 and s["any_stale"] is True and s["latest_report"] is None
    for p in ("open_bets", "settled", "snapshots"):
        assert c.get(f"/api/{p}").json() == {"rows": []}
    assert c.get("/api/series").json() == {"points": []}
    assert c.get("/api/report/latest").status_code == 404
    assert c.get("/").status_code == 200
    assert not (tmp_path / "data" / "paper.sqlite").exists()  # never created by the dashboard


def test_empty_ledger_and_populated(tmp_path):
    root = _root(tmp_path)
    conn = ledger.connect(root / "data" / "paper.sqlite")
    c = TestClient(create_app(root))
    assert c.get("/api/summary").json()["counts"]["open"] == 0

    def bet(i):
        conn.execute(
            "INSERT INTO bets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"b{i}", "2026-10-01T00:00:00+00:00", f"k{i}", "E0", "2026-10-02", "A", "B",
             "2026-10-02T14:00:00+00:00", "1x2", "H", .5, .4, .45, 2.5, "PS", .05, 1.0, .01,
             "shadow", "h", "g", "e"),
        )

    bet(1)
    bet(2)
    conn.execute(
        "INSERT INTO settlements VALUES ('b1','2026-10-03T00:00:00+00:00','H',1,1.5,2.2,'PS',"
        "0.02,0.03)"
    )
    conn.commit()
    (root / "reports" / "weekly").mkdir(parents=True)
    (root / "reports" / "weekly" / "2026-10-06.md").write_text("# hi")
    s = c.get("/api/summary").json()
    assert s["counts"] == {"bets": 2, "open": 1, "settled": 1, "snapshots": 0}
    assert s["settled"]["pnl_units"] == 1.5 and s["latest_report"] == "2026-10-06.md"
    assert [r["bet_id"] for r in c.get("/api/open_bets").json()["rows"]] == ["b2"]
    assert c.get("/api/settled").json()["rows"][0]["log_clv"] == 0.02
    assert c.get("/api/series").json()["points"][0]["cum_pnl"] == 1.5
    assert c.get("/api/report/latest").text == "# hi"


def test_connection_is_read_only(tmp_path):
    root = _root(tmp_path)
    ledger.connect(root / "data" / "paper.sqlite").close()
    from fedge.dashboard.app import _connect_ro

    ro = _connect_ro(root / "data" / "paper.sqlite")
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM bets")
    ro.close()


def test_predictions_endpoint_empty_and_populated(tmp_path):
    from fedge.predict import upcoming as pred

    root = _root(tmp_path)
    c = TestClient(create_app(root))
    empty = c.get("/api/predictions").json()
    assert empty["rows"] == [] and empty["stale"] is True and "Premier League" in empty["scope"]

    conn = ledger.connect(root / "data" / "paper.sqlite")
    row = {
        "match_key": "k1", "run_ts": "2026-10-06T00:00:00+00:00", "status": "modelled",
        "reason": None, "div": "E0", "league": "England Premier League", "home": "Arsenal",
        "away": "Chelsea", "kickoff_utc": "2026-10-10T14:00:00+00:00",
        "p_home": 0.5, "p_draw": 0.27, "p_away": 0.23,
        "mkt_home": 0.45, "mkt_draw": 0.28, "mkt_away": 0.27,
        "xg_home": 1.6, "xg_away": 1.1, "p_over25": 0.52, "p_btts": 0.55,
        "top_scores": '[{"score": "1-1", "p": 0.11}]',
        "grid": "[[0.05, 0.08], [0.1, 0.11]]",
        "detail": '{"method": "dixon-coles lambdas solved to match 1X2", "grid_mass_shown": 0.97}',
    }
    pred.write_predictions(conn, [row])
    got = c.get("/api/predictions").json()
    assert got["stale"] is False and got["last_updated_utc"] == "2026-10-06T00:00:00+00:00"
    r = got["rows"][0]
    assert r["league"] == "England Premier League" and r["home"] == "Arsenal"
    assert r["probs"]["home"] == pytest.approx(0.5)
    assert r["xg"]["home"] == pytest.approx(1.6)
    assert r["top_scores"][0]["score"] == "1-1"
    # the dashboard shows the model alone: prices, margins and the scoreline grid never reach it
    for hidden in ("market", "gap", "grid"):
        assert hidden not in r
    # kickoff shown in Australia/Sydney (2026-10-10T14:00Z -> 2026-10-11T01:00+11:00)
    assert r["kickoff_local"].startswith("2026-10-11T01:00")


def test_main_page_is_model_only_and_the_ledger_holds_the_book(tmp_path):
    """The fixtures page must not show bookmaker prices; the ledger page may."""
    c = TestClient(create_app(_root(tmp_path)))
    assert c.get("/ledger").status_code == 200
    assert c.get("/style.css").status_code == 200
    assert c.get("/style.css").headers["content-type"].startswith("text/css")
    home = c.get("/").text
    assert "Upcoming fixtures" in home
    for banned in ("Market", "Gap", "Edge", "CLV", "de-vigged", "gap"):
        assert banned not in home, f"main page should not mention {banned!r}"
    ledger = c.get("/ledger").text
    assert "Edge" in ledger and "CLV" in ledger
