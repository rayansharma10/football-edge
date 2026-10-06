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
