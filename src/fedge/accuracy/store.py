"""Telemetry store: the append-only ``prediction_log`` and the ``results`` table.

Lives in its own file, ``data/predictions.sqlite``, NOT in the betting ledger
(``data/paper.sqlite``): betting is parked, its tables stay untouched as an archive, and the
accuracy tracker must keep working (and be backed up / wiped) independently of it. The dashboard
opens this file ``mode=ro``; only ``scripts/predict_upcoming.py`` and ``scripts/update_results.py``
write.

``prediction_log`` rules (enforced by SQLite triggers, not just by Python):

* append-only: UPDATE and DELETE abort;
* freeze rule: a row with ``run_ts >= kickoff_utc`` is rejected (INSERT aborts);
* ``snapshot_kind``: ``run`` = every logged prediction run; ``d7`` / ``d3`` = a copy of the FIRST
  run at or after ``kickoff - 7 days`` / ``kickoff - 3 days`` (unique per match, absent when no
  run fell in the window - never back-filled). One run can carry both labels (two rows).
  ``final`` is not stored: it is the latest ``run`` row before kickoff (view ``v_final``).
* ``hours_before_kickoff`` is stored on every row.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_NAME = "predictions.sqlite"
LOG_HORIZON_DAYS = 14  # runs further out than this are shown on the page but not logged
D7_HOURS = 7 * 24.0
D3_HOURS = 3 * 24.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    match_key            TEXT NOT NULL,
    div                  TEXT NOT NULL,
    league               TEXT,
    home                 TEXT NOT NULL,
    away                 TEXT NOT NULL,
    kickoff_utc          TEXT NOT NULL,
    snapshot_kind        TEXT NOT NULL CHECK (snapshot_kind IN ('run', 'd7', 'd3')),
    run_ts               TEXT NOT NULL,
    hours_before_kickoff REAL NOT NULL,
    model_version        TEXT NOT NULL,
    source               TEXT NOT NULL DEFAULT 'live' CHECK (source IN ('live', 'retro')),
    p_home REAL NOT NULL, p_draw REAL NOT NULL, p_away REAL NOT NULL,
    xg_home REAL NOT NULL, xg_away REAL NOT NULL,
    pred_outcome         TEXT NOT NULL CHECK (pred_outcome IN ('H', 'D', 'A')),
    pred_score_home      INTEGER NOT NULL, pred_score_away INTEGER NOT NULL,
    pred_score_p         REAL NOT NULL,
    pred_mode_score_home INTEGER NOT NULL, pred_mode_score_away INTEGER NOT NULL,
    p_over25 REAL NOT NULL, p_btts REAL NOT NULL,
    top_scores           TEXT NOT NULL,   -- JSON [{score, p}, ...] (top 5)
    grid                 TEXT NOT NULL,   -- JSON 8x8, rows = home goals 0-7, cols = away goals 0-7
    UNIQUE (match_key, snapshot_kind, run_ts)
);
CREATE UNIQUE INDEX IF NOT EXISTS prediction_log_d7 ON prediction_log (match_key)
    WHERE snapshot_kind = 'd7';
CREATE UNIQUE INDEX IF NOT EXISTS prediction_log_d3 ON prediction_log (match_key)
    WHERE snapshot_kind = 'd3';
CREATE INDEX IF NOT EXISTS prediction_log_kickoff ON prediction_log (kickoff_utc);

CREATE TRIGGER IF NOT EXISTS prediction_log_no_update BEFORE UPDATE ON prediction_log
BEGIN SELECT RAISE(ABORT, 'prediction_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prediction_log_no_delete BEFORE DELETE ON prediction_log
BEGIN SELECT RAISE(ABORT, 'prediction_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prediction_log_freeze BEFORE INSERT ON prediction_log
WHEN NEW.run_ts >= NEW.kickoff_utc
BEGIN SELECT RAISE(ABORT, 'freeze rule: prediction made at/after kickoff'); END;

CREATE VIEW IF NOT EXISTS v_final AS
SELECT p.* FROM prediction_log p
WHERE p.snapshot_kind = 'run'
  AND p.run_ts = (SELECT MAX(q.run_ts) FROM prediction_log q
                  WHERE q.match_key = p.match_key AND q.snapshot_kind = 'run'
                    AND q.source = p.source AND q.run_ts < q.kickoff_utc);

CREATE TABLE IF NOT EXISTS results (
    match_key   TEXT PRIMARY KEY,
    div         TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    kickoff_utc TEXT NOT NULL,
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    result      TEXT NOT NULL CHECK (result IN ('H', 'D', 'A')),
    source      TEXT NOT NULL,
    fetched_ts  TEXT NOT NULL
);

-- display cache for the Upcoming page: replaced on every run (not an archive)
CREATE TABLE IF NOT EXISTS upcoming (
    match_key TEXT PRIMARY KEY, run_ts TEXT NOT NULL, model_version TEXT NOT NULL,
    div TEXT, league TEXT, home TEXT, away TEXT, kickoff_utc TEXT,
    p_home REAL, p_draw REAL, p_away REAL, xg_home REAL, xg_away REAL,
    pred_outcome TEXT, pred_score_home INTEGER, pred_score_away INTEGER, pred_score_p REAL,
    p_over25 REAL, p_btts REAL, top_scores TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

LOG_COLS = [
    "match_key", "div", "league", "home", "away", "kickoff_utc", "snapshot_kind", "run_ts",
    "hours_before_kickoff", "model_version", "source", "p_home", "p_draw", "p_away", "xg_home",
    "xg_away", "pred_outcome", "pred_score_home", "pred_score_away", "pred_score_p",
    "pred_mode_score_home", "pred_mode_score_away", "p_over25", "p_btts", "top_scores", "grid",
]
RESULT_COLS = [
    "match_key", "div", "home", "away", "kickoff_utc", "home_goals", "away_goals", "result",
    "source", "fetched_ts",
]
UPCOMING_COLS = [
    "match_key", "run_ts", "model_version", "div", "league", "home", "away", "kickoff_utc",
    "p_home", "p_draw", "p_away", "xg_home", "xg_away", "pred_outcome", "pred_score_home",
    "pred_score_away", "pred_score_p", "p_over25", "p_btts", "top_scores", "note",
]


def default_path(data_dir: Path | str = "data") -> Path:
    return Path(data_dir) / DB_NAME


def connect(path: Path | str) -> sqlite3.Connection:
    """Writable connection (scripts only); creates the schema."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def iso(ts) -> str:
    """Canonical timestamp text: UTC, seconds, ``+00:00`` (sorts lexicographically)."""
    return utc(ts).isoformat(timespec="seconds")


def outcome_of(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A")


def log_row(rec: dict, fx: dict, run_ts, model_version: str, source: str = "live") -> dict:
    """A ``prediction_log`` dict (kind ``run``) from a model record and its fixture."""
    run, ko = utc(run_ts), utc(fx["kickoff_utc"])
    grid = np.round(np.asarray(rec["grid_log"], dtype=float), 6).tolist()
    return {
        "match_key": fx["match_key"], "div": fx["div"],
        "league": fx.get("league") or fx["div"], "home": fx["home"], "away": fx["away"],
        "kickoff_utc": iso(ko), "snapshot_kind": "run", "run_ts": iso(run),
        "hours_before_kickoff": round((ko - run).total_seconds() / 3600.0, 4),
        "model_version": model_version, "source": source,
        "p_home": float(rec["p_home"]), "p_draw": float(rec["p_draw"]),
        "p_away": float(rec["p_away"]),
        "xg_home": float(rec["xg_home"]), "xg_away": float(rec["xg_away"]),
        "pred_outcome": rec["pred_outcome"],
        "pred_score_home": int(rec["pred_score_home"]),
        "pred_score_away": int(rec["pred_score_away"]),
        "pred_score_p": float(rec["pred_score_p"]),
        "pred_mode_score_home": int(rec["mode_score_home"]),
        "pred_mode_score_away": int(rec["mode_score_away"]),
        "p_over25": float(rec["p_over25"]), "p_btts": float(rec["p_btts"]),
        "top_scores": json.dumps(
            [{"score": t["score"], "p": round(float(t["p"]), 6)} for t in rec["top_scores"]][:5]
        ),
        "grid": json.dumps(grid, separators=(",", ":")),
    }


def _insert(conn: sqlite3.Connection, row: dict) -> bool:
    cur = conn.execute(
        f"INSERT OR IGNORE INTO prediction_log ({','.join(LOG_COLS)}) "
        f"VALUES ({','.join('?' * len(LOG_COLS))})",
        tuple(row[c] for c in LOG_COLS),
    )
    return cur.rowcount == 1


def append_predictions(conn: sqlite3.Connection, rows: list[dict]) -> dict[str, int]:
    """Append ``run`` rows (and the d7 / d3 labels they earn); returns counts.

    Rejected (never written): ``run_ts >= kickoff_utc`` (freeze rule) and runs further than
    :data:`LOG_HORIZON_DAYS` before kickoff. Re-appending the same (match, run_ts) is a no-op.
    """
    out = {"run": 0, "d7": 0, "d3": 0, "frozen": 0, "too_early": 0, "duplicate": 0}
    with conn:
        for r in rows:
            h = r["hours_before_kickoff"]
            if r["run_ts"] >= r["kickoff_utc"] or h <= 0:
                out["frozen"] += 1
                continue
            if h > LOG_HORIZON_DAYS * 24:
                out["too_early"] += 1
                continue
            if not _insert(conn, {**r, "snapshot_kind": "run"}):
                out["duplicate"] += 1
                continue
            out["run"] += 1
            if r["source"] != "live":
                continue
            for kind, limit in (("d7", D7_HOURS), ("d3", D3_HOURS)):
                if h <= limit and _insert(conn, {**r, "snapshot_kind": kind}):
                    out[kind] += 1
    return out


def replace_upcoming(conn: sqlite3.Connection, rows: list[dict], meta: dict[str, str]) -> int:
    """Replace the Upcoming-page display cache (and its meta) in one transaction."""
    with conn:
        conn.execute("DELETE FROM upcoming")
        conn.executemany(
            f"INSERT INTO upcoming ({','.join(UPCOMING_COLS)}) "
            f"VALUES ({','.join('?' * len(UPCOMING_COLS))})",
            [tuple(r.get(c) for c in UPCOMING_COLS) for r in rows],
        )
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            list(meta.items()),
        )
    return len(rows)


def upsert_results(conn: sqlite3.Connection, rows: list[dict]) -> dict[str, int]:
    """Insert finished matches; existing ones are never changed (idempotent).

    A re-fetch whose score differs from the stored one is counted under ``conflict`` (and left
    alone): a changed score is a data problem a human should look at.
    """
    out = {"inserted": 0, "unchanged": 0, "conflict": 0}
    with conn:
        for r in rows:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO results ({','.join(RESULT_COLS)}) "
                f"VALUES ({','.join('?' * len(RESULT_COLS))})",
                tuple(r[c] for c in RESULT_COLS),
            )
            if cur.rowcount == 1:
                out["inserted"] += 1
                continue
            old = conn.execute(
                "SELECT home_goals, away_goals FROM results WHERE match_key = ?", (r["match_key"],)
            ).fetchone()
            same = old is not None and (old[0], old[1]) == (r["home_goals"], r["away_goals"])
            out["unchanged" if same else "conflict"] += 1
    return out
