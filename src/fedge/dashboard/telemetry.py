"""Telemetry routes of the dashboard: ``/api/predictions`` (Upcoming) and ``/api/accuracy``.

Everything here reads ``data/predictions.sqlite`` with ``mode=ro`` (scripts are the only writers)
and shows model output only - no bookmaker prices exist in that database.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI

from fedge.accuracy import report as rpt
from fedge.accuracy import store

PREDICTIONS_STALE_HOURS = 20.0  # the predictions job runs twice a day (11:30 and 19:00 Sydney)
RECENT_DAYS = 3
SYDNEY = ZoneInfo("Australia/Sydney")


def _connect_ro(db: Path) -> sqlite3.Connection | None:
    if not db.exists():
        return None
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _rows(conn, sql: str, args: tuple = ()) -> list[dict]:
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:  # table missing: the scripts have not run yet
        return []


def register(app: FastAPI, root: Path) -> None:
    db = Path(root) / "data" / store.DB_NAME
    history_file = Path(root) / "data" / "interim" / "matches.parquet"
    cache: dict = {}

    def history() -> pd.DataFrame | None:
        if not history_file.exists():
            return None
        key = ("hist", history_file.stat().st_mtime)
        if cache.get("hist_key") != key:
            h = pd.read_parquet(history_file)
            h = h.loc[h["FTHG"].notna() & h["FTAG"].notna()]
            cache["hist"] = h[["div", "home", "away", "kickoff_utc", "FTHG", "FTAG"]].copy()
            cache["hist_key"] = key
        return cache["hist"]

    @app.get("/api/predictions")
    def predictions():
        """Upcoming Premier League / La Liga predictions (model output only) plus matches that
        finished in the last 3 days with the actual result next to the prediction."""
        c = _connect_ro(db)
        try:
            rows = _rows(c, "SELECT * FROM upcoming ORDER BY kickoff_utc, home")
            meta = {m["key"]: m["value"] for m in _rows(c, "SELECT key, value FROM meta")}
            frames = rpt.load_frames(c) if c is not None else {}
        finally:
            if c:
                c.close()
        now = datetime.now(UTC)
        out = []
        versions = set()
        for r in rows:
            ko = _parse(r["kickoff_utc"])
            if ko is None or ko <= now:
                continue  # already kicked off since the last run
            try:
                top = json.loads(r["top_scores"]) if r["top_scores"] else []
            except ValueError:
                top = []
            versions.add(r["model_version"])
            out.append({
                "match_key": r["match_key"], "status": "modelled", "div": r["div"],
                "league": r["league"], "home": r["home"], "away": r["away"],
                "kickoff_utc": r["kickoff_utc"],
                "kickoff_local": ko.astimezone(SYDNEY).isoformat(timespec="minutes"),
                "probs": {"home": r["p_home"], "draw": r["p_draw"], "away": r["p_away"]},
                "xg": {"home": r["xg_home"], "away": r["xg_away"]},
                "p_over25": r["p_over25"], "p_btts": r["p_btts"], "top_scores": top,
                "pred_outcome": r["pred_outcome"],
                "pred_score": [r["pred_score_home"], r["pred_score_away"]],
                "pred_score_p": r["pred_score_p"], "note": r["note"],
            })
        recent: list[dict] = []
        fin = frames.get("final", pd.DataFrame())
        if len(fin):
            cutoff = (now - timedelta(days=RECENT_DAYS)).isoformat(timespec="seconds")
            sub = fin.loc[fin["kickoff_utc"] >= cutoff]
            sub = sub.sort_values("source", kind="mergesort").drop_duplicates("match_key")
            recent = rpt.clean(rpt.recent_rows(sub, limit=200))
            for r in recent:
                ko = _parse(r["kickoff_utc"])
                r["kickoff_local"] = (
                    ko.astimezone(SYDNEY).isoformat(timespec="minutes") if ko else None
                )
        last = _parse(meta.get("last_run_utc"))
        age = None if last is None else (now - last).total_seconds() / 3600
        try:
            warns = json.loads(meta.get("schedule_warnings") or "[]")
        except ValueError:
            warns = []
        sched_stale = meta.get("schedule_stale") == "1"
        return {
            "rows": out, "recent_results": recent,
            "recent_days": RECENT_DAYS,
            "schedule": {
                "fetched_utc": meta.get("schedule_fetched_utc") or None, "stale": sched_stale,
                "warnings": warns if sched_stale else [],
            },
            "last_updated_utc": meta.get("last_run_utc"),
            "age_hours": None if age is None else round(age, 1),
            "stale_after_hours": PREDICTIONS_STALE_HOURS,
            "stale": age is None or age > PREDICTIONS_STALE_HOURS,
            "model_version": ", ".join(sorted(versions)) or None,
            "scope": "Premier League (E0) + La Liga (SP1) only",
        }

    @app.get("/api/accuracy")
    def accuracy(include_retro: int = 0):
        """Winner + scoreline accuracy of the logged predictions (see ``fedge.accuracy``).

        ``include_retro=1`` adds the retrospective (not live) predictions to the numbers.
        """
        if not db.exists():
            return rpt.build_report({}, None, bool(include_retro))
        stamp = (db.stat().st_mtime_ns, db.stat().st_size, bool(include_retro),
                 history_file.stat().st_mtime if history_file.exists() else 0)
        wal = db.with_name(db.name + "-wal")
        if wal.exists():
            stamp += (wal.stat().st_mtime_ns,)
        if cache.get("acc_key") == stamp:
            return cache["acc"]
        c = _connect_ro(db)
        try:
            frames = rpt.load_frames(c)
        finally:
            if c:
                c.close()
        out = rpt.build_report(frames, history(), bool(include_retro))
        cache["acc_key"], cache["acc"] = stamp, out
        return out
