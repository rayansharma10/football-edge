"""Read-only local dashboard over the paper ledger (``data/paper.sqlite``).

The ledger is opened with ``mode=ro`` (SQLite URI) on every request, so this process can never
write to it. Run via ``scripts/dashboard.py`` (binds 127.0.0.1 only).
"""

from __future__ import annotations

import json
import sqlite3
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).with_name("static")

# A job is "stale" when its last success is older than this many hours.
STALE_HOURS = {"picks": 14.0, "settle": 30.0}
# Upcoming-predictions table: refreshed with each picks run (twice a day normally).
PREDICTIONS_STALE_HOURS = 20.0


def _connect_ro(db: Path) -> sqlite3.Connection | None:
    if not db.exists():
        return None
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _rows(conn: sqlite3.Connection | None, sql: str, args: tuple = ()) -> list[dict]:
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:  # table missing (ledger never initialised)
        return []


def _scalar(conn: sqlite3.Connection | None, sql: str):
    rows = _rows(conn, sql)
    return next(iter(rows[0].values())) if rows else None


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def create_app(root: Path | str = ROOT) -> FastAPI:
    root = Path(root)
    db = root / "data" / "paper.sqlite"
    app = FastAPI(title="fedge dashboard (read-only)", docs_url=None, redoc_url=None)

    def conn() -> sqlite3.Connection | None:
        return _connect_ro(db)

    def config() -> dict:
        try:
            return tomllib.loads((root / "config" / "strategy.toml").read_text("utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def latest_report() -> Path | None:
        files = sorted((root / "reports" / "weekly").glob("*.md"))
        return files[-1] if files else None

    def last_runs() -> dict:
        c = conn()
        try:
            try:
                state = json.loads((root / "data" / "last_run.json").read_text("utf-8"))
            except (OSError, ValueError):
                state = {}
            # Fallback evidence from the ledger itself when the wrapper has not logged a run.
            ledger_ts = {
                "picks": _scalar(c, "SELECT MAX(seen_utc) FROM snapshots"),
                "settle": _scalar(c, "SELECT MAX(settled_utc) FROM settlements"),
            }
        finally:
            if c:
                c.close()
        now = datetime.now(UTC)
        out = {}
        for job, limit in STALE_HOURS.items():
            s = state.get(job, {})
            ok = s.get("last_success_utc") or ledger_ts[job]
            d = _parse(ok)
            age = (now - d).total_seconds() / 3600 if d else None
            out[job] = {
                "last_success_utc": ok,
                "last_attempt_utc": s.get("last_attempt_utc"),
                "age_hours": None if age is None else round(age, 1),
                "stale_after_hours": limit,
                "stale": age is None or age > limit,
            }
        return out

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/ledger", include_in_schema=False)
    def ledger_page():
        """Paper-trading ledger view (the main page shows model predictions only)."""
        return FileResponse(STATIC / "ledger.html")

    @app.get("/style.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(STATIC / "style.css", media_type="text/css")

    @app.get("/api/summary")
    def summary():
        cfg = config()
        c = conn()
        try:
            n_open = _scalar(
                c,
                "SELECT COUNT(*) FROM bets b LEFT JOIN settlements s USING (bet_id) "
                "WHERE s.bet_id IS NULL",
            )
            agg = (
                _rows(
                    c,
                    "SELECT COUNT(*) AS n, COALESCE(SUM(pnl),0) AS pnl, AVG(log_clv) AS clv, "
                    "COALESCE(SUM(won),0) AS wins FROM settlements",
                )
                or [{}]
            )[0]
            n_bets = _scalar(c, "SELECT COUNT(*) FROM bets")
            n_snap = _scalar(c, "SELECT COUNT(*) FROM snapshots")
        finally:
            if c:
                c.close()
        rep = latest_report()
        runs = last_runs()
        return {
            "now_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "ledger_exists": db.exists(),
            "mode": cfg.get("mode"),
            "gate0_passed": cfg.get("gate0_passed"),
            "shadow_threshold": (cfg.get("shadow") or {}).get("shadow_threshold"),
            "markets": {
                k: {"model": v.get("model"), "qualified": v.get("qualified")}
                for k, v in (cfg.get("markets") or {}).items()
            },
            "counts": {
                "bets": n_bets or 0,
                "open": n_open or 0,
                "settled": agg.get("n") or 0,
                "snapshots": n_snap or 0,
            },
            "settled": {
                "pnl_units": round(agg.get("pnl") or 0.0, 4),
                "mean_log_clv": agg.get("clv"),
                "wins": agg.get("wins") or 0,
            },
            "jobs": runs,
            "any_stale": any(j["stale"] for j in runs.values()),
            "latest_report": rep.name if rep else None,
        }

    @app.get("/api/open_bets")
    def open_bets():
        c = conn()
        try:
            rows = _rows(
                c,
                "SELECT b.bet_id, b.kickoff_utc, b.div, b.home, b.away, b.market, b.selection, "
                "b.price_taken, b.edge, b.stake_units, b.mode FROM bets b "
                "LEFT JOIN settlements s USING (bet_id) WHERE s.bet_id IS NULL "
                "ORDER BY b.kickoff_utc LIMIT 200",
            )
        finally:
            if c:
                c.close()
        return {"rows": rows}

    @app.get("/api/settled")
    def settled():
        c = conn()
        try:
            rows = _rows(
                c,
                "SELECT s.settled_utc, b.div, b.home, b.away, b.market, b.selection, "
                "b.price_taken, s.closing_price, s.result, s.won, b.stake_units, s.pnl, "
                "s.log_clv FROM settlements s JOIN bets b USING (bet_id) "
                "ORDER BY s.settled_utc DESC LIMIT 200",
            )
        finally:
            if c:
                c.close()
        return {"rows": rows}

    @app.get("/api/series")
    def series():
        """Cumulative P&L and cumulative mean CLV in settlement order."""
        c = conn()
        try:
            rows = _rows(
                c,
                "SELECT settled_utc, pnl, log_clv FROM settlements ORDER BY settled_utc, bet_id",
            )
        finally:
            if c:
                c.close()
        pts, cum, clv_sum, clv_n = [], 0.0, 0.0, 0
        for i, r in enumerate(rows, 1):
            cum += r["pnl"] or 0.0
            if r["log_clv"] is not None:
                clv_sum += r["log_clv"]
                clv_n += 1
            pts.append(
                {
                    "n": i,
                    "t": r["settled_utc"],
                    "cum_pnl": round(cum, 4),
                    "mean_clv": (clv_sum / clv_n) if clv_n else None,
                }
            )
        return {"points": pts}

    @app.get("/api/snapshots")
    def snapshots():
        """Highest-edge rows from the most recent snapshot day."""
        c = conn()
        try:
            rows = _rows(
                c,
                "SELECT snapshot_day, seen_utc, kickoff_utc, div, home, away, market, selection, "
                "price, model_prob, pooled_prob, edge FROM snapshots "
                "WHERE snapshot_day = (SELECT MAX(snapshot_day) FROM snapshots) "
                "ORDER BY edge DESC LIMIT 25",
            )
        finally:
            if c:
                c.close()
        return {"rows": rows}

    @app.get("/api/predictions")
    def predictions():
        """Upcoming Premier League / La Liga predictions from the ``predictions`` table.

        Written by ``scripts/predict_upcoming.py`` (and the paper-picks run), read-only here.
        """
        c = conn()
        try:
            rows = _rows(
                c,
                "SELECT * FROM predictions ORDER BY kickoff_utc, home",
            )
        finally:
            if c:
                c.close()
        tz = ZoneInfo("Australia/Sydney")
        last = None
        out = []
        for r in rows:
            ko = _parse(r["kickoff_utc"])
            last = max(last, r["run_ts"]) if last else r["run_ts"]
            try:
                detail = json.loads(r["detail"]) if r["detail"] else {}
                top = json.loads(r["top_scores"]) if r["top_scores"] else []
            except ValueError:
                detail, top = {}, []
            # Model output only: the market/price columns and the scoreline grid are deliberately
            # not exposed, because the dashboard shows the model alone (no bookmaker odds).
            out.append(
                {
                    "match_key": r["match_key"],
                    "status": r["status"],
                    "reason": r["reason"],
                    "div": r["div"],
                    "league": r["league"],
                    "home": r["home"],
                    "away": r["away"],
                    "kickoff_utc": r["kickoff_utc"],
                    "kickoff_local": (
                        ko.astimezone(tz).isoformat(timespec="minutes") if ko else None
                    ),
                    "probs": None if r["p_home"] is None else {
                        "home": r["p_home"], "draw": r["p_draw"], "away": r["p_away"],
                    },
                    "xg": None if r["xg_home"] is None else {
                        "home": r["xg_home"], "away": r["xg_away"],
                    },
                    "p_over25": r["p_over25"],
                    "p_btts": r["p_btts"],
                    "top_scores": top,
                    "detail": detail,
                }
            )
        age = None
        if last:
            d = _parse(last)
            age = None if d is None else (datetime.now(UTC) - d).total_seconds() / 3600
        return {
            "rows": out,
            "last_updated_utc": last,
            "age_hours": None if age is None else round(age, 1),
            "stale_after_hours": PREDICTIONS_STALE_HOURS,
            "stale": age is None or age > PREDICTIONS_STALE_HOURS,
            "scope": "Premier League (E0) + La Liga (SP1) only",
        }

    @app.get("/api/report/latest")
    def report():
        rep = latest_report()
        if rep is None:
            return PlainTextResponse("no weekly report yet", status_code=404)
        return PlainTextResponse(rep.read_text("utf-8"), media_type="text/plain; charset=utf-8")

    return app


app = create_app()
