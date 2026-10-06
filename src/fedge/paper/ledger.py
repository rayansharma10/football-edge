"""SQLite paper-bet ledger: bets, settlements and price snapshots.

The desk is credential-free and paper-only (AGENTS.md rule 2). Three tables in
``data/paper.sqlite``:

``bets``
    One row per placed *paper* or *shadow* bet. ``bet_id`` is a content hash of
    ``(match_key, market, selection)`` and a unique index on that triple makes double-betting a
    match/market/selection impossible: :func:`insert_bets` skips anything already in the table and
    returns only the rows it actually wrote (idempotency).
``settlements``
    One row per settled bet: the match result, ``pnl`` after 6% commission on net winnings, the
    closing price of the selection (``BFEC*`` preferred, else ``AvgC*``) and the log-CLV against
    the margin-free closing probability (:mod:`fedge.backtest.stats`).
``snapshots``
    One row per (fixture, market, selection) price snapshot *seen*, bet or not, so the live CLV of
    the model's view can be measured on the whole fixture list and not only on the bets. Unique per
    ``(snapshot_day, match_key, market, selection)``: re-running the picker on the same day with the
    same fixture file records nothing new, while the next football-data refresh (a new day) adds a
    fresh row for the new price.

Every timestamp written is tz-aware UTC ISO-8601 (AGENTS.md rule 5); the Sydney conversion happens
only when a digest or report is rendered.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.backtest.stats import net_odds

BET_COLUMNS = (
    "bet_id",
    "created_utc",
    "match_key",
    "div",
    "date",
    "home",
    "away",
    "kickoff_utc",
    "market",
    "selection",
    "model_prob",
    "market_prob",
    "pooled_prob",
    "price_taken",
    "price_source",
    "edge",
    "stake_units",
    "stake_frac",
    "mode",
    "config_hash",
    "gate0_hash",
    "experiment_id",
)

SNAPSHOT_COLUMNS = (
    "seen_utc",
    "snapshot_day",
    "match_key",
    "div",
    "date",
    "home",
    "away",
    "kickoff_utc",
    "market",
    "selection",
    "price",
    "price_source",
    "model_prob",
    "pooled_prob",
    "edge",
)

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    bet_id       TEXT PRIMARY KEY,
    created_utc  TEXT NOT NULL,
    match_key    TEXT NOT NULL,
    div          TEXT NOT NULL,
    date         TEXT NOT NULL,
    home         TEXT NOT NULL,
    away         TEXT NOT NULL,
    kickoff_utc  TEXT NOT NULL,
    market       TEXT NOT NULL,
    selection    TEXT NOT NULL,
    model_prob   REAL,
    market_prob  REAL,
    pooled_prob  REAL,
    price_taken  REAL NOT NULL,
    price_source TEXT NOT NULL,
    edge         REAL NOT NULL,
    stake_units  REAL NOT NULL,
    stake_frac   REAL NOT NULL,
    mode         TEXT NOT NULL CHECK (mode IN ('paper', 'shadow')),
    config_hash  TEXT NOT NULL,
    gate0_hash   TEXT NOT NULL,
    experiment_id TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS bets_match ON bets (match_key, market, selection);
CREATE INDEX IF NOT EXISTS bets_created ON bets (created_utc);
CREATE INDEX IF NOT EXISTS bets_mode ON bets (mode);

CREATE TABLE IF NOT EXISTS settlements (
    bet_id         TEXT PRIMARY KEY REFERENCES bets (bet_id),
    settled_utc    TEXT NOT NULL,
    result         TEXT NOT NULL,
    won            INTEGER NOT NULL,
    pnl            REAL NOT NULL,
    closing_price  REAL,
    closing_source TEXT,
    log_clv        REAL,
    log_clv_raw    REAL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    seen_utc     TEXT NOT NULL,
    snapshot_day TEXT NOT NULL,
    match_key    TEXT NOT NULL,
    div          TEXT,
    date         TEXT,
    home         TEXT,
    away         TEXT,
    kickoff_utc  TEXT,
    market       TEXT NOT NULL,
    selection    TEXT NOT NULL,
    price        REAL,
    price_source TEXT,
    model_prob   REAL,
    pooled_prob  REAL,
    edge         REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS snapshots_seen
    ON snapshots (snapshot_day, match_key, market, selection);
"""


def default_path(data_dir: Path | str = "data") -> Path:
    """``data/paper.sqlite`` (``data/`` is gitignored)."""
    return Path(data_dir) / "paper.sqlite"


def bet_id(match_key: str, market: str, selection: str) -> str:
    """Deterministic id of a (match, market, selection) bet: sha1 of the triple, 16 hex chars."""
    key = f"{match_key}|{market}|{selection}".encode()
    return hashlib.sha1(key).hexdigest()[:16]


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating) the ledger database and make sure the schema exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection, commission: float = 0.06) -> None:
    """Bring an older ledger up to ``SCHEMA_VERSION`` (tracked in ``PRAGMA user_version``).

    v0/v1 -> v2: ``settlements.log_clv`` used to hold the RAW log-CLV. It moves to the new
    ``log_clv_raw`` column and ``log_clv`` is recomputed NET of commission from ``price_taken``.
    """
    ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if ver > SCHEMA_VERSION:
        raise RuntimeError(f"ledger schema v{ver} is newer than this code (v{SCHEMA_VERSION})")
    if ver == SCHEMA_VERSION:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(settlements)")}
    if "log_clv_raw" not in cols:
        conn.execute("ALTER TABLE settlements ADD COLUMN log_clv_raw REAL")
        rows = conn.execute(
            "SELECT s.bet_id, s.log_clv, b.price_taken FROM settlements s "
            "JOIN bets b USING (bet_id) WHERE s.log_clv IS NOT NULL"
        ).fetchall()
        for bet_id_, raw, price in rows:
            net = raw + float(np.log(float(net_odds(price, commission)) / price))
            conn.execute(
                "UPDATE settlements SET log_clv_raw = ?, log_clv = ? WHERE bet_id = ?",
                (raw, net, bet_id_),
            )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def insert_bets(conn: sqlite3.Connection, rows: Iterable[dict]) -> list[dict]:
    """Insert bets, skipping any (match, market, selection) already recorded.

    Returns the subset of ``rows`` that was actually written, so callers can report only genuinely
    new bets (idempotency: re-running the picker with the same fixtures writes nothing).
    """
    rows = list(rows)
    if not rows:
        return []
    keys = {(r["match_key"], r["market"], r["selection"]) for r in rows}
    have = existing_triples(conn, keys)
    fresh = [r for r in rows if (r["match_key"], r["market"], r["selection"]) not in have]
    if not fresh:
        return []
    conn.executemany(
        f"INSERT INTO bets ({', '.join(BET_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(BET_COLUMNS))})",
        [tuple(r[c] for c in BET_COLUMNS) for r in fresh],
    )
    conn.commit()
    return fresh


def existing_triples(conn: sqlite3.Connection, triples: Iterable[tuple[str, str, str]]) -> set:
    """Which ``(match_key, market, selection)`` triples are already in ``bets``."""
    out: set[tuple[str, str, str]] = set()
    keys = list(triples)
    for i in range(0, len(keys), 200):
        chunk = keys[i : i + 200]
        marks = ",".join("(?,?,?)" for _ in chunk)
        params = [v for k in chunk for v in k]
        sql = (
            "SELECT match_key, market, selection FROM bets "
            f"WHERE (match_key, market, selection) IN ({marks})"
        )
        out |= {(r[0], r[1], r[2]) for r in conn.execute(sql, params)}
    return out


def insert_snapshots(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Insert snapshot rows, skipping ones already recorded for the same day; returns the count."""
    rows = list(rows)
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        f"INSERT OR IGNORE INTO snapshots ({', '.join(SNAPSHOT_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(SNAPSHOT_COLUMNS))})",
        [tuple(r[c] for c in SNAPSHOT_COLUMNS) for r in rows],
    )
    conn.commit()
    return conn.total_changes - before


def record_settlements(conn: sqlite3.Connection, rows: Iterable[dict]) -> list[dict]:
    """Insert settlement rows, skipping bets already settled; returns the rows actually written."""
    rows = list(rows)
    if not rows:
        return []
    marks = ",".join("?" * len(rows))
    have = {
        r[0]
        for r in conn.execute(
            f"SELECT bet_id FROM settlements WHERE bet_id IN ({marks})",
            [r["bet_id"] for r in rows],
        )
    }
    fresh = [r for r in rows if r["bet_id"] not in have]
    if not fresh:
        return []
    cols = (
        "bet_id",
        "settled_utc",
        "result",
        "won",
        "pnl",
        "closing_price",
        "closing_source",
        "log_clv",
        "log_clv_raw",
    )
    conn.executemany(
        f"INSERT INTO settlements ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r[c] if c in r.keys() else None for c in cols) for r in fresh],
    )
    conn.commit()
    return fresh


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Whole table as a DataFrame (empty frame with the right columns when the table is empty)."""
    if table not in ("bets", "settlements", "snapshots"):
        raise ValueError(f"unknown table {table!r}")
    return pd.read_sql_query(f"SELECT * FROM {table}", conn)  # noqa: S608 (fixed whitelist)


def bets_with_settlements(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every bet left-joined with its settlement (``pnl`` / ``log_clv`` NaN while unsettled)."""
    return pd.read_sql_query(
        """
        SELECT b.*, s.settled_utc, s.result, s.won AS settled_won, s.pnl,
               s.closing_price, s.closing_source, s.log_clv, s.log_clv_raw
        FROM bets b LEFT JOIN settlements s ON s.bet_id = b.bet_id
        ORDER BY b.kickoff_utc, b.match_key, b.market
        """,
        conn,
    )


def unsettled_bets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Bets with no settlement row yet."""
    return pd.read_sql_query(
        """
        SELECT b.* FROM bets b
        LEFT JOIN settlements s ON s.bet_id = b.bet_id
        WHERE s.bet_id IS NULL
        ORDER BY b.kickoff_utc, b.match_key
        """,
        conn,
    )


def bets_placed_on(conn: sqlite3.Connection, day: str) -> int:
    """How many bets were created on the UTC day ``YYYY-MM-DD`` (used by the daily cap)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM bets WHERE substr(created_utc, 1, 10) = ?", (day,)
    ).fetchone()
    return int(row[0])


def settled_frame(conn: sqlite3.Connection, mode: str | None = None) -> pd.DataFrame:
    """Settled bets (optionally only ``paper`` or ``shadow``) with their settlement columns."""
    df = bets_with_settlements(conn)
    df = df[df["pnl"].notna()]
    if mode is not None:
        df = df[df["mode"] == mode]
    return df


def mode_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """``{'paper': n, 'shadow': m}`` bet counts by mode."""
    rows = conn.execute("SELECT mode, COUNT(*) FROM bets GROUP BY mode").fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


def settled_mode_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Settled bet counts by mode."""
    rows = conn.execute(
        "SELECT b.mode, COUNT(*) FROM bets b JOIN settlements s USING (bet_id) GROUP BY b.mode"
    ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}

