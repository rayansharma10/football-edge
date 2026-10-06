"""Score upcoming fixtures into the ``predictions`` table of ``data/paper.sqlite``.

The writer is :func:`write_predictions` (called by ``scripts/predict_upcoming.py`` and, best
effort, at the end of ``scripts/paper_picks.py``); the dashboard only ever reads the table.
Nothing here touches bets, snapshots, stakes or the stdout digest.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from fedge.models import dixon_coles as dc
from fedge.predict.scoreline import predict_match

DIV_NAMES = {
    "E0": "England Premier League", "E1": "England Championship", "E2": "England League One",
    "E3": "England League Two", "SC0": "Scotland Premiership", "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga", "I1": "Italy Serie A", "I2": "Italy Serie B",
    "SP1": "Spain La Liga", "SP2": "Spain Segunda", "F1": "France Ligue 1", "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie", "B1": "Belgium Pro League", "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig", "G1": "Greece Super League",
}

# Desk scope (operator): Premier League + La Liga only; everything else is dropped.
DESK_DIVS = ("E0", "SP1")

def load_desk_divs(path="config/leagues.toml") -> tuple[str, ...]:
    """``desk_divisions`` from the leagues config (falls back to :data:`DESK_DIVS`)."""
    import tomllib

    try:
        with open(path, "rb") as f:
            return tuple(map(str, tomllib.load(f).get("desk_divisions", DESK_DIVS)))
    except OSError:
        return DESK_DIVS


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    match_key    TEXT PRIMARY KEY,
    run_ts       TEXT NOT NULL,
    status       TEXT NOT NULL,          -- 'modelled' | 'not_modelled'
    reason       TEXT,
    div          TEXT,
    league       TEXT,
    home         TEXT,
    away         TEXT,
    kickoff_utc  TEXT,
    p_home       REAL, p_draw REAL, p_away REAL,          -- lgbd_xg 1X2 (market-anchored)
    mkt_home     REAL, mkt_draw REAL, mkt_away REAL,      -- de-vigged snapshot price
    xg_home      REAL, xg_away REAL,
    p_over25     REAL, p_btts REAL,
    top_scores   TEXT,                   -- JSON [{score,p}, ...] (top 5)
    grid         TEXT,                   -- JSON 6x6, rows = home goals 0-5
    detail       TEXT                    -- JSON: method, lambdas, DC 1X2, fallback flags
);
CREATE INDEX IF NOT EXISTS predictions_kickoff ON predictions (kickoff_utc);
"""

COLS = [
    "match_key", "run_ts", "status", "reason", "div", "league", "home", "away", "kickoff_utc",
    "p_home", "p_draw", "p_away", "mkt_home", "mkt_draw", "mkt_away", "xg_home", "xg_away",
    "p_over25", "p_btts", "top_scores", "grid", "detail",
]


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(ts) -> str:
    return _utc(ts).isoformat(timespec="seconds")


def dc_lambdas(params: dc.DCParams, home: str, away: str) -> tuple[float, float, bool]:
    """Expected goals (home, away) from fitted DC params; True when a team had no history."""
    idx = {t: i for i, t in enumerate(params.teams)}
    unknown = home not in idx or away not in idx
    ah = params.attack[idx[home]] if home in idx else dc.PRIOR_ATTACK
    dh = params.defence[idx[home]] if home in idx else dc.PRIOR_DEFENCE
    aa = params.attack[idx[away]] if away in idx else dc.PRIOR_ATTACK
    da = params.defence[idx[away]] if away in idx else dc.PRIOR_DEFENCE
    return float(np.exp(ah + da + params.hfa)), float(np.exp(aa + dh)), unknown


def fit_divisions(played: pd.DataFrame, divs, ref) -> dict:
    """Per-division DC fit (None when the data are insufficient)."""
    return {
        d: dc.fit_dc(played.loc[played["div"] == d], ref=ref) for d in sorted(set(map(str, divs)))
    }


def _league_means(played: pd.DataFrame, div: str, ref) -> tuple[float, float]:
    g = played.loc[(played["div"] == div) & (played["kickoff_utc"] < ref)].tail(380)
    if g.empty:
        return 1.5, 1.2
    return float(g["FTHG"].astype(float).mean()), float(g["FTAG"].astype(float).mean())


def _target_frames(edges: pd.DataFrame):
    e = edges.loc[edges["market"] == "1x2"]
    model = e.pivot_table(index="match_key", columns="selection", values="model_prob")
    mkt = e.pivot_table(index="match_key", columns="selection", values="market_prob")
    return model, mkt


def _modelled_row(f, tgt, mk, params, played, now, run_ts) -> dict:
    fallback = None
    if params is None:
        lh, la = _league_means(played, f.div, now)
        rho = 0.0
        fallback = "league-average goals (no DC fit)"
    else:
        lh, la, unk = dc_lambdas(params, f.home, f.away)
        rho = params.rho
        if unk:
            fallback = "team without history: DC prior"
    pm = predict_match(lh, la, rho, tgt)
    detail = {
        "method": pm["method"],
        "lambda_dc": [lh, la],
        "lambda_used": [pm["lambda_home"], pm["lambda_away"]],
        "rho": rho,
        "dc_1x2": pm["dc_p"],
        "fallback": fallback,
        "grid_mass_shown": pm["grid_mass_shown"],
        "model": "lgbd_xg (market-anchored: initialised from the de-margined price)",
    }
    return {
        "match_key": f.match_key, "run_ts": _iso(run_ts), "status": "modelled", "reason": None,
        "div": f.div, "league": DIV_NAMES.get(f.div, f.div), "home": f.home, "away": f.away,
        "kickoff_utc": _iso(f.kickoff_utc),
        "p_home": pm["p_home"], "p_draw": pm["p_draw"], "p_away": pm["p_away"],
        "mkt_home": float(mk[0]), "mkt_draw": float(mk[1]), "mkt_away": float(mk[2]),
        "xg_home": pm["xg_home"], "xg_away": pm["xg_away"],
        "p_over25": pm["p_over25"], "p_btts": pm["p_btts"],
        "top_scores": json.dumps([{"score": t["score"], "p": t["p"]} for t in pm["top_scores"]]),
        "grid": json.dumps(pm["grid"]),
        "detail": json.dumps(detail),
    }


def _unmodelled(key, run_ts, div, league, home, away, kickoff, reason) -> dict:
    r = dict.fromkeys(COLS)
    r.update(match_key=key, run_ts=_iso(run_ts), status="not_modelled", reason=reason, div=div,
             league=league, home=str(home), away=str(away), kickoff_utc=_iso(kickoff))
    return r


def build_rows(
    edges: pd.DataFrame,
    fixtures: pd.DataFrame,
    played: pd.DataFrame,
    run_ts,
    now=None,
    desk_divs=DESK_DIVS,
) -> list[dict]:
    """One dict per upcoming desk-league fixture (Premier League + La Liga only).

    ``edges`` is the output of :func:`fedge.paper.model_state.score_all` (only its 1x2 rows are
    used); ``fixtures`` every parsed fixture. Other divisions are dropped entirely; a desk
    fixture that could not be scored (no complete price) is kept as a ``not_modelled`` row so
    it is never silently missing.
    """
    run_ts = _utc(run_ts)
    now = _utc(now) if now is not None else run_ts
    fixtures = fixtures.loc[fixtures["div"].isin(set(desk_divs))]
    rows: list[dict] = []
    if len(edges):
        model_p, mkt_p = _target_frames(edges)
    else:
        model_p = mkt_p = pd.DataFrame()
    todo = (
        fixtures.loc[fixtures["match_key"].isin(model_p.index) & (fixtures["kickoff_utc"] > now)]
        if len(model_p) else fixtures.iloc[0:0]
    )
    fits = fit_divisions(played, todo["div"], now) if len(todo) else {}
    for f in todo.itertuples(index=False):
        tgt = model_p.loc[f.match_key, ["H", "D", "A"]].to_numpy(dtype=float)
        mk = mkt_p.loc[f.match_key, ["H", "D", "A"]].to_numpy(dtype=float)
        rows.append(_modelled_row(f, tgt, mk, fits.get(f.div), played, now, run_ts))
    done = {r["match_key"] for r in rows}
    for f in fixtures.itertuples(index=False):
        if f.match_key in done or _utc(f.kickoff_utc) <= now:
            continue
        rows.append(_unmodelled(
            f.match_key, run_ts, f.div, DIV_NAMES.get(f.div, f.div), f.home, f.away,
            f.kickoff_utc, "no complete 1X2 price in the fixture file",
        ))
    rows.sort(key=lambda r: (r["kickoff_utc"], r["home"]))
    return rows


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def write_predictions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Replace the table contents with ``rows`` in one transaction (the set of upcoming
    fixtures changes every run, so stale rows must go)."""
    ensure_table(conn)
    with conn:
        conn.execute("DELETE FROM predictions WHERE 1=1")
        conn.executemany(
            f"INSERT OR REPLACE INTO predictions ({','.join(COLS)}) "
            f"VALUES ({','.join('?' * len(COLS))})",
            [tuple(r.get(c) for c in COLS) for r in rows],
        )
    return len(rows)
