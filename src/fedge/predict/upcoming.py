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
    detail       TEXT,                   -- JSON: method, lambdas, DC 1X2, fallback flags
    model_kind   TEXT                    -- 'lgbd_xg_priced' | 'lgb_xg_price_free'
);
CREATE INDEX IF NOT EXISTS predictions_kickoff ON predictions (kickoff_utc);
CREATE TABLE IF NOT EXISTS prediction_meta (key TEXT PRIMARY KEY, value TEXT);
"""

KIND_PRICED = "lgbd_xg_priced"
KIND_PRICE_FREE = "lgb_xg_price_free"

COLS = [
    "match_key", "run_ts", "status", "reason", "div", "league", "home", "away", "kickoff_utc",
    "p_home", "p_draw", "p_away", "mkt_home", "mkt_draw", "mkt_away", "xg_home", "xg_away",
    "p_over25", "p_btts", "top_scores", "grid", "detail", "model_kind",
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


def _modelled_row(f, tgt, mk, params, played, now, run_ts, kind=KIND_PRICED) -> dict:
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
        "model": (
            "lgbd_xg (market-anchored: initialised from the de-margined price)"
            if kind == KIND_PRICED
            else "lgb_xg (price-free: no market feature; today's ratings and form)"
        ),
    }
    return {
        "match_key": f.match_key, "run_ts": _iso(run_ts), "status": "modelled", "reason": None,
        "div": f.div, "league": DIV_NAMES.get(f.div, f.div), "home": f.home, "away": f.away,
        "kickoff_utc": _iso(f.kickoff_utc),
        "p_home": pm["p_home"], "p_draw": pm["p_draw"], "p_away": pm["p_away"],
        "mkt_home": None if mk is None else float(mk[0]),
        "mkt_draw": None if mk is None else float(mk[1]),
        "mkt_away": None if mk is None else float(mk[2]),
        "xg_home": pm["xg_home"], "xg_away": pm["xg_away"],
        "p_over25": pm["p_over25"], "p_btts": pm["p_btts"],
        "top_scores": json.dumps([{"score": t["score"], "p": t["p"]} for t in pm["top_scores"]]),
        "grid": json.dumps(pm["grid"]),
        "detail": json.dumps(detail),
        "model_kind": kind,
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
    price_free: pd.DataFrame | None = None,
) -> list[dict]:
    """One dict per upcoming desk-league fixture (Premier League + La Liga only).

    ``edges`` is the output of :func:`fedge.paper.model_state.score_all` (only its 1x2 rows are
    used); ``fixtures`` every known fixture (priced or schedule-only). ``price_free`` is an
    optional ``match_key`` x H/D/A frame of price-free 1X2 probabilities for fixtures that have no
    price: those become ``lgb_xg_price_free`` rows. Other divisions are dropped entirely; a desk
    fixture with neither a price nor a price-free estimate is kept as a ``not_modelled`` row so it
    is never silently missing.
    """
    run_ts = _utc(run_ts)
    now = _utc(now) if now is not None else run_ts
    fixtures = fixtures.loc[fixtures["div"].isin(set(desk_divs))]
    rows: list[dict] = []
    if len(edges):
        model_p, mkt_p = _target_frames(edges)
    else:
        model_p = mkt_p = pd.DataFrame()
    pf = price_free if price_free is not None else pd.DataFrame()
    future = fixtures.loc[fixtures["kickoff_utc"] > now]
    priced = (
        future.loc[future["match_key"].isin(model_p.index)] if len(model_p) else future.iloc[0:0]
    )
    free = future.loc[
        ~future["match_key"].isin(set(priced["match_key"])) & future["match_key"].isin(pf.index)
    ]
    todo = pd.concat([priced, free])
    fits = fit_divisions(played, todo["div"], now) if len(todo) else {}
    for f in priced.itertuples(index=False):
        tgt = model_p.loc[f.match_key, ["H", "D", "A"]].to_numpy(dtype=float)
        mk = mkt_p.loc[f.match_key, ["H", "D", "A"]].to_numpy(dtype=float)
        rows.append(_modelled_row(f, tgt, mk, fits.get(f.div), played, now, run_ts))
    for f in free.itertuples(index=False):
        tgt = pf.loc[f.match_key, ["H", "D", "A"]].to_numpy(dtype=float)
        rows.append(
            _modelled_row(f, tgt, None, fits.get(f.div), played, now, run_ts, KIND_PRICE_FREE)
        )
    done = {r["match_key"] for r in rows}
    for f in future.itertuples(index=False):
        if f.match_key in done:
            continue
        rows.append(_unmodelled(
            f.match_key, run_ts, f.div, DIV_NAMES.get(f.div, f.div), f.home, f.away,
            f.kickoff_utc, "no price and no price-free estimate",
        ))
    rows.sort(key=lambda r: (r["kickoff_utc"], r["home"]))
    return rows


def full_fixture_list(priced_fixtures: pd.DataFrame, data_dir, divs, now, refresh=True, delay=1.0):
    """Schedule-backed fixture list: ``(fixtures, ScheduleResult)``.

    The full-season schedule (:mod:`fedge.ingest.schedule`) with football-data's priced fixtures
    merged over it. Never raises: when no schedule is available the priced fixtures stand alone.
    """
    from fedge.ingest import schedule as sch

    try:
        res = sch.load_schedule(tuple(divs), data_dir, now=now, refresh=refresh, delay=delay)
    except Exception as exc:  # defensive: the picks run must not fail on a display feature
        res = sch.ScheduleResult(
            fixtures=pd.DataFrame(columns=sch.COLUMNS), stale=True,
            warnings=[f"schedule unavailable ({type(exc).__name__}: {exc})"],
        )
    desk_priced = priced_fixtures.loc[priced_fixtures["div"].isin(set(map(str, divs)))]
    sched_fx = sch.to_fixtures(sch.upcoming(res.fixtures, now))
    return sch.merge_priced(sched_fx, desk_priced), res


def score_missing_prices(
    data_dir, played, fixtures, edges, now, divs, threads=None
) -> pd.DataFrame:
    """Price-free 1X2 for upcoming desk fixtures that have no priced edge (best effort)."""
    from fedge.paper import model_state as ms

    have = set(edges.loc[edges["market"] == "1x2", "match_key"]) if len(edges) else set()
    miss = fixtures.loc[
        fixtures["div"].isin(set(map(str, divs)))
        & (fixtures["kickoff_utc"] > _utc(now))
        & ~fixtures["match_key"].isin(have)
    ]
    if miss.empty:
        return pd.DataFrame(columns=["H", "D", "A"])
    return ms.score_price_free(data_dir, played, miss, now, threads or ms.THREADS)


def build_all(
    data_dir, played, priced_fixtures, edges, now, divs, refresh=True, delay=1.0, threads=None
):
    """Whole display pipeline: ``(rows, meta, schedule_result)``.

    Full-season schedule + priced fixtures -> priced rows from ``edges`` (lgbd_xg) and price-free
    rows (lgb_xg) for every other upcoming fixture. Nothing here touches bets or stdout.
    """
    fixtures, res = full_fixture_list(priced_fixtures, data_dir, divs, now, refresh, delay)
    price_free = score_missing_prices(data_dir, played, fixtures, edges, now, divs, threads)
    rows = build_rows(edges, fixtures, played, now, now, desk_divs=divs, price_free=price_free)
    return rows, res.meta, res


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    if "model_kind" not in have:  # table created by an older version
        conn.execute("ALTER TABLE predictions ADD COLUMN model_kind TEXT")


def write_predictions(
    conn: sqlite3.Connection, rows: list[dict], meta: dict[str, str] | None = None
) -> int:
    """Replace the table contents with ``rows`` in one transaction (the set of upcoming
    fixtures changes every run, so stale rows must go). ``meta`` (schedule source, staleness,
    warnings) replaces ``prediction_meta`` when given."""
    ensure_table(conn)
    with conn:
        conn.execute("DELETE FROM predictions WHERE 1=1")
        conn.executemany(
            f"INSERT OR REPLACE INTO predictions ({','.join(COLS)}) "
            f"VALUES ({','.join('?' * len(COLS))})",
            [tuple(r.get(c) for c in COLS) for r in rows],
        )
        if meta is not None:
            conn.execute("DELETE FROM prediction_meta WHERE 1=1")
            conn.executemany(
                "INSERT INTO prediction_meta (key, value) VALUES (?, ?)", list(meta.items())
            )
    return len(rows)
