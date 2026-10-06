"""Accuracy report: ``prediction_log`` x ``results`` -> metrics, baselines, breakdowns.

Pure functions over DataFrames (no writes). :func:`load_frames` reads the telemetry DB
(``mode=ro``); :func:`build_report` turns the frames into the JSON the dashboard serves.

Snapshots: ``final`` = the latest ``run`` row before kickoff (view ``v_final``); ``d7`` / ``d3`` =
the first run inside the 7 / 3 day window. The headline is ``final``. Retrospective rows
(``source = 'retro'``) are excluded unless ``include_retro``.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from fedge.accuracy import baselines as bl
from fedge.accuracy import metrics as mt

KINDS = ("d7", "d3", "final")
RECENT_LIMIT = 60

_SQL = """
SELECT p.*, r.home_goals, r.away_goals, r.result, r.source AS result_source
FROM {table} p JOIN results r USING (match_key)
"""


def load_frames(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """``{kind: frame}`` of predictions joined with finished results (kind in d7 / d3 / final)."""
    out = {}
    for kind in KINDS:
        if kind == "final":
            sql = _SQL.format(table="v_final")
        else:
            sql = _SQL.format(table="prediction_log") + f" WHERE p.snapshot_kind = '{kind}'"
        try:
            df = pd.read_sql_query(sql, conn)
        except Exception:  # tables not created yet
            df = pd.DataFrame()
        out[kind] = df
    return out


def _prep(df: pd.DataFrame) -> dict:
    P = df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    hg = df["home_goals"].to_numpy(dtype=int)
    ag = df["away_goals"].to_numpy(dtype=int)
    grids = np.array([json.loads(g) for g in df["grid"]], dtype=float)
    return {
        "P": P, "y": mt.outcome_index(hg, ag), "hg": hg, "ag": ag, "grids": grids,
        "pred_idx": np.argmax(P, axis=1),
        "ph": df["pred_score_home"].to_numpy(dtype=int),
        "pa": df["pred_score_away"].to_numpy(dtype=int),
        "xgh": df["xg_home"].to_numpy(dtype=float), "xga": df["xg_away"].to_numpy(dtype=float),
        "p_over": df["p_over25"].to_numpy(dtype=float),
        "p_btts": df["p_btts"].to_numpy(dtype=float),
        "clusters": df["match_key"].to_numpy(),
    }


def _ci(values, clusters, n_boot: int):
    lo, hi = mt.bootstrap_ci(values, clusters, n_boot=n_boot, seed=0)
    return [lo, hi]


def _hits_topk(grids, hg, ag, k: int) -> np.ndarray:
    return np.array([mt.topk_hit_rate(grids[i:i + 1], hg[i:i + 1], ag[i:i + 1], k)
                     for i in range(len(hg))])


def slice_metrics(df: pd.DataFrame, history: pd.DataFrame | None, n_boot: int = 1000) -> dict:
    """Every metric (and the baselines) for the rows of ``df``."""
    n = len(df)
    base = {"n": n, "small_sample": n < mt.SMALL_N}
    if n == 0:
        return base
    d = _prep(df)
    P, y, hg, ag, grids = d["P"], d["y"], d["hg"], d["ag"], d["grids"]
    win_hits = (d["pred_idx"] == y).astype(float)
    exact_hits = ((d["ph"] == hg) & (d["pa"] == ag)).astype(float)
    top3 = _hits_topk(grids, hg, ag, 3)
    top5 = _hits_topk(grids, hg, ag, 5)
    gd_hits = ((d["ph"] - d["pa"]) == (hg - ag)).astype(float)
    over = (hg + ag > mt.OU_LINE).astype(float)
    btts = ((hg > 0) & (ag > 0)).astype(float)
    ou_acc, ou_brier = mt.binary_accuracy_brier(d["p_over"], over)
    bt_acc, bt_brier = mt.binary_accuracy_brier(d["p_btts"], btts)
    cl = d["clusters"]
    out = {
        **base,
        "mean_hours_before_kickoff": float(df["hours_before_kickoff"].mean()),
        "winner": {
            "hit_rate": float(win_hits.mean()), "hit_rate_ci": _ci(win_hits, cl, n_boot),
            "brier": mt.brier_multiclass(P, y), "log_loss": mt.log_loss(P, y), "rps": mt.rps(P, y),
            "draw_predicted_rate": mt.draw_predicted_rate(d["pred_idx"]),
            "draw_actual_rate": float(np.mean(y == 1)),
            "confusion": mt.confusion_matrix(d["pred_idx"], y),
            "calibration": mt.calibration_table(P, y, 10),
        },
        "score": {
            "exact_hit_rate": float(exact_hits.mean()),
            "exact_hit_rate_ci": _ci(exact_hits, cl, n_boot),
            "top3_hit_rate": float(top3.mean()), "top3_hit_rate_ci": _ci(top3, cl, n_boot),
            "top5_hit_rate": float(top5.mean()), "top5_hit_rate_ci": _ci(top5, cl, n_boot),
            "goal_diff_hit_rate": float(gd_hits.mean()),
            "goal_diff_hit_rate_ci": _ci(gd_hits, cl, n_boot),
            "mae_total_goals": mt.mae(d["xgh"] + d["xga"], hg + ag),
            "mae_home_goals": mt.mae(d["xgh"], hg), "mae_away_goals": mt.mae(d["xga"], ag),
            "mean_log_score": mt.mean_log_score(grids, hg, ag),
            "mean_actual_score_prob": mt.mean_actual_score_prob(grids, hg, ag),
            "over25_accuracy": ou_acc, "over25_brier": ou_brier,
            "btts_accuracy": bt_acc, "btts_brier": bt_brier,
        },
    }
    if history is not None and len(history):
        out["baselines"] = _baselines(df, d, history, n_boot)
    return out


def _baselines(df: pd.DataFrame, d: dict, history: pd.DataFrame, n_boot: int) -> dict:
    frames = bl.baseline_frames(df[["div", "home", "away", "kickoff_utc"]], history)
    y, hg, ag, cl = d["y"], d["hg"], d["ag"], d["clusters"]
    out = {}
    for name, f in frames.items():
        pred_idx = np.argmax(f["P"], axis=1)
        wh = (pred_idx == y).astype(float)
        e = {"winner_hit_rate": float(wh.mean()), "winner_hit_rate_ci": _ci(wh, cl, n_boot),
             "brier": mt.brier_multiclass(f["P"], y)}
        if name != "always_home":
            e["log_loss"] = mt.log_loss(f["P"], y)
            e["rps"] = mt.rps(f["P"], y)
        eh = ((f["pred_h"] == hg) & (f["pred_a"] == ag)).astype(float)
        e["exact_hit_rate"] = float(eh.mean())
        e["exact_hit_rate_ci"] = _ci(eh, cl, n_boot)
        if f["grid"] is not None:
            e["mean_log_score"] = mt.mean_log_score(f["grid"], hg, ag)
        out[name] = e
    return out


def _by(df: pd.DataFrame, col: str, history, n_boot: int) -> dict:
    return {str(k): slice_metrics(g, history, n_boot) for k, g in df.groupby(col, sort=True)}


def _cumulative(df: pd.DataFrame, history) -> list[dict]:
    """Running winner / exact hit rate in kickoff order (+ the league-frequency baseline)."""
    if df.empty:
        return []
    df = df.sort_values(["kickoff_utc", "match_key"], kind="mergesort").reset_index(drop=True)
    d = _prep(df)
    win = (d["pred_idx"] == d["y"]).astype(float)
    exact = ((d["ph"] == d["hg"]) & (d["pa"] == d["ag"])).astype(float)
    base_hit = None
    if history is not None and len(history):
        f = bl.baseline_frames(df[["div", "home", "away", "kickoff_utc"]], history)["league_freq"]
        base_hit = (np.argmax(f["P"], axis=1) == d["y"]).astype(float)
    n = np.arange(1, len(df) + 1)
    cw, ce = np.cumsum(win) / n, np.cumsum(exact) / n
    cb = None if base_hit is None else np.cumsum(base_hit) / n
    return [
        {"n": int(i), "kickoff_utc": df["kickoff_utc"].iloc[i - 1],
         "winner_cum": float(cw[i - 1]), "exact_cum": float(ce[i - 1]),
         "baseline_winner_cum": None if cb is None else float(cb[i - 1])}
        for i in n
    ]


def recent_rows(df: pd.DataFrame, limit: int = RECENT_LIMIT) -> list[dict]:
    """Newest-first prediction-vs-result rows."""
    if df.empty:
        return []
    df = df.sort_values(["kickoff_utc", "match_key"], ascending=False, kind="mergesort")
    rows = []
    for r in df.head(limit).itertuples(index=False):
        rows.append({
            "match_key": r.match_key, "kickoff_utc": r.kickoff_utc, "league": r.league,
            "div": r.div, "home": r.home, "away": r.away, "source": r.source,
            "hours_before_kickoff": r.hours_before_kickoff,
            "probs": {"home": r.p_home, "draw": r.p_draw, "away": r.p_away},
            "pred_outcome": r.pred_outcome,
            "pred_score": [int(r.pred_score_home), int(r.pred_score_away)],
            "actual_score": [int(r.home_goals), int(r.away_goals)], "actual": r.result,
            "winner_ok": bool(r.pred_outcome == r.result),
            "exact_ok": bool(r.pred_score_home == r.home_goals
                             and r.pred_score_away == r.away_goals),
        })
    return rows


def clean(o):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def build_report(
    frames: dict[str, pd.DataFrame], history: pd.DataFrame | None = None,
    include_retro: bool = False, n_boot: int = 1000,
) -> dict:
    """The full accuracy report (see module docstring); JSON-safe."""

    def pick(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not include_retro:
            return df.loc[df["source"] == "live"]
        # a match with both a live and a retro prediction counts once, as live
        return df.sort_values("source", kind="mergesort").drop_duplicates("match_key")

    final_all = frames.get("final", pd.DataFrame())
    counts = {
        "live": int((final_all["source"] == "live").sum()) if len(final_all) else 0,
        "retro": int((final_all["source"] == "retro").sum()) if len(final_all) else 0,
    }
    fin = pick(final_all).copy()
    out: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "include_retro": include_retro, "counts": counts, "n_boot": n_boot,
        "scored": int(len(fin)), "empty": bool(fin.empty),
        "small_sample_below": mt.SMALL_N,
    }
    if fin.empty:
        return clean(out)
    fin["month"] = pd.to_datetime(fin["kickoff_utc"], utc=True).dt.strftime("%Y-%m")
    out["headline"] = slice_metrics(fin, history, n_boot)
    out["by_league"] = _by(fin, "league", history, n_boot)
    out["by_month"] = _by(fin, "month", history, n_boot)
    picked = {k: pick(frames.get(k, pd.DataFrame())) for k in KINDS}
    out["by_kind"] = {
        k: slice_metrics(df, history, n_boot) if len(df) else {"n": 0}
        for k, df in picked.items()
    }
    keysets = [set(df["match_key"]) if len(df) else set() for df in picked.values()]
    common = set.intersection(*keysets) if all(keysets) else set()
    out["by_kind_common"] = {}
    for k, df in picked.items():
        sub = df.loc[df["match_key"].isin(common)] if len(df) else df
        out["by_kind_common"][k] = slice_metrics(sub, history, n_boot) if len(sub) else {"n": 0}
    out["cumulative"] = _cumulative(fin, history)
    out["recent"] = recent_rows(pick(final_all))
    return clean(out)
