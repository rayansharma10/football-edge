"""Phase 3 baselines: Dixon-Coles, Elo/pi ordered logits, and pool weights vs the market.

Stages (each caches its output under ``data/interim/`` so the report can be regenerated without
re-running the fits)::

    uv run python scripts/run_baselines.py --stage dc       # walk-forward Dixon-Coles
    uv run python scripts/run_baselines.py --stage ratings  # Elo / pi ordered logits
    uv run python scripts/run_baselines.py --stage report   # -> reports/v1_baseline.md

Reference book per (division, season, market, phase): Pinnacle for seasons up to and including
2024/25 (rows flagged ``stale`` are excluded), Betfair Exchange from 2025/26, market average
(``Avg``) as the fallback. The book actually used is written into the report row by row.
De-margining is the power method (``fedge.market``).

Test seasons are 2016/17 onward. Everything here is deterministic: fixed seeds, no random
splits, no LLM in any decision path. Runtime is dominated by the Dixon-Coles refits; the caches
make report iteration cheap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.features import ratings
from fedge.market import devig
from fedge.models import dixon_coles as dc
from fedge.models import pool
from fedge.report import metrics as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
DC_CACHE = INTERIM / "v1_dc.parquet"
DC_INFO = INTERIM / "v1_dc_info.json"
RATINGS_CACHE = INTERIM / "v1_ratings.parquet"
RATINGS_INFO = INTERIM / "v1_ratings_info.json"

START_SEASON = "2016/17"
PS_LAST_SEASON = "2024/25"  # Pinnacle is the reference up to and including this season
MIN_COVER = 0.6  # a reference book must cover this share of the division-season's matches
BOOT_N = 200  # match-clustered bootstrap resamples for the pool weight
SELS = {"1x2": ("H", "D", "A"), "ou25": ("over", "under")}
MODELS_1X2 = ("dc", "elo", "pi", "market_pre", "market_close")
MODELS_OU = ("dc", "market_pre", "market_close")
PHASES = ("pre", "close")
MODEL_COLS = ("dc_h", "dc_d", "dc_a", "dc_over", "dc_under", "elo_h", "elo_d", "elo_a",
              "pi_h", "pi_d", "pi_a")


# ------------------------------------------------------------------ loading
def load_played() -> pd.DataFrame:
    m = pd.read_parquet(INTERIM / "matches.parquet")
    m = m.loc[m["FTHG"].notna() & m["FTAG"].notna()].copy()
    m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True)
    return m.sort_values(["div", "kickoff_utc"], kind="mergesort").reset_index(drop=True)


def load_cache(path: Path, refresh: bool) -> pd.DataFrame | None:
    if refresh or not path.exists():
        return None
    return pd.read_parquet(path).set_index("match_id")


# ------------------------------------------------------------------ market layer
_WIDE_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}


def _wide(odds: pd.DataFrame, book: str, market: str, phase: str) -> pd.DataFrame:
    """Complete de-marginable odds for (book, market, phase), memoised across the run.

    The raw odds table is ~5.7M rows; without the cache the loop over
    division x season x market x phase would re-scan it a thousand times.
    """
    key = (book, market, phase)
    hit = _WIDE_CACHE.get(key)
    if hit is not None:
        return hit
    sels = list(SELS[market])
    o = odds.loc[
        (odds["bookmaker"] == book)
        & (odds["market"] == market)
        & (odds["phase"] == phase)
        & (~odds["stale"])
        & (odds["selection"].isin(sels))
    ]
    if o.empty:
        out = pd.DataFrame(columns=sels)
    else:
        w = o.pivot_table(index="match_id", columns="selection", values="price", aggfunc="first")
        out = w[sels].dropna() if set(sels).issubset(w.columns) else pd.DataFrame(columns=sels)
    _WIDE_CACHE[key] = out
    return out


def _devig(wide: pd.DataFrame, market: str, prefix: str) -> pd.DataFrame:
    p = devig(wide.to_numpy(dtype=float), "power").probs
    return pd.DataFrame(p, index=wide.index, columns=[f"{prefix}_{s}" for s in SELS[market]])


def reference_books(played: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """One row per (div, season, market, phase): the reference book used and its coverage."""
    rows = []
    counts = played.groupby(["div", "season"], observed=True)["match_id"].size()
    for (div, season), n in counts.items():
        season = str(season)
        order = ["PS", "BFE", "Avg"] if season <= PS_LAST_SEASON else ["BFE", "Avg"]
        ids = set(played.loc[(played["div"] == div) & (played["season"] == season), "match_id"])
        for market in SELS:
            for phase in PHASES:
                book, cover = "none", 0.0
                for cand in order:
                    w = _wide(odds, cand, market, phase)
                    c = len(ids & set(w.index)) / max(n, 1)
                    if c >= MIN_COVER:
                        book, cover = cand, c
                        break
                rows.append(
                    {
                        "div": div,
                        "season": season,
                        "market": market,
                        "phase": phase,
                        "book": book,
                        "cover": round(cover, 3),
                        "n_played": int(n),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ dataset
def build_dataset(
    played: pd.DataFrame, odds: pd.DataFrame, dc_preds: pd.DataFrame, rating_preds: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    refs = reference_books(played, odds)
    base = played.set_index("match_id")
    goals = (base["FTHG"] + base["FTAG"]).astype(int)
    frame = pd.DataFrame(
        {
            "div": base["div"],
            "season": base["season"].astype(str),
            "kickoff_utc": base["kickoff_utc"],
            "FTR": base["FTR"],
            "y_1x2": base["FTR"].map({"H": 0, "D": 1, "A": 2}).astype("Int64"),
            "y_ou25": np.where(goals > 2.5, 0, 1),
        }
    )
    frame = frame.join(dc_preds[["p_home", "p_draw", "p_away", "p_over25", "p_under25"]])
    frame = frame.rename(
        columns={
            "p_home": "dc_h", "p_draw": "dc_d", "p_away": "dc_a",
            "p_over25": "dc_over", "p_under25": "dc_under",
        }
    )
    frame = frame.join(
        rating_preds[
            ["p_elo_home", "p_elo_draw", "p_elo_away", "p_pi_home", "p_pi_draw", "p_pi_away"]
        ]
    )
    frame = frame.rename(
        columns={
            "p_elo_home": "elo_h", "p_elo_draw": "elo_d", "p_elo_away": "elo_a",
            "p_pi_home": "pi_h", "p_pi_draw": "pi_d", "p_pi_away": "pi_a",
        }
    )
    out: dict[str, pd.DataFrame] = {}
    ids_by_cell = {
        (d, str(s)): set(g["match_id"])
        for (d, s), g in played.groupby(["div", "season"], observed=True)
    }
    for market in SELS:
        mkt_parts, book_parts = [], []
        for r in refs[refs["market"] == market].itertuples(index=False):
            w = _wide(odds, r.book, r.market, r.phase)
            if w.empty:
                continue
            # keep only this division-season's matches: every cell has its own reference book,
            # and concatenating whole-book tables would let the first cell's book win everywhere
            # (it silently dropped 2025/26+, where the reference is Betfair Exchange, not PS)
            w = w.loc[w.index.intersection(ids_by_cell[(r.div, str(r.season))])]
            if w.empty:
                continue
            mkt_parts.append(_devig(w, market, r.phase))
            book_parts.append(
                pd.DataFrame({f"ref_{r.phase}": r.book}, index=w.index)
            )
        if not mkt_parts:
            continue
        mkt = pd.concat(
            [pd.concat([p for p in mkt_parts if p.columns[0].startswith(f"{ph}_")])
             for ph in PHASES],
            axis=1,
        )
        books = None
        for part in book_parts:  # parts cover disjoint match sets (one book per div-season-phase)
            books = part if books is None else books.combine_first(part)
        cols = [f"{phase}_{s}" for phase in PHASES for s in SELS[market]]
        need = [*cols, "y_1x2", *[c for c in MODEL_COLS if c in frame.columns]]
        sub = frame.join(mkt, how="inner").join(books, how="inner")
        # test seasons only, and only rows where every model and market column is present:
        # without this the pre-2016/17 seasons (no walk-forward predictions) enter with NaN
        # model columns and silently poison every metric and the pooled fit.
        sub = sub.loc[sub["season"] >= START_SEASON].dropna(subset=need)
        out[market] = sub
    return out, refs


# ------------------------------------------------------------------ metrics
def model_cols(model: str, market: str) -> list[str]:
    """Probability columns of ``model``; unknown names are ``<model>_h/_d/_a`` or ``_over/_under``
    (the Phase 4 models, :mod:`scripts.run_main`, register their columns that way)."""
    known = MODELS_1X2 if market == "1x2" else MODELS_OU
    if model not in known:
        sels = ("h", "d", "a") if market == "1x2" else ("over", "under")
        return [f"{model}_{s}" for s in sels]
    if market == "1x2":
        return {
            "dc": ["dc_h", "dc_d", "dc_a"],
            "elo": ["elo_h", "elo_d", "elo_a"],
            "pi": ["pi_h", "pi_d", "pi_a"],
            "market_pre": ["pre_H", "pre_D", "pre_A"],
            "market_close": ["close_H", "close_D", "close_A"],
        }[model]
    return {
        "dc": ["dc_over", "dc_under"],
        "market_pre": ["pre_over", "pre_under"],
        "market_close": ["close_over", "close_under"],
    }[model]


def ycol(market: str) -> str:
    return "y_1x2" if market == "1x2" else "y_ou25"


def score_rows(df: pd.DataFrame, model: str, market: str) -> pd.DataFrame:
    """Per (div, season, model, market) forecast metrics."""
    cols = model_cols(model, market)
    p = df[cols].to_numpy(dtype=float)
    y = df[ycol(market)].to_numpy(dtype=int)
    pos = {mid: i for i, mid in enumerate(df.index)}
    out = []
    for (div, season), idx in df.groupby(["div", "season"], observed=True).groups.items():
        i = np.array([pos[m] for m in idx])
        pp, yy = p[i], y[i]
        out.append(
            {
                "model": model,
                "market": market,
                "div": div,
                "season": str(season),
                "n": len(i),
                "rps": M.rps(pp, yy),
                "log_loss": M.log_loss(pp, yy),
                "brier": M.brier(pp, yy),
                "cal_slope": M.calibration_slope_multi(pp, yy)[1],
            }
        )
    return pd.DataFrame(out)


def pooled_scores(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Whole-sample metrics per model (all leagues and test seasons pooled)."""
    rows = []
    y = df[ycol(market)].to_numpy(dtype=int)
    models = MODELS_1X2 if market == "1x2" else MODELS_OU
    for model in models:
        p = df[model_cols(model, market)].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "n": len(df),
                "rps": M.rps(p, y),
                "log_loss": M.log_loss(p, y),
                "brier": M.brier(p, y),
                "cal_slope": M.calibration_slope_multi(p, y)[1],
            }
        )
    return pd.DataFrame(rows)


def _pool_with_rows(p_model: np.ndarray, p_market: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Log opinion pool with a per-row weight (vectorised form of fedge.models.pool.log_pool)."""
    lp = w[:, None] * np.log(np.clip(p_model, 1e-12, 1.0))
    lp += (1.0 - w[:, None]) * np.log(np.clip(p_market, 1e-12, 1.0))
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def pool_table(
    df: pd.DataFrame, market: str, model: str = "dc"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample pool weight vs the pre-closing and closing market, per division."""
    pcols = model_cols(model, market)
    yc = ycol(market)
    rows, details = [], []
    for div, g in df.groupby("div", observed=True):
        g = g.sort_values("season", kind="mergesort")
        seasons = sorted(g["season"].unique())
        pmodel_all = g[pcols].to_numpy(dtype=float)
        y_all = g[yc].to_numpy(dtype=int)
        clusters = g.index.to_numpy()
        season_arr = g["season"].to_numpy()
        for phase in PHASES:
            mcols = [f"{phase}_{s}" for s in SELS[market]]
            if not all(c in g.columns for c in mcols):
                continue
            pmarket_all = g[mcols].to_numpy(dtype=float)
            w_by_season = {}
            for s in seasons:
                past = season_arr < s
                if past.sum() < 300:
                    continue
                w_by_season[s] = pool.fit_weight(pmodel_all[past], pmarket_all[past], y_all[past])
            if not w_by_season:
                continue
            ws = np.array(list(w_by_season.values()))
            boot = pool.fit_weight_bootstrap(
                pmodel_all, pmarket_all, y_all, clusters=clusters, n_boot=BOOT_N, seed=0
            )
            keep = np.isin(season_arr, list(w_by_season))
            w_apply = pd.Series(season_arr).map(w_by_season).to_numpy(dtype=float)
            pw = _pool_with_rows(pmodel_all[keep], pmarket_all[keep], w_apply[keep])
            pooled_ll = M.log_loss(pw, y_all[keep])
            mkt_ll = M.log_loss(pmarket_all[keep], y_all[keep])
            rows.append(
                {
                    "market": market,
                    "model": model,
                    "div": div,
                    "phase": phase,
                    "n": len(g),
                    "n_seasons": len(w_by_season),
                    "w_expanding": float(ws.mean()),
                    "w_min": float(ws.min()),
                    "w_max": float(ws.max()),
                    "w_pooled": float(boot["w"]),
                    "w_lo": float(boot["lo"]),
                    "w_hi": float(boot["hi"]),
                    "pool_logloss": pooled_ll,
                    "market_logloss": mkt_ll,
                }
            )
            details += [
                {"market": market, "model": model, "div": div, "phase": phase, "season": s,
                 "w": float(w_by_season[s])}
                for s in w_by_season
            ]
            print(
                f"  pool {model}/{market}/{div}/{phase}: w_exp={ws.mean():.3f} "
                f"pooled={boot['w']:.3f} [{boot['lo']:.3f},{boot['hi']:.3f}]",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(details)


# ------------------------------------------------------------------ report helpers
def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return str(v)


def _fmt_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with every ``cal_slope`` column rendered through ``M.fmt_slope``.

    ``md_table`` formats floats itself; pre-transforming to strings routes separated
    (non-estimable) calibration fits through the ``n.e.`` flag without touching any maths
    or the CSV companions.
    """
    df = df.copy()
    for c in df.columns:
        if c == "cal_slope" or c.endswith("_cal_slope"):
            df[c] = df[c].map(M.fmt_slope)
    return df


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    head = headers or cols
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def per_league_season_table(scores: pd.DataFrame, market: str, refs: pd.DataFrame) -> str:
    s = scores[scores["market"] == market]
    piv = s.pivot_table(
        index=["div", "season", "n"], columns="model",
        values=["rps", "log_loss", "brier", "cal_slope"], aggfunc="first",
    )
    piv.columns = [f"{model}_{metric}" for metric, model in piv.columns]
    piv = piv.reset_index()
    books = refs[(refs["market"] == market)].pivot_table(
        index=["div", "season"], columns="phase", values="book", aggfunc="first"
    ).reset_index()
    books["season"] = books["season"].astype(str)
    piv["season"] = piv["season"].astype(str)
    piv = piv.merge(books, on=["div", "season"], how="left")
    models = MODELS_1X2 if market == "1x2" else MODELS_OU
    cols = ["div", "season", "pre", "close", "n"]
    for model in models:
        for metric in ("rps", "log_loss", "brier", "cal_slope"):
            c = f"{model}_{metric}"
            if c in piv.columns:
                cols.append(c)
    piv = _fmt_slopes(piv)
    return md_table(piv.sort_values(["div", "season"]), cols)


# the bounded optimiser returns ~1e-8 rather than exactly 0 at the boundary
CI_ZERO_TOL = 1e-3


def interpretation(pools: pd.DataFrame) -> str:
    pre = pools[(pools["phase"] == "pre")]
    sig = pre[pre["w_lo"] > CI_ZERO_TOL]
    out = []
    if sig.empty:
        out.append(
            "No (league x market) cell has a **pre-closing** pool weight whose 95% bootstrap CI "
            "excludes 0. On this evidence the baseline models add no information to the "
            "pre-closing price - the decision point in REPORT section 5.2 fires: prioritise "
            "timing / execution (v3) over more model work."
        )
    else:
        out.append(
            f"{len(sig)} of {len(pre)} pre-closing (division x market) cells have a 95% "
            "bootstrap CI excluding 0. The CI belongs to `w_pooled`, which is an **in-sample** "
            "weight: it is refit on the whole sample, so the CI measures how tightly a weight "
            "chosen with hindsight is pinned down, not whether that weight would have helped "
            "out of sample. The out-of-sample statistic is `w_expanding` (fitted on strictly "
            "earlier test seasons only), and it is the weight the pooled log-losses below are "
            "scored with. Cells are listed below in `w_expanding` order. Multiple testing: "
            f"{len(pre)} pre-closing cells are searched for a significant CI "
            f"({len(pools)} cells across both phases), so roughly {0.05 * len(pre):.1f} "
            "significant pre-closing cells are expected by chance alone at 5%."
        )
        for r in sig.sort_values("w_expanding", ascending=False).itertuples(index=False):
            out.append(
                f"- {r.div} {r.market} ({r.phase}): w_expanding={r.w_expanding:.3f} "
                f"(out-of-sample); w_pooled={r.w_pooled:.3f} [{r.w_lo:.3f}, {r.w_hi:.3f}] "
                f"(in-sample CI; n={r.n}; pooled log-loss {r.pool_logloss:.4f} vs market "
                f"{r.market_logloss:.4f})"
            )
        worse = sig[sig["pool_logloss"] > sig["market_logloss"]]
        better = sig[~(sig["pool_logloss"] > sig["market_logloss"])]
        out += [
            "",
            f"**Out-of-sample test of those {len(sig)} cells (expanding weight).** "
            f"{len(worse)} of {len(sig)} have a pooled log-loss *worse* than the market on the "
            "same matches:",
        ]
        for r in worse.sort_values("w_expanding", ascending=False).itertuples(index=False):
            out.append(
                f"  - {r.div} {r.market} ({r.phase}): pooled {r.pool_logloss:.4f} vs market "
                f"{r.market_logloss:.4f} (w_expanding={r.w_expanding:.3f})"
            )
        if worse.empty:
            out.append("  - (none)")
        out.append(
            f"So only {len(better)} of the {len(sig)} CI-significant pre-closing cells also "
            "improve on the market **out of sample**; the rest are artefacts of the in-sample "
            "weight and of searching many cells."
        )
    close = pools[(pools["phase"] == "close")]
    close_sig = close[close["w_lo"] > CI_ZERO_TOL]
    out += [
        "",
        f"Against the **closing** price, {len(close_sig)} of {len(close)} cells have an "
        "(in-sample) CI excluding 0 (the closing price is the hardest benchmark; the "
        "literature expects ~0).",
        "",
        "Pre-closing cells sorted by mean expanding-window weight "
        "(`w_pooled`/`w_lo`/`w_hi` are in-sample):",
        "",
    ]
    top = pre.sort_values("w_expanding", ascending=False).head(10)
    out.append(md_table(top, ["div", "market", "w_expanding", "w_pooled", "w_lo", "w_hi", "n"]))
    return "\n".join(out)


def main() -> None:  # pragma: no cover - CLI driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dc", "ratings", "report", "all"], default="all")
    ap.add_argument("--refresh", action="store_true", help="ignore the prediction caches")
    args = ap.parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    played = load_played()
    print(f"played matches: {len(played)} in {played['div'].nunique()} divisions", flush=True)

    if args.stage in ("dc", "all"):
        dc_preds = None if args.refresh else load_cache(DC_CACHE, False)
        if dc_preds is None:
            dc_preds = stage_dc(played)
        print(f"dc predictions: {len(dc_preds)}", flush=True)
    if args.stage in ("ratings", "all"):
        rating_preds = None if args.refresh else load_cache(RATINGS_CACHE, False)
        if rating_preds is None:
            rating_preds = stage_ratings(played)
        print(f"rating predictions: {len(rating_preds)}", flush=True)
    if args.stage in ("dc", "ratings"):
        return

    dc_preds = load_cache(DC_CACHE, False)
    rating_preds = load_cache(RATINGS_CACHE, False)
    if dc_preds is None or rating_preds is None:
        raise SystemExit("missing caches: run --stage dc and --stage ratings first")
    odds = pd.read_parquet(INTERIM / "odds.parquet")
    datasets, refs = build_dataset(played, odds, dc_preds, rating_preds)
    for market, d in datasets.items():
        print(f"joined {market}: {len(d)} rows, {d['div'].nunique()} divisions", flush=True)

    scores, pools, details = [], [], []
    for market in ("1x2", "ou25"):
        d = datasets[market]
        models = MODELS_1X2 if market == "1x2" else MODELS_OU
        for model in models:
            scores.append(score_rows(d, model, market))
        w, det = pool_table(d, market)
        pools.append(w)
        details.append(det)
    scores_df = pd.concat(scores, ignore_index=True)
    pools_df = pd.concat([p for p in pools if len(p)], ignore_index=True)
    details_df = pd.concat([p for p in details if len(p)], ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(REPORTS / "v1_baseline_metrics.csv", index=False)
    pools_df.to_csv(REPORTS / "v1_baseline_pool_weights.csv", index=False)
    details_df.to_csv(REPORTS / "v1_baseline_pool_weights_by_season.csv", index=False)
    refs.to_csv(REPORTS / "v1_baseline_reference_books.csv", index=False)
    write_report(datasets, scores_df, pools_df, details_df, refs)
    print("wrote", REPORTS / "v1_baseline.md", flush=True)


def stage_dc(played: pd.DataFrame) -> pd.DataFrame:
    parts, infos = [], {}
    for div, g in played.groupby("div", observed=True):
        t0 = time.time()
        preds, info = dc.run_league(g)
        infos[div] = {
            "n_folds": info["n_folds"],
            "n_failed": info["n_failed"],
            "xi_by_season": {k: float(v) for k, v in info["xi_by_season"].items()},
            "xi_val_logloss": {
                k: (None if np.isnan(v) else float(v)) for k, v in info["xi_loss"].items()
            },
        }
        parts.append(preds)
        print(
            f"  dc {div}: {len(preds)} preds, {info['n_folds']} folds, "
            f"{info['n_failed']} failed, {time.time() - t0:.1f}s",
            flush=True,
        )
        infos[div]["seconds"] = round(time.time() - t0, 1)
    out = pd.concat(parts)
    out.index.name = "match_id"
    out.reset_index().to_parquet(DC_CACHE, index=False)
    DC_INFO.write_text(json.dumps(infos, indent=1), encoding="utf-8")
    return out


def stage_ratings(played: pd.DataFrame) -> pd.DataFrame:
    parts, infos = [], {}
    for div, g in played.groupby("div", observed=True):
        t0 = time.time()
        preds, info = ratings.run_league(g)
        infos[div] = {
            s: {
                "elo": v["elo"],
                "pi": v["pi"],
                "elo_val_logloss": v["elo_val_logloss"],
                "pi_val_logloss": v["pi_val_logloss"],
            }
            for s, v in info["selected"].items()
        }
        parts.append(preds)
        print(f"  ratings {div}: {len(preds)} preds, {time.time() - t0:.1f}s", flush=True)
    out = pd.concat(parts)
    out.index.name = "match_id"
    out.reset_index().to_parquet(RATINGS_CACHE, index=False)
    RATINGS_INFO.write_text(json.dumps(infos, indent=1), encoding="utf-8")
    return out


def write_report(datasets, scores, pools, details, refs) -> None:  # pragma: no cover
    """Render reports/v1_baseline.md from the computed tables."""
    dc_info = json.loads(DC_INFO.read_text(encoding="utf-8")) if DC_INFO.exists() else {}
    rating_info = (
        json.loads(RATINGS_INFO.read_text(encoding="utf-8")) if RATINGS_INFO.exists() else {}
    )
    d1, d25 = datasets["1x2"], datasets["ou25"]
    s1 = sorted(d1["season"].unique())
    s25 = sorted(d25["season"].unique())
    books = refs.groupby(["market", "phase", "book"], observed=True).size().reset_index()
    books.columns = ["market", "phase", "book", "n_cells"]
    lines = [
        "# football-edge v1 baselines (Phase 3: cards 3.1-3.3)",
        "",
        "Generated by `uv run python scripts/run_baselines.py --stage report` from "
        "`data/interim/` caches. Machine-readable companions in this directory: "
        "`v1_baseline_metrics.csv`, `v1_baseline_pool_weights.csv`, "
        "`v1_baseline_pool_weights_by_season.csv`, `v1_baseline_reference_books.csv`.",
        "",
        "## 1. What was run",
        "",
        f"- Divisions: {d1['div'].nunique()}.",
        f"- **1X2 evaluation window**: {s1[0]} to {s1[-1]} ({len(s1)} seasons, the last one "
        f"partial); {len(d1)} matches with all of DC + Elo + pi + a reference market.",
        f"- **Over/under 2.5 evaluation window**: {s25[0]} to {s25[-1]} ({len(s25)} seasons, "
        f"the last one partial); {len(d25)} matches. This is **not** the same window as 1X2: "
        "the pre-2019/20 ou25 cells have no closing reference book - `reference_books` "
        "returns `book = 'none'` for ou25/close before 2019/20 (126 cells in "
        "`v1_baseline_reference_books.csv`) - so `build_dataset` drops those rows, and every "
        "over/under 2.5 number in this report is 2019/20 onward.",
        "- **Dixon-Coles** (penaltyblog): per-division team parameters on a 5-year rolling "
        "window with exponential time decay `exp(-xi * years)`; refit **every matchweek**; xi "
        "re-selected in-fold once per (division, test season) from "
        f"{tuple(dc.XI_GRID)} by inner-validation log-loss on the last {dc.VAL_MATCHES} "
        "pre-test matches. 3h embargo between a result and its use.",
        "- **Elo** (own, goal-difference scaled, home advantage) and **pi-ratings** "
        "(penaltyblog `PiRatingSystem`), both computed by a single causal sweep with the same "
        "3h embargo (a result enters the state only once `kickoff + 3h <= kickoff_predicted`; "
        "`tests/test_ratings.py` proves this equals an `asof`-restricted recomputation).",
        "- The 1X2 model for each rating is an **ordered logit** on the pre-match rating "
        "difference, with `(k, HFA)` / `(alpha, beta)` selected in-fold per (division, test "
        "season) from small grids (`ratings.ELO_GRID`, `ratings.PI_GRID`) and the logit refit "
        "on the same in-fold window.",
        "- **Market**: de-margined with the **power** method (`fedge.market`) on the "
        "pre-closing (`pre`) and closing (`close`) prices of one reference book per "
        "(division, season, market, phase): Pinnacle to 2024/25 (non-stale rows only), Betfair "
        "Exchange from 2025/26, `Avg` when neither covers " + f"{MIN_COVER:.0%}" + " of the "
        "division-season. The book used is shown per row in section 3.",
        "- **Pool**: logarithmic opinion pool `p ~ p_model**w * p_market**(1-w)` "
        "(w on the model). `w_expanding` = mean over test seasons of the weight fitted on all "
        "strictly earlier test seasons (fully out-of-sample); `w_pooled` and its 95% "
        f"match-clustered bootstrap CI ({BOOT_N} resamples, seed 0) refit the weight on the "
        "whole sample. Pooled probabilities scored in section 4 use the expanding weight.",
        "- No random splits anywhere; all thresholds, decay and hyper-parameters are chosen "
        "in-fold. Seeds and iteration counts are fixed, so re-runs are identical.",
        "",
        "Reference books actually used (cells = division x season x market x phase):",
        "",
        md_table(books, ["market", "phase", "book", "n_cells"]),
        "",
        "## 2. Pooled scores (all divisions; 1X2 test seasons 2016/17 onward, over/under 2.5 "
        "2019/20 onward)",
        "",
        "RPS and Brier are multiclass sums (lower is better); `cal_slope` is the pooled "
        "one-vs-rest logistic recalibration slope (1.0 = calibrated). `n.e.` = not estimable: "
        "the one-vs-rest logistic recalibration separated at this sample size (|slope| > 10), "
        "so the number is an artefact, not a calibration measurement.",
        "",
        "### 1X2",
        "",
        md_table(_fmt_slopes(pooled_scores(d1, "1x2")),
                 ["model", "n", "rps", "log_loss", "brier", "cal_slope"]),
        "",
        "### Over/under 2.5 (test seasons 2019/20 onward)",
        "",
        md_table(
            _fmt_slopes(pooled_scores(d25, "ou25")),
            ["model", "n", "rps", "log_loss", "brier", "cal_slope"],
        ),
        "",
        "## 3. Per league x season",
        "",
        "`pre` / `close` are the reference books for that league-season (Pinnacle `PS`, Betfair "
        "Exchange `BFE`, market average `Avg`).",
        "",
        "### 1X2",
        "",
        per_league_season_table(scores, "1x2", refs),
        "",
        "### Over/under 2.5",
        "",
        per_league_season_table(scores, "ou25", refs),
        "",
        "## 4. Pool weight vs the market (Dixon-Coles)",
        "",
        md_table(
            pools.sort_values(["market", "phase", "div"]),
            ["market", "div", "phase", "n", "n_seasons", "w_expanding", "w_pooled", "w_lo",
             "w_hi", "pool_logloss", "market_logloss"],
        ),
        "",
        "Per-season weights: `v1_baseline_pool_weights_by_season.csv`.",
        "",
        "`w_expanding` is the **out-of-sample** statistic (the weight is fitted only on "
        "strictly earlier test seasons); `w_pooled`, `w_lo` and `w_hi` are **in-sample** (the "
        "weight is refit on the whole sample, so the CI is a within-sample diagnostic, not "
        "out-of-sample evidence). 1X2 cells cover test seasons 2016/17 onward and over/under "
        "2.5 cells 2019/20 onward (section 1).",
        "",
        "## 5. Interpretation",
        "",
        interpretation(pools),
        "",
        "## 6. In-fold parameter selection (audit trail)",
        "",
        "Chosen decay `xi` per division (count of test seasons by value; `edge` counts seasons "
        "where the tuner picked the largest grid value, i.e. the optimum is at or beyond the "
        "grid edge):",
        "",
        md_table(xi_table(dc_info), ["div", "xi_mode", "n_at_edge", "n_seasons", "folds",
                                     "failed_folds"]),
        "",
        "The xi inner-validation log-loss surface is flat (see section 7); these choices are "
        "near-arbitrary within the grid and are reported for auditability rather than as a "
        "performance claim.",
        "",
        "Rating hyper-parameter selection frequency (division-season cells):",
        "",
        md_table(param_freq(rating_info, "elo"), ["param", "n_cells"]),
        "",
        md_table(param_freq(rating_info, "pi"), ["param", "n_cells"]),
        "",
        "## 7. Runtime, limits and UNVERIFIED assumptions",
        "",
        runtime_lines(dc_info, rating_info),
        "",
        "**Limitations / UNVERIFIED**",
        "- `xi` is re-selected per division-season rather than per matchweek (runtime bound); "
        "the inner-validation loss surface is flat, so this is a low-stakes simplification.",
        "- Ratings are per division: a promoted side starts from the initial rating "
        f"({ratings.ELO_START:g} Elo / 0 pi) instead of carrying its lower-division rating.",
        "- Neutral venues (COVID 2020/21) are treated as ordinary home matches.",
        "- Pre-closing `available_at` is `kickoff - 24h` from the P1 ingest; the reviewer "
        "flagged that football-data's real collection cadence is later (Fri 17:00 / Tue 13:00 "
        "UK). This does not affect anything in this report (no bet-time decision is taken "
        "here), but Phase 5 must fix it before any early-line strategy is measured.",
        "- 2026/27 is partial (10 divisions, ~1,100 matches) and only Betfair Exchange "
        "closing prices exist for it.",
        "- Ordered-logit `l2` and the grids are fixed a priori; only the grid members are "
        "selected in-fold.",
        "- The Elo goal-difference multiplier follows the World Football Elo convention "
        "(1 / 1.5 / (11+gd)/8); this specific form is inherited, not tuned.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "v1_baseline.md").write_text("\n".join(lines), encoding="utf-8")


def xi_table(dc_info: dict) -> pd.DataFrame:
    cols = ["div", "xi_mode", "n_at_edge", "n_seasons", "folds", "failed_folds"]
    rows = []
    for div, info in sorted(dc_info.items()):
        vals = list(info.get("xi_by_season", {}).values())
        if not vals:
            continue
        s = pd.Series(vals)
        rows.append(
            {
                "div": div,
                "xi_mode": float(s.mode().iloc[0]),
                "n_at_edge": int((s == dc.XI_GRID[-1]).sum()),
                "n_seasons": len(vals),
                "folds": info.get("n_folds"),
                "failed_folds": info.get("n_failed"),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def param_freq(rating_info: dict, key: str) -> pd.DataFrame:
    vals = [v[key] for seasons in rating_info.values() for v in seasons.values() if key in v]
    if not vals:
        return pd.DataFrame(columns=["param", "n_cells"])
    s = pd.Series(vals).value_counts().rename_axis("param").reset_index(name="n_cells")
    return s


def runtime_lines(dc_info: dict, rating_info: dict) -> str:
    secs = [v.get("seconds") for v in dc_info.values() if v.get("seconds")]
    total = f"{sum(secs):.0f}s" if secs else "measured in the run log (not recorded in the cache)"
    return (
        "- Runtime (this machine, Windows 11 desktop): Dixon-Coles walk-forward "
        f"(18 divisions, per-matchweek refits + in-fold xi selection) = {total}; "
        "ratings are ~1-2s per division. Report stage ~2-4 min (pool bootstraps). "
        "Prediction caches live in `data/interim/v1_*.parquet`, so the report can be "
        "regenerated without re-fitting."
    )


if __name__ == "__main__":
    main()
