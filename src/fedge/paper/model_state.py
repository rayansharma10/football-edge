"""Live model state for the paper desk: refresh results, rebuild features, score fixtures.

The desk re-uses the Phase 1-4 pipeline instead of re-implementing it:

* :func:`refresh_ingest` runs ``fedge.ingest.football_data.ingest`` (finished seasons are served
  from the raw cache, the current season is re-downloaded) so the ratings/form state sees the
  latest results.
* :func:`build_features` rebuilds the Phase 4 feature table (``fedge.features.main_set``) on the
  played matches and caches it under ``data/interim/paper/`` keyed by a hash of the results it was
  built from, so a second run of the desk is seconds rather than a minute.
* :func:`fixture_features` computes the same feature families for the *upcoming* fixtures. Each
  fixture gets its own causal sweep over the played matches strictly before its kickoff (the
  ``available_at < bet_time`` rule of AGENTS.md rule 3, enforced here by construction rather than
  trusted), because the Elo/pi/attack-defence sweeps require a result for every row and must never
  be fed a fixture as if it had played.
* :func:`fit_live` (in :mod:`fedge.models.gbm`) applies the documented Phase 4 protocol to the
  "next fold": fit, early-stop and calibrate on the season before the fixtures, then refit on every
  match before them.
* :func:`score_fixtures` de-margins the snapshot price, applies the live models and pools model and
  market with the weight from ``config/strategy.toml``.

Both configured models are used: the market-residual variant (``lgbd_xg`` for 1X2) is initialised
from the *de-margined pre-closing reference price of the same match* exactly as in Phase 4; for the
fixtures the snapshot price of the fixture file plays that role.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.features import form, ratings
from fedge.features.main_set import ELO_PARAMS, PI_PARAMS, feature_sets
from fedge.ingest import football_data as fd
from fedge.market import devig
from fedge.models import gbm
from fedge.models.pool import log_pool

# name -> (learner, feature variant, decorrelated?). The full P4 registry lives in
# scripts/run_main.py; the desk only needs the two models config/strategy.toml names.
LIVE_MODELS = {"lgb_xg": ("lgb", "xg", False), "lgbd_xg": ("lgb", "xg", True)}
START_SEASON = "2016/17"
THREADS = 4


def refresh_ingest(
    data_dir: Path | str = "data", config_path: Path | str = "config/leagues.toml",
    delay: float = 1.0, offline: bool = False,
):
    """Download + parse football-data (current season re-fetched, finished seasons cached)."""
    return fd.ingest(config_path, data_dir, delay, offline=offline)


def played_matches(data_dir: Path | str = "data") -> pd.DataFrame:
    """Played matches from the ingest cache: tz-aware kickoff, sorted by division then kickoff."""
    m = pd.read_parquet(Path(data_dir) / "interim" / "matches.parquet")
    m = m.loc[m["FTHG"].notna() & m["FTAG"].notna()].copy()
    m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True)
    m["season"] = m["season"].astype(str)
    return m.sort_values(["div", "kickoff_utc"], kind="mergesort").reset_index(drop=True)


def results_signature(played: pd.DataFrame) -> str:
    """Stable hash of the results the features were built from (match id + score + kickoff)."""
    cols = ["match_id", "kickoff_utc", "FTHG", "FTAG"]
    txt = played[cols].to_csv(index=False, float_format="%.0f")
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


def div_code_map(played: pd.DataFrame) -> dict[str, int]:
    """``div -> categorical code``, in the same sorted order ``Series.astype('category')`` uses."""
    cats = sorted(played["div"].astype(str).unique().tolist())
    return {d: i for i, d in enumerate(cats)}


def build_features(
    played: pd.DataFrame, data_dir: Path | str = "data", xg_path: Path | str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Phase 4 feature table for every played match, cached by :func:`results_signature`."""
    from fedge.features.main_set import build_feature_table

    cache_dir = Path(data_dir) / "interim" / "paper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = results_signature(played)
    cached = cache_dir / f"features_{sig}.parquet"
    if cached.exists() and not refresh:
        return pd.read_parquet(cached)
    xg_file = Path(xg_path) if xg_path is not None else (
        Path(data_dir) / "interim" / "match_xg.parquet"
    )
    xg = pd.read_parquet(xg_file) if xg_file.exists() else None
    t0 = time.time()
    feats = build_feature_table(played, xg)
    feats.to_parquet(cached)
    print(f"paper: features {feats.shape} in {time.time() - t0:.1f}s ({cached.name})", flush=True)
    return feats


def season_of(kickoff_utc) -> str:
    """Season label of a date: July-June, e.g. 2026-10-10 -> ``2026/27`` (the ingest format)."""
    ts = pd.Timestamp(kickoff_utc)
    y = ts.year if ts.month >= 7 else ts.year - 1
    return f"{y}/{str(y + 1)[2:]}"


def fixture_features(
    played: pd.DataFrame, fixtures: pd.DataFrame, div_codes: dict[str, int] | None = None
) -> pd.DataFrame:
    """Feature rows for the fixtures, one per ``match_key`` (index = ``match_key``).

    Each fixture is scored from the played matches of its division strictly before its kickoff: the
    fixture row is appended last and is therefore never released into any state (the sweeps only
    release rows with a kickoff at least the 3h embargo before the row being scored). A fixture in
    a division the model was never trained on has no history and is returned as all-NaN.
    """
    codes = div_codes if div_codes is not None else div_code_map(played)
    by_div = {d: g for d, g in played.groupby("div", observed=True)}
    rows, keys = [], []
    for f in fixtures.itertuples(index=False):
        g = by_div.get(f.div)
        if g is None:
            continue
        before = g.loc[g["kickoff_utc"] < f.kickoff_utc]
        row = {
            "match_id": f.match_key,
            "div": f.div,
            "season": season_of(f.kickoff_utc),
            "kickoff_utc": f.kickoff_utc,
            "home": f.home,
            "away": f.away,
            "FTHG": np.nan,
            "FTAG": np.nan,
        }
        sub = pd.concat([before, pd.DataFrame([row])], ignore_index=True)
        sub["kickoff_utc"] = pd.to_datetime(sub["kickoff_utc"], utc=True)
        # the ingest writes nullable Int64 goal columns; the rating sweeps cast to int64 and
        # pandas refuses that for an extension dtype holding NA, so coerce the goals to float64.
        # The fixture's own (missing) goals are never released into any state: it is the last row
        # and the sweeps only release rows strictly before the row being scored.
        for col in ("FTHG", "FTAG"):
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce").astype(float)
        s = sub.sort_values("kickoff_utc", kind="mergesort")
        r = ratings.attack_defence_sweep(s)
        r["elo_diff"] = ratings.elo_sweep(s, *ELO_PARAMS)
        r["pi_diff"] = ratings.pi_sweep(s, *PI_PARAMS)
        frow = form.form_features(s).loc[[f.match_key]]
        last = pd.concat([r.iloc[[-1]].reset_index(drop=True), frow.reset_index(drop=True)], axis=1)
        last["div_code"] = codes.get(f.div, np.nan)
        rows.append(last)
        keys.append(f.match_key)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # LightGBM needs the categorical column to be an integer dtype (the training table's div_code
    # comes from pandas category codes); a missing code can only happen for a division with no
    # history, which is skipped above.
    out["div_code"] = pd.to_numeric(out["div_code"], errors="coerce").fillna(-1).astype("int32")
    out.index = pd.Index(keys, name="match_key")
    return out


# ------------------------------------------------------------------ training / live fit
def training_table(played: pd.DataFrame, feats: pd.DataFrame, market_pre: dict) -> pd.DataFrame:
    """The table :mod:`fedge.models.gbm` consumes: features, outcome, market prior by match id."""
    base = played.set_index("match_id")
    goals = (base["FTHG"] + base["FTAG"]).astype(int)
    t = feats.join(
        pd.DataFrame(
            {
                "kickoff_utc": base["kickoff_utc"],
                "season": base["season"].astype(str),
                "div": base["div"],
                "y_1x2": base["FTR"].map({"H": 0, "D": 1, "A": 2}),
                "y_ou25": np.where(goals > 2.5, 0, 1),
            }
        ),
        how="inner",
    )
    t = t.loc[t["y_1x2"].notna()].copy()
    t["y_1x2"] = t["y_1x2"].astype(int)
    for market, mk in market_pre.items():
        t = t.join(mk.set_index("match_id").add_prefix(f"{market}_"), how="left")
    return t.sort_values("kickoff_utc", kind="mergesort")


def load_market_pre(data_dir: Path | str = "data") -> dict[str, pd.DataFrame]:
    """Phase 4 pre-closing market prior per market (``data/interim/v2_market_pre_*.parquet``)."""
    d = Path(data_dir) / "interim"
    out = {}
    for market in ("1x2", "ou25"):
        p = d / f"v2_market_pre_{market}.parquet"
        if p.exists():
            out[market] = pd.read_parquet(p)
    return out


def fit_live(table: pd.DataFrame, market: str, model: str, asof, threads: int = THREADS):
    """Fit the configured model on history up to ``asof``: ``(learner, model, cal, info)``.

    ``asof`` is the kickoff of the earliest fixture being scored (tz-aware UTC): nothing at/after
    ``asof - 3h`` enters the final fit.
    """
    learner, variant, decor = LIVE_MODELS[model]
    cols = feature_sets([c for c in table.columns])[variant]
    init_cols = [f"{market}_pre_{s}" for s in TASK_COLS[market]] if decor else None
    model_obj, cal, info = gbm.fit_live(
        table,
        cols,
        f"y_{market}",
        market,
        learner,
        asof,
        init_cols=init_cols,
        threads=threads,
        log=lambda s: print(s, flush=True),
    )
    info.update({"model": model, "market": market, "learner": learner, "decor": bool(decor)})
    return learner, model_obj, cal, info


TASK_COLS = {"1x2": ("H", "D", "A"), "ou25": ("over", "under")}


def market_probs(price_rows: pd.DataFrame, market: str) -> pd.DataFrame:
    """De-margined (power) snapshot probabilities: ``match_key`` x selection columns."""
    sels = list(TASK_COLS[market])
    w = price_rows.pivot_table(
        index="match_key", columns="selection", values="price", aggfunc="first"
    )
    w = w.reindex(columns=sels).dropna()
    p = devig(w.to_numpy(dtype=float), "power").probs
    return pd.DataFrame(p, index=w.index, columns=sels)


def score_fixtures(
    fits: dict, x_fix: pd.DataFrame, price_rows: pd.DataFrame, pool_weights: dict
) -> pd.DataFrame:
    """Model and pooled probabilities plus the edge of every (fixture, market, selection).
    market_prob, pooled_prob, edge``. The market probability is the de-margined snapshot price; the
    pooled probability is the log opinion pool with the weight from ``config/strategy.toml``
    (``fedge.models.pool``). Edge is EV per unit stake after 6% commission.
    """
    from fedge.backtest.stats import COMMISSION, net_odds

    out = []
    for market, (learner, model_obj, cal, info) in fits.items():
        sels = list(TASK_COLS[market])
        rows = price_rows[price_rows["market"] == market]
        p_market = market_probs(rows, market)
        if p_market.empty:
            continue
        keys = p_market.index
        x = x_fix.reindex(keys)
        init = gbm.to_scores(p_market.to_numpy(dtype=float), market) if info["decor"] else None
        raw = gbm.predict_model(model_obj, learner, market, x, init)
        model_p = cal(raw)
        pooled = log_pool(model_p, p_market.to_numpy(dtype=float), pool_weights[market])
        price = rows.pivot_table(
            index="match_key", columns="selection", values="price", aggfunc="first"
        ).reindex(keys)
        src = rows.pivot_table(
            index="match_key", columns="selection", values="price_source", aggfunc="first"
        ).reindex(keys)
        nodds = net_odds(price.to_numpy(dtype=float), COMMISSION)
        edge = pooled * nodds - 1.0
        for j, s in enumerate(sels):
            out.append(
                pd.DataFrame(
                    {
                        "match_key": keys,
                        "market": market,
                        "selection": s,
                        "price": price[s].to_numpy(dtype=float),
                        "price_source": src[s].to_numpy(),
                        "model_prob": model_p[:, j],
                        "market_prob": p_market[s].to_numpy(dtype=float),
                        "pooled_prob": pooled[:, j],
                        "edge": edge[:, j],
                        "calibrator": info["calibrator"],
                        "best_iter": info["best_iter"],
                    }
                )
            )
    if not out:
        return pd.DataFrame(
            columns=["match_key", "market", "selection", "price", "price_source", "model_prob",
                     "market_prob", "pooled_prob", "edge"]
        )
    return pd.concat(out, ignore_index=True)


def score_all(
    data_dir,
    strategy: dict,
    played: pd.DataFrame,
    fixtures: pd.DataFrame,
    price_rows: pd.DataFrame,
    weights: dict,
    threads: int = THREADS,
) -> pd.DataFrame:
    """The whole scoring pipeline for a fixture list: features -> live fit -> pooled edges.

    This is the single entry point ``scripts/paper_picks.py`` calls (and the seam the tests stub):
    refresh the feature table, build the fixtures' features, refit the configured models with the
    Phase 4 protocol up to the first kickoff, and pool model and snapshot market price.
    """
    feats = build_features(played, data_dir)
    table = training_table(played, feats, load_market_pre(data_dir))
    x_fix = fixture_features(played, fixtures, div_code_map(played))
    if x_fix.empty:
        return score_fixtures({}, x_fix, price_rows, weights)
    asof = fixtures["kickoff_utc"].min()
    fits = {}
    for market in TASK_COLS:
        model = str(strategy["markets"][market]["model"])
        fits[market] = fit_live(table, market, model, asof, threads=threads)
    return score_fixtures(fits, x_fix, price_rows, weights)
