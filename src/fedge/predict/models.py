"""Price-free prediction models behind one registry keyed by ``model_version``.

Every prediction the accuracy tracker shows or logs comes from :func:`predict` with a
``model_version``. A model is a function

    fn(data_dir, played, fixtures, now, threads=..., table=None) -> {match_key: record}

where ``fixtures`` has ``match_key, div, home, away, kickoff_utc`` and each ``record`` is the output
of :func:`fedge.predict.scoreline.predict_match` plus ``fallback`` (a note when the Dixon-Coles
side had to fall back) and ``method``. **No bookmaker price enters any model here**: the 1X2 comes
from the non-anchored ``lgb_xg`` LightGBM (ratings, form, xG form, schedule features only) and the
scoreline grid from a Dixon-Coles fit reconciled to that 1X2.

Phase 2 (a model bake-off) registers further versions with :func:`register` and flips
:data:`CURRENT_VERSION`; the telemetry tables store the version with every row.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.models import dixon_coles as dc
from fedge.models import gbm
from fedge.paper import model_state as ms
from fedge.predict.scoreline import predict_match

LGB_XG_DC_V1 = "lgb_xg_dc_v1"
CURRENT_VERSION = LGB_XG_DC_V1
MODEL_DESCRIPTIONS = {
    LGB_XG_DC_V1: (
        "price-free lgb_xg (ratings, form, xG form; no market feature) 1X2 + Dixon-Coles "
        "scoreline grid reconciled to it"
    ),
}

ModelFn = Callable[..., dict]
REGISTRY: dict[str, ModelFn] = {}


def register(version: str):
    """Decorator: add a model function to :data:`REGISTRY` under ``version``."""

    def deco(fn: ModelFn) -> ModelFn:
        if version in REGISTRY:
            raise ValueError(f"model version {version!r} already registered")
        REGISTRY[version] = fn
        return fn

    return deco


def predict(version: str, *args, **kwargs) -> dict:
    """Run the model registered as ``version`` (the single entry point for all callers)."""
    try:
        fn = REGISTRY[version]
    except KeyError:
        raise KeyError(f"unknown model_version {version!r}; known: {sorted(REGISTRY)}") from None
    return fn(*args, **kwargs)


def dc_lambdas(params: dc.DCParams, home: str, away: str) -> tuple[float, float, bool]:
    """Expected goals (home, away) from fitted DC params; True when a team had no history."""
    idx = {t: i for i, t in enumerate(params.teams)}
    unknown = home not in idx or away not in idx
    ah = params.attack[idx[home]] if home in idx else dc.PRIOR_ATTACK
    dh = params.defence[idx[home]] if home in idx else dc.PRIOR_DEFENCE
    aa = params.attack[idx[away]] if away in idx else dc.PRIOR_ATTACK
    da = params.defence[idx[away]] if away in idx else dc.PRIOR_DEFENCE
    return float(np.exp(ah + da + params.hfa)), float(np.exp(aa + dh)), unknown


def league_means(played: pd.DataFrame, div: str, ref) -> tuple[float, float]:
    """Mean home/away goals over the last 380 matches of ``div`` before ``ref``."""
    g = played.loc[(played["div"] == div) & (played["kickoff_utc"] < ref)].tail(380)
    if g.empty:
        return 1.5, 1.2
    return float(g["FTHG"].astype(float).mean()), float(g["FTAG"].astype(float).mean())


def price_free_1x2(
    data_dir, played: pd.DataFrame, fixtures: pd.DataFrame, now, threads: int = ms.THREADS,
    table: pd.DataFrame | None = None, match_xg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """1X2 (``match_key`` x H/D/A) from the non-anchored ``lgb_xg`` model fitted as of ``now``.

    ``table`` is the training table (:func:`fedge.paper.model_state.training_table`); built from
    ``played`` when omitted. Callers that score retrospectively pass a table already cut to the
    rows known at ``now``. Fixtures far in the future are scored with today's ratings and form
    (:func:`fedge.paper.model_state.feature_asof_fixtures`).
    """
    cols = ["H", "D", "A"]
    if fixtures.empty:
        return pd.DataFrame(columns=cols)
    if table is None:
        feats = ms.build_features(played, data_dir)
        table = ms.training_table(played, feats, ms.load_market_pre(data_dir))
    if match_xg is None:
        xg_file = Path(data_dir) / "interim" / "match_xg.parquet"
        match_xg = pd.read_parquet(xg_file) if xg_file.exists() else None
    x_fix = ms.fixture_features(
        played, ms.feature_asof_fixtures(fixtures, now, played), ms.div_code_map(played), match_xg
    )
    if x_fix.empty:
        return pd.DataFrame(columns=cols)
    learner, model_obj, cal, _info = ms.fit_live(table, "1x2", ms.PRICE_FREE_MODEL, now, threads)
    x_fix = x_fix[ms.live_feature_columns(table.columns, ms.PRICE_FREE_MODEL)]
    raw = gbm.predict_model(model_obj, learner, "1x2", x_fix, None)
    return pd.DataFrame(cal(raw), index=x_fix.index, columns=cols)


@register(LGB_XG_DC_V1)
def lgb_xg_dc_v1(
    data_dir, played: pd.DataFrame, fixtures: pd.DataFrame, now, threads: int = ms.THREADS,
    table: pd.DataFrame | None = None, match_xg: pd.DataFrame | None = None,
) -> dict:
    """Price-free lgb_xg 1X2 + Dixon-Coles grid reconciled to it. ``{match_key: record}``."""
    now = pd.Timestamp(now)
    p = price_free_1x2(data_dir, played, fixtures, now, threads, table, match_xg)
    fits = {
        d: dc.fit_dc(played.loc[played["div"] == d], ref=now)
        for d in sorted(set(map(str, fixtures["div"])))
    }
    out: dict[str, dict] = {}
    for f in fixtures.itertuples(index=False):
        if f.match_key not in p.index:
            continue
        params = fits.get(f.div)
        fallback = None
        if params is None:
            lh, la = league_means(played, f.div, now)
            rho = 0.0
            fallback = "league-average goals (no DC fit)"
        else:
            lh, la, unknown = dc_lambdas(params, f.home, f.away)
            rho = params.rho
            if unknown:
                fallback = "team without history: DC prior"
        tgt = p.loc[f.match_key, ["H", "D", "A"]].to_numpy(dtype=float)
        pm = predict_match(lh, la, rho, tgt)
        pm["fallback"] = fallback
        pm["lambda_dc"] = [lh, la]
        out[f.match_key] = pm
    return out
