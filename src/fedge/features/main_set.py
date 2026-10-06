"""Assemble the Phase 4 feature table (all families, as-of) for the main model.

Feature families (each has its own leakage test in ``tests/test_form.py`` /
``tests/test_ratings.py``):

``ratings``  Elo difference and pi-rating expected goal difference (fixed a-priori parameters,
             not tuned on any test season) and online Poisson attack/defence ("DC-lite").
``goals``    time-weighted goals for/against, points, home/away splits (+ effective matches).
``shots``    time-weighted shots / shots on target for and against.
``xg``       time-weighted xG for/against (Understat top-5, football-data 2026/27).
``sched``    rest days, games played this season, promoted flag.

Variants: ``goals`` = ratings + goals + sched; ``xg`` = goals variant + shots + xG.
"""

from __future__ import annotations

import pandas as pd

from fedge.features import form, ratings
from fedge.features.xg import attach_xg

ELO_PARAMS = (20.0, 60.0)  # (k, home advantage): a member of ratings.ELO_GRID, fixed
PI_PARAMS = (0.15, 0.10)  # (alpha, beta): a member of ratings.PI_GRID, fixed

RATING_COLS = ["elo_diff", "pi_diff", *ratings.AD_COLS]


def _ratings_block(matches: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in matches.groupby("div", sort=True, observed=True):
        g = g.sort_values("kickoff_utc", kind="mergesort")
        r = ratings.attack_defence_sweep(g)
        r["elo_diff"] = ratings.elo_sweep(g, *ELO_PARAMS)
        r["pi_diff"] = ratings.pi_sweep(g, *PI_PARAMS)
        parts.append(r[RATING_COLS])
    return pd.concat(parts)


def build_feature_table(matches: pd.DataFrame, match_xg: pd.DataFrame | None) -> pd.DataFrame:
    """One row per played match (index ``match_id``) with every feature family.

    ``matches`` must be played matches only, with tz-aware ``kickoff_utc``.
    """
    m = attach_xg(matches, match_xg)
    feats = _ratings_block(m).join(form.form_features(m))
    feats["div_code"] = (
        m.set_index("match_id")["div"].astype("category").cat.codes.reindex(feats.index)
    )
    return feats


def feature_sets(columns) -> dict[str, list[str]]:
    """Column lists for the ``goals`` and ``xg`` variants, given the table's columns."""
    cols = list(columns)
    sched = [c for c in cols if c.endswith(("_rest", "_games", "_promoted"))]
    goals = [
        c
        for c in cols
        if c.startswith(("h_", "a_"))
        and any(t in c for t in ("gf", "ga", "pts", "_n", "home_g", "away_g"))
        and c not in sched
    ]
    shots = [c for c in cols if c.split("_", 1)[-1] in ("sf", "sa", "tf", "ta")]
    xg = [c for c in cols if c.split("_", 1)[-1] in ("xf", "xa")]
    base = [*RATING_COLS, "div_code", *goals, *sched]
    return {"goals": base, "xg": [*base, *shots, *xg]}
