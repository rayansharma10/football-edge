"""xG-based features: attach per-match xG to the match table.

Sources, in order of preference: Understat (``match_xg`` table, top-5 leagues, available_at =
kickoff + 3h) and the football-data ``HxG``/``AxG`` columns (2026/27 onward). The xG columns
are *outcomes of the match itself*: they are only ever read for matches that have already
been released by the causal sweep in :mod:`fedge.features.form` (kickoff + embargo < bet time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_xg(matches: pd.DataFrame, match_xg: pd.DataFrame | None) -> pd.DataFrame:
    """Return ``matches`` with float ``xg_home`` / ``xg_away`` (NaN where unavailable)."""
    out = matches.copy()
    xh = pd.Series(np.nan, index=out.index, dtype=float)
    xa = pd.Series(np.nan, index=out.index, dtype=float)
    if match_xg is not None and len(match_xg):
        mx = match_xg.drop_duplicates("match_id").set_index("match_id")
        xh = out["match_id"].map(mx["xg_home"]).astype(float)
        xa = out["match_id"].map(mx["xg_away"]).astype(float)
    if "HxG" in out.columns:
        xh = xh.fillna(out["HxG"].astype(float))
        xa = xa.fillna(out["AxG"].astype(float))
    out["xg_home"] = xh.to_numpy()
    out["xg_away"] = xa.to_numpy()
    return out
