"""Pure scoreline model: Dixon-Coles grid reconciled to a target 1X2.

Method (documented on the dashboard too):

1. Start from Dixon-Coles expected goals ``lh``/``la`` and ``rho`` (fitted on goals per division by
   :func:`fedge.models.dixon_coles.fit_dc`) and build the score grid with
   :func:`fedge.models.dixon_coles.score_grid`.
2. If a ``target`` 1X2 (the live ``lgbd_xg`` model) is given, solve for the two expected-goal
   rates (log scale, ``rho`` kept) whose grid reproduces the target home and away win
   probabilities (the draw then follows). If the solver does not converge, or to polish residual
   error, the grid is rescaled by outcome class (iterative proportional fitting on the
   home-win / draw / away-win cells), which matches the target exactly.
3. Everything else (expected goals, top scorelines, over/under 2.5, BTTS) is read off the
   reconciled grid.

The 1X2 therefore *is* the LightGBM 1X2 (which is itself anchored on the de-margined market
price); only the scoreline *shape* inside each outcome and the goal total come from Dixon-Coles.
The scoreline probabilities are a model estimate, not a market price.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root

from fedge.models.dixon_coles import score_grid

GRID_SHOW = 6  # scorelines 0..5 are returned as the display grid
MAX_GOALS = 12  # internal grid size (mass beyond is renormalised away, negligible)
LAM_MIN, LAM_MAX = 0.05, 8.0


def _grid(lh: float, la: float, rho: float, max_goals: int) -> np.ndarray:
    g = score_grid(lh, la, rho, max_goals)[0]
    s = g.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("degenerate score grid")
    return g / s


def _outcome_masks(n: int):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return i > j, i == j, i < j  # home win, draw, away win (rows = home goals)


def _outcome_probs(grid: np.ndarray) -> np.ndarray:
    mh, md, ma = _outcome_masks(grid.shape[0])
    return np.array([grid[mh].sum(), grid[md].sum(), grid[ma].sum()])


def _ipf(grid: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rescale cells by outcome class so the 1X2 equals ``target`` exactly."""
    out = grid.copy()
    cur = _outcome_probs(out)
    for mask, c, t in zip(_outcome_masks(grid.shape[0]), cur, target, strict=True):
        if c > 0:
            out[mask] *= t / c
        elif t > 1e-12:
            raise ValueError("cannot reconcile: DC grid has zero mass in a target outcome")
    return out / out.sum()


def _validate_target(target) -> np.ndarray:
    t = np.asarray(target, dtype=float)
    if t.shape != (3,) or not np.all(np.isfinite(t)) or np.any(t < 0) or t.sum() <= 0:
        raise ValueError(f"target 1X2 must be 3 non-negative finite numbers, got {target!r}")
    if abs(t.sum() - 1.0) > 1e-3:
        raise ValueError(f"target 1X2 must sum to 1, got {t.sum():.6f}")
    return t / t.sum()


def predict_match(
    lh: float,
    la: float,
    rho: float = 0.0,
    target=None,
    max_goals: int = MAX_GOALS,
    top_n: int = 5,
) -> dict:
    """Scoreline distribution and derived markets for one match.

    ``lh``/``la``: Dixon-Coles expected goals (home, away); ``rho``: DC low-score correction;
    ``target``: optional (H, D, A) probabilities the grid must agree with.
    """
    if not (np.isfinite(lh) and np.isfinite(la) and lh > 0 and la > 0):
        raise ValueError(f"expected goals must be positive and finite, got {lh!r}, {la!r}")
    if max_goals <= GRID_SHOW:
        raise ValueError("max_goals must exceed the displayed grid size")
    lh0, la0 = float(lh), float(la)
    grid = _grid(lh0, la0, rho, max_goals)
    dc_1x2 = _outcome_probs(grid)
    reconciled, method = False, "dixon-coles"
    lam_h, lam_a = lh0, la0
    if target is not None:
        t = _validate_target(target)

        def resid(x):
            p = _outcome_probs(_grid(*np.clip(np.exp(x), LAM_MIN, LAM_MAX), rho, max_goals))
            return [p[0] - t[0], p[2] - t[2]]

        try:
            sol = root(resid, np.log([lh0, la0]), method="hybr", tol=1e-12)
            cand = np.clip(np.exp(sol.x), LAM_MIN, LAM_MAX)
            ok = sol.success and np.max(np.abs(resid(np.log(cand)))) < 1e-8
        except (ValueError, FloatingPointError):
            ok = False
        if ok:
            lam_h, lam_a = float(cand[0]), float(cand[1])
            grid = _grid(lam_h, lam_a, rho, max_goals)
            method = "dixon-coles lambdas solved to match 1X2"
        else:
            method = "dixon-coles grid rescaled by outcome class (IPF) to match 1X2"
        grid = _ipf(grid, t)  # exact (no-op when the solve already matched)
        reconciled = True

    n = grid.shape[0]
    k = np.arange(n)
    p1x2 = _outcome_probs(grid)
    tot = k[:, None] + k[None, :]
    flat = np.argsort(-grid, axis=None, kind="stable")[:top_n]
    top = [
        {"score": f"{i}-{j}", "home": int(i), "away": int(j), "p": float(grid[i, j])}
        for i, j in (np.unravel_index(f, grid.shape) for f in flat)
    ]
    return {
        "p_home": float(p1x2[0]),
        "p_draw": float(p1x2[1]),
        "p_away": float(p1x2[2]),
        "xg_home": float((grid.sum(axis=1) * k).sum()),
        "xg_away": float((grid.sum(axis=0) * k).sum()),
        "p_over25": float(grid[tot > 2.5].sum()),
        "p_under25": float(grid[tot <= 2.5].sum()),
        "p_btts": float(grid[1:, 1:].sum()),
        "top_scores": top,
        "grid": grid[:GRID_SHOW, :GRID_SHOW].tolist(),
        "grid_mass_shown": float(grid[:GRID_SHOW, :GRID_SHOW].sum()),
        "dc_p": [float(x) for x in dc_1x2],
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "rho": float(rho),
        "reconciled": reconciled,
        "method": method,
    }
