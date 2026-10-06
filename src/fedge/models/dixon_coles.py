"""Dixon-Coles (penaltyblog) with exponential time decay, walk-forward refits.

Design (Phase 3, card 3.1):

* Team parameters are estimated **per division** (a team's attack strength is not comparable
  across divisions), on an expanding window capped at :data:`TRAIN_YEARS` years.
* Matches are weighted by exponential time decay ``exp(-xi * years_before_fold)``.
* ``xi`` is re-selected **in-fold** once per (division, test season) from :data:`XI_GRID` by
  inner-validation log-loss, using only matches strictly before the test season (see
  :func:`select_xi`). It is per-season, not per-matchweek, to bound runtime; xi moves slowly and
  the grid is small.
* The model is **refit once per matchweek** (:func:`fedge.backtest.walkforward.matchweek_folds`),
  each fold trained on everything before the fold's test block minus the 3h embargo.
* Probabilities are computed from the fitted parameters by :func:`score_grid`, a vectorised
  replication of penaltyblog's Dixon-Coles score grid (verified against
  ``DixonColesGoalModel.predict`` in ``tests/test_dixon_coles.py``). This is done so that teams
  with no training history (promoted sides) fall back to the prior parameters
  (:data:`PRIOR_ATTACK` / :data:`PRIOR_DEFENCE`) instead of raising, and so that a matchweek of
  predictions costs milliseconds.
* Neutral venues are not modelled (no flag in the ingest); the 2020/21 COVID neutral-venue
  matches are therefore treated as normal home matches. Documented, not fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel
from scipy.stats import poisson

from fedge.backtest.walkforward import EMBARGO, matchweek_folds
from fedge.report.metrics import log_loss

MAX_GOALS = 16  # matches penaltyblog's default grid size
DEFAULT_XI = 0.0065  # ~1 year half-life (REPORT 5.1)
XI_GRID = (0.0040, 0.0055, 0.0065, 0.0080, 0.0100, 0.0130)
TRAIN_YEARS = 5.0  # rolling training window (M1: 5-year rolling window with time decay)
PRIOR_ATTACK = 1.0  # penaltyblog's init values; exp(1 + -1) = 1 goal vs an average defence
PRIOR_DEFENCE = -1.0
MIN_TRAIN_MATCHES = 200
VAL_MATCHES = 380  # inner-validation block for xi selection (~1 season)
START_SEASON = "2016/17"
_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class DCParams:
    """Fitted Dixon-Coles parameters in penaltyblog's convention."""

    teams: tuple[str, ...]
    attack: np.ndarray  # log-scale, mean 1
    defence: np.ndarray  # log-scale, mean -1
    hfa: float
    rho: float

    @property
    def n_teams(self) -> int:
        return len(self.teams)


def decay_weights(kickoffs: pd.Series, ref, xi: float) -> np.ndarray:
    """``exp(-xi * years)`` for each kickoff, years measured back from ``ref``."""
    age_days = (pd.Timestamp(ref) - pd.to_datetime(kickoffs, utc=True)).dt.total_seconds() / 86400.0
    return np.exp(-xi * np.asarray(age_days, dtype=float) / _DAYS_PER_YEAR)


def _window(matches: pd.DataFrame, ref, years: float) -> pd.DataFrame:
    lo = pd.Timestamp(ref) - pd.Timedelta(days=_DAYS_PER_YEAR * years)
    return matches.loc[matches["kickoff_utc"] >= lo]


def fit_dc(
    matches: pd.DataFrame,
    xi: float = DEFAULT_XI,
    ref=None,
    train_years: float = TRAIN_YEARS,
    min_matches: int = MIN_TRAIN_MATCHES,
) -> DCParams | None:
    """Fit Dixon-Coles on ``matches`` (played, one division) with decay weights anchored at ``ref``.

    Returns ``None`` when there is not enough data or the optimiser fails; the caller must treat
    that as "no prediction for this fold" rather than substituting a guess.
    """
    tr = matches.loc[matches["FTHG"].notna() & matches["FTAG"].notna()]
    if ref is not None:
        tr = tr.loc[tr["kickoff_utc"] < pd.Timestamp(ref)]
        tr = _window(tr, ref, train_years)
    if len(tr) < min_matches or tr["home"].nunique() < 2 or tr["away"].nunique() < 2:
        return None
    if ref is None:
        ref = pd.to_datetime(tr["kickoff_utc"], utc=True).max()
    else:
        ref = pd.Timestamp(ref)
    weights = decay_weights(tr["kickoff_utc"], ref, xi)
    try:
        model = DixonColesGoalModel(
            tr["FTHG"].astype(int).to_numpy(),
            tr["FTAG"].astype(int).to_numpy(),
            tr["home"].to_numpy(),
            tr["away"].to_numpy(),
            weights=weights,
        )
        model.fit()
        params = model.params_array
    except Exception:  # noqa: BLE001 - a failed fit is a missing fold, never a fabricated value
        return None
    if not np.all(np.isfinite(params)):
        return None
    n = len(model.teams)
    return DCParams(
        teams=tuple(str(t) for t in model.teams),
        attack=params[:n].copy(),
        defence=params[n : 2 * n].copy(),
        hfa=float(params[-2]),
        rho=float(params[-1]),
    )


def score_grid(lh, la, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Dixon-Coles score grid P(H=i, A=j) for arrays of expected goals.

    ``tau`` correction: (0,0) 1-lh*la*rho, (0,1) 1+lh*rho, (1,0) 1+la*rho, (1,1) 1-rho.
    """
    lh = np.atleast_1d(np.asarray(lh, dtype=float))
    la = np.atleast_1d(np.asarray(la, dtype=float))
    k = np.arange(max_goals)
    grid = poisson.pmf(k, lh[:, None])[:, :, None] * poisson.pmf(k, la[:, None])[:, None, :]
    grid[:, 0, 0] *= 1.0 - lh * la * rho
    grid[:, 0, 1] *= 1.0 + lh * rho
    grid[:, 1, 0] *= 1.0 + la * rho
    grid[:, 1, 1] *= 1.0 - rho
    np.clip(grid, 0.0, None, out=grid)
    return grid


def _expand(params: DCParams) -> tuple[pd.Series, pd.Series]:
    return (
        pd.Series(params.attack, index=list(params.teams)),
        pd.Series(params.defence, index=list(params.teams)),
    )


def predict(params: DCParams, matches: pd.DataFrame, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """1X2 and over/under 2.5 probabilities for ``matches`` under ``params``.

    Teams absent from the fitted parameters (promoted sides) use the prior attack/defence.
    Returns a frame indexed like ``matches``.
    """
    attack, defence = _expand(params)
    ah = matches["home"].map(attack).fillna(PRIOR_ATTACK).to_numpy(dtype=float)
    dh = matches["home"].map(defence).fillna(PRIOR_DEFENCE).to_numpy(dtype=float)
    aa = matches["away"].map(attack).fillna(PRIOR_ATTACK).to_numpy(dtype=float)
    da = matches["away"].map(defence).fillna(PRIOR_DEFENCE).to_numpy(dtype=float)
    lh = np.exp(ah + da + params.hfa)
    la = np.exp(aa + dh)
    grid = score_grid(lh, la, params.rho, max_goals)
    total = grid.sum(axis=(1, 2), keepdims=True)
    total = np.where(total <= 0, 1.0, total)
    grid = grid / total
    k = np.arange(max_goals)
    over = (k[:, None] + k[None, :]) > 2.5
    return pd.DataFrame(
        {
            "p_home": np.tril(grid, -1).sum(axis=(1, 2)),
            "p_draw": np.einsum("nii->n", grid),
            "p_away": np.triu(grid, 1).sum(axis=(1, 2)),
            "p_over25": grid[:, over].sum(axis=1),
            "p_under25": grid[:, ~over].sum(axis=1),
        },
        index=matches.index,
    )


def select_xi(
    matches: pd.DataFrame,
    test_start,
    xi_grid: tuple[float, ...] = XI_GRID,
    train_years: float = TRAIN_YEARS,
    val_matches: int = VAL_MATCHES,
    min_matches: int = MIN_TRAIN_MATCHES,
) -> tuple[float, float]:
    """Pick xi in-fold by log-loss on an inner-validation block (returns ``(xi, log_loss)``).

    The validation block is the last ``val_matches`` played matches strictly before
    ``test_start - EMBARGO``; the candidates are fitted on the matches before that block only.
    Falls back to :data:`DEFAULT_XI` when there is not enough in-fold data.
    """
    train = matches.loc[
        (matches["kickoff_utc"] < pd.Timestamp(test_start) - EMBARGO) & matches["FTHG"].notna()
    ].sort_values("kickoff_utc", kind="mergesort")
    if len(train) < val_matches + min_matches:
        return DEFAULT_XI, float("nan")
    val = train.iloc[-val_matches:]
    val_start = pd.Timestamp(val["kickoff_utc"].min())
    inner = train.iloc[:-val_matches]
    y = outcome_codes(val)
    best_xi, best_loss = DEFAULT_XI, np.inf
    for xi in xi_grid:
        params = fit_dc(inner, xi=xi, ref=val_start, train_years=train_years,
                        min_matches=min_matches)
        if params is None:
            continue
        p = predict(params, val)
        loss = log_loss(p[["p_home", "p_draw", "p_away"]].to_numpy(), y)
        if loss < best_loss:
            best_xi, best_loss = float(xi), float(loss)
    return best_xi, best_loss


def outcome_codes(matches: pd.DataFrame) -> np.ndarray:
    """Outcome class codes in the ordinal H, D, A order used by fedge.report.metrics."""
    return matches["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)


def run_league(
    matches: pd.DataFrame,
    start_season: str = START_SEASON,
    xi_grid: tuple[float, ...] = XI_GRID,
    train_years: float = TRAIN_YEARS,
    min_train_matches: int = MIN_TRAIN_MATCHES,
    val_matches: int = VAL_MATCHES,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward Dixon-Coles predictions for one division.

    ``matches`` must be one division: with tz-aware ``kickoff_utc``, ``season`` in ``YYYY/YY``,
    ``home``, ``away``, ``FTHG``, ``FTAG``, ``FTR``, ``match_id``.

    Returns ``(predictions, info)``: predictions indexed by ``match_id`` with columns
    ``p_home, p_draw, p_away, p_over25, p_under25, season, fold, xi``; info holds the per-season
    xi choices, the fold count and the number of folds with no usable fit.
    """
    m = matches.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    seasons = sorted(
        {s for s in m["season"].unique() if int(s[:4]) >= int(start_season[:4])},
        key=lambda s: int(s[:4]),
    )
    xi_by_season: dict[str, float] = {}
    xi_loss: dict[str, float] = {}
    for season in seasons:
        test_start = m.loc[m["season"] == season, "kickoff_utc"].min()
        xi, loss = select_xi(m, test_start, xi_grid, train_years, val_matches, min_train_matches)
        xi_by_season[season] = xi
        xi_loss[season] = loss
    info = {"xi_by_season": xi_by_season, "xi_loss": xi_loss, "n_folds": 0, "n_failed": 0}
    if not seasons:
        return _empty_preds(), info
    first_start = m.loc[m["season"] == seasons[0], "kickoff_utc"].min()
    parts: list[pd.DataFrame] = []
    for fold in matchweek_folds(m, min_train_matches=min_train_matches, start=first_start):
        test = m.iloc[fold.test_idx]
        xi = xi_by_season.get(str(test["season"].iloc[0]), DEFAULT_XI)
        params = fit_dc(m.iloc[fold.train_idx], xi=xi, ref=fold.test_start, train_years=train_years,
                        min_matches=min_train_matches)
        info["n_folds"] += 1
        if params is None:
            info["n_failed"] += 1
            continue
        p = predict(params, test)
        p["season"] = test["season"].to_numpy()
        p["fold"] = fold.label
        p["xi"] = xi
        p.index = test["match_id"].to_numpy()
        parts.append(p)
    if not parts:
        return _empty_preds(), info
    out = pd.concat(parts)
    out.index.name = "match_id"
    return out, info


def _empty_preds() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["p_home", "p_draw", "p_away", "p_over25", "p_under25", "season", "fold", "xi"],
        index=pd.Index([], name="match_id"),
    )
