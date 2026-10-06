"""Own Elo and pi-ratings, as-of, plus an ordered-logit 1X2 model on the rating difference.

Design (Phase 3, card 3.2):

* **Causal sweeps.** Both ratings are computed once per division with a single chronological
  sweep (:func:`elo_sweep`, :func:`pi_sweep`). A match's result enters the state only once
  ``kickoff + EMBARGO <= kickoff_of_the_match_being_predicted``, i.e. the same 3h rule
  :mod:`fedge.backtest.walkforward` uses for folds and the same
  ``available_at < bet_time`` discipline as :mod:`fedge.features.asof` (a result's
  ``available_at`` is ``kickoff + 3h``). ``tests/test_ratings.py`` proves the sweep equals a
  brute-force recomputation from ``asof``-restricted history.
* **Both models feed one ordinal model.** The 1X2 model is an ordered logit on a single
  feature: the pre-match rating difference (home minus away). For Elo that difference is
  bookmaker-free; the home-advantage term enters the Elo *update* and the logit cutpoints.
  For pi-ratings it is ``PiRatingSystem.expected_goal_difference`` (home rating minus away
  rating), which already carries the venue asymmetry.
* **Hyper-parameters are chosen in-fold.** ``(k, HFA)`` for Elo and ``(alpha, beta)`` for pi
  are selected per (division, test season) by inner-validation log-loss over :data:`ELO_GRID`
  / :data:`PI_GRID`, using only matches strictly before the test season. The ordered logit is
  refit on the same in-fold window. Grids are deliberately small to bound runtime; the chosen
  parameters are reported so the selection is auditable.
* Ratings are per division (a promoted side starts at the initial rating); no cross-division
  carry-over is modelled. Documented as a limitation, not fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from penaltyblog.ratings import PiRatingSystem
from scipy.optimize import minimize

from fedge.backtest.walkforward import EMBARGO
from fedge.report.metrics import log_loss

START_SEASON = "2016/17"
TRAIN_YEARS = 5.0
VAL_MATCHES = 380
MIN_TRAIN_MATCHES = 200
ELO_START = 1500.0
ELO_GRID = ((10.0, 40.0), (20.0, 60.0), (20.0, 100.0), (30.0, 80.0))  # (k, home advantage)
PI_GRID = ((0.10, 0.05), (0.15, 0.10), (0.20, 0.10))  # (alpha, beta)
PI_K = 0.75
PI_SIGMA = 1.0
_DAYS_PER_YEAR = 365.25


# --------------------------------------------------------------------------- Elo
def goal_diff_weight(goal_diff: int) -> float:
    """Standard World-Football-Elo goal-difference multiplier."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def elo_expected(rating_diff: float, hfa: float) -> float:
    """Home win expectation ``1 / (1 + 10 ** (-(dr + hfa) / 400))``."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_diff + hfa) / 400.0))


def elo_from_results(
    matches: pd.DataFrame, k: float, hfa: float, start: float = ELO_START
) -> dict[str, float]:
    """Final Elo ratings after applying every played match in ``matches`` (test helper)."""
    ratings: dict[str, float] = {}
    for home, away, gh, ga in zip(
        matches["home"], matches["away"], matches["FTHG"], matches["FTAG"], strict=True
    ):
        rh = ratings.get(home, start)
        ra = ratings.get(away, start)
        exp_home = elo_expected(rh - ra, hfa)
        score = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        delta = k * goal_diff_weight(int(gh) - int(ga)) * (score - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def _played(matches: pd.DataFrame) -> pd.DataFrame:
    return matches.loc[matches["FTHG"].notna() & matches["FTAG"].notna()]


def _utc_naive(series: pd.Series) -> np.ndarray:
    return series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


def elo_sweep(
    matches: pd.DataFrame, k: float, hfa: float, start: float = ELO_START, embargo=EMBARGO
) -> np.ndarray:
    """Pre-match Elo rating difference (home minus away) for every row of ``matches``.

    ``matches`` must be one division, sorted by ``kickoff_utc``, played matches only.
    """
    ko = _utc_naive(matches["kickoff_utc"])
    cut = ko - embargo.to_timedelta64()
    home = matches["home"].to_numpy()
    away = matches["away"].to_numpy()
    gh = matches["FTHG"].to_numpy(dtype=int)
    ga = matches["FTAG"].to_numpy(dtype=int)
    out = np.empty(len(matches), dtype=float)
    ratings: dict[str, float] = {}
    ptr = 0
    for i in range(len(matches)):
        while ptr < i and ko[ptr] <= cut[i]:
            h, a = home[ptr], away[ptr]
            rh = ratings.get(h, start)
            ra = ratings.get(a, start)
            score = 1.0 if gh[ptr] > ga[ptr] else (0.0 if gh[ptr] < ga[ptr] else 0.5)
            delta = k * goal_diff_weight(gh[ptr] - ga[ptr]) * (score - elo_expected(rh - ra, hfa))
            ratings[h] = rh + delta
            ratings[a] = ra - delta
            ptr += 1
        out[i] = ratings.get(home[i], start) - ratings.get(away[i], start)
    return out


# --------------------------------------------------------------------------- pi-ratings
def pi_sweep(
    matches: pd.DataFrame,
    alpha: float,
    beta: float,
    k: float = PI_K,
    sigma: float = PI_SIGMA,
    embargo=EMBARGO,
) -> np.ndarray:
    """Pre-match pi-rating expected goal difference (home minus away) for every row.

    ``matches`` must be one division, sorted by ``kickoff_utc``, played matches only.
    """
    ko = _utc_naive(matches["kickoff_utc"])
    cut = ko - embargo.to_timedelta64()
    home = matches["home"].to_numpy()
    away = matches["away"].to_numpy()
    gh = matches["FTHG"].to_numpy(dtype=int)
    ga = matches["FTAG"].to_numpy(dtype=int)
    out = np.empty(len(matches), dtype=float)
    sys = PiRatingSystem(alpha=alpha, beta=beta, k=k, sigma=sigma)
    ptr = 0
    for i in range(len(matches)):
        while ptr < i and ko[ptr] <= cut[i]:
            sys.update_ratings(home[ptr], away[ptr], gh[ptr] - ga[ptr])
            ptr += 1
        out[i] = sys.expected_goal_difference(home[i], away[i])
    return out


# --------------------------------------------------------------------------- ordered logit
@dataclass(frozen=True)
class OrderedLogit:
    """Ordered logit with latent ``z = a0 + b * x`` and cutpoints ``c1 < c2 = c1 + exp(d)``.

    Class indices ascend with ``z``, so with ``x`` = (home minus away) rating difference the
    fitted ``b`` is negative: a stronger home side predicts the *lower* code (H = 0).
    """

    a0: float
    b: float
    c1: float
    d: float

    @property
    def c2(self) -> float:
        return self.c1 + float(np.exp(self.d))


def _sigmoid(v: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow for |v| in the thousands)."""
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    pos = v >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-v[pos]))
    ev = np.exp(v[~pos])
    out[~pos] = ev / (1.0 + ev)
    return out


def ordered_logit_probs(theta: OrderedLogit, x) -> np.ndarray:
    """Class probabilities (n, 3) for feature values ``x``."""
    x = np.asarray(x, dtype=float)
    z = theta.a0 + theta.b * x
    p0 = _sigmoid(theta.c1 - z)
    p2 = 1.0 - _sigmoid(theta.c2 - z)
    p1 = np.clip(1.0 - p0 - p2, 0.0, 1.0)
    out = np.column_stack([p0, p1, p2])
    return out / out.sum(axis=1, keepdims=True)


def _nll(theta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> float:
    a0, b, c1, d = theta
    c2 = c1 + np.exp(d)
    z = a0 + b * x
    p0 = _sigmoid(c1 - z)
    p2 = 1.0 - _sigmoid(c2 - z)
    p1 = np.clip(1.0 - p0 - p2, 1e-12, None)
    p = np.column_stack([np.clip(p0, 1e-12, 1.0), p1, np.clip(p2, 1e-12, 1.0)])
    ll = np.log(p[np.arange(len(y)), y])
    return float(-ll.mean() + l2 * (a0**2 + b**2))


_INIT = np.array([0.0, -0.2, -0.2, 0.09])
_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0), (-8.0, 8.0), (-6.0, 6.0)]


def fit_ordered_logit(x, y, l2: float = 1e-4) -> OrderedLogit:
    """Fit the ordered logit by L-BFGS-B (deterministic; fixed init, no randomness)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must have the same non-zero length")
    res = minimize(_nll, _INIT, args=(x, y, l2), method="L-BFGS-B", bounds=_BOUNDS)
    a0, b, c1, d = (float(v) for v in res.x)
    return OrderedLogit(a0=a0, b=b, c1=c1, d=d)


# --------------------------------------------------------------------------- walk-forward
def outcome_codes(matches: pd.DataFrame) -> np.ndarray:
    """Outcome class codes in the ordinal H, D, A order used by fedge.report.metrics."""
    return matches["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)


def _candidates(matches: pd.DataFrame, elo_grid, pi_grid) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    for k, hfa in elo_grid:
        feats[f"elo_k{k:g}_hfa{hfa:g}"] = elo_sweep(matches, k, hfa)
    for alpha, beta in pi_grid:
        feats[f"pi_a{alpha:g}_b{beta:g}"] = pi_sweep(matches, alpha, beta)
    return feats


def _select(
    feats: dict[str, np.ndarray],
    keys: list[str],
    m: pd.DataFrame,
    test_start,
    train_years: float,
    val_matches: int,
    min_matches: int,
) -> tuple[str, OrderedLogit | None, float]:
    """Inner-validation selection over ``keys``: returns (best key, logit, val log-loss)."""
    train = m.loc[m["kickoff_utc"] < pd.Timestamp(test_start) - EMBARGO]
    if len(train) < val_matches + min_matches:
        return keys[0], None, float("nan")  # type: ignore[return-value]
    val = train.iloc[-val_matches:]
    val_start = pd.Timestamp(val["kickoff_utc"].min())
    lo = val_start - pd.Timedelta(days=_DAYS_PER_YEAR * train_years)
    inner = train.loc[train["kickoff_utc"] < val_start - EMBARGO]
    inner = inner.loc[inner["kickoff_utc"] >= lo]
    y_in, y_val = outcome_codes(inner), outcome_codes(val)
    i_idx, v_idx = inner.index.to_numpy(), val.index.to_numpy()
    best_key, best_theta, best_loss = keys[0], None, np.inf
    for key in keys:
        if len(inner) < min_matches:
            break
        theta = fit_ordered_logit(feats[key][i_idx], y_in)
        loss = log_loss(ordered_logit_probs(theta, feats[key][v_idx]), y_val)
        if loss < best_loss:
            best_key, best_theta, best_loss = key, theta, float(loss)
    return best_key, best_theta, best_loss


def run_league(
    matches: pd.DataFrame,
    start_season: str = START_SEASON,
    elo_grid=ELO_GRID,
    pi_grid=PI_GRID,
    train_years: float = TRAIN_YEARS,
    val_matches: int = VAL_MATCHES,
    min_train_matches: int = MIN_TRAIN_MATCHES,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward Elo-logit and pi-logit predictions for one division.

    ``matches`` must be one division: tz-aware ``kickoff_utc``, ``season``, ``home``, ``away``,
    ``FTHG``, ``FTAG``, ``FTR``, ``match_id``. Returns ``(predictions, info)``; predictions are
    indexed by ``match_id`` with ``p_elo_*`` and ``p_pi_*`` (H, D, A order) plus the chosen
    hyper-parameters per season.
    """
    m = _played(matches).sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    if m.empty:
        return _empty_preds(), {"selected": {}}
    feats = _candidates(m, elo_grid, pi_grid)
    elo_keys = [f"elo_k{k:g}_hfa{hfa:g}" for k, hfa in elo_grid]
    pi_keys = [f"pi_a{a:g}_b{b:g}" for a, b in pi_grid]
    seasons = sorted(
        {s for s in m["season"].unique() if int(s[:4]) >= int(start_season[:4])},
        key=lambda s: int(s[:4]),
    )
    selected: dict[str, dict] = {}
    parts: list[pd.DataFrame] = []
    for season in seasons:
        rows = m.loc[m["season"] == season]
        test_start = rows["kickoff_utc"].min()
        ek, etheta, eloss = _select(feats, elo_keys, m, test_start, train_years, val_matches,
                                    min_train_matches)
        pk, ptheta, ploss = _select(feats, pi_keys, m, test_start, train_years, val_matches,
                                    min_train_matches)
        selected[season] = {
            "elo": ek,
            "pi": pk,
            "elo_val_logloss": eloss,
            "pi_val_logloss": ploss,
            "theta_elo": etheta,
            "theta_pi": ptheta,
        }
        if etheta is None or ptheta is None:
            continue  # not enough in-fold history: no prediction for this season
        idx = rows.index.to_numpy()
        pe = ordered_logit_probs(etheta, feats[ek][idx])
        pp = ordered_logit_probs(ptheta, feats[pk][idx])
        part = pd.DataFrame(
            {
                "p_elo_home": pe[:, 0],
                "p_elo_draw": pe[:, 1],
                "p_elo_away": pe[:, 2],
                "p_pi_home": pp[:, 0],
                "p_pi_draw": pp[:, 1],
                "p_pi_away": pp[:, 2],
                "season": rows["season"].to_numpy(),
                "elo_params": ek,
                "pi_params": pk,
            },
            index=rows["match_id"].to_numpy(),
        )
        parts.append(part)
    if not parts:
        return _empty_preds(), {"selected": selected}
    out = pd.concat(parts)
    out.index.name = "match_id"
    return out, {"selected": selected}


def _empty_preds() -> pd.DataFrame:
    cols = ["p_elo_home", "p_elo_draw", "p_elo_away", "p_pi_home", "p_pi_draw", "p_pi_away",
            "season", "elo_params", "pi_params"]
    return pd.DataFrame(columns=cols, index=pd.Index([], name="match_id"))
