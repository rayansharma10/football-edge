"""Probability calibration: isotonic, Beta, Venn-Abers; chosen in-fold by log-loss.

All calibrators map a raw probability matrix ``(n, k)`` to a calibrated one. For ``k > 2``
each class is calibrated one-vs-rest on its own column and the rows are renormalised (the
standard OvR recipe; it is not a joint multiclass calibrator). Binary forecasts use the same
code with ``k = 2`` (columns ``[over, under]``; both columns are calibrated and renormalised).

* ``isotonic``: ``sklearn.isotonic.IsotonicRegression`` (monotone, clipped out of range).
* ``beta``: Kull et al. beta calibration, ``logit q = a ln p - b ln(1-p) + c``, fitted as a
  2-feature logistic regression (the ``a, b >= 0`` constraint is not enforced; with the
  GBM outputs used here the fitted ``a, b`` are positive in practice).
* ``venn_abers``: the ``venn-abers`` package (inductive VA, ``p' = p1 / (1 - p0 + p1)``).
* ``identity``: no calibration; always a candidate so calibration is never forced on a model
  that is already calibrated.

:func:`select_calibrator` picks the candidate with the lowest log-loss on the *later half* of
a chronologically ordered calibration set after fitting each on the earlier half; the winner
is then refit on the whole calibration set. No random splits.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
METHODS = ("identity", "isotonic", "beta", "venn_abers")

Calibrator = Callable[[np.ndarray], np.ndarray]


def _renorm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return p / p.sum(axis=1, keepdims=True)


def _check(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("p must be (n, k) and match len(y)")
    return p, y


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))))


def fit_identity(p, y) -> Calibrator:
    return lambda q: _renorm(np.asarray(q, dtype=float))


def fit_isotonic(p, y) -> Calibrator:
    p, y = _check(p, y)
    models = []
    for k in range(p.shape[1]):
        iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
        iso.fit(p[:, k], (y == k).astype(float))
        models.append(iso)

    def apply(q):
        q = np.asarray(q, dtype=float)
        return _renorm(np.column_stack([m.predict(q[:, k]) for k, m in enumerate(models)]))

    return apply


def _beta_features(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1 - EPS)
    return np.column_stack([np.log(x), -np.log1p(-x)])


def fit_beta(p, y) -> Calibrator:
    p, y = _check(p, y)
    models = []
    for k in range(p.shape[1]):
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
        lr.fit(_beta_features(p[:, k]), (y == k).astype(int))
        models.append(lr)

    def apply(q):
        q = np.asarray(q, dtype=float)
        return _renorm(
            np.column_stack(
                [m.predict_proba(_beta_features(q[:, k]))[:, 1] for k, m in enumerate(models)]
            )
        )

    return apply


def fit_venn_abers(p, y) -> Calibrator:
    from venn_abers import VennAbers

    p, y = _check(p, y)
    models = []
    for k in range(p.shape[1]):
        va = VennAbers()
        x = np.clip(p[:, k], EPS, 1 - EPS)
        va.fit(np.column_stack([1 - x, x]), (y == k).astype(int))
        models.append(va)

    def apply(q):
        q = np.asarray(q, dtype=float)
        cols = []
        for k, va in enumerate(models):
            x = np.clip(q[:, k], EPS, 1 - EPS)
            p_prime, _ = va.predict_proba(np.column_stack([1 - x, x]))
            cols.append(p_prime[:, 1])
        return _renorm(np.column_stack(cols))

    return apply


FITTERS: dict[str, Callable[..., Calibrator]] = {
    "identity": fit_identity,
    "isotonic": fit_isotonic,
    "beta": fit_beta,
    "venn_abers": fit_venn_abers,
}


def select_calibrator(
    p_cal, y_cal, methods: tuple[str, ...] = METHODS, min_rows: int = 400
) -> tuple[str, Calibrator, dict[str, float]]:
    """Choose the calibrator with the lowest held-out log-loss; refit it on all rows.

    ``p_cal`` / ``y_cal`` must be in chronological order (earlier rows first). Returns
    ``(name, calibrator, {method: held-out log-loss})``. With fewer than ``min_rows`` rows the
    identity is returned (not enough data to calibrate).
    """
    p, y = _check(p_cal, y_cal)
    if len(y) < min_rows:
        return "identity", fit_identity(p, y), {}
    cut = len(y) // 2
    scores: dict[str, float] = {}
    for name in methods:
        try:
            cal = FITTERS[name](p[:cut], y[:cut])
            scores[name] = _log_loss(cal(p[cut:]), y[cut:])
        except Exception:  # a candidate that fails to fit is simply not eligible
            continue
    if not scores:
        return "identity", fit_identity(p, y), {}
    best = min(scores, key=lambda k: (scores[k], methods.index(k)))
    return best, FITTERS[best](p, y), scores
