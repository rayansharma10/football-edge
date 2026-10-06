"""Pure metric functions for the accuracy tracker (numpy only; no I/O).

Conventions: outcome index 0 = home win (H), 1 = draw (D), 2 = away win (A); ``P`` is an
``(n, 3)`` array of probabilities in that order; ``grids`` is ``(n, G, G)`` with ``grids[k, i, j]``
the probability of home goals ``i`` and away goals ``j`` (G = 8: 0-7 goals, rows = home).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6  # probability floor for log scores (a score beyond the 0-7 grid gets this)
OU_LINE = 2.5
OUTCOMES = ("H", "D", "A")
SMALL_N = 100


def outcome_index(home_goals, away_goals) -> np.ndarray:
    h, a = np.asarray(home_goals), np.asarray(away_goals)
    return np.where(h > a, 0, np.where(h == a, 1, 2)).astype(int)


# ----------------------------------------------------------------------------- winner
def hit_rate(pred_idx, y) -> float:
    """Share of matches whose predicted outcome (argmax) was right."""
    return float(np.mean(np.asarray(pred_idx) == np.asarray(y)))


def brier_multiclass(P, y) -> float:
    """Mean over matches of ``sum_k (p_k - 1[k = y])^2`` (0 = perfect, 2 = worst)."""
    P = np.asarray(P, dtype=float)
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def log_loss(P, y) -> float:
    """Mean ``-log p(actual outcome)`` (probabilities floored at ``EPS``)."""
    P = np.asarray(P, dtype=float)
    p = np.clip(P[np.arange(len(P)), np.asarray(y, dtype=int)], EPS, 1.0)
    return float(np.mean(-np.log(p)))


def rps(P, y) -> float:
    """Ranked probability score for the ordered outcomes H < D < A (lower is better)."""
    P = np.asarray(P, dtype=float)
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    cum = np.cumsum(P, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum(cum**2, axis=1) / 2.0))


def calibration_table(P, y, bins: int = 10) -> list[dict]:
    """One-vs-rest reliability table: every (match, outcome) probability pooled into ``bins``
    equal-width bins -> ``n``, mean predicted probability, observed frequency."""
    P = np.asarray(P, dtype=float)
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    p, o = P.ravel(), onehot.ravel()
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append({
            "lo": b / bins, "hi": (b + 1) / bins, "n": n,
            "mean_pred": float(p[m].mean()) if n else None,
            "observed": float(o[m].mean()) if n else None,
        })
    return out


def confusion_matrix(pred_idx, y) -> list[list[int]]:
    """Rows = actual (H, D, A), columns = predicted (H, D, A)."""
    cm = np.zeros((3, 3), dtype=int)
    for a, p in zip(np.asarray(y, dtype=int), np.asarray(pred_idx, dtype=int), strict=True):
        cm[a, p] += 1
    return cm.tolist()


def draw_predicted_rate(pred_idx) -> float:
    return float(np.mean(np.asarray(pred_idx) == 1))


# ----------------------------------------------------------------------------- scoreline
def _cell(grids, hg, ag):
    """Grid probability of the actual score (``EPS`` when beyond the grid)."""
    g = np.asarray(grids, dtype=float)
    n, size = g.shape[0], g.shape[1]
    hg, ag = np.asarray(hg, dtype=int), np.asarray(ag, dtype=int)
    inside = (hg < size) & (ag < size)
    p = np.full(n, EPS)
    k = np.nonzero(inside)[0]
    p[k] = g[k, hg[k], ag[k]]
    return p, inside


def exact_hit_rate(pred_h, pred_a, hg, ag) -> float:
    return float(np.mean((np.asarray(pred_h) == np.asarray(hg))
                         & (np.asarray(pred_a) == np.asarray(ag))))


def topk_hit_rate(grids, hg, ag, k: int) -> float:
    """Share of matches whose actual score is among the ``k`` most probable grid cells.

    Ties are broken towards the lower flat index (stable); a score outside the 0-7 grid is a miss.
    """
    g = np.asarray(grids, dtype=float)
    n, size = g.shape[0], g.shape[1]
    flat = g.reshape(n, -1)
    order = np.argsort(-flat, axis=1, kind="stable")[:, :k]
    hg, ag = np.asarray(hg, dtype=int), np.asarray(ag, dtype=int)
    inside = (hg < size) & (ag < size)
    target = np.where(inside, hg * size + ag, -1)
    return float(np.mean([t in row for t, row in zip(target, order, strict=True)]))


def goal_diff_hit_rate(pred_h, pred_a, hg, ag) -> float:
    return float(np.mean((np.asarray(pred_h) - np.asarray(pred_a))
                         == (np.asarray(hg) - np.asarray(ag))))


def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float))))


def mean_log_score(grids, hg, ag) -> float:
    """Mean ``log p(actual scoreline)`` (higher is better; <= 0)."""
    p, _ = _cell(grids, hg, ag)
    return float(np.mean(np.log(np.clip(p, EPS, 1.0))))


def mean_actual_score_prob(grids, hg, ag) -> float:
    p, _ = _cell(grids, hg, ag)
    return float(np.mean(p))


def binary_accuracy_brier(p_event, event) -> tuple[float, float]:
    """``(accuracy at the 0.5 threshold, Brier)`` for a binary event (e.g. over 2.5, BTTS)."""
    p, e = np.asarray(p_event, dtype=float), np.asarray(event, dtype=float)
    return float(np.mean((p >= 0.5) == (e >= 0.5))), float(np.mean((p - e) ** 2))


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_ci(
    values, clusters=None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI of a mean with match-clustered resampling.

    ``values`` are per-row (0/1 hits, say); ``clusters`` labels the rows (default: each row its
    own cluster). Whole clusters are drawn with replacement ``n_boot`` times (RNG seeded with
    ``seed``) and the mean of all rows in the drawn clusters is recorded.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    labels = np.arange(len(v)) if clusters is None else np.asarray(clusters)
    _, inv = np.unique(labels, return_inverse=True)
    k = int(inv.max()) + 1
    sums = np.bincount(inv, weights=v, minlength=k)
    cnts = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    means = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
