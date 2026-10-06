"""Forecast metrics: RPS, log-loss, Brier, calibration slope, ECE, reliability helper.

Convention: ``probs`` has shape (n, k) with outcome classes in a fixed *ordinal* order
(for 1X2: H, D, A). ``y`` holds integer class indices (n,). Binary helpers take p (n,) and
y in {0, 1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-15

# A logistic recalibration whose |slope| exceeds this is "separated": a handful of near-0/1
# forecasts fit by an unbounded Newton step, so the value is an artefact of the sample size, not
# a calibration measurement. Reports render such values through ``fmt_slope``.
SLOPE_NOT_ESTIMABLE = 10.0


def slope_not_estimable(b) -> bool:
    """True when a calibration slope is a separated fit rather than a measurable slope."""
    if b is None:
        return True
    b = float(b)
    return not np.isfinite(b) or abs(b) > SLOPE_NOT_ESTIMABLE


def fmt_slope(b, nd: int = 4) -> str:
    """Format a calibration slope, flagging separated (non-estimable) fits as ``n.e.``."""
    if b is None:
        return "-"
    b = float(b)
    if not np.isfinite(b):
        return "-"
    if slope_not_estimable(b):
        return "n.e."
    return f"{b:.{nd}f}"


def _prep(probs, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probs must be (n, k) and match len(y)")
    return p, y


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k)[y]


def rps(probs, y) -> float:
    """Mean Ranked Probability Score: 1/(k-1) * sum_{i<k} (cumP_i - cumO_i)^2."""
    p, y = _prep(probs, y)
    k = p.shape[1]
    diff = np.cumsum(p, axis=1)[:, :-1] - np.cumsum(_onehot(y, k), axis=1)[:, :-1]
    return float(np.mean(np.sum(diff**2, axis=1) / (k - 1)))


def log_loss(probs, y) -> float:
    """Mean negative log-probability assigned to the realised class."""
    p, y = _prep(probs, y)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0))))


def brier(probs, y) -> float:
    """Multiclass Brier score: mean over matches of sum_k (p_k - o_k)^2."""
    p, y = _prep(probs, y)
    return float(np.mean(np.sum((p - _onehot(y, p.shape[1])) ** 2, axis=1)))


def calibration_slope(p, y, iters: int = 50) -> tuple[float, float]:
    """Logistic recalibration of binary forecasts: y ~ sigmoid(a + b * logit(p)).

    Returns (intercept a, slope b). Perfect calibration: (0, 1); b < 1 means overconfident.
    Newton-Raphson; no external dependency.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-X @ beta))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        grad = X.T @ (y - mu)
        hess = (X * w[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def calibration_slope_multi(probs, y, iters: int = 50) -> tuple[float, float]:
    """One-vs-rest pooled calibration slope for a multiclass forecast.

    Every (class, match) pair contributes ``(p_k, 1[y = k])`` to a single logistic
    recalibration ``y ~ sigmoid(a + b * logit(p))``; returns ``(a, b)``. Perfectly calibrated:
    ``(0, 1)``; ``b < 1`` means overconfident.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probs must be (n, k) and match len(y)")
    flat_p = p.ravel()
    flat_y = np.eye(p.shape[1], dtype=float)[y].ravel()
    return calibration_slope(flat_p, flat_y, iters)


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability bins: mean forecast, observed frequency and count per bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((b, float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return pd.DataFrame(rows, columns=["bin", "mean_pred", "obs_freq", "n"])


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted mean |obs_freq - mean_pred| over bins."""
    t = reliability_table(p, y, n_bins)
    return float(np.sum(t["n"] * (t["obs_freq"] - t["mean_pred"]).abs()) / t["n"].sum())


def plot_reliability(p, y, path=None, n_bins: int = 10, title: str = "Reliability"):
    """Reliability diagram; saves to ``path`` if given. Returns the matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = reliability_table(p, y, n_bins)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(t["mean_pred"], t["obs_freq"], "o-")
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig


def score_all(probs, y) -> dict[str, float]:
    """Convenience bundle of the multiclass metrics."""
    return {"rps": rps(probs, y), "log_loss": log_loss(probs, y), "brier": brier(probs, y)}
