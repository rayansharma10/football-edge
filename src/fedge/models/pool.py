"""Logarithmic opinion pool of a structural model and a de-margined market.

The pooled per-row distribution is the normalised geometric mixture
``p ∝ p_model**w * p_market**(1-w)``, i.e. ``log p = w*log(p_model) + (1-w)*log(p_market)``
renormalised to sum to 1 per row. ``w`` is the weight on the MODEL: w=0 recovers the market,
w=1 the model. ``fit_weight`` picks the w minimising the mean multiclass log-loss against the
realised outcomes, so a weight of 0 means the model adds nothing to the market price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

EPS = 1e-12


def _as_rows(p, name: str) -> np.ndarray:
    """Coerce ``p`` to a float64 (n, k) array, raising ValueError if it is not 2-D."""
    a = np.asarray(p, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, k); got shape {a.shape}")
    return a


def _check_pair(p_model, p_market, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce ``p_model``/``p_market``/``y`` to matching-shape arrays."""
    pm = _as_rows(p_model, "p_model")
    pk = _as_rows(p_market, "p_market")
    if pm.shape != pk.shape:
        raise ValueError(f"p_model and p_market shape mismatch: {pm.shape} vs {pk.shape}")
    yy = np.asarray(y)
    if yy.ndim != 1 or len(yy) != pm.shape[0]:
        raise ValueError(f"y must be 1-D of length {pm.shape[0]}; got shape {yy.shape}")
    return pm, pk, yy


def log_pool(p_model, p_market, w: float) -> np.ndarray:
    """Pooled probabilities with weight ``w`` on the model and ``1-w`` on the market.

    Rows of the result sum to 1. Probabilities are clipped to ``EPS`` before the logs.
    """
    pm, pk, _ = _check_pair(p_model, p_market, np.zeros(len(np.asarray(p_model))))
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1]; got {w}")
    logp = w * np.log(np.clip(pm, EPS, None)) + (1.0 - w) * np.log(np.clip(pk, EPS, None))
    logp -= logp.max(axis=1, keepdims=True)  # shift before exp for numerical stability
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def pool_log_loss(p_model, p_market, w: float, y) -> float:
    """Mean multiclass negative log-likelihood of the pooled rows against classes ``y``."""
    pm, pk, yy = _check_pair(p_model, p_market, y)
    p = log_pool(pm, pk, w)
    prob = p[np.arange(len(yy)), yy]
    return float(-np.mean(np.log(np.clip(prob, EPS, None))))


def fit_weight(p_model, p_market, y) -> float:
    """Weight on the model minimising ``pool_log_loss`` over w in [0, 1]; deterministic."""
    pm, pk, yy = _check_pair(p_model, p_market, y)
    res = minimize_scalar(
        lambda w: pool_log_loss(pm, pk, w, yy),
        method="bounded",
        bounds=(0.0, 1.0),
        options={"xatol": 1e-7},
    )
    return float(res.x)


def fit_weight_bootstrap(
    p_model,
    p_market,
    y,
    clusters,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Match-clustered bootstrap CI for ``fit_weight``.

    Whole clusters (e.g. match ids) are resampled with replacement and w is refit on each
    replicate. Seeded: same inputs give identical output.
    """
    pm, pk, yy = _check_pair(p_model, p_market, y)
    codes, uniq = pd.factorize(pd.Series(clusters), sort=True)
    k = len(uniq)
    codes = np.asarray(codes)
    groups = [np.flatnonzero(codes == c) for c in range(k)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    ws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        rows = np.concatenate([groups[c] for c in idx[b]])
        ws[b] = fit_weight(pm[rows], pk[rows], yy[rows])
    w_hat = fit_weight(pm, pk, yy)
    lo, hi = np.quantile(ws, [alpha / 2, 1 - alpha / 2])
    return {
        "w": float(w_hat),
        "mean": float(ws.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(yy)),
        "n_clusters": int(k),
    }
