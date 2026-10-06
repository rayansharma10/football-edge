"""De-margining (power, Shin, multiplicative) and implied probabilities.

All functions are vectorised over rows: ``odds`` is an array of shape (n, k) (k = 3 for 1X2,
k = 2 for over/under) or a single (k,) vector. Solvers use fixed-iteration bisection, so results
are deterministic. Returned parameters (power exponent ``c``, Shin ``z``) are diagnostics: an
implausible value signals stale or bad odds.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

METHODS = ("power", "shin", "multiplicative")
_ITERS = 100


class Devig(NamedTuple):
    probs: np.ndarray  # fair probabilities, rows sum to 1
    param: np.ndarray  # power: exponent c (p = r**c); shin: z; multiplicative: booksum
    overround: np.ndarray  # sum(1/odds) - 1


def _as2d(odds) -> tuple[np.ndarray, bool]:
    a = np.asarray(odds, dtype=float)
    single = a.ndim == 1
    a = np.atleast_2d(a)
    if a.shape[1] < 2:
        raise ValueError("need at least two outcomes")
    if not np.all(a > 1.0):
        raise ValueError("all decimal odds must be > 1.0")
    return a, single


def _out(p, param, over, single) -> Devig:
    if single:
        return Devig(p[0], param[0], over[0])
    return Devig(p, param, over)


def multiplicative(odds) -> Devig:
    """Proportional removal: p_i = r_i / sum(r)."""
    a, single = _as2d(odds)
    r = 1.0 / a
    s = r.sum(axis=1)
    return _out(r / s[:, None], s, s - 1.0, single)


def power(odds) -> Devig:
    """Power method: p_i = r_i ** c with c chosen so that sum(p) = 1 (c > 1 for an overround)."""
    a, single = _as2d(odds)
    r = 1.0 / a
    s = r.sum(axis=1)
    lo = np.zeros(len(r))  # sum(r**0) = k > 1
    hi = np.full(len(r), 2.0)
    while np.any(np.sum(r ** hi[:, None], axis=1) > 1.0):
        hi = np.where(np.sum(r ** hi[:, None], axis=1) > 1.0, hi * 2.0, hi)
    for _ in range(_ITERS):
        mid = (lo + hi) / 2.0
        too_big = np.sum(r ** mid[:, None], axis=1) > 1.0  # decreasing in c
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    c = (lo + hi) / 2.0
    p = r ** c[:, None]
    p = p / p.sum(axis=1, keepdims=True)  # remove residual ~1e-16 error
    return _out(p, c, s - 1.0, single)


def _shin_p(r: np.ndarray, s: np.ndarray, z: np.ndarray) -> np.ndarray:
    zz = z[:, None]
    return (np.sqrt(zz**2 + 4.0 * (1.0 - zz) * r**2 / s[:, None]) - zz) / (2.0 * (1.0 - zz))


def shin(odds) -> Devig:
    """Shin (1992/93): a fraction z of money is insider. Bisection on z in [0, 1).

    p_i = (sqrt(z^2 + 4(1-z) r_i^2 / S) - z) / (2(1-z)), S = sum(r).
    """
    a, single = _as2d(odds)
    r = 1.0 / a
    s = r.sum(axis=1)
    lo = np.zeros(len(r))
    hi = np.full(len(r), 0.999)
    for _ in range(_ITERS):
        mid = (lo + hi) / 2.0
        too_big = _shin_p(r, s, mid).sum(axis=1) > 1.0  # decreasing in z
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    z = (lo + hi) / 2.0
    p = _shin_p(r, s, z)
    p = p / p.sum(axis=1, keepdims=True)
    return _out(p, z, s - 1.0, single)


def devig(odds, method: str = "power") -> Devig:
    """De-margin decimal odds with ``method`` in METHODS (default: power)."""
    fns = {"power": power, "shin": shin, "multiplicative": multiplicative}
    if method not in fns:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return fns[method](odds)


def implied_1x2(h, d, a, method: str = "power") -> Devig:
    """3-way (home/draw/away) de-margining for arrays of odds."""
    return devig(np.column_stack([h, d, a]), method)


def implied_2way(over, under, method: str = "power") -> Devig:
    """2-way (e.g. over/under 2.5) de-margining for arrays of odds."""
    return devig(np.column_stack([over, under]), method)
