"""Baselines scored on the same matches as the model (price-free, history-only).

* ``always_home`` - predict a home win every time (winner hit rate and Brier only).
* ``league_freq`` - league-average W/D/L frequencies and league-average scoreline distribution
  (its predicted score is the league's most common scoreline).
* ``prev_season_poisson`` - a naive independent Poisson on each team's goals for / against per
  game in the previous completed season (promoted teams, with no previous-season data, get the
  league average).

``history`` is a frame of finished matches with ``div, home, away, kickoff_utc, FTHG, FTAG``;
every baseline for a match only uses rows kicked off strictly before the earliest scored match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

GRID = 8
BIG = 14  # internal Poisson grid before cropping to GRID x GRID
FREQ_WINDOW = 1900  # most recent matches per division behind the league-average baselines


def season_year(ts) -> int:
    t = pd.Timestamp(ts)
    return t.year if t.month >= 7 else t.year - 1


def _class_mask(size: int) -> np.ndarray:
    i, j = np.indices((size, size))
    return np.where(i > j, 0, np.where(i == j, 1, 2))


def constrained_score(grid: np.ndarray, outcome_idx: int) -> tuple[int, int]:
    """Most likely cell of ``grid`` within outcome class ``outcome_idx`` (0 H, 1 D, 2 A)."""
    masked = np.where(_class_mask(grid.shape[0]) == outcome_idx, grid, -1.0)
    i, j = np.unravel_index(int(np.argmax(masked)), grid.shape)
    return int(i), int(j)


def poisson_grid(lh: float, la: float, size: int = BIG) -> np.ndarray:
    k = np.arange(size)
    g = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    return g / g.sum()


def outcome_probs(grid: np.ndarray) -> np.ndarray:
    m = _class_mask(grid.shape[0])
    return np.array([grid[m == c].sum() for c in range(3)])


def _history_before(history: pd.DataFrame, cutoff) -> pd.DataFrame:
    h = history.copy()
    h["kickoff_utc"] = pd.to_datetime(h["kickoff_utc"], utc=True)
    return h.loc[h["kickoff_utc"] < pd.Timestamp(cutoff)]


def league_freq_tables(history: pd.DataFrame, cutoff) -> dict[str, dict]:
    """Per division: W/D/L frequencies, empirical scoreline grid and its mode."""
    h = _history_before(history, cutoff)
    out: dict[str, dict] = {}
    for div, g in h.groupby("div"):
        g = g.sort_values("kickoff_utc").tail(FREQ_WINDOW)
        hg, ag = g["FTHG"].astype(int).to_numpy(), g["FTAG"].astype(int).to_numpy()
        y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        freq = np.bincount(y, minlength=3) / len(y)
        grid = np.zeros((GRID, GRID))
        for a, b in zip(hg, ag, strict=True):
            if a < GRID and b < GRID:
                grid[a, b] += 1
        grid /= grid.sum()
        mi, mj = np.unravel_index(int(np.argmax(grid)), grid.shape)
        out[str(div)] = {"freq": freq, "grid": grid, "mode": (int(mi), int(mj))}
    return out


def _team_rates(history: pd.DataFrame, cutoff) -> dict:
    """``(div, season_year) -> (mu_home, mu_away, {team: (gf_rate, ga_rate)})`` from finished
    seasons (rates are goals per game, home and away pooled)."""
    h = _history_before(history, cutoff)
    h = h.assign(sy=[season_year(t) for t in h["kickoff_utc"]])
    out = {}
    for (div, sy), g in h.groupby(["div", "sy"]):
        hg, ag = g["FTHG"].astype(float), g["FTAG"].astype(float)
        rows = pd.concat([
            pd.DataFrame({"team": g["home"], "gf": hg, "ga": ag}),
            pd.DataFrame({"team": g["away"], "gf": ag, "ga": hg}),
        ])
        per = rows.groupby("team").agg(gf=("gf", "mean"), ga=("ga", "mean"), n=("gf", "size"))
        out[(str(div), int(sy))] = (
            float(hg.mean()), float(ag.mean()),
            {t: (float(r.gf), float(r.ga)) for t, r in per.iterrows() if r.n >= 10},
        )
    return out


def baseline_frames(df: pd.DataFrame, history: pd.DataFrame) -> dict[str, dict]:
    """For the scored frame ``df`` (``div, home, away, kickoff_utc``): per baseline
    ``{"P": (n,3), "grid": (n,8,8) or None, "pred_h", "pred_a"}``."""
    n = len(df)
    cutoff = pd.to_datetime(df["kickoff_utc"], utc=True).min()
    tabs = league_freq_tables(history, cutoff)
    rates = _team_rates(history, cutoff)
    # always home
    always = {
        "P": np.tile([1.0, 0.0, 0.0], (n, 1)), "grid": None,
        "pred_h": np.full(n, 1), "pred_a": np.full(n, 0),
    }
    fP, fG, fh, fa = [], [], [], []
    pP, pG, ph, pa = [], [], [], []
    for r in df.itertuples(index=False):
        t = tabs.get(r.div)
        if t is None:  # no history for the division: uniform fallbacks
            fP.append([1 / 3] * 3)
            fG.append(np.full((GRID, GRID), 1.0 / GRID**2))
            fh.append(1)
            fa.append(1)
        else:
            fP.append(t["freq"])
            fG.append(t["grid"])
            fh.append(t["mode"][0])
            fa.append(t["mode"][1])
        sy = season_year(r.kickoff_utc) - 1
        prev = rates.get((r.div, sy))
        if prev is None:
            lh, la = 1.5, 1.2
        else:
            mu_h, mu_a, team = prev
            mu = (mu_h + mu_a) / 2  # goals per team per game
            gh, gaa = team.get(r.home, (mu, mu)), team.get(r.away, (mu, mu))
            lh = mu_h * (gh[0] / mu) * (gaa[1] / mu)
            la = mu_a * (gaa[0] / mu) * (gh[1] / mu)
        big = poisson_grid(max(lh, 0.05), max(la, 0.05))
        probs = outcome_probs(big)
        pP.append(probs)
        pG.append(big[:GRID, :GRID])
        i, j = constrained_score(big, int(np.argmax(probs)))
        ph.append(i)
        pa.append(j)
    return {
        "always_home": always,
        "league_freq": {"P": np.array(fP), "grid": np.array(fG),
                        "pred_h": np.array(fh), "pred_a": np.array(fa)},
        "prev_season_poisson": {"P": np.array(pP), "grid": np.array(pG),
                                "pred_h": np.array(ph), "pred_a": np.array(pa)},
    }
