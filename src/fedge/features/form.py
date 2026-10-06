"""Time-weighted form, schedule and venue-split features (Phase 4, card 4.1).

One causal sweep per division. A finished match enters the state only once
``kickoff_j + embargo < kickoff_i`` (the same rule as ``fedge.features.asof``: a result's
``available_at`` is ``kickoff + 3h`` and must be strictly before the bet time, which here is the
kickoff of the match being predicted). ``tests/test_form.py`` proves the sweep equals a brute
force recomputation through ``asof`` and that mutating any result at/after the cut changes
nothing.

Feature families (``FAMILIES``); every stat is an exponentially time-decayed mean with
half-life ``half_life_days``, shrunk with ``prior_k`` pseudo-matches towards the division's
decayed mean (venue splits are shrunk towards the team's own all-venue mean):

* ``goals``  - goals for/against, points per game, plus home-only / away-only goals
* ``shots``  - shots and shots on target for/against (NaN where football-data lacks them)
* ``xg``     - xG for/against (NaN outside Understat's top-5 and football-data 2026/27)
* ``sched``  - rest days, games played this season, promoted (new to the division) flag. Fixture
  dates are known in advance, so these use kickoff times only, never results.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fedge.backtest.walkforward import EMBARGO

STATS = ("gf", "ga", "pts", "sf", "sa", "tf", "ta", "xf", "xa")
FAMILIES: dict[str, tuple[str, ...]] = {
    "goals": ("gf", "ga", "pts"),
    "shots": ("sf", "sa", "tf", "ta"),
    "xg": ("xf", "xa"),
}
SPLIT_STATS = ("gf", "ga")  # venue splits are computed for goals only
HALF_LIFE_DAYS = 120.0
PRIOR_K = 3.0
REST_CAP = 21.0
_S = len(STATS)
_NS = 1_000_000_000 * 86400.0


def _row(gh, ga, hs, as_, hst, ast, xh, xa, home: bool) -> np.ndarray:
    """Per-team stat vector (order ``STATS``) for the home or away side of one match."""
    if home:
        gf, gA, sf, sa, tf, ta, xf, xA = gh, ga, hs, as_, hst, ast, xh, xa
    else:
        gf, gA, sf, sa, tf, ta, xf, xA = ga, gh, as_, hs, ast, hst, xa, xh
    pts = 3.0 if gf > gA else (1.0 if gf == gA else 0.0)
    return np.array([gf, gA, pts, sf, sa, tf, ta, xf, xA], dtype=float)


class _State:
    """Decayed numerator / denominator per stat (NaN observations are skipped)."""

    __slots__ = ("num", "den", "t")

    def __init__(self) -> None:
        self.num = np.zeros(_S)
        self.den = np.zeros(_S)
        self.t = 0.0

    def decayed(self, t: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        f = math.exp(-lam * (t - self.t)) if self.den.any() else 1.0
        return self.num * f, self.den * f

    def add(self, x: np.ndarray, t: float, lam: float) -> None:
        self.num, self.den = self.decayed(t, lam)
        ok = ~np.isnan(x)
        self.num[ok] += x[ok]
        self.den[ok] += 1.0
        self.t = t


def _shrunk(state: _State | None, t: float, lam: float, prior: np.ndarray, k: float):
    """(mean, effective matches) with ``k`` pseudo-matches at ``prior``."""
    if state is None:
        return prior.copy(), np.zeros(_S)
    num, den = state.decayed(t, lam)
    return (num + k * prior) / (den + k), den


def _league_mean(state: _State, t: float, lam: float) -> np.ndarray:
    num, den = state.decayed(t, lam)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def form_features(
    matches: pd.DataFrame,
    families: tuple[str, ...] = ("goals", "shots", "xg", "sched"),
    half_life_days: float = HALF_LIFE_DAYS,
    prior_k: float = PRIOR_K,
    embargo: pd.Timedelta = EMBARGO,
) -> pd.DataFrame:
    """As-of form features, one row per match, indexed by ``match_id``.

    ``matches`` needs ``match_id, div, season, kickoff_utc, home, away, FTHG, FTAG`` and
    optionally ``HS, AS, HST, AST, xg_home, xg_away`` (see :func:`fedge.features.xg.attach_xg`).
    Matches without a result (``FTHG`` NaN) get features but never enter the state.
    """
    lam = math.log(2.0) / half_life_days
    cols_by_stat = [s for fam in ("goals", "shots", "xg") if fam in families for s in FAMILIES[fam]]
    stat_idx = [STATS.index(s) for s in cols_by_stat]
    split_on = "goals" in families
    out_parts: list[pd.DataFrame] = []
    for _, g in matches.groupby("div", sort=True, observed=True):
        g = g.sort_values("kickoff_utc", kind="mergesort")
        out_parts.append(
            _sweep_division(g, families, lam, prior_k, embargo, stat_idx, cols_by_stat, split_on)
        )
    if not out_parts:
        return pd.DataFrame()
    out = pd.concat(out_parts)
    return out.reindex(matches["match_id"].to_numpy())


def _sweep_division(g, families, lam, k, embargo, stat_idx, stat_names, split_on) -> pd.DataFrame:
    ko = g["kickoff_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    t_days = (ko.astype("datetime64[ns]").astype("int64") / _NS).astype(float)
    emb_days = embargo / pd.Timedelta(days=1)
    home, away = g["home"].to_numpy(), g["away"].to_numpy()
    season = g["season"].astype(str).to_numpy()

    def col(name):
        return (
            g[name].to_numpy(dtype=float, na_value=np.nan)
            if name in g.columns
            else np.full(len(g), np.nan)
        )

    gh, ga = col("FTHG"), col("FTAG")
    hs, as_, hst, ast = col("HS"), col("AS"), col("HST"), col("AST")
    xh, xa = col("xg_home"), col("xg_away")
    played = ~(np.isnan(gh) | np.isnan(ga))

    # promoted: team has no match in this division in the previous season (and a previous
    # season exists in the data). Fixture lists are public at season start.
    seasons_sorted = sorted(set(season), key=lambda s: int(s[:4]))
    members = {s: set() for s in seasons_sorted}
    for s, h, a in zip(season, home, away, strict=True):
        members[s].add(h)
        members[s].add(a)
    prev = {s: (seasons_sorted[i - 1] if i else None) for i, s in enumerate(seasons_sorted)}

    overall: dict[str, _State] = {}
    venue: dict[tuple[str, str], _State] = {}
    league = _State()
    last_ko: dict[str, float] = {}
    n_season: dict[tuple[str, str], int] = {}
    nan_vec = np.full(_S, np.nan)
    rows = []
    ptr = 0
    n = len(g)
    for i in range(n):
        t = t_days[i]
        while ptr < i and t_days[ptr] + emb_days < t:
            if played[ptr]:
                xh_row = _row(
                    gh[ptr], ga[ptr], hs[ptr], as_[ptr], hst[ptr], ast[ptr], xh[ptr], xa[ptr], True
                )
                xa_row = _row(
                    gh[ptr], ga[ptr], hs[ptr], as_[ptr], hst[ptr], ast[ptr], xh[ptr], xa[ptr], False
                )
                tj = t_days[ptr]
                for team, v, key in ((home[ptr], xh_row, "H"), (away[ptr], xa_row, "A")):
                    overall.setdefault(team, _State()).add(v, tj, lam)
                    venue.setdefault((team, key), _State()).add(v, tj, lam)
                    league.add(v, tj, lam)
            ptr += 1
        prior = _league_mean(league, t, lam) if league.den.any() else nan_vec
        feat: dict[str, float] = {}
        h, a = home[i], away[i]
        mh, nh = _shrunk(overall.get(h), t, lam, prior, k)
        ma, na = _shrunk(overall.get(a), t, lam, prior, k)
        for si, name in zip(stat_idx, stat_names, strict=True):
            feat[f"h_{name}"] = mh[si]
            feat[f"a_{name}"] = ma[si]
        if "goals" in families:
            feat["h_n"] = nh[0]
            feat["a_n"] = na[0]
        if split_on:
            vh, _ = _shrunk(venue.get((h, "H")), t, lam, mh, k)
            va, _ = _shrunk(venue.get((a, "A")), t, lam, ma, k)
            for name in SPLIT_STATS:
                si = STATS.index(name)
                feat[f"h_home_{name}"] = vh[si]
                feat[f"a_away_{name}"] = va[si]
        if "sched" in families:
            feat["h_rest"] = min((t - last_ko[h]), REST_CAP) if h in last_ko else np.nan
            feat["a_rest"] = min((t - last_ko[a]), REST_CAP) if a in last_ko else np.nan
            feat["h_games"] = n_season.get((h, season[i]), 0)
            feat["a_games"] = n_season.get((a, season[i]), 0)
            p = prev[season[i]]
            feat["h_promoted"] = float(p is not None and h not in members[p])
            feat["a_promoted"] = float(p is not None and a not in members[p])
        rows.append(feat)
        for team in (h, a):  # kickoff times are known in advance: schedule state, not results
            last_ko[team] = t
            n_season[(team, season[i])] = n_season.get((team, season[i]), 0) + 1
    return pd.DataFrame(rows, index=g["match_id"].to_numpy())
