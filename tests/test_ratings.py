"""Ratings: as-of equivalence, causality/no-lookahead, ordered logit and walk-forward."""

import numpy as np
import pandas as pd
import pytest

from fedge.features import ratings as R
from fedge.features.asof import asof


def _league(n_matches=900, n_teams=12, seed=0, strength: bool = True) -> pd.DataFrame:
    """Synthetic division (three seasons). With ``strength`` the i-th team is stronger."""
    rng = np.random.default_rng(seed)
    seasons = ["2020/21", "2021/22", "2022/23"]
    rows = []
    start = pd.Timestamp("2020-08-01 15:00", tz="UTC")
    for i in range(n_matches):
        h, a = rng.choice(n_teams, size=2, replace=False)
        lam_h, lam_a = 1.4, 1.1
        if strength:
            lam_h += 0.35 * (h - a) / n_teams
            lam_a -= 0.35 * (h - a) / n_teams
        gh, ga = rng.poisson(max(lam_h, 0.05)), rng.poisson(max(lam_a, 0.05))
        rows.append(
            {
                "match_id": f"m{i:04d}",
                "home": f"T{h:02d}",
                "away": f"T{a:02d}",
                "FTHG": gh,
                "FTAG": ga,
                "FTR": "H" if gh > ga else ("D" if gh == ga else "A"),
                "kickoff_utc": start + pd.Timedelta(hours=19 * i),
                "season": seasons[min(int(3 * i / n_matches), 2)],
            }
        )
    m = pd.DataFrame(rows)
    # available_at for a match result: kickoff + 3h (the harness embargo)
    m["available_at"] = m["kickoff_utc"] + pd.Timedelta(hours=3)
    return m


def test_elo_sweep_equals_asof_restricted_recomputation():
    """The causal sweep must equal rebuilding the ratings from only the visible history."""
    m = _league(seed=1)
    sweep = R.elo_sweep(m, k=20.0, hfa=60.0)
    for i in (1, 5, 40, 123, 400, len(m) - 1):
        bet_time = m["kickoff_utc"].iloc[i]
        visible = asof(m, bet_time)  # strict available_at < bet_time
        final = R.elo_from_results(visible, k=20.0, hfa=60.0)
        dr = final.get(m["home"].iloc[i], R.ELO_START) - final.get(m["away"].iloc[i], R.ELO_START)
        assert np.isclose(dr, sweep[i], atol=1e-9)


def test_elo_and_pi_sweeps_are_causal_under_future_poisoning():
    m = _league(seed=2)
    k, hfa = 20.0, 60.0
    elo = R.elo_sweep(m, k=k, hfa=hfa)
    pi = R.pi_sweep(m, alpha=0.15, beta=0.10)
    poisoned = m.copy()
    cut = m["kickoff_utc"].iloc[300]
    later = poisoned["kickoff_utc"] >= cut
    poisoned.loc[later, "FTHG"] = 12
    poisoned.loc[later, "FTAG"] = 0
    elo2 = R.elo_sweep(poisoned, k=k, hfa=hfa)
    pi2 = R.pi_sweep(poisoned, alpha=0.15, beta=0.10)
    early = m["kickoff_utc"] < cut
    assert np.allclose(elo[early], elo2[early])
    assert np.allclose(pi[early], pi2[early])
    assert not np.allclose(elo[later], elo2[later])  # the poison really did move the future


def test_same_day_matches_do_not_share_results_within_the_embargo():
    """A match whose result is not yet in (kickoff + 3h) must not inform a later same-day one."""
    m = _league(n_matches=4, seed=3)
    ko = pd.Timestamp("2022-01-01 15:00", tz="UTC")
    m.loc[0, ["home", "away", "FTHG", "FTAG", "FTR"]] = ["A", "B", 7, 0, "H"]
    m.loc[1, ["home", "away", "FTHG", "FTAG", "FTR"]] = ["A", "C", 1, 1, "D"]
    m.loc[0, "kickoff_utc"] = ko
    m.loc[1, "kickoff_utc"] = ko + pd.Timedelta(hours=1)
    m.loc[0, "available_at"] = ko + pd.Timedelta(hours=3)
    m.loc[1, "available_at"] = ko + pd.Timedelta(hours=4)
    sweep = R.elo_sweep(m.iloc[:2].reset_index(drop=True), k=20.0, hfa=60.0)
    assert sweep[1] == 0.0  # A's 7-0 has not landed yet: no rating information


def test_ordered_logit_probs_are_valid_and_monotone():
    m = _league(seed=4)
    x = R.elo_sweep(m, k=20.0, hfa=60.0)
    y = R.outcome_codes(m)
    theta = R.fit_ordered_logit(x, y)
    p = R.ordered_logit_probs(theta, x)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()
    grid = np.linspace(-400, 400, 41)
    pg = R.ordered_logit_probs(theta, grid)
    assert theta.b < 0  # larger x (=home edge) -> smaller latent -> lower code (H = 0)
    assert pg[0, 0] < pg[-1, 0]  # home-win probability rises with the home rating edge
    assert pg[0, 2] > pg[-1, 2]  # away-win probability falls
    assert pg[:, 1].max() > pg[0, 1]  # a draw band exists in the middle


def test_ordered_logit_recovers_a_planted_direction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    z = 0.0 - 1.5 * x + rng.logistic(size=4000)
    y = np.digitize(z, [-0.6, 0.6])
    theta = R.fit_ordered_logit(x, y)
    assert theta.b < -1.0
    assert theta.c2 > theta.c1


def test_run_league_no_lookahead_and_deterministic():
    m = _league(seed=5)
    preds, info = R.run_league(m, val_matches=120, min_train_matches=60)
    assert len(preds) > 0
    assert preds.index.is_unique
    again, _ = R.run_league(m, val_matches=120, min_train_matches=60)
    pd.testing.assert_frame_equal(preds, again)
    seasons = sorted(preds["season"].unique())
    assert len(seasons) >= 2
    poisoned = m.copy()
    later = poisoned["season"] > seasons[0]
    poisoned.loc[later, "FTHG"] = 9
    poisoned.loc[later, "FTAG"] = 0
    poisoned.loc[later, "FTR"] = "H"
    preds2, _ = R.run_league(poisoned, val_matches=120, min_train_matches=60)
    keep = preds.index[(preds["season"] == seasons[0]) & preds.index.isin(preds2.index)]
    assert len(keep) > 0
    pd.testing.assert_frame_equal(preds.loc[keep], preds2.loc[keep])


def test_in_fold_selection_uses_only_past_data():
    """The chosen hyper-parameters for a season must not react to future results."""
    m = _league(seed=6)
    _, info = R.run_league(m, val_matches=120, min_train_matches=60)
    poisoned = m.copy()
    later = poisoned["season"] == "2022/23"
    poisoned.loc[later, "FTHG"] = 11
    poisoned.loc[later, "FTAG"] = 0
    poisoned.loc[later, "FTR"] = "H"
    _, info2 = R.run_league(poisoned, val_matches=120, min_train_matches=60)
    assert info["selected"]["2021/22"]["elo"] == info2["selected"]["2021/22"]["elo"]
    assert info["selected"]["2021/22"]["pi"] == info2["selected"]["2021/22"]["pi"]


def test_probabilities_valid():
    m = _league(seed=7)
    preds, _ = R.run_league(m, val_matches=120, min_train_matches=60)
    assert len(preds) > 0
    groups = (["p_elo_home", "p_elo_draw", "p_elo_away"],
              ["p_pi_home", "p_pi_draw", "p_pi_away"])
    for cols in groups:
        p = preds[cols].to_numpy()
        assert np.allclose(p.sum(axis=1), 1.0)
        assert (p > 0).all()


def test_empty_input_is_handled():
    m = _league(seed=8).iloc[0:0]
    preds, info = R.run_league(m)
    assert len(preds) == 0
    with pytest.raises(ValueError):
        R.fit_ordered_logit(np.array([]), np.array([]))
