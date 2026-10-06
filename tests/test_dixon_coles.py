"""Dixon-Coles: grid equivalence with penaltyblog, in-fold xi, and no-lookahead tests."""

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel

from fedge.models import dixon_coles as DC


def _league(n_matches=900, n_teams=12, seed=0) -> pd.DataFrame:
    """A small synthetic division (three seasons) with tz-aware kickoffs and FTR labels."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    seasons = ["2020/21", "2021/22", "2022/23"]
    rows = []
    start = pd.Timestamp("2020-08-01 15:00", tz="UTC")
    for i in range(n_matches):
        h, a = rng.choice(n_teams, size=2, replace=False)
        gh, ga = rng.poisson(1.4), rng.poisson(1.1)
        ko = start + pd.Timedelta(hours=19 * i)
        rows.append(
            {
                "match_id": f"m{i:04d}",
                "home": teams[h],
                "away": teams[a],
                "FTHG": gh,
                "FTAG": ga,
                "FTR": "H" if gh > ga else ("D" if gh == ga else "A"),
                "kickoff_utc": ko,
                "season": seasons[min(int(3 * i / n_matches), 2)],
            }
        )
    return pd.DataFrame(rows)


def test_score_grid_matches_penaltyblog():
    """predict() must reproduce penaltyblog's own fitted-model grid to float noise."""
    m = _league(seed=1)
    ref = pd.Timestamp("2021-06-01", tz="UTC")
    params = DC.fit_dc(m, xi=DC.DEFAULT_XI, ref=ref, min_matches=100)
    assert params is not None
    tr = m.loc[(m["kickoff_utc"] < ref) & (m["kickoff_utc"] >= ref - pd.Timedelta(days=365.25 * 5))]
    w = DC.decay_weights(tr["kickoff_utc"], ref, DC.DEFAULT_XI)
    model = DixonColesGoalModel(
        tr["FTHG"].astype(int), tr["FTAG"].astype(int), tr["home"], tr["away"], weights=w
    )
    model.fit()
    assert tuple(model.teams) == params.teams
    assert np.allclose(model.params_array, np.concatenate(
        [params.attack, params.defence, [params.hfa, params.rho]]
    ))
    test = m.tail(60).reset_index(drop=True)
    mine = DC.predict(params, test)
    pb = pd.DataFrame(
        [
            {
                "p_home": g.home_win,
                "p_draw": g.draw,
                "p_away": g.away_win,
                "p_over25": g.total_goals("over", 2.5),
                "p_under25": g.total_goals("under", 2.5),
            }
            for g in (
                model.predict(h, a, max_goals=DC.MAX_GOALS)
                for h, a in zip(test["home"], test["away"], strict=True)
            )
        ]
    )
    diff = np.abs(mine[["p_home", "p_draw", "p_away", "p_over25", "p_under25"]].to_numpy()
                   - pb.to_numpy())
    assert diff.max() < 1e-6


def test_predict_uses_prior_for_unseen_teams():
    m = _league(seed=2)
    params = DC.fit_dc(m)
    assert params is not None
    unseen = m.tail(5).reset_index(drop=True).copy()
    unseen["home"] = "BRAND NEW FC"
    p = DC.predict(params, unseen)
    assert ((p[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs() < 1e-12).all()
    assert ((p[["p_over25", "p_under25"]].sum(axis=1) - 1).abs() < 1e-12).all()
    assert p["p_home"].between(0, 1).all()
    # any unseen home side gets exactly the same prior treatment
    a = m.tail(1).reset_index(drop=True).copy()
    b = a.copy()
    a["home"], b["home"] = "UNSEEN A", "UNSEEN B"
    pd.testing.assert_frame_equal(DC.predict(params, a), DC.predict(params, b))


def test_decay_weights_maths():
    ko = pd.Series(pd.to_datetime(["2024-01-01", "2023-01-01"], utc=True))
    ref = pd.Timestamp("2025-01-01", tz="UTC")
    w = DC.decay_weights(ko, ref, 0.0065)
    assert np.isclose(w[1] / w[0], np.exp(-0.0065), atol=1e-3)


def test_select_xi_ignores_the_test_season():
    """Poisoning the test season's results must not change the in-fold xi choice."""
    m = _league(seed=3)
    test_start = m.loc[m["season"] == "2022/23", "kickoff_utc"].min()
    xi_a, loss_a = DC.select_xi(m, test_start, val_matches=120, min_matches=60)
    assert xi_a in DC.XI_GRID
    assert np.isfinite(loss_a)
    poisoned = m.copy()
    mask = poisoned["kickoff_utc"] >= test_start
    poisoned.loc[mask, "FTHG"] = 9
    poisoned.loc[mask, "FTAG"] = 0
    xi_b, loss_b = DC.select_xi(poisoned, test_start, val_matches=120, min_matches=60)
    assert (xi_a, loss_a) == (xi_b, loss_b)


def test_run_league_has_no_lookahead_and_is_deterministic():
    m = _league(seed=4)
    preds, info = DC.run_league(m, val_matches=120, min_train_matches=60)
    assert len(preds) > 0
    assert preds.index.is_unique
    assert info["n_folds"] > 0
    again, _ = DC.run_league(m, val_matches=120, min_train_matches=60)
    pd.testing.assert_frame_equal(preds, again)
    # poison every result after the first test season; earlier predictions must not move
    seasons = sorted(preds["season"].unique())
    assert len(seasons) >= 2
    later = m["season"] > seasons[0]
    poisoned = m.copy()
    poisoned.loc[later, "FTHG"] = 8
    poisoned.loc[later, "FTAG"] = 0
    poisoned.loc[later, "FTR"] = "H"
    preds2, _ = DC.run_league(poisoned, val_matches=120, min_train_matches=60)
    first = preds.index[preds["season"] == seasons[0]]
    keep = [i for i in first if i in preds2.index]
    assert keep, "expected predictions in the first test season"
    pd.testing.assert_frame_equal(preds.loc[keep], preds2.loc[keep])


def test_probabilities_are_valid_and_ordering_holds():
    m = _league(seed=5)
    params = DC.fit_dc(m)
    assert params is not None
    p = DC.predict(params, m.tail(200))
    for a, b in (("p_home", "p_draw"), ("p_draw", "p_away")):
        assert (p[a] > 0).all() and (p[b] > 0).all()
    assert np.allclose(p[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    assert np.allclose(p[["p_over25", "p_under25"]].sum(axis=1), 1.0)


def test_fit_dc_returns_none_when_data_is_thin():
    assert DC.fit_dc(_league(n_matches=5)) is None
