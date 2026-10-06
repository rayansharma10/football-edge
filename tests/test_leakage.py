"""Leakage tests (AGENTS.md rule 3, Gate 0 B0.3/B0.4) on a synthetic league with a planted signal.

World: 16 teams with latent strengths; goals ~ Poisson. Feature = shrunk as-of mean goal
difference of each team, built ONLY through fedge.features.asof. Model = logistic regression
fit walk-forward. "Edge" = baseline log-loss minus model log-loss (higher = better).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fedge.backtest.walkforward import run_walkforward, season_folds
from fedge.features.asof import LeakageError, asof, assert_no_leak, build_features

N_TEAMS = 16
SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24"]


def make_world(seed=0, signs=(1, 1, 1, 1), spread=0.6) -> pd.DataFrame:
    """Synthetic matches; ``signs`` flips the strength sign per season."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, spread, N_TEAMS)
    rows = []
    for si, season in enumerate(SEASONS):
        start = pd.Timestamp(f"{2020 + si}-08-08 15:00", tz="UTC")
        t = list(range(N_TEAMS))
        rnd = 0
        for leg in range(2):
            for _ in range(N_TEAMS - 1):
                for i in range(N_TEAMS // 2):
                    h, a = (t[i], t[-1 - i]) if leg == 0 else (t[-1 - i], t[i])
                    d = signs[si] * (strength[h] - strength[a])
                    hg = rng.poisson(np.exp(0.2 + d / 2))
                    ag = rng.poisson(np.exp(-d / 2))
                    rows.append(
                        (season, start + pd.Timedelta(days=7 * rnd), f"T{h}", f"T{a}", hg, ag)
                    )
                t = [t[0], t[-1], *t[1:-1]]
                rnd += 1
    m = pd.DataFrame(rows, columns=["season", "kickoff_utc", "home", "away", "FTHG", "FTAG"])
    m = m.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    m["match_id"] = [f"m{i}" for i in range(len(m))]
    return m


def results_table(m: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, available 3h after kickoff."""
    h = pd.DataFrame({"team": m.home, "gd": m.FTHG - m.FTAG, "available_at": m.kickoff_utc})
    a = pd.DataFrame({"team": m.away, "gd": m.FTAG - m.FTHG, "available_at": m.kickoff_utc})
    t = pd.concat([h, a], ignore_index=True)
    t["available_at"] = t["available_at"] + pd.Timedelta(hours=3)
    return t.sort_values("available_at", kind="mergesort").reset_index(drop=True)


def feat_fn(visible: pd.DataFrame, bet: pd.Series) -> dict:
    def f(team):
        g = visible.loc[visible.team == team, "gd"]
        return g.sum() / (len(g) + 5.0)

    return {"x": f(bet.home) - f(bet.away)}


def featurise(m: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    bets = m.assign(bet_time=m.kickoff_utc - pd.Timedelta(hours=1))
    feats = build_features(table, bets, feat_fn)
    out = m.copy()
    out["x"] = feats["x"]
    out["y"] = (m.FTHG > m.FTAG).astype(int)
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    clf = LogisticRegression().fit(train[["x"]], train["y"])
    return pd.DataFrame(
        {
            "match_id": test.match_id,
            "y": test.y,
            "p": clf.predict_proba(test[["x"]])[:, 1],
            "base": train.y.mean(),
        }
    )


def edge(pred: pd.DataFrame) -> float:
    def ll(p, y):
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    y = pred.y.to_numpy()
    return float(ll(pred.base.to_numpy(), y) - ll(pred.p.to_numpy(), y))


def pipeline(m: pd.DataFrame, min_train_seasons=1) -> pd.DataFrame:
    df = featurise(m, results_table(m))
    return run_walkforward(df, season_folds(df, min_train_seasons), fit_predict)


@pytest.fixture(scope="module")
def world():
    return make_world(seed=7)


@pytest.fixture(scope="module")
def real_pred(world):
    return pipeline(world)


def test_planted_signal_gives_positive_edge(real_pred):
    assert edge(real_pred) > 0.02


def test_a_future_row_is_ignored_and_detected(world):
    table = results_table(world)
    bet_time = world.kickoff_utc.iloc[100] - pd.Timedelta(hours=1)
    bets = world.iloc[[100]].assign(bet_time=bet_time)
    base = build_features(table, bets, feat_fn)
    future = pd.DataFrame(
        {
            "team": [world.home.iloc[100]] * 3,
            "gd": [50, 50, 50],
            "available_at": [
                bet_time,
                bet_time + pd.Timedelta(seconds=1),
                bet_time + pd.Timedelta(days=30),
            ],
        }
    )
    dirty = pd.concat([table, future], ignore_index=True)
    # injected rows (available_at >= bet_time, including EXACTLY bet_time) never reach the feature
    pd.testing.assert_frame_equal(base, build_features(dirty, bets, feat_fn))
    # and a raw, unfiltered table is flagged
    with pytest.raises(LeakageError):
        assert_no_leak(dirty, bet_time)
    assert_no_leak(asof(dirty, bet_time), bet_time)


def test_a_naive_leaky_feature_changes_with_future_row(world):
    """Sanity: a feature that bypasses asof DOES move, so the test above is meaningful."""
    table = results_table(world)
    t = world.kickoff_utc.iloc[100] - pd.Timedelta(hours=1)
    bet = world.iloc[100]
    clean = feat_fn(table, bet)["x"]
    row = pd.DataFrame(
        {"team": [bet.home], "gd": [50], "available_at": [t + pd.Timedelta(days=1)]}
    )
    spiked = pd.concat([table, row])
    assert feat_fn(spiked, bet)["x"] != clean
    assert feat_fn(asof(spiked, t), bet)["x"] == feat_fn(asof(table, t), bet)["x"]


def test_b_shuffled_targets_collapse_edge(world, real_pred):
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(world))
    shuf = world.copy()
    shuf[["FTHG", "FTAG"]] = world[["FTHG", "FTAG"]].to_numpy()[perm]
    e = edge(pipeline(shuf))
    assert abs(e) < 0.01
    assert e < edge(real_pred) / 4


def test_c_regime_reversal_flips_sign():
    same = make_world(seed=7, signs=(1, 1, 1, 1))
    rev = make_world(seed=7, signs=(1, 1, 1, -1))
    assert edge(pipeline(same, min_train_seasons=3)) > 0.02
    assert edge(pipeline(rev, min_train_seasons=3)) < -0.02


def test_d_determinism(world, real_pred):
    again = pipeline(world)
    assert real_pred.to_csv().encode() == again.to_csv().encode()
    assert real_pred.to_parquet() == again.to_parquet()


def test_walkforward_folds_are_chronological(world):
    df = featurise(world, results_table(world))
    folds = list(season_folds(df))
    assert [f.label for f in folds] == SEASONS[1:]
    for f in folds:
        tr, te = df.iloc[f.train_idx], df.iloc[f.test_idx]
        assert tr.kickoff_utc.max() < te.kickoff_utc.min()
        assert set(tr.season).isdisjoint(set(te.season))
