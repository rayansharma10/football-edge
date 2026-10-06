"""Calibrators and the GBM walk-forward (small synthetic data; no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from fedge.models import calibrate as C
from fedge.models import gbm
from fedge.report import metrics as M


def _overconfident(n=4000, k=3, seed=0):
    """Forecasts that are too extreme: true probs are a tempered version of the stated ones."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.2, size=(n, k))
    true = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    stated = np.exp(2.0 * logits) / np.exp(2.0 * logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in true])
    return stated, y


@pytest.mark.parametrize("method", ["isotonic", "beta", "venn_abers"])
def test_calibrators_fix_overconfidence(method):
    p, y = _overconfident()
    cut = 2500
    cal = C.FITTERS[method](p[:cut], y[:cut])
    out = cal(p[cut:])
    assert out.shape == p[cut:].shape and np.allclose(out.sum(axis=1), 1.0)
    assert M.log_loss(out, y[cut:]) < M.log_loss(p[cut:], y[cut:]) - 0.01
    assert abs(M.calibration_slope_multi(out, y[cut:])[1] - 1.0) < abs(
        M.calibration_slope_multi(p[cut:], y[cut:])[1] - 1.0
    )


def test_binary_calibration_and_identity():
    rng = np.random.default_rng(1)
    q = rng.uniform(0.1, 0.9, 3000)
    y = (rng.uniform(size=3000) < q).astype(int)
    p = np.column_stack([q, 1 - q])
    yy = 1 - y  # column 0 = positive class
    ident = C.fit_identity(p, yy)(p)
    assert np.allclose(ident, p, atol=1e-5)
    for method in ("isotonic", "beta", "venn_abers"):
        out = C.FITTERS[method](p, yy)(p)
        assert np.allclose(out.sum(axis=1), 1.0)
        assert abs(out[:, 0].mean() - q.mean()) < 0.03  # already calibrated: stays calibrated


def test_select_calibrator_prefers_a_real_fix_and_keeps_identity_when_calibrated():
    p, y = _overconfident(n=6000, seed=2)
    name, cal, scores = C.select_calibrator(p, y)
    assert name != "identity" and scores["identity"] > scores[name]
    assert cal(p[:5]).shape == (5, 3)
    rng = np.random.default_rng(3)
    q = rng.dirichlet([2, 2, 2], size=6000)
    y2 = np.array([rng.choice(3, p=r) for r in q])
    name2, _, scores2 = C.select_calibrator(q, y2)
    assert scores2[name2] <= scores2["identity"] + 1e-12
    name3, _, scores3 = C.select_calibrator(q[:50], y2[:50])
    assert name3 == "identity" and scores3 == {}


def test_select_calibrator_uses_only_chronological_halves():
    """Fit on the first half, score on the second: poisoning the first half's order changes
    nothing about which rows are scored (a later-row-only evaluation)."""
    p, y = _overconfident(n=1000, seed=4)
    _, _, s1 = C.select_calibrator(p, y)
    y_bad = y.copy()
    y_bad[:500] = (y_bad[:500] + 1) % 3  # corrupt the fitting half only
    _, _, s2 = C.select_calibrator(p, y_bad)
    assert s1["identity"] == pytest.approx(s2["identity"])  # identity scored on rows 500: only


def _gbm_table(n=3600, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-08-01 15:00", tz="UTC")
    season = np.repeat(["2018/19", "2019/20", "2020/21", "2021/22"], n // 4)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    z = 0.9 * x1
    p_h = 1 / (1 + np.exp(-(z + 0.1)))
    y = np.where(rng.uniform(size=n) < p_h, 0, np.where(rng.uniform(size=n) < 0.4, 1, 2))
    goals = rng.poisson(np.exp(0.2 + 0.5 * x2))
    t = pd.DataFrame(
        {
            "kickoff_utc": start + pd.to_timedelta(np.arange(n) * 6, unit="h"),
            "season": season,
            "x1": x1,
            "x2": x2,
            "y_1x2": y,
            "y_ou25": np.where(goals > 2.5, 0, 1),
        },
        index=[f"m{i}" for i in range(n)],
    )
    return t


@pytest.mark.parametrize("learner", ["lgb", "cb"])
def test_walk_forward_is_deterministic_and_beats_climatology(learner):
    t = _gbm_table()
    kw = dict(
        feature_cols=["x1", "x2"],
        y_col="y_1x2",
        task="1x2",
        learner=learner,
        test_seasons=["2020/21", "2021/22"],
        threads=2,
    )
    a, info = gbm.walk_forward(t, **kw)
    b, _ = gbm.walk_forward(t, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["season"]) == {"2020/21", "2021/22"} and len(info) == 2
    p = a[["cal_h", "cal_d", "cal_a"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    y = t.loc[a.index, "y_1x2"].to_numpy()
    base = np.tile(np.bincount(y) / len(y), (len(y), 1))
    assert M.log_loss(a[["raw_h", "raw_d", "raw_a"]].to_numpy(), y) < M.log_loss(base, y)


def test_walk_forward_never_trains_on_the_test_season():
    """Flip every label from the first test season on: its predictions must not change."""
    t = _gbm_table(seed=1)
    kw = dict(
        feature_cols=["x1", "x2"],
        y_col="y_1x2",
        task="1x2",
        learner="lgb",
        test_seasons=["2020/21"],
        threads=2,
    )
    a, _ = gbm.walk_forward(t, **kw)
    t2 = t.copy()
    later = t2["season"] >= "2020/21"
    t2.loc[later, "y_1x2"] = (t2.loc[later, "y_1x2"] + 1) % 3
    b, _ = gbm.walk_forward(t2, **kw)
    pd.testing.assert_frame_equal(a, b)


def test_binary_task_and_residual_variant_tracks_the_market():
    t = _gbm_table(seed=2)
    rng = np.random.default_rng(5)
    # a market that already knows x2 perfectly: the residual model should stay near it
    true_over = 1 / (1 + np.exp(-(0.5 * t["x2"].to_numpy())))
    t["mk_over"], t["mk_under"] = true_over, 1 - true_over
    t["y_ou25"] = np.where(rng.uniform(size=len(t)) < true_over, 0, 1)
    kw = dict(
        feature_cols=["x1", "x2"],
        y_col="y_ou25",
        task="ou25",
        learner="lgb",
        test_seasons=["2021/22"],
        threads=2,
    )
    plain, _ = gbm.walk_forward(t, **kw)
    resid, info = gbm.walk_forward(t, init_cols=["mk_over", "mk_under"], **kw)
    ids = resid.index
    mk = t.loc[ids, ["mk_over", "mk_under"]].to_numpy()
    assert np.abs(resid[["raw_over", "raw_under"]].to_numpy() - mk).max() < 0.15
    assert np.allclose(plain[["cal_over", "cal_under"]].sum(axis=1), 1.0)
    assert info["best_iter"].iloc[0] >= 1
    y = t.loc[ids, "y_ou25"].to_numpy()
    assert M.log_loss(resid[["raw_over", "raw_under"]].to_numpy(), y) < M.log_loss(
        np.column_stack([np.full(len(y), 0.5)] * 2), y
    )
