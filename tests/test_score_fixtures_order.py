"""Regression (t_187194d2): priced score_fixtures feeds LightGBM the training column order."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from fedge.paper import model_state


def test_priced_scoring_is_invariant_to_fixture_column_order():
    rng = np.random.default_rng(0)
    cols = ["elo_diff", "pi_diff", "ad_att_h", "h_gf"]
    xtr = pd.DataFrame(rng.normal(size=(300, 4)), columns=cols)
    ytr = (xtr["elo_diff"] * 2 + rng.normal(size=300) > 0).astype(int) + (
        xtr["h_gf"] > 1
    ).astype(int)
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbose": -1, "min_data_in_leaf": 5},
        lgb.Dataset(xtr, ytr),
        num_boost_round=20,
    )
    keys = ["f0", "f1", "f2"]
    x_train_order = pd.DataFrame(rng.normal(size=(3, 4)), columns=cols, index=keys)
    x_fixture_order = x_train_order[cols[::-1]]  # fixture_features order differs from training
    sels = list(zip("HDA", (2.0, 3.5, 4.0), strict=True))
    prices = pd.DataFrame(
        [(k, "1x2", s, p, "test") for k in keys for s, p in sels],
        columns=["match_key", "market", "selection", "price", "price_source"],
    )
    info = {"decor": False, "calibrator": "none", "best_iter": 20}
    fits = {"1x2": ("lgb", booster, lambda p: p, info)}
    w = {"1x2": 0.5}
    a = model_state.score_fixtures(fits, x_train_order, prices, w)
    b = model_state.score_fixtures(fits, x_fixture_order, prices, w)
    pd.testing.assert_frame_equal(a, b)
    expect = booster.predict(x_train_order)
    got = a[a["selection"] == "H"]["model_prob"].to_numpy()
    np.testing.assert_allclose(got, expect[:, 0])
