"""Phase 5 strategy tests: edge maths, in-fold selection, no lookahead, baselines."""

from __future__ import annotations

import tomllib

import numpy as np
import pandas as pd
import pytest

from fedge.backtest import strategy as S
from fedge.backtest.stats import net_odds


def _world(
    n_per=300, seasons=("2016/17", "2017/18", "2018/19", "2019/20", "2020/21", "2021/22"), seed=0
):
    """Synthetic 1X2 world: true probs, a noisy model, a market with margin."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2016-08-01", tz="UTC")
    for si, s in enumerate(seasons):
        for i in range(n_per):
            rows.append(
                (
                    f"m{si}_{i}",
                    "E0" if i % 2 else "D1",
                    s,
                    t0 + pd.Timedelta(days=365 * si + i // 2),
                )
            )
    meta = pd.DataFrame(rows, columns=["match_id", "div", "season", "kickoff_utc"]).set_index(
        "match_id"
    )
    n = len(meta)
    true = rng.dirichlet([4, 3, 4], size=n)
    y = np.array([rng.choice(3, p=p) for p in true])
    model = true * np.exp(rng.normal(0, 0.15, true.shape))
    model /= model.sum(axis=1, keepdims=True)
    mkt = true * np.exp(rng.normal(0, 0.05, true.shape))
    mkt /= mkt.sum(axis=1, keepdims=True)
    price = 1.0 / (mkt * 1.04)
    close = mkt
    return meta, y, model, mkt, price, close


def test_candidate_picks_max_edge_with_commission():
    meta = pd.DataFrame(
        {
            "div": ["E0"],
            "season": ["2020/21"],
            "kickoff_utc": [pd.Timestamp("2021-01-01", tz="UTC")],
        },
        index=pd.Index(["a"], name="match_id"),
    )
    p = np.array([[0.5, 0.3, 0.2]])
    price = np.array([[2.1, 3.6, 5.0]])
    out = S.candidate_table(
        meta, "1x2", price, np.array([[0.45, 0.3, 0.25]]), {"m": p}, np.array([2])
    )["m"]
    edges = p * net_odds(price) - 1
    assert out["sel"].iloc[0] == "HDA"[int(edges.argmax())]
    assert out["edge"].iloc[0] == pytest.approx(edges.max())
    assert out["won"].iloc[0] == (out["sel"].iloc[0] == "A")
    # commission is only charged on winnings: a loss costs exactly the stake
    assert out["pnl_flat"].iloc[0] == (-1.0 if not out["won"].iloc[0] else out["pnl_flat"].iloc[0])


def test_edge_formula_exact():
    meta = pd.DataFrame(
        {"div": ["E0"], "season": ["s"], "kickoff_utc": [pd.Timestamp("2021-01-01", tz="UTC")]},
        index=pd.Index(["a"], name="match_id"),
    )
    out = S.candidate_table(
        meta,
        "ou25",
        np.array([[2.0, 1.9]]),
        np.array([[0.5, 0.5]]),
        {"m": np.array([[0.6, 0.4]])},
        np.array([0]),
    )["m"]
    # over: 0.6 * (1 + 1*0.94) - 1 = 0.164 ; won -> pnl 0.94
    assert out["edge"].iloc[0] == pytest.approx(0.6 * 1.94 - 1)
    assert out["pnl_flat"].iloc[0] == pytest.approx(0.94)
    assert out["clv_net"].iloc[0] == pytest.approx(np.log(1.94 * 0.5))
    assert out["clv_raw"].iloc[0] == pytest.approx(np.log(2.0 * 0.5))


def test_pooled_by_season_uses_only_earlier_seasons():
    meta, y, model, mkt, *_ = _world(n_per=200)
    season = meta["season"].to_numpy()
    pooled, wts = S.pooled_by_season({"m": model}, mkt, y, season)
    first = season == "2016/17"
    assert np.isnan(pooled["m"][first]).all()
    assert not np.isnan(pooled["m"][~first]).any()
    # alter outcomes of the LAST season: nothing in any season's pooled probs may change
    y2 = y.copy()
    y2[season == "2021/22"] = (y2[season == "2021/22"] + 1) % 3
    pooled2, wts2 = S.pooled_by_season({"m": model}, mkt, y2, season)
    np.testing.assert_array_equal(pooled["m"], pooled2["m"])
    assert wts.equals(wts2)
    # alter outcomes of 2018/19: seasons up to and including 2018/19 unchanged, later ones change
    y3 = y.copy()
    y3[season == "2018/19"] = (y3[season == "2018/19"] + 1) % 3
    p3, _ = S.pooled_by_season({"m": model}, mkt, y3, season)
    upto = np.isin(season, ["2016/17", "2017/18", "2018/19"])
    np.testing.assert_array_equal(pooled["m"][upto], p3["m"][upto])
    assert not np.allclose(pooled["m"][season == "2019/20"], p3["m"][season == "2019/20"])


def _best_tables():
    meta, y, model, mkt, price, close = _world()
    season = meta["season"].to_numpy()
    pooled, _ = S.pooled_by_season({"m": model, "n": mkt}, mkt, y, season)
    ll = S.pooled_log_loss_table(pooled, y, season)
    # price offered 30% better than fair so some edges exist
    best = {"1x2": S.candidate_table(meta, "1x2", price * 1.3, close, pooled, y)}
    seasons = sorted(set(season))[1:]
    return best, {"1x2": ll}, seasons, meta


def test_walk_forward_in_fold_selection_ignores_the_future():
    best, ll, seasons, _ = _best_tables()
    kw = {"min_bets": 20, "min_cell": 5, "min_hist_seasons": 2}
    bets, cfgs = S.run_walk_forward(best, ll, seasons, **kw)
    assert len(bets) and set(bets["season"]) == set(seasons[2:])  # first folds have no history
    # regime reversal on the LAST season only: earlier folds' choices and bets must not move
    last = seasons[-1]
    altered = {"1x2": {}}
    for m, df in best["1x2"].items():
        d = df.copy()
        sel = d["season"] == last
        d.loc[sel, "won"] = ~d.loc[sel, "won"]
        d.loc[sel, "clv_net"] = -d.loc[sel, "clv_net"]
        d.loc[sel, "pnl_flat"] = -d.loc[sel, "pnl_flat"]
        altered["1x2"][m] = d
    bets2, cfgs2 = S.run_walk_forward(altered, ll, seasons, **kw)
    a = cfgs[cfgs["season"] != last].reset_index(drop=True)
    b = cfgs2[cfgs2["season"] != last].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    ea = bets[bets["season"] != last]
    eb = bets2[bets2["season"] != last]
    assert list(ea.index) == list(eb.index)
    assert ea["selected"].tolist() == eb["selected"].tolist()


def test_selected_bets_only_in_test_seasons_and_respect_threshold():
    best, ll, seasons, _ = _best_tables()
    bets, _ = S.run_walk_forward(best, ll, seasons, min_bets=20, min_cell=5)
    sel = bets[bets["selected"]]
    assert (sel["edge"] > sel["threshold"]).all()
    assert sel["whitelisted"].all()
    assert len(sel) > 0 and sel["season"].min() >= seasons[2]


def test_select_market_config_threshold_whitelist_and_min_bets():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "season": ["a"] * n,
            "div": np.where(np.arange(n) % 2 == 0, "E0", "D1"),
            "edge": rng.uniform(0, 0.1, n),
        }
    )
    # E0 has positive CLV, D1 negative
    df["clv_net"] = np.where(df["div"] == "E0", 0.02, -0.02) + rng.normal(0, 0.01, n)
    cfg = S.select_market_config(df, ["a"], min_bets=50, min_cell=10)
    assert cfg is not None and cfg["whitelist"] == ["E0"]
    assert S.select_market_config(df, ["a"], min_bets=10_000) is None
    assert S.select_market_config(df, ["zzz"], min_bets=1) is None  # no history, no config


def test_pick_model_uses_history_only():
    ll = pd.DataFrame(
        {
            "model": ["a", "b", "a", "b"],
            "season": ["s1", "s1", "s2", "s2"],
            "n": [10, 10, 10, 10],
            "ll": [1.0, 1.1, 2.0, 0.5],
        }
    )
    assert S.pick_model(ll, ["s1"]) == "a"
    assert S.pick_model(ll, ["s1", "s2"]) == "b"
    assert S.pick_model(ll, []) is None


def test_select_kelly_falls_back_to_smallest():
    assert S.select_kelly(pd.DataFrame()) == 0.1


def test_favourite_is_shortest_price_and_baseline_is_deterministic():
    meta, y, model, mkt, price, close = _world(n_per=100)
    fav = S.favourite_table(meta, "1x2", price, close, y)
    assert (fav["price"].to_numpy() == price.min(axis=1)).all()
    cand = S.long_candidates(meta, "1x2", price, close, y)
    assert len(cand) == 3 * len(meta)
    counts = pd.DataFrame(
        {"div": ["E0", "D1"], "season": ["2017/18"] * 2, "market": ["1x2"] * 2, "n": [20, 15]}
    )
    r1 = S.random_baseline(cand, counts, n_rep=10, seed=3)
    r2 = S.random_baseline(cand, counts, n_rep=10, seed=3)
    pd.testing.assert_frame_equal(r1, r2)
    assert (r1["n"] == 35).all()
    # random 1X2 selection at a 4% margin and 6% commission cannot have CLV above 0 on average
    assert r1["clv_net"].mean() < 0


def test_ledger_metrics_known_values():
    idx = pd.Index(["a", "b", "c", "d"], name="match_id")
    b = pd.DataFrame(
        {
            "price": [2.0, 2.0, 3.0, 3.0],
            "p_close": [0.5, 0.5, 1 / 3, 1 / 3],
            "won": [True, False, True, False],
            "kickoff_utc": pd.date_range("2021-01-01", periods=4, tz="UTC"),
            "market": "1x2",
            "p": [0.55, 0.55, 0.36, 0.36],
        },
        index=idx,
    )
    b["clv_raw"] = np.log(b["price"] * b["p_close"])
    b["clv_net"] = np.log(net_odds(b["price"]) * b["p_close"])
    b["pnl_flat"] = np.where(b["won"], (b["price"] - 1) * 0.94, -1.0)
    b["market_id"] = idx.astype(str) + "|1x2"
    m = S.ledger_metrics(b, n_boot=50)
    assert m["n"] == 4
    assert m["clv_raw"] == pytest.approx(0.0, abs=1e-12)
    assert m["clv_net"] < 0
    assert m["roi_flat"] == pytest.approx((0.94 - 1 + 2 * 0.94 - 1) / 4)
    cm = S.cell_metrics(b, ["market"], n_boot=50)
    assert cm["n"].tolist() == [4]


def test_render_strategy_toml_parses():
    txt = S.render_strategy_toml(
        {
            "comments": ["hello"],
            "top": {"gate0_passed": False, "mode": "shadow", "kelly_fraction": 0.1},
            "markets": {
                "1x2": {
                    "model": "lgb_xg",
                    "pool_weight": 0.05,
                    "threshold": 0.04,
                    "whitelist": ["E0", "D1"],
                }
            },
            "shadow": {"shadow_threshold": 0.03},
        }
    )
    d = tomllib.loads(txt)
    assert d["gate0_passed"] is False and d["mode"] == "shadow"
    assert d["markets"]["1x2"]["whitelist"] == [["E0", "1x2"], ["D1", "1x2"]]
    assert d["shadow"]["shadow_threshold"] == pytest.approx(0.03)
    # a cfg with no [shadow] section stays valid (the desk then falls back to its own default)
    assert "shadow" not in tomllib.loads(
        S.render_strategy_toml({"comments": [], "top": {}, "markets": {}})
    )


def test_select_market_config_has_no_positivity_floor():
    """Documented behaviour (gate0.md section 1, B0.9 PARTIAL, review M2).

    The threshold search maximises mean net CLV x sqrt(n) and accepts the argmax whenever the
    subset has MIN_HIST_BETS rows, so a threshold whose in-fold history LOST money is still
    selected; only the whitelist can produce a fold with no bets. Pinned here so that adding a
    positivity floor is a deliberate, test-visible change (it would empty the headline ledger).
    """
    n = 400
    df = pd.DataFrame(
        {
            "season": ["a"] * n,
            "div": np.where(np.arange(n) % 2 == 0, "E0", "D1"),
            "edge": 0.03,
            "clv_net": -0.10,
            "clv_raw": -0.02,
        }
    )
    cfg = S.select_market_config(df, ["a"], min_bets=50, min_cell=10)
    assert cfg is not None  # a losing history still yields a configuration
    assert cfg["hist_clv"] < 0 and cfg["hist_score"] < 0
    assert cfg["hist_clv_raw"] < 0  # the raw (skill) metric is recorded beside the net one
    assert cfg["hist_n"] == n
    assert cfg["whitelist"] == []  # the whitelist is the only no-bet gate


def test_paired_clv_diff_is_paired_and_deterministic():
    idx = pd.Index(["a", "b", "c", "d"], name="match_id")
    strat = pd.DataFrame({"market": "1x2", "clv_net": [0.10, 0.20, -0.05, 0.00]}, index=idx)
    base = pd.DataFrame({"market": "1x2", "clv_net": [0.00, 0.10, -0.05, -0.10]}, index=idx)
    d1 = S.paired_clv_diff(strat, base, n_boot=200, seed=0)
    d2 = S.paired_clv_diff(strat, base, n_boot=200, seed=0)
    assert d1 == d2  # seeded: same inputs, same output
    assert d1["n"] == 4 and d1["n_strategy"] == 4
    assert d1["mean"] == pytest.approx(np.mean([0.10, 0.10, 0.0, 0.10]))
    assert d1["lo"] <= d1["mean"] <= d1["hi"]
    # only the (match_id, market) pairs both sides share are used
    d3 = S.paired_clv_diff(strat, base.iloc[:2], n_boot=100, seed=0)
    assert d3["n"] == 2 and d3["n_strategy"] == 4
    assert S.paired_clv_diff(strat.iloc[0:0], base, n_boot=10) == {"n": 0}
