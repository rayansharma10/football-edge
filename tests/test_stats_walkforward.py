"""Tests for walk-forward matchweek folds, CLV, bootstrap and bankroll simulation."""

import numpy as np
import pandas as pd
import pytest

from fedge.backtest import stats as S
from fedge.backtest import walkforward as W
from fedge.backtest.walkforward import matchweek_folds
from fedge.market import devig

# Results are only usable as a training row ~3h after kickoff (AGENTS.md rule 3, EMBARGO).
# Kept as an independent floor here so a mutated default cannot make these tests agree with it.
EMBARGO_FLOOR = pd.Timedelta(hours=3)
# Monday 2024-09-02 00:30 UTC: first kickoff of the test block (W-SUN weeks start on Monday).
TEST_START = pd.Timestamp("2024-09-02 00:30", tz="UTC")


def test_matchweek_folds_no_overlap_and_chronological():
    ko = pd.date_range("2024-08-01 15:00", periods=400, freq="19h", tz="UTC")
    m = pd.DataFrame({"kickoff_utc": ko})
    folds = list(matchweek_folds(m, min_train_matches=50))
    assert folds
    seen = set()
    for f in folds:
        assert ko[f.train_idx].max() < ko[f.test_idx].min()
        assert not (seen & set(f.test_idx))
        seen |= set(f.test_idx)


def _embargo_frame() -> pd.DataFrame:
    """Hourly history plus three matches planted around ``TEST_START``.

    The planted rows land in the previous (Sunday) matchweek, so they can only ever be training
    rows: 1h before TEST_START and exactly ``EMBARGO_FLOOR`` before it are inside the embargo
    window, 1h earlier still is legal.
    """
    hist = pd.date_range(end=TEST_START - pd.Timedelta(days=20), periods=400, freq="h", tz="UTC")
    ko = pd.DatetimeIndex(
        [
            *hist,
            TEST_START - pd.Timedelta(hours=1),
            TEST_START - EMBARGO_FLOOR,
            TEST_START - EMBARGO_FLOOR - pd.Timedelta(hours=1),
            TEST_START,
            TEST_START + pd.Timedelta(hours=2),
        ]
    )
    m = pd.DataFrame({"kickoff_utc": ko, "season": ["2023/24"] * (len(ko) - 2) + ["2024/25"] * 2})
    m["match_id"] = [f"m{i}" for i in range(len(m))]
    return m.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)


def _embargo_folds(m: pd.DataFrame) -> dict[str, list]:
    return {
        "season": list(W.season_folds(m, min_train_seasons=1)),
        "matchweek": list(W.matchweek_folds(m, min_train_matches=100)),
    }


def test_default_embargo_is_not_below_the_floor():
    """The 3h embargo is the only guard keeping same-day results out of train; keep it >= 3h."""
    assert W.EMBARGO >= EMBARGO_FLOOR


def test_every_fold_respects_the_embargo_and_never_overlaps():
    m = _embargo_frame()
    ko = m.kickoff_utc
    for kind, folds in _embargo_folds(m).items():
        assert folds, kind
        for f in folds:
            train_max = ko.iloc[f.train_idx].max()
            assert not (set(f.train_idx) & set(f.test_idx)), (kind, f.label)
            assert train_max <= f.test_start - W.EMBARGO, (kind, f.label)
            # independent of the live constant: no training row within 3h of the test block
            assert train_max <= f.test_start - EMBARGO_FLOOR, (kind, f.label, train_max)


def test_matches_inside_the_embargo_window_are_not_training_rows():
    """Planted case: a match 1h before the test block (and one exactly EMBARGO before it, the
    comparison is strict) must not be a training row; 1h earlier must be."""
    m = _embargo_frame()
    ko = m.kickoff_utc
    inside = m.index[ko == TEST_START - pd.Timedelta(hours=1)]
    boundary = m.index[ko == TEST_START - EMBARGO_FLOOR]
    legal = m.index[ko == TEST_START - EMBARGO_FLOOR - pd.Timedelta(hours=1)]
    assert len(inside) == len(boundary) == len(legal) == 1
    for kind, folds in _embargo_folds(m).items():
        matches = [f for f in folds if f.test_start == TEST_START]
        assert len(matches) == 1, kind
        train = set(matches[0].train_idx)
        assert not (train & set(inside)), f"{kind}: match 1h before test_start used as training"
        assert not (train & set(boundary)), f"{kind}: exactly {EMBARGO_FLOOR} before is unusable"
        assert train & set(legal), f"{kind}: {ko[legal[0]]} must be usable as training"


def test_clv_zero_at_fair_close_and_overround_at_raw_close():
    rng = np.random.default_rng(0)
    raw = np.column_stack(
        [rng.uniform(1.5, 3, 500), rng.uniform(3, 4, 500), rng.uniform(3, 8, 500)]
    )
    d = devig(raw, "multiplicative")
    sel = rng.integers(0, 3, 500)
    p_sel = d.probs[np.arange(500), sel]
    # bet "at the close" at the margin-free price: CLV is exactly 0
    clv = S.log_clv(1.0 / p_sel, p_sel)
    assert np.abs(clv).max() < 1e-12
    # the raw margined closing price is worse than fair: CLV = -log(booksum) < 0
    raw_clv = S.log_clv(raw[np.arange(500), sel], p_sel)
    np.testing.assert_allclose(raw_clv, -np.log(1 + d.overround), atol=1e-12)


def test_clv_bet_at_close_simulation_mean_near_zero():
    """Outcomes drawn from the fair close; backing at fair odds -> CLV 0, ROI ~ 0."""
    rng = np.random.default_rng(1)
    p = rng.dirichlet([4, 3, 3], 20000)
    y = np.array([rng.choice(3, p=r) for r in p])
    sel = rng.integers(0, 3, len(p))
    ps = p[np.arange(len(p)), sel]
    clv = S.log_clv(1 / ps, ps)
    assert abs(clv.mean()) < 1e-12
    won = y == sel
    roi = np.where(won, 1 / ps - 1, -1).mean()
    assert abs(roi) < 0.05


def test_bootstrap_seeded_and_clustered():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(300), 3)
    # strong within-cluster correlation: each cluster has a shared shock
    v = rng.normal(0, 1, 300)[clusters] + rng.normal(0, 0.1, 900)
    a = S.bootstrap_ci(v, clusters, n_boot=1000, seed=5)
    assert a == S.bootstrap_ci(v, clusters, n_boot=1000, seed=5)
    assert a != S.bootstrap_ci(v, clusters, n_boot=1000, seed=6)
    assert a["lo"] < a["mean"] < a["hi"] and a["n_clusters"] == 300
    # naive (bet-level) CI would be ~sqrt(3) too narrow; clustered must be wider
    naive = S.bootstrap_ci(v, np.arange(900), n_boot=1000, seed=5)
    assert (a["hi"] - a["lo"]) > 1.3 * (naive["hi"] - naive["lo"])


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(3)
    v = rng.normal(0.5, 1, 2000)
    ci = S.bootstrap_ci(v, np.arange(2000), seed=0)
    assert ci["lo"] < 0.5 < ci["hi"]


def _bets(rows):
    return pd.DataFrame(rows, columns=["market_id", "date", "price", "p", "won"])


def test_flat_commission_on_net_market_winnings():
    d = "2025-01-01"
    # same market: win 10@3.0 (+20), lose 10 on another selection (-10) -> net +10, comm 0.6
    b = _bets([("m1", d, 3.0, 0.4, True), ("m1", d, 2.0, 0.4, False)])
    ledger, s = S.simulate_bankroll(b, "flat", bankroll0=1000, flat_stake=10)
    assert ledger.gross_pnl.iloc[0] == pytest.approx(10.0)
    assert ledger.commission.iloc[0] == pytest.approx(0.6)
    assert s["profit"] == pytest.approx(9.4) and s["turnover"] == pytest.approx(20.0)
    # losing market pays no commission
    b2 = _bets([("m2", d, 3.0, 0.4, False)])
    _, s2 = S.simulate_bankroll(b2, "flat", flat_stake=10)
    assert s2["profit"] == pytest.approx(-10.0) and s2["commission_paid"] == 0.0


def test_kelly_stakes_and_same_day_bankroll():
    # p=0.5, price 2.2 -> b_net = 1.2*0.94 = 1.128; f = (0.5*1.128-0.5)/1.128
    f = S.kelly_fraction(0.5, 2.2)
    assert f == pytest.approx((0.5 * 1.128 - 0.5) / 1.128)
    assert S.kelly_fraction(0.3, 2.0) == 0.0  # no edge -> no bet
    b = _bets([("m1", "2025-01-01", 2.2, 0.5, True), ("m2", "2025-01-01", 2.2, 0.5, True)])
    _, s = S.simulate_bankroll(b, "kelly", bankroll0=1000, kelly_mult=0.25)
    stake = 0.25 * f * 1000  # both stakes use the day-start bankroll
    per = stake * 1.2
    assert s["turnover"] == pytest.approx(2 * stake)
    assert s["profit"] == pytest.approx(2 * (per - 0.06 * per))


def test_kelly_stake_capped():
    b = _bets([("m1", "2025-01-01", 5.0, 0.9, False)])
    _, s = S.simulate_bankroll(b, "kelly", bankroll0=1000, max_stake_frac=0.05)
    assert s["turnover"] == pytest.approx(50.0)


def test_fair_bets_lose_commission_only_on_average():
    """Betting at fair odds with true probs: flat yield is negative, driven by the 6% commission."""
    rng = np.random.default_rng(4)
    n = 20000
    p = rng.uniform(0.3, 0.7, n)
    won = rng.random(n) < p
    b = pd.DataFrame(
        {"market_id": [f"m{i}" for i in range(n)],
         "date": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(np.arange(n) // 10, "D"),
         "price": 1 / p, "p": p, "won": won}
    )
    _, s = S.simulate_bankroll(b, "flat", flat_stake=10)
    assert -0.06 < s["yield"] < 0.0
