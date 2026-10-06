"""Tests for fedge.models.pool (log opinion pool of a structural model vs the market)."""

import numpy as np
import pytest

from fedge.models.pool import fit_weight, fit_weight_bootstrap, log_pool, pool_log_loss


def _market_and_outcomes(seed: int = 0, n: int = 4000, k: int = 3):
    """Dirichlet market rows with outcomes drawn from the market (market is the truth)."""
    rng = np.random.default_rng(seed)
    p_market = rng.dirichlet(np.ones(k), size=n)
    y = np.array([rng.choice(k, p=row) for row in p_market])
    return p_market, y


def _good_model(p_market, y, tilt: float = 0.6):
    """Market rows tilted toward the realised outcome: a genuinely better predictor."""
    oh = np.eye(p_market.shape[1])[y]
    p = p_market * (1.0 + tilt * oh)
    return p / p.sum(axis=1, keepdims=True)


def _bad_model(p_market, seed: int = 1, noise: float = 0.3):
    """Market rows plus extra uniform noise and no signal: worse than the market."""
    rng = np.random.default_rng(seed)
    p = p_market + rng.uniform(0.0, noise, size=p_market.shape)
    return p / p.sum(axis=1, keepdims=True)


def _half_informative(p_market, y, seed: int = 5, frac: float = 0.5, tilt: float = 0.6):
    """Informative on a fraction of rows, uniform (noise) on the rest: interior optimum."""
    rng = np.random.default_rng(seed)
    oh = np.eye(p_market.shape[1])[y]
    p = p_market.copy()
    m = rng.random(len(y)) < frac
    p[m] = p_market[m] * (1.0 + tilt * oh[m])
    p[~m] = 1.0 / p_market.shape[1]
    return p / p.sum(axis=1, keepdims=True)


def test_endpoints_recover_market_and_model():
    p_market, _ = _market_and_outcomes(n=200)
    p_model = np.random.default_rng(9).dirichlet(np.ones(3), size=len(p_market))
    np.testing.assert_allclose(log_pool(p_model, p_market, 0.0), p_market)
    np.testing.assert_allclose(log_pool(p_model, p_market, 1.0), p_model)


def test_rows_sum_to_one_for_range_of_weights():
    p_market, _ = _market_and_outcomes(n=200)
    p_model = np.random.default_rng(9).dirichlet(np.ones(3), size=len(p_market))
    for w in np.linspace(0.0, 1.0, 11):
        p = log_pool(p_model, p_market, w)
        assert p.shape == p_market.shape
        assert (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_good_model_gets_high_weight():
    p_market, y = _market_and_outcomes()
    w = fit_weight(_good_model(p_market, y), p_market, y)
    assert isinstance(w, float)
    assert w >= 0.8


def test_bad_model_gets_low_weight():
    p_market, y = _market_and_outcomes()
    w = fit_weight(_bad_model(p_market), p_market, y)
    assert isinstance(w, float)
    assert w <= 0.2


def test_identical_model_market_is_flat():
    p_market, y = _market_and_outcomes(n=1000)
    ref = pool_log_loss(p_market, p_market, 0.0, y)
    for w in (0.0, 0.5, 1.0):
        assert pool_log_loss(p_market, p_market, w, y) == pytest.approx(ref, abs=1e-10)


def test_bootstrap_seeded_ordered_and_clustered():
    p_market, y = _market_and_outcomes(n=3000)
    p_model = _half_informative(p_market, y)
    clusters = np.repeat(np.arange(750), 4)
    d1 = fit_weight_bootstrap(p_model, p_market, y, clusters, n_boot=150, seed=7)
    d2 = fit_weight_bootstrap(p_model, p_market, y, clusters, n_boot=150, seed=7)
    assert d1 == d2  # seeded: identical output
    assert d1["n"] == len(y) == 3000
    assert d1["n_clusters"] == 750
    assert d1["lo"] <= d1["w"] <= d1["hi"]
    assert d1["lo"] < d1["hi"]
    assert isinstance(d1["w"], float) and isinstance(d1["mean"], float)
    assert isinstance(d1["n"], int) and isinstance(d1["n_clusters"], int)


def test_validation_errors():
    a = np.full((4, 3), 1.0 / 3.0)
    b = np.full((5, 3), 1.0 / 3.0)
    with pytest.raises(ValueError):
        log_pool(a, b, 0.5)  # shape mismatch
    with pytest.raises(ValueError):
        log_pool(a, a, 1.5)  # weight above 1
    with pytest.raises(ValueError):
        log_pool(a, a, -0.1)  # weight below 0
    with pytest.raises(ValueError):
        log_pool(np.zeros(3), np.zeros(3), 0.5)  # not 2-D
    with pytest.raises(ValueError):
        pool_log_loss(a, a, 0.5, [0, 1, 2])  # len(y) mismatch
    with pytest.raises(ValueError):
        fit_weight(a, a, np.zeros((4, 1), dtype=int))  # y not 1-D
