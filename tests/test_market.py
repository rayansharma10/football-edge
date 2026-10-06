"""Tests for fedge.market (de-margining) vs penaltyblog and analytic cases."""

import numpy as np
import penaltyblog as pb
import pytest

from fedge.market import devig, implied_1x2, implied_2way

CASES = [[2.7, 2.3, 4.4], [1.5, 4.2, 7.5], [1.9, 1.95], [3.0, 3.0, 3.0]]


@pytest.mark.parametrize("odds", CASES)
@pytest.mark.parametrize("method", ["power", "shin", "multiplicative"])
def test_matches_penaltyblog(odds, method):
    ref = np.array(pb.implied.calculate_implied(odds, method=method).probabilities)
    got = devig(odds, method).probs
    np.testing.assert_allclose(got, ref, atol=1e-6)
    assert got.sum() == pytest.approx(1.0, abs=1e-12)


def test_known_values_multiplicative():
    # penaltyblog docstring example: [2.7, 2.3, 4.4] -> 0.35873804, 0.42112726, 0.2201347
    p = devig([2.7, 2.3, 4.4], "multiplicative").probs
    np.testing.assert_allclose(p, [0.35873804, 0.42112726, 0.2201347], atol=1e-7)


def test_fair_book_unchanged():
    p = devig([2.0, 4.0, 4.0]).probs
    np.testing.assert_allclose(p, [0.5, 0.25, 0.25], atol=1e-9)


def test_power_exponent_above_one_with_overround():
    d = devig([2.7, 2.3, 4.4], "power")
    assert d.param > 1.0 and d.overround > 0


def test_vectorised_matches_rowwise():
    arr = np.array(CASES[:2])
    for m in ("power", "shin", "multiplicative"):
        batch = devig(arr, m).probs
        for i, row in enumerate(arr):
            np.testing.assert_allclose(batch[i], devig(row, m).probs, atol=1e-12)


def test_helpers_and_2way():
    d = implied_1x2([2.7, 1.5], [2.3, 4.2], [4.4, 7.5])
    assert d.probs.shape == (2, 3)
    d2 = implied_2way([1.9], [1.95])
    np.testing.assert_allclose(d2.probs.sum(axis=1), 1.0)
    # Shin == additive for two outcomes: p_i = r_i - (sum r - 1)/2 ~ checked loosely
    r = 1 / np.array([1.9, 1.95])
    add = r - (r.sum() - 1) / 2
    np.testing.assert_allclose(devig([1.9, 1.95], "shin").probs, add, atol=1e-3)


def test_bad_input():
    with pytest.raises(ValueError):
        devig([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        devig([2.0, 2.0], "nope")


def test_deterministic():
    a = devig(CASES[0], "shin").probs.tobytes()
    assert a == devig(CASES[0], "shin").probs.tobytes()
