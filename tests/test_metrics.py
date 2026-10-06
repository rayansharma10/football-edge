"""Metric tests against hand-computed examples."""

import numpy as np
import pytest

from fedge.report import metrics as M


def test_rps_hand():
    # one match, forecast (0.5, 0.3, 0.2), outcome H (class 0):
    # cumP = (0.5, 0.8), cumO = (1, 1) -> ((0.5)^2 + (0.2)^2)/2 = 0.145
    assert M.rps([[0.5, 0.3, 0.2]], [0]) == pytest.approx(0.145)
    # outcome A: cumO = (0, 0) -> (0.25 + 0.64)/2 = 0.445
    assert M.rps([[0.5, 0.3, 0.2]], [2]) == pytest.approx(0.445)
    # mean of both
    assert M.rps([[0.5, 0.3, 0.2]] * 2, [0, 2]) == pytest.approx((0.145 + 0.445) / 2)


def test_rps_perfect_and_worst():
    assert M.rps([[1, 0, 0]], [0]) == 0.0
    assert M.rps([[1, 0, 0]], [2]) == pytest.approx(1.0)


def test_log_loss_hand():
    assert M.log_loss([[0.5, 0.3, 0.2]], [1]) == pytest.approx(-np.log(0.3))
    assert M.log_loss([[0.25, 0.25, 0.5]] * 2, [0, 2]) == pytest.approx(
        (-np.log(0.25) - np.log(0.5)) / 2
    )


def test_brier_hand():
    # (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    assert M.brier([[0.5, 0.3, 0.2]], [0]) == pytest.approx(0.38)


def test_ece_hand():
    # bin [0.7,0.8): p = 0.75 x4, 3 of 4 hit -> gap 0; bin [0.2,0.3): p=0.25 x4, 2 hit -> gap 0.25
    p = [0.75] * 4 + [0.25] * 4
    y = [1, 1, 1, 0] + [1, 1, 0, 0]
    assert M.ece(p, y) == pytest.approx((4 * 0 + 4 * 0.25) / 8)


def test_reliability_table_counts():
    t = M.reliability_table([0.05, 0.05, 0.95], [0, 1, 1])
    assert list(t["n"]) == [2, 1]
    assert list(t["obs_freq"]) == [0.5, 1.0]


def test_calibration_slope_recovers_known():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 40000)
    x = np.log(p / (1 - p))
    true = 1 / (1 + np.exp(-(0.0 + 0.5 * x)))  # forecasts overconfident: true slope 0.5
    y = rng.random(40000) < true
    a, b = M.calibration_slope(p, y)
    assert b == pytest.approx(0.5, abs=0.05) and abs(a) < 0.05
    y2 = rng.random(40000) < p  # calibrated
    assert M.calibration_slope(p, y2)[1] == pytest.approx(1.0, abs=0.05)


def test_plot_reliability(tmp_path):
    out = tmp_path / "r.png"
    M.plot_reliability([0.1, 0.5, 0.9, 0.9], [0, 1, 1, 1], out)
    assert out.exists() and out.stat().st_size > 0
