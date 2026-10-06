"""Scoreline model + upcoming-predictions table: pure functions, no I/O surprises."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedge.models.dixon_coles import score_grid
from fedge.paper import ledger
from fedge.predict import upcoming as pred
from fedge.predict.scoreline import GRID_SHOW, predict_match


def test_probabilities_and_grid_sum_to_one():
    pm = predict_match(1.6, 1.1, rho=-0.05)
    assert pm["p_home"] + pm["p_draw"] + pm["p_away"] == pytest.approx(1.0, abs=1e-12)
    assert pm["p_over25"] + pm["p_under25"] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < pm["p_btts"] < 1.0
    assert pm["xg_home"] > pm["xg_away"] > 0
    # the displayed 6x6 grid is a slice of a normalised grid: it can hold nearly all of it
    assert 0.9 < pm["grid_mass_shown"] <= 1.0
    assert pm["reconciled"] is False and pm["method"] == "dixon-coles"


def test_top_scores_sorted_and_plausible():
    pm = predict_match(1.4, 1.2, rho=-0.03)
    top = pm["top_scores"]
    assert len(top) == 5
    assert [t["p"] for t in top] == sorted((t["p"] for t in top), reverse=True)
    assert top[0]["p"] > 1 / 25  # a 5-goal-each grid has 25 cells, the mode must beat the mean
    assert all(0 <= t["home"] < 12 and 0 <= t["away"] < 12 for t in top)
    assert len(pm["grid"]) == GRID_SHOW and all(len(r) == GRID_SHOW for r in pm["grid"])


def test_grid_is_normalised_dixon_coles_when_no_target():
    pm = predict_match(2.0, 0.8, rho=-0.02)
    g = score_grid(2.0, 0.8, -0.02, 12)[0]
    g = g / g.sum()
    assert pm["xg_home"] == pytest.approx((g.sum(axis=1) * np.arange(12)).sum(), abs=1e-12)
    assert pm["dc_p"] == pytest.approx(
        [g[np.tril_indices(12, -1)].sum(), np.trace(g), g[np.triu_indices(12, 1)].sum()],
        abs=1e-12,
    )


def test_reconciled_grid_matches_target_exactly():
    target = (0.55, 0.25, 0.20)
    pm = predict_match(1.9, 1.0, rho=-0.04, target=target)
    assert pm["p_home"] == pytest.approx(target[0], abs=1e-9)
    assert pm["p_draw"] == pytest.approx(target[1], abs=1e-9)
    assert pm["p_away"] == pytest.approx(target[2], abs=1e-9)
    assert pm["reconciled"] is True
    assert "1X2" in pm["method"]
    # the reconciled grid is still a probability grid over 12x12
    full = np.array(pm["grid"])
    assert 0.0 < full.sum() <= 1.0


def test_reconciliation_moves_lambdas_but_keeps_xg_sane():
    pm = predict_match(1.2, 1.2, rho=0.0, target=(0.7, 0.2, 0.1))
    assert pm["p_home"] == pytest.approx(0.7, abs=1e-9)
    assert pm["lambda_home"] != 1.2 and pm["lambda_home"] != pm["lambda_away"]
    assert 0.05 <= pm["lambda_home"] <= 8.0 and 0.05 <= pm["lambda_away"] <= 8.0
    assert pm["xg_home"] > pm["xg_away"]  # a 70/20/10 target must imply a home-leaning grid


def test_extreme_target_falls_back_to_rescaling():
    """A target the two-parameter solve cannot reach still yields an exact 1X2."""
    pm = predict_match(1.0, 1.0, rho=0.0, target=(0.98, 0.01, 0.01))
    assert pm["p_home"] == pytest.approx(0.98, abs=1e-9)
    assert pm["p_draw"] == pytest.approx(0.01, abs=1e-9)


@pytest.mark.parametrize(
    "target", [(0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.6, -0.1), (-1, 1.5, 0.5)]
)
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        predict_match(1.5, 1.1, target=target)


@pytest.mark.parametrize("lh,la", [(0, 1.0), (1.0, -1), (np.nan, 1.0)])
def test_invalid_lambdas_rejected(lh, la):
    with pytest.raises(ValueError):
        predict_match(lh, la)


# ------------------------------------------------------------------ upcoming-predictions rows
def _played():
    rows = [
        ("E0", "2026-08-15T14:00:00+00:00", "Arsenal", "Chelsea", 2, 1),
        ("E0", "2026-08-16T14:00:00+00:00", "Chelsea", "Arsenal", 1, 1),
        ("E2", "2026-08-15T14:00:00+00:00", "Wrexham", "Bristol Rovers", 3, 0),
    ]
    return pd.DataFrame(
        [
            {"match_id": f"m{i}", "div": d, "kickoff_utc": pd.Timestamp(k, tz="UTC"),
             "home": h, "away": a, "FTHG": fh, "FTAG": fa, "FTR": "H"}
            for i, (d, k, h, a, fh, fa) in enumerate(rows)
        ]
    )


def _fixtures():
    return pd.DataFrame(
        [
            {"match_key": "k1", "div": "E0", "date": "2026-10-10", "time": "15:00",
             "time_unknown": False, "home": "Arsenal", "away": "Chelsea",
             "kickoff_utc": pd.Timestamp("2026-10-10T14:00:00Z")},
            {"match_key": "k2", "div": "E2", "date": "2026-10-10", "time": "15:00",
             "time_unknown": False, "home": "Wrexham", "away": "Bristol Rovers",
             "kickoff_utc": pd.Timestamp("2026-10-10T14:00:00Z")},
            {"match_key": "k3", "div": "SP1", "date": "2026-10-11", "time": "20:00",
             "time_unknown": False, "home": "Real Madrid", "away": "Barcelona",
             "kickoff_utc": pd.Timestamp("2026-10-11T19:00:00Z")},
        ]
    )


def _edges():
    return pd.DataFrame(
        [
            {"match_key": "k1", "market": "1x2", "selection": "H", "model_prob": 0.50,
             "market_prob": 0.45, "price": 2.2, "price_source": "PS", "pooled_prob": 0.48,
             "edge": 0.05},
            {"match_key": "k1", "market": "1x2", "selection": "D", "model_prob": 0.27,
             "market_prob": 0.28, "price": 3.4, "price_source": "PS", "pooled_prob": 0.27,
             "edge": 0.0},
            {"match_key": "k1", "market": "1x2", "selection": "A", "model_prob": 0.23,
             "market_prob": 0.27, "price": 3.6, "price_source": "PS", "pooled_prob": 0.25,
             "edge": 0.0},
        ]
    )


def test_build_rows_scopes_to_desk_divisions_and_flags_unscored():
    rows = pred.build_rows(
        _edges(), _fixtures(), _played(), pd.Timestamp("2026-10-06T00:00:00Z"),
        desk_divs=("E0", "SP1"),
    )
    keys = [r["match_key"] for r in rows]
    assert "k2" not in keys  # E2 is out of scope, silently dropped (no not-modelled row)
    by_key = {r["match_key"]: r for r in rows}
    assert by_key["k1"]["status"] == "modelled"
    assert by_key["k3"]["status"] == "not_modelled"
    assert "price" in by_key["k3"]["reason"]
    assert rows == sorted(rows, key=lambda r: (r["kickoff_utc"], r["home"]))


def test_modelled_row_targets_the_lightgbm_1x2():
    r = pred.build_rows(
        _edges(), _fixtures(), _played(), pd.Timestamp("2026-10-06T00:00:00Z"),
        desk_divs=("E0",),
    )[0]
    assert r["p_home"] == pytest.approx(0.50, abs=1e-9)
    assert r["p_draw"] == pytest.approx(0.27, abs=1e-9)
    assert r["p_away"] == pytest.approx(0.23, abs=1e-9)
    assert (r["mkt_home"], r["mkt_draw"], r["mkt_away"]) == (0.45, 0.28, 0.27)
    assert len(json.loads(r["top_scores"])) == 5
    assert len(json.loads(r["grid"])) == GRID_SHOW
    detail = json.loads(r["detail"])
    assert detail["fallback"] == "league-average goals (no DC fit)"  # too little history here
    assert detail["lambda_dc"] == [1.5, 1.0]  # league means of the two E0 rows in the fixture


def test_write_predictions_replaces_table(tmp_path):
    conn = ledger.connect(tmp_path / "paper.sqlite")
    rows = pred.build_rows(
        _edges(), _fixtures(), _played(), pd.Timestamp("2026-10-06T00:00:00Z"), desk_divs=("E0",)
    )
    assert pred.write_predictions(conn, rows) == 1
    first = conn.execute("SELECT match_key, p_home, run_ts FROM predictions").fetchall()
    assert first[0][0] == "k1" and first[0][1] == pytest.approx(0.50)
    # a second run without any scored fixture must not leave k1 behind
    assert pred.write_predictions(conn, []) == 0
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0


def test_unmodelled_row_has_no_probabilities():
    r = pred.build_rows(
        pd.DataFrame(), _fixtures().iloc[[2]], _played(),
        pd.Timestamp("2026-10-06T00:00:00Z"), desk_divs=("E0", "SP1"),
    )[0]
    assert r["status"] == "not_modelled" and r["p_home"] is None and r["grid"] is None
    assert r["league"] == "Spain La Liga" and r["kickoff_utc"] == "2026-10-11T19:00:00+00:00"


def test_desk_divs_from_config():
    assert pred.load_desk_divs("config/leagues.toml") == ("E0", "SP1")
    assert pred.load_desk_divs("does-not-exist.toml") == pred.DESK_DIVS


# ------------------------------------------------------------------ lookup CLI (`--home/--away`)
def _script(name):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lookup_prints_the_live_model_numbers(capsys):
    """Regression: ``row.div`` on a pandas Series is the truediv METHOD, not the column."""
    import argparse

    mod = _script("predict_upcoming")
    fix = _fixtures().iloc[[0]]
    args = argparse.Namespace(home="Arsenal", away="Chelsea", json=False)
    assert mod.lookup(args, _played(), fix, _edges(), pd.Timestamp("2026-10-06T00:00:00Z")) == 0
    out = capsys.readouterr().out
    assert "bound method" not in out and "<" not in out
    assert out.splitlines()[0] == "Arsenal v Chelsea  (E0)"
    assert "40.6%" not in out  # the target is 0.50/0.27/0.23 below, not the real model's numbers
    assert " 50.0% /  27.0% /  23.0%" in out
    assert "mkt   45.0% /  28.0% /  27.0%" in out
    assert "source: live lgbd_xg 1X2 (market-anchored)" in out


def test_lookup_without_a_priced_fixture_says_dixon_coles_only(capsys):
    import argparse

    mod = _script("predict_upcoming")
    args = argparse.Namespace(home="Arsenal", away="Chelsea", json=False)
    rc = mod.lookup(args, _played(), _fixtures(), pd.DataFrame(),
                    pd.Timestamp("2026-10-06T00:00:00Z"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "no Dixon-Coles fit available" in err  # 2 played rows: no fit, never a fake number
