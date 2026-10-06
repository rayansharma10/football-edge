"""Phase 6 paper-desk tests: ledger idempotency, settlement maths, risk caps, stdout contract.

The scripts are loaded as modules (:func:`_script`) so their ``main()`` can be run end to end with
the model stage stubbed: everything except the LightGBM fit and the football-data download is the
real code path.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedge.ingest.football_data import make_match_id
from fedge.paper import fixtures as fx
from fedge.paper import ledger, model_state, picks, risk, settle

ROOT = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2027-11-01T15:00:00Z")


# ------------------------------------------------------------------ helpers
def _script(name: str):
    """Load ``scripts/<name>.py`` as a module (the trick run_strategy.py uses for run_baselines)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _played(divs=("E2",), n=2):
    """Minimal played-matches frame (only the columns the picker reads before it scores)."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "match_id": f"p{i}",
                "div": divs[i % len(divs)],
                "season": "2027/28",
                "kickoff_utc": T0 - pd.Timedelta(days=30 + i),
                "home": f"H{i}",
                "away": f"A{i}",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
            }
        )
    return pd.DataFrame(rows)


def _fixture_csv(path: Path, rows: list[dict], columns=None) -> Path:
    cols = columns or [
        "Div", "Date", "Time", "HomeTeam", "AwayTeam",
        "BFEH", "BFED", "BFEA", "BFE>2.5", "BFE<2.5",
        "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
        "MaxH", "MaxD", "MaxA", "Max>2.5", "Max<2.5",
    ]
    df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])
    df.to_csv(path, index=False)
    return path


def _future_date(days=30) -> str:
    return (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=days)).strftime("%d/%m/%Y")


def _edges(rows) -> pd.DataFrame:
    """Synthetic edge table in the shape ``model_state.score_fixtures`` returns."""
    return pd.DataFrame(rows)


def _strategy(gate0=False, shadow_threshold=0.03, wl_1x2=(), wl_ou25=()):
    return {
        "gate0_passed": gate0,
        "mode": "paper" if gate0 else "shadow",
        "markets": {
            "1x2": {"model": "lgbd_xg", "pool_weight": 0.46, "threshold": 0.06,
                    "whitelist": list(wl_1x2)},
            "ou25": {"model": "lgb_xg", "pool_weight": 0.29, "threshold": 1.0,
                     "whitelist": list(wl_ou25)},
        },
        "shadow_threshold": shadow_threshold,
    }


def _fixtures_frame(n=2, div="E2"):
    return pd.DataFrame(
        {
            "match_key": [f"k{i}" for i in range(n)],
            "div": [div] * n,
            "date": ["2027-11-01"] * n,
            "home": [f"H{i}" for i in range(n)],
            "away": [f"A{i}" for i in range(n)],
            "kickoff_utc": pd.to_datetime(
                [T0 + pd.Timedelta(days=i) for i in range(n)], utc=True
            ),
        }
    )


def _bet(match_key="k0", market="1x2", selection="H", price=4.0, stake=1.0, mode="shadow",
         kickoff=None):
    return {
        "bet_id": ledger.bet_id(match_key, market, selection),
        "created_utc": "2026-10-06T08:00:00+00:00",
        "match_key": match_key,
        "div": "E2",
        "date": "2026-10-03",
        "home": "Burton",
        "away": "Huddersfield",
        "kickoff_utc": (kickoff or T0).isoformat(),
        "market": market,
        "selection": selection,
        "model_prob": 0.3,
        "market_prob": 0.25,
        "pooled_prob": 0.3,
        "price_taken": price,
        "price_source": "BFE",
        "edge": 0.1,
        "stake_units": stake,
        "stake_frac": 0.001,
        "mode": mode,
        "config_hash": "cfg",
        "gate0_hash": "gate",
        "experiment_id": "exp",
    }


@pytest.fixture
def conn(tmp_path):
    c = ledger.connect(tmp_path / "paper.sqlite")
    yield c
    c.close()


# ------------------------------------------------------------------ ledger
def test_bet_id_is_deterministic():
    assert ledger.bet_id("k0", "1x2", "H") == ledger.bet_id("k0", "1x2", "H")
    assert ledger.bet_id("k0", "1x2", "H") != ledger.bet_id("k0", "1x2", "A")


def test_insert_bets_is_idempotent(conn):
    first = ledger.insert_bets(conn, [_bet("k0"), _bet("k1", selection="over", market="ou25")])
    assert len(first) == 2
    assert ledger.mode_counts(conn) == {"shadow": 2}
    again = ledger.insert_bets(conn, [_bet("k0"), _bet("k1", selection="over", market="ou25")])
    assert again == []
    assert len(ledger.read_table(conn, "bets")) == 2


def test_insert_bets_refuses_a_second_row_per_match_market(conn):
    ledger.insert_bets(conn, [_bet("k0", selection="H")])
    ledger.insert_bets(conn, [_bet("k0", selection="A")])  # same match+market: never double-bet
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO bets (bet_id, created_utc, match_key, div, date, home, away, "
            "kickoff_utc, market, selection, price_taken, price_source, edge, stake_units, "
            "stake_frac, mode, config_hash, gate0_hash, experiment_id) VALUES "
            "('x','t','k0','E2','d','h','a','ko','1x2','A',2,'BFE',0,1,0.001,'shadow','c','g','e')"
        )


def test_snapshots_are_idempotent_per_day_and_add_a_row_on_a_new_day(conn):
    rows = [
        {"seen_utc": "2026-10-06T08:00:00+00:00", "snapshot_day": "2026-10-06", "match_key": "k0",
         "div": "E2", "date": "2026-10-03", "home": "a", "away": "b",
         "kickoff_utc": "2026-10-03T14:00:00+00:00", "market": "1x2", "selection": "H",
         "price": 4.0, "price_source": "BFE", "model_prob": 0.3, "pooled_prob": 0.3, "edge": 0.1},
        {"seen_utc": "2026-10-06T08:00:00+00:00", "snapshot_day": "2026-10-06", "match_key": "k0",
         "div": "E2", "date": "2026-10-03", "home": "a", "away": "b",
         "kickoff_utc": "2026-10-03T14:00:00+00:00", "market": "1x2", "selection": "D",
         "price": 3.5, "price_source": "BFE", "model_prob": 0.25, "pooled_prob": 0.25, "edge": 0.0},
    ]
    assert ledger.insert_snapshots(conn, rows) == 2
    assert ledger.insert_snapshots(conn, rows) == 0
    later = [{**r, "snapshot_day": "2026-10-09", "seen_utc": "2026-10-09T08:00:00+00:00"}
             for r in rows]
    assert ledger.insert_snapshots(conn, later) == 2
    assert len(ledger.read_table(conn, "snapshots")) == 4


# ------------------------------------------------------------------ settlement maths
def test_settlement_pnl_after_commission():
    assert settle.pnl(1.0, 4.0, True, 0.06) == pytest.approx(3.0 * 0.94)
    assert settle.pnl(2.0, 4.0, True, 0.06) == pytest.approx(2 * 3.0 * 0.94)
    assert settle.pnl(1.5, 4.0, False, 0.06) == pytest.approx(-1.5)


def test_market_result_from_ftr_and_from_goals():
    assert settle.market_result("1x2", 2, 0, "H") == "H"
    assert settle.market_result("1x2", 2, 0, None) == "H"
    assert settle.market_result("1x2", 1, 1, None) == "D"
    assert settle.market_result("1x2", 0, 3, None) == "A"
    assert settle.market_result("ou25", 2, 1, "H") == "over"
    assert settle.market_result("ou25", 1, 1, "D") == "under"
    assert settle.won("ou25", "under", 1, 1, "D")


def test_settle_bet_writes_pnl_and_leaves_clv_null_without_a_closing_price():
    row = settle.settle_bet(_bet(price=4.0, stake=1.0), 2, 0, "H", "now", commission=0.06)
    assert row["result"] == "H"
    assert row["won"] == 1
    assert row["pnl"] == pytest.approx(2.82)
    assert row["closing_price"] is None and row["log_clv"] is None
    lost = settle.settle_bet(
        _bet(selection="A", price=2.0, stake=1.0), 2, 0, "H", "now", commission=0.06
    )
    assert (lost["won"], lost["result"], lost["pnl"]) == (0, "H", -1.0)


def test_settle_bet_log_clv_uses_the_fair_closing_probability():
    close = {"price": 3.2, "source": "BFE", "p_fair": 0.3}
    row = settle.settle_bet(_bet(price=4.0), 2, 0, "H", "now", commission=0.06, close=close)
    assert row["closing_price"] == 3.2
    assert row["closing_source"] == "BFE"
    assert row["log_clv"] == pytest.approx(float(np.log(4.0 * 0.3)))


def _odds_long() -> pd.DataFrame:
    rows = []
    for book, market, sel, price in (
        ("BFE", "1x2", "H", 3.5), ("BFE", "1x2", "D", 3.4), ("BFE", "1x2", "A", 2.1),
        ("Avg", "1x2", "H", 3.4), ("Avg", "1x2", "D", 3.3), ("Avg", "1x2", "A", 2.0),
        ("BFE", "ou25", "over", 1.9), ("BFE", "ou25", "under", 1.95),
        ("BFE", "1x2", "H", 3.3), ("BFE", "1x2", "D", 3.3),
    ):
        rows.append(
            {"match_id": "m1", "bookmaker": book, "market": market, "selection": sel,
             "phase": "close", "price": price}
        )
    return pd.DataFrame(rows)


def test_closing_probabilities_prefer_bfe_and_fall_back_to_avg():
    odds = _odds_long()
    probs = settle.closing_probabilities(odds, "1x2")
    assert list(probs.loc["m1", ["H", "D", "A"]].index) == ["H", "D", "A"]
    assert probs.loc["m1", "source"] == "BFE"
    assert probs.loc["m1", ["H", "D", "A"]].sum() == pytest.approx(1.0)
    # drop one BFE leg: the complete Avg book must take over
    partial = odds[~((odds["bookmaker"] == "BFE") & (odds["selection"] == "A"))]
    assert settle.closing_probabilities(partial, "1x2").loc["m1", "source"] == "Avg"
    # no complete book at all -> no row
    none = odds[odds["market"] == "ou25"]
    assert settle.closing_probabilities(none[none["bookmaker"] == "Max"], "ou25").empty


# ------------------------------------------------------------------ risk
def test_stake_is_capped_and_zero_edge_stakes_nothing():
    limits = risk.Limits(max_stake_frac=0.01, kelly_fraction=0.25)
    assert risk.stake_frac(0.6, 2.0, limits) == pytest.approx(0.01)  # full Kelly capped at 1%
    assert risk.stake_frac(0.4, 2.0, limits) == 0.0  # no edge (p < 1/price)
    frac, units = risk.stake_units(0.6, 2.0, limits, bankroll=1000.0)
    assert 0 < frac <= 0.01 and units == pytest.approx(frac * 1000.0)
    # a thin edge is scaled by the Kelly fraction rather than capped
    assert risk.stake_frac(0.53, 2.0, limits) == pytest.approx(
        0.25 * (0.53 * 0.94 - 0.47) / 0.94
    )


def test_limits_refuse_live_true(tmp_path):
    p = tmp_path / "limits.toml"
    p.write_text("LIVE = true\ncommission = 0.06\nmax_stake_frac = 0.01\n"
                 "kelly_fraction = 0.25\nmax_bets_per_day = 20\ndrawdown_kill = 0.2\n")
    with pytest.raises(RuntimeError, match="LIVE = true"):
        risk.load_limits(p)


def test_kill_switch_file_blocks_the_desk(conn, tmp_path):
    limits = risk.Limits()
    assert risk.daily_state(conn, limits, tmp_path)["remaining"] == limits.max_bets_per_day
    risk.kill_path(tmp_path).write_text("stop\n")
    assert risk.kill_switch_active(tmp_path)
    state = risk.daily_state(conn, limits, tmp_path)
    assert state["blocked"] and state["remaining"] == 0
    rows = picks.plan_bets(
        _edges([dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                     model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.1)]),
        _fixtures_frame(1),
        _strategy(),
        limits,
        state["remaining"],
        pd.Timestamp("2026-10-06T08:00:00Z"),
        "cfg",
        "gate",
        "exp",
    )
    assert rows == []


def test_realised_drawdown_blocks_the_desk(conn, tmp_path):
    limits = risk.Limits(drawdown_kill=0.2)
    ledger.insert_bets(conn, [_bet("k0", price=2.0, stake=10.0, mode="paper")])
    ledger.record_settlements(
        conn,
        [{"bet_id": ledger.bet_id("k0", "1x2", "H"), "settled_utc": "t", "result": "A", "won": 0,
          "pnl": -300.0, "closing_price": None, "closing_source": None, "log_clv": None}],
    )
    assert risk.realised_drawdown(ledger.settled_frame(conn)) == pytest.approx(0.3)
    state = risk.daily_state(conn, limits, tmp_path)
    assert state["blocked"] and "drawdown" in state["reason"] and state["remaining"] == 0


# ------------------------------------------------------------------ fixtures parsing
def test_fixture_parse_converts_uk_local_time_and_builds_the_ingest_match_key(tmp_path):
    p = _fixture_csv(
        tmp_path / "fixtures.csv",
        [{"Div": "E2", "Date": "01/11/2027", "Time": "15:00", "HomeTeam": "Burton",
          "AwayTeam": "Huddersfield", "BFEH": 4.0, "BFED": 3.95, "BFEA": 1.97,
          "BFE>2.5": 1.87, "BFE<2.5": 1.98}],
    )
    raw = fx.read_fixture_csv(p)
    fix = fx.parse_fixtures(raw)
    assert len(fix) == 1
    assert fix.loc[0, "kickoff_utc"].isoformat() == "2027-11-01T15:00:00+00:00"  # GMT in November
    assert fix.loc[0, "match_key"] == make_match_id("E2", "2027-11-01", "Burton", "Huddersfield")
    rows = fx.market_rows(raw, fix, phase="pre")
    assert len(rows) == 5  # 3 x 1x2 + 2 x ou25
    assert set(rows["price_source"]) == {"BFE"}
    assert rows[rows["selection"] == "H"]["price"].iloc[0] == 4.0


def test_price_source_priority_bfe_then_avg_then_max(tmp_path):
    def row(i, **odds):
        return {"Div": "E2", "Date": "01/11/2027", "Time": "15:00", "HomeTeam": f"H{i}",
                "AwayTeam": f"A{i}", **odds}

    p = _fixture_csv(
        tmp_path / "fixtures.csv",
        [
            row(0, BFEH=4.0, BFED=3.9, BFEA=2.0),          # BFE complete
            row(1, AvgH=4.1, AvgD=3.9, AvgA=2.0),          # only Avg complete
            row(2, MaxH=4.2, MaxD=3.9, MaxA=2.0),          # only Max complete
            row(3, BFEH=4.0, BFED=3.9, AvgH=4.1, AvgD=3.9, AvgA=2.0),  # BFE incomplete
        ],
    )
    raw = fx.read_fixture_csv(p)
    fix = fx.parse_fixtures(raw)
    rows = fx.market_rows(raw, fix, phase="pre", markets=("1x2",))
    got = [rows[rows["match_key"] == k]["price_source"].iloc[0] for k in fix["match_key"]]
    assert got == ["BFE", "Avg", "Max", "Avg"]  # incomplete BFE falls back to the complete Avg book


def test_price_column_names_for_the_closing_phase():
    assert fx.column("BFE", "1x2", "H", "close") == "BFECH"
    assert fx.column("BFE", "ou25", "over", "close") == "BFEC>2.5"
    assert fx.column("Avg", "1x2", "A", "close") == "AvgCA"
    assert fx.column("Max", "ou25", "under", "pre") == "Max<2.5"


# ------------------------------------------------------------------ selection rules
def test_shadow_mode_uses_the_shadow_threshold_and_every_division():
    edges = _edges(
        [
            dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.05),
            dict(match_key="k0", market="1x2", selection="A", price=2.1, price_source="BFE",
                 model_prob=0.45, market_prob=0.45, pooled_prob=0.45, edge=-0.05),
            dict(match_key="k1", market="ou25", selection="over", price=2.5, price_source="BFE",
                 model_prob=0.4, market_prob=0.38, pooled_prob=0.4, edge=0.02),
        ]
    )
    cand = picks.candidate_bets(
        edges, _fixtures_frame(2), _strategy(shadow_threshold=0.03), "shadow"
    )
    assert list(cand["match_key"]) == ["k0"]  # only the 5% edge clears 3%; the 2% one does not
    assert list(cand["selection"]) == ["H"]


def test_gate0_mode_uses_the_per_market_threshold_and_the_whitelist():
    edges = _edges(
        [
            dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.07),
            dict(match_key="k1", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.07),
        ]
    )
    f = _fixtures_frame(2)
    f.loc[1, "div"] = "D1"
    cand = picks.candidate_bets(edges, f, _strategy(gate0=True, wl_1x2=("E2",)), "paper")
    assert list(cand["match_key"]) == ["k0"]  # D1 is not whitelisted
    out = picks.plan_bets(edges, f, _strategy(gate0=True, wl_1x2=("E2",)), risk.Limits(), 5,
                          pd.Timestamp("2026-10-06T08:00:00Z"), "cfg", "gate", "exp")
    assert [b["mode"] for b in out] == ["paper"]


def test_daily_cap_keeps_the_highest_edges_first():
    rows = [
        dict(match_key=f"k{i}", market="1x2", selection="H", price=4.0, price_source="BFE",
             model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.04 + i / 100)
        for i in range(5)
    ]
    out = picks.plan_bets(
        _edges(rows), _fixtures_frame(5), _strategy(), risk.Limits(), 2,
        pd.Timestamp("2026-10-06T08:00:00Z"), "cfg", "gate", "exp",
    )
    assert len(out) == 2
    assert [b["match_key"] for b in out] == ["k4", "k3"]
    assert all(b["stake_frac"] <= 0.01 and b["mode"] == "shadow" for b in out)
    assert all(b["config_hash"] == "cfg" and b["gate0_hash"] == "gate" for b in out)


def test_digest_lists_sydney_kickoff_and_edge():
    bets = picks.plan_bets(
        _edges([dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                     model_prob=0.5, market_prob=0.4, pooled_prob=0.5, edge=0.9)]),
        _fixtures_frame(1), _strategy(), risk.Limits(), 5,
        pd.Timestamp("2026-10-06T08:00:00Z"), "cfg", "gate", "exp",
    )
    text = picks.format_digest(bets, _strategy(), pd.Timestamp("2026-10-06T08:00:00Z"))
    lines = text.splitlines()
    assert lines[0].startswith("fedge shadow picks") and "1 new bet" in lines[0]
    assert "E2 | H0 v A0 | H @ 4.00 (BFE)" in lines[1]
    assert "edge +90.0%" in lines[1] and "u" in lines[1]
    # 2027-11-01 15:00 UTC = 2027-11-02 02:00 AEDT
    assert "Tue 02 Nov 02:00 AEDT" in lines[1]


# ------------------------------------------------------------------ end to end (model stubbed)
@pytest.fixture
def pick_env(tmp_path, monkeypatch):
    """A temp repo-ish environment: strategy/limits config, fixture CSVs, stubbed model stage."""
    data = tmp_path / "data"
    data.mkdir()
    cfg = tmp_path / "strategy.toml"
    cfg.write_text(
        "gate0_passed = false\nmode = \"shadow\"\ncommission = 0.06\nkelly_fraction = 0.1\n"
        "[markets.1x2]\nmodel = \"lgbd_xg\"\npool_weight = 0.46\nthreshold = 0.06\nwhitelist = []\n"
        "[markets.ou25]\nmodel = \"lgb_xg\"\npool_weight = 0.29\nthreshold = 1.0\nwhitelist = []\n"
        "[shadow]\nshadow_threshold = 0.03\n"
    )
    limits = tmp_path / "limits.toml"
    limits.write_text(
        "LIVE = false\ncommission = 0.06\nmax_stake_frac = 0.01\nkelly_fraction = 0.25\n"
        "max_bets_per_day = 20\ndrawdown_kill = 0.2\n"
    )
    gates = tmp_path / "gate0.md"
    gates.write_text("# gate 0\nNOT PASSED\n")
    date = _future_date()
    fixcsv = _fixture_csv(
        tmp_path / "fixtures.csv",
        [
            {"Div": "E2", "Date": date, "Time": "15:00", "HomeTeam": "Burton",
             "AwayTeam": "Huddersfield", "BFEH": 4.0, "BFED": 3.95, "BFEA": 1.97,
             "BFE>2.5": 1.87, "BFE<2.5": 1.98},
            {"Div": "SC3", "Date": date, "Time": "15:00", "HomeTeam": "Annan",
             "AwayTeam": "Stirling", "BFEH": 1.6, "BFED": 3.9, "BFEA": 4.2},
        ],
    )
    newcsv = _fixture_csv(
        tmp_path / "new_league_fixtures.csv",
        [{"Country": "Argentina", "League": "Liga Profesional", "Date": date, "Time": "20:00",
          "Home": "Boca", "Away": "Union", "BFEH": 1.6, "BFED": 4.0, "BFEA": 7.0}],
        columns=["Country", "League", "Date", "Time", "Home", "Away", "BFEH", "BFED", "BFEA"],
    )
    mod = _script("paper_picks")
    monkeypatch.setattr(mod.fx, "fetch_fixture_files", lambda *a, **k: (fixcsv, newcsv))
    monkeypatch.setattr(mod.model_state, "refresh_ingest", lambda *a, **k: None)
    monkeypatch.setattr(mod.model_state, "played_matches", lambda *a, **k: _played())
    monkeypatch.setattr(mod.model_state, "score_all", _fake_score_all)
    argv = [
        "--data-dir", str(data), "--config", str(cfg), "--limits", str(limits),
        "--gates", str(gates), "--no-refresh",
    ]
    return mod, argv, data


def _fake_score_all(data_dir, strategy, played, fixtures, price_rows, weights, threads=4):
    """Deterministic stand-in for the model: a 9% edge on the home side, nothing elsewhere."""
    rows = []
    for r in price_rows.itertuples(index=False):
        edge = 0.09 if r.selection in ("H", "over") else -0.09
        rows.append(
            dict(match_key=r.match_key, market=r.market, selection=r.selection, price=r.price,
                 price_source=r.price_source, model_prob=0.5, market_prob=0.45,
                 pooled_prob=0.5, edge=edge)
        )
    return pd.DataFrame(rows)


def test_picks_prints_the_digest_only_when_it_places_new_bets(pick_env, capsys):
    mod, argv, data = pick_env
    assert mod.main(argv) == 0
    out, err = capsys.readouterr()
    lines = out.strip().splitlines()
    assert lines[0].startswith("fedge shadow picks") and "1 new bet" in lines[0]
    assert "Burton v Huddersfield" in lines[1]
    assert "shadow" in lines[1] or "Burton" in lines[1]
    assert "recorded" in err and "unmodelled_divs=['SC3']" in err
    conn = ledger.connect(ledger.default_path(data))
    bets = ledger.read_table(conn, "bets")
    snaps = ledger.read_table(conn, "snapshots")
    assert len(bets) == 1 and bets.loc[0, "mode"] == "shadow"
    assert bets.loc[0, "gate0_hash"] == picks.file_hash(Path(argv[argv.index("--gates") + 1]))
    assert len(snaps) == 5  # the modelled fixture's three 1x2 and two ou25 prices

    # a second run on the same fixture file must be silent and write nothing
    assert mod.main(argv) == 0
    out2, _ = capsys.readouterr()
    assert out2 == ""
    assert len(ledger.read_table(conn, "bets")) == 1
    assert len(ledger.read_table(conn, "snapshots")) == 5
    conn.close()


def test_picks_is_silent_when_the_kill_switch_is_set(pick_env, capsys):
    mod, argv, data = pick_env
    risk.kill_path(data).write_text("stop\n")
    assert mod.main(argv) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "kill switch" in err
    conn = ledger.connect(ledger.default_path(data))
    assert len(ledger.read_table(conn, "bets")) == 0
    assert len(ledger.read_table(conn, "snapshots")) == 5  # snapshots are not bets
    conn.close()


# ------------------------------------------------------------------ settle + weekly scripts
@pytest.fixture
def settle_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    limits = tmp_path / "limits.toml"
    limits.write_text(
        "LIVE = false\ncommission = 0.06\nmax_stake_frac = 0.01\nkelly_fraction = 0.25\n"
        "max_bets_per_day = 20\ndrawdown_kill = 0.2\n"
    )
    conn = ledger.connect(ledger.default_path(data))
    mkey = make_match_id("E2", "2026-10-03", "Burton", "Huddersfield")
    ledger.insert_bets(
        conn,
        [_bet(mkey, price=4.0, stake=1.0, kickoff=pd.Timestamp("2026-10-03T14:00:00Z"))],
    )
    conn.close()
    mod = _script("paper_settle")
    monkeypatch.setattr(mod.model_state, "refresh_ingest", lambda *a, **k: None)
    monkeypatch.setattr(
        mod.model_state,
        "played_matches",
        lambda *a, **k: pd.DataFrame(
            [{"match_id": mkey, "FTHG": 2, "FTAG": 0, "FTR": "H"}]
        ),
    )
    monkeypatch.setattr(
        mod,
        "load_odds",
        lambda *a, **k: pd.DataFrame(
            [{"match_id": mkey, "bookmaker": b, "market": "1x2", "selection": s, "phase": "close",
              "price": p}
             for b, s, p in (("BFE", "H", 3.2), ("BFE", "D", 3.4), ("BFE", "A", 2.6))]
        ),
    )
    return mod, ["--data-dir", str(data), "--limits", str(limits), "--no-refresh"], data


def test_settle_digest_then_silence(settle_env, capsys):
    mod, argv, data = settle_env
    assert mod.main(argv) == 0
    out, err = capsys.readouterr()
    lines = out.strip().splitlines()
    assert lines[0].startswith("fedge settled") and "1 new settlement" in lines[0]
    assert "WON" in lines[1] and "+2.82u" in lines[1]
    assert lines[-1].strip().startswith("totals: n=1 settled")
    assert "settled 1 of 1" in err
    assert mod.main(argv) == 0
    out2, _ = capsys.readouterr()
    assert out2 == ""
    conn = ledger.connect(ledger.default_path(data))
    assert len(ledger.read_table(conn, "settlements")) == 1
    conn.close()


def test_settle_reports_nothing_when_the_match_has_no_result_yet(settle_env, monkeypatch, capsys):
    mod, argv, data = settle_env
    monkeypatch.setattr(mod.model_state, "played_matches", lambda *a, **k: pd.DataFrame(
        columns=["match_id", "FTHG", "FTAG", "FTR"]
    ))
    assert mod.main(argv) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "nothing newly settled" in err


def test_weekly_report_states_gate0_failure_and_gate1_progress(tmp_path, capsys, monkeypatch):
    mod = _script("paper_weekly")
    data = tmp_path / "data"
    data.mkdir()
    cfg = tmp_path / "strategy.toml"
    cfg.write_text(
        "gate0_passed = false\nmode = \"shadow\"\ncommission = 0.06\nkelly_fraction = 0.1\n"
        "[markets.1x2]\nmodel = \"lgbd_xg\"\npool_weight = 0.46\nthreshold = 0.06\nwhitelist = []\n"
        "[markets.ou25]\nmodel = \"lgb_xg\"\npool_weight = 0.29\nthreshold = 1.0\nwhitelist = []\n"
        "[shadow]\nshadow_threshold = 0.03\n"
    )
    limits = tmp_path / "limits.toml"
    limits.write_text(
        "LIVE = false\ncommission = 0.06\nmax_stake_frac = 0.01\nkelly_fraction = 0.25\n"
        "max_bets_per_day = 20\ndrawdown_kill = 0.2\n"
    )
    conn = ledger.connect(ledger.default_path(data))
    ledger.insert_bets(conn, [_bet("k0", mode="shadow"), _bet("k1", mode="paper", selection="A")])
    ledger.record_settlements(conn, [
        {"bet_id": ledger.bet_id("k0", "1x2", "H"), "settled_utc": "t", "result": "A", "won": 0,
         "pnl": -1.0, "closing_price": 3.4, "closing_source": "BFE", "log_clv": -0.05},
    ])
    conn.close()
    reports = tmp_path / "weekly"
    argv = ["--data-dir", str(data), "--reports", str(reports), "--config", str(cfg),
            "--limits", str(limits), "--now", "2026-10-06T08:00:00Z"]
    assert mod.main(argv) == 0
    out, _ = capsys.readouterr()
    assert "Gate 0 NOT passed" in out and "Gate 1 0/1000" in out
    report = reports / "2026-10-06.md"
    text = report.read_text(encoding="utf-8")
    assert "**Gate 0 status: NOT passed**" in text
    assert "shadow bets do **not** count towards Gate 1" in text.replace("Shadow", "shadow")
    assert "Settled **paper** bets: 0 / 1000" in text
    assert "Settled shadow bets (do not count): 1" in text
    assert "n=1" in text  # one settled bet, mean log-CLV -5.00% (no CI below 30)


def test_fixture_features_accept_nullable_int_goals(tmp_path):
    """Regression: the ingest writes Int64 goal columns, which the rating sweeps cannot cast."""
    played = _played(("E2",), n=2)
    played["FTHG"] = played["FTHG"].astype("Int64")
    played["FTAG"] = played["FTAG"].astype("Int64")
    fix = pd.DataFrame(
        [{"match_key": "k0", "div": "E2", "date": "2027-11-01", "home": "H0", "away": "A1",
          "kickoff_utc": pd.Timestamp("2027-11-01T15:00:00Z")}]
    )
    x = model_state.fixture_features(played, fix, model_state.div_code_map(played))
    assert list(x.index) == ["k0"]
    assert x.loc["k0", "h_games"] == 1
    assert np.isfinite(x.loc["k0", "elo_diff"])


def test_training_table_and_fixture_features_shapes(tmp_path):
    """The live-model plumbing keeps the Phase 4 column contract (no model fit involved)."""
    played = _played(("E2", "E0"), n=4)
    played["div"] = ["E2", "E2", "E0", "E0"]
    feats = model_state.build_features(played, tmp_path, xg_path=tmp_path / "nope.parquet")
    assert {"elo_diff", "pi_diff", "ad_lam_h", "h_gf", "h_rest", "h_promoted", "div_code"} <= set(
        feats.columns
    )
    codes = model_state.div_code_map(played)
    assert codes == {"E0": 0, "E2": 1}
    fix = pd.DataFrame(
        [{"match_key": "k0", "div": "E2", "date": "2027-11-01", "home": "H0", "away": "A1",
          "kickoff_utc": pd.Timestamp("2027-11-01T15:00:00Z")}]
    )
    x = model_state.fixture_features(played, fix, codes)
    assert list(x.index) == ["k0"]
    assert set(feats.columns) - set(x.columns) == set()
    assert x.loc["k0", "div_code"] == 1
    assert x.loc["k0", "h_games"] == 1  # only the played E2 match of H0 is before this fixture
    assert x.loc["k0", "h_rest"] == 21.0  # capped at REST_CAP
    # a division the model has no history for is skipped rather than invented
    unknown = fix.assign(div="SC3", match_key="k1")
    assert model_state.fixture_features(played, unknown, codes).empty
    assert model_state.season_of(pd.Timestamp("2027-06-30T12:00:00Z")) == "2026/27"
    assert model_state.season_of(pd.Timestamp("2027-07-01T12:00:00Z")) == "2027/28"
    assert model_state.season_of(pd.Timestamp("2027-11-01T12:00:00Z")) == "2027/28"
