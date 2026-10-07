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

from fedge.backtest.stats import net_odds
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
    # log_clv is NET of commission (the gate variable); log_clv_raw is the pre-commission value
    assert row["log_clv_raw"] == pytest.approx(float(np.log(4.0 * 0.3)))
    assert row["log_clv"] == pytest.approx(float(np.log(net_odds(4.0, 0.06) * 0.3)))


def test_settle_bet_net_clv_is_the_raw_clv_shifted_by_the_commission():
    """Pin the relation the migration and the desk's gate both rely on."""
    for price in (1.5, 2.0, 2.76, 4.0, 6.0):
        close = {"price": price, "source": "BFE", "p_fair": 0.3}
        row = settle.settle_bet(
            _bet(price=price), 2, 0, "H", "now", commission=0.06, close=close
        )
        assert row["log_clv"] == pytest.approx(
            row["log_clv_raw"] + float(np.log(net_odds(price, 0.06) / price))
        )
        assert row["log_clv"] < row["log_clv_raw"]  # raw is the optimistic one


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


def test_price_source_priority_bfe_then_avg_never_max(tmp_path):
    def row(i, **odds):
        return {"Div": "E2", "Date": "01/11/2027", "Time": "15:00", "HomeTeam": f"H{i}",
                "AwayTeam": f"A{i}", **odds}

    p = _fixture_csv(
        tmp_path / "fixtures.csv",
        [
            row(0, BFEH=4.0, BFED=3.9, BFEA=2.0),          # BFE complete
            row(1, AvgH=4.1, AvgD=3.9, AvgA=2.0),          # only Avg complete
            row(2, MaxH=4.2, MaxD=3.9, MaxA=2.0),          # only Max complete: never used
            row(3, BFEH=4.0, BFED=3.9, AvgH=4.1, AvgD=3.9, AvgA=2.0),  # BFE incomplete
        ],
    )
    raw = fx.read_fixture_csv(p)
    fix = fx.parse_fixtures(raw)
    rows = fx.market_rows(raw, fix, phase="pre", markets=("1x2",))
    got = [rows.loc[rows["match_key"] == k, "price_source"].tolist() for k in fix["match_key"]]
    # incomplete BFE falls back to the complete Avg book; a Max-only fixture gets no price
    assert got == [["BFE"] * 3, ["Avg"] * 3, [], ["Avg"] * 3]


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


def test_gate0_whitelist_accepts_the_rendered_pair_format(tmp_path):
    """``render_strategy_toml`` writes ``[["E0", "1x2"], ...]``: pairs, not flat division codes."""
    from fedge.backtest import strategy as S

    txt = S.render_strategy_toml(
        {
            "comments": [],
            "top": {"gate0_passed": True, "mode": "paper"},
            "markets": {
                "1x2": {"model": "lgbd_xg", "pool_weight": 0.46, "threshold": 0.06,
                        "whitelist": ["E2", "D1"]},
                "ou25": {"model": "lgb_xg", "pool_weight": 0.29, "threshold": 0.04,
                         "whitelist": ["D1"]},
            },
            "shadow": {"shadow_threshold": 0.03},
        }
    )
    path = tmp_path / "strategy.toml"
    path.write_text(txt, encoding="utf-8")
    cfg = picks.load_strategy(path)  # the loader the desk really uses
    assert cfg["mode"] == "paper" and cfg["gate0_passed"] is True
    assert picks.market_whitelist(cfg, "1x2", "paper") == ["E2", "D1"]
    assert picks.market_whitelist(cfg, "ou25", "paper") == ["D1"]  # E2 only whitelists 1x2
    assert picks.market_whitelist(cfg, "1x2", "shadow") == []  # shadow opens every division
    assert picks.market_threshold(cfg, "ou25", "paper") == pytest.approx(0.04)

    edges = _edges(
        [
            dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.07),
            dict(match_key="k1", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.07),
        ]
    )
    f = _fixtures_frame(2)
    f.loc[1, "div"] = "SC3"  # not whitelisted for 1x2
    cand = picks.candidate_bets(edges, f, cfg, "paper")
    assert list(cand["match_key"]) == ["k0"]


def test_repo_strategy_toml_keeps_the_gate0_fields_and_adds_the_shadow_table():
    """The committed config the desk reads: shadow bar explicit, Gate 0 fields untouched."""
    text = (ROOT / "config" / "strategy.toml").read_text(encoding="utf-8")
    assert "[shadow]" in text  # required by the P6 card: an explicit, reviewable shadow bar
    cfg = picks.load_strategy(ROOT / "config" / "strategy.toml")
    assert cfg["shadow_threshold"] == pytest.approx(0.03)
    assert cfg["mode"] in ("paper", "shadow") and cfg["gate0_passed"] is (cfg["mode"] == "paper")
    # shadow mode ignores the Gate 0 per-market bar and opens every division for both markets
    assert picks.market_threshold(cfg, "1x2", "shadow") == pytest.approx(0.03)
    assert picks.market_whitelist(cfg, "ou25", "shadow") == []


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
    # the desk scores only the divisions named in the leagues config (config/leagues.toml
    # desk_divisions; E0 + SP1 in production), so this test declares its own E2 desk
    leagues = _leagues_toml(tmp_path / "leagues.toml", ["E2"])
    argv = [
        "--data-dir", str(data), "--config", str(cfg), "--limits", str(limits),
        "--gates", str(gates), "--leagues", str(leagues), "--no-refresh",
    ]
    return mod, argv, data


def _leagues_toml(path: Path, desk: list[str]) -> Path:
    items = ", ".join(f'"{d}"' for d in desk)
    path.write_text(f'divisions = ["E0", "SP1"]\ndesk_divisions = [{items}]\n')
    return path


def test_picks_ignores_fixtures_outside_the_desk_divisions(pick_env, capsys):
    """A fixture in a league the desk does not cover is not scored and not bet."""
    mod, argv, data = pick_env
    leagues = _leagues_toml(Path(argv[argv.index("--leagues") + 1]).with_name("other.toml"), ["E0"])
    assert mod.main([*argv, "--leagues", str(leagues)]) == 0
    out, err = capsys.readouterr()
    assert out == ""  # nothing bet, so the stdout digest contract stays silent
    assert "modelled_fixtures=0" in err and "unmodelled_divs=['E2', 'SC3']" in err
    conn = ledger.connect(ledger.default_path(data))
    assert len(ledger.read_table(conn, "bets")) == 0
    conn.close()


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


def test_picks_reads_a_fresh_clock_after_scoring(pick_env, monkeypatch, capsys):
    """M1: the kickoff re-check and created_utc use the time after score_all, not script start."""
    mod, argv, data = pick_env
    start = pd.Timestamp.now(tz="UTC")
    clock = {"now": start}
    monkeypatch.setattr(mod, "_utc_now", lambda: clock["now"])

    def slow_score(*a, **k):
        clock["now"] = start + pd.Timedelta(hours=1)  # the fit "took an hour"
        return _fake_score_all(*a, **k)

    monkeypatch.setattr(mod.model_state, "score_all", slow_score)
    assert mod.main(argv) == 0
    conn = ledger.connect(ledger.default_path(data))
    bets = ledger.read_table(conn, "bets")
    snaps = ledger.read_table(conn, "snapshots")
    assert len(bets) == 1
    assert bets.loc[0, "created_utc"] == picks._iso(start + pd.Timedelta(hours=1))
    assert set(snaps["seen_utc"]) == {picks._iso(start + pd.Timedelta(hours=1))}
    conn.close()


def test_picks_rejects_a_fixture_that_kicked_off_while_scoring(pick_env, monkeypatch, capsys):
    mod, argv, data = pick_env
    start = pd.Timestamp.now(tz="UTC")
    clock = {"now": start}
    monkeypatch.setattr(mod, "_utc_now", lambda: clock["now"])

    def slow_score(*a, **k):
        clock["now"] = start + pd.Timedelta(days=45)  # fixtures are ~30 days out: kicked off
        return _fake_score_all(*a, **k)

    monkeypatch.setattr(mod.model_state, "score_all", slow_score)
    assert mod.main(argv) == 0
    conn = ledger.connect(ledger.default_path(data))
    assert len(ledger.read_table(conn, "bets")) == 0
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


# ------------------------------------------------- P6 follow-ups from review_p6 (M2, m1, m2, m9)
def test_paper_mode_empty_whitelist_selects_nothing_not_everything():
    """The committed in-fold shape: qualified=false, threshold=0.06, whitelist=[].

    reports/gate0_fold_configs.csv has exactly this for the exchange folds 2022/23..2026/27. The
    backtest reads an empty whitelist as "no division" (strategy.py: fold["div"].isin(cfg[...])),
    so paper mode must too - never as "every division".
    """
    strategy = _strategy(gate0=True)  # gate0=True -> mode "paper", whitelists empty
    for market in strategy["markets"].values():
        market["qualified"] = False
    edges = _edges(
        [
            dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.3, market_prob=0.25, pooled_prob=0.3, edge=0.07),
            dict(match_key="k1", market="ou25", selection="over", price=2.5, price_source="BFE",
                 model_prob=0.4, market_prob=0.37, pooled_prob=0.4, edge=0.07),
        ]
    )
    cand = picks.candidate_bets(edges, _fixtures_frame(2), strategy, "paper")
    assert cand.empty
    bets = picks.plan_bets(
        edges, _fixtures_frame(2), strategy, risk.Limits(), 5,
        pd.Timestamp("2026-10-06T08:00:00Z"), "cfg", "gate", "exp",
    )
    assert bets == []


def test_plan_bets_never_bets_a_fixture_that_has_already_kicked_off():
    """The model fits take minutes, so the kickoff is re-checked at insert time (m2)."""
    edges = _edges(
        [
            dict(match_key="k0", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.5, market_prob=0.4, pooled_prob=0.5, edge=0.9),
            dict(match_key="k1", market="1x2", selection="H", price=4.0, price_source="BFE",
                 model_prob=0.5, market_prob=0.4, pooled_prob=0.5, edge=0.9),
        ]
    )
    f = _fixtures_frame(2)  # kickoffs T0, T0+1d (2027-11-01/02)
    now = T0 + pd.Timedelta(minutes=5)  # after k0's kickoff, before k1's
    out = picks.plan_bets(edges, f, _strategy(), risk.Limits(), 5, now, "cfg", "gate", "exp")
    assert [b["match_key"] for b in out] == ["k1"]


def test_settle_matches_a_moved_fixture_by_division_and_teams():
    """A rescheduled fixture keeps its teams but changes match_id: settle via the +/-3d fallback."""
    settle_script = _script("paper_settle")
    bet = {
        "match_key": make_match_id("SP2", "03/10/2026", "Sabadell", "Andorra"),
        "div": "SP2",
        "home": "Sabadell",
        "away": "Andorra",
        "kickoff_utc": "2026-10-03T16:30:00+00:00",
    }
    moved = make_match_id("SP2", "04/10/2026", "Sabadell", "Andorra")
    played = pd.DataFrame(
        [
            {"match_id": moved, "div": "SP2", "home": "Sabadell", "away": "Andorra",
             "kickoff_utc": pd.Timestamp("2026-10-04T19:00:00Z"), "FTHG": 2.0, "FTAG": 1.0,
             "FTR": "H"},
            {"match_id": make_match_id("E0", "04/10/2026", "Arsenal", "Chelsea"), "div": "E0",
             "home": "Arsenal", "away": "Chelsea",
             "kickoff_utc": pd.Timestamp("2026-10-04T14:00:00Z"), "FTHG": 1.0, "FTAG": 0.0,
             "FTR": "H"},
        ]
    )
    results = settle_script.load_results(played)
    found = settle_script.match_result(results, played, bet)
    assert found is not None
    res, mid = found
    assert mid == moved and float(res["FTHG"]) == 2.0

    # more than 3 days away (a genuinely different fixture): no match, so nothing settles
    far = dict(bet, kickoff_utc="2026-10-15T16:30:00+00:00")
    assert settle_script.match_result(results, played, far) is None


def test_settle_ignores_a_nonsense_closing_price():
    """One bad closing quote must not abort the settlement run (m8): settle without CLV."""
    settle_script = _script("paper_settle")
    data = pd.DataFrame(
        [
            {"match_id": "m1", "bookmaker": "BFE", "market": "1x2", "selection": "H",
             "phase": "close", "price": 0.0},
            {"match_id": "m1", "bookmaker": "BFE", "market": "1x2", "selection": "D",
             "phase": "close", "price": 3.3},
            {"match_id": "m1", "bookmaker": "BFE", "market": "1x2", "selection": "A",
             "phase": "close", "price": 2.2},
        ]
    )
    cache = settle_script.close_lookup(data)
    assert settle_script.closing_info(cache, "1x2", "m1", "H") is None


def test_load_strategy_refuses_paper_mode_without_a_passed_gate0(tmp_path):
    """m6: paper bets count towards Gate 1, so mode=paper requires gate0_passed = true."""
    p = tmp_path / "s.toml"
    p.write_text(
        "gate0_passed = false\nmode = \"paper\"\n"
        "[markets.1x2]\nmodel = \"m\"\npool_weight = 0.5\nthreshold = 0.06\nwhitelist = []\n"
        "[markets.ou25]\nmodel = \"m\"\npool_weight = 0.5\nthreshold = 0.06\nwhitelist = []\n"
    )
    with pytest.raises(ValueError, match="gate0_passed"):
        picks.load_strategy(p)
    p.write_text(p.read_text().replace("gate0_passed = false", "gate0_passed = true"))
    assert picks.load_strategy(p)["mode"] == "paper"
    # an out-of-range threshold or a bad mode is refused too
    p.write_text(p.read_text().replace("threshold = 0.06", "threshold = 1.5", 1))
    with pytest.raises(ValueError, match="threshold"):
        picks.load_strategy(p)


def test_load_limits_refuses_out_of_range_caps(tmp_path):
    """m6: a mis-typed cap must stop the desk, not silently size stakes."""
    p = tmp_path / "limits.toml"
    p.write_text(
        "LIVE = false\ncommission = 0.06\nmax_stake_frac = 1.5\nkelly_fraction = 0.25\n"
        "max_bets_per_day = 20\ndrawdown_kill = 0.2\n"
    )
    with pytest.raises(ValueError, match="max_stake_frac"):
        risk.load_limits(p)


def test_shadow_mode_drawdown_does_not_block_the_desk(tmp_path):
    """m3: shadow P&L is hypothetical, so it must not trip the paper kill switch."""
    conn = ledger.connect(tmp_path / "paper.sqlite")
    try:
        bets = [
            _bet(match_key=f"k{i}", price=2.0, stake=10.0, mode="shadow",
                 kickoff=T0 + pd.Timedelta(days=i)) for i in range(3)
        ]
        ledger.insert_bets(conn, bets)
        ledger.record_settlements(conn, [
            {"bet_id": b["bet_id"], "settled_utc": "t", "result": "A", "won": 0, "pnl": -300.0,
             "closing_price": None, "closing_source": None, "log_clv": None} for b in bets
        ])
        limits = risk.Limits(drawdown_kill=0.2)
        shadow_state = risk.daily_state(conn, limits, tmp_path, today="2027-11-01", mode="shadow")
        assert shadow_state["drawdown"] > 0.2  # the hypothetical loss is visible...
        assert shadow_state["blocked"] is False  # ...but does not block
        paper_state = risk.daily_state(conn, limits, tmp_path, today="2027-11-01", mode="paper")
        assert paper_state["blocked"] is False  # paper ledger is untouched by shadow P&L
        mixed = risk.daily_state(conn, limits, tmp_path, today="2027-11-01")
        assert mixed["blocked"] is True  # unscoped (legacy) view still sees the drawdown
    finally:
        conn.close()


def test_no_order_placement_code_in_the_paper_path():
    """AGENTS.md rule 2: the desk is paper-only until Gate 2 is signed off."""
    forbidden = ("place_orders", "placeOrders", "flumine", "betfairlightweight")
    hits = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{path.relative_to(ROOT)}: {tok}" for tok in forbidden if tok in text]
    assert hits == []


def test_connect_migrates_a_raw_clv_ledger_to_net_clv(tmp_path):
    """A pre-existing data/paper.sqlite stores RAW log_clv; connect() moves it to log_clv_raw."""
    p = tmp_path / "old.sqlite"
    old = sqlite3.connect(str(p))
    old.executescript(
        """
        CREATE TABLE bets (
            bet_id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, match_key TEXT NOT NULL,
            div TEXT NOT NULL, date TEXT NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL,
            kickoff_utc TEXT NOT NULL, market TEXT NOT NULL, selection TEXT NOT NULL,
            model_prob REAL, market_prob REAL, pooled_prob REAL, price_taken REAL NOT NULL,
            price_source TEXT NOT NULL, edge REAL NOT NULL, stake_units REAL NOT NULL,
            stake_frac REAL NOT NULL, mode TEXT NOT NULL, config_hash TEXT NOT NULL,
            gate0_hash TEXT NOT NULL, experiment_id TEXT NOT NULL
        );
        CREATE TABLE settlements (
            bet_id TEXT PRIMARY KEY, settled_utc TEXT NOT NULL, result TEXT NOT NULL,
            won INTEGER NOT NULL, pnl REAL NOT NULL, closing_price REAL,
            closing_source TEXT, log_clv REAL
        );
        """
    )
    raw = float(np.log(4.0 * 0.3))
    old.execute(
        "INSERT INTO bets VALUES ('b1','t','k0','E2','2027-11-01','H','A','2027-11-01T15:00:00Z',"
        "'1x2','H',0.3,0.25,0.3,4.0,'BFE',0.1,1.0,0.01,'shadow','c','g','e')"
    )
    old.execute("INSERT INTO settlements VALUES ('b1','t','H',1,2.82,3.2,'BFE',?)", (raw,))
    old.commit()
    old.close()

    conn = ledger.connect(p)
    try:
        out = ledger.bets_with_settlements(conn)
        assert out.loc[0, "log_clv_raw"] == pytest.approx(raw)
        assert out.loc[0, "log_clv"] == pytest.approx(
            raw + float(np.log(net_odds(4.0, 0.06) / 4.0))
        )
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == ledger.SCHEMA_VERSION
        # idempotent: reconnecting does not re-migrate or double-shift
        conn.close()
        again = ledger.connect(p)
        assert again.execute("PRAGMA user_version").fetchone()[0] == ledger.SCHEMA_VERSION
        assert again.execute("SELECT log_clv FROM settlements").fetchone()[0] == pytest.approx(
            raw + float(np.log(net_odds(4.0, 0.06) / 4.0))
        )
        again.close()
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - already closed above
            pass



# ------------------------------------------------------------------ P6 leftovers (t_1e550b87)
def test_effective_limits_uses_the_strategy_kelly_capped_by_limits():
    lim = risk.Limits(kelly_fraction=0.25)
    assert risk.effective_limits(lim, {"kelly_fraction": 0.1}).kelly_fraction == 0.1
    assert risk.effective_limits(lim, {"kelly_fraction": 0.5}).kelly_fraction == 0.25
    assert risk.effective_limits(lim, {}).kelly_fraction == 0.25
    with pytest.raises(ValueError):
        risk.effective_limits(lim, {"kelly_fraction": 1.5})
    cfg = picks.load_strategy(ROOT / "config" / "strategy.toml")
    assert risk.effective_limits(risk.load_limits(ROOT / "config" / "limits.toml"), cfg)


def test_insert_bets_survives_a_duplicate_inside_one_batch(conn):
    out = ledger.insert_bets(conn, [_bet("k0"), _bet("k0"), _bet("k1")])
    assert [b["match_key"] for b in out] == ["k0", "k1"]
    assert len(ledger.read_table(conn, "bets")) == 2


def test_insert_bets_rechecks_the_daily_allowance_inside_the_transaction(conn):
    # another run already wrote 2 bets today after this run read its allowance
    ledger.insert_bets(conn, [_bet("k0"), _bet("k1")])
    out = ledger.insert_bets(conn, [_bet("k2"), _bet("k3"), _bet("k4")], daily_cap=3)
    assert [b["match_key"] for b in out] == ["k2"]
    assert ledger.bets_placed_on(conn, "2026-10-06") == 3


def test_blank_time_is_start_of_day_and_flagged_unknown(tmp_path):
    p = _fixture_csv(
        tmp_path / "fixtures.csv",
        [{"Div": "E2", "Date": "01/11/2027", "Time": "", "HomeTeam": "Burton",
          "AwayTeam": "Huddersfield"}],
    )
    fix = fx.parse_fixtures(fx.read_fixture_csv(p))
    assert fix.loc[0, "kickoff_utc"].isoformat() == "2027-11-01T00:00:00+00:00"
    assert bool(fix.loc[0, "time_unknown"])


def test_format_digest_accepts_naive_timestamps():
    bet = {**_bet("k0"), "kickoff_utc": "2027-11-01 15:00:00"}
    text = picks.format_digest([bet], {"mode": "shadow"}, pd.Timestamp("2027-11-01 08:00:00"))
    assert "Huddersfield" in text


def test_market_rows_aligns_on_raw_rows_when_parse_drops_rows(tmp_path):
    p = _fixture_csv(
        tmp_path / "fixtures.csv",
        [
            {"Div": "E2", "Date": "", "Time": "15:00", "HomeTeam": "X", "AwayTeam": "Y",
             "BFEH": 9.0, "BFED": 9.0, "BFEA": 9.0},  # dropped: no date
            {"Div": "E2", "Date": "01/11/2027", "Time": "15:00", "HomeTeam": "Burton",
             "AwayTeam": "Huddersfield", "BFEH": 4.0, "BFED": 3.9, "BFEA": 2.0},
        ],
    )
    raw = fx.read_fixture_csv(p)
    fix = fx.parse_fixtures(raw)
    rows = fx.market_rows(raw, fix, phase="pre", markets=("1x2",))
    assert len(rows) == 3
    assert rows[rows["selection"] == "H"]["price"].iloc[0] == 4.0
