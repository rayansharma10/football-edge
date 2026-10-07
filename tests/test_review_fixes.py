"""Regression tests for REVIEW-P6-and-full fixes that live in scripts/ and ops/ (M2, N2)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ M2: Gate 0 is achievable
def test_gate0_can_pass_when_computable_items_pass_and_partials_become_prerequisites():
    rs = _load("scripts/run_strategy.py", "run_strategy_under_test")
    gates = {f"B0.{i}": ("PASS", "ev") for i in range(1, 11)}
    assert rs.gate0_decision(gates, True) == (True, [])  # all-PASS synthetic

    gates["B0.8"] = ("PARTIAL", "ev")
    gates["B0.9"] = ("PARTIAL", "ev")
    passed, prereqs = rs.gate0_decision(gates, True)
    assert passed is True and prereqs == ["B0.8", "B0.9"]  # PARTIAL no longer blocks Gate 0

    assert rs.gate0_decision(gates, False)[0] is False  # headline CLV bar still applies
    gates["B0.6"] = ("FAIL", "ev")
    assert rs.gate0_decision(gates, True)[0] is False  # a computable FAIL still blocks


def test_gate0_computed_items_fail_on_bad_data():
    rs = _load("scripts/run_strategy.py", "run_strategy_under_test")
    good = pd.DataFrame(
        [
            {"market": m, "season": "2020/21", "model": name, "n": 10, "rps": 0.2,
             "log_loss": 1.0, "brier": 0.6, "slope": 1.0, "ece": 0.02}
            for m in ("1x2", "ou25") for name in ("market_close", "market_pre", "lgb_xg")
        ]
    )
    assert rs.benchmark_ok(good)
    assert not rs.benchmark_ok(good.iloc[:0])
    assert not rs.benchmark_ok(good.assign(rps=float("nan")))
    assert not rs.benchmark_ok(good[good["model"] != "market_close"])

    seasons = ["2018/19", "2019/20", "2020/21", "2021/22"]
    ok = {"s": {"seasons": seasons, "bets": pd.DataFrame({"season": ["2020/21"]}),
                "cfgs": pd.DataFrame({"season": ["2020/21", "2021/22"]})}}
    assert rs.walkforward_ok(ok)
    ok["s"]["bets"] = pd.DataFrame({"season": ["2018/19"]})  # a bet in a season with no history
    assert not rs.walkforward_ok(ok)

    cell = pd.DataFrame({"n": [3, 2]})
    out = {s: {"overall": {"n": 5}, "by_cell": cell} for s in rs.SCENARIOS}
    assert rs.reporting_ok(out)
    out["max"]["overall"] = {"n": 6}  # a bet missing from the per-cell table
    assert not rs.reporting_ok(out)


# ------------------------------------------------------------------ N2: cron state file
def _cron(tmp_path, monkeypatch):
    mod = _load("ops/hermes_cron/fedge_paper.py", "fedge_paper_under_test")
    monkeypatch.setattr(mod, "STATE", tmp_path / "data" / "last_run.json")
    return mod


def test_cron_record_keeps_concurrent_updates(tmp_path, monkeypatch):
    mod = _cron(tmp_path, monkeypatch)
    jobs = [f"job{i}" for i in range(12)]
    threads = [threading.Thread(target=mod._record, args=(j, True)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = json.loads(mod.STATE.read_text(encoding="utf-8"))
    assert set(st) == set(jobs)  # no lost update
    assert not list(mod.STATE.parent.glob("*.tmp")) and not list(mod.STATE.parent.glob("*.lock"))


def test_cron_timeout_is_recorded_as_a_failed_attempt(tmp_path, monkeypatch, capsys):
    mod = _cron(tmp_path, monkeypatch)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "update_results.py").write_text("pass\n")
    monkeypatch.setattr(mod, "REPO", tmp_path)

    calls = []

    def hang(*a, **k):
        calls.append(1)
        raise subprocess.TimeoutExpired(cmd="uv", timeout=1500, output=b"partial", stderr=None)

    monkeypatch.setattr(mod.subprocess, "run", hang)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    assert mod.main("results") == 1
    assert len(calls) == 1  # a hang is not retried (it would double the wait)
    st = json.loads(mod.STATE.read_text(encoding="utf-8"))
    assert "last_attempt_utc" in st["results"] and "last_success_utc" not in st["results"]
    assert "timed out" in capsys.readouterr().out
