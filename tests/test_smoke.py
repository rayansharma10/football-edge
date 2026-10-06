import tomllib

import fedge


def _load(name: str) -> dict:
    with (fedge.CONFIG_DIR / name).open("rb") as f:
        return tomllib.load(f)


def test_import():
    assert fedge.__version__


def test_leagues_config():
    cfg = _load("leagues.toml")
    assert len(cfg["divisions"]) == 18
    assert "E0" in cfg["divisions"]
    assert cfg["seasons"]["all"][0] == "2012/13"
    assert cfg["seasons"]["all"][-1] == "2026/27"
    assert len(cfg["seasons"]["all"]) == 15
    assert cfg["markets"]["enabled"] == ["1x2", "ou25"]


def test_limits_config():
    cfg = _load("limits.toml")
    assert cfg["LIVE"] is False
    assert cfg["commission"] == 0.06
    assert cfg["max_stake_frac"] == 0.01
    assert cfg["kelly_fraction"] == 0.25
    assert cfg["max_bets_per_day"] == 20
    assert cfg["drawdown_kill"] == 0.2
