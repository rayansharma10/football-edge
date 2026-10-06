"""Stake caps, daily bet cap and the drawdown kill switch for the paper desk.

This module is the *only* thing standing between a model opinion and a recorded bet, and it is
deliberately tiny. It contains **no order-placement code and no exchange client**: the desk is
paper-only (AGENTS.md rule 2) and Betfair execution is out of scope until Gate 2.

Enforced here:

* ``config/limits.toml`` is parsed and ``LIVE = true`` is a hard error, so a mis-edited config
  stops the desk instead of silently enabling live betting.
* Stake size is ``kelly_fraction * full Kelly`` on commission-adjusted odds, capped at
  ``max_stake_frac`` of the bankroll (``limits.toml``).
* At most ``max_bets_per_day`` bets are written per UTC day; the picker sorts by edge and takes the
  top of the list.
* The kill-switch file (``data/KILL``) blocks every new bet while it exists. A realised drawdown
  over ``drawdown_kill`` blocks new bets for the rest of the run too (the same switch, computed
  rather than typed).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.backtest.stats import net_odds
from fedge.paper import ledger

DEFAULT_LIMITS = "config/limits.toml"
KILL_FILE = "KILL"
BANKROLL0 = 1000.0  # paper bankroll in units, the same starting bankroll as the backtests


@dataclass(frozen=True)
class Limits:
    """The risk limits, read from ``config/limits.toml``."""

    commission: float = 0.06
    max_stake_frac: float = 0.01
    kelly_fraction: float = 0.25
    max_bets_per_day: int = 20
    drawdown_kill: float = 0.2


def load_limits(path: Path | str = DEFAULT_LIMITS) -> Limits:
    """Read ``limits.toml``. Refuses to run with ``LIVE = true`` or an out-of-range cap (m6)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if cfg.get("LIVE", False):
        raise RuntimeError(
            f"{path}: LIVE = true. The paper desk never places a real bet (AGENTS.md rule 2); "
            "live trading needs Gate 2 sign-off and a separate execution card."
        )
    limits = Limits(
        commission=float(cfg["commission"]),
        max_stake_frac=float(cfg["max_stake_frac"]),
        kelly_fraction=float(cfg["kelly_fraction"]),
        max_bets_per_day=int(cfg["max_bets_per_day"]),
        drawdown_kill=float(cfg["drawdown_kill"]),
    )
    for name, value in (
        ("commission", limits.commission),
        ("max_stake_frac", limits.max_stake_frac),
        ("kelly_fraction", limits.kelly_fraction),
        ("drawdown_kill", limits.drawdown_kill),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}: {name} must be a fraction in [0, 1], got {value}")
    if limits.max_bets_per_day < 1:
        raise ValueError(f"{path}: max_bets_per_day must be >= 1")
    return limits


def kill_path(data_dir: Path | str = "data") -> Path:
    """``data/KILL``: while this file exists the desk places no bets."""
    return Path(data_dir) / KILL_FILE


def kill_switch_active(data_dir: Path | str = "data") -> bool:
    """True when the kill-switch file exists."""
    return kill_path(data_dir).exists()


def stake_frac(p: float, price: float, limits: Limits) -> float:
    """Fraction of the bankroll to stake: fractional Kelly, capped at ``max_stake_frac``.

    Kelly on commission-adjusted odds (``fedge.backtest.stats.kelly_fraction``): a selection with no
    edge gets 0, never a negative stake.
    """
    b = float(net_odds(price, limits.commission)) - 1.0
    if b <= 0.0:
        return 0.0
    f = (p * b - (1.0 - p)) / b
    return float(min(max(f, 0.0) * limits.kelly_fraction, limits.max_stake_frac))


def stake_units(p: float, price: float, limits: Limits, bankroll: float = BANKROLL0) -> tuple:
    """``(stake_frac, stake_units)`` for a bet at ``price`` with model probability ``p``."""
    f = stake_frac(p, price, limits)
    return f, f * bankroll


def realised_drawdown(settled: pd.DataFrame, bankroll0: float = BANKROLL0) -> float:
    """Max peak-to-trough drawdown of the settled ledger's bankroll path (0 when unsettled).

    Bets are ordered by kickoff so the path is the one a human would have lived through; P&L is
    already net of commission.
    """
    if settled is None or len(settled) == 0:
        return 0.0
    df = settled.sort_values(["kickoff_utc", "match_key", "market"], kind="mergesort")
    bank = bankroll0 + df["pnl"].cumsum().to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[bankroll0], bank]))[:-1]
    dd = (peak - bank) / peak
    return float(dd.max()) if len(dd) else 0.0


def daily_state(
    conn,
    limits: Limits,
    data_dir: Path | str = "data",
    today: str | None = None,
    bankroll0: float = BANKROLL0,
    mode: str | None = None,
) -> dict:
    """Whether the desk may bet today, and why not if it may not.

    Returns ``{'today', 'placed_today', 'remaining', 'drawdown', 'blocked', 'reason'}``. ``blocked``
    is True for the kill-switch file or a realised drawdown at/over ``drawdown_kill``; ``remaining``
    is how many more bets may be written today (0 when blocked).

    ``mode`` scopes the drawdown to one ledger (``paper``/``shadow``). Shadow bets are hypotheses
    with no money at risk, and blocking them on a hypothetical drawdown would freeze the evidence
    loop (no new bets -> no new settlements -> never unblocked), so the drawdown switch only gates
    ``paper`` mode; shadow mode is gated by the file switch alone (m3).
    """
    today = today or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    placed = ledger.bets_placed_on(conn, today)
    dd = realised_drawdown(ledger.settled_frame(conn, mode), bankroll0)
    reason = None
    if kill_switch_active(data_dir):
        reason = f"kill switch file {kill_path(data_dir)} exists"
    elif mode in (None, "paper") and dd >= limits.drawdown_kill:
        reason = f"realised drawdown {dd:.1%} >= limits drawdown_kill {limits.drawdown_kill:.0%}"
    remaining = 0 if reason else max(limits.max_bets_per_day - placed, 0)
    return {
        "today": today,
        "placed_today": placed,
        "remaining": remaining,
        "drawdown": dd,
        "blocked": bool(reason),
        "reason": reason,
    }
