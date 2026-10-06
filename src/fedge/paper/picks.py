"""Pick planning and the Slack-ready digest: pure functions, no I/O, no model.

Given the scored fixtures (:func:`fedge.paper.model_state.score_fixtures`) this module decides
*which* bets the desk records and formats the stdout digest. It is deliberately separated from the
scripts so the selection rules and the stdout contract can be tested without network, a model or a
database.

Selection rules (AGENTS.md rule 4: parameters are configuration, never fitted here):

* Gate 0 **passed**: the (division, market) whitelist and the per-market edge threshold from
  ``config/strategy.toml`` apply, and bets carry ``mode='paper'``.
* Gate 0 **not** passed (the current state, ``reports/gate0.md``): the desk runs in *shadow* mode.
  Both markets are then open in every division the model can score, with the single
  ``[shadow].shadow_threshold`` edge (0.03 after 6% commission) and ``mode='shadow'``. Shadow bets
  are hypotheses for evidence collection; they never count towards Gate 1.
* At most one bet per (match, market): the largest edge, as in the Phase 5 backtest.
* Bets are ranked by edge and capped by the remaining daily allowance from ``fedge.paper.risk``.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import pandas as pd

from fedge.paper import ledger, risk
from fedge.paper.settle import SELS

SYDNEY = "Australia/Sydney"
DEFAULT_SHADOW_THRESHOLD = 0.03


def load_strategy(path: Path | str = "config/strategy.toml") -> dict:
    """Read and validate ``config/strategy.toml`` (the Gate 0 decision written by Phase 5)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "gate0_passed" not in cfg or "mode" not in cfg:
        raise ValueError(f"{path}: missing gate0_passed/mode")
    for market in SELS:
        sec = cfg.get("markets", {}).get(market)
        if sec is None:
            raise ValueError(f"{path}: missing [markets.{market}]")
        for key in ("model", "pool_weight", "threshold", "whitelist"):
            if key not in sec:
                raise ValueError(f"{path}: [markets.{market}] missing {key}")
    shadow = cfg.get("shadow", {})
    cfg["shadow_threshold"] = float(shadow.get("shadow_threshold", DEFAULT_SHADOW_THRESHOLD))
    return cfg


def file_hash(path: Path | str) -> str:
    """sha256 of a file's bytes (used for the config and Gate 0 report stamps on every bet)."""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def pool_weights(strategy: dict) -> dict:
    """``{market: weight}`` on the model in the log pool, from the strategy config."""
    return {m: float(strategy["markets"][m]["pool_weight"]) for m in SELS}


def market_threshold(strategy: dict, market: str, mode: str) -> float:
    """Edge threshold for ``market``: the shadow threshold in shadow mode, else the Gate 0 one."""
    if mode == "shadow":
        return float(strategy["shadow_threshold"])
    return float(strategy["markets"][market]["threshold"])


def market_whitelist(strategy: dict, market: str, mode: str) -> list[str]:
    """Divisions allowed for ``market``: every division in shadow mode, else the Gate 0 list.

    ``config/strategy.toml`` stores the whitelist as ``[["E0", "1x2"], ...]`` (division, market)
    pairs, one per whitelisted cell, as written by ``run_strategy.render_strategy_toml``; a flat
    list of division codes is accepted too. Pairs belonging to another market are ignored, and the
    division codes are returned flat so they can be matched against a fixture's ``div``.
    """
    if mode == "shadow":
        return []
    out: list[str] = []
    for item in strategy["markets"][market].get("whitelist", []):
        if isinstance(item, (list, tuple)):
            if len(item) < 2 or str(item[1]) != market:
                continue
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def candidate_bets(
    edges: pd.DataFrame, fixtures: pd.DataFrame, strategy: dict, mode: str
) -> pd.DataFrame:
    """Best-edge selection per (match, market) that clears the threshold and the whitelist."""
    if edges.empty:
        return edges.assign(match_key=pd.Series(dtype=str))
    df = edges.merge(
        fixtures[["match_key", "div", "date", "home", "away", "kickoff_utc"]],
        on="match_key",
        how="inner",
    )
    keep = []
    for market in SELS:
        thr = market_threshold(strategy, market, mode)
        wl = market_whitelist(strategy, market, mode)
        g = df[(df["market"] == market) & (df["edge"] >= thr)]
        if wl:
            g = g[g["div"].isin(wl)]
        if g.empty:
            continue
        order = g.sort_values(
            ["edge", "match_key", "selection"], ascending=[False, True, True]
        )
        keep.append(order.groupby("match_key", sort=False).head(1))
    if not keep:
        return edges.iloc[0:0].assign(div=None)
    out = pd.concat(keep, ignore_index=True)
    return out.sort_values(["edge", "match_key", "market"], ascending=[False, True, True])


def plan_bets(
    edges: pd.DataFrame,
    fixtures: pd.DataFrame,
    strategy: dict,
    limits: risk.Limits,
    remaining: int,
    now_utc,
    config_hash: str,
    gate0_hash: str,
    experiment_id: str,
) -> list[dict]:
    """Bet rows for the top ``remaining`` candidates, highest edge first (never above the cap)."""
    mode = str(strategy["mode"])
    cand = candidate_bets(edges, fixtures, strategy, mode)
    rows = []
    for r in cand.head(max(remaining, 0)).itertuples(index=False):
        frac, units = risk.stake_units(float(r.pooled_prob), float(r.price), limits)
        if frac <= 0.0:
            continue
        rows.append(
            {
                "bet_id": ledger.bet_id(r.match_key, r.market, r.selection),
                "created_utc": _iso(now_utc),
                "match_key": r.match_key,
                "div": r.div,
                "date": r.date,
                "home": r.home,
                "away": r.away,
                "kickoff_utc": _iso(r.kickoff_utc),
                "market": r.market,
                "selection": r.selection,
                "model_prob": float(r.model_prob),
                "market_prob": float(r.market_prob),
                "pooled_prob": float(r.pooled_prob),
                "price_taken": float(r.price),
                "price_source": str(r.price_source),
                "edge": float(r.edge),
                "stake_units": float(units),
                "stake_frac": float(frac),
                "mode": mode,
                "config_hash": config_hash,
                "gate0_hash": gate0_hash,
                "experiment_id": experiment_id,
            }
        )
    # sort for a deterministic insert order (and a stable digest), edge first
    return sorted(rows, key=lambda b: (-b["edge"], b["match_key"], b["market"], b["selection"]))


def snapshot_rows(edges: pd.DataFrame, fixtures: pd.DataFrame, now_utc) -> list[dict]:
    """One snapshot row per (fixture, market, selection) seen, bet or not."""
    if edges.empty:
        return []
    df = edges.merge(
        fixtures[["match_key", "div", "date", "home", "away", "kickoff_utc"]],
        on="match_key",
        how="inner",
    )
    seen = _iso(now_utc)
    if isinstance(now_utc, str):
        day = str(now_utc)[:10]
    else:
        day = pd.Timestamp(now_utc).strftime("%Y-%m-%d")
    rows = []
    for r in df.sort_values(["match_key", "market", "selection"], kind="mergesort").itertuples(
        index=False
    ):
        rows.append(
            {
                "seen_utc": seen,
                "snapshot_day": day,
                "match_key": r.match_key,
                "div": r.div,
                "date": r.date,
                "home": r.home,
                "away": r.away,
                "kickoff_utc": _iso(r.kickoff_utc),
                "market": r.market,
                "selection": r.selection,
                "price": _f(r.price),
                "price_source": r.price_source,
                "model_prob": _f(r.model_prob),
                "pooled_prob": _f(r.pooled_prob),
                "edge": _f(r.edge),
            }
        )
    return rows


def format_digest(bets: list[dict], strategy: dict, now_utc) -> str:
    """The Slack-ready digest: header line, then one line per new bet (Sydney kickoff time)."""
    mode = str(strategy["mode"])
    day = pd.Timestamp(now_utc).tz_convert(SYDNEY).strftime("%Y-%m-%d %H:%M %Z")
    gate = (
        f"Gate 0 parked the whitelist (mode={mode})"
        if mode != "paper"
        else "Gate 0 whitelist active"
    )
    head = f"fedge {mode} picks {day} | {len(bets)} new bet(s) | {gate}"
    lines = [head]
    for b in bets:
        ko = pd.Timestamp(b["kickoff_utc"]).tz_convert(SYDNEY).strftime("%a %d %b %H:%M %Z")
        lines.append(
            f"  {ko} | {b['div']} | {b['home']} v {b['away']} | {b['selection']} @ "
            f"{b['price_taken']:.2f} ({b['price_source']}) | edge {b['edge'] * 100:+.1f}% | "
            f"{b['stake_units']:.2f}u"
        )
    the_bets = "" if len(bets) == 1 else "s"
    lines[0] = lines[0].replace("new bet(s)", f"new bet{the_bets}")
    return "\n".join(lines)


def _iso(ts) -> str:
    """tz-aware UTC ISO-8601 string."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").isoformat()


def _f(v):
    """float or None (SQLite-friendly)."""
    return None if v is None or pd.isna(v) else float(v)
