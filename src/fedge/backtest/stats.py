"""CLV, clustered bootstrap, bankroll simulation (flat and fractional Kelly, with commission).

CLV definition: log(price_taken * p_close_fair) = log(price_taken / fair_close_odds), where
p_close_fair is the margin-free closing probability of the selection (see fedge.market). Taking
exactly the fair closing price gives CLV = 0. The raw margined closing price is worse than fair:
CLV = -log(booksum) < 0, the cost of the margin.
For exchange prices pass commission-adjusted odds (see ``net_odds``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMISSION = 0.06  # Betfair AU market base rate, charged on net winnings per market


def net_odds(price, commission: float = COMMISSION):
    """Back odds after commission on a winning bet: 1 + (price - 1) * (1 - commission)."""
    price = np.asarray(price, dtype=float)
    return 1.0 + (price - 1.0) * (1.0 - commission)


def log_clv(price_taken, p_close_fair) -> np.ndarray:
    """Per-bet log closing-line value vs the margin-free closing probability."""
    return np.log(np.asarray(price_taken, dtype=float) * np.asarray(p_close_fair, dtype=float))


def bootstrap_ci(
    values,
    clusters,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Match-clustered bootstrap CI for the mean of ``values``.

    Whole clusters (e.g. match ids) are resampled with replacement, so bets on the same match
    stay together. Seeded: same inputs give identical output.
    """
    v = np.asarray(values, dtype=float)
    codes, uniq = pd.factorize(pd.Series(clusters), sort=True)
    k = len(uniq)
    sums = np.bincount(codes, weights=v, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(v.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(v)),
        "n_clusters": int(k),
    }


def kelly_fraction(p, price, commission: float = COMMISSION) -> np.ndarray:
    """Full-Kelly bankroll fraction for a back bet at commission-adjusted odds (0 if no edge)."""
    p = np.asarray(p, dtype=float)
    b = net_odds(price, commission) - 1.0
    f = (p * b - (1.0 - p)) / b
    return np.clip(f, 0.0, None)


def simulate_bankroll(
    bets: pd.DataFrame,
    mode: str = "flat",
    bankroll0: float = 1000.0,
    flat_stake: float = 10.0,
    kelly_mult: float = 0.25,
    max_stake_frac: float = 0.05,
    commission: float = COMMISSION,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate a ledger of back bets.

    ``bets`` columns: market_id, date (UTC date or timestamp, used for ordering), price (decimal,
    before commission), won (bool), and for Kelly mode ``p`` (model probability).

    Rules:
      * Bets are grouped by UTC day. All stakes in a day use the bankroll at the start of that
        day; the day's markets are settled together afterwards (no intra-day leakage).
      * Commission is charged per market on net winnings (sum of bet P&L in that market, if > 0).
      * Kelly stake = kelly_mult * full Kelly (on commission-adjusted odds) * bankroll, capped at
        ``max_stake_frac`` of bankroll. Bets with no edge are skipped.
    Returns (per-market ledger, summary dict).
    """
    if mode not in ("flat", "kelly"):
        raise ValueError("mode must be 'flat' or 'kelly'")
    df = bets.copy()
    df["day"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
    df = df.sort_values(["day", "market_id"], kind="mergesort")
    bank = bankroll0
    peak = bankroll0
    max_dd = 0.0
    rows = []
    turnover = 0.0
    n_bets = 0
    for day, g in df.groupby("day", sort=True):
        if mode == "flat":
            stakes = np.full(len(g), float(flat_stake))
        else:
            f = kelly_fraction(g["p"].to_numpy(), g["price"].to_numpy(), commission)
            stakes = np.minimum(kelly_mult * f, max_stake_frac) * bank
        gross = np.where(g["won"].to_numpy(), stakes * (g["price"].to_numpy() - 1.0), -stakes)
        gross = np.where(stakes > 0, gross, 0.0)
        tmp = pd.DataFrame({"market_id": g["market_id"].to_numpy(), "pnl": gross})
        net = tmp.groupby("market_id", sort=True)["pnl"].sum()
        comm = commission * net.clip(lower=0.0)
        day_pnl = float((net - comm).sum())
        turnover += float(stakes.sum())
        n_bets += int((stakes > 0).sum())
        bank += day_pnl
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
        for m in net.index:
            rows.append((day, m, float(net[m]), float(comm[m])))
    ledger = pd.DataFrame(rows, columns=["day", "market_id", "gross_pnl", "commission"])
    profit = bank - bankroll0
    summary = {
        "n_bets": n_bets,
        "n_markets": int(len(ledger)),
        "turnover": turnover,
        "profit": profit,
        "yield": profit / turnover if turnover > 0 else 0.0,
        "final_bankroll": bank,
        "max_drawdown": max_dd,
        "commission_paid": float(ledger["commission"].sum()) if len(ledger) else 0.0,
    }
    return ledger, summary
