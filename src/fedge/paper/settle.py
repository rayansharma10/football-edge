"""Settlement maths for the paper ledger: results, P&L after commission, and closing line value.

Definitions, all reused from earlier cards rather than re-derived:

* ``won``: the selection won the market (1X2 from ``FTR``; over/under 2.5 from the total goals).
* ``pnl``: ``stake * (price - 1) * (1 - commission)`` for a winner, ``-stake`` for a loser, i.e. 6%
  commission on net winnings (AGENTS.md rule 10).
* ``log_clv`` (NET, the gate variable): ``log(net_odds(price_taken) * p_close_fair)``, the price
  after commission on winnings (AGENTS.md rule 10). ``log_clv_raw`` is the same without
  commission, ``log(price_taken * p_close_fair)``; it is optimistic by 3-5 pp and kept for
  reference only. ``p_close_fair`` is the *margin-free*
  (power) closing probability of the selection (:mod:`fedge.market`; see
  :func:`fedge.backtest.stats.log_clv`). The closing price itself is the Betfair Exchange close
  (``BFEC*``) when both sides of the market are quoted, otherwise the market average close
  (``AvgC*``); the source is stored with the number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fedge.backtest.stats import log_clv, net_odds
from fedge.market import devig

SELS = {"1x2": ("H", "D", "A"), "ou25": ("over", "under")}


def market_result(market: str, fthg: float, ftag: float, ftr: str | None) -> str:
    """The winning selection of a finished match (``H``/``D``/``A`` or ``over``/``under``)."""
    if market == "1x2":
        if ftr in ("H", "D", "A"):
            return str(ftr)
        if pd.isna(fthg) or pd.isna(ftag):
            raise ValueError("no result available")
        return "H" if fthg > ftag else ("D" if fthg == ftag else "A")
    if market == "ou25":
        return "over" if float(fthg) + float(ftag) > 2.5 else "under"
    raise ValueError(f"unknown market {market!r}")


def won(market: str, selection: str, fthg: float, ftag: float, ftr: str | None) -> bool:
    """Did ``selection`` win? (``market_result`` == ``selection``.)"""
    return market_result(market, fthg, ftag, ftr) == selection


def pnl(stake_units: float, price_taken: float, is_winner: bool, commission: float) -> float:
    """P&L in units: net winnings after commission when it wins, ``-stake`` when it loses."""
    if is_winner:
        return float(stake_units) * (float(price_taken) - 1.0) * (1.0 - commission)
    return -float(stake_units)


def wide_prices(odds: pd.DataFrame, book: str, market: str, phase: str) -> pd.DataFrame:
    """Complete de-marginable prices for (book, market, phase), indexed by ``match_id``.

    Rows with a non-positive/nonsense quote (``<= 1.0``) are dropped: one bad quote would otherwise
    make the de-vig raise and abort the whole settlement run (m8).

    Assumes the ingest layer already de-duplicated the odds table to one row per (match_id, book,
    market, phase, selection): the pivot takes ``aggfunc="first"`` and so would silently pick an
    arbitrary row of a duplicate pair (n9).
    """
    sels = list(SELS[market])
    o = odds.loc[
        (odds["bookmaker"] == book) & (odds["market"] == market) & (odds["phase"] == phase)
    ]
    if o.empty:
        return pd.DataFrame(columns=sels)
    w = o.pivot_table(index="match_id", columns="selection", values="price", aggfunc="first")
    if not set(sels).issubset(w.columns):
        return pd.DataFrame(columns=sels)
    w = w[sels].dropna()
    price = w.to_numpy(dtype=float)
    return w[np.all(price > 1.0, axis=1)]


def closing_probabilities(odds: pd.DataFrame, market: str) -> pd.DataFrame:
    """Margin-free closing probabilities: ``BFEC*`` if complete for the match, else ``AvgC*``.

    Returns columns ``[<selection>..., 'source']`` indexed by ``match_id`` (power de-vig,
    :mod:`fedge.market`). ``source`` is ``BFE`` or ``Avg``: which book the *quoted* close came from.
    """
    sels = list(SELS[market])
    bfe = wide_prices(odds, "BFE", market, "close")
    avg = wide_prices(odds, "Avg", market, "close")
    use_bfe = bfe.index
    use_avg = avg.index.difference(bfe.index)
    idx = use_bfe.union(use_avg)
    if len(idx) == 0:
        return pd.DataFrame(columns=[*sels, "source"], index=pd.Index([], name="match_id"))
    flag = idx.isin(use_bfe)
    bfe_p = bfe.reindex(idx)[sels].to_numpy(dtype=float)
    avg_p = avg.reindex(idx)[sels].to_numpy(dtype=float)
    prices = np.where(flag[:, None], bfe_p, avg_p)
    p = devig(prices, "power").probs
    out = pd.DataFrame(p, index=idx, columns=sels)
    out["source"] = np.where(flag, "BFE", "Avg")
    return out


def settle_bet(
    bet: dict,
    fthg: float,
    ftag: float,
    ftr: str | None,
    settled_utc: str,
    commission: float = 0.06,
    close: dict | None = None,
) -> dict:
    """One settlement row for ``bet`` given the match result and, optionally, the closing price.

    ``close`` is ``{'price': quoted closing price, 'source': 'BFE'|'Avg', 'p_fair': margin-free
    closing probability of the selection}`` or None when the closing price is not available yet: the
    bet still settles (``pnl`` is final), but ``closing_price`` and ``log_clv`` stay NULL.
    """
    is_winner = won(bet["market"], bet["selection"], fthg, ftag, ftr)
    row = {
        "bet_id": bet["bet_id"],
        "settled_utc": settled_utc,
        "result": market_result(bet["market"], fthg, ftag, ftr),
        "won": int(is_winner),
        "pnl": pnl(float(bet["stake_units"]), float(bet["price_taken"]), is_winner, commission),
        "closing_price": None,
        "closing_source": None,
        "log_clv": None,
        "log_clv_raw": None,
    }
    if close and close.get("p_fair") is not None:
        row["closing_price"] = None if close.get("price") is None else float(close["price"])
        row["closing_source"] = str(close["source"])
        price = float(bet["price_taken"])
        p_fair = float(close["p_fair"])
        row["log_clv_raw"] = float(log_clv(price, p_fair))
        row["log_clv"] = float(log_clv(float(net_odds(price, commission)), p_fair))
    return row
