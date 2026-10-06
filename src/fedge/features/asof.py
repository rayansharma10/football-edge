"""As-of feature access: only rows with available_at < bet_time.

This is the single gateway for building features (AGENTS.md rule 3). ``available_at`` and
``bet_time`` must be tz-aware (UTC). The comparison is strict: a row that becomes available at
exactly ``bet_time`` is NOT visible.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

# Outcome columns of the match being bet on. Reading them is reading the future (AGENTS.md rule 3),
# so the feature gateway strips them before a feature function ever sees a bet row.
OUTCOME_COLS = ("FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR")


class LeakageError(AssertionError):
    """Raised when data with available_at >= bet_time is used for a bet at bet_time."""


def _check_tz(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("bet_time must be tz-aware (UTC)")
    return ts


def _check_available_at(available_at: pd.Series, col: str) -> None:
    if available_at.dt.tz is None:
        raise ValueError(f"{col} must be tz-aware (UTC)")


def asof(df: pd.DataFrame, bet_time, col: str = "available_at") -> pd.DataFrame:
    """Rows of ``df`` visible at ``bet_time`` (available_at strictly before it)."""
    bet_time = _check_tz(bet_time)
    _check_available_at(df[col], col)
    return df.loc[df[col] < bet_time]


def assert_no_leak(df: pd.DataFrame, bet_time, col: str = "available_at") -> None:
    """Raise LeakageError if any row in ``df`` has available_at >= bet_time."""
    bet_time = _check_tz(bet_time)
    bad = df[col] >= bet_time
    if bad.any():
        raise LeakageError(f"{int(bad.sum())} row(s) available at/after bet_time {bet_time}")


def visible_bet_row(bet: pd.Series) -> pd.Series:
    """The bet row as a feature may see it: the outcome columns (``OUTCOME_COLS``) removed.

    A bet row carries the result of the match being bet on, so a feature could read the answer
    straight from it. The gateway drops those columns; reading one raises KeyError, which
    ``build_features`` reports as a ``LeakageError``.
    """
    return bet.drop(index=[c for c in OUTCOME_COLS if c in bet.index])


def build_features(
    table: pd.DataFrame,
    bets: pd.DataFrame,
    fn: Callable[[pd.DataFrame, pd.Series], dict],
    bet_time_col: str = "bet_time",
    col: str = "available_at",
) -> pd.DataFrame:
    """Build one feature row per bet: ``fn(visible_rows, bet_row) -> dict``.

    Two guarantees, both enforced here rather than trusted to the feature author:

    * ``visible_rows`` is exactly ``asof(table, bet_time)``: the rows of ``table`` available
      strictly before the bet. Future rows cannot reach ``fn``.
    * ``bet_row`` is the bet's row with the outcome columns (``OUTCOME_COLS``: FTHG, FTAG, FTR,
      HTHG, HTAG, HTR) removed, so ``fn`` cannot read the result of the match it is betting on.
      Reading one raises ``LeakageError``.

    Cost is O(bets x table): every bet re-masks the whole feature table. ``fn`` is called once per
    bet and the result is indexed like ``bets``.
    """
    _check_available_at(table[col], col)
    out = []
    for _, bet in bets.iterrows():
        bet_time = _check_tz(bet[bet_time_col])
        try:
            out.append(fn(asof(table, bet_time, col), visible_bet_row(bet)))
        except KeyError as exc:
            key = exc.args[0] if exc.args else None
            if key in OUTCOME_COLS:
                raise LeakageError(
                    f"feature function read outcome column {key!r} of the bet row for bet_time "
                    f"{bet_time}: the result of the match being bet on is not knowable at bet time"
                ) from exc
            raise
    return pd.DataFrame(out, index=bets.index)
