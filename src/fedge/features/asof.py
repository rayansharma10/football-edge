"""As-of feature access: only rows with available_at < bet_time.

This is the single gateway for building features (AGENTS.md rule 3). ``available_at`` and
``bet_time`` must be tz-aware (UTC). The comparison is strict: a row that becomes available at
exactly ``bet_time`` is NOT visible.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


class LeakageError(AssertionError):
    """Raised when data with available_at >= bet_time is used for a bet at bet_time."""


def _check_tz(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("bet_time must be tz-aware (UTC)")
    return ts


def asof(df: pd.DataFrame, bet_time, col: str = "available_at") -> pd.DataFrame:
    """Rows of ``df`` visible at ``bet_time`` (available_at strictly before it)."""
    bet_time = _check_tz(bet_time)
    if df[col].dt.tz is None:
        raise ValueError(f"{col} must be tz-aware (UTC)")
    return df.loc[df[col] < bet_time]


def assert_no_leak(df: pd.DataFrame, bet_time, col: str = "available_at") -> None:
    """Raise LeakageError if any row in ``df`` has available_at >= bet_time."""
    bet_time = _check_tz(bet_time)
    bad = df[col] >= bet_time
    if bad.any():
        raise LeakageError(f"{int(bad.sum())} row(s) available at/after bet_time {bet_time}")


def build_features(
    table: pd.DataFrame,
    bets: pd.DataFrame,
    fn: Callable[[pd.DataFrame, pd.Series], dict],
    bet_time_col: str = "bet_time",
    col: str = "available_at",
) -> pd.DataFrame:
    """Build one feature row per bet: ``fn(visible_rows, bet_row) -> dict``.

    ``visible_rows`` is only ever ``asof(table, bet_time)``, so ``fn`` cannot see the future.
    The result is indexed like ``bets``.
    """
    out = []
    for _, bet in bets.iterrows():
        out.append(fn(asof(table, bet[bet_time_col], col), bet))
    return pd.DataFrame(out, index=bets.index)
