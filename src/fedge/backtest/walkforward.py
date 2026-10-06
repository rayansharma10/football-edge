"""Walk-forward folds: season and matchweek. No random splits, ever.

Folds are expanding-window: train = everything strictly earlier than the test block (minus an
embargo), test = the block. Inputs need a tz-aware ``kickoff_utc`` column (and ``season``
for season folds). Returned index arrays refer to the positional order of ``matches``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

EMBARGO = pd.Timedelta(hours=3)  # a match is only a usable training row ~3h after kickoff


@dataclass(frozen=True)
class Fold:
    label: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp


def _season_start(season: str) -> int:
    return int(season[:4])


def season_folds(
    matches: pd.DataFrame, min_train_seasons: int = 1, embargo: pd.Timedelta = EMBARGO
) -> Iterator[Fold]:
    """One fold per season after the first ``min_train_seasons``; train = earlier matches."""
    seasons = sorted(matches["season"].unique(), key=_season_start)
    cut = matches["kickoff_utc"]
    for s in seasons[min_train_seasons:]:
        test = np.flatnonzero((matches["season"] == s).to_numpy())
        test_start = cut.iloc[test].min()
        train = np.flatnonzero((cut < test_start - embargo).to_numpy())
        if len(train):
            yield Fold(s, train, test, test_start)


def matchweek_folds(
    matches: pd.DataFrame,
    min_train_matches: int = 100,
    embargo: pd.Timedelta = EMBARGO,
    start: pd.Timestamp | None = None,
) -> Iterator[Fold]:
    """One fold per Monday-based UTC calendar week ("matchweek"); train = all earlier matches.

    Only weeks with at least ``min_train_matches`` earlier matches are yielded, and (if given)
    weeks whose first kickoff is at/after the tz-aware ``start``.
    """
    ko = matches["kickoff_utc"]
    week = ko.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W-SUN")
    for wk in sorted(week.unique()):
        test = np.flatnonzero((week == wk).to_numpy())
        test_start = ko.iloc[test].min()
        if start is not None and test_start < start:
            continue
        train = np.flatnonzero((ko < test_start - embargo).to_numpy())
        if len(train) >= min_train_matches:
            yield Fold(str(wk), train, test, test_start)


def run_walkforward(
    matches: pd.DataFrame,
    folds,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Apply ``fit_predict(train_df, test_df) -> predictions`` per fold and concatenate.

    Predictions are tagged with the fold label. Deterministic given a deterministic fit_predict.
    """
    parts = []
    for f in folds:
        pred = fit_predict(matches.iloc[f.train_idx], matches.iloc[f.test_idx]).copy()
        pred["fold"] = f.label
        parts.append(pred)
    return pd.concat(parts) if parts else pd.DataFrame()
