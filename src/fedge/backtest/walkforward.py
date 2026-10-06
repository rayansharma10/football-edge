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


@dataclass(frozen=True, eq=False)  # eq=False: fields are ndarrays, so == would be ambiguous
class Fold:
    label: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp
    # Identity of the frame the positional indices were computed against (see ``_fingerprint``).
    fingerprint: tuple | None = None


def _fingerprint(matches: pd.DataFrame) -> tuple:
    """Cheap identity of a frame: the ``train_idx``/``test_idx`` arrays are positional.

    Only length and first/last rows are captured (a cheap sentinel, not a hash); enough to catch
    the realistic mistake of running folds built from one frame against a reordered or filtered
    copy of it.
    """
    if not len(matches):
        return (0,)
    if "match_id" in matches.columns:
        first_id, last_id = matches["match_id"].iloc[0], matches["match_id"].iloc[-1]
    else:
        first_id = last_id = None
    return (
        len(matches),
        matches["kickoff_utc"].iloc[0],
        matches["kickoff_utc"].iloc[-1],
        first_id,
        last_id,
    )


def _season_start(season: str) -> int:
    return int(season[:4])


def season_folds(
    matches: pd.DataFrame, min_train_seasons: int = 1, embargo: pd.Timedelta = EMBARGO
) -> Iterator[Fold]:
    """One fold per season after the first ``min_train_seasons``; train = earlier matches."""
    seasons = sorted(matches["season"].unique(), key=_season_start)
    cut = matches["kickoff_utc"]
    fp = _fingerprint(matches)
    for s in seasons[min_train_seasons:]:
        test = np.flatnonzero((matches["season"] == s).to_numpy())
        test_start = cut.iloc[test].min()
        train = np.flatnonzero((cut < test_start - embargo).to_numpy())
        if len(train):
            yield Fold(s, train, test, test_start, fp)


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
    fp = _fingerprint(matches)
    for wk in sorted(week.unique()):
        test = np.flatnonzero((week == wk).to_numpy())
        test_start = ko.iloc[test].min()
        if start is not None and test_start < start:
            continue
        train = np.flatnonzero((ko < test_start - embargo).to_numpy())
        if len(train) >= min_train_matches:
            yield Fold(str(wk), train, test, test_start, fp)


def run_walkforward(
    matches: pd.DataFrame,
    folds,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Apply ``fit_predict(train_df, test_df) -> predictions`` per fold and concatenate.

    Predictions are tagged with the fold label. Deterministic given a deterministic fit_predict.
    A fold built from a different frame than ``matches`` (reordered, filtered) is rejected: its
    positional indices would silently select the wrong rows.
    """
    fp = _fingerprint(matches)
    parts = []
    for f in folds:
        if f.fingerprint is not None and f.fingerprint != fp:
            raise ValueError(
                f"fold {f.label!r} was built from a different frame than the one passed to "
                f"run_walkforward (fingerprint {f.fingerprint!r} != {fp!r})"
            )
        pred = fit_predict(matches.iloc[f.train_idx], matches.iloc[f.test_idx]).copy()
        pred["fold"] = f.label
        parts.append(pred)
    return pd.concat(parts) if parts else pd.DataFrame()
