"""Shared test setup: no test may touch the network for the full-season schedule."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _offline_schedule(monkeypatch):
    """The picks/predict scripts load a full-season schedule over HTTP; tests get none (priced
    fixtures only). Schedule behaviour itself is tested in tests/test_schedule.py with fakes."""
    from fedge.ingest import schedule as sch
    from fedge.predict import upcoming as pred

    def offline(priced_fixtures, data_dir, divs, now, refresh=True, delay=1.0):
        desk = priced_fixtures.loc[priced_fixtures["div"].isin(set(map(str, divs)))]
        res = sch.ScheduleResult(fixtures=pd.DataFrame(columns=sch.COLUMNS))
        return desk.reset_index(drop=True), res

    monkeypatch.setattr(pred, "full_fixture_list", offline)
