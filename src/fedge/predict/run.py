"""Glue between the schedule, the price-free model registry and the telemetry store."""

from __future__ import annotations

import json

import pandas as pd

from fedge.accuracy import store
from fedge.ingest import schedule as sch
from fedge.predict import models
from fedge.predict.upcoming import DIV_NAMES


def upcoming_fixtures(res: sch.ScheduleResult, now) -> pd.DataFrame:
    """Not-yet-played, future fixtures of the schedule in the ``match_key`` layout."""
    fx = sch.to_fixtures(sch.upcoming(res.fixtures, now))
    fx["league"] = fx["div"].map(lambda d: DIV_NAMES.get(d, d))
    return fx.sort_values(["kickoff_utc", "home"], kind="mergesort").reset_index(drop=True)


def records_to_rows(
    records: dict[str, dict], fixtures: pd.DataFrame, run_ts, model_version: str,
    source: str = "live",
) -> tuple[list[dict], list[dict]]:
    """``(log_rows, display_rows)`` for the fixtures a model returned a record for."""
    log_rows: list[dict] = []
    shown: list[dict] = []
    for f in fixtures.to_dict("records"):
        rec = records.get(f["match_key"])
        if rec is None:
            continue
        row = store.log_row(rec, f, run_ts, model_version, source)
        log_rows.append(row)
        shown.append({
            **{c: row[c] for c in store.UPCOMING_COLS if c in row},
            "top_scores": row["top_scores"], "note": rec.get("fallback"),
        })
    return log_rows, shown


def predict_all(
    data_dir, played: pd.DataFrame, fixtures: pd.DataFrame, run_ts,
    model_version: str = models.CURRENT_VERSION, threads: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Run the registered model on ``fixtures`` as of ``run_ts``."""
    if fixtures.empty:
        return [], []
    recs = models.predict(model_version, data_dir, played, fixtures, run_ts, threads=threads)
    return records_to_rows(recs, fixtures, run_ts, model_version)


def schedule_meta(res: sch.ScheduleResult) -> dict[str, str]:
    m = dict(res.meta)
    m["schedule_unmapped"] = json.dumps(res.unmapped)
    return m
