"""Finished-match results for the accuracy tracker (pure assembly; I/O lives in the script).

Primary source: the full-season schedule (fixturedownload.com carries ``HomeTeamScore`` /
``AwayTeamScore``; openfootball carries ``score.ft``). Cross-check and fallback: the football-data
season CSVs (already ingested into ``matches.parquet`` and exposed as the ``played`` frame of
:func:`fedge.paper.model_state.played_matches`). Team names on both sides are already mapped to the
football-data spelling by :mod:`fedge.ingest.schedule`.
"""

from __future__ import annotations

import pandas as pd

from fedge.accuracy.store import iso, outcome_of
from fedge.ingest import schedule as sch

PAIR_TOLERANCE = pd.Timedelta(days=3)


def _row(key, div, home, away, kickoff, hg, ag, source, fetched) -> dict:
    hg, ag = int(hg), int(ag)
    return {
        "match_key": key, "div": div, "home": home, "away": away, "kickoff_utc": iso(kickoff),
        "home_goals": hg, "away_goals": ag, "result": outcome_of(hg, ag), "source": source,
        "fetched_ts": iso(fetched),
    }


def build_results(
    sched: pd.DataFrame, sources: dict[str, str], played: pd.DataFrame | None, divs, fetched,
) -> tuple[list[dict], list[str]]:
    """``(result_rows, warnings)`` for every finished desk-league match we can score.

    ``sched`` is a :class:`~fedge.ingest.schedule.ScheduleResult` fixtures frame (with
    ``home_goals`` / ``away_goals``), ``sources`` its ``div -> source`` map, ``played`` the
    football-data finished matches (``div, home, away, kickoff_utc, FTHG, FTAG``).
    """
    warnings: list[str] = []
    rows: list[dict] = []
    divs = set(map(str, divs))
    fd = pd.DataFrame()
    if played is not None and len(played):
        fd = played.loc[played["div"].isin(divs)].copy()
        fd["kickoff_utc"] = pd.to_datetime(fd["kickoff_utc"], utc=True)
    used_fd: set[int] = set()

    keyed = sch.to_fixtures(sched.loc[sched["div"].isin(divs)]) if len(sched) else pd.DataFrame()
    sched_by_idx = sched.loc[sched["div"].isin(divs)].reset_index(drop=True)
    for i in range(len(keyed)):
        f, s = keyed.iloc[i], sched_by_idx.iloc[i]
        # the football-data twin: same pair within a few days
        twin = None
        if len(fd):
            m = fd.loc[
                (fd["div"] == f["div"]) & (fd["home"] == f["home"]) & (fd["away"] == f["away"])
                & ((fd["kickoff_utc"] - f["kickoff_utc"]).abs() <= PAIR_TOLERANCE)
            ]
            if len(m):
                twin = m.iloc[0]
                used_fd.add(int(m.index[0]))
        has_sched = pd.notna(s["home_goals"]) and pd.notna(s["away_goals"])
        if has_sched:
            src = str(sources.get(f["div"], "schedule")).replace("cache:", "")
            if twin is not None and (int(twin["FTHG"]), int(twin["FTAG"])) != (
                int(s["home_goals"]), int(s["away_goals"])
            ):
                warnings.append(
                    f"score mismatch {f['home']} v {f['away']}: {src} "
                    f"{int(s['home_goals'])}-{int(s['away_goals'])} vs football-data "
                    f"{int(twin['FTHG'])}-{int(twin['FTAG'])} (kept {src})"
                )
            rows.append(_row(f["match_key"], f["div"], f["home"], f["away"], f["kickoff_utc"],
                             s["home_goals"], s["away_goals"], src, fetched))
        elif twin is not None:
            rows.append(_row(f["match_key"], f["div"], f["home"], f["away"], f["kickoff_utc"],
                             twin["FTHG"], twin["FTAG"], "football_data", fetched))
    # football-data matches the schedule does not know: keyed on their own date, flagged
    if len(fd):
        rest = fd.loc[[i for i in fd.index if int(i) not in used_fd]]
        rest = rest.loc[rest["kickoff_utc"] >= (
            sched["kickoff_utc"].min() if len(sched) else pd.Timestamp.max.tz_localize("UTC"))]
        if len(rest):
            keyed_fd = sch.to_fixtures(rest[["div", "home", "away", "kickoff_utc"]])
            for (_, r), key in zip(rest.iterrows(), keyed_fd["match_key"], strict=True):
                warnings.append(
                    f"football-data match not in schedule: {r['home']} v {r['away']} "
                    f"({r['kickoff_utc']:%Y-%m-%d}) - logged under its own key"
                )
                rows.append(_row(key, r["div"], r["home"], r["away"], r["kickoff_utc"],
                                 r["FTHG"], r["FTAG"], "football_data", fetched))
    return rows, warnings
