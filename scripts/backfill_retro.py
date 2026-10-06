"""Leakage-safe retrospective predictions for matches already played (``source = 'retro'``).

    uv run python scripts/backfill_retro.py                  # every finished current-season match
    uv run python scripts/backfill_retro.py --max-groups 6   # only the latest 6 matchdays
    uv run python scripts/backfill_retro.py --dry-run

Matches are grouped into matchdays (kickoff dates at most ``--span-days`` apart). For each group the
model is fitted on, and the features/ratings built from, matches kicked off STRICTLY BEFORE the
group's first date at 00:00 UTC (``asof``); nothing from the group itself or later is visible. The
row's ``run_ts`` is that ``asof`` (the model's knowledge cut-off, so ``hours_before_kickoff`` is
honest). Rows are tagged ``source = 'retro'`` and are shown as "retrospective (not live)" and kept
out of the live headline. Matches that already have any logged prediction are skipped.
Needs ``results`` to be filled first (``scripts/update_results.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedge.accuracy import store  # noqa: E402
from fedge.paper import model_state as ms  # noqa: E402
from fedge.predict import models  # noqa: E402
from fedge.predict import run as prun  # noqa: E402
from fedge.predict.upcoming import DIV_NAMES  # noqa: E402


def matchday_groups(
    kickoffs: pd.Series, span_days: int = 3
) -> list[tuple[pd.Timestamp, list[int]]]:
    """``[(asof, row positions)]``: dates grouped so each group spans <= ``span_days`` days;
    ``asof`` is the group's first date at 00:00 UTC."""
    ko = pd.to_datetime(kickoffs, utc=True).reset_index(drop=True)
    days = ko.dt.normalize()
    groups: list[tuple[pd.Timestamp, list[int]]] = []
    for day in sorted(days.unique()):
        day = pd.Timestamp(day)
        pos = [int(i) for i in days.index[days == day]]
        if groups and (day - groups[-1][0]).days <= span_days:
            groups[-1][1].extend(pos)
        else:
            groups.append((day, pos))
    return groups


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--threads", type=int, default=ms.THREADS)
    ap.add_argument("--model-version", default=models.CURRENT_VERSION)
    ap.add_argument("--span-days", type=int, default=3)
    ap.add_argument("--max-groups", type=int, default=0, help="only the latest N matchdays")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    conn = store.connect(store.default_path(args.data_dir))
    res = pd.read_sql_query(
        "SELECT r.match_key, r.div, r.home, r.away, r.kickoff_utc FROM results r "
        "WHERE r.match_key NOT IN (SELECT match_key FROM prediction_log) ORDER BY r.kickoff_utc",
        conn,
    )
    if res.empty:
        print("backfill_retro: nothing to do (no finished matches without a prediction)",
              file=sys.stderr)
        return 0
    res["kickoff_utc"] = pd.to_datetime(res["kickoff_utc"], utc=True)
    res["league"] = res["div"].map(lambda d: DIV_NAMES.get(d, d))
    groups = matchday_groups(res["kickoff_utc"], args.span_days)
    if args.max_groups:
        groups = groups[-args.max_groups:]
    played = ms.played_matches(args.data_dir)
    feats = ms.build_features(played, args.data_dir)
    table = ms.training_table(played, feats, ms.load_market_pre(args.data_dir))
    print(f"backfill_retro: {sum(len(g[1]) for g in groups)} matches in {len(groups)} matchdays",
          file=sys.stderr)
    total = {"run": 0}
    for n, (asof, pos) in enumerate(groups, 1):
        fx = res.iloc[pos].reset_index(drop=True)
        cut_played = played.loc[played["kickoff_utc"] < asof]
        cut_table = table.loc[table["kickoff_utc"] < asof]
        recs = models.predict(
            args.model_version, args.data_dir, cut_played, fx, asof, threads=args.threads,
            table=cut_table,
        )
        log_rows, _ = prun.records_to_rows(recs, fx, asof, args.model_version, source="retro")
        if args.dry_run:
            counts = {"run": len(log_rows)}
        else:
            counts = store.append_predictions(conn, log_rows)
        total["run"] += counts["run"]
        print(f"backfill_retro: [{n}/{len(groups)}] asof {asof:%Y-%m-%d}: {len(fx)} matches, "
              f"{counts}", file=sys.stderr, flush=True)
    conn.close()
    print(f"backfill_retro: wrote {total['run']} retro rows", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # one-line error contract
        print(f"backfill_retro: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
