"""Predict every upcoming Premier League / La Liga fixture (winner + scoreline) and log it.

    uv run python scripts/predict_upcoming.py                 # refresh, predict, log
    uv run python scripts/predict_upcoming.py --no-refresh    # reuse the football-data cache
    uv run python scripts/predict_upcoming.py --dry-run       # predict and print, write nothing
    uv run python scripts/predict_upcoming.py --home Arsenal --away Chelsea   # one match

One price-free model for every fixture: :data:`fedge.predict.models.CURRENT_VERSION` (non-anchored
``lgb_xg`` 1X2 + a Dixon-Coles scoreline grid reconciled to it). No bookmaker price is read.
Writes ``data/predictions.sqlite``: the ``upcoming`` display cache (replaced each run) and the
append-only ``prediction_log`` (freeze rule, d7/d3 labels - see ``fedge.accuracy.store``).
This script and ``update_results.py`` are the only writers; the dashboard is read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedge.accuracy import store  # noqa: E402
from fedge.ingest import schedule as sch  # noqa: E402
from fedge.paper import model_state  # noqa: E402
from fedge.predict import models  # noqa: E402
from fedge.predict import run as prun  # noqa: E402
from fedge.predict.upcoming import load_desk_divs  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--leagues", default=str(ROOT / "config" / "leagues.toml"))
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=model_state.THREADS)
    ap.add_argument("--model-version", default=models.CURRENT_VERSION)
    ap.add_argument("--no-refresh", action="store_true", help="do not re-download football-data")
    ap.add_argument("--no-refresh-schedule", action="store_true",
                    help="use the cached full-season schedule instead of re-downloading it")
    ap.add_argument("--dry-run", action="store_true", help="predict and print, write nothing")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    ap.add_argument("--home", help="single-match lookup: home team name (as in the data)")
    ap.add_argument("--away", help="single-match lookup: away team name")
    args = ap.parse_args(argv)
    if bool(args.home) != bool(args.away):
        ap.error("--home and --away must be given together")
    return args


def print_rows(rows: list[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        top = json.loads(r["top_scores"])
        print(
            f"{r['kickoff_utc']}  {r['league']:<24s} {r['home']} v {r['away']}\n"
            f"    1X2 {r['p_home'] * 100:5.1f}% / {r['p_draw'] * 100:5.1f}% / "
            f"{r['p_away'] * 100:5.1f}%   predicted {r['pred_outcome']} "
            f"{r['pred_score_home']}-{r['pred_score_away']} ({r['pred_score_p'] * 100:.1f}%)\n"
            f"    xG {r['xg_home']:.2f} - {r['xg_away']:.2f}   O2.5 {r['p_over25'] * 100:.1f}%   "
            f"BTTS {r['p_btts'] * 100:.1f}%   top: "
            + ", ".join(f"{t['score']} {t['p'] * 100:.1f}%" for t in top[:3])
            + (f"   [{r['note']}]" if r.get("note") else "")
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    divs = load_desk_divs(args.leagues)
    if not args.no_refresh:
        model_state.refresh_ingest(args.data_dir, args.leagues, args.delay)
    played = model_state.played_matches(args.data_dir)
    res = sch.load_schedule(
        divs, args.data_dir, now=now, refresh=not args.no_refresh_schedule, delay=args.delay
    )
    for w in res.warnings + [f"UNMAPPED team name: {u}" for u in res.unmapped]:
        print(f"predict_upcoming: schedule: {w}", file=sys.stderr)
    fixtures = prun.upcoming_fixtures(res, now)
    if args.home:
        want = (args.home.strip().lower(), args.away.strip().lower())
        fixtures = fixtures.loc[
            (fixtures["home"].str.lower() == want[0]) & (fixtures["away"].str.lower() == want[1])
        ]
        if fixtures.empty:
            print(f"predict_upcoming: no upcoming fixture {args.home} v {args.away}",
                  file=sys.stderr)
            return 1
    log_rows, shown = prun.predict_all(
        args.data_dir, played, fixtures, now, args.model_version, args.threads
    )
    if args.dry_run or args.home:
        print_rows(shown, args.json)
    by_div = {d: int((fixtures["div"] == d).sum()) for d in divs}
    print(
        f"predict_upcoming: {len(fixtures)} upcoming fixtures {by_div}, {len(log_rows)} predicted "
        f"with {args.model_version}; schedule sources {res.sources}"
        + (" STALE" if res.stale else ""),
        file=sys.stderr,
    )
    if args.dry_run or args.home:
        return 0
    if not log_rows:
        return 0
    conn = store.connect(store.default_path(args.data_dir))
    try:
        meta = prun.schedule_meta(res)
        meta["last_run_utc"] = store.iso(now)
        store.replace_upcoming(conn, shown, meta)
        counts = store.append_predictions(conn, log_rows)
    finally:
        conn.close()
    print(f"predict_upcoming: wrote {len(shown)} upcoming rows; prediction_log {counts}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # one-line error contract
        print(f"predict_upcoming: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
