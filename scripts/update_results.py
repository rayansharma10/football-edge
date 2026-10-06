"""Fill the ``results`` table with finished Premier League / La Liga matches.

    uv run python scripts/update_results.py               # refresh schedule + football-data, write
    uv run python scripts/update_results.py --no-refresh --no-refresh-schedule   # cached inputs
    uv run python scripts/update_results.py --dry-run

Primary source: the full-season schedule (fixturedownload.com / openfootball scores); the
football-data.co.uk season CSVs cross-check it and fill any gap. Idempotent: an already stored
match is never changed (a differing re-fetch is reported as a conflict). Unmapped team names are
logged to stderr. Writes ``data/predictions.sqlite`` (``results``); the dashboard stays read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedge.accuracy import results as res_mod  # noqa: E402
from fedge.accuracy import store  # noqa: E402
from fedge.ingest import schedule as sch  # noqa: E402
from fedge.paper import model_state  # noqa: E402
from fedge.predict.upcoming import load_desk_divs  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--leagues", default=str(ROOT / "config" / "leagues.toml"))
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--no-refresh", action="store_true", help="do not re-download football-data")
    ap.add_argument("--no-refresh-schedule", action="store_true",
                    help="use the cached schedule instead of re-downloading it")
    ap.add_argument("--dry-run", action="store_true", help="assemble and print, write nothing")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    divs = load_desk_divs(args.leagues)
    if not args.no_refresh:
        model_state.refresh_ingest(args.data_dir, args.leagues, args.delay)
    played = model_state.played_matches(args.data_dir)
    sres = sch.load_schedule(
        divs, args.data_dir, now=now, refresh=not args.no_refresh_schedule, delay=args.delay
    )
    for w in sres.warnings:
        print(f"update_results: schedule: {w}", file=sys.stderr)
    for u in sres.unmapped:
        print(f"update_results: UNMAPPED team name: {u}", file=sys.stderr)
    rows, warns = res_mod.build_results(sres.fixtures, sres.sources, played, divs, now)
    for w in warns:
        print(f"update_results: {w}", file=sys.stderr)
    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"update_results: {len(rows)} finished matches found {by_src}", file=sys.stderr)
    if args.dry_run:
        for r in rows[:5]:
            print(r)
        return 0
    conn = store.connect(store.default_path(args.data_dir))
    try:
        counts = store.upsert_results(conn, rows)
        total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    finally:
        conn.close()
    print(f"update_results: {counts}; {total} results stored", file=sys.stderr)
    if counts["conflict"]:
        print("update_results: WARNING stored scores differ from the re-fetch (kept stored)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # one-line error contract
        print(f"update_results: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
