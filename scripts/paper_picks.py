"""Phase 6 paper picks: download fixtures, score them, record shadow/paper bets.

    uv run python scripts/paper_picks.py                 # the cron/daily entry point
    uv run python scripts/paper_picks.py --dry-run       # score only, write nothing
    uv run python scripts/paper_picks.py --no-refresh    # skip the football-data download

What it does, in order:

1. ``fedge.ingest.football_data.ingest`` refreshes the results (current season re-downloaded,
   finished seasons from the raw cache) and ``fedge.paper.model_state`` rebuilds the Phase 4
   feature table for the played matches, cached by a hash of those results.
2. Both fixture files are downloaded (``fixtures.csv``, ``new_league_fixtures.csv``) and parsed.
   Only divisions the model was trained on can be scored; fixtures of the extra leagues are counted
   and reported.
3. The model state is refreshed for the fixtures: each fixture gets its own causal feature sweep
   over the matches before its kickoff, and the two models named in ``config/strategy.toml`` are
   refit with the documented Phase 4 protocol (:func:`fedge.models.gbm.fit_live`).
4. Model and de-margined snapshot price are pooled with the configured weight; every
   (fixture, market, selection) is written to the ``snapshots`` table, and the bets that clear the
   Gate 0 (or shadow) threshold are written to ``bets`` with the stake caps of
   ``config/limits.toml`` enforced by ``fedge.paper.risk``.

STDOUT CONTRACT: a short Slack-ready digest (header + one line per *new* bet) when new bets were
placed, and **nothing** otherwise. Errors exit non-zero with a single line on stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedge.paper import fixtures as fx  # noqa: E402
from fedge.paper import ledger, model_state, picks, risk  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--config", default=str(ROOT / "config" / "strategy.toml"))
    ap.add_argument("--limits", default=str(ROOT / "config" / "limits.toml"))
    ap.add_argument("--gates", default=str(ROOT / "reports" / "gate0.md"))
    ap.add_argument("--leagues", default=str(ROOT / "config" / "leagues.toml"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between HTTP requests")
    ap.add_argument("--threads", type=int, default=model_state.THREADS)
    ap.add_argument("--no-refresh", action="store_true", help="do not re-download football-data")
    ap.add_argument("--cached-fixtures", action="store_true", help="use the cached fixture files")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="score and report, write nothing to the ledger (also reports past fixtures)",
    )
    return ap.parse_args(argv)


def _fixture_inputs(args):
    """Download (or reuse) both fixture files and parse them into frames."""
    path, new_path = fx.fetch_fixture_files(
        args.data_dir, delay=args.delay, refresh=not args.cached_fixtures
    )
    raw = fx.read_fixture_csv(path)
    fixtures = fx.parse_fixtures(raw)
    rows = fx.market_rows(raw, fixtures, phase="pre")
    new = fx.parse_new_league(fx.read_fixture_csv(new_path))
    return fixtures, rows, new


def main(argv=None) -> int:
    args = parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")

    strategy = picks.load_strategy(args.config)
    limits = risk.load_limits(args.limits)
    conn = ledger.connect(ledger.default_path(args.data_dir))
    state = risk.daily_state(conn, limits, args.data_dir, mode=str(strategy["mode"]))

    if not args.no_refresh:
        model_state.refresh_ingest(args.data_dir, args.leagues, args.delay)
    played = model_state.played_matches(args.data_dir)
    modelled = set(played["div"].astype(str).unique())

    fixtures, rows, new = _fixture_inputs(args)
    fix_modelled = fixtures[fixtures["div"].isin(modelled)].copy()
    upcoming = fix_modelled[fix_modelled["kickoff_utc"] > now]
    rows_up = rows[rows["match_key"].isin(set(upcoming["match_key"]))]
    extra = {
        "fixtures_total": int(len(fixtures)),
        "modelled_fixtures": int(len(fix_modelled)),
        "modelled_upcoming": int(len(upcoming)),
        "unmodelled_fixtures": int(len(fixtures) - len(fix_modelled)),
        "new_league_fixtures": int(len(new)),
    }
    print(
        "paper_picks: "
        + ", ".join(f"{k}={v}" for k, v in extra.items())
        + f", unmodelled_divs={sorted(set(fixtures['div']) - modelled)}"
        + f", new_leagues={sorted(set(new['league'])) if len(new) else []}",
        file=sys.stderr,
    )

    if not args.dry_run and (rows_up.empty or upcoming.empty):
        print(
            "paper_picks: no upcoming fixture in a modelled division had a complete price; "
            "0 bets, 0 snapshots (see reports/weekly for the ledger totals)",
            file=sys.stderr,
        )
        return 0

    scored = fix_modelled if args.dry_run else upcoming
    rows_scored = rows if args.dry_run else rows_up
    if rows_scored.empty:
        print("paper_picks: no priceable fixture to score", file=sys.stderr)
        return 0
    scored = scored[scored["match_key"].isin(set(rows_scored["match_key"]))]

    edges = model_state.score_all(
        args.data_dir,
        strategy,
        played,
        scored,
        rows_scored,
        picks.pool_weights(strategy),
        threads=args.threads,
    )
    if edges.empty:
        print("paper_picks: no fixture could be scored", file=sys.stderr)
        return 0

    cfg_hash = picks.file_hash(args.config)
    gate_hash = picks.gate0_hash(args.gates)
    experiment = f"{strategy['mode']}-{cfg_hash[:8]}"
    snapshots = picks.snapshot_rows(edges, scored, now)
    remaining = 0 if state["blocked"] else state["remaining"]
    bets = picks.plan_bets(
        edges, scored, strategy, limits, remaining, now, cfg_hash, gate_hash, experiment
    )

    if args.dry_run:
        print(f"DRY RUN (no writes) | mode={strategy['mode']} | scored={len(scored)} fixtures")
        print(f"would record {len(snapshots)} snapshot rows, {len(bets)} bets")
        if bets:
            print(picks.format_digest(bets, strategy, now))
        else:
            top = edges.sort_values("edge", ascending=False).head(5)
            print("no candidate clears the shadow threshold; best edges:")
            for r in top.itertuples(index=False):
                print(
                    f"  {r.match_key[:8]} {r.market:4s} {r.selection:5s} "
                    f"edge {r.edge * 100:+.2f}%"
                )
        return 0

    n_snap = ledger.insert_snapshots(conn, snapshots)
    inserted = ledger.insert_bets(conn, bets)
    if state["blocked"]:
        print(f"paper_picks: no bets placed: {state['reason']}", file=sys.stderr)
    if inserted:
        print(picks.format_digest(inserted, strategy, now))
    print(
        f"paper_picks: recorded {n_snap} snapshots, {len(inserted)} new bets "
        f"({len(bets)} planned, {remaining} of {limits.max_bets_per_day} daily slots left)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # one-line error contract
        print(f"paper_picks: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
