"""Score the upcoming Premier League / La Liga fixtures into the ``predictions`` table.

    uv run python scripts/predict_upcoming.py                # full run (ingest refresh + write)
    uv run python scripts/predict_upcoming.py --no-refresh   # reuse the ingest cache
    uv run python scripts/predict_upcoming.py --cached-fixtures --dry-run
    uv run python scripts/predict_upcoming.py --home Arsenal --away Chelsea   # single match

The heavy lifting is the same live pipeline the paper desk uses
(:func:`fedge.paper.model_state.score_all`: lgbd_xg 1X2, market-anchored) plus the Dixon-Coles
scoreline grid reconciled to that 1X2 (:mod:`fedge.predict.scoreline`). The dashboard only reads
the resulting ``predictions`` table; this script is the only writer.

The output is display-only: it never touches bets, snapshots, stakes, or the paper-desk digest.
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

from fedge.models import dixon_coles as dc  # noqa: E402
from fedge.paper import fixtures as fx  # noqa: E402
from fedge.paper import ledger, model_state, picks  # noqa: E402
from fedge.predict import upcoming as pred  # noqa: E402
from fedge.predict.scoreline import predict_match  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--config", default=str(ROOT / "config" / "strategy.toml"))
    ap.add_argument("--leagues", default=str(ROOT / "config" / "leagues.toml"))
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=model_state.THREADS)
    ap.add_argument("--no-refresh", action="store_true", help="do not re-download football-data")
    ap.add_argument("--cached-fixtures", action="store_true", help="use the cached fixture files")
    ap.add_argument(
        "--fixtures-csv",
        help="read the fixture list from this file instead of downloading (testing/staging)",
    )
    ap.add_argument(
        "--no-refresh-schedule", action="store_true",
        help="use the cached full-season schedule instead of re-downloading it",
    )
    ap.add_argument("--dry-run", action="store_true", help="score and print, write nothing")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON (with --dry-run)")
    ap.add_argument("--home", help="single-match lookup: home team name (as in the data)")
    ap.add_argument("--away", help="single-match lookup: away team name")
    args = ap.parse_args(argv)
    if bool(args.home) != bool(args.away):
        ap.error("--home and --away must be given together")
    return args


def fixture_inputs(args):
    """Both fixture files (downloaded or cached) and the parsed frames."""
    if args.fixtures_csv:
        path = Path(args.fixtures_csv)
    else:
        path, _new_path = fx.fetch_fixture_files(
            args.data_dir, delay=args.delay, refresh=not args.cached_fixtures
        )
    raw = fx.read_fixture_csv(path)
    fixtures = fx.parse_fixtures(raw)
    return fixtures, fx.market_rows(raw, fixtures, phase="pre")


def print_rows(rows, as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        head = f"{r['kickoff_utc']}  {r['league']:<24s} {r['home']} v {r['away']}"
        if r["status"] != "modelled":
            print(f"{head}\n    not modelled: {r['reason']}")
            continue
        d = json.loads(r["detail"])
        top = json.loads(r["top_scores"])
        mkt = "   [price-free estimate]" if r["mkt_home"] is None else (
            f"   market {r['mkt_home'] * 100:5.1f}% / {r['mkt_draw'] * 100:5.1f}% / "
            f"{r['mkt_away'] * 100:5.1f}%"
        )
        print(
            f"{head}\n"
            f"    1X2 {r['p_home'] * 100:5.1f}% / {r['p_draw'] * 100:5.1f}% / "
            f"{r['p_away'] * 100:5.1f}%{mkt}\n"
            f"    xG {r['xg_home']:.2f} - {r['xg_away']:.2f}   O2.5 {r['p_over25'] * 100:.1f}%   "
            f"BTTS {r['p_btts'] * 100:.1f}%\n"
            f"    top scores {', '.join(f'{t['score']} {t['p'] * 100:.1f}%' for t in top)}\n"
            f"    {d['method']}"
            + (f"   [{d['fallback']}]" if d["fallback"] else "")
        )


def _team_division(played: pd.DataFrame, team_lower: str) -> str | None:
    """Division the team appears in (any side), most recent row first."""
    mask = (
        played["home"].str.lower().eq(team_lower) | played["away"].str.lower().eq(team_lower)
    )
    g = played.loc[mask].sort_values("kickoff_utc")
    return str(g["div"].iloc[-1]) if len(g) else None


def lookup(args, played: pd.DataFrame, fixtures: pd.DataFrame, edges: pd.DataFrame, now) -> int:
    """``--home``/``--away``: score one named match (from the live edges when present)."""
    want = (args.home.strip().lower(), args.away.strip().lower())
    fix = fixtures.loc[
        (fixtures["home"].str.lower() == want[0]) & (fixtures["away"].str.lower() == want[1])
    ]
    div = str(fix["div"].iloc[0]) if len(fix) else pred.DESK_DIVS[0]
    model_p, mkt = pd.DataFrame(), pd.DataFrame()
    if len(edges):
        e = edges.loc[(edges["market"] == "1x2")
                      & edges["match_key"].isin(fix["match_key"])]
        model_p = e.pivot_table(index="match_key", columns="selection", values="model_prob")
        mkt = e.pivot_table(index="match_key", columns="selection", values="market_prob")
    if len(fix) and len(model_p):
        row = fix.loc[fix["match_key"].isin(model_p.index)].iloc[0]
        div = str(row["div"])
        key = row["match_key"]
        tgt = model_p.loc[key, ["H", "D", "A"]].to_numpy(dtype=float)
        mk = mkt.loc[key, ["H", "D", "A"]].to_numpy(dtype=float)
        params = dc.fit_dc(played.loc[played["div"] == div], ref=now)
        if params is None:
            lh, la, rho = 1.5, 1.2, 0.0
        else:
            lh, la, _ = pred.dc_lambdas(params, row["home"], row["away"])
            rho = params.rho
        pm = predict_match(lh, la, rho, tgt)
        src = "live lgbd_xg 1X2 (market-anchored) + Dixon-Coles scoreline grid"
    else:
        # no priced upcoming fixture: pure Dixon-Coles, clearly labelled lower confidence
        div = _team_division(played, want[0]) or div
        params = dc.fit_dc(played.loc[played["div"] == div], ref=now)
        if params is None:
            print(f"predict_upcoming: no Dixon-Coles fit available for division {div}",
                  file=sys.stderr)
            return 1
        lh, la, _ = pred.dc_lambdas(params, args.home, args.away)
        pm = predict_match(lh, la, params.rho)
        mk = None
        src = ("Dixon-Coles ONLY (no priced upcoming fixture; the 1X2 is NOT the lgbd_xg model) "
               "- lower confidence")
    out = {
        "home": args.home, "away": args.away, "source": src, "division": div,
        "p_home": pm["p_home"], "p_draw": pm["p_draw"], "p_away": pm["p_away"],
        "xg_home": pm["xg_home"], "xg_away": pm["xg_away"],
        "p_over25": pm["p_over25"], "p_btts": pm["p_btts"],
        "top_scores": pm["top_scores"], "grid": pm["grid"], "method": pm["method"],
    }
    if mk is not None:
        out["market"] = {"home": float(mk[0]), "draw": float(mk[1]), "away": float(mk[2])}
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"{args.home} v {args.away}  ({div})")
    print(f"  1X2  {out['p_home'] * 100:5.1f}% / {out['p_draw'] * 100:5.1f}% / "
          f"{out['p_away'] * 100:5.1f}%")
    if mk is not None:
        print(f"  mkt  {mk[0] * 100:5.1f}% / {mk[1] * 100:5.1f}% / {mk[2] * 100:5.1f}%")
    print(f"  xG   {out['xg_home']:.2f} - {out['xg_away']:.2f}   "
          f"O2.5 {out['p_over25'] * 100:.1f}%   BTTS {out['p_btts'] * 100:.1f}%")
    print("  top scores: " + ", ".join(f"{t['score']} {t['p'] * 100:.1f}%" for t in
                                        out["top_scores"]))
    print(f"  method: {pm['method']}")
    print(f"  source: {src}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    strategy = picks.load_strategy(args.config)
    divs = pred.load_desk_divs(args.leagues)

    if not args.no_refresh:
        model_state.refresh_ingest(args.data_dir, args.leagues, args.delay)
    played = model_state.played_matches(args.data_dir)
    fixtures, rows = fixture_inputs(args)
    desk = fixtures.loc[fixtures["div"].isin(set(divs))].copy()
    upcoming = desk.loc[desk["kickoff_utc"] > now]
    rows_up = rows.loc[rows["match_key"].isin(set(upcoming["match_key"]))]

    if args.home:
        edges = pd.DataFrame()
        if not rows_up.empty:
            edges = model_state.score_all(
                args.data_dir, strategy, played, upcoming, rows_up,
                picks.pool_weights(strategy), threads=args.threads,
            )
        return lookup(args, played, desk, edges, now)

    edges = pd.DataFrame()
    if not rows_up.empty:
        edges = model_state.score_all(
            args.data_dir, strategy, played, upcoming, rows_up,
            picks.pool_weights(strategy), threads=args.threads,
        )
    built, meta, res = pred.build_all(
        args.data_dir, played, desk, edges, now, divs,
        refresh=not args.cached_fixtures and not args.no_refresh_schedule,
        delay=args.delay, threads=args.threads,
    )
    print_rows(built, args.json)
    for w in res.warnings + [f"UNMAPPED team name: {u}" for u in res.unmapped]:
        print(f"predict_upcoming: schedule: {w}", file=sys.stderr)
    n_mod = sum(r["status"] == "modelled" for r in built)
    n_free = sum(r.get("model_kind") == pred.KIND_PRICE_FREE for r in built)
    by_div = {d: sum(r["div"] == d for r in built) for d in divs}
    print(
        f"predict_upcoming: {len(built)} upcoming fixtures in {list(divs)} {by_div} "
        f"({n_mod} modelled: {n_mod - n_free} priced lgbd_xg + {n_free} price-free lgb_xg, "
        f"{len(built) - n_mod} not modelled); schedule sources {res.sources}"
        + (" STALE" if res.stale else ""),
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    if not built:
        return 0
    conn = ledger.connect(ledger.default_path(args.data_dir))
    n = pred.write_predictions(conn, built, meta)
    print(f"predict_upcoming: wrote {n} rows to predictions", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # one-line error contract
        print(f"predict_upcoming: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
