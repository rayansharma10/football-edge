"""Phase 6 settlement: refresh results, settle finished bets, attach the closing price and log-CLV.

    uv run python scripts/paper_settle.py
    uv run python scripts/paper_settle.py --no-refresh

1. ``fedge.ingest.football_data.ingest`` refreshes the season CSVs (the current one is
   re-downloaded, finished seasons come from the raw cache), so the results *and* the closing
   prices (``BFEC*``, ``AvgC*``) of the just-played fixtures are available.
2. Every unsettled bet whose match now has a result is settled: ``pnl`` after 6% commission on net
   winnings (``fedge.paper.settle``), the quoted closing price and its source, and the log-CLV
   against the margin-free (power) closing probability.
3. STDOUT: a digest of the newly settled bets plus running totals (settled count, mean log-CLV with
   a match-clustered 95% CI once n >= 30, flat ROI) — printed **only** when something settled.
   Errors exit non-zero with one line on stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedge.backtest.stats import bootstrap_ci  # noqa: E402
from fedge.paper import ledger, model_state, risk, settle  # noqa: E402
from fedge.paper.settle import SELS  # noqa: E402

CI_MIN_N = 30


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--limits", default=str(ROOT / "config" / "limits.toml"))
    ap.add_argument("--leagues", default=str(ROOT / "config" / "leagues.toml"))
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--no-refresh", action="store_true")
    return ap.parse_args(argv)


def load_odds(data_dir: Path | str = "data") -> pd.DataFrame:
    """The long odds table from the ingest cache."""
    return pd.read_parquet(Path(data_dir) / "interim" / "odds.parquet")


def load_results(played: pd.DataFrame) -> pd.DataFrame:
    """``match_id -> (FTHG, FTAG, FTR)`` for finished matches."""
    return played.set_index("match_id")[["FTHG", "FTAG", "FTR"]]


def close_lookup(odds: pd.DataFrame) -> dict:
    """``{market: (probabilities frame, wide quoted prices per book)}`` for the closing prices."""
    return {
        market: (settle.closing_probabilities(odds, market), odds) for market in SELS
    }


def closing_info(cache: dict, market: str, match_id: str, selection: str) -> dict | None:
    """Quoted closing price + margin-free closing probability of one selection, or None."""
    probs, odds = cache[market]
    if match_id not in probs.index:
        return None
    row = probs.loc[match_id]
    source = str(row["source"])
    wide = settle.wide_prices(odds, source, market, "close")
    if match_id not in wide.index:
        return None
    return {
        "price": float(wide.loc[match_id, selection]),
        "source": source,
        "p_fair": float(row[selection]),
    }


def running_totals(settled: pd.DataFrame) -> str:
    """One-line running totals for the stdout digest (CLV CI only once n >= ``CI_MIN_N``)."""
    n = len(settled)
    if n == 0:
        return "totals: n=0 settled"
    parts = [f"totals: n={n} settled"]
    clv = settled["log_clv"].dropna()
    if len(clv):
        if len(clv) >= CI_MIN_N:
            ci = bootstrap_ci(
                clv.to_numpy(dtype=float), settled.loc[clv.index, "match_key"].to_numpy()
            )
            parts.append(
                f"mean log-CLV {ci['mean'] * 100:+.2f}% "
                f"[{ci['lo'] * 100:+.2f}, {ci['hi'] * 100:+.2f}] (n={len(clv)})"
            )
        else:
            parts.append(
                f"mean log-CLV {clv.mean() * 100:+.2f}% (no CI yet, n={len(clv)} < {CI_MIN_N})"
            )
    turnover = float(settled["stake_units"].sum())
    roi = float(settled["pnl"].sum()) / turnover * 100
    parts.append(f"flat ROI {roi:+.2f}% on {turnover:.2f}u")
    return ", ".join(parts)


def format_digest(new: list[dict], settled: pd.DataFrame, now) -> str:
    """Digest of the newly settled bets, then the running totals."""
    day = pd.Timestamp(now).tz_convert("Australia/Sydney").strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"fedge settled {day} | {len(new)} new settlement(s)"]
    for r in new:
        clv = "n/a" if r["log_clv"] is None else f"{r['log_clv'] * 100:+.2f}%"
        lines.append(
            f"  {r['label']} | {'WON' if r['won'] else 'lost'} | {r['pnl']:+.2f}u | CLV {clv}"
        )
    lines.append("  " + running_totals(settled))
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    limits = risk.load_limits(args.limits)
    conn = ledger.connect(ledger.default_path(args.data_dir))
    pending = ledger.unsettled_bets(conn)
    if pending.empty:
        print("paper_settle: no unsettled bets", file=sys.stderr)
        return 0

    if not args.no_refresh:
        model_state.refresh_ingest(args.data_dir, args.leagues, args.delay)
    played = model_state.played_matches(args.data_dir)
    results = load_results(played)
    cache = close_lookup(load_odds(args.data_dir))

    now = pd.Timestamp.now(tz="UTC")
    now_iso = now.isoformat()
    rows, skipped = [], 0
    for bet in pending.to_dict("records"):
        mid = bet["match_key"]
        if mid not in results.index:
            skipped += 1
            continue
        res = results.loc[mid]
        close = closing_info(cache, bet["market"], mid, bet["selection"])
        row = settle.settle_bet(
            bet,
            float(res["FTHG"]),
            float(res["FTAG"]),
            res["FTR"],
            now_iso,
            commission=limits.commission,
            close=close,
        )
        row["label"] = (
            f"{bet['div']} | {bet['home']} v {bet['away']} | {bet['selection']} @ "
            f"{bet['price_taken']:.2f}"
        )
        rows.append(row)

    new = ledger.record_settlements(conn, rows)
    if not new:
        print(
            f"paper_settle: nothing newly settled ({len(pending)} open, "
            f"{skipped} without a result)",
            file=sys.stderr,
        )
        return 0
    settled = ledger.settled_frame(conn)
    print(format_digest(new, settled, now))
    print(
        f"paper_settle: settled {len(new)} of {len(pending)} open bets ({skipped} still without "
        f"a result); ledger holds {len(settled)} settled bets",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"paper_settle: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
