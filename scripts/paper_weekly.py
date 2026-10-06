"""Phase 6 weekly report: ledger totals, live CLV, ROI, per-league split and Gate 1 progress.

    uv run python scripts/paper_weekly.py

Writes ``reports/weekly/YYYY-MM-DD.md`` (Sydney date) and prints a short Slack-ready summary.

Gate 0 did **not** pass (``reports/gate0.md``): the desk runs in ``shadow`` mode, so every bet it
records is a hypothesis to be measured, not a stake. The report says so at the top and Gate 1
progress counts **paper** settlements only — shadow bets can never move the Gate 1 counter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedge.backtest.stats import bootstrap_ci, simulate_bankroll  # noqa: E402
from fedge.paper import ledger, picks, risk  # noqa: E402

GATE1_BETS = 1000
CI_MIN_N = 30


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "weekly"))
    ap.add_argument("--config", default=str(ROOT / "config" / "strategy.toml"))
    ap.add_argument("--limits", default=str(ROOT / "config" / "limits.toml"))
    ap.add_argument("--now", default=None, help="override the report timestamp (tests)")
    return ap.parse_args(argv)


def metrics(settled: pd.DataFrame, limits: risk.Limits) -> dict:
    """Settled count, mean log-CLV with CI, flat and Kelly ROI and max drawdown."""
    n = len(settled)
    if n == 0:
        return {"n": 0}
    out = {"n": n, "turnover": float(settled["stake_units"].sum())}
    clv = settled["log_clv"].dropna()
    out["clv_n"] = int(len(clv))
    if len(clv):
        out["clv_mean"] = float(clv.mean())
        if len(clv) >= CI_MIN_N:
            ci = bootstrap_ci(
                clv.to_numpy(dtype=float), settled.loc[clv.index, "match_key"].to_numpy()
            )
            out.update({"clv_lo": ci["lo"], "clv_hi": ci["hi"]})
    out["roi_flat"] = float(settled["pnl"].sum()) / out["turnover"] if out["turnover"] else np.nan
    sim = settled.assign(
        date=settled["kickoff_utc"],
        price=settled["price_taken"],
        p=settled["pooled_prob"],
        market_id=settled["match_key"] + "|" + settled["market"],
        won=settled["settled_won"].astype(bool),
    )
    _, flat = simulate_bankroll(sim, mode="flat", max_stake_frac=limits.max_stake_frac,
                                commission=limits.commission)
    _, kelly = simulate_bankroll(
        sim, mode="kelly", kelly_mult=limits.kelly_fraction, max_stake_frac=limits.max_stake_frac,
        commission=limits.commission,
    )
    out.update(
        {
            "roi_kelly": kelly["yield"],
            "max_drawdown": kelly["max_drawdown"],
            "mdd_flat": flat["max_drawdown"],
        }
    )
    return out


def fmt(v, pct=True, nd=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v * 100:+.{nd}f}%" if pct else f"{v:.{nd}f}"


def clv_text(m: dict) -> str:
    if "clv_mean" not in m:
        return "no settled bet has a closing price yet"
    if "clv_lo" in m:
        return (
            f"mean log-CLV {fmt(m['clv_mean'])} "
            f"[{fmt(m['clv_lo'])}, {fmt(m['clv_hi'])}] (n={m['clv_n']})"
        )
    return f"mean log-CLV {fmt(m['clv_mean'])} (no CI yet, n={m['clv_n']} < {CI_MIN_N})"


def build_report(conn, strategy: dict, limits: risk.Limits, now: pd.Timestamp) -> tuple:
    """``(markdown, summary dict)`` for the weekly report."""
    bets = ledger.bets_with_settlements(conn)
    settled = ledger.settled_frame(conn)
    counts = ledger.mode_counts(conn)
    settled_counts = ledger.settled_mode_counts(conn)
    m = metrics(settled, limits)
    m_paper = metrics(ledger.settled_frame(conn, "paper"), limits)
    m_shadow = metrics(ledger.settled_frame(conn, "shadow"), limits)
    snaps = ledger.read_table(conn, "snapshots")
    gate_passed = bool(strategy["gate0_passed"])

    sydney_day = now.tz_convert("Australia/Sydney").strftime("%Y-%m-%d")
    cfg_hash = picks.file_hash(ROOT / "config" / "strategy.toml")
    gate_hash = picks.file_hash(ROOT / "reports" / "gate0.md")
    lines = [
        f"# football-edge paper desk: weekly report {sydney_day}",
        "",
        (
            "**Gate 0 status: NOT passed** (`reports/gate0.md`), so the desk runs in **shadow** "
            "mode: recorded bets are hypothetical. Shadow bets do **not** count towards Gate 1."
            if not gate_passed
            else "**Gate 0 passed**: bets are recorded in `paper` mode."
        ),
        f"Stamped with config hash `{cfg_hash[:16]}` and Gate 0 report hash "
        f"`{gate_hash[:16]}`; experiment `{strategy['mode']}-{cfg_hash[:8]}`.",
        "",
        "## Ledger",
        "",
        f"- Bets: {len(bets)} total ({counts.get('paper', 0)} paper, "
        f"{counts.get('shadow', 0)} shadow)",
        f"- Settled: {len(settled)} ({settled_counts.get('paper', 0)} paper, "
        f"{settled_counts.get('shadow', 0)} shadow)",
        f"- Price snapshots seen: {len(snaps)}",
        f"- Open (kickoff in the future or no result yet): {len(bets) - len(settled)}",
        "",
        "## Live CLV and ROI (settled bets)",
        "",
        f"- {clv_text(m)}",
        f"- Flat ROI {fmt(m.get('roi_flat'))} on {fmt(m.get('turnover'), pct=False)}u turnover; "
        f"Kelly({limits.kelly_fraction:g}) ROI {fmt(m.get('roi_kelly'))}",
        f"- Max drawdown: Kelly {fmt(m.get('max_drawdown'), nd=2)} / flat {fmt(m.get('mdd_flat'))}",
        f"- Paper only: n={m_paper.get('n', 0)}, {clv_text(m_paper)}",
        f"- Shadow only: n={m_shadow.get('n', 0)}, {clv_text(m_shadow)}",
        "",
        "## By league (settled bets)",
        "",
        "| Div | n | mean log-CLV | flat ROI |",
        "|---|---|---|---|",
    ]
    if len(settled):
        for div, g in settled.groupby("div", sort=True, observed=True):
            clv = g["log_clv"].dropna()
            mean = fmt(clv.mean()) if len(clv) else "-"
            turnover = float(g["stake_units"].sum())
            roi = fmt(float(g["pnl"].sum()) / turnover) if turnover else "-"
            lines.append(f"| {div} | {len(g)} | {mean} | {roi} |")
    else:
        lines.append("| - | 0 | - | - |")
    lines += [
        "",
        "## Gate 1 progress",
        "",
        f"- Settled **paper** bets: {settled_counts.get('paper', 0)} / {GATE1_BETS}",
        f"- Settled shadow bets (do not count): {settled_counts.get('shadow', 0)}",
        f"- Bets placed so far: {len(bets)}; snapshots recorded: {len(snaps)}",
        "",
        "## Caveats",
        "",
        "- The desk trades the football-data.co.uk pre-closing snapshot (`BFE`, else `Avg`, else "
        "`Max`); the Pinnacle column is excluded (stale from 2025-07-23).",
        "- The real collection time of the football-data `pre` price is unverified; the snapshot "
        "is stamped with the download time (`seen_utc`), not the kickoff-24h assumption of the "
        "backtest.",
        "- CLV is measured against the margin-free (power de-vig) closing probability, `BFEC*` if "
        "both sides are quoted, else `AvgC*`.",
        "- Shadow bets are logged only when the model is confident enough to bet a fixture whose "
        f"kickoff is in the future; the shadow edge threshold is "
        f"{strategy['shadow_threshold']:.2%}.",
        "",
    ]
    summary = {
        "bets": len(bets),
        "settled": len(settled),
        "settled_paper": settled_counts.get("paper", 0),
        "settled_shadow": settled_counts.get("shadow", 0),
        "snapshots": len(snaps),
        "clv": m.get("clv_mean"),
        "clv_n": m.get("clv_n", 0),
        "roi_flat": m.get("roi_flat"),
        "roi_kelly": m.get("roi_kelly"),
        "gate_passed": gate_passed,
    }
    return "\n".join(lines) + "\n", summary


def format_summary(summary: dict, strategy: dict, limits: risk.Limits) -> str:
    """Short Slack-ready stdout summary."""
    gate = "PASSED" if summary["gate_passed"] else "NOT passed (shadow mode; no Gate 1 credit)"
    clv = "-" if summary["clv"] is None else f"{summary['clv'] * 100:+.2f}%"
    roi = "-" if summary["roi_flat"] is None else f"{summary['roi_flat'] * 100:+.2f}%"
    kroi = "-" if summary["roi_kelly"] is None else f"{summary['roi_kelly'] * 100:+.2f}%"
    return (
        f"fedge weekly: Gate 0 {gate} | bets {summary['bets']} "
        f"({summary['settled']} settled: {summary['settled_paper']} paper / "
        f"{summary['settled_shadow']} shadow) | snapshots {summary['snapshots']} | "
        f"mean log-CLV {clv} (n={summary['clv_n']}) | flat ROI {roi}, "
        f"Kelly({limits.kelly_fraction:g}) ROI {kroi} | "
        f"Gate 1 {summary['settled_paper']}/{GATE1_BETS}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    strategy = picks.load_strategy(args.config)
    limits = risk.load_limits(args.limits)
    conn = ledger.connect(ledger.default_path(args.data_dir))
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    md, summary = build_report(conn, strategy, limits, now)
    out = Path(args.reports) / f"{now.tz_convert('Australia/Sydney').strftime('%Y-%m-%d')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8", newline="\n")
    print(format_summary(summary, strategy, limits))
    print(f"paper_weekly: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI driver
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"paper_weekly: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
