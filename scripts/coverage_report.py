"""Coverage report: per division x season availability of PS/BFE/Avg odds and xG.

Reads data/fedge.duckdb (built by scripts/ingest_football_data.py) and writes
reports/data_coverage.md. Percentages are of all rows in the season file (including
not-yet-played fixtures in the current season).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedge import store  # noqa: E402

COVERAGE_SQL = """
WITH h AS (
    SELECT match_id,
        max(CASE WHEN bookmaker='PS'  AND phase='pre'   THEN 1 ELSE 0 END) AS ps_pre,
        max(CASE WHEN bookmaker='PS'  AND phase='close' THEN 1 ELSE 0 END) AS ps_close,
        max(CASE WHEN bookmaker='BFE' AND phase='pre'   THEN 1 ELSE 0 END) AS bfe_pre,
        max(CASE WHEN bookmaker='BFE' AND phase='close' THEN 1 ELSE 0 END) AS bfe_close,
        max(CASE WHEN bookmaker='Avg' AND phase='close' THEN 1 ELSE 0 END) AS avg_close
    FROM odds WHERE market='1x2' AND selection='H'
    GROUP BY match_id
)
SELECT m.div, m.season, count(*) AS n,
    coalesce(sum(h.ps_pre), 0) AS ps_pre, coalesce(sum(h.ps_close), 0) AS ps_close,
    coalesce(sum(h.bfe_pre), 0) AS bfe_pre, coalesce(sum(h.bfe_close), 0) AS bfe_close,
    coalesce(sum(h.avg_close), 0) AS avg_close,
    count(m.HxG) AS xg
FROM matches m LEFT JOIN h USING (match_id)
GROUP BY m.div, m.season ORDER BY m.div, m.season
"""


def pct(a: int, n: int) -> str:
    return f"{100 * a / n:.0f}%" if n else "-"


def build_report(data_dir: Path | str = "data") -> str:
    con = store.connect(data_dir)
    rows = con.execute(COVERAGE_SQL).fetchall()
    tot = con.execute("SELECT count(*) FROM matches").fetchone()[0]
    odds_n = con.execute("SELECT count(*) FROM odds").fetchone()[0]
    e0 = con.execute(
        "SELECT count(DISTINCT o.match_id), "
        "CAST(timezone('UTC', max(m.kickoff_utc)) AS VARCHAR) FROM odds o "
        "JOIN matches m "
        "USING (match_id) WHERE m.div='E0' AND m.season='2025/26' AND o.bookmaker='PS' "
        "AND o.phase='close' AND o.market='1x2' AND o.selection='H'"
    ).fetchone()
    e0_n = con.execute(
        "SELECT count(*) FROM matches WHERE div='E0' AND season='2025/26'"
    ).fetchone()[0]
    e1_bfe = con.execute(
        "SELECT season, count(DISTINCT match_id) FROM odds o JOIN matches m USING (match_id) "
        "WHERE m.div='E1' AND o.bookmaker='BFE' AND o.phase='close' AND o.market='1x2' "
        "GROUP BY season ORDER BY season"
    ).fetchall()
    xg_seasons = con.execute(
        "SELECT season, count(HxG) FROM matches GROUP BY season HAVING count(HxG) > 0 "
        "ORDER BY season"
    ).fetchall()
    stale_n = con.execute("SELECT count(*) FROM odds WHERE stale").fetchone()[0]
    con.close()

    out = ["# football-data.co.uk data coverage", ""]
    out.append(f"Generated from `data/fedge.duckdb`: {tot} matches, {odds_n} odds rows "
               f"({stale_n} rows flagged `stale`, Pinnacle kickoff >= 2025-07-23).")
    out.append("")
    out.append("Percentages are of all fixture rows in each season file (the current season "
               "includes unplayed fixtures). Prices counted on the 1X2 home selection. "
               "xG = rows with HxG/AxG present.")
    out.append("")
    out.append("| div | season | matches | PS pre | PS close | BFE pre | BFE close "
               "| Avg close | xG |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for div, season, n, a, b, c, d, e, x in rows:
        out.append(f"| {div} | {season} | {n} | {pct(a, n)} | {pct(b, n)} | {pct(c, n)} "
                   f"| {pct(d, n)} | {pct(e, n)} | {pct(x, n)} |")
    out.append("")
    out.append("## Sanity checks against expected facts")
    out.append("")
    last = e0[1][:10] if e0[1] else "n/a"
    out.append(f"- E0 2025/26 Pinnacle 1X2 closing: **{e0[0]}/{e0_n}** rows filled, last "
               f"kickoff with a PS closing price **{last}** (expected ~210/380, last "
               f"2026-01-08).")
    first_bfe = e1_bfe[0][0] if e1_bfe else "none"
    out.append(f"- E1 Betfair Exchange closing first present in season **{first_bfe}** "
               f"(expected 2024/25). Per season: "
               + ", ".join(f"{s}: {c}" for s, c in e1_bfe) + ".")
    out.append("- Seasons with HxG/AxG populated: "
               + (", ".join(f"{s} ({c})" for s, c in xg_seasons) or "none")
               + " (expected: 2026/27).")
    return "\n".join(out) + "\n"


def main() -> None:
    text = build_report(ROOT / "data")
    path = ROOT / "reports" / "data_coverage.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(text[-1500:])


if __name__ == "__main__":
    main()
