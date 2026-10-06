"""Append/replace the Understat join-rate section in reports/data_coverage.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedge.ingest.understat import join_rates  # noqa: E402

MARK = "## Understat xG join rate"


def main() -> None:
    m = pd.read_parquet(ROOT / "data" / "interim" / "matches.parquet")
    x = pd.read_parquet(ROOT / "data" / "interim" / "match_xg.parquet")
    g = join_rates(m, x)
    piv = g.pivot(index="season", columns="div", values="rate")
    tot = g.groupby("div")[["joined", "matches"]].sum()
    lines = [MARK, "",
             "Played matches (FTR present) in E0/SP1/D1/I1/F1 from 2014/15, joined to Understat "
             "team-match xG (`match_xg` table). Join key: (div, UK-local date, canonical home, "
             "canonical away); fallback on (div, season, home, away) when Understat's date is "
             "within 3 days and the full-time score agrees "
             f"({int((x.match_key == 'fixture').sum())} of {len(x)} rows use the fallback).", "",
             f"Overall: {int(g.joined.sum())}/{int(g.matches.sum())} = "
             f"{g.joined.sum() / g.matches.sum():.2%}", "",
             "| season | " + " | ".join(piv.columns) + " |",
             "|---|" + "---|" * len(piv.columns)]
    for s, row in piv.iterrows():
        lines.append(f"| {s} | " + " | ".join(f"{v:.1%}" for v in row) + " |")
    lines.append("| **all** | " + " | ".join(
        f"{tot.loc[d, 'joined'] / tot.loc[d, 'matches']:.1%}" for d in piv.columns) + " |")
    low = g[g.rate < 0.98]
    lines += ["", f"League-seasons below 98%: {len(low)}"]
    for r in low.itertuples():
        lines.append(f"- {r.div} {r.season}: {r.joined}/{r.matches} ({r.rate:.1%})")
    section = "\n".join(lines) + "\n"
    p = ROOT / "reports" / "data_coverage.md"
    txt = p.read_text(encoding="utf-8")
    if MARK in txt:
        txt = txt[: txt.index(MARK)].rstrip() + "\n\n"
    else:
        txt = txt.rstrip() + "\n\n"
    p.write_text(txt + section, encoding="utf-8")
    print(section)


if __name__ == "__main__":
    main()
