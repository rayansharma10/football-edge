"""CLI: download + parse football-data.co.uk into Parquet and data/fedge.duckdb."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedge.ingest.football_data import ingest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "leagues.toml"))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between HTTP requests")
    ap.add_argument("--div", action="append", help="restrict to division(s), repeatable")
    ap.add_argument("--season", action="append", help="restrict to season(s), e.g. 2024/25")
    ap.add_argument("--offline", action="store_true", help="parse cached raw files only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matches, odds, status = ingest(args.config, args.data_dir, args.delay, args.div,
                                   args.season, args.offline)
    counts: dict[str, int] = {}
    for s in status:
        counts[s.status] = counts.get(s.status, 0) + 1
    print(f"files: {counts}")
    print(f"matches rows: {len(matches)}")
    print(f"odds rows: {len(odds)}")


if __name__ == "__main__":
    main()
