"""Fetch (cached) Understat league-season JSON and build the match_xg table."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedge.ingest.understat import ingest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mx, status = ingest(Path(args.data_dir), args.delay, args.offline)
    counts: dict[str, int] = {}
    for s in status:
        counts[s.status] = counts.get(s.status, 0) + 1
    print(f"fetch: {counts}")
    print(f"match_xg rows: {len(mx)}")


if __name__ == "__main__":
    main()
