"""Team-name normalisation across sources (football-data, Understat, Betfair)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ALIASES_CSV = Path(__file__).resolve().parents[2] / "config" / "team_aliases.csv"
SOURCES = ("football_data", "understat", "betfair")


@lru_cache(maxsize=4)
def load_aliases(path: Path = ALIASES_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def canon_map(source: str, path: Path = ALIASES_CSV) -> dict[str, str]:
    """Return {source_name: canonical_id}. Raises if a name maps to two ids."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    df = load_aliases(path)
    out: dict[str, str] = {}
    for cid, names in zip(df["canonical_id"], df[source], strict=True):
        for name in filter(None, (n.strip() for n in names.split("|"))):
            if out.get(name, cid) != cid:
                raise ValueError(f"{source} name {name!r} maps to {out[name]} and {cid}")
            out[name] = cid
    return out
