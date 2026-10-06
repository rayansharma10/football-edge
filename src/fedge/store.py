"""Parquet + DuckDB access with available_at discipline."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from fedge.schemas import matches_schema, odds_schema


def paths(data_dir: Path | str = "data") -> dict[str, Path]:
    d = Path(data_dir)
    return {
        "matches": d / "interim" / "matches.parquet",
        "odds": d / "interim" / "odds.parquet",
        "db": d / "fedge.duckdb",
    }


def write_tables(matches: pd.DataFrame, odds: pd.DataFrame, data_dir: Path | str = "data") -> None:
    """Validate with pandera, write Parquet, then (re)load DuckDB tables. Idempotent."""
    matches = matches_schema.validate(matches)
    odds = odds_schema.validate(odds)
    p = paths(data_dir)
    p["matches"].parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(p["matches"], index=False, compression="zstd")
    odds.to_parquet(p["odds"], index=False, compression="zstd")
    con = duckdb.connect(str(p["db"]))
    try:
        for name in ("matches", "odds"):
            con.execute(
                f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet(?)",
                [p[name].as_posix()],
            )
    finally:
        con.close()


def connect(data_dir: Path | str = "data", read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(paths(data_dir)["db"]), read_only=read_only)
