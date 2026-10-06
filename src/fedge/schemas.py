"""pandera schemas (matches, odds, snapshots, bets)."""

from __future__ import annotations

import pandera.pandas as pa

_NF = {"nullable": True, "coerce": True}

MATCH_STATS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

matches_schema = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, unique=True),
        "div": pa.Column(str),
        "season": pa.Column(str, pa.Check.str_matches(r"^\d{4}/\d{2}$")),
        "kickoff_utc": pa.Column("datetime64[ns, UTC]"),
        "time_unknown": pa.Column(bool),
        "home": pa.Column(str),
        "away": pa.Column(str),
        "FTHG": pa.Column("Int64", pa.Check.ge(0), **_NF),
        "FTAG": pa.Column("Int64", pa.Check.ge(0), **_NF),
        "FTR": pa.Column(str, pa.Check.isin(["H", "D", "A"]), nullable=True),
        "HTHG": pa.Column("Int64", pa.Check.ge(0), **_NF),
        "HTAG": pa.Column("Int64", pa.Check.ge(0), **_NF),
        "HTR": pa.Column(str, pa.Check.isin(["H", "D", "A"]), nullable=True),
        "Referee": pa.Column(str, nullable=True),
        **{c: pa.Column("Float64", **_NF) for c in MATCH_STATS},
        "HxG": pa.Column("Float64", **_NF),
        "AxG": pa.Column("Float64", **_NF),
    },
    strict=True,
    ordered=False,
)

odds_schema = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str),
        "bookmaker": pa.Column(str),
        "market": pa.Column(str, pa.Check.isin(["1x2", "ou25", "ah"])),
        "selection": pa.Column(str, pa.Check.isin(["H", "D", "A", "over", "under"])),
        "phase": pa.Column(str, pa.Check.isin(["pre", "close"])),
        "line": pa.Column(float, nullable=True, coerce=True),
        "price": pa.Column(float, pa.Check.gt(1.0), coerce=True),
        "available_at": pa.Column("datetime64[ns, UTC]"),
        "stale": pa.Column(bool),
    },
    strict=True,
    ordered=False,
)
