"""Form / schedule / attack-defence features: as-of equivalence and leakage tests."""

import math

import numpy as np
import pandas as pd
import pytest

from fedge.features import form as F
from fedge.features import ratings as R
from fedge.features.asof import asof
from fedge.features.main_set import build_feature_table, feature_sets

LAM = math.log(2.0) / F.HALF_LIFE_DAYS


def _league(n_matches=700, n_teams=10, seed=0) -> pd.DataFrame:
    """Synthetic division, three seasons, with shots and (partial) xG; one team swaps each year."""
    rng = np.random.default_rng(seed)
    seasons = ["2020/21", "2021/22", "2022/23"]
    rows = []
    start = pd.Timestamp("2020-08-01 15:00", tz="UTC")
    for i in range(n_matches):
        si = min(int(3 * i / n_matches), 2)
        teams = [f"T{t:02d}" for t in range(n_teams)]
        teams = [t if not (t == "T00" and si >= 1) else "P00" for t in teams]  # T00 relegated
        h, a = rng.choice(n_teams, size=2, replace=False)
        gh, ga = rng.poisson(1.5), rng.poisson(1.1)
        rows.append(
            {
                "match_id": f"m{i:04d}",
                "div": "X1",
                "home": teams[h],
                "away": teams[a],
                "FTHG": gh,
                "FTAG": ga,
                "HS": float(rng.integers(5, 20)),
                "AS": float(rng.integers(5, 20)),
                "HST": float(rng.integers(1, 9)),
                "AST": float(rng.integers(1, 9)),
                "xg_home": float(rng.uniform(0.3, 3.0)) if i % 3 else np.nan,
                "xg_away": float(rng.uniform(0.3, 3.0)) if i % 3 else np.nan,
                # irregular spacing, including a 3h gap so the strict embargo is exercised
                "kickoff_utc": start + pd.Timedelta(hours=19 * i + (i % 5)),
                "season": seasons[si],
            }
        )
    m = pd.DataFrame(rows)
    m["available_at"] = m["kickoff_utc"] + pd.Timedelta(hours=3)
    return m


def _persp(v: pd.DataFrame, team: str, venue: str | None) -> dict[str, np.ndarray]:
    """Independent re-derivation of the per-team stat rows (time, stats) from visible rows."""
    sel_h = v["home"] == team
    sel_a = v["away"] == team
    parts = []
    for sel, home in ((sel_h, True), (sel_a, False)):
        if venue == "H" and not home or venue == "A" and home:
            continue
        r = v.loc[sel]
        gf, ga = (r["FTHG"], r["FTAG"]) if home else (r["FTAG"], r["FTHG"])
        sf, sa = (r["HS"], r["AS"]) if home else (r["AS"], r["HS"])
        xf, xa = (r["xg_home"], r["xg_away"]) if home else (r["xg_away"], r["xg_home"])
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        parts.append(
            pd.DataFrame(
                {
                    "t": r["kickoff_utc"],
                    "gf": gf,
                    "ga": ga,
                    "pts": pts,
                    "sf": sf,
                    "sa": sa,
                    "xf": xf,
                    "xa": xa,
                }
            )
        )
    return pd.concat(parts) if parts else pd.DataFrame(columns=["t"])


def _wmean(rows: pd.DataFrame, stat: str, t_i, prior: float, k: float):
    if rows.empty:
        return prior, 0.0
    w = np.exp(-LAM * ((t_i - rows["t"]).dt.total_seconds().to_numpy() / 86400.0))
    x = rows[stat].to_numpy(dtype=float)
    ok = ~np.isnan(x)
    num, den = float((w[ok] * x[ok]).sum()), float(w[ok].sum())
    return (num + k * prior) / (den + k), den


def _reference_row(m: pd.DataFrame, i: int, stat: str) -> dict:
    t_i = m["kickoff_utc"].iloc[i]
    v = asof(m, t_i)  # strict available_at < bet_time
    h, a = m["home"].iloc[i], m["away"].iloc[i]
    both = pd.concat([_persp(v, tm, None) for tm in pd.unique(pd.concat([v["home"], v["away"]]))])
    lg, _ = _wmean(both, stat, t_i, np.nan, 0.0) if len(both) else (np.nan, 0.0)
    if len(both):
        w = np.exp(-LAM * ((t_i - both["t"]).dt.total_seconds().to_numpy() / 86400.0))
        x = both[stat].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        lg = float((w[ok] * x[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    out = {}
    for who, tm, venue in (("h", h, "H"), ("a", a, "A")):
        mean, den = _wmean(_persp(v, tm, None), stat, t_i, lg, F.PRIOR_K)
        out[f"{who}_{stat}"] = mean
        out[f"{who}_{stat}_n"] = den
        if stat in F.SPLIT_STATS:
            sv, _ = _wmean(_persp(v, tm, venue), stat, t_i, mean, F.PRIOR_K)
            out[f"{who}_{'home' if venue == 'H' else 'away'}_{stat}"] = sv
    return out


def test_form_equals_asof_recomputation():
    m = _league(seed=1)
    feats = F.form_features(m)
    for i in (2, 9, 57, 150, 401, len(m) - 1):
        for stat in ("gf", "ga", "pts", "sf", "xf"):
            ref = _reference_row(m, i, stat)
            for col in (f"h_{stat}", f"a_{stat}", f"h_home_{stat}", f"a_away_{stat}"):
                if col in ref:
                    got = feats.loc[m["match_id"].iloc[i], col]
                    assert np.isclose(got, ref[col], atol=1e-9, equal_nan=True), (i, col)
            if stat == "gf":
                assert np.isclose(feats.loc[m["match_id"].iloc[i], "h_n"], ref["h_gf_n"], atol=1e-9)


def test_strict_embargo_hides_a_result_exactly_three_hours_old():
    m = _league(n_matches=120, seed=3)
    j = 60
    # next match for the same home team, exactly 3h after match j, vs 3h + 1 min after
    base = m.copy()
    base.loc[base.index[j + 1], "home"] = base["home"].iloc[j]
    for delta, visible in (
        (pd.Timedelta(hours=3), False),
        (pd.Timedelta(hours=3, minutes=1), True),
    ):
        d = base.copy()
        d.loc[d.index[j + 1], "kickoff_utc"] = d["kickoff_utc"].iloc[j] + delta
        d["available_at"] = d["kickoff_utc"] + pd.Timedelta(hours=3)
        d = d.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
        k = d.index[d["match_id"] == base["match_id"].iloc[j + 1]][0]
        poisoned = d.copy()
        jj = d.index[d["match_id"] == base["match_id"].iloc[j]][0]
        poisoned.loc[jj, ["FTHG", "FTAG"]] = [9, 0]
        a = F.form_features(d, families=("goals",)).loc[d["match_id"].iloc[k], "h_gf"]
        b = F.form_features(poisoned, families=("goals",)).loc[d["match_id"].iloc[k], "h_gf"]
        assert (not np.isclose(a, b)) == visible


FAMILY_SOURCES = {
    "goals": ["FTHG", "FTAG"],
    "shots": ["HS", "AS", "HST", "AST"],
    "xg": ["xg_home", "xg_away"],
}


def _feat_cols(feats: pd.DataFrame, family: str) -> list[str]:
    names = {
        "goals": ("gf", "ga", "pts", "n"),
        "shots": ("sf", "sa", "tf", "ta"),
        "xg": ("xf", "xa"),
    }[family]
    return [c for c in feats.columns if c.split("_")[-1] in names]


@pytest.mark.parametrize("family", ["goals", "shots", "xg"])
def test_family_does_not_depend_on_future_source_columns(family):
    """Poison the family's raw columns from row ``cut`` on: no earlier row's features move."""
    m = _league(seed=4)
    cut = 350
    base = F.form_features(m)
    poisoned = m.copy()
    for c in FAMILY_SOURCES[family]:
        poisoned.loc[poisoned.index[cut:], c] = 77.0
    got = F.form_features(poisoned)
    ids = m["match_id"].iloc[: cut + 1]  # row `cut` itself is the first poisoned match
    cols = _feat_cols(base, family)
    assert cols, family
    pd.testing.assert_frame_equal(base.loc[ids, cols], got.loc[ids, cols])


def test_schedule_features_never_read_results():
    m = _league(seed=5)
    base = F.form_features(m)
    poisoned = m.copy()
    poisoned[["FTHG", "FTAG", "HS", "AS", "HST", "AST", "xg_home", "xg_away"]] = 5.0
    got = F.form_features(poisoned)
    sched = [c for c in base.columns if c.endswith(("_rest", "_games", "_promoted"))]
    assert len(sched) == 6
    pd.testing.assert_frame_equal(base[sched], got[sched])


def test_rest_games_and_promoted():
    m = _league(seed=6)
    f = F.form_features(m)
    i = 200
    row = m.iloc[i]
    prev = m.iloc[:i]
    last = prev.loc[
        (prev["home"] == row["home"]) | (prev["away"] == row["home"]), "kickoff_utc"
    ].max()
    exp_rest = min((row["kickoff_utc"] - last).total_seconds() / 86400.0, F.REST_CAP)
    assert np.isclose(f.loc[row["match_id"], "h_rest"], exp_rest)
    same = prev.loc[prev["season"] == row["season"]]
    n = ((same["home"] == row["home"]) | (same["away"] == row["home"])).sum()
    assert f.loc[row["match_id"], "h_games"] == n
    # P00 replaces T00 from season 2: promoted in 2021/22, not in 2022/23; nothing in season 1
    p = m.loc[(m["home"] == "P00") & (m["season"] == "2021/22")].iloc[0]["match_id"]
    q = m.loc[(m["home"] == "P00") & (m["season"] == "2022/23")].iloc[0]["match_id"]
    assert f.loc[p, "h_promoted"] == 1.0 and f.loc[q, "h_promoted"] == 0.0
    assert (f.loc[m.loc[m["season"] == "2020/21", "match_id"], "h_promoted"] == 0).all()


def test_form_is_deterministic_and_handles_unplayed_rows():
    m = _league(n_matches=200, seed=7)
    a, b = F.form_features(m), F.form_features(m)
    pd.testing.assert_frame_equal(a, b)
    fixtures = m.copy()
    fixtures.loc[fixtures.index[-5:], ["FTHG", "FTAG"]] = np.nan
    out = F.form_features(fixtures)
    assert len(out) == len(m) and out.index.is_unique


def test_attack_defence_sweep_is_causal_and_sane():
    m = _league(seed=8)
    base = R.attack_defence_sweep(m)
    cut = 300
    poisoned = m.copy()
    poisoned.loc[poisoned.index[cut:], ["FTHG", "FTAG"]] = [9, 0]
    got = R.attack_defence_sweep(poisoned)
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], got.iloc[: cut + 1])
    assert not base.iloc[cut + 5 :].equals(got.iloc[cut + 5 :])  # later rows do see the change
    assert (base["ad_lam_h"] > 0).all() and (base["ad_lam_a"] > 0).all()
    assert abs(base["ad_lam_h"].mean() - 1.5) < 0.3  # synthetic mean home goals = 1.5


def test_feature_table_columns_and_variants():
    m = _league(n_matches=300, seed=9)
    xg = pd.DataFrame({"match_id": m["match_id"], "xg_home": m["xg_home"], "xg_away": m["xg_away"]})
    base = m.drop(columns=["xg_home", "xg_away", "available_at"])
    t = build_feature_table(base, xg)
    sets = feature_sets(t.columns)
    assert set(sets["goals"]) < set(sets["xg"])
    assert all(c in t.columns for c in sets["xg"])
    assert not any(c.endswith(("_xf", "_xa", "_sf", "_sa", "_tf", "_ta")) for c in sets["goals"])
    for c in ("FTHG", "FTAG", "FTR"):
        assert c not in sets["xg"]


def test_full_feature_table_has_no_lookahead():
    """Every family together (ratings, form, shots, xG, schedule): poison the future."""
    m = _league(n_matches=400, seed=11).drop(columns=["available_at"])
    xg = m[["match_id", "xg_home", "xg_away"]].dropna()
    base_m = m.drop(columns=["xg_home", "xg_away"])
    cut = 250
    base = build_feature_table(base_m, xg)
    poisoned = base_m.copy()
    poisoned.loc[poisoned.index[cut:], ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]] = 40
    xg_p = xg.copy()
    xg_p.loc[
        xg_p["match_id"].isin(set(poisoned["match_id"].iloc[cut:])), ["xg_home", "xg_away"]
    ] = 40
    got = build_feature_table(poisoned, xg_p)
    ids = base_m["match_id"].iloc[: cut + 1]
    pd.testing.assert_frame_equal(base.loc[ids], got.loc[ids])
    assert not base.iloc[cut + 10 :].equals(got.iloc[cut + 10 :])
