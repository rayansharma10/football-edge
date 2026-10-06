"""Phase 4: main model (LightGBM / CatBoost) + calibration + decorrelation, vs DC and the market.

Stages (each caches under ``data/interim/``)::

    uv run python scripts/run_main.py --stage dataset            # reuse run_baselines' join
    uv run python scripts/run_main.py --stage features           # as-of feature table
    uv run python scripts/run_main.py --stage models --model lgb_xg [--threads 4]
    uv run python scripts/run_main.py --stage report              # -> reports/v2_main.md

Models (see :mod:`fedge.models.gbm` for the protocol): ``lgb_goals``, ``lgb_xg``, ``cb_goals``,
``cb_xg`` and ``lgbd_xg`` (the decorrelated / market-residual variant of ``lgb_xg``). Each is
fitted for 1X2 (multiclass) and over/under 2.5 (binary). The final probabilities are the
in-fold-calibrated ones (``<model>_*``); the uncalibrated ones are ``<model>_raw_*``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.features.main_set import build_feature_table, feature_sets
from fedge.models import gbm
from fedge.report import metrics as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
IMG = REPORTS / "img"
START_SEASON = "2016/17"
BOOT_N = 100  # match-clustered bootstrap resamples per pool-weight cell (v1 used 200)

# name -> (learner, feature variant, decorrelated?)
MODELS = {
    "lgb_goals": ("lgb", "goals", False),
    "lgb_xg": ("lgb", "xg", False),
    "cb_goals": ("cb", "goals", False),
    "cb_xg": ("cb", "xg", False),
    "lgbd_xg": ("lgb", "xg", True),
}
HEADLINE = "lgb_xg"  # fixed a priori, before any result was seen
MARKET_SELS = {"1x2": ("H", "D", "A"), "ou25": ("over", "under")}


def _load_baselines():
    spec = importlib.util.spec_from_file_location(
        "run_baselines", ROOT / "scripts" / "run_baselines.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_baselines"] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load_baselines()


# ------------------------------------------------------------------ data
def load_matches_xg() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    played = rb.load_played()
    p = INTERIM / "match_xg.parquet"
    return played, (pd.read_parquet(p) if p.exists() else None)


def stage_dataset() -> None:
    """The v1 evaluation join (DC + Elo + pi + reference market), cached."""
    played = rb.load_played()
    odds = pd.read_parquet(INTERIM / "odds.parquet")
    dc_preds, rating_preds = (
        rb.load_cache(rb.DC_CACHE, False),
        rb.load_cache(rb.RATINGS_CACHE, False),
    )
    datasets, refs = rb.build_dataset(played, odds, dc_preds, rating_preds)
    for market, d in datasets.items():
        d.reset_index().to_parquet(INTERIM / f"v2_eval_{market}.parquet", index=False)
    refs.to_parquet(INTERIM / "v2_refs.parquet", index=False)
    # all-seasons pre-closing market per row (training input of the decorrelated variant)
    for market in rb.SELS:
        parts = []
        for r in refs[(refs["market"] == market) & (refs["phase"] == "pre")].itertuples(
            index=False
        ):
            if r.book == "none":
                continue
            w = rb._wide(odds, r.book, r.market, r.phase)
            ids = played.loc[
                (played["div"] == r.div) & (played["season"].astype(str) == r.season), "match_id"
            ]
            w = w.loc[w.index.intersection(ids)]
            if len(w):
                parts.append(rb._devig(w, market, "pre"))
        pd.concat(parts).reset_index(names="match_id").to_parquet(
            INTERIM / f"v2_market_pre_{market}.parquet", index=False
        )
    print("dataset cached", {k: len(v) for k, v in datasets.items()}, flush=True)


def stage_features() -> None:
    played, xg = load_matches_xg()
    t0 = time.time()
    feats = build_feature_table(played, xg)
    feats.to_parquet(INTERIM / "v2_features.parquet")
    print(f"features {feats.shape} in {time.time() - t0:.1f}s", flush=True)


def training_table(market_pre: dict[str, pd.DataFrame]) -> pd.DataFrame:
    played = rb.load_played().set_index("match_id")
    feats = pd.read_parquet(INTERIM / "v2_features.parquet")
    goals = played["FTHG"] + played["FTAG"]
    t = feats.join(
        pd.DataFrame(
            {
                "kickoff_utc": played["kickoff_utc"],
                "season": played["season"].astype(str),
                "div": played["div"],
                "y_1x2": played["FTR"].map({"H": 0, "D": 1, "A": 2}),
                "y_ou25": np.where(goals > 2.5, 0, 1),
            }
        ),
        how="inner",
    )
    t = t.loc[t["y_1x2"].notna()].copy()
    t["y_1x2"] = t["y_1x2"].astype(int)
    for market, mk in market_pre.items():
        t = t.join(mk.set_index("match_id").add_prefix(f"{market}_"), how="left")
    return t


def load_market_pre() -> dict[str, pd.DataFrame]:
    return {m: pd.read_parquet(INTERIM / f"v2_market_pre_{m}.parquet") for m in rb.SELS}


def pred_path(model: str, task: str) -> Path:
    return INTERIM / f"v2_pred_{model}_{task}.parquet"


def stage_models(names: list[str], threads: int) -> None:
    t = training_table(load_market_pre() if any(MODELS[n][2] for n in names) else {})
    sets = feature_sets([c for c in t.columns])
    seasons = sorted(s for s in t["season"].unique() if s >= START_SEASON)
    for name in names:
        learner, variant, decor = MODELS[name]
        for task in gbm.TASKS:
            t0 = time.time()
            init_cols = None
            if decor:
                init_cols = [f"{task}_pre_{s}" for s in MARKET_SELS[task]]
            preds, info = gbm.walk_forward(
                t,
                sets[variant],
                f"y_{task}",
                task,
                learner,
                seasons,
                init_cols=init_cols,
                threads=threads,
                log=lambda s: print(s, flush=True),
            )
            preds.reset_index(names="match_id").to_parquet(pred_path(name, task), index=False)
            info.to_csv(INTERIM / f"v2_info_{name}_{task}.csv", index=False)
            print(f"{name}/{task}: {len(preds)} preds in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ report
def load_eval() -> dict[str, pd.DataFrame]:
    """v1 evaluation sets with every available Phase 4 model joined on."""
    out = {}
    for market in rb.SELS:
        d = pd.read_parquet(INTERIM / f"v2_eval_{market}.parquet").set_index("match_id")
        cols = ["h", "d", "a"] if market == "1x2" else ["over", "under"]
        for name in MODELS:
            p = pred_path(name, market)
            if not p.exists():
                continue
            pr = pd.read_parquet(p).set_index("match_id")
            ren = {f"cal_{c}": f"{name}_{c}" for c in cols}
            ren.update({f"raw_{c}": f"{name}_raw_{c}" for c in cols})
            d = d.join(pr[list(ren)].rename(columns=ren), how="left")
        out[market] = d
    return out


def available_models(d: pd.DataFrame, market: str) -> list[str]:
    cols = ["h", "d", "a"] if market == "1x2" else ["over", "under"]
    return [n for n in MODELS if f"{n}_{cols[0]}" in d.columns]


def common_rows(d: pd.DataFrame, market: str) -> pd.DataFrame:
    """Rows where every Phase 4 model has a prediction (the decorrelated one needs a market)."""
    need = [c for n in available_models(d, market) for c in rb.model_cols(n, market)]
    return d.dropna(subset=need)


def metric_row(d: pd.DataFrame, model: str, market: str) -> dict:
    p = d[rb.model_cols(model, market)].to_numpy(dtype=float)
    y = d[rb.ycol(market)].to_numpy(dtype=int)
    return {
        "model": model,
        "n": len(d),
        "rps": M.rps(p, y),
        "log_loss": M.log_loss(p, y),
        "brier": M.brier(p, y),
        "cal_slope": M.calibration_slope_multi(p, y)[1],
    }


def pooled_table(d: pd.DataFrame, market: str) -> pd.DataFrame:
    base = (
        ["dc", "elo", "pi", "market_pre", "market_close"]
        if market == "1x2"
        else ["dc", "market_pre", "market_close"]
    )
    rows = [metric_row(d, m, market) for m in base]
    for n in available_models(d, market):
        rows.append(metric_row(d, n, market))
        rows.append({**metric_row(d, f"{n}_raw", market), "model": f"{n} (uncalibrated)"})
    return pd.DataFrame(rows)


def per_div_table(d: pd.DataFrame, market: str, models: list[str]) -> pd.DataFrame:
    y_all = d[rb.ycol(market)].to_numpy(dtype=int)
    rows = []
    for div, idx in d.groupby("div", observed=True).indices.items():
        row = {"div": div, "n": len(idx)}
        for m in models:
            p = d[rb.model_cols(m, market)].to_numpy(dtype=float)[idx]
            row[f"{m}_ll"] = M.log_loss(p, y_all[idx])
            row[f"{m}_rps"] = M.rps(p, y_all[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def per_season_table(d: pd.DataFrame, market: str, models: list[str]) -> pd.DataFrame:
    y_all = d[rb.ycol(market)].to_numpy(dtype=int)
    rows = []
    for season, idx in d.groupby("season", observed=True).indices.items():
        row = {"season": season, "n": len(idx)}
        for m in models:
            p = d[rb.model_cols(m, market)].to_numpy(dtype=float)[idx]
            row[f"{m}_ll"] = M.log_loss(p, y_all[idx])
            row[f"{m}_slope"] = M.calibration_slope_multi(p, y_all[idx])[1]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_dc(
    d: pd.DataFrame, market: str, models: list[str], n_boot: int = 1000
) -> pd.DataFrame:
    """Mean per-match log-loss difference (model - DC) with a match bootstrap 95% CI."""
    y = d[rb.ycol(market)].to_numpy(dtype=int)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    pdc = d[rb.model_cols("dc", market)].to_numpy(dtype=float)
    ll_dc = -np.log(np.clip(pdc[np.arange(len(y)), y], 1e-15, 1))
    rows = []
    for m in [*models, "market_pre"]:
        p = d[rb.model_cols(m, market)].to_numpy(dtype=float)
        diff = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)) - ll_dc
        means = diff[idx].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        rows.append({"model": m, "mean_dlogloss_vs_dc": diff.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def corr_table(d: pd.DataFrame, market: str, models: list[str]) -> pd.DataFrame:
    mk = d[rb.model_cols("market_pre", market)].to_numpy(dtype=float)
    rows = []
    for m in models:
        p = d[rb.model_cols(m, market)].to_numpy(dtype=float)
        cs = [np.corrcoef(p[:, k], mk[:, k])[0, 1] for k in range(p.shape[1])]
        lo = np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(mk, 1e-9, 1))
        rows.append(
            {
                "model": m,
                "mean_corr_with_market": float(np.mean(cs)),
                "rmse_to_market": float(np.sqrt(np.mean((p - mk) ** 2))),
                "mean_abs_logodds_gap": float(np.abs(lo).mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_plot(d: pd.DataFrame, market: str, model: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = d[rb.ycol(market)].to_numpy(dtype=int)
    k = 0  # home win (1X2) / over (O/U)
    target = (y == k).astype(float)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for label, cols in (
        ("dc", rb.model_cols("dc", market)),
        (f"{model} raw", rb.model_cols(f"{model}_raw", market)),
        (f"{model} calibrated", rb.model_cols(model, market)),
    ):
        p = d[cols[k]].to_numpy(dtype=float)
        t = M.reliability_table(p, target, 10)
        ax.plot(t["mean_pred"], t["obs_freq"], "o-", ms=3, lw=1, label=label)
    ax.set_xlabel("forecast P(" + ("home win" if market == "1x2" else "over 2.5") + ")")
    ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7)
    ax.set_title(f"{market} reliability, test seasons", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def stage_report() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    data = load_eval()
    played, mx = load_matches_xg()
    xg_divs = (
        sorted(played.loc[played["match_id"].isin(set(mx["match_id"])), "div"].unique())
        if mx is not None
        else []
    )
    datasets, tables = {}, {}
    pools, weights_by_season = [], []
    metrics_rows, calib_rows, ls_rows = [], [], []
    for market, d in data.items():
        d = common_rows(d, market)
        datasets[market] = d
        names = available_models(d, market)
        tab = pooled_table(d, market)
        tab["market"] = market
        metrics_rows.append(tab)
        tables[market] = {
            "pooled": tab,
            "div": per_div_table(d, market, ["dc", "market_pre", *names]),
            "season": per_season_table(d, market, ["dc", HEADLINE, "market_pre"]),
            "paired": paired_vs_dc(d, market, names),
            "corr": corr_table(d, market, ["dc", *names]),
        }
        sc = pd.concat(
            [rb.score_rows(d, m, market) for m in ["dc", "market_pre", "market_close", *names]],
            ignore_index=True,
        )
        sc["season"] = sc["season"].astype(str)
        ls_rows.append(sc)
        piv = sc[sc["model"].isin(["dc", HEADLINE, "market_pre"])].pivot_table(
            index=["div", "season", "n"],
            columns="model",
            values=["rps", "log_loss"],
            aggfunc="first",
        )
        piv.columns = [f"{m}_{k}" for k, m in piv.columns]
        tables[market]["ls"] = piv.reset_index().sort_values(["div", "season"])
        for n in names:
            info = INTERIM / f"v2_info_{n}_{market}.csv"
            if info.exists():
                i = pd.read_csv(info)
                i["model"], i["market"] = n, market
                calib_rows.append(i)
        if HEADLINE in names:
            calibration_plot(d, market, HEADLINE, IMG / f"cal_{market}_{HEADLINE}.png")
        for n in names:
            w, det = rb.pool_table(d, market, n)
            pools.append(w)
            weights_by_season.append(det)
        wdc, detdc = rb.pool_table(d, market, "dc")
        pools.append(wdc)
        weights_by_season.append(detdc)
    pools_df = pd.concat([p for p in pools if len(p)], ignore_index=True)
    pd.concat(metrics_rows).to_csv(REPORTS / "v2_main_metrics.csv", index=False)
    pools_df.to_csv(REPORTS / "v2_main_pool_weights.csv", index=False)
    pd.concat(ls_rows).to_csv(REPORTS / "v2_main_metrics_by_league_season.csv", index=False)
    pd.concat([p for p in weights_by_season if len(p)]).to_csv(
        REPORTS / "v2_main_pool_weights_by_season.csv", index=False
    )
    calib_df = pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame()
    calib_df.to_csv(REPORTS / "v2_main_fold_info.csv", index=False)
    write_report(datasets, tables, pools_df, calib_df, xg_divs)
    print("wrote", REPORTS / "v2_main.md", flush=True)


def _t(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    return rb.md_table(df, cols, headers)


def _slopes(df: pd.DataFrame, col: str = "cal_slope") -> pd.DataFrame:
    """Copy ``df`` with the calibration-slope column rendered through ``M.fmt_slope``.

    ``md_table`` formats floats to 4 dp but passes strings through, so the separated
    (non-estimable) slopes must be strings before they reach the renderer. The stored CSVs are
    untouched (this copies).
    """
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].map(M.fmt_slope)
    return out


def pool_summary(pools: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, market, phase), g in pools.groupby(["model", "market", "phase"]):
        rows.append(
            {
                "model": model,
                "market": market,
                "phase": phase,
                "cells": len(g),
                "cells_ci_gt0": int((g["w_lo"] > rb.CI_ZERO_TOL).sum()),
                "mean_w_expanding": float(g["w_expanding"].mean()),
                "max_w_pooled": float(g["w_pooled"].max()),
                "mean_ll_gain_vs_market": float((g["market_logloss"] - g["pool_logloss"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(datasets, tables, pools, calib, xg_divs) -> None:  # pragma: no cover
    d1, d25 = datasets["1x2"], datasets["ou25"]
    summ = pool_summary(pools)
    lines = [
        "# football-edge v2: main model (Phase 4, cards 4.1-4.4)",
        "",
        "Generated by `uv run python scripts/run_main.py --stage report` from `data/interim/` "
        "caches. Companion CSVs in this directory: `v2_main_metrics.csv`, "
        "`v2_main_pool_weights.csv`, `v2_main_pool_weights_by_season.csv`, "
        "`v2_main_fold_info.csv`. The evaluation join (leagues, test seasons from 2016/17 for "
        "1X2 and 2019/20 for over/under 2.5, reference-book rule, power de-vig, pooled log "
        "opinion pool, 95% match-clustered bootstrap) is exactly the one in "
        "`reports/v1_baseline.md`, restricted to rows where every model below has a prediction.",
        "",
        "## 1. What was run",
        "",
        f"- Evaluation rows: {len(d1)} for 1X2 over {d1['season'].nunique()} seasons "
        f"({d1['season'].min()} to {d1['season'].max()}, 2026/27 partial) and {len(d25)} for "
        f"over/under 2.5 over {d25['season'].nunique()} seasons ({d25['season'].min()} to "
        f"{d25['season'].max()}), across {d1['div'].nunique()} divisions. The two markets do "
        "not share a window: the pre-2019/20 over/under cells have no closing reference book "
        "(126 `none` cells at the close in `reports/v1_baseline_reference_books.csv`) and so "
        "`run_baselines.build_dataset` drops them.",
        "- **Features** (all as-of, `fedge.features.*`; leakage tests in `tests/test_form.py`, "
        "`tests/test_ratings.py`): Elo difference and pi-rating expected goal difference (fixed "
        'a-priori parameters), online Poisson attack/defence ratings ("DC-lite"), '
        "exponentially time-weighted (half-life 120 days, shrunk to the division mean with 3 "
        "pseudo-matches) goals for/against, points, home-only/away-only goals, rest days, games "
        "played this season, new-to-division (promoted) flag and the division code. The `xg` "
        "variant adds time-weighted shots, shots on target and xG for/against (NaN where "
        "absent: LightGBM/CatBoost handle it natively). `goals` = no shots or xG.",
        f"- xG exists for {', '.join(xg_divs) or 'no division'} (Understat) and for "
        "football-data's "
        "2026/27 `HxG/AxG`; all other rows have NaN xG features.",
        "- **Models**: LightGBM (`lgb_*`) and CatBoost (`cb_*`) multiclass for 1X2 and binary for "
        "over 2.5, one model across all 18 divisions, goals-only and xG variants. `lgbd_xg` is the "
        "decorrelated variant (section 7). `lgb_xg` was named the headline model before any "
        "result was seen.",
        "- **Walk-forward** per test season S: model A on seasons before S-1 (early stopping on "
        "the "
        "first half of S-1), calibrator chosen in-fold on the second half of S-1 (identity / "
        "isotonic / Beta / Venn-Abers, lowest held-out log-loss), final model refit on every "
        "match before S with 1.1x the early-stopped tree count, 3h embargo on every cut. "
        "Seeds fixed (`seed=0`), deterministic LightGBM, fixed thread counts.",
        "- **Pool**: same log opinion pool as v1 with the model weight `w` against the pre-closing "
        "and closing de-margined reference price; bootstrap resamples per cell = "
        f"{rb.BOOT_N} (v1: 200).",
        "",
        "## 2. Pooled scores (all divisions; 1X2 test seasons 2016/17+, over/under 2.5 2019/20+)",
        "",
        "RPS/Brier are multiclass sums; `cal_slope` is the pooled one-vs-rest logistic "
        "recalibration slope (1.0 = calibrated). Lower is better except slope.",
        "`n.e.` = not estimable: the one-vs-rest logistic recalibration separated at this n "
        "(|slope| > 10), so the value is a sample-size artefact, not a calibration measurement.",
        "",
        "### 1X2",
        "",
        _t(
            _slopes(tables["1x2"]["pooled"]),
            ["model", "n", "rps", "log_loss", "brier", "cal_slope"],
        ),
        "",
        "### Over/under 2.5",
        "",
        _t(
            _slopes(tables["ou25"]["pooled"]),
            ["model", "n", "rps", "log_loss", "brier", "cal_slope"],
        ),
        "",
        "## 3. Comparison vs Dixon-Coles",
        "",
        "Mean per-match log-loss difference to Dixon-Coles (negative = better than DC) with a "
        "1000-resample match bootstrap 95% CI:",
        "",
        "### 1X2",
        "",
        _t(tables["1x2"]["paired"], ["model", "mean_dlogloss_vs_dc", "lo", "hi"]),
        "",
        "### Over/under 2.5",
        "",
        _t(tables["ou25"]["paired"], ["model", "mean_dlogloss_vs_dc", "lo", "hi"]),
        "",
        "Per league (log-loss, all test seasons):",
        "",
        "### 1X2",
        "",
        _t(
            tables["1x2"]["div"],
            ["div", "n", *[c for c in tables["1x2"]["div"].columns if c.endswith("_ll")]],
        ),
        "",
        "### Over/under 2.5",
        "",
        _t(
            tables["ou25"]["div"],
            ["div", "n", *[c for c in tables["ou25"]["div"].columns if c.endswith("_ll")]],
        ),
        "",
        "Per test season (headline model vs DC vs pre-closing market):",
        "",
        "### 1X2",
        "",
        _t(tables["1x2"]["season"], list(tables["1x2"]["season"].columns)),
        "",
        "### Over/under 2.5",
        "",
        _t(tables["ou25"]["season"], list(tables["ou25"]["season"].columns)),
        "",
    ]
    lines += ["## 3b. Per league x season (headline vs DC vs pre-closing market)", ""]
    lines += ["All models: `v2_main_metrics_by_league_season.csv`.", ""]
    for market in ("1x2", "ou25"):
        ls = tables[market]["ls"]
        lines += [f"### {market}", "", _t(ls, list(ls.columns)), ""]
    # xG vs goals-only on the xG leagues
    lines += ["## 4. xG variant vs goals-only (leagues with Understat xG)", ""]
    for market, d in (("1x2", d1), ("ou25", d25)):
        sub = d[d["div"].isin(xg_divs)]
        if not len(sub):
            continue
        names = [
            m
            for m in ("lgb_goals", "lgb_xg", "cb_goals", "cb_xg")
            if f"{m}_" + ("h" if market == "1x2" else "over") in sub.columns
        ]
        rows = [metric_row(sub, m, market) for m in ["dc", *names, "market_pre"]]
        lines += [
            f"### {market} ({len(sub)} matches)",
            "",
            _t(
                _slopes(pd.DataFrame(rows)),
                ["model", "n", "rps", "log_loss", "brier", "cal_slope"],
            ),
            "",
        ]
    # pool weights
    lines += [
        "## 5. Pool weight vs the market",
        "",
        "`w` is the weight on the model in the log pool (0 = the model adds nothing to the "
        "market). Cells counted as significant have a bootstrap lower bound above "
        f"{rb.CI_ZERO_TOL:g}. Summary over the 18 league cells per (model, market, phase):",
        "",
        _t(
            summ.sort_values(["market", "phase", "model"]),
            [
                "model",
                "market",
                "phase",
                "cells",
                "cells_ci_gt0",
                "mean_w_expanding",
                "max_w_pooled",
                "mean_ll_gain_vs_market",
            ],
        ),
        "",
    ]
    for model in (HEADLINE, "lgbd_xg"):
        for market in ("1x2", "ou25"):
            g = pools[(pools["model"] == model) & (pools["market"] == market)]
            if not len(g):
                continue
            lines += [
                f"### {model}, {market}: w vs pre-closing and closing price, per league",
                "",
                _t(
                    g.sort_values(["phase", "div"]),
                    [
                        "div",
                        "phase",
                        "n",
                        "n_seasons",
                        "w_expanding",
                        "w_pooled",
                        "w_lo",
                        "w_hi",
                        "pool_logloss",
                        "market_logloss",
                    ],
                ),
                "",
            ]
    lines += [
        "Dixon-Coles weights (same code) and all other models: `v2_main_pool_weights.csv`.",
        "",
    ]
    # calibration
    lines += ["## 6. Calibration", ""]
    if len(calib):
        # Count evaluated seasons, not fitted ones: the three pre-2019/20 over/under folds are
        # fitted for every model but dropped from the evaluation join (m1).
        eval_seasons = {m: set(datasets[m]["season"].astype(str)) for m in datasets}
        keep = [
            str(s) in eval_seasons.get(str(mkt), set())
            for s, mkt in zip(calib["season"], calib["market"], strict=True)
        ]
        excluded = int(len(calib) - sum(keep))
        cnt = (
            calib[keep]
            .groupby(["model", "market", "calibrator"])
            .size()
            .reset_index(name="seasons")
        )
        lines += [
            "In-fold calibrator chosen per (model, market, test season) by held-out log-loss "
            "(identity is always a candidate); counts are evaluated seasons only:",
            "",
            _t(cnt, ["model", "market", "calibrator", "seasons"]),
            "",
            f"{excluded} (model, market, season) rows were fitted but never evaluated: the "
            "three pre-2019/20 over/under folds are fitted for every model but dropped from "
            "the evaluation join (m1). `reports/v2_main_fold_info.csv` intentionally keeps "
            "the full fitted list.",
            "",
        ]
    lines += [
        f"Reliability plots for `{HEADLINE}` (raw vs calibrated vs DC): "
        f"`img/cal_1x2_{HEADLINE}.png`, `img/cal_ou25_{HEADLINE}.png`.",
        "",
        f"![1x2](img/cal_1x2_{HEADLINE}.png) ![ou25](img/cal_ou25_{HEADLINE}.png)",
        "",
        "## 7. Decorrelation variant (card 4.4)",
        "",
        "`lgbd_xg` trains on the *residual of the market*: boosting starts from "
        "`init_score = log p_market(pre-closing)` (logit for the binary task), so the output is "
        "`softmax(log p_market + f(x))`; early stopping on the previous season decides how far "
        "`f` may move from the market (zero trees = the market). It is trained only on rows "
        "with a pre-closing price. The aim: a model whose *only* signal is what the market does "
        "not already say, which the pool then weights. Distance to the market (pre-closing):",
        "",
        "### 1X2",
        "",
        _t(tables["1x2"]["corr"], list(tables["1x2"]["corr"].columns)),
        "",
        "### Over/under 2.5",
        "",
        _t(tables["ou25"]["corr"], list(tables["ou25"]["corr"].columns)),
        "",
    ]
    lines += ["", "## 8. Limits and UNVERIFIED", "", *LIMITS, ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "v2_main.md").write_text("\n".join(lines), encoding="utf-8")


LIMITS = [
    "- **Correction to v1.** `run_baselines.build_dataset` concatenated whole-book price tables "
    "per (division, season) cell and kept the first book's columns, so every evaluation row "
    "used Pinnacle and the 2025/26 and 2026/27 rows (reference book: Betfair Exchange) were "
    "silently dropped (59,193 -> 66,373 1X2 rows after the fix, 39,539 -> 46,417 over/under). "
    "The join now restricts each cell's prices to that cell's matches; `v1_baseline.md` was "
    "regenerated. The v1 headline changed with it: 5 of 36 pre-closing cells (SC0, G1, N1, SP2 "
    "over/under; G1 1X2) now have a Dixon-Coles pool-weight CI above 0 (the earlier "
    "'0 of 36' no longer holds). That CI is in-sample (`w_pooled` refits the weight on the "
    "whole sample; the out-of-sample `w_expanding` is printed beside it in `v1_baseline.md`), "
    "so read it as 36 cells x 2 markets of multiple testing, and 3 of those 5 (SC0, G1, SP2 "
    "over/under) have expanding-window pooled log-loss *worse* than the market. Against the "
    "closing price it is still 0 of 36.",
    "- **How to read the pre-closing over/under weights.** The pre-closing over/under price is "
    "visibly mis-calibrated (slope 0.92 vs 1.02 at the close) and 0.005 log-loss worse than "
    "the close, so a pool with any smoother model gains a little. For the GBM models against "
    "the pre-closing over/under price, the `mean_ll_gain_vs_market` rows of section 5 span "
    "0.0003-0.0006 (mean ~0.00045); the gain is absent against the close. This is not evidence "
    "of an exploitable edge: it most likely reflects the staleness of the football-data "
    "pre-closing price (timestamp unverified, see next item).",
    "- Features use the same cut as v1: a result is usable once `kickoff + 3h < kickoff of the "
    "predicted match`. The pre-closing price in football-data is dated `kickoff - 24h` by the P1 "
    "ingest, which is almost certainly earlier than the real collection time; a result that "
    "lands in the 24h before kickoff (another team's midweek match) can therefore be in the "
    "features but not in the price. Phase 5 must fix the cut-off clock before any early-line "
    "strategy is measured.",
    "- Pi/Elo parameters are fixed a priori (Elo k=20/HFA=60, pi alpha=0.15/beta=0.10), the "
    "attack/defence step (0.06), form half-life (120 days), prior strength (3 matches) and all "
    "GBM hyper-parameters are fixed and not tuned; only the tree count and the calibrator are "
    "chosen in-fold.",
    "- Beta calibration is the unconstrained 2-feature logistic form (no a,b >= 0 constraint). "
    "Venn-Abers is the `venn-abers` package, one-vs-rest with renormalisation for 1X2; the "
    "interval width is discarded.",
    "- The calibrator is fitted on predictions of a model trained one season earlier than the "
    "final model that it is applied to (the final model sees S-1 as well).",
    "- Early stopping uses only the first half of S-1 and the calibrator the second half, so the "
    "calibration set is ~3,300 matches per season (18 divisions pooled).",
    "- One model is fitted across all divisions (division code as a categorical feature); "
    "per-division models were not tried.",
    "- A promoted team is one absent from its division in the previous season of the data "
    "(relegated-from-above teams count too); in 2012/13 nobody is flagged.",
    "- xG is only available for the top-5 leagues (Understat) and 2026/27 (football-data); the "
    "xG variant is NaN elsewhere.",
    "- The decorrelated model is trained on football-data reference prices (Pinnacle/BFE/Avg "
    "chosen per division-season as in v1), mixing books across training seasons.",
]


def main() -> None:  # pragma: no cover - CLI driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dataset", "features", "models", "report"], required=True)
    ap.add_argument("--model", action="append", choices=list(MODELS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    if args.stage == "dataset":
        stage_dataset()
    elif args.stage == "features":
        stage_features()
    elif args.stage == "models":
        stage_models(args.model or list(MODELS), args.threads)
    else:
        rb.BOOT_N = BOOT_N
        stage_report()


if __name__ == "__main__":
    main()
