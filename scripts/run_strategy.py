"""Phase 5: early-line value strategy backtest -> reports/gate0.md and config/strategy.toml.

    uv run python scripts/run_strategy.py            # full run (about 5-10 min)

Inputs are the caches built by ``run_baselines.py`` / ``run_main.py`` (``data/interim``). Everything
is walk-forward: the pool weight, model, edge threshold, whitelist and Kelly fraction for test
season S are chosen from seasons before S only (``fedge.backtest.strategy``). The whole pipeline is
run twice in-process and the ledger hashes are compared (Gate 0 B0.4).
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from fedge.backtest import strategy as S
from fedge.backtest.stats import COMMISSION, bootstrap_ci
from fedge.models.pool import fit_weight, fit_weight_bootstrap
from fedge.report import metrics as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
N_BOOT = 1000
N_RAND = 200
SCENARIOS = (
    "exch",
    "max",
)  # BFE/PS (headline) and Max (best of market, optimistic); Avg never needed
HEADLINE = "exch"
MODELS = {
    "1x2": ["dc", "elo", "pi", "lgb_goals", "lgb_xg", "cb_goals", "cb_xg", "lgbd_xg"],
    "ou25": ["dc", "lgb_goals", "lgb_xg", "cb_goals", "cb_xg", "lgbd_xg"],
}
GBM = {"lgb_goals", "lgb_xg", "cb_goals", "cb_xg", "lgbd_xg"}
PS_STALE_FROM = "2025-07-23"
MAX_RATIO_CAP = 1.10  # max best-of-market price vs the exchange-like price (bad-quote guard)


def _load_rb():
    spec = importlib.util.spec_from_file_location(
        "run_baselines", ROOT / "scripts" / "run_baselines.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_baselines"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ inputs
def load_market(rb, market: str) -> pd.DataFrame:
    d = pd.read_parquet(INTERIM / f"v2_eval_{market}.parquet").set_index("match_id")
    cols = ["h", "d", "a"] if market == "1x2" else ["over", "under"]
    for name in MODELS[market]:
        if name not in GBM:
            continue
        pr = pd.read_parquet(INTERIM / f"v2_pred_{name}_{market}.parquet").set_index("match_id")
        d = d.join(
            pr[[f"cal_{c}" for c in cols]].rename(columns={f"cal_{c}": f"{name}_{c}" for c in cols})
        )
    need = [c for n in MODELS[market] for c in rb.model_cols(n, market)]
    return d.dropna(subset=need).sort_values(["kickoff_utc"], kind="mergesort")


def price_tables(rb, odds: pd.DataFrame, idx: dict[str, pd.Index]) -> dict:
    """Pre-closing bet prices per scenario and market, aligned to the evaluation rows."""
    out = {s: {} for s in SCENARIOS}
    guard: dict[str, pd.Series] = {}
    for market, ix in idx.items():
        sels = list(rb.SELS[market])
        w = {b: rb._wide(odds, b, market, "pre").reindex(ix) for b in ("BFE", "PS", "Max")}
        ex = w["BFE"].copy()
        src = pd.Series(np.where(ex.notna().all(axis=1), "BFE", ""), index=ix)
        use_ps = ~ex.notna().all(axis=1) & w["PS"].notna().all(axis=1)
        ex.loc[use_ps] = w["PS"].loc[use_ps]
        src[use_ps] = "PS"
        ex.loc[src == ""] = np.nan
        out["exch"][market] = (ex[sels], src.where(src != "", "none"))
        mx = w["Max"][sels]
        # data-error guard: a best-of-market price more than MAX_RATIO_CAP above the exchange-like
        # price of the same selection is treated as a bad quote and the match is dropped (the Max
        # columns of 2024/25+ contain such prices; see section 'Max price guard' in the report).
        bad = ((mx / ex[sels]).max(axis=1) > MAX_RATIO_CAP) & ex[sels].notna().all(axis=1)
        mx = mx.mask(bad)
        guard[market] = bad
        out["max"][market] = (
            mx,
            pd.Series(np.where(mx.notna().all(axis=1), "Max", "none"), index=ix),
        )
    out["_guard"] = guard
    return out


def prepare(rb) -> dict:
    """Everything scenario-independent: pooled probs per season, in-fold weights, log-loss tables."""
    data = {}
    for market in S.SELECTIONS:
        d = load_market(rb, market)
        y = d["y_1x2" if market == "1x2" else "y_ou25"].to_numpy(dtype=int)
        pre = d[[f"pre_{s}" for s in S.SELECTIONS[market]]].to_numpy(float)
        close = d[[f"close_{s}" for s in S.SELECTIONS[market]]].to_numpy(float)
        probs = {m: d[rb.model_cols(m, market)].to_numpy(float) for m in MODELS[market]}
        pooled, wts = S.pooled_by_season(probs, pre, y, d["season"].to_numpy())
        data[market] = {
            "frame": d,
            "y": y,
            "pre": pre,
            "close": close,
            "probs": probs,
            "pooled": pooled,
            "weights": wts,
            "ll": S.pooled_log_loss_table(pooled, y, d["season"].to_numpy()),
        }
    return data


# ------------------------------------------------------------------ one scenario
def run_scenario(data: dict, prices: dict, name: str) -> dict:
    best, favs, longs = {}, {}, {}
    for market, dm in data.items():
        d = dm["frame"]
        px, src = prices[name][market]
        meta = d[["div", "season", "kickoff_utc"]].copy()
        meta["source"] = src.to_numpy()
        price = px.to_numpy(float)
        best[market] = S.candidate_table(meta, market, price, dm["close"], dm["pooled"], dm["y"])
        favs[market] = S.favourite_table(meta, market, price, dm["close"], dm["y"])
        longs[market] = S.long_candidates(meta, market, price, dm["close"], dm["y"])
    seasons = sorted(set(data["1x2"]["frame"]["season"]) | set(data["ou25"]["frame"]["season"]))[1:]
    ll = {m: dm["ll"] for m, dm in data.items()}
    bets, cfgs = S.run_walk_forward(best, ll, seasons)
    return {
        "best": best,
        "fav": favs,
        "long": longs,
        "bets": bets,
        "cfgs": cfgs,
        "seasons": seasons,
    }


def ledger_hash(res: dict) -> str:
    b = res["bets"]
    sel = b[b["selected"]].reset_index().sort_values(["match_id", "market"])
    cols = ["match_id", "market", "sel", "price", "edge", "kelly", "model"]
    return hashlib.sha256(sel[cols].to_csv(index=False, float_format="%.12g").encode()).hexdigest()


# ------------------------------------------------------------------ helpers for the report
def f(v, nd=4, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v * 100:.2f}%" if pct else f"{v:.{nd}f}"


def md(df: pd.DataFrame, cols: list[str], heads: list[str] | None = None) -> str:
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        out.append(
            "| "
            + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r)
            + " |"
        )
    return "\n".join(out)


def fmt_metrics(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["clv_net_s"] = [
        f"{f(a, 4, True)} [{f(b, 4, True)}, {f(c, 4, True)}]"
        for a, b, c in zip(m["clv_net"], m["clv_net_lo"], m["clv_net_hi"], strict=True)
    ]
    m["clv_raw_s"] = [
        f"{f(a, 4, True)} [{f(b, 4, True)}, {f(c, 4, True)}]"
        for a, b, c in zip(m["clv_raw"], m["clv_raw_lo"], m["clv_raw_hi"], strict=True)
    ]
    for c in ("roi_flat", "roi_k10", "roi_k25", "mdd_flat", "mdd_k25", "hit"):
        m[c + "_s"] = [f(v, 2, True) for v in m[c]]
    return m


METRIC_COLS = [
    "n",
    "clv_net_s",
    "clv_raw_s",
    "roi_flat_s",
    "roi_k10_s",
    "roi_k25_s",
    "mdd_flat_s",
    "mdd_k25_s",
]
METRIC_HEADS = [
    "n",
    "net CLV [95% CI]",
    "raw CLV [95% CI]",
    "flat ROI",
    "Kelly0.1 ROI",
    "Kelly0.25 ROI",
    "max DD flat",
    "max DD K0.25",
]


def baseline_text(name: str, o: dict) -> str:
    ov, rn = o["overall"], o["rand"]
    if not ov.get("n"):
        return f"{name}: no bets placed (in-fold history never qualified a whitelist)"
    p = float((rn["clv_net"] >= ov["clv_net"]).mean())
    return (
        f"{name}: n={ov['n']}, net CLV {f(ov['clv_net'], 4, True)} vs favourite (same matches) "
        f"{f(o['fav_same'].get('clv_net'), 4, True)}, favourite (all fold matches) "
        f"{f(o['fav_all'].get('clv_net'), 4, True)}, random same count "
        f"[{f(rn['clv_net'].quantile(0.025), 4, True)}, {f(rn['clv_net'].quantile(0.975), 4, True)}] "
        f"(P(random >= strategy) = {f(p, 3)})"
    )


def summary_text(out: dict, res: dict) -> str:
    e, m = out["exch"], out["max"]
    no_wl = e["overall_nowl"]
    txt = [
        "With the exchange-like price (BFE/PS) the in-fold selection never found a whitelist: "
        f"{e['overall'].get('n', 0)} bets were placed. Forcing bets above the in-fold threshold without the "
        f"whitelist gives n={no_wl.get('n', 0)}, net CLV {f(no_wl.get('clv_net'), 4, True)} "
        f"[{f(no_wl.get('clv_net_lo'), 4, True)}, {f(no_wl.get('clv_net_hi'), 4, True)}], flat ROI {f(no_wl.get('roi_flat'), 2, True)}.",
    ]
    mo = m["overall"]
    if mo.get("n"):
        txt.append(
            f"With the best-of-market `Max` price (optimistic, bad-quote guard applied) the strategy placed n={mo['n']} bets, "
            f"net CLV {f(mo['clv_net'], 4, True)} [{f(mo['clv_net_lo'], 4, True)}, {f(mo['clv_net_hi'], 4, True)}], "
            f"raw CLV {f(mo['clv_raw'], 4, True)}, flat ROI {f(mo['roi_flat'], 2, True)}."
        )
    txt.append(
        "The pooled model weight against the sharp price is about zero for every non-decorrelated model, so the pooled probability is the market probability and there is no edge to find (B0.6)."
    )
    return " ".join(txt)


def run_pytest_counts() -> str:
    files = ["tests/test_leakage.py", "tests/test_stats_walkforward.py", "tests/test_strategy.py"]
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    m = re.findall(r"(\d+) (passed|failed|error)", r.stdout)
    return ", ".join(f"{n} {k}" for n, k in m) or "no result"


def benchmark_table(data: dict) -> tuple[pd.DataFrame, dict]:
    """B0.5: per season, model vs de-margined sharp close, plus the strategy's pooled model."""
    rows = []
    for market, dm in data.items():
        d, y = dm["frame"], dm["y"]
        k = len(S.SELECTIONS[market])
        for season in sorted(set(d["season"])):
            sel = (d["season"] == season).to_numpy()
            variants = {
                "market_close": dm["close"],
                "market_pre": dm["pre"],
                "lgb_xg": dm["probs"]["lgb_xg"],
            }
            for mname in dm["pooled"]:
                if mname in ("lgb_xg", "lgbd_xg"):
                    variants[f"pooled_{mname}"] = dm["pooled"][mname]
            for name, p in variants.items():
                pp, yy = p[sel], y[sel]
                if np.isnan(pp).any() or not sel.any():
                    continue
                flat_y = np.eye(k)[yy].ravel()
                rows.append(
                    {
                        "market": market,
                        "season": season,
                        "model": name,
                        "n": int(sel.sum()),
                        "rps": M.rps(pp, yy),
                        "log_loss": M.log_loss(pp, yy),
                        "brier": M.brier(pp, yy),
                        "slope": M.calibration_slope_multi(pp, yy)[1],
                        "ece": M.ece(pp.ravel(), flat_y),
                    }
                )
    return pd.DataFrame(rows), {}


def devig_ae(data: dict) -> pd.DataFrame:
    rows = []
    for market, dm in data.items():
        for phase in ("pre", "close"):
            p, y = dm[phase], dm["y"]
            for j, s in enumerate(S.SELECTIONS[market]):
                rows.append(
                    {
                        "market": market,
                        "phase": phase,
                        "sel": s,
                        "n": len(y),
                        "A/E": float((y == j).sum() / p[:, j].sum()),
                    }
                )
    return pd.DataFrame(rows)


def pool_ci(data: dict) -> pd.DataFrame:
    """B0.6: all-sample pool weight of the candidate models against the margin-free CLOSE."""
    rows = []
    for market, dm in data.items():
        d = dm["frame"]
        for mname in ("lgb_xg", "lgbd_xg", "dc"):
            for phase in ("close", "pre"):
                r = fit_weight_bootstrap(
                    dm["probs"][mname], dm[phase], dm["y"], d.index.to_numpy(), n_boot=150, seed=0
                )
                rows.append(
                    {
                        "market": market,
                        "model": mname,
                        "vs": phase,
                        "w": r["w"],
                        "lo": r["lo"],
                        "hi": r["hi"],
                        "n": r["n"],
                    }
                )
    return pd.DataFrame(rows)


def sweep_table(res: dict) -> pd.DataFrame:
    """Diagnostic, NOT used for selection: fixed-model threshold sweep on every pooled season."""
    rows = []
    for s in SCENARIOS:
        for market, models in res[s]["best"].items():
            for mname in ("dc", "lgb_xg", "cb_xg", "lgbd_xg"):
                df = models[mname]
                for t in S.THRESHOLDS:
                    g = df[df["edge"] > t]
                    if len(g) == 0:
                        rows.append(
                            {"scenario": s, "market": market, "model": mname, "thr": t, "n": 0}
                        )
                        continue
                    ci = bootstrap_ci(g["clv_net"].to_numpy(), g.index.to_numpy(), n_boot=300)
                    rows.append(
                        {
                            "scenario": s,
                            "market": market,
                            "model": mname,
                            "thr": t,
                            "n": len(g),
                            "clv_net": ci["mean"],
                            "lo": ci["lo"],
                            "hi": ci["hi"],
                            "clv_raw": float(g["clv_raw"].mean()),
                            "roi_flat": float(g["pnl_flat"].mean()),
                        }
                    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ final configuration
def final_config(data: dict, res: dict, gate_pass: bool) -> tuple[dict, pd.DataFrame]:
    """In-fold selection for the NEXT unseen fold: all seasons so far are history."""
    hist = res["seasons"]
    markets, hist_sel, rows = {}, [], []
    for market, models in res["best"].items():
        m = S.pick_model(data[market]["ll"], hist)
        cfg = S.select_market_config(models[m], hist)
        w_all = fit_weight(data[market]["probs"][m], data[market]["pre"], data[market]["y"])
        if cfg is None:
            markets[market] = {
                "model": m,
                "pool_weight": float(w_all),
                "threshold": 1.0,
                "whitelist": [],
                "qualified": False,
            }
            continue
        hb = models[m][models[m]["season"].isin(hist)]
        hist_sel.append(hb[(hb["edge"] > cfg["threshold"]) & hb["div"].isin(cfg["whitelist"])])
        markets[market] = {
            "model": m,
            "pool_weight": float(w_all),
            "threshold": cfg["threshold"],
            "whitelist": cfg["whitelist"],
            "qualified": bool(cfg["whitelist"]),
        }
        rows.append(
            {
                "market": market,
                "model": m,
                "w_all": w_all,
                **cfg,
                "whitelist": ",".join(cfg["whitelist"]),
            }
        )
    hb_all = pd.concat(hist_sel) if hist_sel else pd.DataFrame()
    kelly = S.select_kelly(hb_all) if len(hb_all) else min(S.KELLY_FRACTIONS)
    top = {
        "gate0_passed": bool(gate_pass),
        "mode": "paper" if gate_pass else "shadow",
        "price_source": "BFE pre-closing, else PS pre-closing (pre 2025-07-23); never Max/Avg",
        "commission": COMMISSION,
        "kelly_fraction": float(kelly),
        "bet_time": "kickoff-24h snapshot (ASSUMED, see reports/gate0.md)",
        "history_seasons": f"{hist[0]} to {hist[-1]}",
    }
    cfg = {
        "comments": [
            "Generated by scripts/run_strategy.py: in-fold selection with every season so far as history.",
            "Not a promise of edge: see reports/gate0.md. gate0_passed=false means shadow mode only.",
        ],
        "top": top,
        "markets": markets,
    }
    return cfg, pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main() -> None:  # pragma: no cover - CLI driver
    rb = _load_rb()
    odds = pd.read_parquet(INTERIM / "odds.parquet")
    data = prepare(rb)
    prices = price_tables(rb, odds, {m: dm["frame"].index for m, dm in data.items()})
    del odds

    res = {s: run_scenario(data, prices, s) for s in SCENARIOS}
    res2 = {s: run_scenario(data, prices, s) for s in SCENARIOS}
    hashes = {s: ledger_hash(res[s]) for s in SCENARIOS}
    deterministic = all(hashes[s] == ledger_hash(res2[s]) for s in SCENARIOS)
    print("hashes", hashes, "deterministic", deterministic, flush=True)

    out = {}
    for s in SCENARIOS:
        r = res[s]
        b = r["bets"]
        sel = b[b["selected"]]
        above = b[b["above_threshold"]]
        o = {"sel": sel, "above": above}
        o["overall"] = S.ledger_metrics(sel, N_BOOT)
        o["overall_nowl"] = S.ledger_metrics(above, N_BOOT)
        o["by_market"] = S.cell_metrics(sel, ["market"], N_BOOT)
        o["by_market_nowl"] = S.cell_metrics(above, ["market"], N_BOOT)
        o["by_season"] = S.cell_metrics(sel, ["season"], N_BOOT)
        o["by_season_nowl"] = S.cell_metrics(above, ["season"], N_BOOT)
        o["by_div_nowl"] = S.cell_metrics(above, ["market", "div"], N_BOOT)
        o["by_source_nowl"] = S.cell_metrics(above, ["source"], N_BOOT)
        o["by_cell"] = S.cell_metrics(sel, ["div", "market", "season"], N_BOOT)
        o["by_div"] = S.cell_metrics(sel, ["market", "div"], N_BOOT)
        # baselines on the same ledger
        fold_seasons = set(b["season"])
        fav_all = pd.concat(r["fav"].values())
        fav_all = fav_all[fav_all["season"].isin(fold_seasons)]
        fav_same = fav_all[
            fav_all.set_index("market", append=True).index.isin(
                sel.set_index("market", append=True).index
            )
        ]
        cand = pd.concat(r["long"].values())
        cand = cand[cand["season"].isin(fold_seasons)]
        counts = sel.groupby(["div", "season", "market"]).size().rename("n").reset_index()
        rnd = S.random_baseline(cand, counts, N_RAND)
        o["fav_same"] = S.ledger_metrics(fav_same, N_BOOT) if len(fav_same) else {"n": 0}
        o["fav_all"] = S.ledger_metrics(fav_all, N_BOOT)
        o["rand"] = rnd
        out[s] = o
        print(s, "selected bets", len(sel), "above", len(above), flush=True)

    rows = []
    for market, dm in data.items():
        bad = prices["_guard"][market]
        g = pd.DataFrame({"season": dm["frame"]["season"].to_numpy(), "bad": bad.to_numpy()})
        for season, gg in g.groupby("season"):
            rows.append(
                {
                    "market": market,
                    "season": season,
                    "n_rows": len(gg),
                    "n_dropped": int(gg["bad"].sum()),
                    "share": f(gg["bad"].mean(), 3),
                }
            )
    out["_guard_tbl"] = pd.DataFrame(rows)

    ae = devig_ae(data)
    bench, _ = benchmark_table(data)
    pools = pool_ci(data)
    sweep = sweep_table(res)
    pytest_txt = run_pytest_counts()

    # ---- gate evaluation
    h = out[HEADLINE]
    ov = h["overall"]
    rnd = h["rand"]
    n_sel = ov.get("n", 0)
    beat_fav = (
        n_sel > 0
        and ov["clv_net"] > h["fav_same"].get("clv_net", np.inf)
        and ov["clv_net"] > h["fav_all"]["clv_net"]
    )
    p_rand = float((rnd["clv_net"] >= ov["clv_net"]).mean()) if n_sel else np.nan
    beat_rand = n_sel > 0 and p_rand <= 0.025
    sizing = ae["A/E"].sub(1).abs().max()
    wc = pools[pools["vs"] == "close"]
    ok_models = [
        m
        for m, g in wc.groupby("model")
        if len(g) == 2 and (g["lo"] > 0.001).all()  # weights at the bound are ~1e-8, not > 0
    ]
    w_ok = bool(ok_models)
    clv_pos = n_sel > 0 and ov["clv_net_lo"] > 0
    gates = {
        "B0.1": (
            "PASS" if sizing < 0.03 else "FAIL",
            f"max |A/E-1| = {sizing:.3f} over pre/close x selection on the evaluation rows (<0.03 required); full per-league report reports/devig_sanity.md",
        ),
        "B0.2": (
            "PASS",
            "season-expanding folds only (`fedge.backtest.walkforward`, tests/test_stats_walkforward.py); this card's pool weight, model, threshold, whitelist, Kelly fraction use seasons < test season; no random split anywhere",
        ),
        "B0.3": (
            "PASS"
            if "failed" not in pytest_txt and "error" not in pytest_txt and "passed" in pytest_txt
            else "FAIL",
            f"tests/test_leakage.py (leak tests incl. regime reversal), tests/test_stats_walkforward.py (embargo) and tests/test_strategy.py (selection ignores future seasons; flipping later outcomes cannot change earlier bets): {pytest_txt}",
        ),
        "B0.4": (
            "PASS" if deterministic else "FAIL",
            "this script runs the full strategy pipeline twice in-process and compares sha256 of the selected-bet ledger: "
            + "; ".join(f"{k}={v[:12]}" for k, v in hashes.items())
            + (" (identical)" if deterministic else " (DIFFERENT)"),
        ),
        "B0.5": (
            "PASS",
            "per-season RPS / log loss / Brier / slope / ECE for the pooled model, lgb_xg and the de-margined market close and pre-close: section 6 and reports/v2_main_metrics_by_league_season.csv. The model does NOT beat the close (see table)",
        ),
        "B0.6": (
            "PASS" if w_ok else "FAIL",
            "pool weight vs margin-free close, all test seasons (CI must exclude 0 in BOTH markets): "
            + "; ".join(
                f"{r.model} {r.market} w={r.w:.3f} CI [{r.lo:.3f},{r.hi:.3f}]"
                for r in wc[wc["model"].isin(["lgb_xg", "lgbd_xg"])].itertuples()
            )
            + (f". Passing: {', '.join(ok_models)}" if ok_models else ". No model passes"),
        ),
        "B0.7": (
            "PASS" if (beat_fav and beat_rand) else "FAIL",
            "; ".join(baseline_text(s_, out[s_]) for s_ in SCENARIOS)
            + ". Headline (exch) must beat both baselines on net CLV; it placed no bets, so it cannot",
        ),
        "B0.8": (
            "PARTIAL",
            "modelled: 6% commission on net winnings per market (`stats.simulate_bankroll`), bet-time policy (pre-closing snapshot, kickoff-24h ASSUMED not verified), per-bet stake cap 1% of bankroll (limits.toml), one bet per match-market. NOT demonstrable on football-data history: order-book depth/liquidity and takeable-price checks (no depth in the data); deferred to Phase 6 (stake only if snapshot depth >= stake)",
        ),
        "B0.9": (
            "PASS",
            "threshold, Kelly fraction, whitelist, model and pool weight chosen in-fold per test season (section 4, config table); test_strategy.py checks the choice is unchanged when future seasons are altered",
        ),
        "B0.10": (
            "PASS",
            "losing runs reported with CIs: every scenario, the no-whitelist variant, and per-cell results below (including negative cells); nothing was dropped",
        ),
    }
    gate_pass = all(v[0] == "PASS" for v in gates.values()) and clv_pos
    cfg, cfg_rows = final_config(data, res[HEADLINE], gate_pass)
    (CONFIG / "strategy.toml").write_text(
        S.render_strategy_toml(cfg), encoding="utf-8", newline="\n"
    )
    tomllib.loads((CONFIG / "strategy.toml").read_text(encoding="utf-8"))  # must parse

    # ---- csv artefacts
    cells = pd.concat(
        [out[s]["by_cell"].assign(scenario=s) for s in SCENARIOS if len(out[s]["by_cell"])]
    )
    cells.to_csv(REPORTS / "gate0_cells.csv", index=False, float_format="%.6g")
    lcols = [
        "match_id",
        "div",
        "season",
        "kickoff_utc",
        "market",
        "sel",
        "source",
        "model",
        "edge",
        "p",
        "price",
        "p_close",
        "won",
        "clv_net",
        "kelly",
        "whitelisted",
        "selected",
    ]
    for s_ in SCENARIOS:
        out[s_]["above"].reset_index()[lcols].to_csv(
            REPORTS / f"gate0_ledger_{s_}.csv", index=False, float_format="%.6g"
        )
    pd.concat([res[s]["cfgs"].assign(scenario=s) for s in SCENARIOS]).to_csv(
        REPORTS / "gate0_fold_configs.csv", index=False, float_format="%.6g"
    )

    write_report(
        data,
        res,
        out,
        ae,
        bench,
        pools,
        gates,
        gate_pass,
        cfg,
        cfg_rows,
        hashes,
        p_rand,
        pytest_txt,
        sweep,
    )
    print("done; gate0_passed =", gate_pass, flush=True)


def write_report(
    data,
    res,
    out,
    ae,
    bench,
    pools,
    gates,
    gate_pass,
    cfg,
    cfg_rows,
    hashes,
    p_rand,
    pytest_txt,
    sweep,
):  # pragma: no cover
    L: list[str] = []
    h = out[HEADLINE]
    L += [
        "# football-edge Gate 0 report: early-line value strategy backtest (Phase 5)",
        "",
        "Generated by `uv run python scripts/run_strategy.py`; companion files `gate0_cells.csv` (div x market x season x scenario metrics), `gate0_ledger_<scenario>.csv` (every bet above the in-fold threshold, whitelisted or not; `selected` bets are those in whitelisted divisions), `gate0_fold_configs.csv` (in-fold choices), `../config/strategy.toml`.",
        "",
        f"**Result: Gate 0 {'PASSED' if gate_pass else 'NOT passed'}.** Gate 0 certifies the measurement, not the edge (M3 section 5). The strategy was evaluated honestly and is reported whatever it did.",
        "",
        "**Summary.** " + summary_text(out, res),
        "",
        "## 1. What was run",
        "",
        "- Test seasons: the P3/P4 walk-forward predictions (18 divisions, 2016/17 to 2026/27 partial). Pool weights use seasons before the test season, so the first usable season is 2017/18; the in-fold selection needs two prior seasons of ledger, so strategy bets start in 2019/20.",
        "- Fair probability: log opinion pool of one model with the de-margined (power) pre-closing reference price of the same match; `w` fit on all earlier seasons per market and model. Candidate models: dc, elo, pi (1X2 only), lgb_goals, lgb_xg, cb_goals, cb_xg, lgbd_xg. Model per fold: lowest pooled log loss on history.",
        "- Bet price: the pre-closing price. Source rule per match: Betfair Exchange (BFE) if present, else Pinnacle (PS, never after the 2025-07-23 staleness cut, stale rows are excluded upstream), (an Avg fallback was never needed: every evaluation row has a BFE or PS price, because the evaluation join already required a reference book). Scenarios: `exch` = BFE/PS (headline), `max` = best-of-market `Max` price (optimistic). All are run through the same machinery.",
        f"- Edge = p x (1 + (price-1) x (1-{COMMISSION:.2f})) - 1, i.e. EV per unit stake with 6% commission on winnings. At most one bet per match-market (largest edge). A bet needs edge > threshold and (div, market) in the in-fold whitelist.",
        "- In-fold choices from prior seasons only: model (min pooled log loss); threshold in {0, 0.02, 0.04, 0.06, 0.08} (maximise mean net CLV x sqrt(n), at least 100 history bets); whitelist = divisions with >= 25 history bets and mean net CLV > 0; Kelly fraction in {0.1, 0.25} (best log growth on the history ledger subject to a 20% drawdown cap, stake cap 1% of bankroll).",
        "- CLV = log(price x p_close_fair), p_close_fair = margin-free (power) closing probability of the reference close book. **net CLV** uses the commission-adjusted price (the one the strategy is judged on); **raw CLV** uses the quoted price. CIs: match-clustered bootstrap, 1000 resamples, seed 0.",
        "- ROI: flat = 1 unit per bet, net of 6% commission on winnings; Kelly = fractional Kelly on commission-adjusted odds, daily bankroll updates, per-market commission, 1000 starting bankroll. Max drawdown on that bankroll path.",
        "- Baselines on the same ledger: back-the-favourite (shortest price) on (a) the matches the strategy bet and (b) every match in the test folds; random selection with the same bet count per (div, season, market) cell, 200 replicates.",
        "",
        "## 2. Headline results (exch = BFE/PS pre-closing price)",
        "",
    ]
    for s in SCENARIOS:
        o = out[s]
        L += [f"### Scenario `{s}`" + (" (headline)" if s == HEADLINE else ""), ""]
        rows = []
        for label, m in (
            ("overall (whitelist+threshold)", o["overall"]),
            ("overall, whitelist NOT applied", o["overall_nowl"]),
        ):
            rows.append({"slice": label, **m})
        for r in o["by_market"].to_dict("records"):
            rows.append({**r, "slice": f"market {r['market']}"})
        for r in o["by_market_nowl"].to_dict("records"):
            rows.append({**r, "slice": f"market {r['market']}, no whitelist"})
        t = pd.DataFrame(rows)
        if len(t) and "clv_net" in t:
            t = fmt_metrics(t)
            L += [md(t, ["slice", *METRIC_COLS], ["slice", *METRIC_HEADS]), ""]
        else:
            L += ["No bets placed.", ""]
    L += [
        "### Max price guard (data-quality filter, chosen after inspecting the 2026/27 Max prices)",
        "",
    ]
    L += [
        f"A first run showed `Max` over/under prices of 2.25 to 2.50 where Avg and BFE quoted 1.5 to 2.0, giving a spurious +13% net CLV on 25 bets in 2026/27. The Max columns of 2024/25+ contain such quotes. `max` rows are therefore dropped when any selection's Max price exceeds {MAX_RATIO_CAP:.2f} x the BFE/PS price of that selection. This guard was added after seeing that result; it removes the data errors but is blunt: it also drops legitimate rows where Max is more than 10% above BFE/PS on a long-priced selection (6-12% of 1X2 rows in every season). It was not tuned on results. Rows dropped:",
        "",
        md(out["_guard_tbl"], ["market", "season", "n_rows", "n_dropped", "share"]),
        "",
    ]
    for s in SCENARIOS:
        o = out[s]
        L += [f"### Per season, scenario `{s}`", ""]
        for label, key in (
            ("selected bets (threshold + whitelist)", "by_season"),
            ("threshold only, no whitelist", "by_season_nowl"),
        ):
            t = o[key]
            L += [
                f"{label}:",
                "",
                md(fmt_metrics(t), ["season", *METRIC_COLS], ["season", *METRIC_HEADS])
                if len(t)
                else "No bets.",
                "",
            ]
        L += [f"### Per price source, scenario `{s}` (threshold only, no whitelist)", ""]
        t = o["by_source_nowl"]
        L += [
            md(fmt_metrics(t), ["source", *METRIC_COLS], ["source", *METRIC_HEADS])
            if len(t)
            else "No bets.",
            "",
        ]

    L += [
        "## 3. Per league and market (all test seasons pooled; every cell, winners and losers)",
        "",
    ]
    for s in SCENARIOS:
        o = out[s]
        for label, key in (
            ("selected bets", "by_div"),
            ("threshold only, no whitelist", "by_div_nowl"),
        ):
            t = o[key]
            L += [f"### Scenario `{s}`, {label}", ""]
            L += [
                md(
                    fmt_metrics(t),
                    ["market", "div", *METRIC_COLS],
                    ["market", "div", *METRIC_HEADS],
                )
                if len(t)
                else "No bets.",
                "",
            ]
    L += [
        "Per division x market x season cells (selected bets): `reports/gate0_cells.csv`.",
        "",
    ]

    L += ["## 4. In-fold configuration chosen for each test season", ""]
    for s in SCENARIOS:
        L += [f"### Scenario `{s}`", ""]
        cf = res[s]["cfgs"]
        L += [
            md(
                cf.assign(
                    threshold=cf["threshold"].map(lambda v: f"{v:.2f}"),
                    kelly=cf["kelly"].map(lambda v: f"{v:.2f}"),
                    hist_clv=cf["hist_clv"].map(lambda v: f(v, 4, True)).values,
                ),
                [
                    "season",
                    "market",
                    "model",
                    "threshold",
                    "kelly",
                    "hist_n",
                    "hist_clv",
                    "whitelist",
                ],
            )
            if len(cf)
            else "No fold had enough history to pick a threshold (needs 100 history bets above a threshold): no bets were placed.",
            "",
        ]

    L += [
        "### Diagnostic: fixed-model threshold sweep on every pooled season (NOT used for any selection)",
        "",
        "Pooled probability from in-fold weights (`w` fit on earlier seasons), 2017/18 to 2026/27. Shows what the strategy would do if forced to bet, including where it loses. CLV CI: 300 resamples.",
        "",
    ]
    sw = sweep.copy()
    for c in ("clv_net", "lo", "hi", "clv_raw", "roi_flat"):
        sw[c] = sw[c].map(lambda v: f(v, 4, True))
    sw["clv_net_ci"] = sw["clv_net"] + " [" + sw["lo"] + ", " + sw["hi"] + "]"
    sw["thr"] = sw["thr"].map(lambda v: f"{v:.2f}")
    L += [
        md(
            sw,
            ["scenario", "market", "model", "thr", "n", "clv_net_ci", "clv_raw", "roi_flat"],
            [
                "scenario",
                "market",
                "model",
                "edge >",
                "n",
                "net CLV [95% CI]",
                "raw CLV",
                "flat ROI",
            ],
        ),
        "",
    ]

    L += ["## 5. Baselines on the same ledger", ""]
    rows = []
    for s in SCENARIOS:
        o = out[s]
        if not o["overall"].get("n"):
            continue
        rn = o["rand"]
        rows.append(
            {
                "scenario": s,
                "strategy n": o["overall"]["n"],
                "strategy net CLV": f(o["overall"]["clv_net"], 4, True),
                "fav (same matches) n": o["fav_same"].get("n"),
                "fav (same) net CLV": f(o["fav_same"].get("clv_net"), 4, True),
                "fav (all fold matches) net CLV": f(o["fav_all"].get("clv_net"), 4, True),
                "random net CLV mean [2.5%, 97.5%]": f"{f(rn['clv_net'].mean(), 4, True)} [{f(rn['clv_net'].quantile(0.025), 4, True)}, {f(rn['clv_net'].quantile(0.975), 4, True)}]",
                "P(random >= strategy)": f(
                    float((rn["clv_net"] >= o["overall"]["clv_net"]).mean()), 3
                ),
                "strategy flat ROI": f(o["overall"]["roi_flat"], 2, True),
                "fav (same) flat ROI": f(o["fav_same"].get("roi_flat"), 2, True),
                "fav (all) flat ROI": f(o["fav_all"].get("roi_flat"), 2, True),
                "random raw CLV mean": f(rn["clv_raw"].mean(), 4, True),
                "random flat ROI mean": f(rn["roi_flat"].mean(), 2, True),
            }
        )
    if rows:
        t = pd.DataFrame(rows)
        L += [md(t, list(t.columns)), ""]
    else:
        L += ["No bets placed in any scenario.", ""]

    L += [
        "## 6. B0.5 benchmark: model vs de-margined sharp close, every season",
        "",
        "Slope = pooled one-vs-rest calibration slope; ECE on the pooled one-vs-rest probabilities. `pooled_*` rows are the in-fold pooled probabilities the strategy actually used (NaN/absent in 2016/17).",
        "",
    ]
    for market in S.SELECTIONS:
        bm = bench[bench["market"] == market].copy()
        for c in ("rps", "log_loss", "brier", "slope", "ece"):
            bm[c] = bm[c].map(lambda v: f"{v:.4f}")
        L += [
            f"### {market}",
            "",
            md(bm, ["season", "model", "n", "rps", "log_loss", "brier", "slope", "ece"]),
            "",
        ]

    L += [
        "## 7. B0.6 pool weight on the model, all test seasons (match-clustered bootstrap, 150 resamples)",
        "",
    ]
    pt = pools.copy()
    for c in ("w", "lo", "hi"):
        pt[c] = pt[c].map(lambda v: f"{v:.3f}")
    L += [
        md(
            pt,
            ["market", "model", "vs", "n", "w", "lo", "hi"],
            ["market", "model", "vs", "n", "w", "CI lo", "CI hi"],
        ),
        "",
    ]

    L += ["## 8. B0.1 de-vig A/E on the evaluation rows", ""]
    at = ae.copy()
    at["A/E"] = at["A/E"].map(lambda v: f"{v:.3f}")
    L += [md(at, ["market", "phase", "sel", "n", "A/E"]), ""]

    L += [
        "## 9. Gate 0 checklist (M3 section 5)",
        "",
        "Status: PASS / FAIL / PARTIAL (PARTIAL = part of the acceptance test cannot be demonstrated on historical data). Gate 0 passes only if every item is PASS **and** the headline net CLV CI excludes 0 (this card's additional bar).",
        "",
        "| # | Requirement | Status | Evidence |",
        "|---|---|---|---|",
    ]
    names = {
        "B0.1": "De-vig sanity",
        "B0.2": "Walk-forward only",
        "B0.3": "No lookahead (tested)",
        "B0.4": "Determinism",
        "B0.5": "Market benchmark reported",
        "B0.6": "Model adds information",
        "B0.7": "Baselines beaten",
        "B0.8": "Execution modelled",
        "B0.9": "Parameters chosen in-fold",
        "B0.10": "Honest reporting of failures",
    }
    for k, (st, ev) in gates.items():
        L.append(f"| {k} | {names[k]} | **{st}** | {ev} |")
    ov = h["overall"]
    L += [
        "",
        f"Headline strategy: n={ov.get('n', 0)}, net CLV {f(ov.get('clv_net'), 4, True)} [{f(ov.get('clv_net_lo'), 4, True)}, {f(ov.get('clv_net_hi'), 4, True)}], flat ROI {f(ov.get('roi_flat'), 2, True)}. **Gate 0: {'PASS' if gate_pass else 'NOT PASSED'}.**",
        "",
    ]

    L += [
        "## 10. Caveats and unverified assumptions",
        "",
        "- The `pre` price in football-data.co.uk has no true timestamp; P1 assumed kickoff-24h (`available_at`). If the snapshot is actually earlier or later, bet timing, CLV and the pool weights (which are fit against this price) change. UNVERIFIED.",
        "- BFE prices exist only from 2024/25 (and PS is used before): `exch` mixes two sources, reported separately in section 2. Exchange commission is applied to PS prices too (a PS bet would not pay commission but would face a margin; the exchange-equivalent assumption is the conservative one).",
        "- `max` is the best price across many books at an unknown time and is not obtainable at size; treat it as an upper bound. Its RAW CLV of about +3% (above the 3% leakage alarm in AGENTS.md) was investigated: (1) a first run showed +13% net CLV on 25 bets in 2026/27 that were bad Max quotes (guard section); (2) after the guard, the random same-count baseline on the same Max prices has clearly negative CLV (section 5: random raw and net CLV columns) and the favourite baseline on the same matches +0.14% net, so the strategy's lead over random is the market-pooled edge selection picking the largest Max-vs-pool gaps, i.e. best-of-market price optimism plus selecting extreme quotes, not model skill; (3) net CLV (commission applied, though a real bookmaker bet pays none) is +0.47% with a CI including 0 and flat ROI is negative. No lookahead path was found: pool weights, model, threshold, whitelist and Kelly fraction only see earlier seasons (tests).",
        "- Liquidity, price-moving and bet limits are not modelled (B0.8 PARTIAL).",
        "- Whitelist/threshold/model selection is a multiple-comparison exercise on a short history; the out-of-sample folds are the only fair read.",
        "- 2026/27 is a partial season.",
        "",
    ]

    L += ["## 11. Recommended paper-trading config", ""]
    if gate_pass:
        L += [
            "Gate 0 passed: run the Phase 6 paper desk with the config below (`config/strategy.toml`, mode `paper`)."
        ]
    else:
        L += [
            'Gate 0 did **not** pass, so `config/strategy.toml` has `gate0_passed = false` and `mode = "shadow"`: the desk may snapshot prices and log hypothetical bets, but this card does not recommend treating any of it as an edge. It is the best in-fold configuration, written so Phase 6 can run in shadow mode and gather the live, timestamped evidence (T-24h snapshot, depth, closing price) that history cannot provide.'
        ]
    L += ["", "```toml", (CONFIG / "strategy.toml").read_text(encoding="utf-8").rstrip(), "```", ""]
    if len(cfg_rows):
        L += [
            "In-fold selection behind it (history = every season to date):",
            "",
            md(
                cfg_rows.assign(
                    w_all=cfg_rows["w_all"].map(lambda v: f"{v:.3f}"),
                    hist_clv=cfg_rows["hist_clv"].map(lambda v: f(v, 4, True)),
                ),
                ["market", "model", "w_all", "threshold", "hist_n", "hist_clv", "whitelist"],
            ),
            "",
        ]
    L += [
        f"Selected-bet ledger hashes (sha256, exch/max): {', '.join(f'{k} {v[:16]}' for k, v in hashes.items())}",
        "",
    ]
    (REPORTS / "gate0.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":  # pragma: no cover
    main()
