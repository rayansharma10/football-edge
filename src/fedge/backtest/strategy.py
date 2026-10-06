"""Early-line value strategy: bet the pre-closing price when the pooled edge beats a threshold.

Pipeline (all walk-forward; nothing here sees a season it is not allowed to):

1. ``pooled_by_season``: for each test season S and model, the log-opinion-pool weight ``w`` is fit
   on seasons strictly before S only, then used to pool the model with the de-margined
   pre-closing reference price of S. The first season has no history and gets NaN (no bet).
2. ``candidate_table``: per match-market, the selection with the largest edge
   ``p * net_odds(price) - 1`` (commission-adjusted back odds, 6% of net winnings).
3. ``select_market_config`` / ``select_kelly``: model, edge threshold, (div) whitelist and Kelly
   fraction are chosen on the ledger of seasons strictly before the fold, never on the fold.
4. ``run_walk_forward``: applies the in-fold choice to each test fold; ``cell_metrics``,
   ``favourite_table`` and ``random_baseline`` score it on the same ledger.

CLV is ``log(price * p_close_fair)`` (``fedge.backtest.stats.log_clv``) against the margin-free
closing probability; the ``net`` version uses the commission-adjusted price, the ``raw`` one the
quoted price. Two metrics, two jobs:

* **raw log-CLV is the skill metric**: did the price taken beat the fair closing price? This is the
  M3 P1.2 definition and the only version comparable across scenarios and bet prices.
* **net log-CLV is the money metric** and the in-fold selection target: commission is charged on
  net market winnings, so a long price really is worth less than its quote (at the headline
  ledger's mean price 7.46 the commission term is about -4.7%, turning +0.97% raw CLV into -3.76%
  net). ``select_market_config``/``select_kelly`` therefore optimise the quantity the desk is paid.

Both are reported for every scenario; ``net`` sets the gate bar and ``raw`` is quoted beside it.

``select_market_config`` deliberately has **no positivity floor**: the argmax threshold is accepted
whenever its history subset has ``MIN_HIST_BETS`` rows, even when its mean net CLV is negative. The
committed Gate 0 run selects exactly such folds (reports/gate0.md section 1), so the threshold rule
alone can never return "no bets" - only the whitelist can. A ``hist_clv > 0`` floor was considered
and rejected: it would empty the headline ``exch`` ledger in every fold, hiding the configuration
instead of fixing it. The behaviour is documented and pinned by a test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fedge.backtest.stats import (
    COMMISSION,
    bootstrap_ci,
    log_clv,
    net_odds,
    simulate_bankroll,
)
from fedge.models.pool import fit_weight, log_pool

SELECTIONS = {"1x2": ("H", "D", "A"), "ou25": ("over", "under")}
THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08)  # minimum edge (EV per unit stake after commission)
KELLY_FRACTIONS = (0.1, 0.25)
MIN_HIST_BETS = 100  # history bets needed before a threshold can win the in-fold search
# Phase 6 desk edge bar while Gate 0 has not passed (written to config/strategy.toml as [shadow]).
SHADOW_THRESHOLD = 0.03
MIN_CELL_BETS = 25  # history bets needed before a div can enter the whitelist
DD_CAP = 0.20  # drawdown cap used when choosing the Kelly fraction (limits.toml drawdown_kill)
MIN_HIST_SEASONS = 2  # seasons of prior ledger required before the strategy bets at all


# ------------------------------------------------------------------ pooling
def pooled_by_season(
    model_p: dict[str, np.ndarray],
    market_p: np.ndarray,
    y: np.ndarray,
    season: Sequence[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Pool each model with the market using a weight fit on strictly earlier seasons.

    Returns ``({model: pooled (n, k) probs, NaN for the first season}, weights table)`` where the
    table has one row per (model, season) with the weight used and the rows it was fit on.
    """
    season = np.asarray(season).astype(str)
    seasons = sorted(set(season))
    out = {m: np.full(market_p.shape, np.nan) for m in model_p}
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        fit = np.isin(season, seasons[:i])
        test = season == s
        for m, p in model_p.items():
            w = fit_weight(p[fit], market_p[fit], y[fit])
            out[m][test] = log_pool(p[test], market_p[test], w)
            rows.append({"model": m, "season": s, "w": w, "n_fit": int(fit.sum())})
    return out, pd.DataFrame(rows, columns=["model", "season", "w", "n_fit"])


def pooled_log_loss_table(
    pooled: dict[str, np.ndarray], y: np.ndarray, season: Sequence[str]
) -> pd.DataFrame:
    """Mean pooled log-loss per (model, season) (rows with NaN pooled probs are skipped)."""
    season = np.asarray(season).astype(str)
    rows = []
    for m, p in pooled.items():
        ok = ~np.isnan(p).any(axis=1)
        pr = np.clip(p[np.arange(len(y)), y], 1e-12, None)
        for s in sorted(set(season[ok])):
            sel = ok & (season == s)
            rows.append(
                {"model": m, "season": s, "n": int(sel.sum()), "ll": -np.log(pr[sel]).mean()}
            )
    return pd.DataFrame(rows, columns=["model", "season", "n", "ll"])


# ------------------------------------------------------------------ ledger
def _finish(df: pd.DataFrame, commission: float) -> pd.DataFrame:
    df["clv_net"] = log_clv(net_odds(df["price"], commission), df["p_close"])
    df["clv_raw"] = log_clv(df["price"], df["p_close"])
    df["pnl_flat"] = np.where(df["won"], (df["price"] - 1.0) * (1.0 - commission), -1.0)
    df["market_id"] = df.index.astype(str) + "|" + df["market"]
    return df


def candidate_table(
    meta: pd.DataFrame,
    market: str,
    price: np.ndarray,
    p_close: np.ndarray,
    pooled: dict[str, np.ndarray],
    y: np.ndarray,
    commission: float = COMMISSION,
) -> dict[str, pd.DataFrame]:
    """Per model: one row per match with the max-edge selection (the only bet we would place).

    ``meta`` has index ``match_id`` and columns ``div``, ``season``, ``kickoff_utc``.
    ``price`` / ``p_close`` / each ``pooled[model]`` are (n, k) arrays aligned with ``meta``;
    rows with NaN in any of them are dropped. ``y`` is the winning column index.
    """
    sels = np.asarray(SELECTIONS[market])
    nodds = net_odds(price, commission)
    out = {}
    for m, p in pooled.items():
        ok = ~(
            np.isnan(p).any(axis=1) | np.isnan(price).any(axis=1) | np.isnan(p_close).any(axis=1)
        )
        edge = np.where(ok[:, None], p * nodds - 1.0, -np.inf)
        k = edge.argmax(axis=1)  # first max wins ties: deterministic
        r = np.arange(len(k))
        df = meta.copy()
        df["market"] = market
        df["sel"] = sels[k]
        df["edge"] = edge[r, k]
        df["p"] = p[r, k]
        df["price"] = price[r, k]
        df["p_close"] = p_close[r, k]
        df["won"] = y == k
        out[m] = _finish(df.loc[ok].copy(), commission)
    return out


def favourite_table(
    meta: pd.DataFrame,
    market: str,
    price: np.ndarray,
    p_close: np.ndarray,
    y: np.ndarray,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """Back the shortest-priced selection of every match (same price source as the strategy)."""
    ok = ~(np.isnan(price).any(axis=1) | np.isnan(p_close).any(axis=1))
    k = np.where(ok[:, None], price, np.inf).argmin(axis=1)
    r = np.arange(len(k))
    df = meta.copy()
    df["market"] = market
    df["sel"] = np.asarray(SELECTIONS[market])[k]
    df["price"] = price[r, k]
    df["p_close"] = p_close[r, k]
    df["won"] = y == k
    return _finish(df.loc[ok].copy(), commission)


def long_candidates(
    meta: pd.DataFrame,
    market: str,
    price: np.ndarray,
    p_close: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """Every (match, selection) with a price: the population the random baseline draws from."""
    sels = SELECTIONS[market]
    ok = ~(np.isnan(price).any(axis=1) | np.isnan(p_close).any(axis=1))
    parts = []
    for j, s in enumerate(sels):
        d = meta.loc[ok].copy()
        d["market"] = market
        d["sel"] = s
        d["price"] = price[ok, j]
        d["p_close"] = p_close[ok, j]
        d["won"] = y[ok] == j
        parts.append(d)
    return pd.concat(parts)


# ------------------------------------------------------------------ in-fold selection
def _score(clv: np.ndarray) -> float:
    """Selection score on history: mean net CLV x sqrt(n) (a t-statistic-like trade-off)."""
    n = len(clv)
    return float(np.mean(clv) * np.sqrt(n)) if n else -np.inf


def pick_model(ll: pd.DataFrame, hist: Sequence[str]) -> str | None:
    """Model with the lowest row-weighted pooled log-loss over the history seasons."""
    h = ll[ll["season"].isin(list(hist))]
    if h.empty:
        return None
    g = h.assign(t=h["ll"] * h["n"]).groupby("model")[["t", "n"]].sum()
    return str((g["t"] / g["n"]).sort_values(kind="mergesort").index[0])


def select_market_config(
    best: pd.DataFrame,
    hist: Sequence[str],
    thresholds: Sequence[float] = THRESHOLDS,
    min_bets: int = MIN_HIST_BETS,
    min_cell: int = MIN_CELL_BETS,
) -> dict | None:
    """Threshold and whitelist for one market and one model, from history seasons only.

    There is no positivity floor on the chosen threshold (see the module docstring and
    reports/gate0.md section 1): the argmax of ``_score`` wins as long as its subset has
    ``min_bets`` rows, so ``hist_clv`` may be negative. ``hist_clv_raw`` is the same mean on the
    quoted-price CLV, the skill metric quoted beside the money metric.
    """
    h = best[best["season"].isin(list(hist))]
    best_t, best_s = None, -np.inf
    for t in thresholds:
        sel = h[h["edge"] > t]
        if len(sel) < min_bets:
            continue
        s = _score(sel["clv_net"].to_numpy())
        if s > best_s:
            best_t, best_s = t, s
    if best_t is None:
        return None
    sel = h[h["edge"] > best_t]
    cells = sel.groupby("div")["clv_net"].agg(["size", "mean"])
    wl = sorted(cells.index[(cells["size"] >= min_cell) & (cells["mean"] > 0)].tolist())
    return {
        "threshold": float(best_t),
        "whitelist": wl,
        "hist_n": int(len(sel)),
        "hist_clv": float(sel["clv_net"].mean()),
        "hist_clv_raw": (
            float(sel["clv_raw"].mean()) if "clv_raw" in sel.columns else float("nan")
        ),
        "hist_score": float(best_s),
    }


def select_kelly(
    hist_bets: pd.DataFrame,
    fractions: Sequence[float] = KELLY_FRACTIONS,
    max_stake_frac: float = 0.01,
    dd_cap: float = DD_CAP,
    commission: float = COMMISSION,
) -> float:
    """Kelly fraction with the best log-growth on the history ledger, subject to the drawdown cap.

    Falls back to the smallest fraction when there is no history or every fraction breaches the cap.
    """
    fr = sorted(fractions)
    if hist_bets.empty:
        return float(fr[0])
    best_f, best_g = None, -np.inf
    for f in fr:
        _, s = simulate_bankroll(
            hist_bets.rename(columns={"kickoff_utc": "date"}),
            mode="kelly",
            kelly_mult=f,
            max_stake_frac=max_stake_frac,
            commission=commission,
        )
        if s["max_drawdown"] > dd_cap:
            continue
        g = float(np.log(max(s["final_bankroll"], 1e-9) / 1000.0))
        if g > best_g:
            best_f, best_g = f, g
    return float(best_f if best_f is not None else fr[0])


def run_walk_forward(
    best_by_model: dict[str, dict[str, pd.DataFrame]],
    ll_by_market: dict[str, pd.DataFrame],
    seasons: Sequence[str],
    thresholds: Sequence[float] = THRESHOLDS,
    fractions: Sequence[float] = KELLY_FRACTIONS,
    min_hist_seasons: int = MIN_HIST_SEASONS,
    max_stake_frac: float = 0.01,
    min_bets: int = MIN_HIST_BETS,
    min_cell: int = MIN_CELL_BETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the in-fold selection to every test season.

    ``best_by_model[market][model]`` is a ``candidate_table`` frame. For test season S the
    configuration is chosen from rows with season < S only. Returns (bets, configs): ``bets``
    holds every candidate of the test seasons with ``selected`` (passes threshold and whitelist),
    ``whitelisted`` and ``kelly`` columns; ``configs`` has one row per (season, market).
    """
    seasons = sorted(seasons)
    parts, cfgs = [], []
    for i, s in enumerate(seasons):
        hist = seasons[:i]
        if len(hist) < min_hist_seasons:
            continue
        chosen, hist_sel = {}, []
        for market, models in best_by_model.items():
            m = pick_model(ll_by_market[market], hist)
            if m is None or m not in models:
                continue
            cfg = select_market_config(
                models[m], hist, thresholds, min_bets=min_bets, min_cell=min_cell
            )
            if cfg is None:
                continue
            chosen[market] = (m, cfg)
            hb = models[m][models[m]["season"].isin(hist)]
            hist_sel.append(hb[(hb["edge"] > cfg["threshold"]) & hb["div"].isin(cfg["whitelist"])])
        hist_bets = pd.concat(hist_sel) if hist_sel else pd.DataFrame()
        kelly = (
            select_kelly(hist_bets, fractions, max_stake_frac)
            if len(hist_bets)
            else float(min(fractions))
        )
        for market, (m, cfg) in chosen.items():
            fold = best_by_model[market][m]
            fold = fold[fold["season"] == s].copy()
            fold["model"] = m
            fold["whitelisted"] = fold["div"].isin(cfg["whitelist"])
            fold["above_threshold"] = fold["edge"] > cfg["threshold"]
            fold["selected"] = fold["above_threshold"] & fold["whitelisted"]
            fold["threshold"] = cfg["threshold"]
            fold["kelly"] = kelly
            parts.append(fold)
            cfgs.append(
                {
                    "season": s,
                    "market": market,
                    "model": m,
                    "threshold": cfg["threshold"],
                    "kelly": kelly,
                    "whitelist": ",".join(cfg["whitelist"]),
                    "hist_n": cfg["hist_n"],
                    "hist_clv": cfg["hist_clv"],
                    "hist_clv_raw": cfg.get("hist_clv_raw", float("nan")),
                }
            )
    if parts:
        bets = pd.concat(parts)
    else:  # nothing qualified in any fold: an empty ledger with the full column set
        some = next(iter(next(iter(best_by_model.values())).values()))
        extra = {
            "model": str,
            "whitelisted": bool,
            "above_threshold": bool,
            "selected": bool,
            "threshold": float,
            "kelly": float,
        }
        bets = some.iloc[0:0].assign(**{c: pd.Series(dtype=t) for c, t in extra.items()})
    return bets, pd.DataFrame(cfgs)


# ------------------------------------------------------------------ metrics
def _sim(bets: pd.DataFrame, mode: str, kelly_mult: float, max_stake_frac: float) -> dict:
    _, s = simulate_bankroll(
        bets.rename(columns={"kickoff_utc": "date"}),
        mode=mode,
        kelly_mult=kelly_mult,
        max_stake_frac=max_stake_frac,
    )
    return s


def ledger_metrics(
    bets: pd.DataFrame, n_boot: int = 1000, seed: int = 0, max_stake_frac: float = 0.01
) -> dict:
    """n, CLV (net and raw) with match-clustered bootstrap CI, flat / Kelly ROI, max drawdown."""
    n = len(bets)
    if n == 0:
        return {"n": 0}
    cl = bets.index.to_numpy()
    net = bootstrap_ci(bets["clv_net"].to_numpy(), cl, n_boot=n_boot, seed=seed)
    raw = bootstrap_ci(bets["clv_raw"].to_numpy(), cl, n_boot=n_boot, seed=seed)
    flat = _sim(bets, "flat", 0.0, max_stake_frac)
    k10 = _sim(bets, "kelly", 0.1, max_stake_frac) if "p" in bets else {"yield": np.nan}
    k25 = _sim(bets, "kelly", 0.25, max_stake_frac) if "p" in bets else {"yield": np.nan}
    return {
        "n": n,
        "clv_net": net["mean"],
        "clv_net_lo": net["lo"],
        "clv_net_hi": net["hi"],
        "clv_raw": raw["mean"],
        "clv_raw_lo": raw["lo"],
        "clv_raw_hi": raw["hi"],
        "roi_flat": float(bets["pnl_flat"].mean()),
        "roi_k10": k10["yield"],
        "roi_k25": k25["yield"],
        "mdd_flat": flat["max_drawdown"],
        "mdd_k25": k25.get("max_drawdown", np.nan),
        "hit": float(bets["won"].mean()),
        "mean_price": float(bets["price"].mean()),
    }


def cell_metrics(
    bets: pd.DataFrame,
    by: Sequence[str],
    n_boot: int = 1000,
    min_n: int = 1,
    max_stake_frac: float = 0.01,
) -> pd.DataFrame:
    """``ledger_metrics`` per group (e.g. div x market x season)."""
    rows = []
    for key, g in bets.groupby(list(by), sort=True, observed=True):
        if len(g) < min_n:
            continue
        key = key if isinstance(key, tuple) else (key,)
        met = ledger_metrics(g, n_boot, 0, max_stake_frac)
        rows.append({**dict(zip(by, key, strict=True)), **met})
    return pd.DataFrame(rows)


def paired_clv_diff(
    strategy: pd.DataFrame,
    baseline: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Match-clustered bootstrap of the paired difference strategy - baseline, per bet.

    ``strategy`` and ``baseline`` are ledgers with ``clv_net`` and a ``market`` column, indexed by
    ``match_id`` (one bet per match-market). Only the (match_id, market) pairs present in BOTH
    frames are used, so the two means are estimated on the same matches and the difference is a
    paired statistic (a point comparison of two independent CIs is not a test).
    """
    if not len(strategy) or not len(baseline):
        return {"n": 0}
    s = pd.Series(
        strategy["clv_net"].to_numpy(float),
        index=pd.MultiIndex.from_arrays(
            [strategy.index.astype(str), strategy["market"].astype(str)]
        ),
    )
    b = pd.Series(
        baseline["clv_net"].to_numpy(float),
        index=pd.MultiIndex.from_arrays(
            [baseline.index.astype(str), baseline["market"].astype(str)]
        ),
    )
    common = s.index.intersection(b.index)
    if not len(common):
        return {"n": 0}
    diff = (s.loc[common] - b.loc[common]).to_numpy()
    clusters = np.asarray([k[0] for k in common])
    codes, uniq = pd.factorize(pd.Series(clusters), sort=True)
    k = len(uniq)
    sums = np.bincount(codes, weights=diff, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(len(diff)),
        "n_strategy": int(len(strategy)),
        "mean": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_gt0": float((means > 0).mean()),
    }


def random_baseline(
    cand: pd.DataFrame,
    counts: pd.DataFrame,
    n_rep: int = 200,
    seed: int = 0,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """Random selection with the same count per (div, season, market) cell as the strategy.

    ``cand`` is a long frame of every candidate (one row per match x selection) with ``div``,
    ``season``, ``market``, ``price``, ``p_close``, ``won`` (index = match_id); ``counts`` has the
    strategy's ``div``, ``season``, ``market``, ``n``. Each replicate draws ``n`` distinct matches
    in the cell and one random selection per match. Returns per-replicate pooled mean net / raw
    CLV and flat ROI after commission.
    """
    rng = np.random.default_rng(seed)
    groups = dict(list(cand.groupby(["div", "season", "market"], sort=True, observed=True)))
    cells = []
    for r in counts.itertuples(index=False):
        g = groups.get((r.div, r.season, r.market))
        if g is None or r.n == 0:
            continue
        uniq, inv = np.unique(g.index.to_numpy(), return_inverse=True)
        order = np.argsort(inv, kind="mergesort")
        cnts = np.bincount(inv, minlength=len(uniq))
        starts = np.concatenate([[0], np.cumsum(cnts)[:-1]])
        cells.append(
            (
                min(int(r.n), len(uniq)),
                starts,
                cnts,
                g["price"].to_numpy()[order],
                g["p_close"].to_numpy()[order],
                g["won"].to_numpy()[order],
            )
        )
    out = []
    for rep in range(n_rep):
        sums, tot = np.zeros(3), 0
        for n, starts, cnts, price, pc, won in cells:
            pick = rng.choice(len(starts), size=n, replace=False)
            idx = starts[pick] + (rng.random(n) * cnts[pick]).astype(int)
            pr, p_c, w = price[idx], pc[idx], won[idx]
            sums[0] += log_clv(net_odds(pr, commission), p_c).sum()
            sums[1] += log_clv(pr, p_c).sum()
            sums[2] += np.where(w, (pr - 1.0) * (1.0 - commission), -1.0).sum()
            tot += n
        m = sums / tot if tot else np.full(3, np.nan)
        out.append({"rep": rep, "n": tot, "clv_net": m[0], "clv_raw": m[1], "roi_flat": m[2]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ config output
def render_strategy_toml(cfg: dict) -> str:
    """Deterministic TOML text for ``config/strategy.toml`` (no TOML-writer dependency)."""

    def lit(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace('"', '\\"') + '"'
        if isinstance(v, float):
            return repr(round(v, 6))
        return str(v)

    lines = [f"# {c}" for c in cfg.get("comments", [])]
    for k, v in cfg["top"].items():
        lines.append(f"{k} = {lit(v)}")
    for market, sec in cfg["markets"].items():
        lines += ["", f"[markets.{market}]"]
        for k, v in sec.items():
            if k == "whitelist":
                items = ", ".join(f'["{d}", "{market}"]' for d in v)
                lines.append(f"whitelist = [{items}]")
            else:
                lines.append(f"{k} = {lit(v)}")
    shadow = cfg.get("shadow")
    if shadow:
        lines += ["", "[shadow]"]
        for k, v in shadow.items():
            lines.append(f"{k} = {lit(v)}")
    return "\n".join(lines) + "\n"
