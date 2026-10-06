"""LightGBM / CatBoost 1X2 (multiclass) and over/under 2.5 (binary), walk-forward by season.

Protocol for test season ``S`` (``S1`` = the season before it; everything chronological):

1. **Model A** is trained on rows of seasons before ``S1`` (result available: kickoff + embargo
   before ``S1``'s first kickoff). The first half of ``S1`` (by kickoff) is its early-stopping
   set -> ``best_iter``.
2. **Calibrator**: the second half of ``S1`` is the calibration set. Model A's probabilities on
   it are used by :func:`fedge.models.calibrate.select_calibrator`, which picks (in-fold, by
   log-loss) among identity / isotonic / Beta / Venn-Abers.
3. **Final model** is refit on every row before ``S`` with ``round(best_iter * 1.1)`` trees
   (CatBoost: same), predicts ``S`` and the chosen calibrator is applied.

Decorrelated variant (``init_cols`` given): the model learns the *residual* of the outcome over
a market prior. Boosting starts from ``init_score = log p_market`` (multiclass) or
``logit p_market`` (binary) and only adds trees that improve the log-loss beyond the market
price, so the output is ``softmax(log p_market + f(x))``. Early stopping keeps ``f`` small (zero
useful trees = the market). Rows without a market price are dropped from every fit in that
variant.

Seeds, thread counts and iteration caps are fixed, so a re-run is byte-identical on the same
machine and library versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fedge.backtest.walkforward import EMBARGO
from fedge.models import calibrate

SEED = 0
MAX_ROUNDS = 1500
ES_ROUNDS = 50
TASKS = ("1x2", "ou25")


def lgb_params(task: str, threads: int) -> dict:
    p = {
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "max_bin": 127,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": threads,
        "verbose": -1,
    }
    if task == "1x2":
        p.update(objective="multiclass", num_class=3, metric="multi_logloss")
    else:
        p.update(objective="binary", metric="binary_logloss")
    return p


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_scores(prob: np.ndarray, task: str) -> np.ndarray:
    """Market probabilities -> boosting init scores (log for multiclass, logit for binary)."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    if task == "1x2":
        return np.log(p)
    return np.log(p[:, 0] / p[:, 1])  # columns are [over, under]: logit of P(over)


def _to_probs(raw: np.ndarray, init: np.ndarray | None, task: str) -> np.ndarray:
    """Raw boosting scores (+ init) -> probability matrix in dataset column order."""
    z = raw if init is None else raw + init
    if task == "1x2":
        return _softmax(z)
    over = 1.0 / (1.0 + np.exp(-z))
    return np.column_stack([over, 1.0 - over])


def _target(y: np.ndarray, task: str) -> np.ndarray:
    """Dataset labels (1x2: 0/1/2; ou25: 0=over, 1=under) -> the boosting target."""
    return y if task == "1x2" else (1 - y)


def fit_model(
    learner: str,
    task: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
    init_tr: np.ndarray | None = None,
    init_va: np.ndarray | None = None,
    rounds: int | None = None,
    threads: int = 4,
):
    """Fit one booster. With a validation set it early-stops and returns ``best_iter``."""
    t_tr = _target(y_tr, task)
    if learner == "lgb":
        import lightgbm as lgb

        cat = [c for c in X_tr.columns if c == "div_code"]
        dtr = lgb.Dataset(
            X_tr, t_tr, init_score=init_tr, categorical_feature=cat, free_raw_data=False
        )
        params = lgb_params(task, threads)
        if X_va is not None and rounds is None:
            dva = lgb.Dataset(
                X_va,
                _target(y_va, task),
                init_score=init_va,
                categorical_feature=cat,
                reference=dtr,
            )
            booster = lgb.train(
                params,
                dtr,
                MAX_ROUNDS,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)],
            )
            return booster, max(int(booster.best_iteration), 1)
        booster = lgb.train(params, dtr, int(rounds))
        return booster, int(rounds)
    if learner == "cb":
        from catboost import CatBoostClassifier, Pool

        loss = "MultiClass" if task == "1x2" else "Logloss"
        kw = dict(
            loss_function=loss,
            learning_rate=0.08,
            depth=5,
            l2_leaf_reg=10.0,
            random_seed=SEED,
            thread_count=threads,
            verbose=0,
            allow_writing_files=False,
        )
        ptr = Pool(X_tr, t_tr, baseline=init_tr)
        if X_va is not None and rounds is None:
            pva = Pool(X_va, _target(y_va, task), baseline=init_va)
            model = CatBoostClassifier(iterations=800, early_stopping_rounds=ES_ROUNDS, **kw)
            model.fit(ptr, eval_set=pva, use_best_model=True)
            return model, max(int(model.get_best_iteration()) + 1, 1)
        model = CatBoostClassifier(iterations=int(rounds), **kw)
        model.fit(ptr)
        return model, int(rounds)
    raise ValueError(f"unknown learner {learner!r}")


def predict_model(model, learner: str, task: str, X: pd.DataFrame, init=None) -> np.ndarray:
    """Probability matrix for ``X`` (shape (n, 3) for 1x2, (n, 2) [over, under] for ou25)."""
    if learner == "lgb":
        raw = model.predict(X, raw_score=True)
    else:
        raw = model.predict(X, prediction_type="RawFormulaVal")
    return _to_probs(np.asarray(raw, dtype=float), init, task)


def fit_live(
    table: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    task: str,
    learner: str,
    asof,
    init_cols: list[str] | None = None,
    threads: int = 4,
    embargo: pd.Timedelta = EMBARGO,
    log=None,
) -> tuple[object, object, dict]:
    """The "next fold" of :func:`walk_forward`, for fixtures that have not been played yet.

    ``asof`` is the bet/fixture time (tz-aware UTC). The protocol is exactly the walk-forward one
    with the test season replaced by everything at/after ``asof``:

    1. ``S1`` = the last season present in ``table``. Model A trains on rows before ``S1``,
       early-stopping on the chronological first half of ``S1``.
    2. The calibrator is chosen on model A's probabilities for the second half of ``S1``.
    3. The final model is refit on every row with ``kickoff < asof - embargo`` with
       ``round(best_iter * 1.1)`` trees, and the chosen calibrator is returned with it.

    Returns ``(model, calibrator, info)``; ``info`` holds ``best_iter``, ``rounds``, ``calibrator``,
    ``n_train``, ``n_cal`` and the calibrator scores. Nothing here reads a row at/after the embargo
    cut, so the fixtures' own results can never enter their features or their fit.
    """
    d = table.sort_values("kickoff_utc", kind="mergesort")
    if init_cols is not None:
        d = d.loc[d[init_cols].notna().all(axis=1)]
    seasons = sorted(d["season"].astype(str).unique(), key=_season_key)
    if len(seasons) < 2:
        raise ValueError(f"need at least two seasons to fit live; got {seasons}")
    ko = d["kickoff_utc"]
    sea = d["season"].astype(str)
    y_all = d[y_col].to_numpy(dtype=int)
    X_all = d[feature_cols]
    init_all = to_scores(d[init_cols].to_numpy(dtype=float), task) if init_cols else None
    s1 = seasons[-1]
    s1_rows = np.flatnonzero((sea == s1).to_numpy())
    if len(s1_rows) < 2:
        raise ValueError(f"last season {s1} has {len(s1_rows)} rows: cannot calibrate on it")
    half = len(s1_rows) // 2
    es_rows, cal_rows = s1_rows[:half], s1_rows[half:]
    s1_start = ko.iloc[s1_rows].min()
    asof_ts = pd.Timestamp(asof)
    asof_ts = asof_ts.tz_localize("UTC") if asof_ts.tzinfo is None else asof_ts
    tr_a = np.flatnonzero((ko < s1_start - embargo).to_numpy())
    tr_f = np.flatnonzero((ko < asof_ts - embargo).to_numpy())

    def sl(idx):
        return (X_all.iloc[idx], y_all[idx], None if init_all is None else init_all[idx])

    Xa, ya, ia = sl(tr_a)
    Xe, ye, ie = sl(es_rows)
    model_a, best = fit_model(learner, task, Xa, ya, Xe, ye, ia, ie, threads=threads)
    Xc, yc, ic = sl(cal_rows)
    p_cal = predict_model(model_a, learner, task, Xc, ic)
    name, cal, scores = calibrate.select_calibrator(p_cal, yc)
    Xf, yf, if_ = sl(tr_f)
    rounds = max(int(round(best * 1.1)), 1)
    model_f, _ = fit_model(learner, task, Xf, yf, None, None, if_, None, rounds=rounds,
                           threads=threads)
    info = {
        "best_iter": int(best),
        "rounds": int(rounds),
        "calibrator": name,
        "n_train": int(len(tr_f)),
        "n_cal": int(len(cal_rows)),
        "asof": str(asof_ts),
        **{f"ll_{k}": v for k, v in scores.items()},
    }
    if log:
        log(f"  live {learner}/{task}: iter={best} cal={name} n_train={len(tr_f)}")
    return model_f, cal, info


def _season_key(s: str) -> int:
    return int(str(s)[:4])


def walk_forward(
    table: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    task: str,
    learner: str,
    test_seasons: list[str],
    init_cols: list[str] | None = None,
    threads: int = 4,
    embargo: pd.Timedelta = EMBARGO,
    log=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season walk-forward predictions for ``test_seasons``.

    ``table``: indexed by match id with ``kickoff_utc``, ``season``, ``y_col``, the feature
    columns and (optionally) the market columns ``init_cols`` for the decorrelated variant.
    Returns ``(preds, info)``: ``preds`` has ``raw_*`` and ``cal_*`` probability columns and the
    chosen calibrator per row; ``info`` one row per season (best_iter, calibrator, scores).
    """
    k_cols = ["h", "d", "a"] if task == "1x2" else ["over", "under"]
    d = table.sort_values("kickoff_utc", kind="mergesort")
    if init_cols is not None:
        d = d.loc[d[init_cols].notna().all(axis=1)]
    seasons = sorted(d["season"].astype(str).unique(), key=_season_key)
    ko = d["kickoff_utc"]
    sea = d["season"].astype(str)
    y_all = d[y_col].to_numpy(dtype=int)
    X_all = d[feature_cols]
    init_all = to_scores(d[init_cols].to_numpy(dtype=float), task) if init_cols else None
    parts, infos = [], []
    for s in test_seasons:
        i = seasons.index(s) if s in seasons else None
        if i is None or i < 2:
            continue
        s1 = seasons[i - 1]
        s1_rows = np.flatnonzero((sea == s1).to_numpy())
        s_rows = np.flatnonzero((sea == s).to_numpy())
        s1_start = ko.iloc[s1_rows].min()
        s_start = ko.iloc[s_rows].min()
        tr_a = np.flatnonzero((ko < s1_start - embargo).to_numpy())
        half = len(s1_rows) // 2
        es_rows, cal_rows = s1_rows[:half], s1_rows[half:]  # chronological halves of S1
        tr_f = np.flatnonzero((ko < s_start - embargo).to_numpy())

        def sl(idx):
            return (X_all.iloc[idx], y_all[idx], None if init_all is None else init_all[idx])

        Xa, ya, ia = sl(tr_a)
        Xe, ye, ie = sl(es_rows)
        model_a, best = fit_model(learner, task, Xa, ya, Xe, ye, ia, ie, threads=threads)
        Xc, yc, ic = sl(cal_rows)
        p_cal = predict_model(model_a, learner, task, Xc, ic)
        name, cal, scores = calibrate.select_calibrator(p_cal, yc)
        Xf, yf, if_ = sl(tr_f)
        rounds = max(int(round(best * 1.1)), 1)
        model_f, _ = fit_model(
            learner, task, Xf, yf, None, None, if_, None, rounds=rounds, threads=threads
        )
        Xs, _, is_ = sl(s_rows)
        raw = predict_model(model_f, learner, task, Xs, is_)
        out = pd.DataFrame(index=d.index[s_rows])
        for j, c in enumerate(k_cols):
            out[f"raw_{c}"] = raw[:, j]
        calp = cal(raw)
        for j, c in enumerate(k_cols):
            out[f"cal_{c}"] = calp[:, j]
        out["season"] = s
        out["calibrator"] = name
        parts.append(out)
        infos.append(
            {
                "season": s,
                "best_iter": best,
                "rounds": rounds,
                "calibrator": name,
                "n_train": len(tr_f),
                "n_cal": len(cal_rows),
                **{f"ll_{k}": v for k, v in scores.items()},
            }
        )
        if log:
            log(f"  {learner}/{task} {s}: iter={best} cal={name} n_train={len(tr_f)}")
    if not parts:
        return pd.DataFrame(), pd.DataFrame(infos)
    return pd.concat(parts), pd.DataFrame(infos)
