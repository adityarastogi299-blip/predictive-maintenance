"""Train, tune and compare the 13 regressors on the SAME engine-wise split.

Examples
--------
python -m src.train_all                       # all 13 models
python -m src.train_all --members varnika     # only one member's models
python -m src.train_all --models "SVR" "KNN"  # specific models
python -m src.train_all --fast                # quick smoke test (tiny search)
python -m src.train_all --finalize            # pick best on validation, refit on all
                                              # training engines, save best_pipeline.joblib
Each model writes results/per_model/<name>.json, so members can run separately and
the table is merged automatically (model_comparison.csv).
"""
from __future__ import annotations
import argparse
import json
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.base import clone

from . import config as C
from . import data_loader, features, metrics, plots
from .models import get_all

warnings.filterwarnings("ignore")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def prepare(cap=C.RUL_CAP):
    train = data_loader.load_train()
    sensors = features.informative_sensors(train)
    X, y, y_raw, groups = features.make_xy(train, sensors, cap)
    return train, sensors, X, y, y_raw, groups


def engine_split(groups, n_val=C.N_VAL_ENGINES):
    """Hold out WHOLE engines (never rows) -> no leakage between train and validation."""
    gss = GroupShuffleSplit(n_splits=1, test_size=n_val, random_state=C.RANDOM_STATE)
    tr, va = next(gss.split(np.zeros(len(groups)), groups=groups))
    return tr, va


def run_models(selected: dict, fast: bool, n_iter: int, cap: int):
    train, sensors, X, y, y_raw, groups = prepare(cap)
    tr, va = engine_split(groups)
    split = {"val_engines": sorted(map(int, groups.iloc[va].unique())),
             "n_train_engines": int(groups.iloc[tr].nunique()),
             "cap": cap, "sensors": sensors, "n_features": int(X.shape[1])}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "validation_split.json").write_text(json.dumps(split, indent=2))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
    Xva, yva = X.iloc[va], y.iloc[va]
    print(f"features: {X.shape[1]} | train rows {len(tr)} ({split['n_train_engines']} engines)"
          f" | val rows {len(va)} ({len(split['val_engines'])} engines)")

    cv = GroupKFold(n_splits=2 if fast else C.CV_FOLDS)
    C.PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (member, pipe, params) in selected.items():
        print(f"\n>>> [{member}] {name}")
        t0 = time.time()
        if params:
            sizes = [len(v) if isinstance(v, list) else np.inf for v in params.values()]
            grid_size = float(np.prod(sizes))
            if np.isfinite(grid_size) and grid_size <= n_iter:
                search = GridSearchCV(pipe, params, cv=cv, scoring="neg_root_mean_squared_error",
                                      n_jobs=1, refit=True)
            else:
                search = RandomizedSearchCV(pipe, params, n_iter=n_iter, cv=cv,
                                            scoring="neg_root_mean_squared_error",
                                            random_state=C.RANDOM_STATE, n_jobs=1, refit=True)
            search.fit(Xtr, ytr, groups=gtr)
            model, best = search.best_estimator_, search.best_params_
            cv_rmse = float(-search.best_score_)
        else:
            model = clone(pipe).fit(Xtr, ytr)
            best, cv_rmse = {}, float("nan")
        secs = time.time() - t0
        pred = model.predict(Xva)
        rec = {"Model": name, "Member": member, **metrics.compute_all(yva, pred),
               "CV_RMSE": cv_rmse, "Training_Time_Seconds": round(secs, 2),
               "best_params": best}
        (C.PER_MODEL_DIR / f"{slug(name)}.json").write_text(
            json.dumps(rec, indent=2, default=_jsonable))
        print(f"    MAE {rec['MAE']:.2f} | RMSE {rec['RMSE']:.2f} | R2 {rec['R2']:.3f} | "
              f"NASA {rec['NASA_Score']:.0f} | {secs:.0f}s | {best}")


def merge_results() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(C.PER_MODEL_DIR.glob("*.json"))]
    if not rows:
        raise SystemExit("No per-model results found - train first.")
    df = pd.DataFrame(rows)
    cols = ["Model", "Member", "MAE", "MSE", "RMSE", "R2", "NASA_Score",
            "Training_Time_Seconds", "CV_RMSE"]
    df = df[cols + ["best_params"]]
    # no single-metric choice: average rank over MAE, RMSE, R2 and NASA score
    ranks = pd.concat([df["MAE"].rank(), df["RMSE"].rank(), df["NASA_Score"].rank(),
                       df["R2"].rank(ascending=False)], axis=1)
    df["Avg_Rank"] = ranks.mean(axis=1)
    df = df.sort_values(["Avg_Rank", "RMSE"]).reset_index(drop=True)
    df.drop(columns="best_params").to_csv(C.COMPARISON_CSV, index=False)
    plots.plot_comparison(df)
    print("\n=== model comparison (validation engines) ===")
    print(df.drop(columns="best_params").round(3).to_string(index=False))
    return df


def finalize(cap: int):
    df = merge_results()
    best_row = df.iloc[0]
    name = best_row["Model"]
    print(f"\nBest model on validation (avg rank over MAE/RMSE/R2/NASA): {name}")
    registry = get_all(fast=False)
    _, pipe, _ = registry[name]
    pipe = clone(pipe).set_params(**best_row["best_params"])
    train, sensors, X, y, _, _ = prepare(cap)
    pipe.fit(X, y)                       # refit on ALL 100 training engines
    C.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "sensors": sensors, "cap": cap, "model_name": name,
                 "feature_names": list(X.columns), "best_params": best_row["best_params"]},
                C.BEST_PIPELINE_PATH)
    print(f"Saved fitted pipeline -> {C.BEST_PIPELINE_PATH}")
    print("Now run:  python -m src.evaluate_final   (the test file is scored ONCE)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=["all"],
                    help="rajeshwari varnika aditya | all")
    ap.add_argument("--models", nargs="*", default=None, help="model names to run")
    ap.add_argument("--fast", action="store_true", help="tiny search for smoke tests")
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--cap", type=int, default=C.RUL_CAP, help="RUL cap (0 = no cap)")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()

    if a.finalize:
        return finalize(a.cap)
    if a.merge_only:
        return merge_results()
    n_iter = a.n_iter or (3 if a.fast else C.N_SEARCH_ITER)
    reg = get_all(fast=a.fast)
    sel = {n: v for n, v in reg.items()
           if ("all" in a.members or v[0] in a.members)
           and (a.models is None or n in a.models)}
    if not sel:
        raise SystemExit(f"Nothing selected. Available: {list(reg)}")
    run_models(sel, a.fast, n_iter, a.cap)
    merge_results()


if __name__ == "__main__":
    main()
