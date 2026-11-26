"""Score the held-out test engines ONCE with the selected model.

* the test file is only opened here, never during model building
* features are rebuilt from each engine's cycles up to its last recorded cycle
* only the LAST row of each engine is scored against its official `rul` value
"""
import json
import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data_loader, features, metrics, plots


def main():
    bundle = joblib.load(C.BEST_PIPELINE_PATH)
    pipe, sensors, cap = bundle["pipeline"], bundle["sensors"], bundle["cap"]

    test = data_loader.load_test()
    X = features.build_features(test, sensors)[bundle["feature_names"]]
    test = test.loc[X.index]
    pred_all = pipe.predict(X)

    idx = features.last_rows(test, X)
    last = test.loc[idx]
    y_true = last[C.TARGET].values
    y_pred = pd.Series(pred_all, index=X.index).loc[idx].values.clip(min=0)

    out = metrics.compute_all(y_true, y_pred)
    out_cap = metrics.compute_all(np.minimum(y_true, cap), y_pred)
    summary = {"model": bundle["model_name"], "n_test_engines": int(len(last)),
                "label_range": [int(y_true.min()), int(y_true.max())],
                "official_labels": out,
                f"labels_capped_at_{cap}": out_cap,
                "late_predictions": int((y_pred > y_true).sum()),
                "early_predictions": int((y_pred < y_true).sum())}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "final_test_metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"unit": last["unit"].values, "last_cycle": last["cycle"].values,
                    "actual_rul": y_true, "predicted_rul": np.round(y_pred, 2),
                    "error": np.round(y_pred - y_true, 2)}).to_csv(
        C.RESULTS_DIR / "final_test_predictions.csv", index=False)

    plots.plot_actual_vs_pred(y_true, y_pred)
    plots.plot_residuals(y_true, y_pred)
    curves = {}
    for u in sorted(test["unit"].unique())[:3]:
        m = (test["unit"] == u).values
        d = test[m]
        curves[int(u)] = {"cycle": d["cycle"].values,
                            "actual": d[C.TARGET].values,
                            "pred": pd.Series(pred_all, index=X.index)[m].clip(lower=0).values}
    plots.plot_pred_vs_cycle(curves)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
