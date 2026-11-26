"""Rajeshwari Sarkar - Elastic Net, Extra Trees, AdaBoost, XGBoost."""
import warnings
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .. import config as C

RS = C.RANDOM_STATE


def get_models(fast: bool = False) -> dict:
    m = {}
    m["Elastic Net"] = (
        Pipeline([("scale", StandardScaler()),
                ("model", ElasticNet(max_iter=20000, random_state=RS))]),
        {"model__alpha": loguniform(1e-3, 10),
        "model__l1_ratio": uniform(0.05, 0.9)})

    m["Extra Trees"] = (
        Pipeline([("model", ExtraTreesRegressor(random_state=RS, n_jobs=-1))]),
        {"model__n_estimators": [50] if fast else randint(100, 400),
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": randint(1, 15),
        "model__max_features": [0.3, 0.5, 1.0]})

    m["AdaBoost"] = (
        Pipeline([("model", AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=5, random_state=RS),
            random_state=RS))]),
        {"model__n_estimators": [30] if fast else randint(50, 300),
        "model__learning_rate": loguniform(0.01, 1.0),
        "model__loss": ["linear", "square"],
        "model__estimator__max_depth": [3, 5, 8]})

    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = (
            Pipeline([("model", XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                            random_state=RS, n_jobs=-1))]),
            {"model__n_estimators": [100] if fast else randint(200, 800),
            "model__max_depth": randint(3, 9),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_alpha": loguniform(1e-3, 10),
            "model__reg_lambda": loguniform(0.1, 20)})
    except ImportError:
        warnings.warn("xgboost is not installed - XGBoost skipped (pip install xgboost)")
    return m
