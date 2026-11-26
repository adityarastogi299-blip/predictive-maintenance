"""Varnika Sharma - Linear Regression, Lasso, KNN, Random Forest."""
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as C

RS = C.RANDOM_STATE


def get_models(fast: bool = False) -> dict:
    m = {}
    # Baseline - nothing to tune
    m["Linear Regression"] = (
        Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]), {})

    m["Lasso"] = (
        Pipeline([("scale", StandardScaler()),
                ("model", Lasso(max_iter=20000, random_state=RS))]),
        {"model__alpha": loguniform(1e-3, 5)})

    # distance based -> scaling is essential
    m["KNN"] = (
        Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_jobs=-1))]),
        {"model__n_neighbors": randint(5, 60),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]})

    m["Random Forest"] = (
        Pipeline([("model", RandomForestRegressor(random_state=RS, n_jobs=-1))]),
        {"model__n_estimators": [50] if fast else randint(100, 400),
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_leaf": randint(1, 15),
        "model__max_features": [0.3, 0.5, "sqrt"]})
    return m
