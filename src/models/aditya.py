"""Aditya Rastogi - Polynomial, Ridge, SVR, Decision Tree, Gradient Boosting."""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .. import config as C

RS = C.RANDOM_STATE


def get_models(fast: bool = False) -> dict:
    m = {}
    # Polynomial terms explode with many features -> keep the k best first
    m["Polynomial Regression"] = (
        Pipeline([("scale", StandardScaler()),
                ("select", SelectKBest(f_regression)),
                ("poly", PolynomialFeatures(include_bias=False)),
                ("model", LinearRegression())]),
        {"select__k": [10, 15, 20],
        "poly__degree": [2, 3] if not fast else [2]})

    m["Ridge"] = (
        Pipeline([("scale", StandardScaler()), ("model", Ridge(random_state=RS))]),
        {"model__alpha": loguniform(1e-2, 1e3)})

    m["SVR"] = (
        Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", cache_size=500))]),
        {"model__C": loguniform(1, 200),
        "model__epsilon": uniform(0.1, 5),
        "model__gamma": loguniform(1e-4, 1e-1)})

    m["Decision Tree"] = (
        Pipeline([("model", DecisionTreeRegressor(random_state=RS))]),
        {"model__max_depth": randint(3, 16),
        "model__min_samples_leaf": randint(5, 80),
        "model__min_samples_split": randint(2, 40)})

    m["Gradient Boosting"] = (
        Pipeline([("model", GradientBoostingRegressor(random_state=RS))]),
        {"model__n_estimators": [60] if fast else randint(100, 500),
        "model__learning_rate": loguniform(0.02, 0.3),
        "model__max_depth": randint(2, 6),
        "model__subsample": uniform(0.6, 0.4),
        "model__min_samples_leaf": randint(5, 60)})
    return m
