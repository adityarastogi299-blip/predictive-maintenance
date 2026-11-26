"""Feature engineering - past-only, per-engine, leakage free.

Every feature at cycle t uses readings from cycles <= t of the SAME engine.
Because of that, the same function can be applied to train, validation and test data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def informative_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors that vary in the TRAINING file (constant ones carry no information)."""
    return [s for s in C.SENSOR_COLS if train[s].nunique() >= C.MIN_UNIQUE_VALUES]


def cap_rul(rul, cap: int | None = C.RUL_CAP):
    """Piecewise-linear target: early-life RUL is clipped at `cap`."""
    return np.minimum(rul, cap) if cap else rul


def build_features(df: pd.DataFrame, sensors: list[str],
                    window: int = C.ROLL_WINDOW, lag: int = C.LAG_DIFF) -> pd.DataFrame:
    """Return a feature frame (same index as df) with columns:
    raw sensors, rolling mean, rolling std, lag-difference and the cycle count."""
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")[sensors]
    out = {}
    out["cycle"] = df["cycle"].astype(float)
    for s in sensors:
        out[s] = df[s]
    roll = g.rolling(window, min_periods=1)
    mean = roll.mean().reset_index(level=0, drop=True)
    std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
    diff = (df[sensors] - g.shift(lag)).fillna(0.0)
    for s in sensors:
        out[f"{s}_mean{window}"] = mean[s]
        out[f"{s}_std{window}"] = std[s]
        out[f"{s}_diff{lag}"] = diff[s]
    return pd.DataFrame(out, index=df.index)


def make_xy(df: pd.DataFrame, sensors: list[str], cap: int | None = C.RUL_CAP):
    """Features X, capped target y, raw (uncapped) rul and engine groups."""
    X = build_features(df, sensors)
    d = df.loc[X.index]
    y = pd.Series(cap_rul(d[C.TARGET].values, cap), index=X.index, name="rul_capped")
    return X, y, d[C.TARGET], d["unit"]


def last_rows(df: pd.DataFrame, X: pd.DataFrame) -> pd.Index:
    """Index of the last recorded cycle of each engine (test evaluation rows)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).index
