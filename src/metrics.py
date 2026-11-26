"""MAE, MSE, RMSE, R^2 and the asymmetric NASA C-MAPSS score."""
import numpy as np


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def mse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((y - p) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nasa_score(y, p):
    """d = predicted - actual. Late predictions (d >= 0) are penalised more."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def compute_all(y, p) -> dict:
    return {"MAE": mae(y, p), "MSE": mse(y, p), "RMSE": rmse(y, p),
            "R2": r2(y, p), "NASA_Score": nasa_score(y, p)}
