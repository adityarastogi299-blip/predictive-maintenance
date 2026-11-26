"""All figures used in the report / poster. Every plot reads real results only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ------------------------------- EDA ---------------------------------------
def plot_lifetimes(train):
    life = train.groupby("unit")["cycle"].max()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(life, bins=20, kde=True, ax=ax, color="#2a6f97")
    ax.set(title="Engine lifetimes (cycles to failure)", xlabel="cycles", ylabel="engines")
    return _save(fig, "eda_lifetimes.png")


def plot_sensor_trends(train, sensors, units=(1, 2, 3, 4, 5)):
    n = len(sensors)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    for ax, s in zip(np.ravel(axes), sensors):
        for u in units:
            d = train[train.unit == u]
            ax.plot(d["cycle"], d[s], lw=0.8, label=f"E{u}")
        ax.set_title(s)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=7)
    fig.suptitle("Sensor trends vs cycle (5 engines)")
    return _save(fig, "eda_sensor_trends.png")


def plot_corr_with_rul(train, sensors):
    c = train[sensors + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    c.plot.barh(ax=ax, color="#bc4749")
    ax.set(title="Correlation of each sensor with RUL")
    return _save(fig, "eda_corr_with_rul.png")


def plot_corr_matrix(train, sensors):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(train[sensors].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Sensor correlation matrix")
    return _save(fig, "eda_corr_matrix.png")


def plot_rul_target(train):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train[C.TARGET], bins=40, alpha=.6, label="raw RUL", color="#2a6f97")
    ax.hist(np.minimum(train[C.TARGET], C.RUL_CAP), bins=40, alpha=.6,
            label=f"capped at {C.RUL_CAP}", color="#f4a261")
    ax.legend()
    ax.set(title="Distribution of the RUL target", xlabel="RUL (cycles)")
    return _save(fig, "eda_rul_target.png")


def plot_cleaning(raw_counts: dict):
    items = {k: v for k, v in raw_counts.items() if isinstance(v, int) and v > 0}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pd.Series(items).sort_values().plot.barh(ax=ax, color="#6a994e")
    ax.set(title="Data-quality problems fixed in the dirty file")
    return _save(fig, "cleaning_summary.png")


# ------------------------------ comparison ---------------------------------
def plot_comparison(df: pd.DataFrame):
    paths = []
    for metric, better in [("MAE", "lower"), ("RMSE", "lower"),
                            ("R2", "higher"), ("NASA_Score", "lower")]:
        d = df.sort_values(metric, ascending=(better == "lower"))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(d["Model"], d[metric], color=sns.color_palette("viridis", len(d)))
        ax.invert_yaxis()
        ax.set(title=f"{metric} on validation engines ({better} is better)")
        if metric == "NASA_Score":
            ax.set_xscale("log")
        paths.append(_save(fig, f"compare_{metric.lower()}.png"))
    if "Training_Time_Seconds" in df:
        d = df.sort_values("Training_Time_Seconds")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(d["Model"], d["Training_Time_Seconds"], color="#8d99ae")
        ax.set(title="Training time (seconds, incl. tuning)")
        paths.append(_save(fig, "compare_training_time.png"))
    return paths


# ------------------------------ final test ---------------------------------
def plot_actual_vs_pred(y, p, title="Actual vs predicted RUL (test engines)"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, s=25, alpha=.75, color="#2a6f97")
    lim = [0, max(np.max(y), np.max(p)) + 5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="actual RUL", ylabel="predicted RUL", title=title)
    return _save(fig, "final_actual_vs_pred.png")


def plot_residuals(y, p):
    res = np.asarray(p) - np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, res, s=25, alpha=.75, color="#bc4749")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set(xlabel="actual RUL", ylabel="predicted - actual",
                title="Residuals (>0 = late warning, dangerous)")
    sns.histplot(res, bins=20, kde=True, ax=axes[1], color="#bc4749")
    axes[1].set(title="Residual distribution", xlabel="predicted - actual")
    return _save(fig, "final_residuals.png")


def plot_pred_vs_cycle(curves: dict):
    fig, axes = plt.subplots(1, len(curves), figsize=(4.5 * len(curves), 4), squeeze=False)
    for ax, (u, d) in zip(axes[0], curves.items()):
        ax.plot(d["cycle"], d["actual"], label="actual", color="k")
        ax.plot(d["cycle"], d["pred"], label="predicted", color="#e76f51")
        ax.set(title=f"Test engine {u}", xlabel="cycle", ylabel="RUL")
    axes[0][0].legend()
    return _save(fig, "final_pred_vs_cycle.png")
