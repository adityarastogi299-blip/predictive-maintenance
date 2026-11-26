"""Load the C-MAPSS FD001 files, check rul, clean the dirty training file.

CLEANING of FD001_train_dirty.csv (function clean_dirty):
1. Inconsistent engine_id ("fd001_train_001", "FD001-train-001", trailing spaces)
    -> engine_id is rebuilt from the reliable integer `unit` column.
2. Exact duplicate rows and duplicate (unit, cycle) rows with different values
    -> keep the first record of every engine-cycle.
3. Text noise: "?" tokens, "642.12units" suffixes, -999 sentinels, empty cells
    -> converted to numbers / NaN.
4. Sign flips (negative sensor values; all true sensors are positive)
    -> absolute value (settings may legitimately be negative, so they are skipped).
5. Spikes / outliers injected into sensors
    -> per-engine Hampel filter (rolling median + robust sigma); flagged points become NaN.
    Real degradation is slow and survives the filter - noise is NOT blindly removed.
6. Missing values (original + those created above)
    -> linear interpolation along `cycle` inside each engine (never across engines),
        then edge fill, then column median as last resort.
7. Misspelt / mixed-case health_status ("Healthl", "DEGRADING", NaN)
    -> normalised; missing values restored from RUL thresholds learnt from labelled rows.
    health_status is derived from RUL, so it is kept for EDA only and never used as a feature.
8. rul column re-derived as (last cycle - cycle) and compared with the supplied one.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C


def _read_any(path: Path) -> pd.DataFrame:
    """Read .csv or .xlsx (looks for the other extension if the file is absent)."""
    path = Path(path)
    if not path.exists():
        for ext in (".csv", ".xlsx", ".xls"):
            alt = path.with_suffix(ext)
            if alt.exists():
                path = alt
                break
        else:
            raise FileNotFoundError(f"Could not find {path} (.csv / .xlsx)")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["unit", "cycle"]:
        df[c] = df[c].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_train(path=C.CLEAN_TRAIN_PATH) -> pd.DataFrame:
    return _tidy(_read_any(path))


def load_test(path=C.TEST_PATH) -> pd.DataFrame:
    return _tidy(_read_any(path))


def test_labels(test: pd.DataFrame) -> pd.DataFrame:
    """Official label = `rul` of each engine's LAST recorded cycle."""
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    return last[["unit", "cycle", C.TARGET]].reset_index(drop=True)


def check_rul(train: pd.DataFrame) -> bool:
    """rul must equal (last cycle - current cycle) for each training engine."""
    expected = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    return bool((expected == train[C.TARGET]).all())


def data_checks(train: pd.DataFrame, test: pd.DataFrame | None = None) -> dict:
    """Section 4.3 'checks to record'."""
    life = train.groupby("unit")["cycle"].max()
    sensors = [c for c in C.SENSOR_COLS if c in train]
    std = train[sensors].std()
    out = {
        "train_rows": int(len(train)),
        "train_engines": int(train["unit"].nunique()),
        "lifetime_min": int(life.min()),
        "lifetime_mean": float(round(life.mean(), 2)),
        "lifetime_max": int(life.max()),
        "missing_values_total": int(train.isna().sum().sum()),
        "constant_sensors": [s for s in sensors if train[s].nunique() < C.MIN_UNIQUE_VALUES],
        "zero_std_sensors": std[std == 0].index.tolist(),
        "rul_equals_last_minus_cycle": check_rul(train),
        "identifier_columns": C.ID_COLS,
    }
    if test is not None:
        lab = test_labels(test)
        out.update(
            test_rows=int(len(test)),
            test_engines=int(test["unit"].nunique()),
            test_label_count=int(len(lab)),
            test_label_min=int(lab[C.TARGET].min()),
            test_label_max=int(lab[C.TARGET].max()),
        )
    return out


# =====================  CLEANING OF THE DIRTY FILE  =====================
SETTINGS_AND_SENSORS = ["setting_1", "setting_2", "setting_3"] + C.SENSOR_COLS
SENSOR_ONLY = C.SENSOR_COLS
STATUS_MAP = {"healthy": "Healthy", "healthl": "Healthy",
              "degrading": "Degrading", "degradinl": "Degrading",
              "critical": "Critical", "criticai": "Critical"}


def _to_number(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(r"(?i)units?$", "", regex=True).str.strip()
    s = s.mask(s.str.lower().isin([t.lower() for t in C.MISSING_TOKENS]))
    return pd.to_numeric(s, errors="coerce").astype(float)


def _hampel_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """True where a value is a spike w.r.t. its engine's rolling median."""
    med = df.groupby("unit")[col].transform(
        lambda x: x.rolling(C.HAMPEL_WINDOW, center=True, min_periods=3).median())
    resid = (df[col] - med)
    mad = (resid - resid.median()).abs().median() * 1.4826   # global robust sigma
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=df.index)
    return resid.abs() > C.HAMPEL_K * mad


def _fill_status(df: pd.DataFrame) -> pd.Series:
    lab = df["health_status"]
    known = df[lab.notna()]
    if known.empty:
        return lab
    crit_max = known.loc[known["health_status"] == "Critical", "rul"].max()
    degr_max = known.loc[known["health_status"] == "Degrading", "rul"].max()
    guess = np.where(df["rul"] <= crit_max, "Critical",
                    np.where(df["rul"] <= degr_max, "Degrading", "Healthy"))
    return lab.fillna(pd.Series(guess, index=df.index))


def clean_dirty(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"rows_in": int(len(raw))}
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # --- numeric coercion --------------------------------------------------
    for c in ["unit", "cycle", "rul"]:
        df[c] = _to_number(df[c])
    df = df.dropna(subset=["unit", "cycle"]).copy()
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    tokens_before = {}
    for c in SETTINGS_AND_SENSORS:
        orig = df[c].astype("string").str.strip()
        num = _to_number(df[c])
        tokens_before[c] = int((num.isna() & orig.notna()).sum() - 0)
        df[c] = num
    sentinel_cells = int(df[SETTINGS_AND_SENSORS].isin(C.SENTINELS).sum().sum())
    df[SETTINGS_AND_SENSORS] = df[SETTINGS_AND_SENSORS].replace(C.SENTINELS, np.nan)
    rep["sentinel_-999_cells"] = sentinel_cells

    # --- engine id ---------------------------------------------------------
    rep["engine_id_variants_before"] = int(raw["engine_id"].nunique())
    df["engine_id"] = df["unit"].map(lambda u: f"FD001_train_{u:03d}")

    # --- duplicates --------------------------------------------------------
    rep["exact_duplicate_rows"] = int(raw.duplicated().sum())
    n0 = len(df)
    df = df.sort_values(["unit", "cycle"], kind="stable")
    df = df.drop_duplicates(subset=["unit", "cycle"], keep="first")
    rep["rows_removed_as_duplicate_engine_cycle"] = int(n0 - len(df))

    # --- sign flips --------------------------------------------------------
    neg = (df[SENSOR_ONLY] < 0)
    rep["sign_flips_fixed"] = int(neg.sum().sum())
    df[SENSOR_ONLY] = df[SENSOR_ONLY].abs()

    # --- spikes (Hampel) ---------------------------------------------------
    df = df.reset_index(drop=True)
    spikes = {}
    for c in SENSOR_ONLY:   # operating settings are pure noise in FD001, left untouched
        if df[c].nunique(dropna=True) < C.MIN_UNIQUE_VALUES:
            continue
        m = _hampel_mask(df, c)
        spikes[c] = int(m.sum())
        df.loc[m, c] = np.nan
    rep["spikes_removed_per_column"] = spikes
    rep["spikes_removed_total"] = int(sum(spikes.values()))

    # --- imputation --------------------------------------------------------
    rep["missing_cells_before_imputation"] = int(df[SETTINGS_AND_SENSORS].isna().sum().sum())
    df[SETTINGS_AND_SENSORS] = (
        df.groupby("unit")[SETTINGS_AND_SENSORS]
            .transform(lambda x: x.interpolate(method="linear", limit_direction="both")))
    df[SETTINGS_AND_SENSORS] = df[SETTINGS_AND_SENSORS].fillna(df[SETTINGS_AND_SENSORS].median())
    rep["missing_cells_after_imputation"] = int(df[SETTINGS_AND_SENSORS].isna().sum().sum())

    # --- RUL ---------------------------------------------------------------
    derived = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    rep["rul_mismatch_with_supplied"] = int((derived != df["rul"]).sum())
    df["rul"] = derived.astype(int)

    # --- health status -----------------------------------------------------
    st = df["health_status"].astype("string").str.strip().str.lower().map(STATUS_MAP)
    df["health_status"] = st
    rep["health_status_missing_before"] = int(st.isna().sum())
    df["health_status"] = _fill_status(df)

    cols = (["engine_id", "unit", "cycle"] + C.SETTING_COLS + C.SENSOR_COLS
            + ["rul", "health_status"])
    df = df[cols].sort_values(["unit", "cycle"]).reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    rep["engines_out"] = int(df["unit"].nunique())
    return df, rep


def clean_and_save(in_path=C.DIRTY_TRAIN_PATH, out_path=C.CLEAN_TRAIN_PATH,
        report_path=C.CLEANING_REPORT_PATH):
    raw = pd.read_csv(in_path, dtype=str)
    clean, rep = clean_dirty(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    with open(report_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nSaved clean file -> {out_path}")
    return clean, rep


if __name__ == "__main__":
    clean_and_save()
