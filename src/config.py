"""Central configuration: paths and design choices (documented, tunable)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PER_MODEL_DIR = RESULTS_DIR / "per_model"
MODEL_DIR = ROOT / "saved_models"

# ---- files ---------------------------------------------------------------
DIRTY_TRAIN_PATH = RAW_DIR / "FD001_train_dirty.csv"
CLEAN_TRAIN_PATH = PROC_DIR / "fd001_train_clean.csv"
CLEANING_REPORT_PATH = PROC_DIR / "cleaning_report.json"
# the loader also accepts .xlsx with the same stem
TEST_PATH = RAW_DIR / "fd001_test.csv"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
BEST_PIPELINE_PATH = MODEL_DIR / "best_pipeline.joblib"

# ---- columns -------------------------------------------------------------
ID_COLS = ["engine_id", "unit"]          # identifiers, never features
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
TARGET = "rul"
GROUP_COL = "unit"                       # engine-wise splitting

# ---- design choices ------------------------------------------------------
RANDOM_STATE = 42
RUL_CAP = 125                 # piecewise-linear target cap (tune on validation)
MIN_UNIQUE_VALUES = 3         # sensors with fewer unique values are "constant"
ROLL_WINDOW = 10              # past-only rolling window (cycles)
LAG_DIFF = 5                  # difference against the reading 5 cycles ago
N_VAL_ENGINES = 20            # engines held out for validation
CV_FOLDS = 3                  # GroupKFold folds used inside hyper-parameter search
N_SEARCH_ITER = 10            # RandomizedSearchCV iterations (see --fast)

# ---- cleaning ------------------------------------------------------------
SENTINELS = [-999, -999.0]    # placeholder for "missing"
MISSING_TOKENS = ["?", "", "nan", "NaN", "NA", "N/A", "null", "None", "-"]
HAMPEL_WINDOW = 9             # centred rolling-median window (per engine)
HAMPEL_K = 6.0                # |residual| > K * robust sigma  -> spike
