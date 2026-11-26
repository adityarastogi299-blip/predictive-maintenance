"""01 - Clean the dirty file, check the data, make EDA plots. Just press Run."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import json
from src import config as C, data_loader, features, plots

# 1) clean FD001_train_dirty.csv -> data/processed/fd001_train_clean.csv
clean, report = data_loader.clean_and_save()

# 2) checks to record (section 4.3)
train = data_loader.load_train()
checks = data_loader.data_checks(train)
C.RESULTS_DIR.mkdir(exist_ok=True)
(C.RESULTS_DIR / "data_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))

# 3) EDA figures -> results/plots/
sensors = features.informative_sensors(train)
plots.plot_lifetimes(train)
plots.plot_sensor_trends(train, sensors)
plots.plot_corr_with_rul(train, sensors)
plots.plot_corr_matrix(train, sensors)
plots.plot_rul_target(train)
plots.plot_cleaning({k: v for k, v in report.items() if k in (
    "sentinel_-999_cells", "exact_duplicate_rows", "rows_removed_as_duplicate_engine_cycle",
    "sign_flips_fixed", "spikes_removed_total", "health_status_missing_before")})
print("EDA figures saved to", C.PLOTS_DIR)
