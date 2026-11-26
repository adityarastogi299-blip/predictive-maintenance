# Predictive Maintenance System - NASA C-MAPSS FD001 (RUL prediction)

Course: Machine Learning (5.0CS04O015) - Bake-Off Competition
Team: Rajeshwari Sarkar (leader), Varnika Sharma, Aditya Rastogi

Predict the Remaining Useful Life (cycles left) of turbofan engines from sensor history.
13 regressors are trained independently and compared on MAE, MSE, RMSE, R2 and the NASA score.

## Run in VS Code (plain Python files, no notebooks, no virtual environment)
1. `File > Open Folder...` -> select the `predictive-maintenance` folder.
2. In the VS Code terminal (Ctrl+`) run once: `pip install -r requirements.txt`
3. Put the test file at `data/raw/fd001_test.csv` (or `.xlsx`).
4. Open each file in `notebooks/` in this order and press the **Run** button (top right):
   `01_eda.py` -> `02_preprocessing.py` -> `03_modelling.py` -> `04_final_test.py`
5. Results appear in `results/` (tables, plots, final test RMSE / R2).

## Folder map (same as the flowchart)
```
predictive-maintenance/
|-- data/raw/            FD001_train_dirty.csv, fd001_test.csv
|-- data/processed/      cleaned / feature files
|-- notebooks/           01_eda.py  02_preprocessing.py  03_modelling.py  04_final_test.py
|-- src/
|   |-- data_loader.py   load CSVs, check rul, CLEAN the dirty file
|   |-- features.py      drop constants, rolling and lag features
|   |-- metrics.py       MAE, MSE, RMSE, R2, NASA score
|   |-- models/          rajeshwari.py  varnika.py  aditya.py
|   |-- train_all.py     runs every model on the same split
|   |-- evaluate_final.py scores the test file's last rows
|   |-- config.py        paths + settings      plots.py   figures
|-- results/             model_comparison.csv, plots/
|-- saved_models/        best_pipeline.joblib
`-- requirements.txt, README.md, .gitignore
```
Terminal alternative: `python -m src.data_loader`, `python -m src.train_all`, `python -m src.train_all --finalize`, `python -m src.evaluate_final`.

## Data
* `data/raw/FD001_train_dirty.csv` - the dirty training file (given)
* `data/raw/fd001_test.csv` (or `.xlsx`) - test engines, 28 columns, `rul` of each engine's last row = official label

## What the cleaning step fixes (`clean_dirty` in `src/data_loader.py`)
inconsistent engine ids, duplicate rows / engine-cycles, `?` and `units` text, `-999` sentinels,
sign-flipped sensors, spikes (per-engine Hampel filter), missing values (per-engine interpolation),
misspelt `health_status`, and re-derives `rul` = last cycle - cycle.
`health_status` is derived from RUL, so it is **never** a feature (leakage).

## Leakage rules (enforced in code)
* validation = whole engines held out (`GroupShuffleSplit`), tuning = `GroupKFold` on engine
* features are past-only per engine; scalers live inside the model pipelines
* `engine_id`, `unit`, `health_status`, `rul` are never features
* the test file is opened only by `evaluate_final.py`; selection uses validation only

## Team allocation (`src/models/`, one folder per member, one file per algorithm)
```
src/models/
|-- rajeshwari/  elastic_net.py  extra_trees.py  adaboost.py  xgboost_model.py
|-- varnika/     linear_regression.py  lasso.py  knn.py  random_forest.py
`-- aditya/      polynomial_regression.py  ridge.py  svr.py  decision_tree.py  gradient_boosting.py
```
Each file has a `build(fast)` function returning the pipeline and its tuning ranges.

## Design choices to document in the report
RUL cap = 125 (try `--cap`), rolling window 10, lag-diff 5, 20 validation engines,
model chosen by average rank over MAE/RMSE/R2/NASA (not a single metric).
Validation metrics are over all rows of held-out engines vs the capped target; final test metrics
are on the last row of each test engine vs the official label (a capped-label view is also reported).
