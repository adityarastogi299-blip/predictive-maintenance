"""02 - Show the preprocessing / feature engineering (constant sensors dropped,
rolling + lag features, capped RUL target). Just press Run."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from src import config as C, data_loader, features

train = data_loader.load_train()
sensors = features.informative_sensors(train)
print("kept sensors   :", sensors)
print("dropped sensors:", [s for s in C.SENSOR_COLS if s not in sensors])

X, y, y_raw, groups = features.make_xy(train, sensors, cap=C.RUL_CAP)
print("feature matrix :", X.shape)
print(X.head())
X.to_csv(C.PROC_DIR / "fd001_train_features.csv", index=False)
print("saved ->", C.PROC_DIR / "fd001_train_features.csv")
