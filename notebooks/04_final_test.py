"""04 - Score the test file ONCE with the selected model (RMSE, R2, MAE, MSE, NASA score).
Just press Run. Do not change the model after seeing this score."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from src import evaluate_final
evaluate_final.main()
