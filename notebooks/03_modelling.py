"""03 - Train and compare the 13 models (engine-wise validation split).
Set MEMBERS to ["rajeshwari"], ["varnika"] or ["aditya"] to train only one person's models.
Set FAST = True for a quick test. Then it selects the best model on validation
and saves saved_models/best_pipeline.joblib."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
MEMBERS = ["all"]
FAST = False

from src import train_all
sys.argv = [sys.argv[0], "--members", *MEMBERS] + (["--fast"] if FAST else [])
train_all.main()                      # trains + writes results/model_comparison.csv
sys.argv = [sys.argv[0], "--finalize"]
train_all.main()                      # best model -> refit on all engines -> saved_models/
