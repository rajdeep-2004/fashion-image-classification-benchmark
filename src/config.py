"""Project-wide constants: seed, sizes, class names, download info and paths.

Every other module imports settings from here, so nothing is hard-coded twice (docs/decisions.md D-007).
"""
from pathlib import Path

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Split and compute sizes ----------------------------------------------
VAL_SIZE = 6000            # validation images carved from the official 60k train set (D-003)
SVM_TUNE_SUBSET = 10000    # train subset used to tune the RBF SVM (D-005)
N_JOBS = 4                 # parallel workers; capped for an 8 GB laptop
LOGREG_MAX_ITER = 5000     # allow the full LR grid to converge (D-020)

# --- Labels ------------------------------------------------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 28

# --- HOG features --------------------------------------------------------------
# Chosen in phase v1/02 by the HOG sweep on the validation set (results/hog_sweep.csv, D-017): 1,728 features.
HOG_PARAMS = {"orientations": 12, "pixels_per_cell": (4, 4), "cells_per_block": (2, 2)}

# --- Raw data download (D-006, D-012) ------------------------------------------
RAW_URL_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
RAW_FILES = {  # filename -> official MD5 from the Fashion-MNIST README
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

# --- Paths (all relative to the repo root, so notebooks work from any folder) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
VAL_INDICES_PATH = SPLITS_DIR / "val_indices.npy"
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TEST_LOG_PATH = RESULTS_DIR / "test_log.csv"
HOG_SWEEP_PATH = RESULTS_DIR / "hog_sweep.csv"

FIGURE_DPI = 300

# --- Phase 2 CNN (D-034, D-036, D-038) -------------------------------------------
CNN_BATCH_SIZE = 64        # images per training step (~844 steps per epoch on 54k)
CNN_LR = 0.001             # Adam learning rate
CNN_MAX_EPOCHS = 30        # upper bound; early stopping usually ends sooner
CNN_PATIENCE = 5           # stop after this many epochs without a lower val loss
CNN_DROPOUT = 0.5          # dropout before the last layer
CNN_PREDICT_BATCH = 1000   # images per forward pass when predicting (bounds memory)
CNN_FEATURES_FROM = "cnn3"  # the final CNN (D-040); its last hidden layer feeds the hybrid RBF SVM (D-044)
MODELS_DIR = ROOT / "models"             # best checkpoints (gitignored)
TRAINING_DIR = RESULTS_DIR / "training"  # per-epoch history CSVs
