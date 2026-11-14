"""Feature extraction: raw pixels, HOG and (Phase 2) CNN features, cached on disk in features/.

Pixels and HOG are fixed formulas applied to each image on its own. CNN features are the last hidden layer of the
final CNN, whose weights were learned on the 54k train split only (D-044). Nothing is *fitted* here, so computing any
of them on val/test cannot leak information. Scaling happens later, inside each model's Pipeline (D-004).
See docs/data-formats.md §5 and docs/module-interfaces.md.
"""
import itertools
import time

import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.cnn import extract_cnn_features, load_trained
from src.utils import display_path
from src.data import load_splits

KINDS = ("pixels", "hog", "cnn")
SPLITS = ("train", "val", "trainval", "test")  # trainval = train rows followed by val rows (60k)


# --- Extractors ------------------------------------------------------------------

def _check_images(X: np.ndarray) -> None:
    if X.ndim != 3 or X.shape[1:] != (config.IMAGE_SIZE, config.IMAGE_SIZE):
        raise ValueError(f"Expected images of shape (N, 28, 28), got {X.shape}")


def _square(value) -> tuple:
    """Accept 4 or (4, 4) and return (4, 4). Only square cells/blocks are used in this project."""
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or pair[0] != pair[1]:
        raise ValueError(f"Only square cells/blocks are supported, got {value}")
    return pair


def extract_pixels(X: np.ndarray) -> np.ndarray:
    """(N, 28, 28) -> (N, 784) float32. Just flattening; values stay in [0, 1]."""
    _check_images(X)
    return X.reshape(len(X), -1).astype(np.float32)


def extract_hog(X: np.ndarray, orientations: int, pixels_per_cell, cells_per_block) -> np.ndarray:
    """(N, 28, 28) -> (N, d) float32 HOG descriptors (skimage, L2-Hys block normalisation)."""
    _check_images(X)
    ppc, cpb = _square(pixels_per_cell), _square(cells_per_block)
    features = [
        hog(img, orientations=orientations, pixels_per_cell=ppc, cells_per_block=cpb,
            block_norm="L2-Hys", feature_vector=True)
        for img in X
    ]
    return np.asarray(features, dtype=np.float32)


# --- Cached access ---------------------------------------------------------------

def _validate(kind: str, split: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"Unknown feature kind {kind!r}; use one of {KINDS}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; use one of {SPLITS}")


def hog_tag(params: dict) -> str:
    """Short name for a HOG config, e.g. 'o9_p4_c2'. Used in cache filenames so a param change means a new file (D-007)."""
    ppc, cpb = _square(params["pixels_per_cell"]), _square(params["cells_per_block"])
    return f"o{params['orientations']}_p{ppc[0]}_c{cpb[0]}"


def _cache_path(kind: str, split: str):
    prefixes = {"pixels": "pixels", "hog": f"hog_{hog_tag(config.HOG_PARAMS)}", "cnn": config.CNN_FEATURES_FROM}
    prefix = prefixes[kind]
    return config.FEATURES_DIR / f"{prefix}_{split}.npy"


def _split_arrays(split: str):
    """Images and labels of one split. 'trainval' = train followed by val."""
    s = load_splits(verbose=False)
    if split == "trainval":
        return np.concatenate([s["X_train"], s["X_val"]]), np.concatenate([s["y_train"], s["y_val"]])
    return s[f"X_{split}"], s[f"y_{split}"]


def _extract(kind: str, X: np.ndarray) -> np.ndarray:
    if kind == "pixels":
        return extract_pixels(X)
    if kind == "hog":
        return extract_hog(X, **config.HOG_PARAMS)
    return extract_cnn_features(load_trained(config.CNN_FEATURES_FROM), X)


def get_features(kind: str, split: str) -> np.ndarray:
    """Feature matrix for one split, loaded from features/ or built and cached on first use."""
    _validate(kind, split)
    path = _cache_path(kind, split)
    if path.exists():
        return np.load(path)

    if split == "trainval":  # reuse the train and val caches, so the row order always matches get_labels
        features = np.concatenate([get_features(kind, "train"), get_features(kind, "val")])
    else:
        X, _ = _split_arrays(split)
        features = _extract(kind, X)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)
    print(f"cached {display_path(path)} {features.shape}")
    return features


def get_labels(split: str) -> np.ndarray:
    """Labels for one split, in the same row order as get_features(kind, split)."""
    _validate("pixels", split)
    return _split_arrays(split)[1]


# --- HOG parameter sweep (F-06, D-016) ---------------------------------------------

HOG_SWEEP_GRID = {
    "orientations": [6, 9, 12],
    "pixels_per_cell": [4, 7],
    "cells_per_block": [1, 2],
}


def _proxy_model():
    """Fast linear SVM used only to *rank* HOG configs (D-016).

    SGDClassifier with hinge loss = a linear SVM trained by stochastic gradient descent: seconds instead of
    the many minutes the exact LinearSVC solver needs on 54k x ~1,700 features.
    """
    # Fixed budget of 30 passes for every config (tol=None), so all configs are compared on equal terms.
    sgd = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=30, tol=None,
                        random_state=config.SEED, n_jobs=config.N_JOBS)
    return Pipeline([("scale", StandardScaler()), ("clf", sgd)])


def _score_hog_config(params: dict, X_train, y_train, X_val, y_val) -> dict:
    """Extract HOG with `params`, fit the proxy on train, score on val. Never touches test."""
    start = time.perf_counter()
    F_train = extract_hog(X_train, **params)
    extract_s = time.perf_counter() - start
    F_val = extract_hog(X_val, **params)
    model = _proxy_model().fit(F_train, y_train)
    return {
        "orientations": params["orientations"],
        "pixels_per_cell": params["pixels_per_cell"],
        "cells_per_block": params["cells_per_block"],
        "dim": F_train.shape[1],
        "extract_s_per_1000": round(extract_s / len(X_train) * 1000, 3),
        "val_acc": round(accuracy_score(y_val, model.predict(F_val)), 4),
    }


def run_hog_sweep(grid: dict = None) -> pd.DataFrame:
    """Score every HOG config in `grid` on the validation set and save the table to results/hog_sweep.csv."""
    grid = grid or HOG_SWEEP_GRID
    s = load_splits(verbose=False)
    rows = []
    for o, p, c in itertools.product(grid["orientations"], grid["pixels_per_cell"], grid["cells_per_block"]):
        params = {"orientations": o, "pixels_per_cell": p, "cells_per_block": c}
        row = _score_hog_config(params, s["X_train"], s["y_train"], s["X_val"], s["y_val"])
        print(f"{hog_tag(params):10s} dim={row['dim']:5d}  val_acc={row['val_acc']:.4f}")
        rows.append(row)
    table = pd.DataFrame(rows)
    config.HOG_SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.HOG_SWEEP_PATH, index=False)
    print(f"saved {display_path(config.HOG_SWEEP_PATH)}")
    return table
