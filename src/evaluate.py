"""The shared evaluation harness: every model's test numbers come from evaluate() (F-07).

Same metrics, same timing method and same size measure for every model, so the results table is a fair comparison.
See docs/data-formats.md §8 and docs/decisions.md D-008 (size/timing), D-011 (stored predictions).
"""
import pickle
import platform
import subprocess
import time
from datetime import datetime

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src import config
from src.utils import save_json

TIMING_REPEATS = 3  # predict() is timed this many times; the fastest run is kept (least background noise)


# --- Model size (D-008) ------------------------------------------------------------

def model_size_bytes(model) -> int:
    """Size of the fitted model when serialized with pickle: one comparable number for every model type."""
    return len(pickle.dumps(model))


def model_size_detail(model) -> dict:
    """The model-specific size measure, for the write-up (support vectors, tree nodes, stored samples, coefficients,
    or trainable parameters for a CNN wrapped in CNNPredictor, D-037)."""
    if isinstance(getattr(model, "model", None), torch.nn.Module):
        return {"type": "parameters", "value": int(sum(p.numel() for p in model.model.parameters() if p.requires_grad))}
    est = model.steps[-1][1] if hasattr(model, "steps") else model  # last step of a Pipeline
    if hasattr(est, "n_support_"):
        return {"type": "support_vectors", "value": int(est.n_support_.sum())}
    if hasattr(est, "estimators_") and hasattr(est.estimators_[0], "tree_"):
        return {"type": "tree_nodes", "value": int(sum(t.tree_.node_count for t in est.estimators_))}
    if hasattr(est, "n_samples_fit_"):
        return {"type": "stored_samples", "value": int(est.n_samples_fit_)}
    if hasattr(est, "coef_"):
        return {"type": "coefficients", "value": int(est.coef_.size + np.size(est.intercept_))}
    return {"type": "none", "value": 0}


# --- Environment info (so timings can be interpreted later) --------------------------

def _cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
            return out.stdout.strip()
        with open("/proc/cpuinfo") as f:  # Linux / Colab
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass  # CPU name is informational only; fall back to the architecture below
    return platform.machine()


def environment() -> dict:
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "torch": torch.__version__,
            "cpu": _cpu_name(), "n_jobs": config.N_JOBS}


# --- Evaluation -------------------------------------------------------------------

def _timed_predict(model, X: np.ndarray):
    """Predict TIMING_REPEATS times; return the predictions and the best time in ms per 1,000 images."""
    best_s, y_pred = float("inf"), None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        y_pred = model.predict(X)
        best_s = min(best_s, time.perf_counter() - start)
    return np.asarray(y_pred), best_s / len(X) * 1000 * 1000


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str, kind: str,
             best_params: dict, train_time_s: float) -> dict:
    """Score a fitted model on the test set, write results/metrics/{name}_{kind}_test.json and return the metrics.

    Called only by experiment.final_test_run() (or a Phase 2 equivalent), never during tuning.
    """
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test has {len(X_test)} rows but y_test has {len(y_test)}")
    y_pred, ms_per_1000 = _timed_predict(model, X_test)
    labels = list(range(config.NUM_CLASSES))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)

    metrics = {
        "experiment": f"{name}_{kind}",
        "model": name,
        "features": kind,
        "split": "test",
        "best_params": best_params,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "per_class_precision": np.round(precision, 4).tolist(),
        "per_class_recall": np.round(recall, 4).tolist(),
        "per_class_f1": np.round(f1, 4).tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "y_pred": y_pred.astype(int).tolist(),
        "train_time_s": round(float(train_time_s), 2),
        "inference_ms_per_1000": round(float(ms_per_1000), 2),
        "model_size_bytes": model_size_bytes(model),
        "model_size_detail": model_size_detail(model),
        "env": environment(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(metrics, config.METRICS_DIR / f"{name}_{kind}_test.json")
    print(f"{name}_{kind}: test accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    return metrics
