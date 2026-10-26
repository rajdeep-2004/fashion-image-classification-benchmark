"""Phase 05: summary table, comparison plots, best-model choice and error analysis.

Everything here reads saved results (results/metrics/*.json); nothing is retrained or re-tested. The error analysis also
reads the true test labels and images via load_splits(), the phase 05 exception in AGENTS.md §4: it only looks at
predictions that final_test_run() already saved. Tasks 05.1-05.3 by Anand, 05.4 by Omved (D-031).
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, utils
from src.data import load_splits

MODEL_ORDER = ["dummy", "logreg", "knn", "rf", "linear_svm", "rbf_svm"]
FEATURE_ORDER = ["pixels", "hog"]
RECALL_COLS = [f"recall_c{i}" for i in range(config.NUM_CLASSES)]


# --- 1. Loading, best model and summary table (Task 05.1) ----------------------

def _read_json(path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing {utils.display_path(path)}: run the notebook of the phase that creates it first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_metrics(split: str = "test") -> list[dict]:
    """Read every results/metrics/*_{split}.json ('val', 'test' or 'cv'), sorted by file name."""
    if split not in ("val", "test", "cv"):
        raise ValueError(f"split must be 'val', 'test' or 'cv', got {split!r}")
    return [_read_json(p) for p in sorted(config.METRICS_DIR.glob(f"*_{split}.json"))]


def comparable_val_acc(experiment: str) -> float:
    """Val accuracy from a model trained on all 54k train images (D-029).

    Subset-tuned experiments (RBF SVM) have a {experiment}_val_full.json; every other one uses best_val_acc.
    """
    full_path = config.METRICS_DIR / f"{experiment}_val_full.json"
    if full_path.exists():
        return float(_read_json(full_path)["val_acc"])
    return float(_read_json(config.METRICS_DIR / f"{experiment}_val.json")["best_val_acc"])


def pick_best_experiment() -> str:
    """The experiment with the highest comparable val accuracy. Test scores play no part (D-005, D-029)."""
    experiments = [d["experiment"] for d in load_all_metrics("val")]
    if not experiments:
        raise RuntimeError("No *_val.json files in results/metrics/: run notebooks 03 and 04 first.")
    return max(experiments, key=comparable_val_acc)  # max() keeps the first of equal scores


def build_summary() -> pd.DataFrame:
    """One row per *_test.json with the data-formats.md §11 columns; writes results/summary.csv.

    When timing_comparable is false (the two LR runs, D-024), both timing columns are NaN, never zero.
    """
    test_metrics = load_all_metrics("test")
    if not test_metrics:
        raise RuntimeError("No *_test.json files in results/metrics/: run notebooks 03 and 04 first.")

    rows = []
    for tm in test_metrics:
        timing_ok = tm.get("timing_comparable", True)
        rows.append({
            "experiment": tm["experiment"], "model": tm["model"], "features": tm["features"],
            "val_acc": round(comparable_val_acc(tm["experiment"]), 4),
            "test_acc": round(float(tm["accuracy"]), 4),
            "test_macro_f1": round(float(tm["macro_f1"]), 4),
            **{col: round(float(r), 4) for col, r in zip(RECALL_COLS, tm["per_class_recall"], strict=True)},
            "train_time_s": round(float(tm["train_time_s"]), 2) if timing_ok else np.nan,
            "inference_ms_per_1000": round(float(tm["inference_ms_per_1000"]), 2) if timing_ok else np.nan,
            "model_size_bytes": int(tm["model_size_bytes"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "features"], key=lambda col: col.map(
        lambda v: (MODEL_ORDER + FEATURE_ORDER).index(v))).reset_index(drop=True)
    path = config.RESULTS_DIR / "summary.csv"
    df.to_csv(path, index=False)
    print(f"saved {utils.display_path(path)} ({len(df)} experiments)")
    return df


# --- 2. Comparison plots (Task 05.2) --------------------------------------------

def plot_pixels_vs_hog(df: pd.DataFrame) -> plt.Figure:
    """RQ2: grouped bars of test accuracy, pixels vs HOG, for every model. Saves results_pixels_vs_hog.png."""
    names = {"dummy": "Dummy", "logreg": "Logistic Reg", "knn": "k-NN", "rf": "Random Forest",
             "linear_svm": "Linear SVM", "rbf_svm": "RBF SVM"}
    acc = df.pivot(index="model", columns="features", values="test_acc").reindex(MODEL_ORDER).dropna() * 100
    x, width = np.arange(len(acc)), 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    for offset, kind, label, color in ((-width / 2, "pixels", "Raw Pixels (784)", "#4C72B0"),
                                       (width / 2, "hog", "HOG (1,728)", "#DD8452")):
        bars = ax.bar(x + offset, acc[kind], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8.5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("RQ2: Effect of Feature Representation on Test Accuracy (Pixels vs HOG)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in acc.index], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
    fig.tight_layout()
    utils.save_fig(fig, "results_pixels_vs_hog")
    return fig


def plot_acc_vs_inference(df: pd.DataFrame) -> plt.Figure:
    """RQ3: test accuracy vs inference time per 1,000 images (log x-axis). Saves results_acc_vs_time.png.

    Leaves out experiments with NaN timings (LR, D-024) and the dummy baseline (0.00 ms can't sit on a log
    axis); the footnote names them. Raw timings stay in summary.csv.
    """
    shown = df[df["inference_ms_per_1000"].notna() & (df["model"] != "dummy")]
    # Label positions in points, hand-placed so the labels don't overlap: (dx, dy, horizontal alignment)
    label_pos = {"rbf_svm_hog": (-9, 4, "right"), "rf_pixels": (-9, 5, "right"), "rf_hog": (9, -10, "left")}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    for kind, label, color, marker in (("pixels", "Pixels", "#4C72B0", "o"), ("hog", "HOG", "#DD8452", "s")):
        sub = shown[shown["features"] == kind]
        ax.scatter(sub["inference_ms_per_1000"], sub["test_acc"] * 100, c=color, marker=marker, s=90,
                   edgecolors="black", linewidth=0.8, label=f"{label} features", zorder=4)
    for _, row in shown.iterrows():
        dx, dy, ha = label_pos.get(row["experiment"], (9, -4, "left"))
        ax.annotate(row["experiment"], (row["inference_ms_per_1000"], row["test_acc"] * 100),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlim(0.8, 50000)
    ax.set_ylim(80, 94)
    ax.set_xlabel("Inference time (ms per 1,000 images, log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("RQ3: Accuracy vs Inference Speed Trade-off", fontsize=13, pad=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right", fontsize=10)
    fig.tight_layout()

    excluded = ", ".join(df.loc[~df.index.isin(shown.index), "experiment"])
    fig.text(0.02, -0.01, (
        f"Not shown: {excluded}. LR timings are not comparable (D-024); dummy timings are ~0 ms and come from another "
        "machine.\nTimings: phase 03 models (kNN) on an Apple M3, phase 04 models (RF, SVMs) on an Apple M3 Pro, so "
        "cross-phase speed gaps are approximate.\nrbf_svm_hog's 28.9 s / 1,000 looks ~2x slower than expected; "
        "treat it as an upper bound (D-028)."), ha="left", va="top", fontsize=8, color="#444444")
    utils.save_fig(fig, "results_acc_vs_time")
    return fig


def plot_recall_heatmap(df: pd.DataFrame) -> plt.Figure:
    """RQ4: per-class recall of every experiment, one cell per (experiment, class). Saves results_recall_heatmap.png."""
    matrix = df[RECALL_COLS].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["experiment"], fontsize=9.5)
    ax.set_title("RQ4: Per-Class Recall Heatmap Across All Experiments", fontsize=13, pad=14)
    ax.set_xlabel("Fashion-MNIST Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Experiment", fontsize=11, labelpad=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label("Class Recall", fontsize=10)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if matrix[i, j] > 0.65 else "black")  # white text on dark cells
    fig.tight_layout()
    utils.save_fig(fig, "results_recall_heatmap")
    return fig


# --- 3. Error analysis of saved test predictions (Task 05.4, D-032) ---------------

def _saved_test(experiment: str) -> dict:
    return _read_json(config.METRICS_DIR / f"{experiment}_test.json")


def _class_index(c) -> int:
    """Accept a class index (0-9) or a class name such as 'Shirt'."""
    if isinstance(c, str):
        if c not in config.CLASS_NAMES:
            raise ValueError(f"Unknown class name {c!r}; use one of {config.CLASS_NAMES}")
        return config.CLASS_NAMES.index(c)
    if not 0 <= int(c) < config.NUM_CLASSES:
        raise ValueError(f"Class index must be 0-{config.NUM_CLASSES - 1}, got {c}")
    return int(c)


def plot_confusion(experiment: str, normalize: bool = True) -> plt.Figure:
    """Confusion matrix from the saved test JSON (rows = true class). Saves errors_confusion_{experiment}.png.

    normalize=True divides each row by its total, so the diagonal is each class's recall.
    """
    cm = np.asarray(_saved_test(experiment)["confusion_matrix"])
    matrix = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    vmax = 1.0 if normalize else cm.max()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(f"Confusion Matrix: {experiment} ({'row-normalised' if normalize else 'counts'})", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Class", fontsize=11, labelpad=8)
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Proportion" if normalize else "Count", fontsize=10)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, f"{matrix[i, j]:.2f}" if normalize else str(matrix[i, j]), ha="center", va="center",
                fontsize=8.5, color="white" if matrix[i, j] > vmax / 2 else "black")
    fig.tight_layout()
    utils.save_fig(fig, f"errors_confusion_{experiment}")
    return fig


def top_confused_pairs(experiment: str, k: int = 5) -> pd.DataFrame:
    """The k largest off-diagonal confusion-matrix cells as (true, pred, count, rate); rate = count / class total."""
    cm = np.asarray(_saved_test(experiment)["confusion_matrix"])
    rows = [{"true": config.CLASS_NAMES[i], "pred": config.CLASS_NAMES[j], "count": int(cm[i, j]),
             "rate": round(cm[i, j] / cm[i].sum(), 4)}
            for i, j in np.ndindex(cm.shape) if i != j]
    return pd.DataFrame(rows).sort_values("count", ascending=False, kind="stable").head(k).reset_index(drop=True)


def plot_misclassified(experiment: str, true_c, pred_c, n: int = 16) -> plt.Figure:
    """Grid of the first n test images (in test order, not hand-picked) of class true_c predicted as pred_c.

    Classes can be indices or names. Saves errors_misclassified_{true}_vs_{pred}.png.
    """
    true_c, pred_c = _class_index(true_c), _class_index(pred_c)
    y_pred = np.asarray(_saved_test(experiment)["y_pred"])
    splits = load_splits(verbose=False)
    matches = np.flatnonzero((splits["y_test"] == true_c) & (y_pred == pred_c))
    if len(matches) == 0:
        raise ValueError(f"{experiment} never predicts {config.CLASS_NAMES[pred_c]} for a true "
                         f"{config.CLASS_NAMES[true_c]}; pick a pair from top_confused_pairs().")
    shown = matches[:n]
    ncols = min(4, len(shown))
    nrows = int(np.ceil(len(shown) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, shown):
        ax.imshow(splits["X_test"][idx], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Test #{idx}", fontsize=9)
    true_name, pred_name = config.CLASS_NAMES[true_c], config.CLASS_NAMES[pred_c]
    fig.suptitle(f"Misclassified: True '{true_name}' -> Pred '{pred_name}'\n"
                 f"({experiment}, showing {len(shown)} of {len(matches)})", fontsize=12, y=1.02)
    fig.tight_layout()
    slug = lambda name: name.lower().replace("/", "_").replace(" ", "_")
    utils.save_fig(fig, f"errors_misclassified_{slug(true_name)}_vs_{slug(pred_name)}")
    return fig


def model_agreement(exp_a: str, exp_b: str) -> dict:
    """Compare two experiments' saved test predictions image by image.

    Returns counts: both_correct, both_incorrect, only_a_incorrect, only_b_incorrect, same_wrong_label (both wrong with
    the same label), total, and agreement_pct (% of images where both predict the same label, right or wrong).
    """
    pred_a = np.asarray(_saved_test(exp_a)["y_pred"])
    pred_b = np.asarray(_saved_test(exp_b)["y_pred"])
    y_test = load_splits(verbose=False)["y_test"]
    right_a, right_b = pred_a == y_test, pred_b == y_test
    return {
        "both_correct": int(np.sum(right_a & right_b)),
        "both_incorrect": int(np.sum(~right_a & ~right_b)),
        "only_a_incorrect": int(np.sum(~right_a & right_b)),
        "only_b_incorrect": int(np.sum(right_a & ~right_b)),
        "same_wrong_label": int(np.sum(~right_a & ~right_b & (pred_a == pred_b))),
        "total": len(y_test),
        "agreement_pct": round(float(np.mean(pred_a == pred_b) * 100), 2),
    }
