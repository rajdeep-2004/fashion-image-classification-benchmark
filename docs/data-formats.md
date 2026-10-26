# Data Formats

> Last updated: 2026-10-24 — Every file the pipeline reads or writes, and exactly what is inside it.
> This project has no database. These files play that role. Change a format here **before** changing the code (see D-010).

## 1. Naming conventions
| Thing | Allowed values |
|---|---|
| Model `name` | `dummy`, `logreg`, `knn`, `rf`, `linear_svm`, `rbf_svm` (Phase 2 will add `cnn`, …) |
| Feature `kind` | `pixels`, `hog` |
| Data `split` | `train` (54k), `val` (6k), `trainval` (60k = train + val), `test` (10k) |
| Experiment id | `{name}_{kind}`, e.g. `rbf_svm_hog` |

Class labels 0–9 = T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot (`config.CLASS_NAMES`).

## 2. Raw data — `data/raw/` (gitignored)
Downloaded from the official repository `https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/`.

| File | Contents | Official MD5 |
|---|---|---|
| `train-images-idx3-ubyte.gz` | 60,000 × 28 × 28 uint8 | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| `train-labels-idx1-ubyte.gz` | 60,000 uint8 labels | `25c81989df183df01b3e8a0aad5dffbe` |
| `t10k-images-idx3-ubyte.gz` | 10,000 × 28 × 28 uint8 | `bef4ecab320f06d8554ea6380940ec79` |
| `t10k-labels-idx1-ubyte.gz` | 10,000 uint8 labels | `bb300cfdad3c16e7a12a480ee83cd310` |

The MD5 values are taken from the official README. They are confirmed against a real download in phase 01 (D-006).
IDX format: gzip file, then a big-endian header (16 bytes for images, 8 bytes for labels), then raw uint8 bytes. It is read with
`np.frombuffer(data, np.uint8, offset=16 or 8)`.

## 3. Split — `data/splits/val_indices.npy` (COMMITTED)
| Field | Value |
|---|---|
| dtype / shape | `int64`, `(6000,)` |
| Contents | Sorted indices into the official 60k train set that form the val split. 600 per class. |
| Train split | All indices 0–59,999 **not** in this file (54,000) |
| Checksum | MD5 of the file, printed by `load_splits()` and recorded in the README in phase 01 |

## 4. In-memory splits — return value of `load_splits()`
| Key | dtype | Shape | Range |
|---|---|---|---|
| `X_train`, `X_val`, `X_test` | float32 | (54000/6000/10000, 28, 28) | [0, 1] |
| `y_train`, `y_val`, `y_test` | int64 | (54000/6000/10000,) | 0–9 |

## 5. Feature caches — `features/` (gitignored)
| File pattern | dtype | Shape |
|---|---|---|
| `pixels_{split}.npy` | float32 | (N, 784). Unscaled values in [0,1]. Scaling happens inside model Pipelines (D-004) |
| `hog_o{orientations}_p{ppc}_c{cpb}_{split}.npy` | float32 | (N, d), where d depends on the params. Chosen config o12/p4/c2 → **1,728** (D-017) |

Rows are in the same order as the matching `y_{split}`. `trainval` rows are train rows followed by val rows.

## 6. HOG sweep table — `results/hog_sweep.csv` (COMMITTED)
| Column | Type | Meaning |
|---|---|---|
| `orientations` | int | 6 / 9 / 12 |
| `pixels_per_cell` | int | 4 or 7 (square cells) |
| `cells_per_block` | int | 1 or 2 |
| `dim` | int | Feature length |
| `extract_s_per_1000` | float | Seconds to extract 1,000 images |
| `val_acc` | float | Proxy linear-SVM (SGD, hinge loss) accuracy on val, trained on train (D-016) |

## 7. Tuning results — `results/metrics/{name}_{kind}_val.json` (COMMITTED)
```json
{
  "experiment": "knn_hog",
  "model": "knn",
  "features": "hog",
  "split": "val",
  "train_subset": null,
  "grid": {"n_neighbors": [1, 3, 5, 7, 9], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
  "rows": [
    {"params": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}, "val_acc": 0.0, "val_macro_f1": 0.0, "fit_s": 0.0}
  ],
  "best_params": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
  "best_val_acc": 0.0,
  "timestamp": "2026-10-24T12:00:00"
}
```
- `train_subset` is `null`, or the subset size (e.g. `10000` for `rbf_svm`).
- The "best model" for phase 05 is the experiment with the highest val accuracy (never test accuracy). For experiments tuned
  on a subset (`train_subset` not null), the val accuracy comes from `{name}_{kind}_val_full.json` instead (D-029).

**Full-train val score — `results/metrics/{name}_{kind}_val_full.json`** (COMMITTED, only for subset-tuned experiments)
```json
{"experiment": "rbf_svm_hog", "model": "rbf_svm", "features": "hog", "split": "val", "train_rows": 54000,
 "params": {"C": 10, "gamma": "scale"}, "fit_s": 0.0, "val_acc": 0.0, "val_macro_f1": 0.0, "timestamp": "2026-10-25T12:00:00"}
```
Written by `experiment.full_train_val_score()`: the chosen params refit on all 54k train images and scored on the same 6k val.

## 8. Test results — `results/metrics/{name}_{kind}_test.json` (COMMITTED)
```json
{
  "experiment": "rbf_svm_hog",
  "model": "rbf_svm",
  "features": "hog",
  "split": "test",
  "best_params": {"C": 10, "gamma": "scale"},
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "per_class_precision": [10 floats],
  "per_class_recall": [10 floats],
  "per_class_f1": [10 floats],
  "confusion_matrix": [[10 x 10 ints]],
  "y_pred": [10000 ints],
  "train_time_s": 0.0,
  "inference_ms_per_1000": 0.0,
  "model_size_bytes": 0,
  "model_size_detail": {"type": "support_vectors | tree_nodes | stored_samples | coefficients | none", "value": 0},
  "env": {"python": "3.12.x", "sklearn": "1.x.y", "cpu": "Apple M3", "n_jobs": 4},
  "timestamp": "2026-10-24T12:00:00"
}
```
- `confusion_matrix[i][j]` = number of test images with true class `i` predicted as class `j`.
- `y_pred` is in the same order as `y_test`. It's stored so phase 05 can draw misclassified images and compare
  models without retraining anything (~30 KB per file).
- `train_time_s` is the final fit on all 60k (`trainval`).
- Optional `timing_comparable` (bool, defaults to true when absent) and `timing_note` (str) record known timing contamination. When false, preserve the raw timing values for provenance but exclude both from speed comparisons. This flag does not invalidate accuracy or other predictive metrics.

## 9. Stability check — `results/metrics/{name}_{kind}_cv.json` (COMMITTED, best model only)
```json
{
  "experiment": "rbf_svm_hog", "folds": 5, "data": "trainval", "subset": 10000,
  "best_params": {"C": 10, "gamma": "scale"},
  "fold_acc": [5 floats], "mean_acc": 0.0, "std_acc": 0.0,
  "timestamp": "2026-10-24T12:00:00"
}
```

## 10. Test audit log — `results/test_log.csv` (COMMITTED, append-only)
| Column | Example |
|---|---|
| `timestamp` | `2026-10-24T12:00:00` |
| `experiment` | `logreg_pixels` |
| `accuracy` | `0.8431` |

One row is added each time `final_test_run()` runs. In the final repo, each experiment should appear once.
Any repeat must be explained in `decisions.md`.

## 11. Summary table — `results/summary.csv` (COMMITTED)
One row per `*_test.json`, built by `results.build_summary()`.
For `timing_comparable=false`, retain the experiment and its accuracy metrics, but write empty/NaN training and inference times. Never substitute zero. Speed plots and rankings must exclude these missing times and name the excluded experiments in a note. Raw measurements remain in the source JSON.
| Column | Type |
|---|---|
| `experiment`, `model`, `features` | str |
| `val_acc` | float: the comparable val score, from `_val_full.json` if it exists (subset-tuned RBF SVM), else `best_val_acc` from `_val.json` (D-029) |
| `test_acc`, `test_macro_f1` | float |
| `recall_c0` … `recall_c9` | float |
| `train_time_s`, `inference_ms_per_1000` | float |
| `model_size_bytes` | int |

## 12. Figures — `results/figures/` (COMMITTED)
- PNG, 300 dpi, one plot per file, saved only via `utils.save_fig()`.
- Name pattern: `{stage}_{what}.png`. `stage` is one of `eda`, `features`, `linear`, `treekernel`, `results`, `errors`.
  Examples: `eda_sample_grid.png`, `features_hog_examples.png`, `linear_knn_val_curve_pixels.png`,
  `results_acc_vs_time.png`, `errors_misclassified_shirt_vs_tshirt.png`.

Phase 03 produces `linear_logreg_val_curve_pixels.png`, `linear_logreg_val_curve_hog.png`,
`linear_knn_val_curve_pixels.png`, `linear_knn_val_curve_hog.png`, and `linear_logreg_weights.png`.
LR curves show every C; kNN curves hold the validation-winning weights/metric fixed while varying k.
Weight maps are 2×5 standardised-pixel coefficient maps with one symmetric colour scale.

Phase 05 produces `results_pixels_vs_hog.png`, `results_acc_vs_time.png`, `results_recall_heatmap.png`,
`errors_confusion_{experiment}.png` (the best model: `errors_confusion_rbf_svm_hog.png`) and one
`errors_misclassified_{true}_vs_{pred}.png` per top-5 confused pair. Class names are lower-cased with `/` and spaces
turned into `_`, e.g. `errors_misclassified_shirt_vs_t-shirt_top.png`.
