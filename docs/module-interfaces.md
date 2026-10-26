# Module Interfaces

> Last updated: 2026-10-24 — What each shared function in `src/` takes and returns.
> This project has no API. These functions are the agreements between phases. Change a signature here **before**
> changing the code, and update every caller (see D-010). File formats are in `data-formats.md`.

Conventions:
- All functions are plain functions. Classes are used only where sklearn needs them (Pipelines) and for `Timer`.
- Inputs are never modified in place. Functions return new arrays/dicts.
- Errors raise `ValueError` (bad argument) or `RuntimeError` (missing/corrupt data), with a message saying how to fix it.

## `src/config.py` — constants only
`SEED = 42`, `VAL_SIZE = 6000`, `SVM_TUNE_SUBSET = 10000`, `N_JOBS = 4`, `LOGREG_MAX_ITER = 5000`, `CLASS_NAMES` (10 str),
`HOG_PARAMS` (dict), `RAW_URL_BASE`, `RAW_FILES` (filename → MD5), and paths `ROOT`, `DATA_RAW`, `SPLITS_DIR`,
`VAL_INDICES_PATH`, `FEATURES_DIR`, `METRICS_DIR`, `FIGURES_DIR`, `RESULTS_DIR`, `TEST_LOG_PATH`, `HOG_SWEEP_PATH`.
Modules read these at call time (`config.X` inside functions), so a scratch check can point them at a temp folder.

## `src/utils.py`
| Function | Takes | Returns | Notes |
|---|---|---|---|
| `set_seed(seed=SEED)` | int | None | Seeds `random` and `numpy` |
| `Timer(label)` | str | context manager with `.seconds` | `with Timer("fit") as t: ...` prints `fit: 12.3s` |
| `save_fig(fig, name)` | matplotlib Figure, filename without extension | `Path` | Saves `FIGURES_DIR/{name}.png` at 300 dpi, creating the folder if needed |
| `save_json(obj, path)` | dict, `Path` | None | Writes indented JSON, converting NumPy numbers/arrays; creates the folder |
| `display_path(path)` | `Path` | str | Path relative to the repo for messages; full path if outside the repo (e.g. Google Drive) |

## `src/data.py`
| Function | Takes | Returns | Errors |
|---|---|---|---|
| `download_raw()` | — | None | `RuntimeError` on a network failure or MD5 mismatch (the bad file is deleted) |
| `load_raw()` | — | `(X_train_full, y_train_full, X_test, y_test)`, uint8 images (N,28,28), int64 labels | `RuntimeError` if the files are missing (tells you to run `download_raw()`) |
| `run_sanity_checks()` | — | dict with keys `train_shape`, `test_shape`, `image_dtype`, `label_dtype`, `pixel_range`, `label_range`, `nan_count`, `train_class_counts`, `test_class_counts`, `duplicates_within_train`, `duplicates_within_test`, `test_images_also_in_train`. Also prints them | `RuntimeError` if shapes, label range, NaNs or class counts are wrong |
| `load_splits(verbose=True)` | bool | dict with keys from `data-formats.md` §4 | Creates and saves `val_indices.npy` on the first call. Prints its MD5 unless `verbose=False`. `RuntimeError` if the saved file isn't a valid 6,000-index, 600-per-class split |

## `src/features.py`
| Function | Takes | Returns | Notes |
|---|---|---|---|
| `extract_pixels(X)` | (N,28,28) float32 | (N,784) float32 | Flatten only |
| `extract_hog(X, orientations, pixels_per_cell, cells_per_block)` | (N,28,28) float32 + params | (N,d) float32 | `skimage.feature.hog`, L2-Hys block norm |
| `get_features(kind, split)` | `"pixels"\|"hog"`, `"train"\|"val"\|"trainval"\|"test"` | (N,d) float32 | Loads from cache or builds and saves. `ValueError` for an unknown kind/split |
| `get_labels(split)` | split name | (N,) int64 | Same row order as `get_features` |
| `hog_tag(params)` | HOG params dict | str, e.g. `"o9_p4_c2"` | Used in cache filenames and the sweep table |
| `run_hog_sweep(grid)` | dict of lists | `pandas.DataFrame` (columns in `data-formats.md` §6) | Uses train/val only. Writes `results/hog_sweep.csv` |

## `src/models_linear.py` and `src/models_tree_kernel.py`
Every function returns a **new, unfitted** `sklearn.pipeline.Pipeline`. Scaling is included where the model needs it.
| Function | Hyperparameters | Pipeline steps |
|---|---|---|
| `make_dummy()` | — | `DummyClassifier(strategy="most_frequent")` |
| `make_logreg(C)` | C | `StandardScaler` → `LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=SEED)` |
| `make_knn(n_neighbors, weights, metric)` | k, weights, metric | `StandardScaler` → `KNeighborsClassifier(n_jobs=N_JOBS)` |
| `make_rf(n_estimators, max_depth, max_features)` | as named | `RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS)` (no scaler, as trees don't need one) |
| `make_linear_svm(C)` | C | `StandardScaler` → `LinearSVC(random_state=SEED)` |
| `make_rbf_svm(C, gamma)` | C, gamma | `StandardScaler` → `SVC(kernel="rbf")` |

Default tuning grids live next to these functions as constants (e.g. `KNN_GRID`), copied from the PRD/idea.md.

Model files only build unfitted Pipelines and hold grids. Figures (e.g. the LR weight maps) live in the notebooks (D-025).

## `src/experiment.py` — the only module that runs train → val → test
| Function | Takes | Returns | Side effects |
|---|---|---|---|
| `tune_on_val(name, make_fn, grid, kind, train_subset=None)` | model name, factory, grid dict, feature kind, optional subset size | `(best_params: dict, table: DataFrame)` | Writes `{name}_{kind}_val.json`. **Never loads test.** |
| `full_train_val_score(name, make_fn, params, kind)` | as above + chosen params | dict (`data-formats.md` §7) | Refits on the **full 54k train**, scores on val. For subset-tuned models (RBF SVM) so their val accuracy is comparable (D-029). Writes `{name}_{kind}_val_full.json`. **Never loads test.** |
| `final_test_run(name, make_fn, params, kind)` | as above + chosen params | metrics dict (`data-formats.md` §8) | Fits on `trainval`, calls `evaluate()`, appends to `test_log.csv` |
| `cross_validate_best(name, make_fn, params, kind, subset=None, folds=5)` | as above | dict (`data-formats.md` §9) | Stratified K-fold on `trainval` (or a stratified subset). Writes `{name}_{kind}_cv.json` |
| `stratified_subset(X, y, size)` | arrays, int | `(X_sub, y_sub)` | Class-balanced subset, seed 42, original row order (used for the SVM tuning subset and CV subset) |

## `src/evaluate.py`
| Function | Takes | Returns | Notes |
|---|---|---|---|
| `evaluate(model, X_test, y_test, name, kind, best_params, train_time_s)` | fitted model + test data + metadata | metrics dict | Times `predict` (best of 3), computes all metrics, writes `{name}_{kind}_test.json`. Usable by any model with `.predict()`, including the Phase 2 CNN wrapper |
| `model_size_bytes(model)` | fitted model | int | `len(pickle.dumps(model))` |
| `model_size_detail(model)` | fitted model | dict `{type, value}` | Support vectors / tree nodes / stored samples / coefficients |
| `environment()` | — | dict `{python, sklearn, cpu, n_jobs}` | Stored in every test JSON as `env` |

## `src/results.py`
| Function | Takes | Returns | Notes |
|---|---|---|---|
| `load_all_metrics(split="test")` | `"val"\|"test"\|"cv"` | list of dicts | Reads `results/metrics/*_{split}.json` |
| `build_summary()` | — | DataFrame | Writes `results/summary.csv` |
| `comparable_val_acc(experiment)` | experiment id | float | `val_acc` from `_val_full.json` if it exists, else `best_val_acc` from `_val.json` (D-029). Used by `pick_best_experiment()` and `build_summary()` |
| `pick_best_experiment()` | — | experiment id | Highest **val** accuracy, using `_val_full.json` for subset-tuned experiments (D-029) |
| `plot_pixels_vs_hog(df)` | summary df | Figure | RQ2 |
| `plot_acc_vs_inference(df)` | summary df | Figure | RQ3 |
| `plot_recall_heatmap(df)` | summary df | Figure | RQ4 |
| `plot_confusion(experiment, normalize=True)` | experiment id | Figure | From the saved `confusion_matrix`; rows = true class; saves `errors_confusion_{experiment}.png` |
| `top_confused_pairs(experiment, k=5)` | experiment id | DataFrame `(true, pred, count, rate)` | Off-diagonal cells of the confusion matrix |
| `plot_misclassified(experiment, true_c, pred_c, n=16)` | experiment id, class pair (index 0–9 or class name) | Figure | Uses stored `y_pred` + `load_splits()["X_test"]`; shows the first `n` matches in test order (not hand-picked) |
| `model_agreement(exp_a, exp_b)` | two experiment ids | dict: `both_correct`, `both_incorrect`, `only_a_incorrect`, `only_b_incorrect`, `same_wrong_label`, `total`, `agreement_pct` | From the two saved `y_pred` + `load_splits()["y_test"]` |

## Typical notebook usage
```python
from src.models_linear import make_knn, KNN_GRID
from src.experiment import tune_on_val, final_test_run

best, table = tune_on_val("knn", make_knn, KNN_GRID, kind="hog")
metrics = final_test_run("knn", make_knn, best, kind="hog")
```
