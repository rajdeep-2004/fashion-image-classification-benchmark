# System Design

> Last updated: 2026-10-26 — How the pipeline behaves when it runs: order of steps, how leakage is prevented, what can fail. Structure lives in architecture.md.

## 1. Requirements that drive the design
| Requirement | Target | Why it matters |
|---|---|---|
| Reproducibility | Same split and same metrics on any machine | Teammates and the examiner must get identical numbers |
| No leakage | Test set never influences any choice | The most common reason ML coursework loses marks |
| Runtime | The full pipeline finishes in a few hours on an M3 with 8 GB | It has to be practical to re-run before submission |
| Memory | Peak RAM below ~5 GB | 8 GB laptop, with VS Code and a browser also open |
| Fair comparison | Every model measured with the same `evaluate()` | Results tables are only meaningful if the measurements match |

## 2. Key flows

### Flow A: Data step (phase 01, run once)
```mermaid
sequenceDiagram
  participant NB as 01_data_eda.ipynb
  participant D as data.py
  participant Web as Zalando mirror
  NB->>D: download_raw()
  D->>Web: GET 4 .gz files (skipped if present)
  D->>D: verify MD5 of each file → raise on mismatch
  NB->>D: run_sanity_checks()
  NB->>D: load_splits()
  D->>D: if data/splits/val_indices.npy missing → stratified split (seed 42), save it
  D-->>NB: X_train, y_train (54k), X_val, y_val (6k), X_test, y_test (10k), float32 [0,1]
```
Notes:
- `val_indices.npy` is committed, so everyone uses the exact same split even if sklearn's splitting code changes in a future version.
- EDA (PCA, t-SNE, mean images) runs on the **train split only**. Val and test stay unseen.

### Flow B: Features (phase 02)
`get_features(kind, split)` returns the cached `.npy` if it exists, otherwise builds it, saves it and returns it.
- `pixels`: flatten to 784-d. **No scaling here.** Scaling happens inside each model's Pipeline (see §3).
- `hog`: `skimage.feature.hog` per image with `config.HOG_PARAMS`. The cache filename includes the params
  (e.g. `hog_o9_p4_c2_train.npy`), so changing params can never silently reuse old features.
- `split` ∈ {`train`, `val`, `trainval`, `test`}. `trainval` = all 60k, used only for the final refit.
- HOG sweep: for each config, a proxy linear SVM (`SGDClassifier`, hinge loss, D-016) is fit on the train features and scored on val. Only val is used.

### Flow C: One experiment (phases 03–04)
```mermaid
sequenceDiagram
  participant NB as Notebook
  participant E as experiment.py
  participant M as models_*.py
  participant F as features.py
  participant V as evaluate.py
  NB->>E: tune_on_val(make_fn, grid, kind)
  E->>F: get_features(kind, "train"), get_features(kind, "val")
  loop each param combo
    E->>M: make_fn(**params) → unfitted Pipeline
    E->>E: fit on train, score accuracy on val
  end
  E->>V: save tuning table → results/metrics/{name}_{kind}_val.json
  E-->>NB: best_params, tuning table (for validation curve)
  NB->>E: final_test_run(make_fn, best_params, kind, name)
  E->>F: get_features(kind, "trainval"), get_features(kind, "test")
  E->>E: fit on all 60k (timed)
  E->>V: evaluate(model, X_test, y_test) → {name}_{kind}_test.json
  E->>E: append row to results/test_log.csv
```
Notes:
- RBF SVM: `tune_on_val` gets a stratified 10k subset of train (seed 42). The val set is still the same 6k.
- If `final_test_run` is called again for the same experiment, it still runs, but `test_log.csv` shows the repeat. That makes test-set reuse visible instead of hidden.

### Flow D: Results (phase 05)
`results.py` reads every `*_test.json` and builds `summary.csv` and the comparison plots.
The "best model" for error analysis and 5-fold CV is chosen by **val accuracy, not test accuracy**. All val scores must come
from models trained on the full 54k train split: for subset-tuned RBF SVM that's `_val_full.json`, written by
`experiment.full_train_val_score()` (D-029). Result: `rbf_svm_hog` (val 0.9157).
Error analysis reads the saved `confusion_matrix` / `y_pred` of each test JSON plus the true test labels/images from
`load_splits()`; nothing is predicted again (D-011, D-032). 5-fold CV of the best model runs on a 10k stratified subset of `trainval`.

## 3. Leakage controls (the core of this design)
| Possible loophole | How we close it |
|---|---|
| Scaler fit on val/test | `StandardScaler` lives inside the sklearn `Pipeline`, so `fit()` only ever sees training rows |
| Hyperparameters picked on test | `tune_on_val()` only loads `train` and `val`, and the test arrays are never passed in |
| HOG params picked on test | The sweep scores on val only |
| "Best model" picked on test | Selection uses val accuracy, recorded in the `_val.json` files (`_val_full.json` for subset-tuned RBF SVM, D-029) |
| EDA insights drawn from test | EDA runs on the train split only |
| Different splits per person | Committed `val_indices.npy`, plus its checksum printed and documented |
| Stale features after a param change | HOG params are encoded in the cache filename |
| Peeking at test repeatedly | Each experiment has one `final_test_run`, and every call is logged to `test_log.csv` |
| CV folds leaking scaling | CV runs `cross_val_score` on the whole Pipeline, so the scaler is refit per fold |
| Error analysis re-using test | Phase 05 reads saved predictions only; no model predicts on test again (D-011, D-032) |
| Duplicate images across train/val/test | Counted and reported in the sanity checks. Not removed, so results stay comparable to published benchmarks (D-004) |

## 4. Concurrency and race conditions
None that matter: one person runs one notebook at a time. Parallelism is limited to sklearn's
`n_jobs=config.N_JOBS` (4) inside RF and kNN. CV runs its folds one after another (`n_jobs=1`, D-018).

## 5. Async work and realtime
Not applicable.

## 6. Caching
| What | Where | Invalidation |
|---|---|---|
| Raw `.gz` files | `data/raw/` | Re-downloaded if missing or the MD5 check fails |
| Feature matrices | `features/{kind}_{params}_{split}.npy` | Filename changes when params change. Delete the folder to force a rebuild. |
| Split | `data/splits/val_indices.npy` | Never. It is fixed for the whole project. |

## 7. Failure modes
| Failure | Detection | Behavior / recovery |
|---|---|---|
| Download fails / no internet | urllib error | Stop with a message naming the URL. Retry later or copy the files in manually. |
| Corrupted / wrong file | MD5 mismatch | Delete the file and stop with an error. The next run re-downloads it. |
| Feature cache missing | File not found | Built automatically. |
| Out of memory (SVM/kNN) | Kernel crash | Lower `N_JOBS`, use `SVM_TUNE_SUBSET`, and close other apps. Documented in the README. |
| RBF SVM too slow | Wall-clock time | Tune on the 10k subset. The final 60k fit is run once: 196 s (pixels) and 451 s (HOG) on an M3 Pro. |
| Model doesn't converge (LR) | sklearn `ConvergenceWarning` | Raise `max_iter`. The warning is left visible, not suppressed. |

## 8. Security
- Downloads go to a fixed HTTPS-hosted URL in `config.py` and are verified by checksum before any parsing.
- No pickles are loaded from outside the project. Models are pickled only to measure size, never loaded back from untrusted sources.
- No secrets exist in this project.

## 9. Scaling plan
Not a concern for 70k images. The slowest steps are the RBF-SVM fit (O(n²)) and kNN prediction (brute force).
If they get too slow, the mitigation is the documented subset, or PCA-reduced pixels (parked idea).

## 10. Observability
- `Timer` prints the duration of every fit and prediction.
- Each metrics JSON records Python and sklearn versions, CPU model, `n_jobs` and a timestamp, so the timing numbers can be interpreted later.
- `results/test_log.csv` is the audit trail for test-set use.

## 11. Trade-offs accepted
| Trade-off | Why acceptable now | Revisit when |
|---|---|---|
| Single holdout val (6k) instead of k-fold for tuning | 6k is large enough for stable estimates, and it's simple and fast. k-fold is used once on the best model to show stability. | Val scores of two configs are within ~0.3% of each other |
| RBF SVM tuned on 10k, not 54k | Full-size tuning would take hours on a laptop | Never for v1. It's documented in the report. |
| Timing measured on laptops with `n_jobs=4` | Phase 03 ran on an Apple M3 and phase 04 on an M3 Pro; gaps of 10× or more are still clear, small gaps are approximate (D-028) | A single-machine re-timing, or Phase 2 GPU timings (report them separately) |
| Duplicates kept | Matches the official benchmark protocol | Sanity checks find a large number of duplicates |
