# Product Requirements

> Status: approved; v1 delivered 2026-10-26 — Last updated: 2026-10-26
> Scope: **Capstone Phase 1 (classical ML) = release v1.** Capstone Phase 2 (CNN + hybrid) will be planned later as v2.

## 1. Problem
A small online clothing catalogue wants to auto-tag product photos (shirt, trouser, sneaker, …)
instead of tagging them by hand. Before reaching for deep learning, we want to know how far
**classical machine learning with handcrafted features** can go on this task, which model gives the
best accuracy / speed / size trade-off, and *why* these methods fail where they fail. The answer is a
reproducible, leak-free benchmark on Fashion-MNIST whose error analysis motivates the Phase 2 CNN.

## 2. Users
| User / actor | What they need | How often |
|---|---|---|
| Our team (builders) | One pipeline that gives identical numbers on every machine | Every working session |
| Professor / examiners | A correct methodology with no leakage, clear results, a defensible error analysis | At submission and viva |
| Each team member at the viva | To explain every stage (data → features → models → evaluation) | Viva |
| Future us (Phase 2) | A shared split and `evaluate()` so the CNN plugs in and compares directly | Phase 2 |

## 3. Goals
Each goal answers one of the research questions:
- **RQ1:** How accurately can classical ML classify Fashion-MNIST from raw pixels?
- **RQ2:** Do HOG features beat raw pixels, and for which models?
- **RQ3:** Which model gives the best accuracy / inference-speed / model-size trade-off?
- **RQ4:** Which classes are systematically confused, and why do handcrafted features fail there?

Measurable outcomes:
- 4 models (LR, kNN, RF, RBF-SVM) × 2 feature sets (pixels, HOG) = **8 tuned experiments**, plus a dummy baseline. All are in one results table.
  *Delivered:* these plus the should-have LinearSVC (S-01) = 10 tuned experiments + 2 dummy runs = **12** in `summary.csv`.
- Re-running the pipeline from scratch reproduces the same split (checked by checksum) and the same metrics.
- **Zero leakage.** The test set is never used for tuning, and scalers are fit only on training data.
- Pixel-feature test accuracies fall within the published sanity ranges (§8). Anything outside them is investigated as a bug.

## 4. Non-goals
- No neural networks, CNNs or data augmentation. Those belong to Phase 2 (v2).
- No dataset other than Fashion-MNIST.
- No web app, API, deployment or model serving.
- No experiment-tracking platforms (MLflow, W&B), Docker, CI or committed tests (lean profile).
- No advanced code architecture. Plain functions in a few small modules, plus notebooks.
- The LaTeX report and slides are not built in this plan. They will go in the repo later (parked).

## 5. Features
### Must have (v1)
| ID | Feature | Acceptance criteria |
|---|---|---|
| F-01 | Data download + integrity check | Downloads the 4 official IDX files and verifies each one against the MD5 checksum published by Zalando (see D-006). A mismatch stops the run with an error. |
| F-02 | Loader + fixed split | `load_splits()` returns the 54k train / 6k val (stratified, seed 42) and 10k test as float32 arrays in [0,1]. Split indices are saved to disk, and a checksum of them is printed and documented. |
| F-03 | Data sanity checks | The code checks shapes, dtypes, label range, NaNs, class counts, exact duplicates (by hash) and train/test overlap, and reports the results. |
| F-04 | EDA notebook | Sample grid, per-class mean/std images, intensity histogram, PCA 2-D projection, t-SNE on a ~5k subsample, PCA explained-variance curve, class-mean similarity matrix. At least 6 PNGs are saved, plus a paragraph naming the expected hard classes. |
| F-05 | Pixel features | 784-d flattened vectors. Scaling is done inside each model's sklearn `Pipeline`, so it is always fit on training data only. |
| F-06 | HOG features + sweep | A sweep over orientations {6,9,12} × pixels_per_cell {4,7} × cells_per_block {1,2}, scored on val with a fast proxy model. The table records dimensionality, extraction time and val accuracy. The chosen config is saved, and the HOG features are cached to disk. |
| F-07 | Evaluation harness | `evaluate()` computes accuracy, macro F1, per-class precision/recall, the confusion matrix, train time, inference ms per 1,000 images and model size (serialized bytes), and writes one JSON per experiment. |
| F-08 | Test-set guard | Tuning code only ever sees train/val. Test metrics are produced in a single final-evaluation step per experiment, after refitting on the full 60k. |
| F-09 | Dummy + Logistic Regression + kNN | Tuned on val over the grids in idea.md, with validation curves. Test JSONs exist for both feature sets. LR weight heatmaps per class (pixels). |
| F-10 | Random Forest + RBF SVM | Tuned on val. The RBF SVM is tuned on a stratified 10k train subset, then refit on the full 60k. Test JSONs exist for both feature sets. RF importance heatmap, SVM support-vector count. |
| F-11 | Stability check | 5-fold stratified CV (mean ± std) for the best overall config, on the 60k train set (on a subset if it's an SVM, documented). |
| F-12 | Aggregation + comparison | `results/summary.csv` holds every experiment. Plots: pixels-vs-HOG grouped bars (RQ2), accuracy-vs-inference-time scatter (RQ3), per-class recall heatmap (RQ4). |
| F-13 | Error analysis | Normalised confusion matrix of the best model, the top-5 confused pairs, a grid of misclassified images per pair, a check of whether the best two models make the same mistakes, and a written explanation linked back to the EDA. |
| F-14 | README | Setup (conda env + VS Code / Colab fallback), how to run each stage in order, results table, folder map. |

### Should have
| ID | Feature | Notes |
|---|---|---|
| S-01 | LinearSVC on both feature sets | Cheap. Shows the gap between a linear and a kernel SVM. |
| S-02 | HOG visualisation per class | Original image next to its HOG image, for the report and viva. |

### Later / maybe
- PCA-reduced pixels as a third feature set (speeds up SVM/kNN, shows the curse of dimensionality).
- LBP texture features; L1 vs L2 Logistic Regression.
- IEEE LaTeX report and slides inside `report/` (parked; planned after the code is done).
- Phase 2 (v2): small CNN using the same split and `evaluate()`, ablations, a hybrid CNN-features → SVM model.

## 6. Core user flows
### Flow 1: Reproduce the whole benchmark from a clean clone
1. Create the conda env (Python 3.12) and `pip install -r requirements.txt`.
2. Run the data step: download, verify checksums, split, save indices.
3. Run the feature step: build and cache pixel + HOG features.
4. Run each model notebook/script: tune on val, refit on 60k, evaluate once on test, write JSON.
5. Run the results notebook: aggregate into `summary.csv` and generate all comparison/error figures.
6. Compare the numbers with the README table. They match.

### Flow 2: Add a new model (later the Phase 2 CNN)
1. Load the same splits via `load_splits()` / `get_features()`.
2. Train it, then call `evaluate()`. The new JSON appears automatically in `summary.csv` and in every comparison plot.

## 7. Constraints
- Deadline: none fixed yet. Planned by phase, with no dates.
- Scale: 70k images of 28×28. Total data is about 30 MB raw and about 200 MB as float32 arrays.
- Platform: MacBook Air M3 (8 cores, 8 GB RAM), local Jupyter in VS Code. Colab is the fallback. The 8 GB RAM means parallel jobs are capped and RBF-SVM tuning uses a subset.
- Budget: free tools only.
- Compliance / privacy: public dataset (MIT licence). No personal data.

## 8. Success metrics
- All 8 experiments plus the dummy baseline appear in `summary.csv`, and every figure is regenerated by code (delivered: 12 rows incl. LinearSVC).
- Sanity ranges for pixel test accuracy: dummy ~10%, LR ~84%, kNN ~85–86%, RF ~87–88%, RBF SVM ~89–90%.
- Shirt (class 6) is expected to have the lowest recall. The error analysis explains why.
- Any team member can defend every stage at the viva.

## 9. Open questions
| # | Question | Owner | Resolved? |
|---|---|---|---|
| 1 | Who owns each of the 5 phases? | User + team | Yes: see `docs/progress.md` (phase 05 split in two, D-031) |
| 2 | Submission deadline | User | No |
| 3 | Exact report/slide requirements from the brief | User | No, parked |
