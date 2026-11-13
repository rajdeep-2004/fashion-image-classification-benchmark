# idea.md — Project 11, Phase 1 (Machine Learning)

**Project:** Automatic Clothing Category Recognition — Inventory Image Tagging via Handcrafted Features and a Small CNN
**This document covers:** Phase 1 only (classical ML). Phase 2 (CNN + hybrid) is previewed at the end.
**Team size:** 5. Phase 1 is split into 5 workstreams, one owner each.

---

## 1. The idea in one paragraph

A small online catalogue needs to auto-tag product photos (shirt, trouser, sneaker, …). We test how far **classical machine learning with hand-designed features** can go on this task before any deep learning is used. We describe each image two ways — **raw pixels** and **HOG edge features** — and train four classical models on each: Logistic Regression, k-Nearest Neighbours, Random Forest and SVM. We then measure accuracy, per-class behaviour and speed, and explain *why* every model fails on visually similar tops (shirt / T-shirt / pullover / coat). That failure analysis is the motivation for Phase 2's CNN.

## 2. Research questions

| # | Question | Answered by |
|---|----------|-------------|
| RQ1 | How accurately can classical ML classify Fashion-MNIST using raw pixels? | WS3, WS4 |
| RQ2 | Do HOG features improve over raw pixels, and for which models? | WS2, WS3, WS4 |
| RQ3 | Which classical model gives the best accuracy / speed / size trade-off? | WS5 |
| RQ4 | Which classes are systematically confused, and why do handcrafted features fail there? | WS1, WS5 |

## 3. Goals and non-goals

**Goals**
- A reproducible pipeline: same split, same seed, same metrics for every model.
- 4 models × 2 feature sets = **8 tuned experiments**, plus a trivial baseline.
- A results table, confusion matrices, misclassified examples and a written error analysis.
- A GitHub repo, an IEEE-format LaTeX report and a presentation.
- Every member can explain the full pipeline in the individual viva.

**Non-goals (Phase 1)**
- No neural networks or CNNs. These are Phase 2.
- No data augmentation. This is Phase 2.
- No new or different dataset. Fashion-MNIST meets the brief.

## 4. Dataset

| Item | Value |
|------|-------|
| Name | Fashion-MNIST (Zalando Research, Xiao et al. 2017) |
| Source | Official IDX files from github.com/zalandoresearch/fashion-mnist |
| Size | 60,000 train + 10,000 test, 28×28 grayscale, pixel values 0–255 |
| Classes (10) | 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot |
| Balance | 6,000 per class in train, 1,000 per class in test |

**Split protocol (fixed for the whole team):**
- The official 10k **test** set is locked. It is used once, for the final numbers only.
- The official 60k train set is split by **stratified sampling, `random_state=42`**, into:
  - **train** (54,000)
  - **val** (6,000)
- All hyperparameter tuning uses train/val only.
- Final models are refit on the full 60k with the best hyperparameters, then evaluated on test.
- Any scaler or PCA is **fit on train only**, then applied to val and test.

## 5. Pipeline overview

```
 raw IDX files
      │
      ▼
 [WS1] load → scale to [0,1] → stratified train/val split → EDA
      │                                 saves: data/processed/*.npy + split indices
      ▼
 [WS2] feature extraction
      ├── pixels  : flatten 28×28 → 784-d   (+ StandardScaler)
      └── hog     : skimage HOG → ~324–1296-d (tuned)
                                        saves: features/{pixels,hog}_{train,val,test}.npy
      ▼
 [WS3] Logistic Regression, kNN    ─┐
 [WS4] Random Forest, SVM          ─┼─► each uses WS5's evaluate() → results/metrics/*.json
      ▼                             │
 [WS5] evaluation harness, aggregation, confusion matrices, error analysis, final comparison
```

## 6. Repository structure

```
fashion-mnist-ml/
├── README.md                  # what, how to run, results table, team + ownership
├── requirements.txt
├── config.yaml                # seed, paths, split sizes, HOG params
├── data/
│   ├── raw/                   # IDX files (gitignored; download script provided)
│   └── processed/             # .npy arrays + split indices (gitignored)
├── features/                  # cached feature matrices (gitignored)
├── src/
│   ├── data.py                # WS1: download, load, split
│   ├── features.py            # WS2: pixel + HOG extractors
│   ├── models_linear.py       # WS3: LR + kNN training/tuning
│   ├── models_tree_kernel.py  # WS4: RF + SVM training/tuning
│   ├── evaluate.py            # WS5: metrics, timing, confusion matrix, JSON writer
│   └── utils.py               # seeding, logging, timer
├── notebooks/
│   ├── 01_eda.ipynb                 # WS1
│   ├── 02_features.ipynb            # WS2
│   ├── 03_linear_knn.ipynb          # WS3
│   ├── 04_rf_svm.ipynb              # WS4
│   └── 05_results_analysis.ipynb    # WS5
├── results/
│   ├── metrics/               # one JSON per experiment
│   ├── figures/               # all plots used in report/slides
│   └── summary.csv            # aggregated table
└── report/                    # LaTeX (IEEEtran) source
```

**Stack:**
- Python 3.11, NumPy, pandas
- scikit-learn, scikit-image
- matplotlib (+ seaborn, optional)
- joblib, PyYAML

## 7. Shared conventions (everyone follows these)

- **Seed:** `42` everywhere: NumPy, sklearn models and the split. Set it via `utils.set_seed()`.
- **Interfaces:** workstreams talk through files and functions, not copy-pasted code.
  - `data.load_splits() -> dict` with keys `X_train, y_train, X_val, y_val, X_test, y_test`. The X arrays have shape (N, 28, 28), float32, values in [0,1].
  - `features.get_features(kind, split) -> np.ndarray`, where `kind` is `"pixels"` or `"hog"`. Loads from cache, or builds and caches.
  - `evaluate.evaluate(model, X, y, name, feature_kind) -> dict`. Writes `results/metrics/{name}_{feature_kind}.json`.
- **Metrics JSON schema:**
  ```json
  {
    "model": "svm_rbf", "features": "hog", "split": "test",
    "best_params": {"C": 10, "gamma": "scale"},
    "accuracy": 0.0, "macro_f1": 0.0,
    "per_class_recall": [10 floats],
    "confusion_matrix": [[10x10 ints]],
    "train_time_s": 0.0, "inference_ms_per_1000": 0.0,
    "model_size": {"type": "params|support_vectors|nodes|stored_samples", "value": 0}
  }
  ```
- **Timing:** measure on one agreed machine (or Colab CPU), reported as ms per 1,000 test images.
- **Git:**
  - One branch per workstream (`ws1-data`, `ws2-features`, …), merged by pull request.
  - Another member reviews each pull request.
  - No direct pushes to `main`.
  - Small, descriptive commits, since commit history shows individual contribution.
- **Figures:** saved to `results/figures/` as PNG (300 dpi) with descriptive names. One plot = one file.

---

## 8. The five workstreams

Each workstream has:
- an owner
- a clear scope
- deliverables
- "done when" criteria
- a list of what its owner must be able to defend in the viva

Each owner also writes the matching part of the report's literature review and methods section.

### WS1 — Data Foundation & Exploratory Data Analysis

**One-line role:** *"Built the data pipeline and validation protocol, and ran the exploratory analysis that identified the core difficulty of the task."*

**Scope**
- Write a download script for the official IDX files and a loader into NumPy.
- Run sanity checks:
  - shapes, dtypes, label range
  - no NaNs
  - class counts
  - exact-duplicate detection (hash each image)
  - train/test overlap check
- Scale pixels to [0,1].
- Make the stratified 54k/6k train/val split and save the split indices so everyone uses identical data.
- EDA:
  - a sample grid (e.g. 10 images per class)
  - the mean image and the pixel-std image per class
  - pixel-intensity histograms
  - a 2-D projection with PCA, plus t-SNE on a ~5k subsample, coloured by class. This shows which classes overlap.
  - PCA explained-variance curve (how many components cover 90/95% of variance). WS2 uses this as an input.
  - pairwise similarity of class mean images, as a first hint of which classes will be confused.

**Deliverables**
- `src/data.py`
- `notebooks/01_eda.ipynb`
- EDA figures in `results/figures/`
- Report sections: **Dataset** and **EDA**
- Lit review: Xiao et al. 2017 (Fashion-MNIST) and how it compares to MNIST

**Done when**
- `load_splits()` returns identical arrays on every teammate's machine (verify with a checksum).
- The EDA notebook runs top-to-bottom with no errors.
- At least 6 figures are saved.
- A written paragraph names the expected hard classes.

**Viva prep**
- Why a separate validation set? Why stratify? What is data leakage?
- What does PCA do, and why does t-SNE look different from PCA?
- Why do shirt, T-shirt, pullover and coat overlap in the projection?

---

### WS2 — Feature Engineering (Pixels vs HOG)

**One-line role:** *"Designed and tuned the handcrafted feature representations (raw pixels and HOG) that every model in the study is built on."*

**Scope**

*Pixel features*
- Flatten 28×28 to 784-d.
- Fit a `StandardScaler` on train, then apply it to val and test.

*HOG features*
- Use `skimage.feature.hog`.
- Tune these parameters with a fast proxy model (Logistic Regression or LinearSVC on train → val accuracy):
  - `orientations` ∈ {6, 9, 12}
  - `pixels_per_cell` ∈ {(4,4), (7,7)}
  - `cells_per_block` ∈ {(1,1), (2,2)}
- Record feature dimensionality and extraction time for each setting.
- Choose the final HOG config and write it to `config.yaml`.
- Visualise the HOG output: original image next to its HOG image, one per class.

*Optional extras (bonus depth)*
- PCA-reduced pixels, e.g. 50–150 components at 95% variance. This also speeds up SVM and kNN.
- LBP texture features.

*Caching*
- Save every feature matrix to `features/` so WS3 and WS4 never recompute it.

**Deliverables**
- `src/features.py`
- `notebooks/02_features.ipynb`
- A HOG parameter-sweep table and HOG visualisations
- Report section: **Feature Engineering**
- Lit review: Dalal & Triggs 2005 (HOG), and handcrafted features in general

**Done when**
- `get_features("pixels"|"hog", split)` works from cache.
- A parameter-sweep table exists with dimensionality, extraction time and proxy val accuracy.
- The final config is committed.

**Viva prep**
- How HOG is computed step by step: gradients → orientation bins per cell → block normalisation.
- Why HOG might help: shape and edges are robust to brightness changes.
- Why HOG might hurt: on 28×28 images, fine texture is lost.
- Why flattening pixels throws away spatial neighbourhood information.
- Why scaling matters for LR, kNN and SVM but not for Random Forest.

---

### WS3 — Linear & Instance-Based Models (Logistic Regression, kNN)

**One-line role:** *"Implemented, tuned and theoretically analysed the linear and distance-based classifiers across both feature sets."*

**Scope**

*Baseline*
- Add a `DummyClassifier` (most-frequent) as the floor every model must beat. Expect ~10%.

*Logistic Regression*
- Multinomial (softmax), solver `saga` or `lbfgs`.
- Tune `C` ∈ {0.01, 0.1, 1, 10}.
- Compare L2 vs L1 penalty if time allows.
- Run on pixels and on HOG.

*kNN*
- Tune `k` ∈ {1, 3, 5, 7, 9}, `weights` ∈ {uniform, distance}, metric ∈ {euclidean, manhattan}.
- Run on pixels and on HOG.
- Optionally run on PCA-reduced pixels to show the curse-of-dimensionality effect.

*Final runs*
- Plot a validation curve for each tuned hyperparameter (val accuracy vs C, val accuracy vs k).
- Refit on the full 60k with the best params.
- Evaluate on test via `evaluate()`.
- Visualise the learned LR weights for each class as a 28×28 heatmap (pixels only). This shows what a linear model "looks at".

**Deliverables**
- `src/models_linear.py`
- `notebooks/03_linear_knn.ipynb`
- 4 metrics JSONs plus the dummy baseline
- Validation curves and LR weight maps
- Report subsections: **Logistic Regression** and **kNN** (theory + results)
- Lit review: classical linear and instance-based classifiers

**Done when**
- All 4 experiments are tuned on val and evaluated once on test.
- The JSONs follow the shared schema.
- Theory write-up is done.

**Viva prep**
- Logistic Regression:
  - The softmax function and cross-entropy loss.
  - What `C` does: inverse regularisation strength.
  - Why LR draws linear decision boundaries.
- kNN:
  - Why kNN has no training phase but slow prediction.
  - How k trades off bias and variance.
  - The curse of dimensionality and why distances become less meaningful in 784-d.

---

### WS4 — Tree Ensemble & Kernel Models (Random Forest, SVM)

**One-line role:** *"Implemented, tuned and theoretically analysed the ensemble and kernel classifiers, including the best-performing classical model."*

**Scope**

*Random Forest*
- Tune `n_estimators` ∈ {100, 300}, `max_depth` ∈ {None, 20, 40}, `max_features` ∈ {sqrt, log2}.
- Run on pixels and on HOG.
- Report the feature-importance heatmap (pixels → 28×28 map).

*SVM*
- Run `LinearSVC` and `SVC(kernel="rbf")`.
- Tune `C` ∈ {0.1, 1, 10} and `gamma` ∈ {scale, 0.01, 0.001}.
- Run on pixels and on HOG.
- **Compute note:** RBF SVM on 54k × 784 is slow. Tune on a stratified 10k subset of train, then refit the best config on the full 60k. Using PCA-reduced pixels from WS2 is an acceptable speed-up. Document whatever you do.
- Report the number of support vectors (model size).

*Final runs*
- Refit on 60k with the best params.
- Evaluate on test via `evaluate()`.
- Run 5-fold CV on the best config and report mean ± std, to show stability.

**Deliverables**
- `src/models_tree_kernel.py`
- `notebooks/04_rf_svm.ipynb`
- 4–6 metrics JSONs
- The RF importance map and tuning tables
- Report subsections: **Random Forest** and **SVM** (theory + results)
- Lit review: Breiman 2001 (Random Forests), Cortes & Vapnik 1995 (SVM)

**Done when**
- All experiments are tuned and evaluated.
- The subsampling strategy is documented.
- CV mean ± std is reported for the best model.

**Viva prep**
- Random Forest:
  - Bagging and random feature subsets: why many trees reduce variance.
  - Gini impurity.
  - Why RF doesn't need feature scaling.
- SVM:
  - The maximum-margin idea and hinge loss.
  - The kernel trick: what RBF does, and what `C` and `gamma` control.
  - Why SVM training scales badly with sample count.

---

### WS5 — Evaluation Framework, Error Analysis & Integration

**One-line role:** *"Designed the shared evaluation framework and metrics, led the cross-model comparison and error analysis, and integrated results into the report and repository."*

**Scope**

*Evaluation harness (build this first, by Day 2)*
- Build `src/evaluate.py`. It computes:
  - accuracy, macro F1 and per-class recall/precision
  - the confusion matrix
  - train time, and inference time per 1,000 images
  - model size
- It writes the JSON in the shared schema and is used by WS3 and WS4.
- Write a `utils.py` timer and the seeding helper.

*Aggregation*
- Collect all JSONs into `results/summary.csv`.
- Build the main results table: model × features → accuracy, macro F1, inference time, size.

*Comparison plots*
- Grouped bar chart of pixels vs HOG for each model (answers RQ2).
- Accuracy vs inference-time scatter (answers RQ3).
- Per-class recall heatmap across all models (answers RQ4).

*Error analysis*
- Normalised confusion matrix for the best model.
- List the top-5 most-confused class pairs.
- Show a grid of misclassified test images (true vs predicted) for those pairs.
- Write the explanation of *why* handcrafted features fail here. This connects to WS1's EDA overlap finding.
- Check agreement between the best two models: where do they make the same mistakes?

*Integration*
- README results table and run instructions.
- Own the LaTeX report skeleton (IEEEtran on Overleaf) and merge everyone's sections.
- Assemble the presentation deck.
- Write the "Conclusion & Motivation for Phase 2" section.

**Deliverables**
- `src/evaluate.py`
- `notebooks/05_results_analysis.ipynb`
- `results/summary.csv`
- Comparison and error-analysis figures
- The integrated report and slides
- Lit review: evaluation metrics (why macro F1 and not just accuracy) and published Fashion-MNIST classical benchmarks

**Done when**
- Every experiment appears in one table.
- All figures are generated from code, not by hand.
- The report compiles.
- The slides are ready.

**Viva prep**
- Precision vs recall vs F1, and macro vs micro averaging.
- How to read a confusion matrix.
- Why the test set is touched only once.
- What the results say about RQ1–RQ4.

---

## 9. Dependencies and timeline

WS1 and WS5 unblock everyone else, so they start first. Dates are relative, so shift them to the actual deadline once it's announced.

| Day | WS1 Data/EDA | WS2 Features | WS3 LR/kNN | WS4 RF/SVM | WS5 Eval/Integration |
|-----|--------------|--------------|------------|------------|----------------------|
| 1 | Loader + split, push `data.py` | Read HOG paper, prototype HOG on samples | Read LR/kNN theory, write tuning code skeleton | Read RF/SVM theory, write tuning skeleton | Repo setup, `evaluate.py`, report skeleton |
| 2–3 | EDA figures | Pixel features + HOG sweep | Pixel-feature runs | Pixel-feature runs (SVM on subset) | Harness review, start lit-review merge |
| 4–5 | EDA write-up | Final HOG config, cache all features | HOG-feature runs, validation curves | HOG-feature runs, CV on best | Aggregation script, comparison plots |
| 6–7 | Help WS5 with error-analysis figures | Feature write-up | Final test eval + write-up | Final test eval + write-up | Error analysis, results table |
| 8–9 | Review report | Review report | Review report | Review report | Integrate report + slides |
| 10 | **Full-team dry run of presentation and mock viva (everyone explains every stage)** | | | | |

**Critical-path rule:** WS3 and WS4 can begin with pixel features as soon as WS1 pushes `data.py`. They don't need to wait for HOG.

## 10. Expected results (sanity ranges, not targets)

These are rough figures from published Fashion-MNIST benchmarks. If a number lands far outside these ranges, suspect a bug such as a wrong split, unscaled data or leakage.

| Model | Pixels (approx. test accuracy) |
|-------|-------------------------------|
| Dummy (most frequent) | ~10% |
| Logistic Regression | ~84% |
| kNN | ~85–86% |
| Random Forest | ~87–88% |
| SVM (RBF) | ~89–90% |

The HOG results are unknown. That is what this phase finds out. Shirt (class 6) will almost certainly have the lowest recall for every model.

## 11. Report outline (IEEE, LaTeX)

| Section | Owner |
|---------|-------|
| Abstract | WS5 |
| I. Introduction (problem, motivation, RQs) | WS5 |
| II. Related Work | Each WS writes its part; WS5 merges |
| III. Dataset & EDA | WS1 |
| IV. Feature Engineering | WS2 |
| V-A. Linear & instance-based models | WS3 |
| V-B. Ensemble & kernel models | WS4 |
| VI. Experimental Setup (split, tuning, metrics, hardware) | WS1 + WS5 |
| VII. Results & Comparison | WS5 (with WS3/WS4 inputs) |
| VIII. Error Analysis | WS5 + WS1 |
| IX. Conclusion & Phase 2 Motivation | WS5 |
| References | All |
| Appendix: contribution statement (who did what) | All |

**Core references**
- Xiao, Rasul & Vollgraf (2017). Fashion-MNIST.
- LeCun et al. (1998). Gradient-Based Learning Applied to Document Recognition.
- Dalal & Triggs (2005). Histograms of Oriented Gradients for Human Detection.
- Cortes & Vapnik (1995). Support-Vector Networks.
- Breiman (2001). Random Forests.
- Cover & Hart (1967). Nearest Neighbor Pattern Classification.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.

## 12. Presentation outline (~12 slides)

1. Problem and motivation
2. Dataset
3. EDA highlights
4. Features: pixels vs HOG
5. Models overview
6. Experimental protocol
7. Results table
8. Pixels vs HOG chart
9. Accuracy vs speed
10. Confusion matrix and error examples
11. Conclusions
12. What Phase 2 will do

Each member presents their own workstream's slides.

## 13. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| SVM/kNN too slow on full data | Tune on a 10k subset, use PCA features, run on Colab. Document it. |
| Teammates get different numbers | Fixed seed, saved split indices, cached features, one timing machine |
| Test-set leakage through repeated peeking | Test is evaluated only in the final run, from a single script |
| Viva questions outside one's own workstream | Day-10 mock viva where everyone explains every stage |
| Merge conflicts in notebooks | One notebook per workstream; shared logic lives in `src/` |

## 14. Phase 1 definition of done

- [ ] 8 tuned experiments + dummy baseline, all in `results/summary.csv`
- [ ] All figures reproducible from notebooks/scripts
- [ ] README with setup, run commands, results table, team ownership
- [ ] IEEE LaTeX report compiled to PDF
- [ ] Slides ready
- [ ] Every member has meaningful commits on their branch
- [ ] Every member can answer the viva-prep questions of **all five** workstreams

---

## 15. Phase 2 preview (for context only)

Phase 2 adds a **2–3 block CNN** (conv → pool → dropout, early stopping) trained on the same split and evaluated with the **same `evaluate()` harness**, so results compare directly with Phase 1.

Planned additions:
- ablation studies (number of blocks, dropout, augmentation)
- a DL literature review
- a possible **hybrid** model for bonus marks, e.g. CNN features fed into the Phase 1 SVM, or HOG + CNN feature fusion aimed at the shirt/T-shirt/pullover confusions

Building a clean, reusable Phase 1 pipeline now makes Phase 2 much faster.
