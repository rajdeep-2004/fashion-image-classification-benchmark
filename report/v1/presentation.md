# The Story of Our Fashion-MNIST Benchmark

*A walkthrough for every team member: what we built, why, how the code works, and what to say at the viva.*

Read the **Prologue** first, then **your own part** closely, then skim the other parts so you can follow the whole story.
Each part ends with "Numbers to remember" and "Questions you may get". All numbers come from the files in `results/`.

| Part | Presenter | Enrollment No. | Topic |
|---|---|---|---|
| 1 | Tanmay Singh | 2401010476 | Data pipeline & EDA |
| 2 | Anand Mishra | 2401010065 | Feature engineering & evaluation framework |
| 3 | Rajdeep Sanyal | 2401010370 | Linear & distance-based models |
| 4 | Rajat Srivastav | 2401020101 | Tree & kernel models |
| 5 | Omved Nagre | 2401010311 | Results, error analysis & conclusion |

---

## Prologue: what we set out to do

### The question
Imagine a small online clothing shop that wants to tag product photos automatically: "shirt", "trouser", "sneaker".
Everyone reaches for deep learning, but we asked a more careful question first: **how far can classical machine
learning go, which model is the best trade-off, and where and why does it fail?** The answer becomes a baseline that our
Phase 2 neural network has to beat, and the failure analysis tells Phase 2 what to fix.

We turned that into four research questions:
- **RQ1:** How accurate is classical ML on raw pixels?
- **RQ2:** Do HOG features (edge directions) beat raw pixels, and for which models?
- **RQ3:** Which model gives the best accuracy / speed / size trade-off?
- **RQ4:** Which classes get confused, and why?

### The one rule that mattered most: no leakage
"Leakage" means information from the test set sneaking into a decision. If you pick a model because it scored well on
test, the test score is no longer an honest estimate of how it will do on new data. Examiners check this first, so we
designed the project so that leakage is structurally hard:

1. **Three splits with three jobs.** Train (54,000 images) is for fitting. Validation (6,000) is for **every** decision.
   Test (10,000) is opened **once per experiment**, at the very end.
2. **One door to the test set.** Only one function, `final_test_run()` in `src/experiment.py`, reads test data. Every
   call writes a row to `results/test_log.csv`. The final log has **12 rows for 12 experiments**: nothing was tested twice.
3. **Scaling inside pipelines.** Models that need feature scaling have a `StandardScaler` *inside* their scikit-learn
   `Pipeline`, so the scaler learns its mean and spread only from the rows the model trains on.
4. **EDA on the training split only.**
5. **Error analysis reads saved predictions.** Nobody predicts on test again to make plots.

### How the code is organised
```
src/config.py              every constant: seed 42, split size, HOG settings, paths
src/utils.py               set_seed, Timer, save_fig, save_json
src/data.py                download + checksum, sanity checks, the fixed split (load_splits)
src/features.py            pixel and HOG features, cached to disk; the HOG sweep
src/models_linear.py       make_dummy, make_logreg, make_knn  (+ their tuning grids)
src/models_tree_kernel.py  make_rf, make_linear_svm, make_rbf_svm (+ grids)
src/experiment.py          tune_on_val, final_test_run, full_train_val_score, cross_validate_best
src/evaluate.py            evaluate(): all metrics, timing, model size → one JSON per experiment
src/results.py             summary table, best-model choice, comparison and error-analysis plots
notebooks/01 … 05          one per stage; they call src/ and explain the results
results/                   metrics/*.json, summary.csv, test_log.csv, hog_sweep.csv, figures/*.png
docs/                      PRD, architecture, decisions (D-001 … D-032), phase files
```
The rule: **logic lives in `src/`, notebooks only call it and explain it.** That keeps the code testable and the same
for every model.

### The life of one experiment
Every model on every feature set goes through the same four steps:
1. `tune_on_val(...)`: fit each setting in the grid on the 54k train split, score it on validation, keep the best.
   Writes `results/metrics/{model}_{features}_val.json`.
2. Pick the setting with the highest validation accuracy.
3. `final_test_run(...)`: refit that setting on all 60,000 train + validation images, then evaluate **once** on test.
4. `evaluate(...)` records accuracy, macro F1, per-class precision/recall, the confusion matrix, the 10,000 predictions,
   training time, prediction time per 1,000 images and model size, in `{model}_{features}_test.json`.

An "experiment" is one model on one feature set, e.g. `rbf_svm_hog`. We have 6 model types × 2 feature sets = **12**.

### Words you should be comfortable with
- **Validation set:** data held back from training and used to choose settings.
- **Hyperparameter:** a setting you choose, not learn (e.g. `C`, `k`, number of trees).
- **Grid search:** try every combination in a list of settings.
- **Accuracy / macro F1:** share of correct predictions / the average F1 over the 10 classes (treats classes equally).
- **Recall of a class:** of all images of that class, the share the model got right.
- **Confusion matrix:** a 10×10 table; row = true class, column = predicted class.
- **Stratified:** keeps the same share of each class in every part.

---

## Part 1: Data pipeline & EDA (Tanmay Singh)

### The story
Before any model, we had to be sure the data was the real data, that everyone used the same split, and that we
understood what makes the task hard.

**Getting the data.** `download_raw()` in `src/data.py` downloads the four official Fashion-MNIST files over HTTPS from
Zalando's GitHub repository (D-012). Each file is checked against the MD5 checksum Zalando publishes (D-006). If a file
is corrupted, it is deleted and the run stops with a clear message. A download writes to a temporary `.part` file first,
so a crash can never leave a half-written file that looks real.

**Checking the data.** `run_sanity_checks()` confirmed: 60,000 training and 10,000 test images of 28×28 pixels, pixel
values 0–255, labels 0–9, no missing values, exactly 6,000 training and 1,000 test images per class, and **0 duplicate
images** inside train, inside test, or shared between them (found by hashing every image).

**Fixing the split.** `load_splits()` takes 6,000 images (600 per class, seed 42) from the official training set as the
validation set, leaving 54,000 for training. The chosen indices are saved to `data/splits/val_indices.npy` and **committed
to git** (D-003), so even a future version of scikit-learn can't produce a different split. Its fingerprint is
`54ff6b78b27109cdfc3ebe7204040a07`; anyone can check it with `md5 -q data/splits/val_indices.npy`. Images come back as
float32 values in [0, 1]. We don't keep a separate processed copy of the data: reading the compressed files takes under a
second (D-014).

**Exploring the data (notebook 01).** EDA looked **only at the 54k training images** (D-015):
- **Sample grid and class-mean images:** the average T-shirt, Pullover, Coat and Shirt look almost the same.
- **Similarity of class means** (cosine similarity, 1 = identical shape): Coat–Shirt **0.987**, Pullover–Coat **0.984**,
  Pullover–Shirt **0.983**, T-shirt/top–Shirt **0.950**. These are the four most similar pairs.
- **PCA:** 84 of the 784 pixel directions keep 90% of the variance, 188 keep 95%, so raw pixels are very redundant.
- **t-SNE** on a 5,000-image subsample (after reducing to 50 PCA components): Pullover, Coat and Shirt form one mixed
  cloud; Trouser and Bag form clean islands; the three shoe classes sit close but separate.

**The prediction we made before training anything:** *Shirt will be the hardest class, confused with Coat, Pullover and
T-shirt/top.* At the end of the project (Part 5) the models confirmed this exactly.

### The code you own
| Function | File | What it does |
|---|---|---|
| `download_raw()` | `src/data.py` | Downloads the 4 files if missing, verifies MD5, deletes and errors on mismatch |
| `load_raw()` | `src/data.py` | Reads the compressed IDX files into NumPy arrays (skips the 16/8-byte header) |
| `run_sanity_checks()` | `src/data.py` | Shapes, dtypes, label range, NaNs, class counts, duplicates via hashing |
| `load_splits()` | `src/data.py` | Applies the committed split; returns train/val/test as float32 in [0, 1] |
| notebook | `notebooks/01_data_eda.ipynb` | All EDA figures (`eda_*.png`, 8 in total) |

### Numbers to remember
- 70,000 images, 10 classes, 28×28 grayscale; split 54,000 / 6,000 / 10,000
- 0 duplicates, 0 missing values; split MD5 `54ff6b78…`
- PCA: 84 components → 90% variance; class-mean similarity Coat–Shirt 0.987

### Questions you may get
- **Why commit the split file instead of re-creating it with the seed?** A library update could change how the seed is
  used. A committed file guarantees the identical split forever.
- **Why do EDA on the training split only?** Anything we learn from validation or test could influence our decisions,
  which is a form of leakage.
- **What if there had been duplicates?** There were none, but our rule was to count them, not remove them, so results
  stay comparable to published benchmarks (D-004).
- **What did EDA predict, and was it right?** Shirt hardest, confused with the other tops. Yes: Shirt is the lowest-recall
  class for every trained model.

---

## Part 2: Feature engineering & evaluation framework (Anand Mishra)

### The story
A model never sees a "picture"; it sees a list of numbers. We compared two ways of turning an image into numbers, and
built the shared machinery that every model runs through.

**Feature 1: raw pixels.** Flatten the 28×28 image into 784 numbers between 0 and 1. Simple, but it throws away the
2-D layout: the model doesn't know which pixels are neighbours, and a garment shifted by two pixels becomes a very
different vector.

**Feature 2: HOG (Histogram of Oriented Gradients).** HOG describes **edges and their directions**:
1. At every pixel, measure how fast brightness changes left-right and up-down: that gives an edge strength and direction.
2. Split the image into small cells (e.g. 4×4 pixels). In each cell, build a histogram of edge directions (`orientations`
   bins), each pixel voting with its edge strength.
3. Group neighbouring cells into blocks (e.g. 2×2) and normalise them together, so lighting and contrast matter less.
4. Join all block histograms into one long vector.
Length = ((28 / cell) − block + 1)² × block² × orientations. For our setting: 6² × 4 × 12 = **1,728 numbers**.

**Choosing the HOG settings on validation (notebook 02).** We tried 3 × 2 × 2 = **12** settings: orientations {6, 9, 12},
cell size {4×4, 7×7}, block size {1×1, 2×2}. Each was scored by a quick **proxy model** (a linear SVM trained with
stochastic gradient descent, fixed 30 passes, D-016) fitted on train features and scored on validation. The proxy only
ranks the settings; it is not one of our study's models. What we learned:
- **Small cells win by a lot** (4×4 beats 7×7 by 4–8 points): a 7×7 cell is a quarter of the image and blurs the outline.
- **Block normalisation helps** (+2–3 points). **More orientation bins help a little** (12 > 9 > 6).
- The rule, written down before looking: *highest validation accuracy; within 0.3 points prefer the smaller vector.*
  Winner: **12 orientations, 4×4 cells, 2×2 blocks, 1,728-d, 0.8818**. Runner-up 9 orientations (1,296-d) scored 0.8783,
  0.35 points lower, so the winner stands (D-017).
- Extraction costs about **0.22 s per 1,000 images**. This matters later for speed comparisons.

**Caching.** `get_features(kind, split)` builds features once and saves them to `features/` (not committed). The HOG
settings are part of the file name (`hog_o12_p4_c2_train.npy`), so changing the settings can never silently reuse old
features (D-007). **Test features are never built in notebook 02**: they are created only inside `final_test_run()`,
so the rule "test is read in exactly one place" stays absolute and easy to audit (D-019).

**Scaling: who needs it and why.** Logistic Regression and SVMs are trained with a penalty that treats every weight
equally, and k-NN measures distances, so features with a large spread would dominate. These models get a
`StandardScaler` (subtract mean, divide by standard deviation) **inside their pipeline**. Random Forest splits on one
feature at a time ("is pixel 300 > 0.4?"), and rescaling doesn't change which split is best, so it needs no scaler.

**The evaluation framework (`src/experiment.py`, `src/evaluate.py`).** This is the machinery all models share:
- `tune_on_val(name, make_fn, grid, kind, train_subset=None)` loads only train and val features, fits every grid
  combination, scores it on validation and returns the best. A scratch check that made any test read crash proved it never
  reads test (D-018). Ties go to the first combination in grid order.
- `final_test_run(name, make_fn, params, kind)` refits on all 60,000 images, loads test features (the only place that does),
  calls `evaluate()` and appends a row to `test_log.csv`. If an experiment was already tested, it prints a **warning**
  instead of hiding the repeat.
- `evaluate(...)` computes everything and writes the test JSON. Prediction time is the **best of 3 runs** per 1,000 images;
  model size is the number of bytes when the model is pickled (D-008), one comparable number for every model type. It also
  saves the 10,000 predictions (`y_pred`), so later error analysis never has to predict on test again (D-011).
- `cross_validate_best(...)` runs a 5-fold stratified cross-validation of the whole pipeline (so scaling is redone in every
  fold), one fold at a time to stay within 8 GB of RAM.

### The code you own
| Function | File | What it does |
|---|---|---|
| `extract_pixels`, `extract_hog` | `src/features.py` | Flatten / HOG with scikit-image (L2-Hys block normalisation) |
| `get_features`, `get_labels` | `src/features.py` | Cached features and labels for train / val / trainval / test |
| `run_hog_sweep` | `src/features.py` | The 12-setting sweep → `results/hog_sweep.csv` |
| `tune_on_val`, `final_test_run`, `cross_validate_best`, `stratified_subset` | `src/experiment.py` | The shared tuning / testing / CV machinery |
| `evaluate`, `model_size_bytes`, `environment` | `src/evaluate.py` | Metrics, timing, size, machine info → JSON |
| notebook | `notebooks/02_features.ipynb` | HOG explanation, sweep, HOG pictures, scaling explanation |

### Numbers to remember
- Pixels: 784 features. HOG: 1,728 features (12 orientations, 4×4 cells, 2×2 blocks), proxy val accuracy 0.8818
- 12 HOG settings tried; small cells +4–8 points; HOG extraction ≈ 0.22 s per 1,000 images
- Every test result is one JSON; the audit log has exactly 12 rows

### Questions you may get
- **Why HOG and not something else?** It's the classic handcrafted shape descriptor: it captures edges and ignores
  brightness, which is what distinguishes many garments. LBP and PCA features were parked as extra ideas.
- **Why a proxy model for the sweep?** Tuning every real model for each of 12 HOG settings would multiply the cost by 12.
  A fast linear model is enough to rank the settings.
- **How can you prove tuning never touched test?** `tune_on_val` only calls `get_features(..., "train"/"val")`, and a check
  that made any test read crash ran fine.
- **Why measure model size by pickling?** It gives one comparable number for very different models (trees, support
  vectors, stored images, coefficients).

---

## Part 3: Linear & distance-based models (Rajdeep Sanyal)

### The story
With data and features ready, the first models went through the framework: a dummy baseline, Logistic Regression and
k-Nearest Neighbours, each on pixels and on HOG (notebook 03).

**Dummy baseline.** Always predicts the most common training class. Because every class is equally common, it scores
**10.00%**: the floor any real model must clear.

**Logistic Regression.** For each class it computes a score (a weighted sum of the features plus a bias); softmax turns the
ten scores into probabilities. Training minimises cross-entropy plus a penalty on large weights. **C** controls that
penalty: smaller C = stronger penalty = simpler model.
- Grid: C ∈ {0.01, 0.1, 1, 10}. Validation accuracy rose steadily as C got smaller, and **C = 0.01 won** on both feature
  sets (pixels 0.8642, HOG 0.8927).
- Some large-C fits on pixels didn't converge in 1,000 iterations, so the limit was raised to **5,000** (D-020).
- Test: **84.72% on pixels → 88.86% on HOG (+4.14 points).** HOG gives a linear model the shape information it cannot
  build itself.
- **Weight maps** (D-022): each class's 784 weights drawn as a 28×28 picture. Trouser has a strong negative strip between
  the legs (fabric there means "not a trouser"); shoes put weight on the sole and ankle area; the four tops share broad
  torso and sleeve regions, which is why a linear model struggles with them.

**k-Nearest Neighbours.** k-NN "trains" by storing all the training images; to predict, it finds the k closest stored
images and lets them vote.
- Grid: k ∈ {1, 3, 5, 7, 9} × uniform/distance-weighted votes × Euclidean/Manhattan distance = 20 combinations per feature set.
- Best: **k = 3, distance-weighted**, with Manhattan distance on pixels (val 0.8747) and Euclidean on HOG (val 0.8728).
- Test: **86.11% on pixels, 86.31% on HOG**: HOG barely helps, because k-NN already compares whole images.
- Cheap to fit (under 2 s) but slow to predict (**0.7–6.9 s per 1,000 images**) and large (it stores all 60,000
  images: **189–415 MB**).

**Two honest problems and how we handled them.**
- *Overlapping runs (D-023, D-024).* The final Logistic Regression test runs happened while other validation work was
  still running on the same machine, so their **timings** are unreliable. Accuracy is unaffected. We kept the raw numbers
  but marked the two results `timing_comparable = false`, and they are excluded from every speed comparison. This led to a
  project rule: finish all tuning for a model before running its test.
- *Re-running the notebook safely (D-021).* Notebook 03 reuses saved results instead of testing again; if a saved test
  result doesn't match the chosen settings, it stops with an error rather than re-testing silently.

### The code you own
| Function | File | What it does |
|---|---|---|
| `make_dummy()` | `src/models_linear.py` | Most-frequent-class baseline |
| `make_logreg(C)` | `src/models_linear.py` | StandardScaler → LogisticRegression (max_iter 5,000) |
| `make_knn(n_neighbors, weights, metric)` | `src/models_linear.py` | StandardScaler → KNeighborsClassifier |
| `LOGREG_GRID`, `KNN_GRID` | `src/models_linear.py` | The tuning grids |
| notebook | `notebooks/03_linear_knn.ipynb` | Tuning, validation curves, tests, weight maps |

### Numbers to remember
- Dummy 10.00%. Logistic Regression 84.72% → 88.86% (+4.14), best C = 0.01 (smallest tried)
- k-NN 86.11% → 86.31%, best k = 3 distance-weighted; 0.7–6.9 s per 1,000 predictions; 189–415 MB
- Logistic Regression HOG was 3rd best by validation overall (0.8927)

### Questions you may get
- **The best C is the smallest you tried. Isn't that a problem?** It's a stated limitation: the optimum might be lower. We
  didn't widen the grid afterwards because that would mean re-tuning and testing again.
- **Why is k-NN so big and slow?** It stores every training image and compares each new image with all of them.
- **Why are Logistic Regression timings missing from the speed chart?** Those runs overlapped other work, so the times are
  not comparable. Accuracy is still valid (D-024).
- **What do the weight maps show?** Which pixel regions push a class's score up or down, e.g. the gap between trouser legs.

---

## Part 4: Tree & kernel models (Rajat Srivastav)

### The story
Next came the more powerful models: Random Forest, a linear SVM and an RBF (kernel) SVM (notebook 04). This part also
contains the most important fairness fix of the project.

**Random Forest.** Many decision trees, each trained on a random resample of the data and a random subset of features at
each split; the forest votes.
- Grid: trees {100, 300} × max depth {none, 20, 40} × features per split {√p, log₂ p} = 12 combinations.
- Best: **300 trees, √p features**; depth 20 on pixels (val 0.8867), unlimited depth on HOG (val 0.8735).
- Test: **87.56% on pixels, 87.03% on HOG.** HOG does **not** help. A plausible reason: each split looks at one feature,
  and one raw pixel ("is there fabric here?") is more informative on its own than one HOG value (one edge direction in one
  small block). We did not test this directly.
- No scaler needed. Fast to predict (**12–19 ms per 1,000**) but large (**366–501 MB**, 2.5–3.5 million tree nodes).
- **Importance map:** the pixels the forest relies on most form a vertical strip in the lower centre (the gap between
  trouser legs), plus the shoulder and neckline area; the empty border is unused.

**Linear SVM.** Finds the straight boundary with the widest margin between classes.
- Grid: C ∈ {0.01, 0.1, 1}. **C = 0.01 won** on both (val 0.8583 pixels, 0.8900 HOG).
- Test: **83.78% → 88.98% on HOG (+5.20 points, the biggest HOG gain of all).** Tiny model: **181 KB**.
- Slow to tune: each fit on 54k images took **105–924 s**, growing sharply with C. We let scikit-learn pick the faster
  solver form automatically (`dual="auto"`, D-026).

**RBF SVM.** Uses a kernel to draw curved boundaries: it compares a new image with stored "support vectors" using
similarity exp(−γ·distance²).
- Grid: C ∈ {0.1, 1, 10} × γ ∈ {scale, 0.01, 0.001}. Training time grows faster than linearly with data, so the grid was
  searched on a **10,000-image stratified subset** (1,000 per class), then the winner was refit on all 60,000 (D-005, D-027).
- γ = 0.01 is far too narrow for this many standardised features: on HOG, validation accuracy collapsed to 11–29%.
- Best: **C = 10** (largest tried) with γ = 0.001 on pixels and γ = scale on HOG.
- Test: **89.70% on pixels, 91.08% on HOG**, the best results of the project. Final 60k fits took 196 s and 451 s.
- Slow to predict: each image is compared with every support vector (**19,710** on pixels, **26,343** on HOG), so about
  **4.8 s and 28.9 s per 1,000 images**. The HOG number looks about 2× slower than expected, so we treat it as an upper bound (D-028).

**The fairness fix (D-029).** When we lined up validation scores to pick the overall best model, we noticed a problem:
the RBF SVM's validation scores (0.8793 pixels, 0.8867 HOG) came from models trained on **10,000** images, while every
other model's came from **54,000**. That made the RBF SVM look worse than it really is, and the rule would have picked
Logistic Regression for the wrong reason. Instead of changing the rule, we fixed the comparison, and **wrote the plan down
before running it**: refit the *already chosen* RBF settings on the full 54k training split and score them on the same
validation set. No new search, no test data. Result: **0.9082 (pixels) and 0.9157 (HOG)**. The fair ranking became
RBF HOG 0.9157 > RBF pixels 0.9082 > Logistic Regression HOG 0.8927 > Linear SVM HOG 0.8900.

### The code you own
| Function | File | What it does |
|---|---|---|
| `make_rf(...)` | `src/models_tree_kernel.py` | RandomForestClassifier (no scaler) |
| `make_linear_svm(C)` | `src/models_tree_kernel.py` | StandardScaler → LinearSVC (dual="auto") |
| `make_rbf_svm(C, gamma)` | `src/models_tree_kernel.py` | StandardScaler → SVC(kernel="rbf") |
| `RF_GRID`, `LINEAR_SVM_GRID`, `RBF_SVM_GRID` | `src/models_tree_kernel.py` | The tuning grids |
| `full_train_val_score(...)` | `src/experiment.py` | The D-029 fairness refit (never touches test) |
| notebook | `notebooks/04_rf_svm.ipynb` | Tuning, curves, tests, RF importance map, runtime notes, D-029 section |

### Numbers to remember
- Random Forest 87.56% / 87.03%; Linear SVM 83.78% → 88.98% (+5.20); RBF SVM 89.70% / **91.08%**
- RBF tuned on 10k, refit on 60k; fair validation scores 0.9082 / **0.9157**
- RBF HOG: 26,343 support vectors, ~28.9 s per 1,000 predictions (upper bound)

### Questions you may get
- **Why tune the RBF SVM on a subset?** Its training cost grows roughly with the square of the data. One 60k fit already
  took 451 s; a full grid on 54k would take hours on a laptop.
- **Wasn't the D-029 refit a way to make your favourite model win?** No. The settings were already chosen, the plan was
  written down before the run, and test data was never used. It only made the comparison equal (54k for everyone).
- **Why doesn't Random Forest need scaling?** Splits are thresholds on single features; rescaling doesn't change them.
- **Why is the RBF SVM so slow?** Every prediction is compared with tens of thousands of support vectors in 1,728 dimensions.

---

## Part 5: Results, error analysis & conclusion (Omved Nagre)

### The story
With all 12 experiments tested exactly once, the last stage put everything side by side, chose the best model honestly,
and asked *why* the models fail (notebook 05, `src/results.py`).

**The summary table.** `build_summary()` reads all 12 test JSONs and writes `results/summary.csv`. The validation column
uses the fair score (the 54k refit for the RBF SVMs). The two Logistic Regression timings are left blank, not zero.

| Model | Pixels test | HOG test |
|---|---|---|
| Dummy | 10.00% | 10.00% |
| Logistic Regression | 84.72% | 88.86% |
| k-NN | 86.11% | 86.31% |
| Random Forest | 87.56% | 87.03% |
| Linear SVM | 83.78% | 88.98% |
| RBF SVM | 89.70% | **91.08%** |

**Choosing the best model.** `pick_best_experiment()` returns the highest **validation** score: **`rbf_svm_hog` (0.9157)**.
Its test accuracy (91.08%, macro F1 0.9108) is reported but played no part in the choice. Validation on 6,000 images has
about ±0.4 points of noise, so the 0.75-point lead over the pixel RBF SVM is real but modest.

**Stability check.** 5-fold stratified cross-validation of the best model on a 10,000-image subset: **0.8775 ± 0.0054**
(folds 0.8765, 0.8860, 0.8715, 0.8725, 0.8810). The small spread means the model is stable. The mean is lower than
validation because each fold trains on only 8,000 images.

**Answering the research questions.**
- **RQ1 (pixels):** best is the RBF SVM at **89.70%**; linear models manage only 83.78–84.72%.
- **RQ2 (HOG):** big gains for linear models (+5.20, +4.14), +1.38 for the RBF SVM, nothing for Random Forest (−0.53) or
  k-NN (+0.20).
- **RQ3 (trade-off):** speed must include feature extraction (~216 ms per 1,000 images for HOG, D-032). Then:
  fastest above 87% = **Random Forest on pixels** (~12 ms, but 366 MB); best accuracy for its size = **Linear SVM on HOG**
  (88.98%, 181 KB, ~219 ms); most accurate = **RBF SVM on HOG** (91.08%, ~29 s, about 2,400× slower than the forest for
  +3.5 points). k-NN is never the best choice: slower and less accurate than the forest.
- **RQ4 (errors):** see below.

**Error analysis.** Everything here reads saved predictions; nothing is predicted again (D-011, D-032).
- **Per-class recall:** Shirt is the hardest class for **all 10 trained models** (50.9% to 75.2%), then Pullover and Coat.
- **Top confusions of the best model** (out of 1,000 test images per class): Shirt → T-shirt/top **104**,
  T-shirt/top → Shirt **91**, Pullover → Shirt **66**, Pullover → Coat **61**, Shirt → Coat **59**.
- Mistakes among T-shirt/top, Pullover, Coat and Shirt are **56–66% of every model's errors**: exactly the group EDA
  flagged in Part 1 before any model existed.
- **Misclassified images:** grids of the first 16 test images for each top-5 pair (in test order, not hand-picked). Many
  "shirts" called T-shirts are short-sleeved, collarless or patterned; at 28×28 a collar or button row is a few pixels.
- **Do the two best models make the same mistakes?** RBF SVM on HOG vs on pixels: 8,593 images right for both, **515 wrong
  for both**, 377 wrong only with HOG, 515 wrong only with pixels. HOG fixes 138 more images than it breaks, exactly the
  1.38-point gap. **391 of the 515 shared mistakes have the same wrong label** and 397 are upper-body garments: when two
  very different models make the same mistake, the image itself is probably ambiguous.

**Limitations we state openly.**
- Best C at the edge of the grid for three models (we didn't widen grids afterwards, to avoid extra test runs).
- Validation is above test for all 10 trained models (by 0.02–2.05 points): choosing the best setting on validation
  flatters it slightly. Test is the honest number.
- Timings come from two laptops (k-NN on an Apple M3, RF and SVMs on an Apple M3 Pro), so only big gaps are firm.
- CV used a 10k subset, so it measures stability, not final accuracy.

**What's next (Phase 2).** A small convolutional network on the **same split** with the **same `evaluate()`**, so it
drops straight into our table. Hypothesis: learned filters can catch collars, buttons and cuffs and fix the
Shirt / T-shirt / Pullover / Coat confusions, beating 91.08%.

### The code you own
| Function | File | What it does |
|---|---|---|
| `load_all_metrics`, `comparable_val_acc`, `pick_best_experiment`, `build_summary` | `src/results.py` | Summary table and validation-only best-model choice |
| `plot_pixels_vs_hog`, `plot_acc_vs_inference`, `plot_recall_heatmap` | `src/results.py` | RQ2 / RQ3 / RQ4 charts |
| `plot_confusion`, `top_confused_pairs`, `plot_misclassified`, `model_agreement` | `src/results.py` | Error analysis from saved predictions |
| notebook | `notebooks/05_results_analysis.ipynb` | All results, answers to RQ1–RQ4, Phase 2 motivation |

### Numbers to remember
- Best: RBF SVM on HOG, val 0.9157, **test 91.08%**, CV 0.8775 ± 0.0054
- Shirt recall 50.9–75.2%; top confusion Shirt → T-shirt/top (104)
- Upper-body mistakes = 56–66% of errors; 515 shared mistakes, 391 with the same wrong label

### Questions you may get
- **Why is validation higher than test for every model?** Selection bias: we picked each model's best setting on
  validation, which slightly flatters its validation score. The ranking barely changes.
- **Why is CV lower than validation?** Each CV fold trains on 8,000 images, not 54,000.
- **Is the most accurate model the one you'd deploy?** Depends on the need: for speed, Random Forest on pixels; for a tiny
  model, Linear SVM on HOG; for accuracy, RBF SVM on HOG if 29 s per 1,000 images is acceptable.
- **How do you know the error analysis didn't use test again?** It reads the saved `y_pred` and confusion matrices; the
  audit log still has exactly 12 rows.

---

## Epilogue: how it all fits together

1. **Part 1** gave everyone the same verified data and split, and predicted where the difficulty would be.
2. **Part 2** turned images into two kinds of numbers and built one honest pipeline every model runs through.
3. **Parts 3 and 4** tuned each model on validation and tested it exactly once, and fixed an unfair comparison before it
   could mislead us.
4. **Part 5** chose the best model on validation only, answered the four questions, and showed the failures match the EDA.

**What makes this project trustworthy:** a committed split; one door to the test set and a 12-row audit log; scaling
inside pipelines; every choice on validation; a decision log (`docs/decisions.md`, D-001 to D-032) explaining every
choice; and a fresh clone reproduces every result file byte-for-byte in under two minutes.

### Full results (from `results/summary.csv`)
| Experiment | Val acc | Test acc | Macro F1 | Inference ms / 1,000 | Size |
|---|---|---|---|---|---|
| dummy_pixels | 0.1000 | 0.1000 | 0.0182 | ~0 | 709 B |
| dummy_hog | 0.1000 | 0.1000 | 0.0182 | ~0 | 709 B |
| logreg_pixels | 0.8642 | 0.8472 | 0.8464 | n/a | 51.4 KB |
| logreg_hog | 0.8927 | 0.8886 | 0.8885 | n/a | 111.8 KB |
| knn_pixels | 0.8747 | 0.8611 | 0.8614 | 6,926.61 | 188.7 MB |
| knn_hog | 0.8728 | 0.8631 | 0.8623 | 731.13 | 415.2 MB |
| rf_pixels | 0.8867 | 0.8756 | 0.8741 | 12.16 | 366.5 MB |
| rf_hog | 0.8735 | 0.8703 | 0.8692 | 19.02 | 500.7 MB |
| linear_svm_pixels | 0.8583 | 0.8378 | 0.8353 | 1.22 | 82.7 KB |
| linear_svm_hog | 0.8900 | 0.8898 | 0.8892 | 2.62 | 180.8 KB |
| rbf_svm_pixels | 0.9082 | 0.8970 | 0.8967 | 4,766.94 | 125.1 MB |
| **rbf_svm_hog** | **0.9157** | **0.9108** | **0.9108** | 28,876.96 | 366.2 MB |

### Where to look in the repo
| If you want to see… | Open |
|---|---|
| The data checks and EDA | `notebooks/01_data_eda.ipynb` |
| HOG and the sweep | `notebooks/02_features.ipynb`, `results/hog_sweep.csv` |
| Logistic Regression and k-NN | `notebooks/03_linear_knn.ipynb` |
| Random Forest and SVMs | `notebooks/04_rf_svm.ipynb` |
| Final results and error analysis | `notebooks/05_results_analysis.ipynb`, `results/summary.csv` |
| Why any choice was made | `docs/decisions.md` (search for the D-number) |
| Every test-set use | `results/test_log.csv` (12 rows) |
