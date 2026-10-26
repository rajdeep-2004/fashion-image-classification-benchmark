# Fashion-MNIST Classical ML Benchmark

A leak-free, reproducible benchmark comparing classical machine learning algorithms on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) clothing classification dataset. We evaluate four core model families across two distinct feature representations (raw pixels vs. Histogram of Oriented Gradients), evaluate performance along accuracy, latency, and memory dimensions, and conduct comprehensive error analysis to motivate a Phase 2 Convolutional Neural Network (CNN).

> **Contributors:** Built collaboratively across five sequential phases.
> - **Phase 01:** Foundation & Data (Rajdeep)
> - **Phase 02:** Features & Evaluation Harness (Rajdeep)
> - **Phase 03:** Dummy, Logistic Regression & k-NN (Rajat Srivastav)
> - **Phase 04:** Random Forest & Support Vector Machines (Tanmay Singh)
> - **Phase 05:** Results Aggregation, Plots & CV (Anand Mishra, Part A) | Error Analysis, Synthesis & README (Omved Nagre, Part B)
>
> Developer workflow: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Architecture rules & leakage controls: [`AGENTS.md`](AGENTS.md) · Design specs: [`docs/`](docs/)

---

## 1. Dataset & Fixed Data Splits

Fashion-MNIST consists of 70,000 28×28 grayscale images spanning 10 balanced clothing categories (6,000 train / 1,000 test per class):
*0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot.*

### Data Splits
To ensure complete reproducibility and prevent data leakage, the dataset is partitioned as follows:
- **Training split (54,000 images):** Used for model fitting during hyperparameter tuning.
- **Validation split (6,000 images):** Stratified holdout (600 per class, random seed 42) carved from the official 60k training set. Committed to version control at [`data/splits/val_indices.npy`](data/splits/val_indices.npy) to ensure every team member and experiment uses an identical validation partition.
- **Test split (10,000 images):** The official benchmark test set. **Evaluated strictly once per model** via `src.experiment.final_test_run()` after hyperparameter selection is complete.

Verify split checksum:
```bash
md5 -q data/splits/val_indices.npy        # macOS  (Linux/Colab: md5sum data/splits/val_indices.npy)
# expected: 54ff6b78b27109cdfc3ebe7204040a07
```

---

## 2. Environment Setup

### Option A: Local Installation (Recommended)
Requires Python 3.12, conda (or virtualenv), and VS Code.
```bash
git clone https://github.com/rajdeep-2004/fashion-image-classification-benchmark.git
cd fashion-image-classification-benchmark

# Create and activate conda environment
conda create -n fmnist python=3.12 -y
conda activate fmnist

# Install pinned dependencies
pip install -r requirements.txt
```

*Without conda:*
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code:
1. Press `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** → select `fmnist` (or `.venv`).
2. When opening any notebook in `notebooks/`, click **Select Kernel** (top right) and pick the `fmnist` interpreter.

### Option B: Google Colab (Fallback)
Run this in the first cell of a Colab notebook:
```python
!git clone https://github.com/rajdeep-2004/fashion-image-classification-benchmark.git
%cd fashion-image-classification-benchmark
!pip install -q -r requirements.txt
```

### Download and Verify Raw Dataset
Download the four official IDX files and automatically verify their published Zalando MD5 checksums:
```bash
python -c "from src.data import download_raw; download_raw()"
```
Raw files are saved in `data/raw/` (gitignored). If a checksum fails, the bad file is deleted and an error tells you to download again.

---

## 3. Benchmark Execution Order

Open each notebook in `notebooks/` and run **Restart & Run All**, in order 01 → 05. Later notebooks use code and results
from earlier ones.

The repo ships with every tuning and test result (`results/metrics/*.json`), so notebooks 03–05 **reuse** them instead of
re-tuning or re-testing, and the test set is never touched again. Delete a result JSON only if you really mean to redo that
experiment (a repeated test run must be explained in `docs/decisions.md`).

| Step | Notebook | What it does | ⏱ With the committed results (measured, fresh clone, Apple M3) | ⏱ From scratch (no result JSONs) |
|---|---|---|---|---|
| 1 | `01_data_eda.ipynb` | Data checks, class balance, sample grids, mean images, PCA / t-SNE views (train split only) | ~15 s | same |
| 2 | `02_features.ipynb` | Pixel and HOG features (cached in `features/`), HOG settings sweep (12 configs, linear proxy) | ~20 s | a few min |
| 3 | `03_linear_knn.ipynb` | Dummy, Logistic Regression, kNN: tune on val, one test run each | ~20 s | tens of minutes (kNN grid) |
| 4 | `04_rf_svm.ipynb` | Random Forest, LinearSVC, RBF SVM (tuned on a 10k subset, refit on 60k) | ~40 s | ~2 h (LinearSVC tuning ~45 min, final fits ~45 min) |
| 5 | `05_results_analysis.ipynb` | `summary.csv`, comparison plots, 5-fold CV, error analysis, RQ answers | ~10 s | ~2 min (CV) |

Total with the committed results: under 2 minutes, and every file in `results/` comes out identical.

*Hardware note:* the saved timings come from an Apple M3 (phase 03, except the dummy) and an Apple M3 Pro (phase 04). Runtimes vary with CPU cores.

---

## 4. Benchmark Results

Aggregated from all 12 completed experiments stored in `results/summary.csv`:

| Experiment | Model | Features | Val Acc | Test Acc | Macro F1 | Train Time (s) | Inference (ms/1k) | Model Size |
|---|---|---|---|---|---|---|---|---|
| `dummy_pixels` | Dummy | Pixels | 0.1000 | 0.1000 | 0.0182 | 0.00 | 0.00* | 709 B |
| `dummy_hog` | Dummy | HOG | 0.1000 | 0.1000 | 0.0182 | 0.16 | 0.01* | 709 B |
| `logreg_pixels` | Logistic Regression | Pixels | 0.8642 | 0.8472 | 0.8464 | NaN† | NaN† | 51.4 KB |
| `logreg_hog` | Logistic Regression | HOG | 0.8927 | 0.8886 | 0.8885 | NaN† | NaN† | 111.8 KB |
| `knn_pixels` | k-NN | Pixels | 0.8747 | 0.8611 | 0.8614 | 0.49 | 6,926.61 | 188.7 MB |
| `knn_hog` | k-NN | HOG | 0.8728 | 0.8631 | 0.8623 | 1.29 | 731.13 | 415.2 MB |
| `rf_pixels` | Random Forest | Pixels | 0.8867 | 0.8756 | 0.8741 | 32.83 | 12.16 | 366.5 MB |
| `rf_hog` | Random Forest | HOG | 0.8735 | 0.8703 | 0.8692 | 68.53 | 19.02 | 500.7 MB |
| `linear_svm_pixels` | Linear SVM | Pixels | 0.8583 | 0.8378 | 0.8353 | 99.24 | 1.22 | 82.7 KB |
| `linear_svm_hog` | Linear SVM | HOG | 0.8900 | 0.8898 | 0.8892 | 162.04 | 2.62 | 180.8 KB |
| `rbf_svm_pixels` | RBF SVM | Pixels | 0.9082 | 0.8970 | 0.8967 | 195.67 | 4,766.94 | 125.1 MB |
| `rbf_svm_hog` | RBF SVM | HOG | **0.9157** | **0.9108** | **0.9108** | 451.43 | 28,876.96‡ | 366.2 MB |

*Footnotes on benchmark measurements:*
- † **Logistic Regression timings:** Preserved as `NaN` (per D-024) because their test runs overlapped validation jobs causing CPU contention. Predictive metrics (accuracy, F1, recall) are fully valid and unaffected.
- \* **Dummy baseline timings:** ~0 ms, measured on a different machine (`env.cpu = arm64`); not used in speed comparisons.
- ‡ **RBF SVM HOG inference latency:** slow because each prediction is compared with all 26,343 support vectors in 1,728 dimensions; it also looks ~2× slower than expected, so treat 28.9 s as an upper bound (D-028).
- Inference times are model-only. HOG models also need HOG extraction: ~216 ms per 1,000 images (`results/hog_sweep.csv`).
- Model size is the pickled model in bytes (KB/MB = 1,000/1,000,000 bytes).

---

## 5. Key Findings & Synthesis

### Best Model Selection
The overall best model is **`rbf_svm_hog`**, selected **strictly using comparable validation accuracy** (0.9157 on 54k train, D-029). Its test result, reported for information only, is **91.08% accuracy** and **0.9108 macro F1**. Val accuracy has about ±0.4 points of noise (6,000 images). Stratified 5-fold cross-validation on `trainval` yielded a stable **0.8775 ± 0.0054** (the CV score is lower because each fold trains on 8,000 images rather than the full 54,000).

### Core Conclusions
1. **RQ1 (Raw Pixels Ceiling):** Classical models on raw pixels max out at **89.70%** (RBF SVM). Linear models reach only 83.8%–84.7%: a pixel only says how bright one spot is, so a straight boundary can't capture shape. kNN reaches 86.1% (k=3, Manhattan) and Random Forest 87.6%.
2. **RQ2 (Impact of HOG):** HOG boosts the linear models (**+5.20 points for LinearSVC**, **+4.14 for Logistic Regression**) and pushes RBF SVM to 91.08% (+1.38). It does not help Random Forest (**−0.53**) or kNN (+0.20); a plausible reason is that one raw pixel is more informative on its own than one HOG bin, which is what a tree split looks at.
3. **RQ3 (Speed vs accuracy vs size):** counting HOG extraction (~216 ms per 1,000 images), end-to-end per 1,000 images:
   - **Fastest above 87%:** `rf_pixels`, ~12 ms, 87.56%, but a 366 MB model.
   - **Best accuracy for its size:** `linear_svm_hog`, ~219 ms (almost all HOG extraction), 88.98%, 181 KB.
   - **Most accurate:** `rbf_svm_hog`, 91.08%, ~29 s (upper bound), too slow for high-volume use without kernel approximations.
4. **RQ4 (Systematic Error Modes):**
   - Errors are heavily concentrated in the upper-body clothing cluster: **Shirt (Class 6), T-shirt/top (Class 0), Pullover (Class 2), and Coat (Class 4)**.
   - **Shirt is the hardest class** for all 10 trained models (recall 50.9% to 75.2%). Mistakes among these four classes are 56–66% of each model's errors.
   - The top two confused directed pairs are **Shirt → T-shirt/top** (10.4% error rate) and **T-shirt/top → Shirt** (9.1% error rate). At 28×28 grayscale, collars, buttons and cuffs cover only a few pixels, and the outlines of these garments are very similar.
   - The two RBF SVMs (HOG and pixels) both get 515 test images wrong, and 391 of those with the *same* wrong label: many of these images are ambiguous in the data itself.

---

## 6. Scientific Rigor & No-Leakage Policy

The rules every phase followed (`AGENTS.md` §4):
- **No tuning on test:** hyperparameters, HOG settings and the best model are chosen on the validation set only. The test set is used once per experiment; error analysis reads the saved predictions.
- **Pipeline-Encapsulated Scaling:** `StandardScaler` lives strictly inside scikit-learn `Pipeline` objects. Feature scalers never see validation or test distributions.
- **Append-Only Test Audit Trail:** Every execution of `final_test_run()` appends to `results/test_log.csv`. It has exactly 12 rows: one test run per experiment.
- **Reproducible Figures:** All figures in `results/figures/` are generated directly from code and tracked in version control.

---

## 7. Future Work: Capstone Phase 2 (CNN)

Our best classical model reaches 91.08% test accuracy. The primary bottleneck is the representational limit of fixed, handcrafted spatial extractors (HOG) on overlapping garment silhouettes (Shirt vs. T-shirt vs. Coat).

**Capstone Phase 2** will introduce deep learning architectures using the identical data split and `evaluate()` harness:
1. **Hierarchical Representation Learning:** Convolutional layers will learn localized feature detectors directly from data, automatically progressing from low-level edges to mid-level collar/button textures and high-level garment compositions.
2. **Translation Equivariance:** Weight sharing in convolutional filters will detect fine garment attributes regardless of slight posture or centering shifts.
3. **Targeted error resolution (hypothesis to test):** a small CNN (2–3 conv blocks, batch norm, dropout) on the same split should beat 91.08%, mainly by fixing the upper-body confusions (397 of the 515 images both RBF SVMs get wrong are T-shirt/Pullover/Coat/Shirt).

---

## 8. Repository Layout

```
fashion-image-classification-benchmark/
├── src/                    # Reusable pipeline modules
│   ├── config.py           # Constants, seeds, paths, class names, HOG params
│   ├── utils.py            # Timer, set_seed, save_fig, save_json
│   ├── data.py             # Download, MD5 verification, load_splits
│   ├── features.py         # Pixel flattening, HOG extraction & sweep
│   ├── models_linear.py    # Dummy, Logistic Regression, k-NN Pipelines
│   ├── models_tree_kernel.py # Random Forest, LinearSVC, RBF SVM Pipelines
│   ├── evaluate.py         # Evaluation metrics, latency timing, model sizing
│   ├── experiment.py       # Validation tuning, full refits, final_test_run, CV
│   └── results.py          # Summary aggregation, comparison plots, error analysis
├── notebooks/              # Sequential analysis notebooks (with committed outputs)
│   ├── 01_data_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_linear_knn.ipynb
│   ├── 04_rf_svm.ipynb
│   └── 05_results_analysis.ipynb
├── data/splits/            # val_indices.npy (committed fixed validation split)
├── results/                # Benchmark outputs
│   ├── metrics/            # Validation, test, and CV JSON files per experiment
│   ├── figures/            # 300 dpi plots (eda_*, features_*, linear_*, treekernel_*, results_*, errors_*)
│   ├── summary.csv         # Aggregated 12-experiment benchmark table
│   └── test_log.csv        # Append-only audit log of final test runs
├── docs/                   # Full capstone engineering documentation
│   ├── prd.md              # Requirements and research questions
│   ├── architecture.md     # Components and folder layout
│   ├── system-design.md    # Runtime flows and leakage controls
│   ├── decisions.md        # Decision log with reasons (D-001 to D-032)
│   ├── module-interfaces.md # Function contracts and signatures
│   ├── data-formats.md     # File formats, schemas, and naming conventions
│   ├── progress.md         # Phase tracking and session log
│   └── phases/v1/          # Individual phase specification files
├── CONTRIBUTING.md         # Contributor step-by-step workflow guide
├── AGENTS.md               # Rules for human and AI contributors
├── requirements.txt        # Pinned Python 3.12 dependencies
└── README.md               # Project overview, setup, results, and synthesis
```
