# idea-v2.md — Project 11, Phase 2 (Deep Learning)

**Project:** Automatic Clothing Category Recognition — Inventory Image Tagging via Handcrafted Features and a Small CNN
**This document covers:** Capstone Phase 2 only (deep learning), built **in this same repository** as v2.
Phase 1 (classical ML, `idea.md`, v1) is complete and frozen at tag `v1.0.0`.
**Team:** The lead (Rajdeep Sanyal) builds the foundation (WS1). Each of WS2–WS6 has one owner.
**Scope rule:** Phase 2 does **exactly what the Project 11 brief asks** (§2), done well. The hybrid in §15 is optional
and only started after every required deliverable is finished.

> **For Claude Code:** read `AGENTS.md` first. Its rules and workflow apply unchanged to Phase 2. Phase 2 only **adds**
> to the repo (new `src/` modules, notebooks 06+, new result files, new rows in `summary.csv`, new docs). Never edit
> Phase 1 results, figures, notebooks 01–05, the split file or the seed. Small, backward-compatible changes to shared
> Phase 1 modules (`config.py`, `utils.py`, `evaluate.py`, `experiment.py`, `results.py`) are allowed only where this
> file says so, and each one needs a decision entry.

---

## 1. The idea in one paragraph

Phase 1 showed how far **classical ML with handcrafted features** goes: the best model, an **RBF SVM on HOG features,
reached 91.08% test accuracy**, and every model struggled with visually similar tops (**Shirt** vs T-shirt/top,
Pullover and Coat). The brief names the gap: flattened pixels discard neighbourhood structure, and HOG fixes the
representation before the model sees any examples. In Phase 2 we train a **small convolutional neural network (CNN)**
directly on the 28×28 images. Its filters are **learned from the data**, so it can pick up small details (collars,
buttons, cuffs) that HOG misses. We build a 2-block and a 3-block CNN with pooling, dropout and early stopping, choose
between them on validation, score them with the **same split and the same `evaluate()`** as Phase 1, and check whether
the CNN actually fixes the Shirt confusion.

### 1.1 Phase 1 vs Phase 2 at a glance

| | Phase 1: Machine Learning (v1, done) | Phase 2: Deep Learning (v2, this file) |
|---|---|---|
| Course evaluation | ML evaluation | DL evaluation |
| Question | How far can classical ML with handcrafted features go? | Does a CNN that learns its own features do better, and why? |
| Model input | Features computed beforehand: flattened pixels (784-d) or HOG (1,728-d) | The image itself, from `load_splits()`, reshaped to 1×28×28 |
| Who designs the features | We do: fixed rules (pixel values, HOG gradients) | The network: conv filters are learned during training |
| Models | Dummy, Logistic Regression, k-NN, Random Forest, Linear SVM, RBF SVM | CNN2 (2 conv blocks), CNN3 (3 conv blocks) |
| Training | scikit-learn `.fit()`; tuned on val; refit on train+val (60k) | PyTorch loop (Adam, cross-entropy, mini-batches, epochs); early stopping on val; trained on train (54k) only |
| Regularization | Model-specific (C, k, tree depth, scaling in Pipelines) | Pooling, dropout, early stopping |
| Experiments | 12 (6 models × 2 feature sets) | 2 (`cnn2_images`, `cnn3_images`); final model chosen on val |
| Split, test set, metrics | 54k / 6k / 10k, `evaluate()` | **Identical**: same split file, same test set, same `evaluate()` |
| Best result | `rbf_svm_hog`: 91.08% test, Shirt recall 75.2% | To be measured and compared in `summary.csv` |
| Stack | NumPy, pandas, scikit-learn, scikit-image, matplotlib | Same, plus PyTorch (`torch`, `torchvision`) |

## 2. The Project 11 brief (verbatim from the course's project ideas document)

> **Project 11: Automatic Clothing Category Recognition**
> **Title:** Inventory Image Tagging via Handcrafted Features and a Small CNN
>
> **Project Overview & Abstract:** Classify basic clothing images for a small catalogue and examine whether a CNN
> resolves visually similar categories better than fixed image descriptors.
>
> **The 'Hybrid' Logic: Handcrafted Visual Features to Hierarchical Visual Features**
> **AML baseline:** Normalize the 28 by 28 images; compare flattened pixels and HOG descriptors with Logistic
> Regression, k nearest neighbors, Random Forest and SVM.
> **Identify the gap:** Flattened pixels discard neighbourhood structure, while HOG fixes the representation before the
> model sees category specific examples.
> **DL model:** Train a two or three block CNN with pooling, dropout and early stopping.
> **Fair comparison:** Use the official train test split with a validation set drawn only from training data. Report
> accuracy, macro F1, per class recall, parameter count and inference time.
>
> **Suggested Pre-reads:** Xiao et al. (2017). Fashion MNIST. LeCun et al. (1998). Gradient Based Learning Applied to
> Document Recognition.
> **Datasets & Stack:** Fashion MNIST, available through torchvision or TensorFlow datasets. Suggested stack: Python,
> NumPy, scikit image, scikit learn, PyTorch or TensorFlow and matplotlib.
> **Feature Engineering & Setup Tips:** Scale pixels to zero one, keep the class labels intact, compute HOG only after
> fixing image orientation, and inspect confusion between shirts, T shirts and pullovers.
> **Expected Deliverable:** A classifier notebook with AML versus CNN confusion examples and a concise error analysis.

**Course rules for Phase 2 (instructions PDF):** Phase 2 is the Deep Learning evaluation. Rubric items: Architecture
Logic, Hybrid Innovation, Ablation Studies, Deep Learning Literature Review, DL Dataset & Regularization, Technical
Validation, Theoretical Rigor, GitHub Repository & Code Quality, Project Presentation, Viva Voce, Project Report
(LaTeX/IEEE). A hybrid ML + DL approach earns **additional** marks only if it is meaningful, justified and successfully
implemented. The **viva is individual**: every member must be able to explain the whole project.

**What Phase 1 already covered:** the AML baseline and the gap analysis. **What Phase 2 covers:** the DL model, the
fair comparison, the setup tips that apply to a CNN, and the expected deliverable (AML vs CNN confusion examples +
error analysis).

## 3. Research questions (continuing from Phase 1's RQ1–RQ4)

| # | Question | Answered by |
|---|---|---|
| RQ5 | Does a small CNN beat the best classical model (`rbf_svm_hog`, 91.08% test) on accuracy and macro F1? | WS3, WS4 |
| RQ6 | Does the CNN reduce the Shirt / T-shirt / Pullover / Coat confusion that limited every Phase 1 model? | WS4 |
| RQ7 | Does a third conv block help, and what do pooling, dropout and early stopping contribute? | WS1, WS2 |
| RQ8 | What does the CNN cost compared with the classical models (parameters, model size, inference time)? | WS3, WS4 |

## 4. Goals and non-goals

**Goals**
- A reproducible PyTorch pipeline on the **same split, seed (42) and metrics** as Phase 1.
- Two CNNs (CNN2, CNN3) with pooling, dropout and early stopping; the final one chosen **on validation**.
- Training curves (train vs val loss and accuracy per epoch) for both.
- One test run per CNN through the shared `evaluate()`, logged in `test_log.csv`; two new rows in `summary.csv`.
- AML vs CNN confusion examples and a concise error analysis focused on Shirt / T-shirt / Pullover / Coat.
- DL literature review, theory, IEEE LaTeX report, presentation; every member can explain every stage in the viva.

**Non-goals (Phase 2)**
- No changes to Phase 1 results, figures, notebooks 01–05, the split file or the seed.
- No re-running of Phase 1 models or HOG (their saved results are reused).
- No transfer learning, pretrained models, ResNet or Transformers.
- No data augmentation, batch normalization or learning-rate schedulers (not in the brief).
- No hyperparameter search. Fixed, documented settings only (§7).
- No new dataset, web app or demo. No tests, CI or Docker (as `AGENTS.md` §7 says).
- No hybrid unless §15's condition is met.

## 5. Dataset and split (reused from Phase 1, unchanged)

| Item | Value |
|---|---|
| Dataset | Fashion-MNIST (Xiao et al. 2017), official IDX files, MD5-verified by `src.data.download_raw()` |
| Split | `src.data.load_splits()`: 54,000 train / 6,000 val (stratified, seed 42, `data/splits/val_indices.npy`) / 10,000 official test |
| Format returned | `X_*`: float32 in [0, 1], shape (N, 28, 28). `y_*`: int64 labels 0–9 |
| Split check | `md5sum data/splits/val_indices.npy` → `54ff6b78b27109cdfc3ebe7204040a07` |
| EDA | Already done in Phase 1 (notebook 01). Not repeated; the report cites it |

The brief's "scale pixels to zero one" is already satisfied by `load_splits()`. The only new data step for the CNN is
adding the channel dimension (N, 28, 28) → (N, 1, 28, 28) and wrapping the arrays in PyTorch `DataLoader`s.

## 6. Pipeline overview

```
 data/raw/*.gz ─► [Phase 1, reused] src.data.load_splits()
                    X_train 54k · X_val 6k · X_test 10k  (float32 [0,1], N×28×28)
      │
      ▼
 [WS1] src/cnn_data.py   add channel dim → N×1×28×28 → DataLoaders (train shuffled, batch 64)
 [WS1] src/models_cnn.py CNN2, CNN3, count_params(), CNNPredictor (.predict() wrapper)
 [WS1] src/train.py      train_cnn(): Adam + cross-entropy, epochs, early stopping on val loss,
                         best checkpoint → models/, per-epoch history → results/training/
      │
      ▼
 [WS2] train CNN2 + CNN3 → compare on val → choose final model     → results/metrics/cnn*_images_val.json
      │
      ▼
 [WS3] src.experiment.final_test_run_cnn() → shared evaluate()     → results/metrics/cnn*_images_test.json
                                                                     results/test_log.csv, results/summary.csv
      │
      ▼
 [WS4] CNN vs rbf_svm_hog: confusion, per-class recall, misclassified examples, agreement, speed/size
      │
 [WS5] DL literature review + theory  (parallel from day 1)
 [WS6] README Phase 2, IEEE report v2, slides, viva walkthrough (integrates everything)
```

## 7. Model and training specification (defaults; any change needs a decision entry)

**CNN2 (baseline, 2 conv blocks)**
| Layer | Output shape | Parameters |
|---|---|---|
| Input | 1 × 28 × 28 | 0 |
| Conv2d(1→32, 3×3, padding 1) + ReLU | 32 × 28 × 28 | (3·3·1 + 1)·32 = 320 |
| MaxPool2d(2) | 32 × 14 × 14 | 0 |
| Conv2d(32→64, 3×3, padding 1) + ReLU | 64 × 14 × 14 | (3·3·32 + 1)·64 = 18,496 |
| MaxPool2d(2) | 64 × 7 × 7 | 0 |
| Flatten | 3,136 | 0 |
| Linear(3136→128) + ReLU | 128 | (3136 + 1)·128 = 401,536 |
| Dropout(0.5) | 128 | 0 |
| Linear(128→10) | 10 class scores | (128 + 1)·10 = 1,290 |
| **Total** | | **421,642** |

**CNN3 (variant, 3 conv blocks).** CNN2 plus `Conv2d(64→128, 3×3, padding 1) + ReLU + MaxPool2d(2)` → 128 × 3 × 3 =
1,152 → Linear(1152→128) + ReLU → Dropout(0.5) → Linear(128→10).
Parameters: 320 + 18,496 + 73,856 + 147,584 + 1,290 = **241,546**. CNN3 has *fewer* parameters because the extra
pooling shrinks the input to the large dense layer (a viva point). Verify both totals with `count_params()`.

**Training (both models)**
| Setting | Value | Why |
|---|---|---|
| Loss | `nn.CrossEntropyLoss` (softmax inside) | Standard for multi-class; same loss as Phase 1's Logistic Regression |
| Optimizer | Adam, lr 0.001 | Reliable default; no tuning needed |
| Batch size | 64 | ~844 updates per epoch on 54k images |
| Max epochs | 30 | Upper bound; early stopping ends earlier |
| Early stopping | Stop when val loss hasn't improved for 5 epochs (patience 5); restore the best checkpoint | Required by the brief; prevents overfitting |
| Seed | `config.SEED = 42` for Python, NumPy and PyTorch; `torch.use_deterministic_algorithms(True)` where supported | Reproducibility |
| Device | CPU (a few minutes per model). Record the device in the result JSON | One comparable timing setup |

**Model choice:** the final CNN is the one with the higher **validation accuracy** (ties: fewer parameters).
**Not refit on train + val** (unlike Phase 1), because early stopping needs the val set. Log as D-033.

## 8. Shared conventions (everyone follows these)

All Phase 1 conventions in `AGENTS.md` still apply: seed 42, logic in `src/`, thin notebooks, one task = one commit,
decision log in the full format, figures written only by code. Phase 2 additions:

- **Experiment names:** model `cnn2` / `cnn3`, feature kind `images` → `cnn2_images`, `cnn3_images`.
- **Interfaces (new):**
  - `cnn_data.make_loaders(batch_size=64) -> dict` with keys `train`, `val` (built from `load_splits()`; no test loader).
  - `models_cnn.CNN2()`, `models_cnn.CNN3()` (`nn.Module`); `models_cnn.count_params(model) -> int`.
  - `models_cnn.CNNPredictor(model)`: `.predict(X)` takes (N, 28, 28) float32 images, returns class indices (runs
    `model.eval()` + `torch.no_grad()`). This lets the unchanged `evaluate()` score a CNN.
  - `train.train_cnn(name, model, loaders) -> dict`: trains with early stopping, saves `models/{name}_best.pt` and
    `results/training/{name}_history.csv`, writes `results/metrics/{name}_images_val.json`, returns the summary.
  - `experiment.final_test_run_cnn(name, model) -> dict`: loads the best checkpoint, calls `evaluate()` on test once,
    appends to `test_log.csv`. **The only place Phase 2 touches the test set.**
- **Val JSON (`{name}_images_val.json`):** follows `data-formats.md` §7 closely enough that `results.comparable_val_acc()`
  works (must contain `best_val_acc`), plus `best_epoch`, `epochs_run`, `final_train_acc`, `param_count`, `settings`.
- **Test JSON (`{name}_images_test.json`):** the Phase 1 schema (`data-formats.md` §8) unchanged, plus
  `param_count` and `device`. `env` also records the PyTorch version.
- **Figures:** prefix `cnn_` (e.g. `cnn_training_curves.png`, `cnn_confusion_cnn2_images.png`), PNG, 300 dpi, saved with
  `utils.save_fig`.
- **Git:** branches `phase/v2-0N-<slug>`; phase files in `docs/phases/v2/`; decisions continue from **D-033**;
  `docs/progress.md` gets a "v2 — Capstone Phase 2" phase table. PRs reviewed by the lead with Claude Code, merged with
  a merge commit (not squash).

## 9. The six workstreams

Each workstream has an owner, a scope, deliverables, "done when" criteria and viva prep. Each owner also writes the
matching report section (WS6 merges them). Every member must still be able to explain **all** of Phase 2 in the viva.

---

### WS1 — Setup, CNN Models & Training Pipeline (Rajdeep Sanyal)

**One-line role:** *"Set up the deep learning stack and built the CNN architectures and the training pipeline that every Phase 2 experiment runs on."*

**Scope**
- Confirm Phase 1 is tagged `v1.0.0`. Add pinned `torch` and `torchvision` to `requirements.txt` (CPU build is fine).
- `config.py`: CNN constants (batch size, lr, max epochs, patience, dropout, model and history paths).
- `utils.py`: extend `set_seed()` to seed PyTorch.
- `src/cnn_data.py`: `make_loaders()` from `load_splits()` (add channel dim, tensors, DataLoaders; no test loader).
- `src/models_cnn.py`: `CNN2`, `CNN3`, `count_params()`, `CNNPredictor`.
- `src/train.py`: `train_cnn()` (training loop, early stopping, best checkpoint, history CSV, val JSON).
- `notebooks/06_cnn_training.ipynb`: shape walk-through of one batch through CNN2 (print each layer's output shape),
  parameter counts for both models checked against §7, one full CNN2 training run, its training curves.
- Docs: plan v2 in `docs/prd.md`, `architecture.md`, `system-design.md`, `module-interfaces.md`, `data-formats.md`,
  `progress.md`; add the Phase 2 leakage rules (§10) to `AGENTS.md`; create the v2 phase files.

**Deliverables**
- `src/cnn_data.py`, `src/models_cnn.py`, `src/train.py`, notebook 06
- `models/cnn2_best.pt`, `results/training/cnn2_history.csv`, `results/metrics/cnn2_images_val.json`
- `cnn_training_curves_cnn2.png`
- Report section: **CNN Architecture & Training Setup**

**Done when**
- `count_params(CNN2()) == 421,642` and `count_params(CNN3()) == 241,546`.
- CNN2 trains end-to-end with early stopping and its val accuracy is in the sanity range (§12).
- Re-running notebook 06 gives the same val accuracy (same seed).
- No code path in WS1 touches the test set.

**Viva prep**
- What a convolution, a filter and a feature map are; why 3×3 filters and padding 1.
- What ReLU and max pooling do; how the shape goes 1×28×28 → 32×14×14 → 64×7×7 → 3,136.
- How to count parameters for a conv layer and a linear layer.
- The training loop: forward pass, cross-entropy, backpropagation, Adam step; batches vs epochs.
- Dropout and early stopping: what they prevent and how they work.

---

### WS2 — Model Choice: 2-Block vs 3-Block CNN (ablation)

**One-line role:** *"Ran the architecture comparison between the 2-block and 3-block CNNs and chose the final model on validation."*

**Scope**
- Train CNN3 with `train_cnn()` (same settings and seed as CNN2).
- Compare CNN2 vs CNN3 on validation: val accuracy, val macro F1, parameter count, best epoch / epochs run,
  train–val accuracy gap (overfitting), training time.
- Overlay training curves for both models; mark the early-stopping epoch on each.
- Show what early stopping did: compare the best-epoch val loss with the last-epoch val loss.
- Choose the final model by val accuracy (§7) and log the choice as a decision entry with the numbers.
- `notebooks/07_cnn_model_choice.ipynb`.

**Deliverables**
- `models/cnn3_best.pt`, `results/training/cnn3_history.csv`, `results/metrics/cnn3_images_val.json`
- `results/cnn_model_choice.csv` (one row per model, the columns above)
- `cnn_training_curves_comparison.png`, `cnn_val_comparison.png`
- Decision entry: final CNN and why
- Report section: **Ablation: Network Depth** (+ the architecture-logic paragraph)

**Done when**
- Both val JSONs exist and the comparison table is generated by code.
- The final model is chosen on validation only, and the decision is logged.
- No test data used.

**Viva prep**
- What an ablation study is and why you change one thing at a time.
- Why CNN3 has fewer parameters than CNN2, and what deeper layers learn (edges → parts).
- How to read training curves: underfitting vs overfitting; what the train–val gap shows.
- Why model choice must use validation, never test.

---

### WS3 — Test Evaluation & Metrics

**One-line role:** *"Integrated the CNNs into the shared evaluation framework and produced the final, leak-free test metrics."*

**Scope**
- Add `final_test_run_cnn()` to `src/experiment.py` (mirrors `final_test_run()`; logs to `test_log.csv`).
- Small, backward-compatible additions to `src/evaluate.py`: `param_count` and `device` fields, PyTorch version in
  `environment()`, a `"parameters"` case in `model_size_detail()`. Model size stays `len(pickle.dumps(model))` (D-008)
  so it is comparable with Phase 1. Phase 1 JSONs are not regenerated.
- Add `cnn2` / `cnn3` / `images` to the ordering lists in `src/results.py` so `build_summary()` includes them.
- Run **one** test evaluation per CNN, after WS2's model choice is merged.
- Rebuild `results/summary.csv` (14 rows) and a CNN results table: accuracy, macro F1, per-class recall, parameter
  count, inference ms / 1,000 images, model size, train time.
- Normalized confusion matrix for both CNNs; per-class recall chart.
- `notebooks/08_cnn_evaluation.ipynb`.

**Deliverables**
- `results/metrics/cnn2_images_test.json`, `cnn3_images_test.json`
- `results/test_log.csv` with exactly **14** rows; `results/summary.csv` with 14 rows
- `cnn_confusion_cnn2_images.png`, `cnn_confusion_cnn3_images.png`, `cnn_per_class_recall.png`
- Report section: **Experimental Setup & Test Results**

**Done when**
- Each CNN appears exactly once in `test_log.csv`; Phase 1's 12 rows are untouched.
- `summary.csv` regenerates byte-for-byte from the JSONs.
- All five brief metrics are reported for both CNNs.

**Viva prep**
- Accuracy vs macro F1 vs per-class recall; why macro F1 matters.
- How inference time and model size are measured, and why the same method as Phase 1 is used.
- Why the test set is used once, and why the CNN is not refit on train + val.
- `model.eval()` and `torch.no_grad()`: what they change at prediction time.

---

### WS4 — Comparison with Phase 1 & Error Analysis

**One-line role:** *"Compared the CNN against the classical models and analysed which confusions the CNN fixed, and which remain."*

**Scope**
- Final CNN vs `rbf_svm_hog` (best classical) and `rbf_svm_pixels`: accuracy, macro F1, per-class recall side by side.
- Accuracy vs inference-time chart and a model-size comparison with all 14 experiments.
- `top_confused_pairs()` for the final CNN vs `rbf_svm_hog`; focus on Shirt, T-shirt/top, Pullover, Coat.
- **AML vs CNN confusion examples (the brief's deliverable):** misclassified-image grids for the same class pairs from
  both models (`plot_misclassified()`), plus "SVM wrong, CNN right" and "CNN wrong, SVM right" grids.
- `model_agreement()` between the final CNN and `rbf_svm_hog`: how many errors they share.
- Visualize the CNN's first-layer filters (32 3×3 kernels) and a few feature maps for one shirt image, contrasted with
  HOG: learned vs fixed features.
- Write the concise error analysis and answer RQ5–RQ8.
- `notebooks/09_cnn_vs_classical.ipynb`. Uses saved predictions only; no model predicts on test again.

**Deliverables**
- `cnn_vs_svm_recall.png`, `cnn_acc_vs_time.png`, `cnn_errors_*` grids, `cnn_filters.png`, `cnn_feature_maps.png`
- Written error analysis and RQ5–RQ8 answers (notebook 09 + report section)
- Report section: **Comparison with Classical Models & Error Analysis**

**Done when**
- Every figure is generated from saved JSONs/predictions; `test_log.csv` is unchanged by this workstream.
- The analysis states, with numbers, whether the Shirt confusion improved and which errors both models share.

**Viva prep**
- Why learned filters can separate shirt / T-shirt / pullover better than HOG (and where they still fail).
- How to read a confusion matrix and the agreement numbers.
- The accuracy / speed / size trade-off between the CNN and the RBF SVM.
- Which images are probably ambiguous in the data itself, and why no model gets them right.

---

### WS5 — Deep Learning Literature Review & Theory

**One-line role:** *"Wrote the deep learning literature review and the theoretical foundations behind every design choice."*

**Scope**
- Literature review (one paragraph each, tied to our design): LeCun et al. 1998 (LeNet, CNNs), Xiao et al. 2017
  (Fashion-MNIST), Srivastava et al. 2014 (Dropout), Kingma & Ba 2015 (Adam), Prechelt 1998 (early stopping),
  Krizhevsky et al. 2012 (deep CNNs, ReLU), Goodfellow et al. 2016 (Deep Learning book, CNN chapter).
- Theory section: convolution and feature maps; receptive field; ReLU; max pooling; softmax + cross-entropy;
  backpropagation and gradient descent; Adam; overfitting and the three regularizers we use; parameter counting with
  CNN2 as the worked example; why CNNs suit images (local connectivity, weight sharing) vs pixels and HOG.
- A one-page glossary for the team's viva prep.

**Deliverables**
- `report/sections/related_work_dl.tex`, `report/sections/theory_dl.tex`, `report/viva_glossary.md`
- BibTeX entries in `report/references.bib`

**Done when**
- Every design choice in §7 is backed by a cited source or a theory paragraph.
- The parameter-count worked example matches `count_params()`.

**Viva prep**
- Explain every term in the glossary without notes.
- How LeNet relates to our CNN; what dropout, Adam and early stopping each solve.

---

### WS6 — Write-Up: README, Report, Slides & Viva Walkthrough

**One-line role:** *"Integrated the whole of Phase 2 into the README, the IEEE report and the presentation, and prepared the team's viva walkthrough."*

**Scope**
- README: Phase 2 section (what, how to run notebooks 06–09, results table generated from `summary.csv`, key findings,
  leakage rules, limitations), team table updated with Phase 2 roles.
- IEEE LaTeX report v2 (`report/report_v2.tex` → PDF): owns the skeleton, merges every workstream's section (§13).
- Slides (~12, §14) and a viva walkthrough (`report/presentation_v2.md`) covering every stage for every member.
- Final docs audit with the lead: docs match the code, every decision logged, `progress.md` closed.

**Deliverables**
- Updated `README.md`, `report/report_v2.tex` + `.pdf`, `report/ppt_v2.pdf` + source, `report/presentation_v2.md`

**Done when**
- The report compiles; every number in README/report/slides matches `summary.csv` and the JSONs.
- A fresh clone runs notebooks 06–09 top-to-bottom with no errors.

**Viva prep**
- The full Phase 2 story in 3 minutes: gap → CNN → training → choice → results → what's still hard.

## 10. Leakage rules for Phase 2 (add to `AGENTS.md` §4)

1. The test set is used **once per experiment**, only through `final_test_run_cnn()`. No test DataLoader exists.
2. **All choices on validation**: early stopping, CNN2 vs CNN3, every setting. Never by test score.
3. CNNs are trained on the **54k train split only**; val is used for early stopping and model choice, never for weight
   updates. No refit on train + val (D-033).
4. No statistics are computed from any data: pixels are already in [0, 1] from `load_splits()`.
5. Always `model.eval()` + `torch.no_grad()` for val and test predictions (dropout must be off).
6. Error analysis reads saved `y_pred`; no model predicts on test again.
7. Finish WS2's model choice before WS3's test runs. Never run training and testing at the same time.

## 11. Dependencies and timeline

| Day | WS1 Setup/CNN/training | WS2 Model choice | WS3 Evaluation | WS4 Comparison | WS5 Lit/theory | WS6 Write-up |
|---|---|---|---|---|---|---|
| 1–2 | Tag v1.0.0, docs, `cnn_data`, `models_cnn` | Study CNNs, read §7 | Read `evaluate.py`, `experiment.py` | Read Phase 1 notebook 05 | Literature review | Report skeleton |
| 3–4 | `train.py`, CNN2 run, notebook 06 | — | Draft `final_test_run_cnn()` | Draft analysis plan | Theory section | README outline |
| 5 | Review WS2 PR | CNN3 run, comparison, choice | — | — | Glossary | Slides outline |
| 6 | Review WS3 PR | — | Test runs, summary, notebook 08 | — | Review | — |
| 7–8 | Review WS4 PR | Report section | Report section | Analysis, notebook 09 | Review | Merge sections |
| 9 | Final review | Review | Review | Report section | Review | Report + slides final |
| 10 | **Full-team dry run of the presentation and mock viva (everyone explains every stage)** | | | | | |

**Critical path:** WS1 → WS2 → WS3 → WS4. WS5 and WS6 start on day 1. Days are relative; shift them to the deadline.

## 12. Expected results (sanity ranges, not targets)

| Model | Approx. test accuracy | If outside |
|---|---|---|
| CNN2 | ~91–93% | Below ~89%: suspect a bug (wrong reshape, `model.eval()` missing, labels shuffled, lr wrong) |
| CNN3 | ~91–93% | Same |
| Shirt recall | Lowest class again, likely ~70–80% | Shirt being easy would be surprising; check the confusion matrix |
| Early stopping | Best epoch roughly 8–20 | Best epoch = 1 or = 30: check lr and patience |

Beating 91.08% is likely but **not guaranteed**. Whatever the result, report it honestly; a tie on accuracy with a
model that is far smaller and faster is still a finding.

## 13. Report outline (IEEE, LaTeX)

| Section | Owner |
|---|---|
| Abstract, I. Introduction (Phase 1 recap, gap, RQ5–RQ8) | WS6 |
| II. Related Work (DL) | WS5 |
| III. Theoretical Background | WS5 |
| IV. Dataset & Protocol (reused split, leakage rules) | WS1 + WS3 |
| V. CNN Architecture & Training | WS1 |
| VI. Ablation: Network Depth | WS2 |
| VII. Test Results | WS3 |
| VIII. Comparison with Classical Models & Error Analysis | WS4 |
| IX. Conclusion & Limitations | WS6 |
| References; Appendix: contribution statement | All |

## 14. Presentation outline (~12 slides)

1. Recap: Phase 1 result and the gap · 2. Why a CNN · 3. Architecture (CNN2 / CNN3) · 4. Training setup ·
5. Training curves · 6. Ablation: 2 vs 3 blocks · 7. Test results table · 8. CNN vs RBF SVM per class ·
9. Confusion examples (AML vs CNN) · 10. Learned filters vs HOG · 11. Speed / size trade-off · 12. Conclusions.
Each member presents their own workstream's slides.

## 15. Optional extension (only if everything in §9 is done on time)

**CNN features + SVM (hybrid).** Cut the trained final CNN before its last layer, use its 128-number output as features
for every image, and train the Phase 1 RBF SVM pipeline on them (experiment `rbf_svm_cnn`, feature kind `cnn`).
Comparing it with `rbf_svm_hog` keeps the classifier fixed and changes only the features, which tests directly whether
handcrafted features were Phase 1's bottleneck. The CNN is trained on the train split only, so its val/test features
are leak-free. **Do not start this until the lead says so.**

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Teammates get different numbers | Fixed seed, deterministic settings, CPU runs, committed split and checkpoints |
| Slow laptop | CPU training is minutes per model; Colab as fallback (record the device) |
| Test-set leakage | No test DataLoader; test only via `final_test_run_cnn()`; `test_log.csv` audit |
| Phase 1 work accidentally changed | Tag `v1.0.0` first; Phase 2 only adds files; reviews check `git diff` on Phase 1 paths |
| Team doesn't understand the DL parts | WS5 glossary, each workstream's viva prep, day-10 mock viva |

## 17. How Claude Code should start

1. Read `AGENTS.md`, `CONTRIBUTING.md`, `docs/README.md` and the docs it lists, then this file.
2. Check that the tag `v1.0.0` exists. If not, ask the lead to create it before anything else.
3. Summarise this plan back to the lead and ask about anything unclear or conflicting (`AGENTS.md` §7).
4. Plan v2 in the docs: `prd.md` (Phase 2 features F-xx, RQ5–RQ8, sanity ranges), `architecture.md`,
   `system-design.md`, `module-interfaces.md`, `data-formats.md`, `progress.md` (v2 phase table), `AGENTS.md`
   (§10 rules); log D-033 onward.
5. Create `docs/phases/v2/phase-01-setup-cnn-training.md`, `phase-02-model-choice.md`, `phase-03-evaluation.md`,
   `phase-04-comparison-analysis.md`, `phase-05-literature-theory.md`, `phase-06-write-up.md` from
   `docs/phases/_phase-template.md`, with tasks and done criteria from §9.
6. Start WS1 on branch `phase/v2-01-setup-cnn-training`.
