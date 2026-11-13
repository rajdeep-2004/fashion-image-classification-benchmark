# Design: Capstone Phase 2 (CNN) as v2 in the existing benchmark repo

Date: 2026-11-14
Status: Approved

## Problem Statement
Course capstone, Project 11, Phase 2 (deep learning). Phase 1 (v1 of this repo) is done: best model RBF SVM on HOG,
91.08% test accuracy; every model confused Shirt with T-shirt/top, Pullover and Coat. Phase 2 trains a small CNN on the
raw 28x28 images under the identical split, seed and `evaluate()`, and answers one question: does the CNN beat the
classical models, and does it fix the Shirt confusion? Deliverables: a classifier notebook with AML vs CNN confusion
examples and a concise error analysis, an IEEE LaTeX report, and slides.

## Constraints
- Scope is what the Project 11 brief and the deliverables require, nothing more.
- Lean profile stays: no tests, CI or Docker; throwaway checks in `.scratch/`.
- Never edit Phase 1 results, figures, notebooks 01-05, the split file or the seed. Zero leakage.

## What the brief requires (and nothing else)
| Brief item | How it is met |
|---|---|
| Two or three block CNN with pooling, dropout, early stopping | CNN2 and CNN3 (idea-v2 §7 specs), early stopping on val loss |
| Official train/test split, val drawn from train only | Reuse `load_splits()` (54k/6k/10k, same split file) |
| Accuracy, macro F1, per-class recall, parameter count, inference time | Shared `evaluate()` plus `param_count` |
| AML vs CNN confusion examples + concise error analysis | One comparison notebook vs `rbf_svm_hog`, using saved predictions |
| Report (LaTeX/IEEE) and presentation | `report/v2/` |

CNN2 vs CNN3 is kept (not cut to one model): it is one extra training run with code that already exists, and it covers
the "Ablation Studies" rubric item.

## Premises
1. Phase 2 lives in this repo as v2 and only appends. No phase1/ and phase2/ folders.
2. Tag `v1.0.0` on `main` before v2 work.
3. Out of scope: dropout-off ablation, CNN-features + SVM hybrid (parked), literature-review/glossary as separate deliverables
   (a short related-work section goes inside the report), filter/feature-map visualizations, extra charts.

## Approaches Considered
- **A: Same repo, append (chosen).**
- B: Same repo, phase folders. Rejected: breaks Phase 1 imports, paths, README and reproducibility; splits the results.
- C: Separate repo. Rejected: Phase 1 results would have to be copied in to compare.

## Recommended Approach
| Area | Change |
|---|---|
| Root | Clean: early brainstorming notes live in `docs/ideas/` |
| `requirements.txt` | Add pinned `torch` (CPU build) |
| `src/` | New `cnn.py` (data loaders, CNN2/CNN3, `count_params`, predictor wrapper, `train_cnn` with early stopping). Small, logged additions to `config`, `utils.set_seed`, `experiment` (`final_test_run_cnn`), `evaluate` (`param_count`), `results` (ordering lists) |
| `notebooks/` | `06_cnn.ipynb` (train CNN2 + CNN3, curves, choose on val, one test run each) and `07_cnn_vs_classical.ipynb` (comparison table, confusion matrices, AML vs CNN misclassified examples, error analysis) |
| `results/` | `cnn_` figures, `results/training/*_history.csv`, val/test JSONs, `summary.csv` to 14 rows, `test_log.csv` to 14 rows |
| `models/` | Checkpoints, gitignored (error analysis reads saved predictions; checkpoints retrain from seed) |
| `docs/` | Append short v2 sections to prd, architecture, interfaces, formats; v2 table in progress.md; decisions from D-033 |
| `report/` | v1 files stay where they are; new `report/v2/` for report and slides |

## v2 phases (in order)
1. **v2/01 CNN + training:** torch dep, `src/cnn.py`, config/seed changes, notebook 06 training part; choose CNN on val.
2. **v2/02 Test + comparison:** `final_test_run_cnn`, one test run per CNN, summary rebuild, notebook 07 error analysis.
3. **v2/03 Write-up:** README Phase 2 section, IEEE report v2, slides.

## Success Criteria
- `count_params(CNN2) == 421,642`, `count_params(CNN3) == 241,546`.
- Each CNN appears once in `test_log.csv`; Phase 1's 12 rows untouched.
- The five brief metrics reported for both CNNs; error analysis states with numbers whether Shirt recall improved.
- Notebooks 06-07 run top-to-bottom from a fresh clone; report compiles; slides done.

## Distribution Plan
Course submission: GitHub repo (tag `v2.0.0` at the end), report PDF and slides in `report/v2/`.

