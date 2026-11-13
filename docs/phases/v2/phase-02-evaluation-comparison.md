# Phase v2/02 — CNN Test Evaluation & Comparison

| | |
|---|---|
| Status | done |
| Branch | `phase/v2-02-evaluation-comparison` |
| Depends on | Phase v2/01 |
| PRD features | F-19, F-20 |

## Goal
Each CNN is scored once on the test set by the shared `evaluate()`, the all-model table is built without changing any
Phase 1 file, and notebook 07 compares the CNN with the best classical model and answers RQ5–RQ8.

## Previous phase summary
From v2/01: `src/cnn.py` provides `CNN2`, `CNN3`, `CNNPredictor` and `train_cnn(name, make_model)` (seeds, builds
the model and loaders itself, D-041). Val results: CNN2 0.9228, CNN3 0.9308, both above `rbf_svm_hog`'s 0.9157;
**final CNN = CNN3** (D-040), the depth gap is within run-to-run variation. Checkpoints `models/cnn{2,3}_images_best.pt`
are gitignored; to rebuild one, delete the CNN's val JSON and re-run notebook 06 (identical numbers). The test set has
not been used: `test_log.csv` has 12 rows.

## In scope — exact features
1. **Harness additions** (D-037): `model_size_detail` "parameters" case, PyTorch version in `environment()`;
   `results.py` `V1_MODELS` defaults, `summary_all.csv`, `plot_misclassified(tag=...)`, `plot_recall_comparison()`.
   - Acceptance: re-running notebook 05 leaves `summary.csv` and every Phase 1 figure byte-identical (`git status` clean).
2. **`final_test_run_cnn()`** (F-19) in `experiment.py`.
   - Acceptance: loads the checkpoint, calls `evaluate()` once, appends one row to `test_log.csv`; clear `RuntimeError` if the checkpoint is missing.
3. **Test runs** (F-19): notebook 06 test cells, guarded by "test JSON exists → skip"; `build_summary(models=MODEL_ORDER, filename="summary_all.csv")`.
   - Acceptance: `test_log.csv` has 14 rows, each CNN once, after Phase 1's 12 untouched rows; `summary_all.csv` has 14 rows.
4. **Notebook 07** (F-20): table of accuracy, macro F1, per-class recall, parameter count, inference time and size for
   both CNNs vs `rbf_svm_hog` and `rbf_svm_pixels`; confusion matrices; `cnn_recall_vs_classical.png`; top confused pairs
   of the final CNN vs `rbf_svm_hog`; misclassified grids for the same pairs from both models; `model_agreement()`;
   written error analysis answering RQ5–RQ8.
   - Acceptance: every number comes from saved files; the text states with numbers whether Shirt recall improved and how many errors both models share.

## Out of scope
- Any retraining, any second test run, re-predicting the test set in notebook 07.
- Filter/feature-map visualizations, hybrid models, extra charts.

## Relevant docs
- `module-interfaces.md` → v2 additions: `experiment`, `evaluate`, `results`
- `data-formats.md` → §8, §16, §17
- `system-design.md` → §12 Flow E steps 4–5, v2 leakage controls
- `decisions.md` → D-035, D-037, D-038

## Tasks
- [x] 02.1 — `evaluate.py` and `results.py` additions; check notebook 05 output unchanged → `feat(results): support CNN experiments without changing Phase 1 outputs`
- [x] 02.2 — `final_test_run_cnn()` in `experiment.py` → `feat(experiment): single test run for CNNs`
- [x] 02.3 — notebook 06 test cells; run once per CNN; build `summary_all.csv` → `feat(cnn): test evaluation of CNN2 and CNN3`
- [x] 02.4 — `notebooks/07_cnn_vs_classical.ipynb`: comparison, error analysis, RQ5–RQ8 → `feat(analysis): CNN vs classical comparison and error analysis`
- [x] 02.5 — update docs touched by this phase, fill the phase summary → `docs(phase-v2-02): close phase`

## Done criteria
- [x] `test_log.csv`: 14 rows; Phase 1 rows unchanged; each CNN once, with timestamps after the model choice.
- [x] Phase 1 files unchanged: no Phase 1 file in `results/` or `notebooks/0[1-5]*` differs from `v1.0.0` (re-running notebook 05 rewrites them byte-identically).
- [x] All five brief metrics reported for both CNNs.
- [x] Notebooks 06 and 07 run top-to-bottom with no errors and no new test run.
- [x] Docs match what was built; decisions logged.

## How to verify
1. `wc -l results/test_log.csv` → 15 (header + 14); `tail -2 results/test_log.csv` shows `cnn2_images`, `cnn3_images`.
2. Restart & Run All on notebook 05, then `git status` → no changes.
3. Restart & Run All on notebooks 06 and 07 → no errors; `test_log.csv` unchanged.

## Phase summary
- **Built:** `final_test_run_cnn()`; CNN support in `evaluate.py` (parameter count, PyTorch version) and `results.py`
  (Phase 1 defaults, `summary_all.csv`, tagged figure names, `plot_recall_comparison`); test cells and the all-model
  table in notebook 06; notebook 07 with the comparison, confusion matrices, AML vs CNN example grids, model agreement
  and answers to RQ5–RQ8.
- **Results (test, once per CNN):** CNN3 92.09% / macro F1 0.9208, CNN2 91.85% / 0.9176, vs `rbf_svm_hog` 91.08% /
  0.9108. Shirt recall unchanged (CNN3 75.1% vs 75.2%); gains on T-shirt/top, Pullover, Coat. 475 errors shared with
  the SVM (398 with the same label). CNN3: 94 ms per 1,000 images and 0.97 MB vs the SVM's ~29 s and 366 MB.
- **Key decisions:** D-037 (harness integration, verified: notebook 05 output unchanged), D-042 (comparison choices).
- **Known limitations:** one training run per CNN (D-040); timings from two laptops (CNNs on M3, SVMs on M3 Pro).
- **Next phase should know:** every number for the write-up is in `results/summary_all.csv`, the test JSONs and
  notebook 07 §7. `test_log.csv` has 14 rows; do not run any more test evaluations.

## Next phase preview
v2/03 writes the README Phase 2 section, the IEEE report and the slides from `summary_all.csv`, the JSONs and notebook 07.
