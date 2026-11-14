# Phase v2/04 — Hybrid CNN Features + RBF SVM

| | |
|---|---|
| Status | in-progress |
| Branch | `phase/v2-04-hybrid-svm` |
| Depends on | Phase v2/02 (CNN3 checkpoint and test results) |
| PRD features | F-22 |

## Goal
Separate the two things CNN3 changed relative to Phase 1's best model: the features and the classifier. CNN3's last
hidden layer (128 learned features) is fed to the RBF SVM under exactly the Phase 1 protocol, and the result answers
RQ9: does an RBF SVM on CNN3's features beat CNN3's own softmax head?

## Previous phase summary
From v2/02: CNN3 is the final CNN (val 0.9308, test 92.09%, macro F1 0.9208); `rbf_svm_hog` 91.08% / 0.9108.
`test_log.csv` has 14 rows. The CNN3 checkpoint is `models/cnn3_images_best.pt` (gitignored; recreated exactly by
notebook 06, D-041).

## In scope — exact features
1. **CNN features** (F-22): `cnn.load_trained()`, `cnn.extract_cnn_features()`, feature kind `cnn` in
   `features.get_features()` (cache `features/cnn3_{split}.npy`).
   - Acceptance: shape (N, 128) float32; CNN3's own head applied to the val features gives val accuracy 0.9308.
2. **`rbf_svm_cnn` experiment**: Phase 1 protocol (`tune_on_val` on the 10k subset with `RBF_SVM_GRID`,
   `full_train_val_score`, then one `final_test_run`).
   - Acceptance: one new row in `test_log.csv` (15 rows); `summary_all.csv` has 15 rows; `summary.csv` unchanged (12 rows).
3. **Notebook 08**: feature check, tuning, val comparison, test run, comparison with CNN3 and `rbf_svm_hog`
   (accuracy, macro F1, McNemar, Shirt recall, cost), answer to RQ9.
   - Acceptance: runs top-to-bottom; a re-run reuses saved results and never re-tests.

## Out of scope
- Other classifiers or other layers on CNN features; CNN2 features; retraining the CNN.

## Relevant docs
- `prd.md` §10, `decisions.md` D-044, D-005, D-027, D-029

## Tasks
- [x] 04.1 — CNN features, `features` filter in `results.py`, D-044 → `feat(hybrid): CNN3 features as a feature set`
- [ ] 04.2 — notebook 08: tune, val score, test run, comparison → `feat(hybrid): RBF SVM on CNN3 features`
- [ ] 04.3 — README, docs, phase summary → `docs(phase-v2-04): close phase`

## Done criteria
- [ ] `test_log.csv` 15 rows, `rbf_svm_cnn` once; Phase 1 outputs byte-identical.
- [ ] Notebook 08 runs top-to-bottom with no errors.
- [ ] Docs match what was built; decisions logged.

## How to verify
1. Run notebooks 06, 07 and 08 top-to-bottom; `git status` shows no change to Phase 1 files.
2. `results/test_log.csv` has 15 rows, each experiment once.

## Phase summary
(filled at close)
