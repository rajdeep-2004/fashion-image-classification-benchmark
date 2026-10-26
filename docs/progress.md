# Progress

**Current phase:** v1/05 — Results, Error Analysis & README (`review`)
**Branch:** Part A: `phase/v1-05-anand-results` (merged); Part B: `phase/v1-05-omved-analysis`
**Profile:** lean

## Phases
### v1 — Capstone Phase 1: classical ML benchmark
| # | Phase | Owner | Status | Branch | Started | Closed |
|---|---|---|---|---|---|---|
| 01 | [Foundation & Data](phases/v1/phase-01-foundation.md) | Rajdeep | done | phase/v1-01-foundation | 2026-10-24 | 2026-10-24 |
| 02 | [Features & Evaluation Harness](phases/v1/phase-02-features-harness.md) | Rajdeep | done | phase/v1-02-features-harness | 2026-10-24 | 2026-10-24 |
| 03 | [Dummy, Logistic Regression & kNN](phases/v1/phase-03-linear-knn.md) | Rajat Srivastav | done | phase/v1-03-linear-knn | 2026-10-24 | 2026-10-24 |
| 04 | [Random Forest & SVM](phases/v1/phase-04-rf-svm.md) | Tanmay Singh | done | phase/v1-04-rf-svm | 2026-10-24 | 2026-10-25 |
| 05 | [Results, Error Analysis & README](phases/v1/phase-05-results-analysis.md) | Anand Mishra (A), Omved Nagre (B) | review | phase/v1-05-anand-results (A), phase/v1-05-omved-analysis (B) | 2026-10-25 | |

Status values: `not-started` → `in-progress` → `review` → `done`

**Team workflow:** each phase is owned by one team member → built on its own branch → opened as a PR into `main` →
reviewed by the repo owner (with Claude Code) against the phase file and docs → merged. Owners are filled in when
each phase starts. Full workflow: [`AGENTS.md`](../AGENTS.md) §6.

## Session log
<!-- Newest first. 2–4 lines per session. Keep only the last ~10 sessions; older ones are in git history. -->
### 2026-10-26 (v1/05 Part B review, lead)
- Audited PR #4: no leakage; error-analysis numbers match the saved predictions. Fixed on the PR branch (D-032): wrong numbers in the write-up (HOG extraction 216 ms not ~30 ms, kNN k=3, recall ranges, error shares, val vs test), simpler error-analysis code, interfaces and figure names synced.
- Verified: fresh clone, notebooks 01→05 run in ~100 s with 0 errors and every file in `results/` regenerates identically; test_log.csv still 12 rows.
- Next: merge PR #4, mark v1/05 and v1 done.

### 2026-10-26 (v1/05 Part B — Omved Nagre)
- Done: Tasks 05.4 (error analysis in results.py: plot_confusion, top_confused_pairs, plot_misclassified, model_agreement; 6 publication figures), 05.5 (notebook 05_results_analysis.ipynb top-to-bottom run with verified outputs, answers to RQ1–RQ4, Phase 2 CNN motivation), 05.6 (complete README.md), and 05.7 (phase close and docs).
- Verified: summary.csv has 12 rows; best model is rbf_svm_hog (91.08% test acc, 0.8775 CV); top confusion Shirt vs T-shirt/top (10.4%); test_log.csv untouched (MD5 identical, 12 rows). All Done criteria met.
- Next: Open PR from Omved's fork targeting upstream main for review and merge by team lead.

### 2026-10-26 (v1/05 Part A review, lead)
- Audited PR #3: summary.csv, 3 figures and CV JSON all reproduced exactly from code; every summary row re-checked against the raw JSONs; no test use.
- Fixed on the PR branch: simpler `results.py` (shared `comparable_val_acc()`), RQ3 chart layout and footnote, D-030 claims. D-031 logs the two-owner split.
- Next: merge PR #3; Omved starts Part B (05.4–05.7) from `main` following the handover notes in the phase file.

### 2026-10-25 (phase v1/05 — Anand's contribution)
- Done: Tasks 05.1 (results aggregation into results/summary.csv, pick_best_experiment using comparable validation accuracy), 05.2 (comparison plots: results_pixels_vs_hog.png, results_acc_vs_time.png, results_recall_heatmap.png), and 05.3 (5-fold stratified CV on rbf_svm_hog: 0.8775 ± 0.0054 on 10k subset). D-030 logged for aggregation rules and timing handling.
- Verified: summary.csv has 12 rows, exact schema, NaN for non-comparable logreg timings; rbf_svm_hog is best by comparable val (0.9157); test_log.csv has not changed (MD5 identical).
- Next: Open PR for review. Omved will complete Part B (tasks 05.4–05.7: error analysis, final notebook 05, README, and phase closure).

### 2026-10-25 (v1/04 merged + fair val for RBF, lead)
- Merged PR #2 (v1/04 done). Found that RBF SVM's val score came from a 10k-subset model, unlike all others.
- D-029, committed before running: refit RBF's chosen params on full 54k train, score on val, keep "best by val". Result: rbf_svm_hog 0.9157 > rbf_svm_pixels 0.9082 > logreg_hog 0.8927. No test use.
- Next: v1/05 (owner TBD) starts from the updated main.

### 2026-10-25 (phase v1/04)
- Done: all 6 tasks of v1/04. Built RF, LinearSVC, and RBF SVM models on pixels and HOG (6 new validation and test JSONs). 12/12 experiments completed in test_log.csv without leakage. Highest test accuracy: RBF SVM HOG (91.08%) and RBF SVM Pixels (89.70% test acc); LinearSVC HOG achieved 88.98% (+5.2 pts over pixels). Generated all tuning curves and 28x28 RF pixel importance heatmap.
- Verified: notebook 04 runs top-to-bottom and reproduces analysis in <15s without test re-evaluation. Decisions D-026 (LinearSVC dual/primal convergence) and D-027 (RBF SVM 10k subset tuning and full 60k refit scaling) documented.
- Next: PR review and merge into `main`. Phase 05 (Results, Error Analysis & README) can begin.

### 2026-10-25 (v1/04 review fixes, lead)
- Audited PR #2: all 6 test results recomputed from real labels and match; one test per experiment, each after its tuning; no leakage.
- Fixed on the PR branch (D-028): figures one-per-file and generated by the notebook, notebook-03 test guard, ⏱ runtime notes, D-026/D-027 corrected, false "best by val" claim fixed (by val, logreg_hog leads). No new test use; result files byte-identical.

### 2026-10-24 (v1/03 review + merge, lead)
- Done: audited PR #1 (all 6 test results recomputed from real labels; leak-free); approved and merged. Post-merge fixes (D-025): weight-map plot moved to notebook, D-020–D-024 expanded, decision-log format made mandatory for all agents, "finish tuning before testing" rule.
- Next: v1/04 (RF + SVM). Add teammates as GitHub collaborators so PRs come from repo branches, not forks.

### 2026-10-24 (phase v1/03)
- Done: all 6 tasks, six validation/test pairs and unique audit entries, four validation curves, LR weight maps and fully executed notebook. Test accuracy: Dummy 10%; LR pixels/HOG 84.72%/88.86%; kNN pixels/HOG 86.11%/86.31%.
- Verified: exact grids/split, JSON schemas, 10k predictions/confusion totals, notebook restart without new test use. Decisions D-020–D-023 cover convergence, result reuse, weight maps and LR timing contention.
- Next: owner review and merge; Phase 04 starts after merge. Direct push denied; submitted from Rajat's fork. No implementation blockers remain.

### 2026-10-24 (phase v1/02)
- Done: all 6 tasks. Pixel + HOG features with cache; HOG sweep → o12_p4_c2 (val 0.8818); evaluate/experiment harness verified (test never touched in tuning); notebook 02. Fresh clone reproduces 01–02.
- Next: approved and merged to `main` (2026-10-24). v1/03 (Dummy, LR, kNN) is next.
- Blockers: none. Note: exact LinearSVC is slow on HOG (relevant for phase 04).

### 2026-10-24 (phase v1/01)
- Done: all 7 tasks of v1/01. Data verified (official MD5s), sanity checks (0 duplicates/overlap), fixed split committed, EDA notebook + 8 figures. Fresh-clone run reproduces everything.
- Next: approved by the team lead and merged to `main` directly (no PR for this phase, lead's call). The v1/02 owner starts next.
- Blockers: none.

### 2026-10-24
- Done: Ran /init-project. Wrote PRD, architecture, system design, data formats, module interfaces, decisions D-001 to D-012, and the 5 v1 phase files.
- Next: assign owners; owner of v1/01 starts the phase (AGENTS.md §6.1, or `/phase-start v1/01` in Claude Code).
- Blockers: none.

## Known debt
<!-- Shortcuts taken on purpose. Remove when paid off. -->
- None yet.

## Parked ideas
<!-- Out-of-scope ideas that came up while building. Never built without approval. -->
- IEEE LaTeX report (`report/`) and presentation slides inside the repo. Planned after v1 code is done.
- PCA-reduced pixels as a third feature set (faster SVM/kNN; curse-of-dimensionality demo).
- LBP texture features; L1 vs L2 Logistic Regression.
- Capstone Phase 2 as `docs/phases/v2/`: small CNN on the same split + `evaluate()`, ablations (blocks, dropout,
  augmentation), hybrid CNN-features → SVM or HOG + CNN fusion aimed at shirt/T-shirt/pullover confusion.
