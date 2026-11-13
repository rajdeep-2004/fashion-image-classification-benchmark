# Phase v2/03 — README, Report & Slides

| | |
|---|---|
| Status | in-progress |
| Branch | `phase/v2-03-write-up` |
| Depends on | Phase v2/02 |
| PRD features | F-21 |

## Goal
The Phase 2 results are written up: a README section, an IEEE LaTeX report and slides, with every number matching the
saved results. The release is tagged `v2.0.0`.

## Previous phase summary
From v2/02: every number is in `results/summary_all.csv`, the test JSONs and notebook 07 §7. CNN3 (final) 92.09% /
macro F1 0.9208 vs `rbf_svm_hog` 91.08% / 0.9108; Shirt recall unchanged (75.1% vs 75.2%); CNN3 ~300× faster and
~377× smaller. `test_log.csv` has 14 rows; no more test evaluations.

## In scope — exact features
1. **README Phase 2 section**: what was built, how to run notebooks 06–07, results table generated from
   `summary_all.csv`, answers to RQ5–RQ8, v2 leakage rules, limitations.
   - Acceptance: every number matches `summary_all.csv` / the JSONs.
2. **IEEE LaTeX report** in `report/v2/` (`report.tex` → `report.pdf`): introduction (Phase 1 recap, the gap, RQ5–RQ8),
   short related work (LeCun et al. 1998, Xiao et al. 2017, Srivastava et al. 2014, Kingma & Ba 2015, Prechelt 1998),
   theory (convolution, pooling, ReLU, cross-entropy, backpropagation, Adam, dropout, early stopping, parameter
   counting with CNN2), dataset and protocol, architecture and training, depth ablation, test results, comparison and
   error analysis, conclusion and limitations.
   - Acceptance: compiles; figures come from `results/figures/`; numbers match the saved results.
3. **Slides** in `report/v2/` (~12): recap and gap, why a CNN, architectures, training setup and curves, CNN2 vs CNN3,
   test results, CNN vs RBF SVM per class, AML vs CNN confusion examples, speed/size trade-off, conclusions.
   - Acceptance: PDF in `report/v2/`; numbers match the report.

## Out of scope
- New experiments or figures not produced in v2/01–v2/02.

## Relevant docs
- `prd.md` §10, `decisions.md` D-033–D-039, notebook 07's written analysis

## Tasks
- [x] 03.1 — README Phase 2 section → `docs(readme): add Phase 2 CNN results`
- [x] 03.2 — IEEE report in `report/v2/` → `docs(report): Phase 2 report`
- [x] 03.3 — slides in `report/v2/` → `docs(report): Phase 2 slides`
- [ ] 03.4 — final docs check, fill the phase summary; tag `v2.0.0` after merge → `docs(phase-v2-03): close phase`

## Done criteria
- [ ] Report compiles; slides exported to PDF.
- [ ] Every number in README, report and slides matches `summary_all.csv` and the JSONs.
- [ ] A fresh clone runs notebooks 01–07 top-to-bottom with no errors.
- [ ] Docs match what was built; decisions logged.

## How to verify
1. Compile `report/v2/report.tex`; open the PDF.
2. Spot-check 5 numbers in the README/report/slides against `results/summary_all.csv`.
3. Fresh clone → set up env → run notebooks 01–07 → `git status` clean.

## Phase summary
- **Built:**
- **Key decisions:**
- **Known limitations:**
- **Next phase should know:**

## Next phase preview
None: v2 is the final release of the capstone.
