# AGENTS.md — How to work on this repo

This file is for **every contributor and every coding agent** (Claude Code, Antigravity, Cursor, Copilot, Codex, or
no AI at all). Read it together with [`CONTRIBUTING.md`](CONTRIBUTING.md): the step-by-step workflow and troubleshooting, required for humans and agents alike. Read it fully before writing any code. If you are an AI agent, treat every rule here as an instruction.

---

## 1. What this project is
A benchmark of **classical machine learning on Fashion-MNIST** (capstone Phase 1). We compare Logistic Regression, kNN,
Random Forest and SVM on two feature sets (raw pixels and HOG), measure accuracy / speed / size, and explain which clothing
classes get confused and why. It is an **offline Python pipeline**: `src/` modules + Jupyter notebooks. There is no app,
no server and no database.

The work is split into **5 phases**, each normally owned by one person and delivered as one Pull Request (phase 05 was
split into two PRs, D-031). v1 is complete; owners are in `docs/progress.md`:

| Phase | File | Builds |
|---|---|---|
| v1/01 | `docs/phases/v1/phase-01-foundation.md` | Environment, data download, fixed split, EDA |
| v1/02 | `docs/phases/v1/phase-02-features-harness.md` | Pixel + HOG features, shared evaluation/tuning code |
| v1/03 | `docs/phases/v1/phase-03-linear-knn.md` | Dummy, Logistic Regression, kNN |
| v1/04 | `docs/phases/v1/phase-04-rf-svm.md` | Random Forest, LinearSVC, RBF SVM |
| v1/05 | `docs/phases/v1/phase-05-results-analysis.md` | Summary table, comparison plots, error analysis, README |

Phases are **sequential**: each one uses code merged by the previous one. Start your phase only after the previous
phase's PR is merged into `main`. You can read your phase file and the docs earlier to prepare.

## 2. Read these first (in this order)
1. **This file.**
2. **`CONTRIBUTING.md`**: the step-by-step workflow (exact git commands, commit/PR steps) and the troubleshooting table. AGENTS.md says *what* the rules are; CONTRIBUTING.md says *how* to carry them out.
3. `docs/progress.md`: which phase is current, who owns it, and what is already done.
4. **Your phase file** in `docs/phases/v1/`: the exact tasks, done criteria and how to verify. **Build only what it lists.**
5. The docs your phase file lists under "Relevant docs". Usually these are:
   - `docs/module-interfaces.md`: the exact function names, inputs and outputs you must implement or call.
   - `docs/data-formats.md`: the exact format of every file you save (JSON keys, CSV columns, figure names).
   - `docs/system-design.md` §3: the leakage rules (see section 4 below).
6. `docs/decisions.md`: why things are the way they are. Read the entries your phase file mentions.

Do **not** use `idea.md` as instructions. It was early brainstorming. Where it differs from `docs/`, `docs/` wins.

## 3. Environment setup (once)
Python **3.12**. Running locally in VS Code is recommended. Colab works as a fallback.
```bash
git clone <repo-url> && cd fashion-image-classification-benchmark
conda create -n fmnist python=3.12 -y
conda activate fmnist
pip install -r requirements.txt        # exists from phase 01 onwards
```
In VS Code, open a notebook and pick the **fmnist** kernel (top right). Notebooks import code with `from src.<module> import ...`,
so always run them with the repo root as the working directory. If you don't have conda, use `python3.12 -m venv .venv`
and activate it instead.

## 4. Rules that must never be broken (no-leakage rules)
These are what the professor will check. Breaking any of them invalidates our results.
1. **The test set is used once per experiment**, only through `src.experiment.final_test_run()`. Never call
   `get_features(..., "test")` or use `X_test` anywhere else (exceptions: the `data.py` sanity checks and phase 05
   error analysis, which reads the stored `y_pred` plus the true test labels/images and never predicts again).
2. **All choices are made on the validation set.** This covers hyperparameters, HOG settings and which model is "best". Never pick anything by test score.
3. **Scalers live inside sklearn `Pipeline`s** (in `models_*.py`). Never call `StandardScaler().fit(...)` on its own.
4. **EDA uses the train split only.**
5. **Never change** `SEED = 42`, the split file `data/splits/val_indices.npy`, or the tuning grids without a PR
   discussion and a new entry in `docs/decisions.md`.
6. **Finish all tuning for a model before running its `final_test_run()`.** Never run tuning and testing at the same time
   (e.g. in two terminals). Otherwise the audit trail shows a test before tuning ended (see D-023).
7. Every run of `final_test_run()` appends a row to `results/test_log.csv`. Don't delete or edit that file by hand.
   If you must re-run a test evaluation (e.g. after fixing a bug), explain why in `docs/decisions.md` and the PR.

## 5. Coding rules (keep it simple)
- Plain functions in `src/`. No classes unless sklearn needs them, and no frameworks, no YAML configs, no clever abstractions.
- Function names and signatures **exactly** as in `docs/module-interfaces.md`. If you need to change one, edit that doc
  first, in the same PR, and say so in the PR description.
- Every constant (seed, paths, sizes, HOG params, n_jobs) comes from `src/config.py`. Never hard-code paths.
- Don't modify inputs in place. Return new arrays or dicts.
- Errors: raise `ValueError` / `RuntimeError` with a message that says how to fix the problem. No bare `except:`, and never swallow errors.
- Short functions (< ~50 lines). Short comments only where the *why* isn't obvious.
- New dependencies are not allowed without asking. The allowed ones are numpy, pandas, scikit-learn, scikit-image, matplotlib, ipykernel.

**Notebooks**
- One notebook per phase (`notebooks/0N_*.ipynb`). They call `src/` functions, show plots and contain the written
  explanation (markdown cells). No model or feature logic is defined inside notebooks.
- Every notebook's first code cell makes `src` importable (copy it from `notebooks/01_data_eda.ipynb`):
  ```python
  import sys, pathlib
  ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / "src").exists() else pathlib.Path.cwd().parent
  sys.path.insert(0, str(ROOT))
  ```
- Before committing a notebook: **Kernel → Restart & Run All**, and make sure there are no errors. Commit it *with* outputs.
- Mark slow cells with a markdown note giving the expected runtime (e.g. "⏱ ~15 min").

**Figures**: save only via `utils.save_fig(fig, name)` → `results/figures/{name}.png`, using the name pattern in `data-formats.md` §12.

**Lean repo (important):** do **not** commit tests, test frameworks, CI workflows, Dockerfiles or other devops files.
You may write throwaway checks in `.scratch/` (it's gitignored) and run them with plain `python .scratch/check_x.py`.
Never commit `data/raw/`, `features/`, `.env` or anything in `.scratch/`.

## 6. Workflow: from phase start to merge

### 6.1 Start your phase
```bash
git checkout main && git pull
git checkout -b phase/v1-0N-<slug>        # exact branch name is in your phase file's header
```
Then make your first commit (`docs(phase-0N): start phase`) with:
- `docs/progress.md`: set your phase's status to `in-progress`, fill **Started** with today's date and your name in **Owner**,
  and update "Current phase" at the top.
- Your phase file: set Status to `in-progress` and fill **Previous phase summary** by copying the essentials from the
  previous phase file's "Phase summary" section.

### 6.2 Build: one task = one commit
Work through the **Tasks** list in your phase file in order. Each task is one commit, using the commit message given
there (Conventional Commits: `feat(scope): ...`, `fix(...)`, `docs(...)`, `chore(...)`, `refactor(...)`).
Tick the task's checkbox `[x]` in the phase file in the same commit. Push after every commit:
```bash
git push -u origin phase/v1-0N-<slug>
```
Small, honest commits matter: commit history shows individual contribution.

### 6.3 Log decisions as you go (mandatory format)
**When:** whenever you choose something that isn't already written in the docs. Examples: a parameter value, a workaround for
slowness, a bug you found and how you handled it, a re-run of a test evaluation, or any deviation from a doc. Log it **in
the same commit** as the change, not in a batch at the end.

**Where and how:**
1. Append the entry at the **bottom** of `docs/decisions.md` with the **next free number** (`D-0NN`). Never renumber,
   edit the meaning of, or delete an old entry. If a decision changes, add a new one and mark the old one
   `Superseded by D-0NN`.
2. Add a matching row to the **Index** table at the top: `| D-0NN | date | v1/0N | short title | Active |`.
3. Copy the **Entry template** from the top of `decisions.md`. **Every heading is required:**
   ```
   ### D-0NN — <short title>
   - **Date:** YYYY-MM-DD
   - **Phase:** v1/0N — <phase name>
   - **Status:** Active | Superseded by D-0NN | Reverted
   - **Type:** stack | architecture | data | api | algorithm | edge-case | tooling | assumption
   **Context**                 what problem came up, what constraints mattered
   **Decision**                what you chose, in 1–2 sentences
   **Alternatives considered** a table: | Option | Why not chosen |  (at least one real alternative)
   **Why**                     the reasoning
   **How it works**            step by step in plain language, as if teaching a classmate; include numbers/results
   **Files touched**           each file and what changed
   **Concepts worth learning** terms the decision relies on, one line each
   **Revisit if**              when this should be reconsidered
   ```
4. Write for a learner, not a reviewer: someone reading only `decisions.md` after the project should understand what was
   built and why. Entries with only Context / Decision / Why are **incomplete** and will be sent back in review.

If you change a function or file format, also update `module-interfaces.md` / `data-formats.md` in the same commit.

### 6.4 Close your phase
1. Go through **Done criteria** in your phase file. Every box must be true. Run every step in **How to verify**.
2. Fill the **Phase summary** section (Built / Key decisions / Known limitations / Next phase should know).
   The next person reads this first, so make it useful.
3. In `docs/progress.md`, set your phase to `review` and add a 2–4 line entry to the Session log.
4. Commit (`docs(phase-0N): close phase`) and push.

### 6.5 Open the Pull Request
- Open a PR from your phase branch **into `main`**, titled `Phase v1/0N — <phase name>`.
- The description is pre-filled from `.github/pull_request_template.md`. **Fill every section.** Don't delete the
  checklists. A PR with an empty template will not be reviewed.
- Request review from the repo owner. **Don't merge your own PR.**

### 6.6 Review and merge
- The repo owner reviews the PR (with Claude Code) against your phase file, `module-interfaces.md`, `data-formats.md`
  and the no-leakage rules.
- Fix review comments with new commits on the same branch (`fix(...): ...`) and push. **Never force-push**, and don't rewrite history.
- Once approved, the owner merges with **"Create a merge commit"** (not squash, so your individual commits stay visible),
  then sets the phase to `done` in `docs/progress.md` and fills **Closed**.
- The next phase owner then starts from the updated `main` (6.1).

## 7. Extra instructions for AI coding agents
- Before writing code, open and read: this file, `CONTRIBUTING.md`, `docs/progress.md`, the current phase file, and the docs it lists.
  Follow CONTRIBUTING.md's steps for branching, committing, closing the phase and writing the PR, and use its
  troubleshooting table before inventing a workaround.
  Summarise the tasks back to the human before starting.
- **Do only the current phase's tasks.** If you notice something worth doing outside them, don't build it. Add it to
  "Parked ideas" in `docs/progress.md` and mention it to the human.
- **If something is unclear or the docs conflict, stop and ask the human.** Don't guess.
- Never touch the test set outside `final_test_run()`. Never change the seed, split file or grids on your own.
- **Log every decision you make in `docs/decisions.md` in the full format of §6.3** (all headings, next free D-number,
  Index row), in the same commit as the change. If you aren't sure whether something counts as a decision, log it.
- Don't add dependencies, tests, CI, Docker or config frameworks.
- Keep the code at a level a second-year CS student can explain in a viva. Prefer obvious code over clever code.
- Use the exact commit messages from the phase file, and push after each commit.
- Before saying a task is done, run the notebook top-to-bottom (or the relevant `python -c ...` check) and report the
  actual output, including any number that falls outside the sanity ranges in `docs/prd.md` §8.
- When the phase is finished, write the PR description by filling `.github/pull_request_template.md` completely.

## 8. Where things live
```
src/            reusable code (config, utils, data, features, models_*, experiment, evaluate, results)
notebooks/      one notebook per phase: runs src/ code, shows plots, explains results
data/splits/    val_indices.npy, the fixed validation split (committed)
results/        metrics JSONs, summary.csv, test_log.csv, figures (committed, written only by code)
docs/           source of truth: PRD, architecture, formats, interfaces, decisions, progress, phase files
.github/        PR template
```
Full layout: `docs/architecture.md` §4.
