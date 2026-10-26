# Contributing: a step-by-step guide for team members

Welcome! This project is built in **phases**, and each team member normally owns one phase from start to Pull Request
(a large phase can be split into two PRs, as phase 05 was, D-031).
This guide walks you through it in order. The **rules** (what you must and must not do) are in [`AGENTS.md`](AGENTS.md);
read that too, especially §4 "no-leakage rules".

> **Short version:** set up once → read your phase file → make a branch → one commit per task → fill the phase summary →
> open a PR into `main` with the template filled → fix review comments → the team lead merges.

---

## 0. Before you start
You need:
- **GitHub access** to `rajdeep-2004/fashion-image-classification-benchmark` (ask the team lead to add you as a collaborator).
- **git**, **conda** (or any Python 3.12) and **VS Code** with the *Python* and *Jupyter* extensions.
- Optional: an AI coding assistant (see section 9).

## 1. One-time setup
```bash
git clone https://github.com/rajdeep-2004/fashion-image-classification-benchmark.git
cd fashion-image-classification-benchmark
conda create -n fmnist python=3.12 -y
conda activate fmnist
pip install -r requirements.txt
python -c "from src.data import download_raw; download_raw()"   # ~5 min the first time
md5 -q data/splits/val_indices.npy      # macOS; Linux/Colab: md5sum data/splits/val_indices.npy
# must print 54ff6b78b27109cdfc3ebe7204040a07
```
In VS Code: open the folder, then press **Cmd/Ctrl+Shift+P → "Python: Select Interpreter" → fmnist**. In each notebook,
**Select Kernel → fmnist**.

Check that everything works: open `notebooks/01_data_eda.ipynb` → **Restart & Run All**. It should finish with no errors.

## 2. Know your phase
1. Open [`docs/progress.md`](docs/progress.md). The table shows each phase, its owner and its status.
2. **Start only when the previous phase is `done`** (merged into `main`). Phases build on each other.
3. Open **your phase file** in [`docs/phases/v1/`](docs/phases/v1/) and read it fully:
   - **Goal** and **In scope**: what to build, with acceptance criteria.
   - **Previous phase summary**: what already exists and how to use it.
   - **Tasks**: your commits, in order, with the exact commit message for each.
   - **Done criteria** and **How to verify**: how you prove you're finished.
4. Read the docs listed under **Relevant docs**. Usually `docs/module-interfaces.md` (function names) and
   `docs/data-formats.md` (file formats).

Not sure about something? **Ask the team lead before building.** A 2-minute question saves a re-do.

## 3. Start your phase
```bash
git checkout main
git pull
git checkout -b phase/v1-0N-<slug>          # exact name is in your phase file's header, e.g. phase/v1-03-linear-knn
```
Then edit two files:
- `docs/progress.md`: in your row put your name under **Owner**, set **Status** to `in-progress`, and fill **Started** with today's date.
  Also update "Current phase" at the top.
- Your phase file: set **Owner** and **Status**.

```bash
git add docs
git commit -m "docs(phase-0N): start phase"
git push -u origin phase/v1-0N-<slug>
```

## 4. Build: one task = one commit
For each task in your phase file, in order:
1. Write the code (in `src/`) and/or notebook cells (in `notebooks/0N_*.ipynb`).
2. Check it works (see "How to verify" in your phase file).
3. If it's a notebook: **Kernel → Restart & Run All** with no errors, then save. Commit notebooks **with** their outputs.
4. Tick the task in your phase file: `- [ ]` → `- [x]`.
5. If you made a choice that isn't already in the docs, add an entry to `docs/decisions.md` (section 5 below).
6. Commit with the message from the phase file, and push:
   ```bash
   git status                              # check what you're about to commit
   git add src notebooks results docs      # only what belongs to this task
   git commit -m "feat(linear): tune and evaluate kNN"
   git push
   ```

**Never commit:** `data/raw/`, `features/`, `.scratch/`, `.env`, large model files, or test/CI files. `.gitignore` already blocks most of these.
**Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org): `feat(...)`, `fix(...)`, `docs(...)`, `chore(...)`, `refactor(...)`.

Small, honest commits matter: the commit history is how your contribution is seen.

## 5. Logging decisions
`docs/decisions.md` is how the whole team (and the professor) learns *why* things are built the way they are.
Add an entry when you:
- pick a value that isn't in the docs (e.g. a timeout, a subset size),
- work around a problem (e.g. "SVM too slow, so …"),
- re-run a test evaluation (this one is **mandatory**, see AGENTS.md §4),
- do anything differently from your phase file.

Copy the template at the top of `decisions.md`, give it the next number (`D-0NN`), add a row to the Index table, and
explain it so a classmate could learn from it. **Fill every heading** (Context, Decision, Alternatives considered, Why,
How it works, Files touched, Concepts worth learning, Revisit if). The exact format is in `AGENTS.md` §6.3. Entries with
missing sections get sent back in review. Tell your AI assistant to follow §6.3 too.

## 6. Finish your phase
1. Go through **Done criteria** in your phase file. Every box must be true. Run every step in **How to verify**.
2. Fill **Phase summary** in your phase file: *Built / Key results / Key decisions / Known limitations / Next phase should know*.
   The next person reads this first.
3. In `docs/progress.md`, set your status to `review` and add a 2–4 line entry to the **Session log**.
4. Commit and push:
   ```bash
   git add docs
   git commit -m "docs(phase-0N): close phase"
   git push
   ```

## 7. Open the Pull Request
1. On GitHub you'll see a yellow banner **"Compare & pull request"** for your branch. Click it
   (or go to *Pull requests → New pull request*).
2. **base: `main`** ← **compare: `phase/v1-0N-<slug>`**.
3. Title: `Phase v1/0N — <phase name>`.
4. The description is pre-filled from the PR template. **Fill every section:** summary, tasks, key results table,
   both checklists, decisions, deviations, how to verify, notes for the next phase. Empty templates won't be reviewed.
5. Add the team lead as **Reviewer**, then click **Create pull request**.

## 8. Review and merge
- The team lead reviews your PR (with Claude Code) against your phase file, the interfaces/formats docs and the no-leakage rules.
- Answer comments on GitHub. Fix things with **new commits** on the same branch and `git push`.
  **Never** use `git push --force`, and don't rewrite history.
- When it's approved, **the team lead merges** with "Create a merge commit". Don't merge your own PR.
- After the merge, the next phase owner starts from the updated `main` (section 3).

## 9. Using an AI coding assistant
Any assistant is fine (Claude Code, Antigravity, Cursor, Copilot, …). You are responsible for what it writes and must be
able to explain it at the viva.
- **Claude Code** reads `CLAUDE.md` (which includes `AGENTS.md`) automatically. You can run `/phase-start v1/0N`.
- **Other tools:** start every session with a prompt like:
  > Read `AGENTS.md`, `CONTRIBUTING.md`, `docs/progress.md` and `docs/phases/v1/phase-0N-<slug>.md`, plus the docs that phase file lists
  > under "Relevant docs". Summarise the tasks back to me, then do **only task 0N.1**. Follow the no-leakage rules
  > and the function names in `docs/module-interfaces.md` exactly. Ask me if anything is unclear.
- Do one task at a time, check its output yourself, then commit.
- Don't let it add libraries, tests, CI, Docker or "improvements" outside your phase file.

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` in a notebook | Make sure the first code cell is the `sys.path` cell copied from `notebooks/01_data_eda.ipynb` |
| VS Code underlines `numpy` / `sklearn` imports | Cmd/Ctrl+Shift+P → *Python: Select Interpreter* → `fmnist` |
| Notebook can't find the `fmnist` kernel | `conda activate fmnist && pip install -r requirements.txt`, then reload VS Code |
| `Checksum mismatch` when downloading | The file was corrupted and has been deleted. Just run `download_raw()` again |
| Download is very slow | Normal for GitHub's file server (~5 min). It's only needed once |
| Kernel dies / out of memory (SVM, kNN) | Close other apps, keep `N_JOBS = 4` (or lower it), use the documented subset sizes |
| A cell takes very long | Expected for SVM/kNN. Phase files and notebooks mark slow cells with ⏱ |
| `WARNING: … evaluated on test 2 times` | You ran `final_test_run` again. Fine if needed, but explain why in `decisions.md` and the PR |
| Merge conflict in `docs/progress.md` | `git pull origin main` into your branch, keep both sides' rows, commit the fix |
| `val_indices.npy` MD5 is different | Don't continue. Run `git checkout -- data/splits/val_indices.npy` and tell the team lead |

## 11. Getting help
Stuck for more than ~30 minutes? Message the team lead with: your phase, the task number, what you tried, and the exact error.
