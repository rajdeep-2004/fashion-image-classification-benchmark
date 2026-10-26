# Fashion-MNIST Classical ML Benchmark (Capstone Phase 1)

Leak-free, reproducible benchmark of classical ML (LR, kNN, RF, SVM) on raw pixels vs HOG features for Fashion-MNIST,
with error analysis that motivates the Phase 2 CNN. Built by a 5-person student team, normally one phase per person (phase 05 split in two, D-031), delivered as PRs. v1 is complete. Details: `docs/prd.md`.

## Shared rules for all contributors and agents
@AGENTS.md

## Step-by-step contributor workflow and troubleshooting
- **Working for a team member** (`git config user.name` is not `Rajdeep`): read `CONTRIBUTING.md` fully before doing
  anything, and follow its steps for branching, commits, closing the phase and the PR.
- **Working for the team lead** (`git config user.name` is `Rajdeep`): no need to read `CONTRIBUTING.md`; the lead
  already knows the workflow. Read it only when asked or when editing it. (This overrides AGENTS.md §2 / §7 for the lead.)

## Profile
**profile: lean**: no tests, CI, Docker or devops files committed. Throwaway checks go in `.scratch/` (gitignored), run with `python .scratch/<file>.py`.

## Stack
- Language: Python 3.12 (conda env `fmnist`, deps via `pip install -r requirements.txt`)
- ML: scikit-learn, scikit-image (HOG), NumPy, pandas
- Plots: matplotlib
- Notebooks: Jupyter in VS Code (local, M3 Mac, 8 GB RAM); Colab as fallback
- No database, no API, no server

## Commands
```bash
conda create -n fmnist python=3.12 -y && conda activate fmnist
pip install -r requirements.txt                       # from phase 01 onwards
python -c "from src.data import download_raw; download_raw()"   # get + verify data
# then run notebooks/01 → 05 in order (Restart & Run All)
```

## Codebase layout
```
src/          config, utils, data, features, models_linear, models_tree_kernel, experiment, evaluate, results
notebooks/    01_data_eda … 05_results_analysis (one per phase)
data/splits/  val_indices.npy (committed); data/raw/, features/ are gitignored
results/      metrics/*.json, figures/*.png, summary.csv, test_log.csv (committed, code-generated)
docs/         prd, architecture, system-design, data-formats, module-interfaces, decisions, progress, phases/v1/
AGENTS.md     workflow + rules for every contributor and AI tool
CONTRIBUTING.md  step-by-step guide + troubleshooting for team members
.github/      pull_request_template.md
```

## Docs
Index: `docs/README.md`. Read `docs/progress.md` and the current phase file at session start;
read other docs only when the task touches them. `database-schema.md` / `api-contracts.md` from the standard kit are
renamed here to `data-formats.md` / `module-interfaces.md` (D-010).

Current status:
@docs/progress.md

## Project-specific rules (for Claude)
- **Team workflow overrides the solo flow:** phases are built by teammates on `phase/v1-0N-<slug>` branches and merged
  via GitHub PRs ("Create a merge commit"), not by local `git merge`. Claude may build a phase only when the user says so.
- **Reviewing a teammate's PR:** when asked, check the PR against, in this order:
  1. the phase file (every task done, every done criterion met, phase summary filled),
  2. the no-leakage rules (AGENTS.md §4, system-design.md §3),
  3. `module-interfaces.md` signatures and `data-formats.md` schemas,
  4. lean rules and simplicity (no extra deps, tests, CI or abstractions),
  5. that notebooks were run top-to-bottom, results are in the sanity ranges (prd.md §8), and decisions are logged,
  6. that the PR template is fully filled in.
  Report findings ranked by severity (leakage > wrong results > interface drift > style) and suggest exact fixes.
- Keep code simple enough for a student to explain at the viva. Never add infrastructure the phase file doesn't list.
