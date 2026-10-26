# Decision Log

Every non-trivial decision made while building this project, with the reasoning behind it.
Written so that someone reading it after the project is finished can understand **what** was built,
**why** it was built that way, and **how** the logic works.

Rules:
- Append new entries at the bottom. Never delete an entry; if a decision changes, add a new entry
  and mark the old one `Superseded by D-###`.
- Log as you go, not at the end of a phase.
- **Every entry fills every heading of the template** (AGENTS.md §6.3). Small choices keep each section short;
  architectural or tricky ones get the full teaching depth. Add a row to the Index for every entry.

## Index
| ID | Date | Phase | Title | Status |
|---|---|---|---|---|
| D-001 | 2026-10-24 | planning | Python + scikit-learn stack, run locally in VS Code Jupyter | Active |
| D-002 | 2026-10-24 | planning | Logic in small `src/` modules, thin notebooks | Active |
| D-003 | 2026-10-24 | planning | Fixed stratified 54k/6k split with committed val indices | Active |
| D-004 | 2026-10-24 | planning | Leakage controls: Pipelines, test guard, val-based selection | Active |
| D-005 | 2026-10-24 | planning | Tune on a single holdout val set; RBF SVM tuned on a 10k subset | Active |
| D-006 | 2026-10-24 | planning | Verify downloads with Zalando's official MD5 checksums | Active |
| D-007 | 2026-10-24 | planning | Settings as constants in `config.py`; HOG params in cache filenames | Active |
| D-008 | 2026-10-24 | planning | Model size = pickled bytes; inference time per 1,000 test images | Active |
| D-009 | 2026-10-24 | planning | Commit results, figures and notebook outputs; gitignore raw data and caches | Active |
| D-010 | 2026-10-24 | planning | Rename DB/API template docs to data-formats.md and module-interfaces.md | Active |
| D-011 | 2026-10-24 | planning | Store test predictions (y_pred) in each test JSON | Active |
| D-012 | 2026-10-24 | planning | Download raw files over HTTPS from the official GitHub repo | Active |
| D-013 | 2026-10-24 | planning | One owner per phase, delivered as a reviewed PR; AGENTS.md + PR template | Active |
| D-014 | 2026-10-24 | v1/01 | No `data/processed/` cache; load straight from the .gz files | Active |
| D-015 | 2026-10-24 | v1/01 | EDA design: train-only, 5k subsample, PCA→50 before t-SNE, notebook import cell | Active |
| D-016 | 2026-10-24 | v1/02 | HOG sweep proxy = SGD linear SVM with a fixed 30-epoch budget | Active |
| D-017 | 2026-10-24 | v1/02 | Final HOG config: orientations 12, cells 4×4, blocks 2×2 (1,728-d) | Active |
| D-018 | 2026-10-24 | v1/02 | Harness details: best = strict max val acc, sequential CV, test-log warning | Active |
| D-019 | 2026-10-24 | v1/02 | Test features are only built inside final_test_run() | Active |
| D-020 | 2026-10-24 | v1/03 | Increase LR iteration budget to reach convergence | Active |
| D-021 | 2026-10-24 | v1/03 | Reuse saved experiment results when restarting notebook 03 | Active |
| D-022 | 2026-10-24 | v1/03 | Plot LR coefficients in standardised-pixel coordinates | Active |
| D-023 | 2026-10-24 | v1/03 | LR validation rerun chronology and timing limitation | Active |
| D-024 | 2026-10-24 | v1/03 | Exclude contaminated LR timings from speed comparisons | Active |
| D-025 | 2026-10-24 | v1/03 | Post-review consistency: weight-map plot moved to notebook; result annotation rule | Active |
| D-026 | 2026-10-25 | v1/04 | LinearSVC dual optimization and convergence behavior across feature sets | Active |
| D-027 | 2026-10-25 | v1/04 | RBF SVM stratified 10k subset tuning and full 60k refit scaling | Active |
| D-028 | 2026-10-25 | v1/04 | Review fixes: reproducible figures, test guard, runtime notes, val and timing caveats | Active |
| D-029 | 2026-10-25 | v1/05 prep | Comparable val scores: refit subset-tuned RBF SVM on full train before choosing the best model | Active |
| D-030 | 2026-10-25 | v1/05 | Phase 05 aggregation: validation-selected model ranking and non-comparable timing rules | Active |
| D-031 | 2026-10-26 | v1/05 | Phase 05 split between two owners and two PRs | Active |
| D-032 | 2026-10-26 | v1/05 | Error analysis from saved predictions; end-to-end speed counts HOG extraction | Active |

---

## Entry template (copy below the last entry)

### D-### — <Short title>
- **Date:** YYYY-MM-DD
- **Phase:** NN — <phase name>
- **Status:** Active | Superseded by D-### | Reverted
- **Type:** stack | architecture | data | api | algorithm | edge-case | tooling | assumption

**Context**
What problem or question came up, and what constraints mattered.

**Decision**
What was chosen, in one or two sentences.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| | |

**Why**
The reasoning. What made this the right choice for this project at this stage.

**How it works**
Walk through the logic step by step in plain language, as if teaching it. Include a small code
snippet or pseudo-code if it helps. Mention the edge cases it handles.

**Files touched**
- `path/to/file.ts` — what changed

**Concepts worth learning**
Terms, patterns or ideas this decision relies on (e.g. "optimistic concurrency", "idempotency key"),
each with one line explaining it.

**Revisit if**
The conditions under which this decision should be reconsidered.

---

<!-- Entries start below -->

### D-001 — Python + scikit-learn stack, run locally in VS Code Jupyter
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** stack

**Context**
The capstone's Phase 1 must use classical ML only. The team has used Colab and Jupyter before, but hasn't
run notebooks locally. The machine is an M3 MacBook (8 cores, 8 GB RAM) with conda and VS Code installed.

**Decision**
Python 3.12 with NumPy, pandas, scikit-learn, scikit-image and matplotlib. Notebooks run locally in VS Code using a
conda environment. Colab is the documented fallback.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Google Colab as the main platform | Free tier has 2 slow CPUs, sessions time out and wipe state, and importing from `src/` needs a re-clone each session |
| Python 3.14 (already installed) | Newer than Colab (3.12); some wheels may lag. 3.12 keeps local and Colab identical |
| seaborn for plots | One more dependency for little gain; matplotlib does everything we need |
| uv / poetry for envs | Extra tooling to learn; conda is already installed and familiar |

**Why**
scikit-learn is the standard library for classical ML and has every model we need behind one consistent API
(`fit` / `predict`). scikit-image has a reference HOG implementation. Running locally is faster than free Colab
and never loses state.

**How it works**
`conda create -n fmnist python=3.12` → `conda activate fmnist` → `pip install -r requirements.txt` →
open the repo in VS Code → pick the `fmnist` kernel in the notebook. conda provides only the Python
interpreter; all libraries come from pip with pinned versions, so a teammate or Colab gets the same versions.

**Files touched**
- `requirements.txt`, `README.md` (written in phase 01)

**Concepts worth learning**
- Virtual environment: an isolated set of installed packages per project, so versions don't clash.
- Pinning: writing exact versions (`scikit-learn==1.x.y`) so results don't change when a library updates.
- Jupyter kernel: the Python process a notebook runs in; choosing the env's kernel means the notebook uses that env's packages.

**Revisit if**
Phase 2 CNN training is too slow locally. Then use Colab GPU for that phase only.

---

### D-002 — Logic in small `src/` modules, thin notebooks
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** architecture

**Context**
idea.md proposes `src/` files plus one notebook per workstream. We want simple code, no duplicated logic, and
notebooks that are easy to present.

**Decision**
All reusable logic (loading, features, models, tuning, evaluation, aggregation) lives in plain functions in `src/`.
Notebooks import those functions, run them, display plots and hold the written explanations.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Notebooks only | The same split/evaluate code would be copy-pasted into 5 notebooks. One copy drifts, and the results stop being comparable |
| Scripts only | Loses the visual, explainable format the team is used to and the professor expects |
| Config-driven experiment framework (YAML + CLI) | Too much machinery for 9 experiments. Adds learning time with no marks gained |

**Why**
One definition of `load_splits()` and `evaluate()` means every model is guaranteed to use the same data and metrics.
Notebooks stay short and readable, which makes viva preparation easier.

**How it works**
A notebook cell does `from src.experiment import tune_on_val, final_test_run`, then calls them. If the same
code shows up in two notebooks, it gets moved into `src/`. Model files only *build* unfitted sklearn Pipelines
(e.g. `make_knn(k=5, weights="distance")`). They never load data, so they are tiny.

**Files touched**
- `src/*.py`, `notebooks/*.ipynb` (created across phases 01–05)

**Concepts worth learning**
- Separation of concerns: each module has one job (data, features, models, evaluation).
- Factory function: a function that builds and returns a new object (here, an unfitted model) from parameters.

**Revisit if**
A notebook starts holding more than a few lines of non-plotting logic.

---

### D-003 — Fixed stratified 54k/6k split with committed val indices
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
Hyperparameters must be tuned without touching the official 10k test set, so we need a validation set carved from
the 60k training images. Everyone, on every machine, must get the identical split.

**Decision**
Take 6,000 validation images from the 60k train set with `train_test_split(..., stratify=y, random_state=42)`.
Save the val indices to `data/splits/val_indices.npy` and commit that file (~48 KB). Print an MD5 of it and record that value in the README.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Re-create the split from the seed each time | Usually identical, but a future sklearn/NumPy version could change the shuffle. A committed file can't drift |
| k-fold cross-validation for all tuning | 5× the compute for every model, and SVM would be impractical. See D-005 |
| Non-stratified random split | Class proportions in val could be slightly off. Stratifying guarantees exactly 600 per class |

**Why**
A committed index file is the simplest proof that the split is fixed. The professor can verify that val and test never overlap.

**How it works**
1. `load_splits()` checks whether `val_indices.npy` exists.
2. If not, it builds a stratified split of indices 0–59,999 (seed 42), sorts the val indices and saves them.
3. Train = every index not in val. Test = the official 10k, untouched.
4. A check asserts there are 54,000 train and 6,000 val indices, no overlap, and 600 images per class in val.

**Files touched**
- `src/data.py`, `data/splits/val_indices.npy` (phase 01)

**Concepts worth learning**
- Stratified sampling: sample so that each class keeps its original proportion in every subset.
- Validation vs test set: val is for making choices, test is for reporting the final number once.

**Revisit if**
Never in v1. Phase 2 must reuse this exact file.

---

### D-004 — Leakage controls: Pipelines, test guard, val-based selection
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** algorithm

**Context**
Data leakage (information from val/test sneaking into training or model choice) is the main way an ML project loses
credibility. The user explicitly asked for "no loopholes".

**Decision**
Close every known leak with the simplest mechanism:
1. Put any scaler inside an sklearn `Pipeline` with the model.
2. Only `experiment.final_test_run()` loads test features, and every call is logged to `results/test_log.csv`.
3. Pick the best HOG config, best hyperparameters and "best model" by **val** accuracy only.
4. Run EDA on the train split only.
5. Exact-duplicate images are counted and reported but **not removed**.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Fit `StandardScaler` once on train and save the scaled arrays (idea.md) | Correct for the holdout, but leaks in 5-fold CV (scaler sees all folds) and on the 60k refit unless redone by hand. A Pipeline handles every case automatically |
| Lock the test file behind a password / flag | Over-engineering. A single code path plus an audit log is enough to prove discipline |
| Remove duplicates | Would make our numbers incomparable with published Fashion-MNIST results |

**Why**
Each control is one or two lines of code but closes a specific question an examiner might ask
("was your scaler fit on test?", "how did you pick the best model?").

**How it works**
- `make_logreg(C)` returns `Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C))])`.
  Calling `.fit(X_train)` fits the scaler on X_train only. Calling `.predict(X_val)` only *applies* it.
- `tune_on_val()` takes only train/val arrays. It has no parameter through which test data could arrive.
- `final_test_run()` refits on train+val (60k) with the chosen params, evaluates on test once and appends
  `timestamp, experiment, accuracy` to `test_log.csv`. Duplicate rows for the same experiment would show a re-run.
- HOG has no fitted state (it's a fixed formula applied per image), so computing it on test cannot leak.

**Files touched**
- `src/models_*.py`, `src/experiment.py`, `src/data.py` (phases 01–04)

**Concepts worth learning**
- Data leakage: any path by which information from evaluation data influences training or model choice, inflating scores.
- sklearn Pipeline: chains preprocessing and model so `fit` and `predict` apply each step correctly.
- Stateless vs fitted transform: HOG/flatten have no learned parameters, while a scaler learns mean/std from data.

**Finding (2026-10-24, phase 01):** `run_sanity_checks()` found **0** exact duplicates inside train, 0 inside test,
and 0 test images that also appear in train (MD5 hash per image). So rule 5 (keep duplicates) changes nothing in practice,
and there is no image-level train/test leakage in the official split.

**Revisit if**
A new preprocessing step that learns from data (e.g. PCA) is added. It must go inside the Pipeline.

---

### D-005 — Tune on a single holdout val set; RBF SVM tuned on a 10k subset
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** algorithm

**Context**
We need to tune about 9 models × several hyperparameters on an 8 GB laptop. RBF-SVM training time grows roughly with the square
of the sample count, so tuning it on 54k samples would take hours.

**Decision**
Tune every model by fitting on the 54k train split and scoring on the 6k val split, looping over a small grid.
The RBF SVM is tuned on a stratified 10k subset of train (scored on the same 6k val). The final models are all refit
on the full 60k. 5-fold stratified CV is run once, on the best config, to report mean ± std.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| `GridSearchCV` with 5-fold CV for every model | 5× compute. SVM/kNN become impractical on a laptop |
| `GridSearchCV` + `PredefinedSplit` | Same result as our loop but harder to read and explain |
| Tune SVM on PCA-reduced pixels | Changes the feature set being compared. Kept as a parked idea |

**Why**
With 6,000 val images, one percentage point of accuracy is 60 images, which is stable enough to rank configs.
A plain `for params in ParameterGrid(grid)` loop is easy to read, prints progress, and gives the data for the
validation curves directly.

**How it works**
```
for params in ParameterGrid(grid):
    model = make_fn(**params); model.fit(X_train, y_train)
    rows.append({**params, "val_acc": accuracy(y_val, model.predict(X_val)), "fit_s": t})
best = row with highest val_acc  (ties → simpler/faster model)
```
For RBF SVM, `X_train` is replaced by a stratified 10k subset drawn with seed 42.

**Files touched**
- `src/experiment.py` (phase 02), used in phases 03–04

**Concepts worth learning**
- Holdout validation: a single fixed slice of data reserved for comparing configurations.
- Validation curve: val score plotted against one hyperparameter, showing under- vs overfitting.
- Refit: after choosing params, retrain on all available training data so the final model sees as much data as possible.

**Revisit if**
Two top configs are within ~0.3% on val. Then confirm with CV before choosing.

---

### D-006 — Verify downloads with Zalando's official MD5 checksums
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
The PRD draft said SHA-256, but Zalando publishes MD5 checksums for the four files in the official repository README.
We should verify against values published by the source, not values we computed ourselves.

**Decision**
Hard-code the four official MD5 values in `config.py`. `download_raw()` checks each file after download (and on every
run) and stops with an error on a mismatch. PRD F-01 was updated to match.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| SHA-256 computed by us on first download | Only proves the file didn't change after *our* download, not that it was correct |
| No check | A truncated download would give silently wrong data |
| `tensorflow`/`torchvision` dataset loaders | Pulls a huge dependency into a classical-ML phase |

**Why**
MD5 is fine for catching corruption or wrong files. We are not defending against an attacker forging a file,
and these are the checksums the dataset authors publish.

**How it works**
Stream each file through `hashlib.md5()` and compare the hex digest to the expected value. On a mismatch, delete the
file and raise `RuntimeError("checksum mismatch for <file>: re-run to download again")`. The expected values are copied
from github.com/zalandoresearch/fashion-mnist during phase 01 and verified against the actual download.
**Confirmed 2026-10-24 (phase 01):** all four official MD5s matched the real download, and a truncated file was correctly
detected and deleted. Download over HTTPS from GitHub took ~5 minutes in total.

**Files touched**
- `src/config.py`, `src/data.py` (phase 01); `docs/prd.md` F-01

**Concepts worth learning**
- Checksum / hash: a short fingerprint of a file; any change in the file changes the fingerprint.

**Revisit if**
The official mirror moves. Then update the URL in `config.py`; the checksums stay the same.

---

### D-007 — Settings as constants in `config.py`; HOG params in cache filenames
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
idea.md proposed `config.yaml`. We want one place for the seed, paths and HOG settings, with as few dependencies as possible.

**Decision**
Use a plain Python module `src/config.py` with UPPER_CASE constants. After the HOG sweep in phase 02, the chosen params
are written into `config.HOG_PARAMS` by hand and committed. Feature cache filenames encode the HOG params.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| `config.yaml` + PyYAML | An extra dependency and a parsing step. YAML turns tuples into lists, which needs conversion for `pixels_per_cell` |
| Auto-writing the best HOG config from code | Hides a methodological choice. Picking it by hand from the sweep table is clearer to explain |

**Why**
`from src.config import SEED` is the simplest possible config. Putting params in the cache filename prevents a subtle
bug where old HOG features are reused after params change.

**How it works**
`hog_cache_name(split)` → `f"hog_o{o}_p{p}_c{c}_{split}.npy"` from `HOG_PARAMS`. If the params change, the filename
changes, the cache misses, and the features are rebuilt.

**Files touched**
- `src/config.py` (phase 01), `src/features.py` (phase 02)

**Concepts worth learning**
- Cache invalidation: making sure cached results are thrown away when their inputs change; encoding inputs in the key does this automatically.

**Revisit if**
We need to run many configurations from the command line (not planned).

---

### D-008 — Model size = pickled bytes; inference time per 1,000 test images
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** algorithm

**Context**
RQ3 compares accuracy against speed and size. idea.md measured size differently per model (params, support vectors,
nodes, stored samples), and those numbers can't be put on one axis.

**Decision**
Model size = `len(pickle.dumps(fitted_pipeline))` in bytes, the same for every model. The model-specific count
(support vectors, tree nodes, stored samples, coefficients) is *also* recorded as a secondary field for the write-up.
Inference time = wall-clock time of `predict()` on the full 10k test set, reported as ms per 1,000 images, best of 3 runs.
Train time = wall-clock time of the final 60k fit.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Only model-specific counts | Not comparable across model families |
| `joblib.dump` to disk and measure the file | Same number, but slower and leaves files behind |

**Why**
A serialized size is what a deployment would actually store, so it's fair to compare across models. It also
makes kNN's weakness visible (it stores the whole training set).

**How it works**
`evaluate()` times `model.predict(X_test)` 3 times with `time.perf_counter()` and keeps the minimum (it's the least
affected by background noise). It divides by 10 to get ms per 1,000 images and pickles the model to count its bytes. The
hardware/version info is written into the JSON next to the timings.

**Files touched**
- `src/evaluate.py` (phase 02)

**Concepts worth learning**
- Serialization (pickling): turning an in-memory object into bytes so it can be saved or sent.
- Wall-clock vs CPU time: real elapsed time, which is what a user waiting on predictions experiences.

**Revisit if**
Phase 2 adds GPU models. Then report CPU and GPU timings separately.

---

### D-009 — Commit results, figures and notebook outputs; gitignore raw data and caches
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
The professor will look at the GitHub repo. Raw data and feature caches are large and can be rebuilt; results are small and are the actual deliverable.

**Decision**
Commit `results/` (metrics JSON, `summary.csv`, `test_log.csv`, PNG figures at 300 dpi), `data/splits/val_indices.npy`
and notebooks **with their outputs**. Gitignore `data/raw/`, `data/processed/` and `features/`.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Strip notebook outputs before committing | Cleaner diffs, but the professor would then have to run everything just to see results |
| Commit feature caches | Hundreds of MB. They're rebuilt by code in minutes |

**Why**
Anyone opening the repo on GitHub sees the results immediately, and anyone cloning it can regenerate them.

**How it works**
`.gitignore` lists the three big folders. Each notebook is run top-to-bottom ("Restart & Run All") before
being committed, so its outputs match the current code.

**Files touched**
- `.gitignore` (phase 01)

**Concepts worth learning**
- Build artefacts vs source: generated files that can be rebuilt usually stay out of git; deliverables go in.

**Revisit if**
The repo grows past ~50 MB because of figures.

---

### D-010 — Rename DB/API template docs to data-formats.md and module-interfaces.md
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
The standard docs kit has `database-schema.md` and `api-contracts.md`. This project has no database and no API.
The user pointed out that a developer would open those files expecting a DB and endpoints, and be misled.

**Decision**
Rename them to `data-formats.md` (the files the pipeline saves and what's inside them) and `module-interfaces.md`
(the inputs/outputs of the shared `src/` functions). Update `docs/README.md` and the phase template to match.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Keep template names, explain inside | The filename is the first thing people see; a misleading name costs every reader time |
| Delete both files | We still need one place that pins down the metrics JSON schema and the function signatures shared across phases |

**Why**
File names should describe their content. The content plays the same role as a DB schema and API contract
(a fixed agreement between parts), just for files and functions instead of tables and endpoints.

**Files touched**
- `docs/data-formats.md`, `docs/module-interfaces.md` (renamed), `docs/README.md`, `docs/phases/_phase-template.md`

**Revisit if**
A real database or API is ever added (e.g. a demo app in Phase 2). Then add those docs alongside.

---

### D-011 — Store test predictions (y_pred) in each test JSON
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
Phase 05 needs misclassified-image grids and a "do the best two models make the same mistakes?" check.
Both need each model's per-image predictions. Retraining models in phase 05 would be slow and would mean
running the test set a second time.

**Decision**
`evaluate()` saves the 10,000 test predictions as `y_pred` in `{name}_{kind}_test.json`.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Save fitted models (`.joblib`) and re-predict | kNN/SVM files are tens of MB each; re-predicting touches test again |
| Separate `.npy` file per model | One more file type to track for ~30 KB of data |

**Why**
Error analysis becomes pure reading of saved results, with no retraining and no extra test-set use.

**How it works**
`y_pred = model.predict(X_test)` is computed once inside `evaluate()` (the same call that's timed), converted to a list of ints
and written with the other metrics. Phase 05 loads it next to `y_test` from `load_splits()`, and `y_pred != y_test`
gives the mistakes.

**Files touched**
- `src/evaluate.py` (phase 02), `src/results.py` (phase 05), `docs/data-formats.md` §8

**Revisit if**
Phase 2 needs prediction probabilities (e.g. for ensembling). Then add `y_proba` the same way.

---

### D-012 — Download raw files over HTTPS from the official GitHub repo
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
The common Fashion-MNIST mirror is a plain-HTTP S3 website. The same files are also hosted in the official
GitHub repository and served over HTTPS.

**Decision**
`RAW_URL_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"`. Files are verified by MD5 (D-006).

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| S3 HTTP mirror | Unencrypted; works, but HTTPS from the official repo is the better default |

**Why**
Same official files, encrypted transport, and a URL anyone can check in the browser.

**Files touched**
- `src/config.py`, `src/data.py` (phase 01)

**Revisit if**
GitHub rate-limits or blocks the download. Then fall back to the S3 mirror (same MD5s).

---

### D-013 — One owner per phase, delivered as a reviewed PR; AGENTS.md + PR template
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
The 5 phases will be built by 5 different team members. Some use Claude Code, others use other AI tools
(e.g. Antigravity) or none. The repo owner reviews every phase before it reaches `main`.

**Decision**
Each phase is built on its own branch by its owner, then opened as a GitHub PR into `main` using a fixed PR template,
reviewed by the repo owner (with Claude Code), and merged with a merge commit. `AGENTS.md` holds the tool-neutral rules
and workflow. The project `CLAUDE.md` imports it, so every tool follows the same rules.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Rules only in CLAUDE.md | Teammates without Claude would never see them |
| Separate rule files per tool (.cursorrules, etc.) | Copies drift apart; AGENTS.md is a widely read convention and one source |
| Squash-merge PRs | Hides individual commits, but commit history is evidence of each member's contribution |
| CI checks on PRs | Lean profile forbids CI; the PR checklist + owner review replaces it |

**Why**
A phase file says *what* to build. AGENTS.md says *how to work* (setup, leakage rules, branch/commit/PR steps). The PR
template makes every PR summarise its results and tick the leakage checklist, so review is fast and consistent for a human or for Claude.

**How it works**
1. Owner pulls `main`, creates `phase/v1-0N-<slug>`, marks the phase `in-progress` (AGENTS.md §6.1).
2. One commit per task, pushed each time; decisions logged as they happen.
3. Owner closes the phase (done criteria, phase summary, progress → `review`) and opens a PR. GitHub pre-fills the template.
4. Repo owner reviews with Claude (CLAUDE.md lists the review order), the owner fixes with new commits, then it's merged.
5. Next phase starts from the updated `main`.

**Files touched**
- `AGENTS.md`, `.github/pull_request_template.md`, `CLAUDE.md`, `docs/progress.md` (Owner column), all `docs/phases/v1/*.md` (Owner row, AGENTS.md pointer)

**Concepts worth learning**
- Pull request: a request to merge a branch, where code is reviewed and discussed before it reaches `main`.
- Merge commit vs squash: a merge commit keeps every individual commit; squash collapses them into one.

**Revisit if**
Phases need to run in parallel. Then split shared code (harness) earlier, or let later phases branch from an unmerged branch.

---

### D-014 — No `data/processed/` cache; load straight from the .gz files
- **Date:** 2026-10-24
- **Phase:** v1/01 — Foundation & Data
- **Status:** Active
- **Type:** data

**Context**
`architecture.md` planned a `data/processed/` folder of cached `.npy` arrays. In practice `load_raw()` reads and
decompresses all 70k images in well under a second.

**Decision**
Drop `data/processed/`. `load_splits()` always reads the verified `.gz` files and applies the committed split.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Cache float32 arrays in `data/processed/` | Saves <1 s, adds a second copy of the data that could go stale |

**Why**
One source of truth (the checksummed raw files + `val_indices.npy`) is simpler and cannot drift.

**Files touched**
- `.gitignore`, `docs/architecture.md`, `docs/phases/v1/phase-01-foundation.md`

**Revisit if**
Loading ever becomes slow (it won't for 70k 28×28 images). Feature caching in phase 02 is a separate thing and stays.

---

### D-015 — EDA design: train-only, 5k subsample, PCA→50 before t-SNE, notebook import cell
- **Date:** 2026-10-24
- **Phase:** v1/01 — Foundation & Data
- **Status:** Active
- **Type:** algorithm

**Context**
The EDA has to show which classes overlap, without leaking information from val/test, and run quickly on a laptop.

**Decision**
- EDA uses only `X_train`/`y_train`. The notebook deletes the split dict right after taking the train arrays.
- PCA is fit on all 54k train images (explained-variance curve). The 2-D PCA plot and t-SNE use the **same 5k random subsample** (seed 42).
- t-SNE runs on the first 50 PCA components (perplexity 30, `init="pca"`, seed 42). It takes ~10 s on the M3.
- Class similarity = cosine similarity between class-mean images (model-free).
- Every notebook starts with a small cell that adds the repo root to `sys.path`, because VS Code runs notebooks from `notebooks/`.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| t-SNE on raw 784-d pixels | Slower and noisier; PCA→50 is standard practice |
| t-SNE on all 54k | Minutes instead of seconds and an unreadable plot |
| `pip install -e .` package / VS Code settings file | More setup for teammates than a 3-line cell |

**Why**
These are the standard, explainable choices. The subsample keeps plots readable. Train-only keeps the "no leakage" story clean.

**How it works**
Cosine similarity of two mean images = dot product of the flattened images divided by the product of their lengths.
1.0 means the same shape (ignoring overall brightness). Results: Coat–Shirt 0.987, Pullover–Coat 0.984,
Pullover–Shirt 0.983, T-shirt–Shirt 0.950. 84 PCA components keep 90% of the variance, and 188 keep 95%.
To execute the notebook outside VS Code during development, `nbformat`/`nbclient` were installed into the local env only. They are
not project dependencies.

**Files touched**
- `notebooks/01_data_eda.ipynb`, `results/figures/eda_*.png`, `AGENTS.md` (import cell rule), `README.md` (interpreter tip)

**Concepts worth learning**
- PCA: rotates the data onto the directions of largest variance; keeping the first k gives the best k-dim linear summary.
- t-SNE: non-linear map that keeps close neighbours close; cluster *distances* and *sizes* on the plot are not meaningful.
- Cosine similarity: compares direction (shape), not magnitude (brightness).

**Revisit if**
Phase 05's confusion matrices disagree with the EDA's predicted hard classes. Then discuss why in the error analysis.

---

### D-016 — HOG sweep proxy = SGD linear SVM with a fixed 30-epoch budget
- **Date:** 2026-10-24
- **Phase:** v1/02 — Features & Evaluation Harness
- **Status:** Active
- **Type:** algorithm

**Context**
The plan was to score each of the 12 HOG configs with a scaled `LinearSVC`. On 54k × 1,728 features the exact
(liblinear, primal) solver had not finished **one** config after 10 minutes, so all 12 would take hours.

**Decision**
Use `Pipeline(StandardScaler, SGDClassifier(loss="hinge", alpha=1e-4, max_iter=30, tol=None, random_state=42))` as the proxy.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| `LinearSVC` (exact solver) | > 10 min per config, hours for the sweep |
| `LinearSVC` on a 10k subset | Faster, but ranks configs on less data than the real models get |
| `LogisticRegression` (lbfgs) | Also slow on 1,700-d features with 54k rows |
| SGD with the default stopping rule | Emits a `ConvergenceWarning` (SGD keeps wobbling) and stops at different epochs per config |

**Why**
The proxy's job is only to **rank** HOG configs fairly, not to be a final model. SGD with hinge loss optimises the same
objective as a linear SVM, takes ~25 s per config, and a fixed 30-epoch budget gives every config identical treatment
(no early stopping that could favour one config).

**How it works**
For each config: extract HOG for the 54k train and 6k val images → fit scaler + SGD on train → accuracy on val.
`tol=None` turns off the convergence check, so it always runs exactly 30 passes over the data. Seeded, so it's repeatable.
The whole sweep takes ~2.5 minutes on the M3.

**Files touched**
- `src/features.py`, `docs/data-formats.md` §6, `docs/system-design.md` Flow B, phase-02 file

**Concepts worth learning**
- Stochastic gradient descent: update the model after small random batches instead of solving exactly; fast but noisy.
- Hinge loss: the SVM loss, which is 0 for points beyond the margin and grows linearly for points inside it or misclassified.
- Proxy model: a cheap model used only to compare *inputs* (here feature configs), not to report final accuracy.

**Revisit if**
The best two configs are within noise of each other and the choice matters. Then re-check them with a stronger model.

---

### D-017 — Final HOG config: orientations 12, cells 4×4, blocks 2×2 (1,728-d)
- **Date:** 2026-10-24
- **Phase:** v1/02 — Features & Evaluation Harness
- **Status:** Active
- **Type:** algorithm

**Context**
Sweep results (`results/hog_sweep.csv`, proxy val accuracy): o12_p4_c2 **0.8818** (1,728-d), o9_p4_c2 0.8783 (1,296-d),
o6_p4_c2 0.8757 (864-d) … down to o6_p7_c1 0.7612 (96-d). Extraction takes ~0.22 s per 1,000 images for 4×4 cells.

**Decision**
`HOG_PARAMS = {"orientations": 12, "pixels_per_cell": (4, 4), "cells_per_block": (2, 2)}`.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| o9_p4_c2 (the classic Dalal–Triggs choice, 1,296-d) | 0.35 points lower on val. That's above the 0.3-point tie threshold set in D-005, so the pre-stated rule picks o12 |
| 7×7 cells | 4–8 points worse: at 28×28 a 7×7 cell covers a quarter of the image width and blurs away the shape |

**Why**
The rule fixed in advance was "highest val accuracy, ties within 0.3 points go to the simpler config". o12_p4_c2 wins by
0.35 points. The extra dimensions (1,728 vs 784 for pixels) make kNN/SVM somewhat slower but stay manageable.

**How it works**
The sweep's three clear patterns:
1. **Smaller cells win** (4×4 ≫ 7×7). Fine edge detail is what separates classes at 28×28.
2. **Block normalisation helps** (2×2 > 1×1 by 2–3 points). Normalising over neighbouring cells makes the
   gradients robust to local contrast.
3. **More orientation bins help a little** (12 > 9 > 6). Finer angle resolution separates e.g. sleeves from torso edges.

**Files touched**
- `src/config.py`, `results/hog_sweep.csv`

**Concepts worth learning**
- HOG: per cell, a histogram of gradient directions weighted by gradient strength; blocks of cells are normalised and concatenated.
- Feature dimension for 28×28, cell c, block b, o bins: ((28/c) − b + 1)² × b² × o, e.g. 6² × 4 × 12 = 1,728.

**Revisit if**
Phases 03–04 show HOG is far slower than pixels for kNN/SVM. Then o9_p4_c2 is the documented fallback (−0.35 points).

---

### D-018 — Harness details: best = strict max val acc, sequential CV, test-log warning
- **Date:** 2026-10-24
- **Phase:** v1/02 — Features & Evaluation Harness
- **Status:** Active
- **Type:** algorithm

**Context**
Building `experiment.py` raised a few small choices that phases 03–05 rely on.

**Decision**
1. `tune_on_val` returns the combo with the **highest val accuracy**. On an exact tie, the first combo in
   `ParameterGrid` order wins. D-005's "within 0.3 points → prefer the simpler model" is **not automated**: if the top
   two are that close, the phase owner may pick the simpler one *by hand* and must log it here.
2. `cross_validate_best` runs folds **one after another** (`n_jobs=1`), because RF/kNN already use `N_JOBS=4` inside.
   Running both in parallel would start 16 processes and could run out of memory on 8 GB.
3. `final_test_run` always runs, but `_log_test_use` prints a **WARNING** when an experiment already appears in
   `test_log.csv`. Re-runs are visible, never hidden.
4. `utils.display_path()` prints paths relative to the repo, or in full when outside it (e.g. features on Google Drive in Colab).
   Without it, saving outside the repo crashed while printing the message (found by the harness check).
5. `stratified_subset()` is shared by SVM tuning and CV subsets, so both use the same seeded, class-balanced sampling.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Automatic "simplest within 0.3 points" rule | Needs a per-model definition of "simpler"; a logged human choice is clearer |
| Block a second `final_test_run` | A real bug fix may need a re-run; a warning plus a mandatory explanation is more honest |
| Parallel CV (`n_jobs=-1`) | Memory risk on 8 GB with models that are already parallel |

**How it works**
`tune_on_val` gets data only via `features.get_features(kind, "train" | "val")`. The scratch check replaced
`get_features` with a version that raises on `"test"`, and tuning ran fine, which proves it never reads test.
The same check confirmed that the scaler's learned mean equals the training rows' mean, not the val mean.

**Files touched**
- `src/experiment.py`, `src/utils.py`, `src/data.py`, `src/features.py`, `docs/module-interfaces.md`

**Revisit if**
Phase 2's CNN needs a different training loop. It should still finish by calling `evaluate()` and logging the test use.

---

### D-019 — Test features are only built inside final_test_run()
- **Date:** 2026-10-24
- **Phase:** v1/02 — Features & Evaluation Harness
- **Status:** Active
- **Type:** edge-case

**Context**
The phase 02 plan said to cache features "for all 4 splits". But AGENTS.md rule 1 says nothing may call
`get_features(..., "test")` outside `final_test_run()`.

**Decision**
Notebook 02 caches train / val / trainval only. The test cache is created automatically the first time
`final_test_run()` asks for it (pixels: instant; HOG: ~3 s).

**Why**
HOG and flattening are stateless, so building test features early would not leak anything. But keeping the rule
**absolute** ("test is read in exactly one function") makes it trivial to audit, both for the professor and for
reviewers of teammates' PRs. The cost is a few seconds in phase 03.

**Files touched**
- `notebooks/02_features.ipynb`, phase-02 done criteria

**Revisit if**
Never for v1.

### D-020 — Increase LR iteration budget to reach convergence
- **Date:** 2026-10-24
- **Phase:** v1/03 — Dummy, Logistic Regression & kNN (author: Rajat; expanded to full template by the lead in D-025)
- **Status:** Active
- **Type:** algorithm

**Context**
Initial validation fits for pixel LR at C=1 and C=10 emitted `ConvergenceWarning` at the planned 1,000 iterations.

**Decision**
Use `config.LOGREG_MAX_ITER = 5000` for all LR fits (following system-design.md §7: "raise max_iter, don't hide the
warning"), and rerun the full validation grid. The chosen C stayed 0.01 for both feature sets. See D-023 for the timing of runs.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Keep 1,000 and ignore the warning | An unfinished optimisation makes large-C points look worse than they are, distorting the validation curve |
| Suppress the warning | Hides a real problem; the project rule is to keep warnings visible |
| Drop C=1 / C=10 from the grid | Changes the planned grid just to avoid a warning |
| Switch solver (e.g. `saga`) | A bigger change than needed; lbfgs converges with more iterations |

**Why**
Raising the budget keeps the model, solver, seed and C grid identical and only lets the optimiser finish.

**How it works**
Logistic Regression is trained by an iterative optimiser (lbfgs) that improves the weights step by step until changes are
tiny. With weak regularisation (large C) the loss surface is flatter, so it needs more steps. `max_iter` is just the
step limit; 5,000 is enough for every grid point here. Small-C models (the winners) had already converged at 1,000.

**Files touched**
- `src/config.py`, `src/models_linear.py`, `docs/module-interfaces.md`

**Concepts worth learning**
- Convergence: the optimiser has stopped improving the loss meaningfully; before that, the weights are not the true optimum.
- C in LR: inverse regularisation strength. Small C = strong penalty on large weights = simpler model.

**Revisit if**
A future LR (e.g. Phase 2 features) warns at 5,000. Then consider scaling checks or another solver.

---

### D-021 — Reuse saved experiment results when restarting notebook 03
- **Date:** 2026-10-24
- **Phase:** v1/03 — Dummy, Logistic Regression & kNN (author: Rajat; expanded by the lead in D-025)
- **Status:** Active
- **Type:** edge-case

**Context**
Notebook 03 must run top-to-bottom ("Restart & Run All") with outputs, yet `test_log.csv` must contain exactly one
test evaluation per experiment. Calling `final_test_run` unconditionally would re-test on every restart.

**Decision**
The notebook reads saved validation/test JSONs if present. A missing validation file runs the full grid. A missing test
file calls `final_test_run` once, **unless** the audit log already lists that experiment (then it stops and asks you to
recover the metrics). A saved test result whose params differ from the val choice raises an error.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Always re-run tuning and testing | Every restart would add test-log rows (repeat test use) |
| Split into a "run" notebook and a "report" notebook | Two notebooks per phase; more for teammates to follow |
| Cache fitted models to disk | kNN models are 189–415 MB; not committable |

**Why**
Restarting becomes safe and figures stay reproducible, while the append-only log still exposes any deliberate re-run.

**How it works**
For each experiment: `if test JSON exists → load it and check its params match the val choice; elif experiment is in
test_log.csv → stop (metrics lost, investigate); else → final_test_run()`. The last cell asserts each of the six
experiments appears in the log exactly once.

**Files touched**
- `notebooks/03_linear_knn.ipynb`

**Concepts worth learning**
- Idempotent step: running it twice has the same effect as running it once. Here, "Restart & Run All" is made idempotent for test use.

**Revisit if**
Phases 04–05 need the same pattern. Reuse it (copy the guard) rather than inventing a new one.

---

### D-022 — Plot LR coefficients in standardised-pixel coordinates
- **Date:** 2026-10-24
- **Phase:** v1/03 — Dummy, Logistic Regression & kNN (author: Rajat; expanded by the lead in D-025)
- **Status:** Active
- **Type:** algorithm

**Context**
`final_test_run` returns metrics, not the fitted model, but phase 03 needs per-class weight maps for pixel LR.

**Decision**
Refit the validation-selected pixel LR Pipeline on `trainval` just for the figure (no test data loaded or predicted).
Plot the classifier's coefficients as they are, i.e. "per standardised pixel", with one symmetric colour scale for all 10 classes.
(The code lives in notebook 03; it was moved out of `models_linear.py` in D-025.)

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Return the fitted model from `final_test_run` | Changes the shared harness interface used by every phase |
| Divide coefficients by `scaler.scale_` (raw-brightness units) | Inflates rarely-used border pixels with tiny variance; maps look noisy |
| One colour scale per class | Makes classes look equally "confident"; a shared scale keeps them comparable |

**Why**
Keeps the harness unchanged and the maps honest and comparable.

**How it works**
With a StandardScaler in front, the model sees z = (pixel − mean) / std. Coefficient w[c, j] is how much one standard
deviation more ink at pixel j raises the score for class c. Red = evidence for the class, blue = against.
Reshaping each class's 784 weights to 28×28 shows the "template" the linear model looks for (e.g. two legs for Trouser).

**Files touched**
- `notebooks/03_linear_knn.ipynb`, `results/figures/linear_logreg_weights.png`

**Concepts worth learning**
- Linear model as template matching: the score is a dot product of the image with a weight image per class.
- Standardised feature: feature minus its mean, divided by its standard deviation (mean 0, std 1).

**Revisit if**
The report needs raw-pixel interpretation. Then plot w / scale_ with clipping and say so.

---

### D-023 — LR validation rerun chronology and timing limitation
- **Date:** 2026-10-24
- **Phase:** v1/03 — Dummy, Logistic Regression & kNN (author: Rajat; timestamps added by the lead in D-025)
- **Status:** Active
- **Type:** tooling

**Context**
Both complete initial LR grids (max_iter 1,000) selected C=0.01, and those selected fits had converged. Only pixel C=1
and C=10 warned. A convergence-only rerun of the grids at 5,000 iterations (D-020) was still finishing when the final LR
test runs started.

**Exact timestamps (from the committed files):**
| Experiment | `_val.json` written | `test_log.csv` row |
|---|---|---|
| logreg_pixels | 22:06:29 | 22:07:55 (after val) |
| logreg_hog | **22:08:45** | **22:08:43 (2 s before the rerun's val file)** |

**Decision**
Keep the original validation-selected C=0.01 for both final tests and their single recorded evaluations. The completed
5,000-iteration sweep independently selected the same C for both. Do not re-run test evaluations to "fix" timings.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Re-run the LR final tests after the rerun | Creates a second test use per experiment, which is worse for the audit than a documented overlap |
| Delete and redo the val files | Rewrites history; the overlap would just be hidden |

**Why**
No test result influenced any hyperparameter: C was fixed by complete validation sweeps before testing, and the rerun
confirmed it. The overlap only affects **timing** (two processes competing for CPU), which D-024 handles.

**How it works**
An examiner comparing timestamps will see that logreg_hog's test row predates its val file by 2 s. This entry, the
identical `best_params` in both files, and the D-024 flag together explain it.

**Files touched**
- `docs/decisions.md`, phase 03 handoff and PR description

**Concepts worth learning**
- Audit trail: a record that lets someone else reconstruct what happened and in what order; disclosure beats tidying.

**Revisit if**
Never. Future phases should finish all tuning before starting any `final_test_run` (now in AGENTS.md §4).

---

### D-024 — Exclude contaminated LR timings from speed comparisons
- **Date:** 2026-10-24
- **Phase:** v1/03 — PR review fix (author: Rajat; expanded by the lead in D-025)
- **Status:** Active
- **Type:** data

**Context**
D-023 disclosed timing contention in prose, but phase 05 could still read the numeric timing fields as normal measurements.

**Decision**
Add `timing_comparable: false` and a `timing_note` to both LR test JSONs, keeping every original field (raw timings,
predictions, timestamps). The flag is optional and means "true" when absent. Phase 05 must write NaN for both timing columns
of flagged experiments in `summary.csv`, exclude them from speed plots/rankings and name them in a note, while keeping
their accuracy results. Notebook 03 shows a caution next to the raw timings.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Leave it as prose only | Easy for phase 05 code to miss |
| Re-time by predicting on test again | Another test-set use |
| Re-time on the validation set | Different data size and protocol than every other model's timing |

**Why**
A machine-readable flag plus a written aggregation rule stops these numbers from silently entering RQ3, without
repeating any test evaluation or inventing replacement measurements.

**How it works**
`build_summary()` (phase 05) reads `metrics.get("timing_comparable", True)`. If false, it sets `train_time_s` and
`inference_ms_per_1000` to NaN in the summary row. Plots drop NaN timings and list the excluded experiments.

**Files touched**
- `results/metrics/logreg_pixels_test.json`, `results/metrics/logreg_hog_test.json` (annotation only),
  `docs/data-formats.md` §8 and §11, phase 03/05 files, notebook 03

**Concepts worth learning**
- Provenance: keep the raw observation and record why it's excluded, rather than deleting or overwriting it.

**Revisit if**
LR timings are essential for the report. Then re-time LR on a separate, clearly documented benchmark run that doesn't count as a test evaluation.

---

### D-025 — Post-review consistency: weight-map plot moved to notebook; result annotation rule
- **Date:** 2026-10-24
- **Phase:** v1/03 — post-merge follow-up by the lead
- **Status:** Active
- **Type:** architecture

**Context**
The PR #1 audit found `plot_logreg_weights()` in `src/models_linear.py`. It loaded data, trained a model and drew a figure, but
architecture.md §5 says model files must not load data or train. The late commit for D-024 also edited two result JSONs,
while architecture.md §4 says `results/` is written only by code.

**Decision**
1. Move the weight-map code into notebook 03 (same logic, same output); `models_linear.py` again only builds Pipelines and holds grids.
2. Allow exactly one kind of manual edit in `results/`: **adding** annotation fields to a JSON (never changing or
   removing values), and only with a decisions.md entry. Recorded in architecture.md.
3. Expand D-020–D-024 to the full decision template, and add the exact timestamps to D-023.
4. Make the decision-log format mandatory and visible to every agent (AGENTS.md §6.3 and §7, PR template).
5. Add a rule: finish **all** tuning for a model before running its `final_test_run` (prevents the D-023 situation).

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Send the PR back to the author | Slower; the lead chose to merge and fix directly |
| Move the plot to `src/results.py` | That module belongs to phase 05; creating it now would pre-empt that phase |

**How it works**
Notebook 03 was re-run top to bottom after the move: 0 errors, `test_log.csv` byte-identical (SHA-256 prefix
`513b16e7…`), all test JSONs unchanged, and the weight-map PNG identical.

**Files touched**
- `src/models_linear.py`, `notebooks/03_linear_knn.ipynb`, `docs/module-interfaces.md`, `docs/architecture.md`,
  `docs/decisions.md`, `AGENTS.md`, `CONTRIBUTING.md`, `.github/pull_request_template.md`, phase 03/04/05 files, `docs/progress.md`

**Concepts worth learning**
- Module boundary: a rule about what a file may do; it keeps code easy to find and review.

**Revisit if**
Phase 05 needs the weight maps again. Then it can read the PNG or move the code into `results.py` as part of its own scope.

---

### D-026 — LinearSVC dual optimization and convergence behavior across feature sets
- **Date:** 2026-10-25
- **Phase:** v1/04 — Random Forest & SVM (author: Tanmay Singh)
- **Status:** Active
- **Type:** algorithm

**Context**
LinearSVC optimization on 54k/60k images with $d=784$ (pixels) and $d=1,728$ (HOG) poses significant dual vs primal convergence trade-offs as $C$ increases. In Phase 02, exact LinearSVC on HOG was flagged as slow.

**Decision**
Configure `make_linear_svm` with `dual="auto"` (scikit-learn then solves the **primal** problem, i.e. `dual=False`, because $n_{\text{samples}} > n_{\text{features}}$: $54,000 > 784$ and $54,000 > 1,728$), keeping the default `max_iter=1000`. Sweep $C \in [0.01, 0.1, 1.0]$. Both feature sets chose $C=0.01$ as the validation winner, which was also the fastest fit (105 s on pixels, 151 s on HOG during tuning on 54k).

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Fixed `dual=True` | The dual problem has one variable per training image (54,000); liblinear recommends the primal solver when there are far more samples than features |
| Subsetting training set | Unnecessary because primal formulation with $C=0.01$ solves in ~1–2 minutes on the full 54k training set |
| SGDClassifier(loss='hinge') | Approximates LinearSVC via stochastic gradient descent, but requires learning rate scheduling and doesn't guarantee the exact coordinate descent solution |

**Why**
The primal solver (which `dual="auto"` picks here) converges quickly when $N \gg d$, providing an exact linear SVM baseline across all 60k samples without downsampling. Higher $C$ values ($C=1.0$) caused fit times to spike ($924$s on pixels) due to strict penalties on margin violations on non-separable image data, confirming $C=0.01$ as both the optimal and fastest regularisation choice.

**How it works**
`make_linear_svm(C)` wraps `StandardScaler` and `LinearSVC(C=C, dual="auto", random_state=42)`; `max_iter` stays at its default of 1000. Convergence warnings were not recorded. The C = 1 fits (924 s pixels / 811 s HOG) may have hit that cap, which does not change the choice: C = 0.01 won on val and is the fastest. The final 60k refits took 99 s (pixels) and 162 s (HOG). On raw pixels, test accuracy was 83.78%; on HOG, test accuracy reached 88.98% (+5.2 percentage points), demonstrating the immense power of edge-orientation histograms for linear hyperplanes.

**Files touched**
- `src/models_tree_kernel.py`, `results/metrics/linear_svm_*`, `notebooks/04_rf_svm.ipynb`

**Concepts worth learning**
- Primal (trust-region Newton) vs. dual (coordinate descent) solvers in liblinear, and why the sample/feature ratio decides which is faster.
- The effect of regularisation $C$ on the number of margin violations and solver iteration count.

**Revisit if**
Phase 05 wants an exact optimum below C = 0.01 (the winner is at the grid edge), or LinearSVC is chosen for CV and its fits are too slow.

*Corrected in review (D-028): the original entry said `max_iter=5000` and a 34 s HOG fit; neither matched the code or the JSONs.*

---

### D-027 — RBF SVM stratified 10k subset tuning and full 60k refit scaling
- **Date:** 2026-10-25
- **Phase:** v1/04 — Random Forest & SVM (author: Tanmay Singh)
- **Status:** Active
- **Type:** algorithm

**Context**
Non-linear kernel SVM training with LibSVM has $O(N^2 d)$ to $O(N^3 d)$ time complexity. Evaluating 9 grid combinations of $(C, \gamma)$ on the full $N=54,000$ training set would require $>2.9$ billion kernel operations per fit and would take hours/days.

**Decision**
Perform hyperparameter selection on a class-stratified 10,000 training subset (`config.SVM_TUNE_SUBSET=10000`, seed 42) scored on the full 6,000 validation set. Refit the winning hyperparameters on all 60,000 images (`trainval`) for the final test evaluation.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Full 54k tuning for all 9 combos | Impractical compute requirement (>6 hours) without altering the winning hyperparameter choices |
| Random non-stratified subset | Might skew class proportions and misrepresent hard boundary classes like Shirt/Coat/Pullover |
| Nyström / RBF feature approximation | Changes the model architecture away from exact RBF SVM (PRD S-01 requirement) |

**Why**
Stratified subsampling reduces tuning time by $(54/10)^2 \approx 29.2\times$, while preserving class balance (1,000 images per class) and representative margin boundaries. Validation accurately favored moderate regularisation ($C=10$) with well-conditioned bandwidths ($\gamma=0.001$ for pixels, $\gamma=\text{'scale'}$ for HOG), while identifying that $\gamma=0.01$ collapses kernel distances in 784/1,728 dimensions. The 60k refit then captures all available training data, achieving 89.70% (pixels) and 91.08% (HOG).

**How it works**
`tune_on_val(..., train_subset=config.SVM_TUNE_SUBSET)` records `"train_subset": 10000` in `rbf_svm_*_val.json`. `final_test_run()` trains on full 60k, yielding 19,710 support vectors for pixels and 26,343 for HOG.

**Files touched**
- `src/models_tree_kernel.py`, `results/metrics/rbf_svm_*`, `notebooks/04_rf_svm.ipynb` (uses the existing `tune_on_val(train_subset=...)`; `src/experiment.py` is unchanged)

**Concepts worth learning**
- Quadratic solver scaling in kernel machines; Mercer's theorem.
- Bandwidth $\gamma$ conditioning in high dimensions ($d=784, 1728$); support vector sparsity.

**Revisit if**
Phase 05 uses RBF SVM for 5-fold CV; 10k subset or parallel fold execution may be required.

---

### D-028 — Review fixes: reproducible figures, test guard, runtime notes, val and timing caveats
- **Date:** 2026-10-25
- **Phase:** v1/04 — review fixes by the lead on PR #2 (before merge)
- **Status:** Active
- **Type:** edge-case

**Context**
The review of PR #2 found the results themselves honest: all 6 test JSONs were recomputed from the real test labels and
matched, and `test_log.csv` shows one test per experiment, each after its tuning. But several things around them were wrong:
1. The phase summary, notebook and PR said RBF SVM on HOG was the best model "by validation". It is not (see below).
2. Some committed figures could not be regenerated by the notebook. It drew pixels and HOG in one figure and saved it as
   `..._val_curve_pixels` (breaking "one plot per file", `data-formats.md` §12), while the committed `..._val_curve_hog.png`
   files and the RF heatmap came from code that was never committed.
3. D-026 described `max_iter=5000` and a 34 s HOG fit; the code uses the default 1000 and the JSON says 151 s. D-027 listed
   `src/experiment.py` as changed; it wasn't.
4. The notebook's test cell was weaker than notebook 03's guard (D-021): it didn't check `test_log.csv`, and on a parameter
   mismatch it would silently run the test again.
5. No slow cell had a "⏱" runtime marker, although the done criterion was ticked.
6. Timings come from a different machine (Apple M3 Pro) than phase 03 (Apple M3).

**Decision**
Fix everything in the notebook and docs **without any new test evaluation**, and write down two caveats Phase 05 must carry:
RBF SVM val scores are not comparable with the others, and timings differ across machines.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Re-tune RBF SVM on all 54k so its val score is comparable | Hours of compute, and it contradicts the documented subset plan (D-027, system-design §11) for no change in the test results |
| Re-measure RBF/RF timings on one machine | Needs a refit and a new prediction on test, i.e. a second test use per experiment (AGENTS.md §4.1) |
| Merge as-is and let Phase 05 clean up | Leaves non-reproducible figures and a false "best by val" claim that could push Phase 05 into choosing by test |

**Why**
The project's credibility rests on two things the professor will check: every figure is produced by code, and every choice is
made on validation. Both were at risk, while the fixes are cheap and touch no result file.

**How it works**
- **Figures:** the LinearSVC and RBF curve cells now loop over `pixels`/`hog` and save one figure each with `save_fig`
  (`treekernel_{linear_svm,rbf_svm}_val_curve_{pixels,hog}.png`). The heatmap cell always refits the tested `rf_pixels`
  configuration on train+val (seed 42, so the same forest; ~50 s) and saves `treekernel_rf_importance.png`. No test data.
- **Test guard:** copied from notebook 03. A saved test JSON is reused only if its params equal the val winner, otherwise
  it raises; if the JSON is missing but the experiment is already in `test_log.csv`, it raises instead of re-testing.
- **Runtimes:** ⏱ notes before each slow cell, plus a table built from the JSONs (tuning total, slowest fit, final 60k fit,
  inference per 1,000) and a written section. Restart & Run All took 51 s with 0 errors; `test_log.csv` and every metrics
  JSON are byte-identical before and after (SHA-256 checked).
- **Validation ranking (what Phase 05's "best by val" rule sees):** logreg_hog 0.8927 > linear_svm_hog 0.8900 >
  rbf_svm_hog 0.8867 = rf_pixels 0.8867. RBF SVM leads only on test (91.08%). Its val score comes from a model trained
  on 10k images, while the others use 54k, so it understates the final 60k model.
- **Timing caveat:** RBF SVM prediction cost should grow with support vectors × dimensions: (26,343 × 1,728) /
  (19,710 × 784) ≈ 2.9× from pixels to HOG. The measured ratio is 28,877 / 4,767 ≈ 6.1×, so the HOG latency may include
  memory pressure or other load. Treat it as an upper bound.
- **Docs:** D-026/D-027 corrected (marked in D-026), phase summary and progress fixed, Phase 05 file notes added, PR
  description table corrected to the JSON values.

**Files touched**
- `notebooks/04_rf_svm.ipynb`: per-feature figures, generated heatmap, notebook-03 test guard, ⏱ notes, runtime section, LaTeX fix (`\rho`, `\text`), corrected summary
- `results/figures/treekernel_*`: regenerated by the notebook
- `docs/decisions.md`: D-026, D-027 corrections; this entry
- `docs/phases/v1/phase-04-rf-svm.md`, `docs/phases/v1/phase-05-results-analysis.md`, `docs/progress.md`

**Concepts worth learning**
- **Comparable validation scores:** a model selection rule is only fair if each candidate's val score was produced the same way (same training size).
- **Reproducibility:** a committed figure without the code that made it can't be defended or regenerated.
- **Confounders in timing:** different machines or background load change timings, never accuracies.

**Revisit if**
Phase 05 decides to change the "best by val" rule because of the RBF subset caveat (that needs its own decision entry), or
the team re-measures all timings on one machine.

---

### D-029 — Comparable val scores: refit subset-tuned RBF SVM on full train before choosing the best model
- **Date:** 2026-10-25
- **Phase:** v1/05 prep — decided by the lead after the PR #2 review, **written before the new run**
- **Status:** Active
- **Type:** algorithm

**Context**
Phase 05 picks the "best" experiment by validation accuracy (D-005, system-design §3). But the val scores weren't measured
the same way. Every model's val score comes from a model trained on all 54k train images, except RBF SVM, which was tuned on
a 10k subset to save time (D-005, D-027). A model trained on less data scores lower, so the rule was unfair to RBF SVM.
The tuning-file ranking was logreg_hog 0.8927 > linear_svm_hog 0.8900 > rbf_svm_hog = rf_pixels 0.8867.
**We noticed this after the test results were visible** (rbf_svm_hog has the best test accuracy, 91.08%). That is exactly
why the rule below is fixed and committed *before* the new val scores exist.

**Decision**
Keep "best = highest val accuracy", but make every val score comparable: refit RBF SVM's already-chosen params (pixels: C=10,
γ=0.001; HOG: C=10, γ='scale') once on the full 54k train split and score on the same 6k val. Phase 05 uses these numbers for
`rbf_svm_*` and accepts whichever experiment then has the highest val accuracy, **including if it is still logreg_hog**.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Keep the tuning-file val scores, just footnote the caveat | Leaves a known-unfair comparison in the one choice the professor checks most |
| Pick the best by test accuracy | Leakage: the test score of a model chosen by test is no longer an unbiased estimate (AGENTS.md §4.2) |
| A weighted score of accuracy, speed and size | Arbitrary weights and units, chosen after seeing results; it would look tuned to a winner. Trade-offs go in the accuracy-vs-time plot instead (RQ3) |
| Re-tune RBF SVM's whole grid on 54k | ~5 hours (18 fits); unnecessary because the params are already chosen. Only the comparison is unfair, not the choice |

**Why**
The rule itself is sound; the measurement fed into it wasn't. Fixing the measurement costs 2 fits (~10–15 min), uses only
train and val, and keeps "chosen on validation, never on test" true. Writing the decision before the run means the result
can't have influenced the rule.

**How it works**
1. `experiment.full_train_val_score(name, make_fn, params, kind)` builds the pipeline with the given params, fits it on
   `get_features(kind, "train")` (54,000 rows), predicts the 6,000 val images, and writes `{name}_{kind}_val_full.json`
   (`data-formats.md` §7). It never calls `get_features(..., "test")`, so `test_log.csv` can't change.
2. Notebook 04 runs it for `rbf_svm_pixels` and `rbf_svm_hog` (reusing the JSONs if they already exist).
3. Phase 05's `pick_best_experiment()` takes `val_acc` from `_val_full.json` for experiments whose `_val.json` has a
   `train_subset`, and `best_val_acc` from `_val.json` for all others, then picks the highest.
4. The 6k val set gives each accuracy an uncertainty of about ±0.4 points (√(0.89·0.11/6000) ≈ 0.004), so the write-up
   should say the top models are close, and the 5-fold CV of the winner shows its stability.

**Files touched**
- `src/experiment.py`: new `full_train_val_score()`
- `docs/module-interfaces.md`, `docs/data-formats.md` §7: new function and file format; best-model rule wording
- `docs/phases/v1/phase-05-results-analysis.md`: selection rule note
- `notebooks/04_rf_svm.ipynb`, `results/metrics/rbf_svm_*_val_full.json` (in the follow-up commit that runs it)

**Concepts worth learning**
- **Like-for-like comparison:** a selection rule is only fair if every candidate's score was produced under the same conditions.
- **Learning curves:** more training data usually raises accuracy, so a model trained on 10k is handicapped against one on 54k.
- **Pre-registration:** fixing the rule before seeing the numbers it will be applied to, so the outcome can't steer it.

**Result (added after the run, rule unchanged)**
Run on 2026-10-25 (notebook 04 §6, Apple M3), after the rule above was committed in f76e192:

| Experiment | Val acc, 10k-subset model | Val acc, 54k-train model | 54k fit |
|---|---|---|---|
| rbf_svm_pixels | 0.8793 | **0.9082** | 198 s |
| rbf_svm_hog | 0.8867 | **0.9157** | 367 s |

New val ranking: **rbf_svm_hog 0.9157** > rbf_svm_pixels 0.9082 > logreg_hog 0.8927 > linear_svm_hog 0.8900. The winner is
rbf_svm_hog by 2.3 points over the best non-RBF model, far more than the ~±0.4-point val noise. The 10k subset had cost RBF
about 3 points of val accuracy. `test_log.csv` and every existing metrics JSON are byte-identical (SHA-256 checked).

**Revisit if**
Another subset-tuned model is added (e.g. a Phase 2 model tuned on a subset), or the full-train refit doesn't fit in memory.

---

### D-030 — Phase 05 aggregation: validation-selected model ranking and non-comparable timing rules
- **Date:** 2026-10-25
- **Phase:** v1/05 — Results, Error Analysis & README (Anand Mishra)
- **Status:** Active
- **Type:** algorithm

**Context**
Phase 05 aggregates the benchmark results across all 12 experiments into `results/summary.csv` and produces comparison visualizations (RQ2: pixels vs HOG, RQ3: accuracy vs inference latency, RQ4: per-class recall heatmap) and a 5-fold cross-validation stability check. Two integrity constraints are required:
1. Identifying the "best model" must be done strictly on validation accuracy without touching or peeking at test results (D-005, system-design §3), using comparable scores from full 54k-train models (D-029).
2. Phase 03 noted that Logistic Regression fits ran under validation contention (`timing_comparable: false`, D-023, D-024). In aggregating results, predictive metrics (accuracy, F1, recall) must be preserved while non-comparable timings must not distort comparisons.

**Decision**
1. In `src/results.py`: `pick_best_experiment()` and `build_summary()` determine validation accuracy by inspecting for `{experiment}_val_full.json` (created for subset-tuned models such as RBF SVM per D-029) and falling back to `best_val_acc` in `{experiment}_val.json`. Validation accuracy is never read from test data. The selection rule chooses `rbf_svm_hog` (val accuracy 0.9157).
2. In `build_summary()`: When `timing_comparable` is False in the test JSON (for `logreg_pixels` and `logreg_hog`), `train_time_s` and `inference_ms_per_1000` are emitted as `NaN` (blank in CSV). Zero is never substituted.
3. In `plot_acc_vs_inference()`: Filter out rows with `NaN` timing (Logistic Regression per D-024) and the dummy baseline where zero latency (`dummy_pixels` has 0.00 ms measured latency) is mathematically undefined on a logarithmic axis. Preserves original measurements in `results/summary.csv` and source JSONs without substitution, and explicitly states all exclusions in the figure footnote alongside cross-phase hardware notes (Phase 03 on Apple M3 vs Phase 04 on Apple M3 Pro; Dummy sandbox environment) and RBF SVM HOG inference upper bound (D-028).
4. In `cross_validate_best()` (Task 05.3): Stratified 5-fold cross-validation is performed strictly on `trainval` (10,000 stratified subset for RBF SVM per D-005 and D-027), never touching the test set.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Pick the best model using test accuracy | Information leakage: invalidates test set as an unbiased performance estimator |
| Substitute 0.0 or mean values for contaminated LR timings | Distortion: 0 ms suggests instantaneous inference; violates benchmark accuracy |
| Substitute an arbitrary small positive number (e.g. 0.001) for dummy zero latency | Falsifies measurement: original metrics must be preserved truthfully; mathematical incompatibilities on log scales are properly handled via documented exclusion |
| Discard the entire LR experiments from summary table | Unnecessary: predictive metrics (accuracy, F1, class recall) remain fully valid and informative |
| Run 5-fold CV on all 60k images for RBF SVM | Too slow: each fold fits on ~48k images, and one 60k RBF HOG fit already takes 451 s, so 5 folds would take well over 30 min (estimate, not measured). The 10k stratified subset is the planned protocol (phase file, D-005); it took ~2 min (130 s, re-run in review) |

**Why**
Preserves benchmark scientific integrity and prevents leakage. Test evaluation occurs exactly once per experiment and is never used for model selection. Handling non-comparable timings via explicit `NaN` avoids corrupting latency trade-off comparisons.

**How it works**
1. `results.load_all_metrics(split="test")` reads all 12 `*_test.json` files without touching test features.
2. `results.build_summary()` merges comparable val scores, test metrics, and class recalls into `results/summary.csv` (12 rows, 19 columns).
3. `results.pick_best_experiment()` checks `_val_full.json` then `_val.json`, selecting `rbf_svm_hog`.
4. `experiment.cross_validate_best()` runs 5-fold CV on the 10k subset of `trainval`, producing `results/metrics/rbf_svm_hog_cv.json` (mean 0.8775 ± 0.0054).
5. `test_log.csv` is never written to; its row count and MD5 remain unchanged.

**Files touched**
- `src/results.py`: new module implementing aggregation, comparison plots, and best model selection
- `results/summary.csv`: aggregated 12-experiment summary table
- `results/figures/results_pixels_vs_hog.png`, `results/figures/results_acc_vs_time.png`, `results/figures/results_recall_heatmap.png`
- `results/metrics/rbf_svm_hog_cv.json`: stability check output
- `docs/decisions.md`: this decision log entry
- Review changes (lead, D-031): the shared val rule moved into `comparable_val_acc()`; the RQ3 chart footnote now lists the excluded experiments by name and calls rbf_svm_hog's timing an upper bound rather than blaming contention

**Concepts worth learning**
- **Validation vs Test Selection:** Selection on validation preserves unbiased generalization estimates on the final test split.
- **Handling Missing/Contaminated Data:** Missing indicators (`NaN`) preserve sample inclusion for valid dimensions while isolating contaminated dimensions.
- **Cross-Validation Stability:** Stratified k-fold CV within training data assesses estimator variance without risking test set leakage.

**Result**
5-fold CV of `rbf_svm_hog` (10k stratified subset):
- Folds: `[0.8765, 0.8860, 0.8715, 0.8725, 0.8810]`
- Mean accuracy: **0.8775** (87.75%)
- Std accuracy: **0.0054** (±0.54%)
- Summary table: 12 rows, `test_log.csv` MD5 unchanged.

**Revisit if**
A standardized single-machine re-benchmarking run is conducted for all 12 experiments to re-measure uniform inference and training latencies.

---

### D-031 — Phase 05 split between two owners and two PRs
- **Date:** 2026-10-26
- **Phase:** v1/05 — Results, Error Analysis & README
- **Status:** Active
- **Type:** tooling

**Context**
The team has five members but phases 01 and 02 were both built by the lead, so one member had no phase. Phase 05 is also
the largest phase (7 tasks: tables, plots, CV, error analysis, notebook, README, docs). The workflow (AGENTS.md §1, §6)
assumes one owner, one branch and one PR per phase.

**Decision**
Phase 05 is split in two, approved by the lead: **Part A** (tasks 05.1–05.3: summary table, comparison plots, CV) by
Anand Mishra in PR #3 on `phase/v1-05-anand-results`; **Part B** (tasks 05.4–05.7: error analysis, notebook 05, README,
docs and phase close) by Omved Nagre in a second PR, on a new branch from `main` after PR #3 is merged.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| One owner for all of phase 05 | Leaves one team member without a phase, and every member needs visible commits |
| Both people commit to one branch and one PR | Two people pushing to one fork branch is easy to break (conflicts, force-push temptation); review would wait for both halves |
| Split by topic (e.g. one person does all plots) | The tasks are already ordered so that Part B only needs Part A's finished functions; splitting by task is simpler |

**Why**
Each person gets their own commits and PR, and Part B starts from a reviewed, merged Part A instead of an unreviewed branch.

**How it works**
1. PR #3 (Part A) is reviewed and merged. `main` then has `build_summary()`, `pick_best_experiment()`,
   `comparable_val_acc()`, the three `results_*` figures, `summary.csv` and `rbf_svm_hog_cv.json`.
2. The four error-analysis functions in `results.py` are stubs that raise `NotImplementedError` until Part B.
3. The phase stays `in-progress` after PR #3. Only Part B fills the Phase summary, sets the phase to `review` and
   ticks the remaining done criteria.
4. Part B's notebook 05 must *call* the Part A functions (so every committed output is reproducible from the repo) and must
   **reuse** `rbf_svm_hog_cv.json` if it exists with the same params, the same guard pattern as the test JSONs (D-021).

**Files touched**
- `docs/decisions.md`: this entry
- `docs/phases/v1/phase-05-results-analysis.md`: owners, branches, handover notes; one done criterion unticked (error analysis not built yet)
- `docs/progress.md`: owners and session log
- `src/results.py`: stub messages point to D-031

**Concepts worth learning**
- **Vertical split by task order:** later tasks depend only on the merged output of earlier ones, so two people never edit the same function.
- **Reproducibility:** a committed result must be re-creatable by running committed code, not a private script.

**Revisit if**
Part B finds it needs to change a Part A function's signature or output; then update `module-interfaces.md` in Part B's PR.

---

### D-032 — Error analysis from saved predictions; end-to-end speed counts HOG extraction
- **Date:** 2026-10-26
- **Phase:** v1/05 — Results, Error Analysis & README (Part B, Omved Nagre; review fixes by the lead)
- **Status:** Active
- **Type:** algorithm

**Context**
Task 05.4 needs a confusion matrix, the top-5 confused pairs, image grids of those mistakes, and how often the two best
models agree, without any new test predictions (AGENTS.md §4). The RQ3 answer also has to compare speeds fairly: the
saved `inference_ms_per_1000` times only the model's `predict()`, but HOG models first need HOG features computed.
The first version of the write-up assumed HOG extraction took ~30 ms per 1,000 images; the measured value is 216 ms.

**Decision**
1. All error analysis reads the saved `confusion_matrix` and `y_pred` from `{experiment}_test.json`; the grids and the
   agreement counts also read the true test labels/images from `load_splits()`. Nothing is predicted again.
2. Image grids show the **first n** matching test images in test-set order, so nobody can be accused of picking examples.
3. RQ3 is answered with **end-to-end** time per 1,000 images: model time + 216 ms HOG extraction for HOG models
   (`extract_s_per_1000` for the chosen setting in `results/hog_sweep.csv`). The RQ3 chart still plots model-only time.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Re-predict the test set inside the error-analysis functions | A second test use per model; breaks the one-test rule and the audit log |
| Show the "most confident" or hand-picked mistakes | Needs model scores we didn't save, or invites cherry-picking |
| Compare model-only times in RQ3 | Makes LinearSVC HOG look ~5× faster than RF pixels, when end to end it is ~18× slower |
| Re-time HOG extraction on the M3 Pro | Not needed for an order-of-magnitude answer; the phase 02 number is recorded and reproducible |

**Why**
Saved predictions are exactly what was tested, so the analysis is consistent with the reported accuracy and costs nothing.
End-to-end time is what a user of the model would actually wait for.

**How it works**
1. `top_confused_pairs("rbf_svm_hog")` scans the 90 off-diagonal cells of the saved 10×10 matrix and keeps the 5 largest:
   Shirt→T-shirt/top 104, T-shirt/top→Shirt 91, Pullover→Shirt 66, Pullover→Coat 61, Shirt→Coat 59 (out of 1,000 per class).
2. `plot_misclassified` finds test indices where the true label is A and the saved prediction is B, and draws the first 16.
3. `model_agreement("rbf_svm_hog", "rbf_svm_pixels")`: 8,593 both right, 515 both wrong (391 with the same wrong label,
   397 of the 515 in T-shirt/Pullover/Coat/Shirt), 377 only HOG wrong, 515 only pixels wrong. HOG's net gain is
   515 − 377 = 138 images = the 1.38-point test gap.
4. End-to-end per 1,000 images: rf_pixels ~12 ms (87.56%), linear_svm_hog 2.6 + 216 ≈ 219 ms (88.98%, 181 KB),
   rbf_svm_hog ~29 s (91.08%). So "fastest above 87%" is RF pixels and "best accuracy for its size" is LinearSVC HOG.

**Files touched**
- `src/results.py`: `plot_confusion`, `top_confused_pairs`, `plot_misclassified`, `model_agreement`
- `notebooks/05_results_analysis.ipynb`, `README.md`: RQ3 end-to-end table, corrected numbers
- `docs/module-interfaces.md`, `docs/data-formats.md` §12: return keys, class-name arguments, figure names

**Concepts worth learning**
- **Confusion matrix:** rows are the true class, columns the prediction; the diagonal divided by the row total is recall.
- **Directed confusion pair:** Shirt→T-shirt and T-shirt→Shirt are different mistakes with different counts.
- **Model agreement:** if two different models make the same mistake, the image is probably ambiguous, not the model unlucky.
- **End-to-end latency:** everything between "image in" and "label out", including feature extraction.

**Revisit if**
Timings are re-measured on one machine, or Phase 2 adds a model whose features cost more than its prediction.
