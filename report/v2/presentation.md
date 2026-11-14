# The Story of Phase 2: Can a CNN Do Better Than Handcrafted Features?

*A walkthrough for every team member: what we built in Phase 2, why, how the code works, and what to say at the viva.*

Read the **Prologue** first, then **your own part** closely, then skim the other parts so you can follow the whole story.
Each part ends with "Numbers to remember" and "Questions you may get". All numbers come from the files in `results/`.

| Part | Presenter | Enrollment No. | Topic |
|---|---|---|---|
| 1 | Tanmay Singh | 2401010476 | Recap, hypotheses & protocol |
| 2 | Anand Mishra | 2401010065 | How a CNN works & our architectures |
| 3 | Rajdeep Sanyal | 2401010370 | Training, early stopping & model choice |
| 4 | Rajat Srivastav | 2401020101 | Test results, significance & cost |
| 5 | Omved Nagre | 2401010311 | Error analysis & conclusion |

---

## Prologue: what we set out to do

### The question
Phase 1 showed that the best classical model, an **RBF SVM on HOG features**, classifies Fashion-MNIST at **91.08%**, and
that every classical model fails on the same four tops: T-shirt/top, Pullover, Coat and, worst of all, Shirt. Phase 2
asks whether a **convolutional neural network**, which learns its own features instead of using fixed ones, does better,
and *where* it does better.

We wrote two hypotheses down before training anything:
- **H1:** a small CNN beats 91.08% under exactly the same protocol.
- **H2:** it wins *mainly by fixing* the Shirt / T-shirt / Pullover / Coat confusions.

And four research questions continue Phase 1's RQ1–RQ4:
- **RQ5:** Does the CNN beat the best classical model on accuracy and macro F1?
- **RQ6:** Does it reduce the Shirt / T-shirt / Pullover / Coat confusion?
- **RQ7:** Does a third convolutional block help?
- **RQ8:** What does the CNN cost (parameters, size, inference time)?

### The answer in one paragraph
**H1 is supported, H2 is rejected.** The final CNN (CNN3) reaches **92.09%** test accuracy, 101 fewer errors than the
SVM, a statistically significant gain (McNemar p = 2.2 × 10⁻⁴), while being about **300× faster** and **377× smaller**.
But Shirt recall does not move (75.1% vs 75.2%): the two models swap shirts instead of fixing them, and most of the
remaining errors are images both models get wrong in the same way, a sign that the images or labels are ambiguous.

### The one rule that still matters most: no leakage
Phase 2 keeps every Phase 1 rule and adds a few:
1. Same split, same test set, same `evaluate()` code: nothing changes except the model.
2. The CNNs train on the **54k** train images only. Validation decides when to stop and which CNN wins; it never
   updates weights.
3. There is **no test data loader**. The test set is read only by `final_test_run_cnn()`, once per CNN, after the
   choice. `results/test_log.csv` ends with **14 rows** (12 classical + 2 CNNs).
4. Scoring always runs with dropout off (`model.eval()` and `torch.no_grad()`).
5. Phase 2 never rewrites Phase 1 files: `summary.csv` keeps 12 rows; all 14 models are in `summary_all.csv`.

### How the Phase 2 code is organised
| File | Role |
|---|---|
| `src/cnn.py` | Data loaders, CNN2 and CNN3, parameter counting, `CNNPredictor`, `train_cnn()` |
| `src/experiment.py` | `final_test_run_cnn()`: the single test run per CNN |
| `src/evaluate.py` | The **same** metrics as Phase 1, plus the CNN's parameter count and PyTorch version |
| `src/results.py` | Summary tables, recall chart, confusion matrices, error grids, model agreement |
| `notebooks/06_cnn_training.ipynb` | Shapes, parameter counts, training, curves, choice, test runs |
| `notebooks/07_cnn_vs_classical.ipynb` | Comparison with the SVM, error analysis, answers to RQ5–RQ8 |

### Words you should be comfortable with
- **Convolution / filter / feature map:** a small grid of learned weights slid over the image; the output map shows
  where the filter's pattern appears.
- **Pooling:** keeps the strongest value in each 2×2 window, halving the size.
- **Epoch:** one pass over all 54,000 training images. **Batch:** 64 images used for one weight update.
- **Cross-entropy loss:** how wrong the predicted probabilities are; training pushes it down.
- **Adam:** the optimizer that decides how far to move each weight after each batch.
- **Dropout:** randomly switching off half the units of one layer during training so the network can't rely on any one.
- **Early stopping:** stop when validation loss stops improving and keep the best weights.
- **Seed:** the number that fixes all randomness, so runs repeat exactly.
- **McNemar test:** compares two models on the same images, using only the images where they disagree.

---

## Part 1: Recap, hypotheses & protocol (Tanmay Singh)

### The story
Phase 1 built a fair benchmark of classical ML on Fashion-MNIST: 12 experiments, every choice made on a fixed
6,000-image validation split, the test set used exactly once per experiment. The winner was an RBF SVM on HOG features
at **91.08%**. Its error analysis was very clear: **Shirt** was the hardest class for every model, and confusions among the
four upper-body garments made up 56–66% of every model's errors.

The project brief names the reason classical features fall short: **flattened pixels discard which pixels are neighbours,
and HOG fixes the representation before the model sees any examples.** A CNN does neither. It keeps the 2-D layout and
learns its filters from the data. That led to our two hypotheses: H1 (the CNN wins) and H2 (it wins by fixing the tops).

**Why the comparison is fair.** Nothing changes except the model:
- the same 54k / 6k / 10k split (`load_splits()`, split file MD5 `54ff6b78b27109cdfc3ebe7204040a07`);
- pixels already in [0, 1], so no statistics are computed from any data (nothing that could leak);
- the same `evaluate()` produces the same metrics;
- validation makes every choice; the test set is read once per CNN; every read is logged.

**The one difference, and why it favours Phase 1.** Classical models were refit on all 60k images before testing. A CNN
uses the validation set *during* training (early stopping), so it learns from **54k** only (D-035). The CNN has 10% less
data and still wins, so the comparison, if anything, understates its advantage.

### The code behind this part
| Function | File | What it does |
|---|---|---|
| `load_splits()` | `src/data.py` | The fixed Phase 1 split (unchanged) |
| `make_loaders()` | `src/cnn.py` | Adds the channel dimension (N, 1, 28, 28), builds a shuffled train loader and a val loader; **no test loader** |
| `_log_test_use()` | `src/experiment.py` | Appends one row to `test_log.csv` per test run |

### Numbers to remember
- Phase 1 best: RBF SVM on HOG, **91.08%** test, Shirt recall **75.2%**
- Split: **54,000 / 6,000 / 10,000**; test log: **14 rows**
- CNNs train on **54k**; classical models were refit on **60k**

### Questions you may get
- **Why didn't you refit the CNN on 60k like Phase 1?** Early stopping needs validation during training. Refitting on 60k
  would mean guessing the epoch count without any check, for an expected gain of 0.1–0.3 points; we decided this before
  testing (D-035).
- **How do you know the test set wasn't used for tuning?** There is no test loader; only `final_test_run_cnn()` reads it,
  and `test_log.csv` shows exactly one run per CNN, after the model choice.
- **Why state hypotheses in advance?** So the result can prove us wrong. H2 did turn out to be wrong, and that is a finding.

---

## Part 2: How a CNN works & our architectures (Anand Mishra)

### The story
**Convolution.** A 3×3 filter is a grid of 9 learned weights. It slides over the image; at each position it multiplies
the 9 pixels under it by its weights and adds them up. The result is a **feature map** that is bright wherever the
filter's pattern (an edge, a corner, a stripe) appears. Two ideas make this efficient:
- **Local connectivity:** each output looks at a small patch, not the whole image.
- **Weight sharing:** the same filter is used at every position, so a collar detector works anywhere in the image, and the
  layer needs only (3·3·channels + 1) × filters parameters.

**ReLU** keeps positive values and sets negatives to zero. **Max pooling** (2×2) keeps the strongest value in each
window, halving the size and making the network tolerant to small shifts.

**Receptive field.** Each layer lets a unit "see" a bigger patch of the original image. After 2 blocks, one unit sees a
**10×10** pixel patch; after 3 blocks, **22×22**, almost the whole garment.

**Our two networks.** One *block* = Conv 3×3 (padding 1) + ReLU + MaxPool 2×2.
| | CNN2 | CNN3 |
|---|---|---|
| Blocks (filters) | 2 (32, 64) | 3 (32, 64, 128) |
| Shape before flatten | 64 × 7 × 7 = 3,136 | 128 × 3 × 3 = 1,152 |
| Then | Dense 128 + ReLU → Dropout 0.5 → Dense 10 | same |
| Parameters | **421,642** | **241,546** |
| Multiply-adds per image | 4.24 M | 7.60 M |

**Counting parameters (CNN2, a classic viva question).** Conv: (kernel h × kernel w × input channels + 1 bias) × filters.
Dense: (inputs + 1) × outputs.
- Conv1: (3·3·1 + 1) · 32 = **320**
- Conv2: (3·3·32 + 1) · 64 = **18,496**
- Dense: (3,136 + 1) · 128 = **401,536**
- Output: (128 + 1) · 10 = **1,290**
- Total **421,642**. Notebook 06 checks this with `count_params()`.

**Why is the deeper network smaller?** 95% of CNN2's parameters sit in the first dense layer, which takes 3,136 inputs.
CNN3's third pooling step shrinks the map to 3×3, so that layer takes only 1,152 inputs. The third conv block adds 73,856
parameters but removes 253,952 from the dense layer. It does cost more **compute**, though, because the extra convolution
runs at every position of a 7×7 map.

### The code behind this part
| Function | File | What it does |
|---|---|---|
| `CNN2`, `CNN3` | `src/cnn.py` | The two networks, built from `_conv_block()` |
| `count_params(model)` | `src/cnn.py` | Counts trainable weights and biases |
| notebook §2 | `notebooks/06_cnn_training.ipynb` | Sends one batch through CNN2 and prints every layer's output shape |

### Numbers to remember
- CNN2 **421,642** parameters, CNN3 **241,546** (43% fewer)
- Shapes: 1×28×28 → 32×14×14 → 64×7×7 (→ 128×3×3 in CNN3)
- Receptive field: **10×10** (CNN2), **22×22** (CNN3); compute: 4.24 M vs 7.60 M multiply-adds

### Questions you may get
- **Why 3×3 filters and padding 1?** Small filters stacked give a large receptive field with few parameters (the VGG
  idea); padding 1 keeps the size the same, so only pooling shrinks the map.
- **How is a CNN different from HOG?** HOG computes fixed edge-direction histograms with rules we choose; a CNN learns its
  filters from the training images, and deeper layers combine them into parts.
- **Why does 7×7 pool to 3×3, not 3.5×3.5?** Pooling takes whole 2×2 windows; the last row and column are dropped.

---

## Part 3: Training, early stopping & model choice (Rajdeep Sanyal)

### The story
**The training loop** (one epoch): shuffle the 54k training images, then for each batch of 64: run the network forward,
compute the cross-entropy loss, **backpropagate** to get each weight's gradient, and let **Adam** update the weights
(about 844 updates per epoch). After each epoch, score the 6k validation images with dropout off.

**Settings** (the same for both networks, no tuning, D-034): Adam with learning rate 0.001, batch 64, at most 30 epochs,
dropout 0.5, early stopping with patience 5.

**Early stopping.** After every epoch we check validation loss. We remember the weights with the lowest value; if 5 epochs
pass without beating it, we stop and put those weights back.
- CNN2: best at epoch **9**, stopped at 14. CNN3: best at epoch **12**, stopped at 17.
- After the best epoch, training loss keeps falling (CNN3: 0.175 → 0.127) while validation loss rises
  (0.193 → 0.208). That is **overfitting**: the network starts memorising the training images.
- At the best epoch, training and validation accuracy are within 0.5 points: dropout keeps overfitting small.
- Early on, validation accuracy is *higher* than training accuracy, because training accuracy is averaged while the
  weights are still improving, and dropout is on during training only.

**Loss vs accuracy.** Early stopping watches loss, not accuracy. CNN2's accuracy kept creeping up to 0.9297 by epoch 14
while its loss never beat epoch 9. Loss also punishes confident wrong answers, so it turns upward first.

**Choosing the final CNN** (rule fixed before training: higher validation accuracy wins, D-035):
| | Best epoch / run | Val loss | Val acc | Train time |
|---|---|---|---|---|
| CNN2 | 9 / 14 | 0.1976 | 0.9228 | 198 s |
| **CNN3** | 12 / 17 | 0.1926 | **0.9308** | 416 s |

**CNN3 is the final CNN** (D-040). Both beat the best classical model on validation (0.9157).

**The honest caveat.** An earlier CNN2 run, identical except for the order of its training batches, reached **0.9320**.
So the same network varies by about one point from randomness alone, and the CNN3–CNN2 gap is within that noise. We
follow the rule but do not claim that depth helps.

**Reproducibility, and a bug we found.** Retraining in a fresh process first gave a *different* result: the data loader's
shuffle state had been used up by earlier notebook cells. We fixed it at the root: `train_cnn()` now seeds everything
and builds its own model and loaders, so a retrain reproduces every number exactly (D-041).

### The code behind this part
| Function | File | What it does |
|---|---|---|
| `train_cnn(name, make_model)` | `src/cnn.py` | Seeds, builds model and loaders, trains with early stopping, saves checkpoint, history and val JSON |
| `_run_epoch()` | `src/cnn.py` | One pass: trains with an optimizer, or scores with dropout off and no gradients |
| `training_settings()` | `src/cnn.py` | The fixed settings stored with the results |
| notebook §3–5 | `notebooks/06_cnn_training.ipynb` | Training, curves (`cnn_training_curves.png`), the choice |

### Numbers to remember
- Adam 0.001, batch 64, ≤ 30 epochs, patience 5, dropout 0.5
- Best epochs: CNN2 **9/14**, CNN3 **12/17**
- Val accuracy: CNN2 **0.9228**, CNN3 **0.9308** (final); best classical 0.9157
- Same-network variation: about **1 point** (0.9228 vs 0.9320)

### Questions you may get
- **How do you know it overfits?** After the best epoch, training loss keeps falling while validation loss rises.
- **Why stop on loss, not accuracy?** Loss is smoother and catches over-confidence earlier; it's the standard choice.
- **Is CNN3 really better than CNN2?** Not clearly. The gap is within run-to-run variation, and on test the McNemar
  p-value is 0.31. We chose CNN3 by our pre-set rule and report the comparison as inconclusive.
- **Why CPU instead of the Mac's GPU?** CPU runs are exactly reproducible, and CPU timings are comparable with the
  Phase 1 models (D-036).

---

## Part 4: Test results, significance & cost (Rajat Srivastav)

### The story
**One test run per CNN**, after the choice was recorded, through the same `evaluate()` as Phase 1.
`CNNPredictor` gives the network a `.predict()` like an sklearn model; it runs 1,000 images at a time with dropout off.

| Model | Test acc (95% CI) | Macro F1 | Errors | ms / 1,000 | Size |
|---|---|---|---|---|---|
| **CNN3** | **92.09%** (91.54–92.60) | **0.9208** | **791** | 94 | 0.97 MB |
| CNN2 | 91.85% (91.30–92.37) | 0.9176 | 815 | 58 | 1.69 MB |
| RBF SVM, HOG | 91.08% (90.51–91.62) | 0.9108 | 892 | ~29,000 | 366 MB |
| RBF SVM, pixels | 89.70% | 0.8967 | 1,030 | 4,767 | 125 MB |

**Is the gain real?** We compare the two models image by image (McNemar test). CNN3 is right on **417** images the SVM
gets wrong and wrong on **316** the SVM gets right: a net 101. If both models were equally good, those two numbers would
be close; 417 vs 316 gives χ² = 13.6, **p = 2.2 × 10⁻⁴**. **H1 is supported.** CNN2 also beats the SVM (p = 0.005). CNN3
vs CNN2 is 265 vs 241, p = 0.31: no real difference.

**Validation was slightly optimistic** (CNN3: 0.9308 val vs 0.9209 test), as expected when you choose the best model on
validation. The ranking does not change.

**Cost (RQ8).**
- **Speed:** 94 ms per 1,000 images vs about 29 s for the SVM on HOG (including HOG extraction): **~300× faster**. The SVM
  compares each image with all 26,343 stored support vectors; the CNN does a fixed amount of work per image.
- **Size:** 0.97 MB vs 366 MB: **~377× smaller**.
- **Inside the CNNs,** time follows compute, not parameters: CNN3 is smaller but slower (94 vs 58 ms), because it does
  79% more multiply-adds.
- Only Random Forest on pixels predicts faster (~12 ms), but it is 4.5 points less accurate and 366 MB.

### The code behind this part
| Function | File | What it does |
|---|---|---|
| `final_test_run_cnn(name, model)` | `src/experiment.py` | Loads the best checkpoint, calls `evaluate()` once, logs it |
| `CNNPredictor` | `src/cnn.py` | `.predict()` for `evaluate()`: batches of 1,000, `eval()` + `no_grad()` |
| `evaluate()` | `src/evaluate.py` | Same metrics as Phase 1; size detail = parameter count for a CNN |
| `build_summary(MODEL_ORDER, "summary_all.csv")` | `src/results.py` | The 14-row all-model table |

### Numbers to remember
- CNN3 **92.09%**, macro F1 **0.9208**, **791** errors; SVM **91.08%**, **892** errors
- McNemar CNN3 vs SVM: **417 vs 316, p = 2.2 × 10⁻⁴**; CNN3 vs CNN2: p = 0.31
- **94 ms vs ~29 s** per 1,000 images; **0.97 MB vs 366 MB**

### Questions you may get
- **Why McNemar and not just compare accuracies?** Both models are tested on the same images, so their results are
  paired. McNemar uses exactly the images where they disagree; it is the standard test for this (Dietterich, 1998).
- **Why test CNN2 at all if CNN3 was chosen?** So the depth comparison also has test numbers. The choice was recorded
  first, so testing CNN2 could not change it.
- **Are the timings fair?** Same thread count (4) and the same timing code; CNN on an M3, SVMs on an M3 Pro. Only large
  gaps like 300× should be read as firm.

---

## Part 5: Error analysis & conclusion (Omved Nagre)

### The story
**Did the CNN fix the shirts (RQ6)? No.**
- Shirt recall: CNN3 **75.1%**, SVM **75.2%**. The biggest confusion is still Shirt → T-shirt/top (110 vs 104).
- Of the 1,000 test shirts: 661 right for both, **90** right only for CNN3, **91** right only for the SVM, **158** wrong for
  both. The models **swap** shirts instead of fixing them (McNemar p = 1.0).
- Where the CNN *does* gain: T-shirt/top (+2.2 points), Pullover (+2.8), Coat (+1.5), plus Trouser, Sneaker and Sandal.
- Because it removes easier errors first, the four tops now hold **68%** of the CNN's errors (SVM: 64%).
- **H2 is rejected:** the gain is real but broad, not where we predicted.

**Why does Shirt stay hard?**
- **475** test images are wrong for both models, and on **398** of them both give the *same* wrong label. The two models
  see images completely differently (gradient histograms vs learned filters); when they agree on a mistake, the image is
  the problem.
- The grids (first 8 images in test order, not hand-picked) show "shirts" that are collarless short-sleeved tops, and
  even sleeveless tank tops (test #612, #1129). At 28×28 a collar is about two pixels wide, and some labels are debatable.
- Image by image, each model catches things the other misses: CNN3 correctly calls test #40 a Shirt (its button placket
  is visible), while the SVM gets #474, #572, #612, #1056 and #1129 right.

**Conclusion.** A 241,546-parameter CNN beats 12 tuned classical pipelines: 92.09% vs 91.08%, significant, ~300× faster,
~377× smaller. It does not solve Shirt, because most remaining errors are ambiguous images that no choice of features can
fix at this resolution. A third conv block makes the network smaller but not measurably more accurate.

**Limitations:** one training run per network; fixed hyperparameters (no augmentation or batch norm, as the brief
specifies); CNNs trained on 54k vs 60k; timings from two laptops.

**Future work:** several seeds per network; feed the CNN's 128 learned features into the Phase 1 SVM (same classifier,
new features); check the shared errors for label noise.

### The code behind this part
| Function | File | What it does |
|---|---|---|
| `plot_recall_comparison()` | `src/results.py` | Per-class recall chart (`cnn_recall_vs_classical.png`) |
| `plot_confusion()`, `top_confused_pairs()` | `src/results.py` | Confusion matrices and largest confusions (from saved predictions) |
| `plot_misclassified(..., tag=...)` | `src/results.py` | First-n misclassified images per class pair, per model |
| `model_agreement()` | `src/results.py` | Image-by-image comparison of two models |
| notebook | `notebooks/07_cnn_vs_classical.ipynb` | The whole comparison and the written answers to RQ5–RQ8 |

### Numbers to remember
- Shirt recall **75.1% vs 75.2%**; shirts: **90 / 91 / 158** (only CNN3 / only SVM / both wrong)
- Both wrong: **475**, same wrong label: **398**
- Tops' share of errors: **68%** (CNN3) vs **64%** (SVM)
- T-shirt **+2.2**, Pullover **+2.8**, Coat **+1.5** points

### Questions you may get
- **So what is the use of the CNN if Shirt isn't fixed?** It is more accurate overall (significantly), 300× faster and
  377× smaller. The brief asks us to *examine whether* a CNN resolves similar categories better; the honest answer is
  "partly": better on three of the four tops, not on Shirt.
- **Could a bigger CNN fix Shirt?** It might gain a point or two, but the shared errors suggest an information limit:
  much larger published networks still stop around 96%.
- **Why not train a new model now to fix it?** We have seen the test results; choosing a new model because of them would
  be using the test set to make a decision, the leakage this whole project avoids.

---

## Epilogue: how it all fits together

1. **Part 1** kept everything from Phase 1 fixed so that only the model changes, and wrote down H1 and H2 in advance.
2. **Part 2** built two small CNNs whose parameter counts can be derived by hand.
3. **Part 3** trained them reproducibly with early stopping and chose CNN3 on validation, with an honest caveat about noise.
4. **Part 4** tested each CNN once, proved the gain is significant and showed the CNN is far cheaper to run.
5. **Part 5** showed the gain is not on Shirt and traced the remaining errors to ambiguous images.

**What makes this trustworthy:** the same split and evaluation code as Phase 1; no test loader and a 14-row audit log;
every choice on validation with rules set before training; deterministic, reproducible training; and a decision log
(`docs/decisions.md`, D-033 to D-043 for Phase 2) explaining every choice.

### Full Phase 2 results (from `results/summary_all.csv`)
| Experiment | Val acc | Test acc | Macro F1 | Shirt recall | Inference ms / 1,000 | Size |
|---|---|---|---|---|---|---|
| **cnn3_images** | **0.9308** | **0.9209** | **0.9208** | 0.751 | 94.43 | 972 KB |
| cnn2_images | 0.9228 | 0.9185 | 0.9176 | 0.698 | 58.36 | 1.69 MB |
| rbf_svm_hog | 0.9157 | 0.9108 | 0.9108 | 0.752 | 28,876.96 | 366.2 MB |
| rbf_svm_pixels | 0.9082 | 0.8970 | 0.8967 | 0.710 | 4,766.94 | 125.1 MB |

### Where to look in the repo
| If you want to see… | Open |
|---|---|
| The networks and training loop | `src/cnn.py` |
| Training, curves, the choice and the test runs | `notebooks/06_cnn_training.ipynb` |
| The comparison and error analysis | `notebooks/07_cnn_vs_classical.ipynb` |
| All 14 models in one table | `results/summary_all.csv` |
| Per-epoch training history | `results/training/` |
| Why any choice was made | `docs/decisions.md` (search for the D-number) |
| Every test-set use | `results/test_log.csv` (14 rows) |
| The Phase 2 report | `report/v2/report.tex` |
