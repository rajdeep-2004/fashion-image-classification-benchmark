# Fashion-MNIST Classical ML Benchmark: slide deck source

> **Instructions for NotebookLM:** Build a presentation of about 16 slides from this document, one slide per
> "Slide" section below, in the same order. Use the title, the bullet points and the key numbers exactly as written;
> do not invent numbers. Keep each slide to 3–5 short bullets. Use the "Visual" line to decide what to show and the
> "Speaker notes" as the talk track. The audience is a professor and examiners at a university capstone viva.
> Tone: clear, confident, plain English.

**Project:** How Far Can Classical Machine Learning Go on Fashion-MNIST? A leak-free benchmark of raw pixels vs HOG
features, with error analysis. Capstone Phase 1 (classical ML); Phase 2 will be a CNN.

**Team and who presents which part**
| Part | Presenter | Enrollment No. | Topic |
|---|---|---|---|
| 1 | Tanmay Singh | 2401010476 | Data pipeline & EDA |
| 2 | Anand Mishra | 2401010065 | Feature engineering & evaluation framework |
| 3 | Rajdeep Sanyal | 2401010370 | Linear & distance-based models |
| 4 | Rajat Srivastav | 2401020101 | Tree & kernel models |
| 5 | Omved Nagre | 2401010311 | Results, error analysis & conclusion |

---

## Slide 1: Title
- How Far Can Classical Machine Learning Go on Fashion-MNIST?
- A leak-free benchmark of raw pixels vs HOG features, with error analysis
- Capstone Phase 1 · Team of 5 (names and enrollment numbers from the table above)

**Visual:** a clean title slide; optionally a strip of 10 small clothing images, one per class.
**Speaker notes:** We wanted to know how good classical machine learning really is at recognising clothes, which model
is the best trade-off, and exactly where and why these methods fail. That failure analysis is what motivates our Phase 2 CNN.

## Slide 2: The problem and our four questions
- Goal: auto-tag clothing photos (shirt, trouser, sneaker…) with classical ML, before reaching for deep learning
- RQ1: How accurate is classical ML on raw pixels?
- RQ2: Do HOG (edge-direction) features beat raw pixels, and for which models?
- RQ3: Which model gives the best accuracy / speed / size trade-off?
- RQ4: Which classes get confused, and why?

**Visual:** four boxes, one per research question.
**Speaker notes:** Everything in this talk answers one of these four questions. We return to each with a number at the end.

## Slide 3: The headline result
- Best model (chosen on validation, never on test): **RBF SVM on HOG features**
- **91.08% test accuracy** (validation 91.57%)
- Hardest class: **Shirt**, mostly confused with T-shirt/top, Pullover and Coat
- 12 experiments, test set used exactly once each, all reproducible from a fresh clone in under 2 minutes

**Visual:** one big number "91.08%" with the model name underneath.
**Speaker notes:** We show the result up front; the rest of the talk explains how we got there honestly and what it means.

---
### Part 1: Data pipeline & EDA (Tanmay Singh)

## Slide 4: The dataset and a fixed, honest split
- Fashion-MNIST: 70,000 grayscale 28×28 images, 10 balanced clothing classes
- Downloaded from the official source and verified with MD5 checksums; sanity checks found 0 duplicates and 0 missing values
- Split: **54,000 train · 6,000 validation (600 per class, seed 42) · 10,000 test**
- The validation indices are committed to the repo (MD5 54ff6b78…), so every team member uses the identical split

**Visual:** a horizontal bar split into three parts: train 54k, validation 6k, test 10k (test shaded as "locked").
**Speaker notes:** Train is for fitting, validation is for every decision, and test is opened once per experiment at the
very end. Committing the split file means nobody can accidentally use a different split.

## Slide 5: What the data told us before training any model
- EDA used the training split only, so nothing we learned could leak from validation or test
- The four tops (T-shirt, Pullover, Coat, Shirt) have almost identical average images: cosine similarity 0.95–0.99
- t-SNE: Pullover, Coat and Shirt form one mixed cloud; Trouser and Bag form clean, separate clusters
- Pixels are very redundant: 84 of 784 PCA components keep 90% of the variance
- **Prediction before any model: Shirt will be the hardest class**

**Visual:** the 10 class-mean images in a row, and a t-SNE scatter plot with the tops overlapping.
**Speaker notes:** This prediction matters: at the end we check whether the models fail exactly where the data said they would. They do.

---
### Part 2: Feature engineering & evaluation framework (Anand Mishra)

## Slide 6: Two ways to describe an image
- **Raw pixels:** flatten 28×28 into 784 brightness values; simple, but a shift of 2 pixels changes everything
- **HOG (Histogram of Oriented Gradients):** describes edge directions in small cells, so it captures shape, not brightness
- HOG settings were chosen on validation with a 12-configuration sweep: **12 orientations, 4×4-pixel cells, 2×2 blocks → 1,728 features**
- Small cells mattered most (+4–8 points); extraction costs about 0.22 s per 1,000 images

**Visual:** one image per class next to its HOG picture (little "stars" of edge directions).
**Speaker notes:** The rule for choosing HOG settings was written down before looking: highest validation accuracy, and
within 0.3 points prefer the smaller vector. The winner beat the runner-up by 0.35 points.

## Slide 7: One harness, no leakage
- Every model goes through the same code: tune on validation → refit on all 60k → test **once** → save results to JSON
- The test set is read by exactly one function, and every use is written to an audit log: **12 rows for 12 experiments**
- Scaling happens inside each model's pipeline, so it only learns from training rows
- The same evaluation function measures accuracy, macro F1, per-class recall, confusion matrix, time and model size for every model

**Visual:** a flow diagram: Train → Validation (choose) → Refit on 60k → Test once → JSON + audit log.
**Speaker notes:** Leakage is the most common way ML coursework goes wrong. We made it structurally hard: the only door to
the test set is one function, and it leaves a record every time it is opened.

---
### Part 3: Linear & distance-based models (Rajdeep Sanyal)

## Slide 8: Baseline, Logistic Regression and k-NN
- Dummy baseline (always predicts one class): **10.00%**, the floor every model must beat
- Logistic Regression: **84.72%** on pixels → **88.86%** on HOG (+4.14 points); best C = 0.01
- k-NN: **86.11%** on pixels, **86.31%** on HOG (k = 3, distance-weighted); HOG barely helps
- k-NN is cheap to train but slow to predict (0.7–6.9 s per 1,000 images) and stores all 60k images (189–415 MB)

**Visual:** grouped bars: pixels vs HOG for Logistic Regression and k-NN, with the 10% dummy line.
**Speaker notes:** HOG gives a linear model the shape information it cannot build itself. k-NN already compares whole images,
so swapping the description doesn't change its neighbours much.

## Slide 9: What a linear model looks at
- Each class's 784 Logistic Regression weights can be drawn as a 28×28 heat map
- Trouser: a strong negative strip between the legs; shoes: weight on the sole and ankle region
- The four tops share broad torso and sleeve regions, which is why a linear model struggles with them
- Best C sits at the edge of the grid (0.01, the smallest tried), so the true optimum might be a bit lower: a stated limitation

**Visual:** the 2×5 grid of Logistic Regression weight maps.
**Speaker notes:** We did not widen the grid afterwards, because that would have meant re-tuning and testing again.

---
### Part 4: Tree & kernel models (Rajat Srivastav)

## Slide 10: Random Forest and Linear SVM
- Random Forest: **87.56%** on pixels (300 trees, depth 20), **87.03%** on HOG; trees need no scaling
- HOG does **not** help the forest: each split looks at one feature, and one pixel says more than one HOG bin
- Linear SVM: **83.78%** on pixels → **88.98%** on HOG (+5.20 points, the biggest HOG gain); best C = 0.01
- Linear SVM on HOG is tiny (181 KB), but slow to tune: each fit on 54k images took 105–924 s, growing with C

**Visual:** Random Forest pixel-importance heat map (28×28) next to the pixels-vs-HOG bars for RF and Linear SVM.
**Speaker notes:** The forest's importance map is brightest in a vertical strip in the lower centre (the gap between trouser
legs), with some weight on the shoulder and neckline area, and almost none on the empty border. It reads shape directly from raw pixels.

## Slide 11: RBF SVM and a fair comparison
- RBF SVM: **89.70%** on pixels, **91.08%** on HOG (C = 10), the most accurate models
- Too slow to tune on 54k images, so the 9-setting grid was searched on a 10,000-image stratified subset, then refit on 60k
- Problem found: its validation score came from 10k training images while every other model used 54k, so it was unfairly low
- Fix, written down before running: refit the already-chosen settings on 54k and score on validation (no new search, no test)
- Fair ranking: **RBF HOG 0.9157 > RBF pixels 0.9082 > Logistic Regression HOG 0.8927**

**Visual:** a before/after bar chart of validation scores: "tuned on 10k" vs "refit on 54k".
**Speaker notes:** Without this fix, we would have picked Logistic Regression as "best by validation" for the wrong reason.
We fixed the comparison, not the rule, and we committed the plan before seeing the result.

---
### Part 5: Results, error analysis & conclusion (Omved Nagre)

## Slide 12: All 12 experiments
| Model | Pixels test | HOG test |
|---|---|---|
| Dummy | 10.00% | 10.00% |
| Logistic Regression | 84.72% | 88.86% |
| k-NN | 86.11% | 86.31% |
| Random Forest | 87.56% | 87.03% |
| Linear SVM | 83.78% | 88.98% |
| RBF SVM | 89.70% | **91.08%** |

- Best by validation = best on test: RBF SVM on HOG
- 5-fold cross-validation on a 10k subset: 0.8775 ± 0.0054, so the model is stable

**Visual:** the table above, with the RBF SVM HOG cell highlighted.
**Speaker notes:** RQ1: 89.70% from raw pixels. RQ2: HOG helps linear models a lot, the RBF SVM a little, and not trees or k-NN.

## Slide 13: Speed vs accuracy vs size (RQ3)
- Counting HOG extraction (~216 ms per 1,000 images):
- **Fastest above 87%:** Random Forest on pixels: 87.56%, ~12 ms per 1,000 images, but a 366 MB model
- **Best accuracy for its size:** Linear SVM on HOG: 88.98%, 181 KB model, ~219 ms per 1,000 (mostly HOG)
- **Most accurate:** RBF SVM on HOG: 91.08%, but ~29 s per 1,000 images, about 2,400× slower than the forest for +3.5 points

**Visual:** scatter plot: test accuracy (y) vs prediction time per 1,000 images on a log scale (x).
**Speaker notes:** There is no single winner. The RBF SVM compares each image with 26,343 stored support vectors, which is why it is so slow.

## Slide 14: Where the models fail (RQ4)
- **Shirt is the hardest class for all 10 trained models** (recall 50.9% to 75.2%)
- Best model's top mistakes: Shirt → T-shirt/top (104 of 1,000), T-shirt/top → Shirt (91), Pullover → Shirt (66), Pullover → Coat (61), Shirt → Coat (59)
- Mistakes among the four tops are **56–66% of every model's errors**, exactly the group the EDA flagged
- At 28×28, collars, buttons and cuffs cover only a few pixels

**Visual:** the per-class recall heat map (Shirt column clearly lighter) and the confusion matrix of the best model.
**Speaker notes:** The EDA predicted this before any model was trained. The models confirm it.

## Slide 15: Same mistakes, different features
- RBF SVM on HOG vs on pixels: 8,593 images right for both, **515 wrong for both**
- 391 of those 515 get the **same wrong label** from both models; 397 are upper-body garments
- HOG fixes 515 of the pixel model's errors and adds 377 new ones: a net gain of 138 images = the 1.38-point gap
- When two very different models make the same mistake, the image itself is probably ambiguous

**Visual:** a 2×2 grid (both right / only HOG wrong / only pixels wrong / both wrong) plus a few misclassified shirt images.
**Speaker notes:** Some of these shirts would fool a person too. Fixed handcrafted features have hit their ceiling here.

## Slide 16: Conclusions and Phase 2
- Classical ML reaches **91.08%** on Fashion-MNIST, chosen with zero test-set decisions
- HOG is the biggest help for linear models; the most accurate model is also by far the slowest
- Limitations: best settings at grid edges, timings from two laptops, validation slightly optimistic (val > test for every model)
- **Phase 2:** a small CNN on the same split and the same evaluation code. Hypothesis: learned filters catch collars and buttons and fix the Shirt / T-shirt / Pullover / Coat confusions
- Everything is reproducible: code, split, results, figures and a 32-entry decision log on GitHub

**Visual:** a simple "Phase 1 → Phase 2" arrow, with the four confused tops in the middle.
**Speaker notes:** Thank the audience and invite questions. Each presenter can answer questions on their own part.

---

## Backup: likely viva questions (short answers)
- **Why not choose the model by test accuracy?** Then the test score would no longer be an unbiased estimate; every choice was made on validation.
- **Why is validation higher than test for every model?** We picked the best setting on validation, which flatters it slightly (selection bias). Test is the honest number.
- **Why is CV (87.75%) lower than validation (91.57%)?** Each CV fold trains on only 8,000 images; it measures stability, not final accuracy.
- **Why does Random Forest not need scaling?** It splits one feature at a time on a threshold; rescaling doesn't change which split is best.
- **Why tune the RBF SVM on a subset?** Its training time grows faster than linearly; one 60k HOG fit took 451 s, a full grid on 54k would take hours.
- **How do we know there was no leakage?** One function reads the test set, the audit log has exactly 12 rows, scaling is inside pipelines, and EDA used training data only.
