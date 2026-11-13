# Fashion-MNIST Phase 2: Handcrafted vs Learned Features — slide deck source

> **Instructions for NotebookLM:** Build a presentation of about 15 slides from this document, one slide per
> "Slide" section below, in the same order. Use the title, the bullet points and the key numbers exactly as written;
> do not invent numbers. Keep each slide to 3–5 short bullets. Use the "Visual" line to decide what to show and the
> "Speaker notes" as the talk track. The audience is a professor and examiners at a university capstone viva.
> Tone: clear, confident, plain English.

**Project:** Do Learned Features Resolve Fine-Grained Garment Confusions? A controlled comparison of small CNNs and
handcrafted-feature classifiers on Fashion-MNIST. Capstone Phase 2 (deep learning), building on Phase 1 (classical ML).

**Team and who presents which part**
| Part | Presenter | Enrollment No. | Topic | Slides |
|---|---|---|---|---|
| 1 | Tanmay Singh | 2401010476 | Recap, hypotheses & protocol | 1–4 |
| 2 | Anand Mishra | 2401010065 | How a CNN works & our architectures | 5–6 |
| 3 | Rajdeep Sanyal | 2401010370 | Training, early stopping & model choice | 7–8 |
| 4 | Rajat Srivastav | 2401020101 | Test results, significance & cost | 9–11 |
| 5 | Omved Nagre | 2401010311 | Error analysis & conclusion | 12–15 |

---

## Slide 1: Title
- Do Learned Features Resolve Fine-Grained Garment Confusions?
- Small CNNs vs handcrafted-feature classifiers on Fashion-MNIST
- Capstone Phase 2 · Team of 5 (names and enrollment numbers from the table above)

**Visual:** a clean title slide; a strip of 10 small clothing images, one per class.
**Speaker notes:** In Phase 1 we measured how far classical machine learning goes on clothing images. In Phase 2 we
ask whether a neural network that learns its own features does better, and, more importantly, where it does better.

## Slide 2: Where Phase 1 left us
- Best classical model: RBF SVM on HOG features, **91.08%** test accuracy
- Shirt was the hardest class for all 10 trained models (best recall 75.2%)
- Confusions among T-shirt/top, Pullover, Coat and Shirt = 56–66% of every model's errors
- The gap: pixels throw away which pixels are neighbours; HOG fixes the features before seeing any examples

**Visual:** the Phase 1 confusion matrix with the four upper-body garments highlighted.
**Speaker notes:** Phase 1 told us exactly where classical models fail. Every one of them struggled with the same four
tops. That gives Phase 2 a precise target.

## Slide 3: Two hypotheses, stated before training
- **H1 (accuracy):** a small CNN beats 91.08% under the same protocol
- **H2 (mechanism):** it wins *chiefly* by fixing the Shirt / T-shirt / Pullover / Coat confusions
- RQ5: does the CNN win? · RQ6: does it fix Shirt? · RQ7: does a third block help? · RQ8: what does it cost?

**Visual:** two boxes (H1, H2) with "tested in this talk" stamps; four RQ chips underneath.
**Speaker notes:** We wrote these down at the end of Phase 1, before any network existed. H2 is the interesting one,
because it predicts not just that learned features help, but where.

## Slide 4: A fair fight: same data, same test, same code
- Same split as Phase 1: 54,000 train / 6,000 validation / 10,000 official test
- Same `evaluate()` code scores the CNN and every classical model
- Validation decides everything (when to stop, which CNN); the test set is read once per network
- Audit log: 14 test evaluations in total, one per experiment
- CNNs learn from 54k images, classical models were refit on 60k, a handicap for the CNN

**Visual:** the protocol diagram: train → early stopping on validation → choose on validation → one locked test run → audit log.
**Speaker notes:** The comparison is only meaningful if nothing else changes. The CNN even has 10% less training data,
because early stopping needs the validation set, so any advantage we find is, if anything, understated.

## Slide 5: How a CNN sees an image
- A convolution slides a small learned 3×3 filter over the image and marks where its pattern appears
- **Local connectivity + weight sharing:** few parameters, same detector everywhere in the image
- ReLU keeps positive responses; 2×2 max pooling halves the size and tolerates small shifts
- Stacking blocks grows the receptive field: 10×10 pixels after 2 blocks, 22×22 after 3

**Visual:** a 3×3 filter sliding over a shirt image, producing a feature map; a small receptive-field diagram.
**Speaker notes:** Unlike HOG, nobody tells the network which patterns matter. It learns its filters from the training
images by gradient descent.

## Slide 6: Our two networks
- Block = Conv 3×3 (padding 1) + ReLU + MaxPool 2×2
- **CNN2:** 2 blocks (32, 64 filters) → 3,136 → Dense 128 → Dropout 0.5 → 10 · **421,642 parameters**
- **CNN3:** 3 blocks (32, 64, 128) → 1,152 → Dense 128 → Dropout 0.5 → 10 · **241,546 parameters**
- Deeper but smaller: the extra pooling shrinks the big dense layer
- But more compute: 7.6M vs 4.2M multiply-adds per image

**Visual:** the architecture diagram with the third block dashed; a small parameter-count table.
**Speaker notes:** A nice surprise: the deeper network has 43% fewer parameters, because almost all of CNN2's parameters
sit in the first dense layer, and the third pooling step makes that layer much smaller.

## Slide 7: Training with early stopping
- Adam, learning rate 0.001, cross-entropy loss, batches of 64, up to 30 epochs
- Early stopping: keep the weights with the lowest validation loss; stop after 5 epochs with no improvement
- CNN2 best at epoch 9 of 14; CNN3 best at epoch 12 of 17
- After the best epoch: training loss keeps falling, validation loss rises → overfitting, cut off by early stopping
- Seeded, deterministic, CPU: re-training gives identical numbers

**Visual:** the training-curves figure (`cnn_training_curves.png`), dashed line at the best epoch.
**Speaker notes:** The curves show the textbook picture: the two losses separate after the best epoch. Dropout keeps the
train–validation gap under half a point at the chosen epoch.

## Slide 8: Choosing the final CNN (validation only)
- Rule fixed before training: higher validation accuracy wins
- CNN2 **0.9228**, CNN3 **0.9308** → **CNN3 is the final CNN**
- Both beat the best classical model on validation (0.9157)
- Honest caveat: a CNN2 run differing only in batch order reached 0.9320, so the depth gap is within run-to-run noise
- What the third block clearly buys: 43% fewer parameters

**Visual:** a two-row table (CNN2, CNN3) with the winner highlighted and a "within noise" note.
**Speaker notes:** We follow the rule we set in advance, but we do not claim depth helps: the gap is smaller than the
variation we observed between two runs of the same network.

## Slide 9: Test results (RQ5)
- **CNN3: 92.09%** test accuracy (95% CI 91.54–92.60), macro F1 **0.9208**
- CNN2: 91.85%, macro F1 0.9176
- RBF SVM on HOG: 91.08%, macro F1 0.9108
- CNN3 makes **791 errors vs 892**: 101 fewer mistakes out of 10,000

**Visual:** a results table: CNN3, CNN2, RBF SVM HOG, RBF SVM pixels; CNN3 row highlighted.
**Speaker notes:** Each network was tested exactly once, after the choice was made. Both CNNs beat every classical model.

## Slide 10: Is the gain real? (significance)
- Paired comparison on the same 10,000 images: CNN3 fixes 417 images the SVM gets wrong, breaks 316
- McNemar test: χ² = 13.6, **p = 2.2 × 10⁻⁴** → significant
- CNN3 vs CNN2: 265 vs 241, p = 0.31 → **not** significant
- **H1 supported**

**Visual:** the 2×2 agreement table (both right 8,792 · only CNN3 right 417 · only SVM right 316 · both wrong 475).
**Speaker notes:** McNemar's test looks only at the images where the two models disagree. 417 against 316 is far too
lopsided to be luck; 265 against 241 is not.

## Slide 11: What does the CNN cost? (RQ8)
- CNN3: **94 ms** per 1,000 images vs ~29 s for the RBF SVM on HOG → **~300× faster**
- CNN3: **0.97 MB** vs 366 MB → **~377× smaller** (the SVM stores 26,343 support vectors)
- Training: a few minutes on a laptop CPU for both
- The CNN is more accurate, faster and smaller at the same time

**Visual:** two bar charts on a log scale: inference time and model size, CNN3 vs RBF SVM.
**Speaker notes:** The SVM compares every new image with every stored support vector; the CNN's cost is fixed by its
architecture. For the catalogue-tagging use case, this is the decisive argument.

## Slide 12: Did it fix the shirts? (RQ6)
- **No.** Shirt recall: CNN3 75.1% vs SVM 75.2%
- Shirt → T-shirt/top is still the biggest confusion (110 vs 104)
- The CNN improves the *other* tops: T-shirt +2.2, Pullover +2.8, Coat +1.5 points
- Of 1,000 test shirts: 90 right only for CNN3, 91 right only for the SVM, 158 wrong for both (p = 1.0)

**Visual:** the per-class recall chart (`cnn_recall_vs_classical.png`) with the Shirt bars circled.
**Speaker notes:** This is the surprise of the project. The two models trade shirts rather than fix them.

## Slide 13: Why Shirt stays hard
- 475 test images are wrong for both models; on 398 of them both give the **same** wrong label
- Two models that see the image completely differently agree on the mistake → the image is the problem
- Many "shirts" are collarless short-sleeved tops; some are sleeveless tank tops (test #612, #1129)
- The CNN does catch details HOG misses (test #40: visible buttons) and misses some HOG catches

**Visual:** the two "Shirt predicted as T-shirt/top" grids side by side (SVM above, CNN3 below).
**Speaker notes:** At 28×28 pixels a collar is two pixels wide, and some labels are genuinely debatable. No choice of
features can recover information that is not in the image.

## Slide 14: Verdict on the hypotheses
- **H1 supported:** CNN3 beats the best classical model, 92.09% vs 91.08%, p = 2.2 × 10⁻⁴
- **H2 rejected:** the gain is spread over many classes; Shirt is unchanged
- The four tops now hold an even larger share of the CNN's errors (68% vs 64%)
- Depth: no measurable accuracy gain, but a 43% smaller model

**Visual:** H1 with a green tick, H2 with a red cross, one supporting number under each.
**Speaker notes:** A rejected hypothesis is a result. It tells us the remaining error is an information limit of the
data, not a weakness of handcrafted features.

## Slide 15: Conclusion and next steps
- A 241,546-parameter CNN beats 12 tuned classical pipelines, and is ~300× faster and ~377× smaller
- Most remaining errors are ambiguous images shared by both model families
- Limitations: one training run per network, fixed hyperparameters, CNNs trained on 54k vs 60k
- Next: multi-seed runs; CNN features fed into the Phase 1 SVM; auditing the shared errors for label noise

**Visual:** a short summary panel; the repository link.
**Speaker notes:** Everything is reproducible from the repository: the split, the audit log, every result file and a
decision log explaining each choice.
