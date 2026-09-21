# Task 2 - implementation and study notes

These are engineering notes that explain what the code does and why, so the
report can be written from real understanding. They are not report text.

## Fixed protocol (shared by every method)

* Backbone: torchvision ResNet-18, `IMAGENET1K_V1`, `fc` replaced by a 7-class
  linear layer. **Full fine-tuning** for all methods (`shared/backbone.py`).
* Preprocessing: Resize(256) -> RandomCrop(224) + HFlip (train) / CenterCrop(224)
  (eval) -> ToTensor -> ImageNet normalize (`shared/pacs_protocol.py`).
* Batch per update: 8 Photo + 8 Art + 8 Cartoon source images, plus 24 Sketch
  images for target-aware methods (DAN/DANN/CDAN). Loaders are cycled, so the
  three domains stay balanced even when their sizes differ.
* Optimizer AdamW, lr 1e-4, weight decay 1e-4; <=30 epochs; early stop after 5
  epochs without improvement in **mean source-validation macro-F1**; seed 6304.
* One epoch = `sum(source train sizes) / 24` updates ~ one pass over the source
  training set with equal domain weight.

## Frozen BatchNorm running statistics (`shared/bn_policy.py`)

Source and target come from different visual distributions. If BN updated its
running mean/variance on the mixed batches, the backbone would silently adapt
to the target mixture - an implicit adaptation method competing with the one
being studied. The assignment therefore pins `running_mean`, `running_var` and
`num_batches_tracked` to their pretrained ImageNet values for every method.

In PyTorch this requires care: `model.train()` switches BN modules back to
training mode. The code calls `model.train()` and then puts *only* the BN
modules into `eval()`; the affine parameters gamma/beta remain ordinary
trainable parameters and keep receiving gradients. The smoke test asserts that
the running statistics do not change after an optimizer step and that gamma
still has a finite gradient.

## Source-only ERM

`CE(f(x_s), y_s)` on the three source domains. Establishes the unadapted domain
gap and doubles as the Task 3 ERM baseline, so the Task 2 checkpoint is reused
unchanged later (do not retrain it for Task 3).

## DAN (`task2/methods/dan.py`, `shared/mmd.py`)

    L = L_cls + lambda * MMD^2(F(X_s), F(X_t))

MMD compares the mean embeddings of the two feature distributions in a kernel
Hilbert space: it is zero exactly when all kernel moments match. The kernel
trick avoids constructing phi; the biased estimator used here is

    MMD^2 = mean(k(x,x)) + mean(k(y,y)) - 2 * mean(k(x,y))

with `k` a sum of three RBF kernels. The bandwidths are 0.5, 1 and 2 times the
median pairwise **squared** distance of the current combined batch, so the
kernel scale adapts to feature magnitudes as training progresses but cannot be
gamed by the optimizer (the median is computed under `no_grad`).

Why a sum of kernels: a single bandwidth either smooths away differences (too
wide) or only sees nearest neighbours (too narrow). Summing three scales is the
usual DAN recipe.

Note the penalty is marginal: it compares overall feature distributions and does
not know which target example belongs to which class. That is exactly the
question CDAN addresses.

## DANN (`task2/methods/dann.py`, `shared/grl.py`)

A binary discriminator (512 -> 256 -> ReLU -> dropout 0.5 -> 2) tries to
separate source from target features. The backbone should instead make them
indistinguishable, i.e. maximise the discriminator loss, which is implemented by
negating and scaling the gradient in a custom autograd function:

    forward(x) = x            backward(g) = -alpha * g

The adversarial strength follows

    alpha(p) = 2 / (1 + exp(-10 p)) - 1,   p = global step / total steps

At p = 0 alpha is 0, so the representation first learns the class task; by
p ~ 0.5 alpha ~ 0.99, so domain confusion dominates later. Only source examples
contribute to the class loss; both source and target contribute to the domain
loss (weight 1). The controlled study varies `max_alpha` in {0.25, 0.5, 1}
while keeping the same schedule shape.

## CDAN (`task2/methods/cdan.py`)

The discriminator sees `g(x) = vec(f(x) outer p(x))` with `p = softmax(C(f(x)))`,
i.e. the 512-d feature multiplied by the classifier's 7 class probabilities
(3584-d input). This conditions the domain decision on the predicted class, so
alignment can happen between semantically corresponding regions of the two
domains instead of mixing classes across domains. The assignment requires that
`f` and `p` are **not** detached, so the domain loss shapes both the features
and the classifier's confidence; entropy conditioning is explicitly forbidden.

## Checkpointing and early stopping

Every epoch each source validation domain is evaluated separately and the
checkpoint score is the mean of their macro-F1 values (classes are balanced in
PACS, but macro-F1 protects against a majority-class illusion). The best epoch's
`best.pt` and the latest `last.pt` (with optimizer state, for `--resume`) go to
`checkpoints/<run_id>/`. Reporting per-domain values, not just the mean, is
required evidence: pooling the three source domains can hide one being
sacrificed.

## Domain separability (`task2/evaluation/domain_separability.py`)

Freeze the adapted backbone, pool equal numbers of source-validation and target
features (seed 6304 picks which), split 70/30 with seed 6304, train a balanced
logistic regression with C = 1, and report held-out accuracy. 50% = chance =
domain information was removed. Important interpretation guard: a low score is
evidence that domain information is harder to recover, **not** that class
information survived; a collapsed representation can also produce 50%. Always
read it together with source performance and per-class target behaviour.

## Scale stability of alignment objectives under frozen BatchNorm

With BN running statistics frozen, the feature *scale* becomes a free parameter
that every alignment objective can exploit, in both directions:

* **DANN / CDAN** (domain loss): the backbone inflates features to make the
  discriminator's task easier to distort; norms went 27 -> 130,000 in 30 local
  steps and the loss exploded. DANN diverged inside epoch 1 and CDAN peaked at
  epoch 2 before exploding at epoch 4.
* **DAN-DG** (pairwise MMD): the optimizer does the opposite - it *shrinks*
  features (mean norm 20 -> 5) until the pairwise discrepancy is tiny, which
  destroys the classifier (class loss pinned at ln 7 ~ 1.946, validation F1 at
  chance). Task 2's DAN reduces MMD the same way in principle.

The uniform stabilisation is to make **every alignment term consume
L2-normalised 512-d features** (`normalize_features: true` in the DAN, DANN,
CDAN and DAN-DG configs). The classification path always uses the
unnormalised feature, so the objectives, architectures and hyperparameters are
unchanged; only the degenerate scale direction is removed. Unstabilised runs
are kept as documented failures (`results/` history files); the same
normalisation is what makes the Task 2 vs Task 3 "target access" comparison
fair.

## Task 1 - cue-conflict representation stability (the pairing)

The assignment requires cosine stability between each transformed image and its
clean counterpart for grayscale, cue conflict, translation and patch shuffle.
Grayscale/hue/translation/shuffle are derived from test-subset images, so the
clean and transformed features live in the same cached file and are paired by
position.

Cue conflicts are *generated*: a content image from class A is stylised with a
style image from class B, so there is no "transformed version of a test image"
to pair. The pairing used here is content image vs conflict image:

1. `make_cue_conflicts.py` records, for every accepted conflict, the STL-10
   index of its content and style source images in `manifest.json`
   (`items[].content_index`).
2. `run_task1.py` (`_conflict_stability`) maps that index to the row of the
   already-cached clean train features (`<backbone>_train.npz`, whose `indices`
   array holds the original STL-10 index of each cached image) and computes
   `cos(f(content), f(conflict))`, averaged over accepted conflicts whose
   content image is present in the cache.
3. The value is stored as `stability.conflict.<backbone>` in
   `results/task1/analysis.json`.

No new forward passes are needed (train features were already extracted for the
linear probes), and the pairing is model-free: it uses the generator's record,
not any model prediction.

## What to inspect when real results arrive

* **Curves** (`curves_train.png`, `curves_val.png`): does the discriminator
  accuracy fall towards chance (DANN/CDAN)? Does the MMD penalty decrease (DAN)?
  Both near chance with collapsing source accuracy may mean alignment is
  dominating the class objective.
* **Main table**: per-domain source macro-F1, target accuracy/macro-F1, change
  vs source-only, separability score. Watch for gains that come with large
  source drops (over-alignment) and for class-specific negative transfer
  (`target_per_class.csv`, `target_confusions.csv`).
* **Failure cases**: `target_outputs.npz` keeps logits and paths so specific
  images can be inspected for the report.
