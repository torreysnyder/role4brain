# Targets: whitening, number of PCs, output head, evaluation

All numbers here are for subject 1: the 7,604 most reliable ventral vertices,
the builder's split, and 1,000 shared images with three repeats each for the
signal estimates. The script that produced them is described in the last section.

## 1. Whitening: a limitation of the current pipeline

### What happens now

`role_approx_fmri.py` z-scores every target dimension separately:
- `compute_target_stats` returns one SD per dimension;
- `_standardize` divides both predictions and targets by it inside the loss
  (`train_epoch_role`) and the evaluation (`evaluate_role`);
- R² is averaged over dimensions with equal weight (`uniform_average`).

The targets are PCA scores, and `build_role_inputs.py` writes them raw. So the
training script **whitens** them: every principal component ends up with unit
variance, whatever its actual variance.

### Why that is a problem

The loss the model minimizes is

    L = sum_j (yhat_j - y_j)^2 / sigma_j^2

so component j is weighted by 1 / its variance. For ventral PCs, variance falls
steeply and the share of it that is signal (repeatable across presentations)
falls with it:

| PC | Variance | Relative to PC 1 | Whitening multiplies its weight by | Signal share |
|---|---|---|---|---|
| 1 | 744 | 1 | 1x | 0.52 |
| 2 | 434 | 0.58 | 1.7x | 0.73 |
| 8 | 45.7 | 0.061 | 16x | 0.27 |
| 32 | 8.4 | 0.011 | 88x | 0.33 |
| 128 | 2.4 | 0.0032 | 314x | 0.13 |

That makes **noise** a much larger share of what the loss sees. Values are from
`signal_profile.py`:

| k PCs | Subject 1, unwhitened | Subject 1, whitened | Subject 8, unwhitened | Subject 8, whitened |
|---|---|---|---|---|
| 8 | 0.40 | 0.42 | 0.60 | 0.62 |
| 16 | 0.42 | 0.52 | 0.64 | 0.74 |
| 32 | 0.44 | 0.59 | 0.66 | 0.80 |
| 64 | 0.47 | 0.69 | 0.68 | 0.84 |
| 128 | 0.50 | 0.78 | 0.70 | 0.88 |

Here "noise share" is the non-repeatable fraction of the training target,
weighted as the loss weights it.

Consequences:

1. **The legacy 8 PCs is nearly unaffected** (0.42 against 0.40 for subject 1),
   so earlier results are not invalidated by this.
2. **Whitening is what blocks raising k.** Each added component brings
   relatively more noise but enters the loss with the same weight as PC 1. At
   128 PCs, more than three quarters of the whitened objective is noise. For a
   noisier subject such as subject 8 it is close to 90%.
3. **It matters most for parameters shared across outputs.** For a model with
   separate parameters per output, such as ridge, whitening rescales each
   output's fit without changing its predictions. In ROLE, the role assigner,
   squeeze layer and projection hidden layer are shared by all outputs, so
   their gradients are sums over dimensions. Under whitening those sums are
   dominated by the many low-variance, noisy components, which means the role
   assignments are learned largely from noise. This harm is argued, not yet
   measured for ROLE: the ablation in section 5 measures it.
4. **R² is not comparable across k.** `uniform_average` R² over 8 whitened PCs
   and over 128 whitened PCs are different quantities, so they cannot rank runs.

### The fix

Keep the targets in vertex units and preserve their relative variances.

- **The builder writes unwhitened scores by default.** With an orthonormal
  basis, MSE on them equals vertex-space MSE inside the k-dimensional subspace.
  `--whiten` reproduces the legacy convention.
- **The patch adds `TARGET_SCALING = "global"`:** one shared scale (the root mean
  variance) instead of one per dimension. It keeps the optimizer's step sizes
  sensible without reweighting the components.
- **The patch adds `R2_MULTIOUTPUT = "variance_weighted"`:** 1 − total SSE / total SS.
- **Final scores are computed in vertex space** with `vertex_eval.py` (section 4).

## 2. How many PCs

"Noise" means two different things here:

- **Measurement noise** is what does not repeat across presentations of the same
  image. It is model-independent.
- **What a given model cannot explain** depends on the model.

The number of PCs should be set by the first. Setting it by the second bakes the
current model's limits into the targets: a model with richer fillers (visual
features, contextual embeddings) could explain components that category labels
cannot.

**Where the repeatable signal is.** This is the cross-repeat covariance projected
on the PCs, from `signal_profile.py`. The PCA is fit on non-shared images, and
the signal is estimated on the shared images.

| Top k PCs | Subject 1, share of variance | Subject 1, share of signal | Subject 8, share of variance | Subject 8, share of signal |
|---|---|---|---|---|
| 8 | 52% | 80% | 47% | 81% |
| 16 | 59% | 87% | 54% | 86% |
| 32 | 65% | 93% | 62% | 91% |
| 64 | 70% | 96% | 69% | 94% |
| 128 | 76% | 97% | 75% | 97% |
| 256 | 82% | 98% | 82% | 98% |

Signal is far more concentrated than total variance, and the profile is similar
for a complete and an incomplete subject.

**Where the label-predictable signal is.** These are vertex-space test R² scores
of models whose predictions are restricted to the top k PCs. The noise ceiling
is about 0.42.

| Top k PCs | Additive filler model | Its share of the full-rank score | Gain from pairwise co-occurrence |
|---|---|---|---|
| 3 | 0.109 | 80% | +0.007 |
| 8 | 0.130 | 95% | +0.008 |
| 32 | 0.136 | 100% | +0.008 |
| all | 0.137 | 100% | +0.006 |

- The additive filler model (bag of fillers) is ROLE with one role and a linear head.
- The pairwise co-occurrence term is fit on top of the additive model, as a
  rough stand-in for the non-additive structure ROLE's roles can capture. With no
  positional encoding, ROLE's output depends only on which categories are
  present, and pairwise co-occurrence is the first non-additive term of such a
  function.

So, for category labels, everything predictable sits in the top ~32 PCs.

**Recommendation:**
- The default is **128 PCs**. That is a model-independent margin: it holds
  about 97% of the repeatable variance in both subjects checked. Going to 256
  gains about 1 point.
- `--pcs` accepts any k, or `all` (every vertex, no PCA).
- With unwhitened targets a generous k costs little. Extra components add only
  their own small variance to the loss, and that variance is mostly noise.
- Keep `--pcs 8 --whiten` only to reproduce earlier results.
- **Check the signal profile per subject before relying on 128**, with
  `python multisubject_handoff/signal_profile.py --inputs <builder dir> --subject N`.
  Only subjects 1 and 8 were checked here, and other ROIs will differ.

## 3. The output head

The current projection head (`final_proj` in `role_learning_tensor_product_encoder.py`) is:

    TPR (1280) -> Linear(1280, D) -> ReLU -> Dropout -> Linear(D, D) -> LayerNorm(D)

Here D is the number of targets. Two problems:

- **The hidden width equals D.** At 8 PCs the 1,280-dimensional TPR is squeezed
  through 8 ReLU units. At D = 7,604 vertices the second layer would have 58M
  parameters. The patch adds `PROJECTION_HIDDEN_DIM`; 256 is a reasonable
  default, and it should exceed the signal dimensionality (about 100).
- **The output LayerNorm fixes each prediction's norm.** It subtracts each
  prediction vector's mean and divides by its SD across dimensions before the
  learned affine. Every prediction therefore has the same norm, and the model
  cannot predict a weak versus a strong response. For regression onto brain
  responses that is mis-specified. The patch adds `OUTPUT_LAYER_NORM`; set it to
  `False`.

The defaults in the patch reproduce the legacy head exactly, so existing runs and
checkpoints are unaffected.

## 4. Evaluation in vertex space

`vertex_eval.py` maps predictions back to vertices through the saved basis
(`basis_sub<NN>.npz`) and scores them against the subject's real betas. Every
run, whatever its k or scaling, then lands on one scale. It reports:

- **`r2_total`:** variance-weighted R² over vertices. This is the headline number.
- **`noise_ceiling`:** from NSD's noise-ceiling SNR (ncsnr) and each test image's
  actual repeat count. It is about 0.42 for subject 1.
- **`fraction_of_ceiling`:** `r2_total` / `noise_ceiling`.
- **`subspace_ceiling`:** the R² of the truth's own projection onto the target
  subspace, i.e. the most a model trained on k PCs could reach. It shows what
  truncation alone costs.

The patched `role_approx_fmri.py` saves raw test and validation predictions
(`role_predictions.npz`) for this script.

## 5. Ablations worth running

These settle whitening and k empirically for ROLE, on the single-subject inputs.
Fix everything else, including the head.

| Run | Targets | `TARGET_SCALING` | Purpose |
|---|---|---|---|
| a | 8 PCs, whitened | per_dim | legacy reference |
| b | 8 PCs | global | whitening at small k (expect little difference) |
| c | 128 PCs, whitened | per_dim | whitening at large k (expect harm) |
| d | 128 PCs | global | recommended default |
| e | all vertices | global | no PCA (needs `PROJECTION_HIDDEN_DIM`) |

Compare `r2_total` in vertex space, and the role diagnostics (filler-role mutual
information, role usage). Run c against d is the direct test of whether
whitening hurts the shared role assigner.

## Reproducing the numbers

**Signal and noise-share tables:** `signal_profile.py`.

**Per-component variance table, predictability table:** a one-off analysis
script, not included here. Its signal estimates used a PCA fit on training rows
that included the shared images, which is slightly optimistic for high
components. It takes:
- the PCA of the reference's training rows;
- the cross-repeat covariance of the three single-trial presentations of the
  1,000 shared images, projected on those PCs;
- ridge models (summed filler embeddings; plus pairwise category indicators seen
  at least 5 times, fit to the additive model's residuals), with their
  predictions projected onto the top-k subspace and scored on the test split
  against all 7,604 vertices.

Ridge is linear in its targets, so the projected all-vertex prediction equals a
model trained on k-PC targets with the same alpha.
