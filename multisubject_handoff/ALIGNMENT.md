# Approach A: ridge alignment, then pool

Train the **unchanged** ROLE model (`role_approx_fmri.py` with both patches) on
the reference subject's own rows plus other subjects' rows mapped into the
reference's target space. Everything here is implemented, apart from running it.

## What `build_nsd_role_inputs.py --mode aligned` does

For each source subject `k` (by default, all subjects except the reference):

1. **Fitting images:** the shared images that both the reference and `k` saw,
   excluding `shared_test`. That is 707-800 images, depending on the subject.
2. **Ridge fit:** from `k`'s 7,604 vertex responses to the reference's targets,
   which are its unwhitened PCA scores by default. Alpha is chosen by 5-fold
   cross-validated, variance-weighted R² inside the fitting images.
   - Regressing onto the targets is exactly equivalent to Ferrante et al.'s
     vertex-to-vertex map followed by projection, because ridge is linear in
     its targets.
   - R² is used rather than correlation because the aligned values become
     training targets, so their scale matters.
3. **Mapped rows:** `k`'s own **training** images (unique images only) are
   mapped and added as rows with `aligned: true`, `subject: <reference>` and
   `source_subject: k`.
   - Using only `k`'s training images keeps the image sets identical to what the
     joint model trains on.
   - `k`'s shared images are never added: the reference already has real
     responses for them, and the alignment fit has seen them.

`build_summary.json -> alignment` records, per source subject: the number of
fitting images, alpha, cross-validated R² and r, and the variance ratio.

With subject 1 as reference: 7,100 own training rows become 49,494 pooled rows,
42,394 of them aligned.

**Training.** With `SPLIT_FROM_METADATA = True`, the patched
`role_approx_fmri.py` trains on all `train` rows, own and aligned, and
validates and tests on the reference's real `val`/`test` rows only. The target
statistics come from real rows only, so pooled and single-subject runs are in
identical units.

## Experiments

Use the same target settings and seed throughout. Score every run with
`vertex_eval.py`.

**A1. Additive baseline (seconds).**
```bash
python multisubject_handoff/additive_baseline.py --inputs nsd_inputs/aligned_pca128
```
It reports the own-data learning curve, own + aligned, and aligned-only.

For subject 1 at 128 PCs:
- own data reaches 0.138 vertex R² and is nearly saturated;
- own + aligned gives 0.128;
- aligned only gives 0.123.

So pooling hurts the additive model. ROLE has to show that it is different.

**A2. Is ROLE limited by data? (the key experiment).** Build single-subject
inputs at reduced training fractions. The PCA basis is fit before thinning, so
the targets are the same at every fraction:
```bash
for F in 0.25 0.5; do
  python multisubject_handoff/build_nsd_role_inputs.py --mode single \
      --train-fraction $F --out-dir nsd_inputs/single_pca128_f$F
done
```
Train ROLE on `f0.25`, `f0.5` and the full `single_pca128`. Plot vertex R²
against the number of training images, next to the additive baseline's curve.

- **ROLE's margin over the baseline still growing at 100%:** more data should
  help ROLE, so pool.
- **ROLE flat like the baseline:** neither pooling approach can help much.

**A3. Pooled runs.** Point `FMRI_METADATA_JSON` and `FMRI_TARGETS_NPY` at
`nsd_inputs/aligned_pca128/` and train as for single-subject. Compare vertex R²
on the reference's test images with the single-subject run. Use paired
per-image differences (the predictions files hold `fmri_row`), not two point
estimates.

**A4. Dose.** Rebuild with `--subjects 1 2 5 7` (the four complete subjects)
and with `--subjects 1 2`. If pooling helps, the gain should grow with the
number of source subjects. If it shrinks or reverses, alignment bias dominates.

**A5. Role diagnostics.** Compare pooled and single-subject runs on what the
script already tracks:
- filler-role mutual information;
- filler independence;
- role usage, entropy and sparsity;
- the role snapshots.

A pooled model that predicts better but uses fewer roles, or assigns roles more
by filler identity alone, has not found more role structure.

## Regression dilution: a bias more shared images cannot remove

Ridge from a *noisy* source subject attenuates the stimulus-driven signal, and
this happens even with unlimited shared images. Write the reference response as
y = f(s) + noise and the source response as x = g(s) + e, with f = M g. The
population regression of y on x is

    W = M * Sigma_g * (Sigma_g + Sigma_e)^(-1)

So the signal part of an aligned target, `W g(s)`, is a shrunk version of
`f(s) = M g(s)`. Each direction is shrunk by roughly that direction's source
reliability. A model trained on aligned rows learns that shrunk target.

The builder's `cv_variance_ratio` (0.26-0.32 at 128 PCs) shows the total
shrinkage. Part of that is legitimate removal of source noise; the attenuation
of signal is the part that hurts.

**Evidence so far:** at 8 PCs the additive model's pooled loss was almost all
scale, with R² down 0.020 and the per-component correlation down only 0.003.
At 128 PCs the correlation drops too, so not all of the loss is scale.

**Possible remedy (untested):** rescale aligned rows per component. Estimate the
attenuation from the shared images' repeats, where split-half covariance gives
`Sigma_g`. Alternatively, weight aligned rows below own rows in the loss.

## Compute

- **Train-set evaluation is the slow part.** The script evaluates the full
  training set every epoch, one image at a time (`evaluate_role(train_set)`).
  With 49k pooled rows that pass dominates the epoch. Evaluate a fixed random
  subset of the training rows instead, or batch the evaluation.
- **Training cost grows with the data.** At `BATCH_SIZE = 4`, a pooled epoch is
  about 12k optimizer steps against 1.8k for single-subject. Use a larger batch
  for both runs being compared, so the batch size is not a hidden difference.

## Pitfalls

- **Aligned rows never enter `val` or `test`.** The builder guarantees this and
  the patch respects it.
- **Don't compare runs built with different `--scope`, `--vertices`,
  `--shared-test`, `--seed`, `--pcs` or `--whiten`,** or run with a different
  `MAX_SEQ_LEN`.
- **Role indices are arbitrary across runs.** A pooled model's role 3 and a
  single-subject model's role 3 are unrelated. Compare them only after matching
  roles, for example with Hungarian matching on filler-by-role usage.
