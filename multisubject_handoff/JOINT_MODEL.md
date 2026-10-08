# Approach B: one ROLE model, per-subject output heads

Train a single ROLE model on every subject's own responses. Each subject keeps
its targets in its own basis. Everything up to a shared hidden layer is shared;
each subject has its own linear output head. There is no alignment step, so
there is no alignment bias.

The input builder is done (`--mode joint`). The model wrapper and training loop
are specified here but not implemented, because they involve choices. Each
choice comes with a recommended default. **It requires the binding fix**
(`role_binding_fix.patch`; see the README).

## Architecture

    fillers (frozen BERT, 768) -> squeeze Linear(768,128)              shared
    role assigner (transformer over the filler set) -> roles (10-D)     shared
    TPR binding: sum_j filler_j (x) role_j -> 1280                      shared
    Linear(1280, h) -> ReLU -> Dropout                                  shared, h = 256
    Linear(h, D_k) + bias_k                                             per subject k

`D_k` is subject k's number of targets: k PCs in its own basis, or its 7,604
vertices with `--pcs all`. There is no output LayerNorm (see
`TARGETS_AND_WHITENING.md`, section 3).

With one subject, this is exactly the single-subject model run with
`PROJECTION_HIDDEN_DIM = h` and `OUTPUT_LAYER_NORM = False`. That equivalence is
the first test below.

**Roles are shared by construction.** The role assigner sees only the image's
fillers, so a given image gets the same roles whichever subject's response is
being predicted. Subjects differ only in their head. Whether a single
stimulus-driven role assignment *should* serve all brains is therefore an
assumption of this design. Evaluation 5 below tests it.

### Why the shared part is wide and the per-subject part small

The per-subject head is **not** low-rank in a restrictive sense. It is a full
h x D_k matrix, so every vertex or PC of every subject has its own weights. What
is constrained is that all subjects' predictions are linear readouts of one
shared h-dimensional representation. With h = 256, above the roughly 100
signal dimensions per subject (`TARGETS_AND_WHITENING.md`, section 2), that caps
no subject.

The reasons for putting the capacity in the shared part:

1. **Data.** Shared parameters see about 55k training rows; each head sees about 7k.
2. **It is what makes "shared roles" meaningful.** A deep, nonlinear head per
   subject could reinterpret the shared representation, or do the roles' work
   itself. Linear heads encode the hypothesis that subjects share *what* is
   represented and differ in *how it is laid out* on cortex. That is the
   assumption behind hyperalignment and shared-response models, and behind the
   linear alignment of approach A. It is also the design of shared-core,
   per-neuron-readout encoding models (for example, Sensorium).
3. **It is testable.** Compare linear with small nonlinear heads on held-out
   data, and run the leave-one-subject-out test (evaluation 4).

## Implementation notes

**Encoder.** Build `RoleLearningTensorProductEncoder` with
`final_layer_width=None`. With the binding fix it then returns the TPR,
`(B, 1280)`, together with the role predictions.

**Wrapper.** Hold the shared layer and the heads in a wrapper, for example
`MultiSubjectROLE`:

```python
z = self.shared(tpr)                                      # (B, h)
y = torch.einsum("bdh,bh->bd", self.head_weight[subject_index], z) \
    + self.head_bias[subject_index]                       # (B, D)
```

- `head_weight` has shape `(S, D, h)`, where S is the number of subjects.
- `subject_index` is a LongTensor of shape `(B,)`.
- Map subject numbers to 0..S-1 and store the mapping in the checkpoint.
- All subjects have the same D (the same `--pcs`, or the same vertex set), so
  mixed-subject batches need no padding.

**Data.** The patched loader already puts `subject`, `fmri_row`, `split` and
`aligned` on every instance, so batches can carry `subject_index`.

**Target scaling.** Standardize each subject separately:
- the per-dimension mean of its own training rows;
- **one global scale per subject** (`compute_target_stats(..., "global")` on that
  subject's training rows).

Subjects then contribute comparably to the loss without any of them being
whitened.

**Loss.**
- Use the mean over rows of the standardized MSE, so subjects are weighted by
  their number of training rows. Subjects 3, 4, 6 and 8 have about 10% fewer.
  A variant weights subjects equally.
- The role regularizers (pairwise diversity, filler independence) act on the
  role predictions, which do not depend on the subject, so they carry over
  unchanged.

**Batches.**
- Shuffle rows across subjects.
- A shared image appears once per subject. An optional optimization is to
  group these rows, so the encoder runs once per image and feeds several heads.

**Selection.**
- Early-stop and select the checkpoint on the **reference subject's** validation
  score, so the comparison with approach A and single-subject is like for like.
- Log every subject's validation score as well.

**Outputs.**
- After selection, evaluate `test` once for every subject.
- Save one predictions file per subject (`fmri_row`, `prediction`), and score
  each with `vertex_eval.py --inputs nsd_inputs/joint_pca128 --subject N`.

**Run time.** About 54,800 training rows means epochs about 7.7 times longer than
single-subject. The per-epoch train-set evaluation should use a subset (see
`ALIGNMENT.md`, Compute).

## Tests to write before training on real data

1. **Reduces to single-subject ROLE.** With S=1, copy the weights from a
   single-subject model with the same head settings. Predictions and role
   predictions must match to floating-point tolerance.
2. **Batch independence.** An image's output must not change when other images
   in its batch change. This would have caught the binding bug.
3. **Roles do not depend on the subject.** For the same fillers, the role
   predictions must be identical whatever `subject_index` is.
4. **Gradients stay with their own subject.** A loss on subject-k rows only must
   give exactly zero gradient to every other subject's head.
5. **Mixed-subject batches.** Shapes are correct, and each row uses its own
   subject's head.

## Evaluation

All of these use held-out rows, with the same builder flags and seed as the
other approaches, and are scored in vertex space.

1. **Head to head on the reference.** Vertex R² on the reference's 1,350 test
   images, which are identical to those in `single_*` and `aligned_*`. Use
   paired per-image differences against single-subject and approach-A runs.
2. **Negative transfer.** Joint versus single-subject on the reference. If joint
   is worse, the shared core is being pulled toward other subjects. Try a larger
   h, or weighting the reference more, before drawing conclusions.
3. **Every subject.** Report test vertex R² as a fraction of each subject's own
   noise ceiling; the ceilings differ between subjects. Where you can afford it,
   compare with single-subject runs for one or two other subjects (say 2 and 7).
4. **Leave one subject out.** This is the strongest test that the shared core
   transfers across brains.
   - Train the joint model on subjects 2-8.
   - Freeze everything up to the shared hidden layer.
   - Fit only subject 1's head. With the representation fixed and a linear head,
     this is closed-form ridge from `z` (256-D) to subject 1's targets: seconds.
   - Compare with subject 1's single-subject ROLE and with the full joint model.
   - If a core learned only from other brains matches subject 1's own model,
     the representation, roles included, generalizes across brains.
5. **Is a shared role assignment justified?** In this architecture roles are
   shared by construction, so test the assumption outside it.
   - Train single-subject ROLE models for two or three subjects independently,
     and two seeds for one of them.
   - On the `shared_test` images, compare the token-to-role partitions (argmax
     role per filler token) with the adjusted Rand index. It is invariant to
     how roles are labelled.
   - Compare **cross-subject** agreement with **cross-seed** agreement for the
     same subject (the ceiling) and with shuffled images (chance).

   How to read it:
   - **Cross-subject close to cross-seed:** the role structure is shared across
     brains, and approach B's assumption holds.
   - **Cross-subject near chance:** each brain induces different roles. That
     would be a finding in its own right, and a reason to prefer per-subject
     models.
6. **Head capacity.** Linear heads versus a small per-subject MLP. If the MLP
   wins clearly, subjects differ in more than layout.
7. **Role diagnostics.** Filler-role mutual information, role usage and the
   snapshots, against single-subject runs (as in `ALIGNMENT.md`, A5).

## Pitfalls

- **Rows of different subjects are in different units.** Each subject has its
  own basis and scale, and noise ceilings differ (subject 8's reliability is
  about 0.43, subject 2's about 0.73). Never compare raw MSE across subjects.
- **The split between the shared layer and the heads is not unique.** The shared
  layer's output can be re-mixed by any invertible linear map if the heads undo
  it, so do not interpret `z`'s coordinates. Predictions and roles are
  unaffected.
- **A good fit does not show that the roles are shared.** The architecture
  forces it. Evaluations 4 and 5 are what support that claim.
