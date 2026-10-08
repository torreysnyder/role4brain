# Training the ROLE model on several NSD subjects

This folder holds what is needed to test whether training the transformer ROLE
model (`role_approx_fmri.py`, `role_learning_tensor_product_encoder.py`,
`role_assigner.py`) on more than one NSD subject helps, and which way of pooling
works better:

| | Approach A: align, then pool | Approach B: shared model, per-subject heads |
|---|---|---|
| Idea | Map every source subject's responses into the reference subject's target space with ridge regression fit on the shared images, then train the unchanged ROLE model on the pooled rows | One ROLE model for all subjects: shared fillers, role assigner and TPR binding, plus a small linear output head per subject, each predicting that subject's own responses |
| Code needed | Input builder and patches (provided) | Input builder (provided) + a multi-subject wrapper and training loop (spec in `JOINT_MODEL.md`, to implement) |
| Main risk | Alignment bias: about 900 shared images limit how well one brain predicts another, and ridge shrinks the signal | Negative transfer: the shared role assigner may be pulled away from what is best for the reference subject |
| Details | [`ALIGNMENT.md`](ALIGNMENT.md) | [`JOINT_MODEL.md`](JOINT_MODEL.md) |

## Read first: two problems in the current single-subject pipeline

Both affect any ROLE run, pooled or not.

### 1. The TPR binding sums over the batch, not over each image's fillers

`role_binding_fix.patch` fixes this.

**What changed.** Commit `e683bde` (2026-03-17) started transposing the fillers
to sequence-first, (S, B, filler_dim), before binding. But
`SumFlattenedOuterProduct` computes `bmm(fillers^T, roles)`, which sums over
dimension 1, so it assumes batch-first inputs. All three binders do.

**What happens now**, checked on the committed code:
- the outer products are summed over the **batch** instead of the sequence;
- every forward pass returns shape (1, S, D), one output per sequence position
  rather than per image;
- each image's output changes when the *other* images in its batch change.

**Consequence.** With the repo's 16-token inputs, the training loop compares a
flattened (1, 16·D) prediction with (B, D) targets and fails with a shape error.

**Before applying it, check what you actually run.** If your local copy differs
from the committed one, check which binding it uses. If your runs worked, your
copy must differ. If it matches the committed code, apply the patch.

**What the patch does:**
- keeps the fillers batch-first;
- transposes the roles to batch-first;
- drops the binder's leading singleton dimension;
- makes the encoder's `eval()` return `self`, as `nn.Module.eval()` does, so
  `model.eval()` can be chained.

**Verified:** with the fix, a batch of 4 gives a (4, D) output, and image 0's
output no longer changes when images 1-3 do.

### 2. The targets are whitened

`role_approx_fmri.py` z-scores every PCA component, so the loss weighs the
128th component like the first. At 8 PCs this barely matters. At 128 PCs it
turns more than three quarters of the objective into noise.

This, the choice of how many PCs to use, and the output-head problems are
explained in **[`TARGETS_AND_WHITENING.md`](TARGETS_AND_WHITENING.md)**. Read it
before choosing targets.

## Contents

| File | What it is |
|---|---|
| `TARGETS_AND_WHITENING.md` | Whitening, number of PCs, output head, vertex-space evaluation, ablations |
| `ALIGNMENT.md` | Approach A: what the builder does, experiments, controls, pitfalls |
| `JOINT_MODEL.md` | Approach B: architecture, training changes, tests, evaluation |
| `build_nsd_role_inputs.py` | Builds ROLE inputs from real NSD betas: `--mode single / aligned / joint`, `--pcs K` or `all`, optional `--whiten` |
| `vertex_eval.py` | Scores saved predictions in vertex space, with noise and subspace ceilings |
| `signal_profile.py` | Per-subject check of where the repeatable signal is, and of the noise share with and without whitening |
| `additive_baseline.py` | Closed-form bag-of-fillers floor: learning curve, pooled versus own data, aligned-only. Takes seconds |
| `role_split_scaling_head.patch` | `role_approx_fmri.py` + encoder: splits from metadata, train-only statistics, target scaling, R² option, head options, saved predictions. Defaults reproduce current behaviour |
| `role_binding_fix.patch` | The binding fix above. Separate, so you can decide on it independently |

## Prerequisites

- **Conda environment:** `role4brain`.
- **Repository files:** `role_approx_fmri.py`, `role_learning_tensor_product_encoder.py`,
  `role_assigner.py` and `binding_operations.py`.
- **Filler files, shipped in `data/`:**
  - `bert_filler_vocab.json`: the 80 COCO category names, numbered 1-80
    alphabetically, with 0 for padding;
  - `bert_embedding_matrix.pt`: the matching 81 x 768 matrix, built by
    `build_role_inputs.py` (not part of this package). It holds mean-pooled
    `bert-base-uncased` vectors, centered and rescaled, with row 0 the zero
    padding vector.

  They are a matched pair: the builder writes filler ids with this vocabulary,
  so the ROLE run must use this matrix. They live in `data/` rather than at the
  repo root so they cannot overwrite your own root copies, which may have been
  built differently. Both match the repo's `*.pt`/`*.json` ignore rules and were
  added with `git add -f`.
- **Per-subject betas:** `sub<NN>_ventral.npz` for subjects 1-8, written by
  `nsd_real_betas.py --subject N`. On the lab server they are in
  `/data/giacomo/natural-scenes-dataset/nsd_ventral_fsaverage/`.
- **NSD and COCO metadata:** `nsd_stim_info_merged.csv` and the COCO 2017
  `instances_*.json` annotations. The defaults point at the lab copies; all
  paths are flags.
- **Cluster gotcha: limit the BLAS threads in every job.** The environment's
  numpy uses an OpenBLAS that starts up to 64 threads. On the shared DGX they
  thrash: a 4000 x 768 Gram matrix took 12 s instead of 0.07 s. Add this after
  `conda activate`:
  ```bash
  export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  ```

## Applying the patches and configuring a run

```bash
patch -p1 < multisubject_handoff/role_split_scaling_head.patch
patch -p1 < multisubject_handoff/role_binding_fix.patch   # after checking your binding
```

The two patches apply in either order. The first one adds constants at the top
of `main()` in `role_approx_fmri.py`. For builder outputs, set the following.
`BERT_EMBEDDING_PATH` is not in that block; it is set further down in `main()`,
just before the matrix is loaded. Point it at
`multisubject_handoff/data/bert_embedding_matrix.pt` as well.

```python
FMRI_METADATA_JSON = "nsd_inputs/single_pca128/fmri_metadata.json"
FMRI_TARGETS_NPY   = "nsd_inputs/single_pca128/fmri_targets.npy"
TARGET_ENCODING_DIM = None            # infer from the targets
SPLIT_FROM_METADATA = True            # train/val/test from the builder; test used once
TARGET_SCALING = "global"             # not "per_dim" (whitening)
R2_MULTIOUTPUT = "variance_weighted"
FILLER_VOCAB_JSON = "multisubject_handoff/data/bert_filler_vocab.json"
PROJECTION_HIDDEN_DIM = 256           # head width independent of the output size
OUTPUT_LAYER_NORM = False             # see TARGETS_AND_WHITENING.md, section 3
PLOT_ROLE_BBOXES = True               # the builder writes COCO ids and coco_split for it;
                                      # untested on NSD inputs (needs FiftyOne)
```

After training, the script evaluates the best checkpoint once on `test` and
writes `role_predictions.npz`. Score it with:

```bash
python multisubject_handoff/vertex_eval.py --inputs nsd_inputs/single_pca128 \
    --predictions role_predictions.npz
```

**Note on `MAX_SEQ_LEN = 15`.** It drops images with more than 15 categories.
The builder pads to 18, the NSD maximum. Keep the setting identical across runs
you compare. `vertex_eval.py` scores whatever rows were predicted and says how
many it skipped.

## The three input modes share one evaluation set

Every mode writes:
- `fmri_targets.npy`;
- `fmri_metadata.json`;
- `basis_sub<NN>.npz`, one per subject with its own rows;
- `build_summary.json`.

Records keep the legacy loader's fields (`filler_ids`, `fmri_row`, `image_id` =
COCO id, `split`). They add `subject`, `source_subject`, `aligned`, `shared`,
`nsd_id`, `nsd_index` and `coco_split`.

The following are fixed by `--scope`, `--vertices`, `--shared-test` and `--seed`,
**never by `--mode`**:

- **Vertices:** the top 7,604 ventral vertices by noise-ceiling SNR (ncsnr)
  averaged over all 8 subjects. fsaverage gives every subject the same vertex
  indices.
- **The split is by image:**
  - 200 of the 907 shared images all 8 subjects saw are `shared_test`, held out
    from training, alignment and selection;
  - the other shared images are `train` for every subject;
  - each subject's unique images get a seeded 70/15/15 `train`/`val`/`test` split.
- **Reference targets** are fit on subject 1's training rows. The reference rows
  are identical in every mode.

**The primary comparison** is vertex-space test R² on the reference subject's
1,350 `test` images. These are real responses in every approach.

## Suggested order

1. **Decide on the binding fix.**
2. **Run the target ablations on single-subject inputs** (`TARGETS_AND_WHITENING.md`,
   section 5). They fix the target settings everything else uses.
3. **Build the pooled inputs** (CPU; about 20-60 s each when storage is idle):
   ```bash
   python multisubject_handoff/build_nsd_role_inputs.py --mode single  --out-dir nsd_inputs/single_pca128
   python multisubject_handoff/build_nsd_role_inputs.py --mode aligned --out-dir nsd_inputs/aligned_pca128
   python multisubject_handoff/build_nsd_role_inputs.py --mode joint   --out-dir nsd_inputs/joint_pca128
   ```
4. **Run the additive baseline** (seconds):
   `python multisubject_handoff/additive_baseline.py --inputs nsd_inputs/aligned_pca128`
5. **Check whether ROLE is limited by data.** Run single-subject ROLE learning
   curves (`ALIGNMENT.md`, A2). If ROLE barely improves between 50% and 100% of
   the reference's data, pooling cannot help much.
6. **Approach A:** pooled ROLE runs on `aligned_pca128` (`ALIGNMENT.md`).
7. **Approach B:** implement and test the joint model (`JOINT_MODEL.md`).
8. **Compare** A, B and single-subject on the reference test images in vertex space.

## Results from verifying the package (2026-10-08)

These use seed 42, all 8 subjects and subject 1 as reference.

**Inputs at 128 PCs:**

| Mode | Rows | Training rows | Notes |
|---|---|---|---|
| single | 10,000 | 7,100 | 1,350 val, 1,350 test, 200 shared_test |
| aligned | 52,394 | 49,494 | 42,394 of the training rows aligned |
| joint | 77,240 | 54,768 | each subject in its own basis |

- **Variance explained:** 128 PCs capture 72-82% of each subject's vertex variance.
- **Consistency checks passed:**
  - the reference rows are identical across modes (images, splits, fillers,
    COCO ids; target difference 0);
  - no image carries two split labels;
  - `--whiten` gives unit-SD components, and the default keeps raw variances
    (PC 1: 744, PC 128: 2.3).

**Alignment onto subject 1 at 128 PCs:**
- cross-validated variance-weighted R² of 0.28-0.35 across source subjects;
- alpha 1e4 for every subject;
- aligned targets keep 26-32% of the real targets' variance (ridge shrinkage).

**Additive filler baseline** (vertex-space test R² on subject 1's 1,350 test
images; noise ceiling 0.421):

| Training data | Targets | n | Vertex R² | % of ceiling |
|---|---|---|---|---|
| Own 12.5% | 128 PCs | 888 | 0.114 | 27% |
| Own 50% | 128 PCs | 3,550 | 0.134 | 32% |
| Own 100% | 128 PCs | 7,100 | 0.138 | 33% |
| Own 100% | 8 PCs, whitened | 7,100 | 0.130 | 31% |
| Own 100% | all vertices | 7,100 | 0.137 | 32% |
| Own + aligned | 128 PCs | 49,494 | 0.128 | 30% |
| Aligned only | 128 PCs | 42,394 | 0.123 | 29% |

What this says:
- **The additive model is close to saturation:** the last doubling of data adds 0.004.
- **Pooling aligned data hurts it.**
- **8 PCs lose a little against 128.** The most a model trained on 8 PCs could
  reach (its subspace ceiling) is 0.52 R², against 0.75 for 128 PCs.
- **Whether ROLE behaves differently is the open question.**

**ROLE smoke test.** Both patches were applied, with the settings above, on
`single_pca128`, for 1 epoch at batch size 64 (45 s on a 20 GB GPU slice). It
showed that:
- training runs;
- checkpoint selection uses `val`;
- `test` is evaluated once and `role_predictions.npz` is written;
- `vertex_eval.py` scores all 1,350 test images.

Test R² was about 0 after one epoch at the start of the temperature schedule.
That shows the plumbing works and says nothing about the model.

Not run: a full training run, `PLOT_ROLE_BBOXES` (FiftyOne), and the unpatched
legacy path. The patch defaults reproduce the legacy code path, apart from the
binding bug, which makes that path fail.

## What is deliberately left open

- **The joint model's implementation.** `JOINT_MODEL.md` gives defaults and the
  reasons for them.
- **Weighting aligned rows below own rows**, or rescaling them to undo ridge
  shrinkage (`ALIGNMENT.md`).
- **Per-subject checks of the 128-PC choice.** Only subjects 1 and 8 were
  checked: 97% of the repeatable signal in both. Use `signal_profile.py` for
  the others.
