# Agent Guide

## Project Snapshot

This repository is a Python research codebase for characterizing
representations of natural scenes in visual cortex by decomposing them into
filler and role representations. The main scientific target is human fMRI data
from the Natural Scenes Dataset (NSD), with possible monkey visual-cortex data
added later. The intended decomposition uses embeddings of category labels
present in each image as fillers, learns roles with the ROLE architecture, and
binds fillers and roles using tensor products in the Smolensky/McCoy TPR style.

The repo also contains earlier fixed-role TPDN experiments based on McCoy et al.
2018/2019-style Tensor Product Decomposition Networks, plus toy number-sequence
experiments for checking TPDN and ROLE behavior in a controlled setting. The
main toy path described in `README.md` trains a Transformer-based role learner
to approximate TPDN encodings for fixed-length number sequences.

The repository is script-oriented, not packaged as an installable module. Most
configuration is held as constants inside individual scripts.

## Component Map

- Scientific target: natural-scene representations in visual cortex. Human NSD
  fMRI is the current focus; future non-human primate data should be treated as
  the same high-level target with different data loaders/metadata.
- Fillers in the current NSD/fMRI ROLE experiments: category-label embeddings
  for objects/categories present in each image, typically using BERT-derived
  vectors or IDs that index a BERT embedding matrix.
- Learned-role model: `role_approx_fmri.py` applies the ROLE architecture to
  fMRI targets. It uses category-label filler sequences, learns role
  assignments, binds fillers to learned roles via a tensor product, and projects
  the result to PCA-reduced fMRI response targets.
- ROLE/TPR core modules: `role_learning_tensor_product_encoder.py` implements
  the filler embedding, role assignment, binding, and optional projection logic;
  `binding_operations.py` implements tensor-product and related binding
  operations; `role_assigner.py` is the role-assignment module location.
- fMRI encoding baselines: `regression_encoding_model.py` maps aggregated BERT
  category-label embeddings to PCA-reduced fMRI targets;
  `regression_encoding_model_NSD.py` maps NSD-keyed BERT image embeddings to raw
  left/right hemisphere fMRI voxel responses with RidgeCV.
- Fixed-role TPDN experiments: `tpdn_test4role.py` is the present fixed-role
  TPDN implementation in this checkout. For toy number sequences it supports
  `l2r`, `r2l`, `bow`, and `bidi` role schemes and writes TPDN encodings for
  downstream ROLE approximation.
- Caption/syntax fixed-role lineage: the project also includes or is expected
  to include earlier fixed-role TPDN experiments where fillers are single words
  from image captions and roles are word positions or syntax-tree positions. In
  this checkout, the visible code for that lineage is mostly legacy evaluation
  scaffolding (`evaluation.py`) and missing imports such as `data_loader`,
  `role_assignment_functions`, and `rolelearner.*`; verify the relevant files or
  data are present before editing or running that path.
- Toy architecture tests: `gen_numbers_data.py`, `tpdn_test4role.py`, and
  `role_approx.py` form the controlled number-sequence workflow used to test
  TPDN encodings and ROLE recovery of fixed role schemes.

## Important Files

- `README.md`: project overview, experiment description, and reported results.
- `gen_numbers_data.py`: generates `random_sequences.csv` and
  `random_sequences.txt`.
- `tpdn_test4role.py`: trains the fixed-role TPDN on `random_sequences.txt` and
  writes `tpdn_encodings_test_<role_scheme>.json` plus model/output artifacts.
- `role_approx.py`: toy ROLE approximation experiment for number-sequence TPDN
  encodings. It expects `tpdn_encodings_test_l2r.json`.
- `role_learning_tensor_product_encoder.py`: ROLE encoder module. It embeds
  fillers, obtains learned roles, binds filler-role pairs, and optionally
  projects the resulting representation.
- `role_assigner.py`: currently contains `RoleAssignmentLSTM`.
- `binding_operations.py`: tensor product, circular convolution, and elementwise
  binding operations.
- `role_approx_fmri.py`: main learned-role fMRI experiment using category-label
  fillers and PCA-reduced fMRI targets.
- `regression_encoding_model.py`, `regression_encoding_model_NSD.py`:
  fMRI/NSD-related regression baselines that require external data files.
- `evaluation.py`: legacy evaluation helpers. It imports modules/packages that
  are not present in this checkout.

## Environment And Dependencies

There is currently no `requirements.txt`, `pyproject.toml`, or environment file.
The scripts use these Python packages:

- Core: `torch`, `numpy`, `matplotlib`, `tqdm`
- fMRI/regression scripts: `scikit-learn`, `scipy`, `pandas`

Use Python 3 with PyTorch installed. CUDA is used automatically when available;
otherwise scripts fall back to CPU, but full training may be slow.

## Data And Artifact Flow

For the number-sequence ROLE experiment, the expected workflow is:

1. Generate input sequences:
   `python gen_numbers_data.py`
2. Train/generate TPDN outputs:
   `python tpdn_test4role.py`
3. Train/evaluate ROLE approximation:
   `python role_approx.py`

Common generated files include:

- `random_sequences.csv`
- `random_sequences.txt`
- `best_tpdn_model.pt`
- `tpdn_encodings_test_l2r.json`
- `best_role_model_encodings.pt`
- `best_tpdn_decoder_encodings.pt`
- training plots such as `role_model_mse_encodings_additional_epochs.png`

Treat generated `.pt`, `.json`, `.csv`, and plot files as experiment artifacts.
Do not delete or overwrite existing artifacts unless the user explicitly asks or
the task clearly requires regenerating them.

The fMRI/NSD scripts expect local external data such as
`fmri_metadata_pca8.json`, `ventral_encodings_fmri_300.json`,
`bert_embedding_matrix.pt`, and `bert_embeddings_with_nsdIDs.json`. These files
are not present in the repository by default.

## Known Current Caveats

- `role_learning_tensor_product_encoder.py` imports `RoleAssignmentTransformer`
  from `role_assigner.py`, and `role_approx.py` imports it too. In this checkout,
  `role_assigner.py` defines `RoleAssignmentLSTM` but not
  `RoleAssignmentTransformer`. Verify or restore that class before expecting the
  current Transformer-based ROLE scripts to run end to end.
- `evaluation.py` imports `data_loader`, `role_assignment_functions`, and
  `rolelearner.role_learning_tensor_product_encoder`; these are legacy
  dependencies absent from this checkout.
- `binding_operations.py` has CUDA-specific code in the circular convolution
  helper (`.cuda()`). The default ROLE path uses `binder="tpr"`, which avoids
  that branch.

## Development Guidelines

- Keep edits close to the relevant experiment script or module. Avoid broad
  refactors unless the task requires them.
- Preserve tensor shape conventions. ROLE modules commonly use:
  - fillers: `(batch, sequence_length)`
  - role predictions: `(sequence_length, batch, num_roles)`
  - role embeddings: `(sequence_length, batch, role_dim)`
  - TPR outputs: `(batch, encoding_dim)` or a squeezed equivalent depending on
    the binding path
- Keep device handling explicit and compatible with both CUDA and CPU.
- Maintain reproducibility helpers such as `set_seed()` when touching training
  loops.
- Be careful with script-level constants. The project currently does not use a
  central config system or command-line parser.
- Prefer adding small, focused helpers over introducing new frameworks or package
  structure.
- Use ASCII in new code and docs unless a file already uses non-ASCII notation
  for mathematical text.

## Validation

Before finishing changes, run the lightest relevant checks:

- Syntax check all scripts:
  `python -m py_compile *.py`
- For data generation changes:
  `python gen_numbers_data.py`
- For TPDN changes, run `python tpdn_test4role.py` only when the runtime and
  artifact writes are acceptable.
- For ROLE training changes, run `python role_approx.py` only after confirming
  `tpdn_encodings_test_l2r.json` exists and `RoleAssignmentTransformer` is
  available.

Full training scripts can be slow and write checkpoints/plots. If you cannot run
them, state exactly which checks were run and which were skipped.
