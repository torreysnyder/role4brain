#!/usr/bin/env python
"""Build ROLE inputs from real NSD betas for single-subject, aligned and joint runs.

Three modes, one set of conventions, so that every approach is scored on the
same held-out images in the same units:

  single   The reference subject's own responses only. This is the baseline
           every pooled run is compared against.

  aligned  The single-mode rows, plus the training images of every source
           subject mapped into the reference subject's target space by a ridge
           regression fit on the shared images (Ferrante et al. 2024, adapted
           to regress onto the reference's targets). Aligned rows are only
           ever used for training; validation and test stay the reference's
           real responses.

  joint    Every subject's own responses, each in that subject's own basis,
           tagged with `subject`. For a shared ROLE model with per-subject
           output heads; role_approx_fmri.py cannot train on this file as is.

Targets (--pcs, --whiten):

  --pcs K      top-K PCA scores, fit on the subject's training rows. By default
               they are NOT whitened: scores stay in vertex units, so the MSE on
               them equals the vertex-space MSE inside the K-dim subspace and
               each component weighs in proportion to its variance. See
               TARGETS_AND_WHITENING.md for why this matters.
  --pcs all    every vertex, centered on the training mean. Equivalent to
               unwhitened PCA with all components, without the rotation.
  --whiten     divide each PC by its training SD (the legacy convention).

Each subject with its own rows gets basis_sub<NN>.npz (mean, components, scale)
so that predictions can be mapped back to vertices by vertex_eval.py.

Invariants that make the modes comparable (fixed by --scope, --vertices,
--shared-test and --seed, never by --mode):

  * Vertex set: the top --vertices ventral vertices by ncsnr averaged over the
    scope subjects. fsaverage gives every subject the same vertex indices.
  * Split by image, not by row. The shared images that every scope subject saw
    are split once into `shared_test` (held out from everything, including the
    alignment fit) and `train`; a given shared image has the same label in
    every subject, so no image can leak across subjects. Each subject's unique
    images get a seeded 70/15/15 train/val/test split.
  * Reference targets: fit on the reference subject's training rows. The
    reference rows are identical in all three modes for the same --pcs.

Metadata records keep the legacy loader's contract (`filler_ids`, `fmri_row`,
`image_id` = COCO id, `split`) and add `subject`, `source_subject`, `aligned`,
`shared`, `nsd_id`, `nsd_index` and `coco_split`.

Alignment: for each source subject, ridge from the source's vertices to the
reference's targets, fit on the shared images valid for both (excluding
shared_test). Fitting onto the targets is exactly equivalent to fitting
vertex->vertex and projecting afterwards, because ridge is linear in its
targets. Alpha is chosen by k-fold variance-weighted R^2 inside the fitting
images; R^2 rather than correlation, because the aligned values become training
targets and their scale matters.

Prerequisite: the per-subject files written by nsd_real_betas.py
(`sub<NN>_<stream>.npz` with betas, image_count, nsd_ids, ncsnr). Missing
images are exact zeros in `betas`, so this script only ever indexes through
`image_count > 0`.

Examples (role4brain conda environment, inside a Slurm job):
    python build_nsd_role_inputs.py --mode single  --pcs 128 --out-dir nsd_inputs/single_pca128
    python build_nsd_role_inputs.py --mode aligned --pcs 128 --out-dir nsd_inputs/aligned_pca128
    python build_nsd_role_inputs.py --mode joint   --pcs 128 --out-dir nsd_inputs/joint_pca128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

NSD_ROOT = "/data/giacomo/natural-scenes-dataset"
DEFAULT_BETAS_DIR = os.path.join(NSD_ROOT, "nsd_ventral_fsaverage")
DEFAULT_STIM_INFO = os.path.join(
    NSD_ROOT, "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
)
DEFAULT_COCO_ANNOTATIONS = os.path.join(NSD_ROOT, "coco_2017/annotations")
# Filler vocabulary and BERT matrix shipped with the package (a matched pair).
HANDOFF_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NUM_SHARED = 1000
ALL_SUBJECTS = list(range(1, 9))
OWN_SPLITS = ("train", "val", "test", "shared_test")


# --------------------------------------------------------------------------
# subject files


def subject_path(args: argparse.Namespace, subject: int) -> str:
    return os.path.join(args.betas_dir, f"sub{subject:02d}_{args.stream}.npz")


def read_light(path: str) -> dict[str, np.ndarray]:
    """Everything except the betas. np.savez members load lazily, so this is cheap."""
    with np.load(path) as data:
        return {
            "ncsnr": data["ncsnr"].astype(np.float64),
            "valid": data["image_count"] > 0,
            "nsd_ids": data["nsd_ids"].astype(np.int64),
        }


def read_betas(path: str, keep: np.ndarray) -> np.ndarray:
    with np.load(path) as data:
        return data["betas"][:, keep].astype(np.float64)


# --------------------------------------------------------------------------
# splits


def choose_shared_test(
    light: dict[int, dict[str, np.ndarray]], scope: list[int], size: int, seed: int
) -> tuple[np.ndarray, int]:
    """Held-out shared images, drawn from those every scope subject saw."""
    common = np.ones(NUM_SHARED, dtype=bool)
    for subject in scope:
        common &= light[subject]["valid"][:NUM_SHARED]
    candidates = np.where(common)[0]
    if size > len(candidates):
        raise SystemExit(f"--shared-test {size} exceeds the {len(candidates)} "
                         f"shared images valid for every scope subject")
    chosen = np.random.default_rng(seed).choice(candidates, size=size, replace=False)
    return np.sort(chosen), len(candidates)


def split_labels(
    valid: np.ndarray, shared_test: np.ndarray, subject: int, seed: int
) -> np.ndarray:
    """Per-row split label; '' marks images the subject never saw."""
    labels = np.full(len(valid), "", dtype=object)
    shared = np.arange(NUM_SHARED)
    labels[shared[valid[:NUM_SHARED]]] = "train"
    labels[shared_test] = "shared_test"   # valid by construction

    unique = np.where(valid[NUM_SHARED:])[0] + NUM_SHARED
    order = np.random.default_rng(seed + 1000 * subject).permutation(unique)
    num_train = int(0.70 * len(order))
    num_val = int(0.15 * len(order))
    labels[order[:num_train]] = "train"
    labels[order[num_train:num_train + num_val]] = "val"
    labels[order[num_train + num_val:]] = "test"
    return labels


# --------------------------------------------------------------------------
# targets and alignment


def fit_targets(
    betas: np.ndarray, labels: np.ndarray, pcs: int | None, whiten: bool, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Targets for every seen row, plus the basis that maps them back to vertices.

    vertices = mean + (targets * scale) @ components, with components the
    identity when pcs is None (all vertices).
    """
    train = labels == "train"
    seen = labels != ""
    mean = betas[train].mean(axis=0)
    targets = np.zeros((len(betas), pcs or betas.shape[1]))
    if pcs is None:
        targets[seen] = betas[seen] - mean
        basis = {"kind": "vertices", "explained_variance_ratio": 1.0}
        return targets, {"mean": mean, **basis}

    from sklearn.decomposition import PCA

    pca = PCA(n_components=pcs, random_state=seed).fit(betas[train])
    targets[seen] = (betas[seen] - mean) @ pca.components_.T
    scale = targets[train].std(axis=0).clip(1e-8) if whiten else np.ones(pcs)
    targets[seen] /= scale
    return targets, {
        "kind": "pca", "mean": mean, "components": pca.components_.astype(np.float32),
        "scale": scale, "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def ridge_predict(
    source: np.ndarray, target: np.ndarray, evaluation: np.ndarray, alphas: list[float]
) -> dict[float, np.ndarray]:
    """Dual-form ridge for every alpha from one eigendecomposition.

    Inputs must already be centered on the fitting rows. With ~800 fitting
    images and ~7600 vertices the (n, n) Gram matrix is the cheap side.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(source @ source.T)
    projected = eigenvectors.T @ target
    cross = evaluation @ source.T @ eigenvectors
    return {
        alpha: cross @ (projected / (eigenvalues + alpha)[:, None]) for alpha in alphas
    }


def total_r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Variance-weighted R^2: 1 - total SSE / total SS around the column means."""
    residual = ((truth - prediction) ** 2).sum()
    return float(1.0 - residual / ((truth - truth.mean(axis=0)) ** 2).sum())


def mean_component_r(prediction: np.ndarray, truth: np.ndarray) -> float:
    p = prediction - prediction.mean(axis=0)
    t = truth - truth.mean(axis=0)
    r = (p * t).sum(axis=0) / np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0)).clip(1e-12)
    return float(r.mean())


def fit_alignment(
    source: np.ndarray,
    target: np.ndarray,
    apply_to: np.ndarray,
    alphas: list[float],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose alpha by k-fold R^2 on the fitting images, refit, map `apply_to`."""
    order = np.random.default_rng(seed).permutation(len(source))
    predictions = {alpha: np.zeros_like(target) for alpha in alphas}
    for fold in range(folds):
        held = order[fold::folds]
        kept = np.setdiff1d(order, held)
        source_mean = source[kept].mean(axis=0)
        target_mean = target[kept].mean(axis=0)
        fold_predictions = ridge_predict(
            source[kept] - source_mean, target[kept] - target_mean,
            source[held] - source_mean, alphas,
        )
        for alpha, prediction in fold_predictions.items():
            predictions[alpha][held] = prediction + target_mean
    scores = {alpha: total_r2(p, target) for alpha, p in predictions.items()}
    best = max(scores, key=scores.get)
    held_out = predictions[best]
    if best in (min(alphas), max(alphas)):
        print(f"    WARNING: alpha {best:g} is at the edge of the grid")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    mapped = ridge_predict(
        source - source_mean, target - target_mean, apply_to - source_mean, [best]
    )[best] + target_mean
    diagnostics = {
        "num_fit_images": int(len(source)),
        "alpha": float(best),
        "cv_r2_by_alpha": {f"{a:g}": float(s) for a, s in scores.items()},
        "cv_target_r2": float(scores[best]),
        "cv_target_r": mean_component_r(held_out, target),
        # Ridge shrinks: aligned targets have less variance than real ones.
        "cv_variance_ratio": float(held_out.var(axis=0).sum() / target.var(axis=0).sum()),
    }
    return mapped, diagnostics


# --------------------------------------------------------------------------
# fillers


def load_filler_lookup(args: argparse.Namespace):
    """nsdId (1-based) -> (COCO id, COCO split), and COCO id -> category areas."""
    coco_by_nsd: dict[int, tuple[int, str]] = {}
    with open(args.stim_info, newline="") as handle:
        for row in csv.DictReader(handle):
            # The CSV's nsdId is 0-based; subjectim (and nsd_ids) is 1-based.
            coco_by_nsd[int(row["nsdId"]) + 1] = (int(row["cocoId"]), row["cocoSplit"])

    areas: dict[int, dict[str, float]] = {}
    for split in ("train2017", "val2017"):
        path = os.path.join(args.coco_annotations, f"instances_{split}.json")
        with open(path) as handle:
            payload = json.load(handle)
        names = {c["id"]: c["name"] for c in payload["categories"]}
        sizes = {i["id"]: float(i["width"] * i["height"]) for i in payload["images"]}
        for annotation in payload["annotations"]:
            image_id = annotation["image_id"]
            name = names[annotation["category_id"]]
            per_image = areas.setdefault(image_id, {})
            per_image[name] = per_image.get(name, 0.0) + annotation["area"] / sizes[image_id]
    return coco_by_nsd, areas


def filler_ids_for(nsd_id: int, coco_by_nsd, areas, vocab) -> list[int]:
    """Same rule as build_role_inputs.py: unique categories, largest area first."""
    per_image = areas.get(coco_by_nsd[nsd_id][0], {})
    ordered = sorted(per_image, key=lambda name: -per_image[name])
    return [vocab[name] for name in ordered if name in vocab]


# --------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    started = time.time()
    pcs = None if args.pcs == "all" else int(args.pcs)
    if pcs is None and args.whiten:
        raise SystemExit("--whiten applies to PCA targets only")
    reference = args.reference
    scope = sorted(set(args.scope))
    if reference not in scope:
        raise SystemExit("--reference must be in --scope")
    others = [s for s in args.subjects if s != reference]
    if not set(others) <= set(scope):
        raise SystemExit("every --subjects entry must be in --scope")
    row_subjects = [reference] if args.mode == "single" else [reference] + others
    if args.mode != "single" and not others:
        raise SystemExit(f"--mode {args.mode} needs at least one non-reference subject")
    if args.train_fraction < 1.0 and args.mode != "single":
        print("NOTE: --train-fraction thins only the reference subject's own rows")

    light = {s: read_light(subject_path(args, s)) for s in scope}
    mean_ncsnr = np.mean([light[s]["ncsnr"] for s in scope], axis=0)
    num_vertices = args.vertices if args.vertices else len(mean_ncsnr)
    keep = np.sort(np.argsort(mean_ncsnr)[::-1][:num_vertices])
    print(f"Vertices: {len(keep)} of {len(mean_ncsnr)} by mean ncsnr over {scope}")
    if pcs is not None and pcs > len(keep):
        raise SystemExit(f"--pcs {pcs} exceeds the {len(keep)} vertices")

    shared_test, num_common = choose_shared_test(light, scope, args.shared_test, args.seed)
    print(f"Shared images valid for all scope subjects: {num_common}; "
          f"{len(shared_test)} held out as shared_test")
    labels = {s: split_labels(light[s]["valid"], shared_test, s, args.seed)
              for s in row_subjects}

    os.makedirs(args.out_dir, exist_ok=True)
    rows: list[dict[str, object]] = []
    blocks: list[np.ndarray] = []
    summary: dict[str, object] = {
        "mode": args.mode, "reference": reference, "subjects": row_subjects,
        "scope": scope, "pcs": args.pcs, "whiten": args.whiten, "stream": args.stream,
        "num_vertices": int(len(keep)), "seed": args.seed,
        "train_fraction": args.train_fraction,
        "num_common_shared": int(num_common),
        "shared_test_nsd_ids": [int(light[reference]["nsd_ids"][i]) for i in shared_test],
        "explained_variance_ratio": {}, "alignment": {}, "rows": {},
    }

    def save_basis(subject, basis):
        summary["explained_variance_ratio"][str(subject)] = basis["explained_variance_ratio"]
        np.savez(os.path.join(args.out_dir, f"basis_sub{subject:02d}.npz"),
                 vertices=keep, betas_file=subject_path(args, subject), **basis)

    def add_rows(subject, source, indices, values, aligned, split_of):
        nsd_ids = light[source]["nsd_ids"]
        for index in indices:
            rows.append({
                "subject": subject, "source_subject": source,
                "aligned": aligned, "split": split_of(index),
                "shared": bool(index < NUM_SHARED),
                "nsd_index": int(index), "nsd_id": int(nsd_ids[index]),
            })
        blocks.append(values.astype(np.float32))

    # --- reference subject: own rows -------------------------------------
    print(f"Loading sub{reference:02d} (reference) ...")
    betas_ref = read_betas(subject_path(args, reference), keep)
    targets_ref, basis = fit_targets(betas_ref, labels[reference], pcs, args.whiten, args.seed)
    save_basis(reference, basis)
    print(f"  targets: {targets_ref.shape[1]} dims ({basis['kind']}), "
          f"{basis['explained_variance_ratio']:.1%} of vertex variance")
    del betas_ref

    ref_labels = labels[reference].copy()
    if args.train_fraction < 1.0:
        # Thin after the basis fit, so the basis and units do not move with the
        # fraction and runs along the learning curve stay directly comparable.
        train = np.where(ref_labels == "train")[0]
        kept = np.random.default_rng(args.seed + 7).choice(
            train, size=int(round(args.train_fraction * len(train))), replace=False
        )
        ref_labels[np.setdiff1d(train, kept)] = ""
    own = np.where(np.isin(ref_labels, OWN_SPLITS))[0]
    add_rows(reference, reference, own, targets_ref[own], False, lambda i: ref_labels[i])

    # --- other subjects -----------------------------------------------------
    for subject in row_subjects[1:]:
        print(f"Loading sub{subject:02d} ...")
        betas = read_betas(subject_path(args, subject), keep)
        subject_labels = labels[subject]
        if args.mode == "aligned":
            # Fit on shared images both subjects saw, never on shared_test.
            fit = np.where(
                (labels[reference][:NUM_SHARED] == "train")
                & (subject_labels[:NUM_SHARED] == "train")
            )[0]
            apply_to = np.where(subject_labels == "train")[0]
            apply_to = apply_to[apply_to >= NUM_SHARED]   # reference has the shared ones
            mapped, diagnostics = fit_alignment(
                betas[fit], targets_ref[fit], betas[apply_to],
                args.alphas, args.alignment_folds, args.seed,
            )
            diagnostics["num_rows"] = int(len(apply_to))
            summary["alignment"][str(subject)] = diagnostics
            print(f"  aligned on {len(fit)} shared images: alpha {diagnostics['alpha']:g}  "
                  f"CV R2 {diagnostics['cv_target_r2']:+.3f}  "
                  f"r {diagnostics['cv_target_r']:+.3f}  "
                  f"var ratio {diagnostics['cv_variance_ratio']:.2f}  "
                  f"-> {len(apply_to)} training rows")
            add_rows(reference, subject, apply_to, mapped, True, lambda i: "train")
        else:  # joint
            targets, basis = fit_targets(betas, subject_labels, pcs, args.whiten, args.seed)
            save_basis(subject, basis)
            own = np.where(np.isin(subject_labels, OWN_SPLITS))[0]
            print(f"  targets: {basis['explained_variance_ratio']:.1%} of vertex variance, "
                  f"{len(own)} rows")
            add_rows(subject, subject, own, targets[own], False,
                     lambda i, labels_=subject_labels: labels_[i])
        del betas

    # --- fillers -------------------------------------------------------------
    print("Building filler sequences from COCO annotations ...")
    with open(args.vocab) as handle:
        vocab = json.load(handle)
    coco_by_nsd, areas = load_filler_lookup(args)
    # Fix the length over every image the scope subjects saw, not over this
    # mode's rows, so the reference rows are identical across modes.
    scope_ids = {int(i) for s in scope for i in light[s]["nsd_ids"][light[s]["valid"]]}
    max_fillers = args.max_fillers or max(
        len(filler_ids_for(i, coco_by_nsd, areas, vocab)) for i in scope_ids
    )
    empty = truncated = 0
    for index, row in enumerate(rows):
        ids = filler_ids_for(row["nsd_id"], coco_by_nsd, areas, vocab)
        empty += not ids
        truncated += max(0, len(ids) - max_fillers)
        ids = ids[:max_fillers]
        row["filler_ids"] = ids + [0] * (max_fillers - len(ids))
        row["image_id"], row["coco_split"] = coco_by_nsd[row["nsd_id"]]
        row["fmri_row"] = index
    targets = np.concatenate(blocks, axis=0)

    # --- invariants ----------------------------------------------------------
    split_of_image: dict[int, str] = {}
    for row in rows:
        if row["aligned"]:
            continue
        previous = split_of_image.setdefault(row["nsd_id"], row["split"])
        if previous != row["split"]:
            raise RuntimeError(f"image {row['nsd_id']} is in both {previous} and {row['split']}")
    held_out = {r["nsd_id"] for r in rows if r["split"] != "train"}
    if any(r["aligned"] and r["nsd_id"] in held_out for r in rows):
        raise RuntimeError("an aligned training row reuses a held-out image")

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"sub{row['source_subject']:02d}" + ("_aligned" if row["aligned"] else "")
        counts.setdefault(key, {}).setdefault(row["split"], 0)
        counts[key][row["split"]] += 1
    summary["rows"] = counts
    summary.update({
        "num_rows": len(rows), "target_dim": int(targets.shape[1]),
        "max_fillers": int(max_fillers), "truncated_fillers": int(truncated),
        "rows_without_fillers": int(empty), "seconds": round(time.time() - started, 1),
    })

    np.save(os.path.join(args.out_dir, "fmri_targets.npy"), targets)
    with open(os.path.join(args.out_dir, "fmri_metadata.json"), "w") as handle:
        json.dump(rows, handle)
    with open(os.path.join(args.out_dir, "build_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nRows per source and split: {json.dumps(counts, indent=1)}")
    print(f"Targets {targets.shape}; sequence length {max_fillers} "
          f"({truncated} fillers truncated, {empty} rows without any COCO category)")
    print(f"Wrote {args.out_dir} in {summary['seconds']}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("single", "aligned", "joint"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pcs", default="128",
                        help="number of PCA components, or 'all' for every vertex")
    parser.add_argument("--whiten", action="store_true",
                        help="divide each PC by its training SD (legacy; not recommended)")
    parser.add_argument("--reference", type=int, default=1)
    parser.add_argument("--subjects", type=int, nargs="+", default=ALL_SUBJECTS,
                        help="subjects contributing rows (aligned/joint); default all 8")
    parser.add_argument("--scope", type=int, nargs="+", default=ALL_SUBJECTS,
                        help="subjects that fix the vertex set and shared_test images; "
                             "keep identical across runs that will be compared")
    parser.add_argument("--vertices", type=int, default=7604,
                        help="top-N by mean ncsnr over --scope; 0 keeps the whole ROI")
    parser.add_argument("--shared-test", type=int, default=200,
                        help="shared images held out from training and alignment")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="keep this fraction of the reference's training rows "
                             "(learning curves; intended for --mode single)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    parser.add_argument("--alignment-folds", type=int, default=5)
    parser.add_argument("--max-fillers", type=int, default=None,
                        help="sequence length; default is the observed maximum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream", default="ventral")
    parser.add_argument("--betas-dir", default=DEFAULT_BETAS_DIR)
    parser.add_argument("--stim-info", default=DEFAULT_STIM_INFO)
    parser.add_argument("--coco-annotations", default=DEFAULT_COCO_ANNOTATIONS)
    parser.add_argument("--vocab", default=os.path.join(HANDOFF_DATA, "bert_filler_vocab.json"),
                        help="must be the vocabulary the embedding matrix was built with")
    args = parser.parse_args()
    if args.pcs != "all" and not args.pcs.isdigit():
        parser.error("--pcs must be a positive integer or 'all'")
    if not 0.0 < args.train_fraction <= 1.0:
        parser.error("--train-fraction must be in (0, 1]")
    return args


if __name__ == "__main__":
    main(parse_args())
