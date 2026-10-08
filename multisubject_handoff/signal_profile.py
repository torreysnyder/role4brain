#!/usr/bin/env python
"""Where is a subject's repeatable signal, and how much noise does whitening add?

Model-independent check for choosing --pcs (see TARGETS_AND_WHITENING.md):

  * PCA on the subject's valid non-shared images (the vertex set comes from a
    build_nsd_role_inputs.py output directory, so it matches the targets).
  * Signal variance per component from the shared images: the three single-trial
    presentations are projected on the components, and the covariance between
    *different* presentations of the same image estimates the signal, because
    their noise is independent.
  * For each k: the share of total variance and of signal variance in the top k
    components, and the noise share of the training target under unwhitened
    and whitened (per-component z-scored) targets.

Only shared images seen three times are used for the signal estimate. Beyond
the point where the cumulative signal share reaches ~100%, the estimate is
noise and can overshoot.

    python signal_profile.py --inputs nsd_inputs/single_pca128 --subject 2
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

NUM_SHARED = 1000


def main(args: argparse.Namespace) -> None:
    basis_files = sorted(glob.glob(os.path.join(args.inputs, "basis_sub*.npz")))
    if not basis_files:
        raise SystemExit(f"no basis_sub*.npz in {args.inputs}")
    keep = np.load(basis_files[0])["vertices"]
    path = os.path.join(args.betas_dir, f"sub{args.subject:02d}_{args.stream}.npz")
    with np.load(path) as data:
        counts = data["image_count"]
        unique = np.where(counts[NUM_SHARED:] > 0)[0] + NUM_SHARED
        betas = data["betas"][unique][:, keep].astype(np.float64)
        reps = data["shared_reps"][:, :, keep].astype(np.float64)
        if "shared_rep_valid" in data.files:
            rep_valid = data["shared_rep_valid"]
        else:   # older files: presentations were filled in occurrence order
            rep_valid = np.arange(3)[:, None] < counts[None, :NUM_SHARED]

    mean = betas.mean(axis=0)
    _, singular, components = np.linalg.svd(betas - mean, full_matrices=False)
    variance = singular ** 2 / (len(betas) - 1)

    full = np.where(rep_valid.all(axis=0))[0]
    projected = [(reps[r, full] - reps[r, full].mean(axis=0)) @ components.T for r in range(3)]
    raw = [reps[r, full] - reps[r, full].mean(axis=0) for r in range(3)]
    pairs = [(0, 1), (0, 2), (1, 2)]
    signal = np.mean([(projected[a] * projected[b]).sum(axis=0) / (len(full) - 1)
                      for a, b in pairs], axis=0)
    signal_total = np.mean([(raw[a] * raw[b]).sum() / (len(full) - 1) for a, b in pairs])
    share = signal / variance

    print(f"sub{args.subject:02d}: {len(unique)} images for PCA, {len(full)} shared images "
          f"with three presentations, {len(keep)} vertices")
    print(f"{'k':>6} {'variance':>9} {'signal':>8} {'noise share':>12} {'noise share':>12}")
    print(f"{'':>6} {'cum.':>9} {'cum.':>8} {'unwhitened':>12} {'whitened':>12}")
    for k in args.ks:
        if k > len(variance):
            continue
        print(f"{k:>6} {variance[:k].sum() / variance.sum():9.3f} "
              f"{signal[:k].sum() / signal_total:8.3f} "
              f"{1 - signal[:k].sum() / variance[:k].sum():12.2f} "
              f"{1 - share[:k].mean():12.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True,
                        help="any build_nsd_role_inputs.py output dir (for the vertex set)")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--betas-dir",
                        default="/data/giacomo/natural-scenes-dataset/nsd_ventral_fsaverage")
    parser.add_argument("--stream", default="ventral")
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[3, 8, 16, 32, 64, 128, 256, 512])
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
