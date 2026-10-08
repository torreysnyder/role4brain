#!/usr/bin/env python
"""Closed-form bag-of-fillers baseline: the floor a ROLE model has to clear.

Ridge regression from the sum of a scene's frozen BERT filler embeddings to the
targets. This is what ROLE reduces to with a single role and a linear output
head: no role structure, no interactions. It answers three questions in
seconds, before any ROLE run:

  learning curve   Vertex-space test R^2 when training on 12.5% ... 100% of the
                   reference subject's own training rows. A flat curve near
                   100% means this model is limited by its features, not by
                   data, and pooling cannot help it.

  pooled           Own rows + every aligned row (aligned-mode inputs only).

  aligned only     Aligned rows alone. The gap to own data at a similar n is the
                   alignment bias as this model sees it.

Alpha is chosen on the reference's real `val` rows; every number is scored once
on its real `test` rows, in vertex space via vertex_eval.py, so runs with
different --pcs settings compare directly. `r` (mean correlation over target
dimensions) and the prediction-to-target SD ratio separate lost structure from
lost scale: ridge alignment shrinks its targets.

    python additive_baseline.py --inputs nsd_inputs/aligned_pca128
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vertex_eval import VertexScorer  # noqa: E402

HANDOFF_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def total_r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(1.0 - ((truth - prediction) ** 2).sum()
                 / ((truth - truth.mean(axis=0)) ** 2).sum())


def mean_component_r(prediction: np.ndarray, truth: np.ndarray) -> float:
    p = prediction - prediction.mean(axis=0)
    t = truth - truth.mean(axis=0)
    r = (p * t).sum(axis=0) / np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0)).clip(1e-12)
    return float(r.mean())


def fit_ridge(features, targets, alphas):
    """One predictor per alpha, sharing a single Gram matrix."""
    centre, offset = features.mean(axis=0), targets.mean(axis=0)
    x = features - centre
    gram, cross = x.T @ x, x.T @ (targets - offset)
    eye = np.eye(x.shape[1])
    predictors = {}
    for alpha in alphas:
        weights = np.linalg.solve(gram + alpha * eye, cross)
        predictors[alpha] = lambda f, w=weights: (f - centre) @ w + offset
    return predictors


def evaluate(features, targets, train, val, scorer, alphas):
    """Alpha by validation R^2 in target space, then one vertex-space test score."""
    predictors = fit_ridge(features[train], targets[train], alphas)
    scores = {a: total_r2(p(features[val]), targets[val]) for a, p in predictors.items()}
    alpha = max(scores, key=scores.get)
    if alpha in (min(alphas), max(alphas)):
        print(f"  NOTE: alpha {alpha:g} at the edge of the grid (n={len(train)}); "
              f"harmless if the validation score is flat there")
    test = scorer.fmri_rows
    prediction = predictors[alpha](features[test])
    result = scorer.score(prediction)
    result.update({
        "num_train": int(len(train)), "alpha": float(alpha),
        "target_r": mean_component_r(prediction, targets[test]),
        "sd_ratio": float(prediction.std(axis=0).sum() / targets[test].std(axis=0).sum()),
    })
    return result


def describe(name, result):
    print(f"  {name:<24} n={result['num_train']:<6} vertex R2 {result['r2_total']:+.4f} "
          f"({result['fraction_of_ceiling']:.0%} of ceiling)  r {result['target_r']:+.4f}  "
          f"sd ratio {result['sd_ratio']:.2f}  alpha {result['alpha']:g}")


def main(args: argparse.Namespace) -> None:
    with open(os.path.join(args.inputs, "fmri_metadata.json")) as handle:
        rows = json.load(handle)
    targets = np.load(os.path.join(args.inputs, "fmri_targets.npy")).astype(np.float64)
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    scorer = VertexScorer(args.inputs, split="test")
    reference = scorer.subject

    # Summed filler embeddings, accumulated per position: indexing all positions
    # at once would materialize a (rows, sequence, 768) array of several GB.
    filler_ids = np.array([row["filler_ids"] for row in rows])
    features = np.zeros((len(rows), embeddings.shape[1]))
    for position in range(filler_ids.shape[1]):
        ids = filler_ids[:, position]
        features += embeddings[ids] * (ids != 0)[:, None]

    def select(predicate):
        return np.array([row["fmri_row"] for row in rows if predicate(row)])

    own = lambda row: row["subject"] == reference and not row["aligned"]
    own_train = select(lambda r: own(r) and r["split"] == "train")
    aligned_train = select(lambda r: r["aligned"] and r["split"] == "train")
    val = select(lambda r: own(r) and r["split"] == "val")
    print(f"Reference sub{reference:02d}: {len(own_train)} own training rows, "
          f"{len(aligned_train)} aligned rows, {len(val)} val, {len(scorer.fmri_rows)} test; "
          f"noise ceiling {scorer.noise_ceiling:.3f}, "
          f"subspace ceiling {scorer.subspace_ceiling:.3f}")

    results: dict[str, object] = {"reference": reference, "learning_curve": []}
    rng = np.random.default_rng(args.seed)
    print("\nOwn-data learning curve (vertex-space test R^2; mean over draws below 100%):")
    for fraction in args.fractions:
        draws = 1 if fraction == 1.0 else args.draws
        runs = [evaluate(features, targets,
                         rng.choice(own_train, size=int(round(fraction * len(own_train))),
                                    replace=False),
                         val, scorer, args.alphas) for _ in range(draws)]
        entry = {key: float(np.mean([run[key] for run in runs]))
                 for key in ("r2_total", "fraction_of_ceiling", "target_r", "sd_ratio", "alpha")}
        entry.update({"fraction": fraction, "num_train": runs[0]["num_train"],
                      "r2_total_sd": float(np.std([run["r2_total"] for run in runs]))})
        results["learning_curve"].append(entry)
        describe(f"own {fraction:.3f}", entry)

    if len(aligned_train):
        print()
        results["pooled"] = evaluate(features, targets,
                                     np.concatenate([own_train, aligned_train]),
                                     val, scorer, args.alphas)
        describe("own + aligned", results["pooled"])
        results["aligned_only"] = evaluate(features, targets, aligned_train,
                                           val, scorer, args.alphas)
        describe("aligned only", results["aligned_only"])

    path = os.path.join(args.inputs, "additive_baseline.json")
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="a build_nsd_role_inputs.py output dir")
    parser.add_argument("--embeddings",
                        default=os.path.join(HANDOFF_DATA, "bert_embedding_matrix.pt"))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--draws", type=int, default=3,
                        help="random subsets per fraction below 1")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
