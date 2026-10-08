#!/usr/bin/env python
"""Score predictions in vertex space, so that runs with different targets compare.

R^2 on PCA targets is a different quantity for every number of components, and
for whitened versus unwhitened scores. Mapping predictions back to vertices and
scoring them against the subject's real betas puts every run on one scale:

    vertices = mean + (prediction * scale) @ components     (PCA targets)
    vertices = mean + prediction                            (--pcs all)

using the basis_sub<NN>.npz written by build_nsd_role_inputs.py.

Reported, over the chosen subject's real rows in one split:

  r2_total          1 - SSE / SST summed over vertices (variance-weighted).
  r2_mean_vertex    mean of the per-vertex R^2.
  noise_ceiling     variance-weighted ceiling from NSD's ncsnr and each image's
                    actual repeat count: the most any model can reach.
  subspace_ceiling  R^2 of the truth's own projection onto the target subspace:
                    the most any model trained on these targets can reach.
  fraction_of_ceiling  r2_total / noise_ceiling.

CLI, for predictions saved by the patched role_approx_fmri.py (an .npz with
`fmri_row` and `prediction`, the latter in target units):

    python vertex_eval.py --inputs nsd_inputs/single_pca128 \
        --predictions role_test_predictions.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


class VertexScorer:
    """Real vertex responses for one subject's rows in one split, plus the basis."""

    def __init__(self, inputs: str, subject: int | None = None, split: str = "test",
                 betas_file: str | None = None):
        with open(os.path.join(inputs, "build_summary.json")) as handle:
            summary = json.load(handle)
        with open(os.path.join(inputs, "fmri_metadata.json")) as handle:
            rows = json.load(handle)
        self.subject = summary["reference"] if subject is None else subject
        basis = np.load(os.path.join(inputs, f"basis_sub{self.subject:02d}.npz"))
        self.kind = str(basis["kind"])
        self.mean = basis["mean"].astype(np.float64)
        if self.kind == "pca":
            self.components = basis["components"].astype(np.float64)
            self.scale = basis["scale"].astype(np.float64)
        keep = basis["vertices"]

        selected = [r for r in rows if r["subject"] == self.subject
                    and not r["aligned"] and r["split"] == split]
        if not selected:
            raise ValueError(f"no real rows for subject {self.subject} in split {split!r}")
        self.fmri_rows = np.array([r["fmri_row"] for r in selected])
        nsd_index = np.array([r["nsd_index"] for r in selected])

        with np.load(betas_file or str(basis["betas_file"])) as data:
            self.truth = data["betas"][nsd_index][:, keep].astype(np.float64)
            self.ncsnr = data["ncsnr"][keep].astype(np.float64)
            self.repeats = data["image_count"][nsd_index].astype(np.float64)
        self._prepare()

    def restrict(self, mask: np.ndarray) -> None:
        """Keep only some rows (e.g. those a run actually predicted) and rescore."""
        self.fmri_rows, self.truth, self.repeats = (
            self.fmri_rows[mask], self.truth[mask], self.repeats[mask])
        self._prepare()

    def _prepare(self) -> None:
        centered = self.truth - self.truth.mean(axis=0)
        self.sst_vertex = (centered ** 2).sum(axis=0)
        ceiling_vertex = self.ncsnr ** 2 / (self.ncsnr ** 2 + np.mean(1.0 / self.repeats))
        self.noise_ceiling = float((ceiling_vertex * self.sst_vertex).sum() / self.sst_vertex.sum())
        if self.kind == "pca":
            projected = self.mean + ((self.truth - self.mean) @ self.components.T) @ self.components
            self.subspace_ceiling = self._r2_total(projected)
        else:
            self.subspace_ceiling = 1.0

    def to_vertices(self, prediction: np.ndarray) -> np.ndarray:
        if self.kind == "pca":
            return self.mean + (prediction * self.scale) @ self.components
        return self.mean + prediction

    def _r2_total(self, vertices: np.ndarray) -> float:
        return float(1.0 - ((self.truth - vertices) ** 2).sum() / self.sst_vertex.sum())

    def score(self, prediction: np.ndarray) -> dict[str, float]:
        """`prediction` holds one row per entry of self.fmri_rows, in target units."""
        vertices = self.to_vertices(prediction)
        per_vertex = 1.0 - ((self.truth - vertices) ** 2).sum(axis=0) / self.sst_vertex.clip(1e-12)
        r2 = self._r2_total(vertices)
        return {
            "r2_total": r2,
            "r2_mean_vertex": float(per_vertex.mean()),
            "noise_ceiling": self.noise_ceiling,
            "subspace_ceiling": self.subspace_ceiling,
            "fraction_of_ceiling": r2 / self.noise_ceiling,
            "num_images": int(len(self.fmri_rows)),
        }


def main(args: argparse.Namespace) -> None:
    scorer = VertexScorer(args.inputs, args.subject, args.split, args.betas_file)
    saved = np.load(args.predictions)
    position = {int(row): i for i, row in enumerate(saved["fmri_row"])}
    missing = [int(r) for r in scorer.fmri_rows if int(r) not in position]
    if missing:
        print(f"NOTE: {len(missing)} of {len(scorer.fmri_rows)} {args.split} rows have no "
              f"prediction (e.g. dropped by MAX_SEQ_LEN); scoring the rest")
        scorer.restrict(np.array([int(r) in position for r in scorer.fmri_rows]))
    prediction = saved["prediction"][[position[int(r)] for r in scorer.fmri_rows]]
    result = scorer.score(prediction)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="a build_nsd_role_inputs.py output dir")
    parser.add_argument("--predictions", required=True,
                        help=".npz with fmri_row (n,) and prediction (n, target_dim)")
    parser.add_argument("--subject", type=int, default=None,
                        help="default: the inputs' reference subject")
    parser.add_argument("--split", default="test")
    parser.add_argument("--betas-file", default=None,
                        help="override the betas path recorded in the basis file")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
