#!/usr/bin/env python
"""Prepare real NSD fsaverage betas for the ventral stream, one file per subject.

Turns 80 per-session surface files into a single (10000, n_vertices) array in
which row i is the trial-averaged response to `subjectim[subject][i]`, the
subject's i-th NSD image. Rows 0-999 are the shared 1000 that every subject
saw; this is verified against `sharedix` rather than assumed.

Three preprocessing choices, all standard for NSD and all recorded in the
output so a later analysis cannot silently disagree about them:

  session z-scoring   Each vertex is z-scored within each scan session before
                      anything is averaged. Session is the unit across which
                      scanner drift and head position change, so this is what
                      makes betas from session 3 and session 37 comparable, and
                      it is what BERG's synthetic outputs already emulate.

  repeat averaging    Every image was shown exactly 3 times (checked). Averaging
                      the repeats raises the effective SNR by about sqrt(3) and
                      is what makes a 1000-image alignment set usable at all.

  shared vertex set   The ROI is the `streams` atlas ventral label on fsaverage,
                      so both hemispheres of both subjects index the *same*
                      vertices. That is what lets an identity map serve as the
                      anatomical-alignment baseline later.

Run in the `role4brain` conda environment:
    python nsd_real_betas.py --subject 1
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

NSD_ROOT = "/data/giacomo/natural-scenes-dataset"
BETAS_ROOT = os.path.join(NSD_ROOT, "nsd_fsaverage_betas")
LABEL_ROOT = os.path.join(NSD_ROOT, "nsddata/freesurfer/fsaverage/label")
EXPDESIGN = os.path.join(NSD_ROOT, "nsddata/experiments/nsd/nsd_expdesign.mat")
DEFAULT_OUT = os.path.join(NSD_ROOT, "nsd_ventral_fsaverage")

TRIALS_PER_SESSION = 750
NUM_SHARED = 1000
# streams.mgz.ctab: 0 Unknown, 1 early, 2 midventral, 3 midlateral,
# 4 midparietal, 5 ventral, 6 lateral, 7 parietal
STREAM_LABELS = {"early": 1, "midventral": 2, "midlateral": 3,
                 "midparietal": 4, "ventral": 5, "lateral": 6, "parietal": 7}


@dataclass
class SubjectBetas:
    """One subject's prepared ventral betas, with the gaps made explicit.

    `betas` rows for images the subject never saw are exactly zero, which is a
    legal response value, so the only safe way to use this array is through
    `valid`. Subjects 3, 4, 6 and 8 have 70-93 of the shared 1000 missing
    outright and roughly half the rest on fewer than three presentations.
    """

    subject: int
    betas: np.ndarray            # (n_images, n_vertices); invalid rows are zero
    ncsnr: np.ndarray            # (n_vertices,)
    nsd_ids: np.ndarray          # (n_images,) 1-based nsdId
    image_count: np.ndarray      # (n_images,) presentations actually recorded
    shared_reps: np.ndarray      # (3, NUM_SHARED, n_vertices)
    shared_rep_valid: np.ndarray # (3, NUM_SHARED) bool
    num_sessions: int

    @property
    def valid(self) -> np.ndarray:
        """Images with at least one recorded presentation."""
        return self.image_count > 0

    def shared_valid(self) -> np.ndarray:
        return self.valid[:NUM_SHARED]


def load_subject(subject: int, stream: str = "ventral",
                 directory: str = DEFAULT_OUT) -> SubjectBetas:
    """Load a prepared subject, reconstructing the repeat mask when absent.

    Files written before the mask existed are still usable: presentations are
    filled in occurrence order, so cell (r, i) was filled exactly when
    r < image_count[i]. The saved array is preferred when present so that this
    derivation is not load-bearing if the fill order ever changes.
    """
    path = os.path.join(directory, f"sub{subject:02d}_{stream}.npz")
    data = np.load(path)
    counts = data["image_count"]
    if "shared_rep_valid" in data.files:
        valid = data["shared_rep_valid"]
    else:
        valid = np.arange(3)[:, None] < counts[None, :NUM_SHARED]
        print(f"  sub{subject:02d}: no stored repeat mask; derived from "
              f"image_count ({int(valid.sum())} of {3 * NUM_SHARED} cells)")
    return SubjectBetas(
        subject=subject,
        betas=data["betas"],
        ncsnr=data["ncsnr"],
        nsd_ids=data["nsd_ids"],
        image_count=counts,
        shared_reps=data["shared_reps"],
        shared_rep_valid=valid,
        num_sessions=int(data["num_sessions"]),
    )


def shared_validity(subject: int, stream: str = "ventral",
                    directory: str = DEFAULT_OUT) -> np.ndarray:
    """Shared-image validity for one subject without loading its betas.

    np.savez writes members uncompressed, so pulling `image_count` alone costs a
    40 KB read rather than the 1 GB the betas would. That matters when a study
    fixes its image set by intersecting over all eight subjects but only
    analyses two of them.
    """
    with np.load(os.path.join(directory, f"sub{subject:02d}_{stream}.npz")) as data:
        return data["image_count"][:NUM_SHARED] > 0


def common_shared_images(subjects: list[SubjectBetas]) -> np.ndarray:
    """Shared-image indices every given subject actually saw at least once."""
    mask = np.ones(NUM_SHARED, dtype=bool)
    for subject in subjects:
        mask &= subject.shared_valid()
    return np.where(mask)[0]


def load_surface(path: str) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(path).dataobj).squeeze().astype(np.float32)


def roi_vertices(streams: list[str]) -> dict[str, np.ndarray]:
    """Vertex indices per hemisphere for the requested stream labels."""
    wanted = {STREAM_LABELS[name] for name in streams}
    selected = {}
    for hemisphere in ("lh", "rh"):
        labels = load_surface(os.path.join(LABEL_ROOT, f"{hemisphere}.streams.mgz"))
        selected[hemisphere] = np.where(np.isin(labels.astype(int), sorted(wanted)))[0]
    return selected


def main(args: argparse.Namespace) -> None:
    from scipy.io import loadmat

    design = loadmat(EXPDESIGN)
    subject_images = design["subjectim"][args.subject - 1]      # (10000,) 1-based nsdId
    ordering = design["masterordering"].ravel()                 # (30000,) 1-based
    shared = design["sharedix"].ravel()
    if not np.array_equal(subject_images[:1000], shared):
        raise SystemExit("ERROR: the first 1000 images are not the shared set")

    vertices = roi_vertices(args.streams)
    counts = {h: len(v) for h, v in vertices.items()}
    total = sum(counts.values())
    print(f"ROI {'+'.join(args.streams)}: {counts['lh']} lh + {counts['rh']} rh "
          f"= {total} vertices")

    subject_dir = os.path.join(BETAS_ROOT, f"subj{args.subject:02d}")
    sessions = sorted(
        int(name.split("session")[1].split(".")[0])
        for name in os.listdir(subject_dir)
        if name.startswith("lh.betas_session")
    )
    trials = len(sessions) * TRIALS_PER_SESSION
    if trials > len(ordering):
        raise SystemExit(f"ERROR: {trials} trials exceeds the 30000-trial design")
    print(f"Subject {args.subject}: {len(sessions)} sessions, {trials} trials")

    # Accumulate straight into per-image sums; holding all 30000 trials would
    # cost 2.3 GB for nothing, since every trial is used exactly once.
    image_sum = np.zeros((len(subject_images), total), dtype=np.float64)
    image_count = np.zeros(len(subject_images), dtype=np.int32)
    # The shared images are also kept un-averaged, one array per presentation.
    # Averaging three repeats is what makes the betas usable, but it also
    # destroys the only handle on how much of the result is noise, and every
    # cross-subject number needs a reliability ceiling to be readable. 1000
    # images is small enough that keeping all three costs ~230 MB.
    shared_reps = np.zeros((3, NUM_SHARED, total), dtype=np.float32)
    # Which (presentation, image) cells were actually filled. Subjects 3, 4, 6
    # and 8 stopped at 30-32 of the 40 sessions, and because the shared images
    # are spread across sessions those subjects are missing shared presentations
    # entirely -- 70 to 93 of the 1000 images have no trial at all. Without this
    # mask the gaps are indistinguishable from a measured response of zero, and
    # a consumer would quietly correlate real data against zeros.
    shared_rep_valid = np.zeros((3, NUM_SHARED), dtype=bool)

    started = time.time()
    for position, session in enumerate(sessions):
        blocks = []
        for hemisphere in ("lh", "rh"):
            path = os.path.join(
                subject_dir, f"{hemisphere}.betas_session{session:02d}.mgh"
            )
            data = load_surface(path)[vertices[hemisphere]]     # (n_vert, 750)
            blocks.append(data.T)                              # (750, n_vert)
        session_betas = np.concatenate(blocks, axis=1).astype(np.float64)
        if session_betas.shape[0] != TRIALS_PER_SESSION:
            raise SystemExit(f"ERROR: session {session} has "
                             f"{session_betas.shape[0]} trials")

        # Z-score within session, per vertex.
        session_betas -= session_betas.mean(axis=0, keepdims=True)
        session_betas /= session_betas.std(axis=0, keepdims=True).clip(1e-8)

        start = position * TRIALS_PER_SESSION
        image_index = ordering[start : start + TRIALS_PER_SESSION] - 1
        for trial, image in enumerate(image_index):
            repeat = image_count[image]
            image_sum[image] += session_betas[trial]
            image_count[image] = repeat + 1
            if image < NUM_SHARED and repeat < 3:
                shared_reps[repeat, image] = session_betas[trial]
                shared_rep_valid[repeat, image] = True
        if (position + 1) % 10 == 0 or position == len(sessions) - 1:
            elapsed = time.time() - started
            print(f"  session {session:02d} ({position + 1}/{len(sessions)}) "
                  f"| {elapsed:.0f}s")

    seen = image_count > 0
    print(f"Images with at least one trial: {int(seen.sum())} / {len(subject_images)}")
    print(f"Shared images with at least one trial: "
          f"{int(seen[:NUM_SHARED].sum())} / {NUM_SHARED}   "
          f"with all three: {int((image_count[:NUM_SHARED] == 3).sum())}")
    print(f"Repeat counts: {dict(zip(*[c.tolist() for c in np.unique(image_count, return_counts=True)]))}")
    betas = np.zeros_like(image_sum, dtype=np.float32)
    betas[seen] = (image_sum[seen] / image_count[seen, None]).astype(np.float32)

    ncsnr = np.concatenate([
        load_surface(os.path.join(subject_dir, f"{hemisphere}.ncsnr.mgh"))[
            vertices[hemisphere]
        ]
        for hemisphere in ("lh", "rh")
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"sub{args.subject:02d}_{'_'.join(args.streams)}.npz")
    np.savez(
        path,
        betas=betas,
        shared_reps=shared_reps,
        shared_rep_valid=shared_rep_valid,
        image_count=image_count,
        nsd_ids=subject_images,
        ncsnr=ncsnr,
        lh_vertices=vertices["lh"],
        rh_vertices=vertices["rh"],
        num_sessions=len(sessions),
        streams=np.array(args.streams),
    )
    print(f"\nWrote {path}  betas {betas.shape}  "
          f"({os.path.getsize(path) / 1e9:.2f} GB, {time.time() - started:.0f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--streams", nargs="+", default=["ventral"],
                        choices=sorted(STREAM_LABELS))
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
