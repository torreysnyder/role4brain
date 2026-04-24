"""
Train regression encoding models from BERT image embeddings to raw Sub 1 fMRI voxels.

Changes from the original PyTorch MLP/PCA script:
  1. Uses bert_embeddings_with_nsdIDs.json as input features.
  2. Restricts BERT embeddings to the NSD IDs present in Sub 1 training_images.
  3. Uses raw lh_training_fmri.npy and rh_training_fmri.npy voxel responses.
  4. Removes PCA and deep-learning MLP training.
  5. Fits separate RidgeCV models for left and right hemispheres.

Expected files in the working directory, unless paths are changed below:
  - bert_embeddings_with_nsdIDs.json
  - lh_training_fmri.npy
  - rh_training_fmri.npy
  - Sub 1 training image directory containing files named like train-*.png/jpg

Optional ROI plotting files, matching get_train_val.py:
  - mapping_prf-visualrois.npy, mapping_streams.npy, mapping_floc-bodies.npy,
    mapping_floc-places.npy
  - lh./rh. ROI challenge-space .npy files listed in ROI config below
"""

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# --------------------
# Config
# --------------------

TRAIN_IMAGE_DIR = Path("C:/Users/torre/OneDrive/Desktop/RP2/sub1_nsd_images/training_images")
BERT_EMBEDDINGS_JSON = Path("bert_embeddings_with_nsdIDs.json")
LH_FMRI_PATH = Path("lh_training_fmri.npy")
RH_FMRI_PATH = Path("rh_training_fmri.npy")

OUTPUT_DIR = Path("ridgecv_encoding_outputs")
LH_MODEL_PATH = OUTPUT_DIR / "ridgecv_lh_model.pkl"
RH_MODEL_PATH = OUTPUT_DIR / "ridgecv_rh_model.pkl"
MATCHED_EMBEDDINGS_CSV = OUTPUT_DIR / "matched_bert_embeddings.csv"
MATCHED_NSD_IDS_PATH = OUTPUT_DIR / "matched_nsd_ids.npy"
METRICS_JSON = OUTPUT_DIR / "ridgecv_metrics.json"
ROI_PLOT_PATH = OUTPUT_DIR / "ridgecv_roi_correlations.png"

SEED = 42
TRAIN_SPLIT = 0.90
ALPHAS = np.logspace(-6, 6, 100)
RIDGE_CV_FOLDS = 5

# Set to True to attempt ROI summary plotting, as in get_train_val.py.
# Leave False if you only want to train/evaluate the raw voxel models.
MAKE_ROI_PLOT = True

ROI_MAPPING_FILES = [
    "mapping_prf-visualrois.npy",
    "mapping_streams.npy",
    "mapping_floc-bodies.npy",
    "mapping_floc-places.npy",
]
LH_CHALLENGE_ROI_FILES = [
    "lh.prf-visualrois_challenge_space.npy",
    "lh.streams_challenge_space.npy",
    "lh.floc-bodies_challenge_space.npy",
    "lh.floc-places_challenge_space.npy",
]
RH_CHALLENGE_ROI_FILES = [
    "rh.prf-visualrois_challenge_space.npy",
    "rh.streams_challenge_space.npy",
    "rh.floc-bodies_challenge_space.npy",
    "rh.floc-places_challenge_space.npy",
]


# --------------------
# Utilities
# --------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_nsd_ids_from_training_images(directory: Path) -> List[int]:
    """Extract Sub 1 NSD IDs from filenames using the user's train-* convention."""
    if not directory.exists():
        raise FileNotFoundError(f"Training image directory not found: {directory}")

    nsd_ids: List[int] = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("train-"):
            nsd_id_str = filename[-9:-4]
            try:
                nsd_ids.append(int(nsd_id_str))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse NSD ID from {filename!r}; expected filename[-9:-4] to be numeric."
                ) from exc

    if not nsd_ids:
        raise ValueError(f"No train-* images found in {directory}")
    return nsd_ids


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flatten_embedding(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Embedding value is scalar; expected a vector or matrix.")
    return arr.reshape(-1) if arr.ndim > 1 else arr


def load_bert_embeddings_by_nsd_id(json_path: Path) -> Dict[int, np.ndarray]:
    """
    Load BERT embeddings and return {nsd_id: flattened_embedding}.

    This supports common JSON formats:
      - list of records with keys such as nsd_id / nsdID / linex_index / instance_id
        and pooled_output / embedding / bert_embedding / features
      - dict keyed by NSD ID, with either embedding values or nested records
    """
    if not json_path.exists():
        raise FileNotFoundError(f"BERT embeddings JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings: Dict[int, np.ndarray] = {}
    id_keys = ("nsd_id", "nsdID", "nsdId", "NSD_ID", "image_id", "instance_id", "linex_index")
    embedding_keys = (
        "pooled_output",
        "embedding",
        "bert_embedding",
        "bert_embeddings",
        "image_bert_embedding",
        "features",
        "model_output",
        "output",
    )

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # If top-level dict has a records-like list, use it; otherwise treat keys as IDs.
        records_like = _first_present(data, ("data", "records", "embeddings", "items"))
        if isinstance(records_like, list):
            records = records_like
        else:
            records = []
            for key, value in data.items():
                if isinstance(value, dict):
                    record = dict(value)
                    record.setdefault("nsd_id", key)
                    records.append(record)
                else:
                    records.append({"nsd_id": key, "embedding": value})
    else:
        raise ValueError("Unsupported JSON structure: expected a list or dictionary.")

    for record in records:
        if not isinstance(record, dict):
            continue
        nsd_id = _as_int_or_none(_first_present(record, id_keys))
        embedding_value = _first_present(record, embedding_keys)
        if nsd_id is None or embedding_value is None:
            continue
        embeddings[nsd_id] = _flatten_embedding(embedding_value)

    if not embeddings:
        raise ValueError(
            "No embeddings could be parsed. Check the JSON keys for NSD IDs and BERT vectors."
        )
    return embeddings


def build_matched_arrays(
    nsd_ids: List[int],
    embeddings_by_nsd_id: Dict[int, np.ndarray],
    lh_fmri: np.ndarray,
    rh_fmri: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match embeddings and fMRI rows by the Sub 1 nsd_ids order."""
    matched_nsd_ids: List[int] = []
    matched_embeddings: List[np.ndarray] = []
    fmri_index: List[int] = []

    for i, nsd_id in enumerate(nsd_ids):
        if nsd_id in embeddings_by_nsd_id:
            matched_nsd_ids.append(nsd_id)
            matched_embeddings.append(embeddings_by_nsd_id[nsd_id])
            fmri_index.append(i)

    if not matched_embeddings:
        raise ValueError("No overlap between Sub 1 NSD IDs and BERT embedding NSD IDs.")

    X = np.vstack(matched_embeddings).astype(np.float32)
    lh_y = lh_fmri[np.asarray(fmri_index)]
    rh_y = rh_fmri[np.asarray(fmri_index)]

    if len({row.shape[0] for row in matched_embeddings}) != 1:
        raise ValueError("Matched embeddings do not all have the same flattened dimensionality.")

    return np.asarray(matched_nsd_ids, dtype=np.int64), X, lh_y, rh_y


def train_val_split_ordered(
    X: np.ndarray,
    lh_y: np.ndarray,
    rh_y: np.ndarray,
    train_split: float,
) -> Tuple[np.ndarray, ...]:
    """Use the same ordered split style as get_train_val.py."""
    n_samples = X.shape[0]
    num_train = int(np.round(n_samples / 100 * (train_split * 100)))
    if num_train <= 0 or num_train >= n_samples:
        raise ValueError(
            f"Invalid train/validation split: {num_train} train out of {n_samples} samples."
        )

    idxs = np.arange(n_samples)
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    return (
        X[idxs_train], X[idxs_val],
        lh_y[idxs_train], lh_y[idxs_val],
        rh_y[idxs_train], rh_y[idxs_val],
    )


def fit_ridgecv(X_train: np.ndarray, y_train: np.ndarray, label: str) -> RidgeCV:
    cv = min(RIDGE_CV_FOLDS, len(X_train))
    if cv < 2:
        raise ValueError("Need at least 2 training samples for RidgeCV.")
    print(f"\nFitting {label} RidgeCV with {len(ALPHAS)} alphas and cv={cv}...")
    return RidgeCV(alphas=ALPHAS, cv=cv).fit(X_train, y_train)


def voxelwise_correlations(y_pred: np.ndarray, y_true: np.ndarray, label: str) -> np.ndarray:
    correlations = np.zeros(y_pred.shape[1], dtype=np.float32)
    for v in tqdm(range(y_pred.shape[1]), desc=f"{label} voxel correlations"):
        pred_v = y_pred[:, v]
        true_v = y_true[:, v]
        if np.std(pred_v) == 0 or np.std(true_v) == 0:
            correlations[v] = np.nan
        else:
            correlations[v] = pearsonr(pred_v, true_v)[0]
    return correlations


def maybe_plot_roi_correlations(lh_corr: np.ndarray, rh_corr: np.ndarray) -> Optional[Path]:
    required = ROI_MAPPING_FILES + LH_CHALLENGE_ROI_FILES + RH_CHALLENGE_ROI_FILES
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print("\nSkipping ROI plot because these files are missing:")
        for path in missing:
            print(f"  - {path}")
        return None

    roi_name_map = [np.load(path, allow_pickle=True).item() for path in ROI_MAPPING_FILES]
    lh_challenge_roi = [np.load(path) for path in LH_CHALLENGE_ROI_FILES]
    rh_challenge_roi = [np.load(path) for path in RH_CHALLENGE_ROI_FILES]

    roi_names: List[str] = []
    lh_roi_correlation: List[np.ndarray] = []
    rh_roi_correlation: List[np.ndarray] = []

    for r1 in range(len(lh_challenge_roi)):
        for roi_id, roi_name in roi_name_map[r1].items():
            if roi_id != 0:
                roi_names.append(roi_name)
                lh_roi_idx = np.where(lh_challenge_roi[r1] == roi_id)[0]
                rh_roi_idx = np.where(rh_challenge_roi[r1] == roi_id)[0]
                lh_roi_correlation.append(lh_corr[lh_roi_idx])
                rh_roi_correlation.append(rh_corr[rh_roi_idx])

    roi_names.append("All vertices")
    lh_roi_correlation.append(lh_corr)
    rh_roi_correlation.append(rh_corr)

    lh_mean_roi_correlation = [float(np.nanmean(values)) for values in lh_roi_correlation]
    rh_mean_roi_correlation = [float(np.nanmean(values)) for values in rh_roi_correlation]

    plt.figure(figsize=(18, 6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width / 2, lh_mean_roi_correlation, width, label="Left Hemisphere")
    plt.bar(x + width / 2, rh_mean_roi_correlation, width, label="Right Hemisphere")
    plt.xlabel("ROIs")
    plt.xticks(ticks=x, labels=roi_names, rotation=45, ha="right")
    plt.ylabel("Mean Pearson's r")
    plt.title("RidgeCV encoding performance by ROI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROI_PLOT_PATH, dpi=200)
    plt.close()
    return ROI_PLOT_PATH


# --------------------
# Main
# --------------------

def main() -> None:
    print("=" * 80)
    print("RIDGECV ENCODING MODEL — BERT IMAGE EMBEDDINGS -> RAW SUB 1 fMRI VOXELS")
    print("=" * 80)

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading NSD IDs from training images: {TRAIN_IMAGE_DIR}")
    nsd_ids = load_nsd_ids_from_training_images(TRAIN_IMAGE_DIR)
    print(f"  Found {len(nsd_ids)} Sub 1 training-image NSD IDs")

    print(f"\nLoading fMRI arrays: {LH_FMRI_PATH}, {RH_FMRI_PATH}")
    lh_fmri = np.load(LH_FMRI_PATH)
    rh_fmri = np.load(RH_FMRI_PATH)
    print(f"  LH fMRI shape: {lh_fmri.shape}")
    print(f"  RH fMRI shape: {rh_fmri.shape}")

    if lh_fmri.shape[0] != len(nsd_ids) or rh_fmri.shape[0] != len(nsd_ids):
        raise ValueError(
            "The number of fMRI rows must match the number of Sub 1 training-image NSD IDs. "
            f"Got len(nsd_ids)={len(nsd_ids)}, lh rows={lh_fmri.shape[0]}, rh rows={rh_fmri.shape[0]}."
        )

    print(f"\nLoading BERT embeddings from: {BERT_EMBEDDINGS_JSON}")
    embeddings_by_nsd_id = load_bert_embeddings_by_nsd_id(BERT_EMBEDDINGS_JSON)
    print(f"  Parsed {len(embeddings_by_nsd_id)} NSD-keyed embeddings")

    matched_nsd_ids, X, lh_y, rh_y = build_matched_arrays(
        nsd_ids=nsd_ids,
        embeddings_by_nsd_id=embeddings_by_nsd_id,
        lh_fmri=lh_fmri,
        rh_fmri=rh_fmri,
    )
    print(f"\nMatched samples: {len(matched_nsd_ids)}")
    print(f"  BERT feature matrix: {X.shape}")
    print(f"  Matched LH fMRI:     {lh_y.shape}")
    print(f"  Matched RH fMRI:     {rh_y.shape}")

    pd.DataFrame(X).assign(nsd_id=matched_nsd_ids).to_csv(MATCHED_EMBEDDINGS_CSV, index=False)
    np.save(MATCHED_NSD_IDS_PATH, matched_nsd_ids)

    X_train, X_val, lh_train, lh_val, rh_train, rh_val = train_val_split_ordered(
        X, lh_y, rh_y, TRAIN_SPLIT
    )
    print(f"\nTrain samples: {X_train.shape[0]} | Validation samples: {X_val.shape[0]}")
    print(f"  Train features: {X_train.shape}")
    print(f"  Val features:   {X_val.shape}")

    reg_lh = fit_ridgecv(X_train, lh_train, "left hemisphere")
    reg_rh = fit_ridgecv(X_train, rh_train, "right hemisphere")

    print(f"\nSelected alpha for left hemisphere:  {reg_lh.alpha_}")
    print(f"Selected alpha for right hemisphere: {reg_rh.alpha_}")

    print("\nPredicting validation fMRI responses...")
    lh_pred = reg_lh.predict(X_val)
    rh_pred = reg_rh.predict(X_val)

    lh_mse = float(mean_squared_error(lh_val, lh_pred))
    rh_mse = float(mean_squared_error(rh_val, rh_pred))
    print(f"  LH validation MSE: {lh_mse:.6f}")
    print(f"  RH validation MSE: {rh_mse:.6f}")

    lh_corr = voxelwise_correlations(lh_pred, lh_val, "LH")
    rh_corr = voxelwise_correlations(rh_pred, rh_val, "RH")
    print(f"  LH mean voxel Pearson r: {np.nanmean(lh_corr):.6f}")
    print(f"  RH mean voxel Pearson r: {np.nanmean(rh_corr):.6f}")

    with open(LH_MODEL_PATH, "wb") as f:
        pickle.dump(reg_lh, f)
    with open(RH_MODEL_PATH, "wb") as f:
        pickle.dump(reg_rh, f)

    metrics = {
        "n_sub1_nsd_ids": int(len(nsd_ids)),
        "n_matched_samples": int(len(matched_nsd_ids)),
        "bert_feature_dim": int(X.shape[1]),
        "lh_voxels": int(lh_y.shape[1]),
        "rh_voxels": int(rh_y.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "train_split": TRAIN_SPLIT,
        "ridge_alphas": ALPHAS.tolist(),
        "lh_alpha": float(reg_lh.alpha_),
        "rh_alpha": float(reg_rh.alpha_),
        "lh_val_mse": lh_mse,
        "rh_val_mse": rh_mse,
        "lh_mean_voxel_pearson_r": float(np.nanmean(lh_corr)),
        "rh_mean_voxel_pearson_r": float(np.nanmean(rh_corr)),
    }

    roi_plot = None
    if MAKE_ROI_PLOT:
        roi_plot = maybe_plot_roi_correlations(lh_corr, rh_corr)
        if roi_plot is not None:
            metrics["roi_plot"] = str(roi_plot)
            print(f"\nSaved ROI correlation plot to: {roi_plot}")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"LH model saved to:       {LH_MODEL_PATH}")
    print(f"RH model saved to:       {RH_MODEL_PATH}")
    print(f"Metrics saved to:        {METRICS_JSON}")
    print(f"Matched embeddings CSV:  {MATCHED_EMBEDDINGS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        sys.exit(1)
