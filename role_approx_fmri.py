"""
Trains the ROLE model to approximate PCA-reduced fMRI responses from
COCO category label sequences, using frozen BERT filler embeddings.

Modifications from original ROLE (Soulos et al. 2020):
  - Transformer-based role assigner (replaces LSTM)
  - Temperature annealing during training
  - Entropy regularization
  - fMRI responses as target encodings (replaces TPDN encodings)
  - Checkpoint selection via pure val MSE
  - Straight-through Gumbel-Softmax training (hard forward, soft backward)
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from role_learning_tensor_product_encoder import RoleLearningTensorProductEncoder
from sklearn.metrics import r2_score

# --------------------
# Utilities
# --------------------

def set_seed(seed):
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def get_temperature(epoch, max_epochs, T_start=5.0, T_end=0.5):
    """Anneal temperature from T_start to T_end over training"""
    progress = epoch / max_epochs
    return T_start * (T_end / T_start) ** progress


def entropy_loss(role_probs):
    """Encourage low-entropy (confident) role assignments

    Args:
        role_probs: (seq_len, batch, n_roles) or (batch, seq_len, n_roles)
    """
    # Add small epsilon for numerical stability
    entropy = -(role_probs * torch.log(role_probs + 1e-10)).sum(dim=-1)
    return entropy.mean()


def _role_logits_to_bsr(role_logits, batch_size):
    """Return ROLE logits/probabilities in shape (batch, seq_len, n_roles)."""
    if role_logits.dim() != 3:
        raise RuntimeError(f"Expected role logits to be 3D, got {tuple(role_logits.shape)}")

    # Common ROLE output shape is (seq_len, batch, n_roles).
    if role_logits.size(1) == batch_size:
        return role_logits.transpose(0, 1)

    # Some implementations already return (batch, seq_len, n_roles).
    if role_logits.size(0) == batch_size:
        return role_logits

    raise RuntimeError(
        f"Could not infer role-logit layout from shape {tuple(role_logits.shape)} "
        f"and batch_size={batch_size}"
    )

def filler_independence_loss_from_probs(
    role_probs,
    filler_ids,
    pad_mask=None,
    eps=1e-8,
):
    """
    Equal-per-filler KL independence loss:

        mean_f KL(P(role | filler=f) || P(role))

    Each unique filler category receives equal weight.
    """

    B, S, R = role_probs.shape

    role_probs_flat = role_probs.reshape(-1, R)
    filler_ids_flat = filler_ids.reshape(-1)

    # Remove PAD positions
    if pad_mask is not None:
        valid = pad_mask.reshape(-1).bool()
        role_probs_flat = role_probs_flat[valid]
        filler_ids_flat = filler_ids_flat[valid]

    if role_probs_flat.shape[0] == 0:
        return role_probs.sum() * 0.0

    # P(role): marginal role distribution across valid tokens
    p_role = role_probs_flat.mean(dim=0)

    unique_fillers = torch.unique(filler_ids_flat)

    losses = []

    for filler in unique_fillers:

        mask_f = filler_ids_flat == filler

        # P(role | filler=f)
        p_role_given_filler = role_probs_flat[mask_f].mean(dim=0)

        # KL(P(role | filler=f) || P(role))
        kl = (
            p_role_given_filler
            * (
                torch.log(p_role_given_filler + eps)
                - torch.log(p_role + eps)
            )
        ).sum()

        losses.append(kl)

    if len(losses) == 0:
        return role_probs.sum() * 0.0

    return torch.stack(losses).mean()

def pairwise_diversity_loss_from_probs(role_probs, pad_mask=None):
    sim = torch.bmm(role_probs, role_probs.transpose(1, 2))

    B, S, _ = sim.shape
    eye = torch.eye(S, device=sim.device).unsqueeze(0)
    offdiag = 1.0 - eye

    if pad_mask is not None:
        valid = (pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)).float()
        offdiag = offdiag * valid

    return (sim * offdiag).sum() / offdiag.sum().clamp_min(1.0)


def token_entropy_loss_from_probs(role_probs, pad_mask=None, eps=1e-8):
    """Encourage confident (low-entropy) role assignments at non-PAD positions."""
    token_entropy = -(role_probs * torch.log(role_probs + eps)).sum(dim=-1)
    if pad_mask is not None:
        valid = pad_mask.float()
        return (token_entropy * valid).sum() / valid.sum().clamp_min(1.0)
    return token_entropy.mean()


def one_hot_loss_from_probs(role_probs, pad_mask=None):
    """Encourage one-hot role assignments using confirmed probabilities.

    For a valid probability vector p, sum_r p_r(1-p_r) is non-negative,
    equals 0 for a one-hot assignment, and is maximized by the uniform
    distribution (0.875 for 8 roles). PAD positions are excluded.
    """
    per_token = (role_probs * (1.0 - role_probs)).sum(dim=-1)
    if pad_mask is not None:
        valid = pad_mask.float()
        return (per_token * valid).sum() / valid.sum().clamp_min(1.0)
    return per_token.mean()


def role_balance_loss_from_probs(role_probs, pad_mask=None, eps=1e-8):
    """Encourage balanced global use of all roles via KL(P(role) || Uniform).

    This is equivalent, up to an additive constant, to maximizing the entropy
    of the marginal role distribution. It complements token entropy: token
    entropy sharpens each assignment, while this term encourages the full
    role vocabulary to be used across tokens.
    """
    B, S, R = role_probs.shape
    probs = role_probs.reshape(-1, R)

    if pad_mask is not None:
        valid = pad_mask.reshape(-1).bool()
        probs = probs[valid]

    if probs.shape[0] == 0:
        return role_probs.sum() * 0.0

    p_role = probs.mean(dim=0)
    uniform = torch.full_like(p_role, 1.0 / R)

    return (
        p_role
        * (torch.log(p_role + eps) - torch.log(uniform + eps))
    ).sum()


class RoleLearningInstance(object):
    def __init__(
        self,
        filler,
        target_encoding,
        target_roles=None,
        image_id=None,
        split=None,
    ):
        self.filler = filler                    # LongTensor [1, seq_len]
        self.target_encoding = target_encoding  # FloatTensor [1, encoding_dim]
        self.target_roles = target_roles        # LongTensor [1, seq_len]
        self.image_id = image_id
        self.split = split

# --------------------
# Data loading
# --------------------

def load_fmri_encodings(metadata_filename, targets_filename, device, expected_dim=None):
    """
    Load PCA-reduced fMRI targets from the files generated by generate_full_fmri.py.

    generate_full_fmri.py writes:
      - fmri_targets_pca16.npy: array of shape (n_images, 16)
      - fmri_metadata_pca16.json: records with filler_ids, role_ids, and fmri_row

    This replaces the older JSON-only loader that expected each record to contain
    a target_encoding field directly.
    """
    print(f"Loading metadata from {metadata_filename}...")
    try:
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Metadata file {metadata_filename} not found.")
        print("Please run generate_full_fmri.py first.")
        sys.exit(1)

    print(f"Loading PCA fMRI targets from {targets_filename}...")
    try:
        targets = np.load(targets_filename)
    except FileNotFoundError:
        print(f"ERROR: Target file {targets_filename} not found.")
        print("Please run generate_full_fmri.py first.")
        sys.exit(1)

    if not metadata:
        print("ERROR: No metadata records found.")
        sys.exit(1)
    if targets.ndim != 2:
        print(f"ERROR: Expected targets to be 2D, got shape {targets.shape}")
        sys.exit(1)
    if expected_dim is not None and targets.shape[1] != expected_dim:
        print(
            f"ERROR: Expected PCA target dimension {expected_dim}, "
            f"but {targets_filename} has shape {targets.shape}."
        )
        sys.exit(1)

    print(f"Loaded {len(metadata)} metadata records")
    print(f"Loaded target matrix with shape {targets.shape}")

    # Determine role offset (role_ids are usually 0-indexed, but remap for safety)
    all_role_ids = []
    for entry in metadata:
        if entry.get('role_ids') is not None:
            all_role_ids.extend(entry['role_ids'])
    role_offset = min(all_role_ids) if all_role_ids else 0
    print(f"Role offset: {role_offset}")

    data = []
    skipped = 0
    for default_row, entry in enumerate(metadata):
        filler_ids = entry['filler_ids']
        fmri_row = int(entry.get('fmri_row', default_row))
        image_id = entry.get('image_id')
        split = entry.get('split')
        if fmri_row < 0 or fmri_row >= targets.shape[0]:
            skipped += 1
            continue

        target_encoding = targets[fmri_row]  # PCA-reduced fMRI vector, e.g. 16 dims
        role_ids = entry.get('role_ids', None)

        filler_t = Variable(torch.LongTensor(filler_ids), requires_grad=False).unsqueeze(0)
        encoding_t = Variable(torch.FloatTensor(target_encoding), requires_grad=False).unsqueeze(0)

        if role_ids is not None:
            remapped_role_ids = [r - role_offset for r in role_ids]
            role_t = Variable(torch.LongTensor(remapped_role_ids), requires_grad=False).unsqueeze(0)
        else:
            role_t = None

        filler_t = filler_t.to(device)
        encoding_t = encoding_t.to(device)
        if role_t is not None:
            role_t = role_t.to(device)

        data.append(
            RoleLearningInstance(
                filler_t,
                encoding_t,
                role_t,
                image_id=image_id,
                split=split,
            )
        )

    if not data:
        print("ERROR: No valid ROLE training instances were created.")
        sys.exit(1)

    if skipped:
        print(f"Skipped {skipped} metadata records with invalid fmri_row values")

    print(f"Created {len(data)} instances")
    print(f"  Sequence length:           {len(metadata[0]['filler_ids'])}")
    print(f"  Target encoding dimension: {targets.shape[1]}")
    return data, role_offset


# --------------------
# Training helpers
# --------------------

# --------------------
# Training helpers
# --------------------

def compute_target_stats(data):
    """Compute mean and std over target_encoding features."""
    with torch.no_grad():
        xs = [inst.target_encoding.view(-1).cpu().numpy() for inst in data]
        X = np.stack(xs, axis=0)
        mu = torch.from_numpy(X.mean(axis=0)).float()
        sigma = torch.from_numpy(X.std(axis=0) + 1e-6).float()
    return mu, sigma


def _standardize(x, mu, sigma):
    return (x - mu.to(x.device).unsqueeze(0)) / sigma.to(x.device).unsqueeze(0)


def batchify_with_roles(data, batch_size):
    """Create batches that preserve all instance information."""
    batches = []
    for bi in range(0, len(data), batch_size):
        batch_data = data[bi:min(bi + batch_size, len(data))]
        batches.append(batch_data)
    return batches


# --------------------
# ROLE Training
# --------------------

def train_epoch_role(
        model, train_batches, optimizer, criterion,
        use_regularization=True,
        temperature=1.0,
        entropy_weight=0.01,
        mu=None, sigma=None,
        role_diversity_weight=0.01,
        filler_independence_weight=0.01,
        token_entropy_weight=0.0,
        role_balance_weight=0.0,
        one_hot_weight=0.0,
        lambda_div=1.0
):
    """Train ROLE for one epoch to match TPDN encodings."""
    model.train()
    import random
    random.shuffle(train_batches)

    total_loss = 0.0
    total_mse = 0.0
    total_one_hot = 0.0
    total_l2 = 0.0
    total_pairwise_div = 0.0
    total_filler_independence = 0.0
    total_token_entropy = 0.0
    total_role_balance = 0.0
    num_updates = 0

    for batch in tqdm(train_batches, desc=f"Training ROLE (T={temperature:.2f})"):
        fillers = []
        targets = []

        for inst in batch:
            fillers.append(inst.filler)
            targets.append(inst.target_encoding)

        filler_batch = torch.cat(fillers, dim=0)
        target_batch = torch.cat(targets, dim=0)

        optimizer.zero_grad()

        # Forward - get TPDN encoding from ROLE (without final projection)
        pred, role_predictions = model(filler_batch, None, temperature=temperature)

        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        targ_flat = target_batch.view(target_batch.size(0), -1)

        # Optionally standardize
        if mu is not None and sigma is not None:
            pred_flat = _standardize(pred_flat, mu, sigma)
            targ_flat = _standardize(targ_flat, mu, sigma)

        # MSE reconstruction loss
        mse_loss = criterion(pred_flat, targ_flat)
        loss = mse_loss

        # Pairwise role diversity regularization.
        # This directly penalizes pairs of non-PAD tokens within the same image
        # when they have similar role-assignment distributions.
        if any(w > 0 for w in (
            role_diversity_weight,
            filler_independence_weight,
            token_entropy_weight,
            role_balance_weight,
            one_hot_weight,
        )):
            role_probs = None

            if hasattr(model, "role_assigner") and hasattr(model.role_assigner, "last_role_probs"):
                role_probs = (model.role_assigner.last_role_soft_probs
                              if model.role_assigner.use_gumbel_st
                              else model.role_assigner.last_role_probs)

            if role_probs is None:
                role_logits = _role_logits_to_bsr(
                    role_predictions,
                    batch_size=filler_batch.size(0)
                )
                role_probs = F.softmax(
                    role_logits / max(temperature, 1e-8),
                    dim=-1
                )

            if role_probs.size(0) != filler_batch.size(0):
                role_probs = role_probs.transpose(0, 1)

            pad_mask = filler_batch != 0

            role_div_loss = pairwise_diversity_loss_from_probs(
                role_probs,
                pad_mask=pad_mask
            )

            filler_indep_loss = filler_independence_loss_from_probs(role_probs, filler_batch, pad_mask=pad_mask)
            token_ent_loss = token_entropy_loss_from_probs(role_probs, pad_mask=pad_mask)
            role_balance_loss = role_balance_loss_from_probs(role_probs, pad_mask=pad_mask)
            one_hot_loss = one_hot_loss_from_probs(role_probs, pad_mask=pad_mask)

            total_pairwise_div += float(role_div_loss.item())
            total_filler_independence += float(filler_indep_loss.item())
            total_token_entropy += float(token_ent_loss.item())
            total_role_balance += float(role_balance_loss.item())

            weighted_role_div = role_diversity_weight * role_div_loss
            weighted_filler_indep = filler_independence_weight * filler_indep_loss
            weighted_token_entropy = token_entropy_weight * token_ent_loss
            weighted_role_balance = role_balance_weight * role_balance_loss
            weighted_one_hot = one_hot_weight * one_hot_loss

            loss = (
                loss
                + weighted_role_div
                + weighted_filler_indep
                + weighted_token_entropy
                + weighted_role_balance
                + weighted_one_hot
            )

            # Store the weighted one-hot contribution, matching the historical
            # role_train_one_hot history semantics.
            total_one_hot += float(weighted_one_hot.item())

            # Debug print once per epoch
            if num_updates == 0:
                print(
                    f"MSE={mse_loss.item():.4f} | "
                    f"role_div={role_div_loss.item():.4f} | "
                    f"weighted_role_div={weighted_role_div.item():.4f} | "
                    f"filler_indep={filler_indep_loss.item():.4f} | "
                    f"weighted_filler_indep={weighted_filler_indep.item():.4f} | "
                    f"token_entropy={token_ent_loss.item():.4f} | "
                    f"weighted_token_entropy={weighted_token_entropy.item():.4f} | "
                    f"role_balance={role_balance_loss.item():.4f} | "
                    f"weighted_role_balance={weighted_role_balance.item():.4f} | "
                    f"one_hot={one_hot_loss.item():.4f} | "
                    f"weighted_one_hot={weighted_one_hot.item():.4f}"
                )

        # Legacy model.get_regularization_loss(role_predictions) is intentionally
        # not used here. One-hot regularization is computed above from confirmed
        # role probabilities and excludes PAD tokens. L2/unique-role terms remain off.
        total_l2 += 0.0

        # Legacy entropy block disabled: PAD-aware token entropy is applied above.

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        num_updates += 1

    denom = max(1, num_updates)
    return (
        total_loss / denom,
        total_mse / denom,
        total_one_hot / denom,
        total_l2 / denom,
        total_pairwise_div / denom,
        total_filler_independence / denom,
        total_token_entropy / denom,
        total_role_balance / denom,
    )


@torch.no_grad()
def evaluate_role(role_model, data, temperature=1.0, mu=None, sigma=None):
    """Evaluate ROLE MSE and R² on dataset."""
    role_model.eval()

    preds = []
    targets = []

    for inst in data:
        pred, _ = role_model(inst.filler, None, temperature=temperature)

        pred = pred.view(pred.size(0), -1)
        targ = inst.target_encoding.view(inst.target_encoding.size(0), -1)

        if mu is not None and sigma is not None:
            pred = _standardize(pred, mu, sigma)
            targ = _standardize(targ, mu, sigma)

        preds.append(pred.cpu().numpy())
        targets.append(targ.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse = np.mean((preds - targets) ** 2)

    r2 = r2_score(
        targets,
        preds,
        multioutput="uniform_average"
    )

    return mse, r2

@torch.no_grad()
def evaluate_filler_independence(
    role_model,
    data,
    batch_size=64,
    temperature=1.0,
):
    """
    Compute filler-role independence loss across a validation dataset.

    The complete validation set is accumulated before calculating the loss,
    so repeated filler IDs can be compared across different images.

    Returns
    -------
    float
        Mean KL divergence between P(role | filler) and P(role).
        Lower is better.
    """
    role_model.eval()

    all_role_probs = []
    all_filler_ids = []

    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]

        filler_batch = torch.cat(
            [inst.filler for inst in batch],
            dim=0,
        )

        _, role_predictions = role_model(
            filler_batch,
            None,
            temperature=temperature,
        )

        role_probs = None

        if (
            hasattr(role_model, "role_assigner")
            and hasattr(role_model.role_assigner, "last_role_probs")
        ):
            role_probs = role_model.role_assigner.last_role_probs

        if role_probs is None:
            role_logits = _role_logits_to_bsr(
                role_predictions,
                batch_size=filler_batch.size(0),
            )

            role_probs = F.softmax(
                role_logits / max(temperature, 1e-8),
                dim=-1,
            )

        # Ensure (B, S, R)
        if role_probs.size(0) != filler_batch.size(0):
            if role_probs.size(1) == filler_batch.size(0):
                role_probs = role_probs.transpose(0, 1)
            else:
                raise RuntimeError(
                    f"Could not align role_probs shape "
                    f"{tuple(role_probs.shape)} with batch size "
                    f"{filler_batch.size(0)}"
                )

        all_role_probs.append(role_probs.detach())
        all_filler_ids.append(filler_batch.detach())

    if not all_role_probs:
        return float("nan")

    role_probs = torch.cat(all_role_probs, dim=0)
    filler_ids = torch.cat(all_filler_ids, dim=0)

    independence_loss = filler_independence_loss_from_probs(
        role_probs,
        filler_ids,
    )

    return float(independence_loss.item())

@torch.no_grad()
def compute_filler_role_mutual_information(
    role_model,
    dataset,
    device,
    temperature=1.0,
):
    """
    Computes I(Filler ; Role) using soft role assignments.

    Returns
    -------
    mi_bits
    """

    import numpy as np
    import torch
    import torch.nn.functional as F

    role_model.eval()

    num_roles = role_model.n_roles
    num_fillers = role_model.n_fillers

    joint = np.zeros((num_fillers, num_roles), dtype=np.float64)

    for inst in dataset:

        fillers = inst.filler.to(device)

        if fillers.dim() == 1:
            fillers = fillers.unsqueeze(0)

        _, role_logits = role_model(
            fillers,
            None,
            temperature=temperature,
        )

        role_probs = role_model.role_assigner.last_role_probs

        if role_probs is None:
            if role_logits.size(1) == fillers.size(0):
                role_logits = role_logits.transpose(0,1)

            role_probs = F.softmax(
                role_logits / temperature,
                dim=-1,
            )

        filler_ids = fillers.squeeze(0).cpu().numpy()
        probs = role_probs.squeeze(0).cpu().numpy()

        mask = filler_ids != 0

        filler_ids = filler_ids[mask]
        probs = probs[mask]

        for fid, p in zip(filler_ids, probs):
            joint[fid] += p

    joint /= joint.sum()

    p_filler = joint.sum(axis=1, keepdims=True)
    p_role = joint.sum(axis=0, keepdims=True)

    eps = 1e-12

    mi = (
        joint *
        np.log2(
            (joint + eps) /
            ((p_filler @ p_role) + eps)
        )
    ).sum()

    return float(mi)

@torch.no_grad()
def evaluate_role_utilization_diagnostics(
    role_model,
    data,
    batch_size=64,
    temperature=1.0,
    eps=1e-12,
):
    """Compute validation-set soft/argmax utilization diagnostics over non-PAD tokens.

    Returns
    -------
    dict with:
      soft_utilization: mean P(role) across valid tokens, shape [R]
      argmax_utilization: fraction of valid tokens whose winning role is r, shape [R]
      dsa: total-variation distance between soft and argmax utilization
           = 0.5 * sum_r |soft_r - argmax_r|. Lower means closer agreement.
      argmax_entropy: entropy of the argmax-utilization distribution in nats.
      active_argmax_roles: number of roles with at least one argmax-winning token.
    """
    role_model.eval()
    soft_sum = None
    argmax_counts = None
    n_valid = 0

    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        filler_batch = torch.cat([inst.filler for inst in batch], dim=0)

        _, role_predictions = role_model(
            filler_batch, None, temperature=temperature
        )

        role_probs = None
        if (
            hasattr(role_model, "role_assigner")
            and hasattr(role_model.role_assigner, "last_role_probs")
        ):
            role_probs = role_model.role_assigner.last_role_probs

        if role_probs is None:
            role_logits = _role_logits_to_bsr(
                role_predictions, batch_size=filler_batch.size(0)
            )
            role_probs = F.softmax(
                role_logits / max(temperature, 1e-8), dim=-1
            )

        if role_probs.size(0) != filler_batch.size(0):
            if role_probs.size(1) == filler_batch.size(0):
                role_probs = role_probs.transpose(0, 1)
            else:
                raise RuntimeError(
                    f"Could not align role_probs shape {tuple(role_probs.shape)} "
                    f"with batch size {filler_batch.size(0)}"
                )

        valid = (filler_batch != 0).reshape(-1)
        probs = role_probs.reshape(-1, role_probs.size(-1))[valid]
        if probs.numel() == 0:
            continue

        if soft_sum is None:
            soft_sum = torch.zeros(
                probs.size(-1), device=probs.device, dtype=probs.dtype
            )
            argmax_counts = torch.zeros_like(soft_sum)

        soft_sum += probs.sum(dim=0)
        winners = probs.argmax(dim=-1)
        argmax_counts += torch.bincount(
            winners, minlength=probs.size(-1)
        ).to(argmax_counts.dtype)
        n_valid += int(probs.size(0))

    if n_valid == 0:
        return {
            "soft_utilization": np.array([]),
            "argmax_utilization": np.array([]),
            "dsa": float("nan"),
            "argmax_entropy": float("nan"),
            "active_argmax_roles": 0,
        }

    soft = soft_sum / float(n_valid)
    argmax = argmax_counts / float(n_valid)
    dsa = 0.5 * torch.abs(soft - argmax).sum()
    argmax_entropy = -(argmax * torch.log(argmax + eps)).sum()
    active = int((argmax_counts > 0).sum().item())

    return {
        "soft_utilization": soft.detach().cpu().numpy(),
        "argmax_utilization": argmax.detach().cpu().numpy(),
        "dsa": float(dsa.item()),
        "argmax_entropy": float(argmax_entropy.item()),
        "active_argmax_roles": active,
    }


def plot_role_utilization_diagnostics(
    eval_epochs,
    dsa_hist,
    argmax_entropy_hist,
    active_argmax_roles_hist,
    n_roles,
):
    """Plot validation soft-vs-argmax discrepancy diagnostics over training."""
    if not eval_epochs:
        return

    eval_epochs = np.asarray(eval_epochs, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, dsa_hist, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Dsa (total variation distance)")
    plt.title("ROLE Model - Soft vs. Argmax Utilization Discrepancy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("role_dsa_over_epochs.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, argmax_entropy_hist, marker="o", label="Argmax entropy")
    plt.axhline(np.log(n_roles), linestyle="--", label=f"Maximum = ln({n_roles})")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy (nats)")
    plt.title("ROLE Model - Argmax Utilization Entropy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("role_argmax_entropy_over_epochs_oh.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, active_argmax_roles_hist, marker="o")
    plt.axhline(n_roles, linestyle="--", label=f"All roles = {n_roles}")
    plt.xlabel("Epoch")
    plt.ylabel("Number of active argmax roles")
    plt.title("ROLE Model - Active Argmax Roles")
    plt.yticks(range(0, n_roles + 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("role_active_argmax_roles_over_epochs_oh.png", dpi=200)
    plt.close()


def build_coco_bbox_lookup(instances, label_field="ground_truth"):
    """Build a COCO ground-truth bbox lookup for the requested ROLE instances."""
    from collections import defaultdict
    from pathlib import Path

    try:
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise ImportError(
            "Role-colored bounding-box snapshots require FiftyOne. "
            "Install FiftyOne or set PLOT_ROLE_BBOXES = False."
        ) from exc

    ids_by_split = defaultdict(list)
    for inst in instances:
        if inst.image_id is None or inst.split is None:
            continue
        ids_by_split[str(inst.split)].append(inst.image_id)

    for split in list(ids_by_split.keys()):
        ids_by_split[split] = list(dict.fromkeys(ids_by_split[split]))

    lookup = {}

    for split, image_ids in ids_by_split.items():
        if not image_ids:
            continue

        print(
            f"[role bbox] loading {len(image_ids)} COCO images "
            f"from split '{split}'"
        )

        split_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            label_types=["detections"],
            image_ids=image_ids,
        )

        for sample in split_dataset:
            coco_id = None

            if sample.has_field("coco_id"):
                coco_id = sample["coco_id"]

            if coco_id is None:
                try:
                    coco_id = int(Path(sample.filepath).stem)
                except (TypeError, ValueError):
                    continue

            labels = sample[label_field]
            detections = []

            if labels is not None:
                for det in labels.detections:
                    detections.append(
                        {
                            "label": det.label,
                            "bounding_box": list(det.bounding_box),
                        }
                    )

            lookup[int(coco_id)] = {
                "filepath": sample.filepath,
                "detections": detections,
            }

    print(f"[role bbox] prepared annotations for {len(lookup)} images")
    return lookup


def save_role_colored_bbox_image(
    image_id,
    role_probs_for_image,
    filler_ids_for_image,
    bbox_lookup,
    label_to_filler_id,
    output_path,
):
    """Draw ground-truth COCO boxes with colors determined by ROLE argmax."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np

    if image_id is None:
        print("[role bbox] skipping image with missing image_id")
        return False

    try:
        image_id_int = int(image_id)
    except (TypeError, ValueError):
        image_id_int = image_id

    record = bbox_lookup.get(image_id_int)
    if record is None:
        print(f"[role bbox] no COCO bbox record found for image_id={image_id}")
        return False

    filler_ids_for_image = np.asarray(filler_ids_for_image)
    nonpad_mask = filler_ids_for_image != 0
    nonpad_filler_ids = filler_ids_for_image[nonpad_mask]
    nonpad_probs = np.asarray(role_probs_for_image)[nonpad_mask]

    filler_to_role = {
        int(fid): int(np.argmax(prob))
        for fid, prob in zip(nonpad_filler_ids, nonpad_probs)
    }

    image = Image.open(record["filepath"]).convert("RGB")
    width, height = image.size

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    cmap = plt.get_cmap("tab10")
    roles_present = set()

    for det in record["detections"]:
        label = det["label"]
        filler_id = label_to_filler_id.get(label)

        if filler_id is None or int(filler_id) not in filler_to_role:
            continue

        role_idx = filler_to_role[int(filler_id)]
        roles_present.add(role_idx)

        x, y, w, h = det["bounding_box"]
        x_px = x * width
        y_px = y * height
        w_px = w * width
        h_px = h * height

        color = cmap(role_idx % cmap.N)

        rect = patches.Rectangle(
            (x_px, y_px),
            w_px,
            h_px,
            linewidth=3,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x_px,
            max(0, y_px - 4),
            f"{label} | filler {int(filler_id)} | role {role_idx}",
            fontsize=9,
            color="white",
            bbox={
                "facecolor": color,
                "edgecolor": color,
                "alpha": 0.85,
                "pad": 2,
            },
        )

    legend_handles = [
        patches.Patch(
            facecolor=cmap(role_idx % cmap.N),
            edgecolor=cmap(role_idx % cmap.N),
            label=f"Role {role_idx}",
        )
        for role_idx in sorted(roles_present)
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            framealpha=0.9,
            title="ROLE assignment",
        )

    ax.set_title(f"Role-colored ground-truth boxes | image {image_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------
# Role assignment tracking
# --------------------

@torch.no_grad()
def snapshot_role_distributions(
    role_model,
    data,
    device,
    out_dir="role_plots_fmri",
    epoch_idx=0,
    max_items=32,
    temperature=1.0,
    plot_role_bboxes=False,
    bbox_lookup=None,
    label_to_filler_id=None,
):
    """
    Save one role-distribution heatmap per image and optionally a role-colored bbox image.

    Returns
    -------
    {
        "entropy": [...],
        "sparsity": [...]
    }
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    os.makedirs(out_dir, exist_ok=True)

    epoch_dir = os.path.join(out_dir, f"epoch_{epoch_idx + 1:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    n = min(max_items, len(data))

    if n <= 0:
        return {
            "entropy": [],
            "sparsity": [],
        }

    fillers = []

    for i in range(n):
        f = data[i].filler

        if f.dim() == 2 and f.size(0) == 1:
            f = f.squeeze(0)
        elif f.dim() != 1:
            f = f.view(-1)

        fillers.append(f)

    filler_batch = torch.stack(fillers, dim=0).to(device)

    role_model.eval()

    _, role_logits = role_model(
        filler_batch,
        None,
        temperature=temperature,
    )

    role_probs = None

    if (
        hasattr(role_model, "role_assigner")
        and hasattr(role_model.role_assigner, "last_role_probs")
    ):
        role_probs = role_model.role_assigner.last_role_probs

    if role_probs is None:
        if role_logits.dim() != 3:
            raise RuntimeError(
                f"Unexpected role_logits shape: {tuple(role_logits.shape)}"
            )

        if role_logits.size(1) == filler_batch.size(0):
            role_logits = role_logits.transpose(0, 1)

        role_probs = F.softmax(
            role_logits / max(temperature, 1e-8),
            dim=-1,
        )

    if role_probs.dim() != 3:
        raise RuntimeError(
            f"Unexpected role_probs shape: {tuple(role_probs.shape)}"
        )

    if (
        role_probs.size(0) != filler_batch.size(0)
        and role_probs.size(1) == filler_batch.size(0)
    ):
        role_probs = role_probs.transpose(0, 1)

    entropy_values = []
    sparsity_values = []

    for image_idx in range(role_probs.size(0)):

        image_id = data[image_idx].image_id

        image_probs = role_probs[image_idx].detach().cpu().numpy()

        filler_ids = filler_batch[image_idx].detach().cpu().numpy()
        nonpad_mask = filler_ids != 0
        true_filler_ids = filler_ids[nonpad_mask]

        plot_dist = image_probs[nonpad_mask]

        if plot_dist.shape[0] == 0:
            continue

        S, R = plot_dist.shape

        argmax_roles = np.argmax(plot_dist, axis=1)

        print(
            f"[role snapshot] epoch={epoch_idx + 1}, "
            f"image={image_id}, "
            f"nonpad_positions={S}, "
            f"argmax_roles={argmax_roles.tolist()}"
        )

        p = np.clip(plot_dist, 1e-12, 1.0)

        entropy_per_position = -(p * np.log(p)).sum(axis=-1)
        sparsity_per_position = np.max(p, axis=-1)

        entropy_values.extend(entropy_per_position.tolist())
        sparsity_values.extend(sparsity_per_position.tolist())

        plt.figure(figsize=(10, max(3, min(8, 0.35 * S))))

        plt.imshow(
            plot_dist,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=max(1e-6, float(plot_dist.max())),
        )

        plt.colorbar(label="P(role | position)")
        plt.xlabel("Role index")
        plt.ylabel("Non-PAD filler ID")

        plt.title(
            f"ROLE assignment distribution | "
            f"epoch {epoch_idx + 1} | "
            f"image {image_id}"
        )

        plt.xticks(range(R))
        plt.yticks(range(S), true_filler_ids)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                epoch_dir,
                f"role_dist_epoch_{epoch_idx + 1:03d}"
                f"_image_{image_id}.png",
            ),
            dpi=200,
        )

        plt.close()

        if plot_role_bboxes:
            if bbox_lookup is None or label_to_filler_id is None:
                raise ValueError(
                    "plot_role_bboxes=True requires bbox_lookup and "
                    "label_to_filler_id."
                )

            bbox_output_path = os.path.join(
                epoch_dir,
                f"role_bbox_epoch_{epoch_idx + 1:03d}"
                f"_image_{image_id}.png",
            )

            save_role_colored_bbox_image(
                image_id=image_id,
                role_probs_for_image=image_probs,
                filler_ids_for_image=filler_ids,
                bbox_lookup=bbox_lookup,
                label_to_filler_id=label_to_filler_id,
                output_path=bbox_output_path,
            )

    return {
        "entropy": entropy_values,
        "sparsity": sparsity_values,
    }


def plot_role_summaries(role_entropy_hist, role_sparsity_hist, role_epochs=None, out_dir="role_plots_fmri"):
    """Plot role-distribution entropy and sparsity over training."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    role_entropy_hist = np.asarray(role_entropy_hist, dtype=float)
    role_sparsity_hist = np.asarray(role_sparsity_hist, dtype=float)

    if role_entropy_hist.size == 0 or role_sparsity_hist.size == 0:
        print("No role-distribution snapshots to plot.")
        return

    if role_epochs is None:
        role_epochs = np.arange(1, role_entropy_hist.shape[0] + 1)
    else:
        role_epochs = np.asarray(role_epochs)

    mean_entropy = np.nanmean(role_entropy_hist, axis=1)
    mean_sparsity = np.nanmean(role_sparsity_hist, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(role_epochs, mean_entropy, marker="o", label="Mean entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.title("ROLE assignment entropy across training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "role_entropy_over_epochs_oh.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(role_epochs, mean_sparsity, marker="o", label="Mean max probability")
    plt.xlabel("Epoch")
    plt.ylabel("Max role probability")
    plt.title("ROLE assignment sparsity across training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "role_sparsity_over_epochs_oh.png"), dpi=200)
    plt.close()

def plot_training_curves(
    role_train_mse,
    role_val_mse,
    role_train_r2,
    role_val_r2,
    role_train_one_hot=None,
    role_train_l2=None,
    role_train_filler_independence=None,
    role_filler_mi=None,
    role_eval_epochs=None,
    role_train_pairwise_diversity=None,
):
    """
    Plot ROLE training curves using two history cadences.

    Every training epoch
    --------------------
    role_train_mse  (full train set, model.eval())

    Evaluation epochs only
    ----------------------
    role_val_mse
    role_train_r2
    role_val_r2
    role_filler_mi
    role_eval_epochs

    Every training epoch
    --------------------
    role_train_one_hot
    role_train_l2
    role_train_filler_independence
    role_train_pairwise_diversity

    This function is safe to call repeatedly during training. Each call
    overwrites the same PNG files, so the plots always show the latest
    available training history.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # Evaluation-epoch histories
    # ============================================================
    if role_eval_epochs is None:
        eval_epochs = np.arange(1, len(role_val_mse) + 1)
    else:
        eval_epochs = np.asarray(role_eval_epochs, dtype=int)

    eval_histories = {
        "role_val_mse": role_val_mse,
        "role_train_r2": role_train_r2,
        "role_val_r2": role_val_r2,
    }

    for name, values in eval_histories.items():
        if values is None:
            raise ValueError(f"{name} cannot be None.")

        if len(values) != len(eval_epochs):
            raise ValueError(
                f"{name} has {len(values)} values, but "
                f"role_eval_epochs has {len(eval_epochs)} values."
            )

    if role_filler_mi is not None:
        if len(role_filler_mi) != len(eval_epochs):
            raise ValueError(
                f"role_filler_mi has {len(role_filler_mi)} values, but "
                f"role_eval_epochs has {len(eval_epochs)} values."
            )

    # ============================================================
    # Every-training-epoch histories
    # ============================================================
    per_epoch_histories = [
        role_train_mse,
        role_train_one_hot,
        role_train_l2,
        role_train_filler_independence,
        role_train_pairwise_diversity,
    ]

    per_epoch_lengths = [
        len(values)
        for values in per_epoch_histories
        if values is not None
    ]

    if per_epoch_lengths:
        n_training_epochs = max(per_epoch_lengths)
    elif len(eval_epochs) > 0:
        n_training_epochs = int(eval_epochs[-1])
    else:
        n_training_epochs = 0

    epochs_role = np.arange(1, n_training_epochs + 1)

    def validate_per_epoch_history(name, values):
        if values is None:
            return

        if len(values) != n_training_epochs:
            raise ValueError(
                f"{name} has {len(values)} values, but expected "
                f"{n_training_epochs} values—one per actual training epoch."
            )

    validate_per_epoch_history(
        "role_train_mse",
        role_train_mse,
    )
    validate_per_epoch_history(
        "role_train_one_hot",
        role_train_one_hot,
    )
    validate_per_epoch_history(
        "role_train_l2",
        role_train_l2,
    )
    validate_per_epoch_history(
        "role_train_filler_independence",
        role_train_filler_independence,
    )
    validate_per_epoch_history(
        "role_train_pairwise_diversity",
        role_train_pairwise_diversity,
    )

    # ============================================================
    # MSE:
    # ============================================================
    if len(eval_epochs) > 0:

        plt.figure(figsize=(10, 6))

        # Train MSE is recomputed in model.eval() after EVERY epoch.
        train_mse_epochs = np.arange(1, len(role_train_mse) + 1)
        plt.plot(
            train_mse_epochs,
            role_train_mse,
            marker="o",
            label="Train MSE",
        )

        # Validation MSE is available only at evaluation epochs.
        plt.plot(
            eval_epochs,
            role_val_mse,
            marker="s",
            label="Validation MSE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(
            "ROLE Model Reconstruction MSE (fMRI Encodings)"
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.5,
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            "role_mse_oh.png",
            dpi=200,
        )

        plt.close()

        # ========================================================
        # R²
        # ========================================================
        plt.figure(figsize=(10, 6))

        plt.plot(
            eval_epochs,
            role_train_r2,
            marker="o",
            label="Train R²",
        )
        plt.plot(
            eval_epochs,
            role_val_r2,
            marker="s",
            label="Validation R²",
        )

        plt.xlabel("Epoch")
        plt.ylabel("R²")
        plt.title("ROLE Model Variance Explained (fMRI Encodings)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            "role_r2_oh.png",
            dpi=200,
        )
        plt.close()

    # ============================================================
    # Filler-independence loss
    # ============================================================
    if (
        role_train_filler_independence is not None
        and len(role_train_filler_independence) > 0
    ):
        filler_independence = np.asarray(
            role_train_filler_independence,
            dtype=float,
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs_role,
            filler_independence,
            marker="d",
            label="Filler independence loss",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ROLE Model - Filler Independence Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            "role_model_filler_indep_oh.png",
            dpi=200,
        )
        plt.close()

    # ============================================================
    # Pairwise-diversity loss
    # ============================================================
    if (
        role_train_pairwise_diversity is not None
        and len(role_train_pairwise_diversity) > 0
    ):
        pairwise_diversity = np.asarray(
            role_train_pairwise_diversity,
            dtype=float,
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs_role,
            pairwise_diversity,
            marker="d",
            label="Pairwise diversity loss",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ROLE Model - Pairwise Diversity Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            "role_model_pairwise_diversity_loss_oh.png",
            dpi=200,
        )
        plt.close()

    # ============================================================
    # Filler-role mutual information
    # ============================================================
    if role_filler_mi is not None and len(role_filler_mi) > 0:
        plt.figure(figsize=(8, 5))

        plt.plot(
            eval_epochs,
            role_filler_mi,
            marker="o",
            linewidth=2,
            label="Validation filler-role MI",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Mutual information (bits)")
        plt.title("Filler-Role Mutual Information")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            "role_model_filler_role_mutual_information_oh.png",
            dpi=200,
        )
        plt.close()

    # ============================================================
    # Legacy model regularization losses
    # ============================================================
    if role_train_one_hot is not None or role_train_l2 is not None:
        plt.figure(figsize=(10, 6))
        eps = 1e-12

        if role_train_one_hot is not None:
            one_hot = np.asarray(
                role_train_one_hot,
                dtype=float,
            )

            plt.plot(
                epochs_role,
                one_hot + eps,
                marker="o",
                label="One-hot regularization",
            )

        if role_train_l2 is not None:
            l2 = np.asarray(
                role_train_l2,
                dtype=float,
            )

            plt.plot(
                epochs_role,
                l2 + eps,
                marker="^",
                label="L2 regularization",
            )

        plt.yscale("symlog", linthresh=1e-6)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (symlog)")
        plt.title("ROLE Model - Legacy Regularization Losses")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            "role_model_reg_loss_oh.png",
            dpi=200,
        )
        plt.close()


# --------------------
# Main
# --------------------

def main():
    print("=" * 80)
    print("ROLE APPROXIMATION WITH PCA-16 fMRI ENCODINGS")
    print("=" * 80)

    # =========================
    # Config
    # =========================
    FMRI_METADATA_JSON = "fmri_metadata_pca8.json"
    FMRI_TARGETS_NPY = "fmri_targets_pca8.npy"
    TARGET_ENCODING_DIM = 8
    SEED = 42

    # Early stopping
    EARLY_STOPPING_PATIENCE = 30
    EARLY_STOPPING_MIN_DELTA = 1e-4
    EARLY_STOPPING_WARMUP = 50

    # Combined validation objective:
    # val_score = val_mse + weight * val_filler_independence
    EARLY_STOP_INDEPENDENCE_WEIGHT = 0.1

    # Batch size used only for validation independence evaluation
    INDEPENDENCE_EVAL_BATCH_SIZE = 64
    # ROLE training (from scratch)
    TOTAL_ROLE_EPOCHS = 150
    BATCH_SIZE = 4              # reduced from 16 to lower GPU memory usage
    ROLE_LR = 1e-4
    USE_ROLE_REG = True
    USE_GUMBEL_ST = True  # hard role selection forward; soft gradients backward
    STANDARDIZE = True

    # Runtime: validate less frequently
    EVAL_INTERVAL = 5  # validate every N epochs (plus first eval epoch and last)

    # Temperature annealing
    T_START = 5.0
    T_END = 0.5

    # Per-token sharpness regularization.
    # Lower entropy means a more confident role distribution for each non-PAD filler.
    # For 8 roles, maximum entropy is ln(8) ~= 2.079 nats.
    TOKEN_ENTROPY_WEIGHT = 0.5

    # PAD-aware one-hot sharpness regularization computed directly from role_probs.
    # For N_ROLES=8, the raw loss is in [0, 0.875] for valid probabilities.
    ONE_HOT_WEIGHT = 0.1

    # Global role-balance regularization.
    # Minimizes KL(P(role) || Uniform), encouraging all N_ROLES to be used
    # across non-PAD tokens. Do not add a separate marginal-entropy loss:
    # for a uniform target it has the same gradient up to a constant.
    ROLE_BALANCE_WEIGHT = 0.05

    # Pairwise role diversity regularization.
    # ROLE_DIVERSITY_WEIGHT controls the overall contribution to the training loss.
    # LAMBDA_DIV controls how strongly pairwise within-image role dissimilarity is
    # penalized relative to sharp token-level assignments.
    # ABLATION: pairwise-diversity term is disabled. The raw diversity
    # diagnostic is still computed/tracked, but contributes 0 to the objective.
    ROLE_DIVERSITY_WEIGHT = 5.0
    FILLER_INDEPENDENCE_WEIGHT = 1.0
    LAMBDA_DIV = 1.0

    # Model architecture
    FILLER_DIM = 128    # post-squeeze binding dimension (BERT 768 -> 64 via embedder_squeeze)
    ROLE_DIM = 10       # dense learned role embedding dim (independent of n_roles=80)
    N_ROLES = 8       # total number of possible category positions play around with
    HIDDEN_DIM = 384   # reduced from 768 to lower GPU memory usage
    NUM_LAYERS = 6     # reduced from 4 to lower GPU memory usage
    N_HEAD = 6 # reduce num
    DROPOUT = 0.1

    # Sequence length cap: sequences longer than this are discarded.
    # Transformer attention memory scales as O(S^2), so capping sequence length
    # is the most impactful way to reduce memory. Set to N_ROLES (80) to keep all.
    MAX_SEQ_LEN = 15

    # Role assignment distribution tracking
    PLOT_ROLE_DISTS = True
    ROLE_OUT_DIR = "role_gumbel_st"
    ROLE_PLOT_EVERY = 5  # snapshot every N epochs
    ROLE_SAMPLE_SIZE = 25

    # Role-colored ground-truth bounding-box snapshots.
    # Uses COCO-2017 detections through FiftyOne, matching load_bbox_images.py.
    PLOT_ROLE_BBOXES = True
    FILLER_VOCAB_JSON = "bert_filler_vocab.json"

    # =========================
    # Setup
    # =========================
    device = set_seed(SEED)

    # =========================
    # Load data
    # =========================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    data, role_offset = load_fmri_encodings(
        FMRI_METADATA_JSON,
        FMRI_TARGETS_NPY,
        device,
        expected_dim=TARGET_ENCODING_DIM,
    )

    # Filter to sequences at or below MAX_SEQ_LEN real (non-PAD) tokens.
    # filler_ids[0] is the PAD index (0), so real length = number of non-zero filler ids.
    if MAX_SEQ_LEN < N_ROLES:
        before = len(data)
        data = [inst for inst in data if int((inst.filler != 0).sum()) <= MAX_SEQ_LEN]
        print(f"Filtered sequences to max real length {MAX_SEQ_LEN}: "
              f"{before} -> {len(data)} ({before - len(data)} discarded)")

    seq_len = data[0].filler.size(1)
    encoding_dim = data[0].target_encoding.view(-1).numel()
    num_fillers = max(int(inst.filler.max().item()) for inst in data) + 1

    # n_roles is fixed to N_ROLES (80) regardless of sequence length filter,
    # since the role space covers all 80 possible category positions
    n_roles = N_ROLES

    print(f"\nDataset info:")
    print(f"  Sequences: {len(data)}")
    print(f"  Padded sequence length: {seq_len}")
    print(f"  Max real sequence length: {MAX_SEQ_LEN}")
    print(f"  Num fillers: {num_fillers}")
    print(f"  Num roles: {n_roles}")
    print(f"  Target encoding dimension: {encoding_dim}")
    print(f"  TPR dim (filler_dim * role_dim): {FILLER_DIM * ROLE_DIM}")

    # Compute statistics for standardization
    enc_mu, enc_sigma = compute_target_stats(data) if STANDARDIZE else (None, None)
    if STANDARDIZE:
        print(f"  Encoding mean: {enc_mu.mean().item():.4f}")
        print(f"  Encoding std: {enc_sigma.mean().item():.4f}")

    split_indices = list(range(len(data)))
    split_rng = random.Random(SEED)
    split_rng.shuffle(split_indices)

    split_idx = int(0.8 * len(split_indices))

    train_indices = split_indices[:split_idx]
    test_indices = split_indices[split_idx:]

    train_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in test_indices]
    print(f"\nTrain: {len(train_set)}, Test: {len(test_set)}")

    # Prepare bbox metadata once for the fixed snapshot images, rather than
    # reloading COCO detections at every snapshot epoch.
    bbox_lookup = None
    label_to_filler_id = None

    if PLOT_ROLE_BBOXES:
        with open(FILLER_VOCAB_JSON, "r") as f:
            label_to_filler_id = json.load(f)

        snapshot_instances = test_set[: min(ROLE_SAMPLE_SIZE, len(test_set))]
        bbox_lookup = build_coco_bbox_lookup(snapshot_instances)

    train_batches = batchify_with_roles(train_set, BATCH_SIZE)

    # =========================
    # Load pretrained BERT filler embeddings
    # =========================
    BERT_EMBEDDING_PATH = "bert_embedding_matrix.pt"
    print(f"\nLoading BERT embedding matrix from {BERT_EMBEDDING_PATH}...")
    try:
        pretrained_filler_embeddings = torch.load(BERT_EMBEDDING_PATH, map_location=device)
        print(f"  Loaded embedding matrix: {pretrained_filler_embeddings.shape}")
    except FileNotFoundError:
        print(f"ERROR: {BERT_EMBEDDING_PATH} not found. Run generate_bert_embeddings.py first.")
        sys.exit(1)

    # =========================
    # Create ROLE model
    # =========================
    print("\n" + "=" * 80)
    print("CREATING ROLE MODEL")
    print("=" * 80)

    role_model = RoleLearningTensorProductEncoder(
        n_roles=n_roles,
        n_fillers=num_fillers,
        filler_dim=FILLER_DIM,
        role_dim=ROLE_DIM,
        final_layer_width=TARGET_ENCODING_DIM,  # projects ROLE output -> 16-dim PCA fMRI target
        binder="tpr",
        role_learner_hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        nhead=N_HEAD,
        dim_feedforward=4 * HIDDEN_DIM,
        dropout=DROPOUT,
        softmax_roles=True,
        embedder_squeeze=64,                    # squeezes BERT 768-dim -> 64-dim before binding
        unique_role_regularization_weight=0.0,
        one_hot_regularization_weight=0.0,  # handled explicitly from role_probs below
        pretrained_filler_embeddings=pretrained_filler_embeddings,
    ).to(device)
    role_model.role_assigner.use_gumbel_st = USE_GUMBEL_ST
    role_model.role_assigner.gumbel_hard_eval = True
    role_model.use_regularization(USE_ROLE_REG)

    # =========================
    # Training from scratch setup
    # =========================
    start_epoch_0idx = 0
    total_role_epochs = TOTAL_ROLE_EPOCHS
    print(f"\nROLE training plan:")
    print(f"  Training from scratch for {total_role_epochs} epochs")
    print(f"  Temperature annealing: {T_START} → {T_END}")
    print(f"  Token entropy / sharpness weight: {TOKEN_ENTROPY_WEIGHT}")
    print(f"  One-hot sharpness weight: {ONE_HOT_WEIGHT}")
    print(f"  Role balance weight: {ROLE_BALANCE_WEIGHT}")
    print(f"  Role diversity weight: {ROLE_DIVERSITY_WEIGHT}")
    print("  Pairwise-diversity ablation: ENABLED (weight = 0; diagnostic still tracked)")
    print(f"  Pairwise diversity lambda: {LAMBDA_DIV}")
    print(f"  Validation interval: every {EVAL_INTERVAL} epochs")
    print(
        f"  Checkpoint selection: "
        f"val MSE + {EARLY_STOP_INDEPENDENCE_WEIGHT} × filler independence"
    )

    # =========================
    # Train ROLE (resume)
    # =========================
    print("\n" + "=" * 80)
    print("TRAINING ROLE MODEL")
    print("=" * 80)

    role_optimizer = optim.Adam(role_model.parameters(), lr=ROLE_LR, weight_decay=1e-4)
    role_criterion = nn.MSELoss()

    # Histories
    # Every training epoch:
    role_optimization_mse_hist = []
    role_train_one_hot_hist = []
    role_train_l2_hist = []
    role_pairwise_div_hist = []
    role_filler_independence_hist = []
    role_token_entropy_hist = []
    role_balance_hist = []

    # Evaluation epochs only:
    role_train_mse_hist = []
    role_val_mse_hist = []
    role_train_r2_hist = []
    role_val_r2_hist = []
    role_filler_mi_hist = []
    role_dsa_hist = []
    role_argmax_entropy_hist = []
    role_active_argmax_roles_hist = []
    role_soft_utilization_hist = []
    role_argmax_utilization_hist = []

    role_entropy_hist = []
    role_sparsity_hist = []
    role_snapshot_epochs = []
    role_eval_epochs = []

    best_role_path = "best_role_model_fmri_pca16.pt"
    latest_role_path = "latest_role_training_checkpoint.pt"
    best_score = float("inf")
    best_role_epoch_1idx = None
    best_role_mse = float("inf")
    best_role_filler_independence = float("inf")

    early_stop_counter = 0
    stopped_early = False

    for epoch_0idx in range(start_epoch_0idx, total_role_epochs):
        temperature = get_temperature(epoch_0idx, total_role_epochs, T_START, T_END)

        _, avg_mse, avg_one_hot, avg_l2, avg_pairwise_div, avg_filler_independence, avg_token_entropy, avg_role_balance = train_epoch_role(
            role_model,
            train_batches,
            role_optimizer,
            role_criterion,
            use_regularization=USE_ROLE_REG,
            temperature=temperature,
            entropy_weight=0.0,
            mu=enc_mu,
            sigma=enc_sigma,
            role_diversity_weight=ROLE_DIVERSITY_WEIGHT,
            filler_independence_weight=FILLER_INDEPENDENCE_WEIGHT,
            token_entropy_weight=TOKEN_ENTROPY_WEIGHT,
            role_balance_weight=ROLE_BALANCE_WEIGHT,
            one_hot_weight=ONE_HOT_WEIGHT,
            lambda_div=LAMBDA_DIV
        )

        role_optimization_mse_hist.append(avg_mse)
        role_train_one_hot_hist.append(avg_one_hot)
        role_train_l2_hist.append(avg_l2)
        role_pairwise_div_hist.append(avg_pairwise_div)
        role_filler_independence_hist.append(avg_filler_independence)
        role_token_entropy_hist.append(avg_token_entropy)
        role_balance_hist.append(avg_role_balance)

        # Recompute TRAIN MSE in eval mode after EVERY epoch.
        # This is the blue curve and is directly comparable to validation MSE.
        epoch_1idx = epoch_0idx + 1
        train_eval_mse, train_r2_this_epoch = evaluate_role(
            role_model,
            train_set,
            temperature=temperature,
            mu=enc_mu,
            sigma=enc_sigma,
        )
        role_train_mse_hist.append(train_eval_mse)

        # Validation and the other expensive diagnostics remain every EVAL_INTERVAL epochs.
        do_eval = (
            epoch_1idx == (start_epoch_0idx + 1)
            or epoch_1idx == total_role_epochs
            or epoch_1idx % EVAL_INTERVAL == 0
        )

        if do_eval:
            train_r2 = train_r2_this_epoch

            val_mse, val_r2 = evaluate_role(
                role_model,
                test_set,
                temperature=temperature,
                mu=enc_mu,
                sigma=enc_sigma,
            )

            val_filler_independence = evaluate_filler_independence(
                role_model,
                test_set,
                batch_size=INDEPENDENCE_EVAL_BATCH_SIZE,
                temperature=temperature,
            )

            val_filler_mi = compute_filler_role_mutual_information(
                role_model,
                test_set,
                device,
                temperature=temperature,
            )

            utilization_diag = evaluate_role_utilization_diagnostics(
                role_model,
                test_set,
                batch_size=INDEPENDENCE_EVAL_BATCH_SIZE,
                temperature=temperature,
            )

            role_val_mse_hist.append(val_mse)
            role_train_r2_hist.append(train_r2)
            role_val_r2_hist.append(val_r2)
            role_filler_mi_hist.append(val_filler_mi)
            role_eval_epochs.append(epoch_1idx)
            role_dsa_hist.append(utilization_diag["dsa"])
            role_argmax_entropy_hist.append(utilization_diag["argmax_entropy"])
            role_active_argmax_roles_hist.append(utilization_diag["active_argmax_roles"])
            role_soft_utilization_hist.append(utilization_diag["soft_utilization"].tolist())
            role_argmax_utilization_hist.append(utilization_diag["argmax_utilization"].tolist())

            print(f"Validation filler-role MI = {val_filler_mi:.4f} bits")
            print(
                f"Validation role utilization: Dsa={utilization_diag['dsa']:.6f} | "
                f"argmax entropy={utilization_diag['argmax_entropy']:.6f} nats | "
                f"active argmax roles={utilization_diag['active_argmax_roles']}/{n_roles}"
            )

            combined_val_score = (
                    float(val_mse)
                    + EARLY_STOP_INDEPENDENCE_WEIGHT
                    * float(val_filler_independence)
            )

            improvement = best_score - combined_val_score

            if improvement > EARLY_STOPPING_MIN_DELTA:
                best_score = combined_val_score
                best_role_epoch_1idx = epoch_1idx
                best_role_mse = float(val_mse)
                best_role_filler_independence = float(
                    val_filler_independence
                )

                early_stop_counter = 0

                torch.save(
                    {
                        "model_state_dict": role_model.state_dict(),
                        "epoch": epoch_1idx,
                        "val_mse": best_role_mse,
                        "val_filler_independence": (
                            best_role_filler_independence
                        ),
                        "combined_val_score": best_score,
                        "temperature": temperature,
                    },
                    best_role_path,
                )

                print(
                    f"  ✓ Saved best ROLE model: {best_role_path}\n"
                    f"    epoch={best_role_epoch_1idx} | "
                    f"val_mse={best_role_mse:.6f} | "
                    f"val_filler_indep="
                    f"{best_role_filler_independence:.6f} | "
                    f"combined_score={best_score:.6f}"
                )

            else:
                if epoch_1idx >= EARLY_STOPPING_WARMUP:
                    early_stop_counter += 1

                print(
                    f"  No combined-score improvement: "
                    f"{early_stop_counter}/"
                    f"{EARLY_STOPPING_PATIENCE}"
                )

            print(
                f"Epoch {epoch_1idx}/{total_role_epochs} "
                f"(T={temperature:.3f}): "
                f"Optimization MSE={avg_mse:.6f}, "
                f"Train Eval MSE={train_eval_mse:.6f}, "
                f"Val MSE={val_mse:.6f}, "
                f"Train R²={train_r2:.6f}, "
                f"Val R²={val_r2:.6f}, "
                f"Pairwise diversity={avg_pairwise_div:.6f}, "
                f"Role balance={avg_role_balance:.6f}, "
                f"Train filler independence="
                f"{avg_filler_independence:.6f}, "
                f"Val filler independence="
                f"{val_filler_independence:.6f}, "
                f"Combined score={combined_val_score:.6f}"
            )

            if (
                    epoch_1idx >= EARLY_STOPPING_WARMUP
                    and early_stop_counter >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "\nEarly stopping triggered:\n"
                    f"  Current epoch: {epoch_1idx}\n"
                    f"  Best epoch: {best_role_epoch_1idx}\n"
                    f"  Best validation MSE: "
                    f"{best_role_mse:.6f}\n"
                    f"  Best validation filler independence: "
                    f"{best_role_filler_independence:.6f}\n"
                    f"  Best combined score: {best_score:.6f}"
                )

                stopped_early = True
        else:
            print(
                f"Epoch {epoch_1idx}/{total_role_epochs} (T={temperature:.3f}): "
                f"Optimization MSE={avg_mse:.6f}, "
                f"Train Eval MSE={train_eval_mse:.6f}, "
                f"(skipping validation; interval={EVAL_INTERVAL}) "
                f"Pairwise diversity={avg_pairwise_div:.6f}, "
                f"Role balance={avg_role_balance:.6f}"
            )

        if PLOT_ROLE_DISTS and (epoch_1idx == 1 or epoch_1idx == total_role_epochs or epoch_1idx % ROLE_PLOT_EVERY == 0):
            summaries = snapshot_role_distributions(
                role_model, test_set, device,
                out_dir=ROLE_OUT_DIR,
                epoch_idx=epoch_0idx,
                max_items=ROLE_SAMPLE_SIZE,
                temperature=temperature,
                plot_role_bboxes=PLOT_ROLE_BBOXES,
                bbox_lookup=bbox_lookup,
                label_to_filler_id=label_to_filler_id,
            )
            role_entropy_hist.append(summaries["entropy"])
            role_sparsity_hist.append(summaries["sparsity"])
            role_snapshot_epochs.append(epoch_1idx)

        # ---------------------------------------------------------
        # LIVE TRAINING PLOTS
        # ---------------------------------------------------------
        # Re-generate the plot files after every epoch. The same
        # filenames are overwritten, so opening/refreshing them during
        # training shows the most recent histories available.
        plot_training_curves(
            role_train_mse_hist,
            role_val_mse_hist,
            role_train_r2_hist,
            role_val_r2_hist,
            role_train_one_hot=role_train_one_hot_hist,
            role_train_l2=role_train_l2_hist,
            role_train_filler_independence=role_filler_independence_hist,
            role_filler_mi=role_filler_mi_hist,
            role_eval_epochs=role_eval_epochs,
            role_train_pairwise_diversity=role_pairwise_div_hist,
        )

        plot_role_utilization_diagnostics(
            role_eval_epochs,
            role_dsa_hist,
            role_argmax_entropy_hist,
            role_active_argmax_roles_hist,
            n_roles,
        )

        # Save raw histories every epoch so any plot can be regenerated later.
        torch.save(
            {
                "optimization_mse": role_optimization_mse_hist,
                "train_eval_mse_every_epoch": role_train_mse_hist,
                "val_mse": role_val_mse_hist,
                "train_r2": role_train_r2_hist,
                "val_r2": role_val_r2_hist,
                "pairwise_diversity": role_pairwise_div_hist,
                "filler_independence": role_filler_independence_hist,
                "token_entropy": role_token_entropy_hist,
                "role_balance": role_balance_hist,
                "filler_mi": role_filler_mi_hist,
                "eval_epochs": role_eval_epochs,
                "dsa": role_dsa_hist,
                "argmax_entropy": role_argmax_entropy_hist,
                "active_argmax_roles": role_active_argmax_roles_hist,
                "soft_utilization": role_soft_utilization_hist,
                "argmax_utilization": role_argmax_utilization_hist,
            },
            "role_training_history.pt",
        )

        # Save the latest training state every epoch for interruption recovery.
        # Unlike best_role_path, this always represents the most recently
        # completed epoch and includes optimizer + histories needed to resume.
        torch.save(
            {
                "epoch": epoch_1idx,
                "model_state_dict": role_model.state_dict(),
                "optimizer_state_dict": role_optimizer.state_dict(),
                "temperature": temperature,
                "best_score": best_score,
                "best_role_epoch_1idx": best_role_epoch_1idx,
                "best_role_mse": best_role_mse,
                "best_role_filler_independence": best_role_filler_independence,
                "early_stop_counter": early_stop_counter,
                "histories": {
                    "optimization_mse": role_optimization_mse_hist,
                    "train_eval_mse_every_epoch": role_train_mse_hist,
                    "val_mse": role_val_mse_hist,
                    "train_r2": role_train_r2_hist,
                    "val_r2": role_val_r2_hist,
                    "one_hot": role_train_one_hot_hist,
                    "l2": role_train_l2_hist,
                    "pairwise_diversity": role_pairwise_div_hist,
                    "filler_independence": role_filler_independence_hist,
                    "token_entropy": role_token_entropy_hist,
                    "role_balance": role_balance_hist,
                    "filler_mi": role_filler_mi_hist,
                    "eval_epochs": role_eval_epochs,
                    "dsa": role_dsa_hist,
                    "argmax_entropy": role_argmax_entropy_hist,
                    "active_argmax_roles": role_active_argmax_roles_hist,
                    "soft_utilization": role_soft_utilization_hist,
                    "argmax_utilization": role_argmax_utilization_hist,
                    "role_entropy": role_entropy_hist,
                    "role_sparsity": role_sparsity_hist,
                    "role_snapshot_epochs": role_snapshot_epochs,
                },
            },
            latest_role_path,
        )

        # Update role entropy/sparsity summary plots whenever a new
        # role-distribution snapshot was created.
        if (
            PLOT_ROLE_DISTS
            and role_entropy_hist
            and (
                epoch_1idx == 1
                or epoch_1idx == total_role_epochs
                or epoch_1idx % ROLE_PLOT_EVERY == 0
            )
        ):
            plot_role_summaries(
                role_entropy_hist,
                role_sparsity_hist,
                role_epochs=role_snapshot_epochs,
                out_dir=ROLE_OUT_DIR,
            )

        if stopped_early:
            break

    # Load best ROLE model before decoder training / substitution eval
    try:
        checkpoint = torch.load(
            best_role_path,
            map_location=device,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            role_model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            # Compatibility with older state-dict-only checkpoints
            role_model.load_state_dict(checkpoint)
        print(f"\nLoaded best ROLE model from {best_role_path}")
        if best_role_epoch_1idx is not None:
            print(f"  Best epoch: {best_role_epoch_1idx}")
            print(f"  Best val MSE: {best_role_mse:.6f}")
            print(
                f"  Best val filler independence: "
                f"{best_role_filler_independence:.6f}"
            )
            print(
                f"  Best combined score: "
                f"{best_score:.6f}"
            )
    except FileNotFoundError:
        print(f"\nWARNING: Best ROLE checkpoint not found at {best_role_path}. Using current in-memory model.")

    # Training curves and role-summary plots were updated throughout training.
    if PLOT_ROLE_DISTS and role_entropy_hist:
        print(f"Saved role assignment distribution plots to: {ROLE_OUT_DIR}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nFinal Results:")
    print("  ROLE Model:")

    if best_role_epoch_1idx is not None:
        print(f"    Best epoch: {best_role_epoch_1idx}")
        print(f"    Best val MSE: {best_role_mse:.6f}")
        print(
            f"    Best val filler independence: "
            f"{best_role_filler_independence:.6f}"
        )
        print(
            f"    Best combined score: "
            f"{best_score:.6f}"
        )
    else:
        print("    No checkpoint was saved.")

    print(f"\nRole offset applied: {role_offset}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED!")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 80)
        sys.exit(1)
