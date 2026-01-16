"""
ROLE approximation TPDN decoder evaluation with Gumbel-Softmax
- Trains ROLE to match TPDN encodings (tensor product representations)
- Uses final projection layer of TPDN decoder to reconstruct original number sequences from ROLE-approximated TPDN encodings
- Evaluates substitution accuracy by feeding ROLE's encodings to the TPDN decoder
- Includes eval-only temperature annealing + CSV/PNG outputs
"""

import sys
import json
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from role_learning_tensor_product_encoder import RoleLearningTensorProductEncoder
from role_assigner import RoleAssignmentTransformer


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


class RoleLearningInstance(object):
    def __init__(self, filler, target_encoding, target_output=None, target_roles=None):
        self.filler = filler  # LongTensor [1, seq_len]
        self.target_encoding = target_encoding  # FloatTensor [1, encoding_dim] - TPDN encoding
        self.target_output = target_output  # LongTensor [1, seq_len] - original sequence
        self.target_roles = target_roles  # LongTensor [1, seq_len]


# --------------------
# Simplified TPDN Decoder
# --------------------

class SimplifiedTPDNDecoder(nn.Module):
    """
    Simplified TPDN decoder that just applies the final linear projection
    to reconstruct sequences from TPDN encodings.

    This matches the architecture of the original TPDN encoder's final layer:
    encoding (role_dim * filler_dim) -> sequence (seq_len)
    """

    def __init__(self, encoding_dim, seq_len, num_fillers):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.seq_len = seq_len
        self.num_fillers = num_fillers

        # Single linear layer to project from encoding to sequence logits
        # Output: (seq_len, num_fillers) for each position's token probabilities
        self.projection = nn.Linear(encoding_dim, seq_len * num_fillers)

    def forward(self, encoding):
        """
        Args:
            encoding: (batch, encoding_dim) - flat TPDN encoding

        Returns:
            logits: (batch, seq_len, num_fillers)
        """
        batch_size = encoding.size(0)

        # Project encoding to logits
        logits = self.projection(encoding)  # (B, seq_len * num_fillers)

        # Reshape to (B, seq_len, num_fillers)
        logits = logits.view(batch_size, self.seq_len, self.num_fillers)

        return logits


# --------------------
# Data loading
# --------------------

def load_tpdn_encodings(filename, device):
    """Load TPDN encodings from the JSON file created by the modified TPDN script."""
    print(f"Loading TPDN encodings from {filename}...")
    try:
        with open(filename, 'r') as f:
            outputs = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File {filename} not found.")
        print("Please run the modified tpdn_test4role.py script first to generate TPDN encodings.")
        sys.exit(1)
    if not outputs:
        print("ERROR: No data in file.")
        sys.exit(1)

    print(f"Loaded {len(outputs)} instances")
    data = []

    for entry in outputs:
        filler_ids = entry['filler_ids']
        tpdn_encoding = entry['tpdn_encoding']  # The actual tensor product encoding
        target_sequence = entry['target_sequence']  # Original sequence values (for reference)
        reconstructed_sequence = entry['reconstructed_sequence']  # TPDN's reconstruction
        role_ids = entry.get('role_ids', None)

        # Convert to tensors
        filler_t = Variable(torch.LongTensor(filler_ids), requires_grad=False).unsqueeze(0)
        encoding_t = Variable(torch.FloatTensor(tpdn_encoding), requires_grad=False).unsqueeze(0)

        # Use filler_ids as target output (the discrete tokens to reconstruct)
        target_seq_t = Variable(torch.LongTensor(filler_ids), requires_grad=False).unsqueeze(0)

        role_t = Variable(torch.LongTensor(role_ids), requires_grad=False).unsqueeze(
            0) if role_ids is not None else None

        # Move to device
        filler_t = filler_t.to(device)
        encoding_t = encoding_t.to(device)
        target_seq_t = target_seq_t.to(device)
        if role_t is not None:
            role_t = role_t.to(device)

        data.append(RoleLearningInstance(filler_t, encoding_t, target_seq_t, role_t))

    print(f"Created {len(data)} role learning instances")
    if data:
        print(f"  Sequence length: {len(outputs[0]['filler_ids'])}")
        print(f"  TPDN encoding dimension: {len(outputs[0]['tpdn_encoding'])}")
        print(f"  Role scheme: {outputs[0]['role_scheme']}")
    return data


# --------------------
# Gumbel-Softmax for ROLE
# --------------------

@torch.no_grad()
def encode_with_gumbel(role_model, filler, temperature=1.0, hard=False):
    role_model.eval()

    # Get LOGITS from model
    _, role_logits = role_model(filler, None)  # (S, B, R)

    # Sample with Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(role_logits) + 1e-20) + 1e-20)
    y = (role_logits + gumbel) / max(temperature, 1e-8)
    y_soft = F.softmax(y, dim=-1)

    if hard:
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        role_weights = y_hard - y_soft.detach() + y_soft
    else:
        role_weights = y_soft

    # Get role embeddings and bind
    role_emb_weight = role_model.role_assigner.role_embedding.weight
    roles_embedded = torch.einsum('sbr,rd->sbd', role_weights, role_emb_weight)

    # Get filler embeddings
    fillers_emb = role_model.filler_embedding(filler)
    if role_model.embed_squeeze:
        fillers_emb = role_model.embedding_squeeze_layer(fillers_emb)
    fillers_emb = fillers_emb.transpose(0, 1)

    # Bind using model's binding layer
    bound = role_model.sum_layer(fillers_emb, roles_embedded)

    # DON'T apply final projection - we want the raw TPDN encoding
    # The final projection is what the decoder will learn

    if bound.dim() == 3 and bound.size(0) == 1:
        bound = bound.squeeze(0)
    elif bound.dim() == 2 and bound.size(0) == 1:
        bound = bound.squeeze(0).view(1, -1)

    return bound


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
        mu=None, sigma=None
):
    """Train ROLE for one epoch to match TPDN encodings."""
    model.train()
    import random
    random.shuffle(train_batches)

    total_loss = 0.0
    total_mse = 0.0
    total_one_hot = 0.0
    total_l2 = 0.0
    total_unique = 0.0
    num_updates = 0

    for batch in tqdm(train_batches, desc="Training ROLE"):
        fillers = []
        targets = []

        for inst in batch:
            fillers.append(inst.filler)
            targets.append(inst.target_encoding)

        filler_batch = torch.cat(fillers, dim=0)
        target_batch = torch.cat(targets, dim=0)

        optimizer.zero_grad()

        # Forward - get TPDN encoding from ROLE (without final projection)
        pred, role_predictions = model(filler_batch, None)

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

        # Optional regularization
        if use_regularization:
            one_hot_loss, l2_loss, unique_loss = model.get_regularization_loss(role_predictions)
            reg_loss = one_hot_loss + l2_loss + unique_loss
            loss = loss + reg_loss

            total_one_hot += float(one_hot_loss.item())
            total_l2 += float(l2_loss.item())
            total_unique += float(unique_loss.item())

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
        total_unique / denom,
        total_l2 / denom,
    )


@torch.no_grad()
def evaluate_role(role_model, data, mu=None, sigma=None):
    """Evaluate ROLE MSE on dataset."""
    role_model.eval()
    total_mse = 0.0

    for inst in data:
        pred, _ = role_model(inst.filler, None)

        pred = pred.view(pred.size(0), -1)
        targ = inst.target_encoding.view(inst.target_encoding.size(0), -1)

        if mu is not None and sigma is not None:
            pred = _standardize(pred, mu, sigma)
            targ = _standardize(targ, mu, sigma)

        mse = torch.mean((pred - targ) ** 2).item()
        total_mse += mse

    return total_mse / max(1, len(data))


@torch.no_grad()
def evaluate_role_accuracy(role_model, data, n_roles=None):
    """Evaluate role prediction accuracy."""
    role_model.eval()

    all_pred = []
    all_true = []

    for inst in data:
        if inst.target_roles is None:
            continue

        _, role_logits = role_model(inst.filler, None)

        pred_roles = None
        if hasattr(role_model, "role_assigner") and hasattr(role_model.role_assigner, "last_role_probs"):
            probs = role_model.role_assigner.last_role_probs
            if probs is not None and probs.dim() == 3:
                pred_roles = probs[0].argmax(dim=-1)

        if pred_roles is None:
            pred_roles = role_logits.squeeze(1).argmax(dim=-1)
        true_roles = inst.target_roles.squeeze(0)

        all_pred.append(pred_roles.detach().cpu())
        all_true.append(true_roles.detach().cpu())

    if not all_true:
        return 0.0, 0.0

    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)

    raw_acc = float((pred == true).float().mean().item())

    # Aligned accuracy
    pred_np = pred.view(-1).numpy()
    true_np = true.view(-1).numpy()

    if n_roles is None:
        n_roles = int(max(pred.max().item(), true.max().item())) + 1

    S = all_true[0].numel()
    N = pred.numel() // S
    pred_np = pred_np.reshape(N, S)
    true_np = true_np.reshape(N, S)

    aligned_acc = compute_role_acc_aligned(pred_np, true_np, n_roles=n_roles)
    return raw_acc, aligned_acc


# --------------------
# TPDN Decoder Training
# --------------------

def train_epoch_decoder(decoder, train_batches, optimizer, criterion, mu=None, sigma=None):
    """Train simplified TPDN decoder for one epoch."""
    decoder.train()
    import random
    random.shuffle(train_batches)

    total_loss = 0.0
    num_updates = 0

    for batch in tqdm(train_batches, desc="Training Decoder"):
        encodings = []
        targets = []

        for inst in batch:
            encodings.append(inst.target_encoding.squeeze(0))
            targets.append(inst.target_output.squeeze(0))

        enc_batch = torch.stack(encodings)  # (B, encoding_dim)
        tgt_batch = torch.stack(targets)  # (B, seq_len)

        # Clamp target tokens to valid range
        tgt_batch = tgt_batch.clamp(0, decoder.num_fillers - 1)

        # Flatten and optionally standardize encodings
        enc_batch = enc_batch.view(enc_batch.size(0), -1)
        if mu is not None and sigma is not None:
            enc_batch = _standardize(enc_batch, mu, sigma)

        optimizer.zero_grad()

        # Forward
        logits = decoder(enc_batch)  # (B, S, V)

        # Compute loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_batch.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_updates += 1

    return total_loss / max(1, num_updates)


@torch.no_grad()
def evaluate_decoder(decoder, data, mu=None, sigma=None):
    """Evaluate TPDN decoder accuracy."""
    decoder.eval()
    correct = 0
    total = 0

    for inst in data:
        enc = inst.target_encoding.squeeze(0).unsqueeze(0)  # (1, encoding_dim)
        enc = enc.view(enc.size(0), -1)

        if mu is not None and sigma is not None:
            enc = _standardize(enc, mu, sigma)

        targets = inst.target_output.squeeze(0).clamp(0, decoder.num_fillers - 1)

        logits = decoder(enc)  # (1, S, V)
        preds = logits.argmax(dim=-1).squeeze(0)  # (S,)

        correct += (preds == targets).sum().item()
        total += targets.numel()

    return correct / max(1, total)


# --------------------
# Substitution Evaluation
# --------------------

@torch.no_grad()
def evaluate_substitution_accuracy(role_model, decoder, data,
                                   gumbel_temp=1.0, use_hard_gumbel=False,
                                   mu=None, sigma=None):
    """Evaluate how well ROLE encodings decode using TPDN decoder."""
    role_model.eval()
    decoder.eval()

    correct = 0
    total = 0

    for inst in tqdm(data, desc=f"Eval (T={gumbel_temp:.3f}, hard={use_hard_gumbel})"):
        # Get ROLE encoding (raw, without final projection)
        role_enc = encode_with_gumbel(role_model, inst.filler,
                                      temperature=gumbel_temp, hard=use_hard_gumbel)
        role_enc = role_enc.view(role_enc.size(0), -1)

        if mu is not None and sigma is not None:
            role_enc = _standardize(role_enc, mu, sigma)

        # Decode with simplified TPDN decoder
        logits = decoder(role_enc)  # (1, S, V)
        preds = logits.argmax(dim=-1).squeeze(0)  # (S,)
        targets = inst.target_output.squeeze(0)  # (S,)

        correct += (preds == targets).sum().item()
        total += targets.numel()

    return correct / max(1, total)


@torch.no_grad()
def evaluate_across_temps(role_model, decoder, data, temps=None, T_init=1.0, T_min=0.1, decay=0.95,
                          csv_path="sub_acc_vs_T.csv", png_path="sub_acc_vs_T.png",
                          mu=None, sigma=None):
    """Evaluate substitution accuracy across temperature range."""
    if temps is None:
        temps = []
        T = float(T_init)
        while True:
            temps.append(T)
            if T <= T_min + 1e-12:
                break
            T_next = max(T_min, T * decay) if decay < 1.0 else T_min
            if abs(T_next - T) < 1e-12:
                break
            T = T_next

    rows = []
    results = {"soft": [], "hard": []}

    for mode in ["soft", "hard"]:
        use_hard = (mode == "hard")
        accs = []
        for T in temps:
            acc = evaluate_substitution_accuracy(role_model, decoder, data,
                                                 gumbel_temp=T, use_hard_gumbel=use_hard,
                                                 mu=mu, sigma=sigma)
            accs.append(acc)
            rows.append([T, acc, mode])
            print(f" -> {mode.upper():>4} | T={T:.4f} | substitution_acc={acc:.4f}")
        results[mode] = accs

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "substitution_accuracy", "mode"])
        w.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    # Plot
    plt.figure()
    plt.plot(temps, results["soft"], marker="o", label="SOFT")
    plt.plot(temps, results["hard"], marker="s", label="HARD (ST)")
    plt.gca().set_xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Gumbel temperature T (log scale)")
    plt.ylabel("Token-level accuracy")
    plt.title("TPDN Decoder: ROLE substitution accuracy vs. temperature")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

    return temps, results


# --------------------
# Plotting
# --------------------

def plot_training_curves(role_train_mse, role_val_mse, role_val_acc,
                         dec_train_loss, dec_val_acc,
                         role_train_one_hot=None, role_train_unique=None, role_train_l2=None):
    """Plot training curves."""
    epochs_role = np.arange(1, len(role_train_mse) + 1)
    epochs_dec = np.arange(1, len(dec_train_loss) + 1)

    # ROLE MSE
    plt.figure()
    plt.plot(epochs_role, role_train_mse, marker="o", label="Train MSE")
    plt.plot(epochs_role, role_val_mse, marker="s", label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("ROLE Model Reconstruction MSE (TPDN Encodings)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("role_model_mse_encodings.png", dpi=200)
    plt.close()

    # ROLE Accuracy
    plt.figure()
    plt.plot(epochs_role, role_val_acc, marker="o", label="Val Role Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ROLE Model Validation Role Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("role_model_acc_encodings.png", dpi=200)
    plt.close()

    # ROLE Regularization losses
    # ROLE Regularization losses
    if role_train_one_hot is not None and role_train_unique is not None and role_train_l2 is not None:
        plt.figure()

        eps = 1e-12  # avoids log/scale issues at exactly 0
        one_hot = np.array(role_train_one_hot, dtype=float)
        unique = np.array(role_train_unique, dtype=float)
        l2 = np.array(role_train_l2, dtype=float)

        plt.plot(epochs_role, one_hot + eps, marker="o", label="One-hot reg")
        plt.plot(epochs_role, unique + eps, marker="s", label="Unique-role reg")
        plt.plot(epochs_role, l2 + eps, marker="^", label="L2 reg")

        # Key change: make small values visible even with big spikes
        plt.yscale("symlog", linthresh=1e-6)

        plt.xlabel("Epoch")
        plt.ylabel("Loss (symlog)")
        plt.title("ROLE Model – Regularization losses (train)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("role_model_reg_losses_encodings.png", dpi=200)
        plt.close()

    # Decoder Loss
    plt.figure()
    plt.plot(epochs_dec, dec_train_loss, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Simplified TPDN Decoder Training Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("decoder_loss_encodings.png", dpi=200)
    plt.close()

    # Decoder Accuracy
    plt.figure()
    plt.plot(epochs_dec, dec_val_acc, marker="o", label="Val Token Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Simplified TPDN Decoder Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("decoder_acc_encodings.png", dpi=200)
    plt.close()


def _hungarian_best_map(conf_mat: np.ndarray) -> np.ndarray:
    """conf_mat[pred, true] = count. Returns mapping array map_pred_to_true[pred] = true"""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-conf_mat)
        mapping = np.full(conf_mat.shape[0], -1, dtype=int)
        mapping[row_ind] = col_ind
        return mapping
    except Exception:
        mapping = np.full(conf_mat.shape[0], -1, dtype=int)
        used_true = set()
        for pred in np.argsort(-conf_mat.max(axis=1)):
            true = int(np.argmax(conf_mat[pred]))
            while true in used_true:
                conf_mat[pred, true] = -1
                true = int(np.argmax(conf_mat[pred]))
            mapping[pred] = true
            used_true.add(true)
        return mapping


def compute_role_acc_aligned(pred_roles: np.ndarray, true_roles: np.ndarray, n_roles: int) -> float:
    """pred_roles, true_roles: shape (N, S) integer role ids"""
    conf = np.zeros((n_roles, n_roles), dtype=np.int64)
    for p, t in zip(pred_roles.reshape(-1), true_roles.reshape(-1)):
        conf[int(p), int(t)] += 1
    mapping = _hungarian_best_map(conf.copy())
    mapped_pred = mapping[pred_roles]
    return float((mapped_pred == true_roles).mean())


# --------------------
# Main
# --------------------

def main():
    print("=" * 80)
    print("ROLE APPROXIMATION WITH TPDN ENCODINGS")
    print("=" * 80)

    # =========================
    # Config
    # =========================
    TPDN_ENCODINGS_JSON = "tpdn_encodings_test_l2r.json"
    SEED = 42

    # ROLE training
    EPOCHS_ROLE = 100
    BATCH_SIZE = 16
    ROLE_LR = 1e-3
    USE_ROLE_REG = True
    STANDARDIZE = True  # Whether to standardize encodings

    # Decoder training
    EPOCHS_DEC = 50
    DECODER_LR = 1e-3

    # Model architecture
    FILLER_DIM = 16
    ROLE_DIM = 8
    HIDDEN_DIM = 768
    NUM_LAYERS = 6
    N_HEAD = 12
    DROPOUT = 0.0

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
    data = load_tpdn_encodings(TPDN_ENCODINGS_JSON, device)

    seq_len = data[0].filler.size(1)
    encoding_dim = data[0].target_encoding.view(-1).numel()
    num_fillers = max(int(inst.filler.max().item()) for inst in data) + 1

    if data[0].target_roles is not None:
        n_roles = int(max(inst.target_roles.max().item() for inst in data)) + 1
    else:
        n_roles = seq_len

    print(f"\nDataset info:")
    print(f"  Sequences: {len(data)}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num fillers: {num_fillers}")
    print(f"  Num roles: {n_roles}")
    print(f"  TPDN encoding dimension: {encoding_dim}")
    print(f"  Expected encoding dim (role_dim * filler_dim): {ROLE_DIM * FILLER_DIM}")

    # Compute statistics for standardization
    enc_mu, enc_sigma = compute_target_stats(data) if STANDARDIZE else (None, None)
    if STANDARDIZE:
        print(f"  Encoding mean: {enc_mu.mean().item():.4f}")
        print(f"  Encoding std: {enc_sigma.mean().item():.4f}")

    train_set = data[: int(0.8 * len(data))]
    test_set = data[int(0.8 * len(data)):]
    print(f"\nTrain: {len(train_set)}, Test: {len(test_set)}")

    train_batches = batchify_with_roles(train_set, BATCH_SIZE)

    # =========================
    # Create ROLE model
    # =========================
    print("\n" + "=" * 80)
    print("CREATING ROLE MODEL")
    print("=" * 80)

    # ROLE model outputs raw TPDN encodings (no final projection)
    role_model = RoleLearningTensorProductEncoder(
        n_roles=n_roles,
        n_fillers=num_fillers,
        filler_dim=FILLER_DIM,
        role_dim=ROLE_DIM,
        final_layer_width=None,  # No projection - output raw TPR encoding
        binder="tpr",
        role_learner_hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        nhead=N_HEAD,
        dim_feedforward=4 * HIDDEN_DIM,
        dropout=DROPOUT,
        softmax_roles=True,
    ).to(device)
    role_model.use_regularization(USE_ROLE_REG)

    print(f"ROLE model architecture:")
    print(f"  Binder output dimension: {role_model.binder_output_dim}")
    print(f"  Expected to match encoding_dim: {encoding_dim}")

    # =========================
    # Train ROLE
    # =========================
    print("\n" + "=" * 80)
    print("TRAINING ROLE MODEL")
    print("=" * 80)

    role_optimizer = optim.Adam(role_model.parameters(), lr=ROLE_LR)
    role_criterion = nn.MSELoss()

    role_train_mse_hist = []
    role_train_one_hot_hist = []
    role_train_unique_hist = []
    role_train_l2_hist = []
    role_val_mse_hist = []
    role_val_acc_hist = []
    role_val_acc_aligned_hist = []
    best_role_acc = 0.0

    for epoch in range(EPOCHS_ROLE):
        _, avg_mse, avg_one_hot, avg_unique, avg_l2 = train_epoch_role(
            role_model,
            train_batches,
            role_optimizer,
            role_criterion,
            use_regularization=USE_ROLE_REG,
            mu=enc_mu,
            sigma=enc_sigma
        )

        val_mse = evaluate_role(role_model, test_set, mu=enc_mu, sigma=enc_sigma)
        val_acc_raw, val_acc_aligned = evaluate_role_accuracy(role_model, test_set, n_roles=n_roles)

        role_train_mse_hist.append(avg_mse)
        role_train_one_hot_hist.append(avg_one_hot)
        role_train_unique_hist.append(avg_unique)
        role_train_l2_hist.append(avg_l2)
        role_val_mse_hist.append(val_mse)
        role_val_acc_hist.append(val_acc_raw)
        role_val_acc_aligned_hist.append(val_acc_aligned)

        print(
            f"Epoch {epoch + 1}/{EPOCHS_ROLE}: "
            f"Train MSE={avg_mse:.6f}, Val MSE={val_mse:.6f}, "
            f"Val Role Acc(raw)={val_acc_raw:.4f}, Val Role Acc(aligned)={val_acc_aligned:.4f}"
        )

        if val_acc_aligned > best_role_acc:
            best_role_acc = val_acc_aligned
            torch.save(role_model.state_dict(), "best_role_model_encodings.pt")
            print(" Saved best ROLE model")

    # Reload best ROLE model
    role_model.load_state_dict(torch.load("best_role_model_encodings.pt", map_location=device))

    # =========================
    # Create + Train Simplified TPDN Decoder
    # =========================
    print("\n" + "=" * 80)
    print("CREATING SIMPLIFIED TPDN DECODER")
    print("=" * 80)

    decoder = SimplifiedTPDNDecoder(
        encoding_dim=encoding_dim,
        seq_len=seq_len,
        num_fillers=num_fillers,
    ).to(device)

    print(f"Decoder architecture:")
    print(f"  Input: encoding_dim = {encoding_dim}")
    print(f"  Output: seq_len — num_fillers = {seq_len}— {num_fillers}")

    print("\n" + "=" * 80)
    print("TRAINING SIMPLIFIED TPDN DECODER")
    print("=" * 80)

    dec_optimizer = optim.Adam(decoder.parameters(), lr=DECODER_LR)
    dec_criterion = nn.CrossEntropyLoss()

    dec_train_loss_hist = []
    dec_val_acc_hist = []
    best_dec_acc = 0.0

    for epoch in range(EPOCHS_DEC):
        train_loss = train_epoch_decoder(
            decoder, train_batches, dec_optimizer, dec_criterion, mu=enc_mu, sigma=enc_sigma
        )
        val_acc = evaluate_decoder(decoder, test_set, mu=enc_mu, sigma=enc_sigma)

        dec_train_loss_hist.append(train_loss)
        dec_val_acc_hist.append(val_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS_DEC}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_dec_acc:
            best_dec_acc = val_acc
            torch.save(decoder.state_dict(), "best_tpdn_decoder_encodings.pt")
            print("  â†’ Saved best decoder")

    decoder.load_state_dict(torch.load("best_tpdn_decoder_encodings.pt", map_location=device))

    # Plot training curves
    plot_training_curves(
        role_train_mse_hist,
        role_val_mse_hist,
        role_val_acc_aligned_hist,
        dec_train_loss_hist,
        dec_val_acc_hist,
        role_train_one_hot=role_train_one_hot_hist,
        role_train_unique=role_train_unique_hist,
        role_train_l2=role_train_l2_hist,
    )

    # =========================
    # Substitution accuracy + temp sweep
    # =========================
    print("\n" + "=" * 80)
    print("SUBSTITUTION ACCURACY EVALUATION")
    print("=" * 80)

    acc_soft = evaluate_substitution_accuracy(
        role_model,
        decoder,
        test_set,
        gumbel_temp=1.0,
        use_hard_gumbel=False,
        mu=enc_mu,
        sigma=enc_sigma,
    )
    acc_hard = evaluate_substitution_accuracy(
        role_model,
        decoder,
        test_set,
        gumbel_temp=1.0,
        use_hard_gumbel=True,
        mu=enc_mu,
        sigma=enc_sigma,
    )

    print(f"\nSubstitution accuracy at T=1.0:")
    print(f"  Soft Gumbel: {acc_soft:.4f}")
    print(f"  Hard Gumbel: {acc_hard:.4f}")

    print("\n" + "=" * 80)
    print("TEMPERATURE SWEEP")
    print("=" * 80)

    evaluate_across_temps(
        role_model,
        decoder,
        test_set,
        T_init=1.0,
        T_min=0.1,
        decay=0.95,
        csv_path="sub_acc_vs_T_encodings.csv",
        png_path="sub_acc_vs_T_encodings.png",
        mu=enc_mu,
        sigma=enc_sigma,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  ROLE Model:")
    print(f"    Best Val MSE: {min(role_val_mse_hist):.6f}")
    print(f"    Best Val Role Acc: {best_role_acc:.4f}")
    print(f"  Simplified TPDN Decoder:")
    print(f"    Best Val Acc: {best_dec_acc:.4f}")
    print(f"  Substitution (T=1.0): soft={acc_soft:.4f} hard={acc_hard:.4f}")


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
