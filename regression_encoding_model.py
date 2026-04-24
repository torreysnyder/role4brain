"""
Trains a regression-based encoding model to map aggregated BERT object
category label embeddings to PCA-reduced in silico fMRI responses.

Pipeline:
  1. Load fMRI training records from tpdn_encodings_fmri.json
     (produced by generate_fmri_encodings.py)
  2. For each image, aggregate its filler BERT embeddings into a single
     fixed-size vector (mean pooling over non-PAD tokens)
  3. Train an MLP regressor to predict the 100-dim PCA-reduced fMRI target
  4. Evaluate on held-out test set and plot training curves
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------
# Config
# --------------------

TPDN_ENCODINGS_JSON     = "ventral_encodings_fmri_300.json"
BERT_EMBEDDING_PATH     = "bert_embedding_matrix.pt"
OUTPUT_MODEL_PATH       = "best_regression_model_fmri.pt"
PLOT_PATH               = "regression_model_mse_fmri_300.png"

SEED                    = 42
TOTAL_EPOCHS            = 50
BATCH_SIZE              = 64
LR                      = 1e-3
WEIGHT_DECAY            = 1e-4
TRAIN_SPLIT             = 0.8
EVAL_INTERVAL           = 5

# Architecture
BERT_DIM                = 768       # raw BERT embedding dimension
HIDDEN_DIMS             = [512, 256, 128]   # MLP hidden layer sizes
TARGET_DIM              = 300       # PCA-reduced fMRI dimensionality
DROPOUT                 = 0.2

PAD_FILLER_ID           = 0         # filler ID used for <PAD> tokens


# --------------------
# Reproducibility
# --------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"CUDA available: {use_cuda} | Device: {device}")
    return device


# --------------------
# Dataset
# --------------------

def compute_target_stats(records):
    """Compute per-dimension mean and std over target encodings.
    Should be called on training records only to avoid data leakage.
    """
    targets = np.array([r["target_encoding"] for r in records], dtype=np.float32)
    mu      = torch.from_numpy(targets.mean(axis=0)).float()
    sigma   = torch.from_numpy(targets.std(axis=0) + 1e-6).float()
    print(f"  Target mean (avg across dims): {mu.mean().item():.4f}")
    print(f"  Target std  (avg across dims): {sigma.mean().item():.4f}")
    return mu, sigma


def standardize(x, mu, sigma):
    """Standardize x to zero mean and unit variance per dimension."""
    return (x - mu.to(x.device)) / sigma.to(x.device)


class FMRIRegressionDataset(Dataset):
    """
    Each item is (aggregated_bert_embedding, standardized_target_fmri_vector).

    Filler IDs are looked up in the pretrained BERT embedding matrix and
    mean-pooled over non-PAD tokens to produce a single BERT_DIM vector.
    Targets are standardized using per-dimension mean and std computed
    from the training set only to avoid data leakage.
    """

    def __init__(self, records, bert_embeddings, mu, sigma, pad_id=PAD_FILLER_ID):
        """
        Args:
            records:          list of dicts with keys 'filler_ids' and 'target_encoding'
            bert_embeddings:  FloatTensor of shape (vocab_size, BERT_DIM)
            mu:               FloatTensor of shape (TARGET_DIM,) — training target mean
            sigma:            FloatTensor of shape (TARGET_DIM,) — training target std
            pad_id:           filler ID used for padding tokens
        """
        self.records         = records
        self.bert_embeddings = bert_embeddings  # (vocab_size, BERT_DIM)
        self.mu              = mu               # (TARGET_DIM,)
        self.sigma           = sigma            # (TARGET_DIM,)
        self.pad_id          = pad_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record     = self.records[idx]
        filler_ids = torch.LongTensor(record["filler_ids"])        # (seq_len,)
        target     = torch.FloatTensor(record["target_encoding"])  # (TARGET_DIM,)

        # Standardize target using training set statistics
        target = standardize(target, self.mu, self.sigma)

        # Mean-pool BERT embeddings over non-PAD tokens
        mask = filler_ids != self.pad_id                           # (seq_len,)
        if mask.sum() == 0:
            # Edge case: all PAD — use zero vector
            aggregated = torch.zeros(self.bert_embeddings.size(1))
        else:
            valid_ids  = filler_ids[mask]                          # (n_real,)
            embeds     = self.bert_embeddings[valid_ids]           # (n_real, BERT_DIM)
            aggregated = embeds.mean(dim=0)                        # (BERT_DIM,)

        return aggregated, target


# --------------------
# Model
# --------------------

class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron that maps a mean-pooled BERT embedding
    (BERT_DIM) to a PCA-reduced fMRI vector (TARGET_DIM).

    Architecture:
        Input (BERT_DIM) -> [Linear -> LayerNorm -> ReLU -> Dropout] x N -> Linear (TARGET_DIM)
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: FloatTensor (batch, input_dim)
        Returns:
            FloatTensor (batch, output_dim)
        """
        return self.net(x)


# --------------------
# Training / Evaluation
# --------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_mse   = 0.0
    n_batches   = 0

    for inputs, targets in tqdm(loader, desc="  Train", leave=False):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds   = model(inputs)
        loss    = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_mse   += loss.item()
        n_batches   += 1

    return total_mse / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_mse   = 0.0
    n_batches   = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds   = model(inputs)
        loss    = criterion(preds, targets)
        total_mse   += loss.item()
        n_batches   += 1

    return total_mse / max(1, n_batches)


# --------------------
# Plotting
# --------------------

def plot_training_curves(train_mse_hist, val_mse_hist, eval_interval, total_epochs):
    epochs_train    = np.arange(1, total_epochs + 1)

    # Val MSE is recorded at eval epochs only — reconstruct x-axis
    eval_epochs = []
    for e in range(1, total_epochs + 1):
        if e == 1 or e == total_epochs or e % eval_interval == 0:
            eval_epochs.append(e)
    eval_epochs = eval_epochs[:len(val_mse_hist)]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, train_mse_hist, marker="o", label="Train MSE")
    plt.plot(eval_epochs,  val_mse_hist,   marker="s", label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Regression Encoding Model — MSE (fMRI Encodings)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()
    print(f"Saved training curve to {PLOT_PATH}")


# --------------------
# Main
# --------------------

def main():
    print("=" * 80)
    print("REGRESSION ENCODING MODEL — BERT -> PCA-REDUCED fMRI")
    print("=" * 80)

    device = set_seed(SEED)

    # -------------------------
    # Load data
    # -------------------------
    print(f"\nLoading training records from {TPDN_ENCODINGS_JSON}...")
    try:
        with open(TPDN_ENCODINGS_JSON, "r") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {TPDN_ENCODINGS_JSON} not found. Run generate_fmri_encodings.py first.")
        sys.exit(1)
    print(f"Loaded {len(records)} records")
    print(f"  Sequence length:           {len(records[0]['filler_ids'])}")
    print(f"  Target encoding dimension: {len(records[0]['target_encoding'])}")

    # -------------------------
    # Load BERT embeddings
    # -------------------------
    print(f"\nLoading BERT embedding matrix from {BERT_EMBEDDING_PATH}...")
    try:
        bert_embeddings = torch.load(BERT_EMBEDDING_PATH, map_location="cpu").float()
        print(f"  Embedding matrix shape: {bert_embeddings.shape}")
    except FileNotFoundError:
        print(f"ERROR: {BERT_EMBEDDING_PATH} not found. Run generate_bert_embeddings.py first.")
        sys.exit(1)

    # -------------------------
    # Train / test split
    # -------------------------
    random.shuffle(records)
    n_train     = int(len(records) * TRAIN_SPLIT)
    train_records = records[:n_train]
    test_records  = records[n_train:]
    print(f"\nTrain: {len(train_records)} | Test: {len(test_records)}")

    # Compute standardization statistics from training set only
    print("\nComputing target standardization statistics from training set...")
    mu, sigma = compute_target_stats(train_records)

    train_dataset   = FMRIRegressionDataset(train_records, bert_embeddings, mu, sigma)
    test_dataset    = FMRIRegressionDataset(test_records,  bert_embeddings, mu, sigma)

    train_loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader     = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Build model
    # -------------------------
    print(f"\nBuilding MLP regressor...")
    print(f"  Input dim:    {BERT_DIM}")
    print(f"  Hidden dims:  {HIDDEN_DIMS}")
    print(f"  Output dim:   {TARGET_DIM}")
    print(f"  Dropout:      {DROPOUT}")

    model       = MLPRegressor(
        input_dim   = BERT_DIM,
        hidden_dims = HIDDEN_DIMS,
        output_dim  = TARGET_DIM,
        dropout     = DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    optimizer   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion   = nn.MSELoss()
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # -------------------------
    # Training loop
    # -------------------------
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    train_mse_hist  = []
    val_mse_hist    = []
    best_val_mse    = float("inf")
    best_epoch      = None

    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_mse = train_epoch(model, train_loader, optimizer, criterion, device)
        train_mse_hist.append(train_mse)

        do_eval = (epoch == 1) or (epoch == TOTAL_EPOCHS) or (epoch % EVAL_INTERVAL == 0)

        if do_eval:
            val_mse = evaluate(model, test_loader, criterion, device)
            val_mse_hist.append(val_mse)
            scheduler.step(val_mse)

            if val_mse < best_val_mse:
                best_val_mse    = val_mse
                best_epoch      = epoch
                torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
                print(f"  ✓ Saved best model (epoch={epoch}, val_mse={val_mse:.6f})")

            print(
                f"Epoch {epoch}/{TOTAL_EPOCHS}: "
                f"Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f}"
            )
        else:
            print(
                f"Epoch {epoch}/{TOTAL_EPOCHS}: "
                f"Train MSE={train_mse:.6f}"
            )

    # -------------------------
    # Final evaluation
    # -------------------------
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=device))
    final_val_mse = evaluate(model, test_loader, criterion, device)
    print(f"  Best epoch:    {best_epoch}")
    print(f"  Best val MSE:  {best_val_mse:.6f}")
    print(f"  Final val MSE: {final_val_mse:.6f}")

    # -------------------------
    # Plot
    # -------------------------
    plot_training_curves(train_mse_hist, val_mse_hist, EVAL_INTERVAL, TOTAL_EPOCHS)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Records:          {len(records)}")
    print(f"  Input dim:        {BERT_DIM} (mean-pooled BERT)")
    print(f"  Hidden dims:      {HIDDEN_DIMS}")
    print(f"  Output dim:       {TARGET_DIM} (PCA-reduced fMRI)")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Best val MSE:     {best_val_mse:.6f}")
    print(f"  Model saved to:   {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED!")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
