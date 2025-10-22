"""
ROLE approximation → transformer decoder evaluation with Gumbel-Softmax
- Trains ROLE to match TPDN encodings
- Trains a transformer decoder on true TPDN encodings
- Evaluates substitution accuracy by feeding ROLE’s (gumbelized) encodings to the decoder
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
    def __init__(self, filler, target_rep, target_output=None, target_roles=None):
        self.filler = filler          # LongTensor [1, seq_len]
        self.target_rep = target_rep  # FloatTensor [1, rep_dim] or any shape -> will be flattened
        self.target_output = target_output  # LongTensor [1, seq_len]
        self.target_roles = target_roles    # LongTensor [1, seq_len] (optional)


# --------------------
# Decoder
# --------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, encoding_dim, output_vocab_size, max_seq_len=20,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
                 sos_id=1, eos_id=2, pad_id=0):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.output_vocab_size = output_vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.encoding_proj = nn.Linear(encoding_dim, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_vocab_size)

    def forward(self, encoding, target_seq=None, teacher_forcing_ratio=1.0):
        """
        encoding: (batch, encoding_dim)  [FLATTENED]
        target_seq: (batch, seq_len) of token IDs. If None -> autoregressive generation
        """
        batch_size = encoding.size(0)
        memory = self.encoding_proj(encoding).unsqueeze(1)  # (B, 1, d_model)

        if target_seq is not None and random.random() < teacher_forcing_ratio:
            seq_len = target_seq.size(1)
            tgt_embedded = self.output_embedding(target_seq)  # (B, L, d_model)
            tgt_embedded = self.pos_encoder(tgt_embedded)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(encoding.device)
            decoder_output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
            return self.output_proj(decoder_output)  # (B, L, V)
        else:
            SOS, EOS = self.sos_id, self.eos_id
            cur = torch.full((batch_size, 1), SOS, dtype=torch.long, device=encoding.device)
            outs = []
            for _ in range(self.max_seq_len):
                tgt_emb = self.output_embedding(cur)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur.size(1)).to(encoding.device)
                dec_out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                next_logits = self.output_proj(dec_out[:, -1:, :])  # (B, 1, V)
                outs.append(next_logits)
                next_tok = next_logits.argmax(dim=-1)
                cur = torch.cat([cur, next_tok], dim=1)
                if (next_tok == EOS).all():
                    break
            return torch.cat(outs, dim=1) if outs else torch.zeros(batch_size, 0, self.output_vocab_size, device=encoding.device)


# --------------------
# Data loading
# --------------------

def load_tpdn_outputs(filename, device, load_targets=True):
    print(f"Loading TPDN outputs from {filename}...")
    try:
        with open(filename, 'r') as f:
            outputs = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File {filename} not found.")
        sys.exit(1)
    if not outputs:
        print("ERROR: No data in file.")
        sys.exit(1)

    print(f"Loaded {len(outputs)} instances")
    data = []
    for entry in outputs:
        filler_ids = entry['filler_ids']
        target_output_embedding = entry['model_output']
        target_output_seq = entry.get('target_output', filler_ids) if load_targets else None
        role_ids = entry.get('role_ids', None)

        filler_t = Variable(torch.LongTensor(filler_ids), requires_grad=False).unsqueeze(0)
        target_emb_t = Variable(torch.FloatTensor(target_output_embedding), requires_grad=False).unsqueeze(0)
        target_seq_t = Variable(torch.LongTensor(target_output_seq), requires_grad=False).unsqueeze(0) if target_output_seq else None
        role_t = Variable(torch.LongTensor(role_ids), requires_grad=False).unsqueeze(0) if role_ids is not None else None

        filler_t = filler_t.to(device)
        target_emb_t = target_emb_t.to(device)
        if target_seq_t is not None: target_seq_t = target_seq_t.to(device)
        if role_t is not None: role_t = role_t.to(device)

        data.append(RoleLearningInstance(filler_t, target_emb_t, target_seq_t, role_t))

    print(f"Created {len(data)} role learning instances")
    if data:
        print(f"  Sequence length: {len(outputs[0]['filler_ids'])}")
        print(f"  Target representation size: {len(outputs[0]['model_output'])}")
    return data


# --------------------
# Gumbel-Softmax
# --------------------

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    eps = 1e-20
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y = (logits + gumbel) / max(temperature, 1e-8)
    y_soft = F.softmax(y, dim=-1)
    if not hard:
        return y_soft
    y_hard = torch.zeros_like(y_soft)
    y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
    return y_hard - y_soft.detach() + y_soft  # straight-through


def _find_role_embedding_weight(role_model):
    ra = role_model.role_assigner
    for name in ["role_embedding", "role_embed", "R", "roles", "embedding", "embeddings", "role_proj"]:
        if hasattr(ra, name):
            obj = getattr(ra, name)
            if isinstance(obj, nn.Embedding):
                return obj.weight
            if torch.is_tensor(obj) and obj.dim() == 2:
                return obj
    for m in ra.modules():
        if isinstance(m, nn.Embedding):
            return m.weight
    raise AttributeError("Could not locate role-embedding weights inside role_assigner.")


@torch.no_grad()
def encode_with_gumbel(role_model, filler, temperature=1.0, hard=False):
    """
    Produce ROLE encoding with Gumbel-Softmax discretized role assignments.
    """
    device = filler.device
    fillers_emb = role_model.filler_embedding(filler)
    if getattr(role_model, "embed_squeeze", False):
        fillers_emb = role_model.embedding_squeeze_layer(fillers_emb)

    # role_assigner may return (seq, batch, roles) or (batch, seq, roles)
    _, role_preds = role_model.role_assigner(filler)
    if role_preds.dim() != 3:
        raise RuntimeError(f"Unexpected role_predictions shape: {role_preds.shape}")
    if role_preds.size(0) == filler.size(0):            # (batch, seq, roles)
        rp_bsr = role_preds
    elif role_preds.size(1) == filler.size(0):          # (seq, batch, roles)
        rp_bsr = role_preds.transpose(0, 1)
    else:
        raise RuntimeError(f"Can't align role_predictions shape {role_preds.shape} with filler {filler.shape}")

    # Convert to logits safely
    rp = rp_bsr.clamp_min(1e-8)
    rp = rp / rp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    logits = torch.log(rp)

    role_weights = gumbel_softmax_sample(logits, temperature=temperature, hard=hard)  # (B, S, R)
    R_w = _find_role_embedding_weight(role_model)  # (R, D_r)
    roles_emb = torch.einsum("bsr,rd->bsd", role_weights, R_w)

    bound = role_model.sum_layer(fillers_emb, roles_emb)
    if getattr(role_model, "has_last", 0) == 1:
        bound = role_model.last_layer(bound)
    return bound  # (B, encoding_dim)


# --------------------
# Training helpers
# --------------------

def batchify(data, batch_size):
    batches = []
    for bi in range(0, len(data), batch_size):
        batch_data = data[bi:min(bi + batch_size, len(data))]
        first = batch_data[0]
        cur_filler = first.filler
        cur_target = first.target_rep
        cur_output = first.target_output if first.target_output is not None else None
        for inst in batch_data[1:]:
            cur_filler = torch.cat((cur_filler, inst.filler), 0)
            cur_target = torch.cat((cur_target, inst.target_rep), 0)
            if cur_output is not None and inst.target_output is not None:
                cur_output = torch.cat((cur_output, inst.target_output), 0)
        batches.append((cur_filler, cur_target, cur_output) if cur_output is not None else (cur_filler, cur_target))
    return batches


def train_epoch(model, train_batches, optimizer, criterion, batch_size, use_regularization=False):
    model.train()
    random.shuffle(train_batches)
    epoch_losses, epoch_reg_losses = [], []

    for bi in tqdm(range(0, len(train_batches), batch_size), desc="Training"):
        batch = train_batches[bi:min(bi + batch_size, len(train_batches))]
        if not batch:
            continue
        optimizer.zero_grad()
        total_loss, total_reg = 0, 0

        for batch_item in batch:
            filler_in = batch_item[0]
            target_out = batch_item[1]
            enc_out, role_predictions = model(filler_in, None)
            recon_loss = criterion(enc_out, target_out)
            total_loss += recon_loss

            if use_regularization:
                one_hot_loss, l2_loss, unique_loss = model.get_regularization_loss(role_predictions)
                reg_loss = one_hot_loss + l2_loss + unique_loss
                total_reg += reg_loss
                total_loss += reg_loss

        total_loss.backward()
        optimizer.step()
        epoch_losses.append(recon_loss.item())
        if use_regularization:
            epoch_reg_losses.append(total_reg.item() if total_reg != 0 else 0)

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
    avg_reg = sum(epoch_reg_losses) / len(epoch_reg_losses) if epoch_reg_losses else 0
    return avg_loss, avg_reg


def train_decoder(decoder, data, optimizer, criterion, batch_size, n_epochs=50, patience=10):
    print("\n" + "=" * 80)
    print("TRAINING TRANSFORMER DECODER")
    print("=" * 80)

    n_total = len(data)
    n_train = int(0.85 * n_total)
    train_data = data[:n_train]
    valid_data = data[n_train:]

    print(f"Training decoder on {len(train_data)} sequences")
    print(f"Validation: {len(valid_data)} sequences")

    best_loss = float('inf')
    patience_counter = 0
    train_losses, valid_losses = [], []

    for epoch in range(n_epochs):
        decoder.train()
        epoch_loss, n_batches = 0, 0
        indices = torch.randperm(len(train_data))

        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch + 1}"):
            batch_indices = indices[i:min(i + batch_size, len(train_data))]
            batch_enc = torch.stack([train_data[idx].target_rep.squeeze(0) for idx in batch_indices])
            batch_tgt = torch.stack([train_data[idx].target_output.squeeze(0) for idx in batch_indices])

            # FLATTEN encodings -> shape (B, encoding_dim)
            batch_enc = batch_enc.view(batch_enc.size(0), -1)

            optimizer.zero_grad()
            logits = decoder(batch_enc, batch_tgt[:, :-1], teacher_forcing_ratio=1.0)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item(); n_batches += 1

        avg_train = epoch_loss / max(1, n_batches)
        train_losses.append(avg_train)

        decoder.eval()
        v_loss, v_batches = 0, 0
        with torch.no_grad():
            for j in range(0, len(valid_data), batch_size):
                chunk = valid_data[j:min(j + batch_size, len(valid_data))]
                v_enc = torch.stack([inst.target_rep.squeeze(0) for inst in chunk])
                v_tgt = torch.stack([inst.target_output.squeeze(0) for inst in chunk])

                v_enc = v_enc.view(v_enc.size(0), -1)

                v_logits = decoder(v_enc, v_tgt[:, :-1], teacher_forcing_ratio=1.0)
                loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_tgt[:, 1:].reshape(-1))
                v_loss += loss.item(); v_batches += 1
        avg_valid = v_loss / max(1, v_batches)
        valid_losses.append(avg_valid)

        print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss={avg_train:.4f}, Valid Loss={avg_valid:.4f}")
        if avg_valid < best_loss:
            best_loss = avg_valid; patience_counter = 0
            torch.save(decoder.state_dict(), "best_decoder_model.pt")
            print("  → New best decoder saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping")
                break

    decoder.load_state_dict(torch.load("best_decoder_model.pt"))
    return decoder, train_losses, valid_losses


# --------------------
# Evaluation with Gumbel-Softmax
# --------------------

@torch.no_grad()
def evaluate_substitution_accuracy(role_model, decoder, data, gumbel_temp=1.0, use_hard_gumbel=False):
    role_model.eval()
    decoder.eval()

    correct, total = 0, 0
    for inst in tqdm(data, desc=f"Eval (T={gumbel_temp:.3f}, hard={use_hard_gumbel})"):
        role_enc = encode_with_gumbel(role_model, inst.filler, temperature=gumbel_temp, hard=use_hard_gumbel)

        # FLATTEN role encoding to decoder input
        role_enc = role_enc.view(role_enc.size(0), -1)

        output_logits = decoder(role_enc, target_seq=None, teacher_forcing_ratio=0.0)
        if output_logits.numel() == 0:
            total += 1
            continue

        pred_seq = output_logits.argmax(dim=-1).squeeze(0)
        target_seq = inst.target_output.squeeze(0)

        L = min(pred_seq.size(0), target_seq.size(0))
        if torch.equal(pred_seq[:L], target_seq[:L]):
            correct += 1
        total += 1

    return (correct / total) if total > 0 else 0.0


@torch.no_grad()
def evaluate_across_temps(
    role_model,
    decoder,
    data,
    T_init=1.0,
    T_min=0.1,
    decay=0.95,
    use_hard_gumbel=False,
    csv_path="sub_acc_vs_T.csv",
    png_path="sub_acc_vs_T.png",
):
    assert 0 < decay <= 1.0, "decay must be in (0,1]"
    assert T_init > 0 and T_min > 0, "temperatures must be > 0"

    temps, accs = [], []
    T = float(T_init)
    step = 0

    def _next_temp(cur):
        nxt = cur * decay
        if decay == 1.0:
            return T_min
        return max(T_min, nxt)

    while True:
        acc = evaluate_substitution_accuracy(role_model, decoder, data, gumbel_temp=T, use_hard_gumbel=use_hard_gumbel)
        temps.append(T); accs.append(acc)
        print(f"  -> T={T:.4f} | hard={use_hard_gumbel} | substitution_acc={acc:.4f}")

        if T <= T_min + 1e-12:
            break
        T_next = _next_temp(T)
        if abs(T_next - T) < 1e-12:
            break
        T = T_next
        step += 1
        if step > 10_000:
            break

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "substitution_accuracy", "hard"])
        for t, a in zip(temps, accs):
            w.writerow([t, a, int(use_hard_gumbel)])
    print(f"Saved CSV: {csv_path}")

    plt.figure()
    plt.plot(temps, accs, marker="o")
    plt.gca().set_xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Gumbel temperature T (log scale)")
    plt.ylabel("Substitution accuracy")
    title_mode = "HARD (ST)" if use_hard_gumbel else "SOFT"
    plt.title(f"Decoder substitution accuracy vs. temperature — {title_mode}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

    return list(zip(temps, accs))


# --------------------
# Plain MSE eval and role analysis
# --------------------

@torch.no_grad()
def evaluate(role_model, data):
    role_model.eval()
    mse = 0.0
    for inst in data:
        pred, _ = role_model(inst.filler, None)
        mse += torch.mean((pred - inst.target_rep) ** 2).item()
    return mse / max(1, len(data))


@torch.no_grad()
def analyze_role_assignments(model, data, num_samples=10):
    model.eval()
    print("\n" + "=" * 80)
    print("ROLE ASSIGNMENT ANALYSIS")
    print("=" * 80)
    for i in range(min(num_samples, len(data))):
        instance = data[i]
        _, role_predictions = model(instance.filler, None)
        role_pred_np = role_predictions.squeeze(1).cpu().numpy()  # (seq, roles)
        assigned_roles = np.argmax(role_pred_np, axis=1)
        max_probs = np.max(role_pred_np, axis=1)
        fillers = instance.filler.squeeze(0).cpu().numpy()
        if instance.target_roles is not None:
            l2r_roles = instance.target_roles.squeeze(0).cpu().numpy()
            match = np.array_equal(l2r_roles, assigned_roles)
            print(f"\nInstance {i}:")
            print(f"  Fillers:       {fillers}")
            print(f"  L2R roles:     {l2r_roles}")
            print(f"  Learned roles: {assigned_roles}")
            print(f"  Confidence:    {['%.3f' % p for p in max_probs]}")
            print(f"  Match L2R:     {match}")
        else:
            print(f"\nInstance {i}:")
            print(f"  Fillers:       {fillers}")
            print(f"  Learned roles: {assigned_roles}")
            print(f"  Confidence:    {['%.3f' % p for p in max_probs]}")


# --------------------
# Main (no argparse; hardcoded config)
# --------------------

def main():
    # ---- Config (edit here) ----
    TPDN_JSON = "model_outputs_test_l2r.json"

    SEED = 42
    EPOCHS_ROLE = 50
    EPOCHS_DEC = 50
    BATCH_SIZE = 16

    # One-shot eval (non-annealed)
    GUMBEL_TEMP = 0.5
    RUN_HARD_ONESHOT = True

    # Annealing sweep (eval-only)
    TEMP_INIT = 1.0
    TEMP_MIN = 0.1
    TEMP_DECAY = 0.95
    RUN_SOFT_CURVE = True
    RUN_HARD_CURVE = True

    # Special token ids (adjust to your dataset if different)
    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2

    # ---- Setup ----
    device = set_seed(SEED)

    # 1) Data
    data = load_tpdn_outputs(TPDN_JSON, device, load_targets=True)

    # ---- Infer vocab + check token ranges ----
    all_max = []
    all_min = []
    for inst in data:
        t = inst.target_output.squeeze(0)
        all_max.append(int(t.max().item()))
        all_min.append(int(t.min().item()))
    max_tok = max(all_max)
    min_tok = min(all_min)

    offset = -min(0, min_tok)  # shift negatives up to 0 if needed
    if offset != 0:
        print(f"[WARN] Found negative token ids (min={min_tok}). Shifting all targets by +{offset}.")
        for inst in data:
            inst.target_output = inst.target_output + offset
        max_tok += offset
        min_tok = 0

    out_vocab = max_tok + 1
    print(f"[INFO] Inferred vocab: 0..{max_tok} (size={out_vocab}), PAD={PAD_ID}")

    for inst in data:
        t = inst.target_output
        if (t < 0).any() or (t >= out_vocab).any():
            raise ValueError(f"Target tokens out of bounds. Found range {int(t.min())}..{int(t.max())}, vocab={out_vocab}")

    # 2) ROLE model (sizes should match your data)
    num_fillers = 101  # tokens 0..100 (adjust if needed)
    n_roles = 6
    filler_dim = 64
    role_dim = 32

    # FLATTENED encoding dim
    target_dim = int(data[0].target_rep.view(-1).numel())

    role_model = RoleLearningTensorProductEncoder(
        n_roles=n_roles,
        n_fillers=num_fillers,
        filler_dim=filler_dim,
        role_dim=role_dim,
        final_layer_width=target_dim,
        binder="tpr",
        role_learner_hidden_dim=256,
        bidirectional=False,
        num_layers=4,
        softmax_roles=True,
        one_hot_regularization_weight=1.0,
        l2_norm_regularization_weight=1.0,
        unique_role_regularization_weight=1.0,
    ).to(device)

    # 3) Train ROLE against TPDN encodings
    optimizer = optim.Adam(role_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_set = data[:int(0.8 * len(data))]
    test_set = data[int(0.8 * len(data)):]
    train_batches = batchify(train_set, BATCH_SIZE)

    print("\nTraining ROLE to match TPDN encodings...")
    best = float('inf')
    for ep in range(EPOCHS_ROLE):
        avg_loss, _ = train_epoch(role_model, train_batches, optimizer, criterion, batch_size=1, use_regularization=True)
        val_mse = evaluate(role_model, test_set)
        print(f"ROLE Epoch {ep+1}/{EPOCHS_ROLE}: Train MSE={avg_loss:.6f} | Val MSE={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            torch.save(role_model.state_dict(), "best_role_model.pt")
            print("  → Saved best ROLE model")
    role_model.load_state_dict(torch.load("best_role_model.pt"))

    # 4) Train transformer decoder on *true* TPDN encodings
    max_seq_len = data[0].target_output.size(1)
    decoder = SimpleTransformerDecoder(
        encoding_dim=target_dim,
        output_vocab_size=out_vocab,
        max_seq_len=max_seq_len,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
        sos_id=SOS_ID, eos_id=EOS_ID, pad_id=PAD_ID
    ).to(device)

    dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)
    dec_crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Train decoder (uses true TPDN encodings)
    decoder, train_losses, valid_losses = train_decoder(
        decoder, data, dec_opt, dec_crit, batch_size=BATCH_SIZE, n_epochs=EPOCHS_DEC, patience=10
    )

    # ---- Save decoder loss curves (CSV + PNG) ----
    # CSV
    with open("decoder_losses.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "valid_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, valid_losses), start=1):
            w.writerow([i, tr, va])
    print("Saved CSV: decoder_losses.csv")

    # Plot
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, valid_losses, marker="o", label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Transformer decoder training")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("decoder_losses.png", dpi=150)
    print("Saved plot: decoder_losses.png")

    # 5a) One-shot substitution accuracy at fixed T
    # Build test_set (same as above, but ensure non-empty)
    if len(test_set) == 0:
        test_set = data  # fallback

    acc_soft = evaluate_substitution_accuracy(role_model, decoder, test_set,
                                              gumbel_temp=GUMBEL_TEMP, use_hard_gumbel=False)
    print(f"\n[One-shot] Substitution accuracy (SOFT, T={GUMBEL_TEMP}): {acc_soft:.4f}")

    if RUN_HARD_ONESHOT:
        acc_hard = evaluate_substitution_accuracy(role_model, decoder, test_set,
                                                  gumbel_temp=GUMBEL_TEMP, use_hard_gumbel=True)
        print(f"[One-shot] Substitution accuracy (HARD-ST, T={GUMBEL_TEMP}): {acc_hard:.4f}")

    # 5b) Annealed sweeps (eval-only)
    if RUN_SOFT_CURVE:
        print("\n=== Annealed sweep (SOFT) ===")
        evaluate_across_temps(
            role_model, decoder, test_set,
            T_init=TEMP_INIT, T_min=TEMP_MIN, decay=TEMP_DECAY,
            use_hard_gumbel=False,
            csv_path="sub_acc_vs_T_soft.csv",
            png_path="sub_acc_vs_T_soft.png",
        )

    if RUN_HARD_CURVE:
        print("\n=== Annealed sweep (HARD-ST) ===")
        evaluate_across_temps(
            role_model, decoder, test_set,
            T_init=TEMP_INIT, T_min=TEMP_MIN, decay=TEMP_DECAY,
            use_hard_gumbel=True,
            csv_path="sub_acc_vs_T_hard.csv",
            png_path="sub_acc_vs_T_hard.png",
        )

    # Optional: quick glance at role assignments
    # analyze_role_assignments(role_model, test_set, num_samples=5)


if __name__ == "__main__":
    main()
