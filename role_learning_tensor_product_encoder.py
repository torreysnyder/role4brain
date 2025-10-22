from __future__ import division
from typing import Optional
import torch
import torch.nn as nn

# Import custom binding operations for different ways to combine fillers and roles
from binding_operations import CircularConvolution, EltWise, SumFlattenedOuterProduct
# NOTE: LSTM assigner no longer used
# from role_assigner import RoleAssignmentLSTM

# set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# Positional encoding (sinusoidal)
# -----------------------------
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, S, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -----------------------------
# Transformer-based Role Assigner
# -----------------------------
class RoleAssignmentTransformer(nn.Module):
    """
    Predicts a distribution over roles per token using a Transformer encoder,
    and maps that distribution to role vectors via a learned role embedding table.

    API matches the previous LSTM role assigner:
      forward(fillers) -> (roles_embedded, role_predictions)
        roles_embedded: (S, B, role_dim)
        role_predictions: (S, B, n_roles)  (probs if softmax_roles=True, else logits)
    """
    def __init__(
        self,
        n_roles: int,
        filler_embedding: nn.Embedding,
        model_dim: int,
        role_dim: int,
        *,
        role_assignment_shrink_filler_dim: Optional[int] = None,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        softmax_roles: bool = True,
    ):
        super().__init__()
        self.n_roles = n_roles
        self.filler_embedding = filler_embedding
        self.model_dim = model_dim
        self.role_dim = role_dim
        self.softmax_roles = softmax_roles

        # Exposed so external code can find it (encode_with_gumbel relies on this)
        self.role_embedding = nn.Embedding(n_roles, role_dim)

        # Optionally shrink/expand filler embedding dimension to model_dim
        filler_dim = filler_embedding.embedding_dim
        if role_assignment_shrink_filler_dim is not None:
            # First shrink to the requested size, then project to model_dim
            self.pre_proj = nn.Sequential(
                nn.Linear(filler_dim, role_assignment_shrink_filler_dim),
                nn.ReLU(),
                nn.Linear(role_assignment_shrink_filler_dim, model_dim),
            )
        elif filler_dim != model_dim:
            self.pre_proj = nn.Linear(filler_dim, model_dim)
        else:
            self.pre_proj = nn.Identity()

        self.pos_enc = PositionalEncoding(model_dim, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True  # (B, S, D)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Head to predict role logits at each position
        self.role_head = nn.Linear(model_dim, n_roles)

        # Flag toggled by outer module's train()/eval() for one-hot snapping
        self.snap_one_hot_predictions = False

    def forward(self, filler_indices: torch.Tensor):
        """
        filler_indices: (B, S) Long
        Returns:
          roles_embedded: (S, B, role_dim)
          role_predictions: (S, B, n_roles)   (probs if softmax_roles else logits)
        """
        # Embed fillers then project to model dimension
        x = self.filler_embedding(filler_indices)  # (B, S, filler_dim)
        x = self.pre_proj(x)                       # (B, S, model_dim)
        x = self.pos_enc(x)                        # add positional info

        # Transformer encoder over the sequence
        h = self.encoder(x)                        # (B, S, model_dim)

        # Role logits per position
        role_logits = self.role_head(h)            # (B, S, n_roles)

        if self.softmax_roles:
            role_probs = torch.softmax(role_logits, dim=-1)  # (B, S, n_roles)
        else:
            role_probs = role_logits  # treat as unnormalized if desired

        # Optionally snap to hard one-hot at inference
        if self.snap_one_hot_predictions and self.softmax_roles:
            idx = role_probs.argmax(dim=-1, keepdim=True)         # (B, S, 1)
            one_hot = torch.zeros_like(role_probs).scatter_(-1, idx, 1.0)
            role_weights = one_hot
        else:
            role_weights = role_probs

        # Map role distribution → role vector via learned role embedding table
        # roles_emb: (B, S, role_dim) = (B, S, n_roles) @ (n_roles, role_dim)
        roles_emb = torch.einsum("bsr,rd->bsd", role_weights, self.role_embedding.weight)

        # Return in (S, B, ...) to match the legacy interface used by the binder
        roles_emb = roles_emb.transpose(0, 1)       # (S, B, role_dim)
        role_pred = (role_probs if self.softmax_roles else role_logits).transpose(0, 1)  # (S, B, n_roles)
        return roles_emb, role_pred


# A tensor product encoder layer
# Takes a list of fillers and a list of roles and returns an encoding
class RoleLearningTensorProductEncoder(nn.Module):
    """
    Learns to encode sequences with Tensor Product Representations (TPRs):
      - Filler embeddings (content)
      - Role assignment (now Transformer-based)
      - Binding op (TPR / HRR / eltwise)
      - Optional final linear to match downstream dimensionality
    """
    def __init__(
            self,
            n_roles=6,
            n_fillers=101,
            filler_dim=64,
            role_dim=32,
            final_layer_width=None,
            pretrained_filler_embeddings=None,
            embedder_squeeze=None,
            binder="tpr",
            role_learner_hidden_dim=256,              # now 'model_dim' for the Transformer
            role_assignment_shrink_filler_dim=None,
            bidirectional=False,                     # kept for API compatibility (unused)
            num_layers=4,
            softmax_roles=True,
            pretrained_embeddings=None,
            one_hot_regularization_weight=2.0,
            l2_norm_regularization_weight=1.0,
            unique_role_regularization_weight=2.0,
            # NEW hyperparams for transformer (optional advanced tuning)
            nhead: int = 8,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
    ):
        super(RoleLearningTensorProductEncoder, self).__init__()
        self.n_roles = n_roles
        self.n_fillers = n_fillers

        self.one_hot_regularization_weight = one_hot_regularization_weight
        self.l2_norm_regularization_weight = l2_norm_regularization_weight
        self.unique_role_regularization_weight = unique_role_regularization_weight
        self.regularize = False

        self.filler_dim = filler_dim
        self.role_dim = role_dim

        # ---- Filler embedding (with optional squeeze) ----
        if embedder_squeeze is None:
            self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
            self.embed_squeeze = False
            print("no squeeze")
        else:
            self.embed_squeeze = True
            self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
            self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)
            print("squeeze")

        if pretrained_embeddings is not None:
            self.filler_embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings).to(device)})
            self.filler_embedding.weight.requires_grad = False

        if pretrained_filler_embeddings:
            print('Using pretrained filler embeddings')
            self.filler_embedding.load_state_dict(torch.load(pretrained_filler_embeddings, map_location=device))
            self.filler_embedding.weight.requires_grad = False

        # ---- Transformer-based role assigner (replaces LSTM) ----
        # role_learner_hidden_dim serves as the transformer model dimension
        self.role_assigner = RoleAssignmentTransformer(
            n_roles=self.n_roles,
            filler_embedding=self.filler_embedding,
            model_dim=role_learner_hidden_dim,
            role_dim=self.role_dim,
            role_assignment_shrink_filler_dim=role_assignment_shrink_filler_dim,
            num_layers=max(1, num_layers),      # keep your CLI/API consistent
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            softmax_roles=softmax_roles,
        )

        # ---- Binding op ----
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        elif binder == "hrr":
            self.sum_layer = CircularConvolution(self.filler_dim)
        elif binder in ("eltwise", "elt"):
            self.sum_layer = EltWise()
        else:
            raise ValueError(f"Invalid binder: {binder}")

        # ---- Optional final linear layer ----
        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            if binder == "tpr":
                # replace the single Linear with:
                self.last_layer = nn.Sequential(
                    nn.Linear(self.filler_dim * self.role_dim, max(512, self.final_layer_width)),
                    nn.GELU(),
                    nn.Linear(max(512, self.final_layer_width), self.final_layer_width),
                )

            else:
                self.last_layer = nn.Linear(self.filler_dim, self.final_layer_width)

    def forward(self, filler_list, role_list_not_used=None):
        """
        Args:
          filler_list: (B, S) Long tensor of token ids
          role_list_not_used: legacy arg (roles are learned)
        Returns:
          output: (B, D_out)  encoded sequence
          role_predictions: (S, B, n_roles)  for diagnostics / regularization
        """
        # 1) Filler embeddings (optionally squeezed)
        fillers_embedded = self.filler_embedding(filler_list)  # (B, S, filler_dim or embedder_squeeze)
        if self.embed_squeeze:
            fillers_embedded = self.embedding_squeeze_layer(fillers_embedded)  # → (B, S, filler_dim)

        # 2) Transformer role assignment (returns roles in (S, B, role_dim))
        roles_embedded, role_predictions = self.role_assigner(filler_list)
        # match binder's expected shape: we need roles_embedded as (B, S, role_dim)
        roles_embedded = roles_embedded.transpose(0, 1)  # (B, S, role_dim)

        # 3) Bind fillers × roles (TPR/HRR/Eltwise) and reduce across sequence
        output = self.sum_layer(fillers_embedded, roles_embedded)

        # 4) Optional bottleneck to a specific dimensionality
        if self.has_last:
            output = self.last_layer(output)

        return output, role_predictions  # role_predictions: (S, B, n_roles)

    # ---- Regularization & mode toggles (kept unchanged) ----
    def use_regularization(self, use_regularization: bool):
        self.regularize = use_regularization

    def set_regularization_temp(self, temp: float):
        self.regularization_temp = temp

    def get_regularization_loss(self, role_predictions):
        """
        Same regularizers as before, using role_predictions (S, B, R).
        """
        if not self.regularize:
            return 0, 0, 0

        one_hot_temperature = self.regularization_temp
        batch_size = role_predictions.shape[1]
        softmax_roles = self.role_assigner.softmax_roles

        if softmax_roles:
            one_hot_reg = torch.sum(role_predictions * (1 - role_predictions))
        else:
            one_hot_reg = torch.sum((role_predictions ** 2) * (1 - role_predictions) ** 2)
        one_hot_loss = one_hot_temperature * one_hot_reg / batch_size

        if softmax_roles:
            l2_norm = -torch.sum(role_predictions * role_predictions)
        else:
            l2_norm = (torch.sum(role_predictions ** 2) - 1) ** 2
        l2_norm_loss = one_hot_temperature * l2_norm / batch_size

        exclusive_role_vector = torch.sum(role_predictions, 0)
        unique_role_loss = one_hot_temperature * torch.sum(
            (exclusive_role_vector * (1 - exclusive_role_vector)) ** 2) / batch_size

        return (self.one_hot_regularization_weight * one_hot_loss,
                self.l2_norm_regularization_weight * l2_norm_loss,
                self.unique_role_regularization_weight * unique_role_loss)

    def train(self, mode: bool = True):
        """
        Toggle training mode; also ensure role assigner does NOT snap to one-hot.
        """
        super().train(mode)
        self.role_assigner.snap_one_hot_predictions = False

    def eval(self):
        """
        Eval mode; snap role predictions to one-hot if softmax_roles=True.
        (Your Gumbel evaluation in role_approx.py overrides discretization anyway.)
        """
        super().eval()
        self.role_assigner.snap_one_hot_predictions = False


# (Optional) keep create_encoder_with_bert_embeddings unchanged, or wire it to use the transformer assigner.
