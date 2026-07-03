import math
import torch
import torch.nn as nn
import itertools
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


def sinkhorn_logspace(log_alpha: torch.Tensor, n_iters: int = 20, eps: float = 1e-9) -> torch.Tensor:
    """
    Sinkhorn normalization in log-space to produce a doubly-stochastic matrix.
    log_alpha: (B, S, R)
    returns:   (B, S, R) with rows/cols approximately summing to 1
    """
    for _ in range(n_iters):
        # Row normalize: sum over roles (dim=2) -> 1
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        # Col normalize: sum over positions (dim=1) -> 1
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    P = torch.exp(log_alpha)
    # numerical cleanup (optional)
    return P.clamp_min(eps)


class RoleAssignmentTransformer(nn.Module):
    """
    Transformer-based role assigner that predicts a role distribution
    over `num_roles` for each position in the input sequence.

    MODIFIED: Added temperature parameter support for training stability
    """

    def __init__(
            self,
            num_roles: int,
            filler_embedding: nn.Embedding,
            d_model: int,
            role_embedding_dim: int,
            num_layers: int = 4,
            nhead: int = 8,
            dim_feedforward: Optional[int] = None,
            dropout: float = 0.1,
            softmax_roles: bool = True,
            role_assignment_shrink_filler_dim: Optional[int] = None,
            use_sinkhorn: bool = True,
            sinkhorn_iters: int = 80,
            sinkhorn_tau: float = 2.0
            , sinkhorn_tau_anneal: float = 0.97
            , sinkhorn_tau_min: float = 0.7
            , hard_permutation_eval: bool = True
    ):
        super().__init__()
        self.num_roles = num_roles
        self.softmax_roles = softmax_roles
        self.snap_one_hot_predictions = False
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_tau_anneal = sinkhorn_tau_anneal
        self.sinkhorn_tau_min = sinkhorn_tau_min
        self.hard_permutation_eval = hard_permutation_eval
        self.last_role_probs: Optional[torch.Tensor] = None  # (B,S,R) post-normalization
        self.last_role_logits: Optional[torch.Tensor] = None  # (B,S,R) raw logits

        # ---- Filler embeddings ----
        self.filler_embedding = filler_embedding
        embed_dim = filler_embedding.embedding_dim

        # Optionally shrink filler dim
        self.shrink_filler = False
        if role_assignment_shrink_filler_dim is not None:
            self.shrink_filler = True
            self.filler_shrink_layer = nn.Linear(
                embed_dim, role_assignment_shrink_filler_dim
            )
            embed_dim = role_assignment_shrink_filler_dim

        # Project to model dimension if needed
        self.pre_proj = (
            nn.Identity() if embed_dim == d_model else nn.Linear(embed_dim, d_model)
        )

        # Positional encoding + Transformer
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward or (4 * d_model),
            dropout=dropout,
            batch_first=True,  # use (B, S, E)
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Predict role logits for each token
        self.role_head = nn.Linear(d_model, num_roles)

        # Role embedding table
        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.1)

    def set_sinkhorn_tau(self, tau: float) -> None:
        """Manually set Sinkhorn temperature (tau)."""
        self.sinkhorn_tau = float(max(tau, 1e-6))

    def step_sinkhorn_tau(self) -> float:
        """Anneal Sinkhorn temperature multiplicatively down to sinkhorn_tau_min."""
        self.sinkhorn_tau = float(max(self.sinkhorn_tau_min, self.sinkhorn_tau * self.sinkhorn_tau_anneal))
        return self.sinkhorn_tau

    @staticmethod
    def _best_permutation_from_probs(prob_bsr: torch.Tensor) -> torch.Tensor:
        """
        Convert soft assignment probs (B,S,R) to a hard permutation matrix (B,S,R)
        that maximizes sum log probs. Uses brute-force for S<=7 (OK for S=6).
        """
        B, S, R = prob_bsr.shape
        assert S == R, "Hard permutation requires S==R"
        # Work on CPU for simplicity/stability
        prob_cpu = prob_bsr.detach().cpu()
        out = torch.zeros_like(prob_cpu)
        perms = list(itertools.permutations(range(S)))  # S! permutations
        for b in range(B):
            P = prob_cpu[b]  # (S,R)
            # avoid log(0)
            logP = (P + 1e-12).log()
            best_score = None
            best_perm = None
            for perm in perms:
                score = 0.0
                for s in range(S):
                    score += float(logP[s, perm[s]].item())
                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_perm = perm
            for s in range(S):
                out[b, s, best_perm[s]] = 1.0
        return out.to(prob_bsr.device)

    def forward(self, filler_tensor: torch.LongTensor, temperature: float = 1.0):
        """
        filler_tensor : (B, S) of token indices
        temperature : float, controls sharpness of role assignment (lower = sharper)

        Returns
        --------
        roles_embedded : (S, B, role_dim)
        role_logits    : (S, B, num_roles)   <-- LOGITS, not softmax
        """
        device = filler_tensor.device
        B, S = filler_tensor.shape

        # Get embeddings
        x = self.filler_embedding(filler_tensor)
        if self.shrink_filler:
            x = self.filler_shrink_layer(x)
        x = self.pre_proj(x)

        # Positional encoding + transformer
        #x = self.pos_enc(x.transpose(0, 1)).transpose(0, 1)  # (B, S, D)
        x = self.encoder(x)

        # Predict role logits
        logits = self.role_head(x)  # (B, S, num_roles)
        #Stabilize logits to prevent role-head collapse
        logits = logits - logits.mean(dim=-1, keepdim=True)
        logits = logits / logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = logits.clamp(min=-10.0, max=10.0)
        # MODIFIED: Apply temperature scaling to logits before normalization
        # This allows for temperature annealing during training
        scaled_logits = logits / max(temperature, 1e-6)

        # Role embeddings (use scaled logits -> assignment here)
        if self.use_sinkhorn and (self.num_roles == S):
            # Combine both temperature and sinkhorn_tau
            # Temperature controls the overall sharpness, sinkhorn_tau is for the algorithm
            log_alpha = scaled_logits / max(self.sinkhorn_tau, 1e-6)
            role_probs = sinkhorn_logspace(log_alpha, n_iters=self.sinkhorn_iters)  # (B, S, R)
        else:
            # Standard softmax with temperature
            role_probs = torch.softmax(scaled_logits, dim=-1)  # (B, S, R)

        # Optionally harden to a true permutation at evaluation time
        if (not self.training) and self.hard_permutation_eval and (self.num_roles == S):
            role_probs = self._best_permutation_from_probs(role_probs)

        self.last_role_probs = role_probs

        role_table = self.role_embedding.weight
        role_table = role_table / (role_table.norm(dim=1, keepdim=True) + 1e-9)

        roles_embedded_bs = role_probs @ role_table  # (B, S, role_dim)
        roles_embedded = roles_embedded_bs.transpose(0, 1)

        # IMPORTANT: return **logits**, not probs
        # Note: We return the original logits (not temperature-scaled) for regularization
        role_logits_sb = logits.transpose(0, 1)  # (S, B, num_roles)

        return roles_embedded, role_logits_sb
