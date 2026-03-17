from __future__ import division
from typing import Optional
import torch
import torch.nn as nn

# Import custom binding operations
from binding_operations import CircularConvolution, EltWise, SumFlattenedOuterProduct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RoleLearningTensorProductEncoder(nn.Module):
    """
    Learns to encode sequences with Tensor Product Representations (TPRs):
      - Filler embeddings (content)
      - Role assignment (Transformer-based via external RoleAssignmentTransformer)
      - Binding op (TPR / HRR / eltwise)
      - Optional final linear to match downstream dimensionality

    MODIFIED: Added temperature parameter support for training stability
    """

    # to-do: lower learning rate
    def __init__(
            self,
            n_roles,
            n_fillers,
            filler_dim,
            role_dim,
            final_layer_width=None,
            binder="tpr",
            role_learner_hidden_dim=512,
            bidirectional=False,  # kept for API compat
            num_layers=4,  # try reducing
            nhead=8,  # try reducing
            dim_feedforward=None,
            dropout=0.1,
            softmax_roles=True,
            one_hot_regularization_weight=0.00,  # try removing both one-hot & l2
            l2_norm_regularization_weight=0.00,
            # switch to L1 normalization # increase training data size from 6000 to 60,000
            unique_role_regularization_weight=1.2,
            role_assignment_shrink_filler_dim=None,
            embedder_squeeze=None,
    ):
        super().__init__()

        self.n_roles = n_roles
        self.n_fillers = n_fillers
        self.filler_dim = filler_dim
        self.role_dim = role_dim
        self.softmax_roles = softmax_roles
        self.dropout = dropout
        self.binder = binder.lower()
        self.final_layer_width = final_layer_width

        # ---- Filler embedding (+ optional squeeze) ----
        self.embed_squeeze = False
        if embedder_squeeze is None:
            self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        else:
            self.embed_squeeze = True
            self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
            self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)

        # ---- Role assigner (Transformer) - imported from role_assigner.py ----
        from role_assigner import RoleAssignmentTransformer

        self.role_assigner = RoleAssignmentTransformer(
            num_roles=n_roles,
            filler_embedding=self.filler_embedding,
            d_model=role_learner_hidden_dim,
            role_embedding_dim=role_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward if dim_feedforward is not None else 4 * role_learner_hidden_dim,
            dropout=dropout,
            softmax_roles=softmax_roles,
            role_assignment_shrink_filler_dim=role_assignment_shrink_filler_dim,
        )

        # ---- Binder selection (compute binder_output_dim) ----
        if self.binder in ("tpr", "tensor", "tpr_sum"):
            self.sum_layer = SumFlattenedOuterProduct()
            self.binder_output_dim = self.filler_dim * self.role_dim
        elif self.binder in ("hrr", "cconv", "circular", "circular_convolution"):
            self.sum_layer = CircularConvolution(self.filler_dim)
            self.binder_output_dim = self.filler_dim
        elif self.binder in ("eltwise", "elementwise", "hadamard"):
            self.sum_layer = EltWise()
            self.binder_output_dim = self.filler_dim
        else:
            raise ValueError(f"Unknown binder type: {binder}")

        print(f"[RoleEncoder] Binder '{self.binder}' output dim: {self.binder_output_dim}")

        # ---- Canonical projection head (binder_output_dim -> final_layer_width) ----
        # ALWAYS create this if final_layer_width is specified
        if self.final_layer_width is not None:
            print(f"[RoleEncoder] Creating projection: {self.binder_output_dim} -> {self.final_layer_width}")
            self.final_proj = nn.Sequential(
                nn.Linear(self.binder_output_dim, self.final_layer_width),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.final_layer_width, self.final_layer_width),
            )
            self.post_norm = nn.LayerNorm(self.final_layer_width)
        else:
            print(f"[RoleEncoder] No projection layer (final_layer_width=None)")
            self.final_proj = None
            self.post_norm = None

        # ---- Regularization controls ----
        self.regularize = True
        self.regularization_temp = 1.0
        self.one_hot_regularization_weight = one_hot_regularization_weight
        self.l2_norm_regularization_weight = l2_norm_regularization_weight
        self.unique_role_regularization_weight = unique_role_regularization_weight

    def forward(self, filler_list: torch.LongTensor, role_list_not_used=None, temperature: float = 1.0):
        """
        filler_list: (B, S)
        temperature: float, controls sharpness of role assignment softmax (lower = sharper)

        Returns:
          output: (B, final_layer_width) if configured, else (B, binder_output_dim)
          role_predictions: (S, B, n_roles)
        """
        # 1) Get role assignments from transformer role assigner (with temperature)
        roles_embedded, role_predictions = self.role_assigner(
            filler_list,
            temperature=temperature
        )  # (S, B, role_dim), (S, B, n_roles)

        # 2) Get filler embeddings
        fillers_embedded = self.filler_embedding(filler_list)  # (B, S, filler_dim or squeeze_dim)
        if self.embed_squeeze:
            fillers_embedded = self.embedding_squeeze_layer(fillers_embedded)
        fillers_embedded = fillers_embedded.transpose(0, 1)  # (S, B, filler_dim)

        # 3) Bind fillers × roles -> (B, binder_output_dim)
        bound = self.sum_layer(fillers_embedded, roles_embedded)

        # 4) Project to target width if specified
        if self.final_proj is not None:
            output = self.final_proj(bound)
            if self.post_norm is not None:
                output = self.post_norm(output)
        else:
            output = bound

        return output, role_predictions

    def use_regularization(self, use_regularization: bool):
        """Enable/disable regularization"""
        self.regularize = use_regularization

    def set_regularization_temp(self, temp: float):
        """Set temperature for regularization"""
        self.regularization_temp = temp

    def get_regularization_loss(self, role_predictions):
        """
        Compute regularization losses from role predictions (S, B, R).
        Returns: (one_hot_loss, l2_loss, unique_loss)
        """
        if not self.regularize:
            zero = role_predictions.new_tensor(0.0)
            return zero, zero, zero

        batch_size = role_predictions.shape[1]
        temperature = self.regularization_temp

        # One-hot regularization: encourage peaky distributions
        if self.softmax_roles:
            one_hot_reg = torch.sum(role_predictions * (1 - role_predictions))
        else:
            one_hot_reg = torch.sum((role_predictions ** 2) * (1 - role_predictions) ** 2)
        one_hot_loss = temperature * one_hot_reg / batch_size

        # L2 norm regularization
        if self.softmax_roles:
            uniform_target = 1.0 / self.n_roles
            l2_norm = torch.sum((role_predictions - uniform_target) ** 2)
        else:
            l2_norm = (torch.sum(role_predictions ** 2) - 1) ** 2
        l2_norm_loss = temperature * l2_norm / batch_size

        # Unique role regularization: encourage different positions to use different roles
        exclusive_role_vector = torch.sum(role_predictions, 0)  # (B, R)
        unique_role_loss = temperature * torch.sum(
            (exclusive_role_vector * (1 - exclusive_role_vector)) ** 2
        ) / batch_size

        return (
            self.one_hot_regularization_weight * one_hot_loss,
            self.l2_norm_regularization_weight * l2_norm_loss,
            self.unique_role_regularization_weight * unique_role_loss
        )

    def train(self, mode: bool = True):
        """Toggle training mode"""
        super().train(mode)
        if hasattr(self.role_assigner, 'snap_one_hot_predictions'):
            self.role_assigner.snap_one_hot_predictions = False

    def eval(self):
        """Eval mode"""
        super().eval()
        if hasattr(self.role_assigner, 'snap_one_hot_predictions'):
            self.role_assigner.snap_one_hot_predictions = True
