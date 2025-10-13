"""
Simplified Role Learning Tensor Product Encoder
Works with the RoleAssignmentLSTM module to learn roles from data
"""

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class SumFlattenedOuterProduct(nn.Module):
    """Tensor product binding operation"""
    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()
    
    def forward(self, input1, input2):
        """
        Compute outer product between fillers and roles
        input1: (batch_size, seq_len, filler_dim)
        input2: (seq_len, batch_size, role_dim) or (batch_size, seq_len, role_dim)
        """
        # Ensure input2 is in the right format
        if input2.dim() == 3 and input2.size(0) != input1.size(0):
            # input2 is (seq_len, batch_size, role_dim), transpose to (batch_size, seq_len, role_dim)
            input2 = input2.transpose(0, 1)
        
        # Now both are (batch_size, seq_len, dim)
        # Compute outer product: (batch_size, filler_dim, role_dim)
        outer_product = torch.bmm(input1.transpose(1, 2), input2)
        
        # Flatten: (batch_size, filler_dim * role_dim)
        flattened = outer_product.view(outer_product.size(0), -1)
        
        return flattened


class RoleLearningTensorProductEncoder(nn.Module):
    """
    Tensor Product Encoder with learned role assignments.
    Uses RoleAssignmentLSTM to dynamically assign roles based on input.
    """
    
    def __init__(
        self,
        n_roles,
        n_fillers,
        filler_dim,
        role_dim,
        final_layer_width,
        pretrained_filler_embeddings=None,
        hidden_dim=32,
        num_layers=1,
        bidirectional=False,
        softmax_roles=True,
        binder="tpr",
        # Regularization weights
        one_hot_regularization_weight=0.1,
        l2_norm_regularization_weight=0.1,
        unique_role_regularization_weight=0.1,
        **kwargs  # Catch unused arguments
    ):
        super(RoleLearningTensorProductEncoder, self).__init__()
        
        self.n_roles = n_roles
        self.n_fillers = n_fillers
        self.filler_dim = filler_dim
        self.role_dim = role_dim
        self.final_layer_width = final_layer_width
        
        # Regularization parameters
        self.one_hot_regularization_weight = one_hot_regularization_weight
        self.l2_norm_regularization_weight = l2_norm_regularization_weight
        self.unique_role_regularization_weight = unique_role_regularization_weight
        self.regularize = False
        self.regularization_temp = 1.0
        
        # Create filler embedding
        if pretrained_filler_embeddings is not None:
            self.filler_embedding = nn.Embedding(n_fillers, filler_dim)
            self.filler_embedding.weight.data = torch.from_numpy(pretrained_filler_embeddings).float()
            self.filler_embedding.weight.requires_grad = False
            print("Using pretrained filler embeddings (frozen)")
        else:
            self.filler_embedding = nn.Embedding(n_fillers, filler_dim)
            print("Using random filler embeddings")
        
        # Import and create role assigner
        try:
            from role_assigner import RoleAssignmentLSTM
            
            self.role_assigner = RoleAssignmentLSTM(
                num_roles=n_roles,
                filler_embedding=self.filler_embedding,
                hidden_dim=hidden_dim,
                role_embedding_dim=role_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                softmax_roles=softmax_roles
            )
            print(f"Created LSTM role assigner (hidden_dim={hidden_dim}, softmax={softmax_roles})")
        except ImportError:
            print("ERROR: Could not import RoleAssignmentLSTM from role_assigner.py")
            print("Make sure role_assigner.py is in the same directory")
            raise
        
        # Create binding operation
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        else:
            raise ValueError(f"Only 'tpr' binder is currently supported, got {binder}")
        
        # Create final projection layer
        self.last_layer = nn.Linear(filler_dim * role_dim, final_layer_width)
        
        print(f"Model created: {n_fillers} fillers, {n_roles} roles")
        print(f"  Filler dim: {filler_dim}, Role dim: {role_dim}")
        print(f"  Output dim: {final_layer_width}")
    
    def forward(self, filler_list, role_list_not_used=None, filler_lengths=None):
        """
        Forward pass through the model.
        
        Args:
            filler_list: Tensor of shape (batch_size, seq_length) with filler IDs
            role_list_not_used: Ignored (roles are learned)
            filler_lengths: Optional sequence lengths for padding
        
        Returns:
            output: Final representation (batch_size, final_layer_width)
            role_predictions: Role assignment probabilities (seq_len, batch_size, n_roles)
        """
        # Get learned role embeddings from LSTM
        # roles_embedded: (seq_len, batch_size, role_dim)
        # role_predictions: (seq_len, batch_size, n_roles)
        roles_embedded, role_predictions = self.role_assigner(filler_list)
        
        # Get filler embeddings
        # fillers_embedded: (batch_size, seq_len, filler_dim)
        fillers_embedded = self.filler_embedding(filler_list)
        
        # Bind fillers and roles via tensor product
        # output: (batch_size, filler_dim * role_dim)
        output = self.sum_layer(fillers_embedded, roles_embedded)
        
        # Project to final dimensionality
        # output: (batch_size, final_layer_width)
        output = self.last_layer(output)
        
        return output, role_predictions
    
    def use_regularization(self, use_reg):
        """Enable or disable regularization"""
        self.regularize = use_reg
    
    def set_regularization_temp(self, temp):
        """Set the temperature for regularization"""
        self.regularization_temp = temp
    
    def get_regularization_loss(self, role_predictions):
        """
        Compute regularization losses to encourage discrete role assignments.
        
        Args:
            role_predictions: (seq_len, batch_size, n_roles)
        
        Returns:
            one_hot_loss: Encourages predictions to be close to one-hot
            l2_loss: Encourages L2 norm constraints
            unique_loss: Encourages different roles for different positions
        """
        if not self.regularize:
            return 0, 0, 0
        
        temp = self.regularization_temp
        batch_size = role_predictions.shape[1]
        
        # Encourage one-hot vectors: w * (1 - w) should be small
        # When w is 0 or 1, this term is 0
        one_hot_reg = torch.sum(role_predictions * (1 - role_predictions))
        one_hot_loss = temp * one_hot_reg / batch_size
        
        # Encourage proper L2 normalization
        l2_norm = -torch.sum(role_predictions * role_predictions)
        l2_norm_loss = temp * l2_norm / batch_size
        
        # Encourage unique role assignments across sequence
        # Sum predictions across sequence: (batch_size, n_roles)
        exclusive_role_vector = torch.sum(role_predictions, 0)
        # This should also be close to one-hot (each role used once)
        unique_role_loss = temp * torch.sum(
            (exclusive_role_vector * (1 - exclusive_role_vector)) ** 2
        ) / batch_size
        
        return (
            self.one_hot_regularization_weight * one_hot_loss,
            self.l2_norm_regularization_weight * l2_norm_loss,
            self.unique_role_regularization_weight * unique_role_loss
        )
    
    def get_role_assignments(self, filler_list):
        """
        Get role assignments without full forward pass.
        Useful for analysis.
        """
        with torch.no_grad():
            _, role_predictions = self.role_assigner(filler_list)
            return role_predictions
    
    def train(self, mode=True):
        """Override train to handle role prediction snapping"""
        super().train(mode)
        if mode:
            # During training, use soft attention
            self.role_assigner.snap_one_hot_predictions = False
        else:
            # During evaluation, snap to discrete roles
            self.role_assigner.snap_one_hot_predictions = True
    
    def eval(self):
        """Override eval for consistency"""
        self.train(False)
