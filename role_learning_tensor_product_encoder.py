from __future__ import division

import torch
import torch.nn as nn
import numpy as np

from binding_operations import CircularConvolution, EltWise, SumFlattenedOuterProduct
from .role_assigner_transformer import RoleAssignmentTransformer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# A tensor product encoder layer
# Takes a list of fillers and a list of roles and returns an encoding
class RoleLearningTensorProductEncoder(nn.Module):
    def __init__(
            self,
            n_roles=2,
            n_fillers=2,
            filler_dim=3,
            role_dim=4,
            final_layer_width=None,
            pretrained_filler_embeddings=None,
            embedder_squeeze=None,
            binder="tpr",
            # Transformer-specific parameters
            d_model=64,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            role_assignment_shrink_filler_dim=None,
            use_positional_encoding=True,
            # Gumbel Softmax parameters
            use_gumbel_softmax=True,
            gumbel_temperature=1.0,
            gumbel_hard=False,
            # Legacy parameters (kept for compatibility)
            bidirectional=False,  # Not used in transformer
            num_layers=1,  # Replaced by num_encoder_layers
            softmax_roles=False,
            # TPDN embedding parameters
            tpdn_model_path=None,
            tpdn_embeddings=None,
            freeze_embeddings=True,
            # Regularization parameters
            one_hot_regularization_weight=1.0,
            l2_norm_regularization_weight=1.0,
            unique_role_regularization_weight=1.0,
    ):

        super(RoleLearningTensorProductEncoder, self).__init__()

        self.n_roles = n_roles  # number of roles
        self.n_fillers = n_fillers  # number of fillers

        self.one_hot_regularization_weight = one_hot_regularization_weight
        self.l2_norm_regularization_weight = l2_norm_regularization_weight
        self.unique_role_regularization_weight = unique_role_regularization_weight
        self.regularize = False

        # Set the dimension for the filler embeddings
        self.filler_dim = filler_dim

        # Set the dimension for the role embeddings
        self.role_dim = role_dim
        
        # Store embedding configuration
        self.freeze_embeddings = freeze_embeddings

        # Create an embedding layer for the fillers with TPDN support
        self._initialize_filler_embeddings(
            tpdn_model_path, tpdn_embeddings, embedder_squeeze
        )

        # Create the Transformer-based role assigner
        self.role_assigner = RoleAssignmentTransformer(
            num_roles=self.n_roles,
            filler_embedding=self.filler_embedding,
            d_model=d_model,
            role_embedding_dim=self.role_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            role_assignment_shrink_filler_dim=role_assignment_shrink_filler_dim,
            softmax_roles=softmax_roles,
            use_positional_encoding=use_positional_encoding,
            use_gumbel_softmax=use_gumbel_softmax,
            gumbel_temperature=gumbel_temperature,
            gumbel_hard=gumbel_hard
        )

        # Store whether we're using Gumbel Softmax for later reference
        self.use_gumbel_softmax = use_gumbel_softmax

        # Create a SumFlattenedOuterProduct layer that will
        # take the sum flattened outer product of the filler
        # and role embeddings (or a different type of role-filler
        # binding function, such as circular convolution)
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        elif binder == "hrr":
            self.sum_layer = CircularConvolution(self.filler_dim)
        elif binder == "eltwise" or binder == "elt":
            self.sum_layer = EltWise()
        else:
            print("Invalid binder")

        # This final part if for including a final linear layer that compresses
        # the sum flattened outer product into the dimensionality you desire
        # But if self.final_layer_width is None, then no such layer is used
        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            if binder == "tpr":
                self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
            else:
                self.last_layer = nn.Linear(self.filler_dim, self.final_layer_width)

    # Function for a forward pass through this layer. Takes a list of fillers and
    # a list of roles and returns an single vector encoding it.
    def forward(self, filler_list, role_list_not_used=None, filler_lengths=None):
        """
        :param filler_list: Tensor of shape (batch_size, sequence_length)
        :param role_list_not_used: Not used (roles are learned by the transformer)
        :param filler_lengths: Optional list of actual sequence lengths for padding mask
        """
        # Embed the fillers
        fillers_embedded = self.filler_embedding(filler_list)

        if self.embed_squeeze:
            fillers_embedded = self.embedding_squeeze_layer(fillers_embedded)

        # Create padding mask if filler_lengths is provided
        src_key_padding_mask = None
        if filler_lengths is not None:
            src_key_padding_mask = self.role_assigner.create_padding_mask(filler_list, filler_lengths)

        # Get role embeddings from the transformer
        roles_embedded, role_predictions = self.role_assigner(filler_list, src_key_padding_mask)
        
        # Transpose to match expected shape: (batch_size, sequence_length, role_dim)
        roles_embedded = roles_embedded.transpose(0, 1)

        # Create the sum of the flattened tensor products of the
        # filler and role embeddings
        output = self.sum_layer(fillers_embedded, roles_embedded)

        # If there is a final linear layer to change the output's dimensionality, apply it
        if self.has_last:
            output = self.last_layer(output)

        return output, role_predictions

    def use_regularization(self, use_regularization):
        self.regularize = use_regularization

    def set_regularization_temp(self, temp):
        self.regularization_temp = temp

    def set_gumbel_temperature(self, temperature):
        """Set the Gumbel Softmax temperature for gradual annealing"""
        if hasattr(self.role_assigner, 'set_gumbel_temperature'):
            self.role_assigner.set_gumbel_temperature(temperature)

    def set_gumbel_hard(self, hard):
        """Switch between hard and soft Gumbel sampling"""
        if hasattr(self.role_assigner, 'set_gumbel_hard'):
            self.role_assigner.set_gumbel_hard(hard)

    def get_regularization_loss(self, role_predictions):
        if not self.regularize:
            return 0, 0, 0

        one_hot_temperature = self.regularization_temp
        batch_size = role_predictions.shape[1]

        # Check if we're using softmax roles (either traditional or Gumbel)
        softmax_roles = self.role_assigner.softmax_roles or self.use_gumbel_softmax

        if softmax_roles:
            # For RoleLearningTensorProductEncoder, we encourage one hot vector weight predictions
            # by regularizing the role_predictions by `w * (1 - w)`
            one_hot_reg = torch.sum(role_predictions * (1 - role_predictions))
        else:
            one_hot_reg = torch.sum((role_predictions ** 2) * (1 - role_predictions) ** 2)
        one_hot_loss = one_hot_temperature * one_hot_reg / batch_size

        if softmax_roles:
            l2_norm = -torch.sum(role_predictions * role_predictions)
        else:
            l2_norm = (torch.sum(role_predictions ** 2) - 1) ** 2
        l2_norm_loss = one_hot_temperature * l2_norm / batch_size

        # We also want to encourage the network to assign each filler in a sequence to a
        # different role. To encourage this, we sum the vector predictions across a sequence
        # (call this vector w) and add `(w * (1 - w))^2` to the loss function.
        exclusive_role_vector = torch.sum(role_predictions, 0)
        unique_role_loss = one_hot_temperature * torch.sum(
            (exclusive_role_vector * (1 - exclusive_role_vector)) ** 2) / batch_size
        
        return (self.one_hot_regularization_weight * one_hot_loss,
                self.l2_norm_regularization_weight * l2_norm_loss,
                self.unique_role_regularization_weight * unique_role_loss)

    def train(self, mode=True):
        """Override train method to handle Gumbel vs snap one-hot behavior"""
        super().train(mode)
        
        if mode:  # Training mode
            if self.use_gumbel_softmax:
                # In Gumbel mode, we don't use snap_one_hot_predictions
                self.role_assigner.snap_one_hot_predictions = False
            else:
                # In traditional mode, use soft attention during training
                self.role_assigner.snap_one_hot_predictions = False
        else:  # Evaluation mode
            if self.use_gumbel_softmax:
                # In Gumbel mode, optionally switch to hard sampling
                # This could be controlled by a separate flag if needed
                pass  # Gumbel handles hard/soft via its own parameters
            else:
                # In traditional mode, snap to one-hot during evaluation
                self.role_assigner.snap_one_hot_predictions = True

    def eval(self):
        """Override eval method for consistency"""
        self.train(False)

    def get_role_assignments(self, filler_list, filler_lengths=None):
        """
        Get the role assignments for a given input without computing the full forward pass
        Useful for analysis and visualization
        """
        with torch.no_grad():
            src_key_padding_mask = None
            if filler_lengths is not None:
                src_key_padding_mask = self.role_assigner.create_padding_mask(filler_list, filler_lengths)
            
            _, role_predictions = self.role_assigner(filler_list, src_key_padding_mask)
            return role_predictions

    def anneal_gumbel_temperature(self, epoch, initial_temp=2.0, final_temp=0.1, decay_rate=0.95):
        """
        Convenience method for annealing Gumbel temperature during training
        
        Args:
            epoch: Current training epoch
            initial_temp: Starting temperature (higher = softer)
            final_temp: Final temperature (lower = harder)
            decay_rate: Exponential decay rate
        """
        if self.use_gumbel_softmax:
            temp = max(final_temp, initial_temp * (decay_rate ** epoch))
            self.set_gumbel_temperature(temp)
            return temp
        return None
    
    def _initialize_filler_embeddings(self, tpdn_model_path, tpdn_embeddings, embedder_squeeze):
        """
        Initialize filler embeddings from a trained TPDN model
        
        Args:
            tpdn_model_path: Path to saved TPDN model state dict
            tpdn_embeddings: Pre-extracted TPDN embeddings (numpy array)
            embedder_squeeze: Optional dimension to squeeze embeddings to
        """
        if tpdn_embeddings is not None:
            # Use provided TPDN embeddings
            embedding_dim = tpdn_embeddings.shape[1]
            print(f"Using TPDN embeddings with dimension {embedding_dim}")
            
            if embedder_squeeze is None:
                # Direct use of TPDN embeddings
                self.filler_embedding = nn.Embedding(self.n_fillers, embedding_dim)
                self.embed_squeeze = False
                
                # Load the TPDN weights
                self.filler_embedding.weight.data = torch.from_numpy(tpdn_embeddings).float()
                
                if self.freeze_embeddings:
                    self.filler_embedding.weight.requires_grad = False
                    print("TPDN embeddings frozen")
                else:
                    print("TPDN embeddings trainable")
                
                # Update filler_dim to match embedding dimension
                self.filler_dim = embedding_dim
                    
            else:
                # Use embedder squeeze with TPDN embeddings
                self.embed_squeeze = True
                self.filler_embedding = nn.Embedding(self.n_fillers, embedding_dim)
                self.embedding_squeeze_layer = nn.Linear(embedding_dim, self.filler_dim)
                
                # Load TPDN embeddings
                self.filler_embedding.weight.data = torch.from_numpy(tpdn_embeddings).float()
                
                if self.freeze_embeddings:
                    self.filler_embedding.weight.requires_grad = False
                    print(f"TPDN embeddings frozen with squeeze to {self.filler_dim} dimensions")
                else:
                    print(f"TPDN embeddings trainable with squeeze to {self.filler_dim} dimensions")
                
        elif tpdn_model_path is not None:
            # Load TPDN embeddings from saved model
            print(f"Loading TPDN embeddings from model: {tpdn_model_path}")
            try:
                # Load the saved state dict
                state_dict = torch.load(tpdn_model_path, map_location=device)
                
                # Extract filler embeddings from the TPDN model
                # The TPDN model has either 'filler_embed.weight' or 
                # 'filler_embed.0.weight' (if using Sequential with linear layer)
                if 'filler_embed.weight' in state_dict:
                    tpdn_embed_weights = state_dict['filler_embed.weight'].cpu().numpy()
                elif 'filler_embed.0.weight' in state_dict:
                    tpdn_embed_weights = state_dict['filler_embed.0.weight'].cpu().numpy()
                else:
                    raise KeyError("Could not find filler embeddings in TPDN model state dict")
                
                embedding_dim = tpdn_embed_weights.shape[1]
                print(f"Extracted TPDN embeddings with dimension {embedding_dim}")
                
                # Initialize using the extracted embeddings
                self._initialize_filler_embeddings(None, tpdn_embed_weights, embedder_squeeze)
                
            except Exception as e:
                print(f"Warning: Failed to load TPDN model {tpdn_model_path}: {e}")
                print("Falling back to random embeddings")
                self._initialize_random_embeddings(embedder_squeeze)
                
        else:
            # Use random embeddings (original behavior)
            self._initialize_random_embeddings(embedder_squeeze)
    
    def _initialize_random_embeddings(self, embedder_squeeze):
        """Initialize with random embeddings (original behavior)"""
        if embedder_squeeze is None:
            self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
            self.embed_squeeze = False
            print("Using random embeddings, no squeeze")
        else:
            self.embed_squeeze = True
            self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
            self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)
            print(f"Using random embeddings with squeeze to {self.filler_dim} dimensions")
    
    @classmethod
    def from_tpdn_model(cls, tpdn_model_path, **kwargs):
        """
        Convenient class method to create encoder with TPDN embeddings
        
        Args:
            tpdn_model_path: Path to saved TPDN model state dict
            **kwargs: Other initialization parameters
        """
        return cls(tpdn_model_path=tpdn_model_path, **kwargs)
    
    @classmethod
    def extract_tpdn_embeddings(cls, tpdn_model_path, n_fillers, filler_dim):
        """
        Extract filler embeddings from a trained TPDN model
        
        Args:
            tpdn_model_path: Path to saved TPDN model state dict
            n_fillers: Number of filler tokens
            filler_dim: Expected dimension of filler embeddings
            
        Returns:
            numpy array of shape (n_fillers, filler_dim) containing the embeddings
        """
        try:
            state_dict = torch.load(tpdn_model_path, map_location=device)
            
            # Extract filler embeddings
            if 'filler_embed.weight' in state_dict:
                embeddings = state_dict['filler_embed.weight'].cpu().numpy()
            elif 'filler_embed.0.weight' in state_dict:
                embeddings = state_dict['filler_embed.0.weight'].cpu().numpy()
            else:
                raise KeyError("Could not find filler embeddings in TPDN model")
            
            print(f"Extracted embeddings with shape {embeddings.shape}")
            
            # Verify dimensions match
            if embeddings.shape[0] != n_fillers:
                print(f"Warning: Embedding vocab size {embeddings.shape[0]} != expected {n_fillers}")
            
            return embeddings
            
        except Exception as e:
            print(f"Error extracting TPDN embeddings: {e}")
            return None
