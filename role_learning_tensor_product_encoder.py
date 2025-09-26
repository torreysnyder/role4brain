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
            # BERT embedding parameters
            pretrained_embeddings=None,
            bert_model_name=None,
            freeze_embeddings=True,
            untrained=False,
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
        
        # Store BERT embedding configuration
        self.freeze_embeddings = freeze_embeddings
        self.untrained = untrained

        # Create an embedding layer for the fillers with BERT support
        self._initialize_filler_embeddings(
            pretrained_embeddings, bert_model_name, embedder_squeeze
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
    def forward(self, filler_list, role_list_not_used, filler_lengths=None):
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
    
    def _initialize_filler_embeddings(self, pretrained_embeddings, bert_model_name, embedder_squeeze):
        """
        Initialize filler embeddings with support for BERT pretrained embeddings
        """
        if pretrained_embeddings is not None:
            # Use provided pretrained embeddings (BERT-style initialization)
            embedding_dim = pretrained_embeddings.shape[1]
            print(f"Using pretrained embeddings with dimension {embedding_dim}")
            
            if embedder_squeeze is None:
                # Direct use of pretrained embeddings
                self.filler_embedding = nn.Embedding(self.n_fillers, embedding_dim)
                self.embed_squeeze = False
                
                if not self.untrained:
                    # Load the pretrained weights
                    self.filler_embedding.weight.data = torch.from_numpy(pretrained_embeddings).float()
                
                if self.freeze_embeddings:
                    self.filler_embedding.weight.requires_grad = False
                    print("Pretrained embeddings frozen")
                else:
                    # Add a linear layer to allow fine-tuning while preserving pretrained knowledge
                    frozen_embed = nn.Embedding(self.n_fillers, embedding_dim)
                    frozen_embed.weight.data = torch.from_numpy(pretrained_embeddings).float()
                    frozen_embed.weight.requires_grad = False
                    
                    self.filler_embedding = nn.Sequential(
                        frozen_embed,
                        nn.Linear(embedding_dim, self.filler_dim)
                    )
                    print("Pretrained embeddings with trainable linear layer")
                
                # Update filler_dim to match embedding dimension if no squeeze
                if embedder_squeeze is None:
                    self.filler_dim = embedding_dim
                    
            else:
                # Use embedder squeeze with pretrained embeddings
                self.embed_squeeze = True
                self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
                self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)
                
                if not self.untrained:
                    # For squeeze, we need to project pretrained embeddings to squeeze dimension
                    pretrained_projected = torch.from_numpy(pretrained_embeddings).float()
                    if pretrained_projected.size(1) != embedder_squeeze:
                        # Create a projection layer to match dimensions
                        proj_layer = nn.Linear(pretrained_projected.size(1), embedder_squeeze)
                        with torch.no_grad():
                            projected_embeddings = proj_layer(pretrained_projected)
                        self.filler_embedding.weight.data = projected_embeddings
                    else:
                        self.filler_embedding.weight.data = pretrained_projected
                
                if self.freeze_embeddings:
                    self.filler_embedding.weight.requires_grad = False
                
                print(f"Using pretrained embeddings with squeeze to {embedder_squeeze} dimensions")
                
        elif bert_model_name is not None:
            # Load BERT embeddings directly from model name
            print(f"Loading embeddings from BERT model: {bert_model_name}")
            try:
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
                bert_model = AutoModel.from_pretrained(bert_model_name)
                
                # Extract embedding weights
                bert_embeddings = bert_model.embeddings.word_embeddings.weight.detach().numpy()
                embedding_dim = bert_embeddings.shape[1]
                
                # Initialize with BERT embeddings
                self._initialize_filler_embeddings(bert_embeddings, None, embedder_squeeze)
                print(f"Successfully loaded BERT embeddings with dimension {embedding_dim}")
                
            except ImportError:
                print("Warning: transformers library not available. Using random embeddings.")
                self._initialize_random_embeddings(embedder_squeeze)
            except Exception as e:
                print(f"Warning: Failed to load BERT model {bert_model_name}: {e}")
                self._initialize_random_embeddings(embedder_squeeze)
                
        else:
            # Use random embeddings (original behavior)
            self._initialize_random_embeddings(embedder_squeeze)
            
        # Handle legacy pretrained_filler_embeddings parameter
        if hasattr(self, '_legacy_pretrained_filler_embeddings') and self._legacy_pretrained_filler_embeddings:
            print('Loading legacy pretrained filler embeddings from file')
            try:
                self.filler_embedding.load_state_dict(
                    torch.load(self._legacy_pretrained_filler_embeddings, map_location=device)
                )
                self.filler_embedding.weight.requires_grad = False
            except Exception as e:
                print(f"Warning: Failed to load legacy pretrained embeddings: {e}")
    
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
    def from_bert_model(cls, bert_model_name, tokenizer, id2token, **kwargs):
        """
        Convenient class method to create encoder with BERT embeddings
        
        Args:
            bert_model_name: Name of BERT model (e.g., 'bert-base-uncased')
            tokenizer: BERT tokenizer
            id2token: Dictionary mapping token IDs to tokens
            **kwargs: Other initialization parameters
        """
        try:
            from transformers import AutoModel
            
            # Load BERT model
            bert_model = AutoModel.from_pretrained(bert_model_name)
            
            # Extract embedding weights and create mapping
            bert_weights = bert_model.embeddings.word_embeddings.weight.detach().numpy()
            n_fillers = len(id2token)
            embedding_dim = bert_weights.shape[1]
            
            # Create pretrained embeddings array matching our vocabulary
            pretrained_embeddings = np.random.rand(n_fillers, embedding_dim)
            bert_token_ids = tokenizer.convert_tokens_to_ids([id2token[i] for i in range(n_fillers)])
            
            for i, bert_id in enumerate(bert_token_ids):
                if bert_id < len(bert_weights):
                    pretrained_embeddings[i] = bert_weights[bert_id]
            
            print(f"Created BERT embeddings mapping for {n_fillers} tokens")
            
            # Set default parameters for BERT usage
            if 'filler_dim' not in kwargs:
                kwargs['filler_dim'] = embedding_dim
            if 'n_fillers' not in kwargs:
                kwargs['n_fillers'] = n_fillers
                
            return cls(pretrained_embeddings=pretrained_embeddings, **kwargs)
            
        except ImportError:
            print("Warning: transformers library not available")
            return cls(**kwargs)
        except Exception as e:
            print(f"Warning: Failed to load BERT embeddings: {e}")
            return cls(**kwargs)
