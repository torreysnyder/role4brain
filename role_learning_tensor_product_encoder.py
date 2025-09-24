from __future__ import division

import torch
import torch.nn as nn

from binding_operations import CircularConvolution, EltWise, SumFlattenedOuterProduct
from .role_assigner import RoleAssignmentTransformer

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
            d_model=64,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1
            role_learner_hidden_dim=20,
            role_assignment_shrink_filler_dim=None,
            use_positional_encoding=True,
            use_gumbel_softmax=True,
            gumbel_temperature=1.0,
            gumbel_hard=False,
            bidirectional=False,
            num_layers=1,
            softmax_roles=False,
            pretrained_embeddings=None,
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

        # Create an embedding layer for the fillers
        if embedder_squeeze is None:
            self.filler_embedding = nn.Embedding(
                self.n_fillers,
                self.filler_dim,
            )
            self.embed_squeeze = False
            print("no squeeze")
        else:
            self.embed_squeeze = True
            self.filler_embedding = nn.Embedding(
                self.n_fillers,
                embedder_squeeze,
            )
            self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)
            print("squeeze")

        if pretrained_embeddings is not None:
            self.filler_embedding.load_state_dict(
                {'weight': torch.FloatTensor(pretrained_embeddings).cuda()})
            self.filler_embedding.weight.requires_grad = False

        if pretrained_filler_embeddings:
            print('Using pretrained filler embeddings')
            self.filler_embedding.load_state_dict(
                torch.load(pretrained_filler_embeddings, map_location=device)
            )
            self.filler_embedding.weight.requires_grad = False

        self.role_assigner = RoleAssignmentTransformer(
            num_roles=self.n_roles,
            filler_embedding=self.filler_embedding,
            d_model=d_model
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
    def forward(self, filler_list, role_list_not_used):
        # Embed the fillers
        fillers_embedded = self.filler_embedding(filler_list)

        if self.embed_squeeze:
            fillers_embedded = self.embedding_squeeze_layer(fillers_embedded)

        roles_embedded, role_predictions = self.role_assigner(filler_list)
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
        if hasattr(self.role_assigner, 'set_gumbel_temperature'):
            self.role_assigner.set_gumbel_temperature(temperature)

    def set_gumbel_hard(self, hard):
        if hasattr(self.role_assigner, 'set_gumbel_hard'):
            self.role_assigner.set_gumbel_hard(hard)
            
    def get_regularization_loss(self, role_predictions):
        if not self.regularize:
            return 0, 0, 0

        one_hot_temperature = self.regularization_temp
        batch_size = role_predictions.shape[1]

        softmax_roles = self.role_assigner.softmax_roles

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
        return self.one_hot_regularization_weight * one_hot_loss,\
               self.l2_norm_regularization_weight * l2_norm_loss,\
               self.unique_role_regularization_weight * unique_role_loss

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if self.use_gumbel_softmax:
                self.role_assigner.snap_one_hot_predictions = False
            else:
                self.role_assigner.snap_one_hot_predictions = False
        else:
            if self.use_gumbel_softmax:
                pass
            else:
                self.role_assigner.snap_one_hot_predictions = True

    def eval(self):
        self.train(False)

    def get_role_assignments(self, filler_list, filler_lengths=None):
        with torch.no_grad():
            src_key_padding_mask = None
            if filler_lengths is not None:
                src_key_padding_mask = self.role_assigner.create_padding_mask(filler_list, filler_lengths)
            _, role_predictions = self.role_assigner(filler_list, src_key_padding_mask)
            return role_predictions

    def anneal_gumbel_temperature(self, epoch, initial_temp=2.0, final_temp=0.1, decay_rate=0.95):
        if self.use_gumbel_softmax:
            temp = max(final_temp, initial_temp * (decay_rate ** epoch))
            self.set_gumbel_temperature(temp)
            return temp
        return None
