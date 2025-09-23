import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class RoleAssignmentTransformer(nn.Module):
    def __init__(
        self,
        num_roles,
        filler_embedding,
        d_model,
        role_embedding_dim,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        role_assignment_shrink_filler_dim=None,
        softmax_roles=False,
        use_positional_encoding=True
    ):
        super(RoleAssignmentTransformer, self).__init__()
        self.snap_one_hot_predictions = False
        self.filler_embedding = filler_embedding
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        filler_embedding_dim = filler_embedding.embedding_dim
        self.shrink_filler = False
        if role_assignment_shrink_filler_dim:
            self.shrink_filler = True
            self.filler_shrink_layer = nn.Linear(filler_embedding.embedding_dim, role_assignment_shrink_filler_dim)
            filler_embedding_dim = role_assignment_shrink_filler_dim
        if filler_embedding_dim != d_model:
            self.input_projection = nn.Linear(filler_embedding_dim, d_model)
        else:
            self.input_projection = nn.Identity()
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.num_roles = num_roles
        self.role_weight_predictions = nn.Linear(d_model, num_roles)
        self.softmax_roles = softmax_roles
        if softmax_roles:
            print("Use softmax for role predictions")
            self.softmax = nn.Softmax(dim=2)
        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        self.role_indices = torch.tensor([x for x in range(num_roles)], device=device)
    
    def forward(self, filler_tensor, src_key_padding_mask=None):
        batch_size, seq_length = filler_tensor.shape
        fillers_embedded = self.filler_embedding(filler_tensor)
        if self.shrink_filler:
            fillers_embedded = self.filler_shrink_layer(fillers_embedded)
        fillers_embedded = self.input_projection(fillers_embedded)
        fillers_embedded = torch.transpose(fillers_embedded, 0, 1)
        if self.use_positional_encoding:
            fillers_embedded = self.pos_encoder(fillers_embedded)
        transformer_out = self.transformer_encoder(fillers_embedded, src_key_padding_mask=src_key_padding_mask)
        role_predictions = self.role_weight_predictions(transformer_out)
        if self.softmax_roles:
            role_predictions = self.softmax(role_predictions)
        role_embeddings = self.role_embeddings(self.role_indices)
        role_embeddings = role_embeddings/ torch.norm(role_embeddings, dim=1).unsqueeze(1)
        if self.snap_one_hot_predictions:
            one_hot_predictions = self.one_hot_embedding(torch.argmax(role_predictions, 2), self.num_roles)
            roles = torch.matmul(one_hot_predictions, role_embeddings)
        else:
            roles = torch.matmul(role_predictions, role_embeddings)
        return roles, role_predictions

    def create_padding_mask(self, filler_tensor, filler_lengths):
        batch_size, seq_length = filler_tensor.shape
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        for i, length in enumerate(filler_lengths):
            if length < seq_length:
                mask[i, length:] = True
        return mask
        
    def one_hot_embedding(self, labels, num_classes):
        y = torch.eye(num_classes, device=device)
        return y[labels]
        
if __name__ == "__main__":
    num_roles = 10
    filler_embedding_dim = 20
    num_fillers = 10
    d_model = 64
    role_embedding_dim = 20
    filler_embedding = torch.nn.Embedding(num_fillers + 1, filler_embedding_dim, padding_idx=num_fillers)
    transformer = RoleAssignmentTransformer(
        num_roles=num_roles,
        filler_embedding=filler_embedding,
        d_model=d_model,
        role_embedding_dim=role_embedding_dim,
        nhead=8,
        num_encounter_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        softmax_roles=True,
        use_positional_encoding=True
    )
    print("Testing with padded sequences:")
    data = [[2, 3, 10, 10], [1, 10, 10, 10]]
    filler_lengths = [2, 1]
    data_tensor = torch.tensor(data)
    padding_mask = transformer.create_padding_mask(data_tensor, filler_lengths)
    roles 
