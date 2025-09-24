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
        use_positional_encoding=True,
        gumbel_temperature=1.0,
        gumbel_hard=False
    ):
        super(RoleAssignmentTransformer, self).__init__()
        self.snap_one_hot_predictions = False
        self.filler_embedding = filler_embedding
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_hard = gumbel_hard

        if use_gumbel_softmax:
            print(f"Using Gumbel Softmax with temperature={gumbel_temperature}, hard={gumbel_hard}")
            
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
        if self.use_gumbel_softmax:
            gumbel_predictions = self.gumbel_softmax(role_predictions, temperature=self.gumbel_temperature, hard=self.gumbel_hard)
            roles = torch.matmul(gumbel_predictions, role_embeddings)
        elif self.snap_one_hot_predictions:
            one_hot_predictions = self.one_hot_embedding(torch.argmax(role_predictions, 2), self.num_roles)
            roles = torch.matmul(one_hot_predictions, role_embeddings)
        else:
            roles = torch.matmul(role_predictions, role_embeddings)
        return roles, role_predictions

    def gumbel_softmax(self, logits, temperature=1.0, hard=False, dim=1):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_logits = (logits + gumbel_noise) / temperature
        soft_sample = torch.softmax(gumbel_logits, dim=dim)
        if hard:
            hard_sample = torch.zeros_like(soft_sample)
            hard_sample.scatter_(dim, soft_sample.argmax(dim=dim, keepdim=True), 1.0)
            soft_sample = hard_sample = soft_sample.detach() + soft_sample
        return soft_sample

    def set_gumbel_temperature(self, temperature):
        self.gumbel_temperature = temperature

    def set_gumbel_hard(self, hard):
        self.gumbel_hard = hard

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
        use_positional_encoding=True,
        use_gumbel_softmax=True,
        gumbel_temperature=1.0,
        gumbel_hard=False
    )
    print("Testing with padded sequences:")
    data = [[2, 3, 10, 10], [1, 10, 10, 10]]
    filler_lengths = [2, 1]
    data_tensor = torch.tensor(data)
    padding_mask = transformer.create_padding_mask(data_tensor, filler_lengths)
    roles, role_predictions = transformer(data_tensor, src_key_padding_mask=padding_mask)
    print("Roles shape:", roles.shape)
    print("Role predictions shape:", role_predictions.shape)
    print("Role predictions for first sequence, first position:", role_predictions[0, 0, :])
    print('\nTesting Gumbel temperature annealing:')
    temperatures = [2.0, 1.0, 0.5, 0.1]
    for temp in temperatures:
        transformer.set_gumbel_temperature(temp)
        roles_temp, _ = transformer(data_tensor, src_key_padding_mask=padding_mask)
        print(f"Temperature {temp}: Gumbel output entropy = {-torch.sum(torch.softmax(role_predictions[0, 0, :], dim=0) * torch.log_softmax(role_predictions[0, 0, :], dim=0)):.3f}")
    print('\nTesting hard vs soft Gumbel:')
    transformer.set_gumbel_temperature(0.5)
    transformer.set_gumbel_hard(False)
    roles_soft, _ = transformer(data_tensor, src_key_padding_mask=padding_mask)
    transformer.set_gumbel_hard(True)
    roles_hard, _ = transformer(data_tensor, src_key_padding_mask=padding_mask)
    print("Soft Gumbel max value:", torch.max(roles_soft).item())
    print("Hard Gumbel max value:", torch.max(roles_hard).item())
    print('\nTesting with single sequence:')
    data2 = [[1, 10, 10, 10]]
    filler_lengths2 = [1]
    data_tensor2 = torch.tensor(data2)
    padding_mask2 = transformer.create_padding_mask(data_tensor2, filler_lengths2)
    roles2, role_predictions2 = transformer(data_tensor2, src_key_padding_mask=padding_mask2)
    print("Roles shape:", roles2.shape)
    print("Role predictions shape:", role_predictions2.shape)
