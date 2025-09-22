import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class EmbeddingWithProjection(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.projection = nn.Linear(self.d_embed, self.d_model)
        self.scaling = float(math.sqrt(self.d_model))
        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Drouput(p=dropout)
    
    @staticmethod
    def create_positional_encoding(seq_length, d_model, batch_size=1):
        position = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        return pe

    def forward(self, x):
        assert x.dtype == torch.long, f"Input tensor must have dtype torch.long, got {x.dtype}"
        batch_size, seq_length = x.size()
        token_embedding = self.embedding(x)
        token_embedding = self.projection(token_embedding) * self.scaling
        positional_encoding = self.create_positional_encoding(seq_length, self.d_model, batch_size)
        normalized_sum = self.layernorm(token_embedding + positional_encoding)
        final_output = self.droput(normalized_sum)
        return final output

class TransformerAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model//num_head
        self.dropout_rate = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scaler = float(1.0 / math.sqrt(self.d_head))

    def forward(self, sequence, key_value_states=None, att_mask=None):
        batch_size, seq_len, model_dim = sequence.size()
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
                f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"
        is_cross_attention = key_value_states is not None
        Q_state = self.q_proj(sequence)
        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_len
            K_state = self.k_proj(sequence)
            V_state = self.v_proj(sequence)
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2)
        K_state = K_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)
        Q_state = Q_state * self.scaler
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2))
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")
        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask
        att_score = F.softmax(self.att_matrix, dim=-1)
        att_score = self.dropout(att_score)
        att_output = torch.matmul(att_score, V_state)
        att_output = att_output.transpose(1,2)
        att_output = att_output.contiguous().view(batch_size, seq_len, self.num_head*self.d_head)
        att_output = self.output_proj(att_output)
        assert att_output.size() == (batch_size, seq_len, self.d_model), \
            f"Final output shape {att_output.size()} incorrect"
        return att_output

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.fc2 = nn.Linear(self.d_ff, slef.d_model, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    def forward(self, input):
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"
        f1 = F.relu(self.fc1(input))
        f2 = self.fc2(f1)
        return f2
class RoleAssignmentLSTM(nn.Module):
    def __init__(
            self,
            num_roles,
            filler_embedding,
            hidden_dim,
            role_embedding_dim,
            num_layers=1,
            role_assignment_shrink_filler_dim=None,
            bidirectional=False,
            softmax_roles=False
    ):
        super(RoleAssignmentLSTM, self).__init__()
        # TODO: when we move to language models, we will need to use pre-trained word embeddings.
        # See embedder_squeeze in TensorProductEncoder

        self.snap_one_hot_predictions = False

        self.filler_embedding = filler_embedding
        filler_embedding_dim = filler_embedding.embedding_dim

        self.shrink_filler = False
        if role_assignment_shrink_filler_dim:
            self.shrink_filler = True
            self.filler_shrink_layer = nn.Linear(filler_embedding.embedding_dim,
                                                 role_assignment_shrink_filler_dim)
            filler_embedding_dim = role_assignment_shrink_filler_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_roles = num_roles
        self.bidirectional = bidirectional

        # OPTION we may want the LSTM to be bidirectional for things like RTL roles.
        # Also, should the output size be the number of roles for the weight vector?
        # Or is the output of variable size and we apply a linear transformation
        # to get the weight vector?
        self.lstm = nn.LSTM(filler_embedding_dim, hidden_dim, bidirectional=bidirectional,
                            num_layers=self.num_layers)
        if bidirectional:
            print("The role assignment LSTM is bidirectional")
            self.role_weight_predictions = nn.Linear(hidden_dim * 2, num_roles)
        else:
            self.role_weight_predictions = nn.Linear(hidden_dim, num_roles)

        self.softmax_roles = softmax_roles
        if softmax_roles:
            print("Use softmax for role predictions")
            # The output of role_weight_predictions is shape
            # (sequence_length, batch_size, num_roles)
            # We want to softmax across the roles so set dim=2
            self.softmax = nn.Softmax(dim=2)

        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        self.role_indices = torch.tensor([x for x in range(num_roles)], device=device)

    def forward(self, filler_tensor):
        """
        :param filler_tensor: This input tensor should be of shape (batch_size, sequence_length)
        :param filler_lengths: A list of the length of each sequence in the batch. This is used
            for padding the sequences.
        :return: A tensor of size (sequence_length, batch_size, role_embedding_dim) with the role
            embeddings for the input filler_tensor.
        """
        batch_size = len(filler_tensor)
        hidden = self.init_hidden(batch_size)

        fillers_embedded = self.filler_embedding(filler_tensor)
        if self.shrink_filler:
            fillers_embedded = self.filler_shrink_layer(fillers_embedded)
        # The shape of fillers_embedded should be
        # (batch_size, sequence_length, filler_embedding_dim)
        # Pytorch LSTM expects data in the shape (sequence_length, batch_size, feature_dim)
        fillers_embedded = torch.transpose(fillers_embedded, 0, 1)

        '''
        fillers_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            fillers_embedded,
            filler_lengths,
            batch_first=False
        )

        lstm_out, hidden = self.lstm(fillers_embedded, hidden)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        '''
        lstm_out, hidden = self.lstm(fillers_embedded, hidden)
        role_predictions = self.role_weight_predictions(lstm_out)

        if self.softmax_roles:
            role_predictions = self.softmax(role_predictions)
        # role_predictions is size (sequence_length, batch_size, num_roles)

        role_embeddings = self.role_embedding(self.role_indices)

        # Normalize the embeddings. This is important so that role attention is not overruled by
        # embeddings with different orders of magnitude.
        role_embeddings = role_embeddings / torch.norm(role_embeddings, dim=1).unsqueeze(1)
        # role_embeddings is size (num_roles, role_embedding_dim)

        # During evaluation, we want to snap the role predictions to a one-hot vector
        if self.snap_one_hot_predictions:
            one_hot_predictions = self.one_hot_embedding(torch.argmax(role_predictions, 2),
                                                        self.num_roles)
            roles = torch.matmul(one_hot_predictions, role_embeddings)
        else:
            roles = torch.matmul(role_predictions, role_embeddings)
        # roles is size (sequence_length, batch_size, role_embedding_dim)

        return roles, role_predictions

    def init_hidden(self, batch_size):
        layer_multiplier = 1
        if self.bidirectional:
            layer_multiplier = 2

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        # We need a tuple for the hidden state and the cell state of the LSTM.
        return (torch.zeros(self.num_layers * layer_multiplier, batch_size, self.hidden_dim,
                            device=device),
                torch.zeros(self.num_layers * layer_multiplier, batch_size, self.hidden_dim,
                            device=device))

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes, device=device)
        return y[labels]


if __name__ == "__main__":
    import torch

    num_roles = 10
    filler_embedding_dim = 20
    num_fillers = 10
    hidden_dim = 30
    role_embedding_dim = 20
    filler_embedding = torch.nn.Embedding(
        num_fillers + 1,
        filler_embedding_dim,
        padding_idx=num_fillers
    )

    lstm = RoleAssignmentLSTM(
        num_roles,
        filler_embedding,
        hidden_dim,
        role_embedding_dim,
        num_layers=2,
        bidirectional=True,
    )

    #data = [[1, 2, 3, 4], [1, 8, 1, 0]]
    data = [[2, 3], [1, 10]]
    data_tensor = torch.tensor(data)
    out = lstm(data_tensor, [2, 1])
    print(out)
    print('experiment 2')
    data2 = [[1]]
    data_tensor2 = torch.tensor(data2)
    out2 = lstm(data_tensor2, [1])
    print(out2)
