import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingWithProjection(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model,
                 max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.projection = nn.Linear(self.d_embed, self.d_model)
        self.scaling = float(math.sqrt(self.d_model))

        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def create_positional_encoding(seq_length, d_model, batch_size=1):
        # Create position indices: [seq_length, 1]
        position = torch.arange(seq_length).unsqueeze(1).float()

        # Create dimension indices: [1, d_model//2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Create empty tensor: [seq_length, d_model]
        pe = torch.zeros(seq_length, d_model)

        # Compute sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and expand: [batch_size, seq_length, d_model]
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)

        return pe

    def forward(self, x):
        assert x.dtype == torch.long, f"Input tensor must have dtype torch.long, got {x.dtype}"
        batch_size, seq_length = x.size()  # [batch, seq_length]

        # token embedding
        token_embedding = self.embedding(x)  # [2, 16, 1024]
        # project the scaled token embedding to the d_model space
        token_embedding = self.projection(token_embedding) * self.scaling  # [2, 16, 768]

        # add positional encodings to projected,
        # scaled embeddings before applying layer norm and dropout.
        positional_encoding = self.create_positional_encoding(seq_length, self.d_model, batch_size)  # [2, 16, 768]

        # In addition, we apply dropout to the sums of the embeddings
        # in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.
        normalized_sum = self.layernorm(token_embedding + positional_encoding)
        final_output = self.dropout(normalized_sum)
        return final_output

class TransformerAttention(nn.Module):
    """
    Transformer Scaled Dot Product Attention Module
    Args:
        d_model: Total dimension of the model.
        num_head: Number of attention heads.
        dropout: Dropout rate for attention scores.
        bias: Whether to include bias in linear projections.

    Inputs:
        sequence: input sequence for self-attention and the query for cross-attention
        key_value_state: input for the key, values for cross-attention
    """
    def __init__(self, d_model, num_head, dropout=0.1, bias=True): # infer d_k, d_v, d_q from d_model
        super().__init__()  # Missing in the original implementation
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        self.d_model = d_model
        self.num_head = num_head
        self.d_head=d_model//num_head
        self.dropout_rate = dropout  # Store dropout rate separately

        # linear transformations
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head))  # Store as float in initialization

    def forward(self, sequence, key_value_states=None, att_mask=None):
        """Input shape: [batch_size, seq_len, d_model=num_head * d_head]"""
        batch_size, seq_len, model_dim = sequence.size()

        # Check only critical input dimensions
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
                f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # Linear projections and reshape for multi-head
        Q_state = self.q_proj(sequence)
        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_len
            K_state = self.k_proj(sequence)
            V_state = self.v_proj(sequence)

        # [batch_size, self.num_head, seq_len, self.d_head]
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head).transpose(1, 2)

        # in cross-attention, key/value sequence length might be different from query sequence length
        K_state = K_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1, 2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1, 2)

        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler

        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1, -2))

        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask

            # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim=-1)

        # apply drop out to attention score
        att_score = self.dropout(att_score)

        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)

        # concatinate all attention heads
        att_output = att_output.transpose(1, 2)
        att_output = att_output.contiguous().view(batch_size, seq_len, self.num_head * self.d_head)

        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), \
            f"Final output shape {att_output.size()} incorrect"

        return att_output


class FFN(nn.Module):
    """
    Position-wise Feed-Forward Networks
    This consists of two linear transformations with a ReLU activation in between.

    FFN(x) = max(0, xW1 + b1 )W2 + b2
    d_model: embedding dimension (e.g., 512)
    d_ff: feed-forward dimension (e.g., 2048)

    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Linear transformation y = xW+b
        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.fc2 = nn.Linear(self.d_ff, self.d_model, bias=True)

        # for potential speed up
        # Pre-normalize the weights (can help with training stability)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, input):
        # check input and first FF layer dimension matching
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"

        # First linear transformation followed by ReLU
        # There's no need for explicit torch.max() as F.relu() already implements max(0,x)
        f1 = F.relu(self.fc1(input))

        # max(0, xW_1 + b_1)W_2 + b_2
        f2 = self.fc2(f1)

        return f2
net = FFN(d_model = 512,  d_ff =2048)
print(net)


class TransformerEncoder(nn.Module):
    """
    Encoder layer of the Transformer
    Sublayers: TransformerAttention
               Residual LayerNorm
               FNN
               Residual LayerNorm
    Args:
            d_model: 512 model hidden dimension
            d_embed: 512 embedding dimension, same as d_model in transformer framework
            d_ff: 2048 hidden dimension of the feed forward network
            num_head: 8 Number of attention heads.
            dropout:  0.1 dropout rate

            bias: Whether to include bias in linear projections.

    """

    def __init__(
            self, d_model, d_ff,
            num_head, dropout=0.1,
            bias=True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # attention sublayer
        self.att = TransformerAttention(
            d_model=d_model,
            num_head=num_head,
            dropout=dropout,
            bias=bias
        )

        # FFN sublayer
        self.ffn = FFN(
            d_model=d_model,
            d_ff=d_ff
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # layer-normalization layer
        self.LayerNorm_att = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)

    def forward(self, embed_input, padding_mask=None):
        batch_size, seq_len, _ = embed_input.size()

        ## First sublayer: self attion
        att_sublayer = self.att(sequence=embed_input, key_value_states=None,
                                att_mask=padding_mask)  # [batch_size, sequence_length, d_model]

        # apply dropout before layer normalization for each sublayer
        att_sublayer = self.dropout(att_sublayer)
        # Residual layer normalization
        att_normalized = self.LayerNorm_att(embed_input + att_sublayer)  # [batch_size, sequence_length, d_model]

        ## Second sublayer: FFN
        ffn_sublayer = self.ffn(att_normalized)  # [batch_size, sequence_length, d_model]
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized + ffn_sublayer)  # [batch_size, sequence_length, d_model]

        return ffn_normalized
net = TransformerEncoder( d_model = 512, d_ff =2048, num_head=8, dropout=0.1, bias=True )
print(net)


class Transformer(nn.Module):
    def __init__(
            self,
            num_layer,
            d_model, d_embed, d_ff,
            num_head,
            src_vocab_size,
            tgt_vocab_size,
            max_position_embeddings=512,
            dropout=0.1,
            bias=True
    ):
        super().__init__()

        self.tgt_vocab_size = tgt_vocab_size

        # Source and target embeddings
        self.src_embedding = EmbeddingWithProjection(
            vocab_size=src_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )

        self.tgt_embedding = EmbeddingWithProjection(
            vocab_size=tgt_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )

        # Encoder
        self.encoder_decoder = TransformerEncoder(
            num_layer=num_layer,
            d_model=d_model,
            d_ff=d_ff,
            num_head=num_head,
            dropout=dropout,
            bias=bias
        )
        # Output projection and softmax
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def shift_target_right(self, tgt_tokens):
        # Shift target tokens right by padding with zeros at the beginning
        batch_size, seq_len = tgt_tokens.size()

        # Create start token (zeros)
        start_tokens = torch.zeros(batch_size, 1, dtype=tgt_tokens.dtype, device=tgt_tokens.device)

        # Concatenate start token and remove last token
        shifted_tokens = torch.cat([start_tokens, tgt_tokens[:, :-1]], dim=1)

        return shifted_tokens

    def forward(self, src_tokens, tgt_tokens, padding_mask=None):
        """
        Args:
            src_tokens: source sequence [batch_size, src_len]
            tgt_tokens: target sequence [batch_size, tgt_len]
            padding_mask: padding mask [batch_size, 1, 1, seq_len]
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size] log probabilities
        """
        # Shift target tokens right for teacher forcing
        shifted_tgt_tokens = self.shift_target_right(tgt_tokens)

        # Embed source and target sequences
        src_embedding = self.src_embedding(src_tokens)
        tgt_embedding = self.tgt_embedding(shifted_tgt_tokens)

        # Pass through encoder-decoder stack
        decoder_output = self.encoder_decoder(
            embed_encoder_input=src_embedding,
            embed_decoder_input=tgt_embedding,
            padding_mask=padding_mask
        )

        # Project to vocabulary size and apply log softmax
        logits = self.output_projection(decoder_output)
        log_probs = self.softmax(logits)

        return log_probs
