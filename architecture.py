import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from edl_pytorch import NormalInvGamma
from torch.nn import TransformerEncoderLayer, TransformerEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class LinearNN1(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.5, activation='relu'):
        super(LinearNN1, self).__init__()
        self.hidden = hidden_size
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size  # number of labels
        self.drop_p = drop_p
        self.activation = activation

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, embed),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, out_size),
            nn.Softmax()
            # nn.Softplus()
        )

        self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, data):
        emb = self.encoder(data)
        emb = self.dropout(emb)  # Apply dropout after encoding
        output = self.decoder(emb)
        return output

    def get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError("Invalid activation function")


class LinearNN(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.1, activation='softplus'):
        super(LinearNN, self).__init__()
        self.hidden = hidden_size
        hidden = self.hidden
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size  # number of labels
        self.drop_p = drop_p
        self.activation = activation
        if self.activation == 'softplus':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.Softplus(),
                nn.Linear(hidden, embed),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Linear(hidden, hidden),
                nn.Softplus(),
                nn.Dropout(p=self.drop_p),
                nn.Linear(hidden, out_size),
                nn.Softplus(),
            )

    def forward(self, data):
        emb = self.encoder(data)
        output = self.decoder(emb)
        return output


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        # value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        #  input data consists of simple numerical features rather than sequential data (such as text or time series).
        #  treat each sample as an independent data point without considering sequential properties.
        value_len, key_len, query_len = 1, 1, 1,

        # Split the embedding into self.heads different pieces, [batchsize, sequence_length, num_heads, head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each N
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, self.heads * self.head_dim
        )  # (N, (query_len,) heads, head_dim)

        out = self.fc_out(out)

        # Residual connection and layer normalization
        # out += query
        # out = self.layer_norm(out)
        return out


class LinearNNWithSelfAttention(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.5, activation='relu', num_heads=8):
        super(LinearNNWithSelfAttention, self).__init__()
        self.hidden = hidden_size
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size  # number of labels
        self.drop_p = drop_p
        self.activation = activation
        self.num_heads = num_heads

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, embed),
        )

        # Self-attention layer
        self.self_attention = SelfAttention(embed, num_heads)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, hidden_size),
            self.get_activation(),
            nn.Linear(hidden_size, out_size),
            nn.Softmax(dim=-1)  # Softmax along the last dimension
        )

        self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, data, mask=None):
        emb = self.encoder(data)
        emb = self.dropout(emb)  # Apply dropout after encoding

        # Self-attention mechanism
        self_attended = self.self_attention(emb, emb, emb, mask)

        output = self.decoder(self_attended)
        return output


    def get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError("Invalid activation function")

# Example usage:
# model = LinearNNWithSelfAttention(in_size=100, hidden_size=64, out_size=10, embed=32)
# input_data = torch.randn(32, 100)  # Example input data, batch size 32, input size 100
# output = model(input_data)
# print(output.shape)  # Should print torch.Size([32, 10]), indicating batch size 32, output size 10
