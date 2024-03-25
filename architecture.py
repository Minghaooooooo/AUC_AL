import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from edl_pytorch import NormalInvGamma

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


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


class AttentionResidualNN(nn.Module):
    def __init__(self, in_size=None, hidden_size=None, out_size=None, embed=None,
                 drop_p=0.1, activation='softplus'):
        super(AttentionResidualNN, self).__init__()
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

            self.attention_layer = nn.Linear(embed, 1)

            self.ffn = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed)
            )

            self.dropout = nn.Dropout(p=self.drop_p)

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

        # Attention mechanism
        attention_scores = F.softmax(self.attention_layer(emb), dim=1)
        weighted_emb = torch.sum(attention_scores * emb, dim=1)

        # Feedforward neural network (FFN) with residual connection
        ffn_output = self.ffn(weighted_emb)
        residual_emb = weighted_emb + ffn_output

        # Dropout
        residual_emb = self.dropout(residual_emb)

        # Decoder
        output = self.decoder(weighted_emb)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.fc = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, tgt):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output