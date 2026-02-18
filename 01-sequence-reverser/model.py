import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        pe = torch.zeros(seq_len, d_model)

        # (1, d_model // 2)
        # i has to represent the columns in the final matrix, therefore unsqueeze 0
        i = torch.arange(0, d_model // 2).unsqueeze(0)

        # (seq_len, 1)
        pos = torch.arange(0, seq_len).unsqueeze(1)

        # (seq_len, d_model // 2)
        term = pos / torch.pow(10000, 2 * i / d_model)

        pe[:, 0::2] = torch.sin(term)
        pe[:, 1::2] = torch.cos(term)

        pe = pe.unsqueeze(0)

        # Move to gpu for the below x + self.pe operation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # n_samples, seq_len, d_model
        return x + self.pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by num_heads"

        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        n_samples, seq_len, d_model = x.size()

        # (n_samples, seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # (n_samples, seq_len, h, d_k)
        Q = Q.view(n_samples, seq_len, self.h, self.d_k)
        K = K.view(n_samples, seq_len, self.h, self.d_k)
        V = V.view(n_samples, seq_len, self.h, self.d_k)

        # Swap h and seq_len, matmul only works on last 2 dimensions
        # (n_samples, h, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        qkt = Q @ K.transpose(2, 3)  # (n_samples, h, seq_len, seq_len)
        scaled = qkt / math.sqrt(self.d_k)
        weights = F.softmax(scaled, dim=-1)

        self.attn_weights = weights

        output = weights @ V  # (n_samples, h, seq_len, d_k)

        # (n_samples, seq_len, d_model)
        concat = output.transpose(1, 2).contiguous().view(
            n_samples, seq_len, d_model)

        return self.W_O(concat)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, h)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mha_out = self.mha(x)
        x = self.norm1(x + mha_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        # (batch, seq_len, dims)
        return x


class TransformerReverser(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, num_heads, num_layers):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, num_heads * 4) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.fc_out(x)
        return x
