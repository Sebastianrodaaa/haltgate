"""Tiny looped transformer: one shared block of `layers` pre-LN layers, applied n times with input injection."""
import torch
from torch import nn
from .data import V, T


class Looped(nn.Module):
    def __init__(self, d=256, heads=4, layers=2, nmax=8):
        super().__init__()
        self.nmax = nmax
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.block = nn.ModuleList(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True) for _ in range(layers)
        )
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)

    def embed(self, x):
        return self.emb(x) + self.pos.weight[: x.shape[1]]

    def step(self, h, e, pad):
        """One shared-block pass: h_i = block(h_{i-1} + e). Returns new state and readout logits."""
        h = h + e
        for layer in self.block:
            h = layer(h, src_key_padding_mask=pad)
        return h, self.head(self.ln(h))

    def forward(self, x, n=None):
        """Yields (h_prev, h, logits) for loops 1..n."""
        e, pad = self.embed(x), x == 0
        h = torch.zeros_like(e)
        for _ in range(n or self.nmax):
            prev = h
            h, logits = self.step(h, e, pad)
            yield prev, h, logits
