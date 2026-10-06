"""Bidirectional DiT operating in the latent space R^d.

Token ids enter through `embed_ids` (a V -> d linear map on one-hot vectors); the
rollout state z in R^d is refined by `forward_d`, which returns the network's
estimate of the clean latent; `decode` maps a latent to log-probabilities over
the vocabulary. Time enters every block through adaptive LayerNorm.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = torch.arange(seq_len, dtype=torch.float32)[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, L, D]; rotates the (2i, 2i+1) pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class RopeAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0 and (d_model // n_heads) % 2 == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = apply_rope(qkv[0], cos, sin), apply_rope(qkv[1], cos, sin), qkv[2]
        attn_mask = None
        if key_padding_mask is not None:                         # True = padding
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, L], True = keep
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out(out.transpose(1, 2).contiguous().view(B, L, D))


class TimestepEmbedder(nn.Module):
    """Sinusoidal features -> MLP with `n_layers` linear layers -> time_dim."""

    def __init__(self, time_dim: int, freq_dim: int, n_layers: int):
        super().__init__()
        self.freq_dim = freq_dim
        layers = [nn.Linear(freq_dim, time_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(time_dim, time_dim), nn.SiLU()]
        layers += [nn.Linear(time_dim, time_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, t):
        return self.mlp(sinusoidal_time_embedding(t, self.freq_dim))


class DiTBlock(nn.Module):
    """Pre-norm attention + MLP, both modulated by the time embedding (adaLN-Zero)."""

    def __init__(self, d_model, n_heads, mlp_ratio, dropout, time_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = RopeAttention(d_model, n_heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d_model), nn.Dropout(dropout))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * d_model))

    def forward(self, x, t_emb, cos, sin, key_padding_mask=None):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(t_emb).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn_drop(self.attn(h, cos, sin, key_padding_mask))
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, n_layers: int, n_heads: int,
                 mlp_ratio: float, dropout: float, time_embed_layers: int, freq_dim: int, time_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.in_proj = nn.Linear(vocab_size, d_model)
        cos, sin = build_rope_cache(max_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.t_embedder = TimestepEmbedder(time_dim, freq_dim, time_embed_layers)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, mlp_ratio, dropout, time_dim)
                                     for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * d_model))
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:                     # adaLN-Zero: blocks start as the identity
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids [B, L] -> clean latents [B, L, d]."""
        return self.in_proj(F.one_hot(ids, self.vocab_size).float())

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Latents [..., d] -> log-probabilities [..., V]."""
        return F.log_softmax(self.out_proj(h), dim=-1)

    def forward_d(self, z: torch.Tensor, t: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Latent state z [B, L, d] at time t [B] -> estimate of the clean latent [B, L, d]."""
        B, L, _ = z.shape
        assert L <= self.max_len
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        t_emb = self.t_embedder(t)
        key_padding_mask = None if attention_mask is None else ~attention_mask
        h = z
        for block in self.blocks:
            h = block(h, t_emb, cos, sin, key_padding_mask)
        shift, scale = self.final_ada(t_emb).unsqueeze(1).chunk(2, dim=-1)
        return self.final_norm(h) * (1 + scale) + shift
