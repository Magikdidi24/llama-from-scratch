import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4  # GQA
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    n_experts: int = 0  # 0 = dense, >0 = MoE
    top_k: int = 2  # experts per token
    aux_loss_coef: float = 0.01  # load balancing loss weight


def rope_cache(head_dim, seq_len, theta):
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len).float(), freqs)  # (T, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # rotate each (even, odd) channel pair
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = c.n_heads, c.n_kv_heads, c.dim // c.n_heads
        self.wq = nn.Linear(c.dim, c.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(c.dim, 2 * c.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.wkv(x).view(B, T, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:  # generating, append to the kv cache
            if cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        # no mask when decoding a single token, prefill always starts at pos 0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Mixtral-style sparse MoE layer."""

    def __init__(self, c, hidden):
        super().__init__()
        self.top_k = c.top_k
        self.router = nn.Linear(c.dim, c.n_experts, bias=False)
        self.experts = nn.ModuleList(FeedForward(c.dim, hidden) for _ in range(c.n_experts))

    def forward(self, x):
        tokens = x.flatten(0, 1)
        probs = F.softmax(self.router(tokens), -1, dtype=torch.float32)
        weights, idx = probs.topk(self.top_k, -1)
        weights = weights / weights.sum(-1, keepdim=True)  # renormalize over the top_k, like mixtral
        out = torch.zeros_like(tokens)
        for e, expert in enumerate(self.experts):
            tok, slot = torch.where(idx == e)  # tokens sent to e and which top_k slot
            out.index_add_(0, tok, (expert(tokens[tok]) * weights[tok, slot, None]).type_as(out))
        # switch transformer balancing loss, 1 when perfectly balanced
        f = torch.bincount(idx.flatten(), minlength=len(self.experts)) / idx.numel()
        aux = len(self.experts) * (f * probs.mean(0)).sum()
        return out.view_as(x), aux


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm, self.ffn_norm = nn.RMSNorm(c.dim), nn.RMSNorm(c.dim)
        self.attn = Attention(c)
        hidden = 64 * math.ceil(8 * c.dim / 3 / 64)  # llama's 2/3 * 4 * dim, rounded up to a multiple of 64
        # each expert gets hidden / top_k, same flops per token as dense
        self.ffn = MoE(c, hidden // c.top_k) if c.n_experts else FeedForward(c.dim, hidden)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        if isinstance(self.ffn, MoE):
            y, aux = self.ffn(self.ffn_norm(x))
            return x + y, aux
        return x + self.ffn(self.ffn_norm(x)), 0.0


class Llama(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = nn.RMSNorm(c.dim)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        cos, sin = rope_cache(c.dim // c.n_heads, c.max_seq_len, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        for name, p in self.named_parameters():
            if p.dim() == 2:
                # gpt-2 style: smaller init for the residual projections
                std = 0.02 / math.sqrt(2 * c.n_layers) if name.endswith(("wo.weight", "w2.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def num_params(self, active=False):
        # non-embedding params (the N in scaling laws), active=True counts only top_k experts
        n = sum(p.numel() for p in self.parameters()) - self.tok_emb.weight.numel()
        c = self.config
        if active and c.n_experts:
            expert = sum(p.numel() for p in self.layers[0].ffn.experts[0].parameters())
            n -= c.n_layers * (c.n_experts - c.top_k) * expert
        return n

    def forward(self, idx, targets=None, cache=None):
        # cache: one dict of k/v per layer, generation only
        T = idx.size(1)
        pos = cache[0]["k"].size(2) if cache and cache[0] else 0
        x, aux = self.tok_emb(idx), torch.zeros((), device=idx.device)
        for i, layer in enumerate(self.layers):
            x, layer_aux = layer(x, self.cos[pos:pos + T], self.sin[pos:pos + T], None if cache is None else cache[i])
            aux = aux + layer_aux
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        return logits, loss, aux / len(self.layers)  # aux loss averaged over layers, 0 if dense

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        # kv cache: run the prompt once, then feed one token at a time
        # no sliding window, so this stops at max_seq_len
        cache = [{} for _ in self.layers]
        x = idx
        for _ in range(min(max_new_tokens, self.config.max_seq_len - idx.size(1))):
            logits = self(x, cache=cache)[0][:, -1] / temperature
            kth = torch.topk(logits, top_k).values[:, [-1]]
            logits[logits < kth] = -float("inf")
            x = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, x], 1)
        return idx
