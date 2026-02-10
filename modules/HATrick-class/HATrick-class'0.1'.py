import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HierarchicalCLSAblock(nn.Module):
    def __init__(self, dim, num_heads, chunk_size, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.qkv_word = nn.Linear(dim, dim * 3)

        self.qkv_cls = nn.Linear(dim, dim * 3)

        self.qkv_global_cls = nn.Linear(dim, dim * 3)

        self.q_word = nn.Linear(dim, dim)
        self.kv_cls = nn.Linear(dim, dim * 2)

        self.out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, x, mask):
        B, T, D = x.shape
        n = self.chunk_size
        num_chunks = math.ceil(T / n)

        pad_len = num_chunks * n - T
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
            mask = torch.cat([mask, torch.zeros(B, pad_len, device=x.device)], dim=1)

        x = x.view(B, num_chunks, n, D)
        mask = mask.view(B, num_chunks, n)

        cls = self.cls_token.expand(B, num_chunks, 1, D)
        x = torch.cat([cls, x], dim=2)
        mask = torch.cat([torch.ones(B, num_chunks, 1, device=x.device), mask], dim=2)

        x_flat = x.view(B * num_chunks, n + 1, D)
        mask_flat = mask.view(B * num_chunks, n + 1)

        words = x_flat[:, 1:]
        cls_local = x_flat[:, :1]

        qkv_w = self.qkv_word(words).chunk(3, dim=-1)
        qkv_c = self.qkv_cls(cls_local).chunk(3, dim=-1)

        q = torch.cat([qkv_c[0], qkv_w[0]], dim=1)
        k = torch.cat([qkv_c[1], qkv_w[1]], dim=1)
        v = torch.cat([qkv_c[2], qkv_w[2]], dim=1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn = self.attention(q, k, v, mask_flat.unsqueeze(1).unsqueeze(2))
        out = self.combine_heads(attn)

        x_flat = self.norm1(x_flat + self.out(out))

        cls_tokens = x_flat[:, 0].view(B, num_chunks, D)

        qkv = self.qkv_global_cls(cls_tokens).chunk(3, dim=-1)
        q = self.split_heads(qkv[0])
        k = self.split_heads(qkv[1])
        v = self.split_heads(qkv[2])

        attn = self.attention(q, k, v)
        cls_global = self.combine_heads(attn)
        cls_tokens = self.norm2(cls_tokens + self.out(cls_global))

        words = x_flat[:, 1:].view(B, num_chunks * n, D)
        cls_flat = cls_tokens

        q = self.split_heads(self.q_word(words))
        kv = self.kv_cls(cls_flat).chunk(2, dim=-1)
        k = self.split_heads(kv[0])
        v = self.split_heads(kv[1])

        attn = self.attention(q, k, v)
        out_words = self.combine_heads(attn)

        words = self.norm3(words + self.out(out_words))

        words = words[:, :T]
        return words
