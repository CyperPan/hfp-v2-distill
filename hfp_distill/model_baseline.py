"""
Baseline Student: Standard Attention (no HFP)
Same architecture as teacher, for A/B comparison with HFP student.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import HFPConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int = 131072, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, positions: torch.Tensor):
        # positions: [B, N]
        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq)  # [B, N, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, N, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, N, D], cos, sin: [B, N, D] -> [B, 1, N, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class StandardAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 max_position=131072, rope_theta=10000.0, attention_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, max_position, rope_theta)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BaselineDecoderLayer(nn.Module):
    def __init__(self, config: HFPConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = StandardAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_bias=config.attention_bias,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BaselineForCausalLM(nn.Module):
    def __init__(self, config: HFPConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [BaselineDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, use_gradient_checkpointing=False):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            if use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, positions, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, positions)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "hfp_specific": 0}
