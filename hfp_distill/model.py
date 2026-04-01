"""
HFP V2 Causal Language Model

Architecture: Embedding → [HFP + FFN] × L → RMSNorm → LM Head

Supports Qwen2, Qwen3, LLaMA, Mistral family models — all share the same
decoder-only structure with RMSNorm + SwiGLU MLP + GQA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import HFPConfig
from .hfp_layer import CausalHFPLayer


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


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HFPDecoderLayer(nn.Module):
    """
    X' = X + HFP(LayerNorm(X))
    X'' = X' + FFN(LayerNorm(X'))
    """

    def __init__(self, config: HFPConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hfp = CausalHFPLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            freq_cutoff_ratio=config.freq_cutoff_ratio,
            conv_kernel_size=config.conv_kernel_size,
            attention_bias=config.attention_bias,
            chunk_size=getattr(config, 'chunk_size', 64),
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # HFP block (replaces attention)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.hfp(hidden_states)
        hidden_states = residual + hidden_states

        # FFN block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HFPModel(nn.Module):
    def __init__(self, config: HFPConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HFPDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)  # [B, N, D]

        for layer in self.layers:
            if use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class HFPForCausalLM(nn.Module):
    """
    Full causal language model with HFP architecture.
    Drop-in replacement for HuggingFace CausalLM, with matching weight names
    for easy loading.
    """

    def __init__(self, config: HFPConfig):
        super().__init__()
        self.config = config
        self.model = HFPModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Returns logits: [B, N, V]"""
        hidden = self.model(input_ids, use_gradient_checkpointing)
        return self.lm_head(hidden)

    def count_parameters(self) -> dict:
        """Count total, trainable, and HFP-specific parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        hfp_only = 0
        for name, p in self.named_parameters():
            if any(k in name for k in ["dw_conv", "pw_conv", "gate_linear", "alpha_raw"]):
                hfp_only += p.numel()
        return {"total": total, "trainable": trainable, "hfp_specific": hfp_only}

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation (full recompute each step, no KV cache)."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :]  # [B, V]

            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                # top-p (nucleus) sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(
                    -1, torch.multinomial(sorted_probs, 1)
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
