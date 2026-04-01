from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class HFPConfig:
    """Configuration for HFP V2 model, auto-populated from HuggingFace config."""

    # --- Model architecture (populated from teacher) ---
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: Optional[int] = None
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"

    # --- HFP V2 specific ---
    freq_cutoff_ratio: float = 0.125   # k = M * ratio, controls low/high split
    conv_kernel_size: int = 7           # depthwise conv kernel size

    # --- Runtime ---
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **overrides):
        """Build HFPConfig from a HuggingFace model config."""
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        kwargs = dict(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            head_dim=getattr(hf, "head_dim", None),
            vocab_size=hf.vocab_size,
            rms_norm_eps=getattr(hf, "rms_norm_eps", 1e-6),
            max_position_embeddings=getattr(hf, "max_position_embeddings", 4096),
            rope_theta=getattr(hf, "rope_theta", 10000.0),
            rope_scaling=getattr(hf, "rope_scaling", None),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
            attention_bias=getattr(hf, "attention_bias", False),
            mlp_bias=getattr(hf, "mlp_bias", False),
            hidden_act=getattr(hf, "hidden_act", "silu"),
        )
        kwargs.update(overrides)
        return cls(**kwargs)
