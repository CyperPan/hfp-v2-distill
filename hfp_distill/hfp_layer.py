"""
HFP V2 Layer — Hybrid Frequency Processing
Replaces standard multi-head attention in a Transformer block.

Pipeline:
  FFT → [low-freq: attention] + [high-freq: conv+gate] → fusion → IFFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HFPLayer(nn.Module):
    """
    Y = IFFT[ Attn(FFT[X]_{f<k})  ⊕  (DWConv ⊙ Gate)(FFT[X]_{f≥k}) ]

    - Low-frequency (f < k): standard multi-head attention for global semantics
    - High-frequency (f ≥ k): depthwise conv + pointwise conv + gating for local detail
    - Learnable alpha blends low-freq attention output with residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        freq_cutoff_ratio: float = 0.125,
        conv_kernel_size: int = 7,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.freq_cutoff_ratio = freq_cutoff_ratio
        self.scaling = head_dim ** -0.5

        # ---- Low-frequency: attention projections ----
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # ---- High-frequency: depthwise conv (κ=7) + pointwise conv ----
        pad = conv_kernel_size // 2
        self.dw_conv_real = nn.Conv1d(
            hidden_size, hidden_size, conv_kernel_size,
            padding=pad, groups=hidden_size, bias=False,
        )
        self.pw_conv_real = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)
        self.dw_conv_imag = nn.Conv1d(
            hidden_size, hidden_size, conv_kernel_size,
            padding=pad, groups=hidden_size, bias=False,
        )
        self.pw_conv_imag = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)

        # ---- Gate: global info → per-channel gate ----
        self.gate_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # ---- Learnable alpha for low-freq residual mixing ----
        # sigmoid(0) = 0.5, balanced initialization
        self.alpha_raw = nn.Parameter(torch.zeros(1))

        self._init_hfp_weights()

    def _init_hfp_weights(self):
        """Initialize HFP-specific weights (conv, gate). Attention weights are loaded from teacher."""
        for conv in [self.dw_conv_real, self.pw_conv_real, self.dw_conv_imag, self.pw_conv_imag]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.gate_linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, D]  (time domain)
        Returns:
            output: [B, N, D]  (time domain)
        """
        B, N, D = hidden_states.shape

        # ===== Step 1: FFT along sequence dim =====
        X_hat = torch.fft.rfft(hidden_states.float(), dim=1)  # [B, M, D], complex64
        M = X_hat.shape[1]  # N // 2 + 1
        k = max(1, int(M * self.freq_cutoff_ratio))

        # ===== Step 2: Frequency split =====
        X_low = X_hat[:, :k, :]     # [B, k, D]
        X_high = X_hat[:, k:, :]    # [B, M-k, D]

        # ===== Step 3a: Low-freq → attention (real & imag independently) =====
        A_low_real = self._attention(X_low.real, B, k)
        A_low_imag = self._attention(X_low.imag, B, k)
        A_low = torch.complex(A_low_real, A_low_imag)

        # ===== Step 3b: High-freq → conv + gate =====
        A_high = self._high_freq_conv(X_high, hidden_states)

        # ===== Step 4: Fusion =====
        alpha = torch.sigmoid(self.alpha_raw)
        A_low_fused = alpha * A_low + (1.0 - alpha) * X_low   # residual mixing

        Y_hat = torch.cat([A_low_fused, A_high], dim=1)       # [B, M, D]

        # ===== Step 5: IFFT =====
        Y = torch.fft.irfft(Y_hat, n=N, dim=1)                # [B, N, D]
        return Y.to(hidden_states.dtype)

    # ------------------------------------------------------------------
    def _attention(self, x: torch.Tensor, B: int, seq_len: int) -> torch.Tensor:
        """Standard multi-head attention on real-valued [B, seq_len, D]."""
        x = x.to(self.q_proj.weight.dtype)

        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, k, d]
        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(out).float()

    def _high_freq_conv(
        self, X_high: torch.Tensor, X_time: torch.Tensor
    ) -> torch.Tensor:
        """
        High-freq processing:
          H_r = PW_conv( GELU( DW_conv( Re(X_high) ) ) )
          H_i = PW_conv( GELU( DW_conv( Im(X_high) ) ) )
          g   = sigmoid( W_g · mean(X_time) )
          out = (H_r * g + j * H_i * g) + X_high   (residual)
        """
        dtype = self.dw_conv_real.weight.dtype

        # Real part convolution: [B, M-k, D] → [B, D, M-k]
        real_in = X_high.real.to(dtype).transpose(1, 2)
        H_r = self.pw_conv_real(F.gelu(self.dw_conv_real(real_in))).transpose(1, 2)

        # Imaginary part convolution
        imag_in = X_high.imag.to(dtype).transpose(1, 2)
        H_i = self.pw_conv_imag(F.gelu(self.dw_conv_imag(imag_in))).transpose(1, 2)

        # Gate from time-domain global average pooling
        X_bar = X_time.mean(dim=1, keepdim=True)          # [B, 1, D]
        g = torch.sigmoid(self.gate_linear(X_bar)).float() # [B, 1, D]

        # Apply gate + residual
        A_high = torch.complex(H_r.float() * g, H_i.float() * g) + X_high
        return A_high
