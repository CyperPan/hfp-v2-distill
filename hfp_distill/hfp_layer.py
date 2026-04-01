"""
HFP V2 Layer — Hybrid Frequency Processing
Replaces standard multi-head attention in a Transformer block.

Two variants:
  - HFPLayer:       Original non-causal (for encoder / analysis)
  - CausalHFPLayer: Chunk-causal (for autoregressive LM)

Causal design:
  Sequence is split into chunks of size C. FFT is computed within each chunk
  (no future info leaks across chunks). Low-freq attention uses a block-causal
  mask so chunk j only attends to chunks 0..j. High-freq conv uses causal
  (left) padding. Gate uses per-chunk mean (strictly causal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HFPLayer(nn.Module):
    """
    Non-causal HFP V2 (for reference / encoder tasks).
    Y = IFFT[ Attn(FFT[X]_{f<k})  ⊕  (DWConv ⊙ Gate)(FFT[X]_{f≥k}) ]
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

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        pad = conv_kernel_size // 2
        self.dw_conv_real = nn.Conv1d(hidden_size, hidden_size, conv_kernel_size, padding=pad, groups=hidden_size, bias=False)
        self.pw_conv_real = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)
        self.dw_conv_imag = nn.Conv1d(hidden_size, hidden_size, conv_kernel_size, padding=pad, groups=hidden_size, bias=False)
        self.pw_conv_imag = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)

        self.gate_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self._init_hfp_weights()

    def _init_hfp_weights(self):
        for conv in [self.dw_conv_real, self.pw_conv_real, self.dw_conv_imag, self.pw_conv_imag]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.gate_linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, D = hidden_states.shape
        X_hat = torch.fft.rfft(hidden_states.float(), dim=1)
        M = X_hat.shape[1]
        k = max(1, int(M * self.freq_cutoff_ratio))

        X_low = X_hat[:, :k, :]
        X_high = X_hat[:, k:, :]

        A_low_real = self._attention(X_low.real, B, k)
        A_low_imag = self._attention(X_low.imag, B, k)
        A_low = torch.complex(A_low_real, A_low_imag)

        A_high = self._high_freq_conv(X_high, hidden_states)

        alpha = torch.sigmoid(self.alpha_raw)
        A_low_fused = alpha * A_low + (1.0 - alpha) * X_low
        Y_hat = torch.cat([A_low_fused, A_high], dim=1)
        Y = torch.fft.irfft(Y_hat, n=N, dim=1)
        return Y.to(hidden_states.dtype)

    def _attention(self, x, B, seq_len):
        x = x.to(self.q_proj.weight.dtype)
        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(out).float()

    def _high_freq_conv(self, X_high, X_time):
        dtype = self.dw_conv_real.weight.dtype
        real_in = X_high.real.to(dtype).transpose(1, 2)
        H_r = self.pw_conv_real(F.gelu(self.dw_conv_real(real_in))).transpose(1, 2)
        imag_in = X_high.imag.to(dtype).transpose(1, 2)
        H_i = self.pw_conv_imag(F.gelu(self.dw_conv_imag(imag_in))).transpose(1, 2)
        X_bar = X_time.mean(dim=1, keepdim=True)
        g = torch.sigmoid(self.gate_linear(X_bar)).float()
        return torch.complex(H_r.float() * g, H_i.float() * g) + X_high


# ======================================================================
# Causal HFP V2 — chunk-based, for autoregressive language modeling
# ======================================================================

class CausalHFPLayer(nn.Module):
    """
    Chunk-Causal HFP V2 for autoregressive LM.

    Causality guarantee:
      - Sequence split into non-overlapping chunks of size C
      - FFT computed within each chunk → no cross-chunk future leakage
      - Low-freq attention: block-causal mask (chunk j sees chunks 0..j only)
      - High-freq conv: causal (left) padding within each chunk
      - Gate: per-chunk mean (each chunk uses only its own tokens)
      - Within a chunk (C tokens), positions can see each other (bounded, C≤64)

    This is the same trade-off as block-causal attention (Transformer-XL, etc.):
    strict causality between blocks, full visibility within a block.
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
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.freq_cutoff_ratio = freq_cutoff_ratio
        self.chunk_size = chunk_size

        # ---- Low-frequency: attention projections ----
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # ---- High-frequency: causal depthwise conv + pointwise conv ----
        # Causal: left-pad by (kernel_size - 1), no right padding
        self.conv_kernel_size = conv_kernel_size
        self.dw_conv_real = nn.Conv1d(hidden_size, hidden_size, conv_kernel_size, padding=0, groups=hidden_size, bias=False)
        self.pw_conv_real = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)
        self.dw_conv_imag = nn.Conv1d(hidden_size, hidden_size, conv_kernel_size, padding=0, groups=hidden_size, bias=False)
        self.pw_conv_imag = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)

        # ---- Gate ----
        self.gate_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # ---- Learnable alpha ----
        self.alpha_raw = nn.Parameter(torch.zeros(1))

        self._init_hfp_weights()

    def _init_hfp_weights(self):
        for conv in [self.dw_conv_real, self.pw_conv_real, self.dw_conv_imag, self.pw_conv_imag]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.gate_linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, D]
        Returns:
            output: [B, N, D]
        """
        B, N, D = hidden_states.shape
        C = self.chunk_size

        # ===== Pad to multiple of chunk_size =====
        pad_len = (C - N % C) % C
        if pad_len > 0:
            x_padded = F.pad(hidden_states, (0, 0, 0, pad_len))  # pad seq dim
        else:
            x_padded = hidden_states
        num_chunks = x_padded.shape[1] // C

        # ===== Reshape to chunks: [B, nc, C, D] =====
        chunks = x_padded.view(B, num_chunks, C, D)

        # ===== FFT within each chunk (no cross-chunk leakage) =====
        X_hat = torch.fft.rfft(chunks.float(), dim=2)  # [B, nc, M, D]
        M = X_hat.shape[2]  # C // 2 + 1
        k = max(1, int(M * self.freq_cutoff_ratio))

        X_low = X_hat[:, :, :k, :]     # [B, nc, k, D]
        X_high = X_hat[:, :, k:, :]    # [B, nc, M-k, D]

        # ===== Low-freq: block-causal attention across chunks =====
        A_low_real = self._causal_attention(X_low.real, B, num_chunks, k)
        A_low_imag = self._causal_attention(X_low.imag, B, num_chunks, k)
        A_low = torch.complex(A_low_real, A_low_imag)  # [B, nc, k, D]

        # ===== High-freq: causal conv within chunks =====
        A_high = self._high_freq_causal_conv(X_high, chunks)  # [B, nc, M-k, D]

        # ===== Fusion =====
        alpha = torch.sigmoid(self.alpha_raw)
        A_low_fused = alpha * A_low + (1.0 - alpha) * X_low

        Y_hat = torch.cat([A_low_fused, A_high], dim=2)  # [B, nc, M, D]

        # ===== IFFT per chunk =====
        Y = torch.fft.irfft(Y_hat, n=C, dim=2)  # [B, nc, C, D]

        # ===== Reshape back and remove padding =====
        Y = Y.reshape(B, -1, D)[:, :N, :]
        return Y.to(hidden_states.dtype)

    # ------------------------------------------------------------------
    def _causal_attention(
        self, x: torch.Tensor, B: int, nc: int, k: int
    ) -> torch.Tensor:
        """
        Block-causal attention on flattened chunk frequency bins.
        x: [B, nc, k, D] (real-valued)
        Chunk j's k freq bins can attend to chunks 0..j's freq bins.
        """
        total_len = nc * k
        x_flat = x.reshape(B, total_len, -1).to(self.q_proj.weight.dtype)

        q = self.q_proj(x_flat).view(B, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        kk = self.k_proj(x_flat).view(B, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            kk = kk.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Block-causal mask: query i (in chunk i//k) attends to
        # key j only if chunk(j) <= chunk(i)  (same or earlier chunk)
        chunk_ids = torch.arange(total_len, device=x.device) // k  # [total_len]
        # mask[i, j] = True if chunk(j) <= chunk(i)
        causal_mask = chunk_ids.unsqueeze(1) >= chunk_ids.unsqueeze(0)  # [total_len, total_len]

        out = F.scaled_dot_product_attention(
            q, kk, v, attn_mask=causal_mask.unsqueeze(0).unsqueeze(0)
        )
        out = out.transpose(1, 2).contiguous().view(B, total_len, -1)
        out = self.o_proj(out).float()
        return out.view(B, nc, k, -1)

    def _high_freq_causal_conv(
        self, X_high: torch.Tensor, chunks: torch.Tensor
    ) -> torch.Tensor:
        """
        High-freq processing with causal (left-padded) conv.
        X_high: [B, nc, M-k, D] (complex)
        chunks: [B, nc, C, D]   (time-domain, for gate)
        """
        B, nc = X_high.shape[0], X_high.shape[1]
        freq_len = X_high.shape[2]
        dtype = self.dw_conv_real.weight.dtype
        pad = self.conv_kernel_size - 1

        # Flatten chunks for conv: [B*nc, M-k, D] → [B*nc, D, M-k]
        def causal_conv(x_in, dw_conv, pw_conv):
            x = x_in.reshape(B * nc, freq_len, -1).to(dtype).transpose(1, 2)  # [B*nc, D, M-k]
            x = F.pad(x, (pad, 0))          # causal: left-pad only
            x = dw_conv(x)                   # [B*nc, D, M-k]
            x = pw_conv(F.gelu(x))           # [B*nc, D, M-k]
            return x.transpose(1, 2).reshape(B, nc, freq_len, -1)

        H_r = causal_conv(X_high.real, self.dw_conv_real, self.pw_conv_real)
        H_i = causal_conv(X_high.imag, self.dw_conv_imag, self.pw_conv_imag)

        # Gate: per-chunk mean (causal — only current chunk's tokens)
        X_bar = chunks.mean(dim=2)                         # [B, nc, D]
        g = torch.sigmoid(self.gate_linear(X_bar)).float() # [B, nc, D]
        g = g.unsqueeze(2)                                 # [B, nc, 1, D]

        return torch.complex(H_r.float() * g, H_i.float() * g) + X_high
