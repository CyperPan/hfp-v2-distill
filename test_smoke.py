#!/usr/bin/env python3
"""
Smoke test — verify the full pipeline works with a tiny model.
Runs on CPU, no GPU needed.

Usage:
    cd hfp-inference && python test_smoke.py
"""

import torch
from hfp_distill.config import HFPConfig
from hfp_distill.model import HFPForCausalLM
from hfp_distill.hfp_layer import HFPLayer, CausalHFPLayer


def test_hfp_layer():
    """Test non-causal HFP layer forward pass."""
    print("=== Test HFP Layer (non-causal) ===")
    layer = HFPLayer(
        hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, freq_cutoff_ratio=0.125, conv_kernel_size=7,
    ).float()

    x = torch.randn(2, 32, 64)
    y = layer(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert not torch.isnan(y).any(), "NaN in output"
    print(f"  Input: {x.shape}  Output: {y.shape}  OK\n")


def test_causal_hfp_layer():
    """Test causal HFP layer forward pass."""
    print("=== Test CausalHFPLayer ===")
    layer = CausalHFPLayer(
        hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, freq_cutoff_ratio=0.125, conv_kernel_size=7,
        chunk_size=16,
    ).float()

    # Test with seq_len that's a multiple of chunk_size
    x = torch.randn(2, 32, 64)
    y = layer(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert not torch.isnan(y).any(), "NaN in output"
    print(f"  Aligned input:   {x.shape} → {y.shape}  OK")

    # Test with seq_len NOT a multiple of chunk_size (needs padding)
    x2 = torch.randn(2, 37, 64)
    y2 = layer(x2)
    assert y2.shape == x2.shape, f"Shape mismatch: {y2.shape} != {x2.shape}"
    assert not torch.isnan(y2).any(), "NaN in output"
    print(f"  Unaligned input: {x2.shape} → {y2.shape}  OK")
    print()


def test_causal_masking():
    """Verify block-causal property: changing future chunks doesn't affect past output."""
    print("=== Test Causal Masking ===")
    layer = CausalHFPLayer(
        hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, chunk_size=16,
    ).float()
    layer.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 64, 64)  # 4 chunks of 16

    with torch.no_grad():
        y_full = layer(x)

    # Modify chunk 3 (last chunk) and check chunk 0 output is unchanged
    x_mod = x.clone()
    x_mod[:, 48:, :] = torch.randn(1, 16, 64)  # change chunk 3

    with torch.no_grad():
        y_mod = layer(x_mod)

    # Chunk 0 output should be identical (causal: can't see chunk 3)
    diff_chunk0 = (y_full[:, :16, :] - y_mod[:, :16, :]).abs().max().item()
    # Chunk 3 output should differ
    diff_chunk3 = (y_full[:, 48:, :] - y_mod[:, 48:, :]).abs().max().item()

    print(f"  Chunk 0 diff after modifying chunk 3: {diff_chunk0:.2e} (should be ~0)")
    print(f"  Chunk 3 diff after modifying chunk 3: {diff_chunk3:.2e} (should be >0)")

    assert diff_chunk0 < 1e-5, f"CAUSAL VIOLATION: chunk 0 changed! diff={diff_chunk0}"
    assert diff_chunk3 > 1e-3, f"Chunk 3 should change but didn't! diff={diff_chunk3}"
    print("  Causality verified!  OK\n")


def test_full_model():
    """Test full model forward + generate."""
    print("=== Test Full Model (Causal) ===")
    config = HFPConfig(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=256,
        max_position_embeddings=512,
        freq_cutoff_ratio=0.125, conv_kernel_size=7,
        chunk_size=16, dtype=torch.float32,
    )
    model = HFPForCausalLM(config).float()

    params = model.count_parameters()
    print(f"  Params: {params['total']:,} total, {params['hfp_specific']:,} HFP-specific")

    input_ids = torch.randint(0, 256, (2, 32))
    logits = model(input_ids)
    assert logits.shape == (2, 32, 256), f"Logits shape: {logits.shape}"
    print(f"  Forward: input={input_ids.shape} → logits={logits.shape}")

    prompt = torch.randint(0, 256, (1, 4))
    output = model.generate(prompt, max_new_tokens=8, temperature=1.0)
    assert output.shape[1] == 4 + 8, f"Generated length: {output.shape[1]}"
    print(f"  Generate: prompt={prompt.shape} → output={output.shape}")
    print(f"  OK\n")


def test_distillation_loss():
    """Test loss computation."""
    print("=== Test Distillation Loss ===")
    import torch.nn.functional as F

    B, N, V = 2, 32, 256
    student_logits = torch.randn(B, N, V)
    teacher_logits = torch.randn(B, N, V)
    labels = torch.randint(0, V, (B, N))

    temperature = 2.0
    alpha = 0.5

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(
        student_log_probs.view(-1, V), teacher_probs.view(-1, V), reduction="batchmean"
    ) * (temperature ** 2)
    ce = F.cross_entropy(student_logits.view(-1, V), labels.view(-1))
    loss = alpha * kl + (1 - alpha) * ce

    assert not torch.isnan(loss), "NaN loss"
    print(f"  KL={kl.item():.4f}  CE={ce.item():.4f}  Total={loss.item():.4f}  OK\n")


def test_gradient_flow():
    """Verify gradients flow through causal HFP."""
    print("=== Test Gradient Flow (Causal) ===")
    config = HFPConfig(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=256,
        chunk_size=16, dtype=torch.float32,
    )
    model = HFPForCausalLM(config).float()

    input_ids = torch.randint(0, 256, (1, 32))
    labels = torch.randint(0, 256, (1, 32))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 256), labels.view(-1))
    loss.backward()

    hfp_grads = {}
    for name, p in model.named_parameters():
        if any(k in name for k in ["dw_conv", "pw_conv", "gate_linear", "alpha_raw"]):
            hfp_grads[name] = p.grad is not None and p.grad.abs().sum() > 0

    all_ok = all(hfp_grads.values())
    for name, has_grad in hfp_grads.items():
        status = "OK" if has_grad else "FAIL"
        print(f"  {status}: {name}")

    assert all_ok, "Some HFP params have no gradient!"
    print(f"  OK\n")


if __name__ == "__main__":
    test_hfp_layer()
    test_causal_hfp_layer()
    test_causal_masking()
    test_full_model()
    test_distillation_loss()
    test_gradient_flow()
    print("All tests passed!")
