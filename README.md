# HFP V2 — Hybrid Frequency Processing for Transformers

A distillation framework that transfers knowledge from standard Transformer models into a novel **frequency-domain architecture**, replacing O(N²) attention with O(k²D + ND log N) hybrid processing.

## Core Idea

```
Y = IFFT[ Attn(FFT[X]_{f<k})  ⊕  (DWConv ⊙ Gate)(FFT[X]_{f≥k}) ]
         ╰── global semantics ──╯    ╰──── local details ────╯
```

Standard attention processes all N tokens with O(N²) cost. HFP V2 instead:

1. **FFT** — transform the sequence into frequency domain
2. **Split** — separate low-frequency (global trends) from high-frequency (local details)
3. **Low-freq** — standard attention on only k frequency bins (k = N/8 → **64x less compute**)
4. **High-freq** — lightweight depthwise conv + gating (linear cost)
5. **IFFT** — transform back to time domain

### Complexity Comparison

| Component | Standard Transformer | HFP V2 |
|-----------|---------------------|---------|
| Attention | O(N² D) | O(k² D) |
| High-freq | — | O((M-k) D κ) |
| FFT/IFFT | — | O(ND log N) |
| **Total** | **O(N² D)** | **O(k²D + ND log N)** |

## Architecture

```
Input [B, N, D]
    │
    ▼
┌─────────────────────────────────────────┐
│              LayerNorm                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           FFT (seq dim)                 │
│         X_hat ∈ C^{B×M×D}              │
└────────┬────────────────────┬───────────┘
         │                    │
    f < k (low)          f ≥ k (high)
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌────────────────────┐
│  Multi-Head     │  │  DW Conv (κ=7)     │
│  Attention      │  │  → GELU            │
│  (on k bins)    │  │  → PW Conv         │
│                 │  │  × Sigmoid Gate    │
│  Q,K,V from     │  │  + Residual        │
│  teacher weights│  │                    │
└────────┬────────┘  └──────────┬─────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────┐
│  α·A_low + (1-α)·X_low  ‖  A_high     │
│         Concat & IFFT                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
              + Residual
                  │
                  ▼
┌─────────────────────────────────────────┐
│         LayerNorm → FFN (SwiGLU)        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
              + Residual → Output [B, N, D]
```

## Distillation Approach

```
Teacher (Qwen2.5-0.5B, frozen)  ──logits──┐
                                           ▼
                                    KL Divergence ──┐
                                                    ├── Loss
Input ──► Student (HFP V2) ──logits──► Cross-Entropy ──┘
```

**Loss** = α · T² · KL(student_soft ∥ teacher_soft) + (1-α) · CE(student, labels)

**Weight initialization**: Attention Q/K/V/O, MLP, embeddings copied from teacher. HFP-specific weights (conv, gate, α) initialized randomly with 5x higher learning rate.

## Experiment Results

Verification experiment: Qwen2.5-0.5B teacher, 5000 samples, 1 epoch, V100-SXM2-32GB.

### Loss Convergence

Total distillation loss: **17.49 → 9.48 (45.8% reduction)**

![Total Loss](figures/01_total_loss.png)

### KL Divergence & Cross-Entropy

- KL: 15.64 → 7.43 (**52.5% reduction**) — student increasingly matches teacher
- CE: 18.77 → 7.98 (**57.5% reduction**) — language modeling improving

![KL and CE](figures/02_kl_ce_loss.png)

### Loss Reduction Summary

![Reduction](figures/03_loss_reduction.png)

### Training Phases

![Phases](figures/04_training_phases.png)

> Full experiment report with methodology details: [EXPERIMENT.md](EXPERIMENT.md)

## Quick Start

### Install

```bash
pip install torch>=2.1.0 transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.27.0
```

### Verify

```bash
python test_smoke.py
```

### Train (Distillation)

```bash
# Single GPU — quick verification
python train_distill.py \
    --teacher Qwen/Qwen2.5-0.5B \
    --dataset wikitext --dataset_config wikitext-103-raw-v1 \
    --seq_len 512 --batch_size 4 --epochs 1 \
    --max_samples 5000 \
    --output_dir ./checkpoints/hfp-qwen05b-verify

# Single GPU — full training
python train_distill.py \
    --teacher Qwen/Qwen2.5-0.5B \
    --dataset wikitext --dataset_config wikitext-103-raw-v1 \
    --seq_len 512 --batch_size 4 --epochs 3 \
    --gradient_checkpointing \
    --output_dir ./checkpoints/hfp-qwen05b-full

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train_distill.py \
    --teacher Qwen/Qwen2.5-1.5B \
    --seq_len 512 --batch_size 2 --epochs 3 \
    --gradient_checkpointing \
    --output_dir ./checkpoints/hfp-qwen15b
```

### Inference

```bash
python inference.py \
    --checkpoint ./checkpoints/hfp-qwen05b-verify/checkpoint-best \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --prompt "The future of artificial intelligence"
```

## Project Structure

```
hfp-v2-distill/
├── hfp_distill/
│   ├── config.py            # HFPConfig — auto-populates from HuggingFace models
│   ├── hfp_layer.py         # HFP V2 layer (FFT → Attn/Conv → IFFT)
│   ├── model.py             # Full causal LM with HFP decoder layers
│   ├── model_baseline.py    # Standard attention baseline for A/B comparison
│   └── loader.py            # Teacher weight mapping into HFP student
├── train_distill.py         # Distillation training (HFP student)
├── train_baseline.py        # Distillation training (standard attention baseline)
├── inference.py             # Generation with trained HFP model
├── test_smoke.py            # Unit tests (layer, model, loss, gradient flow)
├── plot_results.py          # Experiment visualization
├── figures/                 # Generated plots
├── EXPERIMENT.md            # Full experiment report
└── pyproject.toml
```

## Supported Teacher Models

Any HuggingFace causal LM with RMSNorm + SwiGLU + GQA architecture:

| Model Family | Example | Tested |
|-------------|---------|--------|
| Qwen2/2.5 | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` | Yes |
| Qwen3 | `Qwen/Qwen3-0.6B` | Planned |
| LLaMA 3 | `meta-llama/Llama-3.2-1B` | Planned |
| Mistral | `mistralai/Mistral-7B-v0.1` | Planned |

## Roadmap

- [x] HFP V2 architecture implementation
- [x] Distillation training pipeline
- [x] Verification experiment (0.5B, 5K samples)
- [ ] Baseline A/B comparison (standard attention vs HFP)
- [ ] Full-scale training (WikiText-103, 3 epochs)
- [ ] Long sequence experiments (2048/4096) — complexity advantage
- [ ] Perplexity evaluation on test set
- [ ] Training speed & memory comparison
- [ ] Multi-model support (LLaMA, Mistral)

## License

MIT
