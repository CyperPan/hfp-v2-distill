#!/usr/bin/env python3
"""
Inference with a distilled HFP model.

Usage:
    python inference.py \
        --checkpoint ./checkpoints/hfp-distill/checkpoint-best \
        --tokenizer Qwen/Qwen2.5-1.5B \
        --prompt "The future of artificial intelligence" \
        --max_new_tokens 128
"""

import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from hfp_distill.config import HFPConfig
from hfp_distill.model import HFPForCausalLM


def load_model(checkpoint_dir: str, device: str = "cuda") -> tuple[HFPForCausalLM, HFPConfig]:
    """Load a trained HFP model from checkpoint."""
    with open(f"{checkpoint_dir}/hfp_config.json") as f:
        config_dict = json.load(f)

    # Parse dtype string back
    dtype_str = config_dict.pop("dtype", "torch.bfloat16")
    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    config = HFPConfig(**config_dict, dtype=dtype)

    model = HFPForCausalLM(config).to(dtype).to(device)
    state_dict = torch.load(f"{checkpoint_dir}/model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="HFP Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name")
    parser.add_argument("--prompt", type=str, default="Hello, I am")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, config = load_model(args.checkpoint, device=device)

    params = model.count_parameters()
    print(f"Model loaded: {params['total']/1e6:.1f}M params ({params['hfp_specific']/1e6:.1f}M HFP-specific)")
    print(f"HFP config: freq_cutoff_ratio={config.freq_cutoff_ratio}, conv_kernel_size={config.conv_kernel_size}")
    print("-" * 60)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0

    generated = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    num_tokens = output_ids.shape[1] - prompt_len

    print(f"Prompt: {args.prompt}")
    print(f"Generated ({num_tokens} tokens, {elapsed:.2f}s, {num_tokens/elapsed:.1f} tok/s):")
    print(generated)


if __name__ == "__main__":
    main()
