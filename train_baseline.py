#!/usr/bin/env python3
"""
Baseline Distillation: Standard Attention Student (no HFP)
Same setup as train_distill.py for fair A/B comparison.

Usage:
    python train_baseline.py \
        --teacher Qwen/Qwen2.5-0.5B \
        --dataset wikitext --dataset_config wikitext-103-raw-v1 \
        --seq_len 512 --batch_size 4 --epochs 1 \
        --max_samples 5000 \
        --output_dir ./checkpoints/baseline-qwen05b-verify
"""

import argparse
import json
import logging
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from hfp_distill.config import HFPConfig
from hfp_distill.model_baseline import BaselineForCausalLM
from hfp_distill.loader import load_teacher_for_distillation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_name, dataset_config, split, seq_len, max_samples=None):
        from datasets import load_dataset
        logger.info(f"Loading dataset {dataset_name}/{dataset_config} split={split}")
        raw = load_dataset(dataset_name, dataset_config, split=split)
        text_col = "text" if "text" in raw.column_names else raw.column_names[0]
        all_text = "\n\n".join(t for t in raw[text_col] if t.strip())
        logger.info(f"Tokenizing {len(all_text):,} characters ...")
        tokens = tokenizer.encode(all_text, add_special_tokens=False)
        logger.info(f"Got {len(tokens):,} tokens, chunking into seq_len={seq_len+1}")
        self.chunks = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            self.chunks.append(tokens[i : i + seq_len + 1])
            if max_samples and len(self.chunks) >= max_samples:
                break
        logger.info(f"Created {len(self.chunks):,} training samples")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    V = student_logits.size(-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(
        student_log_probs.view(-1, V), teacher_probs.view(-1, V), reduction="batchmean"
    ) * (temperature ** 2)
    ce = F.cross_entropy(student_logits.view(-1, V), labels.view(-1))
    loss = alpha * kl + (1.0 - alpha) * ce
    return loss, {"kl": kl.item(), "ce": ce.item(), "loss": loss.item()}


def load_teacher_weights_into_baseline(student, teacher_name_or_path, dtype):
    """Load teacher weights into baseline student (direct copy, same architecture)."""
    from transformers import AutoModelForCausalLM
    from collections import OrderedDict

    logger.info(f"Loading teacher weights into baseline student from {teacher_name_or_path}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name_or_path, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    teacher_sd = teacher.state_dict()
    del teacher
    torch.cuda.empty_cache()

    student_sd = student.state_dict()
    new_sd = OrderedDict()
    loaded, skipped = 0, 0

    # Mapping: HF key -> baseline key
    for teacher_key, teacher_val in teacher_sd.items():
        # self_attn.{q,k,v,o}_proj -> self_attn.{q,k,v,o}_proj (same name)
        student_key = teacher_key
        # Skip rotary_emb (we have our own), q_norm, k_norm
        if any(p in teacher_key for p in ["rotary_emb", "q_norm", "k_norm"]):
            skipped += 1
            continue
        if student_key in student_sd and student_sd[student_key].shape == teacher_val.shape:
            new_sd[student_key] = teacher_val.to(dtype)
            loaded += 1
        else:
            skipped += 1

    student.load_state_dict(new_sd, strict=False)
    logger.info(f"Loaded {loaded} keys, skipped {skipped}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build baseline student
    logger.info("Building baseline (standard attention) student model ...")
    config = HFPConfig.from_pretrained(args.teacher, dtype=dtype)
    student = BaselineForCausalLM(config).to(dtype).to(device)

    load_teacher_weights_into_baseline(student, args.teacher, dtype=dtype)
    student = student.to(device)

    params = student.count_parameters()
    logger.info(f"Baseline student: {params['total']/1e6:.1f}M params (0 HFP-specific)")

    # Load frozen teacher
    logger.info("Loading frozen teacher model ...")
    teacher = load_teacher_for_distillation(args.teacher, dtype=dtype, device=device)

    # Dataset
    dataset = TextDataset(
        tokenizer, args.dataset, args.dataset_config,
        split=args.split, seq_len=args.seq_len, max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Optimizer (single LR, no HFP split)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out

            with torch.amp.autocast("cuda", dtype=dtype):
                student_logits = student(input_ids, use_gradient_checkpointing=args.gradient_checkpointing)
                loss, metrics = distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=args.temperature, alpha=args.alpha,
                )

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss += metrics["loss"]
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr_curr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) * args.batch_size / elapsed
                logger.info(
                    f"[BASELINE] Epoch {epoch+1}/{args.epochs} "
                    f"Step {global_step}/{total_steps} "
                    f"Loss={avg_loss:.4f} KL={metrics['kl']:.4f} CE={metrics['ce']:.4f} "
                    f"LR={lr_curr:.2e} Speed={speed:.1f} samples/s"
                )

            if global_step % args.save_every == 0:
                path = os.path.join(args.output_dir, f"checkpoint-step{global_step}")
                os.makedirs(path, exist_ok=True)
                torch.save(student.state_dict(), os.path.join(path, "model.pt"))
                logger.info(f"Saved checkpoint to {path}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - epoch_start
        logger.info(
            f"[BASELINE] Epoch {epoch+1} done — Avg Loss={avg_epoch_loss:.4f} Time={elapsed/60:.1f}min"
        )

    # Final save
    path = os.path.join(args.output_dir, "checkpoint-final")
    os.makedirs(path, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(path, "model.pt"))
    logger.info(f"[BASELINE] Training complete. Final avg loss: {avg_epoch_loss:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Baseline Distillation (Standard Attention)")
    p.add_argument("--teacher", type=str, required=True)
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--output_dir", type=str, default="./checkpoints/baseline-verify")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
