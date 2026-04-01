"""
Weight loader: map HuggingFace pretrained weights → HFP student model.

Mapping:
  Teacher (HF standard)                  →  Student (HFP)
  ─────────────────────────────────────────────────────────────
  model.embed_tokens.weight              →  model.embed_tokens.weight        ✓ copy
  model.layers.i.self_attn.q_proj.*      →  model.layers.i.hfp.q_proj.*     ✓ copy
  model.layers.i.self_attn.k_proj.*      →  model.layers.i.hfp.k_proj.*     ✓ copy
  model.layers.i.self_attn.v_proj.*      →  model.layers.i.hfp.v_proj.*     ✓ copy
  model.layers.i.self_attn.o_proj.*      →  model.layers.i.hfp.o_proj.*     ✓ copy
  model.layers.i.mlp.*                   →  model.layers.i.mlp.*            ✓ copy
  model.layers.i.input_layernorm.*       →  model.layers.i.input_layernorm.*✓ copy
  model.layers.i.post_attention_layernorm.* → ...                           ✓ copy
  model.norm.weight                      →  model.norm.weight               ✓ copy
  lm_head.weight                         →  lm_head.weight                  ✓ copy
  ─────────────────────────────────────────────────────────────
  (no teacher equivalent)                →  hfp.dw_conv_*, pw_conv_*        ✗ random init
  (no teacher equivalent)                →  hfp.gate_linear.*               ✗ random init
  (no teacher equivalent)                →  hfp.alpha_raw                   ✗ zero init
"""

import re
import logging
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM

from .config import HFPConfig
from .model import HFPForCausalLM

logger = logging.getLogger(__name__)

# Keys that only exist in HFP student (no teacher equivalent)
HFP_ONLY_PATTERNS = ["dw_conv", "pw_conv", "gate_linear", "alpha_raw"]


def _rename_key(teacher_key: str) -> str | None:
    """Convert a teacher state_dict key to student key. Returns None if no mapping."""
    # self_attn.{q,k,v,o}_proj → hfp.{q,k,v,o}_proj
    new_key = teacher_key.replace("self_attn.", "hfp.")

    # Skip teacher-only keys (e.g. rotary_emb, q_norm, k_norm — not used in HFP)
    skip_patterns = ["rotary_emb", "q_norm", "k_norm"]
    if any(p in new_key for p in skip_patterns):
        return None

    return new_key


def load_teacher_weights_into_student(
    student: HFPForCausalLM,
    teacher_name_or_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[str], list[str]]:
    """
    Load teacher weights into student model.

    Returns:
        (loaded_keys, skipped_keys)
    """
    logger.info(f"Loading teacher weights from {teacher_name_or_path}")

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name_or_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    teacher_sd = teacher.state_dict()
    del teacher  # free memory
    torch.cuda.empty_cache()

    student_sd = student.state_dict()
    new_sd = OrderedDict()

    loaded_keys = []
    skipped_keys = []

    for teacher_key, teacher_val in teacher_sd.items():
        student_key = _rename_key(teacher_key)
        if student_key is None:
            skipped_keys.append(teacher_key)
            continue

        if student_key in student_sd:
            if student_sd[student_key].shape == teacher_val.shape:
                new_sd[student_key] = teacher_val.to(dtype)
                loaded_keys.append(f"{teacher_key} → {student_key}")
            else:
                logger.warning(
                    f"Shape mismatch: {teacher_key} {teacher_val.shape} "
                    f"vs {student_key} {student_sd[student_key].shape}, skipping"
                )
                skipped_keys.append(teacher_key)
        else:
            skipped_keys.append(teacher_key)

    # Load matched weights (strict=False keeps HFP-specific random init)
    missing, unexpected = student.load_state_dict(new_sd, strict=False)

    # missing keys should only be HFP-specific
    hfp_missing = [k for k in missing if any(p in k for p in HFP_ONLY_PATTERNS)]
    real_missing = [k for k in missing if not any(p in k for p in HFP_ONLY_PATTERNS)]

    if real_missing:
        logger.warning(f"Non-HFP keys missing from teacher: {real_missing}")

    logger.info(
        f"Loaded {len(loaded_keys)} keys from teacher, "
        f"skipped {len(skipped_keys)}, "
        f"{len(hfp_missing)} HFP-specific keys kept at init"
    )

    return loaded_keys, skipped_keys


def load_teacher_for_distillation(
    teacher_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load and freeze teacher model for distillation."""
    logger.info(f"Loading teacher model: {teacher_name_or_path}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
