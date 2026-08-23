"""
Mini-R1 -- Gradient Accumulation Equivalence Verification
=========================================================
Tests and proves that computing gradients across micro-batches with gradient
accumulation (L_micro / accum_steps) produces mathematically identical parameter
gradients to computing the full batch in a single forward/backward pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config
from src.loss import compute_causal_lm_loss


def run_gradient_accumulation_verification(
    precision: str = "float16",
    device: str | None = None,
) -> dict[str, float]:
    """
    Run deterministic gradient accumulation comparison on 8 fixed sequences.
    
    Path A: micro_batch=8, accum=1
    Path B: micro_batch=4, accum=2
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Setup deterministic model
    config = MiniR1Config(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=64,
        bias=False,
        tie_word_embeddings=True,
    )
    model = MiniR1Transformer(config).to(device)

    # Clone initial state for exact replication
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 2. Fixed input batch (8 sequences of length 32)
    inputs = torch.randint(10, 1000, (8, 32), dtype=torch.long, device=device)
    targets = inputs.clone()
    targets[:, 0] = -100  # mask first token

    amp_dtype = torch.float16 if precision == "float16" else torch.float32
    amp_enabled = precision == "float16" and device.startswith("cuda")

    # =========================================================================
    # PATH A: Single Full Batch (Batch Size 8, Accumulation 1)
    # =========================================================================
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=amp_dtype, enabled=amp_enabled):
        logits_a = model(inputs)
        loss_a = compute_causal_lm_loss(logits_a, targets)

    loss_a.backward()
    grads_a = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    # =========================================================================
    # PATH B: Accumulated Micro-Batches (Batch Size 4, Accumulation 2)
    # =========================================================================
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)

    accum_steps = 2
    # Micro-batch 1 (first 4 samples)
    with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=amp_dtype, enabled=amp_enabled):
        logits_b1 = model(inputs[:4])
        loss_b1 = compute_causal_lm_loss(logits_b1, targets[:4])
        scaled_b1 = loss_b1 / accum_steps

    scaled_b1.backward()

    # Micro-batch 2 (second 4 samples)
    with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=amp_dtype, enabled=amp_enabled):
        logits_b2 = model(inputs[4:])
        loss_b2 = compute_causal_lm_loss(logits_b2, targets[4:])
        scaled_b2 = loss_b2 / accum_steps

    scaled_b2.backward()
    grads_b = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    # =========================================================================
    # Compare Gradients
    # =========================================================================
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_elements = 0
    diff_l2_sq = 0.0
    norm_a_l2_sq = 0.0

    for name in grads_a:
        g_a = grads_a[name].float()
        g_b = grads_b[name].float()
        diff = torch.abs(g_a - g_b)

        param_max = diff.max().item()
        if param_max > max_abs_diff:
            max_abs_diff = param_max

        total_abs_diff += diff.sum().item()
        total_elements += diff.numel()
        diff_l2_sq += (diff ** 2).sum().item()
        norm_a_l2_sq += (g_a ** 2).sum().item()

    mean_abs_diff = total_abs_diff / max(1, total_elements)
    rel_error = (diff_l2_sq ** 0.5) / max(1e-9, (norm_a_l2_sq ** 0.5))
    loss_diff = abs(loss_a.item() - ((loss_b1.item() + loss_b2.item()) / 2.0))

    results = {
        "precision": precision,
        "device": device,
        "loss_path_a": loss_a.item(),
        "loss_path_b_avg": (loss_b1.item() + loss_b2.item()) / 2.0,
        "loss_diff": loss_diff,
        "max_abs_gradient_diff": max_abs_diff,
        "mean_abs_gradient_diff": mean_abs_diff,
        "relative_l2_error": rel_error,
    }

    print("\n" + "=" * 75)
    print(f"   Gradient Accumulation Equivalence Verification ({precision.upper()} on {device.upper()})")
    print("=" * 75)
    print(f"  Loss Path A (Full B=8, K=1)        : {loss_a.item():.6f}")
    print(f"  Loss Path B (Micro B=4, K=2 avg)   : {(loss_b1.item() + loss_b2.item()) / 2.0:.6f}")
    print(f"  Absolute Loss Difference           : {loss_diff:.2e}")
    print(f"  Max Absolute Gradient Difference   : {max_abs_diff:.2e}")
    print(f"  Mean Absolute Gradient Difference  : {mean_abs_diff:.2e}")
    print(f"  Relative L2 Gradient Error         : {rel_error:.2e} ({rel_error * 100:.4f}%)")
    print("=" * 75)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Gradient Accumulation Equivalence.")
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_gradient_accumulation_verification(precision=args.precision, device=args.device)


if __name__ == "__main__":
    main()
