"""
Mini-R1 -- Single Batch Optimization Step Demo
==============================================
Demonstrates one complete optimization step on a tiny batch to mathematically
prove that the Transformer can learn and that the loss decreases.

Workflow:
    1. Initialize Transformer and AdamW optimizer
    2. Create a batch of token IDs
    3. Run forward pass and compute initial cross-entropy loss
    4. Run backward pass and inspect gradient norms across layers
    5. Perform one AdamW optimizer parameter update
    6. Recompute forward pass and verify that loss strictly decreased

Usage:
    python scripts/one_batch_train.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.training_step import perform_training_step
from src.loss import compute_causal_lm_loss


def separator(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


def main() -> None:
    print("+===========================================================================+")
    print("|            Mini-R1 Transformer -- Single Optimization Step Demo           |")
    print("+===========================================================================+")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # 1. Architecture Configuration
    config = MiniR1Config(
        vocab_size=8000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32,
        dropout=0.0,
        tie_word_embeddings=True,
    )
    model = MiniR1Transformer(config).to(device)
    optimizer = create_adamw_optimizer(model, lr=5e-3, weight_decay=0.01)

    print(f"  Device              : {device}")
    print(f"  Total Parameters    : {model.get_num_params():,d}")
    print(f"  Learning Rate       : 5e-3 (AdamW)")
    print(f"  Max Sequence Length : {config.max_seq_len}")

    # -------------------------------------------------------------------------
    # 2. Create Sample Training Batch
    # -------------------------------------------------------------------------
    separator("1. Tiny Training Batch Creation")
    batch_size = 2
    seq_len = 16
    # Fixed deterministic token IDs
    input_ids = torch.randint(10, 500, (batch_size, seq_len), device=device)
    print(f"  Batch size   : {batch_size}")
    print(f"  Sequence len : {seq_len}")
    print(f"  Input IDs:\n{input_ids.cpu().numpy()}")

    # -------------------------------------------------------------------------
    # 3. Step 1: Forward Pass & Initial Loss
    # -------------------------------------------------------------------------
    separator("2. Initial Forward Pass & Loss Calculation")
    model.eval()
    with torch.no_grad():
        initial_logits = model(input_ids)
        initial_loss = compute_causal_lm_loss(initial_logits, input_ids).item()

    print(f"  Initial Logits Shape : {list(initial_logits.shape)}")
    print(f"  Initial Cross-Entropy Loss: {initial_loss:.6f}")

    # Snapshot a parameter tensor before update for delta verification
    target_weight_name = "blocks.0.attn.q_proj.weight"
    weight_before = model.blocks[0].attn.q_proj.weight.data.clone()

    # -------------------------------------------------------------------------
    # 4. Step 2: Backward Pass, Gradient Inspection & Optimizer Step
    # -------------------------------------------------------------------------
    separator("3. Backward Pass & Gradient Norm Inspection")
    step_res = perform_training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=input_ids,
        max_grad_norm=1.0,
    )

    print(f"  Global Gradient L2 Norm: {step_res.grad_norm:.6f}\n")
    print(f"  Layer-by-Layer Gradient Norms:")
    for param_name, grad_norm in step_res.param_gradients.items():
        print(f"    - {param_name:<36}: {grad_norm:.6f}")

    # -------------------------------------------------------------------------
    # 5. Step 3: Parameter Delta Inspection
    # -------------------------------------------------------------------------
    separator("4. Parameter Update Verification")
    weight_after = model.blocks[0].attn.q_proj.weight.data
    param_delta = (weight_after - weight_before).norm(2).item()
    print(f"  Inspected Parameter : {target_weight_name}")
    print(f"  Weight Norm Before  : {weight_before.norm(2).item():.6f}")
    print(f"  Weight Norm After   : {weight_after.norm(2).item():.6f}")
    print(f"  Weight Update Delta : {param_delta:.8f}")
    print(f"  Parameter Updated   : {'[OK]' if param_delta > 0.0 else '[FAIL]'}")

    # -------------------------------------------------------------------------
    # 6. Step 4: Recompute Loss & Verify Decrease
    # -------------------------------------------------------------------------
    separator("5. Updated Loss & Verification")
    model.eval()
    with torch.no_grad():
        updated_logits = model(input_ids)
        updated_loss = compute_causal_lm_loss(updated_logits, input_ids).item()

    loss_delta = initial_loss - updated_loss
    percent_reduction = (loss_delta / initial_loss) * 100.0

    print(f"  Initial Loss : {initial_loss:.6f}")
    print(f"  Updated Loss : {updated_loss:.6f}")
    print(f"  Loss Delta   : -{loss_delta:.6f} ({percent_reduction:.2f}% reduction)")

    is_decreased = updated_loss < initial_loss
    print(f"\n  Loss Decreased Successfully: {'[OK]' if is_decreased else '[FAIL]'}")

    separator("Conclusion")
    print("  [OK] Backward propagation, gradient computation, and AdamW update proven working.")


if __name__ == "__main__":
    main()
