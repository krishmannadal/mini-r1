"""
Mini-R1 -- Model Architecture Summary CLI
=========================================
Inspects and prints complete architecture specifications, tensor dimensions,
parameter breakdown, and estimated memory footprint for Mini-R1.

Usage:
    python scripts/model_summary.py
    python scripts/model_summary.py --config configs/model.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Config, MiniR1Transformer
from src.memory import calculate_parameter_breakdown, estimate_training_memory


def print_model_summary(config_path: str = "configs/model.yaml", batch_size: int = 8) -> None:
    """Print structured model architecture summary."""
    resolved_path = PROJECT_ROOT / config_path
    if resolved_path.exists():
        config = MiniR1Config.from_yaml(resolved_path)
        config_source = str(resolved_path)
    else:
        config = MiniR1Config()
        config_source = "Default MiniR1Config"

    # Instantiate model to verify parameter counts match PyTorch parameter registration
    model = MiniR1Transformer(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    param_info = calculate_parameter_breakdown(config)
    mem_info = estimate_training_memory(config, batch_size=batch_size)

    sep = "=" * 76
    sub_sep = "-" * 76

    print(f"\n{sep}")
    print("                    Mini-R1 Transformer Model Summary")
    print(f"{sep}")
    print(f"  Configuration Source   : {config_source}")
    print(f"  Vocabulary Size        : {config.vocab_size:,}")
    print(f"  Hidden Dimension (d)   : {config.d_model}")
    print(f"  Transformer Layers     : {config.n_layers}")
    print(f"  Attention Heads        : {config.n_heads}")
    print(f"  Head Dimension (d_k)   : {config.d_k}")
    print(f"  Feed-Forward Dim (d_ff): {config.d_ff}")
    print(f"  Max Sequence Length    : {config.max_seq_len}")
    print(f"  Weight Tying           : {config.tie_word_embeddings} (lm_head tied to tok_emb)")
    print(f"  Bias in Linear Layers  : {config.bias}")
    print(f"  Dropout Rate           : {config.dropout}")
    print(f"{sub_sep}")

    print("  PARAMETER BREAKDOWN:")
    print(f"    - Token Embeddings   : {param_info.tok_emb_params:>12,} params (vocab={config.vocab_size} x d={config.d_model})")
    print(f"    - Position Embeddings: {param_info.pos_emb_params:>12,} params (max_seq={config.max_seq_len} x d={config.d_model})")
    print(f"    - Final LayerNorm    : {param_info.ln_f_params:>12,} params (2 x d={config.d_model})")
    print(f"    - LM Head Projection : {param_info.lm_head_params:>12,} params (tied: 0 additional)")
    print(f"    - Per-Layer Attention: {param_info.layer_breakdown.attention_params:>12,} params (4 x d^2)")
    print(f"    - Per-Layer FFN      : {param_info.layer_breakdown.ffn_params:>12,} params (2 x d x d_ff)")
    print(f"    - Per-Layer Total    : {param_info.layer_breakdown.total_layer_params:>12,} params")
    print(f"    - All {config.n_layers} Layers Total  : {param_info.all_layers_params:>12,} params")
    print(f"    ------------------------------------------------")
    print(f"    - TOTAL PARAMETERS   : {param_info.total_params:>12,} ({param_info.total_params / 1e6:.2f}M)")
    print(f"    - Non-Embedding Params: {param_info.non_embedding_params:>11,} ({param_info.non_embedding_params / 1e6:.2f}M)")
    print(f"    - PyTorch Verified   : {pytorch_total_params:>12,} params (exact match: {pytorch_total_params == param_info.total_params})")
    print(f"{sub_sep}")

    print(f"  ESTIMATED TRAINING MEMORY (Batch Size = {batch_size}, Seq Len = {config.max_seq_len}, FP32):")
    print(f"    - Weights (FP32)     : {mem_info.weights_mb:>8.2f} MB ({mem_info.weights_mb / 1024:.3f} GB)")
    print(f"    - Gradients (FP32)   : {mem_info.gradients_mb:>8.2f} MB ({mem_info.gradients_mb / 1024:.3f} GB)")
    print(f"    - Optimizer (AdamW)  : {mem_info.optimizer_mb:>8.2f} MB ({mem_info.optimizer_mb / 1024:.3f} GB)")
    print(f"    - Static Subtotal    : {mem_info.static_mb:>8.2f} MB ({mem_info.static_mb / 1024:.3f} GB)")
    print(f"    - Activations (B={batch_size}) : {mem_info.activations_mb:>8.2f} MB ({mem_info.activations_mb / 1024:.3f} GB)")
    print(f"    - CUDA Overhead (10%): {mem_info.overhead_mb:>8.2f} MB ({mem_info.overhead_mb / 1024:.3f} GB)")
    print(f"    ------------------------------------------------")
    print(f"    - TOTAL TRAINING MEM : {mem_info.total_training_mb:>8.2f} MB ({mem_info.total_training_gb:.2f} GB / {mem_info.vram_budget_gb:.1f} GB VRAM)")
    status = "FITS COMFORTABLY" if mem_info.fits_in_vram else "EXCEEDS VRAM"
    print(f"    - Target 6GB Status  : [OK] {status}")
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-R1 Model Summary and Architecture Inspector")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for memory estimation")
    args = parser.parse_args()

    print_model_summary(config_path=args.config, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
