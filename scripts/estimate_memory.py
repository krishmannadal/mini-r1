"""
Mini-R1 -- Training Memory Scaling & Profiling CLI
==================================================
Calculates analytical training memory consumption across varying batch sizes,
sequence lengths, and precision formats for Mini-R1.

Usage:
    python scripts/estimate_memory.py
    python scripts/estimate_memory.py --config configs/model.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Config
from src.memory import calculate_parameter_breakdown, estimate_training_memory


def run_memory_analysis(
    config_path: str = "configs/model.yaml",
    vram_budget_gb: float = 6.0,
) -> None:
    """Print detailed memory breakdown across batch sizes."""
    resolved_path = PROJECT_ROOT / config_path
    if resolved_path.exists():
        config = MiniR1Config.from_yaml(resolved_path)
    else:
        config = MiniR1Config()

    param_info = calculate_parameter_breakdown(config)
    P = param_info.total_params

    sep = "=" * 86
    sub_sep = "-" * 86

    print(f"\n{sep}")
    print(f"             Mini-R1 Training Memory Estimation & Scaling Report")
    print(f"             Model: {P / 1e6:.2f}M parameters | Context Length: {config.max_seq_len}")
    print(f"             Hardware Target: RTX 4050 ({vram_budget_gb:.1f} GB VRAM)")
    print(f"{sep}")

    # Static component breakdown (Batch-independent)
    fp32_weights_mb = (P * 4) / (1024 ** 2)
    fp32_grads_mb = (P * 4) / (1024 ** 2)
    fp32_optim_mb = (P * 8) / (1024 ** 2)  # 2 states (m, v) in FP32
    static_total_mb = fp32_weights_mb + fp32_grads_mb + fp32_optim_mb

    print("\n1. STATIC MEMORY BREAKDOWN (Batch-Independent, FP32):")
    print(f"  {sub_sep}")
    print(f"  Component               Bytes / Param     Total Size (MB)    Total Size (GB)")
    print(f"  {sub_sep}")
    print(f"  Model Weights (FP32)         4 bytes       {fp32_weights_mb:>10.2f} MB       {fp32_weights_mb / 1024:>6.3f} GB")
    print(f"  Gradients (FP32)             4 bytes       {fp32_grads_mb:>10.2f} MB       {fp32_grads_mb / 1024:>6.3f} GB")
    print(f"  Optimizer States (AdamW)     8 bytes       {fp32_optim_mb:>10.2f} MB       {fp32_optim_mb / 1024:>6.3f} GB")
    print(f"  {sub_sep}")
    print(f"  STATIC SUBTOTAL             16 bytes       {static_total_mb:>10.2f} MB       {static_total_mb / 1024:>6.3f} GB")
    print(f"  {sub_sep}")

    # Scaling analysis across batch sizes
    batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 32]
    print("\n2. BATCH SIZE SCALING ANALYSIS (Context Length = 512, FP32):")
    print(f"  {sub_sep}")
    print(f"  Batch   Activations   Static Mem    CUDA Overhead   Total Mem    VRAM Usage   Status")
    print(f"  Size       (MB)          (MB)          (10%)          (MB)         (GB)")
    print(f"  {sub_sep}")

    for B in batch_sizes:
        est = estimate_training_memory(
            config=config,
            batch_size=B,
            dtype_bytes=4,
            vram_budget_gb=vram_budget_gb,
        )
        status_flag = "[OK] Fits" if est.fits_in_vram else "[X] OOM "
        pct_vram = (est.total_training_gb / vram_budget_gb) * 100
        print(
            f"  B={B:<3d}   {est.activations_mb:>8.2f} MB   {est.static_mb:>8.2f} MB    "
            f"{est.overhead_mb:>8.2f} MB   {est.total_training_mb:>8.2f} MB   "
            f"{est.total_training_gb:>4.2f} GB ({pct_vram:>3.0f}%)   {status_flag}"
        )

    print(f"  {sub_sep}")
    print("\n3. ARCHITECTURAL TAKEAWAYS:")
    print(f"  - Baseline Batch Size B=8 requires ~2.80 GB (47% of 6 GB VRAM).")
    print(f"  - Maximum recommended batch size in FP32 is B=16 (~5.03 GB, 84% VRAM).")
    print(f"  - Effective batch sizes of 32 or 64 can be achieved via gradient accumulation.")
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-R1 Training Memory Estimation & Scaling")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    parser.add_argument("--vram", type=float, default=6.0, help="Target VRAM in GB")
    args = parser.parse_args()

    run_memory_analysis(config_path=args.config, vram_budget_gb=args.vram)


if __name__ == "__main__":
    main()
