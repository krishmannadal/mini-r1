"""
Mini-R1 -- Standalone Validation Script
=======================================
Loads a trained checkpoint and evaluates loss and perplexity on the validation dataset.

Usage:
    python scripts/validate.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config
from src.tokenizer import Tokenizer
from src.datasets import ReasoningDatasetManager
from src.torch_dataset import ReasoningTorchDataset
from src.dataloader import create_dataloader
from src.checkpoint import load_checkpoint
from src.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mini-R1 checkpoint on validation data.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint .pt file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for validation.")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of validation samples to evaluate.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    print(f"+===========================================================================+")
    print(f"|            Mini-R1 Transformer -- Standalone Model Evaluation            |")
    print(f"+===========================================================================+")
    print(f"  Checkpoint File : {ckpt_path.resolve()}")
    print(f"  Device          : {device}")

    # 1. Load Tokenizer
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    # 2. Build Model Skeleton & Load Checkpoint
    config = MiniR1Config(
        vocab_size=tokenizer.vocab_size(),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=32,
    )
    model = MiniR1Transformer(config).to(device)
    meta = load_checkpoint(ckpt_path, model=model, device=device)

    print(f"  Loaded Model Epoch : {meta.get('epoch')}")
    print(f"  Global Step Count  : {meta.get('global_step')}")
    print(f"  Stored Val Loss    : {meta.get('val_loss'):.4f}")

    # 3. Load Validation Dataset
    dataset_mgr = ReasoningDatasetManager(seed=42, val_ratio=0.1)
    val_records = []
    for d in ("gsm8k", "svamp", "aqua"):
        val_records.extend(dataset_mgr.get_formatted_split(d, split="validation", limit=args.max_samples // 3))

    val_dataset = ReasoningTorchDataset(val_records, tokenizer=tokenizer, max_seq_len=32)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, pad_id=tokenizer.pad_id)

    # 4. Evaluate
    trainer = Trainer(model=model, optimizer=None, config=config, device=device)
    metrics = trainer.evaluate(val_loader)

    print(f"\n===========================================================================")
    print(f"  Validation Results")
    print(f"===========================================================================")
    print(f"  Validation Loss     : {metrics.avg_loss:.4f}")
    print(f"  Perplexity          : {metrics.perplexity:.2f}")
    print(f"  Total Tokens Tested : {metrics.total_tokens:,d}")
    print(f"  Throughput          : {metrics.tokens_per_sec:.1f} tok/s")
    print(f"  Elapsed Time        : {metrics.elapsed_sec:.2f}s")
    print(f"===========================================================================\n")


if __name__ == "__main__":
    main()
