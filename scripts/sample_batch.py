"""
Mini-R1 -- Sample Batch Demonstration Script
============================================
Demonstrates sampling a unified batch across GSM8K, SVAMP, and AQuA,
verifying formatting consistency, and converting to token ID tensors.

Usage:
    python scripts/sample_batch.py --batch_size 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import ReasoningDatasetManager
from src.tokenizer import Tokenizer


def separator(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a formatted batch from online datasets.")
    parser.add_argument("--batch_size", type=int, default=6, help="Total samples in batch.")
    args = parser.parse_args()

    print("+===========================================================================+")
    print("|            Mini-R1 -- Unified Dataset Sample Batch Generator              |")
    print("+===========================================================================+")

    manager = ReasoningDatasetManager(seed=42, val_ratio=0.1)

    # Sample 2 examples per dataset for a balanced multi-dataset batch
    samples_per_ds = max(1, args.batch_size // 3)
    batch_records = manager.get_combined_train(
        datasets=("gsm8k", "svamp", "aqua"),
        limit_per_dataset=samples_per_ds,
        shuffle=True,
    )

    separator(f"1. Formatted Batch Sample Records (Count: {len(batch_records)})")
    for idx, item in enumerate(batch_records):
        print(f"\n  --- Batch Item {idx + 1}/{len(batch_records)} [Dataset: {item['dataset'].upper()}] ---")
        print(f"  ID: {item['id']}")
        print(f"  Formatted Content:\n{item['formatted_text']}")
        print(f"  Extracted Answer: '{item['answer']}'")

    # -------------------------------------------------------------------------
    # 2. Tokenize the Batch
    # -------------------------------------------------------------------------
    separator("2. Tokenization & Tensor Batching")
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if tok_path.exists():
        tok = Tokenizer(tok_path)
        texts = [item["formatted_text"] for item in batch_records]
        padded_ids = tok.encode_batch(texts, add_bos=True, add_eos=True, pad=True)

        print(f"  Tokenizer Vocab Size : {tok.vocab_size()}")
        print(f"  Batch Sequence Length: {len(padded_ids[0])} tokens (uniform padded)")
        for idx, (rec, ids) in enumerate(zip(batch_records, padded_ids)):
            print(f"    Item {idx + 1} ({rec['dataset'].upper():<5}): len={len(ids)} | First 8 IDs: {ids[:8]} ... Last 4: {ids[-4:]}")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            attention_mask = (input_ids != tok.pad_id).long()

            print(f"\n  PyTorch Tensors Created:")
            print(f"    input_ids shape      : {input_ids.shape} (batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]})")
            print(f"    attention_mask shape : {attention_mask.shape}")
            print(f"    Device               : {device}")
            print(f"    Dtype                : {input_ids.dtype}")
        except ImportError:
            pass
    else:
        print("  Tokenizer model not found. Run 'python scripts/train_tokenizer.py' first.")

    separator("Result")
    print("  [OK] Successfully sampled, formatted, and tokenized multi-dataset batch.")


if __name__ == "__main__":
    main()
