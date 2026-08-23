"""
Mini-R1 -- General Pretraining Corpus Builder & Token Packer
============================================================
Streams educational text from HuggingFaceTB/smollm-corpus, packs documents
contiguously into 512-token chunks, splits into train/validation partitions,
and generates data/manifest.json.

Usage:
    python scripts/build_pretrain_corpus.py --target_tokens 1000000
    python scripts/build_pretrain_corpus.py --config configs/pretrain_250m.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import Tokenizer
from src.general_dataset import GeneralPretrainStreamer, create_pretrain_manifest


def build_pretrain_corpus(
    target_tokens: int = 1_000_000,
    subsets: tuple[str, ...] = ("cosmopedia-v2",),
    val_ratio: float = 0.05,
    seed: int = 42,
    output_dir: Path | None = None,
    save_chunks: bool = True,
) -> dict:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "pretrain"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if not tok_path.exists():
        print(f"[ERROR] Tokenizer missing at {tok_path}")
        sys.exit(1)
    tokenizer = Tokenizer(tok_path)

    print("\n" + "=" * 80)
    print("      Mini-R1: Building General Pretraining Corpus (Streaming & Packing)")
    print("=" * 80)
    print(f"  Source Dataset       : HuggingFaceTB/smollm-corpus")
    print(f"  Subsets              : {list(subsets)}")
    print(f"  Target Token Budget  : {target_tokens:,} tokens")
    print(f"  Validation Ratio     : {val_ratio * 100:.1f}%")
    print(f"  Sequence Length      : 512 tokens (packed)")
    print(f"  Random Seed          : {seed}")
    print(f"  Output Directory     : {output_dir}\n")

    streamer = GeneralPretrainStreamer(
        dataset_name="HuggingFaceTB/smollm-corpus",
        subsets=subsets,
        split="train",
        tokenizer=tokenizer,
        max_seq_len=512,
        seed=seed,
    )

    t0 = time.time()
    train_chunks, val_chunks, stats = streamer.pack_tokens_from_stream(
        target_tokens=target_tokens,
        val_ratio=val_ratio,
    )
    elapsed = time.time() - t0

    # Save packed token chunks as JSONL if requested
    if save_chunks:
        train_file = output_dir / "train_packed_512.jsonl"
        val_file = output_dir / "val_packed_512.jsonl"

        with open(train_file, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(train_chunks):
                f.write(json.dumps({"id": f"train_{idx}", "input_ids": chunk}) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(val_chunks):
                f.write(json.dumps({"id": f"val_{idx}", "input_ids": chunk}) + "\n")

        print(f"  [OK] Saved train chunks : {train_file} ({len(train_chunks):,} sequences)")
        print(f"  [OK] Saved val chunks   : {val_file} ({len(val_chunks):,} sequences)")

    # Save manifest
    manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    manifest = create_pretrain_manifest(
        target_token_budget=target_tokens,
        stats=stats,
        tokenizer_path=tok_path,
        tokenizer_vocab_size=tokenizer.vocab_size(),
        subsets=subsets,
        seed=seed,
        output_path=manifest_path,
    )

    # Compute packed vs unpacked efficiency
    unpacked_waste_pct = max(0.0, (1.0 - (stats.raw_tokens_untruncated / (stats.raw_examples * 512))) * 100) if stats.raw_examples > 0 else 0.0

    print("\n" + "=" * 80)
    print("                     TOKEN ACCOUNTING & PACKING SUMMARY")
    print("=" * 80)
    print(f"  Raw Documents Streamed    : {stats.raw_examples:,}")
    print(f"  Cleaned Documents Used    : {stats.cleaned_examples:,}")
    print(f"  Raw Tokens (Untruncated)  : {stats.raw_tokens_untruncated:,}")
    print(f"  Packed Sequences (512-len): {stats.packed_sequences:,}")
    print(f"  Packed Training Tokens    : {stats.train_tokens:,} ({len(train_chunks):,} sequences)")
    print(f"  Packed Validation Tokens  : {stats.validation_tokens:,} ({len(val_chunks):,} sequences)")
    print(f"  Boundary Discarded Tokens : {stats.discarded_tokens:,} (incomplete final 512 block)")
    print(f"  Token Utilization Ratio   : {stats.token_utilization_ratio * 100:.2f}%")
    print(f"  Packed Compute Efficiency : 100.0% (Zero padding tokens in any sequence)")
    print(f"  Unpacked Padding Waste    : ~{unpacked_waste_pct:.1f}% padding overhead avoided by packing")
    print(f"  Processing Time           : {elapsed:.2f}s ({stats.packed_tokens / max(0.01, elapsed):,.0f} tok/s)")
    print(f"  Manifest Saved            : {manifest_path}")
    print("=" * 80 + "\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and pack general pretraining corpus.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--target_tokens", type=int, default=1_000_000, help="Target token budget.")
    parser.add_argument("--subsets", nargs="+", default=["cosmopedia-v2"], help="Corpus subsets.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    target_tokens = args.target_tokens
    subsets = tuple(args.subsets)
    val_ratio = args.val_ratio
    seed = args.seed

    if args.config:
        cfg_path = PROJECT_ROOT / args.config
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            target_tokens = cfg.get("target_token_budget", target_tokens)
            subsets = tuple(cfg.get("subsets", subsets))
            val_ratio = cfg.get("val_ratio", val_ratio)
            seed = cfg.get("seed", seed)

    build_pretrain_corpus(
        target_tokens=target_tokens,
        subsets=subsets,
        val_ratio=val_ratio,
        seed=seed,
    )


if __name__ == "__main__":
    main()
