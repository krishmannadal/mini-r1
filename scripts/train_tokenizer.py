"""
Mini-R1 -- Tokenizer Training Script
====================================
Trains a SentencePiece BPE tokenizer on a given text corpus.

Features:
- Configurable vocabulary size from configs/base.yaml or CLI args.
- Registers special reasoning tokens: <think>, </think>, <answer>, </answer>.
- Deterministic special token IDs (unk=0, bos=1, eos=2, pad=3).
- Saves .model and .vocab artifacts to the designated tokenizer directory.

Usage:
    python scripts/train_tokenizer.py --input data/sample_corpus.txt --vocab_size 1000
    python scripts/train_tokenizer.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import Tokenizer, train_sentencepiece, DEFAULT_USER_DEFINED_SYMBOLS


def load_config(config_path: Path) -> dict:
    """Load project YAML configuration."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for Mini-R1.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to training text corpus. Defaults to tokenizer.training_corpus in config.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Output model prefix (without .model extension). Defaults to path from config.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Target vocabulary size. Defaults to value from config or 8000.",
    )

    args = parser.parse_args()

    # Load defaults from config if available
    config = load_config(PROJECT_ROOT / args.config)
    tok_cfg = config.get("tokenizer", {})

    input_path_str = args.input or tok_cfg.get("training_corpus", "data/clean/tokenizer_corpus.txt")
    input_file = Path(input_path_str)
    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_file

    if not input_file.exists():
        print(f"[FAIL] Training corpus not found: {input_file}")
        print("Run 'python scripts/build_sample.py' first to build the dataset sample corpus.")
        sys.exit(1)

    model_prefix = (
        args.output_prefix
        or tok_cfg.get("model_prefix", "artifacts/tokenizer/tokenizer")
    )
    prefix_path = Path(model_prefix)
    if not prefix_path.is_absolute():
        prefix_path = PROJECT_ROOT / prefix_path

    vocab_size = (
        args.vocab_size
        or tok_cfg.get("vocab_size", 8000)
    )

    user_symbols = (
        tok_cfg.get("special_tokens", {}).get("user_defined_symbols")
        or DEFAULT_USER_DEFINED_SYMBOLS
    )

    print("+============================================================+")
    print("|            Mini-R1 -- Training Tokenizer                   |")
    print("+============================================================+")
    print(f"  Input corpus   : {input_file}")
    print(f"  Output prefix  : {prefix_path}")
    print(f"  Target vocab   : {vocab_size}")
    print(f"  Special symbols: {user_symbols}")

    model_file = train_sentencepiece(
        input_file=input_file,
        model_prefix=prefix_path,
        vocab_size=vocab_size,
        model_type=tok_cfg.get("model_type", "bpe"),
        user_defined_symbols=user_symbols,
        unk_id=tok_cfg.get("special_tokens", {}).get("unk_id", 0),
        bos_id=tok_cfg.get("special_tokens", {}).get("bos_id", 1),
        eos_id=tok_cfg.get("special_tokens", {}).get("eos_id", 2),
        pad_id=tok_cfg.get("special_tokens", {}).get("pad_id", 3),
    )

    print(f"\n[OK] Model successfully trained and saved: {model_file}")

    # Load and verify with wrapper
    tokenizer = Tokenizer(model_file)
    actual_vocab_size = tokenizer.vocab_size()

    print("\n============================================================")
    print("  Tokenizer Summary")
    print("============================================================")
    print(f"  Actual Vocabulary Size : {actual_vocab_size}")
    print(f"  UNK ID (<unk>)         : {tokenizer.unk_id} -> '{tokenizer.id_to_token(tokenizer.unk_id)}'")
    print(f"  BOS ID (<s>)           : {tokenizer.bos_id} -> '{tokenizer.id_to_token(tokenizer.bos_id)}'")
    print(f"  EOS ID (</s>)          : {tokenizer.eos_id} -> '{tokenizer.id_to_token(tokenizer.eos_id)}'")
    print(f"  PAD ID (<pad>)         : {tokenizer.pad_id} -> '{tokenizer.id_to_token(tokenizer.pad_id)}'")
    print(f"  <think> ID             : {tokenizer.think_start_id} -> '{tokenizer.id_to_token(tokenizer.think_start_id)}'")
    print(f"  </think> ID            : {tokenizer.think_end_id} -> '{tokenizer.id_to_token(tokenizer.think_end_id)}'")
    print(f"  <answer> ID            : {tokenizer.answer_start_id} -> '{tokenizer.id_to_token(tokenizer.answer_start_id)}'")
    print(f"  </answer> ID           : {tokenizer.answer_end_id} -> '{tokenizer.id_to_token(tokenizer.answer_end_id)}'")
    print("============================================================")


if __name__ == "__main__":
    main()
