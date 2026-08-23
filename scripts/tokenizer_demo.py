"""
Mini-R1 -- Tokenizer Demonstration Script
=========================================
Demonstrates text-to-token encoding, special token preservation,
batch processing with padding, and round-trip decoding across:
1. Ordinary English text
2. Mathematical formulas and expressions
3. GSM8K-style arithmetic word problems
4. Multi-step reasoning sequences with <think> and <answer> markers

Usage:
    python scripts/tokenizer_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import Tokenizer


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def format_piece(p: str) -> str:
    """Safely format subword pieces for Windows CP1252 console printing."""
    return p.replace("\u2581", "_")


def demonstrate_sample(tok: Tokenizer, title: str, text: str) -> None:
    """Helper to encode, inspect pieces/IDs, and decode a text sample."""
    separator(title)
    ids = tok.encode(text, add_bos=True, add_eos=True)
    pieces = [format_piece(tok.id_to_token(i)) for i in ids]
    decoded = tok.decode(ids)

    print(f"  Original Text:\n---\n{text}\n---")
    print(f"\n  Token Count : {len(ids)}")
    print(f"  Token IDs   : {ids}")
    print(f"  Token Pieces: {pieces[:25]}{' ...' if len(pieces) > 25 else ''}")
    print(f"  Decoded Text:\n---\n{decoded}\n---")
    print(f"  Lossless Round-trip: {'[OK]' if decoded == text else '[FAIL]'}")


def main() -> None:
    model_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if not model_path.exists():
        print(f"[FAIL] Tokenizer model not found at {model_path}.")
        print("Run 'python scripts/train_tokenizer.py' first.")
        sys.exit(1)

    tok = Tokenizer(model_path)

    print("+====================================================================+")
    print("|             Mini-R1 -- Tokenizer Comprehensive Demo                |")
    print("+====================================================================+")
    print(f"  Loaded model from : {model_path}")
    print(f"  Vocabulary size   : {tok.vocab_size()}")

    # -------------------------------------------------------------------------
    # Special Tokens Inspection
    # -------------------------------------------------------------------------
    separator("Special Tokens & Control IDs")
    specials = [
        ("UNK", tok.unk_id, tok.id_to_token(tok.unk_id)),
        ("BOS", tok.bos_id, tok.id_to_token(tok.bos_id)),
        ("EOS", tok.eos_id, tok.id_to_token(tok.eos_id)),
        ("PAD", tok.pad_id, tok.id_to_token(tok.pad_id)),
        ("<think>", tok.think_start_id, tok.id_to_token(tok.think_start_id)),
        ("</think>", tok.think_end_id, tok.id_to_token(tok.think_end_id)),
        ("<answer>", tok.answer_start_id, tok.id_to_token(tok.answer_start_id)),
        ("</answer>", tok.answer_end_id, tok.id_to_token(tok.answer_end_id)),
    ]
    for name, tid, piece in specials:
        print(f"  {name:<12} -> ID: {tid:<5} -> Token Piece: '{piece}'")

    # 1. Ordinary English
    demonstrate_sample(
        tok,
        "1. Ordinary English Text (General Pretraining Distribution)",
        "In modern physics, the theory of relativity describes the relationship between space, time, and gravity.",
    )

    # 2. Mathematical Text
    demonstrate_sample(
        tok,
        "2. Mathematical Text & Equations",
        "The quadratic formula states that x = (-b +- sqrt(b^2 - 4*a*c)) / (2*a).",
    )

    # 3. GSM8K-style Word Problem
    demonstrate_sample(
        tok,
        "3. GSM8K-Style Question",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    )

    # 4. Multi-Step Reasoning Sequence with Reasoning Tokens
    demonstrate_sample(
        tok,
        "4. Full Reasoning Sequence with <think> and <answer> Delimiters",
        (
            "Question: Solve for x: 3*x + 15 = 45.\n"
            "<think>\n"
            "1. Subtract 15 from both sides: 3*x = 45 - 15 = 30.\n"
            "2. Divide both sides by 3: x = 30 / 3 = 10.\n"
            "</think>\n"
            "<answer>10</answer>"
        ),
    )

    # 5. Batch Encoding with Padding
    separator("5. Batch Processing & Uniform Tensor Conversion")
    batch_sentences = [
        "What is 25 + 47?",
        "Solve 3*x + 15 = 45.",
        "<answer>72</answer>",
    ]
    padded_batch = tok.encode_batch(batch_sentences, add_bos=True, add_eos=True, pad=True)
    for idx, (sent, enc) in enumerate(zip(batch_sentences, padded_batch)):
        print(f"  Item {idx} (len={len(enc)}): {enc}")

    try:
        import torch
        tensor_batch = torch.tensor(padded_batch, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n  PyTorch Tensor Batch:")
        print(f"    Shape : {tensor_batch.shape} (batch_size=3, seq_len={tensor_batch.shape[1]})")
        print(f"    Device: {tensor_batch.device}")
        print(f"    Dtype : {tensor_batch.dtype}")
        print(f"    Tensor:\n{tensor_batch}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
