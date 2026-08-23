"""
Mini-R1 -- Offline Pretraining Shard Builder
============================================
Streams educational text from HuggingFaceTB/smollm-corpus, applies SentencePiece
tokenization (vocab=8000), enforces 60/40 token budget ratio between cosmopedia-v2
and fineweb-edu-dedup, packs contiguously into 512-token sequences with EOS delimiters,
and serializes raw uint16 binary shards with SHA256/MD5 manifests.

Usage:
    python scripts/build_pretrain_shards.py --target_train_tokens 500000000 --target_val_tokens 10000000 --shard_tokens 10000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Sequence
from datasets import load_dataset
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import Tokenizer
from src.general_dataset import clean_general_text, compute_file_md5


def compute_buffer_hashes(data_bytes: bytes) -> tuple[str, str]:
    """Compute SHA256 and MD5 for binary shard bytes."""
    sha = hashlib.sha256(data_bytes).hexdigest()
    md = hashlib.md5(data_bytes).hexdigest()
    return sha, md


def build_pretrain_shards(
    output_dir: Path | str,
    tokenizer_path: Path | str,
    target_train_tokens: int = 500_000_000,
    target_val_tokens: int = 10_000_000,
    tokens_per_shard: int = 10_000_000,
    seq_len: int = 512,
    source_ratios: dict[str, float] | None = None,
    seed: int = 42,
    buffer_size: int = 10000,
    min_doc_chars: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_path = Path(output_dir).resolve()
    shards_dir = output_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    tok_file = Path(tokenizer_path).resolve()
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {tok_file}")

    tokenizer = Tokenizer(tok_file)
    eos_id = tokenizer.eos_id
    ratios = source_ratios or {"cosmopedia-v2": 0.60, "fineweb-edu-dedup": 0.40}

    # Verify disk space
    total_tokens_needed = target_train_tokens + target_val_tokens
    bytes_needed = total_tokens_needed * 2  # uint16 = 2 bytes/token
    total_gb_needed = bytes_needed / (1024 ** 3)

    total_disk, used_disk, free_disk = shutil.disk_usage(output_path)
    free_gb = free_disk / (1024 ** 3)

    print("\n" + "=" * 85)
    print("           Mini-R1: Offline Binary Pretraining Shard Builder")
    print("=" * 85)
    print(f"  Target Train Tokens  : {target_train_tokens:,} tokens ({target_train_tokens // seq_len:,} sequences)")
    print(f"  Target Val Tokens    : {target_val_tokens:,} tokens ({target_val_tokens // seq_len:,} sequences)")
    print(f"  Tokens per Shard     : {tokens_per_shard:,} tokens ({tokens_per_shard // seq_len:,} sequences)")
    print(f"  Sequence Length      : {seq_len} tokens")
    print(f"  Source Ratios        : {ratios}")
    print(f"  Tokenizer Vocab Size : {tokenizer.vocab_size()} (MD5: {compute_file_md5(tok_file)})")
    print(f"  Storage Format       : Raw uint16 binary (2 bytes/token)")
    print(f"  Estimated Disk Usage : {total_gb_needed:.2f} GB")
    print(f"  Available Free Disk  : {free_gb:.2f} GB")
    print(f"  Destination Path     : {shards_dir}\n")

    if free_disk < bytes_needed * 1.5:
        raise RuntimeError(
            f"Insufficient disk space! Needed: {total_gb_needed:.2f} GB, Available: {free_gb:.2f} GB"
        )

    # Shard Writers Helper
    train_shards_meta: list[dict[str, Any]] = []
    val_shards_meta: list[dict[str, Any]] = []

    seqs_per_shard = tokens_per_shard // seq_len

    # Global streaming loop across subsets
    split_rng = random.Random(seed)
    seen_hashes: set[str] = set()

    for split_name, total_target_split, meta_list, prefix in [
        ("val", target_val_tokens, val_shards_meta, "val"),
        ("train", target_train_tokens, train_shards_meta, "train"),
    ]:
        if total_target_split <= 0:
            continue

        print(f"--- Building {split_name.upper()} Shards (Target: {total_target_split:,} tokens) ---")
        current_split_tokens = 0
        current_shard_idx = 0
        current_shard_sequences: list[list[int]] = []
        token_buffer: list[int] = []

        total_weight = sum(ratios.values())
        norm_ratios = {k: v / total_weight for k, v in ratios.items()}

        for subset, weight in norm_ratios.items():
            subset_target = int(total_target_split * weight)
            subset_tokens_collected = 0

            print(f"  Streaming '{subset}' (Target: {subset_target:,} tokens)...")
            ds = None
            for attempt in range(1, 6):
                try:
                    ds = load_dataset("HuggingFaceTB/smollm-corpus", subset, split="train", streaming=True)
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    print(f"    [WARN] Network error on attempt {attempt}/5 ({e}). Retrying in {2**attempt}s...")
                    time.sleep(2**attempt)

            shuffled_ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

            t0 = time.time()
            doc_count = 0

            for item in shuffled_ds:
                if "text" not in item:
                    continue
                raw_text = item.get("text", "")
                if not raw_text or not raw_text.strip():
                    continue

                cleaned = clean_general_text(raw_text)
                if len(cleaned) < min_doc_chars:
                    continue

                # Normalization MD5 deduplication
                norm_doc = " ".join(cleaned.lower().split())
                doc_hash = hashlib.md5(norm_doc.encode("utf-8")).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)

                # Tokenize
                doc_tokens = tokenizer.encode(cleaned, add_bos=False, add_eos=False)
                doc_tokens.append(eos_id)
                token_buffer.extend(doc_tokens)
                doc_count += 1

                # Pack into sequences of length 512
                while len(token_buffer) >= seq_len and subset_tokens_collected < subset_target:
                    seq = token_buffer[:seq_len]
                    token_buffer = token_buffer[seq_len:]
                    current_shard_sequences.append(seq)
                    subset_tokens_collected += seq_len
                    current_split_tokens += seq_len

                    # Check if shard is full
                    if len(current_shard_sequences) >= seqs_per_shard:
                        shard_filename = f"{prefix}_shard_{current_shard_idx:05d}.bin"
                        shard_filepath = shards_dir / shard_filename

                        # Flatten and write uint16
                        shard_array = np.array(current_shard_sequences, dtype=np.uint16)
                        shard_bytes = shard_array.tobytes()
                        with open(shard_filepath, "wb") as f:
                            f.write(shard_bytes)

                        sha256, md5 = compute_buffer_hashes(shard_bytes)
                        shard_tokens = len(current_shard_sequences) * seq_len
                        meta_list.append({
                            "shard_id": current_shard_idx,
                            "filename": shard_filename,
                            "num_sequences": len(current_shard_sequences),
                            "num_tokens": shard_tokens,
                            "file_size_bytes": len(shard_bytes),
                            "sha256": sha256,
                            "md5": md5,
                        })
                        print(f"    [SHARD EMITTED] {shard_filename}: {len(current_shard_sequences):,} seqs ({shard_tokens:,} tokens, {len(shard_bytes)/(1024**2):.1f} MB)")
                        current_shard_idx += 1
                        current_shard_sequences = []

                if doc_count % 1000 == 0 and subset_tokens_collected > 0:
                    sub_elapsed = max(1e-6, time.time() - t0)
                    pct = (subset_tokens_collected / subset_target) * 100.0
                    print(f"    [{subset}] {doc_count:,} docs | {subset_tokens_collected/1e6:.2f}M / {subset_target/1e6:.1f}M tokens ({pct:.1f}%) | {subset_tokens_collected/sub_elapsed:,.0f} tok/s")

                if subset_tokens_collected >= subset_target:
                    break

            elapsed = time.time() - t0
            print(f"  Finished '{subset}': {doc_count:,} docs tokenized -> {subset_tokens_collected:,} tokens in {elapsed:.1f}s")

        # Flush any remaining sequences to a final shard
        if current_shard_sequences:
            shard_filename = f"{prefix}_shard_{current_shard_idx:05d}.bin"
            shard_filepath = shards_dir / shard_filename

            shard_array = np.array(current_shard_sequences, dtype=np.uint16)
            shard_bytes = shard_array.tobytes()
            with open(shard_filepath, "wb") as f:
                f.write(shard_bytes)

            sha256, md5 = compute_buffer_hashes(shard_bytes)
            shard_tokens = len(current_shard_sequences) * seq_len
            meta_list.append({
                "shard_id": current_shard_idx,
                "filename": shard_filename,
                "num_sequences": len(current_shard_sequences),
                "num_tokens": shard_tokens,
                "file_size_bytes": len(shard_bytes),
                "sha256": sha256,
                "md5": md5,
            })
            print(f"    [FINAL SHARD EMITTED] {shard_filename}: {len(current_shard_sequences):,} seqs ({shard_tokens:,} tokens, {len(shard_bytes)/(1024**2):.1f} MB)")

    # Build Global Manifests
    def write_manifest(manifest_path: Path, split_name: str, meta_list: list[dict[str, Any]]) -> dict[str, Any]:
        total_seqs = sum(s["num_sequences"] for s in meta_list)
        total_tokens = total_seqs * seq_len
        manifest_data = {
            "manifest_name": f"Mini-R1 {split_name.capitalize()} Shard Manifest",
            "split": split_name,
            "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "tokenizer_path": str(tok_file),
            "tokenizer_md5": compute_file_md5(tok_file),
            "vocab_size": tokenizer.vocab_size(),
            "source_dataset": "HuggingFaceTB/smollm-corpus",
            "source_mixture": ratios,
            "seed": seed,
            "sequence_length": seq_len,
            "total_shards": len(meta_list),
            "total_sequences": total_seqs,
            "total_tokens": total_tokens,
            "format": "uint16_binary_memmap",
            "shards": meta_list,
        }
        manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
        manifest_data["manifest_hash"] = hashlib.sha256(manifest_bytes).hexdigest()

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_data

    train_manifest_path = output_path / "train_manifest.json"
    val_manifest_path = output_path / "val_manifest.json"

    train_manifest = write_manifest(train_manifest_path, "train", train_shards_meta)
    val_manifest = write_manifest(val_manifest_path, "val", val_shards_meta)

    print("\n" + "=" * 85)
    print("                 SHARD GENERATION COMPLETE")
    print("=" * 85)
    print(f"  Train Manifest Saved : {train_manifest_path} ({train_manifest['total_tokens']:,} tokens in {train_manifest['total_shards']} shards)")
    print(f"  Val Manifest Saved   : {val_manifest_path} ({val_manifest['total_tokens']:,} tokens in {val_manifest['total_shards']} shards)")
    print("=" * 85 + "\n")

    return train_manifest, val_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mini-R1 Binary Pretraining Shards.")
    parser.add_argument("--output_dir", type=str, default="data/pretrain", help="Directory for shards and manifests.")
    parser.add_argument("--tokenizer_path", type=str, default="artifacts/tokenizer/tokenizer.model", help="Path to tokenizer.model.")
    parser.add_argument("--target_train_tokens", type=int, default=500_000_000, help="Target training tokens.")
    parser.add_argument("--target_val_tokens", type=int, default=10_000_000, help="Target validation tokens.")
    parser.add_argument("--shard_tokens", type=int, default=10_000_000, help="Tokens per shard file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    build_pretrain_shards(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        target_train_tokens=args.target_train_tokens,
        target_val_tokens=args.target_val_tokens,
        tokens_per_shard=args.shard_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
