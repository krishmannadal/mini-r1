"""
Mini-R1 -- Forward Pass Demonstration Script
============================================
Runs a complete forward pass through the Mini-R1 Transformer and prints
the exact tensor shapes at every single architectural stage.

Architecture:
    Vocab Size        : 8000
    Embedding Dim     : 128
    Layers            : 2
    Heads             : 4 (head_dim = 32)
    FFN Dim           : 512
    Max Sequence Len  : 32

Usage:
    python scripts/forward_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config, ModelForwardOutputs


def separator(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


def main() -> None:
    print("+===========================================================================+")
    print("|          Mini-R1 Transformer -- Forward Pass & Tensor Shapes Demo         |")
    print("+===========================================================================+")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model.eval()

    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)

    print(f"  Device              : {device}")
    print(f"  Vocabulary Size     : {config.vocab_size}")
    print(f"  Model Dim (d_model) : {config.d_model}")
    print(f"  Layers (n_layers)   : {config.n_layers}")
    print(f"  Heads (n_heads)     : {config.n_heads} (d_k = {config.d_model // config.n_heads})")
    print(f"  FFN Dim (d_ff)      : {config.d_ff}")
    print(f"  Max Sequence Length : {config.max_seq_len}")
    print(f"  Total Parameters    : {total_params:,d}")
    print(f"  Non-Emb Parameters  : {non_emb_params:,d}")

    # -------------------------------------------------------------------------
    # 1. Create Sample Input
    # -------------------------------------------------------------------------
    separator("1. Input Token Tensor")
    batch_size = 2
    seq_len = 8  # T <= 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print(f"  Input shape               : {list(input_ids.shape)}  -> (B={batch_size}, T={seq_len})")
    print(f"  Input IDs:\n{input_ids}")

    # -------------------------------------------------------------------------
    # 2. Execute Forward Pass Exposing Intermediates
    # -------------------------------------------------------------------------
    with torch.no_grad():
        outputs: ModelForwardOutputs = model(input_ids, return_intermediates=True)

    # -------------------------------------------------------------------------
    # 3. Print Every Tensor Shape in Sequence
    # -------------------------------------------------------------------------
    separator("2. Step-by-Step Tensor Shape Progression")

    # Embeddings
    print(f"  [Embeddings]")
    print(f"    Token & Pos Embedding shape: {list(outputs.embedding_out.shape)} -> (B={batch_size}, T={seq_len}, d_model={config.d_model})")

    # Layer 1 Block
    b1 = outputs.block_outputs[0]
    attn1 = b1.attn_intermediates
    print(f"\n  [Transformer Block 1]")
    print(f"    Q shape                    : {list(attn1.q.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, d_k=32)")
    print(f"    K shape                    : {list(attn1.k.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, d_k=32)")
    print(f"    V shape                    : {list(attn1.v.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, d_k=32)")
    print(f"    Attention matrix shape     : {list(attn1.attn_matrix.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, T={seq_len})")
    print(f"    Attention output shape     : {list(attn1.output.shape)} -> (B={batch_size}, T={seq_len}, d_model={config.d_model})")
    print(f"    Block 1 output shape       : {list(b1.output.shape)} -> (B={batch_size}, T={seq_len}, d_model={config.d_model})")

    # Layer 2 Block
    b2 = outputs.block_outputs[1]
    attn2 = b2.attn_intermediates
    print(f"\n  [Transformer Block 2]")
    print(f"    Q shape                    : {list(attn2.q.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, d_k=32)")
    print(f"    K shape                    : {list(attn2.k.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, d_k=32)")
    print(f"    V shape                    : {list(attn2.v.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, d_k=32)")
    print(f"    Attention matrix shape     : {list(attn2.attn_matrix.shape)} -> (B={batch_size}, n_heads={config.n_heads}, T={seq_len}, T={seq_len})")
    print(f"    Attention output shape     : {list(attn2.output.shape)} -> (B={batch_size}, T={seq_len}, d_model={config.d_model})")
    print(f"    Block 2 output shape       : {list(b2.output.shape)} -> (B={batch_size}, T={seq_len}, d_model={config.d_model})")

    # Final Output Logits
    separator("3. Final Model Output Logits")
    print(f"  Logits shape                 : {list(outputs.logits.shape)} -> (B={batch_size}, T={seq_len}, vocab_size={config.vocab_size})")

    # -------------------------------------------------------------------------
    # 4. Verify Causal Mask on Attention Matrix
    # -------------------------------------------------------------------------
    separator("4. Causal Mask Verification (Head 0, Sample 0)")
    attn_sample = attn1.attn_matrix[0, 0].cpu().numpy()
    print("  Attention weight matrix A[i, j] (rows=Query i, cols=Key j):")
    for i in range(seq_len):
        row_str = " ".join(f"{attn_sample[i, j]:6.3f}" for j in range(seq_len))
        print(f"    Row {i} (Query pos {i}): [{row_str}] -> Row Sum = {attn_sample[i].sum():.3f}")

    is_causal = all(attn_sample[i, j] == 0.0 for i in range(seq_len) for j in range(i + 1, seq_len))
    print(f"\n  Upper-triangle strictly zero (Causal Guarantee): {'[OK]' if is_causal else '[FAIL]'}")

    separator("Result")
    print("  [OK] Forward pass completed successfully across all layers and heads.")


if __name__ == "__main__":
    main()
