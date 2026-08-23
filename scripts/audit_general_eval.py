"""
Mini-R1: Task 16A SFT Evaluation Audit Script

Evaluates the Base, Direct SFT, and CoT SFT checkpoints on the exact 10M general-language
validation shards to rigorously resolve the general language perplexity discrepancy.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_general_language
from src.model import MiniR1Config, MiniR1Transformer

BASE_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "mini_r1_500m_pretrain" / "best.pt"
DIRECT_CKPT_PATH = PROJECT_ROOT / "artifacts" / "sft" / "direct" / "best.pt"
COT_CKPT_PATH = PROJECT_ROOT / "artifacts" / "sft" / "cot" / "best.pt"
VAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "pretrain" / "val_manifest.json"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoint(name: str, ckpt_path: Path, device: str = "cuda"):
    print(f"\n{'='*70}")
    print(f" Auditing: {name}")
    print(f" Path: {ckpt_path}")
    
    sha256 = compute_sha256(ckpt_path)
    print(f" SHA256: {sha256}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})
    cfg = MiniR1Config.from_dict(config_dict)
    
    model = MiniR1Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    res = evaluate_general_language(
        model=model,
        val_manifest_path=VAL_MANIFEST_PATH,
        batch_size=32,
        device=device,
        use_amp=True,
        checkpoint_recorded_loss=ckpt.get("val_loss", None),
    )
    
    print(f" Raw Loss:        {res.avg_loss:.6f}")
    print(f" Perplexity:      {res.perplexity:.6f}")
    print(f" Total Sequences: {res.total_sequences:,}")
    print(f" Total Tokens:    {res.total_tokens:,}")
    print(f" Throughput:      {res.tokens_per_sec:,.1f} tok/s")
    print(f" Elapsed:         {res.elapsed_sec:.2f} s")
    if res.checkpoint_recorded_loss is not None:
        print(f" Checkpoint Val Loss: {res.checkpoint_recorded_loss:.6f}")
        print(f" Discrepancy with Ckpt: {res.loss_discrepancy:.6e}")
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "name": name,
        "path": str(ckpt_path),
        "sha256": sha256,
        "avg_loss": res.avg_loss,
        "perplexity": res.perplexity,
        "total_sequences": res.total_sequences,
        "total_tokens": res.total_tokens,
        "elapsed_sec": res.elapsed_sec,
        "tokens_per_sec": res.tokens_per_sec,
        "checkpoint_val_loss": ckpt.get("val_loss", None),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Task 16A Audit on device: {device}")
    print(f"Validation Manifest: {VAL_MANIFEST_PATH}")
    val_manifest_sha = compute_sha256(VAL_MANIFEST_PATH)
    print(f"Val Manifest SHA256: {val_manifest_sha}")

    # 1. Base Model
    base_res = audit_checkpoint("Base Model (Pretrained step 18k)", BASE_CKPT_PATH, device=device)

    # 2. Direct SFT
    direct_res = audit_checkpoint("Direct-Answer SFT (best.pt)", DIRECT_CKPT_PATH, device=device)

    # 3. CoT SFT
    cot_res = audit_checkpoint("Chain-of-Thought SFT (best.pt)", COT_CKPT_PATH, device=device)

    # Compute comparative shifts
    base_ppl = base_res["perplexity"]
    base_loss = base_res["avg_loss"]

    direct_ppl_shift = ((direct_res["perplexity"] - base_ppl) / base_ppl) * 100.0
    direct_loss_delta = direct_res["avg_loss"] - base_loss

    cot_ppl_shift = ((cot_res["perplexity"] - base_ppl) / base_ppl) * 100.0
    cot_loss_delta = cot_res["avg_loss"] - base_loss

    print("\n" + "=" * 70)
    print(" SUMMARY AUDIT TABLE")
    print("=" * 70)
    print(f"{'Condition':<25} | {'Raw Loss':<10} | {'Perplexity':<10} | {'Loss Delta':<10} | {'PPL Shift (%)':<12}")
    print("-" * 75)
    print(f"{'Base Model (step 18k)':<25} | {base_loss:<10.6f} | {base_ppl:<10.4f} | {'Baseline':<10} | {'0.00%':<12}")
    print(f"{'Direct-Answer SFT':<25} | {direct_res['avg_loss']:<10.6f} | {direct_res['perplexity']:<10.4f} | {direct_loss_delta:<+10.6f} | {direct_ppl_shift:<+10.2f}%")
    print(f"{'Chain-of-Thought SFT':<25} | {cot_res['avg_loss']:<10.6f} | {cot_res['perplexity']:<10.4f} | {cot_loss_delta:<+10.6f} | {cot_ppl_shift:<+10.2f}%")
    print("-" * 75)

    # Save JSON summary
    summary_out = PROJECT_ROOT / "reports" / "task16a_evaluation_audit.json"
    audit_data = {
        "val_manifest_path": str(VAL_MANIFEST_PATH),
        "val_manifest_sha256": val_manifest_sha,
        "base": base_res,
        "direct_sft": direct_res,
        "cot_sft": cot_res,
        "direct_ppl_shift_pct": direct_ppl_shift,
        "cot_ppl_shift_pct": cot_ppl_shift,
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit JSON saved to: {summary_out}")


if __name__ == "__main__":
    main()
