# Mini-R1 — Submission Package

A **42.1M-parameter** decoder-only Transformer built from scratch in pure PyTorch on a single 6 GB RTX 4050, studying whether verified arithmetic supervision transfers to natural-language mathematical reasoning.

**Status**: research closed, model frozen 2026-08-23.

---

## The one-paragraph result

Verified arithmetic SFT gives a 42M model genuinely strong arithmetic **inside the numeric regimes it was trained on** (100% on held-out 2-digit subtraction). That competence **does not transfer** to GSM8K word problems — a controlled A/B measured a transfer delta of −0.30%. Worse, GSM8K fine-tuning **destroys** the arithmetic (100% → ~4%). The project's transferable finding is the fix: interleaving a **25% replay block containing the target arithmetic distribution** preserves it through fine-tuning, restoring 2-digit retention from 4.40% to **94.80%**. Along the way, a parameter-freezing intervention failed instructively — because the model has tied embeddings, freezing the token embedding also froze the output head, causing non-terminating generation; the isolating control recovered **+30.38 points** of equation validity.

## Final model

| | |
| :--- | :--- |
| Checkpoint | `artifacts/sft/task23_75_25/best.pt` |
| SHA256 | `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c` |
| Parameters | 42,132,480 |
| Tokenizer | `submission/tokenizer/tokenizer.model` (`c1143f6e…7044`) |

| Benchmark | Result |
| :--- | ---: |
| GSM8K test (1,319), greedy | **2.43%** |
| GSM8K step-level equation validity | **31.64%** |
| SVAMP (300) | 1.67% |
| AQuA (254) | **0.00%** |
| IID 2-digit subtraction (E1) | **89.60%** |
| IID 3-digit (E3) | **84.60%** |
| IID 4-digit (E5) | 38.40% |
| 5-digit acid test (E7, never trained) | **0.60%** |
| Retention probe 2D / 3D / 4D | **94.80% / 78.20% / 32.40%** |
| General-language PPL (10M held-out) | 50.74 |

**Read these honestly.** GSM8K 2.43% is statistically indistinguishable from plain direct-answer SFT (2.35%) — a one-problem difference. This model does not do word-problem reasoning. Its arithmetic is real but regime-local, and general language quality regressed 2.9× from the 17.68 base.

## Contents

```
submission/
├── README.md                 this file
├── INFERENCE.md              runnable inference code, decoding guidance
├── API.md                    programmatic wrapper + verification gating
├── HASHES.json               SHA256 for every referenced artifact
├── config/
│   ├── model_config.json     architecture
│   └── task23_75_25.yaml     exact training config of the final model
├── tokenizer/
│   ├── tokenizer.model       SentencePiece 8k BPE
│   └── tokenizer.vocab
└── reports/
    ├── Mini-R1_model_card.md            architecture, data, limits, failure modes
    ├── Mini-R1_final_research_summary.md full narrative, Tasks 1–24
    ├── task24_final_integrity.md        8/8 integrity checks
    ├── task24_final_benchmarks.md       final benchmarks + claim audit
    ├── task24_final_benchmarks.json     raw benchmark data
    ├── task23_replay_ratio.md           replay-ratio curve, model selection
    ├── task22b_replay_only.md           freezing isolated as failure cause
    ├── task22_replay_sft.md             replay + freezing (disconfirmed)
    ├── task21b_compact_reasoning_transfer.md  transfer null result
    └── task20_progressive_numeric_curriculum.md  arithmetic generalization
```

**The checkpoint binary is intentionally not duplicated here** (509 MB, and it contains optimizer state). It is referenced by path and hash; `INFERENCE.md` shows how to verify it and how to strip the optimizer state to ~169 MB.

## Quick start

```powershell
Get-FileHash "artifacts\sft\task23_75_25\best.pt" -Algorithm SHA256
# expect d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c
```

```python
import sys; sys.path.insert(0, ".")
# full example in submission/INFERENCE.md
```

Prompt format is exact — deviating degrades output sharply:

```
Question: Calculate: 87 - 43\n
```
→
```
<think>
87 - 43 = 44
</think>
<answer>44</answer>
```

## Verify the model's arithmetic

Every `<think>` line is a checkable equation, and **roughly two in three are wrong on GSM8K**. `API.md` provides `solve()` returning `all_steps_verified`; gate on it rather than trusting output.

## Reproducing

```powershell
python scripts\build_replay_arithmetic.py
python scripts\train_task22_replay_sft.py --config configs\task23_75_25.yaml
python scripts\qualify_final_model.py --batch_size 64
```

Training: 650 optimizer steps, effective batch 32, ~25 min on an RTX 4050.

## Scope and limitations

Research artifact only. Not for education, tutoring, finance, health, or any setting where a wrong number carries cost. It emits fluent, correctly formatted, confidently wrong arithmetic — a more hazardous failure mode than visible breakdown. No safety tuning, alignment, or content filtering was applied. Single seed (42); no variance estimates. See the model card for the full limitations and ethical sections.
