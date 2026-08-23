# Mini-R1: Task 24 Part 1 — Final Model Integrity Verification

**TASK 24 PART 1 STATUS: PASSED (8/8 checks)**

**Date**: 2026-08-23
**Final candidate**: `artifacts/sft/task23_75_25/best.pt`

---

## Integrity checks

| # | Check | Result | Evidence |
| :-- | :--- | :---: | :--- |
| 1 | Checkpoint loads | **PASS** | `torch.load` OK; `load_state_dict(strict=True)` returns no missing/unexpected keys |
| 2 | Parameter count = 42,132,480 | **PASS** | `sum(p.numel())` = **42,132,480** |
| 3 | Tokenizer hash matches project tokenizer | **PASS** | on-disk `c1143f6e…7044` == hash recorded inside checkpoint |
| 4 | Checkpoint SHA256 recorded | **PASS** | `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c` |
| 5 | Config snapshot recorded | **PASS** | `model_config` embedded in checkpoint (below) |
| 6 | No NaN/Inf parameters | **PASS** | all 137 state-dict tensors finite |
| 7 | All required tensors present | **PASS** | embeddings, 12 blocks, `ln_f`, `lm_head` all present |
| 8 | Architecture matches config | **PASS** | 12 blocks found, `n_layers=12`; instantiation from config accepts the state dict strictly |

## Recorded hashes

| Artifact | SHA256 |
| :--- | :--- |
| Final checkpoint | `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c` |
| Tokenizer (`artifacts/tokenizer/tokenizer.model`) | `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044` |
| Init checkpoint (Task 20, recorded in-checkpoint) | `fdcecade71afafc1857d5ee56d221d31928ae1321d0f963dc0f566482af56d5c` |
| Replay corpus train | `4bd131c4e31ce89aa9423b8a879fd1afe463a50267aed386e4b7676c5aba7d14` |
| Progressive-subtraction train | `ee18cd7b2de5acde36c4234f45a704e67b0c6e76fb000b16c54435366d7314c2` |

File size 508,892,602 bytes (includes optimizer state).

## Config snapshot

```json
{"vocab_size": 8000, "d_model": 512, "n_layers": 12, "n_heads": 8,
 "d_ff": 2048, "max_seq_len": 512, "dropout": 0.0, "bias": false,
 "tie_word_embeddings": true}
```

`tie_word_embeddings: true` verified at runtime: `lm_head.weight is embeddings.tok_emb.weight` → `True`.

## Training provenance recorded in the checkpoint

- Label: `Model H (Task 23: 75/25 replay, all trainable)`
- Selected at step **650** of 650
- Per-source dev loss at selection: compact GSM8K **0.5657**, replay arithmetic **0.4725**, progressive subtraction **0.0407**, weighted mixture **0.4884**
- Parameter isolation record: 42,132,480 trainable / 0 frozen

**Verification command**: `python scripts/qualify_final_model.py --batch_size 64`
