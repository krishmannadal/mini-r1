# Mini-R1 — Inference Instructions

## Requirements

PyTorch, SentencePiece, NumPy. No external services. ~1 GB VRAM (FP16 weights ≈ 84 MB); CPU works but is slower.

## Checkpoint

The checkpoint binary is **not duplicated** into `submission/`. It lives at:

```
artifacts/sft/task23_75_25/best.pt      # 508,892,602 bytes
SHA256 d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c
```

Verify before use (PowerShell):

```powershell
Get-FileHash "artifacts\sft\task23_75_25\best.pt" -Algorithm SHA256
Get-FileHash "submission\tokenizer\tokenizer.model" -Algorithm SHA256
```

The file contains optimizer state. For deployment, strip it (≈509 MB → ≈169 MB):

```python
import torch
ck = torch.load("artifacts/sft/task23_75_25/best.pt", map_location="cpu", weights_only=False)
torch.save({"model_state_dict": ck["model_state_dict"], "model_config": ck["model_config"]},
           "mini_r1_final_weights.pt")
```

## Prompt format

The model is trained on exactly this shape — deviating from it degrades output sharply:

```
Question: {problem text}\n
```

It completes with:

```
<think>
{equation}
{equation}
</think>
<answer>{number}</answer>
```

## Minimal runnable example

```python
import torch, sys
sys.path.insert(0, ".")                      # repo root
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer("submission/tokenizer/tokenizer.model")

ck = torch.load("artifacts/sft/task23_75_25/best.pt", map_location="cpu", weights_only=False)
model = MiniR1Transformer(MiniR1Config.from_dict(ck["model_config"])).to(device)
model.load_state_dict(ck["model_state_dict"])
model.eval()

ANSWER_CLOSE = tok.encode("</answer>", add_bos=False, add_eos=False)[0]   # single token, id 7

@torch.inference_mode()
def generate(question, max_new_tokens=256, temperature=0.0, top_p=1.0):
    ids = tok.encode(f"Question: {question.strip()}\n", add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new_tokens):
        ctx = x[:, -model.config.max_seq_len:]
        logits = model(ctx)[:, -1, :].float()
        if temperature <= 1e-6:
            nxt = int(torch.argmax(logits, dim=-1))
        else:
            logits = logits / temperature
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
                rm = cum > top_p
                rm[:, 1:] = rm[:, :-1].clone(); rm[:, 0] = False
                logits = logits.scatter(1, si, sl.masked_fill(rm, -float("inf")))
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        if nxt == tok.eos_id:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == ANSWER_CLOSE:                  # </answer> ends the completion
            break
    return tok.decode(out)

print(generate("Calculate: 87 - 43"))
print(generate("Natalia sold clips to 48 friends in April, and then she sold half as "
               "many clips in May. How many clips did Natalia sell altogether?"))
```

## Extracting the answer

```python
import re
def extract(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not m: return None
    n = re.search(r"-?\d+(?:\.\d+)?", m.group(1).replace(",", "").replace("$", ""))
    return n.group(0) if n else None
```

## Verifying the model's own arithmetic

Every `<think>` line is a checkable equation. Always verify rather than trusting output:

```python
def check(line):
    if line.count("=") != 1: return None
    lhs, rhs = line.split("=")
    if not re.fullmatch(r"[\d\.\s+\-*/()]+", lhs.strip()): return None
    try:    return abs(eval(lhs.strip(), {"__builtins__": None}, {}) - float(rhs)) < 1e-4
    except Exception: return None
```

Measured step-level validity on GSM8K is **31.64%** — roughly two in three emitted equations are wrong. Verification is not optional.

## Recommended decoding

| Use case | Setting | Rationale |
| :--- | :--- | :--- |
| Arithmetic (`Calculate: a - b`) | **greedy** (`temperature=0.0`) | 100% EOS, 0% repetition on E1–E6 |
| GSM8K-style word problems, best accuracy | greedy | 2.43% strict (vs 1.21% sampled) |
| GSM8K-style, healthiest generations | `T=0.6, top_p=0.95, rep_pen=1.2` | repetition 16.4% vs 77.3% greedy |

Batch inference: the model uses **learned absolute position embeddings**. If you batch prompts of differing lengths, **right-pad** and pass position-correct inputs, or run batch size 1. Left-padding without an attention mask shifts every position index and measurably degrades output (the project's own harness had this flaw; see `reports/task22_replay_sft.md` §7–8).

## Scope

Research artifact. Do not use for education, tutoring, finance, health, or any decision where a wrong number carries cost. See the model card's limitations and ethical sections.
