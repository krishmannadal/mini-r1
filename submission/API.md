# Mini-R1 — Programmatic API

A minimal wrapper over the inference code in `INFERENCE.md`. There is no hosted
service and no network dependency; this is a local Python API.

## `MiniR1` wrapper

```python
# submission/mini_r1_api.py  (illustrative; copy into your project)
import re, torch, sys
sys.path.insert(0, ".")
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer


class MiniR1:
    """Local inference wrapper for the Mini-R1 42.1M final checkpoint."""

    def __init__(self,
                 checkpoint="artifacts/sft/task23_75_25/best.pt",
                 tokenizer="submission/tokenizer/tokenizer.model",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer(tokenizer)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = MiniR1Transformer(MiniR1Config.from_dict(ck["model_config"])).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.answer_close = self.tok.encode("</answer>", add_bos=False, add_eos=False)[0]

    # -- generation ------------------------------------------------------
    @torch.inference_mode()
    def generate(self, question, max_new_tokens=256, temperature=0.0,
                 top_p=1.0, repetition_penalty=1.0):
        ids = self.tok.encode(f"Question: {question.strip()}\n", add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        seen, out = set(ids), []
        for _ in range(max_new_tokens):
            logits = self.model(x[:, -self.model.config.max_seq_len:])[:, -1, :].float()
            if repetition_penalty != 1.0 and seen:
                idx = torch.tensor(sorted(seen), device=self.device)
                cur = logits[0, idx]
                logits[0, idx] = torch.where(cur > 0, cur / repetition_penalty,
                                             cur * repetition_penalty)
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
            if nxt == self.tok.eos_id:
                break
            out.append(nxt); seen.add(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=self.device)], dim=1)
            if nxt == self.answer_close:
                break
        return self.tok.decode(out)

    # -- structured result ------------------------------------------------
    def solve(self, question, **kw):
        """Returns {'answer', 'steps', 'all_steps_verified', 'raw'}."""
        raw = self.generate(question, **kw)
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
        ans = None
        if m:
            n = re.search(r"-?\d+(?:\.\d+)?", m.group(1).replace(",", "").replace("$", ""))
            ans = n.group(0) if n else None
        think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        steps = []
        if think:
            for line in think.group(1).split("\n"):
                line = line.strip()
                if line:
                    steps.append({"equation": line, "verified": self.verify(line)})
        return {"answer": ans,
                "steps": steps,
                "all_steps_verified": bool(steps) and all(s["verified"] is True for s in steps),
                "raw": raw}

    @staticmethod
    def verify(line):
        """True / False / None(not an equation) - deterministic Python check."""
        if line.count("=") != 1:
            return None
        lhs, rhs = line.split("=")
        if not re.fullmatch(r"[\d\.\s+\-*/()]+", lhs.strip()):
            return None
        try:
            return abs(eval(lhs.strip(), {"__builtins__": None}, {}) - float(rhs)) < 1e-4
        except Exception:
            return None
```

## Usage

```python
m = MiniR1()

m.generate("Calculate: 87 - 43")
# '<think>\n87 - 43 = 44\n</think>\n<answer>44</answer>'

r = m.solve("Calculate: 1500 - 1200")
r["answer"]              # '300'
r["all_steps_verified"]  # True
```

**Always gate on `all_steps_verified`.** Step-level equation validity on GSM8K is
31.64%, so most multi-step outputs contain at least one wrong equation:

```python
r = m.solve(problem)
if r["all_steps_verified"]:
    use(r["answer"])
else:
    reject(r)          # ~2 in 3 GSM8K generations land here
```

## Parameters

| Param | Default | Notes |
| :--- | :--- | :--- |
| `max_new_tokens` | 256 | Stops early on EOS or `</answer>` |
| `temperature` | 0.0 | 0.0 = greedy. Best arithmetic accuracy. |
| `top_p` | 1.0 | Use 0.95 with `temperature=0.6` |
| `repetition_penalty` | 1.0 | 1.2 cuts GSM8K repetition 77.3% → 16.4% |

## Batching

Not provided deliberately. The model uses learned absolute position embeddings,
so naive left-padded batching shifts position indices and degrades output. Batch
only with right-padding and position-correct inputs, or loop at batch size 1.

## Errors and edge cases

- No `<answer>` block → `solve()` returns `answer=None`. Common on prompt shapes outside training (multiple-choice: 0.0% full-format compliance).
- Prompts beyond 512 tokens are truncated to the last 512.
- `verify()` returns `None` for non-equation lines; do not treat `None` as `True`.
