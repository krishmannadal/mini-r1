"""
interactive_generate.py - Interactive Inference and Generation CLI for Mini-R1.

Usage Examples:
  # 1. Interactive REPL with Task 21B Compact Arithmetic model:
  python scripts/interactive_generate.py --model compact_arith

  # 2. Single-shot prompt evaluation:
  python scripts/interactive_generate.py --model compact_arith --prompt "Calculate: 154 - 38"

  # 3. Test on word problem with Direct SFT:
  python scripts/interactive_generate.py --model direct --prompt "Janet has 16 eggs and eats 4. How many left?"

  # 4. Test with temperature and repetition penalty:
  python scripts/interactive_generate.py --model cot --prompt "A robe takes 2 bolts of blue fiber..." --temperature 0.6 --repetition_penalty 1.2
"""

import argparse
import os
import re
import sys
import time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer

MODEL_ALIASES = {
    "compact_arith": "artifacts/sft/compact_transfer/arithmetic_init/best.pt",
    "compact_base": "artifacts/sft/compact_transfer/base_init/best.pt",
    "direct": "artifacts/sft/direct/best.pt",
    "cot": "artifacts/sft/cot/best.pt",
    "verbose_arith": "artifacts/sft/arithmetic_transfer/best.pt",
    "verbose_base": "artifacts/sft/base_transfer/best.pt",
    "progressive_arith": "artifacts/sft/progressive_subtraction/best.pt",
    "base": "checkpoints/mini_r1_500m_pretrain/best.pt",
}


def load_model(checkpoint_path: str, device: str) -> tuple[MiniR1Transformer, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    config = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def generate_response(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    # Ensure proper prompt prefix if not formatted
    clean_p = prompt.strip()
    if not clean_p.startswith("Question:"):
        formatted_prompt = f"Question: {clean_p}\n"
    else:
        formatted_prompt = f"{clean_p}\n"

    input_ids = tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_id

    generated = []
    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        if input_tensor.shape[1] > model.config.max_seq_len:
            break

        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(input_tensor)
            next_logit = logits[0, -1, :].clone()

        # Repetition penalty
        if repetition_penalty != 1.0 and len(generated) > 0:
            for prev_tok in set(generated):
                if next_logit[prev_tok] < 0:
                    next_logit[prev_tok] *= repetition_penalty
                else:
                    next_logit[prev_tok] /= repetition_penalty

        # Greedy vs Sampling
        if temperature > 0.0:
            next_logit = next_logit / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logit[indices_to_remove] = float("-inf")
            probs = torch.softmax(next_logit, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_token = int(torch.argmax(next_logit).item())

        if next_token == eos_id:
            break

        generated.append(next_token)
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

        # Early exit if answer closing tag completed
        current_text = tokenizer.decode(generated)
        if "</answer>" in current_text:
            break

    elapsed = time.perf_counter() - start_time
    decoded_completion = tokenizer.decode(generated)
    tok_rate = len(generated) / max(1e-4, elapsed)
    return decoded_completion, len(generated), tok_rate


def parse_and_display(raw_completion: str, num_tokens: int, tok_rate: float) -> None:
    print("\n" + "-" * 60)
    print(f" GENERATION OUTPUT ({num_tokens} tokens | {tok_rate:.1f} tok/s):")
    print("-" * 60)
    print(raw_completion)
    print("-" * 60)

    # Parse components
    think_match = re.search(r"<think>(.*?)</think>", raw_completion, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_completion, re.DOTALL)

    if think_match:
        print("\n[Extracted Reasoning <think>]:")
        print(think_match.group(1).strip())

    if answer_match:
        print(f"\n[Extracted Answer <answer>]: {answer_match.group(1).strip()}")
    else:
        # Fallback number extraction
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw_completion)
        if nums:
            print(f"\n[Note: No <answer> tag found. Fallback last number]: {nums[-1]}")
    print("-" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-R1 Interactive Inference CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="compact_arith",
        help=f"Model alias ({', '.join(MODEL_ALIASES.keys())}) or path to .pt checkpoint",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt to evaluate")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (1.0 = disabled)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = MODEL_ALIASES.get(args.model, args.model)
    tok_path = "artifacts/tokenizer/tokenizer.model"

    print("=" * 70)
    print(" MINI-R1 INTERACTIVE INFERENCE CONSOLE")
    print("=" * 70)
    print(f"Device:               {device}")
    print(f"Target Checkpoint:    {ckpt_path}")
    print(f"Sampling Parameters:  Temp={args.temperature}, Top-p={args.top_p}, RepPen={args.repetition_penalty}")
    print("=" * 70)

    tokenizer = Tokenizer(tok_path)
    model, ckpt_meta = load_model(ckpt_path, device)
    print(f"Successfully loaded model weights ({model.get_num_params():,} parameters).\n")

    if args.prompt:
        print(f"Input Prompt: {args.prompt}")
        comp, toks, rate = generate_response(
            model,
            tokenizer,
            args.prompt,
            device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        parse_and_display(comp, toks, rate)
        return

    # Interactive REPL mode
    print("Entering interactive mode. Type your math question or prompt below.")
    print("Type 'exit' or 'quit' to quit.\n")

    while True:
        try:
            user_input = input("Enter Question >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break

            comp, toks, rate = generate_response(
                model,
                tokenizer,
                user_input,
                device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            parse_and_display(comp, toks, rate)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
