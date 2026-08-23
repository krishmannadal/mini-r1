"""
server.py - Lightweight local HTTP server and inference backend for Mini-R1.
Serves static frontend files and provides /api/models and /api/generate endpoints.
"""

import http.server
import json
import os
import re
import socketserver
import sys
import time
from urllib.parse import parse_qs, urlparse

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer

PORT = 8000
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK_PATH = "artifacts/tokenizer/tokenizer.model"

MODEL_REGISTRY = {
    "compact_arith": {
        "name": "Model A: Compact Arithmetic Transfer (Task 21B)",
        "description": "Progressive Subtraction SFT -> Compact Equation Reasoning SFT",
        "path": "artifacts/sft/compact_transfer/arithmetic_init/best.pt",
        "format": "compact_think",
    },
    "compact_base": {
        "name": "Model B: Compact Base Transfer (Task 21B)",
        "description": "Base Pretrained (500M) -> Compact Equation Reasoning SFT",
        "path": "artifacts/sft/compact_transfer/base_init/best.pt",
        "format": "compact_think",
    },
    "direct": {
        "name": "Model C: Direct GSM8K Baseline",
        "description": "Direct SFT targeting <answer>...</answer> without scratchpad",
        "path": "artifacts/sft/direct/best.pt",
        "format": "direct_answer",
    },
    "cot": {
        "name": "Model D: Verbose CoT Baseline",
        "description": "Original Verbose Natural-Language CoT SFT",
        "path": "artifacts/sft/cot/best.pt",
        "format": "verbose_cot",
    },
    "verbose_arith": {
        "name": "Model A (Task 21A): Verbose Arithmetic Transfer",
        "description": "Progressive Subtraction SFT -> Verbose CoT SFT",
        "path": "artifacts/sft/arithmetic_transfer/best.pt",
        "format": "verbose_cot",
    },
    "progressive_arith": {
        "name": "Task 20: Progressive Subtraction Model",
        "description": "Pure 2D, 3D, and 4D verified algorithmic subtraction",
        "path": "artifacts/sft/progressive_subtraction/best.pt",
        "format": "compact_think",
    },
    "base": {
        "name": "Base Pretrained (Step 18k)",
        "description": "Zero-shot base 42.1M Transformer pretrained on 500M tokens",
        "path": "checkpoints/mini_r1_500m_pretrain/best.pt",
        "format": "raw",
    },
}

# Global loaded cache
_TOKENIZER = None
_LOADED_MODELS = {}


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = Tokenizer(TOK_PATH)
    return _TOKENIZER


def get_model(model_key: str):
    global _LOADED_MODELS
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")

    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key]

    ckpt_path = MODEL_REGISTRY[model_key]["path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Backend] Loading {model_key} from {ckpt_path} onto {DEVICE}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    config = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(config).to(DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
    model.load_state_dict(state_dict)
    model.eval()

    _LOADED_MODELS[model_key] = model
    return model


@torch.inference_mode()
def run_model_inference(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
):
    clean_p = prompt.strip()
    if not clean_p.startswith("Question:"):
        formatted_prompt = f"Question: {clean_p}\n"
    else:
        formatted_prompt = f"{clean_p}\n"

    input_ids = tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    eos_id = tokenizer.eos_id

    generated = []
    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        if input_tensor.shape[1] > model.config.max_seq_len:
            break

        with torch.amp.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.float16):
            logits = model(input_tensor)
            next_logit = logits[0, -1, :].clone()

        if repetition_penalty != 1.0 and len(generated) > 0:
            for prev_tok in set(generated):
                if next_logit[prev_tok] < 0:
                    next_logit[prev_tok] *= repetition_penalty
                else:
                    next_logit[prev_tok] /= repetition_penalty

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
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)], dim=1)

        # Stop early if </answer> closing tag was generated
        curr_text = tokenizer.decode(generated)
        if "</answer>" in curr_text:
            break

    elapsed = time.perf_counter() - start_time
    decoded = tokenizer.decode(generated)
    tok_rate = len(generated) / max(1e-4, elapsed)

    # Extract <think> and <answer>
    think_match = re.search(r"<think>(.*?)</think>", decoded, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", decoded, re.DOTALL)

    extracted_think = think_match.group(1).strip() if think_match else ""
    extracted_answer = answer_match.group(1).strip() if answer_match else ""

    if not extracted_answer:
        nums = re.findall(r"-?\d+(?:\.\d+)?", decoded)
        fallback_num = nums[-1] if nums else None
    else:
        fallback_num = extracted_answer

    return {
        "raw_completion": decoded,
        "extracted_think": extracted_think,
        "extracted_answer": extracted_answer,
        "fallback_answer": fallback_num,
        "tokens_generated": len(generated),
        "prompt_tokens": len(input_ids),
        "elapsed_sec": round(elapsed, 4),
        "tok_per_sec": round(tok_rate, 2),
    }


class MiniR1Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            model_list = []
            for k, v in MODEL_REGISTRY.items():
                available = os.path.exists(v["path"])
                model_list.append({
                    "id": k,
                    "name": v["name"],
                    "description": v["description"],
                    "format": v["format"],
                    "available": available,
                })

            resp = {
                "device": DEVICE,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "models": model_list,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                data = json.loads(body.decode("utf-8"))
                model_key = data.get("model", "compact_arith")
                prompt = data.get("prompt", "")
                max_tokens = int(data.get("max_tokens", 256))
                temperature = float(data.get("temperature", 0.0))
                top_p = float(data.get("top_p", 1.0))
                repetition_penalty = float(data.get("repetition_penalty", 1.0))

                tokenizer = get_tokenizer()
                model = get_model(model_key)
                res = run_model_inference(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "result": res}).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_server():
    # Warm up default model and tokenizer on startup
    print("=" * 70)
    print(" STARTING MINI-R1 LOCAL PLAYGROUND SERVER")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"URL:    http://127.0.0.1:{PORT}")
    print("=" * 70)

    try:
        get_tokenizer()
        get_model("compact_arith")
        print("[Backend] Pre-warmed Tokenizer and Model A (Compact Arithmetic) into GPU memory.\n")
    except Exception as e:
        print(f"[Backend Warning] Warm-up error: {e}")

    with socketserver.TCPServer(("", PORT), MiniR1Handler) as httpd:
        print(f"Mini-R1 Playground running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    start_server()
