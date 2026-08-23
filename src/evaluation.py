"""
Mini-R1 -- Evaluation Module for Base & Reasoning Models
=========================================================
Implements:
1. Teacher-forcing general language evaluation over sharded uint16 binary validation sets.
2. Autoregressive text generation with greedy decoding and nucleus (top-p) temperature sampling.
3. Deterministic answer extraction for GSM8K, SVAMP (numerical), and AQuA (multiple-choice).
4. Delimiter and reasoning format validation (<think>...</think>, <answer>...</answer>).
5. Comprehensive metrics computation and qualitative categorization.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import MiniR1Transformer, MiniR1Config
from src.tokenizer import Tokenizer
from src.pretrain_dataset import ShardedPretrainDataset
from src.loss import compute_causal_lm_loss


# =============================================================================
# 1. General Language Validation (Teacher-Forcing)
# =============================================================================

@dataclass
class GeneralLMEvalResult:
    """Results from teacher-forcing evaluation on general language validation data."""
    avg_loss: float
    perplexity: float
    total_sequences: int
    total_tokens: int
    tokens_per_sec: float
    elapsed_sec: float
    checkpoint_recorded_loss: float | None = None
    loss_discrepancy: float | None = None


def evaluate_general_language(
    model: MiniR1Transformer,
    val_manifest_path: str | Path,
    batch_size: int = 32,
    device: str = "cuda",
    use_amp: bool = True,
    checkpoint_recorded_loss: float | None = None,
) -> GeneralLMEvalResult:
    """
    Evaluate the frozen base model on the 10M-token pretraining validation set using teacher forcing.

    Args:
        model: Frozen MiniR1Transformer instance.
        val_manifest_path: Path to val_manifest.json.
        batch_size: Micro-batch size for validation forward passes.
        device: 'cuda' or 'cpu'.
        use_amp: Whether to use FP16 AMP for validation passes.
        checkpoint_recorded_loss: Validation loss stored inside the checkpoint for parity check.

    Returns:
        GeneralLMEvalResult containing average cross-entropy loss, perplexity, and throughput.
    """
    model.eval()
    val_dataset = ShardedPretrainDataset(manifest_path=val_manifest_path, seq_len=model.config.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    total_loss_sum = 0.0
    total_tokens_count = 0
    total_sequences_count = 0

    amp_dtype = torch.float16 if (device == "cuda" and use_amp) else torch.float32

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in val_loader:
            # batch is (B, T) LongTensor
            batch = batch.to(device, non_blocking=True)
            b_size, seq_len = batch.shape

            # Forward pass under AMP
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=amp_dtype, enabled=use_amp):
                logits = model(batch)  # (B, T, V)
                loss = compute_causal_lm_loss(logits, batch, ignore_index=-100)

            # Weight loss by active target tokens: (seq_len - 1) per sequence
            active_tokens = b_size * (seq_len - 1)
            total_loss_sum += loss.item() * active_tokens
            total_tokens_count += active_tokens
            total_sequences_count += b_size

    elapsed = max(1e-5, time.perf_counter() - start_time)
    avg_loss = total_loss_sum / max(1, total_tokens_count)
    ppl = math.exp(min(50.0, avg_loss))
    tokens_per_sec = total_tokens_count / elapsed

    discrepancy = None
    if checkpoint_recorded_loss is not None:
        discrepancy = abs(avg_loss - checkpoint_recorded_loss)

    return GeneralLMEvalResult(
        avg_loss=avg_loss,
        perplexity=ppl,
        total_sequences=total_sequences_count,
        total_tokens=total_tokens_count,
        tokens_per_sec=tokens_per_sec,
        elapsed_sec=elapsed,
        checkpoint_recorded_loss=checkpoint_recorded_loss,
        loss_discrepancy=discrepancy,
    )


# =============================================================================
# 2. Autoregressive Generation
# =============================================================================

@dataclass
class GenerationOutput:
    """Output container for an autoregressive generation call."""
    generated_text: str
    generated_tokens: list[int]
    num_generated_tokens: int
    prompt_tokens: list[int]
    num_prompt_tokens: int
    latency_sec: float
    stopped_by_eos: bool


def generate_autoregressive(
    model: MiniR1Transformer,
    prompt_token_ids: Sequence[int],
    tokenizer: Tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda",
    stop_token_ids: Sequence[int] | None = None,
) -> GenerationOutput:
    """
    Generate tokens autoregressively from the model given a prompt.

    Args:
        model: Frozen MiniR1Transformer in eval mode.
        prompt_token_ids: 1D sequence of prompt token IDs.
        tokenizer: Tokenizer instance for decoding and special token lookups.
        max_new_tokens: Maximum number of newly generated tokens.
        temperature: Sampling temperature (0.0 for greedy argmax).
        top_p: Nucleus sampling threshold (1.0 for full distribution).
        device: 'cuda' or 'cpu'.
        stop_token_ids: Token IDs that signal immediate generation termination (e.g. EOS, </answer>).

    Returns:
        GenerationOutput object.
    """
    model.eval()
    eos_id = tokenizer.eos_id
    if stop_token_ids is None:
        stop_token_ids = [eos_id]
    else:
        stop_token_ids = list(stop_token_ids)
        if eos_id not in stop_token_ids:
            stop_token_ids.append(eos_id)

    max_seq_len = model.config.max_seq_len
    input_ids = list(prompt_token_ids)
    num_prompt_tokens = len(input_ids)
    generated_token_ids: list[int] = []
    stopped_by_eos = False

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(input_ids) >= max_seq_len:
                # Truncate context from the left to respect maximum positional embedding capacity
                context = input_ids[-max_seq_len:]
            else:
                context = input_ids

            curr_tensor = torch.tensor([context], dtype=torch.long, device=device)  # (1, T)
            logits = model(curr_tensor)  # (1, T, V)
            next_token_logits = logits[0, -1, :]  # (V,)

            if temperature <= 1e-6:
                # Deterministic greedy decoding
                next_token = int(torch.argmax(next_token_logits, dim=-1).item())
            else:
                # Scaled temperature sampling
                scaled_logits = next_token_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to right to keep at least first token above threshold
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            generated_token_ids.append(next_token)
            input_ids.append(next_token)

            if next_token in stop_token_ids:
                stopped_by_eos = True
                break

    latency = time.perf_counter() - start_time
    generated_text = tokenizer.decode(generated_token_ids)

    return GenerationOutput(
        generated_text=generated_text,
        generated_tokens=generated_token_ids,
        num_generated_tokens=len(generated_token_ids),
        prompt_tokens=list(prompt_token_ids),
        num_prompt_tokens=num_prompt_tokens,
        latency_sec=latency,
        stopped_by_eos=stopped_by_eos,
    )


# =============================================================================
# 3. Deterministic Answer Extractors & Format Validators
# =============================================================================

class AnswerExtractor:
    """
    Deterministic rule-based answer extractors for reasoning benchmarks.
    Zero LLM dependencies; fully reproducible.
    """

    @staticmethod
    def clean_numerical_string(val_str: str) -> str:
        """Strip currency symbols, commas, percent signs, and whitespace."""
        cleaned = val_str.replace(",", "").replace("$", "").replace("%", "").strip()
        # Handle trailing period if present
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]
        # Normalize -0 or float representations like '42.0' to '42'
        try:
            val_float = float(cleaned)
            if val_float.is_integer():
                return str(int(val_float))
            return str(round(val_float, 6))
        except ValueError:
            return cleaned

    @classmethod
    def extract_gsm8k_answer(cls, text: str) -> tuple[str | None, str]:
        """
        Extract numerical answer from GSM8K generation.
        Returns: (extracted_answer, extraction_method)
        """
        # 1. Standard Mini-R1 markup <answer>...</answer>
        m_tag = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if m_tag:
            tag_content = m_tag.group(1).strip()
            # Extract number from inside tag
            nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", tag_content)
            if nums:
                return cls.clean_numerical_string(nums[-1]), "xml_answer_tag"
            if tag_content:
                return cls.clean_numerical_string(tag_content), "xml_answer_tag_raw"

        # 2. GSM8K native delimiter '####'
        m_hash = re.search(r"####\s*([-\$]?\s*[\d,\.]+)", text)
        if m_hash:
            return cls.clean_numerical_string(m_hash.group(1)), "gsm8k_hash_delimiter"

        # 3. Explicit phrasing 'the answer is ...'
        m_phrase = re.search(
            r"(?:the|final)\s+(?:correct\s+)?answer\s+(?:is|=)\s*[:\s]*([-\$]?\s*[\d,\.]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m_phrase:
            return cls.clean_numerical_string(m_phrase.group(1)), "answer_phrase"

        # 4. Fallback: last numerical token in the generated response
        nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if nums:
            return cls.clean_numerical_string(nums[-1]), "last_number_fallback"

        return None, "extraction_failed"

    @classmethod
    def extract_svamp_answer(cls, text: str) -> tuple[str | None, str]:
        """
        Extract numerical answer from SVAMP generation.
        Returns: (extracted_answer, extraction_method)
        """
        # SVAMP uses identical numerical answer semantics to GSM8K
        return cls.extract_gsm8k_answer(text)

    @classmethod
    def extract_aqua_answer(cls, text: str) -> tuple[str | None, str]:
        """
        Extract multiple-choice letter (A-E) from AQuA-RAT generation.
        Returns: (extracted_option_letter, extraction_method)
        """
        # 1. Markup <answer>A</answer>
        m_tag = re.search(r"<answer>\s*([A-Ea-e])\s*</answer>", text, flags=re.IGNORECASE)
        if m_tag:
            return m_tag.group(1).upper(), "xml_answer_tag"

        # 2. Phrasing 'the answer is (option) [A-E]'
        m_phrase = re.search(
            r"(?:the|correct)\s+answer\s+is\s*[:\s]*(?:option\s+)?([A-Ea-e])\b",
            text,
            flags=re.IGNORECASE,
        )
        if m_phrase:
            return m_phrase.group(1).upper(), "answer_phrase"

        # 3. Phrasing 'Option [A-E]' at line boundary
        m_opt = re.search(r"\b(?:option|choice)\s+([A-Ea-e])\b", text, flags=re.IGNORECASE)
        if m_opt:
            return m_opt.group(1).upper(), "option_keyword"

        # 4. Fallback: last standalone capital letter A-E
        letters = re.findall(r"\b([A-E])\b", text)
        if letters:
            return letters[-1].upper(), "last_capital_letter_fallback"

        return None, "extraction_failed"

    @classmethod
    def compare_answers(cls, predicted: str | None, gold: str, dataset_type: str = "numerical") -> bool:
        """
        Compare extracted prediction against gold target string.
        """
        if predicted is None:
            return False

        gold_clean = str(gold).strip()

        if dataset_type == "multiple_choice":
            return predicted.upper() == gold_clean.upper()

        # Numerical comparison
        pred_norm = cls.clean_numerical_string(predicted)
        gold_norm = cls.clean_numerical_string(gold_clean)

        if pred_norm == gold_norm:
            return True

        # Try floating point equivalence within epsilon (1e-5)
        try:
            p_val = float(pred_norm)
            g_val = float(gold_norm)
            return abs(p_val - g_val) < 1e-5
        except ValueError:
            return False


class FormatValidator:
    """
    Validates structural formatting tags (<think>, </think>, <answer>, </answer>).
    """

    @staticmethod
    def validate_formatting(text: str) -> dict[str, Any]:
        """
        Analyze formatting compliance for Mini-R1 reasoning format.
        """
        has_think_open = "<think>" in text
        has_think_close = "</think>" in text
        has_answer_open = "<answer>" in text
        has_answer_close = "</answer>" in text

        is_complete_think = has_think_open and has_think_close
        is_complete_answer = has_answer_open and has_answer_close

        # Check tag ordering
        ordering_valid = False
        if is_complete_think and is_complete_answer:
            idx_to = text.find("<think>")
            idx_tc = text.find("</think>")
            idx_ao = text.find("<answer>")
            idx_ac = text.find("</answer>")
            if idx_to < idx_tc < idx_ao < idx_ac:
                ordering_valid = True

        is_fully_formatted = is_complete_think and is_complete_answer and ordering_valid

        # Classify malformed category
        malformed_reason = "none"
        if not is_fully_formatted:
            if not has_think_open and not has_think_close and not has_answer_open and not has_answer_close:
                malformed_reason = "no_tags_present"
            elif has_think_open and not has_think_close:
                malformed_reason = "unclosed_think_tag"
            elif has_answer_open and not has_answer_close:
                malformed_reason = "unclosed_answer_tag"
            elif not ordering_valid:
                malformed_reason = "invalid_tag_order"
            else:
                malformed_reason = "missing_partial_tags"

        # Extract reasoning component
        reasoning_text = ""
        if is_complete_think:
            m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
            if m:
                reasoning_text = m.group(1).strip()
        elif has_think_open:
            reasoning_text = text.split("<think>", 1)[1].strip()
        else:
            reasoning_text = text.strip()

        return {
            "has_think_open": has_think_open,
            "has_think_close": has_think_close,
            "has_answer_open": has_answer_open,
            "has_answer_close": has_answer_close,
            "is_fully_formatted": is_fully_formatted,
            "malformed_reason": malformed_reason,
            "extracted_reasoning": reasoning_text,
        }


# =============================================================================
# 4. Evaluation Engine Data Structures & Runner
# =============================================================================

@dataclass
class ExampleEvalRecord:
    """Complete record for an individual benchmark example evaluation."""
    dataset: str
    id: str
    question: str
    gold_answer: str
    decoding_strategy: str
    prompt_text: str
    generated_text: str
    extracted_reasoning: str
    extracted_answer: str | None
    extraction_method: str
    extraction_success: bool
    is_correct: bool
    is_fully_formatted: bool
    malformed_reason: str
    prompt_tokens: int
    generated_tokens: int
    latency_sec: float
    stopped_by_eos: bool


@dataclass
class BenchmarkSummaryMetrics:
    """Summary metrics for a reasoning benchmark dataset evaluation."""
    dataset: str
    decoding_strategy: str
    num_examples: int
    num_correct: int
    accuracy: float
    num_extracted: int
    extraction_success_rate: float
    num_fully_formatted: int
    formatting_success_rate: float
    num_malformed: int
    avg_generated_tokens: float
    median_generated_tokens: float
    min_generated_tokens: int
    max_generated_tokens: int
    avg_latency_sec: float
    median_latency_sec: float
    total_time_sec: float


def evaluate_reasoning_benchmark(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    raw_examples: list[dict[str, Any]],
    dataset_name: str,
    decoding_strategy: str = "greedy",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 192,
    device: str = "cuda",
    progress_callback: Any = None,
) -> tuple[BenchmarkSummaryMetrics, list[ExampleEvalRecord]]:
    """
    Run zero-shot reasoning evaluation on a list of dataset records.

    Args:
        model: Frozen MiniR1Transformer.
        tokenizer: Tokenizer instance.
        raw_examples: List of raw dataset dicts.
        dataset_name: 'gsm8k', 'svamp', or 'aqua'.
        decoding_strategy: 'greedy' or 'sampling'.
        temperature: Temperature value.
        top_p: Top-p threshold.
        max_new_tokens: Token budget for newly generated text.
        device: 'cuda' or 'cpu'.
        progress_callback: Optional callable(step, total).

    Returns:
        (BenchmarkSummaryMetrics, list[ExampleEvalRecord])
    """
    model.eval()
    is_mc = (dataset_name.lower() == "aqua")
    extractor_func = AnswerExtractor.extract_aqua_answer if is_mc else AnswerExtractor.extract_gsm8k_answer
    dataset_type = "multiple_choice" if is_mc else "numerical"

    records: list[ExampleEvalRecord] = []
    generated_token_counts: list[int] = []
    latencies: list[float] = []

    total_examples = len(raw_examples)
    start_all = time.perf_counter()

    for idx, ex in enumerate(raw_examples):
        # Extract standardized question and gold answer
        if dataset_name.lower() == "gsm8k":
            q_text = ex.get("question", "").strip()
            raw_a = ex.get("answer", "")
            gold_ans = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            rec_id = f"gsm8k_test_{idx}"
        elif dataset_name.lower() == "svamp":
            body = ex.get("Body", "").strip()
            q = ex.get("Question", "").strip()
            q_text = f"{body} {q}".strip() if body else q
            gold_ans = str(ex.get("Answer", "")).strip()
            rec_id = str(ex.get("ID", f"svamp_test_{idx}"))
        elif dataset_name.lower() == "aqua":
            q = ex.get("question", "").strip()
            opts = ex.get("options", [])
            if isinstance(opts, (list, tuple)) and opts:
                opt_str = "\nOptions:\n" + "\n".join(f"- {o}" for o in opts)
                q_text = f"{q}\n{opt_str}".strip()
            else:
                q_text = q
            gold_ans = str(ex.get("correct", "")).strip().upper()
            rec_id = f"aqua_test_{idx}"
        else:
            raise ValueError(f"Unsupported benchmark: {dataset_name}")

        # Construct standard reasoning prompt
        prompt_text = f"Question: {q_text}\n<think>\n"
        prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)

        # Autoregressive forward generation
        gen_out = generate_autoregressive(
            model=model,
            prompt_token_ids=prompt_tokens,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        full_output = gen_out.generated_text

        # Formatting validation
        fmt_info = FormatValidator.validate_formatting(prompt_text + full_output)

        # Answer extraction
        extracted_ans, extract_method = extractor_func(full_output)
        extraction_success = (extracted_ans is not None and extract_method != "extraction_failed")

        # Correctness evaluation
        is_correct = AnswerExtractor.compare_answers(extracted_ans, gold_ans, dataset_type=dataset_type)

        rec = ExampleEvalRecord(
            dataset=dataset_name.lower(),
            id=rec_id,
            question=q_text,
            gold_answer=gold_ans,
            decoding_strategy=decoding_strategy,
            prompt_text=prompt_text,
            generated_text=full_output,
            extracted_reasoning=fmt_info["extracted_reasoning"],
            extracted_answer=extracted_ans,
            extraction_method=extract_method,
            extraction_success=extraction_success,
            is_correct=is_correct,
            is_fully_formatted=fmt_info["is_fully_formatted"],
            malformed_reason=fmt_info["malformed_reason"],
            prompt_tokens=gen_out.num_prompt_tokens,
            generated_tokens=gen_out.num_generated_tokens,
            latency_sec=gen_out.latency_sec,
            stopped_by_eos=gen_out.stopped_by_eos,
        )
        records.append(rec)
        generated_token_counts.append(gen_out.num_generated_tokens)
        latencies.append(gen_out.latency_sec)

        if progress_callback is not None and ((idx + 1) % 50 == 0 or (idx + 1) == total_examples):
            progress_callback(idx + 1, total_examples)

    total_time = time.perf_counter() - start_all

    num_correct = sum(1 for r in records if r.is_correct)
    num_extracted = sum(1 for r in records if r.extraction_success)
    num_formatted = sum(1 for r in records if r.is_fully_formatted)
    num_malformed = sum(1 for r in records if not r.is_fully_formatted)

    accuracy = num_correct / max(1, total_examples)
    ext_rate = num_extracted / max(1, total_examples)
    fmt_rate = num_formatted / max(1, total_examples)

    summary = BenchmarkSummaryMetrics(
        dataset=dataset_name.lower(),
        decoding_strategy=decoding_strategy,
        num_examples=total_examples,
        num_correct=num_correct,
        accuracy=accuracy,
        num_extracted=num_extracted,
        extraction_success_rate=ext_rate,
        num_fully_formatted=num_formatted,
        formatting_success_rate=fmt_rate,
        num_malformed=num_malformed,
        avg_generated_tokens=float(np.mean(generated_token_counts)) if generated_token_counts else 0.0,
        median_generated_tokens=float(np.median(generated_token_counts)) if generated_token_counts else 0.0,
        min_generated_tokens=int(np.min(generated_token_counts)) if generated_token_counts else 0,
        max_generated_tokens=int(np.max(generated_token_counts)) if generated_token_counts else 0,
        avg_latency_sec=float(np.mean(latencies)) if latencies else 0.0,
        median_latency_sec=float(np.median(latencies)) if latencies else 0.0,
        total_time_sec=total_time,
    )

    return summary, records


# =============================================================================
# 5. Qualitative Sample Classifier
# =============================================================================

def select_qualitative_samples(
    records: list[ExampleEvalRecord],
    target_count: int = 20,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """
    Select representative qualitative samples covering 5 key forensic categories:
    1. correct_and_coherent
    2. correct_and_incoherent (lucky guess / degenerate reasoning)
    3. wrong_and_plausible (logical/arithmetic slip)
    4. wrong_and_incoherent (hallucination / grammar breakdown)
    5. formatting_failure (missing tags / unclosed markup)
    """
    import random
    rng = random.Random(seed)

    cat_1: list[ExampleEvalRecord] = []  # correct + coherent
    cat_2: list[ExampleEvalRecord] = []  # correct + incoherent
    cat_3: list[ExampleEvalRecord] = []  # wrong + plausible
    cat_4: list[ExampleEvalRecord] = []  # wrong + incoherent
    cat_5: list[ExampleEvalRecord] = []  # formatting failure

    for r in records:
        text = r.generated_text
        has_repetitions = bool(re.search(r"(\b\w+\b)(?:\s+\1){3,}", text))
        is_coherent = not has_repetitions and len(text.strip()) > 10

        if not r.is_fully_formatted:
            cat_5.append(r)

        if r.is_correct:
            if is_coherent and r.is_fully_formatted:
                cat_1.append(r)
            else:
                cat_2.append(r)
        else:
            if is_coherent and r.extraction_success:
                cat_3.append(r)
            else:
                cat_4.append(r)

    # Deterministic sampling from each category bucket
    def sample_from(pool: list[ExampleEvalRecord], k: int) -> list[ExampleEvalRecord]:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled[:k]

    per_bucket = max(1, target_count // 5)

    sampled_1 = sample_from(cat_1, per_bucket)
    sampled_2 = sample_from(cat_2, per_bucket)
    sampled_3 = sample_from(cat_3, per_bucket)
    sampled_4 = sample_from(cat_4, per_bucket)
    sampled_5 = sample_from(cat_5, per_bucket)

    # Collect and ensure total target_count
    collected = {
        "correct_and_coherent": [asdict(x) for x in sampled_1],
        "correct_and_incoherent": [asdict(x) for x in sampled_2],
        "wrong_and_plausible": [asdict(x) for x in sampled_3],
        "wrong_and_incoherent": [asdict(x) for x in sampled_4],
        "formatting_failure": [asdict(x) for x in sampled_5],
    }

    return collected
