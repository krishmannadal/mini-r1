"""
Mini-R1 -- Tokenizer Test Suite
===============================
Pytest unit tests verifying SentencePiece BPE tokenizer operations:
- Model initialization & loading
- Encoding/decoding round-trips
- Vocabulary size validity
- Special tokens (BOS, EOS, PAD, UNK)
- Deterministic reasoning tokens (<think>, </think>, <answer>, </answer>)
- Batch encoding with uniform padding
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import Tokenizer, train_sentencepiece, DEFAULT_USER_DEFINED_SYMBOLS
MODEL_PATH = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
CORPUS_PATH = PROJECT_ROOT / "data" / "sample_corpus.txt"


@pytest.fixture(scope="session")
def tokenizer_model() -> Path:
    """Ensure a trained tokenizer model exists for the test session."""
    if not MODEL_PATH.exists():
        train_sentencepiece(
            input_file=CORPUS_PATH,
            model_prefix=MODEL_PATH.with_suffix(""),
            vocab_size=1000,
            model_type="bpe",
            user_defined_symbols=DEFAULT_USER_DEFINED_SYMBOLS,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
        )
    return MODEL_PATH


@pytest.fixture
def tokenizer(tokenizer_model: Path) -> Tokenizer:
    """Create a Tokenizer instance from the trained model fixture."""
    return Tokenizer(tokenizer_model)


class TestTokenizerInitialization:
    """Tests covering model loading and error handling."""

    def test_missing_model_raises_error(self, tmp_path: Path):
        """Providing a non-existent model path must raise FileNotFoundError."""
        missing_path = tmp_path / "non_existent.model"
        with pytest.raises(FileNotFoundError):
            Tokenizer(missing_path)

    def test_vocab_size_positive(self, tokenizer: Tokenizer):
        """Vocab size must be a positive integer."""
        vsize = tokenizer.vocab_size()
        assert isinstance(vsize, int)
        assert vsize > 0
        assert len(tokenizer) == vsize


class TestSpecialTokens:
    """Verify standard and domain-specific special token IDs."""

    def test_standard_special_token_ids(self, tokenizer: Tokenizer):
        """Special token IDs must match the deterministic assignments."""
        assert tokenizer.unk_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2
        assert tokenizer.pad_id == 3

    def test_reasoning_token_ids(self, tokenizer: Tokenizer):
        """Reasoning tokens must be defined and map back to exact string tokens."""
        assert tokenizer.think_start_id > 0
        assert tokenizer.think_end_id > 0
        assert tokenizer.answer_start_id > 0
        assert tokenizer.answer_end_id > 0

        assert tokenizer.id_to_token(tokenizer.think_start_id) == "<think>"
        assert tokenizer.id_to_token(tokenizer.think_end_id) == "</think>"
        assert tokenizer.id_to_token(tokenizer.answer_start_id) == "<answer>"
        assert tokenizer.id_to_token(tokenizer.answer_end_id) == "</answer>"

    def test_reasoning_tokens_are_not_fragmented(self, tokenizer: Tokenizer):
        """
        Special tokens <think>, </think>, <answer>, </answer> must encode
        as a SINGLE token ID, never split into subwords like '<', 'think', '>'.
        """
        for sym, expected_id in [
            ("<think>", tokenizer.think_start_id),
            ("</think>", tokenizer.think_end_id),
            ("<answer>", tokenizer.answer_start_id),
            ("</answer>", tokenizer.answer_end_id),
        ]:
            encoded = tokenizer.encode(sym)
            assert encoded == [expected_id], (
                f"Token '{sym}' was fragmented into {encoded} instead of single ID [{expected_id}]"
            )


class TestEncodingDecoding:
    """Verify encode, decode, and round-trip consistency."""

    def test_encode_returns_integers(self, tokenizer: Tokenizer):
        """Encoding text must produce a list of ints."""
        ids = tokenizer.encode("What is 10 + 20?")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_bos_eos_flags(self, tokenizer: Tokenizer):
        """add_bos and add_eos must prepend bos_id and append eos_id."""
        text = "Hello world"
        ids_bare = tokenizer.encode(text, add_bos=False, add_eos=False)
        ids_both = tokenizer.encode(text, add_bos=True, add_eos=True)

        assert ids_both[0] == tokenizer.bos_id
        assert ids_both[-1] == tokenizer.eos_id
        assert ids_both[1:-1] == ids_bare

    def test_decode_roundtrip(self, tokenizer: Tokenizer):
        """Decoding encoded IDs should reconstruct the original text faithfully."""
        original = "Solve for x: 3*x + 15 = 45."
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)
        assert decoded == original

    def test_cot_sequence_roundtrip(self, tokenizer: Tokenizer):
        """CoT sequence with markup must roundtrip cleanly."""
        cot_text = "Question: 2+2?\n<think>\n2 plus 2 equals 4.\n</think>\n<answer>4</answer>"
        ids = tokenizer.encode(cot_text)
        decoded = tokenizer.decode(ids)
        assert decoded == cot_text


class TestBatchEncoding:
    """Verify batched encoding and padding functionality."""

    def test_batch_encoding_unpadded(self, tokenizer: Tokenizer):
        """Batch encoding without padding returns variable-length lists."""
        texts = ["Short", "This is a significantly longer mathematical sentence."]
        batch = tokenizer.encode_batch(texts, pad=False)
        assert len(batch) == 2
        assert len(batch[0]) != len(batch[1])

    def test_batch_encoding_padded(self, tokenizer: Tokenizer):
        """Batch encoding with padding returns uniform-length lists."""
        texts = ["Short", "This is a longer sentence.", "Mid"]
        batch = tokenizer.encode_batch(texts, pad=True)
        assert len(batch) == 3
        lengths = [len(seq) for seq in batch]
        assert all(l == lengths[0] for l in lengths)

        # The shortest sequence must have padding tokens at the end
        pad_id = tokenizer.pad_id
        assert batch[0][-1] == pad_id

    def test_batch_encoding_max_length(self, tokenizer: Tokenizer):
        """Explicit max_length pads or truncates to exact length."""
        texts = ["A", "B"]
        target_len = 16
        batch = tokenizer.encode_batch(texts, pad=True, max_length=target_len)
        for seq in batch:
            assert len(seq) == target_len
