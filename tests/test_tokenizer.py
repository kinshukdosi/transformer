"""
Correctness tests for the character-level and BPE tokenizers
"""

import torch
import pytest
from pathlib import Path
from tokenizer import BYTE_RANGE, BPETokenizer, SimpleTokenizer

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"

VOCAB_SIZE = 300


def get_text(num_chars: int = 50_000) -> str:
    """Helper function for tests. BPE training is slow, so use a slice of the dataset"""
    with open(DATA_DIR / "tinyshakespeare.txt", "r") as f:
        text = f.read()
    return text[:num_chars]


@pytest.mark.parametrize(("tokenizer"), [BPETokenizer, SimpleTokenizer])
def test_round_trip(tokenizer):
    """Encoding then decoding the training text should give back the original"""

    text = get_text()
    tokenizer = tokenizer()
    tokenizer.train(text, VOCAB_SIZE)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_round_trip_unseen_text():
    """BPE works on bytes, so it can encode text it has never seen"""

    tokenizer = BPETokenizer()
    tokenizer.train(get_text(), VOCAB_SIZE)
    text = "Ünïcödé, 世界 and emoji 🙂\ttabs"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_simple_tokenizer_unknown_char():
    """The character-level tokenizer has no unknown token, so unseen chars fail"""

    tokenizer = SimpleTokenizer()
    tokenizer.train(get_text())

    with pytest.raises(KeyError):
        tokenizer.encode("世界")


def test_bpe_vocab():
    """Test vocabulary size, merges and that encoding compresses the text"""

    text = get_text()
    tokenizer = BPETokenizer()
    tokenizer.train(text, VOCAB_SIZE)
    ids = tokenizer.encode(text)

    assert tokenizer.vocab_size == VOCAB_SIZE
    assert len(tokenizer.vocab) == VOCAB_SIZE
    assert len(tokenizer.merges) == VOCAB_SIZE - BYTE_RANGE
    assert max(ids) < VOCAB_SIZE
    assert len(ids) < len(text.encode("utf-8"))

    # every merged token decodes to the bytes of the pair it was made from
    for (a, b), token_id in tokenizer.merges.items():
        assert tokenizer.vocab[token_id] == tokenizer.vocab[a] + tokenizer.vocab[b]


def test_simple_tokenizer_vocab():
    """Vocabulary should be exactly the set of characters in the training text"""

    text = get_text()
    tokenizer = SimpleTokenizer()
    tokenizer.train(text)
    ids = tokenizer.encode(text)

    assert tokenizer.vocab_size == len(set(text))
    assert len(ids) == len(text)
    assert max(ids) < tokenizer.vocab_size


def test_bpe_save_and_load(tmp_path):
    """A reloaded tokenizer should behave exactly like the original"""

    text = get_text()
    tokenizer = BPETokenizer()
    tokenizer.train(text, VOCAB_SIZE)
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    tokenizer_readback = BPETokenizer.load(path)

    assert tokenizer_readback.vocab_size == tokenizer.vocab_size
    assert tokenizer_readback.encode(text) == tokenizer.encode(text)
    assert tokenizer_readback.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize(("tokenizer"), [BPETokenizer, SimpleTokenizer])
def test_state_dict(tokenizer, tmp_path):
    """A tokenizer rebuilt from its state (as stored in a checkpoint) should match"""

    text = get_text()
    original = tokenizer()
    original.train(text, VOCAB_SIZE)

    # checkpoints are loaded with torch.load's default weights_only=True, which only
    # allows plain types, so check the state survives that too
    path = tmp_path / "tokenizer.pt"
    torch.save(original.state_dict(), path)
    restored = tokenizer()
    restored.load_state_dict(torch.load(path))

    assert restored.vocab_size == original.vocab_size
    assert restored.encode(text) == original.encode(text)
    assert restored.decode(original.encode(text)) == text
