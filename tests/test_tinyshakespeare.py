"""
Tests on Karpathy's tinyshakespeare dataset
40,000 lines of Shakespeare from a variety of Shakespeare's plays
https://huggingface.co/datasets/karpathy/tiny_shakespeare
"""

import torch
import math
import train
import pytest
import yaml
from pathlib import Path
from data_loader import batch_data
from tokenizer import BPETokenizer, SimpleTokenizer
from train import parse_config
from models.bigram import BigramLanguageModel

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"
CONFIG_DIR = TEST_DIR.parent / "config"


def get_tinyshakespeare_text() -> str:
    """Helper function for tests"""
    with open(DATA_DIR / "tinyshakespeare.txt", "r") as f:
        text = f.read()
    return text


@pytest.mark.parametrize(("tokenizer"), [BPETokenizer, SimpleTokenizer])
def test_tokenizer(tokenizer):
    """Test tokenizer and basic encode/decode round trip"""

    text = get_tinyshakespeare_text()
    tokenizer = tokenizer()
    vocab_size = 270
    input_text = "Hello World!"

    tokenizer.train(text, vocab_size)
    encoded = tokenizer.encode(input_text)
    decoded = tokenizer.decode(encoded)

    assert input_text == decoded, "Encode/decode round trip failed!"


def test_load_and_save_tokens():
    """Test tokenizer load and save functions"""

    text = get_tinyshakespeare_text()
    tokenizer = BPETokenizer()
    vocab_size = 257
    path = DATA_DIR / "test_load_and_save.json"

    tokenizer.train(text, vocab_size)
    tokenizer.save(path)
    tokenizer_readback = BPETokenizer.load(path)
    path.unlink()

    assert tokenizer.vocab == tokenizer_readback.vocab
    assert tokenizer.merges == tokenizer_readback.merges


@pytest.mark.parametrize(("tokenizer"), [BPETokenizer, SimpleTokenizer])
def test_data_loader(tokenizer):
    """Test chunking up dataset for model training purposes"""

    text = get_tinyshakespeare_text()
    tokenizer = tokenizer()
    vocab_size = 257
    batch_size = 10
    block_size = 20

    tokenizer.train(text, vocab_size)
    encoded = torch.tensor(tokenizer.encode(text))  # batch_data() expects torch.Tensor

    inputs, targets = batch_data(encoded, batch_size, block_size)

    assert inputs.shape == (batch_size, block_size)
    assert targets.shape == (batch_size, block_size)


@pytest.mark.parametrize(("tokenizer"), [BPETokenizer, SimpleTokenizer])
def test_bigram_language_model(tokenizer):
    """Test running forward pass on untrained bigram language model"""

    text = get_tinyshakespeare_text()
    tokenizer = tokenizer()
    vocab_size = 257
    batch_size = 10
    block_size = 20
    model = BigramLanguageModel(vocab_size)

    tokenizer.train(text, vocab_size)
    encoded = torch.tensor(tokenizer.encode(text))

    inputs, targets = batch_data(encoded, batch_size, block_size)

    with torch.no_grad():
        logits, loss = model(inputs, targets)

    # negative log of (1 / vocab_size), because (1 / vocab_size is the probability for
    # uniform distribution)
    expected_loss = math.log(vocab_size)
    rmse = math.sqrt((expected_loss - loss) ** 2)

    assert rmse < 1, "loss too far from expected"
    assert logits.shape == (batch_size, block_size, vocab_size)


@pytest.mark.parametrize(
    ("config_file"), ["bigram.yaml", "attention.yaml", "transformer.yaml"]
)
def test_train_bigram_model(config_file):
    """Test very short training on tinyshakespeare dataset"""

    cfg_path = CONFIG_DIR / config_file
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = parse_config(cfg_dict)
    cfg.iterations = 500
    train.main(cfg)
