"""
Tests on Karpathy's tinyshakespeare dataset
40,000 lines of Shakespeare from a variety of Shakespeare's plays
https://huggingface.co/datasets/karpathy/tiny_shakespeare
"""

import torch
from pathlib import Path
from data_loader import batch_data
from tokenizer import Tokenizer

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"


def test_tokenizer():
    """Test tokenizer and basic encode/decode round trip"""

    tokenizer = Tokenizer()
    vocab_size = 270
    input_text = "Hello World!"

    with open(DATA_DIR / "tinyshakespeare.txt", "r") as f:
        text = f.read()

    tokenizer.train(text, vocab_size)
    encoded = tokenizer.encode(input_text)
    decoded = tokenizer.decode(encoded)

    assert input_text == decoded, "Encode/decode round trip failed!"


def test_data_loader():
    """Test chunking up dataset for model training purposes"""

    tokenizer = Tokenizer()
    vocab_size = 257
    batch_size = 10
    block_size = 20

    with open(DATA_DIR / "tinyshakespeare.txt", "r") as f:
        text = f.read()

    tokenizer.train(text, vocab_size)
    encoded = torch.tensor(tokenizer.encode(text))  # batch_data() expects torch.Tensor

    inputs, targets = batch_data(encoded, batch_size, block_size)

    assert inputs.shape == (batch_size, block_size)
    assert targets.shape == (batch_size, block_size)
