from pathlib import Path
from tokenizer import Tokenizer

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"


def test_tinyshakespeare():
    """Test training on tinyshakespeare dataset, and basic encode/decode round trip"""

    tokenizer = Tokenizer()
    vocab_size = 300
    input_text = "Hello World!"

    with open(DATA_DIR / "tinyshakespeare.txt", "r") as f:
        text = f.read()

    tokenizer.train(text, vocab_size)
    encoded = tokenizer.encode(input_text)
    decoded = tokenizer.decode(encoded)

    assert input_text == decoded, "Encode/decode round trip failed!"
