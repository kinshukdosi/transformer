from pathlib import Path
from tokenizer import train, decode, encode

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"


def test_tinyshakespeare():
    """Test training on tinyshakespeare dataset, and basic encode/decode round trip"""

    vocab_size = 300
    input_text = "Hello World!"

    with open(DATA_DIR / "tinyshakespeare.txt", "r") as f:
        text = f.read()
    merges, vocab = train(text, vocab_size)

    encoded = encode(input_text, merges)
    decoded = decode(encoded, vocab)

    assert input_text == decoded, "Encode/decode round trip failed!"
