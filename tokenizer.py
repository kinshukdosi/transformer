import json
import heapq
import argparse
import regex
from pathlib import Path
from collections import Counter, defaultdict

BYTE_RANGE = 256
DATA_DIR = Path(__file__).parent / "data"

# GPT-4's pre-tokenization pattern (from tiktoken's cl100k_base). text is split into
# chunks before BPE, and merges never cross a chunk boundary, so tokens can't span
# words, e.g. "and " or "e t". chunks are: contractions ('s, 't, 're ...), words with
# an optional leading space or punctuation char, numbers of up to 3 digits,
# punctuation runs, and whitespace. \p{L} is any unicode letter and \p{N} any number,
# which is why this needs the regex module instead of re
SPLIT_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class SimpleTokenizer:
    """
    The simple character-level tokenizer that Karpathy implements in his video.
    Effective for tinyshakespeare dataset
    """

    def __init__(self) -> None:
        self.chars: list
        self.vocab_size: int

    def train(self, text, vocab_size=0):
        # vocab_size input doesn't matter here, we have it so that all tokenizers can be
        # trained by calling the same function. the actual vocab size will be derived
        # from chars
        self._build_vocab(sorted(list(set(text))))

    def _build_vocab(self, chars: list):
        self.chars = chars
        self.vocab_size = len(self.chars)
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    def decode(self, ids: list) -> str:
        return "".join([self.itos[i] for i in ids])

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text]

    def state_dict(self) -> dict:
        """Everything needed to rebuild the tokenizer, e.g. to store in a checkpoint"""
        return {"chars": self.chars}

    def load_state_dict(self, state: dict):
        self._build_vocab(state["chars"])


class BPETokenizer:
    """
    Byte-pair encoding (BPE). Many modern LLMs use this algorithm to train their
    tokenizers. The idea is to replace the most common contiguous sequences of
    characters into new tokens until a vocabulary of a predefined size is obtained.

    Like GPT-2 and later, the text is first split into chunks (roughly words) with a
    regex, and BPE runs within each chunk. See Karpathy's minbpe RegexTokenizer.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.vocab_size = 0
        self._cache: dict[str, list] = {}  # chunk -> token ids, see encode()

    def _replace(self, ids: list, pair: tuple[int, int], token_id: int) -> list:
        """Replaces given consecutive pair with new token id"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(token_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _get_counts(self, ids: list) -> dict[tuple[int, int], int]:
        """Get counts of consective byte pairs"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text: str, vocab_size: int):
        assert (
            vocab_size > BYTE_RANGE
        ), "Desired vocabulary size must be greater than 256"

        # merges never cross chunk boundaries, so a chunk that appears 1000 times only
        # needs to be stored once, along with its count. natural text has far fewer
        # distinct chunks than total chunks, which is what makes this fast
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # count every pair once up front, and remember which words contain it. after
        # that, each merge only updates the words that contain the merged pair, instead
        # of recounting the whole text
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for word_idx, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                pair_words[pair].add(word_idx)

        # max-heap of (-count, pair), so the most common pair can be found without
        # scanning every pair. when a count changes we push the new count rather than
        # updating the old entry, and skip old entries when they are popped. ties are
        # broken by the smallest pair, so training is deterministic
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        vocab: dict[int, bytes] = {
            x: bytes([x]) for x in range(BYTE_RANGE)
        }  # used to decode
        merges: dict[tuple[int, int], int] = {}  # used to encode

        # vocab starts with all byte values, so we iterate from 256
        # next_id is the next available entry for the vocabulary
        for next_id in range(BYTE_RANGE, vocab_size):

            max_pair = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if neg_count < 0 and pair_counts[pair] == -neg_count:
                    max_pair = pair
                    break
            if max_pair is None:
                break  # every chunk is a single token, there is nothing left to merge

            # remove the old pairs of every affected word, merge, then add its new pairs
            changed = set()
            for word_idx in pair_words.pop(max_pair):
                word, freq = words[word_idx], freqs[word_idx]
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    changed.add(pair)

                word = self._replace(word, max_pair, next_id)
                words[word_idx] = word

                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq
                    pair_words[pair].add(word_idx)
                    changed.add(pair)
                # pair_words isn't cleaned up for pairs a word no longer contains. if
                # one of those pairs is merged later, _replace leaves the word unchanged
                # and its counts are removed and added back, so the result is the same

            for pair in changed:
                if pair_counts[pair] > 0:
                    heapq.heappush(heap, (-pair_counts[pair], pair))

            merges[max_pair] = next_id
            vocab[next_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._cache = {}

    def decode(self, ids: list) -> str:
        text_bytes = b"".join(self.vocab[x] for x in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> list:
        # each distinct chunk is only encoded once, then looked up. the cache is kept
        # between calls, so encoding the validation split reuses the training split's
        # chunks. it is cleared whenever the merges change
        ids = []
        for chunk in SPLIT_PATTERN.findall(text):
            chunk_ids = self._cache.get(chunk)
            if chunk_ids is None:
                chunk_ids = self._encode_chunk(chunk.encode("utf-8"))
                self._cache[chunk] = chunk_ids
            ids.extend(chunk_ids)
        return ids

    def _encode_chunk(self, chunk_bytes: bytes) -> list:
        """Applies the learned merges to the bytes of a single chunk"""
        ids = list(chunk_bytes)
        while len(ids) >= 2:

            counts = self._get_counts(ids)

            # we need to find the pair with the lowest merge index, because merges must
            # be replayed in the same order that they were learned during training
            # e.g. if t+o -> 256 happened before to+p -> 257, you need to merge t+o
            # before you can see to+p

            min_pair = min(counts, key=lambda x: self.merges.get(x, float("inf")))
            # the default value for .get() is set to inf here, this is because if there
            # are no more merges left, then min() will return the first pair (from
            # counts), so then this will be our condition to return early

            if min_pair not in self.merges:
                break

            merge_id = self.merges[min_pair]
            ids = self._replace(ids, min_pair, merge_id)

        return ids

    def state_dict(self) -> dict:
        """
        Everything needed to rebuild the tokenizer, e.g. to store in a checkpoint. Only
        uses plain types, so it can be saved to JSON and loaded by torch.load with
        weights_only=True
        """
        return {
            # we decode the value here because bytes are not JSON serializable
            # we use latin-1 because some byte sequences between 0-255 aren't valid
            # utf-8, so this could cause errors if the training data had unusual chars
            "vocab": {k: v.decode("latin-1") for k, v in self.vocab.items()},
            # JSON doesn't support tuple keys, so we store merges this way
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }

    def load_state_dict(self, state: dict):
        assert "vocab" in state, "vocab not found in tokenizer state"
        assert "merges" in state, "merges not found in tokenizer state"

        # JSON turns the integer vocab keys into strings, so convert them back
        self.vocab = {int(k): v.encode("latin-1") for k, v in state["vocab"].items()}
        self.merges = {(a, b): c for a, b, c in state["merges"]}
        self.vocab_size = len(self.vocab)
        self._cache = {}

    def save(self, path: Path = DATA_DIR / "tokenizer.json"):
        assert path.suffix == ".json", "Path should point to .json file"

        # good to specify encoding here in case vocabulary includes non-ASCII tokens
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path = DATA_DIR / "tokenizer.json"):
        assert path.exists(), "File does not exist"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.load_state_dict(data)

        return tokenizer


if __name__ == "__main__":
    """Tokenize text using BPE tokenizer and save to .json"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, help="Path to input .txt file")
    parser.add_argument("--output", "-o", type=Path, help="Path to output .json file")
    parser.add_argument("--vocab_size", "-v", type=int, help="Vocab size")
    args = parser.parse_args()

    assert Path.exists(args.input), f"{args.input} does not exist"
    assert args.input.suffix == ".txt", "Input file muxt be .txt file"
    assert args.output.suffix == ".json", "Output file muxt be .json file"
    assert type(args.vocab_size) is int, "Vocab size must be integer value"

    tokenizer = BPETokenizer()
    with open(args.input, "r") as f:
        text = f.read()
    tokenizer.train(text, args.vocab_size)
    tokenizer.save(args.output)
