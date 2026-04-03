"""
Byte-pair encoding (BPE). Many modern LLMs use this algorithm to train their
tokenizers. The idea is to replace the most common contiguous sequences of
characters into new tokens until a vocabulary of a predefined size is obtained.
"""

BYTE_RANGE = 256


class Tokenizer:

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

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

        text_bytes = text.encode("utf-8")  # convert to raw bytes
        ids = list(text_bytes)  # list of ints in range 0 to 255 (byte range)

        vocab: dict[int, bytes] = {
            x: bytes([x]) for x in range(BYTE_RANGE)
        }  # used to decode
        merges: dict[tuple[int, int], int] = {}  # used to encode

        # vocab starts with all byte values, so we iterate from 256
        # next_id is the next available entry for the vocabulary
        for next_id in range(BYTE_RANGE, vocab_size):

            counts = self._get_counts(ids)

            max_pair = max(counts, key=lambda x: counts[x])
            ids = self._replace(ids, max_pair, next_id)

            merges[max_pair] = next_id
            vocab[next_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids: list) -> str:
        text_bytes = b"".join(self.vocab[x] for x in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> list:
        text_bytes = text.encode("utf-8")  # convert to raw bytes
        ids = list(text_bytes)
        while len(ids) >= 2:

            counts = self._get_counts(ids)

            # we need to find the pair with the lowest merge index, because merges must be
            # replayed in the same order that they were learned during training
            # e.g. if t+o -> 256 happened before to+p -> 257, you need to merge t+o before
            # you can see to+p

            min_pair = min(counts, key=lambda x: self.merges.get(x, float("inf")))
            # the default value for .get() is set to inf here, this is because if there are
            # no more merges left, then min() will return the first pair (from counts),
            # so then this will be our condition to return early

            if min_pair not in self.merges:
                break

            merge_id = self.merges[min_pair]
            ids = self._replace(ids, min_pair, merge_id)

        return ids
