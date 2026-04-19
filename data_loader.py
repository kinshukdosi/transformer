"""
You can't feed entire datasets through a model at once, so we have to slice it up into
chunks of length "max_seq_len". One chunk of data actually contains "max_seq_len"
training examples, one for every position in the chunk. We need two tensors; inputs and
targets.

The way to do this is to randomly sample starting positions in the dataset, and extract
chunks of data from there. Random sampling stabilises training.
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def batch_data(
    tokens: torch.Tensor, batch_size: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    tokens: list of token ids (encoded text) after tokenization
    batch_size: number of chunks to sample from the given tokens
    max_seq_len: length of each sampled chunk
    """
    assert len(tokens) > max_seq_len, "Not enough tokens for given sequence length"

    # randomly sample starting positions within tokens
    ix = torch.randint(len(tokens) - max_seq_len, (batch_size,))

    inputs = torch.stack([tokens[i : i + max_seq_len] for i in ix])
    targets = torch.stack([tokens[i + 1 : i + max_seq_len + 1] for i in ix])

    inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets
