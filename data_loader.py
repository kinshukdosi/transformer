"""
You can't feed entire datasets through a model at once, so we have to slice it up into
chunks of length "block_size". One chunk of data actually contains "block_size" training
examples, one for every position in the chunk. We need two tensors; inputs and targets.

The way to do this is to randomly sample starting positions in the dataset, and extract
chunks of data from there. Random sampling stabilises training.
"""

import torch


def batch_data(
    tokens: torch.Tensor, batch_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    tokens: list of token ids (encoded text) after tokenization
    batch_size: number of chunks to sample from the given tokens
    block_size: length of each chunk
    """
    assert len(tokens) > block_size, "Not enough tokens for given block size"

    # randomly sample starting positions within tokens
    ix = torch.randint(len(tokens) - block_size, (batch_size,))

    inputs = torch.stack([tokens[i : i + block_size] for i in ix])
    targets = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])

    return inputs, targets
