"""Simple model, predicts the next token given only the current token. It's
essentially a lookup table with dimensions based on the size of the vocabulary of the
dataset. Each row of the lookup table corresponds to one token, and contains the scores
for the next token."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # this creates a learnable matrix. in this case, both dimensions are vocab_size
        # because each row has the scores for the ENTIRE vocabulary. the embedding here
        # is the entire prediction table for the model
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    # we don't need to call forward() directly because we inherit from nn.Module. Since
    # __call__() wraps forward(), we can call the model object like a function instead
    def forward(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        # this is the forward computation! for each input token, it retrieves the
        # corresponding row from the embedding matrix. so the output shape of logits
        # here is (batch_size, block_size, vocab_size)
        logits = self.embedding(inputs)

        loss = None

        if targets is not None:
            assert inputs.shape == targets.shape, "inputs/targets shape mismatch"
            batch_size, block_size, vocab_size = logits.shape

            # loss calculation. this function does a few things: softmax on the logits
            # to convert them to probabilities, then take the natural log, then negate
            # the log probability of the correct next token (from targets). this gives
            # a low loss if the model is confident and correct, a high loss if confident
            # and incorrect. if its totally random, we expect the probability for each
            # token to be 1/vocab_size, so we expect the loss to be ln(vocab_size)

            loss = F.cross_entropy(
                logits.view(batch_size * block_size, vocab_size),
                targets.view(batch_size * block_size),
            )

        return logits, loss
